// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sub-segment range reads for FlatLayout.

use std::ops::Range;
use std::sync::Arc;

use flatbuffers::root;
use futures::FutureExt;
use futures::future::try_join_all;
use vortex_array::BufferSubRange;
use vortex_array::ChildRangeRead;
use vortex_array::RangeDecodeInfo;
use vortex_array::ValidityRangeRead;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::array as fba;
use vortex_session::SessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::session::SeparateValidityReadsEnabled;

/// Fall back to a full segment when range reads would fetch more than this fraction of the data
/// buffers in the segment.
const RANGE_READ_THRESHOLD: f64 = 0.5;

/// Maximum number of times a node may widen and replan its decode range to synchronize validity
/// or other row-aligned children.
const MAX_REPLAN_ATTEMPTS: usize = 4;

#[derive(Debug)]
pub(super) struct RangeReadPlan {
    /// The single contiguous span covering every non-validity buffer.
    data_read: Option<DataRead>,
    /// Additional reads are only permitted for explicit validity buffers.
    validity_reads: Vec<ValidityRead>,
    buffer_alignments: Vec<Alignment>,
    decode_len: usize,
    post_slice: Option<Range<usize>>,
}

#[derive(Debug)]
struct DataRead {
    segment: SegmentRead,
}

#[derive(Debug)]
struct ValidityRead {
    segment: SegmentRead,
}

#[derive(Debug)]
struct SegmentRead {
    segment_range: Range<usize>,
    buffer_slices: Vec<BufferSlice>,
}

#[derive(Debug)]
struct BufferSlice {
    buffer_idx: usize,
    range: Range<usize>,
    alignment: Alignment,
}

#[derive(Debug, Clone)]
struct BufferLocation {
    offset: usize,
    length: usize,
    alignment: Alignment,
}

#[derive(Debug, Clone)]
enum NeededBuffer {
    Full,
    Range(Range<usize>),
}

#[derive(Debug, Clone)]
struct NeededBuffers {
    entries: Vec<Option<NeededBuffer>>,
}

impl NeededBuffers {
    fn new(num_buffers: usize) -> Self {
        Self {
            entries: vec![None; num_buffers],
        }
    }

    fn need_full(&mut self, buffer_idx: u16) -> VortexResult<()> {
        let idx = buffer_idx as usize;
        let Some(entry) = self.entries.get_mut(idx) else {
            return Err(vortex_err!(
                "Array node references buffer {} but the array tree contains {} buffers",
                buffer_idx,
                self.entries.len()
            ));
        };
        *entry = Some(NeededBuffer::Full);
        Ok(())
    }

    fn need_range(&mut self, buffer_idx: u16, range: Range<usize>) -> VortexResult<()> {
        let idx = buffer_idx as usize;
        let num_buffers = self.entries.len();
        let Some(entry) = self.entries.get_mut(idx) else {
            return Err(vortex_err!(
                "Array node references buffer {} but the array tree contains {} buffers",
                buffer_idx,
                num_buffers
            ));
        };

        match entry {
            Some(NeededBuffer::Full) => {}
            Some(NeededBuffer::Range(existing)) => {
                existing.start = existing.start.min(range.start);
                existing.end = existing.end.max(range.end);
            }
            None => *entry = Some(NeededBuffer::Range(range)),
        }
        Ok(())
    }

    fn merge(&mut self, other: Self) {
        debug_assert_eq!(self.entries.len(), other.entries.len());
        for (entry, other_entry) in self.entries.iter_mut().zip(other.entries) {
            match (entry.as_mut(), other_entry) {
                (_, None) | (Some(NeededBuffer::Full), _) => {}
                (None, Some(other)) => *entry = Some(other),
                (Some(current @ NeededBuffer::Range(_)), Some(NeededBuffer::Full)) => {
                    *current = NeededBuffer::Full;
                }
                (Some(NeededBuffer::Range(current)), Some(NeededBuffer::Range(other_range))) => {
                    current.start = current.start.min(other_range.start);
                    current.end = current.end.max(other_range.end);
                }
            }
        }
    }
}

#[derive(Debug)]
struct SubtreePlan {
    decoded_range: Range<usize>,
    data_needed: NeededBuffers,
    validity_needed: NeededBuffers,
}

enum AnalyzeOutcome {
    Complete(SubtreePlan),
    Replan(Range<usize>),
}

fn compute_buffer_locations(fb_array: &fba::Array<'_>) -> Vec<BufferLocation> {
    let mut offset = 0usize;
    fb_array
        .buffers()
        .unwrap_or_default()
        .iter()
        .map(|buffer| {
            offset += buffer.padding() as usize;
            let location = BufferLocation {
                offset,
                length: buffer.length() as usize,
                alignment: Alignment::from_exponent(buffer.alignment_exponent()),
            };
            offset += location.length;
            location
        })
        .collect()
}

fn union_ranges(left: &Range<usize>, right: &Range<usize>) -> Range<usize> {
    left.start.min(right.start)..left.end.max(right.end)
}

fn contains_range(outer: &Range<usize>, inner: &Range<usize>) -> bool {
    outer.start <= inner.start && outer.end >= inner.end
}

fn node_children(node: fba::ArrayNode<'_>) -> Vec<fba::ArrayNode<'_>> {
    node.children()
        .map_or_else(Vec::new, |children| children.iter().collect())
}

fn apply_buffer_sub_ranges(
    node: fba::ArrayNode<'_>,
    sub_ranges: &[BufferSubRange],
    needed: &mut NeededBuffers,
) -> VortexResult<bool> {
    let buffers = node.buffers().unwrap_or_default();
    if buffers.len() != sub_ranges.len() {
        return Ok(false);
    }

    for (local_idx, sub_range) in sub_ranges.iter().enumerate() {
        let global_idx = buffers.get(local_idx);
        match sub_range {
            BufferSubRange::Full => needed.need_full(global_idx)?,
            BufferSubRange::Range(range) => needed.need_range(global_idx, range.clone())?,
        }
    }
    Ok(true)
}

fn need_all_node_buffers(node: fba::ArrayNode<'_>, needed: &mut NeededBuffers) -> VortexResult<()> {
    if let Some(buffers) = node.buffers() {
        for buffer_idx in buffers.iter() {
            needed.need_full(buffer_idx)?;
        }
    }
    if let Some(children) = node.children() {
        for child in children.iter() {
            need_all_node_buffers(child, needed)?;
        }
    }
    Ok(())
}

fn resolve_decode_range(
    decode_info: &RangeDecodeInfo,
    child_plans: &[Option<SubtreePlan>],
) -> Option<Range<usize>> {
    match decode_info {
        RangeDecodeInfo::Rows(range) => Some(range.clone()),
        RangeDecodeInfo::FromChild { child_idx, divisor } => {
            if *divisor == 0 {
                return None;
            }
            let child_range = &child_plans.get(*child_idx)?.as_ref()?.decoded_range;
            if !child_range.start.is_multiple_of(*divisor)
                || !child_range.end.is_multiple_of(*divisor)
            {
                return None;
            }
            Some(child_range.start / divisor..child_range.end / divisor)
        }
    }
}

fn analyze_encoding(
    node: fba::ArrayNode<'_>,
    requested_range: Range<usize>,
    row_count: usize,
    dtype: &DType,
    ctx: &ReadContext,
    session: &VortexSession,
    num_buffers: usize,
) -> VortexResult<Option<SubtreePlan>> {
    let mut planning_range = requested_range.clone();

    for _ in 0..MAX_REPLAN_ATTEMPTS {
        let Some(outcome) = analyze_encoding_once(
            node,
            planning_range.clone(),
            row_count,
            dtype,
            ctx,
            session,
            num_buffers,
        )?
        else {
            return Ok(None);
        };

        match outcome {
            AnalyzeOutcome::Complete(plan) => {
                return Ok(contains_range(&plan.decoded_range, &requested_range).then_some(plan));
            }
            AnalyzeOutcome::Replan(required_range) => {
                let expanded = union_ranges(&planning_range, &required_range);
                if expanded == planning_range {
                    return Ok(None);
                }
                planning_range = expanded;
            }
        }
    }

    Ok(None)
}

fn analyze_encoding_once(
    node: fba::ArrayNode<'_>,
    requested_range: Range<usize>,
    row_count: usize,
    dtype: &DType,
    ctx: &ReadContext,
    session: &VortexSession,
    num_buffers: usize,
) -> VortexResult<Option<AnalyzeOutcome>> {
    let encoding_id = match ctx.resolve(node.encoding()) {
        Some(encoding_id) => encoding_id,
        None => return Ok(None),
    };
    let plugin = match session.arrays().registry().find(&encoding_id) {
        Some(plugin) => plugin,
        None => return Ok(None),
    };
    let metadata = node
        .metadata()
        .map(|metadata| metadata.bytes())
        .unwrap_or(&[]);
    let Some(plan) =
        plugin.plan_range_read(metadata, requested_range.clone(), row_count, dtype, session)?
    else {
        return Ok(None);
    };

    let children = node_children(node);
    let data_children = plan.children.len();
    let has_validity_child = match plan.validity {
        ValidityRangeRead::None => {
            if children.len() != data_children {
                return Ok(None);
            }
            false
        }
        ValidityRangeRead::Optional => match children.len() {
            len if len == data_children => false,
            len if len == data_children + 1 => true,
            _ => return Ok(None),
        },
    };

    let mut data_needed = NeededBuffers::new(num_buffers);
    let mut validity_needed = NeededBuffers::new(num_buffers);
    if !apply_buffer_sub_ranges(node, &plan.buffer_sub_ranges, &mut data_needed)? {
        return Ok(None);
    }

    let mut child_plans = Vec::with_capacity(data_children);
    for (child_idx, child_action) in plan.children.iter().cloned().enumerate() {
        let child_node = children[child_idx];
        match child_action {
            ChildRangeRead::Recurse {
                row_range,
                row_count,
                dtype,
            } => {
                let Some(child_plan) = analyze_encoding(
                    child_node,
                    row_range,
                    row_count,
                    &dtype,
                    ctx,
                    session,
                    num_buffers,
                )?
                else {
                    return Ok(None);
                };
                data_needed.merge(child_plan.data_needed.clone());
                validity_needed.merge(child_plan.validity_needed.clone());
                child_plans.push(Some(child_plan));
            }
            ChildRangeRead::RecurseExact {
                row_range,
                row_count,
                dtype,
            } => {
                let Some(child_plan) = analyze_encoding(
                    child_node,
                    row_range.clone(),
                    row_count,
                    &dtype,
                    ctx,
                    session,
                    num_buffers,
                )?
                else {
                    return Ok(None);
                };
                if child_plan.decoded_range != row_range {
                    return Ok(None);
                }
                data_needed.merge(child_plan.data_needed.clone());
                validity_needed.merge(child_plan.validity_needed.clone());
                child_plans.push(Some(child_plan));
            }
            ChildRangeRead::RecurseWithDecodedRange {
                child_idx: source_child_idx,
                row_count,
                dtype,
            } => {
                if source_child_idx >= child_idx {
                    return Ok(None);
                }
                let Some(source_range) = child_plans
                    .get(source_child_idx)
                    .and_then(Option::as_ref)
                    .map(|child| child.decoded_range.clone())
                else {
                    return Ok(None);
                };
                let Some(child_plan) = analyze_encoding(
                    child_node,
                    source_range.clone(),
                    row_count,
                    &dtype,
                    ctx,
                    session,
                    num_buffers,
                )?
                else {
                    return Ok(None);
                };
                if child_plan.decoded_range != source_range {
                    return Ok(Some(AnalyzeOutcome::Replan(union_ranges(
                        &source_range,
                        &child_plan.decoded_range,
                    ))));
                }
                data_needed.merge(child_plan.data_needed.clone());
                validity_needed.merge(child_plan.validity_needed.clone());
                child_plans.push(Some(child_plan));
            }
            ChildRangeRead::Full => {
                need_all_node_buffers(child_node, &mut data_needed)?;
                child_plans.push(None);
            }
        }
    }

    let Some(decoded_range) = resolve_decode_range(&plan.decode_info, &child_plans) else {
        return Ok(None);
    };
    if !contains_range(&decoded_range, &requested_range) {
        return Ok(None);
    }

    if has_validity_child {
        let validity_node = children[data_children];
        let validity_dtype = DType::Bool(Nullability::NonNullable);
        let Some(validity_plan) = analyze_encoding(
            validity_node,
            decoded_range.clone(),
            row_count,
            &validity_dtype,
            ctx,
            session,
            num_buffers,
        )?
        else {
            return Ok(None);
        };
        if validity_plan.decoded_range != decoded_range {
            return Ok(Some(AnalyzeOutcome::Replan(union_ranges(
                &decoded_range,
                &validity_plan.decoded_range,
            ))));
        }
        validity_needed.merge(validity_plan.data_needed);
        validity_needed.merge(validity_plan.validity_needed);
    }

    Ok(Some(AnalyzeOutcome::Complete(SubtreePlan {
        decoded_range,
        data_needed,
        validity_needed,
    })))
}

pub(super) fn try_plan_range_read(
    array_tree: &ByteBuffer,
    row_range: Range<usize>,
    row_count: usize,
    dtype: &DType,
    ctx: &ReadContext,
    session: &VortexSession,
) -> VortexResult<Option<RangeReadPlan>> {
    if row_range.is_empty() || row_range.start > row_range.end || row_range.end > row_count {
        return Ok(None);
    }
    if row_range.start == 0 && row_range.end == row_count {
        return Ok(None);
    }

    let fb_array = root::<fba::Array>(array_tree.as_ref())?;
    let locations = compute_buffer_locations(&fb_array);
    let root_node = fb_array
        .root()
        .ok_or_else(|| vortex_err!("Array tree has no root node"))?;
    let Some(subtree) = analyze_encoding(
        root_node,
        row_range.clone(),
        row_count,
        dtype,
        ctx,
        session,
        locations.len(),
    )?
    else {
        return Ok(None);
    };

    let mut data_min_offset = usize::MAX;
    let mut data_max_end = 0usize;
    let mut data_ranges = Vec::with_capacity(locations.len());
    for (needed, location) in subtree.data_needed.entries.iter().zip(&locations) {
        let Some(needed) = needed else {
            data_ranges.push(None);
            continue;
        };

        let range = match needed {
            NeededBuffer::Full => 0..location.length,
            NeededBuffer::Range(range) => range.clone(),
        };
        if range.start > range.end || range.end > location.length {
            return Ok(None);
        }
        if range.is_empty() {
            data_ranges.push(None);
            continue;
        }

        let absolute = location.offset + range.start..location.offset + range.end;
        data_min_offset = data_min_offset.min(absolute.start);
        data_max_end = data_max_end.max(absolute.end);
        data_ranges.push(Some(absolute));
    }

    let full_data_size = locations
        .last()
        .map_or(0, |location| location.offset + location.length);
    // Preserve the original PR's I/O contract: every non-validity buffer is covered by one
    // contiguous segment request, even when the encoding has multiple children.
    let data_segment_range =
        (data_min_offset < data_max_end).then_some(data_min_offset..data_max_end);

    let mut data_read = data_segment_range.map(|segment_range| {
        let mut buffer_slices = Vec::new();
        for (buffer_idx, (absolute, location)) in data_ranges.iter().zip(&locations).enumerate() {
            if let Some(absolute) = absolute {
                buffer_slices.push(BufferSlice {
                    buffer_idx,
                    range: absolute.start - segment_range.start..absolute.end - segment_range.start,
                    alignment: location.alignment,
                });
            }
        }
        DataRead {
            segment: SegmentRead {
                segment_range,
                buffer_slices,
            },
        }
    });
    let mut validity_reads = Vec::new();
    let mut requested_bytes = data_read
        .as_ref()
        .map_or(0, |read| read.segment.segment_range.len());

    for (buffer_idx, (needed, location)) in subtree
        .validity_needed
        .entries
        .iter()
        .zip(&locations)
        .enumerate()
    {
        let Some(needed) = needed else {
            continue;
        };

        let range = match needed {
            NeededBuffer::Full => 0..location.length,
            NeededBuffer::Range(range) => range.clone(),
        };
        if range.start > range.end || range.end > location.length {
            return Ok(None);
        }
        if range.is_empty() {
            continue;
        }

        let absolute = location.offset + range.start..location.offset + range.end;
        if let Some(data_range) = &data_ranges[buffer_idx] {
            if !contains_range(data_range, &absolute) {
                return Ok(None);
            }
            continue;
        }

        if let Some(read) = data_read.as_mut()
            && contains_range(&read.segment.segment_range, &absolute)
        {
            read.segment.buffer_slices.push(BufferSlice {
                buffer_idx,
                range: absolute.start - read.segment.segment_range.start
                    ..absolute.end - read.segment.segment_range.start,
                alignment: location.alignment,
            });
            continue;
        }

        requested_bytes = match requested_bytes.checked_add(absolute.len()) {
            Some(requested_bytes) => requested_bytes,
            None => return Ok(None),
        };
        validity_reads.push(ValidityRead {
            segment: SegmentRead {
                segment_range: absolute.clone(),
                buffer_slices: vec![BufferSlice {
                    buffer_idx,
                    range: 0..absolute.len(),
                    alignment: location.alignment,
                }],
            },
        });
    }

    if !validity_reads.is_empty() && !session.get::<SeparateValidityReadsEnabled>().0 {
        return Ok(None);
    }

    if full_data_size > 0 && requested_bytes as f64 / full_data_size as f64 > RANGE_READ_THRESHOLD {
        return Ok(None);
    }

    let decoded_range = subtree.decoded_range;
    let post_slice = (decoded_range != row_range)
        .then(|| row_range.start - decoded_range.start..row_range.end - decoded_range.start);

    Ok(Some(RangeReadPlan {
        data_read,
        validity_reads,
        buffer_alignments: locations
            .iter()
            .map(|location| location.alignment)
            .collect(),
        decode_len: decoded_range.len(),
        post_slice,
    }))
}

pub(super) fn execute_range_read(
    plan: RangeReadPlan,
    array_tree: ByteBuffer,
    segment_id: SegmentId,
    segment_source: Arc<dyn SegmentSource>,
    dtype: DType,
    ctx: ReadContext,
    session: VortexSession,
) -> SharedArrayFuture {
    async move {
        let RangeReadPlan {
            data_read,
            validity_reads,
            buffer_alignments,
            decode_len,
            post_slice,
        } = plan;
        let reads = data_read
            .iter()
            .map(|read| &read.segment)
            .chain(validity_reads.iter().map(|read| &read.segment));
        let requests = reads
            .map(|read| {
                let future = segment_source.request_range(segment_id, read.segment_range.clone());
                async move {
                    let buffer = future.await?;
                    if buffer.len() != read.segment_range.len() {
                        return Err(vortex_err!(
                            "Range read returned {} bytes for requested range {:?}",
                            buffer.len(),
                            read.segment_range
                        ));
                    }

                    // Range requests are byte-addressed, so discard any stronger source-level
                    // alignment before taking individual buffer slices.
                    let buffer = buffer.ensure_aligned(Alignment::none())?;
                    read.buffer_slices
                        .iter()
                        .map(|buffer_slice| {
                            Ok((
                                buffer_slice.buffer_idx,
                                buffer
                                    .slice(buffer_slice.range.clone())
                                    .ensure_aligned(buffer_slice.alignment)?,
                            ))
                        })
                        .collect::<VortexResult<Vec<_>>>()
                }
            })
            .collect::<Vec<_>>();
        let outputs = try_join_all(requests).await?;

        let mut buffers = std::iter::repeat_with(|| None)
            .take(buffer_alignments.len())
            .collect::<Vec<Option<BufferHandle>>>();
        for output in outputs {
            for (buffer_idx, buffer) in output {
                let Some(slot) = buffers.get_mut(buffer_idx) else {
                    return Err(vortex_err!(
                        "Range-read plan references missing buffer {}",
                        buffer_idx
                    ));
                };
                if slot.replace(buffer).is_some() {
                    return Err(vortex_err!(
                        "Range-read plan provides buffer {} more than once",
                        buffer_idx
                    ));
                }
            }
        }
        let buffers = buffers
            .into_iter()
            .zip(buffer_alignments)
            .map(|(buffer, alignment)| {
                buffer
                    .unwrap_or_else(|| BufferHandle::new_host(ByteBuffer::empty_aligned(alignment)))
            })
            .collect();

        let serialized = SerializedArray::from_flatbuffer_with_buffers(array_tree, buffers)?;
        let mut array = serialized.decode_partial(&dtype, decode_len, &ctx, &session)?;
        if let Some(post_slice) = post_slice {
            array = array.slice(post_slice)?;
        }
        Ok(array)
    }
    .map(|result| result.map_err(Arc::new))
    .boxed()
    .shared()
}

#[cfg(test)]
mod tests;
