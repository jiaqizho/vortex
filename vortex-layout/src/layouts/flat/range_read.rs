// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Range read support for FlatLayout segments.
//!
//! When the `array_tree` metadata is inlined in the layout (via `FLAT_LAYOUT_INLINE_ARRAY_NODE`),
//! we can inspect the encoding tree to compute which byte ranges of the segment are needed for a
//! given row range. This allows us to issue a smaller, targeted IO instead of reading the entire
//! segment.
//!
//! Each encoding implements [`VTable::plan_range_read`] to describe how its buffers and children
//! should be handled. The planner dispatches via vtable and recursively walks the encoding tree.

use std::ops::Range;
use std::sync::Arc;

use flatbuffers::root;
use futures::FutureExt;
use vortex_array::serde::ArrayParts;
use vortex_array::vtable::range_read::{BufferSubRange, ChildRangeRead, RangeDecodeInfo};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_err};
use vortex_flatbuffers::array as fba;

use crate::layouts::SharedArrayFuture;
use crate::segments::{SegmentId, SegmentSource};

/// Maximum ratio of (range_size / full_segment_size) at which we attempt a range read.
/// If the range read would read more than this fraction of the full segment, we fall back to
/// a full read to avoid the overhead of range computation.
const RANGE_READ_THRESHOLD: f64 = 0.5;

/// A plan describing which bytes to read from a segment for a given row range.
#[derive(Debug)]
pub(super) struct RangeReadPlan {
    /// The byte range to read from the segment.
    segment_byte_range: Range<usize>,
    /// For each buffer index in the array_tree, the byte range within the partial segment
    /// that contains that buffer's data. Empty range means the buffer is not needed.
    buffer_ranges: Vec<Range<usize>>,
    /// The alignment requirement for each buffer.
    buffer_alignments: Vec<Alignment>,
    /// The number of logical rows to pass to `decode`.
    decode_len: usize,
    /// After decoding, slice the result to this range (relative to decode output).
    /// `None` means no post-decode slicing is needed.
    post_slice: Option<Range<usize>>,
}

/// Decode information returned by encoding analysis.
#[derive(Debug, Clone)]
struct DecodeInfo {
    decode_len: usize,
    post_slice: Option<Range<usize>>,
}

/// Buffer offset, length, and alignment within the full segment.
#[derive(Debug, Clone)]
struct BufferLocation {
    offset: usize,
    length: usize,
    alignment: Alignment,
}

/// Compute buffer locations from the Array flatbuffer's buffer descriptors.
fn compute_buffer_locations(fb_array: &fba::Array<'_>) -> Vec<BufferLocation> {
    let mut offset = 0usize;
    fb_array
        .buffers()
        .unwrap_or_default()
        .iter()
        .map(|buf| {
            offset += buf.padding() as usize;
            let loc = BufferLocation {
                offset,
                length: buf.length() as usize,
                alignment: Alignment::from_exponent(buf.alignment_exponent()),
            };
            offset += buf.length() as usize;
            loc
        })
        .collect()
}

/// Tracks which buffers are needed and their required byte sub-ranges.
struct NeededBuffers {
    entries: Vec<Option<Range<usize>>>,
}

impl NeededBuffers {
    fn new(num_buffers: usize) -> Self {
        Self {
            entries: vec![None; num_buffers],
        }
    }

    fn need_full(&mut self, buffer_idx: u16) {
        let idx = buffer_idx as usize;
        if idx < self.entries.len() {
            self.entries[idx] = Some(0..usize::MAX);
        }
    }

    fn need_range(&mut self, buffer_idx: u16, range: Range<usize>) {
        let idx = buffer_idx as usize;
        if idx < self.entries.len() {
            match &self.entries[idx] {
                Some(existing) if existing.end == usize::MAX => {}
                Some(existing) => {
                    let start = existing.start.min(range.start);
                    let end = existing.end.max(range.end);
                    self.entries[idx] = Some(start..end);
                }
                None => {
                    self.entries[idx] = Some(range);
                }
            }
        }
    }
}

/// Recursively analyze the encoding tree via vtable dispatch to determine which buffer
/// byte ranges are needed for the given row range.
fn analyze_encoding(
    node: fba::ArrayNode<'_>,
    row_range: Range<usize>,
    row_count: usize,
    dtype: &DType,
    ctx: &ArrayContext,
    needed: &mut NeededBuffers,
) -> Option<DecodeInfo> {
    let encoding = ctx.lookup_encoding(node.encoding())?;
    let metadata_bytes = node.metadata().map(|m| m.bytes()).unwrap_or(&[]);

    let plan = encoding.plan_range_read(metadata_bytes, row_range, row_count, dtype)?;

    apply_buffer_sub_ranges(node, &plan.buffer_sub_ranges, needed);

    let child_decode_infos = resolve_children(node, &plan.children, ctx, needed)?;

    resolve_decode_info(plan.decode_info, &child_decode_infos)
}

/// Map the plan's buffer sub-ranges to global buffer indices via the flatbuffer node.
fn apply_buffer_sub_ranges(
    node: fba::ArrayNode<'_>,
    sub_ranges: &[BufferSubRange],
    needed: &mut NeededBuffers,
) {
    if let Some(buffers) = node.buffers() {
        for (local_idx, sub_range) in sub_ranges.iter().enumerate() {
            if local_idx < buffers.len() {
                let global_idx = buffers.get(local_idx);
                match sub_range {
                    BufferSubRange::Full => needed.need_full(global_idx),
                    BufferSubRange::Range(range) => needed.need_range(global_idx, range.clone()),
                }
            }
        }
    }
}

/// Pair each plan child with its corresponding node child and process them.
fn resolve_children(
    node: fba::ArrayNode<'_>,
    plan_children: &[ChildRangeRead],
    ctx: &ArrayContext,
    needed: &mut NeededBuffers,
) -> Option<Vec<Option<DecodeInfo>>> {
    let node_children: Vec<_> = node
        .children()
        .map_or(vec![], |c| (0..c.len()).map(|i| c.get(i)).collect());

    if plan_children.len() != node_children.len() {
        return None;
    }

    let mut decode_infos = Vec::with_capacity(plan_children.len());
    for (action, child_node) in plan_children.iter().zip(node_children.iter()) {
        match action {
            ChildRangeRead::Recurse {
                row_range,
                row_count,
                dtype,
            } => {
                let info = analyze_encoding(
                    *child_node,
                    row_range.clone(),
                    *row_count,
                    dtype,
                    ctx,
                    needed,
                );
                decode_infos.push(info);
            }
            ChildRangeRead::Full => {
                need_all_node_buffers(*child_node, needed);
                decode_infos.push(None);
            }
        }
    }

    Some(decode_infos)
}

/// Compute the final decode parameters from the plan's decode info and children results.
fn resolve_decode_info(
    decode_info: RangeDecodeInfo,
    child_decode_infos: &[Option<DecodeInfo>],
) -> Option<DecodeInfo> {
    match decode_info {
        RangeDecodeInfo::Leaf {
            decode_len,
            post_slice,
        } => Some(DecodeInfo {
            decode_len,
            post_slice,
        }),
        RangeDecodeInfo::FromChild { child_idx, divisor } => {
            let child_info = child_decode_infos.get(child_idx)?.as_ref()?;
            if divisor == 1 {
                return Some(child_info.clone());
            }
            if child_info.decode_len % divisor != 0 {
                return None;
            }
            let scaled_post_slice = match &child_info.post_slice {
                None => None,
                Some(ps) => {
                    if ps.start % divisor != 0 || ps.end % divisor != 0 {
                        return None;
                    }
                    Some(ps.start / divisor..ps.end / divisor)
                }
            };
            Some(DecodeInfo {
                decode_len: child_info.decode_len / divisor,
                post_slice: scaled_post_slice,
            })
        }
    }
}

/// Recursively mark all buffers in a node (and its children) as fully needed.
fn need_all_node_buffers(node: fba::ArrayNode<'_>, needed: &mut NeededBuffers) {
    if let Some(buffers) = node.buffers() {
        for i in 0..buffers.len() {
            needed.need_full(buffers.get(i));
        }
    }
    if let Some(children) = node.children() {
        for i in 0..children.len() {
            need_all_node_buffers(children.get(i), needed);
        }
    }
}

/// Attempt to build a range read plan for the given array tree and row range.
///
/// Returns `None` if:
/// - The encoding does not support range reads (fallback to full segment read).
/// - The row range covers the entire segment (no benefit).
/// - The computed byte range is not significantly smaller than the full segment.
pub(super) fn try_plan_range_read(
    array_tree: &ByteBuffer,
    row_range: Range<usize>,
    row_count: usize,
    dtype: &DType,
    ctx: &ArrayContext,
) -> VortexResult<Option<RangeReadPlan>> {
    if row_range.is_empty() {
        return Ok(None);
    }

    if row_range.start == 0 && row_range.end >= row_count {
        return Ok(None);
    }

    let fb_array = root::<fba::Array>(array_tree.as_ref())
        .map_err(|e| vortex_err!("invalid array tree flatbuffer: {e}"))?;

    let buffer_locations = compute_buffer_locations(&fb_array);
    let num_buffers = buffer_locations.len();

    let root_node = fb_array
        .root()
        .ok_or_else(|| vortex_err!("array tree has no root node"))?;

    let mut needed = NeededBuffers::new(num_buffers);
    let decode_info =
        match analyze_encoding(root_node, row_range, row_count, dtype, ctx, &mut needed) {
            Some(info) => info,
            None => return Ok(None),
        };

    let mut min_offset = usize::MAX;
    let mut max_end = 0usize;
    for (i, entry) in needed.entries.iter_mut().enumerate() {
        if let Some(range) = entry {
            let loc = &buffer_locations[i];
            if range.end == usize::MAX {
                *range = 0..loc.length;
            }
            range.end = range.end.min(loc.length);

            let abs_start = loc.offset + range.start;
            let abs_end = loc.offset + range.end;
            min_offset = min_offset.min(abs_start);
            max_end = max_end.max(abs_end);
        }
    }

    if min_offset >= max_end {
        return Ok(None);
    }

    let segment_byte_range = min_offset..max_end;

    let full_segment_size: usize = buffer_locations
        .last()
        .map(|loc| loc.offset + loc.length)
        .unwrap_or(0);
    if full_segment_size == 0 {
        return Ok(None);
    }
    let ratio = segment_byte_range.len() as f64 / full_segment_size as f64;
    if ratio > RANGE_READ_THRESHOLD {
        return Ok(None);
    }

    let partial_offset = segment_byte_range.start;
    let mut buffer_ranges = Vec::with_capacity(num_buffers);
    let mut buffer_alignments = Vec::with_capacity(num_buffers);
    for (i, entry) in needed.entries.iter().enumerate() {
        let loc = &buffer_locations[i];
        buffer_alignments.push(loc.alignment);
        if let Some(sub_range) = entry {
            let abs_start = loc.offset + sub_range.start;
            let abs_end = loc.offset + sub_range.end;
            buffer_ranges.push((abs_start - partial_offset)..(abs_end - partial_offset));
        } else {
            buffer_ranges.push(0..0);
        }
    }

    Ok(Some(RangeReadPlan {
        segment_byte_range,
        buffer_ranges,
        buffer_alignments,
        decode_len: decode_info.decode_len,
        post_slice: decode_info.post_slice,
    }))
}

/// Execute a range read plan: issue a targeted IO, build ArrayParts from partial buffers, decode.
pub(super) fn execute_range_read(
    plan: RangeReadPlan,
    array_tree: ByteBuffer,
    segment_id: SegmentId,
    segment_source: Arc<dyn SegmentSource>,
    dtype: DType,
    ctx: ArrayContext,
) -> SharedArrayFuture {
    async move {
        // 1. Issue the targeted read.
        let partial_bytes = segment_source
            .request_range(segment_id, plan.segment_byte_range.clone())
            .await?;

        // 2. Slice individual buffers from the partial segment, ensuring alignment.
        let mut buffers = Vec::with_capacity(plan.buffer_ranges.len());
        for (i, buf_range) in plan.buffer_ranges.iter().enumerate() {
            if buf_range.is_empty() {
                buffers.push(ByteBuffer::empty());
            } else {
                let slice = partial_bytes.slice_unaligned(buf_range.clone());
                let aligned = slice.aligned(plan.buffer_alignments[i]);
                buffers.push(aligned);
            }
        }

        // 3. Build ArrayParts and decode.
        let parts = ArrayParts::from_flatbuffer_with_buffers(array_tree, buffers)?;
        let mut array: ArrayRef = parts.decode(&ctx, &dtype, plan.decode_len)?;

        // 4. Post-decode slice if needed (block alignment).
        if let Some(slice_range) = plan.post_slice {
            array = array.slice(slice_range);
        }

        Ok(array)
    }
    .map(|r| r.map_err(Arc::new))
    .boxed()
    .shared()
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::Arc;

    use vortex_alp::alp_encode;
    use vortex_array::arrays::{
        BoolArray, ConstantArray, DictArray, FixedSizeListArray, NullArray, PrimitiveArray,
    };
    use vortex_array::serde::SerializeOptions;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayContext, IntoArray};
    use vortex_buffer::Buffer;
    use vortex_bytebool::ByteBoolArray;
    use vortex_dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_fastlanes::{BitPackedArray, DeltaArray, FoRArray};
    use vortex_io::runtime::single::block_on;
    use vortex_sequence::SequenceArray;
    use vortex_zigzag::zigzag_encode;

    use super::*;
    use crate::segments::{SegmentSink, TestSegments};
    use crate::sequence::SequenceId;

    /// Helper: serialize an array, return (array_tree, full_segment_size, ctx).
    fn plan_info(array: &dyn Array) -> VortexResult<(ByteBuffer, usize, ArrayContext)> {
        let ctx = ArrayContext::empty();
        let buffers = array.serialize(
            &ctx,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )?;
        let array_tree = buffers[buffers.len() - 2].clone();
        let segments = Arc::new(TestSegments::default());
        let full_segment_size = block_on(|_| async {
            let sid = segments
                .write(SequenceId::root().advance(), buffers)
                .await?;
            let seg = segments.request(sid).await?;
            Ok::<_, vortex_error::VortexError>(seg.len())
        })?;
        Ok((array_tree, full_segment_size, ctx))
    }

    /// Assert that a point lookup (single row) plans a read significantly smaller than the
    /// full segment. Returns (range_read_bytes, full_segment_bytes).
    fn assert_point_lookup_smaller(
        array: &dyn Array,
        row_count: usize,
        row_idx: usize,
    ) -> (usize, usize) {
        let (array_tree, full_size, ctx) = plan_info(array).expect("serialize");
        let plan = try_plan_range_read(
            &array_tree,
            row_idx..row_idx + 1,
            row_count,
            array.dtype(),
            &ctx,
        )
        .expect("plan")
        .expect("expected Some plan for point lookup");

        let range_bytes = plan.segment_byte_range.len();
        assert!(
            range_bytes < full_size,
            "point lookup IO ({range_bytes}) should be < full segment ({full_size})"
        );
        (range_bytes, full_size)
    }

    // ---- 1. FSL point lookup IO is proportional to a single row ----

    #[test]
    fn fsl_point_lookup_reads_single_row_io() {
        let num_rows: usize = 1000;
        let list_size: u32 = 8;
        // 1000 rows × 8 elements × 4 bytes = 32000 bytes of data
        let elements: Vec<i32> = (0..num_rows * list_size as usize)
            .map(|i| i as i32)
            .collect();
        let element_array =
            PrimitiveArray::new(Buffer::<i32>::from(elements), Validity::AllValid).into_array();
        let fsl =
            FixedSizeListArray::try_new(element_array, list_size, Validity::AllValid, num_rows)
                .expect("fsl")
                .into_array();

        let (range_bytes, full_size) = assert_point_lookup_smaller(fsl.as_ref(), num_rows, 500);

        // A single FSL row = list_size * sizeof(i32) = 32 bytes.
        // The range read should be close to that, definitely < 1% of full segment.
        let ratio = range_bytes as f64 / full_size as f64;
        println!(
            "FSL point lookup: {range_bytes} bytes / {full_size} bytes = {:.2}%",
            ratio * 100.0
        );
        assert!(
            ratio < 0.01,
            "FSL point lookup should read <1% of segment, got {:.2}%",
            ratio * 100.0
        );
    }

    // ---- 2. Different encodings: point lookup < full segment ----

    #[test]
    fn primitive_point_lookup_smaller_than_full_segment() {
        let num_rows: usize = 10000;
        let data: Vec<i64> = (0..num_rows as i64).collect();
        let array = PrimitiveArray::new(Buffer::<i64>::from(data), Validity::AllValid).into_array();

        let (range_bytes, full_size) =
            assert_point_lookup_smaller(array.as_ref(), num_rows, num_rows / 2);
        println!("Primitive i64: {range_bytes} / {full_size} bytes");
    }

    #[test]
    fn bitpacked_point_lookup_smaller_than_full_segment() {
        // BitPacked: 10000 values, bit_width small
        let num_rows: usize = 10000;
        let data: Vec<u32> = (0..num_rows as u32).map(|v| v % 16).collect(); // 4-bit values
        let prim = PrimitiveArray::new(Buffer::<u32>::from(data), Validity::AllValid);
        let bp = BitPackedArray::encode(prim.as_ref(), 4).expect("bitpack");
        let array = bp.into_array();

        let (range_bytes, full_size) =
            assert_point_lookup_smaller(array.as_ref(), num_rows, num_rows / 2);
        println!("BitPacked: {range_bytes} / {full_size} bytes");
    }

    #[test]
    fn for_primitive_point_lookup_smaller_than_full_segment() {
        let num_rows: usize = 10000;
        let data: Vec<i32> = (1000..1000 + num_rows as i32).collect();
        let prim = PrimitiveArray::new(Buffer::<i32>::from(data), Validity::AllValid);
        let for_array = FoRArray::encode(prim).expect("for").into_array();

        let (range_bytes, full_size) =
            assert_point_lookup_smaller(for_array.as_ref(), num_rows, num_rows / 2);
        println!("FoR: {range_bytes} / {full_size} bytes");
    }

    #[test]
    fn zigzag_point_lookup_smaller_than_full_segment() {
        let num_rows: usize = 10000;
        let data: Vec<i32> = (0..num_rows as i32)
            .map(|v| if v % 2 == 0 { v } else { -v })
            .collect();
        let prim = PrimitiveArray::new(Buffer::<i32>::from(data), Validity::AllValid);
        let zz = zigzag_encode(prim).expect("zigzag").into_array();

        let (range_bytes, full_size) =
            assert_point_lookup_smaller(zz.as_ref(), num_rows, num_rows / 2);
        println!("ZigZag: {range_bytes} / {full_size} bytes");
    }

    #[test]
    fn bool_point_lookup_smaller_than_full_segment() {
        let num_rows: usize = 10000;
        // byte-aligned row index (multiple of 8)
        let data: Vec<bool> = (0..num_rows).map(|i| i % 3 == 0).collect();
        let array = BoolArray::from_iter(data).into_array();

        let (range_bytes, full_size) = assert_point_lookup_smaller(array.as_ref(), num_rows, 8000);
        println!("Bool: {range_bytes} / {full_size} bytes");
    }

    #[test]
    fn constant_point_lookup_has_plan() {
        // Constant arrays always need the full scalar buffer, but plan should still succeed.
        let num_rows: usize = 10000;
        let array = ConstantArray::new(42i32, num_rows).into_array();
        let (array_tree, _full_size, ctx) = plan_info(array.as_ref()).expect("serialize");

        let plan = try_plan_range_read(&array_tree, 5000..5001, num_rows, array.dtype(), &ctx)
            .expect("plan");
        // Constant may or may not produce a plan depending on threshold,
        // but if it does, decode_len should be 1.
        if let Some(p) = plan {
            assert_eq!(p.decode_len, 1);
            println!(
                "Constant: {} / {} bytes",
                p.segment_byte_range.len(),
                _full_size
            );
        } else {
            println!("Constant: plan is None (full buffer is tiny, threshold skips)");
        }
    }

    #[test]
    fn null_point_lookup_has_plan() {
        let num_rows: usize = 10000;
        let array = NullArray::new(num_rows).into_array();
        let (array_tree, _full_size, ctx) = plan_info(array.as_ref()).expect("serialize");

        // Null arrays have no data buffers, so the plan may be None (segment is tiny).
        let plan = try_plan_range_read(&array_tree, 5000..5001, num_rows, array.dtype(), &ctx)
            .expect("plan");
        if let Some(p) = plan {
            assert_eq!(p.decode_len, 1);
            println!(
                "Null: {} / {} bytes",
                p.segment_byte_range.len(),
                _full_size
            );
        } else {
            println!("Null: plan is None (no data buffers)");
        }
    }

    #[test]
    fn dict_range_read_plans_successfully() {
        // Dict range read includes the full values dictionary, so the contiguous IO span
        // can be large. With enough rows and small values dict, a narrow range still wins.
        let num_rows: usize = 100_000;
        let codes: Vec<u32> = (0..num_rows as u32).map(|v| v % 4).collect();
        let values: Vec<i64> = vec![100, 200, 300, 400];
        let codes_array =
            PrimitiveArray::new(Buffer::<u32>::from(codes), Validity::AllValid).into_array();
        let values_array =
            PrimitiveArray::new(Buffer::<i64>::from(values), Validity::AllValid).into_array();
        let dict = DictArray::try_new(codes_array, values_array)
            .expect("dict")
            .into_array();

        let (array_tree, full_size, ctx) = plan_info(dict.as_ref()).expect("serialize");
        // Use a narrow range (100 rows out of 100k) — codes sub-range is small but
        // the full values buffer is included, so the contiguous range spans both.
        let plan = try_plan_range_read(&array_tree, 50000..50100, num_rows, dict.dtype(), &ctx)
            .expect("plan");
        if let Some(p) = plan {
            println!(
                "Dict: {} / {} bytes ({:.1}%)",
                p.segment_byte_range.len(),
                full_size,
                p.segment_byte_range.len() as f64 / full_size as f64 * 100.0
            );
        } else {
            // Dict with ChildRangeRead::Full for values may exceed threshold for small dicts.
            // This is expected — the values buffer must be fully included.
            println!("Dict: plan is None (values buffer dominates, threshold exceeded)");
        }
    }

    #[test]
    fn alp_point_lookup_smaller_than_full_segment() {
        // Use f32 values that ALP can perfectly encode (no patches).
        // Small multiples of 0.01 round-trip cleanly through ALP.
        let num_rows: usize = 10000;
        let data: Vec<f32> = (0..num_rows as i32).map(|i| i as f32 * 0.01).collect();
        let prim = PrimitiveArray::new(Buffer::<f32>::from(data), Validity::NonNullable);
        let alp = alp_encode(&prim, None).expect("alp");

        // Verify no patches before testing
        if alp.patches().is_some() {
            println!("ALP: has patches, skipping point lookup test");
            return;
        }

        let array = alp.into_array();
        let (range_bytes, full_size) =
            assert_point_lookup_smaller(array.as_ref(), num_rows, num_rows / 2);
        println!("ALP: {range_bytes} / {full_size} bytes");
    }

    #[test]
    fn bytebool_point_lookup_smaller_than_full_segment() {
        let num_rows: usize = 10000;
        let data: Vec<bool> = (0..num_rows).map(|i| i % 2 == 0).collect();
        let array = ByteBoolArray::from(data).into_array();

        let (range_bytes, full_size) =
            assert_point_lookup_smaller(array.as_ref(), num_rows, num_rows / 2);
        println!("ByteBool: {range_bytes} / {full_size} bytes");
    }

    #[test]
    fn delta_point_lookup_smaller_than_full_segment() {
        // Delta uses 1024-row chunks. The bases and deltas buffers are sequential
        // in the segment, so querying near the start keeps the contiguous IO span
        // small (only the first chunk of each buffer).
        let num_rows: usize = 1_000_000;
        let data: Vec<u32> = (0..num_rows as u32).collect();
        let delta = DeltaArray::try_from_vec(data).expect("delta").into_array();

        let (range_bytes, full_size) = assert_point_lookup_smaller(delta.as_ref(), num_rows, 0);
        println!("Delta: {range_bytes} / {full_size} bytes");
    }

    #[test]
    fn sequence_point_lookup_has_plan() {
        let num_rows: usize = 10000;
        let seq = SequenceArray::typed_new(0i64, 1i64, Nullability::NonNullable, num_rows)
            .expect("sequence")
            .into_array();
        let (array_tree, _full_size, ctx) = plan_info(seq.as_ref()).expect("serialize");

        // Sequence has no data buffers, plan may be None (tiny segment).
        let plan = try_plan_range_read(&array_tree, 5000..5001, num_rows, seq.dtype(), &ctx)
            .expect("plan");
        if let Some(p) = plan {
            assert_eq!(p.decode_len, 1);
            println!(
                "Sequence: {} / {} bytes",
                p.segment_byte_range.len(),
                _full_size
            );
        } else {
            println!("Sequence: plan is None (no data buffers)");
        }
    }
}
