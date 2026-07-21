// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use parking_lot::Mutex;
use vortex_alp::ALP;
use vortex_alp::ALPRD;
use vortex_alp::ALPRDArrayExt;
use vortex_alp::Exponents;
use vortex_alp::RDEncoder;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::root;
use vortex_array::expr::stats::Stat;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::serde::SerializeOptions;
use vortex_array::session::ArraySession;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_bytebool::ByteBool;
use vortex_datetime_parts::DateTimeParts;
use vortex_datetime_parts::split_temporal;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::delta_compress;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_sequence::Sequence;
use vortex_session::SessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;
use vortex_zigzag::ZigZag;

use super::super::FlatLayout;
use super::super::reader::FlatReader;
use crate::LayoutReader;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSink;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::session::LayoutSession;
use crate::session::RangeReadEnabled;
use crate::session::SeparateValidityReadsEnabled;

#[derive(Clone)]
struct TrackingSegments {
    inner: Arc<TestSegments>,
    range_requests: Arc<Mutex<Vec<Range<usize>>>>,
    full_requests: Arc<AtomicUsize>,
}

impl TrackingSegments {
    fn new(inner: Arc<TestSegments>) -> Self {
        Self {
            inner,
            range_requests: Arc::new(Mutex::new(Vec::new())),
            full_requests: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn ranges(&self) -> Vec<Range<usize>> {
        self.range_requests.lock().clone()
    }

    fn full_requests(&self) -> usize {
        self.full_requests.load(Ordering::Relaxed)
    }
}

impl SegmentSource for TrackingSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.full_requests.fetch_add(1, Ordering::Relaxed);
        self.inner.request(id)
    }

    fn request_range(&self, id: SegmentId, range: Range<usize>) -> SegmentFuture {
        self.range_requests.lock().push(range.clone());
        self.inner.request_range(id, range).boxed()
    }
}

struct WrittenArray {
    array_tree: vortex_buffer::ByteBuffer,
    segment_id: SegmentId,
    segments: Arc<TestSegments>,
    read_ctx: ReadContext,
    full_size: usize,
}

fn test_session() -> VortexSession {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>();
    session.arrays().register(ByteBool);
    session.arrays().register(ALP);
    session.arrays().register(ALPRD);
    session.arrays().register(DateTimeParts);
    session.arrays().register(DecimalByteParts);
    session.arrays().register(BitPacked);
    session.arrays().register(Delta);
    session.arrays().register(FoR);
    session.arrays().register(Sequence);
    session.arrays().register(ZigZag);
    session
}

fn test_session_with_separate_validity_reads() -> VortexSession {
    let session = test_session();
    session.get_mut::<SeparateValidityReadsEnabled>().0 = true;
    session
}

fn write_array(array: &ArrayRef, session: &VortexSession) -> VortexResult<WrittenArray> {
    let ctx = ArrayContext::empty();
    let buffers = array.serialize(
        &ctx,
        session,
        &SerializeOptions {
            offset: 0,
            include_padding: true,
        },
    )?;
    let array_tree = buffers[buffers.len() - 2].clone();
    let full_size = buffers.iter().map(|buffer| buffer.len()).sum();
    let segments = Arc::new(TestSegments::default());
    let segment_id =
        block_on(|_| async { segments.write(SequenceId::root().advance(), buffers).await })?;

    Ok(WrittenArray {
        array_tree,
        segment_id,
        segments,
        read_ctx: ReadContext::new(ctx.to_ids()),
        full_size,
    })
}

fn read_range(
    array: &ArrayRef,
    row_range: Range<usize>,
    session: VortexSession,
    inline_array_tree: bool,
) -> VortexResult<(ArrayRef, TrackingSegments, usize)> {
    let written = write_array(array, &session)?;
    let source = TrackingSegments::new(written.segments);
    let layout = FlatLayout::new_with_metadata(
        array.len() as u64,
        array.dtype().clone(),
        written.segment_id,
        written.read_ctx,
        inline_array_tree.then_some(written.array_tree),
    );
    let reader = FlatReader::new(
        layout,
        "range-read-test".into(),
        Arc::new(source.clone()),
        session,
    );
    let result = block_on(|_| async {
        reader
            .projection_evaluation(
                &(row_range.start as u64..row_range.end as u64),
                &root(),
                MaskFuture::new_true(row_range.len()),
            )?
            .await
    })?;
    Ok((result, source, written.full_size))
}

fn assert_range_read(
    array: &ArrayRef,
    row_range: Range<usize>,
    session: VortexSession,
) -> VortexResult<TrackingSegments> {
    let expected = array.slice(row_range.clone())?;
    let (actual, source, full_size) = read_range(array, row_range, session, true)?;
    assert_arrays_eq!(actual, expected);
    assert_eq!(source.full_requests(), 0, "unexpected full-segment read");
    let requested_bytes: usize = source.ranges().iter().map(Range::len).sum();
    assert!(requested_bytes < full_size);
    Ok(source)
}

#[test]
fn non_nullable_primitive_uses_one_contiguous_request() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter(0..4096i32).into_array();
    let source = assert_range_read(&array, 100..101, test_session())?;
    let ranges = source.ranges();
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].len(), size_of::<i32>());
    Ok(())
}

#[test]
fn range_read_does_not_restore_full_array_statistics() -> VortexResult<()> {
    let row_range = 104usize..112;
    let selected_values = 104i32..112;
    let array = PrimitiveArray::from_option_iter(
        (0i32..4096)
            .map(|value| (value % 2 == 0 || selected_values.contains(&value)).then_some(value)),
    )
    .into_array();
    let session = test_session_with_separate_validity_reads();
    let mut ctx = session.create_execution_ctx();
    array.statistics().compute_all(
        &[Stat::Min, Stat::Max, Stat::Sum, Stat::NullCount],
        &mut ctx,
    )?;

    let (actual, source, _) = read_range(&array, row_range, session.clone(), true)?;
    assert_eq!(source.full_requests(), 0);

    let mut ctx = session.create_execution_ctx();
    assert_eq!(actual.statistics().compute_min::<i32>(&mut ctx), Some(104));
    assert_eq!(actual.statistics().compute_max::<i32>(&mut ctx), Some(111));
    assert_eq!(
        actual.statistics().compute_as::<i64>(Stat::Sum, &mut ctx),
        Some(860)
    );
    assert_eq!(actual.statistics().compute_null_count(&mut ctx), Some(0));
    Ok(())
}

#[test]
fn range_read_does_not_restore_child_statistics() -> VortexResult<()> {
    let row_range = 104usize..112;
    let encoded = PrimitiveArray::from_iter(0..4096i32).into_array();
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    encoded
        .statistics()
        .compute_all(&[Stat::Min, Stat::Max, Stat::Sum], &mut ctx)?;
    let array = FoR::try_new(encoded, 1000i32.into())?.into_array();

    let (actual, source, _) = read_range(&array, row_range, session.clone(), true)?;
    assert_eq!(source.full_requests(), 0);

    let actual = actual.as_::<FoR>();
    let encoded = actual.encoded();
    let mut ctx = session.create_execution_ctx();
    assert_eq!(encoded.statistics().compute_min::<i32>(&mut ctx), Some(104));
    assert_eq!(encoded.statistics().compute_max::<i32>(&mut ctx), Some(111));
    assert_eq!(
        encoded.statistics().compute_as::<i64>(Stat::Sum, &mut ctx),
        Some(860)
    );
    Ok(())
}

#[test]
fn range_read_dict_min_max_ignores_unreferenced_values() -> VortexResult<()> {
    let session = test_session();
    let codes = PrimitiveArray::from_iter((0..4096).map(|idx| u8::from(idx >= 2048)));
    let values = PrimitiveArray::from_iter([10i32, 1000i32]);
    let array = unsafe {
        DictArray::try_new(codes.into_array(), values.into_array())?
            .set_all_values_referenced(true)
            .into_array()
    };

    let (actual, source, _) = read_range(&array, 4000..4008, session.clone(), true)?;
    assert_eq!(source.full_requests(), 0);

    let mut ctx = session.create_execution_ctx();
    assert_eq!(actual.statistics().compute_min::<i32>(&mut ctx), Some(1000));
    assert_eq!(actual.statistics().compute_max::<i32>(&mut ctx), Some(1000));
    Ok(())
}

#[test]
fn range_read_preserves_alignment_for_empty_dict_values() -> VortexResult<()> {
    let codes = PrimitiveArray::from_option_iter((0..4096).map(|_| None::<u32>));
    let values = PrimitiveArray::from_iter(std::iter::empty::<i32>());
    let array = DictArray::try_new(codes.into_array(), values.into_array())?.into_array();

    assert_range_read(
        &array,
        100..108,
        test_session_with_separate_validity_reads(),
    )?;
    Ok(())
}

#[test]
fn non_nullable_decimal_uses_serialized_value_width() -> VortexResult<()> {
    let array = DecimalArray::new(
        Buffer::from_iter((0..4096).map(|idx| i128::from(idx % 100))),
        DecimalDType::new(2, 0),
        Validity::NonNullable,
    )
    .into_array();

    let source = assert_range_read(&array, 100..101, test_session())?;
    assert_eq!(
        source.ranges(),
        vec![100 * size_of::<i128>()..101 * size_of::<i128>()]
    );
    Ok(())
}

#[test]
fn nullable_decimal_reads_values_and_validity_ranges() -> VortexResult<()> {
    let row_count = 4096;
    let array = DecimalArray::new(
        Buffer::from_iter(0i64..4096),
        DecimalDType::new(12, 2),
        Validity::from_iter((0..row_count).map(|idx| idx % 3 != 0)),
    )
    .into_array();

    let source = assert_range_read(&array, 5..13, test_session_with_separate_validity_reads())?;
    let values_end = 13 * size_of::<i64>();
    let validity_offset = row_count * size_of::<i64>();
    assert_eq!(
        source.ranges(),
        vec![0..values_end, validity_offset..validity_offset + 2]
    );
    Ok(())
}

#[test]
fn nullable_primitive_reads_values_and_validity_ranges() -> VortexResult<()> {
    let array =
        PrimitiveArray::from_option_iter((0..4096).map(|value| (value % 3 != 0).then_some(value)))
            .into_array();

    let source = assert_range_read(&array, 5..13, test_session_with_separate_validity_reads())?;
    let values_end = 13 * size_of::<i32>();
    let validity_offset = 4096 * size_of::<i32>();
    assert_eq!(
        source.ranges(),
        vec![0..values_end, validity_offset..validity_offset + 2]
    );
    Ok(())
}

#[test]
fn separate_validity_reads_are_disabled_by_default() -> VortexResult<()> {
    let array =
        PrimitiveArray::from_option_iter((0..4096).map(|value| (value % 3 != 0).then_some(value)))
            .into_array();
    let expected = array.slice(5..13)?;

    let (actual, source, _) = read_range(&array, 5..13, test_session(), true)?;

    assert_arrays_eq!(actual, expected);
    assert_eq!(source.full_requests(), 1);
    assert!(source.ranges().is_empty());
    Ok(())
}

#[test]
fn all_valid_nullable_primitive_still_uses_range_io_by_default() -> VortexResult<()> {
    let array = PrimitiveArray::new(Buffer::from_iter(0..4096i32), Validity::AllValid).into_array();
    assert!(array.dtype().is_nullable());

    let source = assert_range_read(&array, 100..101, test_session())?;
    assert_eq!(source.ranges().len(), 1);
    Ok(())
}

#[test]
fn bool_supports_unaligned_rows_and_nonzero_metadata_offset() -> VortexResult<()> {
    let array = BoolArray::from_iter((0..9000).map(|idx| idx % 3 == 0))
        .into_array()
        .slice(3..8990)?;

    let source = assert_range_read(&array, 101..109, test_session())?;
    assert_eq!(source.ranges().len(), 1);
    Ok(())
}

#[test]
fn nullable_fixed_size_list_uses_range_io() -> VortexResult<()> {
    let row_count = 4096;
    let list_size = 4u32;
    let list_size_usize = usize::try_from(list_size)?;
    let element_values = (0u32..16)
        .cycle()
        .take(row_count * list_size_usize)
        .collect::<Vec<_>>();
    let elements = PrimitiveArray::from_iter(element_values.clone()).into_array();
    let validity = Validity::from_iter((0..row_count).map(|idx| idx % 5 != 0));
    let array =
        FixedSizeListArray::try_new(elements, list_size, validity.clone(), row_count)?.into_array();

    let source = assert_range_read(&array, 5..11, test_session_with_separate_validity_reads())?;
    assert_eq!(source.ranges().len(), 2);

    let session = test_session_with_separate_validity_reads();
    let mut ctx = session.create_execution_ctx();
    let packed_elements = BitPacked::encode(
        &PrimitiveArray::from_iter(element_values).into_array(),
        4,
        &mut ctx,
    )?;
    assert!(packed_elements.patches().is_none());
    let packed_list =
        FixedSizeListArray::try_new(packed_elements.into_array(), list_size, validity, row_count)?
            .into_array();
    assert_range_read(&packed_list, 260..265, session)?;
    Ok(())
}

#[test]
fn nullable_bitpacked_supports_nonzero_offset() -> VortexResult<()> {
    let values = PrimitiveArray::from_option_iter(
        (0..9000u32).map(|value| (value % 7 != 0).then_some(value % 16)),
    )
    .into_array();
    let mut ctx = test_session().create_execution_ctx();
    let encoded = BitPacked::encode(&values, 4, &mut ctx)?;
    assert!(encoded.patches().is_none());
    let array = encoded.into_array().slice(100..8500)?;

    let source = assert_range_read(
        &array,
        1100..1110,
        test_session_with_separate_validity_reads(),
    )?;
    assert_eq!(source.ranges().len(), 2);
    Ok(())
}

#[test]
fn zero_width_bitpacked_range_read_preserves_buffer_alignment() -> VortexResult<()> {
    let values = PrimitiveArray::from_iter((0..4096).map(|_| 0u32)).into_array();
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let encoded = BitPacked::encode(&values, 0, &mut ctx)?;
    assert!(encoded.packed().is_empty());

    let row_range = 100..108;
    let (actual, source, _) = read_range(&encoded.into_array(), row_range.clone(), session, true)?;
    assert!(
        actual
            .as_::<BitPacked>()
            .packed()
            .is_aligned_to(Alignment::of::<u32>())
    );
    assert_arrays_eq!(actual, values.slice(row_range)?);
    assert_eq!(source.full_requests(), 0);
    assert!(source.ranges().is_empty());
    Ok(())
}

#[test]
fn nullable_wrappers_and_dict_use_pr_compatible_io() -> VortexResult<()> {
    let row_count = 4096;
    let validity = Validity::from_iter((0..row_count).map(|idx| idx % 5 != 0));
    let shifted = PrimitiveArray::new(Buffer::from_iter(0..4096i32), validity.clone());
    let for_array = FoR::try_new(shifted.into_array(), 1000i32.into())?.into_array();
    let source = assert_range_read(
        &for_array,
        13..21,
        test_session_with_separate_validity_reads(),
    )?;
    assert_eq!(source.ranges().len(), 2);

    let zigzag_values =
        PrimitiveArray::new(Buffer::from_iter((0u32..4096).map(|idx| idx * 2)), validity);
    let zigzag = ZigZag::try_new(zigzag_values.into_array())?.into_array();
    let source = assert_range_read(&zigzag, 13..21, test_session_with_separate_validity_reads())?;
    assert_eq!(source.ranges().len(), 2);

    let codes = PrimitiveArray::from_iter((0u8..4).cycle().take(row_count));
    let values = PrimitiveArray::from_iter([10i32, 20, 30, 40]);
    let dict = DictArray::try_new(codes.into_array(), values.into_array())?.into_array();
    let source = assert_range_read(&dict, 4000..4008, test_session())?;
    let values_end = row_count * size_of::<u8>() + 4 * size_of::<i32>();
    assert_eq!(source.ranges(), vec![4000..values_end]);
    Ok(())
}

#[test]
fn bytebool_alp_decimal_and_null_range_reads() -> VortexResult<()> {
    let row_count = 4096;

    let bytebool = ByteBool::from_vec(
        (0..row_count).map(|idx| idx % 3 == 0).collect(),
        Validity::from_iter((0..row_count).map(|idx| idx % 5 != 0)),
    )
    .into_array();
    assert_range_read(
        &bytebool,
        13..21,
        test_session_with_separate_validity_reads(),
    )?;

    let encoded_values = PrimitiveArray::from_iter((0i32..4096).map(|idx| idx % 1024)).into_array();
    let mut ctx = test_session().create_execution_ctx();
    let packed = BitPacked::encode(&encoded_values, 10, &mut ctx)?;
    assert!(packed.patches().is_none());
    let alp = ALP::try_new(packed.into_array(), Exponents { e: 0, f: 0 }, None)?.into_array();
    let source = assert_range_read(&alp, 1030..1040, test_session())?;
    let chunk_bytes = 1024 * 10 / 8;
    assert_eq!(source.ranges(), vec![chunk_bytes..2 * chunk_bytes]);

    let decimal_dtype = DecimalDType::new(12, 2);
    let decimal_values =
        PrimitiveArray::from_option_iter((0i32..4096).map(|idx| (idx % 6 != 0).then_some(idx)));
    let decimal =
        DecimalByteParts::try_new(decimal_values.into_array(), decimal_dtype)?.into_array();
    assert_range_read(
        &decimal,
        13..21,
        test_session_with_separate_validity_reads(),
    )?;

    let nulls = NullArray::new(row_count).into_array();
    let expected = nulls.slice(100..105)?;
    let (actual, source, _) = read_range(&nulls, 100..105, test_session(), true)?;
    assert_arrays_eq!(actual, expected);
    assert_eq!(source.full_requests(), 0);
    assert!(source.ranges().is_empty());
    Ok(())
}

#[test]
fn alprd_wide_contiguous_span_falls_back_to_full_segment() -> VortexResult<()> {
    let values = PrimitiveArray::from_iter((0..4096).map(|idx| match idx % 4 {
        0 => 1.25f64,
        1 => 1.5,
        2 => 2.25,
        _ => 2.5,
    }));
    let encoder = RDEncoder::new(values.as_slice::<f64>());
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let encoded = encoder.encode(values.as_view(), &mut ctx);
    assert!(encoded.left_parts_patches().is_none());

    let array = encoded.into_array();
    let (actual, source, _) = read_range(&array, 1030..1040, session, true)?;
    assert_arrays_eq!(actual, array.slice(1030..1040)?);
    assert_eq!(source.full_requests(), 1);
    assert!(source.ranges().is_empty());
    Ok(())
}

#[test]
fn datetime_parts_wide_contiguous_span_falls_back_to_full_segment() -> VortexResult<()> {
    let timestamps =
        PrimitiveArray::from_iter((0i64..4096).map(|idx| idx * 86_400_000 + idx % 1000))
            .into_array();
    let temporal = TemporalArray::new_timestamp(timestamps, TimeUnit::Milliseconds, None);
    let dtype = temporal.dtype().clone();
    let expected = temporal.clone().into_array();
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let parts = split_temporal(temporal, &mut ctx)?;
    let packed_days = BitPacked::encode(&parts.days, 12, &mut ctx)?;
    assert!(packed_days.patches().is_none());
    let encoded = DateTimeParts::try_new(
        dtype,
        packed_days.into_array(),
        parts.seconds,
        parts.subseconds,
    )?
    .into_array();

    let (actual, source, _) = read_range(&encoded, 1030..1040, session, true)?;
    assert_arrays_eq!(actual, expected.slice(1030..1040)?);
    assert_eq!(source.full_requests(), 1);
    assert!(source.ranges().is_empty());
    Ok(())
}

#[test]
fn delta_with_nullable_bitpacked_deltas_reads_last_padded_chunk() -> VortexResult<()> {
    let values = PrimitiveArray::from_option_iter(
        (0..5000u32).map(|value| (value % 7 != 0).then_some(0u32)),
    );
    let expected = values.clone().into_array();
    let session = test_session_with_separate_validity_reads();
    let mut ctx = session.create_execution_ctx();
    let (bases, deltas) = delta_compress(&values, &mut ctx)?;
    let packed_deltas = BitPacked::encode(&deltas.into_array(), 1, &mut ctx)?;
    assert!(packed_deltas.patches().is_none());
    let array = Delta::try_new(
        bases.into_array(),
        packed_deltas.into_array(),
        0,
        values.len(),
    )?
    .into_array();

    let (actual, source, _) = read_range(&array, 4901..4911, session, true)?;
    assert_arrays_eq!(actual, expected.slice(4901..4911)?);
    assert_eq!(source.full_requests(), 0);
    assert!(!source.ranges().is_empty());
    Ok(())
}

#[test]
fn sequence_preserves_absolute_row_values_without_segment_io() -> VortexResult<()> {
    let array = Sequence::try_new_typed(10i64, 2i64, Nullability::NonNullable, 4096)?.into_array();
    let expected = array.slice(123..129)?;
    let (actual, source, _) = read_range(&array, 123..129, test_session(), true)?;

    assert_arrays_eq!(actual, expected);
    assert_eq!(source.full_requests(), 0);
    assert!(source.ranges().is_empty());
    Ok(())
}

#[test]
fn unsupported_and_disabled_paths_fall_back_to_full_segment() -> VortexResult<()> {
    let mut values = vec![0u32; 4096];
    values[2000] = 100;
    let primitive = PrimitiveArray::from_iter(values).into_array();
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let patched = BitPacked::encode(&primitive, 1, &mut ctx)?;
    assert!(patched.patches().is_some());
    let patched = patched.into_array();
    let (actual, source, _) = read_range(&patched, 2000..2001, session, true)?;
    assert_arrays_eq!(actual, patched.slice(2000..2001)?);
    assert_eq!(source.full_requests(), 1);
    assert!(source.ranges().is_empty());

    let primitive = PrimitiveArray::from_iter(0..4096i32).into_array();
    let disabled_session = test_session();
    disabled_session.get_mut::<RangeReadEnabled>().0 = false;
    let (_, source, _) = read_range(&primitive, 100..101, disabled_session, true)?;
    assert_eq!(source.full_requests(), 1);
    assert!(source.ranges().is_empty());

    let (_, source, _) = read_range(&primitive, 100..101, test_session(), false)?;
    assert_eq!(source.full_requests(), 1);
    assert!(source.ranges().is_empty());
    Ok(())
}
