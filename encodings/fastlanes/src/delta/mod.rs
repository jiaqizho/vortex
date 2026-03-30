// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Range;

pub use compress::*;
use fastlanes::FastLanes;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, ChildRangeRead, EncodingRangeRead, NotSupported, RangeDecodeInfo,
    VTable, ValidityChildSliceHelper, ValidityVTableFromChildSliceHelper,
};
use vortex_array::{
    Array, ArrayEq, ArrayHash, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, Precision,
    vtable,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, PType, match_each_unsigned_integer_ptype};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};

use self::serde::DeltaMetadata;

mod compress;
mod compute;
mod ops;
mod serde;

vtable!(Delta);

impl VTable for DeltaVTable {
    type Array = DeltaArray;
    type Encoding = DeltaEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChildSliceHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("fastlanes.delta")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(DeltaEncoding.as_ref())
    }

    fn plan_range_read(
        metadata: &DeltaMetadata,
        row_range: Range<usize>,
        _row_count: usize,
        dtype: &DType,
    ) -> Option<EncodingRangeRead> {
        if metadata.offset != 0 {
            return None;
        }

        let deltas_len = usize::try_from(metadata.deltas_len).ok()?;
        let byte_width = match dtype {
            DType::Primitive(ptype, _) => ptype.byte_width(),
            _ => return None,
        };
        let lanes = match byte_width {
            1 => 128,
            2 => 64,
            4 => 32,
            8 => 16,
            _ => return None,
        };

        let first_chunk = row_range.start / 1024;
        let last_chunk = row_range.end.saturating_sub(1) / 1024;

        let deltas_row_start = first_chunk * 1024;
        let deltas_row_end = ((last_chunk + 1) * 1024).min(deltas_len);

        let num_full_chunks = deltas_len / 1024;
        let has_remainder = !deltas_len.is_multiple_of(1024);
        let bases_len = num_full_chunks * lanes + if has_remainder { 1 } else { 0 };
        let bases_row_start = first_chunk * lanes;
        let bases_row_end = if last_chunk >= num_full_chunks {
            bases_len
        } else {
            (last_chunk + 1) * lanes
        }
        .min(bases_len);

        let sub_deltas_len = deltas_row_end - deltas_row_start;
        let intra_chunk_offset = row_range.start - deltas_row_start;
        let post_slice = (intra_chunk_offset > 0 || sub_deltas_len > row_range.len())
            .then(|| intra_chunk_offset..(intra_chunk_offset + row_range.len()));

        Some(EncodingRangeRead {
            buffer_sub_ranges: vec![],
            children: vec![
                ChildRangeRead::Recurse {
                    row_range: bases_row_start..bases_row_end,
                    row_count: bases_len,
                    dtype: dtype.clone(),
                },
                ChildRangeRead::Recurse {
                    row_range: deltas_row_start..deltas_row_end,
                    row_count: deltas_len,
                    dtype: dtype.clone(),
                },
            ],
            decode_info: RangeDecodeInfo::Leaf {
                decode_len: sub_deltas_len,
                post_slice,
            },
        })
    }
}

/// A FastLanes-style delta-encoded array of primitive values.
///
/// A [`DeltaArray`] comprises a sequence of _chunks_ each representing 1,024 delta-encoded values,
/// except the last chunk which may represent from one to 1,024 values.
///
/// # Examples
///
/// ```
/// use vortex_fastlanes::DeltaArray;
/// let array = DeltaArray::try_from_vec(vec![1_u32, 2, 3, 5, 10, 11]).unwrap();
/// ```
///
/// # Details
///
/// To facilitate slicing, this array accepts an `offset` and `logical_len`. The offset must be
/// strictly less than 1,024 and the sum of `offset` and `logical_len` must not exceed the length of
/// the `deltas` array. These values permit logical slicing without modifying any chunk containing a
/// kept value. In particular, we may defer decompresison until the array is canonicalized or
/// indexed. The `offset` is a physical offset into the first chunk, which necessarily contains
/// 1,024 values. The `logical_len` is the number of logical values following the `offset`, which
/// may be less than the number of physically stored values.
///
/// Each chunk is stored as a vector of bases and a vector of deltas. If the chunk physically
/// contains 1,024 values, then there are as many bases as there are _lanes_ of this type in a
/// 1024-bit register. For example, for 64-bit values, there are 16 bases because there are 16
/// _lanes_. Each lane is a [delta-encoding](https://en.wikipedia.org/wiki/Delta_encoding) `1024 /
/// bit_width` long vector of values. The deltas are stored in the
/// [FastLanes](https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf) order which splits the 1,024
/// values into one contiguous sub-sequence per-lane, thus permitting delta encoding.
///
/// If the chunk physically has fewer than 1,024 values, then it is stored as a traditional,
/// non-SIMD-amenable, delta-encoded vector.
///
/// Note the validity is stored in the deltas array.
#[derive(Clone, Debug)]
pub struct DeltaArray {
    offset: usize,
    len: usize,
    dtype: DType,
    bases: ArrayRef,
    deltas: ArrayRef,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct DeltaEncoding;

impl DeltaArray {
    // TODO(ngates): remove constructing from vec
    pub fn try_from_vec<T: NativePType>(vec: Vec<T>) -> VortexResult<Self> {
        Self::try_from_primitive_array(&PrimitiveArray::new(
            Buffer::copy_from(vec),
            Validity::NonNullable,
        ))
    }

    pub fn try_from_primitive_array(array: &PrimitiveArray) -> VortexResult<Self> {
        let (bases, deltas) = delta_compress(array)?;

        Self::try_from_delta_compress_parts(bases.into_array(), deltas.into_array())
    }

    /// Create a [`DeltaArray`] from the given `bases` and `deltas` arrays.
    /// Note the `deltas` might be nullable
    pub fn try_from_delta_compress_parts(bases: ArrayRef, deltas: ArrayRef) -> VortexResult<Self> {
        let logical_len = deltas.len();
        Self::try_new(bases, deltas, 0, logical_len)
    }

    pub fn try_new(
        bases: ArrayRef,
        deltas: ArrayRef,
        offset: usize,
        logical_len: usize,
    ) -> VortexResult<Self> {
        if offset >= 1024 {
            vortex_bail!("offset must be less than 1024: {}", offset);
        }
        if offset + logical_len > deltas.len() {
            vortex_bail!(
                "offset + logical_len, {} + {}, must be less than or equal to the size of deltas: {}",
                offset,
                logical_len,
                deltas.len()
            )
        }
        if !bases.dtype().eq_ignore_nullability(deltas.dtype()) {
            vortex_bail!(
                "DeltaArray: bases and deltas must have the same dtype, got {:?} and {:?}",
                bases.dtype(),
                deltas.dtype()
            );
        }
        let DType::Primitive(ptype, _) = bases.dtype().clone() else {
            vortex_bail!(
                "DeltaArray: dtype must be an integer, got {}",
                bases.dtype()
            );
        };

        if !ptype.is_int() {
            vortex_bail!("DeltaArray: ptype must be an integer, got {}", ptype);
        }

        let lanes = lane_count(ptype);

        if (deltas.len() % 1024 == 0) != (bases.len() % lanes == 0) {
            vortex_bail!(
                "deltas length ({}) is a multiple of 1024 iff bases length ({}) is a multiple of LANES ({})",
                deltas.len(),
                bases.len(),
                lanes,
            );
        }

        // SAFETY: validation done above
        Ok(unsafe { Self::new_unchecked(bases, deltas, offset, logical_len) })
    }

    pub(crate) unsafe fn new_unchecked(
        bases: ArrayRef,
        deltas: ArrayRef,
        offset: usize,
        logical_len: usize,
    ) -> Self {
        Self {
            offset,
            len: logical_len,
            dtype: bases.dtype().with_nullability(deltas.dtype().nullability()),
            bases,
            deltas,
            stats_set: Default::default(),
        }
    }

    #[inline]
    pub fn bases(&self) -> &ArrayRef {
        &self.bases
    }

    #[inline]
    pub fn deltas(&self) -> &ArrayRef {
        &self.deltas
    }

    #[inline]
    fn lanes(&self) -> usize {
        let ptype =
            PType::try_from(self.dtype()).vortex_expect("DeltaArray DType must be primitive");
        lane_count(ptype)
    }

    #[inline]
    /// The logical offset into the first chunk of [`Self::deltas`].
    pub fn offset(&self) -> usize {
        self.offset
    }

    fn bases_len(&self) -> usize {
        self.bases.len()
    }

    fn deltas_len(&self) -> usize {
        self.deltas.len()
    }
}

pub(crate) fn lane_count(ptype: PType) -> usize {
    match_each_unsigned_integer_ptype!(ptype, |T| { T::LANES })
}

impl ValidityChildSliceHelper for DeltaArray {
    fn unsliced_child_and_slice(&self) -> (&ArrayRef, usize, usize) {
        let (start, len) = (self.offset(), self.len());
        (self.deltas(), start, start + len)
    }
}

impl ArrayVTable<DeltaVTable> for DeltaVTable {
    fn len(array: &DeltaArray) -> usize {
        array.len
    }

    fn dtype(array: &DeltaArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DeltaArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &DeltaArray, state: &mut H, precision: Precision) {
        array.offset.hash(state);
        array.len.hash(state);
        array.dtype.hash(state);
        array.bases.array_hash(state, precision);
        array.deltas.array_hash(state, precision);
    }

    fn array_eq(array: &DeltaArray, other: &DeltaArray, precision: Precision) -> bool {
        array.offset == other.offset
            && array.len == other.len
            && array.dtype == other.dtype
            && array.bases.array_eq(&other.bases, precision)
            && array.deltas.array_eq(&other.deltas, precision)
    }
}

impl CanonicalVTable<DeltaVTable> for DeltaVTable {
    fn canonicalize(array: &DeltaArray) -> Canonical {
        Canonical::Primitive(delta_decompress(array))
    }
}
