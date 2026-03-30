// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::vtable::{
    BufferSubRange, EncodingRangeRead, NotSupported, RangeDecodeInfo, VTable,
    ValidityVTableFromValidityHelper,
};
use vortex_array::{EncodingId, EncodingRef, vtable};
use vortex_dtype::DType;

use self::serde::BitPackedMetadata;
use crate::BitPackedArray;

mod array;
mod canonical;
mod encode;
mod operations;
mod operator;
mod serde;
mod validity;
mod visitor;

vtable!(BitPacked);

impl VTable for BitPackedVTable {
    type Array = BitPackedArray;
    type Encoding = BitPackedEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;
    type OperatorVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("fastlanes.bitpacked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(BitPackedEncoding.as_ref())
    }

    fn plan_range_read(
        metadata: &BitPackedMetadata,
        row_range: Range<usize>,
        _row_count: usize,
        _dtype: &DType,
    ) -> Option<EncodingRangeRead> {
        let bit_width = metadata.bit_width as usize;

        if bit_width == 0 || metadata.offset != 0 {
            return None;
        }

        if metadata.patches.is_some() {
            return None;
        }

        let first_block = row_range.start / 1024;
        let last_block = row_range.end.saturating_sub(1) / 1024;
        let bytes_per_block = 128 * bit_width;
        let byte_start = first_block * bytes_per_block;
        let byte_end = (last_block + 1) * bytes_per_block;

        let block_start_row = first_block * 1024;
        let decode_len = row_range.end - block_start_row;
        let intra_block_offset = row_range.start - block_start_row;
        let post_slice = (intra_block_offset > 0).then_some(intra_block_offset..decode_len);

        Some(EncodingRangeRead {
            buffer_sub_ranges: vec![BufferSubRange::Range(byte_start..byte_end)],
            children: vec![],
            decode_info: RangeDecodeInfo::Leaf {
                decode_len,
                post_slice,
            },
        })
    }
}

#[derive(Clone, Debug)]
pub struct BitPackedEncoding;
