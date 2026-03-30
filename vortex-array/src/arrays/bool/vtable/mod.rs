// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::DType;

use self::serde::BoolMetadata;
use crate::arrays::BoolArray;
use crate::vtable::{
    BufferSubRange, EncodingRangeRead, NotSupported, RangeDecodeInfo, VTable,
    ValidityVTableFromValidityHelper,
};
use crate::{EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod operations;
mod operator;
mod serde;
mod validity;
mod visitor;

vtable!(Bool);

impl VTable for BoolVTable {
    type Array = BoolArray;
    type Encoding = BoolEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.bool")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(BoolEncoding.as_ref())
    }

    fn plan_range_read(
        metadata: &BoolMetadata,
        row_range: Range<usize>,
        _row_count: usize,
        _dtype: &DType,
    ) -> Option<EncodingRangeRead> {
        // Only support offset=0 and byte-aligned start.
        if metadata.offset != 0 || !row_range.start.is_multiple_of(8) {
            return None;
        }

        let byte_start = row_range.start / 8;
        let byte_end = row_range.end.div_ceil(8);

        Some(EncodingRangeRead {
            buffer_sub_ranges: vec![BufferSubRange::Range(byte_start..byte_end)],
            children: vec![],
            decode_info: RangeDecodeInfo::Leaf {
                decode_len: row_range.len(),
                post_slice: None,
            },
        })
    }
}

#[derive(Clone, Debug)]
pub struct BoolEncoding;
