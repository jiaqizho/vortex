// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::DType;

use crate::arrays::ConstantArray;
use crate::vtable::{BufferSubRange, EncodingRangeRead, NotSupported, RangeDecodeInfo, VTable};
use crate::{EmptyMetadata, EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod encode;
mod operations;
mod operator;
mod serde;
mod validity;
mod visitor;

vtable!(Constant);

#[derive(Clone, Debug)]
pub struct ConstantEncoding;

impl VTable for ConstantVTable {
    type Array = ConstantArray;
    type Encoding = ConstantEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    // TODO(ngates): implement a compute kernel for elementwise operations
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.constant")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ConstantEncoding.as_ref())
    }

    fn plan_range_read(
        _metadata: &EmptyMetadata,
        row_range: Range<usize>,
        _row_count: usize,
        _dtype: &DType,
    ) -> Option<EncodingRangeRead> {
        Some(EncodingRangeRead {
            buffer_sub_ranges: vec![BufferSubRange::Full],
            children: vec![],
            decode_info: RangeDecodeInfo::Leaf {
                decode_len: row_range.len(),
                post_slice: None,
            },
        })
    }
}
