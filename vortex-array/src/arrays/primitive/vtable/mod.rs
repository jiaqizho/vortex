// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::DType;

use crate::arrays::PrimitiveArray;
use crate::vtable::{
    BufferSubRange, EncodingRangeRead, NotSupported, RangeDecodeInfo, VTable,
    ValidityVTableFromValidityHelper,
};
use crate::{EmptyMetadata, EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod operations;
mod operator;
mod serde;
mod validity;
mod visitor;

vtable!(Primitive);

impl VTable for PrimitiveVTable {
    type Array = PrimitiveArray;
    type Encoding = PrimitiveEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type OperatorVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.primitive")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(PrimitiveEncoding.as_ref())
    }

    fn plan_range_read(
        _metadata: &EmptyMetadata,
        row_range: Range<usize>,
        _row_count: usize,
        dtype: &DType,
    ) -> Option<EncodingRangeRead> {
        let byte_width = match dtype {
            DType::Primitive(ptype, _) => ptype.byte_width(),
            _ => return None,
        };
        Some(EncodingRangeRead {
            buffer_sub_ranges: vec![BufferSubRange::Range(
                row_range.start * byte_width..row_range.end * byte_width,
            )],
            children: vec![],
            decode_info: RangeDecodeInfo::Leaf {
                decode_len: row_range.len(),
                post_slice: None,
            },
        })
    }
}

#[derive(Clone, Debug)]
pub struct PrimitiveEncoding;
