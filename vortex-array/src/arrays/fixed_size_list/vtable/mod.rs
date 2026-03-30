// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::DType;

use crate::arrays::FixedSizeListArray;
use crate::vtable::{
    ChildRangeRead, EncodingRangeRead, NotSupported, RangeDecodeInfo, VTable,
    ValidityVTableFromValidityHelper,
};
use crate::{EmptyMetadata, EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod operations;
mod serde;
mod validity;
mod visitor;

vtable!(FixedSizeList);

#[derive(Clone, Debug)]
pub struct FixedSizeListEncoding;

impl VTable for FixedSizeListVTable {
    type Array = FixedSizeListArray;
    type Encoding = FixedSizeListEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.fixed_size_list")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(FixedSizeListEncoding.as_ref())
    }

    fn plan_range_read(
        _metadata: &EmptyMetadata,
        row_range: Range<usize>,
        row_count: usize,
        dtype: &DType,
    ) -> Option<EncodingRangeRead> {
        let (element_dtype, list_size) = match dtype {
            DType::FixedSizeList(element_dtype, list_size, _) => {
                (element_dtype.as_ref().clone(), *list_size as usize)
            }
            _ => return None,
        };

        let element_range = (row_range.start * list_size)..(row_range.end * list_size);
        let element_count = row_count * list_size;

        Some(EncodingRangeRead {
            buffer_sub_ranges: vec![],
            children: vec![ChildRangeRead::Recurse {
                row_range: element_range,
                row_count: element_count,
                dtype: element_dtype,
            }],
            decode_info: RangeDecodeInfo::FromChild {
                child_idx: 0,
                divisor: list_size,
            },
        })
    }
}
