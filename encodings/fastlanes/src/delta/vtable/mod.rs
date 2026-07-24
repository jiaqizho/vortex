// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::ChildRangeRead;
use vortex_array::vtable::EncodingRangeRead;
use vortex_array::vtable::RangeDecodeInfo;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityRangeRead;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::DeltaData;
use crate::delta::array::BASES_SLOT;
use crate::delta::array::DELTAS_SLOT;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::SLOT_NAMES;
use crate::delta::array::delta_decompress::delta_decompress;
use crate::delta::array::lane_count;
use crate::delta_compress;

mod operations;
mod rules;
mod slice;
mod validity;

/// A [`Delta`]-encoded Vortex array.
pub type DeltaArray = Array<Delta>;

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DeltaMetadata {
    #[prost(uint64, tag = "1")]
    deltas_len: u64,
    #[prost(uint32, tag = "2")]
    offset: u32, // must be <1024
}

impl ArrayHash for DeltaData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
    }
}

impl ArrayEq for DeltaData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset
    }
}

impl VTable for Delta {
    type TypedArrayData = DeltaData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("fastlanes.delta");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let bases = slots[BASES_SLOT]
            .as_ref()
            .vortex_expect("DeltaArray bases slot");
        let deltas = slots[DELTAS_SLOT]
            .as_ref()
            .vortex_expect("DeltaArray deltas slot");
        validate_parts(bases, deltas, data.offset, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DeltaArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        rules::RULES.evaluate(array, parent, child_idx)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DeltaMetadata {
                deltas_len: array.deltas().len() as u64,
                offset: array.offset() as u32,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.is_empty(),
            "DeltaArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == 2,
            "DeltaArray expects 2 children, got {}",
            children.len()
        );
        let metadata = DeltaMetadata::decode(metadata)?;
        let ptype = PType::try_from(dtype)?;
        let lanes = lane_count(ptype);

        let serialized_deltas_len = usize::try_from(metadata.deltas_len)
            .map_err(|_| vortex_err!("deltas_len {} overflowed usize", metadata.deltas_len))?;
        let offset = metadata.offset as usize;
        let logical_end = offset
            .checked_add(len)
            .ok_or_else(|| vortex_err!("DeltaArray length overflow: {offset} + {len}"))?;
        let required_deltas_len = logical_end
            .div_ceil(1024)
            .checked_mul(1024)
            .ok_or_else(|| vortex_err!("DeltaArray padded length overflow"))?;
        vortex_ensure!(
            required_deltas_len <= serialized_deltas_len,
            "DeltaArray needs {required_deltas_len} deltas but metadata describes {serialized_deltas_len}"
        );
        let deltas_len = if children.is_partial_decode() {
            required_deltas_len
        } else {
            serialized_deltas_len
        };
        let bases_len = (deltas_len / 1024)
            .checked_mul(lanes)
            .ok_or_else(|| vortex_err!("DeltaArray bases length overflow"))?;

        let bases = children.get(0, dtype, bases_len)?;
        let deltas = children.get(1, dtype, deltas_len)?;

        let data = DeltaData::try_new(offset)?;
        let slots = smallvec![Some(bases), Some(deltas)];
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            delta_decompress(&array, ctx)?.into_array(),
        ))
    }

    fn plan_range_read(
        &self,
        metadata: &[u8],
        row_range: Range<usize>,
        _row_count: usize,
        dtype: &DType,
        _session: &VortexSession,
    ) -> VortexResult<Option<EncodingRangeRead>> {
        let metadata = DeltaMetadata::decode(metadata)?;
        if metadata.offset != 0 {
            return Ok(None);
        }
        let deltas_len = usize::try_from(metadata.deltas_len)
            .map_err(|_| vortex_err!("deltas_len {} overflowed usize", metadata.deltas_len))?;
        if !deltas_len.is_multiple_of(1024) || row_range.end > deltas_len {
            return Ok(None);
        }

        let Ok(ptype) = PType::try_from(dtype) else {
            return Ok(None);
        };
        let lanes = lane_count(ptype);
        let first_chunk = row_range.start / 1024;
        let last_chunk = row_range.end.div_ceil(1024);
        let Some(deltas_start) = first_chunk.checked_mul(1024) else {
            return Ok(None);
        };
        let Some(deltas_end) = last_chunk.checked_mul(1024) else {
            return Ok(None);
        };
        let Some(bases_len) = (deltas_len / 1024).checked_mul(lanes) else {
            return Ok(None);
        };
        let Some(bases_start) = first_chunk.checked_mul(lanes) else {
            return Ok(None);
        };
        let Some(bases_end) = last_chunk.checked_mul(lanes) else {
            return Ok(None);
        };

        Ok(Some(EncodingRangeRead {
            buffer_sub_ranges: vec![],
            children: vec![
                ChildRangeRead::RecurseExact {
                    row_range: bases_start..bases_end,
                    row_count: bases_len,
                    dtype: dtype.clone(),
                },
                ChildRangeRead::RecurseExact {
                    row_range: deltas_start..deltas_end,
                    row_count: deltas_len,
                    dtype: dtype.clone(),
                },
            ],
            decode_info: RangeDecodeInfo::Rows(deltas_start..deltas_end),
            validity: ValidityRangeRead::None,
        }))
    }
}

#[derive(Clone, Debug)]
pub struct Delta;

impl Delta {
    pub fn try_new(
        bases: ArrayRef,
        deltas: ArrayRef,
        offset: usize,
        len: usize,
    ) -> VortexResult<DeltaArray> {
        let dtype = bases.dtype().with_nullability(deltas.dtype().nullability());
        let data = DeltaData::try_new(offset)?;
        let slots = smallvec![Some(bases), Some(deltas)];
        Array::try_from_parts(ArrayParts::new(Delta, dtype, len, data).with_slots(slots))
    }

    /// Compress a primitive array using Delta encoding.
    pub fn try_from_primitive_array(
        array: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<DeltaArray> {
        let logical_len = array.len();
        let (bases, deltas) = delta_compress(array, ctx)?;
        Self::try_new(bases.into_array(), deltas.into_array(), 0, logical_len)
    }
}

fn validate_parts(
    bases: &ArrayRef,
    deltas: &ArrayRef,
    offset: usize,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        offset + len <= deltas.len(),
        "offset + len, {offset} + {len}, must be less than or equal to the size of deltas: {}",
        deltas.len()
    );
    vortex_ensure!(
        bases.dtype().eq_ignore_nullability(deltas.dtype()),
        "DeltaArray: bases and deltas must have the same dtype, got {} and {}",
        bases.dtype(),
        deltas.dtype()
    );

    vortex_ensure!(
        bases.dtype().is_int(),
        "DeltaArray: dtype must be an integer, got {}",
        bases.dtype()
    );

    let expected_dtype = bases.dtype().with_nullability(deltas.dtype().nullability());
    vortex_ensure!(
        dtype == &expected_dtype,
        "DeltaArray dtype mismatch: expected {expected_dtype}, got {dtype}"
    );

    let lanes = lane_count(bases.dtype().as_ptype());

    vortex_ensure!(
        deltas.len().is_multiple_of(1024),
        "deltas length ({}) must be a multiple of 1024",
        deltas.len(),
    );
    vortex_ensure!(
        bases.len().is_multiple_of(lanes),
        "bases length ({}) must be a multiple of LANES ({lanes})",
        bases.len(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::PType;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySession;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::test_harness::check_metadata;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::Delta;
    use super::DeltaMetadata;
    use super::lane_count;
    use crate::delta::array::DeltaArrayExt;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_delta_metadata() {
        check_metadata(
            "delta.metadata",
            &DeltaMetadata {
                offset: u32::MAX,
                deltas_len: u64::MAX,
            }
            .encode_to_vec(),
        );
    }

    #[test]
    fn full_serde_preserves_extra_backing_chunks() -> VortexResult<()> {
        const LOGICAL_LEN: usize = 100;
        const DELTAS_LEN: usize = 2048;

        let bases_len = 2 * lane_count(PType::I32);
        let bases = PrimitiveArray::from_iter(vec![0i32; bases_len]).into_array();
        let deltas = PrimitiveArray::from_iter(vec![0i32; DELTAS_LEN]).into_array();
        let array = Delta::try_new(bases, deltas, 0, LOGICAL_LEN)?;

        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(Delta);
        let ctx = ArrayContext::empty();
        let buffers =
            array
                .clone()
                .into_array()
                .serialize(&ctx, &session, &SerializeOptions::default())?;
        let mut serialized = ByteBufferMut::empty();
        for buffer in buffers {
            serialized.extend_from_slice(buffer.as_ref());
        }

        let serialized = SerializedArray::try_from(serialized.freeze())?;
        let decoded = serialized.decode(
            array.dtype(),
            array.len(),
            &ReadContext::new(ctx.to_ids()),
            &session,
        )?;
        let decoded = decoded.as_::<Delta>();
        assert_eq!(decoded.bases().len(), bases_len);
        assert_eq!(decoded.deltas().len(), DELTAS_LEN);
        Ok(())
    }
}
