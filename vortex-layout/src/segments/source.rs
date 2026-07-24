// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::segments::SegmentId;
/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Request a byte range within a segment.
    ///
    /// The default implementation reads the full segment and slices it. Random-access sources
    /// should override this method to avoid the full-segment I/O. The returned range has no
    /// alignment guarantee beyond byte alignment.
    fn request_range(&self, id: SegmentId, range: Range<usize>) -> SegmentFuture {
        let future = self.request(id);
        async move {
            let buffer = future.await?;
            if range.start > range.end || range.end > buffer.len() {
                vortex_bail!(
                    "Segment {} range {}..{} out of bounds for buffer of length {}",
                    id,
                    range.start,
                    range.end,
                    buffer.len()
                );
            }
            let buffer = buffer.ensure_aligned(Alignment::none())?;
            Ok(buffer.slice(range))
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use futures::future;
    use vortex_buffer::ByteBuffer;
    use vortex_error::vortex_err;

    use super::*;

    struct AlignedSegmentSource {
        buffer: ByteBuffer,
    }

    impl SegmentSource for AlignedSegmentSource {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            future::ready(Ok(BufferHandle::new_host(self.buffer.clone()))).boxed()
        }
    }

    #[tokio::test]
    async fn default_range_request_is_unaligned_zero_copy() -> VortexResult<()> {
        let buffer = ByteBuffer::copy_from_aligned([0, 1, 2, 3, 4, 5, 6, 7], Alignment::new(8));
        let expected_ptr = buffer.as_ptr().wrapping_add(4);
        let source = AlignedSegmentSource { buffer };

        let result = source.request_range(SegmentId::from(0), 4..8).await?;
        let result = result
            .as_host_opt()
            .ok_or_else(|| vortex_err!("expected host buffer"))?;

        assert_eq!(result.as_ref(), &[4, 5, 6, 7]);
        assert_eq!(result.alignment(), Alignment::none());
        assert_eq!(result.as_ptr(), expected_ptr);
        Ok(())
    }
}
