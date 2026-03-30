// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<ByteBuffer>>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Request a sub-range of a segment's bytes.
    /// The default implementation reads the full segment and then slices.
    fn request_range(&self, id: SegmentId, range: std::ops::Range<usize>) -> SegmentFuture {
        let full = self.request(id);
        Box::pin(async move {
            let buffer = full.await?;
            Ok(buffer.slice_unaligned(range))
        })
    }
}
