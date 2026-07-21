// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoding-level planning primitives for FlatLayout sub-segment reads.

use std::ops::Range;

use crate::dtype::DType;

/// Describes the buffers and children required to decode a logical row range.
///
/// Encodings return `None` from their range planning hook when a particular encoded array cannot
/// be read safely from a sub-range. The layout reader then falls back to reading the full segment.
#[derive(Debug, Clone)]
pub struct EncodingRangeRead {
    /// One entry for each buffer owned by this encoding node.
    pub buffer_sub_ranges: Vec<BufferSubRange>,
    /// How each non-validity child should be read.
    pub children: Vec<ChildRangeRead>,
    /// The logical source rows represented by the partially decoded array.
    pub decode_info: RangeDecodeInfo,
    /// Whether this encoding may serialize an additional validity child.
    pub validity: ValidityRangeRead,
}

/// Specifies which bytes of an encoding-owned buffer are required.
#[derive(Debug, Clone)]
pub enum BufferSubRange {
    /// Read the entire buffer.
    Full,
    /// Read only this byte range within the buffer.
    Range(Range<usize>),
}

/// Specifies how a serialized child array should be read.
#[derive(Debug, Clone)]
pub enum ChildRangeRead {
    /// Recursively plan a child. Its decoded range may expand to satisfy encoding alignment.
    Recurse {
        /// Requested logical rows in the child coordinate space.
        row_range: Range<usize>,
        /// Total logical rows in the serialized child.
        row_count: usize,
        /// Logical dtype of the child.
        dtype: DType,
    },
    /// Recursively plan a child, requiring it to decode exactly the requested row range.
    ///
    /// This is used for auxiliary children whose coordinate space cannot be mapped back to the
    /// parent if the child expands its decode range.
    RecurseExact {
        /// Requested logical rows in the child coordinate space.
        row_range: Range<usize>,
        /// Total logical rows in the serialized child.
        row_count: usize,
        /// Logical dtype of the child.
        dtype: DType,
    },
    /// Recursively plan this child using the decoded row range of an earlier child.
    ///
    /// The referenced child must precede this child. If this child needs a wider range, the parent
    /// node is replanned with that wider range so all row-aligned children remain synchronized.
    RecurseWithDecodedRange {
        /// Index of the earlier child whose decoded row range should be reused.
        child_idx: usize,
        /// Total logical rows in the serialized child.
        row_count: usize,
        /// Logical dtype of the child.
        dtype: DType,
    },
    /// Include the entire child tree.
    Full,
}

/// Describes the logical rows produced when decoding the selected buffers.
#[derive(Debug, Clone)]
pub enum RangeDecodeInfo {
    /// This encoding directly determines its decoded source row range.
    Rows(Range<usize>),
    /// Derive the decoded source row range from a child.
    ///
    /// `divisor` maps a child coordinate space back to its parent, for example elements back to
    /// rows in a fixed-size list.
    FromChild {
        /// Child whose decoded range drives the parent.
        child_idx: usize,
        /// Divisor applied to the child range boundaries.
        divisor: usize,
    },
}

/// Describes whether an encoding has its own serialized validity child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidityRangeRead {
    /// The encoding has no independent validity child.
    None,
    /// The encoding may have one trailing validity child.
    ///
    /// The child is absent for non-nullable and all-valid arrays, and present for an explicit
    /// validity array or an all-invalid array.
    Optional,
}
