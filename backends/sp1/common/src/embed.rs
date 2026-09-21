//! The `veritor.embed-gather@1` gates: the committed-table row gather of the
//! embedding relation (`Embedding_v1{V,H}`, lane PROOF-COV), bit for bit as
//! `veritor/core/embed_gates.py` pins them.
//!
//! The registered body is one wide-fan-in `GatherBf16x{V}` per hidden
//! coordinate (`h[j] = table[tok][j]`), which does not map to a scalar-gate
//! `KindProgram` directly (the whole vocabulary column would be an operand).
//! The lowering instead follows the MUFU-table pattern of
//! `veritor.reduce-mufu@1`: the embedding matrix is committed as a Merkle
//! tree of `(flat_index << 16) | bf16_word` leaves (`flat = tok * H + j`, row
//! major), the statement opens the gathered row's leaves as 64-bit witness
//! values, and the gather gate *recomputes* the flat index from the token
//! word and rejects any opened row word whose embedded index differs -- so a
//! witness can only ever use the one true table entry for the claimed token,
//! and the token word itself is opened from the record's committed token
//! leaves (the index is bound, not free).
//!
//! Domain note: the registered primitive's total extension (out-of-range or
//! negative token -> `0x7FFF`) is *not* lowered; an out-of-range token yields
//! a flat index no committed leaf embeds, so every opening rejects and the
//! statement fails closed.  Committed serving records only carry in-range
//! tokens, and the per-row build-time validation against the IR oracle
//! covers exactly that domain.

/// Wrapping 32-bit multiply (the row-base computation `tok * H`; exact --
/// no wrap -- on the declared domain `tok < V`, `H` such that `V * H < 2^32`).
#[cfg_attr(feature = "attribution", inline(never))]
pub fn i32_mul(a: u64, b: u64) -> u64 {
    ((a as u32).wrapping_mul(b as u32)) as u64
}

/// Wrapping 32-bit add (the flat-index chain `flat_{j+1} = flat_j + 1`).
#[cfg_attr(feature = "attribution", inline(never))]
pub fn i32_add(a: u64, b: u64) -> u64 {
    ((a as u32).wrapping_add(b as u32)) as u64
}

/// The gather gate: `(flat: u32 word, row: u64 leaf "(index << 16) | word")
/// -> bf16 word`.  Rejects (`None`) when the opened leaf's embedded index is
/// not the recomputed flat index -- the table-lookup contract of the MUFU
/// gates, verbatim.
#[cfg_attr(feature = "attribution", inline(never))]
pub fn gather_word(flat: u64, row: u64) -> Option<u64> {
    if row >> 16 != flat {
        return None;
    }
    Some(row & 0xFFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_forces_the_embedded_index() {
        let leaf = (576u64 << 16) | 0x3F80;
        assert_eq!(gather_word(576, leaf), Some(0x3F80));
        assert_eq!(gather_word(577, leaf), None);
        assert_eq!(gather_word(0, 0x1234), Some(0x1234));
        assert_eq!(gather_word(1, 0x1234), None);
    }

    #[test]
    fn index_arithmetic_wraps_like_u32() {
        assert_eq!(i32_mul(49151, 576), 49151 * 576);
        assert_eq!(i32_mul(0xFFFF_FFFF, 2), 0xFFFF_FFFE);
        assert_eq!(i32_add(0xFFFF_FFFF, 1), 0);
        assert_eq!(i32_add(28_311_551, 1), 28_311_552);
    }
}
