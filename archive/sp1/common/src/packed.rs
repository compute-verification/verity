//! Packed-word authentication: the `packed/v1` leaf schema and the serving
//! side's leaf framing (`verity-vllm/leaf/v1`), mirroring
//! `veritor/protocol/proofs/packed.py` byte for byte.
//!
//! A logical value `gate -> Value<w>` may live physically inside a committed
//! leaf that packs a whole row (IR §H.4: "weight-row leaf 81, word 237").
//! The statement's `PositionRef` then carries, in place of the one-word
//! schema `u<w>`, the schema
//!
//! ```text
//! packed/v1:<framing>:u<w><le|be>:<count>:<index>
//! ```
//!
//! and the guest authenticates the *whole* leaf at `rank` under `<framing>`
//! -- the only one implemented is `verity-vllm/leaf/v1` -- then reads word
//! `index` of its `count` words of `ceil(w/8)` bytes in the named byte order
//! as the position's value.  `(framing, u<w><order>, count)` is the leaf
//! descriptor every position naming one rank must agree on; `index` is the
//! position's own.
//!
//! The serving side's framing (`vllm-poc/verity_vllm/commit/hashing.py`):
//!
//! ```text
//! H(tag, *parts) = sha256( u32be(len(tag)) || tag || (u64be(len(part)) || part)* )
//! leaf = H("verity-vllm/leaf/v1", canonical_id, dtype, shape_text, value)
//! node = H("verity-vllm/node/v1", left, right)
//! lift = H("verity-vllm/lift/v1", child)      # an odd trailing node, never duplicated
//! ```
//!
//! Level 0 is the leaf hashes in tree order; neighbours pair, an odd trailing
//! node is lifted.  A path is one step per level -- a sibling digest or a
//! lift -- and its sibling/lift pattern is a function of `(count, index)`;
//! `fold_path` refuses a path of another shape before comparing any hash.

use sha2::{Digest, Sha256};

pub const PACKED_SCHEMA_PREFIX: &str = "packed/v1:";
/// The serving side's leaf framing (`verity_vllm.commit.hashing.LEAF_TAG`).
pub const VLLM_LEAF_FRAMING: &str = "verity-vllm/leaf/v1";
pub const VLLM_NODE_TAG: &str = "verity-vllm/node/v1";
pub const VLLM_LIFT_TAG: &str = "verity-vllm/lift/v1";

/// A parsed `packed/v1` schema.  `framing` is always [`VLLM_LEAF_FRAMING`]
/// today (the parser knows no other); it is kept so the descriptor
/// comparison is by value, as the Python side's is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedSchema {
    pub framing: &'static str,
    pub width: u32,
    pub little_endian: bool,
    pub count: u64,
    pub index: u64,
}

/// What every position naming one packed leaf must agree on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafDescriptor {
    pub framing: &'static str,
    pub width: u32,
    pub little_endian: bool,
    pub count: u64,
}

impl PackedSchema {
    pub fn descriptor(&self) -> LeafDescriptor {
        LeafDescriptor {
            framing: self.framing,
            width: self.width,
            little_endian: self.little_endian,
            count: self.count,
        }
    }

    pub fn word_bytes(&self) -> usize {
        (self.width as usize).div_ceil(8)
    }
}

impl LeafDescriptor {
    pub fn word_bytes(&self) -> usize {
        (self.width as usize).div_ceil(8)
    }

    /// The exact byte length a leaf of this descriptor has.
    pub fn leaf_bytes(&self) -> Option<usize> {
        usize::try_from(self.count)
            .ok()?
            .checked_mul(self.word_bytes())
    }
}

/// A decimal field without leading zeros, below `bound`.
fn decimal(text: &str, bound: u64) -> Option<u64> {
    if text.is_empty() || (text.len() > 1 && text.starts_with('0')) {
        return None;
    }
    if !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: u64 = text.parse().ok()?;
    (value < bound).then_some(value)
}

/// The three fields of a descriptor after the prefix, or `None`.
fn parse_descriptor_fields(framing: &str, word: &str, count: &str) -> Option<LeafDescriptor> {
    let framing = if framing == VLLM_LEAF_FRAMING {
        VLLM_LEAF_FRAMING
    } else {
        return None;
    };
    let digits = word.strip_prefix('u')?;
    let (digits, little_endian) = if let Some(d) = digits.strip_suffix("le") {
        (d, true)
    } else if let Some(d) = digits.strip_suffix("be") {
        (d, false)
    } else {
        return None;
    };
    let width = decimal(digits, 65)?;
    if width == 0 {
        return None;
    }
    let count = decimal(count, 1 << 32)?;
    if count == 0 {
        return None;
    }
    Some(LeafDescriptor {
        framing,
        width: width as u32,
        little_endian,
        count,
    })
}

/// `veritor.protocol.proofs.schema.parse_leaf_descriptor`: a packed schema
/// without its index -- `packed/v1:<framing>:u<w><le|be>:<count>`, exactly
/// three fields after the prefix -- strict, or `None`.  What a statement-v3
/// range names.
pub fn parse_leaf_descriptor(text: &str) -> Option<LeafDescriptor> {
    let rest = text.strip_prefix(PACKED_SCHEMA_PREFIX)?;
    let mut parts = rest.split(':');
    let framing = parts.next()?;
    let word = parts.next()?;
    let count = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    parse_descriptor_fields(framing, word, count)
}

/// `veritor.protocol.proofs.packed.parse_packed_schema`: strict, or `None`.
///
/// Exactly four `:`-separated fields after the prefix, a known framing,
/// `u<w>le` / `u<w>be` with `1 <= w <= 64`, `1 <= count < 2^32`,
/// `index < count`, all decimals without leading zeros.
pub fn parse_packed_schema(schema: &str) -> Option<PackedSchema> {
    let rest = schema.strip_prefix(PACKED_SCHEMA_PREFIX)?;
    let mut parts = rest.split(':');
    let framing = parts.next()?;
    let word = parts.next()?;
    let count = parts.next()?;
    let index = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let descriptor = parse_descriptor_fields(framing, word, count)?;
    let index = decimal(index, 1 << 32)?;
    if index >= descriptor.count {
        return None;
    }
    Some(PackedSchema {
        framing: descriptor.framing,
        width: descriptor.width,
        little_endian: descriptor.little_endian,
        count: descriptor.count,
        index,
    })
}

/// Word `index` of a packed leaf's `value` bytes under `descriptor`, or
/// `None` when the leaf is not exactly `count * ceil(w/8)` bytes, the index
/// is not below `count`, or the word is not below `2^w`.  The word is the
/// `ceil(w/8)`-byte slice at `index * ceil(w/8)`, read in the descriptor's
/// byte order.  [`extract_word`] is this at the schema's own index; a range
/// (`check_openings_ranged`) calls it once per word without building a schema.
pub fn extract_word_at(descriptor: &LeafDescriptor, index: u64, value: &[u8]) -> Option<u64> {
    let size = descriptor.word_bytes();
    if value.len() != descriptor.leaf_bytes()? || index >= descriptor.count {
        return None;
    }
    let start = usize::try_from(index).ok()?.checked_mul(size)?;
    let bytes = value.get(start..start + size)?;
    let mut word = 0u64;
    if descriptor.little_endian {
        for &b in bytes.iter().rev() {
            word = (word << 8) | b as u64;
        }
    } else {
        for &b in bytes {
            word = (word << 8) | b as u64;
        }
    }
    if descriptor.width < 64 && word >= 1u64 << descriptor.width {
        return None;
    }
    Some(word)
}

/// Word `schema.index` of a packed leaf's `value` bytes ([`extract_word_at`]).
pub fn extract_word(schema: &PackedSchema, value: &[u8]) -> Option<u64> {
    extract_word_at(&schema.descriptor(), schema.index, value)
}

/// The canonical big-endian bytes of a `width`-bit word (what a packed
/// position's `expected` pin is compared with).
pub fn word_be_bytes(width: u32, word: u64) -> Vec<u8> {
    let size = (width as usize).div_ceil(8);
    word.to_be_bytes()[8 - size..].to_vec()
}

/// `verity_vllm.commit.hashing.H`: SHA-256 over the tag and the length-prefixed parts.
pub fn vllm_hash(tag: &str, parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((tag.len() as u32).to_be_bytes());
    hasher.update(tag.as_bytes());
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// `H("verity-vllm/leaf/v1", canonical_id, dtype, shape_text, value)`.
pub fn vllm_leaf_hash(canonical_id: &str, dtype: &str, shape_text: &str, value: &[u8]) -> [u8; 32] {
    vllm_hash(
        VLLM_LEAF_FRAMING,
        &[
            canonical_id.as_bytes(),
            dtype.as_bytes(),
            shape_text.as_bytes(),
            value,
        ],
    )
}

pub fn vllm_node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    vllm_hash(VLLM_NODE_TAG, &[left, right])
}

pub fn vllm_lift_hash(child: &[u8; 32]) -> [u8; 32] {
    vllm_hash(VLLM_LIFT_TAG, &[child])
}

/// The depth of a tree of `count` leaves (levels above the leaves).
pub fn vllm_tree_depth(count: u64) -> usize {
    let mut width = count;
    let mut depth = 0;
    while width > 1 {
        width = width.div_ceil(2);
        depth += 1;
    }
    depth
}

/// Fold a path from a leaf hash to the root, checking its shape as it goes:
/// at each level the node either has a sibling (`Some`) -- and then it is not
/// the odd trailing node -- or is lifted (`None`) -- and then it is.  A path
/// of the wrong length or shape is `None`; otherwise the root it reaches.
///
/// In the `unsound-skip-authentication` build the caller does not get here;
/// in `unsound-skip-path-fold` the leaf is hashed but never folded (the
/// caller returns `true` without calling this).
pub fn vllm_fold_path(leaf: [u8; 32], count: u64, index: u64, path: &[Option<[u8; 32]>]) -> Option<[u8; 32]> {
    if count == 0 || index >= count || path.len() != vllm_tree_depth(count) {
        return None;
    }
    let mut digest = leaf;
    let mut width = count;
    let mut cursor = index;
    for step in path {
        let lifted = cursor == width - 1 && width % 2 == 1;
        match step {
            None => {
                if !lifted {
                    return None;
                }
                digest = vllm_lift_hash(&digest);
            }
            Some(sibling) => {
                if lifted {
                    return None;
                }
                digest = if cursor % 2 == 0 {
                    vllm_node_hash(&digest, sibling)
                } else {
                    vllm_node_hash(sibling, &digest)
                };
            }
        }
        width = width.div_ceil(2);
        cursor /= 2;
    }
    Some(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_grammar() {
        let s = parse_packed_schema("packed/v1:verity-vllm/leaf/v1:u16le:6144:237").unwrap();
        assert_eq!(
            s,
            PackedSchema {
                framing: VLLM_LEAF_FRAMING,
                width: 16,
                little_endian: true,
                count: 6144,
                index: 237
            }
        );
        assert!(parse_packed_schema("packed/v1:verity-vllm/leaf/v1:u16be:1:0").is_some());
        for bad in [
            "u16",
            "packed/v1:verity-vllm/leaf/v1:u16le:6144:6144",
            "packed/v1:x:u16le:2:0",
            "packed/v1:verity-vllm/leaf/v1:u016le:2:0",
            "packed/v1:verity-vllm/leaf/v1:u65le:2:0",
            "packed/v1:verity-vllm/leaf/v1:u0le:2:0",
            "packed/v1:verity-vllm/leaf/v1:u16:2:0",
            "packed/v1:verity-vllm/leaf/v1:u16le:0:0",
            "packed/v1:verity-vllm/leaf/v1:u16le:2:00",
            "packed/v1:verity-vllm/leaf/v1:u16le:2:0:",
            "packed/v1:verity-vllm/leaf/v1:u16le:4294967296:0",
            "packed/v1:verity-vllm/leaf/v1:u16le:2:-1",
        ] {
            assert!(parse_packed_schema(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn word_extraction() {
        let s = parse_packed_schema("packed/v1:verity-vllm/leaf/v1:u16le:3:1").unwrap();
        assert_eq!(extract_word(&s, &[0x01, 0x02, 0xd1, 0xbe, 0x05, 0x06]), Some(0xbed1));
        assert_eq!(extract_word(&s, &[0x01, 0x02, 0xd1, 0xbe, 0x05]), None);
        let be = parse_packed_schema("packed/v1:verity-vllm/leaf/v1:u16be:3:1").unwrap();
        assert_eq!(extract_word(&be, &[0x01, 0x02, 0xd1, 0xbe, 0x05, 0x06]), Some(0xd1be));
        let narrow = parse_packed_schema("packed/v1:verity-vllm/leaf/v1:u4le:2:1").unwrap();
        assert_eq!(extract_word(&narrow, &[0x0f, 0x1f]), None);
        assert_eq!(extract_word(&narrow, &[0x0f, 0x0e]), Some(0x0e));
        assert_eq!(word_be_bytes(16, 0xbed1), vec![0xbe, 0xd1]);
    }

    #[test]
    fn hashing_matches_python_reference() {
        // verity_vllm.commit.hashing.H("x", b"a", b"bc") computed by the Python side
        let expected = {
            let mut h = Sha256::new();
            h.update(1u32.to_be_bytes());
            h.update(b"x");
            h.update(1u64.to_be_bytes());
            h.update(b"a");
            h.update(2u64.to_be_bytes());
            h.update(b"bc");
            let out: [u8; 32] = h.finalize().into();
            out
        };
        assert_eq!(vllm_hash("x", &[b"a", b"bc"]), expected);
    }

    /// Build the serving side's tree for `leaves` and open every index; every
    /// path folds to the root and a path of another shape is refused.
    #[test]
    fn fold_agrees_with_tree_builder() {
        for count in [1u64, 2, 3, 5, 6, 7, 37, 97] {
            let leaves: Vec<[u8; 32]> = (0..count)
                .map(|i| vllm_leaf_hash(&format!("w/{i}"), "bfloat16", "[4]", &(i as u64).to_le_bytes()))
                .collect();
            let mut levels = vec![leaves.clone()];
            while levels.last().unwrap().len() > 1 {
                let level = levels.last().unwrap();
                let mut next = Vec::new();
                for pair in level.chunks(2) {
                    next.push(if pair.len() == 2 {
                        vllm_node_hash(&pair[0], &pair[1])
                    } else {
                        vllm_lift_hash(&pair[0])
                    });
                }
                levels.push(next);
            }
            let root = levels.last().unwrap()[0];
            assert_eq!(levels.len() - 1, vllm_tree_depth(count));
            for index in 0..count {
                let mut path = Vec::new();
                let mut cursor = index as usize;
                for level in &levels[..levels.len() - 1] {
                    let sibling = cursor ^ 1;
                    path.push(level.get(sibling).copied());
                    cursor /= 2;
                }
                assert_eq!(vllm_fold_path(leaves[index as usize], count, index, &path), Some(root), "{count}/{index}");
                // a sibling turned into a lift: wrong shape, refused before any hash
                if let Some(k) = path.iter().position(|s| s.is_some()) {
                    let mut bad = path.clone();
                    bad[k] = None;
                    assert_eq!(vllm_fold_path(leaves[index as usize], count, index, &bad), None);
                }
                // a lift turned into a sibling: likewise
                if let Some(k) = path.iter().position(|s| s.is_none()) {
                    let mut bad = path.clone();
                    bad[k] = Some([0u8; 32]);
                    assert_eq!(vllm_fold_path(leaves[index as usize], count, index, &bad), None);
                }
                if !path.is_empty() {
                    let mut short = path.clone();
                    short.pop();
                    assert_eq!(vllm_fold_path(leaves[index as usize], count, index, &short), None);
                }
            }
        }
    }
}
