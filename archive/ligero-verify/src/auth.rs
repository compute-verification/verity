//! `authentication = included` (`archive/direct/ligero/auth.py`): the published boundary words are leaves of three
//! SHA-256 indexed Merkle trees of `verity.ml.commitments` (a: activations, b: weights, y: output words; one 16-bit
//! word per leaf, `schema = "u16"`, big-endian value), whose `(binding, owner, count, root)` references and the claimed
//! VU indices are an `AuthBlock` of the statement (`ligero-statement/v3`), absorbed into the statement digest
//! (`digest_bytes`).  The transcript (`ligero-proof/v2`) carries one canonical multiproof per tree; after the Ligero
//! tests the verifier recomputes every leaf from the statement's words and folds the three multiproofs to the roots.
//!
//! Hash framing (wire constants of `verity/ml/commitments/merkle.py` + `identity.py`, mirrored here from the Python):
//!
//! ~~~text
//! frame(tag, parts)   = SHA-256("veritor/protocol/merkle/frame/v3\0" | u32be(len tag) | tag | Σ (u64be(len p) | p))
//! tagged(tag, load)   = SHA-256("veritor/tagged-sha256/v1\0" | u32be(len tag) | tag | u64be(len load) | load)
//! uint(v)             = minimal big-endian bytes of v, at least one byte
//! range_identity(n)   = tagged("veritor/indexed-domain/range/v1", '{"start":0,"step":1,"stop":n}')
//! domain_id           = frame("domain", binding, uint(owner + 2), range_identity(count), uint(count))
//! leaf(rank, word)    = frame("leaf", domain_id, uint(rank), uint(rank), "u16", word as 2 BE bytes)   (position == rank)
//! node(lvl, idx, l, r)= frame("node", domain_id, uint(lvl), uint(idx), l, r)
//! ~~~
//!
//! Leaf ranks of the statement's column-major words: A/B word `o` of column `j` (step `i = j % steps` of chain
//! `v = j / steps`) is leaf `vu_index[v] * K + i * 16 + o`; the chain's output word is leaf `vu_index[v]` of tree y.

use crate::hash::Sha256;
use crate::leaf::LeafScheme;

pub const FRAME: &[u8] = b"veritor/protocol/merkle/frame/v3\0";
pub const TAGGED_PREFIX: &[u8] = b"veritor/tagged-sha256/v1\0";
pub const RANGE_TAG: &[u8] = b"veritor/indexed-domain/range/v1";
pub const VERSION: &str = "ligero-b/auth/v1";
pub const SCHEMA: &str = "u16";
pub const TREE_NAMES: [&str; 3] = ["a", "b", "y"];
/// `verity.ml.commitments.limits.VerificationLimits` defaults the Python verifier enforces.
pub const MAX_POSITIONS: u64 = 10_000_000;
pub const MAX_OPENINGS: u64 = 10_000_000;
pub const MAX_PROOF_BYTES: u64 = 64 << 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeRef {
    pub binding: [u8; 32],
    pub owner_plus_two: u8,
    pub count: u64,
    pub root: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthBlock {
    pub vu_index: Vec<u32>, // strictly increasing (checked at parse)
    pub trees: [TreeRef; 3],
}

pub fn uint(v: u64) -> Vec<u8> {
    let n = ((64 - v.leading_zeros() as usize) + 7) / 8;
    v.to_be_bytes()[8 - n.max(1)..].to_vec()
}

pub fn frame(tag: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(FRAME);
    h.update(&(tag.len() as u32).to_be_bytes());
    h.update(tag);
    for p in parts {
        h.update(&(p.len() as u64).to_be_bytes());
        h.update(p);
    }
    h.finalize()
}

pub fn tagged_sha256(tag: &[u8], payload: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TAGGED_PREFIX);
    h.update(&(tag.len() as u32).to_be_bytes());
    h.update(tag);
    h.update(&(payload.len() as u64).to_be_bytes());
    h.update(payload);
    h.finalize()
}

/// `RangeIndexedDomain(0, count).identity_digest` (canonical JSON: sorted keys, no spaces).
pub fn range_identity(count: u64) -> [u8; 32] {
    tagged_sha256(RANGE_TAG, format!("{{\"start\":0,\"step\":1,\"stop\":{count}}}").as_bytes())
}

pub fn domain_id(t: &TreeRef) -> [u8; 32] {
    let rid = range_identity(t.count);
    let owner = uint(t.owner_plus_two as u64);
    let count = uint(t.count);
    frame(b"domain", &[&t.binding[..], &owner[..], &rid[..], &count[..]])
}

pub fn leaf(domain: &[u8; 32], rank: u64, word: u16) -> [u8; 32] {
    let r = uint(rank);
    let w = word.to_be_bytes();
    frame(b"leaf", &[&domain[..], &r[..], &r[..], SCHEMA.as_bytes(), &w[..]])
}

pub fn node(domain: &[u8; 32], level: u64, index: u64, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let lv = uint(level);
    let ix = uint(index);
    frame(b"node", &[&domain[..], &lv[..], &ix[..], &left[..], &right[..]])
}

/// The pad leaf of an unused rank (`count <= rank < 2^depth`).
pub fn pad(domain: &[u8; 32], rank: u64) -> [u8; 32] {
    let r = uint(rank);
    frame(b"pad", &[&domain[..], &r[..]])
}

pub fn merkle_depth(count: u64) -> u32 {
    if count <= 1 { 0 } else { 64 - (count - 1).leading_zeros() }
}

/// What `statement_digest` absorbs (byte-identical to `auth.digest_bytes`): version, schema, the VU indices as
/// `<u4`, then per tree `|<name>:binding=<32 B>|owner=<decimal owner>|count=<decimal>|root=<32 B>`.
pub fn digest_bytes(a: &AuthBlock) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"|auth=");
    out.extend_from_slice(VERSION.as_bytes());
    out.extend_from_slice(b"|schema=");
    out.extend_from_slice(SCHEMA.as_bytes());
    out.extend_from_slice(b"|vus=");
    for v in &a.vu_index {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for (name, t) in TREE_NAMES.iter().zip(a.trees.iter()) {
        out.extend_from_slice(b"|");
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b":binding=");
        out.extend_from_slice(&t.binding);
        out.extend_from_slice(format!("|owner={}", t.owner_plus_two as i64 - 2).as_bytes());
        out.extend_from_slice(format!("|count={}", t.count).as_bytes());
        out.extend_from_slice(b"|root=");
        out.extend_from_slice(&t.root);
    }
    out
}

// ------------------------------------------------------------------------- authentication = included-hash (v5) ----

/// `hashauth.HASH_AUTH_VERSION`: the leaf is one tensor ROW's digest under the statement's leaf scheme (`leaf.rs`:
/// `LeafScheme.schema` / `params_sha256` / `digest_elems` / `leaf_bytes`), framed like the word leaves under the
/// SHA-256 range-domain.
pub const HASH_VERSION: &str = "ligero-b/auth/v2h";

/// The statement's hash-authentication block (`hashauth.HashAuth` + the three multiproofs, which a v5 statement carries
/// itself) and the published digests: per VU the x row's `digest_elems` lanes then the W column's.
#[derive(Clone, Debug)]
pub struct HashAuthBlock {
    /// The leaf scheme the relation string names (`leaf::split_relation`).
    pub leaf: &'static LeafScheme,
    pub vu_index: Vec<u32>, // strictly increasing
    pub x_index: Vec<u32>,  // the x row (leaf rank in tree a) VU v computed on
    pub w_index: Vec<u32>,  // the W column (leaf rank in tree b)
    pub trees: [TreeRef; 3],
    pub digests: Vec<u32>,  // n_vus x (2 digest_elems) canonical lanes
    pub multiproofs: Vec<Vec<[u8; 32]>>,
    /// `vllm-v1` statements (`leaf::is_vllm`): the relation's shape the ports' domains derive from; `None` = frame-v3 trees.
    pub vllm: Option<crate::vllm_v1::Shape>,
}

impl HashAuthBlock {
    /// Lanes per VU in `digests` (x then W).
    pub fn n_digest(&self) -> usize {
        2 * self.leaf.digest_elems
    }
}

/// What `statement_digest` absorbs for a hashed statement, byte-identical to `hashauth.digest_bytes` (it is appended to
/// the relation tag, not to the statement body): version, the leaf scheme's schema and parameters digest, the (vu, x, w)
/// index rows as `<u4`, then the three tree references.
pub fn hash_digest_bytes(a: &HashAuthBlock) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"|auth=");
    out.extend_from_slice(HASH_VERSION.as_bytes());
    out.extend_from_slice(b"|schema=");
    out.extend_from_slice(a.leaf.schema.as_bytes());
    out.extend_from_slice(b"|p2=");
    out.extend_from_slice(&crate::hash::unhex32(a.leaf.params_sha256));
    out.extend_from_slice(b"|idx=");
    for v in 0..a.vu_index.len() {
        out.extend_from_slice(&a.vu_index[v].to_le_bytes());
        out.extend_from_slice(&a.x_index[v].to_le_bytes());
        out.extend_from_slice(&a.w_index[v].to_le_bytes());
    }
    for (name, t) in TREE_NAMES.iter().zip(a.trees.iter()) {
        out.extend_from_slice(b"|");
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b":binding=");
        out.extend_from_slice(&t.binding);
        out.extend_from_slice(format!("|owner={}", t.owner_plus_two as i64 - 2).as_bytes());
        out.extend_from_slice(format!("|count={}", t.count).as_bytes());
        out.extend_from_slice(b"|root=");
        out.extend_from_slice(&t.root);
        if let Some(s) = &a.vllm {
            // vllm-v1 PROTOCOL.md section 9.1: the verifier-held domain of every port beside its bound root
            out.extend_from_slice(b"|domain=");
            out.extend_from_slice(&crate::vllm_v1::domain_digest(TREE_NAMES.iter().position(|n| n == name).unwrap(), t, s));
        }
    }
    out
}

/// A leaf of schema `schema` with an arbitrary big-endian value (`CommitmentDomain.leaf(rank, rank, schema, value)`).
pub fn leaf_value(domain: &[u8; 32], rank: u64, schema: &str, value: &[u8]) -> [u8; 32] {
    let r = uint(rank);
    frame(b"leaf", &[&domain[..], &r[..], &r[..], schema.as_bytes(), value])
}

/// `hashauth.layout_error`: a VU's x row and W column are fixed by its y leaf and the committed layout, never chosen by
/// the prover.  One leaf per VU in all three trees (unshared): x = W = vu; an nx x nw tile (x tree nx, W tree nw, y tree
/// nx * nw leaves): x = vu / nw, W = vu % nw.  Any other tree shape or claim is refused (red-team-standard-hash R1).
pub fn layout_error(vu_index: &[u32], x_index: &[u32], w_index: &[u32], counts: [u64; 3]) -> Option<String> {
    let [na, nb, ny] = counts;
    if x_index.len() != vu_index.len() || w_index.len() != vu_index.len() {
        return Some("auth: one (x, W, y) leaf triple per VU".into());
    }
    if vu_index.iter().any(|&v| v as u64 >= ny) {
        return Some("auth: y: leaf index out of range".into());
    }
    let tile_nw = if na == nb && nb == ny {
        None
    } else if na.checked_mul(nb) == Some(ny) {
        Some(nb)
    } else {
        return Some(format!("auth: tree counts a={na} b={nb} y={ny} are neither one leaf per VU nor an nx x nw tile"));
    };
    for v in 0..vu_index.len() {
        let vu = vu_index[v] as u64;
        let (ex, ew) = match tile_nw {
            None => (vu, vu),
            Some(nw) => (vu / nw, vu % nw),
        };
        if x_index[v] as u64 != ex || w_index[v] as u64 != ew {
            return Some("auth: a VU's x row / W column is not the one its index fixes in the committed layout".into());
        }
    }
    None
}

/// `hashauth.verify_hash_auth`: the (x, W) leaf of every VU is the one its y leaf fixes (`layout_error`); every VU's x
/// digest is the leaf at that x rank under root a, its W digest under root b, its output word (`y_bits` wide: schema
/// u16 / u32, big-endian) under root y; VUs claiming one leaf must have published one digest.  `y_words`: one per VU
/// (the chain-end columns' words).
pub fn check_hashed(a: &HashAuthBlock, y_words: &[u32], y_bits: u32, threads: usize) -> Result<(), String> {
    let n = a.vu_index.len();
    if a.multiproofs.len() != 3 {
        return Err("auth: expected one multiproof per tree".into());
    }
    let (de, nd) = (a.leaf.digest_elems, a.n_digest());
    if a.digests.len() != n * nd || y_words.len() != n {
        return Err("auth: digest / word table does not match the VU count".into());
    }
    if a.digests.iter().any(|&x| x >= crate::field::P) {
        return Err("auth: digest lane out of the field".into());
    }
    let (y_schema, y_bytes) = if y_bits <= 16 { ("u16", 2usize) } else { ("u32", 4usize) };
    if y_words.iter().any(|&w| (w as u64) >> y_bits != 0) {
        return Err("auth: y word out of range".into());
    }
    if let Some(e) = layout_error(&a.vu_index, &a.x_index, &a.w_index, [a.trees[0].count, a.trees[1].count, a.trees[2].count]) {
        return Err(e);
    }
    if crate::leaf::is_vllm(a.leaf) {
        let s = a.vllm.as_ref().ok_or("auth: a vllm-v1 statement over a relation without a vllm-v1 shape")?;
        if (y_bytes as u64) != s.y_bytes {
            return Err("auth: y value length is not the relation's".into());
        }
        return crate::vllm_v1::check(a, s, y_words, threads);
    }
    for ti in 0..3 {
        let name = TREE_NAMES[ti];
        let tree = &a.trees[ti];
        let sibs: &[[u8; 32]] = &a.multiproofs[ti][..];
        if tree.count > MAX_POSITIONS {
            return Err(format!("auth: {name}: positions {} exceed the limit {MAX_POSITIONS}", tree.count));
        }
        if 32 * sibs.len() as u64 > MAX_PROOF_BYTES {
            return Err(format!("auth: {name}: multiproof bytes exceed the limit {MAX_PROOF_BYTES}"));
        }
        let dom = domain_id(tree);
        // (rank, value) per distinct rank, ascending; a rank two VUs claim must carry one value
        let mut entries: Vec<(u64, Vec<u8>)> = Vec::with_capacity(n);
        if ti == 2 {
            for v in 0..n {
                entries.push((a.vu_index[v] as u64, y_words[v].to_be_bytes()[4 - y_bytes..].to_vec()));
            }
        } else {
            let ranks = if ti == 0 { &a.x_index } else { &a.w_index };
            let lane0 = if ti == 0 { 0 } else { de };
            let mut per: Vec<(u64, Vec<u8>)> = (0..n)
                .map(|v| (ranks[v] as u64, (a.leaf.leaf_bytes)(&a.digests[v * nd + lane0..v * nd + lane0 + de])))
                .collect();
            per.sort_by_key(|e| e.0);
            for (r, val) in per {
                match entries.last() {
                    Some((lr, lv)) if *lr == r => {
                        if lv[..] != val[..] {
                            return Err(format!("auth: {name}: VUs claiming leaf {r} published different digests"));
                        }
                    }
                    _ => entries.push((r, val)),
                }
            }
        }
        if entries.iter().any(|(r, _)| *r >= tree.count) {
            return Err(format!("auth: {name}: leaf index out of range"));
        }
        if entries.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(format!("auth: {name}: duplicate leaf index"));
        }
        if entries.len() as u64 > MAX_OPENINGS {
            return Err(format!("auth: {name}: openings {} exceed the limit {MAX_OPENINGS}", entries.len()));
        }
        let schema = if ti == 2 { y_schema } else { a.leaf.schema };
        let leaves: Vec<(u64, [u8; 32])> =
            crate::verify::par_for(entries.len(), threads, |i| (entries[i].0, leaf_value(&dom, entries[i].0, schema, &entries[i].1)));
        match fold_multiproof(&dom, tree.count, &leaves, sibs, threads) {
            Ok(root) if root == tree.root => {}
            Ok(_) => return Err(format!("auth: {name}: multiproof rejected (root mismatch)")),
            Err(e) => return Err(format!("auth: {name}: multiproof rejected ({e})")),
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Src {
    Known(usize),
    Sibling(usize),
}

/// Fold a canonical multiproof (`verity.ml.commitments.multiproof`): level by level the known nodes are visited in
/// ascending index order; an even node whose right neighbour is known is hashed with it, any other node with the
/// next proof sibling; after `merkle_depth(count)` levels one node is left and the proof must be consumed exactly.
/// `leaves` must be strictly ascending by rank, every rank `< count`.  The pairing plan of a level is fixed by the
/// ranks alone, so the hashes of a level run in parallel.
pub fn fold_multiproof(domain: &[u8; 32], count: u64, leaves: &[(u64, [u8; 32])], siblings: &[[u8; 32]], threads: usize)
                       -> Result<[u8; 32], String> {
    if leaves.is_empty() {
        return Err("no leaves".into());
    }
    if leaves.iter().any(|(r, _)| *r >= count) {
        return Err("leaf index out of range".into());
    }
    if leaves.windows(2).any(|w| w[0].0 >= w[1].0) {
        return Err("leaf indices not strictly increasing".into());
    }
    let mut nodes: Vec<(u64, [u8; 32])> = leaves.to_vec();
    let mut cursor = 0usize;
    for level in 0..merkle_depth(count) as u64 {
        // plan: (parent index, left, right) where a source is a known node (index into `nodes`) or a proof sibling
        let mut plan: Vec<(u64, Src, Src)> = Vec::with_capacity(nodes.len());
        let mut at = 0;
        while at < nodes.len() {
            let index = nodes[at].0;
            let parent = index >> 1;
            if index % 2 == 0 && at + 1 < nodes.len() && nodes[at + 1].0 == index + 1 {
                plan.push((parent, Src::Known(at), Src::Known(at + 1)));
                at += 2;
            } else {
                if cursor >= siblings.len() {
                    return Err("multiproof too short".into());
                }
                let s = Src::Sibling(cursor);
                cursor += 1;
                if index % 2 == 0 { plan.push((parent, Src::Known(at), s)) } else { plan.push((parent, s, Src::Known(at))) }
                at += 1;
            }
        }
        let known = &nodes;
        let get = move |s: Src| -> [u8; 32] {
            match s {
                Src::Known(i) => known[i].1,
                Src::Sibling(c) => siblings[c],
            }
        };
        let parents: Vec<(u64, [u8; 32])> = crate::verify::par_for(plan.len(), threads, |i| {
            let (parent, l, r) = plan[i];
            (parent, node(domain, level, parent, &get(l), &get(r)))
        });
        nodes = parents;
    }
    if cursor != siblings.len() {
        return Err("multiproof too long".into());
    }
    debug_assert!(nodes.len() == 1 && nodes[0].0 == 0);
    Ok(nodes[0].1)
}

/// The authentication check: every published word of the `n_vus * steps` real columns (A, B) and every chain's
/// output word (y) is a leaf at its claimed index under the statement's roots.  `a`, `b`: `l x k_ops` words
/// column-major; `y16`: `l` words (the chain-end columns carry the output words).
pub fn check(auth: &AuthBlock, multiproofs: &[Vec<[u8; 32]>], a: &[u16], b: &[u16], y16: &[u16], n_vus: usize, steps: usize,
             k_ops: usize, threads: usize) -> Result<(), String> {
    if multiproofs.len() != 3 {
        return Err("auth: expected one multiproof per tree".into());
    }
    if auth.vu_index.len() != n_vus {
        return Err("auth: authentication block does not match the layout".into());
    }
    let k = (steps * k_ops) as u64;
    let real = n_vus * steps;
    if a.len() < real * k_ops || b.len() < real * k_ops || y16.len() < real {
        return Err("auth: statement words shorter than the layout".into());
    }
    // ranks per tree, ascending (vu_index strictly increasing => the ranks are strictly increasing in column order)
    let mut ab_ranks: Vec<u64> = Vec::with_capacity(real * k_ops);
    for &vu in auth.vu_index.iter() {
        for i in 0..steps {
            let base = vu as u64 * k + (i * k_ops) as u64;
            for o in 0..k_ops as u64 {
                ab_ranks.push(base + o);
            }
        }
    }
    let y_ranks: Vec<u64> = auth.vu_index.iter().map(|&v| v as u64).collect();
    let y_words: Vec<u16> = (0..n_vus).map(|v| y16[v * steps + steps - 1]).collect();
    for ti in 0..3 {
        let name = TREE_NAMES[ti];
        let tree = &auth.trees[ti];
        let ranks: &[u64] = if ti < 2 { &ab_ranks[..] } else { &y_ranks[..] };
        let words: &[u16] = match ti {
            0 => &a[..real * k_ops],
            1 => &b[..real * k_ops],
            _ => &y_words[..],
        };
        let sibs: &[[u8; 32]] = &multiproofs[ti][..];
        if tree.count > MAX_POSITIONS {
            return Err(format!("auth: {name}: positions {} exceed the limit {MAX_POSITIONS}", tree.count));
        }
        if ranks.len() as u64 > MAX_OPENINGS {
            return Err(format!("auth: {name}: openings {} exceed the limit {MAX_OPENINGS}", ranks.len()));
        }
        if 32 * sibs.len() as u64 > MAX_PROOF_BYTES {
            return Err(format!("auth: {name}: multiproof bytes exceed the limit {MAX_PROOF_BYTES}"));
        }
        if ranks.iter().any(|&r| r >= tree.count) {
            return Err(format!("auth: {name}: leaf index out of range"));
        }
        if ranks.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!("auth: {name}: duplicate leaf index"));
        }
        let dom = domain_id(tree);
        let leaves: Vec<(u64, [u8; 32])> =
            crate::verify::par_for(ranks.len(), threads, |i| (ranks[i], leaf(&dom, ranks[i], words[i])));
        match fold_multiproof(&dom, tree.count, &leaves, sibs, threads) {
            Ok(root) if root == tree.root => {}
            Ok(_) => return Err(format!("auth: {name}: multiproof rejected (root mismatch)")),
            Err(e) => return Err(format!("auth: {name}: multiproof rejected ({e})")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{hex, sha256};

    fn h(s: &str) -> [u8; 32] {
        let mut o = [0u8; 32];
        for i in 0..32 {
            o[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
        }
        o
    }

    // Pinned from the Python reference (verity.ml.commitments.MerkleTree / multiproof on auth.build_word_tree):
    // words [0x3f80, 0x0000, 0xbf80, 0x4000, 0x7f7f], binding = bytes(range(32)), owner -1 (tree "a"), count 5.
    const BINDING: [u8; 32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
    const WORDS: [u16; 5] = [0x3f80, 0x0000, 0xbf80, 0x4000, 0x7f7f];
    const ROOT: &str = "7dc53a78722230c27619305152c2f8008ae19cb2cb8418ebc670f9bb8379e7b8";

    fn tree_a() -> TreeRef {
        TreeRef { binding: BINDING, owner_plus_two: 1, count: 5, root: h(ROOT) }
    }

    #[test]
    fn uint_is_minimal_big_endian() {
        assert_eq!(uint(0), vec![0]);
        assert_eq!(uint(1), vec![1]);
        assert_eq!(uint(255), vec![255]);
        assert_eq!(uint(256), vec![1, 0]);
        assert_eq!(uint(6_291_456), vec![0x60, 0, 0]);
    }

    #[test]
    fn range_identity_and_domain_match_python() {
        assert_eq!(hex(&range_identity(5)), "cac30b672f5392ae0261a286d659ca9ca0d9c0f4a71f47dfbde289a3d2d75164");
        assert_eq!(hex(&domain_id(&tree_a())), "152b53bdcd4ff2e59540a24fe076d904ad95d15146b4ceb0cacf5e034fa9cc67");
    }

    #[test]
    fn leaves_nodes_and_root_match_python() {
        let t = tree_a();
        let dom = domain_id(&t);
        let want = [
            "53c94bb782bf50fbb1b994d1a456cf6a5e29f4d1bed63050f60e05d6082d213b",
            "84232b121e122a7b9a194f56bc56b91d14724db5d4035cdcc5376405a72a58f4",
            "bad8e6963d4f80157eaa33bd844fefbd60e2b8f549c870ada02b4b38a8396c23",
            "d67978e6fdfa4242be4f5065b7bb2733ce7b55b9baefa6b962181f04e60795b9",
            "f51e8e31d728520f0a4b96a431bbc4c60078ed518132071f2016c1238ffa4b5f",
        ];
        for (r, w) in want.iter().enumerate() {
            assert_eq!(hex(&leaf(&dom, r as u64, WORDS[r])), *w, "leaf {r}");
        }
        // the pad leaf (rank 5) and level 1 node 0
        assert_eq!(hex(&pad(&dom, 5)), "8be67fa2ecf6b0a6886509e55a8332fb8b08436e4fa910e5deba8c8595bb6ce8");
        assert_eq!(hex(&node(&dom, 0, 0, &h(want[0]), &h(want[1]))), "516dbaaf3502018e6e7d0f4591fb0f7e2ace0ffff2cc23be3ce8a7f76512a169");
        // full multiproof of ranks {1, 3} as verity.ml.commitments.multiproof emits it
        let sibs = [
            h("53c94bb782bf50fbb1b994d1a456cf6a5e29f4d1bed63050f60e05d6082d213b"),
            h("bad8e6963d4f80157eaa33bd844fefbd60e2b8f549c870ada02b4b38a8396c23"),
            h("1ece3166a87f4bc8ae47a0da8d594bf2abaa456973a3940ffadb5a940d2eae99"),
        ];
        let leaves = [(1u64, leaf(&dom, 1, WORDS[1])), (3u64, leaf(&dom, 3, WORDS[3]))];
        for threads in [1, 4] {
            assert_eq!(hex(&fold_multiproof(&dom, 5, &leaves, &sibs, threads).unwrap()), ROOT);
        }
        // negatives: wrong word, wrong index, truncated / extended proof, unsorted leaves
        let bad = [(1u64, leaf(&dom, 1, WORDS[1] ^ 1)), (3u64, leaf(&dom, 3, WORDS[3]))];
        assert_ne!(hex(&fold_multiproof(&dom, 5, &bad, &sibs, 1).unwrap()), ROOT);
        let wrong_idx = [(1u64, leaf(&dom, 1, WORDS[1])), (4u64, leaf(&dom, 4, WORDS[4]))];
        assert!(fold_multiproof(&dom, 5, &wrong_idx, &sibs, 1).map(|r| hex(&r) != ROOT).unwrap_or(true));
        assert!(fold_multiproof(&dom, 5, &leaves, &sibs[..2], 1).is_err());
        let mut longer = sibs.to_vec();
        longer.push(sibs[0]);
        assert!(fold_multiproof(&dom, 5, &leaves, &longer, 1).is_err());
        let unsorted = [leaves[1], leaves[0]];
        assert!(fold_multiproof(&dom, 5, &unsorted, &sibs, 1).is_err());
        assert!(fold_multiproof(&dom, 5, &[(5u64, leaves[0].1)], &sibs, 1).is_err());
    }

    #[test]
    fn digest_bytes_match_python() {
        // auth.digest_bytes of AuthStatement.for_vus({a, b, y}, [0, 1]) with b: binding [1]*32 (owner -2) over the same
        // five words, y: binding [2]*32 (owner -1) over the first two words; sha256 of the 340 bytes pinned.
        let blk = AuthBlock {
            vu_index: vec![0, 1],
            trees: [
                tree_a(),
                TreeRef { binding: [1; 32], owner_plus_two: 0, count: 5, root: h("5ae6c7331b39619769df51b123bb9c7a334b6f23e3c67f03a9de201241a58256") },
                TreeRef { binding: [2; 32], owner_plus_two: 1, count: 2, root: h("d5dcb571025cec7c4ca779bf71aa8030eaf52aaad344ff299d6c3dab2a9b5562") },
            ],
        };
        let d = digest_bytes(&blk);
        assert_eq!(d.len(), 340);
        assert_eq!(hex(&sha256(&d)), "e79b8500b819c06a4cbc6bbdfc6e595f4796690fef0974020cfdb120f0f26cef");
    }

    #[test]
    fn check_positions_and_roots() {
        // a 2-VU statement with K = 4 (steps = 2, k_ops = 2) over tiny trees built here; y tree over 3 chains
        let a_words: Vec<u16> = (0..12u16).map(|i| 0x3f80 + i).collect();
        let b_words: Vec<u16> = (0..12u16).map(|i| 0x4000 + i).collect();
        let y_words: Vec<u16> = vec![0x3c00, 0x3d00, 0x3e00];
        fn build(binding: [u8; 32], owner_plus_two: u8, words: &[u16]) -> (TreeRef, Vec<Vec<[u8; 32]>>) {
            let count = words.len() as u64;
            let t0 = TreeRef { binding, owner_plus_two, count, root: [0; 32] };
            let dom = domain_id(&t0);
            let depth = merkle_depth(count);
            let width = 1usize << depth;
            let mut level: Vec<[u8; 32]> = (0..width)
                .map(|r| if (r as u64) < count { leaf(&dom, r as u64, words[r]) } else { pad(&dom, r as u64) })
                .collect();
            let mut levels = vec![level.clone()];
            for lv in 0..depth as u64 {
                let next: Vec<[u8; 32]> =
                    (0..level.len() / 2).map(|i| node(&dom, lv, i as u64, &level[2 * i], &level[2 * i + 1])).collect();
                level = next;
                levels.push(level.clone());
            }
            (TreeRef { root: level[0], ..t0 }, levels)
        }
        fn multiproof(levels: &[Vec<[u8; 32]>], ranks: &[u64]) -> Vec<[u8; 32]> {
            let mut idx: Vec<u64> = ranks.to_vec();
            let mut out = Vec::new();
            for lv in 0..levels.len() - 1 {
                let mut at = 0;
                let mut next: Vec<u64> = Vec::new();
                while at < idx.len() {
                    let i = idx[at];
                    if i % 2 == 0 && at + 1 < idx.len() && idx[at + 1] == i + 1 {
                        at += 2;
                    } else {
                        out.push(levels[lv][(i ^ 1) as usize]);
                        at += 1;
                    }
                    next.push(i >> 1);
                }
                next.dedup();
                idx = next;
            }
            out
        }
        let (ta, la) = build([7; 32], 1, &a_words);
        let (tb, lb) = build([8; 32], 0, &b_words);
        let (ty, ly) = build([9; 32], 1, &y_words);
        // VUs 0 and 2 (K = 4 words each: ranks 0..4 and 8..12), chains 0 and 2 of y
        let blk = AuthBlock { vu_index: vec![0, 2], trees: [ta.clone(), tb.clone(), ty.clone()] };
        let ab_ranks: Vec<u64> = vec![0, 1, 2, 3, 8, 9, 10, 11];
        let mp = vec![multiproof(&la, &ab_ranks), multiproof(&lb, &ab_ranks), multiproof(&ly, &[0, 2])];
        // statement words: l = 8 columns (4 real: VU0 step0, VU0 step1, VU2 step0, VU2 step1), k_ops = 2
        let mut a = vec![0u16; 16];
        let mut b = vec![0u16; 16];
        a[..4].copy_from_slice(&a_words[0..4]);
        a[4..8].copy_from_slice(&a_words[8..12]);
        b[..4].copy_from_slice(&b_words[0..4]);
        b[4..8].copy_from_slice(&b_words[8..12]);
        let mut y16 = vec![0u16; 8];
        y16[1] = y_words[0];
        y16[3] = y_words[2];
        assert_eq!(check(&blk, &mp, &a, &b, &y16, 2, 2, 2, 2), Ok(()));
        // a word not under the root
        let mut a2 = a.clone();
        a2[5] ^= 1;
        assert!(check(&blk, &mp, &a2, &b, &y16, 2, 2, 2, 1).unwrap_err().starts_with("auth: a: multiproof rejected"));
        // wrong index (claims VU 1 instead of VU 2), right words
        let blk2 = AuthBlock { vu_index: vec![0, 1], ..blk.clone() };
        assert!(check(&blk2, &mp, &a, &b, &y16, 2, 2, 2, 1).is_err());
        // right words, different root
        let mut blk3 = blk.clone();
        blk3.trees[2].root[0] ^= 1;
        assert!(check(&blk3, &mp, &a, &b, &y16, 2, 2, 2, 1).unwrap_err().starts_with("auth: y: multiproof rejected"));
        // truncated / extended multiproof, index out of range
        let mut mp2 = mp.clone();
        mp2[0].pop();
        assert!(check(&blk, &mp2, &a, &b, &y16, 2, 2, 2, 1).is_err());
        let mut mp3 = mp.clone();
        mp3[1].push([0; 32]);
        assert!(check(&blk, &mp3, &a, &b, &y16, 2, 2, 2, 1).is_err());
        let blk4 = AuthBlock { vu_index: vec![0, 3], ..blk.clone() };
        assert_eq!(check(&blk4, &mp, &a, &b, &y16, 2, 2, 2, 1).unwrap_err(), "auth: a: leaf index out of range");
    }

    /// The hashed fixture's authentication block (`fixtures/fp8-ada-hash/sub_00.stmt`, 20 VUs on 20 x rows / 20 W
    /// columns / 20 y words) folds to its roots; every single-field mutation the statement-level tests cannot
    /// isolate (the statement digest binds the block, so a mutated FILE fails on the challenges first) is refused
    /// here by the path check itself.
    #[test]
    fn hashed_fixture_block_folds_and_every_mutation_is_refused() {
        let st = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/fp8-ada-hash/sub_00.stmt")).unwrap();
        let st = crate::format::Statement::parse(&st).unwrap();
        let ha = st.hash_auth.clone().expect("v5 block");
        let y: Vec<u32> = (0..st.n_vus).map(|v| st.y16[v * st.steps + st.steps - 1]).collect();
        assert_eq!((ha.vu_index.len(), ha.trees[0].count, ha.trees[1].count, ha.trees[2].count), (20, 20, 20, 20));
        assert_eq!((ha.leaf.name, ha.digests.len()), ("poseidon2", 20 * ha.n_digest()));
        check_hashed(&ha, &y, 22, 1).expect("the fixture block");
        check_hashed(&ha, &y, 22, 4).expect("the fixture block, 4 threads");
        let refuse = |what: &str, f: &dyn Fn(&mut HashAuthBlock, &mut Vec<u32>), expect: &str| {
            let (mut b, mut yy) = (ha.clone(), y.clone());
            f(&mut b, &mut yy);
            let e = check_hashed(&b, &yy, 22, 1).expect_err(what);
            assert!(e.contains(expect), "{what}: {e}");
        };
        refuse("x digest lane", &|b, _| b.digests[0] ^= 1, "auth: a: multiproof rejected (root mismatch)");
        refuse("W digest lane", &|b, _| b.digests[3 * 16 + 8] ^= 1, "auth: b: multiproof rejected (root mismatch)");
        refuse("x sibling", &|b, _| b.multiproofs[0][0][5] ^= 1, "auth: a: multiproof rejected (root mismatch)");
        refuse("W sibling dropped", &|b, _| { b.multiproofs[1].pop(); }, "auth: b: multiproof rejected");
        refuse("x root", &|b, _| b.trees[0].root[0] ^= 1, "auth: a: multiproof rejected (root mismatch)");
        refuse("y root", &|b, _| b.trees[2].root[31] ^= 1, "auth: y: multiproof rejected (root mismatch)");
        refuse("tree binding", &|b, _| b.trees[1].binding[0] ^= 1, "auth: b: multiproof rejected (root mismatch)");
        const LAYOUT: &str = "is not the one its index fixes in the committed layout";
        refuse("swapped W column", &|b, _| b.w_index.swap(0, 1), LAYOUT);
        refuse("x row of another VU", &|b, _| b.x_index[0] = 5, LAYOUT);
        // red-team-standard-hash R1: VU 0 claims VU 1's x row AND W column (VU 1's output under an honest x / W pair)
        refuse("VU 0 on VU 1's (x, W)", &|b, _| { b.x_index[0] = 1; b.w_index[0] = 1; }, LAYOUT);
        refuse("y word", &|_, yy| yy[7] ^= 1, "auth: y: multiproof rejected (root mismatch)");
        refuse("y word out of range", &|_, yy| yy[0] = 1 << 22, "auth: y word out of range");
        refuse("leaf beyond the count", &|b, _| b.x_index[19] = 20, LAYOUT);
        refuse("x tree of another shape", &|b, _| b.trees[0].count = 19, "neither one leaf per VU nor an nx x nw tile");
        refuse("one multiproof short", &|b, _| { b.multiproofs.pop(); }, "expected one multiproof per tree");
        refuse("VU dropped", &|b, _| { b.vu_index.pop(); }, "does not match the VU count");
    }

    #[test]
    fn layout_fixes_x_and_w_from_the_vu_index() {
        // unshared: x = W = vu
        assert_eq!(layout_error(&[0, 2, 5], &[0, 2, 5], &[0, 2, 5], [6, 6, 6]), None);
        assert!(layout_error(&[0, 1], &[1, 1], &[1, 1], [2, 2, 2]).unwrap().contains("committed layout"));
        assert!(layout_error(&[0, 1], &[0, 1], &[1, 0], [2, 2, 2]).unwrap().contains("committed layout"));
        // a 2 x 3 tile: x = vu / 3, W = vu % 3 (relchain.tile_indices)
        assert_eq!(layout_error(&[3, 4, 5], &[1, 1, 1], &[0, 1, 2], [2, 3, 6]), None);
        assert!(layout_error(&[3, 4, 5], &[1, 1, 1], &[0, 2, 2], [2, 3, 6]).unwrap().contains("committed layout"));
        assert!(layout_error(&[3], &[0], &[1], [3, 2, 6]).unwrap().contains("committed layout"));
        // one VU: both readings agree
        assert_eq!(layout_error(&[0], &[0], &[0], [1, 1, 1]), None);
        // shapes that are neither, indices past y, ragged triples
        assert!(layout_error(&[0], &[0], &[0], [2, 3, 5]).unwrap().contains("neither"));
        assert!(layout_error(&[6], &[2], &[0], [2, 3, 6]).unwrap().contains("y: leaf index out of range"));
        assert!(layout_error(&[0, 1], &[0], &[0, 1], [2, 2, 2]).unwrap().contains("one (x, W, y) leaf triple per VU"));
        assert_eq!(layout_error(&[], &[], &[], [0, 0, 0]), None);
    }
}
