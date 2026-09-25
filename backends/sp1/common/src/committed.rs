//! `relation-committed/v1` (feature `relation-committed`): the Table 2 full relation on a batch of K = 1536 VUs of one
//! numerical format, with the operands bound to frame-v3 SHA-256 trees over `sha256/row/v1` row leaves.  Shared by the
//! guest (`guest/`, built with `--features relation-committed`) and the host's verifier, which checks the trees natively.
//!
//! **Statement.**  For every `i < B`: `y[i]` is the output word of VU `i`'s chain under `format` ([`bare::check_pair`]),
//! AND `dx[i]` / `dw[i]` are the `sha256/row/v1` digests of the x row and the W column VU `i` computed on.  The rows are
//! private.  The guest commits `y`, `dx` and `dw` as public values; the verifier builds three frame-v3 trees from them --
//! `a` (x rows, owner -1), `b` (W columns, owner -2), `y` (output words, owner -1) -- and accepts only if their roots are
//! the statement's.  So the proof shows that the committed outputs are the subcircuit applied to the committed inputs and
//! weights, and the verifier never sees an operand word.
//!
//! **Row leaf** (`verity.commitments.rowleaf`, PR #15 `sha256/row/v1`):
//!
//! ```text
//! prefix = ("verity/sha256-row/v1\0" || u8 role || u8 word_bits || u32be n_words) zero-padded to 64 bytes
//! digest = SHA-256(prefix || row_bytes)          role 1 = an x row, role 2 = a W column
//! tree leaf = frame-v3 leaf(domain, rank, rank, "sha256/row/v1", digest)      (computed by the verifier)
//! ```
//!
//! `row_bytes` are the row's bytes as the frozen set stores them: BF16 words as u16 little-endian (`word_bits` 16), E4M3
//! bytes (8), and the NVFP4 row's 24 firings of 64 E2M1 code bytes and 4 UE4M3 scale bytes as 1632 bytes (8).  The prefix is
//! one SHA-256 block, so the guest absorbs it once and clones the midstate per row.
//!
//! **Output leaves**: frame-v3 word leaves, schema `u16` (2 bytes big-endian, BF16 formats) or `u32` (4 bytes, the FP32
//! accumulator formats), as B-Ligero's `hashauth.word_schema`.
//!
//! **Domains**: `CommitmentDomain(binding, owner, RangeIndexedDomain(0, B))`, with `binding` =
//! `identity_digest("verity/ligero-b/auth-binding/v2h", {dataset, tier, manifest_sha256, lo, hi, K, tree, schema})` -- the
//! convention of B-Ligero's hashed modes (`backends/direct/ligero/hashauth.py`), so any backend that binds this scheme over
//! the same range of a frozen set has the same roots.
//!
//! **Public values** (chunked input layout only):
//!
//! ```text
//! head = "verity/sp1/relation-committed/v1" || u32le(format) || id || u32le(K) || u32le(B) || u8 verdict
//! accept (verdict 1): head || y[0..B] (y_bytes each, LE) || dx[0..B] (32 each) || dw[0..B] (32 each)
//! reject (verdict 0): head only; a malformed header gives format 0, id 0, K 0, B 0
//! ```
//!
//! `id = sha256("<dataset>|<tier>|<lo>|<hi>|<manifest_sha256>")` ([`bare::identity_digest`]) is carried through: the
//! verifier compares it with its own.

use sha2::{Digest, Sha256};

use crate::bare::{self, Format};
use crate::frame;

pub const TAG: &[u8] = b"verity/sp1/relation-committed/v1";
pub const HEAD_LEN: usize = TAG.len() + 4 + 32 + 4 + 4 + 1;
pub const ROW_SCHEMA: &str = "sha256/row/v1";
pub const ROW_TAG: &[u8] = b"verity/sha256-row/v1\0";
pub const ROLE_X: u8 = 1;
pub const ROLE_W: u8 = 2;
pub const BINDING_TAG: &str = "verity/ligero-b/auth-binding/v2h";
pub const RANGE_DOMAIN_TAG: &str = "veritor/indexed-domain/range/v1";
pub const PAD_TAG: &[u8] = b"pad";
pub const TREE_NAMES: [&str; 3] = ["a", "b", "y"];
/// `owner` of trees `a`, `b`, `y`: -1 (boundary), -2 (weights), -1.
pub const OWNERS: [i64; 3] = [-1, -2, -1];

/// The 64-byte first block of a `sha256/row/v1` digest.
pub fn row_prefix(role: u8, word_bits: u8, n_words: u32) -> [u8; 64] {
    let mut out = [0u8; 64];
    out[..ROW_TAG.len()].copy_from_slice(ROW_TAG);
    let at = ROW_TAG.len();
    out[at] = role;
    out[at + 1] = word_bits;
    out[at + 2..at + 6].copy_from_slice(&n_words.to_be_bytes());
    out
}

/// `sha256/row/v1` of one row (the one-shot reference).
pub fn row_digest(role: u8, word_bits: u8, row: &[u8]) -> [u8; 32] {
    let n_words = (row.len() * 8 / word_bits as usize) as u32;
    let mut h = Sha256::new();
    h.update(row_prefix(role, word_bits, n_words));
    h.update(row);
    h.finalize().into()
}

/// The word width a format's rows are committed at.
pub const fn word_bits(format: Format) -> u8 {
    match format {
        Format::Bf16Ampere | Format::Bf16Hopper => 16,
        Format::Fp8Hopper | Format::Fp8Ada | Format::Fp4Nvf4 => 8,
    }
}

/// Words per row at [`word_bits`].
pub const fn n_words(format: Format) -> u32 {
    (format.row_bytes() * 8 / word_bits(format) as usize) as u32
}

/// The two row hashers of a format with their prefix block already compressed; each row clones one.
#[derive(Clone)]
pub struct RowHasher {
    x: Sha256,
    w: Sha256,
}

impl RowHasher {
    pub fn new(format: Format) -> RowHasher {
        let (bits, n) = (word_bits(format), n_words(format));
        let start = |role| {
            let mut h = Sha256::new();
            h.update(row_prefix(role, bits, n));
            h
        };
        RowHasher { x: start(ROLE_X), w: start(ROLE_W) }
    }

    pub fn x(&self, row: &[u8]) -> [u8; 32] {
        let mut h = self.x.clone();
        h.update(row);
        h.finalize().into()
    }

    pub fn w(&self, row: &[u8]) -> [u8; 32] {
        let mut h = self.w.clone();
        h.update(row);
        h.finalize().into()
    }
}

/// Append the x-row and W-column digests of every VU of a chunk (its VUs' x row then W row, `row_bytes` each).
pub fn digest_chunk(hasher: &RowHasher, format: Format, chunk: &[u8], dx: &mut Vec<u8>, dw: &mut Vec<u8>) {
    let rb = format.row_bytes();
    for vu in chunk.chunks_exact(2 * rb) {
        let (x, w) = vu.split_at(rb);
        dx.extend_from_slice(&hasher.x(x));
        dw.extend_from_slice(&hasher.w(w));
    }
}

/// The head of the public values.
pub fn head(format_id: u32, id: &[u8; 32], k: u32, b: u32, verdict: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEAD_LEN);
    out.extend_from_slice(TAG);
    out.extend_from_slice(&format_id.to_le_bytes());
    out.extend_from_slice(id);
    out.extend_from_slice(&k.to_le_bytes());
    out.extend_from_slice(&b.to_le_bytes());
    out.push(verdict);
    out
}

/// What the guest commits for a header and its chunks (the host's expectation and the tests' reference); it reads and
/// checks the chunks as the guest does: the first failing chunk rejects.
pub fn decide(header: &[u8], buffers: &[Vec<u8>]) -> Vec<u8> {
    let Some(h) = bare::parse_header(header) else {
        return head(0, &[0u8; 32], 0, 0, bare::VERDICT_REJECT);
    };
    let reject = || head(h.format.id(), &h.id, h.k, h.b, bare::VERDICT_REJECT);
    let (b, per, yb) = (h.b as usize, h.vus_per_read as usize, h.format.y_bytes());
    if h.version != bare::VERSION || h.k as usize != bare::K || per == 0 || buffers.len() != b.div_ceil(per) {
        return reject();
    }
    let hasher = RowHasher::new(h.format);
    let (mut dx, mut dw) = (Vec::with_capacity(32 * b), Vec::with_capacity(32 * b));
    let mut i = 0;
    for chunk in buffers {
        let n = per.min(b - i);
        let ok = chunk.len() == 8 * n * h.format.vu_words()
            && bare::check_words(h.format, chunk, &bare::words_of(chunk), &h.y_le[yb * i..yb * (i + n)]);
        if !ok {
            return reject();
        }
        digest_chunk(&hasher, h.format, chunk, &mut dx, &mut dw);
        i += n;
    }
    let mut out = head(h.format.id(), &h.id, h.k, h.b, bare::VERDICT_ACCEPT);
    out.extend_from_slice(h.y_le);
    out.extend_from_slice(&dx);
    out.extend_from_slice(&dw);
    out
}

/// Parsed public values; the three byte slices are empty on a reject.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Public<'a> {
    pub format_id: u32,
    pub id: [u8; 32],
    pub k: u32,
    pub b: u32,
    pub verdict: u8,
    pub y_le: &'a [u8],
    pub dx: &'a [u8],
    pub dw: &'a [u8],
}

impl Public<'_> {
    pub fn format(&self) -> Option<Format> {
        Format::from_id(self.format_id)
    }
}

/// Parse public values; `None` on a wrong tag or a length that is not the verdict's.
pub fn parse_public(bytes: &[u8]) -> Option<Public<'_>> {
    if bytes.len() < HEAD_LEN || &bytes[..TAG.len()] != TAG {
        return None;
    }
    let at = TAG.len();
    let u32_at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let format_id = u32_at(at);
    let mut id = [0u8; 32];
    id.copy_from_slice(&bytes[at + 4..at + 36]);
    let (k, b, verdict) = (u32_at(at + 36), u32_at(at + 40), bytes[at + 44]);
    let rest = &bytes[HEAD_LEN..];
    let empty: &[u8] = &[];
    if verdict != bare::VERDICT_ACCEPT {
        return rest.is_empty().then_some(Public { format_id, id, k, b, verdict, y_le: empty, dx: empty, dw: empty });
    }
    let yb = Format::from_id(format_id)?.y_bytes();
    let b_us = b as usize;
    if rest.len() as u64 != b as u64 * (yb as u64 + 64) {
        return None;
    }
    let (y_le, rest) = rest.split_at(yb * b_us);
    let (dx, dw) = rest.split_at(32 * b_us);
    Some(Public { format_id, id, k, b, verdict, y_le, dx, dw })
}

// --- the trees -------------------------------------------------------------------------------------------------------

fn uint_bytes(value: u64) -> Vec<u8> {
    frame::uint(value)[8 - frame::uint_len(value)..].to_vec()
}

/// `RangeIndexedDomain(0, count).identity_digest`.
pub fn range_identity(count: u64) -> [u8; 32] {
    frame::tagged_sha256(RANGE_DOMAIN_TAG.as_bytes(), format!("{{\"start\":0,\"step\":1,\"stop\":{count}}}").as_bytes())
}

/// `CommitmentDomain(binding, owner, RangeIndexedDomain(0, count)).domain_id`.
pub fn domain(binding: &[u8; 32], owner: i64, count: u64) -> [u8; 32] {
    frame::domain_id(binding, (owner + 2) as u64, &range_identity(count), count)
}

/// A JSON string as Python's `json.dumps(..., ensure_ascii=False)` writes it, for the strings a binding carries; `None`
/// for a string that would need an escape.
fn json_str(s: &str) -> Option<String> {
    (!s.chars().any(|c| c == '"' || c == '\\' || (c as u32) < 0x20)).then(|| format!("\"{s}\""))
}

/// The binding of tree `tree` (`a`, `b`, `y`) over tier ids `[lo, hi)` of a frozen set, under leaf schema `schema`:
/// `identity_digest(BINDING_TAG, {dataset, tier, manifest_sha256, lo, hi, K, tree, schema})` (canonical JSON, sorted keys).
#[allow(clippy::too_many_arguments)]
pub fn binding_digest(dataset: &str, tier: &str, manifest_sha256: &str, lo: u64, hi: u64, k: u64, tree: &str, schema: &str) -> Option<[u8; 32]> {
    let payload = format!(
        "{{\"K\":{k},\"dataset\":{},\"hi\":{hi},\"lo\":{lo},\"manifest_sha256\":{},\"schema\":{},\"tier\":{},\"tree\":{}}}",
        json_str(dataset)?,
        json_str(manifest_sha256)?,
        json_str(schema)?,
        json_str(tier)?,
        json_str(tree)?
    );
    Some(frame::tagged_sha256(BINDING_TAG.as_bytes(), payload.as_bytes()))
}

/// The `y` leaves' word schema and width: `u16` (2 bytes) for the BF16 formats, `u32` (4 bytes) otherwise.
pub const fn word_schema(format: Format) -> (&'static str, usize) {
    if format.y_bytes() <= 2 {
        ("u16", 2)
    } else {
        ("u32", 4)
    }
}

/// The leaf schema of tree `name` for `format`.
pub fn tree_schema(format: Format, name: &str) -> &'static str {
    if name == "y" {
        word_schema(format).0
    } else {
        ROW_SCHEMA
    }
}

/// The frame-v3 root over `leaves`: `pad` leaves up to a power of two, then domain-, level- and index-bound nodes.
pub fn root(domain: &[u8; 32], mut leaves: Vec<[u8; 32]>) -> [u8; 32] {
    if leaves.is_empty() {
        return frame::empty_root(domain);
    }
    let width = 1usize << frame::merkle_depth(leaves.len() as u64);
    for rank in leaves.len()..width {
        leaves.push(frame::framed_hash(PAD_TAG, &[domain, &uint_bytes(rank as u64)]));
    }
    let mut level = 0u64;
    while leaves.len() > 1 {
        leaves = leaves.chunks_exact(2).enumerate().map(|(i, p)| frame::node(domain, level, i as u64, &p[0], &p[1])).collect();
        level += 1;
    }
    leaves[0]
}

/// The row leaves over concatenated 32-byte row digests (rank = position = index).
pub fn row_leaves(domain: &[u8; 32], digests: &[u8]) -> Vec<[u8; 32]> {
    digests.chunks_exact(32).enumerate().map(|(r, d)| frame::leaf(domain, r as u64, r as u64, ROW_SCHEMA.as_bytes(), d)).collect()
}

/// The word leaves over the claimed words (`y_bytes` little-endian each), each written big-endian at the schema's width.
pub fn y_leaves(domain: &[u8; 32], format: Format, y_le: &[u8]) -> Vec<[u8; 32]> {
    let (schema, nbytes) = word_schema(format);
    y_le.chunks_exact(format.y_bytes())
        .enumerate()
        .map(|(r, y)| {
            let mut word = [0u8; 4];
            word[..y.len()].copy_from_slice(y);
            let be = u32::from_le_bytes(word).to_be_bytes();
            frame::leaf(domain, r as u64, r as u64, schema.as_bytes(), &be[4 - nbytes..])
        })
        .collect()
}

/// The public reference of one tree: what the verifier needs to rebuild its domain and check its root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeRef {
    pub binding: [u8; 32],
    pub owner: i64,
    pub count: u64,
    pub root: [u8; 32],
}

/// The three bindings (`a`, `b`, `y`) of a statement over tier ids `[lo, hi)`.
pub fn bindings(format: Format, dataset: &str, tier: &str, manifest_sha256: &str, lo: u64, hi: u64) -> Option<[[u8; 32]; 3]> {
    let one = |i: usize| binding_digest(dataset, tier, manifest_sha256, lo, hi, bare::K as u64, TREE_NAMES[i], tree_schema(format, TREE_NAMES[i]));
    Some([one(0)?, one(1)?, one(2)?])
}

/// The roots of trees `a`, `b`, `y` from the digests and words (`dx`, `dw`: 32 bytes per VU; `y_le`: `y_bytes` per VU).
pub fn roots(format: Format, bindings: &[[u8; 32]; 3], y_le: &[u8], dx: &[u8], dw: &[u8]) -> [TreeRef; 3] {
    let count = (y_le.len() / format.y_bytes()) as u64;
    let domains: Vec<[u8; 32]> = (0..3).map(|i| domain(&bindings[i], OWNERS[i], count)).collect();
    let rs = [
        root(&domains[0], row_leaves(&domains[0], dx)),
        root(&domains[1], row_leaves(&domains[1], dw)),
        root(&domains[2], y_leaves(&domains[2], format, y_le)),
    ];
    core::array::from_fn(|i| TreeRef { binding: bindings[i], owner: OWNERS[i], count, root: rs[i] })
}

/// Why committed public values do not prove the statement `(format, id, count, trees)`, or `None` when they do: verdict 1,
/// the statement's format, id, K and B, and the three roots rebuilt from the published digests and words.
pub fn check(public: &[u8], format: Format, id: &[u8; 32], trees: &[TreeRef; 3]) -> Option<String> {
    let Some(p) = parse_public(public) else {
        return Some("public values are not relation-committed/v1".into());
    };
    if p.verdict != bare::VERDICT_ACCEPT {
        return Some(format!("verdict {} (the guest rejected the batch)", p.verdict));
    }
    if p.format_id != format.id() || &p.id != id || p.k as usize != bare::K {
        return Some(format!("format {} / id {} / K {} are not the statement's", p.format_id, hex_of(&p.id), p.k));
    }
    if trees.iter().any(|t| t.count != p.b as u64) {
        return Some(format!("B = {} is not the trees' count", p.b));
    }
    let bindings = [trees[0].binding, trees[1].binding, trees[2].binding];
    let got = roots(format, &bindings, p.y_le, p.dx, p.dw);
    for (i, (g, t)) in got.iter().zip(trees).enumerate() {
        if g.owner != t.owner {
            return Some(format!("tree {}: owner {} is not {}", TREE_NAMES[i], t.owner, g.owner));
        }
        if g.root != t.root {
            return Some(format!("tree {}: the published values give root {}, the statement's is {}", TREE_NAMES[i], hex_of(&g.root), hex_of(&t.root)));
        }
    }
    None
}

fn hex_of(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    //! The row leaf and trees against the core's own vectors (PR #15's `frame_v3/vectors.json`, sha256 entries, copied to
    //! `fixtures/sp1-committed/frame-v3-sha256-vectors.json`), the statement's bindings and roots against the Python
    //! reference (`verity_sp1.committed`, `fixtures/sp1-committed/committed-vectors.json`), and the public values.

    use super::*;
    use serde_json::Value;

    fn fixture(name: &str) -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../fixtures/sp1-committed/").to_string() + name;
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    fn b32(s: &str) -> [u8; 32] {
        unhex(s).try_into().unwrap()
    }

    #[test]
    fn row_leaves_match_the_core_vectors() {
        let v = fixture("frame-v3-sha256-vectors.json");
        let rows = v["row_leaves"].as_array().unwrap();
        assert!(rows.len() >= 12);
        for r in rows {
            let (role, bits, n) = (r["role"].as_u64().unwrap() as u8, r["word_bits"].as_u64().unwrap() as u8, r["n_words"].as_u64().unwrap() as u32);
            assert_eq!(r["schema"], ROW_SCHEMA);
            assert_eq!(row_prefix(role, bits, n).to_vec(), unhex(r["prefix"].as_str().unwrap()));
            let bytes = unhex(r["row_bytes"].as_str().unwrap());
            assert_eq!(row_digest(role, bits, &bytes).to_vec(), unhex(r["digest"].as_str().unwrap()), "{r}");
        }
        for t in v["row_trees"].as_array().unwrap() {
            let d = &t["domain"];
            let range = d["range"].as_array().unwrap();
            let count = range[1].as_u64().unwrap();
            assert_eq!(range[0].as_u64(), Some(0));
            let dom = domain(&b32(d["binding"].as_str().unwrap()), d["owner"].as_i64().unwrap(), count);
            assert_eq!(dom.to_vec(), unhex(d["domain_id"].as_str().unwrap()));
            let digests: Vec<u8> = t["digests"].as_array().unwrap().iter().flat_map(|x| unhex(x.as_str().unwrap())).collect();
            let (role, bits) = (t["role"].as_u64().unwrap() as u8, t["word_bits"].as_u64().unwrap() as u8);
            for (row, dg) in t["rows"].as_array().unwrap().iter().zip(digests.chunks_exact(32)) {
                assert_eq!(row_digest(role, bits, &unhex(row.as_str().unwrap())).to_vec(), dg.to_vec());
            }
            let leaves = row_leaves(&dom, &digests);
            let want: Vec<Vec<u8>> = t["leaves"].as_array().unwrap().iter().map(|x| unhex(x.as_str().unwrap())).collect();
            assert_eq!(leaves.iter().map(|l| l.to_vec()).collect::<Vec<_>>(), want);
            assert_eq!(root(&dom, leaves).to_vec(), unhex(t["root"].as_str().unwrap()), "tree of {count}");
        }
        for w in v["word_leaves"].as_array().unwrap() {
            let d = &w["domain"];
            let count = d["range"][1].as_u64().unwrap();
            let dom = domain(&b32(d["binding"].as_str().unwrap()), d["owner"].as_i64().unwrap(), count);
            assert_eq!(dom.to_vec(), unhex(d["domain_id"].as_str().unwrap()));
            let pos = w["position"].as_u64().unwrap();
            let leaf = frame::leaf(&dom, pos, pos, b"u16", &unhex(w["value"].as_str().unwrap()));
            assert_eq!(leaf.to_vec(), unhex(w["leaf"].as_str().unwrap()));
        }
    }

    #[test]
    fn statements_match_the_python_reference() {
        let v = fixture("committed-vectors.json");
        let cases = v["statements"].as_array().unwrap();
        assert!(!cases.is_empty());
        for c in cases {
            let format = Format::from_name(c["format"].as_str().unwrap()).unwrap();
            let inst = &c["instances"];
            let (lo, hi) = (inst["lo"].as_u64().unwrap(), inst["hi"].as_u64().unwrap());
            let bs = bindings(format, inst["dataset"].as_str().unwrap(), inst["tier"].as_str().unwrap(), inst["manifest_sha256"].as_str().unwrap(), lo, hi).unwrap();
            let x = unhex(c["x"].as_str().unwrap());
            let w = unhex(c["w"].as_str().unwrap());
            let y_le = unhex(c["y_le"].as_str().unwrap());
            let hasher = RowHasher::new(format);
            let rb = format.row_bytes();
            let dx: Vec<u8> = x.chunks_exact(rb).flat_map(|r| hasher.x(r)).collect();
            let dw: Vec<u8> = w.chunks_exact(rb).flat_map(|r| hasher.w(r)).collect();
            assert_eq!(dx, unhex(c["dx"].as_str().unwrap()), "{}", format.name());
            assert_eq!(dw, unhex(c["dw"].as_str().unwrap()), "{}", format.name());
            let trees = roots(format, &bs, &y_le, &dx, &dw);
            for (i, t) in c["trees"].as_array().unwrap().iter().enumerate() {
                assert_eq!(t["name"], TREE_NAMES[i]);
                assert_eq!(t["schema"], tree_schema(format, TREE_NAMES[i]));
                assert_eq!(trees[i].binding.to_vec(), unhex(t["binding"].as_str().unwrap()), "{} binding {}", format.name(), TREE_NAMES[i]);
                assert_eq!(trees[i].owner, t["owner"].as_i64().unwrap());
                assert_eq!(trees[i].count, t["count"].as_u64().unwrap());
                assert_eq!(trees[i].root.to_vec(), unhex(t["root"].as_str().unwrap()), "{} root {}", format.name(), TREE_NAMES[i]);
            }
        }
    }

    /// `n` all-zero VUs: every format's zero dot product is +0, so y = 0 is honest.
    fn batch(format: Format, n: usize) -> (Vec<u8>, Vec<u8>) {
        (vec![0u8; n * format.row_bytes()], vec![0u8; n * format.row_bytes()])
    }

    #[test]
    fn midstate_hasher_equals_the_one_shot_digest() {
        for format in Format::ALL {
            let rb = format.row_bytes();
            let row: Vec<u8> = (0..rb).map(|i| (i * 131 % 251) as u8).collect();
            let h = RowHasher::new(format);
            assert_eq!(h.x(&row), row_digest(ROLE_X, word_bits(format), &row));
            assert_eq!(h.w(&row), row_digest(ROLE_W, word_bits(format), &row));
            assert_ne!(h.x(&row), h.w(&row), "roles are separated");
            assert_eq!(n_words(format) as usize * word_bits(format) as usize, 8 * rb);
        }
    }

    #[test]
    fn decide_parse_and_check_round_trip_and_reject() {
        let id = bare::identity_digest("d|t|0|3|m");
        for format in Format::ALL {
            let (x, w) = batch(format, 3);
            let y_le = bare::encode_y(format, &[0, 0, 0]).unwrap();
            let header = bare::encode_header(format, bare::K as u32, 2, &id, &y_le);
            let chunks = vec![bare::encode_chunk(format, &x, &w, 0, 2), bare::encode_chunk(format, &x, &w, 2, 3)];
            let public = decide(&header, &chunks);
            let p = parse_public(&public).unwrap();
            assert_eq!((p.format(), p.id, p.k, p.b, p.verdict), (Some(format), id, bare::K as u32, 3, bare::VERDICT_ACCEPT));
            let bs = bindings(format, "d", "t", "m", 0, 3).unwrap();
            let trees = roots(format, &bs, &y_le, p.dx, p.dw);
            assert_eq!(check(&public, format, &id, &trees), None);
            // the verifier rejects: a wrong root, another id, a truncated or extended value list
            for i in 0..3 {
                let mut bad = trees;
                bad[i].root[0] ^= 1;
                assert!(check(&public, format, &id, &bad).unwrap().contains(TREE_NAMES[i]));
            }
            assert!(check(&public, format, &[9; 32], &trees).is_some());
            assert!(parse_public(&public[..public.len() - 1]).is_none());
            assert!(check(&[public.clone(), vec![0]].concat(), format, &id, &trees).is_some());
            // a wrong operand word with the honest y: the relation may still hold, the x root does not
            let mut tampered = x.clone();
            tampered[1] ^= 0x01;
            let chunks_t = vec![bare::encode_chunk(format, &tampered, &w, 0, 2), bare::encode_chunk(format, &tampered, &w, 2, 3)];
            let pt = decide(&header, &chunks_t);
            if parse_public(&pt).unwrap().verdict == bare::VERDICT_ACCEPT {
                assert!(check(&pt, format, &id, &trees).unwrap().contains("tree a"));
            }
            // a wrong y: verdict 0, head only
            let wrong = bare::encode_y(format, &[0, 1, 0]).unwrap();
            let pr = decide(&bare::encode_header(format, bare::K as u32, 2, &id, &wrong), &chunks);
            assert_eq!(pr.len(), HEAD_LEN);
            assert_eq!(parse_public(&pr).unwrap().verdict, bare::VERDICT_REJECT);
            assert!(check(&pr, format, &id, &trees).unwrap().contains("verdict"));
            // the indexed layout and a short chunk list reject
            let (index, table, rows) = bare::index_rows(format, &x, &w, 0, 3);
            let pi = decide(&bare::encode_header_indexed(format, bare::K as u32, rows, &id, &y_le), &[index, table]);
            assert_eq!(parse_public(&pi).unwrap().verdict, bare::VERDICT_REJECT);
            assert_eq!(parse_public(&decide(&header, &chunks[..1])).unwrap().verdict, bare::VERDICT_REJECT);
        }
        assert_eq!(parse_public(&decide(b"junk", &[])).unwrap().format_id, 0);
    }

    #[test]
    fn bindings_name_the_tree_and_the_schema() {
        let f = Format::Fp8Ada;
        let bs = bindings(f, "d", "t", "m", 0, 4).unwrap();
        assert_ne!(bs[0], bs[1]);
        assert_ne!(bs[0], bindings(f, "d", "t", "m", 0, 5).unwrap()[0]);
        assert_ne!(bs[2], bindings(Format::Bf16Ampere, "d", "t", "m", 0, 4).unwrap()[2], "u16 vs u32 word leaves");
        assert!(binding_digest("a\"b", "t", "m", 0, 1, 1536, "a", ROW_SCHEMA).is_none());
    }
}
