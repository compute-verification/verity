//! `relation-committed-vllm/v1` (feature `relation-committed-vllm`): the relation of [`committed`] with the operands
//! and outputs bound to `vllm-v1` trees (`verity.commitments.vllm_v1`, PROTOCOL.md) instead of frame-v3 ones.  The guest
//! (`--features relation-committed-vllm`) is a separate ELF with its own verifying key.
//!
//! **Leaves**: `vllm-v1` position leaves, `pos_leaf(v) = SHA-256("verity/pos-leaf/v0" || u64be(len v) || v)`.  Tree `a`'s
//! value at position `i` is VU `i`'s x row, `b`'s its W column (`row_bytes` each, as the frozen set stores them), and
//! `y`'s its output word (`y_bytes`, little-endian, as the guest commits it).  The guest hashes the rows; the verifier
//! hashes the words.  The leaf digest is the tree leaf.
//!
//! **Trees**: the `vllm-v1` fold (`node`, `lift` of an odd tail, `empty`), each root bound as a host-path step root
//! (`StepDomain(program, ctx, geo, layout, N)`, `SHA-256("verity/fa2c/root/v0" || program || ctx || geo || layout ||
//! u64be N || tree_root)`).  `vllm-v1` binds no domain inside its hashes, so the verifier derives every field from its
//! own data (PROTOCOL.md §9):
//!
//! ```text
//! program = identity_digest(PROGRAM_TAG, {K, scheme: "vllm-v1", statement: "relation-committed-vllm/v1"})
//! ctx     = identity_digest(CTX_TAG, {dataset, hi, lo, manifest_sha256, tier, tree})      tree = "a" | "b" | "y"
//! geo     = identity_digest(GEO_TAG, {format, row_bytes, y_bytes})
//! layout  = identity_digest(LAYOUT_TAG, {leaf: "pos-leaf", tree, value, value_len})       value = x-row | w-column | y-word-le
//! ```
//!
//! and the statement names each tree's `domain_digest` (`identity_digest("verity/vllm-v1/domain/v1", {kind: "step", ...})`).
//!
//! **Public values**: [`committed`]'s layout under the tag `verity/sp1/relation-committed-vllm/v1`: `dx` / `dw` are the
//! position-leaf digests of the rows.

use sha2::{Digest, Sha256};

use crate::bare::{self, Format};
use crate::committed::{Public, TREE_NAMES};
use crate::frame;

pub const TAG: &[u8] = b"verity/sp1/relation-committed-vllm/v1";
pub const HEAD_LEN: usize = TAG.len() + 4 + 32 + 4 + 4 + 1;
pub const STATEMENT: &str = "relation-committed-vllm/v1";
pub const POS_LEAF_TAG: &[u8] = b"verity/pos-leaf/v0";
pub const NODE_TAG: &[u8] = b"verity-vllm/node/v1";
pub const LIFT_TAG: &[u8] = b"verity-vllm/lift/v1";
pub const EMPTY_TAG: &[u8] = b"verity-vllm/empty/v1";
pub const STEP_ROOT_TAG: &[u8] = b"verity/fa2c/root/v0";
pub const DOMAIN_DIGEST_TAG: &str = "verity/vllm-v1/domain/v1";
pub const PROGRAM_TAG: &str = "verity/sp1/committed-vllm/program/v1";
pub const CTX_TAG: &str = "verity/sp1/committed-vllm/ctx/v1";
pub const GEO_TAG: &str = "verity/sp1/committed-vllm/geo/v1";
pub const LAYOUT_TAG: &str = "verity/sp1/committed-vllm/layout/v1";
/// What tree `a`, `b`, `y`'s position values are (the layout digest's `value`).
pub const VALUES: [&str; 3] = ["x-row", "w-column", "y-word-le"];

/// `pos_leaf(value)` (the one-shot reference).
pub fn pos_leaf(value: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(POS_LEAF_TAG);
    h.update((value.len() as u64).to_be_bytes());
    h.update(value);
    h.finalize().into()
}

/// The row hasher of a format: the 26-byte position-leaf prefix of a `row_bytes` value, absorbed once and cloned per row.
/// Rows of `a` and `b` share the rule; the trees' domains tell them apart.
#[derive(Clone)]
pub struct RowHasher {
    row: Sha256,
}

impl RowHasher {
    pub fn new(format: Format) -> RowHasher {
        let mut row = Sha256::new();
        row.update(POS_LEAF_TAG);
        row.update((format.row_bytes() as u64).to_be_bytes());
        RowHasher { row }
    }

    pub fn x(&self, row: &[u8]) -> [u8; 32] {
        let mut h = self.row.clone();
        h.update(row);
        h.finalize().into()
    }

    pub fn w(&self, row: &[u8]) -> [u8; 32] {
        self.x(row)
    }
}

/// Append the x-row and W-column leaves of every VU of a chunk (its VUs' x row then W row, `row_bytes` each).
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

/// What the guest commits for a header and its chunks (the native reference; the first failing chunk rejects).
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

/// `H(tag, parts) = SHA-256(u32be(len tag) || tag || (u64be(len p) || p)*)`, the `vllm-v1` tree framing.
pub fn framed(tag: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update((tag.len() as u32).to_be_bytes());
    h.update(tag);
    for p in parts {
        h.update((p.len() as u64).to_be_bytes());
        h.update(p);
    }
    h.finalize().into()
}

/// The `vllm-v1` tree root over leaf digests: pairs `node`, an odd tail `lift`, no leaves `empty`.
pub fn fold(mut level: Vec<[u8; 32]>) -> [u8; 32] {
    if level.is_empty() {
        return framed(EMPTY_TAG, &[]);
    }
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|p| if p.len() == 2 { framed(NODE_TAG, &[&p[0], &p[1]]) } else { framed(LIFT_TAG, &[&p[0]]) })
            .collect();
    }
    level[0]
}

/// A host-path `StepDomain`: the verifier's own context of one tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepDomain {
    pub program: [u8; 32],
    pub ctx: [u8; 32],
    pub geo: [u8; 32],
    pub layout: [u8; 32],
    pub leaf_count: u64,
}

impl StepDomain {
    pub fn binding_preimage(&self, tree_root: &[u8; 32]) -> Vec<u8> {
        [STEP_ROOT_TAG, &self.program, &self.ctx, &self.geo, &self.layout, &self.leaf_count.to_be_bytes(), tree_root].concat()
    }

    /// The published (bound) root.
    pub fn bind(&self, tree_root: &[u8; 32]) -> [u8; 32] {
        frame::sha256(&self.binding_preimage(tree_root))
    }

    /// `vllm_v1.domain_digest(domain)`.
    pub fn digest(&self) -> [u8; 32] {
        let payload = format!(
            "{{\"chunk\":null,\"ctx_digest\":\"{}\",\"geo_digest\":\"{}\",\"kind\":\"step\",\"layout_digest\":\"{}\",\"leaf_count\":{},\"program_digest\":\"{}\"}}",
            hex_of(&self.ctx),
            hex_of(&self.geo),
            hex_of(&self.layout),
            self.leaf_count,
            hex_of(&self.program)
        );
        frame::tagged_sha256(DOMAIN_DIGEST_TAG.as_bytes(), payload.as_bytes())
    }
}

fn json_str(s: &str) -> Option<String> {
    (!s.chars().any(|c| c == '"' || c == '\\' || (c as u32) < 0x20)).then(|| format!("\"{s}\""))
}

fn identity(tag: &str, canonical_json: &str) -> [u8; 32] {
    frame::tagged_sha256(tag.as_bytes(), canonical_json.as_bytes())
}

/// The value length of tree `i`'s positions.
pub const fn value_len(format: Format, i: usize) -> usize {
    if i == 2 {
        format.y_bytes()
    } else {
        format.row_bytes()
    }
}

/// The three domains (`a`, `b`, `y`) of a statement over tier ids `[lo, hi)` of a frozen set; `None` for a name that
/// would need a JSON escape.
pub fn domains(format: Format, dataset: &str, tier: &str, manifest_sha256: &str, lo: u64, hi: u64) -> Option<[StepDomain; 3]> {
    let program = identity(PROGRAM_TAG, &format!("{{\"K\":{},\"scheme\":\"vllm-v1\",\"statement\":\"{STATEMENT}\"}}", bare::K));
    let geo = identity(
        GEO_TAG,
        &format!("{{\"format\":{},\"row_bytes\":{},\"y_bytes\":{}}}", json_str(format.name())?, format.row_bytes(), format.y_bytes()),
    );
    let (dataset, tier, manifest) = (json_str(dataset)?, json_str(tier)?, json_str(manifest_sha256)?);
    let one = |i: usize| StepDomain {
        program,
        ctx: identity(
            CTX_TAG,
            &format!("{{\"dataset\":{dataset},\"hi\":{hi},\"lo\":{lo},\"manifest_sha256\":{manifest},\"tier\":{tier},\"tree\":\"{}\"}}", TREE_NAMES[i]),
        ),
        geo,
        layout: identity(
            LAYOUT_TAG,
            &format!("{{\"leaf\":\"pos-leaf\",\"tree\":\"{}\",\"value\":\"{}\",\"value_len\":{}}}", TREE_NAMES[i], VALUES[i], value_len(format, i)),
        ),
        leaf_count: hi - lo,
    };
    Some([one(0), one(1), one(2)])
}

/// The public reference of one tree: its domain and its bound root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeRef {
    pub domain: StepDomain,
    pub root: [u8; 32],
}

/// The position leaves of the claimed words (`y_bytes` little-endian each).
pub fn y_leaves(format: Format, y_le: &[u8]) -> Vec<[u8; 32]> {
    y_le.chunks_exact(format.y_bytes()).map(pos_leaf).collect()
}

/// The bound roots of trees `a`, `b`, `y` from the row leaves and the words.
pub fn roots(format: Format, domains: &[StepDomain; 3], y_le: &[u8], dx: &[u8], dw: &[u8]) -> [TreeRef; 3] {
    let leaves = |d: &[u8]| d.chunks_exact(32).map(|c| c.try_into().unwrap()).collect::<Vec<[u8; 32]>>();
    let tree_roots = [fold(leaves(dx)), fold(leaves(dw)), fold(y_leaves(format, y_le))];
    core::array::from_fn(|i| TreeRef { domain: domains[i], root: domains[i].bind(&tree_roots[i]) })
}

/// Why committed public values do not prove the statement `(format, id, trees)`, or `None` when they do: verdict 1, the
/// statement's format, id, K and B, and the three bound roots rebuilt from the published leaves and words.
pub fn check(public: &[u8], format: Format, id: &[u8; 32], trees: &[TreeRef; 3]) -> Option<String> {
    let Some(p) = parse_public(public) else {
        return Some(format!("public values are not {STATEMENT}"));
    };
    if p.verdict != bare::VERDICT_ACCEPT {
        return Some(format!("verdict {} (the guest rejected the batch)", p.verdict));
    }
    if p.format_id != format.id() || &p.id != id || p.k as usize != bare::K {
        return Some(format!("format {} / id {} / K {} are not the statement's", p.format_id, hex_of(&p.id), p.k));
    }
    if trees.iter().any(|t| t.domain.leaf_count != p.b as u64) {
        return Some(format!("B = {} is not the trees' leaf count", p.b));
    }
    let domains = [trees[0].domain, trees[1].domain, trees[2].domain];
    let got = roots(format, &domains, p.y_le, p.dx, p.dw);
    for (i, (g, t)) in got.iter().zip(trees).enumerate() {
        if g.root != t.root {
            return Some(format!("tree {}: the published values give root {}, the statement's is {}", TREE_NAMES[i], hex_of(&g.root), hex_of(&t.root)));
        }
    }
    None
}

pub fn hex_of(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    //! Leaves, fold, step-root binding and domain digest against the core's vectors
    //! (`verity.commitments.vllm_v1/vectors.json`); the statement's domains and roots against the Python reference
    //! (`verity_sp1.committed`, `fixtures/sp1-committed/committed-vllm-vectors.json`); the public values.

    use super::*;
    use serde_json::Value;

    fn read_json(relative: &str) -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../").to_string() + relative;
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    fn b32(v: &Value) -> [u8; 32] {
        unhex(v.as_str().unwrap()).try_into().unwrap()
    }

    fn step_domain(d: &Value, n_key: &str) -> StepDomain {
        StepDomain { program: b32(&d["program_digest"]), ctx: b32(&d["ctx_digest"]), geo: b32(&d["geo_digest"]),
            layout: b32(&d["layout_digest"]), leaf_count: d[n_key].as_u64().unwrap() }
    }

    #[test]
    fn leaves_fold_and_bindings_match_the_core_vectors() {
        let v = read_json("packages/verity/src/verity/commitments/vllm_v1/vectors.json");
        let pos = v["pos_leaf"].as_array().unwrap();
        assert!(pos.len() >= 4);
        for p in pos {
            assert_eq!(pos_leaf(&unhex(p["value"].as_str().unwrap())), b32(&p["leaf"]));
        }
        for h in v["hash"].as_array().unwrap() {
            let parts: Vec<Vec<u8>> = h["parts"].as_array().unwrap().iter().map(|p| unhex(p.as_str().unwrap())).collect();
            let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
            assert_eq!(framed(h["tag"].as_str().unwrap().as_bytes(), &refs), b32(&h["digest"]), "{}", h["tag"]);
        }
        let trees = v["trees"].as_array().unwrap();
        assert!(trees.len() >= 10);
        for t in trees {
            let leaves: Vec<[u8; 32]> = t["leaves"].as_array().unwrap().iter().map(b32).collect();
            assert_eq!(fold(leaves), b32(&t["root"]), "{} leaves", t["leaf_count"]);
        }
        let mut steps = 0;
        for r in v["roots"].as_array().unwrap().iter().filter(|r| r["kind"] == "step") {
            let i = &r["inputs"];
            assert_eq!(step_domain(i, "n").bind(&b32(&i["tree_root"])), b32(&r["root"]));
            steps += 1;
        }
        let mut openings = 0;
        for o in v["openings"].as_array().unwrap().iter().filter(|o| o["kind"] == "step" && o["accept"] == true && o["domain"]["chunk"].is_null()) {
            let dom = step_domain(&o["domain"], "leaf_count");
            assert_eq!(pos_leaf(&unhex(o["value"].as_str().unwrap())), b32(&o["leaf"]));
            assert_eq!(dom.binding_preimage(&b32(&o["tree_root"])), unhex(o["binding_preimage"].as_str().unwrap()));
            assert_eq!(dom.bind(&b32(&o["tree_root"])), b32(&o["root"]));
            assert_eq!(dom.digest(), b32(&o["domain_digest"]));
            openings += 1;
        }
        assert!(steps >= 1 && openings >= 1);
        let layout = v["layouts"].as_array().unwrap().iter().find(|l| l["leaf"] == "pos" && !l["value"].as_str().unwrap().is_empty()).unwrap();
        let value = unhex(layout["value"].as_str().unwrap());
        assert_eq!(unhex(layout["prefix"].as_str().unwrap()), [POS_LEAF_TAG, &(value.len() as u64).to_be_bytes()].concat());
    }

    #[test]
    fn statements_match_the_python_reference() {
        let v = read_json("fixtures/sp1-committed/committed-vllm-vectors.json");
        let cases = v["statements"].as_array().unwrap();
        assert!(!cases.is_empty());
        for c in cases {
            let format = Format::from_name(c["format"].as_str().unwrap()).unwrap();
            let inst = &c["instances"];
            let (lo, hi) = (inst["lo"].as_u64().unwrap(), inst["hi"].as_u64().unwrap());
            let ds = domains(format, inst["dataset"].as_str().unwrap(), inst["tier"].as_str().unwrap(), inst["manifest_sha256"].as_str().unwrap(), lo, hi).unwrap();
            let (x, w, y_le) = (unhex(c["x"].as_str().unwrap()), unhex(c["w"].as_str().unwrap()), unhex(c["y_le"].as_str().unwrap()));
            let hasher = RowHasher::new(format);
            let rb = format.row_bytes();
            let dx: Vec<u8> = x.chunks_exact(rb).flat_map(|r| hasher.x(r)).collect();
            let dw: Vec<u8> = w.chunks_exact(rb).flat_map(|r| hasher.w(r)).collect();
            assert_eq!(dx, unhex(c["dx"].as_str().unwrap()), "{}", format.name());
            assert_eq!(dw, unhex(c["dw"].as_str().unwrap()), "{}", format.name());
            let trees = roots(format, &ds, &y_le, &dx, &dw);
            for (i, t) in c["trees"].as_array().unwrap().iter().enumerate() {
                assert_eq!(t["name"], TREE_NAMES[i]);
                assert_eq!(trees[i].domain, step_domain(&t["domain"], "leaf_count"), "{} domain {}", format.name(), TREE_NAMES[i]);
                assert_eq!(trees[i].domain.digest(), b32(&t["domain_digest"]), "{} domain digest {}", format.name(), TREE_NAMES[i]);
                assert_eq!(trees[i].root, b32(&t["root"]), "{} root {}", format.name(), TREE_NAMES[i]);
            }
        }
    }

    #[test]
    fn hasher_equals_the_one_shot_leaf() {
        for format in Format::ALL {
            let row: Vec<u8> = (0..format.row_bytes()).map(|i| (i * 131 % 251) as u8).collect();
            let h = RowHasher::new(format);
            assert_eq!(h.x(&row), pos_leaf(&row));
            assert_eq!(h.w(&row), pos_leaf(&row));
        }
    }

    #[test]
    fn decide_parse_and_check_round_trip_and_reject() {
        let id = bare::identity_digest("d|t|0|3|m");
        for format in Format::ALL {
            let rb = format.row_bytes();
            let (x, w) = (vec![0u8; 3 * rb], vec![0u8; 3 * rb]);
            let y_le = bare::encode_y(format, &[0, 0, 0]).unwrap();
            let header = bare::encode_header(format, bare::K as u32, 2, &id, &y_le);
            let chunks = vec![bare::encode_chunk(format, &x, &w, 0, 2), bare::encode_chunk(format, &x, &w, 2, 3)];
            let public = decide(&header, &chunks);
            let p = parse_public(&public).unwrap();
            assert_eq!((p.format(), p.id, p.k, p.b, p.verdict), (Some(format), id, bare::K as u32, 3, bare::VERDICT_ACCEPT));
            assert!(crate::committed::parse_public(&public).is_none(), "the frame-v3 parser refuses vllm-v1 values");
            let ds = domains(format, "d", "t", "m", 0, 3).unwrap();
            let trees = roots(format, &ds, &y_le, p.dx, p.dw);
            assert_eq!(check(&public, format, &id, &trees), None);
            for i in 0..3 {
                let mut bad = trees;
                bad[i].root[0] ^= 1;
                assert!(check(&public, format, &id, &bad).unwrap().contains(TREE_NAMES[i]));
            }
            // a and b have the same leaves here: only their domains separate the roots
            assert_ne!(trees[0].root, trees[1].root);
            let mut swapped = trees;
            swapped.swap(0, 1);
            assert!(check(&public, format, &id, &swapped).is_some());
            let mut shifted = trees;
            shifted.iter_mut().zip(domains(format, "d", "t", "m", 1, 4).unwrap()).for_each(|(t, d)| t.domain = d);
            assert!(check(&public, format, &id, &shifted).is_some(), "another range's domains");
            assert!(check(&public, format, &[9; 32], &trees).is_some());
            assert!(parse_public(&public[..public.len() - 1]).is_none());
            let mut tampered = x.clone();
            tampered[1] ^= 0x01;
            let chunks_t = vec![bare::encode_chunk(format, &tampered, &w, 0, 2), bare::encode_chunk(format, &tampered, &w, 2, 3)];
            let pt = decide(&header, &chunks_t);
            if parse_public(&pt).unwrap().verdict == bare::VERDICT_ACCEPT {
                assert!(check(&pt, format, &id, &trees).unwrap().contains("tree a"));
            }
            let wrong = bare::encode_y(format, &[0, 1, 0]).unwrap();
            let pr = decide(&bare::encode_header(format, bare::K as u32, 2, &id, &wrong), &chunks);
            assert_eq!(pr.len(), HEAD_LEN);
            assert!(check(&pr, format, &id, &trees).unwrap().contains("verdict"));
        }
    }
}
