//! Scheme `vllm-v1` of `verity/ml/commitments/vllm_v1` (its `PROTOCOL.md`), for committed-operands
//! statements claimed as `R+vllm-v1`: position leaves over the operand rows' raw little-endian bytes, the unpadded
//! node / lift fold, and a step-root binding (§4) per tree.  The leaf digest *is* the tree leaf, so the per-VU public
//! digests are `pos_leaf(row bytes)` and nothing else is hashed around them.
//!
//! vllm-v1 hashes bind no domain (§5, §9): every root here is `StepDomain(chunk=None).bind(tree_root)`, the domain is
//! derived from `commitment.txt`'s pinned fields by `operand_domain`, and its `domain_digest` must equal the
//! statement's `domain_<port>` line (absorbed with the rest of `commitment.txt`).  `operand_domain`'s program / ctx /
//! geo / layout digests are this backend's provisional mapping of an operand tree onto a vLLM step domain; the
//! integration owns those digests (§4) and may replace them.

use crate::commitments::{identity_digest, Digest, Json, Spec};
use crate::sha256::Sha256;

pub const SCHEME: &str = "vllm-v1";
const POS_LEAF_TAG: &[u8] = b"verity/pos-leaf/v0";
const STEP_ROOT_TAG: &[u8] = b"verity/fa2c/root/v0";
pub const DOMAIN_DIGEST_TAG: &str = "verity/vllm-v1/domain/v1";
const MAP_TAG: &str = "verity/gkr-commit/vllm-v1";

/// §1 `H(tag, parts)`
pub fn h(tag: &str, parts: &[&[u8]]) -> Digest {
    let mut s = Sha256::new();
    s.update(&(tag.len() as u32).to_be_bytes());
    s.update(tag.as_bytes());
    for p in parts {
        s.update(&(p.len() as u64).to_be_bytes());
        s.update(p);
    }
    s.finalize()
}

pub fn pos_leaf(value: &[u8]) -> Digest {
    let mut s = Sha256::new();
    s.update(POS_LEAF_TAG);
    s.update(&(value.len() as u64).to_be_bytes());
    s.update(value);
    s.finalize()
}

pub fn node(left: &Digest, right: &Digest) -> Digest { h("verity-vllm/node/v1", &[left, right]) }
pub fn lift(child: &Digest) -> Digest { h("verity-vllm/lift/v1", &[child]) }
pub fn empty() -> Digest { h("verity-vllm/empty/v1", &[]) }

/// §3: level 0 is the leaves; pairs become nodes, an odd tail is lifted; no padding.
pub fn levels(leaves: &[Digest]) -> Vec<Vec<Digest>> {
    if leaves.is_empty() { return vec![vec![empty()]]; }
    let mut out = vec![leaves.to_vec()];
    while out.last().unwrap().len() > 1 {
        let prev = out.last().unwrap();
        let next = prev.chunks(2).map(|c| if c.len() == 2 { node(&c[0], &c[1]) } else { lift(&c[0]) }).collect();
        out.push(next);
    }
    out
}

pub fn fold(leaves: &[Digest]) -> Digest { levels(leaves).last().unwrap()[0] }

/// `shape(N, i)`: per level from the leaves up, `true` where the node has a sibling, `false` where it is lifted.
pub fn path_shape(leaf_count: u64, index: u64) -> Option<Vec<bool>> {
    if index >= leaf_count { return None; }
    let (mut width, mut i, mut out) = (leaf_count, index, Vec::new());
    while width > 1 {
        out.push(!(i == width - 1 && width % 2 == 1));
        width = width.div_ceil(2);
        i /= 2;
    }
    Some(out)
}

pub fn open_path(levels: &[Vec<Digest>], mut index: usize) -> Vec<Option<Digest>> {
    let mut out = Vec::new();
    for level in &levels[..levels.len() - 1] {
        out.push(level.get(index ^ 1).copied());
        index /= 2;
    }
    out
}

pub fn fold_path(leaf: &Digest, mut index: u64, path: &[Option<Digest>]) -> Digest {
    let mut d = *leaf;
    for s in path {
        d = match s { None => lift(&d), Some(s) if index % 2 == 0 => node(&d, s), Some(s) => node(s, &d) };
        index /= 2;
    }
    d
}

/// `verify_opening`: the shape comes from `(leaf_count, index)` before anything is hashed.
pub fn verify_opening(root: &Digest, leaf_count: u64, index: u64, leaf: &Digest, path: &[Option<Digest>], bind: &dyn Fn(&Digest) -> Digest) -> bool {
    let Some(shape) = path_shape(leaf_count, index) else { return false };
    if shape.len() != path.len() || shape.iter().zip(path).any(|(&has, s)| has != s.is_some()) { return false; }
    &bind(&fold_path(leaf, index, path)) == root
}

/// `StepDomain(chunk=None)`: a host-path step tree of position leaves, bound by the step root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepDomain { pub program: Digest, pub ctx: Digest, pub geo: Digest, pub layout: Digest, pub leaf_count: u64 }

impl StepDomain {
    pub fn binding_preimage(&self, tree_root: &Digest) -> Vec<u8> {
        [STEP_ROOT_TAG, &self.program, &self.ctx, &self.geo, &self.layout, &self.leaf_count.to_be_bytes(), tree_root].concat()
    }
    pub fn bind(&self, tree_root: &Digest) -> Digest {
        let mut s = Sha256::new();
        s.update(&self.binding_preimage(tree_root));
        s.finalize()
    }
    pub fn root(&self, leaves: &[Digest]) -> Result<Digest, String> {
        if leaves.len() as u64 != self.leaf_count { return Err(format!("expected {} leaves, got {}", self.leaf_count, leaves.len())); }
        Ok(self.bind(&fold(leaves)))
    }
    /// §9 `domain_digest`
    pub fn digest(&self) -> Digest {
        let hx = crate::commitments::hex;
        let (p, c, g, l) = (hx(&self.program), hx(&self.ctx), hx(&self.geo), hx(&self.layout));
        identity_digest(DOMAIN_DIGEST_TAG, &[("kind", Json::Str("step")), ("leaf_count", Json::Int(self.leaf_count)),
                                             ("program_digest", Json::Str(&p)), ("ctx_digest", Json::Str(&c)), ("geo_digest", Json::Str(&g)),
                                             ("layout_digest", Json::Str(&l)), ("chunk", Json::Null)])
    }
}

pub const PORTS: [&str; 3] = ["a", "b", "y"];

/// The provisional step domain of port `a` (x rows), `b` (W columns) or `y` (output words, one leaf per word) of the
/// statement's input set; mirrored by `gpu/commit.py vllm_domain`.
pub fn operand_domain(spec: &Spec, port: &str) -> StepDomain {
    let hi = spec.lo + spec.vus;
    let base = spec.relation.strip_suffix(spec.hash.suffix()).unwrap_or(&spec.relation);
    let program = identity_digest(&format!("{MAP_TAG}/program/v0"), &[("relation", Json::Str(base))]);
    let ctx = identity_digest(&format!("{MAP_TAG}/ctx/v0"), &[("dataset", Json::Str(&spec.dataset)), ("tier", Json::Str(&spec.tier)),
                                                             ("manifest_sha256", Json::Str(&spec.manifest_sha256)), ("lo", Json::Int(spec.lo)),
                                                             ("hi", Json::Int(hi)), ("port", Json::Str(port))]);
    let geo = identity_digest(&format!("{MAP_TAG}/geo/v0"), &[("vus", Json::Int(spec.vus)), ("K", Json::Int(spec.k)), ("n_y", Json::Int(spec.n_y as u64))]);
    let (value, bits, words, count) = match port {
        "a" => ("x-row", spec.x_bits as u64, spec.k, spec.vus),
        "b" => ("w-column", spec.w_bits as u64, spec.k, spec.vus),
        _ => ("y-word", 8 * spec.y_width as u64, 1, spec.vus * spec.n_y as u64),
    };
    let layout = identity_digest(&format!("{MAP_TAG}/layout/v0"), &[("port", Json::Str(port)), ("value", Json::Str(value)),
                                                                   ("word_bits", Json::Int(bits)), ("words", Json::Int(words)),
                                                                   ("byte_order", Json::Str("little"))]);
    StepDomain { program, ctx, geo, layout, leaf_count: count }
}

/// A `y` word's position leaf: its `y_width` little-endian bytes.
pub fn word_leaf(width: usize, word: u32) -> Digest { pos_leaf(&word.to_le_bytes()[..width]) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitments::tests::{parse_json, vllm_spec_text, J};
    use crate::commitments::{hex, unhex32};

    fn vectors() -> J {
        parse_json(include_str!("../../../../verity/ml/commitments/vllm_v1/vectors.json"))
    }

    fn opt_path(p: &J) -> Vec<Option<Digest>> { p.arr().iter().map(|e| match e { J::Null => None, e => Some(e.d32()) }).collect() }

    #[test]
    fn vllm_v1_hashes_and_pos_leaves() {
        let v = vectors();
        assert_eq!(v.get("scheme").s(), SCHEME);
        for c in v.get("hash").arr() {
            let tag = c.get("tag").s();
            let parts: Vec<Vec<u8>> = c.get("parts").arr().iter().map(J::bytes).collect();
            let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
            assert_eq!(hex(&h(tag, &refs)), c.get("digest").s(), "{tag}");
        }
        for c in v.get("pos_leaf").arr() { assert_eq!(hex(&pos_leaf(&c.get("value").bytes())), c.get("leaf").s()); }
        let mut pos = 0;
        for c in v.get("layouts").arr().iter().filter(|c| c.get("leaf").s() == "pos") {
            let val = c.get("value").bytes();
            assert_eq!(c.get("prefix").bytes(), [POS_LEAF_TAG, &(val.len() as u64).to_be_bytes()].concat());
            assert_eq!(hex(&pos_leaf(&val)), c.get("digest").s());
            pos += 1;
        }
        assert!(pos >= 3);
    }

    #[test]
    fn vllm_v1_trees() {
        let v = vectors();
        let mut n = 0;
        for t in v.get("trees").arr() {
            let count = t.get("leaf_count").n() as u64;
            let leaves: Vec<Digest> = t.get("leaves").arr().iter().map(J::d32).collect();
            let root = fold(&leaves);
            assert_eq!(hex(&root), t.get("root").s(), "N = {count}");
            let lv = levels(&leaves);
            for (i, p) in t.get("paths").arr().iter().enumerate() {
                let want = opt_path(p);
                assert_eq!(open_path(&lv, i), want, "N = {count} path {i}");
                assert!(verify_opening(&root, count, i as u64, &leaves[i], &want, &|r| *r));
                if count > 1 {
                    assert!(!verify_opening(&root, count, ((i + 1) % count as usize) as u64, &leaves[i], &want, &|r| *r));
                }
            }
            n += 1;
        }
        assert!(n >= 12);
        assert_eq!(fold(&[]), empty());
    }

    #[test]
    fn vllm_v1_step_openings() {
        let v = vectors();
        let mut n = 0;
        for o in v.get("openings").arr() {
            if o.get("kind").s() != "step" || !matches!(o.get("domain").get("chunk"), J::Null) { continue; }
            let d = o.get("domain");
            let dom = StepDomain { program: d.get("program_digest").d32(), ctx: d.get("ctx_digest").d32(), geo: d.get("geo_digest").d32(),
                                   layout: d.get("layout_digest").d32(), leaf_count: d.get("leaf_count").n() as u64 };
            let leaf = pos_leaf(&o.get("value").bytes());
            let path = opt_path(o.get("path"));
            let ok = verify_opening(&o.get("root").d32(), dom.leaf_count, o.get("index").n() as u64, &leaf, &path, &|r| dom.bind(r));
            assert_eq!(ok, matches!(o.get("accept"), J::B(true)));
            if ok {
                assert_eq!(hex(&leaf), o.get("leaf").s());
                let tr = o.get("tree_root").d32();
                assert_eq!(hex(&dom.binding_preimage(&tr)), o.get("binding_preimage").s());
                assert_eq!(hex(&dom.digest()), o.get("domain_digest").s(), "domain_digest");
            }
            n += 1;
        }
        assert_eq!(n, 2);
    }

    #[test]
    fn vllm_v1_rejections() {
        let v = vectors();
        for r in v.get("rejections").arr() {
            let path: Option<Vec<Option<Digest>>> = r.get("path").arr().iter()
                .map(|e| match e { J::Null => Some(None), e => unhex32(e.s()).map(Some) }).collect();
            let ok = path.is_some_and(|p| verify_opening(&r.get("root").d32(), r.get("leaf_count").n() as u64, r.get("index").n() as u64,
                                                           &r.get("leaf").d32(), &p, &|x| *x));
            assert!(!ok, "{}", r.get("why").s());
        }
    }

    #[test]
    fn operand_domains_differ_by_port_and_set() {
        let s = Spec::parse(&vllm_spec_text()).unwrap();
        let ds: Vec<StepDomain> = PORTS.iter().map(|p| operand_domain(&s, p)).collect();
        assert!(ds[0].digest() != ds[1].digest() && ds[1].digest() != ds[2].digest());
        assert_eq!(ds[2].leaf_count, s.vus * s.n_y as u64);
        let mut s2 = s.clone();
        s2.lo = 1;
        assert_ne!(operand_domain(&s2, "a").ctx, ds[0].ctx);
    }
}
