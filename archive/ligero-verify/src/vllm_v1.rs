//! `vllm-v1` statements (`archive/direct/ligero/vllm_tree.py`, `leaf/vllm_v1.py`; core spec
//! `verity/ml/commitments/vllm_v1/PROTOCOL.md`): the published x / W digests are `vllm-v1` position
//! leaves `SHA-256("verity/pos-leaf/v0" || u64be(len) || row)` computed in the relation; the verifier folds them (and the
//! position leaves of the output words' little-endian bytes) through each port's statement multiproof into a `vllm-v1`
//! tree (pairs `node`, an odd trailing child `lift`, no padding) and compares the port's step-root binding over the
//! verifier-held domain with the statement's root (never the bare tree root, section 9.2).
//!
//! ~~~text
//! H(tag, p..)      = SHA-256(u32be(len tag) || tag || (u64be(len p) || p)*)          node / lift tags verity-vllm/{node,lift}/v1
//! identity(tag, d) = SHA-256("veritor/tagged-sha256/v1\0" || u32be(len tag) || tag || u64be(len json) || json)   (sorted keys)
//! program          = identity("verity/ligero-b/vllm-v1/program/v1", {K, scheme: "vllm-v1", subcircuit: "gemm-coordinate"})
//! geo              = identity(".../geo/v1", {K, row_bytes, word_bits, y_bytes})
//! layout(port)     = identity(".../layout/v1", {leaf: "pos-leaf", tree, value, value_len, word_bits})
//! ctx(port)        = the tree reference's binding (the input set and range; recomputed from the set by re-verifiers)
//! domain_digest    = identity("verity/vllm-v1/domain/v1", {chunk: null, ctx_digest, geo_digest, kind: "step", layout_digest,
//!                                                          leaf_count, program_digest})   (hex digests)
//! step root        = SHA-256("verity/fa2c/root/v0" || program || ctx || geo || layout || u64be N || tree_root)
//! ~~~
//!
//! Statement multiproof (vllm-v1 has none in the core, section 8.2): level by level the known nodes ascending; a known
//! even node whose right neighbour is known pairs with it, the last node of an odd-width level is lifted, any other
//! takes the next sibling; the proof must be consumed exactly.

use crate::auth::{tagged_sha256, HashAuthBlock, TreeRef, MAX_OPENINGS, MAX_POSITIONS, MAX_PROOF_BYTES, TREE_NAMES};
use crate::hash::{hex, Sha256};

pub const NAME: &str = "vllm-v1";
const TAG: &str = "verity/ligero-b/vllm-v1";
const DOMAIN_TAG: &[u8] = b"verity/vllm-v1/domain/v1";
const POS_LEAF_TAG: &[u8] = b"verity/pos-leaf/v0";
const STEP_ROOT_TAG: &[u8] = b"verity/fa2c/root/v0";
const NODE_TAG: &[u8] = b"verity-vllm/node/v1";
const LIFT_TAG: &[u8] = b"verity-vllm/lift/v1";
const VALUES: [&str; 3] = ["x-row", "w-column", "y-word-le"];

/// What a port's program / geo / layout digests are derived from: the relation's, never the statement's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub k: u64,
    pub word_bits: u32,
    pub y_bytes: u64,
}

/// The operand word width of each relation a `+vllm-v1` statement may name (the x4 folds the lane composes).
const WORD_BITS: &[(&str, u32)] = &[("fp8-ada-x4", 8), ("fp8-hopper-x4", 8), ("bf16-hopper-x4", 16), ("bf16-ampere-x4", 16)];

/// The shape of `base` (a relation name without its `+vllm-v1` suffix); `None` for a relation this scheme does not cover.
pub fn shape_for(base: &str) -> Option<Shape> {
    let wb = WORD_BITS.iter().find(|(n, _)| *n == base)?.1;
    let rel = crate::relation::for_statement(Some(base)).ok()?;
    Some(Shape { k: crate::relation::HASHED_ROW_K as u64, word_bits: wb, y_bytes: if rel.y_bits <= 16 { 2 } else { 4 } })
}

fn framed(tag: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(&(tag.len() as u32).to_be_bytes());
    h.update(tag);
    for p in parts {
        h.update(&(p.len() as u64).to_be_bytes());
        h.update(p);
    }
    h.finalize()
}

pub fn node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    framed(NODE_TAG, &[left, right])
}

pub fn lift(child: &[u8; 32]) -> [u8; 32] {
    framed(LIFT_TAG, &[child])
}

pub fn pos_leaf(value: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(POS_LEAF_TAG);
    h.update(&(value.len() as u64).to_be_bytes());
    h.update(value);
    h.finalize()
}

fn identity(tag: &str, json: &str) -> [u8; 32] {
    tagged_sha256(tag.as_bytes(), json.as_bytes())
}

pub fn value_len(tree: usize, s: &Shape) -> u64 {
    if tree == 2 { s.y_bytes } else { s.k * s.word_bits as u64 / 8 }
}

/// `(program, geo, layout)` of port `tree` (0 a, 1 b, 2 y): `vllm_tree.port_digests`.
pub fn port_digests(tree: usize, s: &Shape) -> ([u8; 32], [u8; 32], [u8; 32]) {
    let program = identity(&format!("{TAG}/program/v1"), &format!("{{\"K\":{},\"scheme\":\"{NAME}\",\"subcircuit\":\"gemm-coordinate\"}}", s.k));
    let geo = identity(&format!("{TAG}/geo/v1"), &format!("{{\"K\":{},\"row_bytes\":{},\"word_bits\":{},\"y_bytes\":{}}}", s.k,
                                                         s.k * s.word_bits as u64 / 8, s.word_bits, s.y_bytes));
    let wb = if tree == 2 { 8 * s.y_bytes } else { s.word_bits as u64 };
    let layout = identity(&format!("{TAG}/layout/v1"), &format!(
        "{{\"leaf\":\"pos-leaf\",\"tree\":\"{}\",\"value\":\"{}\",\"value_len\":{},\"word_bits\":{}}}",
        TREE_NAMES[tree], VALUES[tree], value_len(tree, s), wb));
    (program, geo, layout)
}

/// `verity.ml.commitments.vllm_v1.domain_digest` of the step domain of `t` (port `tree`).
pub fn domain_digest(tree: usize, t: &TreeRef, s: &Shape) -> [u8; 32] {
    let (program, geo, layout) = port_digests(tree, s);
    identity(std::str::from_utf8(DOMAIN_TAG).unwrap(), &format!(
        "{{\"chunk\":null,\"ctx_digest\":\"{}\",\"geo_digest\":\"{}\",\"kind\":\"step\",\"layout_digest\":\"{}\",\"leaf_count\":{},\"program_digest\":\"{}\"}}",
        hex(&t.binding), hex(&geo), hex(&layout), t.count, hex(&program)))
}

/// The step-root binding of `tree_root` under port `tree`'s domain (ctx = `t.binding`, N = `t.count`).
pub fn step_root(tree: usize, t: &TreeRef, s: &Shape, tree_root: &[u8; 32]) -> [u8; 32] {
    let (program, geo, layout) = port_digests(tree, s);
    let mut h = Sha256::new();
    for p in [STEP_ROOT_TAG, &program[..], &t.binding[..], &geo[..], &layout[..], &t.count.to_be_bytes()[..], &tree_root[..]] {
        h.update(p);
    }
    h.finalize()
}

/// Fold a statement multiproof (module doc): `leaves` strictly ascending by rank, every rank `< count`.
pub fn fold_multiproof(count: u64, leaves: &[(u64, [u8; 32])], siblings: &[[u8; 32]]) -> Result<[u8; 32], String> {
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
    let (mut width, mut cursor) = (count, 0usize);
    while width > 1 {
        let mut next = Vec::with_capacity(nodes.len() / 2 + 1);
        let mut at = 0;
        while at < nodes.len() {
            let (i, v) = nodes[at];
            if i % 2 == 0 && at + 1 < nodes.len() && nodes[at + 1].0 == i + 1 {
                next.push((i >> 1, node(&v, &nodes[at + 1].1)));
                at += 2;
                continue;
            }
            if i == width - 1 && width % 2 == 1 {
                next.push((i >> 1, lift(&v)));
            } else {
                let s = siblings.get(cursor).ok_or("multiproof too short")?;
                cursor += 1;
                next.push((i >> 1, if i % 2 == 0 { node(&v, s) } else { node(s, &v) }));
            }
            at += 1;
        }
        nodes = next;
        width = (width + 1) / 2;
    }
    if cursor != siblings.len() {
        return Err("multiproof too long".into());
    }
    Ok(nodes[0].1)
}

/// `hashauth._verify_vllm`: every port's leaves at the verifier-fixed positions (the layout was checked by the caller:
/// `auth::layout_error`) fold through the port's own multiproof to a tree whose step-root binding is the port's root.
pub fn check(a: &HashAuthBlock, s: &Shape, y_words: &[u32], threads: usize) -> Result<(), String> {
    let n = a.vu_index.len();
    let (de, nd) = (a.leaf.digest_elems, a.n_digest());
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
        let mut per: Vec<(u64, Vec<u8>)> = if ti == 2 {
            (0..n).map(|v| (a.vu_index[v] as u64, y_words[v].to_le_bytes()[..s.y_bytes as usize].to_vec())).collect()
        } else {
            let ranks = if ti == 0 { &a.x_index } else { &a.w_index };
            let lane0 = if ti == 0 { 0 } else { de };
            (0..n).map(|v| (ranks[v] as u64, (a.leaf.leaf_bytes)(&a.digests[v * nd + lane0..v * nd + lane0 + de]))).collect()
        };
        per.sort_by_key(|e| e.0);
        let mut entries: Vec<(u64, Vec<u8>)> = Vec::with_capacity(per.len());
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
        if entries.len() as u64 > MAX_OPENINGS {
            return Err(format!("auth: {name}: openings {} exceed the limit {MAX_OPENINGS}", entries.len()));
        }
        let leaves: Vec<(u64, [u8; 32])> = crate::verify::par_for(entries.len(), threads, |i| {
            let (r, v) = &entries[i];
            if ti == 2 {
                (*r, pos_leaf(v))
            } else {
                let mut d = [0u8; 32];
                if v.len() == 32 { d.copy_from_slice(v) }
                (*r, d)
            }
        });
        if ti < 2 && entries.iter().any(|(_, v)| v.len() != 32) {
            return Err(format!("auth: {name}: a published digest does not frame as a vllm-v1 leaf"));
        }
        let root = fold_multiproof(tree.count, &leaves, sibs).map_err(|e| format!("auth: {name}: multiproof rejected ({e})"))?;
        if step_root(ti, tree, s, &root) != tree.root {
            return Err(format!("auth: {name}: multiproof rejected (step root mismatch)"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::unhex32;

    fn h(s: &str) -> [u8; 32] {
        unhex32(s)
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    #[test]
    fn tree_and_every_opening_of_the_core_five_leaf_vector() {
        // vllm_v1/vectors.json trees[leaf_count = 5]
        let leaves = [
            h("c2a6241fc150fed734162ce4d9275170d893511c8e6d62660f10760717aff704"),
            h("c1fc4595c06884a5eb8ce2bf58ef713a58f02bb2ce18947b13d3eea01a83f613"),
            h("ee97d21d8d747649c4c9d0bad4bf5b363f006d167bd3d0fd8e73abc1773d0af5"),
            h("bf7dc118dd2f36187938c69d2f3211e67deb0054ebad3fed172fb3152418dee1"),
            h("09b89a0fa7e7288b94db34366273332f32f41f2e3ce1a6b2ced095923646cbc6"),
        ];
        let root = h("a044fbedcadda5234607f8168dd7d5efe3e29163f8dd25e6de43644ee3a08b7f");
        let paths: [&[&str]; 5] = [
            &["c1fc4595c06884a5eb8ce2bf58ef713a58f02bb2ce18947b13d3eea01a83f613", "b486683c99b6dc72000fc4f3b57d4b5741355b603f9aec4766f56b8528033480",
              "494e2186fe1057bb5142e30b54b89bc3963770fb9a4beeaf506407c8d46aa484"],
            &["c2a6241fc150fed734162ce4d9275170d893511c8e6d62660f10760717aff704", "b486683c99b6dc72000fc4f3b57d4b5741355b603f9aec4766f56b8528033480",
              "494e2186fe1057bb5142e30b54b89bc3963770fb9a4beeaf506407c8d46aa484"],
            &["bf7dc118dd2f36187938c69d2f3211e67deb0054ebad3fed172fb3152418dee1", "f9494a2365bedc0392c797221ab0b047d666de9d47c90ca3569ebbf062aa7e76",
              "494e2186fe1057bb5142e30b54b89bc3963770fb9a4beeaf506407c8d46aa484"],
            &["ee97d21d8d747649c4c9d0bad4bf5b363f006d167bd3d0fd8e73abc1773d0af5", "f9494a2365bedc0392c797221ab0b047d666de9d47c90ca3569ebbf062aa7e76",
              "494e2186fe1057bb5142e30b54b89bc3963770fb9a4beeaf506407c8d46aa484"],
            &["c0b1ba569af493f144fb9c9e541fd8a3c7ef6ebd923cca5ef43f3ec9b6e89252"],   // two lifts dropped
        ];
        for (i, p) in paths.iter().enumerate() {
            let sibs: Vec<[u8; 32]> = p.iter().map(|x| h(x)).collect();
            assert_eq!(fold_multiproof(5, &[(i as u64, leaves[i])], &sibs).unwrap(), root, "leaf {i}");
            assert!(fold_multiproof(5, &[(i as u64, leaves[i])], &sibs[1..]).is_err() || sibs.len() == 1);
            let mut long = sibs.clone();
            long.push([0u8; 32]);
            assert!(fold_multiproof(5, &[(i as u64, leaves[i])], &long).is_err());
        }
        // all five at once: no siblings at all
        let all: Vec<(u64, [u8; 32])> = leaves.iter().enumerate().map(|(i, l)| (i as u64, *l)).collect();
        assert_eq!(fold_multiproof(5, &all, &[]).unwrap(), root);
        // a leaf at another position does not fold to the root
        assert_ne!(fold_multiproof(5, &[(1, leaves[0])], &paths[0].iter().map(|x| h(x)).collect::<Vec<_>>()).unwrap(), root);
    }

    #[test]
    fn step_opening_vector_and_domain_digest() {
        // vllm_v1/vectors.json openings[kind = step, chunk = null, accept]
        let value = unhex("447621edacd2003d3942b0d05be2389c2aa99c43605277c0ec105b743816a9dfbd304eb42150db65");
        let sibs = [h("ac705f5abfef7f3873ad3a50339d762fb73f7106c5cc3f3d23923dc93621b97a"),
                    h("13cb470d9e55234fc629ed61e7b17707281e15e32cacdd68ffeb7cb949cc2073")];
        let tree_root = fold_multiproof(6, &[(4, pos_leaf(&value))], &sibs).unwrap();
        assert_eq!(tree_root, h("81fe9933d665eb834710df6658186ea76d2016d96808b441d6c053a209498520"));
        let (program, ctx, geo, layout) = (h("506511e550d9a329b253c9d4303a1d67fd8216953ef94417729e31e18325a0a3"),
                                           h("145178591562508490adcbafbee329e07596c8448d3d86d2970894f0774cade3"),
                                           h("09a5f931b7afff1fd468818b013c0adec8b53f27c4ceb090214dfeac5888946f"),
                                           h("d6d6be6eb71f5db00ffd5b285a3bf7b4ce9b45019bb848bfb022947e7a5a41f6"));
        let mut s = Sha256::new();
        for p in [STEP_ROOT_TAG, &program[..], &ctx[..], &geo[..], &layout[..], &6u64.to_be_bytes()[..], &tree_root[..]] {
            s.update(p);
        }
        assert_eq!(s.finalize(), h("18f5e9dc6da12fb1458a91992f3ba09db8307f16cb3bad43abc49c04c467dbe5"));
        let dd = identity(std::str::from_utf8(DOMAIN_TAG).unwrap(), &format!(
            "{{\"chunk\":null,\"ctx_digest\":\"{}\",\"geo_digest\":\"{}\",\"kind\":\"step\",\"layout_digest\":\"{}\",\"leaf_count\":6,\"program_digest\":\"{}\"}}",
            hex(&ctx), hex(&geo), hex(&layout), hex(&program)));
        assert_eq!(dd, h("f127172f9fa97c7217ea9484cd3c0c1e212435b9a6c5022686d909e44f499bbd"));
    }

    #[test]
    fn port_domains_match_python() {
        // vllm_tree.domain(tree, ctx, 4096, K=1536, word_bits=8, y_bytes=4) and .bind(0x11 * 32), ctx = vllm_tree.ctx_digest(
        // dataset="bench-instances-fp8-ada/v1", tier="vu-k1536-fp8-ada", manifest_sha256="ab" * 32, lo=0, hi=4096, tree="a")
        let s = Shape { k: 1536, word_bits: 8, y_bytes: 4 };
        let t = TreeRef { binding: h("5bdb6d775233d25d78c1c81c93450c50f398bb9955f69921ae59ddfb5e35b638"), owner_plus_two: 1, count: 4096,
                          root: [0u8; 32] };
        let want = [
            ("2f8e4d71dd1a44f6e99e113d9cc73ed89520d206c14df3d79d19b3a3463708ed", "8c9e2fe698825dbd417132d1be07b802befdfb0be53a47e73c47b2b908cfcd88",
             "917265b14cb3fff7074a53be5887479176a1a9b0e17ac24a9a720d036811cddc"),
            ("1efe0b323a29174931a75b72b8b64f2581803a42a2ef718ecd9b296778a3997e", "1e46346fc9514137944a92805f2fbe1e1438ef1e41c88dc5b87517af1ccd2010",
             "1b7b367df902f08d798188cd339f2b0f028e1f52d5f7c996c637cd63ed9a5754"),
            ("6e8f6d2ea95d39d2b534b481c0e09687cfd93dbf33cf18bb74013ebf3e795e3d", "4fd3b2b89e2bcb456ff1c06518063b7108cd4175fbfedf9861a7e7d870e2d98f",
             "67de6e1bbce16b1d8e14401bf51b8bd8bb0160683377b968f809e813f52e66eb"),
        ];
        for (ti, (layout, dd, bound)) in want.iter().enumerate() {
            let (program, geo, lay) = port_digests(ti, &s);
            assert_eq!(hex(&program), "bb1839cf3e05e8ea421cf6f80a3da20029fb3f64038be21919df46d9d80369d6");
            assert_eq!(hex(&geo), "fd30fd95a7cf373c0980df7e48905c4471599e62149967812e6853254f7c64a3");
            assert_eq!(hex(&lay), *layout);
            assert_eq!(hex(&domain_digest(ti, &t, &s)), *dd);
            assert_eq!(hex(&step_root(ti, &t, &s, &[0x11u8; 32])), *bound);
        }
    }
}
