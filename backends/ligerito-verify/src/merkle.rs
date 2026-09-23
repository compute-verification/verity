//! BLAKE3 Merkle tree over the rows of a codeword matrix and its multi-opening with shared authentication paths.
//!
//! Leaf s = blake3(row s as little-endian u32 words: base rounds one word per entry, extension rounds 6 words per
//! entry, coordinate 0 first); node = blake3(left || right); a power-of-two number of leaves.  A multi-opening of
//! the leaf set S lists, level by level (leaves first) and in increasing index order within a level, exactly the
//! sibling digests the verifier cannot recompute from S itself -- `plan` (the Python `proto/merkle_multi.plan`).

use crate::hash::blake3;

pub type Digest = [u8; 32];

/// The `(level, index)` sibling digests a verifier of the leaf set `indices` needs, in the order the prover sends them.
pub fn plan(indices: &[usize], depth: usize) -> Vec<(usize, usize)> {
    let mut cur: Vec<usize> = indices.to_vec();
    cur.sort_unstable();
    cur.dedup();
    let mut out = Vec::new();
    for level in 0..depth {
        let mut nxt: Vec<usize> = Vec::with_capacity(cur.len());
        for (pos, &i) in cur.iter().enumerate() {
            let sib = i ^ 1;
            // `cur` is sorted and deduplicated: the sibling, if present, is the neighbour
            let has_sib = (i & 1 == 0 && pos + 1 < cur.len() && cur[pos + 1] == sib)
                || (i & 1 == 1 && pos > 0 && cur[pos - 1] == sib);
            if !has_sib {
                out.push((level, sib));
            }
            let parent = i >> 1;
            if nxt.last() != Some(&parent) {
                nxt.push(parent);
            }
        }
        cur = nxt;
    }
    out
}

/// Hash of one leaf: the row's words as little-endian u32.
pub fn leaf_digest(words: &[u32]) -> Digest {
    let mut buf = Vec::with_capacity(4 * words.len());
    for w in words {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    blake3(&buf)
}

pub fn node_digest(l: &Digest, r: &Digest) -> Digest {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(l);
    buf[32..].copy_from_slice(r);
    blake3(&buf)
}

/// Verify a multi-opening: `leaves` = (index, leaf digest) pairs (any order; duplicates must agree), `siblings` =
/// the prover's digests in `plan` order, `depth` = log2(number of leaves).  Returns the reason on failure.
pub fn verify_multi(root: &Digest, leaves: &[(usize, Digest)], siblings: &[Digest], depth: usize) -> Result<(), String> {
    if leaves.is_empty() {
        return Err("merkle: no leaves opened".into());
    }
    let mut cur: Vec<(usize, Digest)> = leaves.to_vec();
    cur.sort_by_key(|x| x.0);
    // duplicates must carry the same digest
    for w in cur.windows(2) {
        if w[0].0 == w[1].0 && w[0].1 != w[1].1 {
            return Err(format!("merkle: leaf {} opened twice with different contents", w[0].0));
        }
    }
    cur.dedup_by_key(|x| x.0);
    for &(i, _) in &cur {
        if i >= 1usize << depth {
            return Err(format!("merkle: leaf index {i} out of range for depth {depth}"));
        }
    }
    let expected = plan(&cur.iter().map(|x| x.0).collect::<Vec<_>>(), depth);
    if siblings.len() != expected.len() {
        return Err(format!("merkle: {} sibling digests, plan needs {}", siblings.len(), expected.len()));
    }
    let mut pos = 0;
    for _level in 0..depth {
        let mut nxt: Vec<(usize, Digest)> = Vec::with_capacity(cur.len());
        let mut k = 0;
        while k < cur.len() {
            let (i, d) = cur[k];
            let sib = i ^ 1;
            let (l, r);
            if k + 1 < cur.len() && cur[k + 1].0 == sib {
                l = d;
                r = cur[k + 1].1;
                k += 2;
            } else {
                let s = siblings[pos];
                pos += 1;
                if i & 1 == 0 {
                    l = d;
                    r = s;
                } else {
                    l = s;
                    r = d;
                }
                k += 1;
            }
            nxt.push((i >> 1, node_digest(&l, &r)));
        }
        cur = nxt;
    }
    debug_assert_eq!(pos, siblings.len());
    if cur.len() != 1 || cur[0].1 != *root {
        return Err("merkle: root mismatch".into());
    }
    Ok(())
}

/// A full tree (reference prover / tests): `levels[0]` = leaves, `levels[depth]` = [root].
pub struct Tree {
    pub levels: Vec<Vec<Digest>>,
}

impl Tree {
    pub fn build(leaves: Vec<Digest>) -> Tree {
        assert!(leaves.len().is_power_of_two());
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let prev = levels.last().unwrap();
            let next: Vec<Digest> = prev.chunks(2).map(|c| node_digest(&c[0], &c[1])).collect();
            levels.push(next);
        }
        Tree { levels }
    }

    pub fn root(&self) -> Digest {
        self.levels.last().unwrap()[0]
    }

    pub fn depth(&self) -> usize {
        self.levels.len() - 1
    }

    /// The sibling digests of `plan(indices)`.
    pub fn open_multi(&self, indices: &[usize]) -> Vec<Digest> {
        plan(indices, self.depth()).into_iter().map(|(lv, i)| self.levels[lv][i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_opening_roundtrip_and_tamper() {
        let n = 64;
        let leaves: Vec<Digest> = (0..n).map(|i| leaf_digest(&[i as u32, 7, 9])).collect();
        let tree = Tree::build(leaves.clone());
        for idx in [vec![3usize], vec![0, 1], vec![5, 4, 5, 63, 17, 16, 40]] {
            let sib = tree.open_multi(&idx);
            let opened: Vec<(usize, Digest)> = idx.iter().map(|&i| (i, leaves[i])).collect();
            assert!(verify_multi(&tree.root(), &opened, &sib, 6).is_ok());
            // a wrong leaf
            let mut bad = opened.clone();
            bad[0].1 = leaf_digest(&[999]);
            assert!(verify_multi(&tree.root(), &bad, &sib, 6).is_err());
            // a wrong sibling count
            if !sib.is_empty() {
                assert!(verify_multi(&tree.root(), &opened, &sib[1..], 6).is_err());
            }
            // a wrong root
            let mut r = tree.root();
            r[0] ^= 1;
            assert!(verify_multi(&r, &opened, &sib, 6).is_err());
        }
        // all leaves: no siblings at all
        let all: Vec<usize> = (0..n).collect();
        assert!(tree.open_multi(&all).is_empty());
        let opened: Vec<(usize, Digest)> = all.iter().map(|&i| (i, leaves[i])).collect();
        assert!(verify_multi(&tree.root(), &opened, &[], 6).is_ok());
    }

    #[test]
    fn plan_matches_python_reference_shape() {
        // indices {2, 3, 5} in a depth-3 tree: level 0 needs sibling 4 (2 and 3 pair up); level 1 has {1, 2}: needs
        // 0 and 3; level 2 has {0, 1}: nothing.
        assert_eq!(plan(&[5, 2, 3], 3), vec![(0, 4), (1, 0), (1, 3)]);
    }
}
