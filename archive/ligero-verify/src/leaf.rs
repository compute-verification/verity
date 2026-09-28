//! The leaf schemes a hashed statement (`authentication = included-hash`, `ligero-statement/v5`) may name: the Rust
//! side of `archive/direct/ligero/leaf/` (`LeafScheme`).  A v5 statement's relation string is `<rel>` (the
//! Poseidon2 scheme, the historical `+hash`) or `<rel>+<leaf>`; the scheme fixes how many field elements one operand
//! row's digest has, the leaf schema string and parameters digest the authentication block must carry (both absorbed
//! into the statement digest through `hash_digest_bytes`), and how a digest row is framed as leaf bytes under the
//! SHA-256 range-domain (`leaf_bytes`).  The in-circuit gadget is the relation's business: it is pinned by the hashed
//! system's `(sys_id, table_digest)` pair (`relation.rs` for Poseidon2, `PINS` here for every other scheme).
//!
//! Adding a scheme (another lane): one `LeafScheme` static, one entry in `SCHEMES`, and its `(relation, leaf,
//! sys_id, table_digest)` pins in `PINS` once the Python side has gated it.

use crate::field::P;

/// One leaf scheme as the verifier needs it (the Python `LeafScheme` protocol's verifier-visible half).
#[derive(Debug)]
pub struct LeafScheme {
    /// The registry name (`leaf.registry`): the `+<name>` suffix of a v5 relation string.
    pub name: &'static str,
    /// The name in messages ("Poseidon2 parameters differ from the verifier's").
    pub display: &'static str,
    /// `LeafScheme.schema`: the leaf schema string the authentication block carries and the trees frame leaves under.
    pub schema: &'static str,
    /// `LeafScheme.params_sha256`: SHA-256 (hex) of the scheme's parameters; a statement whose digest differs is
    /// refused at parse.
    pub params_sha256: &'static str,
    /// `LeafScheme.digest_elems`: field elements per operand-row digest (the public table publishes 2x per VU).
    pub digest_elems: usize,
    /// `LeafScheme.leaf_bytes`: one digest row (`digest_elems` canonical BabyBear lanes) -> the leaf value bytes.
    pub leaf_bytes: fn(&[u32]) -> Vec<u8>,
}

/// `rowleaf.row_leaf_value`: the 8 lanes as big-endian u32s.
fn poseidon2_leaf_bytes(lanes: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 * lanes.len());
    for x in lanes {
        v.extend_from_slice(&x.to_be_bytes());
    }
    v
}

/// The first plugin: `leaf.poseidon2.Poseidon2Leaf` -- one tensor ROW's Poseidon2-BabyBear (width 24, rate 16, Plonky3
/// constants) sponge digest, 8 lanes.  `schema` is `rowleaf.SCHEMA_ROW`; `params_sha256` is
/// `poseidon2_babybear.CONSTANTS_SHA256` (the round constants and the internal diagonal).
pub static POSEIDON2: LeafScheme = LeafScheme {
    name: "poseidon2",
    display: "Poseidon2",
    schema: "poseidon2-babybear-w24/row/v2h",
    params_sha256: "0b81af1cdcd7e5b1e0e510b632cc692d3c7c0d5923c4f3e0036bc58f69543321",
    digest_elems: 8,
    leaf_bytes: poseidon2_leaf_bytes,
};

/// `leaf.ajtai.AjtaiLeaf.leaf_bytes`: the n lanes as little-endian u32s (ajtai-design's framing).
fn ajtai_leaf_bytes(lanes: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 * lanes.len());
    for x in lanes {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

/// Lane ajtai-leaf: the linear Ring-SIS leaf `ajtai-n<N>` (`leaf/ajtai.py`) -- per chain column
/// `acc' = X acc + B_role s` in `F_p[X]/(X^N + 1)`, `s` the column's 256 committed operand bits, `B_role` the role's
/// 256 seeded ring elements of [`ajtai::matrix`] (x: `b_0 .. b_255`, W: `b_256 .. b_511`); the digest is the state after
/// the VU's last column, N BabyBear elements, N linked carry rows.
/// N = 64 serves the 48-step relations (fp8-ada, fp4), N = 128 the 96-step bf16-hopper (`N >= steps`: injectivity).
/// `params_sha256` = SHA-256 of `leaf.ajtai.params_descriptor` (which absorbs the SHA-256 of the derived `B`:
/// [`ajtai::B_SHA256`]); the derivation is checked against it in the tests, and every hashed system a `+ajtai-n<N>`
/// statement verifies against must carry the derived key in its Ajtai rows ([`check_system_key`]; red-team-leaf F6).
/// Binding only: the digest is linear in the bits (see the Python module docstring).
pub static AJTAI_N64: LeafScheme = LeafScheme {
    name: "ajtai-n64",
    display: "ajtai-babybear-n64-b1-r0/row/v1",
    schema: "ajtai-babybear-n64-b1-r0/row/v1",
    params_sha256: "0f292fbb9043f3630bf1e40848b7767c08a42ec3a77837b98f457c066aeebc69",
    digest_elems: 64,
    leaf_bytes: ajtai_leaf_bytes,
};

pub static AJTAI_N128: LeafScheme = LeafScheme {
    name: "ajtai-n128",
    display: "ajtai-babybear-n128-b1-r0/row/v1",
    schema: "ajtai-babybear-n128-b1-r0/row/v1",
    params_sha256: "d63a594bc0c2adeca97530eb9b71ff354b9f293d54319af99ffb8668b97f4ae3",
    digest_elems: 128,
    leaf_bytes: ajtai_leaf_bytes,
};


/// The Ajtai key: `B` re-derived from the public tag (the verifier never takes it from a statement or a system file).
pub mod ajtai {
    use crate::field::P;

    pub const B_TAG: &[u8] = b"verity/ajtai-babybear/B/v1";
    /// Operand bits per chain column (k words x word_bits = 256 for every registered relation).
    pub const M_COL: usize = 256;
    /// Key blocks (`leaf.ajtai.key_offset`): x rows use `b_0 .. b_255`, W columns `b_256 .. b_511`.
    pub const KEYS: usize = 2;
    /// SHA-256 of `matrix(n)` as n x 512 row-major LE u32 (`leaf.ajtai.B_DIGEST_SHA256`): what `params_sha256` absorbs.
    pub const B_SHA256: &[(usize, &str)] = &[
        (64, "1f3df70fd5de615f8bf3ce77fc5515057e41d2561019036399b3b4b76a68ebca"),
        (128, "37eed28c0803e05c9a90335e0c011b7ba6c19717d4eb63434addaf1c8fcf92c3"),
    ];

    /// `leaf.ajtai.derive_ring_element(i, n)`: the SHAKE-256 XOF of `B_TAG || LE32(i)` read as LE u32 words `u`,
    /// `v = u & 0x7fff_ffff`, accepted iff `v < p`; coefficient k = the k-th accepted value.
    pub fn ring_element(i: u32, n: usize) -> Vec<u32> {
        let mut seed = Vec::with_capacity(B_TAG.len() + 4);
        seed.extend_from_slice(B_TAG);
        seed.extend_from_slice(&i.to_le_bytes());
        let mut len = 4 * (n + n / 5 + 64);
        loop {
            let stream = crate::hash::shake256(&seed, len);
            let out: Vec<u32> = stream.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap()) & 0x7fff_ffff)
                .filter(|&v| v < P).take(n).collect();
            if out.len() == n {
                return out;
            }
            len *= 2;
        }
    }

    /// `leaf.ajtai.derive_matrix(n, 512)`: `B[x][i]` = coefficient x of `b_i`, as `n` rows of 512 (the x key, then the
    /// W key).
    pub fn matrix(n: usize) -> Vec<Vec<u32>> {
        let cols: Vec<Vec<u32>> = (0..(KEYS * M_COL) as u32).map(|i| ring_element(i, n)).collect();
        (0..n).map(|x| cols.iter().map(|c| c[x]).collect()).collect()
    }

    /// SHA-256 (hex) of `matrix(n)` as row-major LE u32.
    pub fn matrix_sha256(b: &[Vec<u32>]) -> String {
        let mut bytes = Vec::with_capacity(4 * b.len() * KEYS * M_COL);
        for row in b {
            for v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        crate::hash::hex(&crate::hash::sha256(&bytes))
    }

    /// `leaf.ajtai.params_descriptor(n, 256, 0, b_sha256)`: the bytes `params_sha256` is the SHA-256 of.
    pub fn params_descriptor(n: usize, b_sha256: &str) -> String {
        format!("ajtai-babybear/v1|p={P}|n={n}|m_col={M_COL}|beta=1|salt_bits=0|ring=F_p[X]/(X^n+1)|M=X|B=verity/ajtai-babybear/B/v1\
|b_i=SHAKE-256(tag||LE32(i)) LE u32 &0x7fffffff reject>=p|keys=x:b_0..b_{},W:b_{M_COL}..b_{}|leaf=LE u32 x n|B_sha256={b_sha256}",
            M_COL - 1, 2 * M_COL - 1)
    }

    /// The ring degree of an `ajtai-n<N>` scheme name.
    pub fn degree(name: &str) -> Option<usize> {
        name.strip_prefix("ajtai-n").and_then(|s| s.parse().ok())
    }
}

/// Red-team-leaf F6: a hashed system a `+ajtai-n<N>` statement verifies against must carry the DERIVED key.  The gadget
/// appends, per operand role (key block `o` = 0 for x, 256 for W), N linear rows
/// `out[x] - (X acc_in)[x] - sum_i B[x][o + i] bit_i = 0`; [`chain_key`] follows the statement's digest lanes to exactly
/// those rows (red-team-leaf-2 G1: not a pattern scan) and checks every coefficient.  A system whose digests come from
/// another key (`B = 0`: trivial digests), with or without decoy rows carrying the derived key, is refused even under
/// `--allow-any-system`.  Other schemes: nothing to check (their gadgets have no key; the system pin covers them).
pub fn check_system_key(sys: &crate::format::System, leaf: &LeafScheme) -> Result<(), String> {
    let n = match ajtai::degree(leaf.name) {
        Some(n) => n,
        None => return Ok(()),
    };
    let b = ajtai::matrix(n);
    // the derivation itself, against the digests both verifiers pin (and the scheme's params digest absorbs)
    let b_sha = ajtai::matrix_sha256(&b);
    match ajtai::B_SHA256.iter().find(|(m, _)| *m == n) {
        Some((_, want)) if *want == b_sha => {}
        Some(_) => return Err(format!("{}: the re-derived key B does not match the pinned digest (verifier build error)", leaf.display)),
        None => return Err(format!("{}: no pinned key digest for n = {n}", leaf.display)),
    }
    if crate::hash::hex(&crate::hash::sha256(ajtai::params_descriptor(n, &b_sha).as_bytes())) != leaf.params_sha256 {
        return Err(format!("{}: params_sha256 is not the digest of the parameter descriptor (verifier build error)", leaf.display));
    }
    chain_key(sys, n, &b).map(|_| ()).map_err(|e| {
        format!("system file: the Ajtai chain the digests come from does not use the derived key B (SHAKE-256 of {:?}): {e}",
            String::from_utf8_lossy(ajtai::B_TAG))
    })
}

/// Red-team-leaf-2 G1: the key rows are found by following the digests the statement publishes into the system, never
/// by their coefficient pattern (a pattern scan accepts a chain on `B = 0` plus decoy rows carrying the derived key over
/// rows nothing else reads).  From each public lane `hash.d[x]` (x < N: the x digest, else the W digest): the one
/// linear row `W[pin row] = d[x]`; the one quadratic touching that row, `is_end * (out - d) = 0` (`hashchain.compose`);
/// the one linear row touching `out`, which must be lane x's key row `out - (X acc_in)[x mod N] - sum_i B[x mod N][o +
/// i] bit_i` with `acc_in` the linked row the chain carries the NEIGHBOUR lane's `out` into (`y = out`, 0 at a chain
/// start) and `o` the lane's key block; the bit rows are the same 256 rows (one per bit position) in all N key rows
/// of a role, increasing in bit position (word-major, LSB first: the compose's allocation order), boolean-constrained,
/// and disjoint from the other role's and from every chain / pin row.  Returns the two roles' bit rows.
/// Scope: under `--allow-any-system` this proves the published digests are the derived-key Ring-SIS digests of those
/// boolean rows; that the rows are the operands the relation multiplies is the system pin's job.
pub fn chain_key(sys: &crate::format::System, n: usize, b: &[Vec<u32>]) -> Result<[Vec<u32>; ajtai::KEYS], String> {
    use std::collections::{HashMap, HashSet};
    let nd = ajtai::KEYS * n;
    let single = |e: &crate::format::Expr| -> Option<u32> {
        (e.c == 0 && e.lin.len() == 1 && e.lin[0].1 == 1).then(|| e.lin[0].0)
    };
    // (1) the public lanes and the one linear row that carries each onto its pin row
    let mut end_pin = None;
    let mut d_pin = vec![None; nd];
    for (pi, (name, _)) in sys.pins.iter().enumerate() {
        if name == "hash.is_end" {
            if end_pin.replace(pi).is_some() {
                return Err("hash.is_end is pinned twice".into());
            }
        } else if let Some(x) = name.strip_prefix("hash.d[").and_then(|s| s.strip_suffix(']')).and_then(|s| s.parse::<usize>().ok()) {
            if x >= nd || d_pin[x].replace(pi).is_some() {
                return Err(format!("pin {name}: not one of the {nd} digest lanes, or pinned twice"));
            }
        }
    }
    let mut carried: Vec<Vec<u32>> = vec![Vec::new(); sys.pins.len()];
    for c in &sys.linear {
        if let Some(pi) = c.pub_pin {
            carried[pi].push(single(&c.e).filter(|&r| r == sys.pins[pi].1).unwrap_or(u32::MAX));
        }
    }
    let pin_row = |pi: Option<usize>, what: &str| -> Result<u32, String> {
        let pi = pi.ok_or_else(|| format!("{what} is not pinned"))?;
        match carried[pi][..] {
            [r] if r != u32::MAX => Ok(r),
            _ => Err(format!("{what} does not enter through exactly one constraint W[pin row] = pub")),
        }
    };
    let end_row = pin_row(end_pin, "hash.is_end")?;
    let d_row: Vec<u32> = (0..nd).map(|x| pin_row(d_pin[x], &format!("hash.d[{x}]"))).collect::<Result<_, _>>()?;
    let lane_of: HashMap<u32, usize> = d_row.iter().enumerate().map(|(x, &r)| (r, x)).collect();
    if lane_of.len() != nd || lane_of.contains_key(&end_row) {
        return Err("two public lanes share a pin row".into());
    }
    // (2) the end tie of each lane: the one quadratic touching its pin row is is_end * (out - d) = 0
    let mut out: Vec<Option<u32>> = vec![None; nd];
    for q in &sys.quadratic {
        let lanes: Vec<usize> = q.iter().flat_map(|e| e.lin.iter()).filter_map(|(r, _)| lane_of.get(r).copied()).collect();
        let x = match lanes[..] {
            [] => continue,
            [x] => x,
            _ => return Err("a quadratic constraint touches two digest lanes".into()),
        };
        let dr = d_row[x];
        let tie = single(&q[0]) == Some(end_row) && q[2].c == 0 && q[2].lin.is_empty() && q[1].c == 0 && q[1].lin.len() == 2
            && q[1].lin.iter().any(|&t| t == (dr, P - 1));
        let o = q[1].lin.iter().find(|&&(r, v)| r != dr && v == 1).map(|&(r, _)| r);
        match (tie, o) {
            (true, Some(o)) if out[x].is_none() => out[x] = Some(o),
            _ => return Err(format!("hash.d[{x}]: not tied by exactly one is_end * (out - d) = 0")),
        }
    }
    let out: Vec<u32> = out.iter().enumerate().map(|(x, o)| o.ok_or_else(|| format!("hash.d[{x}] is tied to no chain row")))
        .collect::<Result<_, _>>()?;
    let out_set: HashSet<u32> = out.iter().copied().collect();
    if out_set.len() != nd || out_set.contains(&end_row) || out.iter().any(|r| lane_of.contains_key(r)) {
        return Err("two digest lanes are tied to one row, or to a pin row".into());
    }
    // (3) the linked row each out is carried into (acc_in at the next column; 0 at a chain start)
    let chain = sys.chain.as_ref().ok_or("the hashed system has no chain")?;
    let mut acc_of: HashMap<u32, u32> = HashMap::with_capacity(nd);
    for (&c, y) in chain.c_rows.iter().zip(&chain.y) {
        if let Some(o) = single(y).filter(|o| out_set.contains(o)) {
            if acc_of.insert(o, c).is_some() {
                return Err(format!("chain row {o} is linked twice"));
            }
        }
    }
    let acc: Vec<u32> = out.iter().map(|o| acc_of.get(o).copied().ok_or_else(|| format!("chain row {o} is tied to a digest lane but never linked")))
        .collect::<Result<_, _>>()?;
    let acc_set: HashSet<u32> = acc.iter().copied().collect();
    if acc_set.len() != nd || acc_set.iter().any(|r| out_set.contains(r) || lane_of.contains_key(r) || *r == end_row) {
        return Err("the carried rows of two lanes coincide, or a carried row is a digest / pin row".into());
    }
    // (4) the one linear row touching each out: lane x's key row, exactly
    let mut defs: HashMap<u32, Vec<usize>> = HashMap::with_capacity(nd);
    for (qi, c) in sys.linear.iter().enumerate() {
        for (r, _) in &c.e.lin {
            if out_set.contains(r) {
                defs.entry(*r).or_default().push(qi);
            }
        }
    }
    let boolean: HashSet<u32> = sys.quadratic.iter().filter_map(|q| {
        let r = single(&q[0])?;
        (single(&q[1]) == Some(r) && single(&q[2]) == Some(r)).then_some(r)
    }).collect();
    let mut all_bits: HashSet<u32> = HashSet::with_capacity(ajtai::KEYS * ajtai::M_COL);
    let mut roles: [Vec<u32>; ajtai::KEYS] = Default::default();
    for (k, role_bits) in roles.iter_mut().enumerate() {
        let mut bit_row: Vec<Option<u32>> = vec![None; ajtai::M_COL];
        for xx in 0..n {
            let x = k * n + xx;
            let (o, (prev, prev_coef)) = (out[x], if xx == 0 { (acc[k * n + n - 1], 1) } else { (acc[x - 1], P - 1) });
            let c = match defs.get(&o).map(|v| &v[..]) {
                Some([qi]) => &sys.linear[*qi],
                _ => return Err(format!("hash.d[{x}]: its chain row is not defined by exactly one linear constraint")),
            };
            if c.pub_pin.is_some() || c.e.c != 0 {
                return Err(format!("hash.d[{x}]: the key row has a constant or a public right-hand side"));
            }
            let blk = &b[xx][k * ajtai::M_COL..(k + 1) * ajtai::M_COL];
            let pos: HashMap<u32, usize> = blk.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (P - v, i)).collect();
            if pos.len() != blk.iter().filter(|&&v| v != 0).count() {
                return Err(format!("row {xx} of the derived B repeats a coefficient (verifier build error)"));
            }
            let (mut seen_o, mut seen_prev, mut nbits) = (false, false, 0usize);
            for &(r, v) in &c.e.lin {
                if r == o && v == 1 {
                    seen_o = true;
                } else if r == prev && v == prev_coef {
                    seen_prev = true;
                } else {
                    let i = *pos.get(&v).ok_or_else(|| format!("hash.d[{x}]: row {r} has coefficient {v}, not an entry of the derived key row"))?;
                    match bit_row[i] {
                        None => bit_row[i] = Some(r),
                        Some(r0) if r0 == r => {}
                        Some(r0) => return Err(format!("key role {k}: bit position {i} is row {r0} in one key row and {r} in another")),
                    }
                    nbits += 1;
                }
            }
            if !(seen_o && seen_prev && nbits == pos.len()) {
                return Err(format!("hash.d[{x}]: the key row is not out - (X acc_in)[{xx}] - B[{xx}] bits over the linked neighbour"));
            }
        }
        *role_bits = bit_row.iter().enumerate().map(|(i, r)| r.ok_or_else(|| format!("key role {k}: bit position {i} is never keyed")))
            .collect::<Result<_, _>>()?;
        if role_bits.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!("key role {k}: the bit rows are not in bit-position order"));
        }
        for r in role_bits.iter() {
            if !boolean.contains(r) {
                return Err(format!("key role {k}: bit row {r} is not boolean-constrained"));
            }
            if !all_bits.insert(*r) || out_set.contains(r) || acc_set.contains(r) || lane_of.contains_key(r) || *r == end_row
                || chain.c_rows.contains(r) {
                return Err(format!("key role {k}: bit row {r} is shared with the other role or is a chain / pin row"));
            }
        }
    }
    Ok(roles)
}

// ---------------------------------------------------------------------------------------------------------------------
// BLAKE3 (`leaf/blake3.py`, lane blake3-leaf): the relation proves the chunk chaining values of the row (keyed mode, one
// key per operand role: `verity/blake3-leaf/v1/x` for x rows, `.../w` for W columns), the digest row is
// `[n_chunks + 4 role | CV_0 as 16 little-endian 16-bit limbs | CV_1 | CV_2]` (unused slots 0) and the leaf is the
// standard BLAKE3 root: the left-heavy parent fold of the CVs with the role's key (`PARENT`, `ROOT` on top) -- i.e.
// exactly `blake3::Hasher::new_keyed(key(role)).update(row_bytes).finalize()`.  The compression function is written out
// here (no crates on the pods); it only runs 1-2 times per committed row.  `leaf_bytes` is total on field rows of the
// right length (the Python contract's conformance suite feeds random rows): a row that is not a well-formed frame maps
// to the tagged SHA-256 of its elements, a domain no BLAKE3 root of a row lands in except by collision.
// ---------------------------------------------------------------------------------------------------------------------

const BLAKE3_KEY_X: [u8; 32] = *b"verity/blake3-leaf/v1/x\0\0\0\0\0\0\0\0\0";
const BLAKE3_KEY_W: [u8; 32] = *b"verity/blake3-leaf/v1/w\0\0\0\0\0\0\0\0\0";
const BLAKE3_MALFORMED_TAG: &[u8] = b"verity/blake3-leaf/malformed-digest/v1\0";
const BLAKE3_MAX_CHUNKS: usize = 3;
const BLAKE3_DIGEST_ELEMS: usize = 1 + 16 * BLAKE3_MAX_CHUNKS;
const BLAKE3_MAX_CHUNKS_SIZED: usize = 63;
const BLAKE3_MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const BLAKE3_IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
const BLAKE3_PARENT: u32 = 4;
const BLAKE3_ROOT: u32 = 8;
const BLAKE3_KEYED_HASH: u32 = 16;
const BABYBEAR_P: u32 = 0x7800_0001;

/// `leaf.blake3.key_words(role)`: role 1 = x rows, 2 = W columns.
fn blake3_key_words(role: u32) -> [u32; 8] {
    let key = if role == 1 { &BLAKE3_KEY_X } else { &BLAKE3_KEY_W };
    let mut k = [0u32; 8];
    for (i, w) in k.iter_mut().enumerate() {
        *w = u32::from_le_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
    }
    k
}

fn blake3_g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

/// The BLAKE3 compression function, truncated to the chaining value (`v[i] ^ v[i + 8]`, 8 words).
fn blake3_compress_cv(cv: &[u32; 8], block: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 8] {
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(cv);
    v[8..12].copy_from_slice(&BLAKE3_IV[..4]);
    v[12] = counter as u32;
    v[13] = (counter >> 32) as u32;
    v[14] = block_len;
    v[15] = flags;
    let mut m = *block;
    for r in 0..7 {
        blake3_g(&mut v, 0, 4, 8, 12, m[0], m[1]);
        blake3_g(&mut v, 1, 5, 9, 13, m[2], m[3]);
        blake3_g(&mut v, 2, 6, 10, 14, m[4], m[5]);
        blake3_g(&mut v, 3, 7, 11, 15, m[6], m[7]);
        blake3_g(&mut v, 0, 5, 10, 15, m[8], m[9]);
        blake3_g(&mut v, 1, 6, 11, 12, m[10], m[11]);
        blake3_g(&mut v, 2, 7, 8, 13, m[12], m[13]);
        blake3_g(&mut v, 3, 4, 9, 14, m[14], m[15]);
        if r < 6 {
            let mut p = [0u32; 16];
            for i in 0..16 {
                p[i] = m[BLAKE3_MSG_PERMUTATION[i]];
            }
            m = p;
        }
    }
    let mut out = [0u32; 8];
    for i in 0..8 {
        out[i] = v[i] ^ v[i + 8];
    }
    out
}

/// The left-heavy parent fold of `nodes` (the left subtree holds the largest power of two strictly below the count).
fn blake3_fold(nodes: &[[u32; 8]], role: u32, top: bool) -> [u32; 8] {
    if nodes.len() == 1 {
        return nodes[0];
    }
    let k = 1usize << (usize::BITS - 1 - (nodes.len() - 1).leading_zeros());
    let (l, r) = (blake3_fold(&nodes[..k], role, false), blake3_fold(&nodes[k..], role, false));
    let mut block = [0u32; 16];
    block[..8].copy_from_slice(&l);
    block[8..].copy_from_slice(&r);
    let flags = BLAKE3_KEYED_HASH | BLAKE3_PARENT | if top { BLAKE3_ROOT } else { 0 };
    blake3_compress_cv(&blake3_key_words(role), &block, 0, 64, flags)
}

/// `leaf.blake3.leaf_bytes`: the 32-byte BLAKE3 root of the published chunk CVs, folded with the header's role key.  A
/// wrong length or a non-field element yields NO bytes (the leaf cannot open); a malformed frame (header not `n_chunks
/// (2..=3) + 4 role (1, 2)`, a limb >= 2^16, a non-zero unused slot) yields the tagged SHA-256 of the elements.  The
/// frame's chunk slots `ncm` are the row's length `1 + 16 ncm`: 3 for the registry frame, more for a SIZED frame
/// (`leaf.blake3.Blake3Leaf.for_row`, [`sized`]), whose header is `n_chunks (2..=ncm) + 64 role`.
fn blake3_leaf_bytes(elems: &[u32]) -> Vec<u8> {
    if elems.len() < BLAKE3_DIGEST_ELEMS || (elems.len() - 1) % 16 != 0 || (elems.len() - 1) / 16 > BLAKE3_MAX_CHUNKS_SIZED
        || elems.iter().any(|&x| x >= BABYBEAR_P) {
        return Vec::new();
    }
    let ncm = (elems.len() - 1) / 16;
    let sh = if ncm <= BLAKE3_MAX_CHUNKS { 2 } else { 6 };
    let (nch, role) = ((elems[0] & ((1 << sh) - 1)) as usize, elems[0] >> sh);
    if !(2..=ncm).contains(&nch) || !(1..=2).contains(&role) || elems[1 + 16 * nch..].iter().any(|&x| x != 0)
        || elems[1..1 + 16 * nch].iter().any(|&x| x >= 1 << 16) {
        let mut h = crate::hash::Sha256::new();
        h.update(BLAKE3_MALFORMED_TAG);
        for &x in elems {
            h.update(&x.to_le_bytes());
        }
        return h.finalize().to_vec();
    }
    let mut cvs = vec![[0u32; 8]; nch];
    for c in 0..nch {
        for w in 0..8 {
            cvs[c][w] = elems[1 + 16 * c + 2 * w] | (elems[1 + 16 * c + 2 * w + 1] << 16);
        }
    }
    let root = blake3_fold(&cvs, role, true);
    let mut out = Vec::with_capacity(32);
    for w in root {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// `leaf.blake3.Blake3Leaf` (lane blake3-leaf): `schema` / `params_sha256` are the module's `SCHEMA` and
/// `params_sha256()` (SHA-256 of the two keys and the framing constants).
pub static BLAKE3: LeafScheme = LeafScheme {
    name: "blake3",
    display: "BLAKE3",
    schema: "blake3-keyed/row/v2",
    params_sha256: "6654cdb9680d9f9c782092d0cb7cf0820bcba2612b28742c2b48cdd8957785a6",
    digest_elems: BLAKE3_DIGEST_ELEMS,
    leaf_bytes: blake3_leaf_bytes,
};

// SHA-256 (`leaf/sha256.py`, lane b-ligero-sha256): the core frame-v3 row leaf `sha256/row/v1` =
// `SHA-256(prefix64(role, word_bits, n_words) || row_bytes)`.  The relation proves the data-block compressions from the
// prefix's midstate and publishes the chaining value after the last data block (`[role + 4 word_bits + 256 n_words |
// H_0..H_7 as 16-bit limbs, low first]`); the verifier runs the padding compression here (the row is whole blocks, so the
// last block is `0x80`, zeros, the bit length of the 64 + n_words word_bits / 8 message bytes) -- the leaf is exactly
// `sha256(prefix || row_bytes)`.  A malformed frame (bad header, a limb >= 2^16) maps to the tagged SHA-256 of the elements.
const SHA256_MALFORMED_TAG: &[u8] = b"verity/sha256-leaf/malformed-digest/v1\0";
const SHA256_DIGEST_ELEMS: usize = 17;

/// `leaf.sha256.leaf_bytes`: the padding compression of the published chaining value, big-endian.  Wrong length or a
/// non-field element: NO bytes (the leaf cannot open).
fn sha256_leaf_bytes(elems: &[u32]) -> Vec<u8> {
    if elems.len() != SHA256_DIGEST_ELEMS || elems.iter().any(|&x| x >= BABYBEAR_P) {
        return Vec::new();
    }
    let h0 = elems[0];
    let (role, wb, n) = (h0 & 3, (h0 >> 2) & 63, (h0 >> 8) as u64);
    let n_bytes = n * u64::from(wb) / 8;
    if !(1..=2).contains(&role) || ![8, 16, 32].contains(&wb) || n == 0 || n_bytes % 64 != 0 || elems[1..].iter().any(|&x| x >= 1 << 16) {
        let mut h = crate::hash::Sha256::new();
        h.update(SHA256_MALFORMED_TAG);
        for &x in elems {
            h.update(&x.to_le_bytes());
        }
        return h.finalize().to_vec();
    }
    let mut cv = [0u32; 8];
    for (w, slot) in cv.iter_mut().enumerate() {
        *slot = elems[1 + 2 * w] | (elems[2 + 2 * w] << 16);
    }
    let mut pad = [0u8; 64];
    pad[0] = 0x80;
    pad[56..].copy_from_slice(&(8 * (64 + n_bytes)).to_be_bytes());
    crate::hash::Sha256::compress(&mut cv, &pad);
    let mut out = Vec::with_capacity(32);
    for w in cv {
        out.extend_from_slice(&w.to_be_bytes());
    }
    out
}

/// `leaf.sha256.Sha256Leaf` (lane b-ligero-sha256): `schema` is the core `rowleaf.SCHEMA_SHA256_ROW`, `params_sha256` the
/// module's `params_sha256()`.
pub static SHA256: LeafScheme = LeafScheme {
    name: "sha256",
    display: "SHA-256",
    schema: "sha256/row/v1",
    params_sha256: "f23f6deff5feda68f5d242933d4634d5921737697c2656a33a9f12fd15c7738b",
    digest_elems: SHA256_DIGEST_ELEMS,
    leaf_bytes: sha256_leaf_bytes,
};

/// `leaf.blake3_xob.Blake3XobLeaf` (lane b-ligero-standard-hash): the `blake3` leaf -- schema, keys, framing, digest
/// layout -- proved by the XOR-output-bits compression circuit, so only its `PINS` differ from `BLAKE3`'s.
pub static BLAKE3_XOB: LeafScheme = LeafScheme {
    name: "blake3-xob",
    display: "BLAKE3 (xob circuit)",
    schema: "blake3-keyed/row/v2",
    params_sha256: "6654cdb9680d9f9c782092d0cb7cf0820bcba2612b28742c2b48cdd8957785a6",
    digest_elems: BLAKE3_DIGEST_ELEMS,
    leaf_bytes: blake3_leaf_bytes,
};

/// The SIZED BLAKE3 frames (`leaf.blake3.Blake3Leaf.for_row`): rows longer than 3 chunks (K = 2048 BF16: 4, K = 8192 E4M3: 8,
/// K = 8192 BF16: 16) publish one CV slot per chunk; `params_sha256` is `leaf.blake3.params_sha256(ncm)`.  Same names as the
/// registry frames (a statement names `+blake3` / `+blake3-xob`), picked by [`sized`] from the relation's row length.
macro_rules! blake3_sized {
    ($name:literal, $display:literal, $ncm:literal, $params:literal) => {
        LeafScheme { name: $name, display: $display, schema: "blake3-keyed/row/v2", params_sha256: $params,
                     digest_elems: 1 + 16 * $ncm, leaf_bytes: blake3_leaf_bytes }
    };
}
static BLAKE3_SIZED: &[(usize, LeafScheme)] = &[
    (4, blake3_sized!("blake3", "BLAKE3", 4, "2f5bbfe9e0f21c504dc2e7c02dcd8bacbf274f658c934b04943eba25dab19686")),
    (8, blake3_sized!("blake3", "BLAKE3", 8, "c5308f856bda25a7fbb9af11fc85e33d227738a81ae4305672a96de2a40d2b84")),
    (16, blake3_sized!("blake3", "BLAKE3", 16, "0b2dfbd3daee31b97044e71ca94c31f0c7d84d35e41b51b0ad2dbf4f5f41f63c")),
    (4, blake3_sized!("blake3-xob", "BLAKE3 (xob circuit)", 4, "2f5bbfe9e0f21c504dc2e7c02dcd8bacbf274f658c934b04943eba25dab19686")),
    (8, blake3_sized!("blake3-xob", "BLAKE3 (xob circuit)", 8, "c5308f856bda25a7fbb9af11fc85e33d227738a81ae4305672a96de2a40d2b84")),
    (16, blake3_sized!("blake3-xob", "BLAKE3 (xob circuit)", 16, "0b2dfbd3daee31b97044e71ca94c31f0c7d84d35e41b51b0ad2dbf4f5f41f63c")),
];

/// `leaf.registry.sized`: `leaf` framed for committed rows of `row_bytes` bytes -- itself unless it is a BLAKE3 frame and the
/// row has more than 3 chunks, then the sized frame with one CV slot per chunk (an error when this build has none).
pub fn sized(leaf: &'static LeafScheme, row_bytes: usize) -> Result<&'static LeafScheme, String> {
    if !std::ptr::eq(leaf, &BLAKE3) && !std::ptr::eq(leaf, &BLAKE3_XOB) {
        return Ok(leaf);
    }
    let nch = row_bytes.div_ceil(1024);
    if nch <= BLAKE3_MAX_CHUNKS {
        return Ok(leaf);
    }
    BLAKE3_SIZED.iter().find(|(n, s)| *n == nch && s.name == leaf.name).map(|(_, s)| s)
        .ok_or_else(|| format!("{}: rows of {nch} chunks ({row_bytes} bytes) have no frame in this build (4, 8, 16)", leaf.display))
}

// vllm-v1 (`leaf/vllm_v1.py`, lane b-ligero-vllm-v1): the relation proves every compression of the position leaf
// `SHA-256("verity/pos-leaf/v0" || u64be(len) || row_bytes)` and publishes the digest as 16 limbs (big-endian words, low
// limb first); the leaf value IS those 32 bytes, the tree leaf itself (`vllm_v1.rs` folds and binds the trees).  A
// malformed row (a limb >= 2^16) maps to the tagged SHA-256 of its elements.
const VLLM_MALFORMED_TAG: &[u8] = b"verity/vllm-v1-leaf/malformed-digest/v1\0";
const VLLM_DIGEST_ELEMS: usize = 16;

fn vllm_leaf_bytes(elems: &[u32]) -> Vec<u8> {
    if elems.len() != VLLM_DIGEST_ELEMS || elems.iter().any(|&x| x >= BABYBEAR_P) {
        return Vec::new();
    }
    if elems.iter().any(|&x| x >= 1 << 16) {
        let mut h = crate::hash::Sha256::new();
        h.update(VLLM_MALFORMED_TAG);
        for &x in elems {
            h.update(&x.to_le_bytes());
        }
        return h.finalize().to_vec();
    }
    let mut out = Vec::with_capacity(32);
    for w in 0..8 {
        out.extend_from_slice(&(elems[2 * w] | (elems[2 * w + 1] << 16)).to_be_bytes());
    }
    out
}

/// `leaf.vllm_v1.VllmV1Leaf`: `schema` / `params_sha256` are the module's `SCHEMA` / `params_sha256()`.
pub static VLLM_V1: LeafScheme = LeafScheme {
    name: "vllm-v1",
    display: "vllm-v1 position leaf (SHA-256)",
    schema: "vllm-v1/pos-leaf/v0",
    params_sha256: "e613f010bd33edecbb4a00b5fb62d9ede1a68a31a5e791e71cb584f6e42ba314",
    digest_elems: VLLM_DIGEST_ELEMS,
    leaf_bytes: vllm_leaf_bytes,
};

/// Whether statements of `leaf` commit under `vllm-v1` port trees (`vllm_v1.rs`) instead of frame-v3 indexed trees.
pub fn is_vllm(leaf: &LeafScheme) -> bool {
    std::ptr::eq(leaf, &VLLM_V1)
}

/// Every scheme this build knows.
pub static SCHEMES: &[&LeafScheme] = &[&POSEIDON2, &BLAKE3, &BLAKE3_XOB, &SHA256, &VLLM_V1, &AJTAI_N64, &AJTAI_N128];

/// `leaf.registry.DEFAULT` / `ALIASES`: a relation string without a suffix, or with `+hash`, names Poseidon2.
pub const DEFAULT: &str = "poseidon2";
const ALIASES: &[(&str, &str)] = &[("hash", "poseidon2")];

/// The `(sys_id, table_digest)` pins of hashed systems composed with a scheme OTHER than Poseidon2 (whose pins live on
/// the `Relation` itself: `hashed_sys_id` / `hashed_table_digest`): `(relation name, leaf name, sys_id, table_digest)`.
pub static PINS: &[(&str, &str, &str, &str)] = &[
    ("fp8-ada", "ajtai-n64", "18d915323c056495f667a6c4286eece77e25e348aed03b335b583e2db1e346c4", "a1f2cb20cb549af68f005bf3efef6112f08bb7a9537dbd876811e46769c5ee50"),
    ("bf16-hopper", "ajtai-n128", "29f18689c05de6f355c97aac1e711f9dad7f7f35eda85026482258c85d6a27e5", "e4dceccece10d7f55f5aed8cb7e444d6f0a736012b6323ee6fd2cdfe520f6438"),
    // lane blake3-leaf-2, schema `blake3-keyed/row/v2`: `hashchain.compose(<rel>, "blake3")`, each gated 2048 VUs + 86
    // negatives on vy-blake3-leaf 2026-09-23 21:00Z.  x1 relations: half a block per operand per column (paired columns).
    ("fp8-ada", "blake3", "71f39e447b62fc8cc510c29bc1b9d35a0e53fbf71d59b5fac919a240dd326932",
     "bbacbe7a1567142eaaf79c40c34df6f6bdaa0c8d65c9bb7a43ff0e082e37fc52"),
    ("bf16-hopper", "blake3", "58ef7097798f138e0740f850bab9ad9af458f53ee60636f4547cff91c60caf47",
     "c681b7a15f97eb9c1a5f4e5f95ba8c0b3dc1f8cbbb328da315d50d2a38e03c84"),
    // the x4 fold: 2 compressions per role per column
    ("fp8-ada-x4", "blake3", "1168788f2cf6ab7deaa0974f17ddd981ae972aedccd5c851c99017ae957b705b",
     "43e0eea0e07a9a9de151867609aef4ae4a99b8b7ff26a5394f59bac67cecd6f1"),
    // lane b-ligero-standard-hash, same schema: bf16-ampere gated 2048 VUs (49 honest sub-batches) + 86 negatives on
    // vy-b-ligero-sh 2026-09-25 06:51Z; fp8-hopper likewise
    ("bf16-ampere", "blake3", "5b7620546f5cfda0391974239efd78988fe41556e23f6c07d98bd4513c952d44",
     "f67ab8174346973350e673f0b1164a3a824e5c9f4f8ae48d136eabdfd970f374"),
    ("fp8-hopper", "blake3", "433bdfc37ef7ce64dcac32cdfd9b99e43b33f0df848112e96fd01601531a3bd1",
     "a27af32fe5ef8b8995b5cc4d874bcbac3a7f5196600e5cce062537ab7d1a55d8"),
    // lane b-ligero-standard-hash, the blake3 leaf under the XOR-output-bits compression (`hashchain.compose(<rel>,
    // "blake3-xob")`: 28,584 rows for fp8-ada against 35,370), fixtures on vy-b-ligero-sh 2026-09-25 11:39Z
    ("fp8-ada", "blake3-xob", "3d6cc67b686f8b146d65acd55598b687c29f4b7fea1ce5893baa6e0b865c0b07",
     "a655b6b8fe461a17ae2ff3e460e6ebe5088d980fab56657291f7d73ff6e92c2b"),
    ("fp8-ada-x4", "blake3-xob", "f90e7b41920155ae9f0e58d8985adf0d00b711aa369922d9054c1444ad3e3dc8",
     "6ecf18dabee9959687be6c2b71a3c932d94738f66f60ca16b43513edb7e921c9"),
    // lane x4-hopper-blake3, schema `blake3-keyed/row/v2`: fp8-hopper-x4 (77,692 rows) gated 2048 VUs (7 honest
    // sub-batches) + 86 negatives, bf16-hopper-x4 (76,751 rows) 2048 VUs (13) + 86, at cd963fd4 on vy-x4-hopper-blake3-h100,
    // r20260925-174812-50ad, 2026-09-25 17:51Z
    ("fp8-hopper-x4", "blake3", "3009b5fb01b69f2705e307469044a1659cbb1acfac3c178b55447928957ad533",
     "5a9f049bd62745656e0f05e74c5d8631f6ab6719c6a911a2594a053229127087"),
    ("bf16-hopper-x4", "blake3", "14b9ba1a0136c04894a6a7fb46b2c0244e5ff010e763492d164f2f10f75ac90b",
     "978537834571eadb47e4f692f1db5cec150ea759e7c801e7d7c42133c6a00ab6"),
    // the same under blake3-xob: fp8-hopper-x4 64,120 rows, bf16-hopper-x4 63,179, each gated 2048 VUs + 86 negatives at
    // 9a78cd68, r20260925-211046-ede9, 2026-09-25 21:12Z
    ("fp8-hopper-x4", "blake3-xob", "be64f3a50a676405661dbd3d30e75161446a907decba582a634dda62205f8f80",
     "98db3f37149cd7d002194f5a3e4f6bced4ee93b000d267031c848b36573790a7"),
    ("bf16-hopper-x4", "blake3-xob", "e456b36ac97a570c8a8ea3fa44eb41d9b7db21c76b95d64be238c4eb55ab93d2",
     "bbc8d0b634c502a9d38fc979631ad40f6d898b5fe6a2fa8ef17f83fd0d94c585"),
    // lane b-ligero-sha256, schema `sha256/row/v1`: `hashchain.compose(<rel>, "sha256")` (data blocks in-circuit, 18,128
    // rows per block), each gated 2048 VUs (7 honest sub-batches) + 86 negatives on vy-b-ligero-sha256 2026-09-25 09:05Z
    ("fp8-hopper-x4", "sha256", "6cf20505b38e142533dfa1011a65bd1fb36b3b4792854916dffaeaddfd7f7e9f",
     "76c1ce7f09b99fac28c5576f95a5f037e68e7195ce2f14d7532256835338c40c"),
    ("fp8-ada-x4", "sha256", "d6b0cd8d69cb7ff504ebeb5cef28dc43b57bd318dcbb325229572959ec3282ff",
     "b04a579ceb48fb57a3c18159b13a86b600c6aadb03fa2f10e3510b75d5af9b3f"),
    // gated 2048 VUs (13 honest sub-batches) + 86 negatives at 98d878ca, r20260925-104053-b438, 2026-09-25 10:43Z
    ("bf16-hopper-x4", "sha256", "a02f283de13acd94d47be4e8c794a6c89b833cc2d9151b92393dd6ac37defe62",
     "1b879d1a3ec16e23f886f792900a089fcbc4b080dbddffe4d45a07d64551525a"),
    // lane x4-sha256-fill: gated 2048 VUs (13 honest sub-batches) + 86 negatives at cd963fd4 on vy-x4-sha256-a100 (A100
    // SXM4 80GB), r20260925-175055-0792, 2026-09-25 17:54Z; m = 90,165 rows/col
    ("bf16-ampere-x4", "sha256", "a862f7a041678ab8feca6dbb6433c91528ed0fe8b63b806fd439c93f766d94e5",
     "848a99cd6ff5b314d46229a2cb27abd4f39cc13f3729b31ad70e396ee6949326"),
    // lane b-ligero-vllm-v1, schema `vllm-v1/pos-leaf/v0`: `hashchain.compose(<rel>, "vllm-v1")` (every position-leaf
    // compression in-circuit, 3 per operand per x4 column; 127,536 rows per column), gated 2048 VUs (13 honest sub-batches)
    // + 86 negatives, 0 failures, on vy-b-ligero-vllm-v1-4090 r20260925-180400-c457 2026-09-25 18:12Z
    ("fp8-ada-x4", "vllm-v1", "f7f316139a176eaed1c52a09869d748fc9fd54ef99e9425a9feae041919446d4",
     "8330132438e6c6adbb25a1371afb3d0dcf580af3d7a5ce53ecd141c983830479"),
    // gated 2048 VUs (7 honest sub-batches) + 86 negatives, 0 failures, on vy-b-ligero-vllm-v1-h100 r20260925-181202-65f3
    ("fp8-hopper-x4", "vllm-v1", "69054deb439010a356cb4de6873b2d8d7b0e9a27125119d51062c441238a2f36",
     "23405e55ff9eadc12d638d9ab2555e59c6f049e9bcfa669dcc5cd953779da3d8"),    // lane bligero-real-k: the x4 folds at vLLM's reduction lengths (relations._at_k), `hashchain.compose(<fold>-k<K>, leaf)`;
    // BLAKE3 frames sized to the row (4 / 16 CV slots for BF16 K = 2048 / 8192, 8 for E4M3 K = 8192), SHA-256 unchanged.
    // Digests of the compiled systems (`ligero-verify system-digest` on `system_bytes(compose(...).sys)`, lane VM, torch-free
    // compile), equal to the pod's fixture system.bin digests; gadget gate r20260926-042639-c562 (vy-bligero-real-k-gate, L40S):
    // every system 8 honest sub-batches accepted + 86 negatives rejected, 0 failures, and its fixture verified pinned
    ("bf16-ampere-x4-k2048", "blake3-xob", "b5ee1ee0b6645eb83d5f973f019c216bdfd0907560f7ab3c3d623447b944c546",
     "3b9f7d64a550c86efe1be8bc0aeb71db57835e91631b630bfb35c38d4de0f626"),
    ("bf16-ampere-x4-k2048", "sha256", "232769b4d5b3b8f8a17dbcd887b6332b22642b6722ff086308d1c829b84f7b7f",
     "b318064545e39fa11a82d3c7e261eff25d63a2b18b5aad2ead7061ed0f4a1612"),
    ("bf16-ampere-x4-k8192", "blake3-xob", "467774bd04f30dd0bb0edc71da233401967e701494fb73177047e06bb8937b02",
     "4d0656ccf55389f53e7d3ee7d9beda478fa0368ec984cc7b669ee30dd5631b8b"),
    ("bf16-ampere-x4-k8192", "sha256", "66f7b5f7dbfaf69875a8fa76c94d5b2a9728bd7262fdf2f8cd3131e1440086ce",
     "1ebba29494e191fceeb5a7aedddb249d960a062475d507ca75cf74ff827f31f1"),
    ("bf16-hopper-x4-k2048", "blake3-xob", "97e39093e76e8fabf8e6baa27bb27bc233bbf68676668e3a26399c988545596c",
     "48a1f6bd689b8d3fc789c2f557bc502961952925eafb7baf4eba7e4b9f30398a"),
    ("bf16-hopper-x4-k2048", "sha256", "0dff3d58df9d5bb258265186d5875a3e003322063c9744ae3ae6277954416250",
     "8399ac852779de99f1a1d922e403ed980159d4a3882ae5cd689568ac01f36172"),
    ("bf16-hopper-x4-k8192", "blake3-xob", "58c3713e8c21bc602d8bf5c3ed53c043fde0c621633efe5041f0ca0bfcf7ee7d",
     "4057035fc39d84252fa19ba660421b95c5e7de8461c8fd4bc9ee92c002e0e386"),
    ("bf16-hopper-x4-k8192", "sha256", "de848ba909129bf8d21e8328d3aca3bc93b744e6ddf8d7df5182d75954424109",
     "0e88fc9a3b8a145764f7fe2471cd7c72e86467f14c9cfec5cb2a0af163bf983c"),
    ("fp8-ada-x4-k2048", "blake3-xob", "3f1e58f8c4619826b8d9b856d71f6fc9e4587955735b874703032a5d37380c6d",
     "b1e6f4fe9578cb60a520266d151e876a21b7811bc67c8a0a0a252b69ca080195"),
    ("fp8-ada-x4-k2048", "sha256", "a3dfdb341bdbf5f1fbe6931f06ea9d7ae625177432fa5a239f48d6e7b7ce37c6",
     "953e5dc516306bb200a149857d9d08e131315f1bd303a8153fd0ee08479fb4d1"),
    ("fp8-ada-x4-k8192", "blake3-xob", "103304c5576050eebb276067b999f0bfa72998b6e84e9553626c264ae1be05db",
     "987ccd8060988fd7835dc76bd3daf8a6e3cd6e0b34fff4d770376bbc38697e1f"),
    ("fp8-ada-x4-k8192", "sha256", "8219187a7d2740d37201b7969f57993341afad41bdf4115a16f2cc95ded8ee53",
     "bec158bb520caa606351ce4254c5ff87f486552cab5b3b35fb74e7c57f36a1b4"),
    ("fp8-hopper-x4-k2048", "blake3-xob", "b6baf112e5c6167ae78274d09a5d6b235c614d06c94f83bc33c8751a86ced044",
     "2ac49ed575cf6c24043ca8aa056d846c553af34ce7c8549ecdb614f724591b4e"),
    ("fp8-hopper-x4-k2048", "sha256", "f8c4a2d773133725d0fc21721c10109d475f5d38f851b403fc2d203f3b706bd6",
     "6a1a3963a5d2ef07a45a1b75368ca97eb786a1dc8034232695577d49666c0207"),
    ("fp8-hopper-x4-k8192", "blake3-xob", "4f27ff21174ef25d97f8c0e0888b102fe1a7e97cc990c8f338b73d3a52f1d008",
     "c60bc51fc6f5c6d9d131006eeada3ab1dcd2d222e6f12f2a59ee71407e984193"),
    ("fp8-hopper-x4-k8192", "sha256", "6058692e9ce6820b3fffa4558831076ed59866759de2d865f899eea97a668ec2",
     "8eb1dac93fa528313ba46af8d59c869646a164c1cda0447503c6738aa2199996"),
];

pub fn names() -> Vec<&'static str> {
    SCHEMES.iter().map(|s| s.name).collect()
}

/// The scheme of a registry name or alias; `None` when this build does not know it.
pub fn by_name(name: &str) -> Option<&'static LeafScheme> {
    let canon = ALIASES.iter().find(|(a, _)| *a == name).map_or(name, |(_, c)| *c);
    SCHEMES.iter().copied().find(|s| s.name == canon)
}

/// `leaf.registry.split_relation_name`: `"fp8-ada"` -> `("fp8-ada", POSEIDON2)`, `"fp8-ada+blake3"` -> `("fp8-ada",
/// BLAKE3)`; an unknown or malformed suffix is an error (the statement is refused).
pub fn split_relation(name: &str) -> Result<(&str, &'static LeafScheme), String> {
    let (base, leaf) = match name.split_once('+') {
        None => (name, DEFAULT),
        Some((b, l)) => (b, l),
    };
    if base.is_empty() || leaf.is_empty() || leaf.contains('+') {
        return Err(format!("relation name {name:?}: expected <relation>[+<leaf>]"));
    }
    let scheme = by_name(leaf).ok_or_else(|| format!("relation {name:?} names leaf scheme {leaf:?}; this build knows {}", names().join(", ")))?;
    Ok((base, scheme))
}

/// `leaf.registry.relation_name`: the relation string a v5 statement of `(rel, leaf)` carries (Poseidon2 carries none,
/// so the historical `fp8-ada` statements stay byte-identical).
pub fn relation_name(base: &str, leaf: &LeafScheme) -> String {
    if leaf.name == DEFAULT { base.to_string() } else { format!("{base}+{}", leaf.name) }
}

/// The pinned hashed-system digests of `(rel, leaf)`: the relation's own for Poseidon2, `PINS` otherwise; `None` when
/// nothing is pinned (the gate refuses the statement unless `--allow-any-system`).
pub fn pins(rel: &crate::relation::Relation, leaf: &LeafScheme) -> Option<(&'static str, &'static str)> {
    if leaf.name == DEFAULT {
        return if rel.hashed_sys_id.is_empty() { None } else { Some((rel.hashed_sys_id, rel.hashed_table_digest)) };
    }
    PINS.iter().find(|(r, l, _, _)| *r == rel.name && *l == leaf.name).map(|(_, _, s, t)| (*s, *t))
}

/// The most columns one VU row may span under `leaf`, whatever the system pin says (`--allow-any-system` too): an
/// `ajtai-n<N>` digest `sum_j X^(steps-1-j) B s_j` over `F_p[X]/(X^N + 1)` wraps column `j` onto `j + N` with the
/// opposite sign once `steps > N`, so two different rows share a digest (red-team-leaf-3 H1).  `None`: no bound.
pub fn max_steps(leaf: &LeafScheme) -> Option<usize> {
    ajtai::degree(leaf.name)
}

/// The `(relation, leaf)` whose non-Poseidon2 pins a system file's digests match, if any.
pub fn by_digests(sys_id_hex: &str, table_digest_hex: &str) -> Option<(&'static crate::relation::Relation, &'static LeafScheme)> {
    PINS.iter().find(|(_, _, s, t)| *s == sys_id_hex && *t == table_digest_hex).and_then(|(r, l, _, _)| {
        Some((crate::relation::for_statement(Some(r)).ok()?, by_name(l)?))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_and_name_round_trip() {
        let (b, s) = split_relation("fp8-ada").unwrap();
        assert_eq!((b, s.name), ("fp8-ada", "poseidon2"));
        let (b, s) = split_relation("fp8-ada+hash").unwrap();
        assert_eq!((b, s.name), ("fp8-ada", "poseidon2"));
        let (b, s) = split_relation("fp8-ada+poseidon2").unwrap();
        assert_eq!((b, s.name), ("fp8-ada", "poseidon2"));
        assert_eq!(relation_name("fp8-ada", &POSEIDON2), "fp8-ada");
        assert!(split_relation("fp8-ada+nope").unwrap_err().contains("leaf scheme"));
        assert!(split_relation("+poseidon2").is_err());
        assert!(split_relation("fp8-ada+a+b").is_err());
    }

    #[test]
    fn ajtai_key_derivation_matches_the_python_digests_and_the_params_digest() {
        // leaf/ajtai.py: derive_ring_element(0, 64)[..4] and B_DIGEST_SHA256 / params_sha256 (regression across the two verifiers)
        let b0 = ajtai::ring_element(0, 64);
        assert_eq!(b0.len(), 64);
        assert!(b0.iter().all(|&v| v < P));
        assert_eq!(&ajtai::ring_element(0, 128)[..64], &b0[..]);       // prefix-consistent: one family
        for &(n, want) in ajtai::B_SHA256 {
            let b = ajtai::matrix(n);
            assert_eq!(b.len(), n);
            assert_eq!(ajtai::matrix_sha256(&b), want, "B digest, n = {n}");
            let scheme = by_name(&format!("ajtai-n{n}")).unwrap();
            let d = crate::hash::hex(&crate::hash::sha256(ajtai::params_descriptor(n, want).as_bytes()));
            assert_eq!(d, scheme.params_sha256, "params digest, n = {n}");
            assert_eq!(scheme.digest_elems, n);
        }
        let v = (AJTAI_N64.leaf_bytes)(&[1, 0x0102_0304]);
        assert_eq!(v, vec![1, 0, 0, 0, 4, 3, 2, 1]);
        assert_eq!(split_relation("fp8-ada+ajtai-n64").unwrap().1.name, "ajtai-n64");
        assert_eq!(relation_name("bf16-hopper", &AJTAI_N128), "bf16-hopper+ajtai-n128");
        assert!(split_relation("fp8-ada+ajtai-n96").is_err());
    }

    fn fixture_system(dir: &str) -> crate::format::System {
        crate::format::System::parse(&std::fs::read(format!("fixtures/{dir}/system.bin")).unwrap()).unwrap()
    }

    /// The indices of the linear rows that key a role's bits (terms over its bit rows).
    fn key_rows(sys: &crate::format::System, bits: &[u32]) -> Vec<usize> {
        let set: std::collections::HashSet<u32> = bits.iter().copied().collect();
        (0..sys.linear.len()).filter(|&q| sys.linear[q].e.lin.iter().any(|(r, _)| set.contains(r))
            && sys.linear[q].e.lin.len() > 200).collect()
    }

    #[test]
    fn ajtai_chain_key_follows_the_digest_lanes_in_both_pinned_systems() {
        for (dir, n) in [("fp8-ada-ajtai-n64", 64), ("bf16-hopper-ajtai-n128", 128)] {
            let sys = fixture_system(dir);
            let b = ajtai::matrix(n);
            let roles = chain_key(&sys, n, &b).unwrap_or_else(|e| panic!("{dir}: {e}"));
            for bits in &roles {
                assert_eq!(bits.len(), ajtai::M_COL);
                assert_eq!(key_rows(&sys, bits).len(), n, "{dir}");
            }
            assert!(check_system_key(&sys, by_name(&format!("ajtai-n{n}")).unwrap()).is_ok());
            // the other degree's key over this system: refused on the lanes (2 x 64 vs 2 x 128)
            let m = if n == 64 { 128 } else { 64 };
            assert!(chain_key(&sys, m, &ajtai::matrix(m)).is_err(), "{dir}");
        }
    }

    /// Red-team-leaf-2 G1 in memory: the chain the digests come from keyed by B = 0 (bit terms dropped from the real key
    /// rows) plus 2n decoy rows carrying the derived key over fresh rows -- the old pattern scan accepted exactly this.
    #[test]
    fn ajtai_chain_key_refuses_a_zero_key_chain_with_decoy_key_rows() {
        let n = 64;
        let b = ajtai::matrix(n);
        let honest = fixture_system("fp8-ada-ajtai-n64");
        let roles = chain_key(&honest, n, &b).unwrap();
        let mut sys = honest.clone();
        let fresh0 = sys.m as u32;
        sys.m += 2 * (ajtai::M_COL + 2) * n;
        let zeros: Vec<u32> = (0..ajtai::M_COL as u32).map(|i| fresh0 + i).collect();
        let mut next = fresh0 + ajtai::M_COL as u32;
        for bits in &roles {
            let set: std::collections::HashSet<u32> = bits.iter().copied().collect();
            for q in key_rows(&honest, bits) {
                let orig = sys.linear[q].clone();
                sys.linear[q].e.lin.retain(|(r, _)| !set.contains(r));
                let mut decoy = orig.clone();
                decoy.e.lin = orig.e.lin.iter().map(|&(r, v)| match bits.iter().position(|&x| x == r) {
                    Some(i) => (zeros[i], v),
                    None => {
                        next += 1;
                        (next, v)
                    }
                }).collect();
                decoy.e.lin.sort();
                sys.linear.push(decoy);
            }
        }
        let e = chain_key(&sys, n, &b).unwrap_err();
        assert!(e.contains("the key row is not out - (X acc_in)"), "{e}");
        let e = check_system_key(&sys, &AJTAI_N64).unwrap_err();
        assert!(e.contains("does not use the derived key B"), "{e}");
    }

    #[test]
    fn ajtai_chain_key_refuses_moved_bits_scaled_keys_and_rerouted_ties() {
        let n = 64;
        let b = ajtai::matrix(n);
        let honest = fixture_system("fp8-ada-ajtai-n64");
        let roles = chain_key(&honest, n, &b).unwrap();
        let rows0 = key_rows(&honest, &roles[0]);
        // one key row reads bit position 5 off another row (the W role's bit 5)
        let mut sys = honest.clone();
        let q = rows0[3];
        for t in sys.linear[q].e.lin.iter_mut() {
            if t.0 == roles[0][5] {
                t.0 = roles[1][5];
            }
        }
        sys.linear[q].e.lin.sort();
        let e = chain_key(&sys, n, &b).unwrap_err();
        assert!(e.contains("bit position 5 is row"), "{e}");
        // every key row of the x role reads bit 5 off the W role's bit 5: consistent, but shared across the roles
        let mut sys = honest.clone();
        for &q in &rows0 {
            for t in sys.linear[q].e.lin.iter_mut() {
                if t.0 == roles[0][5] {
                    t.0 = roles[1][5];
                }
            }
            sys.linear[q].e.lin.sort();
        }
        assert!(chain_key(&sys, n, &b).is_err());
        // the key doubled on one row
        let mut sys = honest.clone();
        let set: std::collections::HashSet<u32> = roles[0].iter().copied().collect();
        for t in sys.linear[rows0[0]].e.lin.iter_mut() {
            if set.contains(&t.0) {
                t.1 = ((2 * t.1 as u64) % P as u64) as u32;
            }
        }
        let e = chain_key(&sys, n, &b).unwrap_err();
        assert!(e.contains("not an entry of the derived key row"), "{e}");
        // lane 0's end tie rerouted to a fresh row (the digest no longer comes from the keyed chain)
        let mut sys = honest.clone();
        let fresh = sys.m as u32;
        sys.m += 1;
        let d0 = sys.pins.iter().find(|(nm, _)| nm == "hash.d[0]").unwrap().1;
        let qi = sys.quadratic.iter().position(|q| q[1].lin.iter().any(|&(r, _)| r == d0)).unwrap();
        sys.quadratic[qi][1].lin = vec![(fresh.min(d0), if fresh < d0 { 1 } else { P - 1 }), (fresh.max(d0), if fresh < d0 { P - 1 } else { 1 })];
        let e = chain_key(&sys, n, &b).unwrap_err();
        assert!(e.contains("never linked"), "{e}");
        // a second tie on lane 0
        let mut sys = honest.clone();
        let dup = sys.quadratic[qi].clone();
        sys.quadratic.push(dup);
        assert!(chain_key(&sys, n, &b).unwrap_err().contains("hash.d[0]: not tied by exactly one"));
    }

    #[test]
    fn poseidon2_leaf_bytes_are_big_endian_lanes() {
        let v = (POSEIDON2.leaf_bytes)(&[1, 2, 3, 4, 5, 6, 7, 0x0102_0304]);
        assert_eq!(v.len(), 32);
        assert_eq!(&v[..4], &[0, 0, 0, 1]);
        assert_eq!(&v[28..], &[1, 2, 3, 4]);
    }

    #[test]
    fn blake3_leaf_bytes_match_the_python_scheme() {
        // leaf/blake3.py: Blake3Leaf().native(rows, word_bits=8, role=) of a 1536-byte row (rng seed 1) for both roles,
        // then leaf_bytes == the `blake3` package's keyed digest of the row bytes under the role's key
        let x: [u32; 49] = [6, 12484, 29774, 53750, 51075, 3602, 4076, 31413, 33829, 7246, 42179, 39884, 50426, 27583, 3456,
            1716, 49874, 18585, 43980, 14096, 36809, 18054, 53302, 5941, 5726, 58036, 37223, 9265, 1409, 39823, 19488, 16867,
            60764, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let w: [u32; 49] = [10, 17377, 12951, 31928, 29282, 34781, 23233, 39917, 6581, 31152, 11858, 37240, 57198, 20263,
            51057, 44814, 8727, 40228, 10503, 35577, 34859, 14385, 31516, 1712, 60419, 59968, 5091, 43407, 44310, 18091, 619,
            51485, 1671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let hex = |v: Vec<u8>| -> String { v.iter().map(|b| format!("{b:02x}")).collect() };
        assert_eq!(hex((BLAKE3.leaf_bytes)(&x)), "fde8e56e3e91fb7a44c37124392abe618bd56f83a3eb49f0bca88353b7429fbd");
        assert_eq!(hex((BLAKE3.leaf_bytes)(&w)), "e17631f2bf9ba60e615f9b0d9ce20213763fc181a6b663bb18fb14c1b2ba8cdc");
        let (b, s) = split_relation("fp8-ada-x4+blake3").unwrap();
        assert_eq!((b, s.name), ("fp8-ada-x4", "blake3"));
        // wrong length / non-field element: no leaf
        assert!((BLAKE3.leaf_bytes)(&x[..48]).is_empty());
        let mut bad = x;
        bad[3] = BABYBEAR_P;
        assert!((BLAKE3.leaf_bytes)(&bad).is_empty());
        // malformed frames: the tagged SHA-256 domain (Python `leaf_bytes([6] + [1] * 48)`), never a root
        let mut m = [1u32; 49];
        m[0] = 6;
        assert_eq!(hex((BLAKE3.leaf_bytes)(&m)), "7c1d4e79525b029c7a23cc1e86d8d6bd5fb1dbf72bee0aba300c921ffdaba271");
        for (i, v) in [(0usize, 2u32), (0, 15), (40, 1), (1, 1 << 16)] {
            let mut bad = x;
            bad[i] = v;
            let out = (BLAKE3.leaf_bytes)(&bad);
            assert_eq!(out.len(), 32);
            assert_ne!(hex(out.clone()), hex((BLAKE3.leaf_bytes)(&x)));
            // the same row through the tag directly
            let mut h = crate::hash::Sha256::new();
            h.update(BLAKE3_MALFORMED_TAG);
            for &e in &bad {
                h.update(&e.to_le_bytes());
            }
            assert_eq!(out, h.finalize().to_vec());
        }
    }

    #[test]
    fn sized_blake3_frames_fold_to_the_keyed_digest_of_long_rows() {
        // row byte i = (7 i + 3) mod 256; expected = the `blake3` package's keyed digest under the role's key (Python:
        // blake3.blake3(row, key=leaf.blake3.key_for(role)).hexdigest(), equal to Blake3Leaf(ncm).leaf_bytes(native(row)))
        let hex = |v: Vec<u8>| -> String { v.iter().map(|b| format!("{b:02x}")).collect() };
        for (nbytes, role, want) in [
            (4096usize, 1u32, "fa70cbb20dfe2b0faab40ec6eb6ed8fbf88a2cd93b50b0824bc39f221166ce79"),
            (8192, 2, "98645d6912503f1a77f893d2566313957bbee8c9d07881777956a22d9107e0ac"),
            (16384, 1, "1df95024a07b4031f66396d00634386ae747cd2cc1fc294bae2c2ee6fbe7b3be"),
        ] {
            let row: Vec<u8> = (0..nbytes).map(|i| ((7 * i + 3) % 256) as u8).collect();
            let nch = nbytes / 1024;
            let mut elems = vec![0u32; 1 + 16 * nch];
            elems[0] = nch as u32 + 64 * role;
            for c in 0..nch {
                let mut cv = blake3_key_words(role);
                for j in 0..16 {
                    let mut block = [0u32; 16];
                    for (w, slot) in block.iter_mut().enumerate() {
                        let o = 1024 * c + 64 * j + 4 * w;
                        *slot = u32::from_le_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]]);
                    }
                    let flags = BLAKE3_KEYED_HASH | if j == 0 { 1 } else { 0 } | if j == 15 { 2 } else { 0 };
                    cv = blake3_compress_cv(&cv, &block, c as u64, 64, flags);
                }
                for w in 0..8 {
                    elems[1 + 16 * c + 2 * w] = cv[w] & 0xFFFF;
                    elems[2 + 16 * c + 2 * w] = cv[w] >> 16;
                }
            }
            let s = sized(&BLAKE3_XOB, nbytes).unwrap();
            assert_eq!((s.name, s.digest_elems), ("blake3-xob", elems.len()));
            assert_eq!(hex((s.leaf_bytes)(&elems)), want, "{nbytes} bytes");
            assert_eq!(sized(&BLAKE3, nbytes).unwrap().params_sha256, s.params_sha256);
            // the 3-slot header convention inside a sized frame is malformed (the tagged domain, never the root)
            let mut m = elems.clone();
            m[0] = nch as u32 + 4 * role;
            assert_ne!(hex((s.leaf_bytes)(&m)), want);
        }
        assert!(std::ptr::eq(sized(&BLAKE3, 3072).unwrap(), &BLAKE3) && std::ptr::eq(sized(&SHA256, 16384).unwrap(), &SHA256));
        assert!(sized(&BLAKE3, 5 * 1024).is_err());
    }

    #[test]
    fn sha256_leaf_bytes_match_the_python_scheme() {
        // leaf/sha256.py: Sha256Leaf().native(rows, word_bits=, role=) of 1536-word rows (rng seeds 1 / 2), then leaf_bytes
        // == verity.ml.commitments.rowleaf.sha256_row_digest (hashlib over the core prefix || row bytes)
        let x: [u32; 17] = [393249, 60336, 42228, 53307, 8065, 20306, 48806, 19787, 63601, 59087, 60567, 13660, 6583, 26673,
            9928, 34867, 32873];
        let w: [u32; 17] = [393250, 1457, 21848, 63781, 48250, 47699, 9984, 32060, 40006, 63166, 30315, 25577, 45471, 16651,
            21663, 5717, 38760];
        let x16: [u32; 17] = [393281, 13221, 13212, 62113, 17770, 42013, 46476, 29751, 43698, 64962, 43923, 57936, 4845, 46076,
            47400, 5138, 40235];
        let hex = |v: Vec<u8>| -> String { v.iter().map(|b| format!("{b:02x}")).collect() };
        assert_eq!(hex((SHA256.leaf_bytes)(&x)), "037adf01086dadbeb3ae2cae92d1e04c60ba9a07dd9289071aec73e1804e0b7d");
        assert_eq!(hex((SHA256.leaf_bytes)(&w)), "a7fadc52b1b8a5bf95674b2285823f9c84cccb643b6cd3976b2cddab81de7266");
        assert_eq!(hex((SHA256.leaf_bytes)(&x16)), "adac69c7694e2c0fcdd770176cdf317d23d318b66b27cf9ec89d8eb74058011b");
        let (b, s) = split_relation("fp8-ada-x4+sha256").unwrap();
        assert_eq!((b, s.name), ("fp8-ada-x4", "sha256"));
        assert!((SHA256.leaf_bytes)(&x[..16]).is_empty());
        let mut bad = x;
        bad[3] = BABYBEAR_P;
        assert!((SHA256.leaf_bytes)(&bad).is_empty());
        // malformed frames: the tagged SHA-256 domain (Python `leaf_bytes([header(1, 8)] + [1] * 16)` is well formed;
        // `[5] + [1] * 16` is not)
        let mut m = [1u32; 17];
        m[0] = 393249;
        assert_eq!(hex((SHA256.leaf_bytes)(&m)), "acff57533da448036b527c5133b3fa21f5e1be73aee6fc4e7837ffb7ed45ab43");
        m[0] = 5;
        assert_eq!(hex((SHA256.leaf_bytes)(&m)), "fa6a31bd116bfdb5902759cc25593702901bfc375e2bf30871b48add29cacfef");
        for (i, v) in [(0usize, 393251u32), (0, 1 + 4 * 12 + 256 * 1536), (0, 1 + 4 * 8 + 256 * 1500), (5, 1 << 16)] {
            let mut bad = x;
            bad[i] = v;
            let out = (SHA256.leaf_bytes)(&bad);
            let mut h = crate::hash::Sha256::new();
            h.update(SHA256_MALFORMED_TAG);
            for &e in &bad {
                h.update(&e.to_le_bytes());
            }
            assert_eq!(out, h.finalize().to_vec());
        }
    }

    #[test]
    fn every_scheme_has_a_32_byte_params_digest_and_a_unique_name() {
        let mut seen = std::collections::HashSet::new();
        for s in SCHEMES {
            assert_eq!(s.params_sha256.len(), 64, "{}", s.name);
            assert!(s.params_sha256.chars().all(|c| c.is_ascii_hexdigit()), "{}", s.name);
            assert!(s.digest_elems >= 1 && s.digest_elems <= 128, "{}", s.name);
            assert!(!s.schema.is_empty() && !s.name.contains('+'), "{}", s.name);
            assert!(seen.insert(s.name), "duplicate leaf scheme {}", s.name);
        }
    }
}
