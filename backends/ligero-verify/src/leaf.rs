//! The leaf schemes a hashed statement (`authentication = included-hash`, `ligero-statement/v5`) may name: the Rust
//! side of `backends/direct/ligero/leaf/` (`LeafScheme`).  A v5 statement's relation string is `<rel>` (the
//! Poseidon2 scheme, the historical `+hash`) or `<rel>+<leaf>`; the scheme fixes how many field elements one operand
//! row's digest has, the leaf schema string and parameters digest the authentication block must carry (both absorbed
//! into the statement digest through `hash_digest_bytes`), and how a digest row is framed as leaf bytes under the
//! SHA-256 range-domain (`leaf_bytes`).  The in-circuit gadget is the relation's business: it is pinned by the hashed
//! system's `(sys_id, table_digest)` pair (`relation.rs` for Poseidon2, `PINS` here for every other scheme).
//!
//! Adding a scheme (another lane): one `LeafScheme` static, one entry in `SCHEMES`, and its `(relation, leaf,
//! sys_id, table_digest)` pins in `PINS` once the Python side has gated it.

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
/// (2..=3) + 4 role (1, 2)`, a limb >= 2^16, a non-zero unused slot) yields the tagged SHA-256 of the elements.
fn blake3_leaf_bytes(elems: &[u32]) -> Vec<u8> {
    if elems.len() != BLAKE3_DIGEST_ELEMS || elems.iter().any(|&x| x >= BABYBEAR_P) {
        return Vec::new();
    }
    let (nch, role) = ((elems[0] & 3) as usize, elems[0] >> 2);
    if !(2..=BLAKE3_MAX_CHUNKS).contains(&nch) || !(1..=2).contains(&role) || elems[1 + 16 * nch..].iter().any(|&x| x != 0)
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

/// Every scheme this build knows.
pub static SCHEMES: &[&LeafScheme] = &[&POSEIDON2, &BLAKE3];

/// `leaf.registry.DEFAULT` / `ALIASES`: a relation string without a suffix, or with `+hash`, names Poseidon2.
pub const DEFAULT: &str = "poseidon2";
const ALIASES: &[(&str, &str)] = &[("hash", "poseidon2")];

/// The `(sys_id, table_digest)` pins of hashed systems composed with a scheme OTHER than Poseidon2 (whose pins live on
/// the `Relation` itself: `hashed_sys_id` / `hashed_table_digest`): `(relation name, leaf name, sys_id, table_digest)`.
pub static PINS: &[(&str, &str, &str, &str)] = &[
    // lane blake3-leaf: `hashchain.compose(fp8-ada-x4, "blake3")` (2 compressions per role per column), gated 2048 VUs +
    // 86 negatives on vy-blake3-leaf 2026-09-23 18:00Z
    ("fp8-ada-x4", "blake3", "ac88040054781359fbc51bd9355495fc67e5efd072b341d04c9c8017a46931f1",
     "120e4a98c27ffc9488b4a6c8e0fcdf510605205beb7fca98d9bc62bedcc8edf6"),
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
    fn every_scheme_has_a_32_byte_params_digest_and_a_unique_name() {
        let mut seen = std::collections::HashSet::new();
        for s in SCHEMES {
            assert_eq!(s.params_sha256.len(), 64, "{}", s.name);
            assert!(s.params_sha256.chars().all(|c| c.is_ascii_hexdigit()), "{}", s.name);
            assert!(s.digest_elems >= 1 && s.digest_elems <= 64, "{}", s.name);
            assert!(!s.schema.is_empty() && !s.name.contains('+'), "{}", s.name);
            assert!(seen.insert(s.name), "duplicate leaf scheme {}", s.name);
        }
    }
}
