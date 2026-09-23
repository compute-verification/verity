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

/// Every scheme this build knows.
pub static SCHEMES: &[&LeafScheme] = &[&POSEIDON2, &AJTAI_N64, &AJTAI_N128];

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
/// `out[x] - (X acc_in)[x] - sum_i B[x][o + i] bit_i = 0` whose terms (rows strictly increasing: the 256 bit rows, then
/// the carried row, then the out row) are `[(bit_i, p - B[x][o + i]) for B[x][o + i] != 0] ++ [(acc_in, 1 or p - 1),
/// (out, 1)]`, constant 0; this finds, for every x, exactly one such row per key block.  A system built with another key
/// (`B = 0`: trivial digests) is refused even under `--allow-any-system`.  Other schemes: nothing to check (their gadgets
/// have no key; the system pin covers them).
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
    let mut want: std::collections::HashMap<Vec<u32>, (usize, usize)> = std::collections::HashMap::with_capacity(ajtai::KEYS * n);
    for (x, row) in b.iter().enumerate() {
        for k in 0..ajtai::KEYS {
            let blk = &row[k * ajtai::M_COL..(k + 1) * ajtai::M_COL];
            let coefs: Vec<u32> = blk.iter().filter(|&&v| v != 0).map(|&v| P - v).collect();
            if want.insert(coefs, (x, k)).is_some() {
                return Err(format!("{}: two rows of the derived B coincide", leaf.display));
            }
        }
    }
    let mut seen = vec![[0usize; ajtai::KEYS]; n];
    for c in &sys.linear {
        if c.pub_pin.is_some() || c.e.c != 0 || c.e.lin.len() < 3 {
            continue;
        }
        let t = &c.e.lin;
        let (acc, out) = (t[t.len() - 2].1, t[t.len() - 1].1);
        if out != 1 || (acc != 1 && acc != P - 1) {
            continue;
        }
        let coefs: Vec<u32> = t[..t.len() - 2].iter().map(|&(_, v)| v).collect();
        if let Some(&(x, k)) = want.get(&coefs) {
            seen[x][k] += 1;
        }
    }
    if seen.iter().any(|k| *k != [1; ajtai::KEYS]) {
        let missing = seen.iter().filter(|k| **k != [1; ajtai::KEYS]).count();
        return Err(format!(
            "system file: {} of the {n} Ajtai state rows do not carry the derived key B (SHAKE-256 of {:?}) for both operand roles",
            missing, String::from_utf8_lossy(ajtai::B_TAG)
        ));
    }
    Ok(())
}

/// `leaf.registry.DEFAULT` / `ALIASES`: a relation string without a suffix, or with `+hash`, names Poseidon2.
pub const DEFAULT: &str = "poseidon2";
const ALIASES: &[(&str, &str)] = &[("hash", "poseidon2")];

/// The `(sys_id, table_digest)` pins of hashed systems composed with a scheme OTHER than Poseidon2 (whose pins live on
/// the `Relation` itself: `hashed_sys_id` / `hashed_table_digest`): `(relation name, leaf name, sys_id, table_digest)`.
pub static PINS: &[(&str, &str, &str, &str)] = &[
    ("fp8-ada", "ajtai-n64", "18d915323c056495f667a6c4286eece77e25e348aed03b335b583e2db1e346c4", "a1f2cb20cb549af68f005bf3efef6112f08bb7a9537dbd876811e46769c5ee50"),
    ("bf16-hopper", "ajtai-n128", "29f18689c05de6f355c97aac1e711f9dad7f7f35eda85026482258c85d6a27e5", "e4dceccece10d7f55f5aed8cb7e444d6f0a736012b6323ee6fd2cdfe520f6438"),
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

    #[test]
    fn poseidon2_leaf_bytes_are_big_endian_lanes() {
        let v = (POSEIDON2.leaf_bytes)(&[1, 2, 3, 4, 5, 6, 7, 0x0102_0304]);
        assert_eq!(v.len(), 32);
        assert_eq!(&v[..4], &[0, 0, 0, 1]);
        assert_eq!(&v[28..], &[1, 2, 3, 4]);
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
