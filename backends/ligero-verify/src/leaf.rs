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

/// Every scheme this build knows.
pub static SCHEMES: &[&LeafScheme] = &[&POSEIDON2];

/// `leaf.registry.DEFAULT` / `ALIASES`: a relation string without a suffix, or with `+hash`, names Poseidon2.
pub const DEFAULT: &str = "poseidon2";
const ALIASES: &[(&str, &str)] = &[("hash", "poseidon2")];

/// The `(sys_id, table_digest)` pins of hashed systems composed with a scheme OTHER than Poseidon2 (whose pins live on
/// the `Relation` itself: `hashed_sys_id` / `hashed_table_digest`): `(relation name, leaf name, sys_id, table_digest)`.
pub static PINS: &[(&str, &str, &str, &str)] = &[];

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
