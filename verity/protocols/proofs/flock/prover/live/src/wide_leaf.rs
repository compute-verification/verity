//! The wide level-0 leaf: a statement version of its own beside fast100's (the Lean `Flock.fast100Wide`). A table of
//! 2^32 to 2^35 bits commits Ligerito's level 0 as `2^(m − 25)` witness lanes where fast100 commits 2^6, so level 0's
//! codeword has 2^19 symbols at every size and the levels after it are m = 31's; a smaller table keeps fast100's
//! parameters. The version's Ligerito configs are upstream's, added by `flock-wide-b684b12.patch` and selected by
//! `log_batch_size`.
//!
//! A statement names one version: the wide one has its own `profile` in the identity and the Hello and its own domain
//! tag, so no session of one version reads as the other's. The version has two statements, the circuit statement and the
//! templated one, each with a domain tag and a name the Lean verifier holds as literals ([`DOMAINS`], [`LEAN_NAMES`]): no
//! rule derives either, and a statement without them, the typed statement, has no wide form and is refused.
use flock_core::pcs::ligerito::{LigeritoProfile, embedded_initial_k_or_default, wide_initial_k};

/// fast100's `profile`, in a statement's identity and a session's Hello.
pub const FAST100_PROFILE: &str = "fast100";
/// The wide version's `profile` there.
pub const PROFILE: &str = "fast100-wide-leaf";
/// The wide version's statements: fast100's domain tag (`circuit::Tags::domain`) of each, and the wide version's own. A
/// rep's index follows the tag; in a session of several tables the table's name and a `/` go before the tag's final `rep`.
pub const DOMAINS: [(&str, &str); 2] = [
    ("flock-circuit/fast100x2/rep", "flock-circuit/fast100-wide-leafx2/rep"),
    ("flock-circuit-tmpl/fast100x2/rep", "flock-circuit-tmpl/fast100-wide-leafx2/rep"),
];
/// The wide version's statements by the Lean verifier's names (`Tags.wideLeafName`, `Tags.tmplWideLeafName`): each
/// statement's name (`circuit::STATEMENT`, `circuit::TMPL_STATEMENT`) and its wide version's.
pub const LEAN_NAMES: [(&str, &str); 2] = [
    ("verity/flock-circuit", "verity/flock-circuit+wide-leaf"),
    ("verity/flock-circuit/tmpl", "verity/flock-circuit/tmpl+wide-leaf"),
];
/// `PcsParams.profile` of the wide version's proofs: fast100's own, the wide config being the one at its `log_batch_size`.
pub const LIGERITO: LigeritoProfile = LigeritoProfile::Fast100;

/// What the wide version's `--zk` identity says of level 0's query count after fast100's own text
/// (`hooks.ligerito_level0_queries`).
pub const LEVEL0_QUERIES: &str = "; wide leaf: level 0 of 2^19 symbols at m = 32..35 (k_0 = m - 25, L = 2^18), where q_0 = 219";

/// `profile` of a statement's identity and Hello in a version.
pub fn profile(wide: bool) -> &'static str {
    if wide { PROFILE } else { FAST100_PROFILE }
}

/// `hooks.ligerito_level0_queries` of a `--zk` statement's identity in a version: fast100's `text`, as it is or with
/// [`LEVEL0_QUERIES`] after it.
pub fn level0_queries(text: &str, wide: bool) -> String {
    if wide { format!("{text}{LEVEL0_QUERIES}") } else { text.to_string() }
}

/// A statement's domain tag in a version: `tag`, fast100's, as it is, or the wide version's tag of the same statement
/// ([`DOMAINS`]). A statement the wide version does not have is an error that names the two it has.
pub fn domain(tag: &'static str, wide: bool) -> Result<&'static str, String> {
    if !wide {
        return Ok(tag);
    }
    DOMAINS.iter().find(|(own, _)| *own == tag).map(|(_, wide)| *wide).ok_or_else(|| {
        format!("--wide-leaf: the wide level-0 leaf has no statement whose domain tag is fast100's `{tag}`. It has two, `{}` (the \
                 circuit statement) and `{}` (the templated one): prove this statement without --wide-leaf",
                DOMAINS[0].1, DOMAINS[1].1)
    })
}

/// The Lean verifier's name of the statement a ZK server session's Lean loop is set up under, in a version: `statement` is
/// the circuit's, `given` is `FC_FIREWALL_STATEMENT`, and `seed_injection` whether the build's identity pins it. fast100's
/// is `given`, else `statement` with `+seed-injection` in such a build. The wide version's is `statement`'s wide name
/// ([`LEAN_NAMES`]), and `given` may only repeat it: a loop set up under any other name is at another version and refuses
/// every session, so that is refused here, with the name to set. The Lean verifier has no wide version of a
/// seed-injection build's statement, so the wide version refuses one.
pub fn lean_statement(statement: &str, given: Option<&str>, seed_injection: bool, wide: bool) -> Result<String, String> {
    if !wide {
        return Ok(given.map_or_else(|| format!("{statement}{}", if seed_injection { "+seed-injection" } else { "" }), str::to_string));
    }
    if seed_injection {
        return Err(format!("--wide-leaf behind the ZK server (FC_FIREWALL=1): the Lean verifier has no wide-leaf version of a \
                            seed-injection build's statement (`{statement}+seed-injection` is fast100's alone), so its session \
                            loop cannot be set up: run the ZK server with the proving build, or this build with FC_FIREWALL=0"));
    }
    let want = LEAN_NAMES.iter().find(|(own, _)| *own == statement).map(|(_, wide)| *wide).ok_or_else(|| {
        format!("--wide-leaf behind the ZK server (FC_FIREWALL=1): the Lean verifier has no wide-leaf version of `{statement}`. \
                 It has two, `{}` and `{}`: prove this statement without --wide-leaf", LEAN_NAMES[0].1, LEAN_NAMES[1].1)
    })?;
    match given {
        Some(g) if g != want => Err(format!(
            "--wide-leaf behind the ZK server (FC_FIREWALL=1): FC_FIREWALL_STATEMENT is `{g}`, not this statement's wide-leaf \
             name, so the Lean session loop would be set up at another version and refuse the session: set \
             FC_FIREWALL_STATEMENT={want}, or unset it")),
        _ => Ok(want.to_string()),
    }
}

/// Level 0's `log_batch_size` for a table of `2^m` bits: the wide version's `m − 25` where it has one (32 ≤ m ≤ 35, under
/// [`LIGERITO`]), else the profile's own.
pub fn log_batch_size(m: usize, lig: LigeritoProfile, wide: bool) -> usize {
    match wide_initial_k(m) {
        Some(k) if wide && lig == LIGERITO => k,
        _ => embedded_initial_k_or_default(m, lig),
    }
}

#[cfg(test)]
mod tests {
    use flock_core::{merkle::HashKind, pcs::PcsParams};

    use super::*;
    use crate::circuit::{TAGS, TMPL_TAGS, TYPED_TAGS, rep_domain};

    fn params(m: usize, lig: LigeritoProfile, wide: bool) -> PcsParams {
        PcsParams { m, log_inv_rate: 1, log_batch_size: log_batch_size(m, lig, wide), profile: lig, num_lanes: None, merkle_hash: HashKind::Hm96Sha512 }
    }

    /// Up to m = 31 the wide version's parameters are fast100's: the bytes a proof carries of them, and the whole Ligerito
    /// config either side builds from them.
    #[test]
    fn tables_up_to_m31_keep_fast100s_parameters() {
        for m in 22..=31 {
            let (own, wide) = (params(m, LigeritoProfile::Fast100, false), params(m, LigeritoProfile::Fast100, true));
            assert_eq!(own.log_batch_size, embedded_initial_k_or_default(m, LigeritoProfile::Fast100));
            assert_eq!(bincode::serialize(&own).unwrap(), bincode::serialize(&wide).unwrap(), "m = {m}");
            assert_eq!(format!("{:?}", own.ligerito_prover_config().unwrap()), format!("{:?}", wide.ligerito_prover_config().unwrap()), "m = {m}");
            assert_eq!(format!("{:?}", own.ligerito_verifier_config().unwrap()), format!("{:?}", wide.ligerito_verifier_config().unwrap()), "m = {m}");
        }
    }

    /// fast100's version is untouched at every size: its batch is upstream's own, its names the ones its statements hash.
    #[test]
    fn fast100s_version_is_as_it_was() {
        for m in 22..=35 {
            for lig in [LigeritoProfile::Fast100, LigeritoProfile::Fast] {
                assert_eq!(log_batch_size(m, lig, false), embedded_initial_k_or_default(m, lig), "m = {m}");
            }
            assert_eq!(log_batch_size(m, LigeritoProfile::Fast, true), embedded_initial_k_or_default(m, LigeritoProfile::Fast), "m = {m}");
        }
        assert_eq!(profile(false), "fast100");
        for tags in [&TAGS, &TYPED_TAGS, &TMPL_TAGS] {
            assert_eq!(domain(tags.domain, false), Ok(tags.domain));
        }
    }

    /// From m = 32 to 35 the wide version is the Lean's schedule: `k_0 = m − 25` lanes at level 0, 2^19 leaves of
    /// `2^(m − 25)` witness symbols (four more under `--zk`: 16,448 bytes at m = 35), then m = 31's levels; `--zk`'s
    /// level 0 takes 219 queries and 438 pads a lane.
    #[test]
    fn wide_tables_take_the_wide_schedule() {
        let m31 = params(31, LigeritoProfile::Fast100, false).ligerito_prover_config().unwrap();
        for m in 32..=35usize {
            let pcs = params(m, LIGERITO, true);
            let cfg = pcs.ligerito_prover_config().unwrap();
            assert_eq!((pcs.log_batch_size, cfg.initial_k, cfg.initial_log_msg_cols), (m - 25, m - 25, 18), "m = {m}");
            assert_eq!((pcs.n_leaves(), pcs.leaf_size_bytes()), (1 << 19, 16 << (m - 25)), "m = {m}");
            assert_eq!((&cfg.log_inv_rates, &cfg.queries, &cfg.recursive_ks, &cfg.recursive_log_msg_cols, &cfg.ood_samples),
                       (&m31.log_inv_rates, &m31.queries, &m31.recursive_ks, &m31.recursive_log_msg_cols, &m31.ood_samples), "m = {m}");
            assert_eq!((&cfg.grinding_bits, &cfg.fold_grinding_bits, &cfg.claim_batch_grinding_bits, &cfg.consistency_batch_grinding_bits),
                       (&m31.grinding_bits, &m31.fold_grinding_bits, &m31.claim_batch_grinding_bits, &m31.consistency_batch_grinding_bits), "m = {m}");
            assert_eq!(cfg.queries, vec![218, 106, 71, 53, 43, 36]);
            let q0 = crate::zk_veil::level0_queries(1 << cfg.initial_log_msg_cols, cfg.queries[0]).unwrap();
            assert_eq!((q0, crate::zk_veil::level0_t_pad(q0)), (219, 438), "m = {m}");
            assert_eq!(pcs.ligerito_verifier_config().unwrap().initial_k, m - 25);
            assert_eq!(params(m, LigeritoProfile::Fast100, false).log_batch_size, 6);
        }
        assert_eq!(((1usize << 10) + flock_core::pcs::ligerito::zk::EXTRA_LANES) * 16, 16448);
    }

    /// The wide version's names are the Lean verifier's literals, byte for byte: the profile, the circuit statement's
    /// rep domain with a rep's index and with a table's name before it (as a session builds them, `rep_domain`), the
    /// templated statement's, and `--zk`'s level-0 text. `PcsParams.profile` stays fast100's index.
    #[test]
    fn the_wide_versions_names_are_the_verifiers_literals() {
        assert_eq!((profile(true), profile(false)), ("fast100-wide-leaf", "fast100"));
        let (flat, tmpl) = (domain(TAGS.domain, true).unwrap(), domain(TMPL_TAGS.domain, true).unwrap());
        assert_eq!((flat, tmpl), ("flock-circuit/fast100-wide-leafx2/rep", "flock-circuit-tmpl/fast100-wide-leafx2/rep"));
        assert_eq!(rep_domain(flat, 1, "", 1), b"flock-circuit/fast100-wide-leafx2/rep1");
        assert_eq!(rep_domain(tmpl, 1, "", 0), b"flock-circuit-tmpl/fast100-wide-leafx2/rep0");
        assert_eq!(rep_domain(flat, 2, "circuit-00", 1), b"flock-circuit/fast100-wide-leafx2/circuit-00/rep1");
        assert_eq!(rep_domain(TAGS.domain, 2, "circuit-00", 0), b"flock-circuit/fast100x2/circuit-00/rep0");
        assert_eq!(DOMAINS.map(|(own, _)| own), [TAGS.domain, TMPL_TAGS.domain]);
        assert_eq!(bincode::serialize(&LIGERITO).unwrap(), bincode::serialize(&1u32).unwrap());
        assert_eq!(level0_queries("q_0", false), "q_0");
        assert_eq!(level0_queries("q_0", true), "q_0; wide leaf: level 0 of 2^19 symbols at m = 32..35 (k_0 = m - 25, L = 2^18), where q_0 = 219");
    }

    /// The typed statement has no wide form: the version refuses it, naming the two statements it has, and derives no tag.
    #[test]
    fn the_typed_statement_has_no_wide_form() {
        let why = domain(TYPED_TAGS.domain, true).unwrap_err();
        assert!(why.starts_with("--wide-leaf: the wide level-0 leaf has no statement whose domain tag is fast100's `flock-circuit-types/fast100x2/rep`"), "{why}");
        assert!(why.contains("`flock-circuit/fast100-wide-leafx2/rep`") && why.contains("`flock-circuit-tmpl/fast100-wide-leafx2/rep`"), "{why}");
        assert!(!why.contains("flock-circuit-types/fast100-wide-leaf"), "{why}");
        assert_eq!(domain("flock-circuit/stacked/fast100/rep0", true).ok(), None);
    }

    /// The ZK server's Lean loop is set up under the session's version: fast100's name as it was, `FC_FIREWALL_STATEMENT`
    /// taken as given; under the wide version the statement's wide name, the Lean verifier's literal, with `+wide-leaf`
    /// once, whether `FC_FIREWALL_STATEMENT` is unset or repeats it.
    #[test]
    fn the_lean_loop_is_set_up_under_the_sessions_version() {
        use crate::{circuit::{STATEMENT, TMPL_STATEMENT}, typed};
        assert_eq!(lean_statement(STATEMENT, None, false, false).as_deref(), Ok("verity/flock-circuit"));
        assert_eq!(lean_statement(STATEMENT, None, true, false).as_deref(), Ok("verity/flock-circuit+seed-injection"));
        assert_eq!(lean_statement(TMPL_STATEMENT, Some("anything"), true, false).as_deref(), Ok("anything"));
        assert_eq!(LEAN_NAMES.map(|(own, _)| own), [STATEMENT, TMPL_STATEMENT]);
        assert_eq!(LEAN_NAMES.map(|(_, wide)| wide), ["verity/flock-circuit+wide-leaf", "verity/flock-circuit/tmpl+wide-leaf"]);
        for (own, wide) in LEAN_NAMES {
            assert_eq!(wide, format!("{own}+wide-leaf"));
            assert_eq!(lean_statement(own, None, false, true).as_deref(), Ok(wide));
            assert_eq!(lean_statement(own, Some(wide), false, true).as_deref(), Ok(wide));
        }
        assert!(lean_statement(typed::STATEMENT, None, false, true).unwrap_err().contains("no wide-leaf version of `verity/flock-circuit/types`"));
    }

    /// Under the wide version a loop that would be set up at another version is refused before it starts, with the fix:
    /// `FC_FIREWALL_STATEMENT` naming fast100's statement or the other wide statement, and a seed-injection build.
    #[test]
    fn the_lean_loop_at_another_version_is_refused() {
        use crate::circuit::{STATEMENT, TMPL_STATEMENT};
        for (statement, given) in [(STATEMENT, STATEMENT), (STATEMENT, "verity/flock-circuit+seed-injection"),
                                   (STATEMENT, "verity/flock-circuit/tmpl+wide-leaf"), (TMPL_STATEMENT, TMPL_STATEMENT)] {
            let why = lean_statement(statement, Some(given), false, true).unwrap_err();
            let fix = format!("set FC_FIREWALL_STATEMENT={statement}+wide-leaf, or unset it");
            assert!(why.contains(&format!("FC_FIREWALL_STATEMENT is `{given}`")) && why.ends_with(&fix), "{why}");
        }
        for given in [None, Some("verity/flock-circuit+wide-leaf")] {
            let why = lean_statement(STATEMENT, given, true, true).unwrap_err();
            assert!(why.contains("no wide-leaf version of a seed-injection build's statement") && why.ends_with("this build with FC_FIREWALL=0"), "{why}");
        }
    }
}
