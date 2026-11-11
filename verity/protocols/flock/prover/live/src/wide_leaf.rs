//! The wide level-0 leaf: a statement version of its own beside fast100's (the Lean `Flock.fast100Wide`). A table of
//! 2^32 to 2^35 bits commits Ligerito's level 0 as `2^(m − 25)` witness lanes where fast100 commits 2^6, so level 0's
//! codeword has 2^19 symbols at every size and the levels after it are m = 31's; a smaller table keeps fast100's
//! parameters. The version's Ligerito configs are upstream's, added by `flock-wide-b684b12.patch` and selected by
//! `log_batch_size`.
//!
//! A statement names one version: the wide one has its own `profile` in the identity and the Hello and its own domain
//! tag, so no session of one version reads as the other's. Its three names are the constants below, one line each.
use flock_core::pcs::ligerito::{LigeritoProfile, embedded_initial_k_or_default, wide_initial_k};

/// fast100's `profile`, in a statement's identity and a session's Hello.
pub const FAST100_PROFILE: &str = "fast100";
/// The wide version's `profile` there.
pub const PROFILE: &str = "fast100-wide-leaf";
/// The part of a statement's domain tag (`circuit::Tags::domain`) that names fast100's version, and the wide version's.
const FAST100_DOMAIN: &str = "/fast100x2/";
pub const DOMAIN: &str = "/fast100-wide-leafx2/";
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

/// A statement's domain tag in a version: `tag`, fast100's, as it is, or with the wide version's name in fast100's place.
pub fn domain(tag: &str, wide: bool) -> String {
    if !wide {
        return tag.to_string();
    }
    assert_eq!(tag.matches(FAST100_DOMAIN).count(), 1, "the domain tag {tag} names fast100's version once");
    tag.replacen(FAST100_DOMAIN, DOMAIN, 1)
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
    use crate::circuit::{TAGS, TMPL_TAGS, TYPED_TAGS};

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
            assert_eq!(domain(tags.domain, false), tags.domain);
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

    /// The wide version's names are its own: neither is fast100's, and its domain tags keep the form a rep's index follows.
    #[test]
    fn the_wide_versions_names_are_its_own() {
        assert_eq!((profile(true), profile(false)), ("fast100-wide-leaf", "fast100"));
        assert_eq!(domain(TAGS.domain, true), "flock-circuit/fast100-wide-leafx2/rep");
        assert_eq!(domain(TMPL_TAGS.domain, true), "flock-circuit-tmpl/fast100-wide-leafx2/rep");
        assert_eq!(bincode::serialize(&LIGERITO).unwrap(), bincode::serialize(&1u32).unwrap());
        assert_eq!(level0_queries("q_0", false), "q_0");
        assert_eq!(level0_queries("q_0", true), "q_0; wide leaf: level 0 of 2^19 symbols at m = 32..35 (k_0 = m - 25, L = 2^18), where q_0 = 219");
        let mut seen = std::collections::BTreeSet::new();
        for tags in [&TAGS, &TYPED_TAGS, &TMPL_TAGS] {
            let wide = domain(tags.domain, true);
            assert!(wide != tags.domain && wide.ends_with("/rep") && !wide.contains(FAST100_DOMAIN), "{wide}");
            assert_eq!(wide.replacen(DOMAIN, FAST100_DOMAIN, 1), tags.domain);
            seen.insert(wide);
            seen.insert(tags.domain.to_string());
        }
        assert_eq!(seen.len(), 6);
    }
}
