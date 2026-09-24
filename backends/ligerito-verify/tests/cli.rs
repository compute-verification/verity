//! The release binary's verdicts: `batch` over the must-reject set, a forgery presented as a proof, and the job-level
//! target (`--n-proofs`, red-team F5).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ligerito-verify")).args(args).output().expect("run ligerito-verify")
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ligerito-verify-cli-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

#[test]
fn batch_rejects_every_forged_fixture() {
    let dir = fixtures().join("forged");
    let o = run(&["batch", "--dir", dir.to_str().unwrap(), "--target-bits", "0"]);
    let s = stdout(&o);
    assert!(o.status.success(), "{s}");
    assert!(s.contains("\"accepted\":true") && s.contains("\"n_accepted\":3") && s.contains("\"problems\":[]"), "{s}");
    // the honest toys verify but are 8-query proofs: the default 2^-128 union target fails the batch
    assert!(!run(&["batch", "--dir", dir.to_str().unwrap()]).status.success());
}

#[test]
fn a_forgery_presented_as_a_proof_fails_the_batch() {
    let d = scratch("forgery");
    for f in ["weakfs_v2", "redteam_weakfs_relabelled_v2", "overflow_v2"] {
        std::fs::copy(fixtures().join(format!("forged/{f}.proof.neg")), d.join(format!("{f}.proof"))).unwrap();
    }
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--target-bits", "0"]);
    assert!(!o.status.success());
    let s = stdout(&o);
    assert!(s.contains("\"n_accepted\":0") && s.contains("sumcheck round") && s.contains("non-canonical"), "{s}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn n_proofs_tightens_verify_and_is_checked_by_batch() {
    let d = scratch("nproofs");
    // default selftest proof: n = 12, 16 queries per round, soundness 2^-9.00
    let o = run(&["selftest", "--emit", d.to_str().unwrap()]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let honest = d.join("honest.proof");
    let h = honest.to_str().unwrap();
    assert!(run(&["verify", "--proof", h, "--target-bits", "8"]).status.success());
    let o = run(&["verify", "--proof", h, "--target-bits", "8", "--n-proofs", "4"]);
    assert!(!o.status.success() && stdout(&o).contains("over 4 proof(s)"), "{}", stdout(&o));
    // the emitted dir holds 3 must-accept proofs (honest, live + coins, proto + z) and the negatives
    let ds = d.to_str().unwrap();
    assert!(run(&["batch", "--dir", ds, "--target-bits", "0", "--n-proofs", "3"]).status.success());
    let o = run(&["batch", "--dir", ds, "--target-bits", "0", "--n-proofs", "4"]);
    assert!(!o.status.success() && stdout(&o).contains("n_proofs mismatch"), "{}", stdout(&o));
    std::fs::write(d.join("manifest.json"), r#"{"n_proofs": 2, "proofs": []}"#).unwrap();
    assert!(!run(&["batch", "--dir", ds, "--target-bits", "0"]).status.success());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn sumcheck_fixture_verifies_lgsc0003_and_rejects_a_tampered_proof() {
    let f = fixtures().join("sumcheck").join("lgsc0003_fp8ada_l64_S2.json");
    let o = run(&["sumcheck-fixture", "--fixture", f.to_str().unwrap()]);
    let s = stdout(&o);
    assert!(o.status.success(), "{s}");
    for want in ["\"accepted\":true", "\"format\":\"LGSC0003\"", "\"scripted\":\"accept, claims match\"", "\"negatives\":6,\"negatives_rejected_same_stage\":6"] {
        assert!(s.contains(want), "{want} not in {s}");
    }
    // one hex digit of the first zero-check message changed
    let text = std::fs::read_to_string(&f).unwrap();
    let at = text.find("\"proof_hex\": \"").unwrap() + "\"proof_hex\": \"".len() + 2 * 22 + 1;
    let flipped = if &text[at..at + 1] == "0" { "1" } else { "0" };
    let bad = scratch("lgsc3").join("tampered.json");
    std::fs::write(&bad, format!("{}{}{}", &text[..at], flipped, &text[at + 1..])).unwrap();
    let o = run(&["sumcheck-fixture", "--fixture", bad.to_str().unwrap()]);
    let s = stdout(&o);
    assert!(!o.status.success() && s.contains("\"accepted\":false") && s.contains("zero-check round 0"), "{s}");
}

/// An LGTO0001 dump in run.py's layout (key.bin, <stem>.stmt, <stem>.lgto / .lgto.neg, manifest.json with the Python
/// verdicts): the pre-V1 fixture verifies only under `--allow-legacy`, and even then carries no soundness and fails the
/// batch (red-team-3 R3-1); a proof listed as must-reject but accepted, or a Python verdict we disagree with, fails it.
#[test]
fn batch_over_an_lgto_dump() {
    let src = fixtures().join("lgto/fp8ada_l256_legacy");
    let d = scratch("lgto");
    std::fs::copy(src.join("key.bin"), d.join("key.bin")).unwrap();
    std::fs::copy(src.join("honest_fiat-shamir.stmt"), d.join("honest_fiat-shamir.stmt")).unwrap();
    std::fs::copy(src.join("honest_fiat-shamir.lgto"), d.join("honest_fiat-shamir.lgto")).unwrap();
    let mut neg = std::fs::read(src.join("honest_fiat-shamir.lgto")).unwrap();
    let at = neg.len() - 100;
    neg[at] ^= 1;
    std::fs::write(d.join("neg_00.lgto.neg"), &neg).unwrap();
    std::fs::copy(src.join("honest_fiat-shamir.stmt"), d.join("neg_00.stmt")).unwrap();
    let manifest = |neg_py: bool| {
        format!(
            r#"{{"format": {{"proof": "LGTO0001"}}, "files": [{{"proof": "honest_fiat-shamir.lgto", "stmt": "honest_fiat-shamir.stmt", "expect": "accept", "coins": "fiat-shamir", "coins_seed": 1, "python_verdict": {{"accepted": true, "reason": "accept"}}}}, {{"proof": "neg_00.lgto.neg", "stmt": "neg_00.stmt", "expect": "reject", "coins": "fiat-shamir", "coins_seed": 1, "python_verdict": {{"accepted": {neg_py}, "reason": "x"}}}}]}}"#
        )
    };
    std::fs::write(d.join("manifest.json"), manifest(false)).unwrap();
    let ds = d.to_str().unwrap();
    let o = run(&["batch", "--dir", ds, "--allow-legacy"]);
    let s = stdout(&o);
    assert!(!o.status.success(), "{s}");
    for want in [
        "\"n_accepted\":1",
        "\"manifest_checked\":2",
        "\"accepted_legacy\":1",
        "\"problems\":[\"1 pre-V1 proof(s) accepted under --allow-legacy",
        "\"union_soundness_log2\":null",
        "\"soundness_log2\":null,\"claimed_log2\":null,\"claim_basis\":\"none: pre-V1",
        "\"sumcheck\":\"LGSC0002\"",
        "\"zk_mode\":\"none\"",
        "\"key_pinned\":true",
    ] {
        assert!(s.contains(want), "{want} not in {s}");
    }
    let o = run(&["batch", "--dir", ds]);
    assert!(!o.status.success() && stdout(&o).contains("pre-V1"), "{}", stdout(&o));
    std::fs::write(d.join("manifest.json"), manifest(true)).unwrap();
    let o = run(&["batch", "--dir", ds, "--allow-legacy"]);
    assert!(!o.status.success() && stdout(&o).contains("python_verdict true"), "{}", stdout(&o));
    // the honest proof presented as must-reject
    std::fs::write(d.join("manifest.json"), manifest(false)).unwrap();
    std::fs::rename(d.join("honest_fiat-shamir.lgto"), d.join("honest_fiat-shamir.lgto.neg")).unwrap();
    assert!(!run(&["batch", "--dir", ds, "--allow-legacy"]).status.success());
    let _ = std::fs::remove_dir_all(&d);
}

/// ligerito-relation-2 32d3d42's V1 proofs in place, no manifest (statements by stem, one key): the honest proof and its
/// `--zk` twin accepted in default mode, the five V1 forgeries and 32e9bd5's proof for a non-canonical statement
/// rejected; both honest proofs share a statement, so the union bound counts it once.
#[test]
fn batch_over_the_v1_fixture() {
    let d = fixtures().join("lgto/fp8ada_l256_v1");
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--target-bits", "128"]);
    let s = stdout(&o);
    assert!(o.status.success(), "{s}");
    assert!(s.contains("{\"name\":\"canon_y_offend.lgto.neg\",\"accepted\":false,\"reason\":\"statement sub-batch 0: claimed word off a chain end (non-canonical)\""), "{s}");
    for want in ["\"accepted\":true", "\"n_proofs\":2", "\"n_negatives\":6", "\"n_accepted\":2", "\"problems\":[]", "\"accepted_unpinned_key\":0", "\"sumcheck\":\"LGSC0003\"", "\"zero_blocks\":6", "\"zk\":true"] {
        assert!(s.contains(want), "{want} not in {s}");
    }
    // relation-2's manifests (e3ad950) count distinct statements: the two honest proofs share one
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--target-bits", "128", "--n-proofs", "1"]);
    assert!(o.status.success(), "{}", stdout(&o));
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--target-bits", "128", "--n-proofs", "2"]);
    // F5: as two proofs of one job each must reach 2^-129; 2^-128.02 does not
    assert!(!o.status.success() && stdout(&o).contains("2^-128.02 > 2^-129"), "{}", stdout(&o));
}

/// relation-2 e61b24fe's `--zk` gate (LGSC0004): the ZK key (8c4cd91f) is pinned by derivation from the pinned LGSC0003
/// key, the honest proof verifies at 2^-128, and the ZK V1 must-rejects (committed next:0 / M_next off their shift), a
/// forged-every-round zero-check and a tampered claim value are rejected where Python rejects them.
#[test]
fn batch_over_the_lgsc0004_zk_gate_subset() {
    let d = fixtures().join("lgto/fp8ada_l256_zk_lgsc4");
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--target-bits", "128"]);
    let s = stdout(&o);
    assert!(o.status.success(), "{s}");
    for want in [
        "\"accepted\":true",
        "\"n_proofs_expected\":1",
        "\"n_proofs\":1",
        "\"n_negatives\":4",
        "\"n_accepted\":1",
        "\"manifest_checked\":5",
        "\"problems\":[]",
        "\"accepted_unpinned_key\":0",
        "\"sumcheck\":\"LGSC0004\"",
        "\"zk_mode\":\"lgsc0004\"",
        "\"key_pinned\":true",
        "\"key_derived_from\":\"da27387bb23ff60a60b4b21ef99f180c84c17456edf41efa31e5797aaed9227b\"",
        "\"key_sha256\":\"8c4cd91f45f28ee47a7c650779068907cf96e7462592c3f80ceb301a186a409a\"",
        "{\"name\":\"zk_next0.lgto.neg\",\"accepted\":false,\"reason\":\"combined round 0: sum_b q(b) != claim\"",
        "{\"name\":\"zk_mnext.lgto.neg\",\"accepted\":false,\"reason\":\"combined round 0: sum_b q(b) != claim\"",
    ] {
        assert!(s.contains(want), "{want} not in {s}");
    }
}

/// `batch --session`: live proofs are authenticated by the verifier's records (R3-7). A second copy of a proof cannot
/// reuse its recorded batch; a label-grinding proof (valid under its own .coins) is accepted by file replay with no
/// claim, and rejected under the records.
#[test]
fn batch_with_verifier_sessions() {
    let (z, s) = (fixtures().join("lgto/fp8ada_l256_zk_lgsc4"), fixtures().join("session/fp8ada_l256_zk_live"));
    let sess = s.join("verifier_sessions");
    let d = scratch("session");
    std::fs::copy(z.join("key.bin"), d.join("key.bin")).unwrap();
    for stem in ["a", "b"] {
        std::fs::copy(s.join("honest.lgto"), d.join(format!("{stem}.lgto"))).unwrap();
        std::fs::copy(z.join("honest.stmt"), d.join(format!("{stem}.stmt"))).unwrap();
    }
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--session", sess.to_str().unwrap()]);
    let out = stdout(&o);
    assert!(!o.status.success(), "{out}");
    assert!(out.contains("{\"name\":\"a.lgto\",\"accepted\":true,\"reason\":\"ok (session c20260924T002130Z-1324 batch 0, 28 rounds)\""), "{out}");
    assert!(out.contains("already authenticates another proof"), "{out}");
    std::fs::remove_file(d.join("b.lgto")).unwrap();
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--session", sess.to_str().unwrap()]);
    let out = stdout(&o);
    assert!(o.status.success(), "{out}");
    for want in ["\"claimed_union_log2\":-126.030", "\"live_session\":{\"session\":\"c20260924T002130Z-1324\",\"batch\":0}", "\"mode\":\"lgto live-session\""] {
        assert!(out.contains(want), "{want} not in {out}");
    }
    // R3-10: attempts are countable only over the whole store: one record dir, or a store with an open session, claims nothing
    let one = sess.join("c20260924T002130Z-1324");
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--session", one.to_str().unwrap()]);
    let out = stdout(&o);
    assert!(o.status.success() && out.contains("\"claimed_union_log2\":null") && out.contains("single session record"), "{out}");
    let store = scratch("store");
    for e in std::fs::read_dir(&sess).unwrap().flatten() {
        let to = store.join(e.file_name());
        std::fs::create_dir_all(&to).unwrap();
        for f in ["session.json", "verdict.json", "hello.json"] {
            std::fs::copy(e.path().join(f), to.join(f)).unwrap();
        }
    }
    std::fs::create_dir_all(store.join("c-open")).unwrap();
    std::fs::copy(one.join("hello.json"), store.join("c-open/hello.json")).unwrap();
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--session", store.to_str().unwrap()]);
    let out = stdout(&o);
    assert!(o.status.success() && out.contains("\"claimed_union_log2\":null") && out.contains("[\\\"c-open\\\"] in the verifier's store have no record"), "{out}");
    let _ = std::fs::remove_dir_all(&store);
    std::fs::remove_file(d.join("a.lgto")).unwrap();
    std::fs::copy(s.join("label_grinding.lgto"), d.join("a.lgto")).unwrap();
    std::fs::copy(s.join("label_grinding.coins"), d.join("a.coins")).unwrap();
    let o = run(&["batch", "--dir", d.to_str().unwrap()]);
    let out = stdout(&o);
    assert!(o.status.success() && out.contains("\"claimed_union_log2\":null") && out.contains("lgto live-replay"), "{out}");
    let o = run(&["batch", "--dir", d.to_str().unwrap(), "--session", sess.to_str().unwrap()]);
    let out = stdout(&o);
    assert!(!o.status.success() && out.contains("proof rejected under the recorded coins"), "{out}");
    let _ = std::fs::remove_dir_all(&d);
}

/// red-team-ligerito-2's V1 forgery (y16[47] += 1 with w' = w - e_(3581, 47)): the sumcheck layer accepts it by
/// construction; the zero claim on the virtual block holding row 3581 is non-zero on the forged w, so the PCS opening the
/// full verifier demands (value 0) cannot be met.
#[test]
fn redteam_v1_forgery_is_caught_by_the_zero_claims() {
    let f = fixtures().join("forged").join("virtrow_y16_lgsc0003.json");
    let o = run(&["sumcheck-fixture", "--fixture", f.to_str().unwrap()]);
    let s = stdout(&o);
    for want in ["\"scripted\":\"accept, claims match\"", "\"v1_caught\":true", "\"nonzero_zero_claims\":[\"w = 0 on rows [3580, 3584)\"]"] {
        assert!(s.contains(want), "{want} not in {s}");
    }
}
