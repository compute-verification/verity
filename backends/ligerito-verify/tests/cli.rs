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
