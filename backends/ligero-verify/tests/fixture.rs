//! The committed B = 1 ZK proof (run r20260922-100145-374a, `small_zk/b1`) must verify, and one-byte mutations of
//! every transcript region must be rejected -- through the binary, so the CLI path is what is tested.
use std::process::Command;

fn bin() -> String {
    env!("CARGO_BIN_EXE_ligero-verify").to_string()
}

fn run(args: &[&str]) -> (bool, String) {
    let out = Command::new(bin()).args(args).output().expect("run ligero-verify");
    (out.status.success(), String::from_utf8_lossy(&out.stdout).to_string())
}

#[test]
fn honest_b1_zk_proof_is_accepted() {
    let (ok, js) = run(&["verify", "--system", "fixtures/system.bin", "--statement", "fixtures/b1.stmt", "--proof", "fixtures/b1.proof", "--threads", "2"]);
    assert!(ok, "{js}");
    assert!(js.contains("\"accepted\":true"));
}

#[test]
fn every_region_of_the_proof_is_bound() {
    let pf = std::fs::read("fixtures/b1.proof").unwrap();
    let dir = std::env::temp_dir().join(format!("ligero-verify-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // header flags, coins, root, w, h, q, v, cols, opened, paths: one flipped byte in each region (byte offsets of
    // the B = 1 ZK layout: D = 6, l = 256, k = 512, n = 2048, t = 189, M = 3546, depth 11)
    let regions = [("header", 8usize), ("coins", 44), ("root", 44 + 192), ("w", 44 + 192 + 32 + 7), ("cols", pf.len() - 3546 * 189 * 4 - 189 * 11 * 32 - 189 * 4 + 5), ("opened", pf.len() - 189 * 11 * 32 - 1000), ("paths", pf.len() - 100)];
    for (name, off) in regions {
        let mut m = pf.clone();
        m[off] ^= 1;
        let p = dir.join(format!("{name}.proof"));
        std::fs::write(&p, &m).unwrap();
        let (ok, js) = run(&["verify", "--system", "fixtures/system.bin", "--statement", "fixtures/b1.stmt", "--proof", p.to_str().unwrap()]);
        assert!(!ok, "mutation of {name} accepted: {js}");
        assert!(js.contains("\"accepted\":false"));
    }
    // truncated and extended files
    for (name, m) in [("truncated", pf[..pf.len() - 1].to_vec()), ("extended", [pf.clone(), vec![0u8]].concat())] {
        let p = dir.join(format!("{name}.proof"));
        std::fs::write(&p, &m).unwrap();
        let (ok, js) = run(&["verify", "--system", "fixtures/system.bin", "--statement", "fixtures/b1.stmt", "--proof", p.to_str().unwrap()]);
        assert!(!ok, "{name} accepted: {js}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_flipped_public_word_is_rejected() {
    let st = std::fs::read("fixtures/b1.stmt").unwrap();
    let mut m = st.clone();
    let i = m.len() - 1; // last byte of y16 (a pad column, part of the statement digest)
    m[i] ^= 1;
    let p = std::env::temp_dir().join(format!("ligero-verify-stmt-{}.stmt", std::process::id()));
    std::fs::write(&p, &m).unwrap();
    let (ok, js) = run(&["verify", "--system", "fixtures/system.bin", "--statement", p.to_str().unwrap(), "--proof", "fixtures/b1.proof"]);
    assert!(!ok, "{js}");
    assert!(js.contains("column challenge mismatch"), "{js}");
    let _ = std::fs::remove_file(&p);
}
