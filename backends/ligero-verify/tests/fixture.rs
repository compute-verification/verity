//! The committed B = 1 ZK interactive-mode proof (run r20260922-111453-609b, `small_interactive_zk/b1`, protocol
//! after b-zk-fix) must verify, and one-byte mutations of every transcript region must be rejected -- through the
//! binary, so the CLI path is what is tested.
use std::process::Command;

fn bin() -> String {
    env!("CARGO_BIN_EXE_ligero-verify").to_string()
}

fn run(args: &[&str]) -> (bool, String) {
    let out = Command::new(bin()).args(args).output().expect("run ligero-verify");
    (out.status.success(), String::from_utf8_lossy(&out.stdout).to_string())
}

fn run_code(args: &[&str]) -> (i32, String) {
    let out = Command::new(bin()).args(args).output().expect("run ligero-verify");
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).to_string())
}

/// A batch directory of `n` copies of the B = 1 fixture as `sub_XX.{stmt,proof}`, with `sub_XX.coins` (the transcript's
/// own openings, bytes 108..236 of the proof) for the copies `with_coins` selects.
fn batch_dir(tag: &str, n: usize, with_coins: impl Fn(usize) -> bool) -> std::path::PathBuf {
    let st = std::fs::read("fixtures/b1.stmt").unwrap();
    let pf = std::fs::read("fixtures/b1.proof").unwrap();
    let dir = std::env::temp_dir().join(format!("ligero-verify-batch-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for i in 0..n {
        std::fs::write(dir.join(format!("sub_{i:02}.stmt")), &st).unwrap();
        std::fs::write(dir.join(format!("sub_{i:02}.proof")), &pf).unwrap();
        if with_coins(i) {
            std::fs::write(dir.join(format!("sub_{i:02}.coins")), &pf[108..236]).unwrap();
        }
    }
    dir
}

fn run_batch(dir: &std::path::Path, jobs: &str, threads: &str) -> (i32, String) {
    run_code(&["batch", "--system", "fixtures/system.bin", "--dir", dir.to_str().unwrap(), "--jobs", jobs, "--threads", threads])
}

/// The batch summary without everything that legitimately varies between runs: the timings, the wall, the job and
/// thread counts.
fn signature(js: &str) -> String {
    fn cut(s: &str, from: &str, to: &str) -> String {
        let mut out = String::new();
        let mut rest = s;
        while let Some(i) = rest.find(from) {
            out.push_str(&rest[..i]);
            let after = &rest[i..];
            let j = after.find(to).unwrap_or_else(|| panic!("{to} after {from} in {after}"));
            rest = &after[j..];
        }
        out.push_str(rest);
        out
    }
    let s = cut(js, "\"timing\":", "\"reps\":");
    let s = cut(&s, "\"jobs\":", "\"soundness_bits\":");
    cut(&s, "\"verify_seconds_sum\":", "\"items\":")
}

fn field<'a>(js: &'a str, key: &str) -> &'a str {
    let i = js.find(key).unwrap_or_else(|| panic!("{key} in {js}")) + key.len();
    let rest = &js[i..];
    let end = rest.find(|c: char| c == ',' || c == '}').unwrap();
    &rest[..end]
}

#[test]
fn batch_verdict_and_statement_do_not_depend_on_jobs() {
    // six honest sub-batches, three of them with their own coins file (checked as `verify --coins`), three without
    // (replayed coins: the conditional accept of DISCREPANCIES D7); the same batch on 1 job and on 8 jobs (clamped to 6)
    let dir = batch_dir("honest", 6, |i| i % 2 == 0);
    let (c1, j1) = run_batch(&dir, "1", "1");
    let (c8, j8) = run_batch(&dir, "8", "2");
    assert_eq!(c1, 0, "{j1}");
    assert_eq!(c8, 0, "{j8}");
    assert_eq!(field(&j1, "\"batch_accepted\":"), "true", "{j1}");
    assert_eq!(field(&j1, "\"accepted\":"), "6");
    assert_eq!(field(&j1, "\"own_coins\":"), "3");
    assert_eq!(field(&j1, "\"jobs\":"), "1");
    assert_eq!(field(&j8, "\"jobs\":"), "6", "clamped to the number of sub-batches");
    assert_eq!(signature(&j1), signature(&j8), "verdicts, reasons and parsed statements differ between --jobs 1 and --jobs 8");
    // the parsed statement (configuration, layout, BLAKE2b digest) is reported per sub-batch and identical for the copies
    let digests: Vec<&str> = j1.match_indices("\"digest\":\"").map(|(i, _)| &j1[i + 10..i + 74]).collect();
    assert_eq!(digests.len(), 6, "{j1}");
    assert!(digests.iter().all(|d| *d == digests[0]), "{digests:?}");
    assert!(j1.contains("\"l\":256,\"n\":2048,\"D\":6,\"t\":189,\"zk\":true,\"mode\":\"interactive\",\"n_proofs\":1"), "{j1}");
    assert_eq!(j1.matches("coins are this verifier's step-0 coins").count(), 3, "{j1}");
    assert_eq!(j1.matches("coins replayed from the transcript").count(), 3, "{j1}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_corrupted_sub_batch_rejects_the_batch_with_the_same_error_on_any_jobs() {
    let dir = batch_dir("corrupt", 6, |_| true);
    // one flipped byte in sub_03's opened columns: its Merkle path fails, the other five are untouched
    let p = dir.join("sub_03.proof");
    let mut pf = std::fs::read(&p).unwrap();
    let off = pf.len() - 189 * 11 * 32 - 1000;
    pf[off] ^= 1;
    std::fs::write(&p, &pf).unwrap();
    let (c1, j1) = run_batch(&dir, "1", "1");
    let (c8, j8) = run_batch(&dir, "8", "1");
    assert_eq!(c1, 1, "{j1}");
    assert_eq!(c8, 1, "{j8}");
    for js in [&j1, &j8] {
        assert_eq!(field(js, "\"batch_accepted\":"), "false", "{js}");
        assert_eq!(field(js, "\"accepted\":"), "5", "{js}");
        assert_eq!(field(js, "\"rejected\":"), "1", "{js}");
        assert!(field(js, "\"batch_reason\":").starts_with("\"sub_03.proof: merkle path "), "{js}");
    }
    assert_eq!(field(&j1, "\"batch_reason\":"), field(&j8, "\"batch_reason\":"));
    assert_eq!(signature(&j1), signature(&j8), "the rejection differs between --jobs 1 and --jobs 8");
    // a wrong coins file for one sub-batch is a rejection of that sub-batch, not a crash
    let mut coins = std::fs::read(dir.join("sub_00.coins")).unwrap();
    coins[5] ^= 1;
    std::fs::write(dir.join("sub_00.coins"), &coins).unwrap();
    let (c, js) = run_batch(&dir, "4", "1");
    assert_eq!(c, 1, "{js}");
    assert_eq!(field(&js, "\"rejected\":"), "2", "{js}");
    assert!(field(&js, "\"batch_reason\":").starts_with("\"sub_00.proof: coins are not this verifier's step-0 coins"), "{js}");
    // a missing statement file is an operator error (exit 2), reported after the other verdicts
    std::fs::remove_file(dir.join("sub_05.stmt")).unwrap();
    let (c, js) = run_batch(&dir, "4", "1");
    assert_eq!(c, 2, "{js}");
    assert!(field(&js, "\"batch_reason\":").contains("could not be read"), "{js}");
    let _ = std::fs::remove_dir_all(&dir);
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
    // the B = 1 ZK layout: D = 6, l = 256, k = 512, n = 2048, t = 189, M = 3516 + 6 D = 3552, depth 11)
    let regions = [("header", 8usize), ("coins", 44), ("root", 44 + 192), ("w", 44 + 192 + 32 + 7), ("cols", pf.len() - 3552 * 189 * 4 - 189 * 11 * 32 - 189 * 4 + 5), ("opened", pf.len() - 189 * 11 * 32 - 1000), ("paths", pf.len() - 100)];
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

#[test]
fn interactive_mode_own_coins_are_checked() {
    // the transcript's openings r1 || s1 || r2 || s2 sit at bytes 108..236 of the proof file (after the header and c1 c2)
    let pf = std::fs::read("fixtures/b1.proof").unwrap();
    let dir = std::env::temp_dir().join(format!("ligero-verify-coins-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mine = dir.join("mine.coins");
    std::fs::write(&mine, &pf[108..236]).unwrap();
    let (ok, js) = run(&["verify", "--system", "fixtures/system.bin", "--statement", "fixtures/b1.stmt", "--proof", "fixtures/b1.proof", "--coins", mine.to_str().unwrap()]);
    assert!(ok, "{js}");
    assert!(js.contains("coins are this verifier's step-0 coins"), "{js}");
    let mut other = pf[108..236].to_vec();
    other[0] ^= 1;
    let theirs = dir.join("theirs.coins");
    std::fs::write(&theirs, &other).unwrap();
    let (ok, js) = run(&["verify", "--system", "fixtures/system.bin", "--statement", "fixtures/b1.stmt", "--proof", "fixtures/b1.proof", "--coins", theirs.to_str().unwrap()]);
    assert!(!ok, "{js}");
    assert!(js.contains("coins are not this verifier's step-0 coins"), "{js}");
    // without --coins the accept is explicitly conditional
    let (ok, js) = run(&["verify", "--system", "fixtures/system.bin", "--statement", "fixtures/b1.stmt", "--proof", "fixtures/b1.proof"]);
    assert!(ok, "{js}");
    assert!(js.contains("coins replayed from the transcript"), "{js}");
    let _ = std::fs::remove_dir_all(&dir);
}
