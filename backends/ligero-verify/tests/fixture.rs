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

/// Byte offset of `u32 n_proofs` in a ligero-statement/v2 file (magic 8 | layout 4 | l n D t 16 | target 8 | zk 1 |
/// t_pad 4 | mode 1).  Not part of the statement digest: it is the count the prover sized `t` for, and the batch rule
/// checks it against the number of sub-batches presented (PROTOCOL.md 8c, "Batch verification").
const N_PROOFS_OFF: usize = 42;

/// A batch directory of `n` copies of the B = 1 fixture as `sub_XX.{stmt,proof}`, each statement's `n_proofs` set to
/// `claimed` (the honest value for a batch of `n` is `n`), with `sub_XX.coins` (the transcript's own openings, bytes
/// 108..236 of the proof) for the copies `with_coins` selects.
fn batch_dir_claiming(tag: &str, n: usize, claimed: u32, with_coins: impl Fn(usize) -> bool) -> std::path::PathBuf {
    let mut st = std::fs::read("fixtures/b1.stmt").unwrap();
    assert_eq!(u32::from_le_bytes(st[N_PROOFS_OFF..N_PROOFS_OFF + 4].try_into().unwrap()), 1, "the fixture was proved at n_proofs = 1");
    st[N_PROOFS_OFF..N_PROOFS_OFF + 4].copy_from_slice(&claimed.to_le_bytes());
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

/// `n` copies claiming `n_proofs = n`.  The fixture's own bound is 2^-128.16 (t = 189 sized for ONE proof), so a batch
/// of `n` copies is sound to 2^-(128.16 - log2 n) only: the tests that want it accepted pass a target it meets.
fn batch_dir(tag: &str, n: usize, with_coins: impl Fn(usize) -> bool) -> std::path::PathBuf {
    batch_dir_claiming(tag, n, n as u32, with_coins)
}

/// The fixture copies as a batch: six of them are 2^-125.57, so `--target-bits 125`.
const COPIES_TARGET: &str = "125";

fn run_batch(dir: &std::path::Path, jobs: &str, threads: &str) -> (i32, String) {
    run_batch_at(dir, jobs, threads, COPIES_TARGET)
}

fn run_batch_at(dir: &std::path::Path, jobs: &str, threads: &str, target_bits: &str) -> (i32, String) {
    run_code(&["batch", "--system", "fixtures/system.bin", "--dir", dir.to_str().unwrap(), "--jobs", jobs, "--threads", threads, "--target-bits", target_bits])
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
    let s = cut(&s, "\"jobs\":", "\"target_bits\":");
    cut(&s, "\"verify_seconds_sum\":", "\"items\":")
}

fn num(js: &str, key: &str) -> f64 {
    field(js, key).trim_matches('"').parse().unwrap_or_else(|_| panic!("{key} is not a number in {js}"))
}

/// The JSON string value at `key`, quotes included (a reason may contain commas, which `field` stops at).
fn sfield<'a>(js: &'a str, key: &str) -> &'a str {
    let i = js.find(key).unwrap_or_else(|| panic!("{key} in {js}")) + key.len();
    let rest = &js[i..];
    assert!(rest.starts_with('"'), "{key} is not a string in {js}");
    let mut esc = false;
    for (j, ch) in rest.char_indices().skip(1) {
        match (esc, ch) {
            (true, _) => esc = false,
            (false, '\\') => esc = true,
            (false, '"') => return &rest[..=j],
            _ => {}
        }
    }
    panic!("unterminated string at {key} in {js}")
}

/// The comma-separated body of the JSON array at `key` (`key` ends with `[`).
fn array<'a>(js: &'a str, key: &str) -> &'a str {
    let i = js.find(key).unwrap_or_else(|| panic!("{key} in {js}")) + key.len();
    let rest = &js[i..];
    &rest[..rest.find(']').unwrap()]
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
    assert!(j1.contains("\"l\":256,\"n\":2048,\"D\":6,\"t\":189,\"zk\":true,\"mode\":\"interactive\",\"n_proofs\":6"), "{j1}");
    assert_eq!(j1.matches("coins are this verifier's step-0 coins").count(), 3, "{j1}");
    assert_eq!(j1.matches("coins replayed from the transcript").count(), 3, "{j1}");
    // the batch accounting: six copies of a 2^-128.16 proof are 2^-125.57 -- reported, and the accept reason carries it
    let bits = num(&j1, "\"batch_bits\":");
    assert!((125.5..125.7).contains(&bits), "{bits}");
    assert!(sfield(&j1, "\"batch_reason\":").starts_with("\"accept: all 6 sub-batches accepted; union bound 2^-125."), "{j1}");
    assert!(j1.contains("<= 2^-125\""), "{j1}");
    assert_eq!(array(&j1, "\"n_proofs_claimed\":["), "6,6,6,6,6,6", "{j1}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_batch_union_bound_and_n_proofs_are_enforced() {
    // (b) the scenario of the finding (PROTOCOL.md 8c "Batch verification", DISCREPANCIES D10): every sub-batch verifies
    // on its own at 2^-128.16, but four of them are only 2^-126.16 -- rejected at the default target, with the reason
    let dir = batch_dir("union", 4, |_| true);
    let (c, js) = run_batch_at(&dir, "4", "1", "128");
    assert_eq!(c, 1, "{js}");
    assert_eq!(field(&js, "\"accepted\":"), "4", "every sub-batch verifies; it is the BATCH that fails: {js}");
    assert_eq!(field(&js, "\"batch_accepted\":"), "false", "{js}");
    let reason = sfield(&js, "\"batch_reason\":").to_string();
    assert!(reason.starts_with("\"batch: union bound 2^-126."), "{js}");
    assert!(reason.ends_with("over 4 sub-batches exceeds the target 2^-128\""), "{js}");
    let bits = num(&js, "\"batch_bits\":");
    let per: Vec<f64> = array(&js, "\"per_proof_bits\":[").split(',').map(|x| x.parse().unwrap()).collect();
    assert_eq!(per.len(), 4);
    assert!((bits - (per[0] - 2.0)).abs() < 1e-9, "four equal bounds: batch = per - log2 4; {bits} vs {}", per[0]);
    assert!((128.1..128.2).contains(&per[0]), "{}", per[0]);
    // the default target is 128 (no flag), --soundness-bits is the alias, a fractional target prints two decimals
    let (c, js0) = run_code(&["batch", "--system", "fixtures/system.bin", "--dir", dir.to_str().unwrap(), "--jobs", "2"]);
    assert_eq!(c, 1, "{js0}");
    assert_eq!(sfield(&js0, "\"batch_reason\":"), reason, "{js0}");
    let (c, jsa) = run_code(&["batch", "--system", "fixtures/system.bin", "--dir", dir.to_str().unwrap(), "--soundness-bits", "126.5"]);
    assert_eq!(c, 1, "{jsa}");
    assert!(sfield(&jsa, "\"batch_reason\":").ends_with("exceeds the target 2^-126.50\""), "{jsa}");
    // (a) the same four at a target they meet: accepted, the bits reported, identical on any --jobs
    let (c1, a1) = run_batch_at(&dir, "1", "1", "126");
    let (c4, a4) = run_batch_at(&dir, "4", "1", "126");
    assert_eq!((c1, c4), (0, 0), "{a1}\n{a4}");
    assert_eq!(field(&a1, "\"batch_accepted\":"), "true", "{a1}");
    assert!(sfield(&a1, "\"batch_reason\":").starts_with("\"accept: all 4 sub-batches accepted; union bound 2^-126."), "{a1}");
    assert_eq!(signature(&a1), signature(&a4));
    assert_eq!(field(&a1, "\"batch_bits\":"), field(&js, "\"batch_bits\":"), "the bound does not depend on the target");
    let _ = std::fs::remove_dir_all(&dir);
    // (c) four proofs each claiming n_proofs = 1 presented together: rejected for the mismatch (checked before the
    // union bound, which they also fail), the same string as the Python verifier's, on any --jobs
    let dir = batch_dir_claiming("claim1", 4, 1, |_| true);
    let want = "\"batch: n_proofs mismatch (sub_00.proof: n_proofs = 1, 4 sub-batches presented)\"";
    for jobs in ["1", "4"] {
        let (c, js) = run_batch_at(&dir, jobs, "1", "128");
        assert_eq!(c, 1, "{js}");
        assert_eq!(field(&js, "\"accepted\":"), "4", "{js}");
        assert_eq!(sfield(&js, "\"batch_reason\":"), want, "{js}");
        assert_eq!(array(&js, "\"n_proofs_claimed\":["), "1,1,1,1", "{js}");
    }
    // a proof sized for a LARGER batch is not this batch's either
    let mut st = std::fs::read(dir.join("sub_02.stmt")).unwrap();
    st[N_PROOFS_OFF..N_PROOFS_OFF + 4].copy_from_slice(&25u32.to_le_bytes());
    std::fs::write(dir.join("sub_02.stmt"), &st).unwrap();
    for i in [0usize, 1, 3] {
        let mut st = std::fs::read(dir.join(format!("sub_{i:02}.stmt"))).unwrap();
        st[N_PROOFS_OFF..N_PROOFS_OFF + 4].copy_from_slice(&4u32.to_le_bytes());
        std::fs::write(dir.join(format!("sub_{i:02}.stmt")), &st).unwrap();
    }
    let (c, js) = run_batch_at(&dir, "4", "1", "126");
    assert_eq!(c, 1, "{js}");
    assert_eq!(sfield(&js, "\"batch_reason\":"), "\"batch: n_proofs mismatch (sub_02.proof: n_proofs = 25, 4 sub-batches presented)\"", "{js}");
    // a rejected sub-batch takes precedence over the accounting
    let p = dir.join("sub_01.proof");
    let mut pf = std::fs::read(&p).unwrap();
    let off = pf.len() - 100;
    pf[off] ^= 1;
    std::fs::write(&p, &pf).unwrap();
    let (c, js) = run_batch_at(&dir, "4", "1", "126");
    assert_eq!(c, 1, "{js}");
    assert!(sfield(&js, "\"batch_reason\":").starts_with("\"sub_01.proof: "), "{js}");
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
        assert!(sfield(js, "\"batch_reason\":").starts_with("\"sub_03.proof: merkle path "), "{js}");
    }
    assert_eq!(sfield(&j1, "\"batch_reason\":"), sfield(&j8, "\"batch_reason\":"));
    assert_eq!(signature(&j1), signature(&j8), "the rejection differs between --jobs 1 and --jobs 8");
    // a wrong coins file for one sub-batch is a rejection of that sub-batch, not a crash
    let mut coins = std::fs::read(dir.join("sub_00.coins")).unwrap();
    coins[5] ^= 1;
    std::fs::write(dir.join("sub_00.coins"), &coins).unwrap();
    let (c, js) = run_batch(&dir, "4", "1");
    assert_eq!(c, 1, "{js}");
    assert_eq!(field(&js, "\"rejected\":"), "2", "{js}");
    assert!(sfield(&js, "\"batch_reason\":").starts_with("\"sub_00.proof: coins are not this verifier's step-0 coins"), "{js}");
    // a missing statement file is an operator error (exit 2), reported after the other verdicts
    std::fs::remove_file(dir.join("sub_05.stmt")).unwrap();
    let (c, js) = run_batch(&dir, "4", "1");
    assert_eq!(c, 2, "{js}");
    assert!(sfield(&js, "\"batch_reason\":").contains("could not be read"), "{js}");
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
