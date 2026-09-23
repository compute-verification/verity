//! Relation-named statements (`ligero-statement/v4`) and the per-relation pinned systems, through the binary.
//!
//! `fixtures/fp8-ada/`: sub-batch 0 of rep 1 of the FP8 Ada dump `run-files` art:c5dff3c6... (attempt
//! r20260922-235542-295b, lane ada-fp8-dumps, code a4b0ac7; interactive non-ZK, B = 4096, 85 VUs x 48 steps at
//! l = 4096, K = 32) with the runner's step-0 coins and the pod's `system.bin` (sha256 cba9a09f..., byte-identical
//! to the laptop compilation of `compile_fp8_unit(ADA, chain=True)`).  `fixtures/systems/`: the bf16-hopper and
//! fp8-hopper systems compiled from lane b-hopper a39de82 (no proofs of theirs are in the store yet).
//!
//! `fixtures/fp4-nvf4/`: sub-batch 0 of rep 1 of the NVFP4 non-ZK interactive dump (run r20260923-024035-f3cf, lane
//! fp4-proof, code 3b03017, RTX 5090; B = 4096, 170 VUs x 24 steps at l = 4096, K = 68 = 64 E2M1 codes + 4 UE4M3
//! scale bytes per operand per step, 25 sub-batches) with the runner's step-0 coins (sha256 7cead39c...) and the
//! pod's `ligero-system/v2` file `system.bin` (sha256 6bb9d77f..., byte-identical to the laptop recompilation of
//! `compile_fp4_unit()`; the chain ends on the FP32 accumulator's sign 31/1, exponent 23/8 and fraction 0/23).  The
//! statement and proof sha256s (8625d1bf..., f57e90de...) are the dump manifest's.
use std::process::Command;

fn bin() -> String {
    env!("CARGO_BIN_EXE_ligero-verify").to_string()
}

fn run(args: &[&str]) -> (i32, String) {
    let out = Command::new(bin()).args(args).output().expect("run ligero-verify");
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).to_string())
}

const BF16_SYS: &str = "fixtures/system.bin";
const FP8_SYS: &str = "fixtures/fp8-ada/system.bin";
const FP8_STMT: &str = "fixtures/fp8-ada/sub_00.stmt";
const FP8_PROOF: &str = "fixtures/fp8-ada/sub_00.proof";
const FP8_COINS: &str = "fixtures/fp8-ada/sub_00.coins";
const FP4_SYS: &str = "fixtures/fp4-nvf4/system.bin";
const FP4_STMT: &str = "fixtures/fp4-nvf4/sub_00.stmt";
const FP4_PROOF: &str = "fixtures/fp4-nvf4/sub_00.proof";
const FP4_COINS: &str = "fixtures/fp4-nvf4/sub_00.coins";
const FP4_SYS_ID: &str = "a825ba0b826a15366bfb8cdd720a5e9978603cbe589491abcbba7346ad35bf3c";
const FP4_TABLE_DIGEST: &str = "b597b003e55b8cf1025b517a04d442d9c6d8704f38fd06df13a21b3273b2d2f6";

fn verify(sys: &str, stmt: &str, proof: &str, extra: &[&str]) -> (i32, String) {
    let mut args = vec!["verify", "--system", sys, "--statement", stmt, "--proof", proof, "--threads", "2"];
    args.extend_from_slice(extra);
    run(&args)
}

fn field<'a>(js: &'a str, key: &str) -> &'a str {
    let i = js.find(key).unwrap_or_else(|| panic!("{key} in {js}")) + key.len();
    let rest = &js[i..];
    let end = rest.find(|c: char| c == ',' || c == '}').unwrap();
    &rest[..end]
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ligero-verify-relations-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn write_tmp(name: &str, bytes: &[u8]) -> String {
    let p = tmp(name);
    std::fs::write(&p, bytes).unwrap();
    p.to_string_lossy().to_string()
}

// ------------------------------------------------------------------------------------------- statement re-encoder ----

/// A parsed statement (any version): the 58-byte v2 header through K, the words as u32 and the trailing bytes.
struct Stmt {
    header: Vec<u8>, // bytes 8..58 of the file: layout .. K
    l: usize,
    k: usize,
    a: Vec<u32>,
    b: Vec<u32>,
    y: Vec<u32>,
    auth: Option<Vec<u8>>,
}

fn u32_at(d: &[u8], at: usize) -> usize {
    u32::from_le_bytes(d[at..at + 4].try_into().unwrap()) as usize
}

fn read_words(d: &[u8], at: &mut usize, n: usize, w: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = 0u32;
        for k in 0..w {
            v |= (d[*at + i * w + k] as u32) << (8 * k);
        }
        out.push(v);
    }
    *at += n * w;
    out
}

fn parse(d: &[u8]) -> Stmt {
    let (l, k) = (u32_at(d, 12), u32_at(d, 54));
    let mut at = 58;
    let (wb, yb) = match &d[..8] {
        b"LIGSTM04" => {
            let (wb, yb) = (d[58] as usize, d[59] as usize);
            let n = u32_at(d, 60);
            at = 64 + n;
            (wb, yb)
        }
        _ => (2, 2),
    };
    let a = read_words(d, &mut at, l * k, wb);
    let b = read_words(d, &mut at, l * k, wb);
    let y = read_words(d, &mut at, l, yb);
    let auth = match &d[..8] {
        b"LIGSTM02" => None,
        b"LIGSTM03" => Some(d[at..].to_vec()),
        _ => {
            let has = d[at];
            at += 1;
            if has == 1 { Some(d[at..].to_vec()) } else { None }
        }
    };
    Stmt { header: d[8..58].to_vec(), l, k, a, b, y, auth }
}

fn put_words(out: &mut Vec<u8>, words: &[u32], w: usize) {
    for x in words {
        out.extend_from_slice(&x.to_le_bytes()[..w]);
    }
}

/// Re-encode as version 2 / 3 (BF16, 2-byte words) or 4 (`relation`, `wb` / `yb` byte words, optional auth block).
fn encode(s: &Stmt, version: u8, relation: &str, wb: usize, yb: usize, auth: Option<&[u8]>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(match version {
        2 => b"LIGSTM02",
        3 => b"LIGSTM03",
        _ => b"LIGSTM04",
    });
    out.extend_from_slice(&s.header);
    if version == 4 {
        out.push(wb as u8);
        out.push(yb as u8);
        out.extend_from_slice(&(relation.len() as u32).to_le_bytes());
        out.extend_from_slice(relation.as_bytes());
    }
    put_words(&mut out, &s.a, wb);
    put_words(&mut out, &s.b, wb);
    put_words(&mut out, &s.y, yb);
    if version == 4 {
        out.push(auth.is_some() as u8);
    }
    if let Some(ab) = auth {
        out.extend_from_slice(ab);
    }
    out
}

/// A well-formed authentication block for one VU at index 0 with K words per step (the roots are arbitrary: the test
/// proofs carry no multiproofs, so the pairing check rejects before any root is used).
fn auth_block(steps: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for s in ["ligero-b/auth/v1", "u16"] {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    out.extend_from_slice(&1u32.to_le_bytes()); // n_vus
    out.extend_from_slice(&0u32.to_le_bytes()); // vu_index[0]
    for (i, count) in [(steps * k) as u64, (steps * k) as u64, 1u64].iter().enumerate() {
        out.extend_from_slice(&[i as u8 + 1; 32]); // binding
        out.push(0); // owner -2
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&[i as u8 + 7; 32]); // root
    }
    out
}

// ------------------------------------------------------------------------------------------------------ tests ----

#[test]
fn fp8_ada_v4_honest_proof_is_accepted_against_its_pinned_system() {
    let (code, js) = verify(FP8_SYS, FP8_STMT, FP8_PROOF, &["--coins", FP8_COINS]);
    assert_eq!(code, 0, "{js}");
    assert_eq!(field(&js, "\"accepted\":"), "true");
    assert_eq!(field(&js, "\"system_pinned\":"), "true");
    assert!(js.contains("\"format\":\"ligero-statement/v4\",\"relation\":\"fp8-ada\""), "{js}");
    assert!(js.contains("\"l\":4096,\"n\":16384,\"D\":6,\"t\":198,\"zk\":false,\"mode\":\"interactive\",\"n_proofs\":49,\"n_vus\":85,\"steps\":48,\"K\":32"), "{js}");
    assert!(js.contains("coins are this verifier's step-0 coins"), "{js}");
    assert!(!js.contains("NOT pinned"), "{js}");
    // the same proof without --coins: the conditional accept, as for the BF16 fixture
    let (code, js) = verify(FP8_SYS, FP8_STMT, FP8_PROOF, &[]);
    assert_eq!(code, 0, "{js}");
    assert!(js.contains("coins replayed from the transcript"), "{js}");
}

#[test]
fn system_digest_names_every_pinned_relation() {
    for (path, rel, m, pins) in [
        (BF16_SYS, "bf16-ampere", "3516", "96"),
        (FP8_SYS, "fp8-ada", "3769", "192"),
        ("fixtures/systems/bf16-hopper.system.bin", "bf16-hopper", "3292", "96"),
        ("fixtures/systems/fp8-hopper.system.bin", "fp8-hopper", "3396", "192"),
        (FP4_SYS, "fp4-nvf4", "1583", "141"),
        // the v2 public-selection systems (lane relmin-lookup): recompiled torch-free, sys_id == protocol.system_id
        ("fixtures/systems/bf16-hopper-v2.system.bin", "bf16-hopper-v2", "282", "55"),
        ("fixtures/systems/bf16-ampere-v2.system.bin", "bf16-ampere-v2", "449", "54"),
        ("fixtures/systems/fp8-hopper-v2.system.bin", "fp8-hopper-v2", "185", "16"),
        ("fixtures/systems/fp8-ada-v2.system.bin", "fp8-ada-v2", "337", "32"),
        // the v3 private-operand systems (lane relmin-private): recompiled torch-free, sys_id == protocol.system_id
        ("fixtures/systems/bf16-hopper-v3.system.bin", "bf16-hopper-v3", "1519", "32"),
        ("fixtures/systems/bf16-ampere-v3.system.bin", "bf16-ampere-v3", "1629", "32"),
        ("fixtures/systems/fp8-hopper-v3.system.bin", "fp8-hopper-v3", "1673", "22"),
        ("fixtures/systems/fp8-ada-v3.system.bin", "fp8-ada-v3", "1806", "22"),
        // the folded x4 units (four instruction steps per column; experimental)
        ("fixtures/systems/bf16-hopper-v2x4.system.bin", "bf16-hopper-v2x4", "1045", "220"),
        ("fixtures/systems/bf16-ampere-v2x4.system.bin", "bf16-ampere-v2x4", "1717", "216"),
        ("fixtures/systems/fp8-hopper-v2x4.system.bin", "fp8-hopper-v2x4", "647", "64"),
        ("fixtures/systems/fp8-ada-v2x4.system.bin", "fp8-ada-v2x4", "1255", "128"),
        // the private-operand-safe folds (lane fold-private): v3 x4 and v1 x4, recompiled torch-free, sys_id == system_id
        ("fixtures/systems/bf16-hopper-v3x4.system.bin", "bf16-hopper-v3x4", "5929", "128"),
        ("fixtures/systems/bf16-ampere-v3x4.system.bin", "bf16-ampere-v3x4", "6369", "128"),
        ("fixtures/systems/fp8-hopper-v3x4.system.bin", "fp8-hopper-v3x4", "6558", "86"),
        ("fixtures/systems/fp8-ada-v3x4.system.bin", "fp8-ada-v3x4", "7090", "86"),
        ("fixtures/systems/bf16-hopper-x4.system.bin", "bf16-hopper-x4", "12792", "384"),
        ("fixtures/systems/bf16-ampere-x4.system.bin", "bf16-ampere-x4", "14576", "384"),
        ("fixtures/systems/fp8-hopper-x4.system.bin", "fp8-hopper-x4", "13383", "768"),
        ("fixtures/systems/fp8-ada-x4.system.bin", "fp8-ada-x4", "14875", "768"),
    ] {
        let (code, js) = run(&["system-digest", "--system", path]);
        assert_eq!(code, 0, "{js}");
        assert_eq!(field(&js, "\"pinned_relation\":"), format!("\"{rel}\""), "{js}");
        assert_eq!(field(&js, "\"m\":"), m, "{js}");
        assert_eq!(field(&js, "\"pins\":"), pins, "{js}");
        assert_eq!(field(&js, "\"chain\":"), "true", "{js}");
        // the file format and the chain end: the four v1 systems and the four public-selection systems end on one
        // public word, the fp4 v2 system on the three components of the FP32 accumulator word
        if rel == "fp4-nvf4" {
            assert!(js.contains("\"format\":\"ligero-system/v2\""), "{js}");
            assert!(js.contains("\"chain_end\":[{\"lo\":31,\"bits\":1},{\"lo\":23,\"bits\":8},{\"lo\":0,\"bits\":23}]"), "{js}");
            assert!(js.contains(&format!("\"sys_id\":\"{FP4_SYS_ID}\",\"table_digest\":\"{FP4_TABLE_DIGEST}\"")), "{js}");
            assert_eq!(field(&js, "\"L\":"), "306", "{js}");
            assert_eq!(field(&js, "\"Q\":"), "1389", "{js}");
        } else {
            assert!(js.contains("\"format\":\"ligero-system/v1\""), "{js}");
            assert!(js.contains("\"chain_end\":\"word\""), "{js}");
        }
    }
    // a system file with other tables is pinned for no relation
    let mut sys = std::fs::read(FP8_SYS).unwrap();
    sys[100] ^= 1;
    let p = write_tmp("other.system.bin", &sys);
    let (code, js) = run(&["system-digest", "--system", &p]);
    assert!(code == 0 || code == 2, "{js}");
    if code == 0 {
        assert_eq!(field(&js, "\"pinned_relation\":"), "null", "{js}");
    }
}

#[test]
fn a_statement_against_another_relations_system_is_rejected_unless_unpinned() {
    // the FP8 statement against the pinned BF16 Ampere system: refused by the pin, before any parsing of the proof
    let (code, js) = verify(BF16_SYS, FP8_STMT, FP8_PROOF, &["--coins", FP8_COINS]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned bf16-ampere system, the statement is a fp8-ada statement"), "{js}");
    assert_eq!(field(&js, "\"system_pinned\":"), "true", "the pin WAS enforced: {js}");
    // and the BF16 statement (v2) against the FP8 system
    let (code, js) = verify(FP8_SYS, "fixtures/b1.stmt", "fixtures/b1.proof", &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned fp8-ada system, the statement is a bf16-ampere statement"), "{js}");
    // --allow-any-system lifts the pin: the verdict is then about the system FILE, flagged system_pinned = false.
    // The mismatched pair still rejects (the proof's row count is the other system's), the honest pairs accept unpinned.
    let (code, js) = verify(FP8_SYS, "fixtures/b1.stmt", "fixtures/b1.proof", &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert_eq!(field(&js, "\"system_pinned\":"), "false", "{js}");
    assert!(js.contains("malformed proof: header M = 3552, the statement fixes 3805"), "{js}");
    for (sys, stmt, proof) in [(BF16_SYS, "fixtures/b1.stmt", "fixtures/b1.proof"), (FP8_SYS, FP8_STMT, FP8_PROOF)] {
        let (code, js) = verify(sys, stmt, proof, &["--allow-any-system"]);
        assert_eq!(code, 0, "{js}");
        assert_eq!(field(&js, "\"accepted\":"), "true", "{js}");
        assert_eq!(field(&js, "\"system_pinned\":"), "false", "{js}");
        assert!(js.contains("[system NOT pinned: --allow-any-system; sys_id "), "{js}");
    }
    // a system whose digests are pinned for no relation: refused with both pairs named, accepted only unpinned
    let mut sys = std::fs::read(FP8_SYS).unwrap();
    let n = sys.len();
    sys[n - 1] ^= 1; // the trailing sys_id: the tables are intact, the Python digest is not
    let p = write_tmp("unpinned.system.bin", &sys);
    let (code, js) = verify(&p, FP8_STMT, FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is not the pinned fp8-ada system"), "{js}");
    // (unpinned it still rejects: the sys_id enters the statement digest, so the challenges no longer match)
    let (code, js) = verify(&p, FP8_STMT, FP8_PROOF, &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("column challenge mismatch"), "{js}");
}

#[test]
fn batch_reports_the_pin_and_reads_the_parent_manifest() {
    // a bench-vu dump layout: proofs/manifest.json naming `rep1/sub_00.proof`, the files under proofs/rep1/
    let root = tmp("dump");
    let rep = root.join("proofs").join("rep1");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&rep).unwrap();
    for ext in ["stmt", "proof", "coins"] {
        std::fs::copy(format!("fixtures/fp8-ada/sub_00.{ext}"), rep.join(format!("sub_00.{ext}"))).unwrap();
    }
    std::fs::write(
        root.join("proofs").join("manifest.json"),
        "{\n \"files\": [\n  {\n   \"rep\": 1,\n   \"proof\": \"rep1/sub_00.proof\",\n   \"python_verdict\": {\n    \"accepted\": true,\n    \"reason\": \"accept (0.03s)\"\n   }\n  }\n ]\n}\n",
    )
    .unwrap();
    let dir = rep.to_string_lossy().to_string();
    let (code, js) = run(&["batch", "--system", FP8_SYS, "--dir", &dir, "--jobs", "1", "--threads", "2"]);
    // the one sub-batch verifies (pinned), but it was sized for a 49-batch: the batch rule (D10) refuses 1 of 49
    assert_eq!(code, 1, "{js}");
    assert_eq!(field(&js, "\"accepted\":"), "1", "{js}");
    assert_eq!(field(&js, "\"batch_accepted\":"), "false", "{js}");
    assert!(js.contains("\"batch_reason\":\"batch: n_proofs mismatch (sub_00.proof: n_proofs = 49, 1 sub-batches presented)\""), "{js}");
    assert_eq!(field(&js, "\"system_pinned\":"), "true", "{js}");
    assert_eq!(field(&js, "\"pinned_relation\":"), "\"fp8-ada\"", "{js}");
    assert_eq!(field(&js, "\"own_coins\":"), "1", "{js}");
    assert_eq!(field(&js, "\"python_agree\":"), "1", "the parent manifest's rep1/ entries are matched: {js}");
    assert!(js.contains("\"sys_id\":\"6b570eef0c7faa07ccd1dddb2508e2699bd5367b108b15920469d46d16073d37\""), "{js}");
    let (code, js) = run(&["batch", "--system", BF16_SYS, "--dir", &dir, "--jobs", "1", "--threads", "1", "--allow-any-system"]);
    assert_eq!(code, 3, "the verdict disagrees with the manifest's python_verdict: {js}");
    assert_eq!(field(&js, "\"system_pinned\":"), "false", "{js}");
    assert_eq!(field(&js, "\"pinned_relation\":"), "\"bf16-ampere\"", "{js}");
    assert!(js.contains("malformed proof: header M = 3769, the statement fixes 3516"), "{js}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn fp8_ada_negatives_are_rejected() {
    let st = std::fs::read(FP8_STMT).unwrap();
    let pf = std::fs::read(FP8_PROOF).unwrap();
    let s = parse(&st);
    assert_eq!((s.l, s.k), (4096, 32));
    let words_at = 58 + 2 + 4 + "fp8-ada".len(); // the first operand byte a[0][0]
    assert_eq!(&st[60..64], &7u32.to_le_bytes());
    // (1) an operand turned into the E4M3 NaN code: rejected by the public decode
    let mut m = st.clone();
    m[words_at] = 0x7F;
    let (code, js) = verify(FP8_SYS, &write_tmp("nan.stmt", &m), FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("public inputs: E4M3 NaN operand"), "{js}");
    // (2) an operand changed to another finite word: the statement digest changes under the transcript's challenges
    let mut m = st.clone();
    m[words_at] = if m[words_at] == 0x38 { 0x30 } else { 0x38 };
    let (code, js) = verify(FP8_SYS, &write_tmp("word.stmt", &m), FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("column challenge mismatch"), "{js}");
    // (3) the chain-end public words: a flipped low byte (digest), a set high bit (not a 22-bit word)
    let y_at = words_at + 2 * s.l * s.k; // the first y word (4 bytes each)
    let mut m = st.clone();
    m[y_at] ^= 1;
    let (code, js) = verify(FP8_SYS, &write_tmp("y_low.stmt", &m), FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("column challenge mismatch"), "{js}");
    let mut m = st.clone();
    m[y_at + 3] |= 0x80;
    let (code, js) = verify(FP8_SYS, &write_tmp("y_high.stmt", &m), FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("is not a 22-bit word"), "{js}");
    // (4) has_auth = 1 on 8-bit words: no leaf schema exists, refused at parse whatever follows
    let mut m = st.clone();
    let last = m.len() - 1;
    assert_eq!(m[last], 0, "the fixture carries no authentication block");
    m[last] = 1;
    m.extend_from_slice(&auth_block(48, 32));
    let (code, js) = verify(FP8_SYS, &write_tmp("auth8.stmt", &m), FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("is defined for 16-bit words only"), "{js}");
    // (5) an unknown relation name, and a known one whose widths are not the file's
    let (code, js) = verify(FP8_SYS, &write_tmp("adb.stmt", &encode(&s, 4, "fp8-adb", 1, 4, None)), FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("statement names relation \\\"fp8-adb\\\"; this build pins bf16-ampere, fp8-ada, bf16-hopper, fp8-hopper, fp4-nvf4, bf16-hopper-v2"), "{js}");
    assert_eq!(field(&js, "\"statement\":"), "null", "no relation, nothing to report: {js}");
    let (code, js) = verify(FP8_SYS, &write_tmp("wide.stmt", &encode(&s, 4, "fp8-ada", 2, 4, None)), FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("word widths 2/4 bytes are not the fp8-ada relation's 1/4"), "{js}");
    let (code, js) = verify(FP8_SYS, &write_tmp("bf16w.stmt", &encode(&s, 4, "bf16", 1, 4, None)), FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("word widths 1/4 bytes are not the bf16-ampere relation's 2/2"), "{js}");
    // (6) the same words named as the fp8-hopper relation: the pin refuses the fp8-ada system; unpinned, the
    // relation tag in the digest still separates the two relations (no proof of one verifies as the other)
    let hop = write_tmp("hopper.stmt", &encode(&s, 4, "fp8-hopper", 1, 4, None));
    let (code, js) = verify(FP8_SYS, &hop, FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned fp8-ada system, the statement is a fp8-hopper statement"), "{js}");
    let (code, js) = verify(FP8_SYS, &hop, FP8_PROOF, &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("column challenge mismatch"), "{js}");
    // (7) the proof: a flipped byte in the opened columns, a truncated transcript, a wrong coins file
    let mut m = pf.clone();
    let off = pf.len() - 198 * 14 * 32 - 1000; // inside `opened` (t = 198, depth 14)
    m[off] ^= 1;
    let (code, js) = verify(FP8_SYS, FP8_STMT, &write_tmp("opened.proof", &m), &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("merkle path"), "{js}");
    let (code, js) = verify(FP8_SYS, FP8_STMT, &write_tmp("short.proof", &pf[..pf.len() - 1]), &[]);
    assert_eq!(code, 1, "{js}");
    let mut coins = std::fs::read(FP8_COINS).unwrap();
    coins[3] ^= 1;
    let (code, js) = verify(FP8_SYS, FP8_STMT, FP8_PROOF, &["--coins", &write_tmp("theirs.coins", &coins)]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("coins are not this verifier's step-0 coins"), "{js}");
}

// ------------------------------------------------------------------------------------------------------ fp4-nvf4 ----

/// The byte offset of `a[0][0]` in the fp4 statement: the 58-byte header, the two widths, the relation name.
const FP4_WORDS_AT: usize = 58 + 2 + 4 + "fp4-nvf4".len();

#[test]
fn fp4_nvf4_v4_honest_proof_is_accepted_against_its_pinned_system() {
    let (code, js) = verify(FP4_SYS, FP4_STMT, FP4_PROOF, &["--coins", FP4_COINS]);
    assert_eq!(code, 0, "{js}");
    assert_eq!(field(&js, "\"accepted\":"), "true");
    assert_eq!(field(&js, "\"system_pinned\":"), "true");
    assert!(js.contains("\"format\":\"ligero-statement/v4\",\"relation\":\"fp4-nvf4\""), "{js}");
    assert!(js.contains("\"l\":4096,\"n\":16384,\"D\":6,\"t\":196,\"zk\":false,\"mode\":\"interactive\",\"n_proofs\":25,\"n_vus\":170,\"steps\":24,\"K\":68"), "{js}");
    assert!(js.contains("coins are this verifier's step-0 coins"), "{js}");
    assert!(!js.contains("NOT pinned"), "{js}");
    // the statement digest (tag `|rel=fp4-nvf4|v1`, 1-byte operand words, 4-byte accumulator words, the sys_id) and
    // the union bound the statement's parameters give over its 25 sub-batches: below 2^-128
    assert!(js.contains("\"digest\":\"500bfdb2b69bce82e730a2387188a120c08ce4217ada8c7522180b810a998fad\""), "{js}");
    assert!(js.contains("\"per_proof_log2\":-132.874,\"union_log2\":-128.231"), "{js}");
    // the same proof without --coins: the conditional accept, as for the BF16 and FP8 fixtures
    let (code, js) = verify(FP4_SYS, FP4_STMT, FP4_PROOF, &[]);
    assert_eq!(code, 0, "{js}");
    assert!(js.contains("coins replayed from the transcript"), "{js}");
    // unpinned it verifies too, flagged
    let (code, js) = verify(FP4_SYS, FP4_STMT, FP4_PROOF, &["--allow-any-system"]);
    assert_eq!(code, 0, "{js}");
    assert_eq!(field(&js, "\"accepted\":"), "true", "{js}");
    assert_eq!(field(&js, "\"system_pinned\":"), "false", "{js}");
}

#[test]
fn fp4_nvf4_statement_negatives_are_rejected() {
    let st = std::fs::read(FP4_STMT).unwrap();
    let s = parse(&st);
    assert_eq!((s.l, s.k), (4096, 68));
    assert_eq!((st[58], st[59]), (1, 4), "1-byte operand words, 4-byte accumulator words");
    assert_eq!(&st[60..64], &8u32.to_le_bytes());
    let (code, honest) = verify(FP4_SYS, FP4_STMT, FP4_PROOF, &[]);
    assert_eq!(code, 0, "{honest}");
    let digest = field(&honest, "\"digest\":").to_string();
    let rejected = |name: &str, bytes: &[u8], reason: &str| -> String {
        let (code, js) = verify(FP4_SYS, &write_tmp(name, bytes), FP4_PROOF, &[]);
        assert_eq!(code, 1, "{name}: {js}");
        assert!(js.contains(reason), "{name}: {js}");
        js
    };
    // (1) a public E2M1 nibble flipped to another code (unit 0, a[0]: 0x0b -> 0x0a): every operand byte is absorbed
    // by the statement digest, so the transcript's column challenge no longer derives
    assert_eq!(st[FP4_WORDS_AT], 0x0b);
    let mut m = st.clone();
    m[FP4_WORDS_AT] ^= 1;
    let js = rejected("nibble.stmt", &m, "column challenge mismatch");
    assert_ne!(field(&js, "\"digest\":"), digest, "the flipped nibble changed the statement digest");
    // a code byte with its high nibble set is not an E2M1 code: off the domain at the public decode, before hashing
    let mut m = st.clone();
    m[FP4_WORDS_AT] |= 0x10;
    rejected("nibble_hi.stmt", &m, "public inputs: E2M1 code outside 4 bits (unit 0)");
    // (2) the UE4M3 scale bytes (a[0]'s group-0 scale is 0x31): another finite scale (digest), the NaN code 0x7F, the
    // padding bit (bit 7) -- the last two off the domain at the public decode, in either operand
    let sc_at = FP4_WORDS_AT + 64;
    assert_eq!(st[sc_at], 0x31);
    let mut m = st.clone();
    m[sc_at] ^= 0x08;
    let js = rejected("scale.stmt", &m, "column challenge mismatch");
    assert_ne!(field(&js, "\"digest\":"), digest);
    let mut m = st.clone();
    m[sc_at] = 0x7F;
    rejected("scale_nan.stmt", &m, "public inputs: UE4M3 NaN scale (unit 0)");
    let mut m = st.clone();
    m[sc_at] |= 0x80;
    rejected("scale_pad.stmt", &m, "public inputs: UE4M3 scale byte with the padding bit set (unit 0)");
    let b_sc_at = FP4_WORDS_AT + s.l * s.k + s.k + 64 + 3; // b[1]'s group-3 scale
    let mut m = st.clone();
    m[b_sc_at] = 0xFF;
    rejected("scale_pad_b.stmt", &m, "public inputs: UE4M3 scale byte with the padding bit set (unit 1)");
    let mut m = st.clone();
    m[b_sc_at] = 0x7F;
    rejected("scale_nan_b.stmt", &m, "public inputs: UE4M3 NaN scale (unit 1)");
    // (3) the FP32 accumulator word (unit 0's is +0.0): its sign, exponent-field and fraction components are each
    // absorbed by the digest -- every 32-bit pattern is a word, so this is the only line (`y_bits` = 32)
    let y_at = FP4_WORDS_AT + 2 * s.l * s.k;
    assert_eq!(&st[y_at..y_at + 4], &[0, 0, 0, 0]);
    for (name, byte, mask) in [("y_sign", 3, 0x80u8), ("y_exp", 3, 0x01), ("y_exp_lo", 2, 0x80), ("y_frac", 0, 0x01)] {
        let mut m = st.clone();
        m[y_at + byte] ^= mask;
        let js = rejected(&format!("{name}.stmt"), &m, "column challenge mismatch");
        assert_ne!(field(&js, "\"digest\":"), digest, "{name}");
    }
    // (4) a truncated statement, the widths of another relation under this name, a K that is not the unit's
    rejected("trunc.stmt", &st[..st.len() - 1], "statement file: truncated file");
    rejected("wide.stmt", &encode(&s, 4, "fp4-nvf4", 2, 4, None), "word widths 2/4 bytes are not the fp4-nvf4 relation's 1/4");
    rejected("y2.stmt", &encode(&s, 4, "fp4-nvf4", 1, 2, None), "word widths 1/2 bytes are not the fp4-nvf4 relation's 1/4");
    let (code, js) = verify(FP4_SYS, &write_tmp("blackwell.stmt", &encode(&s, 4, "fp4-blackwell", 1, 4, None)), FP4_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("statement names relation \\\"fp4-blackwell\\\"; this build pins bf16-ampere, fp8-ada, bf16-hopper, fp8-hopper, fp4-nvf4, bf16-hopper-v2, bf16-ampere-v2, fp8-hopper-v2, fp8-ada-v2, bf16-hopper-v3, bf16-ampere-v3, fp8-hopper-v3, fp8-ada-v3, bf16-hopper-v2x4, bf16-ampere-v2x4, fp8-hopper-v2x4, fp8-ada-v2x4"), "{js}");
    // the FP8 Ada words (K = 32, 1-byte words) named fp4-nvf4 with t and n_proofs patched to the fp4 fixture's, so the
    // proof header and the soundness gate pass: refused at the public decode for the K
    let fp8 = std::fs::read(FP8_STMT).unwrap();
    let s8 = parse(&fp8);
    let mut k32 = encode(&s8, 4, "fp4-nvf4", 1, 4, None);
    assert_eq!((u32_at(&k32, 24), u32_at(&k32, 42), u32_at(&k32, 54)), (198, 49, 32));
    k32[24..28].copy_from_slice(&196u32.to_le_bytes());
    k32[42..46].copy_from_slice(&25u32.to_le_bytes());
    rejected("k32.stmt", &k32, "public inputs: statement has K = 32 words per operand per unit, the fp4-nvf4 unit takes 68");
}

#[test]
fn fp4_nvf4_proof_and_coins_negatives_are_rejected() {
    let pf = std::fs::read(FP4_PROOF).unwrap();
    // the v1 transcript: 8 magic + 4 flag bytes + 8 u32 header, the six 32-byte coin slots, the root; then w (D x l),
    // h (D x (l - 1)), q (D x (2l - 1)), the t column indices, the opened columns (M x t, row-major), the t x depth
    // Merkle path nodes -- every size fixed by the statement (l = 4096, D = 6, t = 196, depth 14) and the system (M = 1583)
    let (d, l, t, m_rows, depth) = (6usize, 4096usize, 196usize, 1583usize, 14usize);
    let w_at = 8 + 4 + 32 + 6 * 32 + 32;
    let h_at = w_at + 4 * d * l;
    let q_at = h_at + 4 * d * (l - 1);
    let cols_at = q_at + 4 * d * (2 * l - 1);
    let opened_at = cols_at + 4 * t;
    let paths_at = opened_at + 4 * m_rows * t;
    assert_eq!(paths_at + 32 * t * depth, pf.len());
    let rejected = |name: &str, bytes: &[u8], extra: &[&str], reason: &str| {
        let (code, js) = verify(FP4_SYS, FP4_STMT, &write_tmp(name, bytes), extra);
        assert_eq!(code, 1, "{name}: {js}");
        assert!(js.contains(reason), "{name}: {js}");
    };
    // a flipped byte per region, each caught by its own test
    for (name, at, reason) in [
        ("w", w_at, "proximity test failed"),
        ("h", h_at + 4, "quadratic constraints failed"),
        ("q", q_at + 8, "chain constraints failed (opened columns)"),
        ("cols", cols_at, "column challenge mismatch"),
        ("opened", opened_at + 4 * t * 5, "merkle path 0 invalid"),
        ("root", w_at - 1, "merkle path 0 invalid"),
        ("path", pf.len() - 1, "merkle path 195 invalid"),
        ("coin_c1", 8 + 4 + 32, "coin slot 1: opening does not match its commitment"),
    ] {
        let mut m = pf.clone();
        m[at] ^= 1;
        rejected(&format!("{name}.proof"), &m, &[], reason);
    }
    // a truncated transcript; a transcript whose header is another statement's
    rejected("short.proof", &pf[..pf.len() - 1], &[], "proof file: truncated file");
    let (code, js) = verify(FP4_SYS, FP4_STMT, FP8_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("malformed proof: header M = 3769, the statement fixes 1583"), "{js}");
    // the interactive gate: coins that are not this verifier's
    let mut coins = std::fs::read(FP4_COINS).unwrap();
    coins[3] ^= 1;
    rejected("own.proof", &pf, &["--coins", &write_tmp("theirs.coins", &coins)], "coins are not this verifier's step-0 coins");
}

#[test]
fn fp4_nvf4_system_negatives_are_rejected() {
    // (1) the wrong pinned system, both ways round: refused by the pin, before any parsing of the proof
    for (sys, stmt, proof, reason) in [
        (BF16_SYS, FP4_STMT, FP4_PROOF, "system file is the pinned bf16-ampere system, the statement is a fp4-nvf4 statement"),
        (FP8_SYS, FP4_STMT, FP4_PROOF, "system file is the pinned fp8-ada system, the statement is a fp4-nvf4 statement"),
        (FP4_SYS, "fixtures/b1.stmt", "fixtures/b1.proof", "system file is the pinned fp4-nvf4 system, the statement is a bf16-ampere statement"),
        (FP4_SYS, FP8_STMT, FP8_PROOF, "system file is the pinned fp4-nvf4 system, the statement is a fp8-ada statement"),
    ] {
        let (code, js) = verify(sys, stmt, proof, &[]);
        assert_eq!(code, 1, "{js}");
        assert!(js.contains(reason), "{js}");
        assert_eq!(field(&js, "\"system_pinned\":"), "true", "{js}");
    }
    // under --allow-any-system the digest pin is lifted but the chain end of the FILE must still be the relation's:
    // a v1 word end cannot stand in for the fp4 components, nor the components for a word
    let (code, js) = verify(BF16_SYS, FP4_STMT, FP4_PROOF, &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file: system chain ends on one public word (ligero-system/v1); the fp4-nvf4 relation ends on 3 components (lo/bits 31/1, 23/8, 0/23)"), "{js}");
    assert_eq!(field(&js, "\"system_pinned\":"), "false", "{js}");
    let (code, js) = verify(FP4_SYS, "fixtures/b1.stmt", "fixtures/b1.proof", &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file: system chain ends on 3 components (lo/bits 31/1, 23/8, 0/23; ligero-system/v2); the bf16-ampere relation ends on one 16-bit public word"), "{js}");
    // (2) a v2 system with a wrong component width: the fraction component's `bits` (the last u32 before the sys_id)
    // 23 -> 22.  The tables' digest changes, so pinned it is refused; unpinned, the components are checked
    let sys = std::fs::read(FP4_SYS).unwrap();
    let n = sys.len();
    assert_eq!(&sys[..8], b"LIGSYS02");
    assert_eq!((u32_at(&sys, n - 40), u32_at(&sys, n - 36)), (0, 23), "the last component is the 23-bit fraction at bit 0");
    let mut m = sys.clone();
    m[n - 36..n - 32].copy_from_slice(&22u32.to_le_bytes());
    let p = write_tmp("bits22.system.bin", &m);
    let (code, js) = run(&["system-digest", "--system", &p]);
    assert_eq!(code, 0, "{js}");
    assert_eq!(field(&js, "\"pinned_relation\":"), "null", "{js}");
    assert!(js.contains("{\"lo\":0,\"bits\":22}"), "{js}");
    let (code, js) = verify(&p, FP4_STMT, FP4_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is not the pinned fp4-nvf4 system"), "{js}");
    let (code, js) = verify(&p, FP4_STMT, FP4_PROOF, &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file: system chain end components (lo/bits 31/1, 23/8, 0/22) are not the fp4-nvf4 relation's (31/1, 23/8, 0/23)"), "{js}");
    // a wrong `lo` likewise
    let mut m = sys.clone();
    m[n - 40..n - 36].copy_from_slice(&1u32.to_le_bytes());
    let (code, js) = verify(&write_tmp("lo1.system.bin", &m), FP4_STMT, FP4_PROOF, &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file: system chain end components (lo/bits 31/1, 23/8, 1/23) are not the fp4-nvf4 relation's (31/1, 23/8, 0/23)"), "{js}");
    // (3) component fields that are not bit fields of a 32-bit word below p: refused at parse, pinned or not
    for (name, at, val, reason) in [
        ("bits31", n - 36, 31u32, "chain end component 2: bit field lo = 0, bits = 31 is not a field of a 32-bit word below p"),
        ("bits0", n - 36, 0, "chain end component 2: bit field lo = 0, bits = 0 is not a field of a 32-bit word below p"),
        ("lo10", n - 40, 10, "chain end component 2: bit field lo = 10, bits = 23 is not a field of a 32-bit word below p"),
    ] {
        let mut m = sys.clone();
        m[at..at + 4].copy_from_slice(&val.to_le_bytes());
        let p = write_tmp(&format!("{name}.system.bin"), &m);
        let (code, js) = run(&["system-digest", "--system", &p]);
        assert_eq!(code, 2, "{name}: {js}");
        for extra in [&[][..], &["--allow-any-system"][..]] {
            let (code, js) = verify(&p, FP4_STMT, FP4_PROOF, extra);
            assert_eq!(code, 1, "{name}: {js}");
            assert!(js.contains(&format!("system file: {reason}")), "{name}: {js}");
        }
    }
    // (4) a truncated system file, a v1 body under the v2 magic and a v2 body under the v1 magic
    let (code, js) = verify(&write_tmp("short.system.bin", &sys[..n - 40]), FP4_STMT, FP4_PROOF, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file: truncated file"), "{js}");
    let mut v1 = std::fs::read(BF16_SYS).unwrap();
    v1[..8].copy_from_slice(b"LIGSYS02");
    let (code, js) = verify(&write_tmp("v1_as_v2.system.bin", &v1), "fixtures/b1.stmt", "fixtures/b1.proof", &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.starts_with("{\"name\":\"fixtures/b1.proof\",\"accepted\":false") && js.contains("\"reason\":\"system file: "), "{js}");
    let mut v2 = sys.clone();
    v2[..8].copy_from_slice(b"LIGSYS01");
    let (code, js) = verify(&write_tmp("v2_as_v1.system.bin", &v2), FP4_STMT, FP4_PROOF, &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("\"reason\":\"system file: "), "{js}");
}

#[test]
fn v4_named_bf16_statement_is_the_v2_statement_and_the_hopper_name_is_pinned_apart() {
    let st = std::fs::read("fixtures/b1.stmt").unwrap();
    assert_eq!(&st[..8], b"LIGSTM02");
    let s = parse(&st);
    assert_eq!((s.l, s.k), (256, 16));
    // the round trip through the re-encoder is the identity on the v2 file
    assert_eq!(encode(&s, 2, "", 2, 2, None), st);
    let (code, v2) = verify(BF16_SYS, "fixtures/b1.stmt", "fixtures/b1.proof", &[]);
    assert_eq!(code, 0, "{v2}");
    let digest = field(&v2, "\"digest\":").to_string();
    // v4 named "bf16" / "bf16-ampere", 2/2-byte words, no auth block: the same statement, the same digest
    for name in ["bf16", "bf16-ampere"] {
        let p = write_tmp(&format!("{name}.stmt"), &encode(&s, 4, name, 2, 2, None));
        let (code, js) = verify(BF16_SYS, &p, "fixtures/b1.proof", &[]);
        assert_eq!(code, 0, "{js}");
        assert!(js.contains("\"format\":\"ligero-statement/v4\",\"relation\":\"bf16-ampere\""), "{js}");
        assert_eq!(field(&js, "\"digest\":"), digest, "{js}");
        assert_eq!(field(&js, "\"system_pinned\":"), "true");
    }
    // named "bf16-hopper": the words, tag and widths are the Ampere unit's, so ONLY the pin tells the relations apart
    // -- pinned it is refused; under --allow-any-system it verifies against the Ampere system, flagged unpinned
    let p = write_tmp("bf16-hopper.stmt", &encode(&s, 4, "bf16-hopper", 2, 2, None));
    let (code, js) = verify(BF16_SYS, &p, "fixtures/b1.proof", &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned bf16-ampere system, the statement is a bf16-hopper statement"), "{js}");
    let (code, js) = verify(BF16_SYS, &p, "fixtures/b1.proof", &["--allow-any-system"]);
    assert_eq!(code, 0, "{js}");
    assert_eq!(field(&js, "\"system_pinned\":"), "false", "{js}");
    assert!(js.contains("\"relation\":\"bf16-hopper\""), "{js}");
    // and against the real bf16-hopper system the pin passes but the proof is the other system's (3292 vs 3516 rows)
    let (code, js) = verify("fixtures/systems/bf16-hopper.system.bin", &p, "fixtures/b1.proof", &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("malformed proof: header M = 3552, the statement fixes 3328"), "{js}");
    assert_eq!(field(&js, "\"system_pinned\":"), "true", "{js}");
    // v3 and v4-with-auth statements parse (the block is well formed: one VU at index 0, K = 96 x 16 words) and are
    // refused for the v1 proof's lack of multiproofs -- the version pairing of the authentication extension, unchanged
    let steps = u32_at(&st, 50);
    assert_eq!((steps, u32_at(&st, 46)), (96, 1));
    let ab = auth_block(steps, 16);
    for (name, bytes) in [("v3", encode(&s, 3, "", 2, 2, Some(&ab))), ("v4auth", encode(&s, 4, "bf16", 2, 2, Some(&ab)))] {
        let (code, js) = verify(BF16_SYS, &write_tmp(&format!("{name}.stmt"), &bytes), "fixtures/b1.proof", &[]);
        assert_eq!(code, 1, "{js}");
        assert!(js.contains("statement has an authentication block but the transcript carries no multiproofs"), "{js}");
    }
    // malformed v4 headers: a width that is not 1 / 2 / 4, an empty relation name, a has_auth flag that is not 0 / 1
    let mut bad = encode(&s, 4, "bf16", 2, 2, None);
    bad[58] = 3;
    let (code, js) = verify(BF16_SYS, &write_tmp("w3.stmt", &bad), "fixtures/b1.proof", &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("word widths 3/2 bytes are not 1, 2 or 4"), "{js}");
    let (code, js) = verify(BF16_SYS, &write_tmp("noname.stmt", &encode(&s, 4, "", 2, 2, None)), "fixtures/b1.proof", &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("v4 statement names no relation"), "{js}");
    let mut bad = encode(&s, 4, "bf16", 2, 2, None);
    let last = bad.len() - 1;
    bad[last] = 2;
    let (code, js) = verify(BF16_SYS, &write_tmp("has2.stmt", &bad), "fixtures/b1.proof", &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("has_auth flag not 0/1"), "{js}");
}

// ------------------------------------------------------------------- lane hash-relation: authentication = included-hash ----
//
// `fixtures/fp8-ada-hash/`: a laptop (CPU) dump of the HASHED fp8-ada relation (`hashchain.compose`: operands private, two
// Poseidon2-BabyBear width-24 sponges along the chain, the 16 digest lanes pinned at every VU's last column), lane
// hash-relation commit 6516efc: interactive non-ZK, 20 VUs x 48 steps at l = 1024, one sub-batch, statement
// `ligero-statement/v5` (6217 bytes: roots + indices + digests + y words + three multiproofs, no operand words), system
// `ligero-system/v3` (19 linked rows).  Python's verifier accepted it (manifest `verify_python.json`, 1/1).

const HASH_SYS: &str = "fixtures/fp8-ada-hash/system.bin";
const HASH_STMT: &str = "fixtures/fp8-ada-hash/sub_00.stmt";
const HASH_PROOF: &str = "fixtures/fp8-ada-hash/sub_00.proof";
const HASH_COINS: &str = "fixtures/fp8-ada-hash/sub_00.coins";
const HASH_SYS_ID: &str = "c4c4b403e07da47f7747b947ad76899ef231112bcc6d62dc0448a83ef8696618";
const HASH_TABLE_DIGEST: &str = "c0e2f48119e2c83ca81cbc8eb1cbba51a676b67a600b8125d75bf438b770e05d";

/// Byte offsets of the v5 fixture's fields (`serialize._statement_bytes_v5`; asserted against the header below).
struct V5 {
    digests: usize,   // n_vus x 16 canonical field elements, 4 bytes each
    y16: usize,       // l words of y_bytes
    indices: usize,   // n_vus x (vu, x, w) u32 triples
    trees: usize,     // 3 x (binding 32, owner 1, count u64, root 32)
    multiproofs: usize,
}

fn v5_offsets(st: &[u8]) -> V5 {
    assert_eq!(&st[..8], b"LIGSTM05");
    let (l, n_vus, steps, k) = (u32_at(st, 12), u32_at(st, 46), u32_at(st, 50), u32_at(st, 54));
    assert_eq!((l, n_vus, steps, k), (1024, 20, 48, 1536));
    let yb = st[58] as usize;
    assert_eq!(yb, 4);
    let mut o = 59;
    let s = |o: &mut usize| { let n = u32_at(st, *o); let v = st[*o + 4..*o + 4 + n].to_vec(); *o += 4 + n; v };
    assert_eq!(s(&mut o), b"fp8-ada");
    assert_eq!(s(&mut o), b"included-hash");
    let digests = o;
    o += 4 * 16 * n_vus;
    let y16 = o;
    o += l * yb;
    assert_eq!(s(&mut o), b"ligero-b/auth/v2h");
    assert_eq!(s(&mut o), b"poseidon2-babybear-w24/row/v2h");
    o += 32; // the Poseidon2 constants digest
    assert_eq!(u32_at(st, o), n_vus);
    o += 4;
    let indices = o;
    o += 12 * n_vus;
    let trees = o;
    o += 3 * 73;
    V5 { digests, y16, indices, trees, multiproofs: o }
}

#[test]
fn fp8_ada_hashed_v5_proof_is_accepted_and_the_hashed_system_is_pinned() {
    let (code, js) = verify(HASH_SYS, HASH_STMT, HASH_PROOF, &["--coins", HASH_COINS]);
    assert_eq!(code, 0, "{js}");
    assert_eq!(field(&js, "\"accepted\":"), "true");
    assert_eq!(field(&js, "\"system_pinned\":"), "true");
    assert!(js.contains("\"format\":\"ligero-statement/v5\",\"relation\":\"fp8-ada\""), "{js}");
    assert!(js.contains("\"l\":1024,\"n\":4096,\"D\":6,\"t\":189,\"zk\":false,\"mode\":\"interactive\",\"n_proofs\":1,\"n_vus\":20,\"steps\":48,\"K\":32"), "{js}");
    assert!(js.contains("authentication: included-hash, 20 VUs, 20 x rows + 20 W columns + 20 y words under the a/b/y roots"), "{js}");
    assert!(js.contains("coins are this verifier's step-0 coins"), "{js}");
    let (_, sd) = run(&["system-digest", "--system", HASH_SYS]);
    assert!(sd.contains(&format!("\"sys_id\":\"{HASH_SYS_ID}\"")), "{sd}");
    assert!(sd.contains(&format!("\"table_digest\":\"{HASH_TABLE_DIGEST}\"")), "{sd}");
    assert!(sd.contains("\"pinned_relation\":\"fp8-ada+hash\"") && sd.contains("\"format\":\"ligero-system/v3\""), "{sd}");
    // the batch command over the fixture directory reports the hashed pin and the authentication
    let (code, js) = run(&["batch", "--dir", "fixtures/fp8-ada-hash", "--system", HASH_SYS, "--threads", "2"]);
    assert_eq!(code, 0, "{js}");
    assert!(js.contains("\"pinned_relation\":\"fp8-ada+hash\"") && js.contains("\"accepted\":1,\"rejected\":0"), "{js}");
}

#[test]
fn fp8_ada_hashed_proof_mutations_are_rejected_by_the_chain_and_quadratic_tests() {
    // The hashed system's chain test has 19 links (accumulator 3 + 16 sponge capacity lanes); a flipped byte in the
    // q rows must still fail it, and one in h the quadratic test (layout as in the fp4 proof test: l = 1024, D = 6,
    // t = 189, M = 6210, depth 12, non-ZK interactive).
    let pf = std::fs::read(HASH_PROOF).unwrap();
    let (d, l, t, m_rows, depth) = (6usize, 1024usize, 189usize, 6210usize, 12usize);
    let w_at = 8 + 4 + 32 + 6 * 32 + 32;
    let h_at = w_at + 4 * d * l;
    let q_at = h_at + 4 * d * (l - 1);
    let cols_at = q_at + 4 * d * (2 * l - 1);
    let opened_at = cols_at + 4 * t;
    let paths_at = opened_at + 4 * m_rows * t;
    assert_eq!(paths_at + 32 * t * depth, pf.len());
    for (name, at, reason) in [
        ("w", w_at, "proximity test failed"),
        ("h", h_at + 4, "quadratic constraints failed"),
        ("q", q_at + 8, "chain constraints failed (opened columns)"),
        ("q_last", q_at + 4 * d * (2 * l - 1) - 4, "chain constraints failed (opened columns)"),
        ("opened_cap", opened_at + 4 * t * 6000, "merkle path 0 invalid"),
    ] {
        let mut m = pf.clone();
        m[at] ^= 1;
        let (code, js) = verify(HASH_SYS, HASH_STMT, &write_tmp(&format!("hash_{name}.proof"), &m), &["--coins", HASH_COINS]);
        assert_eq!(code, 1, "{name}: {js}");
        assert!(js.contains(reason), "{name}: {js}");
    }
}

#[test]
fn fp8_ada_hashed_statement_negatives_are_rejected() {
    // Every field of the v5 statement is bound by the statement digest (the relation tag absorbs the authentication
    // block, the body the digests and the y words), so a statement FILE mutated after the fact fails on the transcript's
    // challenges before the path check runs -- "column challenge mismatch" -- exactly as a mutated v4 statement does.
    // The path check itself is exercised on the parsed block in `auth::tests::hashed_fixture_block_folds_and_every_
    // mutation_is_refused` (wrong digest, wrong path, other root, swapped W column, foreign x row, y word, ranges).
    let st = std::fs::read(HASH_STMT).unwrap();
    let f = v5_offsets(&st);
    let rej = |name: &str, m: &[u8], expect: &str| {
        let (code, js) = verify(HASH_SYS, &write_tmp(name, m), HASH_PROOF, &["--coins", HASH_COINS]);
        assert_eq!(code, 1, "{name}: {js}");
        assert!(js.contains(expect), "{name}: expected {expect:?} in {js}");
    };
    let bound = "column challenge mismatch";
    // (1) a wrong digest exposed: lane 0 of VU 0's x digest; (2) VU 3's W digest (lanes 8..16 are the b sponge's)
    let mut m = st.clone();
    m[f.digests] ^= 1;
    rej("digest.stmt", &m, bound);
    let mut m = st.clone();
    m[f.digests + 4 * (16 * 3 + 8)] ^= 1;
    rej("wdigest.stmt", &m, bound);
    // (3) a wrong path: the multiproof siblings are auxiliary (not absorbed by the digest -- the roots and ranks are),
    // so the Ligero tests pass and the path check itself refuses the statement; (4) a path to a different x root;
    // (5) another y root
    let mut m = st.clone();
    m[f.multiproofs + 4 + 5] ^= 1;
    rej("path.stmt", &m, "auth: a: multiproof rejected (root mismatch)");
    let mut m = st.clone();
    m[f.trees + 41] ^= 1;
    rej("root.stmt", &m, bound);
    let mut m = st.clone();
    m[f.trees + 2 * 73 + 41] ^= 1;
    rej("yroot.stmt", &m, bound);
    // (6) a swapped W row: VU 0 claims W column 1; (7) an x row of another VU; (8) a y word
    let mut m = st.clone();
    m[f.indices + 8..f.indices + 12].copy_from_slice(&1u32.to_le_bytes());
    rej("wswap.stmt", &m, bound);
    let mut m = st.clone();
    m[f.indices + 4..f.indices + 8].copy_from_slice(&5u32.to_le_bytes());
    rej("xswap.stmt", &m, bound);
    let mut m = st.clone();
    m[f.y16 + 4 * 47] ^= 1;
    rej("y.stmt", &m, bound);
    // (9) the ids-tree binding; (10) another Poseidon2 parameter set and (11) an index beyond the committed count and
    // (12) a non-canonical digest lane are refused by the parser, before any challenge is derived
    let mut m = st.clone();
    m[f.trees] ^= 1;
    rej("binding.stmt", &m, bound);
    let mut m = st.clone();
    m[f.indices - 4 - 32] ^= 1;
    rej("p2.stmt", &m, "Poseidon2 parameters differ from the verifier's");
    let mut m = st.clone();
    m[f.indices + 4..f.indices + 8].copy_from_slice(&20u32.to_le_bytes());
    rej("xrange.stmt", &m, "claims leaves beyond the committed count");
    let mut m = st.clone();
    m[f.digests..f.digests + 4].copy_from_slice(&0x7800_0001u32.to_le_bytes());
    rej("noncanon.stmt", &m, "non-canonical field element");
    // (13) the authentication name: a v5 statement is included-hash by definition
    let mut m = st.clone();
    m[f.digests - 1] ^= 1;
    rej("authname.stmt", &m, "names an authentication other than included-hash");
}

#[test]
fn fp8_ada_hashed_system_and_version_mismatches_are_rejected() {
    // the hashed statement against the BARE fp8-ada system (both are "fp8-ada"): the pin refuses it; unpinned, the
    // proof carries the hashed system's 6210 rows where the bare system fixes 3769
    let (code, js) = verify(FP8_SYS, HASH_STMT, HASH_PROOF, &["--coins", HASH_COINS]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned fp8-ada system, the statement is a hashed fp8-ada statement"), "{js}");
    let (code, js) = verify(FP8_SYS, HASH_STMT, HASH_PROOF, &["--coins", HASH_COINS, "--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("malformed proof: header M = 6210, the statement fixes 3769"), "{js}");
    // the bare v4 statement against the hashed system: the pin refuses it (and unpinned the row count does)
    let (code, js) = verify(HASH_SYS, FP8_STMT, FP8_PROOF, &["--coins", FP8_COINS]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned hashed fp8-ada system, the statement is a fp8-ada statement"), "{js}");
    let (code, js) = verify(HASH_SYS, FP8_STMT, FP8_PROOF, &["--coins", FP8_COINS, "--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("malformed proof: header M = 3769, the statement fixes 6210"), "{js}");
    // the two systems' digests are pinned apart
    let (_, bare) = run(&["system-digest", "--system", FP8_SYS]);
    assert!(bare.contains("\"pinned_relation\":\"fp8-ada\"") && !bare.contains(HASH_SYS_ID), "{bare}");
    // a proof byte flipped: the hashed relation's tests catch it like any other
    let pf = std::fs::read(HASH_PROOF).unwrap();
    let mut m = pf.clone();
    let mid = m.len() / 2;
    m[mid] ^= 1;
    let (code, js) = verify(HASH_SYS, HASH_STMT, &write_tmp("hproof.proof", &m), &["--coins", HASH_COINS]);
    assert_eq!(code, 1, "{js}");
    // the hashed proof without --coins: the coins are replayed from the transcript (the conditional accept)
    let (code, js) = verify(HASH_SYS, HASH_STMT, HASH_PROOF, &[]);
    assert_eq!(code, 0, "{js}");
    assert!(js.contains("coins replayed from the transcript"), "{js}");
}

// ------------------------------------------------------------ lane hash-compose: the hashed bf16-ampere relation (v5) ----
//
// `fixtures/bf16-ampere-hash/`: a pod CPU dump (vy-hash-compose, RTX 4090 host, `--device cpu`) of the HASHED Ampere BF16
// relation -- `hashchain.compose(relations.relation("bf16-ampere"))`: vu.py's own `compile_unit(REAL)` unit with private
// operands and the two Poseidon2 sponges, 5861 rows, 19 linked rows -- on the FROZEN `bench-instances/v1` `vu-k1536` set
// (VUs 0..10): interactive non-ZK, 10 VUs x 96 steps at l = 1024, one sub-batch, statement `ligero-statement/v5` (3413
// bytes: 16-bit y words), system `ligero-system/v3`.  Python's verifier accepted it (`verify_python.json`, 1/1).

const AH_SYS: &str = "fixtures/bf16-ampere-hash/system.bin";
const AH_STMT: &str = "fixtures/bf16-ampere-hash/sub_00.stmt";
const AH_PROOF: &str = "fixtures/bf16-ampere-hash/sub_00.proof";
const AH_COINS: &str = "fixtures/bf16-ampere-hash/sub_00.coins";
const AH_SYS_ID: &str = "673461ab446d96869a0ab32d1291667f4af4ba86cdf5fb6479fefaa50dbb8ed4";
const AH_TABLE_DIGEST: &str = "3ba12207d9f221dcb2a471044cabb314cc3e05e4512a27559080f431aa7210c9";

/// The v5 fixture's field offsets for the BF16 widths (y_bytes = 2; `v5_offsets` above asserts the FP8 fixture's shape).
fn ah_offsets(st: &[u8]) -> V5 {
    assert_eq!(&st[..8], b"LIGSTM05");
    let (l, n_vus, steps, k) = (u32_at(st, 12), u32_at(st, 46), u32_at(st, 50), u32_at(st, 54));
    assert_eq!((l, n_vus, steps, k), (1024, 10, 96, 1536));
    let yb = st[58] as usize;
    assert_eq!(yb, 2);
    let mut o = 59;
    let s = |o: &mut usize| { let n = u32_at(st, *o); let v = st[*o + 4..*o + 4 + n].to_vec(); *o += 4 + n; v };
    assert_eq!(s(&mut o), b"bf16-ampere");
    assert_eq!(s(&mut o), b"included-hash");
    let digests = o;
    o += 4 * 16 * n_vus;
    let y16 = o;
    o += l * yb;
    assert_eq!(s(&mut o), b"ligero-b/auth/v2h");
    assert_eq!(s(&mut o), b"poseidon2-babybear-w24/row/v2h");
    o += 32;
    assert_eq!(u32_at(st, o), n_vus);
    o += 4;
    let indices = o;
    o += 12 * n_vus;
    let trees = o;
    o += 3 * 73;
    V5 { digests, y16, indices, trees, multiproofs: o }
}

#[test]
fn bf16_ampere_hashed_v5_proof_is_accepted_and_the_hashed_system_is_pinned() {
    let (code, js) = verify(AH_SYS, AH_STMT, AH_PROOF, &["--coins", AH_COINS]);
    assert_eq!(code, 0, "{js}");
    assert_eq!(field(&js, "\"accepted\":"), "true");
    assert_eq!(field(&js, "\"system_pinned\":"), "true");
    assert!(js.contains("\"format\":\"ligero-statement/v5\",\"relation\":\"bf16-ampere\""), "{js}");
    assert!(js.contains("\"n_vus\":10,\"steps\":96,\"K\":16"), "{js}");
    assert!(js.contains("authentication: included-hash, 10 VUs, 10 x rows + 10 W columns + 10 y words under the a/b/y roots"), "{js}");
    let (_, sd) = run(&["system-digest", "--system", AH_SYS]);
    assert!(sd.contains(&format!("\"sys_id\":\"{AH_SYS_ID}\"")), "{sd}");
    assert!(sd.contains(&format!("\"table_digest\":\"{AH_TABLE_DIGEST}\"")), "{sd}");
    assert!(sd.contains("\"pinned_relation\":\"bf16-ampere+hash\"") && sd.contains("\"format\":\"ligero-system/v3\"") && sd.contains("\"m\":5861"), "{sd}");
    let (code, js) = run(&["batch", "--dir", "fixtures/bf16-ampere-hash", "--system", AH_SYS, "--threads", "2"]);
    assert_eq!(code, 0, "{js}");
    assert!(js.contains("\"pinned_relation\":\"bf16-ampere+hash\"") && js.contains("\"accepted\":1,\"rejected\":0"), "{js}");
    // without --coins: replayed from the transcript (the conditional accept)
    let (code, js) = verify(AH_SYS, AH_STMT, AH_PROOF, &[]);
    assert_eq!(code, 0, "{js}");
    assert!(js.contains("coins replayed from the transcript"), "{js}");
}

#[test]
fn bf16_ampere_hashed_negatives_and_system_mismatches_are_rejected() {
    let st = std::fs::read(AH_STMT).unwrap();
    let f = ah_offsets(&st);
    let rej = |name: &str, m: &[u8], expect: &str| {
        let (code, js) = verify(AH_SYS, &write_tmp(name, m), AH_PROOF, &["--coins", AH_COINS]);
        assert_eq!(code, 1, "{name}: {js}");
        assert!(js.contains(expect), "{name}: expected {expect:?} in {js}");
    };
    let bound = "column challenge mismatch";
    let mut m = st.clone();
    m[f.digests] ^= 1;                                       // VU 0's x digest, lane 0
    rej("ah_digest.stmt", &m, bound);
    let mut m = st.clone();
    m[f.digests + 4 * (16 * 7 + 8)] ^= 1;                     // VU 7's W digest, lane 0
    rej("ah_wdigest.stmt", &m, bound);
    let mut m = st.clone();
    m[f.y16 + 2 * 95] ^= 1;                                  // VU 0's chain-end BF16 word
    rej("ah_y.stmt", &m, bound);
    let mut m = st.clone();
    m[f.indices + 8..f.indices + 12].copy_from_slice(&1u32.to_le_bytes());   // VU 0 claims W column 1
    rej("ah_wswap.stmt", &m, bound);
    let mut m = st.clone();
    m[f.trees + 41] ^= 1;                                    // the x root
    rej("ah_root.stmt", &m, bound);
    let mut m = st.clone();
    m[f.indices + 4..f.indices + 8].copy_from_slice(&10u32.to_le_bytes());
    rej("ah_xrange.stmt", &m, "claims leaves beyond the committed count");
    // the multiproofs of a 10-leaf tree opened at all 10 leaves are empty: truncating the block is a parse error
    let (code, js) = verify(AH_SYS, &write_tmp("ah_trunc.stmt", &st[..f.multiproofs]), AH_PROOF, &["--coins", AH_COINS]);
    assert_eq!(code, 1, "{js}");
    // the hashed statement against the BARE Ampere system (the bf16 / bf16-ampere pin): refused by the pin, and unpinned
    // by the row count (5861 vs the bare 3516)
    let (code, js) = verify(BF16_SYS, AH_STMT, AH_PROOF, &["--coins", AH_COINS]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned bf16-ampere system, the statement is a hashed bf16-ampere statement"), "{js}");
    let (code, js) = verify(BF16_SYS, AH_STMT, AH_PROOF, &["--coins", AH_COINS, "--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("malformed proof: header M = 5861, the statement fixes 3516"), "{js}");
    // the bare v2 Ampere statement against the hashed system
    let (code, js) = verify(AH_SYS, "fixtures/b1.stmt", "fixtures/b1.proof", &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned hashed bf16-ampere system, the statement is a bf16-ampere statement"), "{js}");
    // the hashed bf16-hopper pin is another system: the same-width statement is refused by the pin
    let (_, bare) = run(&["system-digest", "--system", BF16_SYS]);
    assert!(bare.contains("\"pinned_relation\":\"bf16-ampere\"") && !bare.contains(AH_SYS_ID), "{bare}");
    // a proof byte flipped
    let pf = std::fs::read(AH_PROOF).unwrap();
    let mut m = pf.clone();
    let mid = m.len() / 2;
    m[mid] ^= 1;
    let (code, js) = verify(AH_SYS, AH_STMT, &write_tmp("ah_proof.proof", &m), &["--coins", AH_COINS]);
    assert_eq!(code, 1, "{js}");
}

// ------------------------------------------------------------ lane ajtai-leaf: the linear Ring-SIS leaf `+ajtai-n<N>` ----
//
// `fixtures/fp8-ada-ajtai-n64/` and `fixtures/bf16-hopper-ajtai-n128/`: `leaf.fixtures.write_fixture` on the pod
// (vy-ajtai-leaf, CPU) -- `hashchain.compose(rel, "ajtai-n<N>")` with `leaf/ajtai.py`: per chain column
// `acc' = X acc + B s` over F_p[X]/(X^N + 1), `s` the compose's own 256 committed operand bit rows, N linked carry rows per
// operand plus N `out` hint rows with the linear key rows, the two N-element digests pinned at the chain end; 4858 rows
// (fp8-ada; the Poseidon2 composition's 6210) / 4669 (bf16-hopper).  Interactive non-ZK, 2 VUs (ids 1, 3 of 4 synthetic)
// at l = 1024, one sub-batch, statement `ligero-statement/v5` (2 x N digest elements per VU), leaf schema
// `ajtai-babybear-n<N>-b1-r0/row/v1`, little-endian u32 leaf bytes.  Python's verifier accepted both (`manifest.json`).

struct AjtaiFx {
    dir: &'static str,
    rel: &'static str,
    leaf: &'static str,
    schema: &'static str,
    n: usize,
    steps: usize,
    k_words: usize,
    y_bytes: usize,
    rows: usize,
    sys_id: &'static str,
    table_digest: &'static str,
}

const AJTAI_FX: &[AjtaiFx] = &[
    AjtaiFx { dir: "fixtures/fp8-ada-ajtai-n64", rel: "fp8-ada", leaf: "ajtai-n64", schema: "ajtai-babybear-n64-b1-r0/row/v1", n: 64,
              steps: 48, k_words: 1536, y_bytes: 4, rows: 4858,
              sys_id: "18d915323c056495f667a6c4286eece77e25e348aed03b335b583e2db1e346c4", table_digest: "a1f2cb20cb549af68f005bf3efef6112f08bb7a9537dbd876811e46769c5ee50" },
    AjtaiFx { dir: "fixtures/bf16-hopper-ajtai-n128", rel: "bf16-hopper", leaf: "ajtai-n128", schema: "ajtai-babybear-n128-b1-r0/row/v1", n: 128,
              steps: 96, k_words: 1536, y_bytes: 2, rows: 4669,
              sys_id: "29f18689c05de6f355c97aac1e711f9dad7f7f35eda85026482258c85d6a27e5", table_digest: "e4dceccece10d7f55f5aed8cb7e444d6f0a736012b6323ee6fd2cdfe520f6438" },
];

impl AjtaiFx {
    fn sys(&self) -> String { format!("{}/system.bin", self.dir) }
    fn stmt(&self) -> String { format!("{}/sub_00.stmt", self.dir) }
    fn proof(&self) -> String { format!("{}/sub_00.proof", self.dir) }
    fn coins(&self) -> String { format!("{}/sub_00.coins", self.dir) }
    fn relation(&self) -> String { format!("{}+{}", self.rel, self.leaf) }

    /// The v5 field offsets of this fixture's statement (2 VUs at l = 1024; 2 x N digest elements per VU).
    fn offsets(&self, st: &[u8]) -> V5 {
        assert_eq!(&st[..8], b"LIGSTM05");
        let (l, n_vus, steps, k) = (u32_at(st, 12), u32_at(st, 46), u32_at(st, 50), u32_at(st, 54));
        assert_eq!((l, n_vus, steps, k), (1024, 2, self.steps, self.k_words), "{}", self.dir);
        assert_eq!(st[58] as usize, self.y_bytes);
        let mut o = 59;
        let s = |o: &mut usize| { let n = u32_at(st, *o); let v = st[*o + 4..*o + 4 + n].to_vec(); *o += 4 + n; v };
        assert_eq!(s(&mut o), self.relation().as_bytes());
        assert_eq!(s(&mut o), b"included-hash");
        let digests = o;
        o += 4 * 2 * self.n * n_vus;
        let y16 = o;
        o += l * self.y_bytes;
        assert_eq!(s(&mut o), b"ligero-b/auth/v2h");
        assert_eq!(s(&mut o), self.schema.as_bytes());
        o += 32; // the scheme's params digest (descriptor + the SHA-256 of the derived B)
        assert_eq!(u32_at(st, o), n_vus);
        o += 4;
        let indices = o;
        o += 12 * n_vus;
        let trees = o;
        o += 3 * 73;
        V5 { digests, y16, indices, trees, multiproofs: o }
    }
}

#[test]
fn ajtai_v5_proofs_are_accepted_and_both_systems_are_pinned() {
    for f in AJTAI_FX {
        let (code, js) = verify(&f.sys(), &f.stmt(), &f.proof(), &["--coins", &f.coins()]);
        assert_eq!(code, 0, "{}: {js}", f.dir);
        assert_eq!(field(&js, "\"accepted\":"), "true");
        assert_eq!(field(&js, "\"system_pinned\":"), "true");
        assert!(js.contains(&format!("\"format\":\"ligero-statement/v5\",\"relation\":\"{}\"", f.relation())), "{js}");
        assert!(js.contains(&format!("\"n_vus\":2,\"steps\":{},", f.steps)), "{js}");
        assert!(js.contains("authentication: included-hash, 2 VUs, 2 x rows + 2 W columns + 2 y words under the a/b/y roots"), "{js}");
        let (_, sd) = run(&["system-digest", "--system", &f.sys()]);
        assert!(sd.contains(&format!("\"sys_id\":\"{}\"", f.sys_id)), "{}: {sd}", f.dir);
        assert!(sd.contains(&format!("\"table_digest\":\"{}\"", f.table_digest)), "{sd}");
        assert!(sd.contains(&format!("\"pinned_relation\":\"{}\"", f.relation())) && sd.contains(&format!("\"m\":{},", f.rows)), "{sd}");
        assert!(sd.contains(&format!("\"pins\":{},", 1 + 2 * f.n)) && sd.contains("\"chain\":true"), "{sd}");
        let (code, js) = run(&["batch", "--dir", f.dir, "--system", &f.sys(), "--threads", "2"]);
        assert_eq!(code, 0, "{js}");
        assert!(js.contains(&format!("\"pinned_relation\":\"{}\"", f.relation())) && js.contains("\"accepted\":1,\"rejected\":0"), "{js}");
        // the manifest's pins row names the same digests (leaf.fixtures prints it for leaf.rs::PINS)
        let man = std::fs::read_to_string(format!("{}/manifest.json", f.dir)).unwrap();
        assert!(man.contains(&format!("(\\\"{}\\\", \\\"{}\\\", \\\"{}\\\", \\\"{}\\\")", f.rel, f.leaf, f.sys_id, f.table_digest)), "{man}");
    }
}

#[test]
fn ajtai_statement_negatives_are_rejected() {
    for f in AJTAI_FX {
        let st = std::fs::read(f.stmt()).unwrap();
        let o = f.offsets(&st);
        let rej = |name: &str, bytes: &[u8], why: &str| {
            let (code, js) = verify(&f.sys(), &write_tmp(&format!("{}_{name}.stmt", f.leaf), bytes), &f.proof(), &["--coins", &f.coins()]);
            assert_eq!(code, 1, "{}/{name}: {js}", f.dir);
            assert!(js.contains(why), "{}/{name}: expected {why:?} in {js}", f.dir);
        };
        // a digest lane of VU 0's x digest changed: the statement digest (and so the challenges) change
        let mut m = st.clone();
        m[o.digests] ^= 1;
        rej("digest", &m, "column challenge mismatch");
        // the LAST lane of VU 1's W digest (the wider table's end)
        let mut m = st.clone();
        m[o.digests + 4 * 2 * f.n * 2 - 4] ^= 1;
        rej("digest_last", &m, "column challenge mismatch");
        // a digest lane >= p is not a canonical field element
        let mut m = st.clone();
        m[o.digests..o.digests + 4].copy_from_slice(&0x7800_0001u32.to_le_bytes());
        rej("digest_noncanonical", &m, "non-canonical field element");
        // the scheme's params digest tampered: "<schema> parameters differ from the verifier's"
        let at = o.indices - 4 - 32;
        let mut m = st.clone();
        m[at] ^= 1;
        rej("params", &m, &format!("{} parameters differ from the verifier's", f.schema));
        // the schema string of another scheme with the Ajtai relation string
        let sch_len_at = at - 4 - f.schema.len();
        let mut m = st.clone();
        let other = f.schema.replace("r0", "r1");
        m[sch_len_at + 4..sch_len_at + 4 + f.schema.len()].copy_from_slice(other.as_bytes());
        rej("schema", &m, "unknown hash-authentication leaf schema");
        // the relation string renamed to an unknown family member (same length): the leaf is unknown to this build
        let rel_at = 59 + 4;
        let mut m = st.clone();
        let renamed = f.relation().replace(&format!("n{}", f.n), &format!("n{}", f.n + 1));
        m[rel_at..rel_at + renamed.len()].copy_from_slice(renamed.as_bytes());
        rej("relation_leaf", &m, "names leaf scheme");
        // a leaf index changed: the statement digest changes
        let mut m = st.clone();
        m[o.indices + 4] ^= 1;
        rej("index", &m, "column challenge mismatch");
        // an x root byte: the multiproof no longer opens (or the digest changes first -- either is a rejection)
        let mut m = st.clone();
        m[o.trees + 73 - 1] ^= 1;
        let (code, _) = verify(&f.sys(), &write_tmp(&format!("{}_root.stmt", f.leaf), &m), &f.proof(), &["--coins", &f.coins()]);
        assert_eq!(code, 1);
        // a proof byte flipped
        let pf = std::fs::read(f.proof()).unwrap();
        let mut m = pf.clone();
        m[pf.len() / 2] ^= 1;
        let (code, _) = verify(&f.sys(), &f.stmt(), &write_tmp(&format!("{}_p.proof", f.leaf), &m), &["--coins", &f.coins()]);
        assert_eq!(code, 1);
        let _ = o.y16;
    }
}

#[test]
fn ajtai_system_mismatches_and_the_key_check_are_enforced() {
    let f = &AJTAI_FX[0];
    // the Ajtai statement against the Poseidon2 hashed fp8-ada system: the pin refuses it ...
    let (code, js) = verify(HASH_SYS, &f.stmt(), &f.proof(), &["--coins", &f.coins()]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned hashed fp8-ada system, the statement is a hashed (ajtai-n64 leaves) fp8-ada statement"), "{js}");
    // ... and UNPINNED the key check does: that system's digest lanes lead to no chain on the derived B (a system built
    // with another key -- B = 0 -- is refused the same way, before any proof byte is read)
    let (code, js) = verify(HASH_SYS, &f.stmt(), &f.proof(), &["--coins", &f.coins(), "--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("does not use the derived key B") && js.contains("hash.d[16] is not pinned"), "{js}");
    // the bare fp8-ada system too
    let (code, js) = verify(FP8_SYS, &f.stmt(), &f.proof(), &["--coins", &f.coins(), "--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("does not use the derived key B") && js.contains("hash.is_end is not pinned"), "{js}");
    // the Poseidon2 statement against the Ajtai system: the pin refuses it
    let (code, js) = verify(&f.sys(), HASH_STMT, HASH_PROOF, &["--coins", HASH_COINS]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned hashed (ajtai-n64 leaves) fp8-ada system, the statement is a hashed fp8-ada statement"), "{js}");
    // the n64 statement against the n128 system (another relation's): the pin refuses it.  Unpinned, the key check does
    // (b_i's first 64 coefficients are the same SHAKE prefix for n = 64 and 128, so a pattern scan found B_64 in rows
    // x < 64 of the n128 system; following the digest lanes finds 256 of them, not 128)
    let g = &AJTAI_FX[1];
    let (code, js) = verify(&g.sys(), &f.stmt(), &f.proof(), &["--coins", &f.coins()]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is the pinned hashed (ajtai-n128 leaves) bf16-hopper system"), "{js}");
    let (code, js) = verify(&g.sys(), &f.stmt(), &f.proof(), &["--coins", &f.coins(), "--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("does not use the derived key B") && js.contains("pin hash.d[128]: not one of the 128 digest lanes"), "{js}");
    // the two Ajtai systems are pinned apart from each other and from the Poseidon2 ones
    let (_, sd) = run(&["system-digest", "--system", &g.sys()]);
    assert!(sd.contains("\"pinned_relation\":\"bf16-hopper+ajtai-n128\"") && !sd.contains(f.sys_id), "{sd}");
    // without --coins: the coins are replayed from the transcript (the conditional accept), for both
    for f in AJTAI_FX {
        let (code, js) = verify(&f.sys(), &f.stmt(), &f.proof(), &[]);
        assert_eq!(code, 0, "{js}");
        assert!(js.contains("coins replayed from the transcript"), "{js}");
    }
}

// red-team-leaf-2 G1 (must reject): `fixtures/fp8-ada-ajtai-n64-g1-decoy/` (`leaf/ajtai_g1_fixture.py`, CPU): the chain the
// digests come from is keyed by B = 0 (every published digest is 0) and 2n decoy linear rows carry the derived key over
// zero hint rows; the proof satisfies that system.  The pattern-scan key check (1cf9178) ACCEPTED it under
// --allow-any-system; following the digest lanes refuses it before a proof byte is read.  Pinned: not the pinned system.
#[test]
fn ajtai_g1_decoy_fixture_is_refused() {
    let d = "fixtures/fp8-ada-ajtai-n64-g1-decoy";
    let (sys, stmt, proof) = (format!("{d}/system.bin"), format!("{d}/sub_00.stmt"), format!("{d}/sub_00.proof"));
    let (code, js) = verify(&sys, &stmt, &proof, &["--allow-any-system"]);
    assert_eq!(code, 1, "{js}");
    assert_eq!(field(&js, "\"accepted\":"), "false");
    assert!(js.contains("does not use the derived key B") && js.contains("hash.d[0]: the key row is not out - (X acc_in)[0]"), "{js}");
    let (code, js) = verify(&sys, &stmt, &proof, &[]);
    assert_eq!(code, 1, "{js}");
    assert!(js.contains("system file is not the pinned hashed (ajtai-n64 leaves) fp8-ada system"), "{js}");
    let (code, js) = run(&["batch", "--dir", d, "--system", &sys, "--allow-any-system", "--threads", "2"]);
    assert_ne!(code, 0, "{js}");
    assert!(js.contains("\"accepted\":0,\"rejected\":1"), "{js}");
}
