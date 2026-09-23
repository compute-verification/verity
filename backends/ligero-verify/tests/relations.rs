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
        // the folded x4 units (four instruction steps per column; experimental)
        ("fixtures/systems/bf16-hopper-v2x4.system.bin", "bf16-hopper-v2x4", "1045", "220"),
        ("fixtures/systems/bf16-ampere-v2x4.system.bin", "bf16-ampere-v2x4", "1717", "216"),
        ("fixtures/systems/fp8-hopper-v2x4.system.bin", "fp8-hopper-v2x4", "647", "64"),
        ("fixtures/systems/fp8-ada-v2x4.system.bin", "fp8-ada-v2x4", "1255", "128"),
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
    assert!(js.contains("statement names relation \\\"fp4-blackwell\\\"; this build pins bf16-ampere, fp8-ada, bf16-hopper, fp8-hopper, fp4-nvf4, bf16-hopper-v2, bf16-ampere-v2, fp8-hopper-v2, fp8-ada-v2, bf16-hopper-v2x4, bf16-ampere-v2x4, fp8-hopper-v2x4, fp8-ada-v2x4"), "{js}");
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
