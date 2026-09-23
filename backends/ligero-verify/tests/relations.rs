//! Relation-named statements (`ligero-statement/v4`) and the per-relation pinned systems, through the binary.
//!
//! `fixtures/fp8-ada/`: sub-batch 0 of rep 1 of the FP8 Ada dump `run-files` art:c5dff3c6... (attempt
//! r20260922-235542-295b, lane ada-fp8-dumps, code a4b0ac7; interactive non-ZK, B = 4096, 85 VUs x 48 steps at
//! l = 4096, K = 32) with the runner's step-0 coins and the pod's `system.bin` (sha256 cba9a09f..., byte-identical
//! to the laptop compilation of `compile_fp8_unit(ADA, chain=True)`).  `fixtures/systems/`: the bf16-hopper and
//! fp8-hopper systems compiled from lane b-hopper a39de82 (no proofs of theirs are in the store yet).
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
    ] {
        let (code, js) = run(&["system-digest", "--system", path]);
        assert_eq!(code, 0, "{js}");
        assert_eq!(field(&js, "\"pinned_relation\":"), format!("\"{rel}\""), "{js}");
        assert_eq!(field(&js, "\"m\":"), m, "{js}");
        assert_eq!(field(&js, "\"pins\":"), pins, "{js}");
        assert_eq!(field(&js, "\"chain\":"), "true", "{js}");
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
    assert!(js.contains("statement names relation \\\"fp8-adb\\\"; this build pins bf16-ampere, fp8-ada, bf16-hopper, fp8-hopper"), "{js}");
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
