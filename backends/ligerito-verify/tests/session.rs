//! Live coins authenticated by the issuing verifier's session record (red-team-ligerito-3 R3-7 / R3-10).

use ligerito_verify::hash::{sha256, shake256};
use ligerito_verify::json::{self, Json};
use ligerito_verify::lgto::{self, Options};
use ligerito_verify::relation::hex_bytes;
use ligerito_verify::session::{self, Batch, Round};
use ligerito_verify::transcript::Live;
use std::path::Path;

const PROOF: &[u8] = include_bytes!("../fixtures/session/fp8ada_l256_legacy/batch_live_0.lgto");
const STMT: &[u8] = include_bytes!("../fixtures/session/fp8ada_l256_legacy/batch_live_0.stmt");
const KEY: &[u8] = include_bytes!("../fixtures/lgto/fp8ada_l256_v1/key.bin");
const BINDING: &str = include_str!("../fixtures/session/fp8ada_l256_legacy/binding.json");

fn d32(j: &Json) -> [u8; 32] {
    hex_bytes(j.as_str().unwrap()).unwrap().try_into().unwrap()
}

/// the pre-V1 fixture needs `allow_legacy` (it carries no zero claims; the coin path is what is tested here)
fn opt() -> Options {
    Options { allow_legacy: true, ..Options::default() }
}

/// The verifier record `ChallengeServerSession` would have written for the fixture's stream: its coins (the binding's
/// openings), its STMT digest, and per round the label and sha256(MSG) the prover sent (MSG recomputed here by replay).
fn honest_record() -> (Vec<u8>, Batch) {
    let b = json::parse(BINDING).unwrap();
    let stmt = hex_bytes(b.get("statement").unwrap().as_str().unwrap()).unwrap();
    let coins: Vec<([u8; 32], [u8; 32])> = b.get("openings").unwrap().as_arr().unwrap().iter().map(|o| {
        let o = o.as_arr().unwrap();
        (d32(&o[0]), d32(&o[1]))
    }).collect();
    let labels: Vec<String> = b.get("labels").unwrap().as_arr().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
    let mut batch = Batch {
        session: "synthetic".into(),
        batch: 0,
        accepted: true,
        stmt_sha256: sha256(&stmt),
        coins,
        rounds: labels.iter().map(|l| Round { label: l.clone(), msg_sha256: [0; 32] }).collect(),
    };
    let sl = session::slots(&batch, &stmt).unwrap();
    let (_, msgs, seen_labels) = session::replay(STMT, KEY, PROOF, sl, &opt()).unwrap();
    assert_eq!(seen_labels.iter().map(|l| String::from_utf8(l.clone()).unwrap()).collect::<Vec<_>>(), labels);
    for (r, m) in batch.rounds.iter_mut().zip(&msgs) {
        r.msg_sha256 = sha256(m);
    }
    (stmt, batch)
}

/// Rust's STMT, coin commitments and slot derivation are Python's (`prove.StreamCoins` over `live.LocalChallenges`): the
/// slots derived from the record alone make the Python-written proof verify.
#[test]
fn record_derived_slots_reproduce_the_python_stream() {
    let b = json::parse(BINDING).unwrap();
    let (stmt, batch) = honest_record();
    let proof = lgto::read_proof(PROOF).unwrap();
    assert_eq!(session::stream_statement(&proof.params_raw, STMT, KEY), stmt);
    let ccom: Vec<[u8; 32]> = b.get("commitments").unwrap().as_arr().unwrap().iter().map(d32).collect();
    assert_eq!(batch.coins.iter().map(|(r, s)| session::coin_commit(r, s)).collect::<Vec<_>>(), ccom);
    let a = session::authenticate(STMT, KEY, PROOF, &[batch], &opt()).unwrap();
    assert_eq!(a.rounds, 64);
    assert!(a.report.legacy && a.report.claimed_log2.is_none(), "a pre-V1 proof claims nothing, authenticated or not");
}

fn rejects(recs: &[Batch], want: &str) {
    match session::authenticate(STMT, KEY, PROOF, recs, &opt()) {
        Ok(_) => panic!("accepted; wanted a rejection containing {want:?}"),
        Err(e) => assert!(e.contains(want), "{want:?} not in {e:?}"),
    }
}

#[test]
fn a_record_that_is_not_the_proofs_is_rejected() {
    let (_, h) = honest_record();
    let last = h.rounds.len() - 1;
    // a round's MSG: the verifier opened coin k after another message
    for k in [0, 17, last] {
        let mut b = h.clone();
        b.rounds[k].msg_sha256[0] ^= 1;
        rejects(&[b], &format!("round {k}: the verifier opened its coin on another message"));
    }
    // a round's label enters its slot: the proof no longer verifies
    let mut b = h.clone();
    b.rounds[5].label = "zc/r/9".into();
    rejects(&[b], "proof rejected under the recorded coins");
    // any coin's opening (r enters its slot; s every commitment, so every slot)
    for (k, which) in [(3, 0), (40, 1), (last, 1)] {
        let mut b = h.clone();
        if which == 0 {
            b.coins[k].0[7] ^= 1;
        } else {
            b.coins[k].1[7] ^= 1;
        }
        rejects(&[b], "proof rejected under the recorded coins");
    }
    // exact length: one round short, one extra round, coins != rounds
    let mut b = h.clone();
    b.rounds.pop();
    b.coins.pop();
    rejects(&[b], "proof rejected under the recorded coins");
    let mut b = h.clone();
    b.rounds.push(Round { label: "x".into(), msg_sha256: [0; 32] });
    b.coins.push(([9; 32], [9; 32]));
    rejects(&[b], "proof rejected under the recorded coins");
    let mut b = h.clone();
    b.coins.push(([9; 32], [9; 32]));
    rejects(&[b], "64 rounds recorded, 65 coins committed");
    // the session was not accepted by its verifier
    let mut b = h.clone();
    b.accepted = false;
    rejects(&[b], "session not accepted");
    // a record of another statement
    let mut b = h.clone();
    b.stmt_sha256[0] ^= 1;
    rejects(&[b], "no recorded verifier batch on this statement");
}

/// R3-7: a prover that ignores the verifier's coins and replays its own slots (the dump's `.coins`) is accepted by the
/// file replay, never by the record: the verifier's coins are the record's.
#[test]
fn slots_are_the_records_never_the_dumps() {
    let (stmt, h) = honest_record();
    let dump_slots = session::slots(&h, &stmt).unwrap();
    let mut coins_for = |_: &str| -> Result<Box<dyn ligerito_verify::transcript::Coins>, String> { Ok(Box::new(Live::new(dump_slots.clone()))) };
    let file = lgto::verify(STMT, KEY, PROOF, &mut coins_for, &opt()).unwrap();
    assert!(file.claimed_log2.is_none());
    // the verifier's own coins (what it recorded) were others; same labels and messages
    let mut theirs = h.clone();
    for (k, c) in theirs.coins.iter_mut().enumerate() {
        let r: [u8; 32] = shake256(&[b"verifier r".as_slice(), &(k as u32).to_le_bytes()].concat(), 32).try_into().unwrap();
        *c = (r, c.1);
    }
    rejects(&[theirs.clone()], "proof rejected under the recorded coins");
    // with the honest batch recorded too, the honest one authenticates and the attempts count both
    let a = session::authenticate(STMT, KEY, PROOF, &[theirs, h.clone()], &opt()).unwrap();
    assert_eq!(a.batch, 0);
    assert_eq!(session::attempts(&[h.clone(), h.clone(), h], &a.stmt_sha256), 3);
}

/// `read_record` over `ChallengeServerSession`'s JSON: batches, coins, rounds in order, the verdict.
#[test]
fn reads_the_verifier_record_format() {
    let (_, h) = honest_record();
    let hx = ligerito_verify::hash::hex;
    let rounds: Vec<String> = h.rounds.iter().enumerate().map(|(k, r)| format!("{{\"k\":{k},\"label\":\"{}\",\"bytes\":0,\"msg_sha256\":\"{}\",\"t_msg_recv\":1.5}}", r.label, hx(&r.msg_sha256))).collect();
    let coins: Vec<String> = h.coins.iter().map(|(r, s)| format!("{{\"r\":\"{}\",\"s\":\"{}\"}}", hx(r), hx(s))).collect();
    let sess = format!(
        "{{\"hello\":{{\"n_batches\":2,\"rounds\":64}},\"batches\":[{{\"batch\":0,\"t_ccom_sent\":1.0,\"rounds\":[{}],\"stmt_sha256\":\"{}\",\"coins\":[{}]}},{{\"batch\":1,\"rounds\":[],\"coins\":[]}}]}}",
        rounds.join(","),
        hx(&h.stmt_sha256),
        coins.join(",")
    );
    let recs = session::read_record(&sess, "{\"session_id\":\"s1\",\"accepted\":true}", "dir").unwrap();
    assert_eq!(recs.len(), 1, "a batch without a statement authenticates nothing");
    assert_eq!((recs[0].session.as_str(), recs[0].rounds.len(), recs[0].coins.len()), ("s1", 64, 64));
    session::authenticate(STMT, KEY, PROOF, &recs, &opt()).unwrap();
    let recs = session::read_record(&sess, "{\"session_id\":\"s1\",\"accepted\":false}", "dir").unwrap();
    rejects(&recs, "session not accepted");
    let bad = sess.replacen("{\"k\":1,", "{\"k\":2,", 1);
    assert!(session::read_record(&bad, "{\"accepted\":true}", "dir").unwrap_err().contains("out of order"));
    // the store's other live protocol (`live.Session`: `subbatches`, no `batches`): no batch on any statement
    assert!(session::read_record("{\"verdict\":{},\"subbatches\":[{\"sub\":0}]}", "{\"accepted\":true}", "dir").unwrap().is_empty());
    assert_eq!(session::read_record(&format!("{},\"subbatches\":[]}}", &sess[..sess.len() - 1]), "{\"accepted\":true}", "dir").unwrap().len(), 1);
    for bad in ["{\"hello\":{}}", "{\"verdict\":{},\"subbatches\":{}}"] {
        assert!(session::read_record(bad, "{\"accepted\":true}", "dir").unwrap_err().contains("no batches"), "{bad}");
    }
}

fn zk_live_records() -> Vec<Batch> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/session/fp8ada_l256_zk_live/verifier_sessions");
    let mut dirs: Vec<_> = std::fs::read_dir(&root).unwrap().flatten().map(|e| e.path()).collect();
    dirs.sort();
    dirs.iter()
        .flat_map(|d| {
            let rd = |f: &str| std::fs::read_to_string(d.join(f)).unwrap();
            session::read_record(&rd("session.json"), &rd("verdict.json"), "").unwrap()
        })
        .collect()
}

/// relation-3 2b0f87a6's loopback-verifier test (fp8-ada --zk, LGSC0004, real `ChallengeServerSession` records): the
/// honest proof authenticates (every MSG against the real record), the label- and context-grinding proofs verify under
/// their own `.coins` but not under the record, one batch authenticates one proof, and the aborted session counts as an
/// attempt.
#[test]
fn real_verifier_records_lgsc0004_live() {
    const ZSTMT: &[u8] = include_bytes!("../fixtures/lgto/fp8ada_l256_zk_lgsc4/honest.stmt");
    const ZKEY: &[u8] = include_bytes!("../fixtures/lgto/fp8ada_l256_zk_lgsc4/key.bin");
    const HONEST: &[u8] = include_bytes!("../fixtures/session/fp8ada_l256_zk_live/honest.lgto");
    let recs = zk_live_records();
    assert_eq!(recs.len(), 4);
    let strict = Options::default();
    let a = session::authenticate(ZSTMT, ZKEY, HONEST, &recs, &strict).unwrap();
    assert_eq!((a.session.as_str(), a.batch, a.rounds), ("c20260924T002130Z-1324", 0, 28));
    assert_eq!(a.report.sumcheck, "LGSC0004");
    assert!(a.report.key_pinned && a.report.key_derived_from.is_some());
    assert_eq!(session::attempts(&recs, &a.stmt_sha256), 4, "3 accepted sessions + 1 aborted, one statement");
    assert!((a.report.claimed_log2.unwrap() - (a.report.soundness_log2 + 2.0)).abs() < 1e-9);
    // one batch, one proof
    let used = vec![(a.session.clone(), a.batch)];
    let e = session::authenticate_unused(ZSTMT, ZKEY, HONEST, &recs, &used, &strict).unwrap_err();
    assert!(e.contains("already authenticates another proof"), "{e}");
    // a real record's MSG
    let mut bad = recs.clone();
    let i = bad.iter().position(|b| b.session == a.session).unwrap();
    bad[i].rounds[13].msg_sha256[0] ^= 1;
    let e = session::authenticate(ZSTMT, ZKEY, HONEST, &bad[i..=i], &strict).unwrap_err();
    assert!(e.contains("round 13"), "{e}");
    // R3-7 A / B: valid proofs under prover-chosen slots (the dump's .coins), rejected under every recorded batch
    for (pf, coins) in [
        (&include_bytes!("../fixtures/session/fp8ada_l256_zk_live/label_grinding.lgto")[..], &include_bytes!("../fixtures/session/fp8ada_l256_zk_live/label_grinding.coins")[..]),
        (&include_bytes!("../fixtures/session/fp8ada_l256_zk_live/context_grinding.lgto")[..], &include_bytes!("../fixtures/session/fp8ada_l256_zk_live/context_grinding.coins")[..]),
    ] {
        let mut from_file = |_: &str| -> Result<Box<dyn ligerito_verify::transcript::Coins>, String> { Ok(Box::new(lgto::read_coins(coins)?)) };
        let cold = lgto::verify(ZSTMT, ZKEY, pf, &mut from_file, &strict).unwrap();
        assert!(cold.claimed_log2.is_none());
        let e = session::authenticate(ZSTMT, ZKEY, pf, &recs, &strict).unwrap_err();
        assert_eq!(e.matches("proof rejected under the recorded coins").count(), 3, "{e}");
        assert!(e.contains("session not accepted by its verifier"), "{e}");
    }
}

/// Real live dumps with the issuing verifier's records (relation-2's bench, `dump_*-live-ro` + `verifier_session/`):
/// `LIGERITO_LIVE_DUMPS=dir1:dir2 cargo test --release real_live_dumps`.  Skipped when unset (15 MB statements).
#[test]
fn real_live_dumps_authenticate_against_their_verifier_records() {
    let Ok(dirs) = std::env::var("LIGERITO_LIVE_DUMPS") else {
        eprintln!("LIGERITO_LIVE_DUMPS unset: skipped");
        return;
    };
    let strict = Options::default();
    for d in dirs.split(':').filter(|s| !s.is_empty()).map(Path::new) {
        let key = std::fs::read(d.join("key.bin")).unwrap();
        let mut recs = Vec::new();
        for s in std::fs::read_dir(d.join("verifier_session")).unwrap().flatten() {
            let rd = |f: &str| std::fs::read_to_string(s.path().join(f)).unwrap();
            recs.extend(session::read_record(&rd("session.json"), &rd("verdict.json"), &s.file_name().to_string_lossy()).unwrap());
        }
        let mut n = 0;
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".lgto") else { continue };
            let (pf, st) = (std::fs::read(e.path()).unwrap(), std::fs::read(d.join(format!("{stem}.stmt"))).unwrap());
            let a = session::authenticate(&st, &key, &pf, &recs, &strict).unwrap_or_else(|x| panic!("{}: {x}", e.path().display()));
            let tries = session::attempts(&recs, &a.stmt_sha256);
            assert!((a.report.claimed_log2.unwrap() - a.report.soundness_log2 - (tries as f64).log2()).abs() < 1e-9);
            let i = recs.iter().position(|b| b.session == a.session && b.batch == a.batch).unwrap();
            let mut bad = recs.clone();
            bad[i].rounds[a.rounds / 2].msg_sha256[31] ^= 1;
            assert!(session::authenticate(&st, &key, &pf, &bad[i..=i], &strict).is_err());
            let mut bad = recs.clone();
            bad[i].coins[0].1[0] ^= 1;
            assert!(session::authenticate(&st, &key, &pf, &bad[i..=i], &strict).is_err());
            n += 1;
        }
        assert!(n > 0, "{}: no .lgto", d.display());
        eprintln!("{}: {n} live proof(s) authenticated", d.display());
    }
}
