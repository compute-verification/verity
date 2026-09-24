//! Live coins authenticated by the issuing verifier's own session record (`live.ChallengeServerSession`'s
//! `session.json` + `verdict.json`), red-team-ligerito-3 R3-7 / R3-10.
//!
//! A live proof's coin slots are derived from the RECORD only: the verifier batch is found by its `stmt_sha256` (the
//! stream statement `b"lgto-stmt|v1|" || sha256(params) || sha256(stmt) || sha256(key)`, computed here), its commitments
//! are `coin_commit(r_k, s_k)` of the recorded coins (all of them, in order: the exact list the verifier committed to),
//! slot k is `expand_challenge(batch_context(STMT, commitments), k, label_k, r_k)` with the RECORDED label, and the
//! proof must consume exactly the recorded rounds.  While verifying, every round's MSG (BLAKE2b-256 of the prover
//! messages absorbed since the previous slot, framed `u32 len(label) || label || u64 len(data) || data`, as
//! `prove.StreamCoins` sent it) and label must equal the record's `msg_sha256` / `label`: the verifier opened coin k after
//! that message.  The dump's `.coins` file and `stream_binding` are never read.
//!
//! What the run may then claim: the proof's interactive union times A, the batches the supplied records show on the same
//! statement (a prover that aborts and retries had A tries at an accepting coin set; R3-10).  Sessions whose records are
//! not supplied are not counted: the claim is over the records given.

use crate::hash::{blake2b_256, hex, sha256, shake256};
use crate::json::{self, Json};
use crate::lgto::{self, Options, Report};
use crate::transcript::{Coins, Live};
use std::cell::RefCell;
use std::rc::Rc;

/// `ligero/protocol.py` COIN_TAG
pub const COIN_TAG: &[u8] = b"ligero-b/coin|v1|";
/// `ligero/live.py` CHAL_TAG
pub const CHAL_TAG: &[u8] = b"ligero-chal|v1|";
pub const CTX_TAG: &[u8] = b"ligero-chal-ctx|";
/// `prove.statement_for_stream`
pub const STMT_TAG: &[u8] = b"lgto-stmt|v1|";

pub fn coin_commit(r: &[u8; 32], s: &[u8; 32]) -> [u8; 32] {
    blake2b_256(&[COIN_TAG, r, s].concat())
}

pub fn batch_context(stmt: &[u8], commitments: &[[u8; 32]]) -> [u8; 32] {
    let mut b = CTX_TAG.to_vec();
    b.extend_from_slice(&(stmt.len() as u32).to_le_bytes());
    b.extend_from_slice(stmt);
    for c in commitments {
        b.extend_from_slice(c);
    }
    blake2b_256(&b)
}

pub fn expand_challenge(ctx: &[u8; 32], k: usize, label: &[u8], r: &[u8; 32]) -> Vec<u8> {
    let mut b = CHAL_TAG.to_vec();
    b.extend_from_slice(ctx);
    b.extend_from_slice(&(k as u32).to_le_bytes());
    b.extend_from_slice(&(label.len() as u32).to_le_bytes());
    b.extend_from_slice(label);
    b.extend_from_slice(r);
    shake256(&b, 32)
}

/// The stream statement STMT of a batch (fixed before any coin is opened).
pub fn stream_statement(params_raw: &[u8], stmt_bytes: &[u8], key_bytes: &[u8]) -> Vec<u8> {
    [STMT_TAG, &sha256(params_raw), &sha256(stmt_bytes), &sha256(key_bytes)].concat()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Round {
    pub label: String,
    pub msg_sha256: [u8; 32],
}

/// One batch of a verifier session, as the verifier recorded it.
#[derive(Clone, Debug)]
pub struct Batch {
    pub session: String,
    pub batch: usize,
    /// the session's verdict (every batch complete, rounds in order); the verifier does not check proofs
    pub accepted: bool,
    pub stmt_sha256: [u8; 32],
    pub coins: Vec<([u8; 32], [u8; 32])>,
    pub rounds: Vec<Round>,
}

fn digest_of(j: Option<&Json>, what: &str) -> Result<[u8; 32], String> {
    let s = j.and_then(Json::as_str).ok_or_else(|| format!("session record: {what} missing"))?;
    let b = crate::relation::hex_bytes(s)?;
    b.try_into().map_err(|_| format!("session record: {what} is not 32 bytes"))
}

/// `session.json` + `verdict.json` of one session -> its batches (a batch without a statement or coins is kept: it still
/// counts as an attempt on nothing, and cannot authenticate anything).
pub fn read_record(session_json: &str, verdict_json: &str, name: &str) -> Result<Vec<Batch>, String> {
    let s = json::parse(session_json)?;
    let v = json::parse(verdict_json)?;
    let accepted = v.get("accepted").and_then(Json::as_bool).unwrap_or(false);
    let session = v.get("session_id").and_then(Json::as_str).unwrap_or(name).to_string();
    let batches = s.get("batches").and_then(Json::as_arr).ok_or("session record: no batches")?;
    let mut out = Vec::with_capacity(batches.len());
    for (i, b) in batches.iter().enumerate() {
        let Some(stmt) = b.get("stmt_sha256") else { continue };
        let mut coins = Vec::new();
        for c in b.get("coins").and_then(Json::as_arr).unwrap_or(&[]) {
            coins.push((digest_of(c.get("r"), "coin r")?, digest_of(c.get("s"), "coin s")?));
        }
        let mut rounds = Vec::new();
        for (k, r) in b.get("rounds").and_then(Json::as_arr).unwrap_or(&[]).iter().enumerate() {
            if r.get("k").and_then(Json::as_usize) != Some(k) {
                return Err(format!("session record {session} batch {i}: round {k} out of order"));
            }
            let label = r.get("label").and_then(Json::as_str).ok_or("session record: round label")?.to_string();
            rounds.push(Round { label, msg_sha256: digest_of(r.get("msg_sha256"), "msg_sha256")? });
        }
        out.push(Batch {
            session: session.clone(),
            batch: b.get("batch").and_then(Json::as_usize).unwrap_or(i),
            accepted,
            stmt_sha256: digest_of(Some(stmt), "stmt_sha256")?,
            coins,
            rounds,
        });
    }
    Ok(out)
}

/// The verifier's slots for its batch `b` on statement `stmt`: one per recorded round, from the recorded coins and labels.
pub fn slots(b: &Batch, stmt: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    // the verifier commits `rounds` coins per batch up front and an accepted session completes every round
    if b.rounds.len() != b.coins.len() {
        return Err(format!("{} rounds recorded, {} coins committed", b.rounds.len(), b.coins.len()));
    }
    let ccom: Vec<[u8; 32]> = b.coins.iter().map(|(r, s)| coin_commit(r, s)).collect();
    let ctx = batch_context(stmt, &ccom);
    Ok(b.rounds.iter().zip(&b.coins).enumerate().map(|(k, (rd, (r, _)))| expand_challenge(&ctx, k, rd.label.as_bytes(), r)).collect())
}

#[derive(Default)]
struct Seen {
    msgs: Vec<[u8; 32]>,
    labels: Vec<Vec<u8>>,
}

/// `prove.SessionReplayCoins`: `Live` over the verifier's slots that also records every round's MSG and label.
struct SessionCoins {
    inner: Live,
    buf: Vec<u8>,
    seen: Rc<RefCell<Seen>>,
}

impl Coins for SessionCoins {
    fn absorb(&mut self, label: &[u8], data: &[u8]) {
        self.buf.extend_from_slice(&(label.len() as u32).to_le_bytes());
        self.buf.extend_from_slice(label);
        self.buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        self.buf.extend_from_slice(data);
    }

    fn begin_round(&mut self, kind: &[u8], label: &[u8], extra: &[u8]) -> Result<(), String> {
        let mut s = self.seen.borrow_mut();
        s.msgs.push(blake2b_256(&self.buf));
        s.labels.push(label.to_vec());
        self.buf.clear();
        self.inner.begin_round(kind, label, extra)
    }

    fn stream(&mut self, nbytes: usize) -> Vec<u8> {
        self.inner.stream(nbytes)
    }

    fn rounds(&self) -> usize {
        self.inner.rounds()
    }

    fn unused(&self) -> usize {
        self.inner.unused()
    }
}

/// A replay's report, MSG k per slot and the slot labels.
pub type Replayed = (Report, Vec<[u8; 32]>, Vec<Vec<u8>>);

/// Verify `proof` under the live `slots`; also returns MSG k (the BLAKE2b of the absorbs before slot k) and the label of
/// every slot, what the issuing verifier records per round.
pub fn replay(stmt_bytes: &[u8], key_bytes: &[u8], proof_bytes: &[u8], slots: Vec<Vec<u8>>, opt: &Options) -> Result<Replayed, String> {
    let seen = Rc::new(RefCell::new(Seen::default()));
    let mut slots = Some(slots);
    let mut coins_for = |kind: &str| -> Result<Box<dyn Coins>, String> {
        if kind != "live" {
            return Err(format!("coins {kind:?}: a session record authenticates live coins only"));
        }
        let sl = slots.take().ok_or("coins requested twice")?;
        Ok(Box::new(SessionCoins { inner: Live::new(sl), buf: Vec::new(), seen: seen.clone() }))
    };
    let rep = lgto::verify(stmt_bytes, key_bytes, proof_bytes, &mut coins_for, opt)?;
    let s = std::mem::take(&mut *seen.borrow_mut());
    Ok((rep, s.msgs, s.labels))
}

/// One live proof authenticated against a recorded batch.
#[derive(Clone, Debug)]
pub struct Authenticated {
    pub report: Report,
    pub session: String,
    pub batch: usize,
    pub rounds: usize,
    pub stmt_sha256: [u8; 32],
}

/// Verify `proof` with the coins of the recorded batch on its statement.  Every candidate batch (same STMT, session
/// accepted) is tried; the first under which the proof verifies AND every round's MSG / label equals the record wins.
pub fn authenticate(stmt_bytes: &[u8], key_bytes: &[u8], proof_bytes: &[u8], records: &[Batch], opt: &Options) -> Result<Authenticated, String> {
    authenticate_unused(stmt_bytes, key_bytes, proof_bytes, records, &[], opt)
}

/// `authenticate` over the batches not in `used` (session id, batch): one recorded batch authenticates one proof.
pub fn authenticate_unused(
    stmt_bytes: &[u8],
    key_bytes: &[u8],
    proof_bytes: &[u8],
    records: &[Batch],
    used: &[(String, usize)],
    opt: &Options,
) -> Result<Authenticated, String> {
    let proof = lgto::read_proof(proof_bytes)?;
    if proof.params.coins != "live" {
        return Err(format!("not a live proof (coins {:?})", proof.params.coins));
    }
    let stmt = stream_statement(&proof.params_raw, stmt_bytes, key_bytes);
    let d = sha256(&stmt);
    let cands: Vec<&Batch> = records.iter().filter(|b| b.stmt_sha256 == d).collect();
    if cands.is_empty() {
        return Err(format!("no recorded verifier batch on this statement (STMT sha256 {})", hex(&d)));
    }
    let mut why = Vec::new();
    for b in cands {
        let tag = format!("session {} batch {}", b.session, b.batch);
        if used.iter().any(|(s, i)| *s == b.session && *i == b.batch) {
            why.push(format!("{tag}: already authenticates another proof"));
            continue;
        }
        if !b.accepted {
            why.push(format!("{tag}: session not accepted by its verifier"));
            continue;
        }
        let sl = match slots(b, &stmt) {
            Ok(s) => s,
            Err(e) => {
                why.push(format!("{tag}: {e}"));
                continue;
            }
        };
        let (rep, msgs, labels) = match replay(stmt_bytes, key_bytes, proof_bytes, sl, opt) {
            Ok(r) => r,
            Err(e) => {
                why.push(format!("{tag}: proof rejected under the recorded coins: {e}"));
                continue;
            }
        };
        if msgs.len() != b.rounds.len() {
            why.push(format!("{tag}: {} coin rounds in the proof, {} recorded", msgs.len(), b.rounds.len()));
            continue;
        }
        let bad = (0..b.rounds.len()).find(|&k| sha256(&msgs[k]) != b.rounds[k].msg_sha256 || labels[k] != b.rounds[k].label.as_bytes());
        if let Some(k) = bad {
            why.push(format!("{tag}: round {k}: the verifier opened its coin on another message or label than the proof's"));
            continue;
        }
        let mut rep = rep;
        rep.live_session = Some((b.session.clone(), b.batch));
        let tries = attempts(records, &d);
        if !rep.legacy {
            rep.claimed_log2 = Some(rep.soundness_log2 + (tries as f64).log2());
            rep.claim_basis = format!(
                "interactive: coins drawn by the live verifier after each prover message (session {} batch {} accepted; {tries} recorded batch(es) on this statement)",
                b.session, b.batch
            );
        }
        return Ok(Authenticated { report: rep, session: b.session.clone(), batch: b.batch, rounds: b.rounds.len(), stmt_sha256: d });
    }
    Err(why.join("; "))
}

/// The recorded batches on a statement: every one the verifier ran (whatever its verdict), i.e. every coin set the
/// prover saw for that claim.
pub fn attempts(records: &[Batch], stmt_sha256: &[u8; 32]) -> usize {
    records.iter().filter(|b| b.stmt_sha256 == *stmt_sha256).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ligero/live.py + protocol.py by hand: coin_commit, batch_context and expand_challenge framings.
    #[test]
    fn derivations_match_the_python_framing() {
        let (r, s) = ([1u8; 32], [2u8; 32]);
        assert_eq!(coin_commit(&r, &s), blake2b_256(&[b"ligero-b/coin|v1|".as_slice(), &[1u8; 32], &[2u8; 32]].concat()));
        let c = coin_commit(&r, &s);
        let ctx = batch_context(b"abc", &[c, c]);
        assert_eq!(ctx, blake2b_256(&[b"ligero-chal-ctx|".as_slice(), &3u32.to_le_bytes(), b"abc", &c, &c].concat()));
        let e = expand_challenge(&ctx, 5, b"zc/tau", &r);
        assert_eq!(e, shake256(&[b"ligero-chal|v1|".as_slice(), &ctx, &5u32.to_le_bytes(), &6u32.to_le_bytes(), b"zc/tau", &r].concat(), 32));
    }
}
