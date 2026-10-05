//! The GPU prover's Fiat-Shamir transcript (`gpu/transcript.py`), reproduced exactly:
//!
//!   state <- SHA-256(domain);  absorb: state <- SHA-256(state || msg)
//!   challenge: digests d_c = SHA-256(state || "chal" || ctr_LE64), ctr += 1 per digest; the eight LE u32 words of
//!   each digest are taken in order and accepted iff < p; six accepted words make one extension challenge; the
//!   words left in the last digest are discarded (a fresh digest starts the next challenge).
//!
//! This differs from the CPU crate's `transcript.rs` (one digest per *coordinate*, first accepted word only), so
//! the two transcripts are not interchangeable even at equal messages.
//!
//! Live prime coins (route (a), `gpu/transcript.py::LiveTranscript`, PROTOCOL 17.6): [`Transcript::replay`] runs the
//! same absorb chain over the session's recorded rounds instead of deriving challenges. A challenge with an absorb
//! since the previous round (or none left from it) opens the next recorded round: the replayed state must equal the
//! state the prover committed before that coin, the round's count must equal the one requested (`reserve`), and the
//! round's words are the challenges, then chained as `state <- SHA-256(state || LIVE_TAG || words LE)`. Unused coins of
//! a round are discarded at the next absorb. A mismatch is kept and reported by [`Transcript::live_finish`].

use crate::field::{Ext, DEG, P};
use crate::sha256::Sha256;

pub const LIVE_TAG: &[u8] = b"verity-gkr/live-coins/v1";

/// One recorded prime round: the state committed, the challenges requested, and the `DEG * n` words issued.
#[derive(Clone, Debug)]
pub struct Round { pub state: [u8; 32], pub n: usize, pub words: Vec<u32> }

#[derive(Clone, Debug)]
pub struct LiveCoins { pub rounds: Vec<Round>, pub before_commit: usize }

struct Live { rec: LiveCoins, k: usize, buf: std::collections::VecDeque<Ext>, dirty: bool, want: usize, failure: Option<String> }

pub struct Transcript { state: [u8; 32], ctr: u64, pub slots: u64, live: Option<Live> }

impl Transcript {
    pub fn new(domain: &[u8]) -> Self { Transcript { state: Sha256::digest(domain), ctr: 0, slots: 0, live: None } }
    pub fn replay(domain: &[u8], rec: LiveCoins) -> Self {
        let mut t = Self::new(domain);
        t.live = Some(Live { rec, k: 0, buf: Default::default(), dirty: true, want: 0, failure: None });
        t
    }
    pub fn is_live(&self) -> bool { self.live.is_some() }
    pub fn absorb(&mut self, data: &[u8]) {
        let mut s = Sha256::new();
        s.update(&self.state);
        s.update(data);
        self.state = s.finalize();
        if let Some(l) = &mut self.live { l.dirty = true; }
    }
    pub fn absorb_ext(&mut self, x: Ext) { self.absorb(&x.to_bytes()); }
    pub fn absorb_exts(&mut self, xs: &[Ext]) {
        let mut buf = Vec::with_capacity(xs.len() * Ext::BYTES);
        for x in xs { buf.extend_from_slice(&x.to_bytes()); }
        self.absorb(&buf);
    }
    /// The next `n` challenges follow with no absorb between them: one live round (nothing under Fiat-Shamir).
    pub fn reserve(&mut self, n: usize) {
        if let Some(l) = &mut self.live {
            if l.dirty || l.buf.is_empty() { l.want = l.want.max(n); }
        }
    }
    fn digest(&mut self) -> [u8; 32] {
        let mut s = Sha256::new();
        s.update(&self.state);
        s.update(b"chal");
        s.update(&self.ctr.to_le_bytes());
        self.ctr += 1;
        s.finalize()
    }
    fn live_round(&mut self) {
        let state = self.state;
        let l = self.live.as_mut().unwrap();
        let n = l.want.max(1);
        l.want = 0;
        let k = l.k;
        l.k += 1;
        let words = match l.rec.rounds.get(k) {
            None => { l.failure.get_or_insert(format!("R2-prime: round {k}: the verifier issued no such coin")); vec![0; DEG * n] }
            Some(r) if r.n != n => { l.failure.get_or_insert(format!("R2-prime: round {k}: {n} challenges drawn, {} issued", r.n)); vec![0; DEG * n] }
            Some(r) if r.state != state => {
                l.failure.get_or_insert(format!("R2-prime: round {k}: the proof's messages differ from those committed before the coin"));
                r.words.clone()
            }
            Some(r) if r.words.len() != DEG * n || r.words.iter().any(|&w| w >= P) => {
                l.failure.get_or_insert(format!("R2-prime: round {k}: malformed words")); vec![0; DEG * n]
            }
            Some(r) => r.words.clone(),
        };
        l.buf = words.chunks(DEG).map(|c| Ext(c.try_into().unwrap())).collect();
        l.dirty = false;
        let mut s = Sha256::new();
        s.update(&state);
        s.update(LIVE_TAG);
        for w in &words { s.update(&w.to_le_bytes()); }
        self.state = s.finalize();
    }
    pub fn challenge(&mut self) -> Ext {
        self.slots += 1;
        if let Some(l) = &self.live {
            if l.dirty || l.buf.is_empty() { self.live_round(); }
            return self.live.as_mut().unwrap().buf.pop_front().unwrap();
        }
        let mut c = [0u32; DEG];
        let mut got = 0;
        while got < DEG {
            let d = self.digest();
            for i in 0..8 {
                if got == DEG { break; }
                let v = u32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap());
                if v < P { c[got] = v; got += 1; }
            }
        }
        Ext(c)
    }
    pub fn challenges(&mut self, n: usize) -> Vec<Ext> {
        self.reserve(n);
        (0..n).map(|_| self.challenge()).collect()
    }
    /// `state || ctr_LE64`: what root_F (the bit link's commitment of this proof, link.rs) hashes.
    pub fn root_f_input(&self) -> Vec<u8> { [self.state.as_slice(), &self.ctr.to_le_bytes()].concat() }
    /// One whole digest (`gpu/transcript.py::challenge_bytes`).
    pub fn challenge_bytes(&mut self) -> [u8; 32] {
        self.slots += 1;
        if let Some(l) = &mut self.live { l.failure.get_or_insert("live prime coins are BabyBear^6 challenges only".into()); }
        self.digest()
    }
    /// root_F is computed here: exactly the rounds the session issued before `Commit` must precede it.
    pub fn live_at_root_f(&self) -> Result<(), String> {
        match &self.live {
            Some(l) if l.k != l.rec.before_commit => Err(format!(
                "R2-prime: root_F follows {} prime rounds, the session issued {} before Commit", l.k, l.rec.before_commit)),
            _ => Ok(()),
        }
    }
    /// After the proof's last challenge: no mismatch, and every recorded round consumed.
    pub fn live_finish(&self) -> Result<(), String> {
        let Some(l) = &self.live else { return Ok(()) };
        if let Some(f) = &l.failure { return Err(f.clone()); }
        if l.k != l.rec.rounds.len() {
            return Err(format!("R2-prime: {} prime rounds recorded, the proof consumed {}", l.rec.rounds.len(), l.k));
        }
        Ok(())
    }
    pub fn live_rounds(&self) -> usize { self.live.as_ref().map(|l| l.k).unwrap_or(0) }
    /// The first replay mismatch so far, if any.
    pub fn live_failure(&self) -> Option<String> { self.live.as_ref().and_then(|l| l.failure.clone()) }
}

/// `prime_coins.txt` (`tools/cell_gate.py` writes it from the session record; `gpu/transcript.py::coins_text`).
pub fn parse_coins(text: &str) -> Result<LiveCoins, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    if lines.next().map(str::trim) != Some("prime-coins live-coins/v1") { return Err("prime_coins.txt: not a live-coins/v1 record".into()); }
    let mut before = None;
    let mut rounds = Vec::new();
    for ln in lines {
        let t: Vec<&str> = ln.split_whitespace().collect();
        match t[0] {
            "before_commit" if t.len() == 2 => before = Some(t[1].parse::<usize>().map_err(|e| format!("prime_coins.txt: {e}"))?),
            "round" if t.len() >= 3 => {
                let h = t[1];
                if h.len() != 64 { return Err("prime_coins.txt: a state is 32 bytes".into()); }
                let mut state = [0u8; 32];
                for i in 0..32 { state[i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).map_err(|e| format!("prime_coins.txt: {e}"))?; }
                let n: usize = t[2].parse().map_err(|e| format!("prime_coins.txt: {e}"))?;
                let words: Vec<u32> = t[3..].iter().map(|w| w.parse::<u32>()).collect::<Result<_, _>>().map_err(|e| format!("prime_coins.txt: {e}"))?;
                if n == 0 || words.len() != DEG * n || words.iter().any(|&w| w >= P) { return Err("prime_coins.txt: a malformed round".into()); }
                rounds.push(Round { state, n, words });
            }
            _ => return Err(format!("prime_coins.txt: unknown line {ln:?}")),
        }
    }
    let before_commit = before.ok_or("prime_coins.txt: no before_commit")?;
    if before_commit == 0 || before_commit > rounds.len() { return Err("prime_coins.txt: before_commit out of range".into()); }
    Ok(LiveCoins { rounds, before_commit })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A live prover over a fake verifier seeded `seed` (the rules of `gpu/transcript.py::LiveTranscript`), running
    /// a toy protocol: a root, a reserved triple of coins, a message computed from the first coin, a reserved pair.
    /// Returns the verifier's record and the proof's messages.
    fn prove(seed: u32) -> (LiveCoins, Vec<Vec<u8>>) {
        let mut tr = Transcript::new(b"t");
        let (mut rounds, mut buf, mut seed) = (Vec::new(), std::collections::VecDeque::new(), seed);
        let mut draw = |tr: &mut Transcript, n: usize, buf: &mut std::collections::VecDeque<Ext>| {
            let words: Vec<u32> = (0..DEG * n).map(|_| { seed = seed.wrapping_mul(1103515245).wrapping_add(12345); seed % P }).collect();
            rounds.push(Round { state: tr.state, n, words: words.clone() });
            let mut s = Sha256::new();
            s.update(&tr.state);
            s.update(LIVE_TAG);
            for w in &words { s.update(&w.to_le_bytes()); }
            tr.state = s.finalize();
            *buf = words.chunks(DEG).map(|c| Ext(c.try_into().unwrap())).collect();
        };
        let m0 = b"root".to_vec();
        tr.absorb(&m0);
        draw(&mut tr, 3, &mut buf);
        let a = buf.pop_front().unwrap();
        let m1 = a.to_bytes().to_vec();
        tr.absorb(&m1);
        draw(&mut tr, 2, &mut buf);
        (LiveCoins { rounds, before_commit: 1 }, vec![m0, m1])
    }

    fn verify(rec: LiveCoins, msgs: &[Vec<u8>]) -> Result<(), String> {
        let mut tr = Transcript::replay(b"t", rec);
        tr.absorb(&msgs[0]);
        tr.reserve(3);
        let _ = tr.challenge();
        let _ = tr.challenge();
        tr.absorb(&msgs[1]);
        tr.live_at_root_f()?;
        tr.reserve(2);
        let _ = tr.challenge();
        let _ = tr.challenge();
        tr.live_finish()
    }

    #[test]
    fn honest_replay_accepts_and_rounds_group() {
        let (rec, msgs) = prove(7);
        assert_eq!(rec.rounds.iter().map(|r| r.n).collect::<Vec<_>>(), vec![3, 2]);
        verify(rec, &msgs).unwrap();
    }

    #[test]
    fn message_other_than_the_committed_one_is_rejected() {
        // the proof carries a message other than the one whose state preceded the coin (committed after the coin)
        let (rec, mut msgs) = prove(7);
        msgs[1][0] ^= 1;
        assert!(verify(rec, &msgs).unwrap_err().contains("differ from those committed"));
    }

    #[test]
    fn prime_coins_replayed_across_sessions_are_rejected() {
        // session A's proof (its messages follow A's coins) against session B's record (fresh coins)
        let (_, msgs_a) = prove(7);
        let (rec_b, msgs_b) = prove(8);
        verify(rec_b.clone(), &msgs_b).unwrap();
        assert!(verify(rec_b, &msgs_a).unwrap_err().contains("round 1"));
    }

    #[test]
    fn tampered_records_are_rejected() {
        let (rec, msgs) = prove(7);
        // a coin: every later state chains it
        let mut t = rec.clone();
        t.rounds[0].words[5] = (t.rounds[0].words[5] + 1) % P;
        assert!(verify(t, &msgs).is_err());
        // a committed state
        let mut t = rec.clone();
        t.rounds[1].state[3] ^= 1;
        assert!(verify(t, &msgs).is_err());
        // a round the proof never drew, a missing round, a split round, and root_F's position
        let mut t = rec.clone();
        t.rounds.push(t.rounds[1].clone());
        assert!(verify(t, &msgs).unwrap_err().contains("consumed"));
        let mut t = rec.clone();
        t.rounds.pop();
        assert!(verify(t, &msgs).unwrap_err().contains("no such coin"));
        let mut t = rec.clone();
        t.rounds[0].n = 2;
        t.rounds[0].words.truncate(2 * DEG);
        assert!(verify(t, &msgs).is_err());
        let mut t = rec.clone();
        t.before_commit = 2;
        assert!(verify(t, &msgs).unwrap_err().contains("root_F follows"));
    }

    #[test]
    fn coins_text_round_trips() {
        let (rec, msgs) = prove(7);
        let text = format!("prime-coins live-coins/v1\nbefore_commit 1\n{}", rec.rounds.iter().map(|r| format!("round {} {} {}\n",
            r.state.iter().map(|b| format!("{b:02x}")).collect::<String>(), r.n, r.words.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(" "))).collect::<String>());
        verify(parse_coins(&text).unwrap(), &msgs).unwrap();
        assert!(parse_coins(&text.replace("before_commit 1", "before_commit 0")).is_err());
    }
}
