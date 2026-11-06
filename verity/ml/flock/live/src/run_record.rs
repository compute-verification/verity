//! The challenger's interactive-run record of one recursive run (`verity/rec-run/v1`), as its coin servers write it: one
//! event a line, in order. The challenger's operator writes the staging (V*'s shared rows and shared set) before any
//! session and the close after the last; between them each `flock-circuit serve --rec-run FILE --rec-part J` appends an
//! `open` for every `Hello` it answers with coins, before that answer (which carries the session's fresh coin key) leaves.
//! `flock-verify rec-compose --rec-run` reads the events in this order (`Flock.RecCompose.checkRun`): a part opened twice
//! is a retry, and is refused.

use std::io::Write;
use std::sync::{Mutex, PoisonError};

use serde_json::json;

use crate::{Req, Resp, Server, hex};

/// One coin server's side of the run's record: the file the operator staged, and the V* part its sessions are opened for.
pub struct RunLog {
    path: std::path::PathBuf,
    part: usize,
    /// one line at a time from this server's sessions; other servers' lines interleave whole (one `O_APPEND` write each)
    lock: Mutex<()>,
}

impl RunLog {
    /// The record at `path`, which must exist: the operator stages it before any session, so a server never starts one.
    pub fn new(path: impl Into<std::path::PathBuf>, part: usize) -> Result<Self, String> {
        let path = path.into();
        if !path.is_file() {
            return Err(format!("--rec-run {}: no run record there: the challenger's operator stages it before any session",
                               path.display()));
        }
        Ok(Self { path, part, lock: Mutex::new(()) })
    }

    /// `s.handle(req)`, and a `Hello` it answers with coins logged as `session`'s open, with the session's coin-tree key,
    /// before the answer is returned. A log that isn't written whole turns the answer into a refusal: no coin key leaves
    /// that the record doesn't hold.
    pub fn handle(&self, s: &mut Server, session: &str, req: Req) -> Resp {
        let hello = matches!(req, Req::Hello(_));
        let resp = s.handle(req);
        if !hello || !matches!(resp, Resp::Coins(_)) {
            return resp;
        }
        match s.coin_key().ok_or("its answer has no coin-tree key (--rec-run needs the keyed coin tree)".to_string())
            .and_then(|k| self.open(session, k)) {
            Ok(()) => resp,
            Err(e) => Resp::Refused(format!("run record: {e}")),
        }
    }

    fn open(&self, session: &str, key: &[u8]) -> Result<(), String> {
        let line = format!("{}\n", json!({"event": "open", "session": session, "rec_part": self.part, "coin_key": hex(key)}));
        let _whole = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        let at = self.path.display();
        let mut f = std::fs::OpenOptions::new().append(true).open(&self.path).map_err(|e| format!("{at}: {e}"))?;
        match f.write(line.as_bytes()) {
            Ok(n) if n == line.len() => f.sync_data().map_err(|e| format!("{at}: {e}")),
            Ok(n) => Err(format!("{at}: {n} of the open's {} bytes written", line.len())),
            Err(e) => Err(format!("{at}: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Binding, LinkCtx, ReplayChallenger, SessionConfig, StatementVerifier, TableSpec, coin_tree, rep_stream};
    use serde_json::Value;

    struct NoProofs;

    impl StatementVerifier for NoProofs {
        fn verify(&self, _: &str, _: usize, _: &[u8], _: &mut ReplayChallenger, _: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
            Err("no proofs in this test".into())
        }
        fn domain(&self, _: &str, _: usize) -> Vec<u8> {
            Vec::new()
        }
    }

    fn cfg(tree: bool) -> SessionConfig {
        let mut c = SessionConfig::flock_128_r2(vec![TableSpec { name: "t".into(), binding: Binding::R1cs { statement_digest: [0; 32] } }]);
        if tree {
            c.coin_tree = Some(coin_tree::Spec { streams: vec![coin_tree::SESSION.into(), rep_stream("t", 0), rep_stream("t", 1)], rounds_log: 2, block: 3 });
        } else {
            c.coin_seed = true;
        }
        c
    }

    fn hello(tree: bool) -> Req {
        Req::Hello(if tree { cfg(true).hello_with_nonce(&[7; coin_tree::NONCE_BYTES]) } else { cfg(false).hello() })
    }

    fn staged(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("fc-run-record-{}-{name}.jsonl", std::process::id()));
        std::fs::write(&p, "{\"event\":\"stage\"}\n").unwrap();
        p
    }

    fn lines(p: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(p).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    /// Each server's Hello is logged once, after the staging, as its session's open for the server's part with the coin
    /// key its session record names; a refused second Hello logs nothing. Two servers on one record (two parts) append
    /// whole lines from several threads at once.
    #[test]
    fn each_hello_answered_with_coins_is_logged_with_its_sessions_coin_key() {
        let p = staged("opens");
        let (a, b) = (RunLog::new(&p, 0).unwrap(), RunLog::new(&p, 3).unwrap());
        let keys: Vec<(String, usize, String)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..8).map(|i| {
                let (log, part) = if i % 2 == 0 { (&a, 0) } else { (&b, 3) };
                sc.spawn(move || {
                    let mut s = Server::try_new(cfg(true), Box::new(NoProofs)).unwrap();
                    let id = format!("l{i:04}-1");
                    assert!(matches!(log.handle(&mut s, &id, hello(true)), Resp::Coins(ref w) if w.len() == 20));
                    assert!(matches!(log.handle(&mut s, &id, hello(true)), Resp::Refused(ref e) if e.contains("a second Hello")));
                    (id, part, s.record_json()["coin_tree"]["key"].as_str().unwrap().to_string())
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let got = lines(&p);
        assert_eq!(got[0], json!({"event": "stage"}));
        assert_eq!(got.len(), 9);
        for (id, part, key) in keys {
            assert_eq!(key.len(), 2 * coin_tree::KEY_BYTES);
            let want = json!({"event": "open", "session": id, "rec_part": part, "coin_key": key});
            assert_eq!(got.iter().filter(|e| **e == want).count(), 1, "{want}");
        }
        std::fs::remove_file(&p).unwrap();
    }

    /// A server with no staged record never starts; a Hello answered with no coin-tree key (M0's seed commitment) is
    /// refused rather than answered unlogged, and so is one whose record can't be written, which it never creates.
    #[test]
    fn a_hello_the_record_cannot_hold_is_refused() {
        let gone = std::env::temp_dir().join(format!("fc-run-record-{}-none.jsonl", std::process::id()));
        assert!(matches!(RunLog::new(&gone, 0), Err(e) if e.contains("no run record there")));
        let p = staged("seed");
        let log = RunLog::new(&p, 1).unwrap();
        let mut s = Server::try_new(cfg(false), Box::new(NoProofs)).unwrap();
        assert!(matches!(log.handle(&mut s, "l0000-1", hello(false)), Resp::Refused(ref e) if e.contains("no coin-tree key")));
        assert_eq!(lines(&p).len(), 1);
        std::fs::remove_file(&p).unwrap();
        let mut s = Server::try_new(cfg(true), Box::new(NoProofs)).unwrap();
        assert!(matches!(log.handle(&mut s, "l0001-1", hello(true)), Resp::Refused(ref e) if e.starts_with("run record: ")));
        assert!(!p.exists());
    }
}
