//! `firewall/v0`'s release side (PROTOCOL.md §10.5): the decisions of whether an outer session of a recursive session goes
//! on, and when its proofs leave. `flock-circuit prove --firewall-state DIR --session ID --part NAME` runs each outer session
//! through [`outer_session`]. `verity_flock.firewall` is the same record in Python: the inner phase's side (`rec_live` opens,
//! finishes and stops recursive sessions) and the test driver. Both write `record.jsonl` alike, byte for byte (Python's
//! `json.dumps(ev, sort_keys=True)`; `firewall_record_vectors.json` pins it), and take the same two `flock(2)` locks:
//! `record.lock` around every read-and-append of the record, and `slot.lock`, the outer slot.
//!
//! The rules, over the record:
//! 1. A recursive session's V* statements and their order are fixed when it opens (`parts`, public, from K: L0–L6 and the
//!    algebra at K = 4096), and it ends at its first stop, inner or outer. [`Record::admit`] serves only the next
//!    statement not yet released, once: a request for any other (a later one, a released one again) is a stop at that
//!    next statement, never opened, so a statement the worker skips ends the recursive session where it stands. After a
//!    stop, and after the last statement's release, it refuses every outer session of it before anything reaches the
//!    challenger.
//! 2. Outer proofs go one at a time across streams and protocols: [`Record::slot`]. A holder that died before recording its
//!    session's outcome is recorded as that session's stop by the next one to take the slot.
//! 3. A recursive session opened with a release schedule (`release`: `start_ms`, and one public window a statement,
//!    `windows_ms`) releases statement `j`'s proofs at the end of its window, `start_ms + windows_ms[0] + ... +
//!    windows_ms[j]`, never earlier and never late: proofs ready early wait for that time, and proofs not ready by it are a
//!    stop (a missed time). The schedule is recorded at the open, before the session, and depends on nothing secret. With
//!    rule 2 it also fixes how the outer sessions of different streams interleave: each leaves at its own public time.
//! 4. A stream opened with a failure `budget` (P1's `budget`, b) stops at its b-th failure: after it every outer session on
//!    it is refused, and so is every open. A failure is a recursive session's stop (a missed time included), or a refusal
//!    of a recursive session that had neither stopped nor released its last statement, on a stream that had not stopped.
//!
//! The record is read whole at every call. A torn last line (a writer killed inside its append) fails every call, so the
//! firewall serves nothing until the developer repairs the record: it fails closed, never by skipping an event it can't
//! read.
use serde_json::{json, Value};
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub const KIND: &str = "firewall/v0";
pub const INNER: &str = "inner";

/// The firewall's state directory, shared by every process of one firewall.
pub struct Record {
    pub state: PathBuf,
}

/// An admitted outer session: its recursive session's stream, and when its proofs are released (rule 3), if scheduled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admitted {
    pub stream: String,
    pub release_ms: Option<u64>,
}

/// The outer slot, held until dropped.
pub struct Slot(#[allow(dead_code)] Held);

/// A `flock(2)` lock, held until dropped. Dropping it unlocks it before closing the file, as `verity_flock.firewall` does:
/// closing alone frees nothing while a child process forked meanwhile shares the open file, until that child execs.
struct Held(File);

impl Drop for Held {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

struct Events {
    path: PathBuf,
    list: Vec<Value>,
}

impl Events {
    fn append(&mut self, ev: Value) -> io::Result<()> {
        let mut m = match ev {
            Value::Object(m) => m,
            _ => unreachable!("an event is an object"),
        };
        m.insert("seq".into(), json!(self.list.len()));
        let ev = Value::Object(m);
        let mut f = OpenOptions::new().create(true).append(true).open(&self.path)?;
        f.write_all(format!("{}\n", py_json(&ev)).as_bytes())?;
        self.list.push(ev);
        Ok(())
    }

    fn of(&self, event: &str) -> impl Iterator<Item = &Value> {
        let event = event.to_string();
        self.list.iter().filter(move |e| e["event"] == event.as_str())
    }

    fn opened(&self, session: &str) -> Option<&Value> {
        self.of("open").find(|e| e["session"] == session)
    }

    fn stopped(&self, session: &str) -> Option<&Value> {
        self.of("stop").find(|e| e["session"] == session)
    }

    /// The stream's recorded budget: the first open's (every open of a stream records the same one).
    fn budget(&self, stream: &str) -> Option<u64> {
        self.of("open").find(|e| e["stream"] == stream).and_then(|e| e["budget"].as_u64())
    }

    /// The position of the recursive session's next statement, the first of its `parts` not released; None once all are.
    fn next(&self, open: &Value) -> Option<usize> {
        let released: std::collections::HashSet<&str> =
            self.of("release").filter(|e| e["session"] == open["session"]).filter_map(|e| e["part"].as_str()).collect();
        parts(open).iter().position(|p| !released.contains(p))
    }

    fn failures(&self, stream: &str) -> u64 {
        let budget = self.budget(stream);
        let mut mine: std::collections::HashMap<&str, Vec<&str>> = Default::default();
        let mut released: std::collections::HashSet<(&str, &str)> = Default::default();
        let mut stopped = std::collections::HashSet::new();
        let mut n = 0u64;
        for e in &self.list {
            let s = e["session"].as_str().unwrap_or_default();
            match e["event"].as_str() {
                Some("open") if e["stream"] == stream => {
                    mine.insert(s, parts(e));
                }
                Some("release") if mine.contains_key(s) => {
                    released.insert((s, e["part"].as_str().unwrap_or_default()));
                }
                Some("stop") if mine.contains_key(s) => {
                    stopped.insert(s);
                    n += 1;
                }
                Some("refused") if mine.get(s).is_some_and(|ps| !ps.iter().all(|p| released.contains(&(s, *p))))
                    && !stopped.contains(s) && budget.is_none_or(|b| n < b) => n += 1,
                _ => {}
            }
        }
        n
    }
}

/// A recursive session's statements in their order, as its `open` event records them.
fn parts(open: &Value) -> Vec<&str> {
    open["parts"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default()
}

/// A recursive session's statements (rule 1): at least one, distinct, each a name other than `inner`.
pub fn parts_ok(parts: &[&str]) -> Result<(), String> {
    let distinct = parts.iter().collect::<std::collections::HashSet<_>>().len() == parts.len();
    if parts.is_empty() || !distinct || parts.iter().any(|p| p.is_empty() || *p == INNER) {
        return Err(format!("a recursive session's statements are distinct names, at least one, none of them {INNER}, not {}",
                           py_json(&json!(parts))));
    }
    Ok(())
}

impl Record {
    pub fn new(state: impl AsRef<Path>) -> io::Result<Self> {
        std::fs::create_dir_all(state.as_ref())?;
        Ok(Record { state: state.as_ref().to_path_buf() })
    }

    fn lock_file(&self, name: &str) -> io::Result<File> {
        OpenOptions::new().create(true).append(true).open(self.state.join(name))
    }

    fn with<T>(&self, f: impl FnOnce(&mut Events) -> io::Result<T>) -> io::Result<T> {
        let lock = self.lock_file("record.lock")?;
        lock.lock()?;
        let _held = Held(lock);
        let path = self.state.join("record.jsonl");
        let list = match std::fs::read_to_string(&path) {
            Ok(text) => text.lines().filter(|l| !l.is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", path.display())))?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e),
        };
        f(&mut Events { path, list })
    }

    /// Every event so far, in order.
    pub fn events(&self) -> io::Result<Vec<Value>> {
        self.with(|ev| Ok(std::mem::take(&mut ev.list)))
    }

    /// A new recursive session on `stream`, opened by its inner session, with its V* statements in their order (`parts`,
    /// rule 1): its id, or why the firewall refuses it (the stream stopped at its budget). Every open of a stream must give
    /// the same `budget`.
    pub fn open(&self, stream: &str, parts: &[&str], budget: Option<u64>, release: Option<&Value>) -> io::Result<Result<String, String>> {
        parts_ok(parts).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        if budget == Some(0) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "a failure budget is a positive integer, not 0"));
        }
        if let Some(r) = release {
            schedule_ok(r, parts.len()).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        }
        self.with(|ev| {
            if let Some(first) = ev.of("open").find(|e| e["stream"] == stream) {
                if first["budget"].as_u64() != budget {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput,
                        format!("stream {stream}'s budget is {}, not {}", show(first["budget"].as_u64()), show(budget))));
                }
            }
            if let Some(b) = budget {
                let n = ev.failures(stream);
                if n >= b {
                    return Ok(Err(format!("stream {stream} stopped at its failure budget ({n} of {b})")));
                }
            }
            let sid = format!("r{:04}", ev.of("open").count());
            let mut e = json!({"event": "open", "session": sid, "stream": stream, "parts": parts});
            if let Some(b) = budget {
                e["budget"] = json!(b);
            }
            if let Some(r) = release {
                e["release"] = r.clone();
            }
            ev.append(e)?;
            Ok(Ok(sid))
        })
    }

    /// The recursive session's inner session finished: its outer sessions may be served.
    pub fn finish(&self, session: &str) -> io::Result<()> {
        self.with(|ev| ev.append(json!({"event": "finish", "session": session, "part": INNER})))
    }

    /// The recursive session stops at `part`: its one failure, recorded once (false if it had already stopped).
    pub fn stop(&self, session: &str, part: &str, why: &str) -> io::Result<bool> {
        self.with(|ev| {
            if ev.stopped(session).is_some() {
                return Ok(false);
            }
            ev.append(json!({"event": "stop", "session": session, "part": part, "why": why}))?;
            Ok(true)
        })
    }

    /// An outer proof of the recursive session was released (the challenger's verdict accepted it).
    pub fn release(&self, session: &str, part: &str) -> io::Result<()> {
        self.with(|ev| ev.append(json!({"event": "release", "session": session, "part": part})))
    }

    /// Whether an outer session of `session` at `part` may be served now: the recursive session is open, has not stopped,
    /// has a statement not yet released, its stream has not stopped at its budget, its inner session finished, and `part`
    /// is its next statement (rule 1). A request for any other statement is the recursive session's stop at the next one
    /// (never opened), then refused. A refusal is recorded (`refused`).
    pub fn admit(&self, session: &str, part: &str) -> io::Result<Result<Admitted, String>> {
        self.with(|ev| {
            let why = match ev.opened(session).cloned() {
                None => format!("recursive session {session} was never opened"),
                Some(open) => {
                    let stream = open["stream"].as_str().unwrap_or_default().to_string();
                    let budget = ev.budget(&stream);
                    let next = ev.next(&open);
                    if let Some(s) = ev.stopped(session) {
                        format!("recursive session {session} stopped at {}: the firewall serves nothing more of it",
                                s["part"].as_str().unwrap_or_default())
                    } else if next.is_none() {
                        format!("recursive session {session} released all its statements: the firewall serves nothing more of it")
                    } else if let Some(b) = budget.filter(|&b| ev.failures(&stream) >= b) {
                        format!("stream {stream} stopped at its failure budget ({} of {b})", ev.failures(&stream))
                    } else if !ev.of("finish").any(|e| e["session"] == session) {
                        format!("recursive session {session}: its inner session has not finished")
                    } else {
                        let j = next.expect("a statement not yet released");
                        let nxt = parts(&open)[j].to_string();
                        if nxt != part {
                            ev.append(json!({"event": "stop", "session": session, "part": nxt,
                                             "why": format!("never opened: {part} was asked in its place")}))?;
                            format!("recursive session {session} stopped at {nxt}: {part} is not its next statement")
                        } else {
                            return Ok(Ok(Admitted { stream, release_ms: release_ms(&open["release"], j) }));
                        }
                    }
                }
            };
            ev.append(json!({"event": "refused", "session": session, "part": part}))?;
            Ok(Err(why))
        })
    }

    /// The failures on `stream` so far (rule 4).
    pub fn failures(&self, stream: &str) -> io::Result<u64> {
        self.with(|ev| Ok(ev.failures(stream)))
    }

    /// The firewall's one outer slot, held until the returned [`Slot`] drops. A session that finds it taken records `wait`,
    /// calls `waiting`, and blocks until the holder leaves it. Once it holds the slot, a `slot` event whose session has no
    /// outcome after it (its holder died) is recorded as that session's stop, and then `slot` is recorded.
    pub fn slot(&self, session: &str, part: &str, waiting: impl FnOnce()) -> io::Result<Slot> {
        let f = self.lock_file("slot.lock")?;
        match f.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                self.with(|ev| ev.append(json!({"event": "wait", "session": session, "part": part})))?;
                waiting();
                f.lock()?;
            }
            Err(TryLockError::Error(e)) => return Err(e),
        }
        let held = Held(f);
        self.with(|ev| {
            if let Some((seq, s, p)) = orphan(&ev.list) {
                if ev.stopped(&s).is_none() {
                    ev.append(json!({"event": "stop", "session": s, "part": p,
                                     "why": format!("the slot's holder ended before its outcome (slot {seq})")}))?;
                }
            }
            ev.append(json!({"event": "slot", "session": session, "part": part}))
        })?;
        Ok(Slot(held))
    }
}

/// The last `slot` event, if no outcome of its session at its part (`release`, `stop` or `refused`) follows it.
fn orphan(list: &[Value]) -> Option<(u64, String, String)> {
    let k = list.iter().rposition(|e| e["event"] == "slot")?;
    let (s, p) = (&list[k]["session"], &list[k]["part"]);
    let done = list[k + 1..].iter().any(|e| {
        e["session"] == *s && (matches!(e["event"].as_str(), Some("release" | "refused")) && e["part"] == *p || e["event"] == "stop")
    });
    (!done).then(|| (list[k]["seq"].as_u64().unwrap_or_default(), s.as_str().unwrap_or_default().into(), p.as_str().unwrap_or_default().into()))
}

fn show(b: Option<u64>) -> String {
    b.map_or("None".into(), |b| b.to_string())
}

/// A release schedule (rule 3) for a recursive session of `n` statements: `{"start_ms": S, "windows_ms": [one positive
/// width a statement]}`.
pub fn schedule_ok(r: &Value, n: usize) -> Result<(), String> {
    let keys = r.as_object().map_or(0, |m| m.len());
    let windows = r["windows_ms"].as_array().map(|a| (a.len(), a.iter().filter_map(Value::as_u64).filter(|&w| w > 0).count()));
    if keys != 2 || r["start_ms"].as_u64().is_none() || windows != Some((n, n)) {
        return Err(format!("a release schedule is {{\"start_ms\": S, \"windows_ms\": [one width > 0 for each of the {n} statements]}}, not {}",
                           py_json(r)));
    }
    Ok(())
}

/// When statement `j`'s proofs are released under `release` (rule 3): the end of its window, `start_ms` plus the widths of
/// the windows up to and including its own; None when the recursive session has no schedule.
pub fn release_ms(release: &Value, j: usize) -> Option<u64> {
    if release.is_null() {
        return None;
    }
    let widths: u64 = release["windows_ms"].as_array().map_or(0, |w| w[..=j].iter().filter_map(Value::as_u64).sum());
    Some(release["start_ms"].as_u64().unwrap_or_default() + widths)
}

/// JSON as Python's `json.dumps(v, sort_keys=True)` writes it: `", "` and `": "` between items, keys sorted, every
/// non-ASCII character escaped.
pub fn py_json(v: &Value) -> String {
    let mut out = String::new();
    write_py(v, &mut out);
    out
}

fn write_py(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (k, x) in a.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                write_py(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (k, key) in keys.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                write_str(key, out);
                out.push_str(": ");
                write_py(&m[*key], out);
            }
            out.push('}');
        }
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if !(' '..='~').contains(&c) => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The firewall's clock (rule 3), injected so that tests read none.
pub trait Clock {
    fn now_ms(&self) -> u64;
    fn sleep_until_ms(&self, t: u64);
}

/// Wall-clock time, in milliseconds since the Unix epoch.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
    }
    fn sleep_until_ms(&self, t: u64) {
        let now = self.now_ms();
        if t > now {
            std::thread::sleep(std::time::Duration::from_millis(t - now));
        }
    }
}

/// An outer session's outcome, as `verity_flock.firewall.relay_session` reports it: `release`, `stop` or `refused`.
#[derive(Debug)]
pub struct Outcome<T> {
    pub outcome: &'static str,
    pub why: Option<String>,
    /// What `send` returned, when the proofs left.
    pub sent: Option<T>,
}

impl<T> Outcome<T> {
    pub fn json(&self) -> Value {
        json!({"outcome": self.outcome, "why": self.why})
    }
}

/// One outer session of `session` at `part`, held to the module's rules: take the slot, then admit the session; `prove`
/// runs it up to the barrier where its proofs would leave (`Err`: it stopped before, and why); `ready` is the send check's
/// verdict on what it proved (`Err`: refused, and why); then, at its release time if it has one (a missed time is a stop),
/// `send` sends the proofs and `Finish` and returns whether the challenger's verdict accepts them, and what to report.
/// Dropping a proved session unsent must cut its connection: every stop the firewall makes reaches the challenger as a cut.
/// A panic anywhere in `prove`, `ready` or `send` is a stop, never the firewall's end.
pub fn outer_session<P, T>(rec: &Record, session: &str, part: &str, clock: &dyn Clock, waiting: impl FnOnce(),
                           prove: impl FnOnce() -> Result<P, String>, ready: impl FnOnce(&P) -> Result<(), String>,
                           send: impl FnOnce(P) -> (bool, T)) -> io::Result<Outcome<T>> {
    let _slot = rec.slot(session, part, waiting)?;
    let admitted = match rec.admit(session, part)? {
        Ok(a) => a,
        Err(why) => return Ok(Outcome { outcome: "refused", why: Some(why), sent: None }),
    };
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(bool, T), String> {
        let p = prove()?;
        ready(&p).map_err(|e| format!("the send check refused: {e}"))?;
        if let Some(t) = admitted.release_ms {
            let now = clock.now_ms();
            if now > t {
                return Err(format!("missed its release time ({now} ms, after {t} ms)"));
            }
            clock.sleep_until_ms(t);
        }
        Ok(send(p))
    }));
    let (outcome, why, sent) = match run {
        Ok(Ok((true, v))) => {
            rec.release(session, part)?;
            return Ok(Outcome { outcome: "release", why: None, sent: Some(v) });
        }
        Ok(Ok((false, v))) => ("stop", "the challenger's verdict rejects it".to_string(), Some(v)),
        Ok(Err(why)) => ("stop", why, None),
        Err(e) => {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            ("stop", format!("the firewall's fault (a panic: {})", msg.chars().take(200).collect::<String>()), None)
        }
    };
    rec.stop(session, part, &why)?;
    Ok(Outcome { outcome, why: Some(why), sent })
}

/// One op of the shared vectors, run on this record: its result as the vectors write it.
#[cfg(test)]
fn run_op(rec: &Record, op: &[Value]) -> Value {
    let s = |k: usize| op[k].as_str().unwrap_or_default();
    match s(0) {
        "open" => {
            let ps: Vec<&str> = op[2].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
            match rec.open(s(1), &ps, op[3].as_u64(), (!op[4].is_null()).then_some(&op[4])) {
                Ok(Ok(sid)) => json!(sid),
                Ok(Err(why)) => json!({"refused": why}),
                Err(e) => json!({"error": e.to_string()}),
            }
        }
        "finish" => {
            rec.finish(s(1)).unwrap();
            Value::Null
        }
        "stop" => json!(rec.stop(s(1), s(2), s(3)).unwrap()),
        "release" => {
            rec.release(s(1), s(2)).unwrap();
            Value::Null
        }
        "admit" => match rec.admit(s(1), s(2)).unwrap() {
            Ok(a) => json!({"release_ms": a.release_ms}),
            Err(why) => json!({"refused": why}),
        },
        "failures" => json!(rec.failures(s(1)).unwrap()),
        "slot" => {
            drop(rec.slot(s(1), s(2), || {}).unwrap());
            Value::Null
        }
        other => panic!("unknown op {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct Fake {
        now: Cell<u64>,
        slept: Cell<Option<u64>>,
    }

    impl Clock for Fake {
        fn now_ms(&self) -> u64 {
            self.now.get()
        }
        fn sleep_until_ms(&self, t: u64) {
            self.slept.set(Some(t));
            self.now.set(self.now.get().max(t));
        }
    }

    fn fake(now: u64) -> Fake {
        Fake { now: Cell::new(now), slept: Cell::new(None) }
    }

    fn scratch(name: &str) -> Record {
        let d = std::env::temp_dir().join(format!("flock-release-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        Record::new(d).unwrap()
    }

    fn text(rec: &Record) -> String {
        std::fs::read_to_string(rec.state.join("record.jsonl")).unwrap_or_default()
    }

    #[test]
    fn the_shared_vectors_hold() {
        let v: Value = serde_json::from_str(include_str!("firewall_record_vectors.json")).unwrap();
        assert_eq!(v["kind"], "firewall/v0/vectors");
        for case in v["cases"].as_array().unwrap() {
            let rec = scratch(&format!("v-{}", case["name"].as_str().unwrap()));
            for (op, want) in case["ops"].as_array().unwrap().iter().zip(case["results"].as_array().unwrap()) {
                assert_eq!(&run_op(&rec, op.as_array().unwrap()), want, "{}: {op}", case["name"]);
            }
            assert_eq!(text(&rec), case["record"].as_str().unwrap(), "{}", case["name"]);
        }
    }

    #[test]
    fn py_json_matches_python() {
        let v = json!({"b": [1, "x"], "a": {"z": null, "y": true}, "c": "é\u{1F600}\"\\\n\u{1}~\u{7f}"});
        assert_eq!(py_json(&v), r#"{"a": {"y": true, "z": null}, "b": [1, "x"], "c": "\u00e9\ud83d\ude00\"\\\n\u0001~\u007f"}"#);
    }

    const PARTS: [&str; 3] = ["L0", "L1", "alg"];

    fn opened(rec: &Record, release: Option<Value>, budget: Option<u64>) -> String {
        let sid = rec.open("s", &PARTS, budget, release.as_ref()).unwrap().unwrap();
        rec.finish(&sid).unwrap();
        sid
    }

    fn kinds(rec: &Record) -> Vec<String> {
        rec.events().unwrap().iter().map(|e| format!("{}:{}", e["event"].as_str().unwrap(), e["part"].as_str().unwrap_or("-"))).collect()
    }

    fn honest(rec: &Record, sid: &str, part: &str) -> &'static str {
        outer_session(rec, sid, part, &fake(0), || {}, || Ok(()), |_| Ok(()), |_| (true, ())).unwrap().outcome
    }

    /// An outer session the firewall must refuse before its prover runs (a prover that ran would be a stop, not a refusal).
    fn refused(rec: &Record, sid: &str, part: &str) -> String {
        let o = outer_session(rec, sid, part, &fake(0), || {}, || -> Result<(), String> { panic!("{part} was proved") }, |_| Ok(()),
                              |_| (true, ())).unwrap();
        assert_eq!(o.outcome, "refused", "{part}: {:?}", o.why);
        o.why.unwrap()
    }

    fn stop_of(rec: &Record, sid: &str) -> (String, String) {
        let ev = rec.events().unwrap();
        let s = ev.iter().find(|e| e["event"] == "stop" && e["session"] == sid).expect("a stop");
        (s["part"].as_str().unwrap().into(), s["why"].as_str().unwrap().into())
    }

    #[test]
    fn an_honest_session_is_released_and_a_stop_ends_the_recursive_session() {
        let rec = scratch("honest");
        let sid = opened(&rec, None, None);
        let c = fake(0);
        let o = outer_session(&rec, &sid, "L0", &c, || {}, || Ok(()), |_| Ok(()), |_| (true, 7)).unwrap();
        assert_eq!((o.outcome, o.sent), ("release", Some(7)));
        let o = outer_session(&rec, &sid, "L1", &c, || {}, || Ok(()), |_| Err("root 96d8".into()), |_| -> (bool, i32) { panic!("sent") }).unwrap();
        assert_eq!(o.outcome, "stop");
        assert_eq!(o.why.as_deref(), Some("the send check refused: root 96d8"));
        assert_eq!(refused(&rec, &sid, "alg"), format!("recursive session {sid} stopped at L1: the firewall serves nothing more of it"));
        assert_eq!(kinds(&rec), ["open:-", "finish:inner", "slot:L0", "release:L0", "slot:L1", "stop:L1", "slot:alg", "refused:alg"]);
    }

    /// The Rust release side admits an outer session before its prover connects, so L0 cut
    /// before its first request (its connection refused or reset) is L0's stop; L1 is then refused unproved.
    #[test]
    fn l0_cut_before_its_first_request_then_l1_is_refused() {
        let rec = scratch("order-cut");
        let sid = opened(&rec, None, None);
        let o = outer_session(&rec, &sid, "L0", &fake(0), || {}, || -> Result<(), String> { Err("the challenger at 127.0.0.1:9: reset".into()) },
                              |_| Ok(()), |_| (true, ())).unwrap();
        assert_eq!(o.outcome, "stop");
        assert_eq!(refused(&rec, &sid, "L1"), format!("recursive session {sid} stopped at L0: the firewall serves nothing more of it"));
        assert_eq!(kinds(&rec), ["open:-", "finish:inner", "slot:L0", "stop:L0", "slot:L1", "refused:L1"]);
        assert_eq!(rec.failures("s").unwrap(), 1);
    }

    /// A statement asked before its turn is refused, and the statement it skipped, never
    /// opened, is the recursive session's stop where it stands. L0 is then refused too.
    #[test]
    fn l1_asked_before_l0_is_refused() {
        let rec = scratch("order-skip");
        let sid = opened(&rec, None, None);
        assert_eq!(refused(&rec, &sid, "L1"), format!("recursive session {sid} stopped at L0: L1 is not its next statement"));
        assert_eq!(stop_of(&rec, &sid), ("L0".into(), "never opened: L1 was asked in its place".into()));
        assert_eq!(refused(&rec, &sid, "L0"), format!("recursive session {sid} stopped at L0: the firewall serves nothing more of it"));
        assert_eq!(kinds(&rec), ["open:-", "finish:inner", "slot:L1", "stop:L0", "refused:L1", "slot:L0", "refused:L0"]);
        assert_eq!(rec.failures("s").unwrap(), 1);
    }

    /// A released statement is not served again. Its repeat is refused and the recursive
    /// session stops at the statement after it, never opened.
    #[test]
    fn l0_asked_again_after_its_release_is_refused() {
        let rec = scratch("order-repeat");
        let sid = opened(&rec, None, None);
        assert_eq!(honest(&rec, &sid, "L0"), "release");
        assert_eq!(refused(&rec, &sid, "L0"), format!("recursive session {sid} stopped at L1: L0 is not its next statement"));
        assert_eq!(stop_of(&rec, &sid), ("L1".into(), "never opened: L0 was asked in its place".into()));
        assert_eq!(refused(&rec, &sid, "L1"), format!("recursive session {sid} stopped at L1: the firewall serves nothing more of it"));
        assert_eq!(rec.failures("s").unwrap(), 1);
    }

    #[test]
    fn a_recursive_session_released_whole_serves_nothing_more_and_that_is_no_failure() {
        let rec = scratch("whole");
        let sid = opened(&rec, None, Some(1));
        for p in PARTS {
            assert_eq!(honest(&rec, &sid, p), "release", "{p}");
        }
        for p in ["L0", "alg", "L9"] {
            assert_eq!(refused(&rec, &sid, p), format!("recursive session {sid} released all its statements: the firewall serves nothing more of it"));
        }
        assert!(!rec.events().unwrap().iter().any(|e| e["event"] == "stop"));
        assert_eq!(rec.failures("s").unwrap(), 0);
        assert!(rec.open("s", &PARTS, Some(1), None).unwrap().is_ok());
    }

    #[test]
    fn each_way_an_outer_session_stops_is_one_stop() {
        for (name, prove_err, verdict, panics) in [("before", true, true, false), ("verdict", false, false, false), ("panic", false, true, true)] {
            let rec = scratch(&format!("stops-{name}"));
            let sid = opened(&rec, None, None);
            let o = outer_session(&rec, &sid, "L0", &fake(0), || {},
                                  || if prove_err { Err("register refused".to_string()) } else { Ok(()) }, |_| Ok(()),
                                  |_| { assert!(!panics, "the challenger's connection broke"); (verdict, ()) }).unwrap();
            assert_eq!(o.outcome, "stop", "{name}");
            assert!(rec.admit(&sid, "L1").unwrap().is_err(), "{name}");
            assert_eq!(rec.events().unwrap().iter().filter(|e| e["event"] == "stop").count(), 1, "{name}");
        }
    }

    #[test]
    fn proofs_leave_at_their_release_time_and_a_missed_time_is_a_stop() {
        let rec = scratch("schedule");
        let sched = json!({"start_ms": 1000, "windows_ms": [500, 300, 900]});
        let sid = opened(&rec, Some(sched), None);
        for (part, now, at) in [("L0", 1200, 1500), ("L1", 1600, 1800)] {
            let c = fake(now);
            let o = outer_session(&rec, &sid, part, &c, || {}, || Ok(()), |_| Ok(()), |_| (true, ())).unwrap();
            assert_eq!((o.outcome, c.slept.get()), ("release", Some(at)), "{part}");
        }
        let c = fake(2701);
        let o = outer_session(&rec, &sid, "alg", &c, || {}, || Ok(()), |_| Ok(()), |_| -> (bool, ()) { panic!("sent late") }).unwrap();
        assert_eq!((o.outcome, o.why.as_deref()), ("stop", Some("missed its release time (2701 ms, after 2700 ms)")));
        assert_eq!(c.slept.get(), None);
        for bad in [json!({"start_ms": 0, "windows_ms": [1, 1, 0]}), json!({"start_ms": 0, "windows_ms": [1, 1]}),
                    json!({"start_ms": 0, "parts": PARTS, "windows_ms": [1, 1, 1]}), json!({"start_ms": 0}), json!({"start_ms": -1, "windows_ms": [1, 1, 1]})] {
            assert!(rec.open("s", &PARTS, None, Some(&bad)).is_err(), "{bad}");
        }
        for bad in [&[][..], &["L0", "L0"], &["L0", INNER], &[""]] {
            assert!(rec.open("s", bad, None, None).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_stream_stops_at_its_budget() {
        let rec = scratch("budget");
        let a = opened(&rec, None, Some(2));
        let b = opened(&rec, None, Some(2));
        rec.stop(&a, "L0", "cut").unwrap();
        assert_eq!(rec.failures("s").unwrap(), 1);
        assert!(rec.admit(&b, "L0").unwrap().is_ok());
        let c = rec.open("s", &PARTS, Some(2), None).unwrap().unwrap();
        assert!(rec.admit(&c, "L0").unwrap().is_err()); // its inner session has not finished: the second failure
        assert_eq!(rec.failures("s").unwrap(), 2);
        assert_eq!(rec.admit(&b, "L0").unwrap(), Err("stream s stopped at its failure budget (2 of 2)".into()));
        assert_eq!(rec.failures("s").unwrap(), 2);
        assert_eq!(rec.open("s", &PARTS, Some(2), None).unwrap(), Err("stream s stopped at its failure budget (2 of 2)".into()));
        assert!(rec.open("s", &PARTS, Some(3), None).is_err());
    }

    #[test]
    fn outer_proofs_go_one_at_a_time_and_a_dead_holder_is_a_stop() {
        let rec = scratch("slot");
        let (a, b) = (opened(&rec, None, None), opened(&rec, None, None));
        let held = rec.slot(&a, "L0", || {}).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let state = rec.state.clone();
        let t = std::thread::spawn(move || {
            let rec = Record::new(state).unwrap();
            outer_session(&rec, &b, "L0", &fake(0), || tx.send(()).unwrap(), || Ok(()), |_| Ok(()), |_| (true, ())).unwrap().outcome
        });
        rx.recv().unwrap();
        drop(held); // a's holder ends without an outcome
        assert_eq!(t.join().unwrap(), "release");
        let ev = rec.events().unwrap();
        let kinds: Vec<&str> = ev.iter().skip(4).map(|e| e["event"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["slot", "wait", "stop", "slot", "release"]);
        assert_eq!(ev[6]["session"], a.as_str());
        assert_eq!(ev[6]["why"], "the slot's holder ended before its outcome (slot 4)");
    }

    #[test]
    fn a_dropped_slot_is_free_while_a_copy_of_its_file_lives() {
        let rec = scratch("slot-copy");
        let held = rec.slot("a", "L0", || {}).unwrap();
        let copy = held.0.0.try_clone().unwrap(); // what a child process forked meanwhile keeps until it execs
        drop(held);
        drop(rec.slot("b", "L0", || panic!("the slot outlived its holder")).unwrap());
        drop(copy);
        let kinds: Vec<String> = rec.events().unwrap().iter().map(|e| e["event"].as_str().unwrap().into()).collect();
        assert_eq!(kinds, ["slot", "stop", "slot"]);
    }
}
