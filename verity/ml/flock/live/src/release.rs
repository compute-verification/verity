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
//! What the rules read of the record is kept beside it in `index/` ([`Index`]: each recursive session's open, stop, finish
//! and releases, each stream's first open and failures, the last `slot`, the counts), and a call reads only what was
//! appended to the record since the index last read it, by whichever process. The index is rebuilt from the record's start
//! when the record was replaced, cut shorter or no longer holds the last bytes the index read, when the machine has booted
//! since the index was written, or when a writer of it ended mid-write; an edit further back that leaves the record no
//! shorter isn't seen. Since the firewall only appends, the verdicts are the ones a read of the whole record gives. A record
//! the index can't answer is read whole at every call, as before, and each rule's read is a scan of it ([`Events`]): one
//! whose last line has no newline yet, or one holding a line the firewall doesn't write or a name over [`KEY_MAX`] bytes.
//! A torn last line (a writer killed inside its append) fails every call, so the firewall serves nothing until the
//! developer repairs the record: it fails closed, never by skipping an event it can't read.
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
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

/// The record read whole, each rule's read a scan of it.
struct Events {
    path: PathBuf,
    list: Vec<Value>,
}

impl Events {
    fn read(path: &Path) -> io::Result<Self> {
        let list: Vec<Value> = match std::fs::read_to_string(path) {
            Ok(text) => text.lines().filter(|l| !l.is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", path.display())))?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e),
        };
        #[cfg(test)]
        PROBE.with(|p| p.borrow_mut().parsed += list.len() as u64);
        Ok(Events { path: path.to_path_buf(), list })
    }

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

/// The record as the rules read it in one call, under `record.lock`: the index's answers, or a scan of the record read
/// whole (with the index, when it read as far as it could, to be written back).
enum View {
    Index(Index),
    Whole(Events, Option<Index>),
}

impl View {
    fn opens(&self) -> u64 {
        match self {
            View::Index(ix) => ix.head.opens,
            View::Whole(ev, _) => ev.of("open").count() as u64,
        }
    }

    fn opened(&mut self, session: &str) -> io::Result<Option<Value>> {
        match self {
            View::Index(ix) => Ok(ix.session(session)?.open.clone()),
            View::Whole(ev, _) => Ok(ev.opened(session).cloned()),
        }
    }

    fn stopped(&mut self, session: &str) -> io::Result<Option<Value>> {
        match self {
            View::Index(ix) => Ok(ix.session(session)?.stop.clone()),
            View::Whole(ev, _) => Ok(ev.stopped(session).cloned()),
        }
    }

    fn finished(&mut self, session: &str) -> io::Result<bool> {
        match self {
            View::Index(ix) => Ok(ix.session(session)?.finished),
            View::Whole(ev, _) => Ok(ev.of("finish").any(|e| e["session"] == session)),
        }
    }

    /// The stream's first open, if it has one: the budget it recorded.
    fn first(&mut self, stream: &str) -> io::Result<Option<Option<u64>>> {
        match self {
            View::Index(ix) => Ok(ix.stream(stream)?.as_ref().map(|t| t.budget)),
            View::Whole(ev, _) => Ok(ev.of("open").find(|e| e["stream"] == stream).map(|e| e["budget"].as_u64())),
        }
    }

    fn budget(&mut self, stream: &str) -> io::Result<Option<u64>> {
        Ok(self.first(stream)?.flatten())
    }

    fn next(&mut self, open: &Value) -> io::Result<Option<usize>> {
        match self {
            View::Index(ix) => {
                let released = &ix.session(open["session"].as_str().unwrap_or_default())?.released;
                Ok(parts(open).iter().position(|p| !released.iter().any(|r| r == p)))
            }
            View::Whole(ev, _) => Ok(ev.next(open)),
        }
    }

    fn failures(&mut self, stream: &str) -> io::Result<u64> {
        match self {
            View::Index(ix) => Ok(ix.stream(stream)?.as_ref().map_or(0, |t| t.n)),
            View::Whole(ev, _) => Ok(ev.failures(stream)),
        }
    }

    fn orphan(&self) -> Option<(u64, String, String)> {
        match self {
            View::Index(ix) => ix.head.slot.clone().filter(|_| !ix.head.slot_done),
            View::Whole(ev, _) => orphan(&ev.list),
        }
    }

    fn append(&mut self, ev: Value) -> io::Result<()> {
        match self {
            View::Index(ix) => ix.append(ev),
            View::Whole(w, _) => w.append(ev),
        }
    }
}

/// The longest session or stream name the index keeps (its files are named by it); a record holding a longer one is
/// read whole.
pub const KEY_MAX: usize = 100;
/// How many of the last bytes it read the index keeps, to tell that the record still holds them.
const LAST: usize = 256;

/// What the index keeps of a recursive session: its first open and first stop, whether its inner session finished, the
/// statements released, and for each stream it was opened on, what that stream's failures read of it.
#[derive(Default)]
struct Session {
    open: Option<Value>,
    stop: Option<Value>,
    finished: bool,
    released: Vec<String>,
    member: Vec<Member>,
}

/// A recursive session as [`Events::failures`] reads it on one of its streams: its statements as its last open there gave
/// them, and what was released and whether it stopped after its first open there.
struct Member {
    stream: String,
    parts: Vec<String>,
    released: Vec<String>,
    stopped: bool,
}

/// A stream that has been opened: its first open's budget, and its failures so far.
struct Tally {
    budget: Option<u64>,
    n: u64,
}

/// Where the index stands in the record: the record's device and inode, how much of it was read (`offset`) and its last
/// bytes, whether every line so far is one the index answers for (`regular`), the counts, and the last `slot` (its seq,
/// session and part), and whether an outcome of it followed.
struct Head {
    file: Option<(u64, u64)>,
    offset: u64,
    last: Vec<u8>,
    regular: bool,
    count: u64,
    opens: u64,
    slot: Option<(u64, String, String)>,
    slot_done: bool,
}

/// What the rules read of the record, kept as it grows, an event at a time, in `index/` beside it: `head.json`, and a file
/// for each recursive session and each opened stream (`s` or `t`, then its name in hex). Each call loads the files it
/// reads and writes back the ones that changed, under `record.lock`: first the head marked dirty, then the files, then
/// the head, each written whole and renamed into place, so a writer that ends mid-write leaves a dirty head, and the next
/// call rebuilds the index from the record.
struct Index {
    dir: PathBuf,
    path: PathBuf,
    boot: String,
    head: Head,
    /// rebuilt from the record's start: no file of the index is read, and every one is written anew
    fresh: bool,
    /// a file of the index wouldn't read while an event was applied: the next call rebuilds it
    broken: bool,
    /// the head changed, so it is written back
    moved: bool,
    sessions: HashMap<String, Session>,
    streams: HashMap<String, Option<Tally>>,
    changed: (HashSet<String>, HashSet<String>),
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    s.len().is_multiple_of(2).then_some(())?;
    (0..s.len()).step_by(2).map(|k| u8::from_str_radix(s.get(k..k + 2)?, 16).ok()).collect()
}

fn strings(v: &Value) -> Option<Vec<String>> {
    v.as_array()?.iter().map(|x| x.as_str().map(String::from)).collect()
}

fn bad(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("the firewall's index: {what}"))
}

/// The machine's boot: an index written before it may have lost writes the page cache held.
fn boot_id() -> String {
    std::fs::read_to_string("/proc/sys/kernel/random/boot_id").unwrap_or_default().trim().to_string()
}

/// Whether `e` is an event as the firewall writes it, one whose reads [`Index`] answers as [`Events`] does: an object with
/// a string `session`, and either an `open` with a string `stream` and a list of string `parts`, or one of the other
/// events with a string `part`; every name at most [`KEY_MAX`] bytes.
fn regular(e: &Value) -> bool {
    let name = |v: &Value| v.as_str().is_some_and(|s| s.len() <= KEY_MAX);
    if !name(&e["session"]) {
        return false;
    }
    match e["event"].as_str() {
        Some("open") => name(&e["stream"]) && e["parts"].as_array().is_some_and(|a| a.iter().all(Value::is_string)),
        Some("finish" | "stop" | "release" | "refused" | "wait" | "slot") => e["part"].is_string(),
        _ => false,
    }
}

/// A stream a loaded session was opened on.
fn tally<'a>(streams: &'a mut HashMap<String, Option<Tally>>, t: &str) -> &'a mut Tally {
    streams.get_mut(t).and_then(Option::as_mut).expect("a loaded stream, opened")
}

fn insert(set: &mut Vec<String>, x: &str) {
    if !set.iter().any(|y| y == x) {
        set.push(x.to_string());
    }
}

impl Session {
    fn json(&self) -> Value {
        let member: Vec<Value> = self.member.iter()
            .map(|m| json!({"stream": m.stream, "parts": m.parts, "released": m.released, "stopped": m.stopped}))
            .collect();
        json!({"open": self.open, "stop": self.stop, "finished": self.finished, "released": self.released, "member": member})
    }

    fn of(v: &Value) -> Option<Self> {
        let member = v["member"].as_array()?.iter().map(|m| Some(Member {
            stream: m["stream"].as_str()?.to_string(),
            parts: strings(&m["parts"])?,
            released: strings(&m["released"])?,
            stopped: m["stopped"].as_bool()?,
        })).collect::<Option<_>>()?;
        let event = |v: &Value| (!v.is_null()).then(|| v.clone());
        Some(Session { open: event(&v["open"]), stop: event(&v["stop"]), finished: v["finished"].as_bool()?,
                       released: strings(&v["released"])?, member })
    }
}

impl Head {
    fn json(&self, boot: &str, dirty: bool) -> Value {
        json!({"v": 1, "boot": boot, "dirty": dirty, "file": self.file.map(|(d, i)| [d, i]), "offset": self.offset,
               "last": hex(&self.last), "regular": self.regular, "count": self.count, "opens": self.opens,
               "slot": self.slot, "slot_done": self.slot_done})
    }

    /// The head in `dir`, if it is one this machine's boot wrote and no writer left dirty.
    fn read(dir: &Path, boot: &str) -> Option<Self> {
        let v: Value = serde_json::from_slice(&std::fs::read(dir.join("head.json")).ok()?).ok()?;
        if v["v"] != 1 || v["boot"] != boot || v["dirty"] != false {
            return None;
        }
        let file = match &v["file"] {
            Value::Null => None,
            f => Some((f[0].as_u64()?, f[1].as_u64()?)),
        };
        let slot = match &v["slot"] {
            Value::Null => None,
            s => Some((s[0].as_u64()?, s[1].as_str()?.to_string(), s[2].as_str()?.to_string())),
        };
        let last = unhex(v["last"].as_str()?)?;
        let offset = v["offset"].as_u64()?;
        (last.len() as u64 <= offset).then_some(())?;
        Some(Head { file, offset, last, regular: v["regular"].as_bool()?, count: v["count"].as_u64()?,
                    opens: v["opens"].as_u64()?, slot, slot_done: v["slot_done"].as_bool()? })
    }
}

fn put(dir: &Path, name: &str, v: &Value) -> io::Result<()> {
    let tmp = dir.join(format!(".{name}"));
    std::fs::write(&tmp, serde_json::to_vec(v)?)?;
    std::fs::rename(tmp, dir.join(name))
}

impl Index {
    /// The index in `dir` of the record at `path` (its device and inode, and its length; None if there is none yet), or a
    /// new one to be built from the record's start if the one there isn't of this record as it stands.
    fn of(dir: PathBuf, path: &Path, at: Option<((u64, u64), u64)>) -> Self {
        let boot = boot_id();
        let head = Head::read(&dir, &boot).filter(|h| at.map(|a| a.0) == h.file && at.map_or(0, |a| a.1) >= h.offset);
        let mut ix = Index::new(dir, path, boot, at.map(|a| a.0));
        if let Some(h) = head {
            (ix.head, ix.fresh, ix.moved) = (h, false, false);
        }
        ix
    }

    fn new(dir: PathBuf, path: &Path, boot: String, file: Option<(u64, u64)>) -> Self {
        Index { dir, path: path.to_path_buf(), boot,
                head: Head { file, offset: 0, last: Vec::new(), regular: true, count: 0, opens: 0, slot: None, slot_done: false },
                fresh: true, broken: false, moved: true, sessions: HashMap::new(), streams: HashMap::new(),
                changed: Default::default() }
    }

    fn get(&self, name: &str) -> io::Result<Option<Value>> {
        #[cfg(test)]
        PROBE.with(|p| p.borrow_mut().loads += 1);
        match std::fs::read(self.dir.join(name)) {
            Ok(b) => serde_json::from_slice(&b).map(Some).map_err(|e| bad(&format!("{name}: {e}"))),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn session(&mut self, s: &str) -> io::Result<&mut Session> {
        if !self.sessions.contains_key(s) {
            let name = format!("s{}.json", hex(s.as_bytes()));
            let v = if self.fresh || s.len() > KEY_MAX { None } else { self.get(&name)? };
            let entry = match v {
                Some(v) => Session::of(&v).ok_or_else(|| bad(&format!("{name}: not a session")))?,
                None => Session::default(),
            };
            self.sessions.insert(s.to_string(), entry);
        }
        Ok(self.sessions.get_mut(s).expect("loaded"))
    }

    fn stream(&mut self, t: &str) -> io::Result<&mut Option<Tally>> {
        if !self.streams.contains_key(t) {
            let name = format!("t{}.json", hex(t.as_bytes()));
            let v = if self.fresh || t.len() > KEY_MAX { None } else { self.get(&name)? };
            let entry = match v {
                Some(v) => Some(Tally { budget: v["budget"].as_u64(),
                                        n: v["n"].as_u64().ok_or_else(|| bad(&format!("{name}: not a stream")))? }),
                None => None,
            };
            self.streams.insert(t.to_string(), entry);
        }
        Ok(self.streams.get_mut(t).expect("loaded"))
    }

    /// Load what a call may read and append for: `sessions` and `streams`, the session the next open would name, the
    /// slot's session if no outcome of it followed, and each of those sessions' streams. A file of the index that won't
    /// read is an error here, before the call reads anything.
    fn load(&mut self, sessions: &[&str], streams: &[&str]) -> io::Result<()> {
        let next = format!("r{:04}", self.head.opens);
        let orphan = self.head.slot.clone().filter(|_| !self.head.slot_done).map(|s| s.1);
        for s in sessions.iter().copied().chain([next.as_str()]).chain(orphan.as_deref()) {
            let member: Vec<String> = self.session(s)?.member.iter().map(|m| m.stream.clone()).collect();
            for t in &member {
                self.stream(t)?;
            }
        }
        for t in streams {
            self.stream(t)?;
        }
        Ok(())
    }

    /// The record's next event, `e`, is in the index. Everything it changes is loaded first, so a load that fails changes
    /// nothing.
    fn add(&mut self, e: &Value) -> io::Result<()> {
        if !(self.head.regular && regular(e)) {
            (self.head.count, self.head.regular, self.moved) = (self.head.count + 1, false, true);
            return Ok(());
        }
        let (kind, s, part) = (e["event"].as_str().unwrap_or_default(), e["session"].as_str().unwrap_or_default(),
                               e["part"].as_str().unwrap_or_default());
        let stream = e["stream"].as_str().unwrap_or_default();
        if kind != "wait" && kind != "slot" {
            let member: Vec<String> = self.session(s)?.member.iter().map(|m| m.stream.clone()).collect();
            for t in member.iter().map(String::as_str).chain((kind == "open").then_some(stream)) {
                self.stream(t)?;
            }
        }
        (self.head.count, self.moved) = (self.head.count + 1, true);
        if kind == "slot" {
            (self.head.slot, self.head.slot_done) = (Some((e["seq"].as_u64().unwrap_or_default(), s.into(), part.into())), false);
            return Ok(());
        }
        if kind != "open" && self.head.slot.as_ref().is_some_and(|(_, ss, sp)| {
            ss == s && (kind == "stop" || matches!(kind, "release" | "refused") && sp == part)
        }) {
            self.head.slot_done = true;
        }
        if kind == "wait" {
            return Ok(());
        }
        let (sess, streams) = (self.sessions.get_mut(s).expect("loaded"), &mut self.streams);
        match kind {
            "open" => {
                self.head.opens += 1;
                sess.open.get_or_insert_with(|| e.clone());
                let ps: Vec<String> = parts(e).into_iter().map(String::from).collect();
                match sess.member.iter_mut().find(|m| m.stream == stream) {
                    Some(m) => m.parts = ps,
                    None => sess.member.push(Member { stream: stream.into(), parts: ps, released: Vec::new(), stopped: false }),
                }
                streams.get_mut(stream).expect("loaded").get_or_insert_with(|| Tally { budget: e["budget"].as_u64(), n: 0 });
                self.changed.1.insert(stream.into());
            }
            "finish" => sess.finished = true,
            "stop" => {
                sess.stop.get_or_insert_with(|| e.clone());
                for m in &mut sess.member {
                    m.stopped = true;
                    tally(streams, &m.stream).n += 1;
                    self.changed.1.insert(m.stream.clone());
                }
            }
            "release" => {
                insert(&mut sess.released, part);
                for m in &mut sess.member {
                    insert(&mut m.released, part);
                }
            }
            "refused" => {
                for m in &sess.member {
                    let t = tally(streams, &m.stream);
                    if !m.parts.iter().all(|p| m.released.contains(p)) && !m.stopped && t.budget.is_none_or(|b| t.n < b) {
                        t.n += 1;
                        self.changed.1.insert(m.stream.clone());
                    }
                }
            }
            _ => unreachable!("a regular event"),
        }
        self.changed.0.insert(s.into());
        Ok(())
    }

    /// The record's next bytes, whole lines, are in the index.
    fn read(&mut self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }
        self.head.offset += chunk.len() as u64;
        let mut last = std::mem::take(&mut self.head.last);
        last.extend_from_slice(&chunk[chunk.len().saturating_sub(LAST)..]);
        self.head.last = last[last.len().saturating_sub(LAST)..].to_vec();
        self.moved = true;
    }

    fn append(&mut self, ev: Value) -> io::Result<()> {
        let mut m = match ev {
            Value::Object(m) => m,
            _ => unreachable!("an event is an object"),
        };
        m.insert("seq".into(), json!(self.head.count));
        let ev = Value::Object(m);
        let line = format!("{}\n", py_json(&ev));
        let mut f = OpenOptions::new().create(true).append(true).open(&self.path)?;
        f.write_all(line.as_bytes())?;
        if self.head.file.is_none() {
            let m = f.metadata()?;
            self.head.file = Some((m.dev(), m.ino()));
        }
        if let Err(e) = self.add(&ev) {
            self.broken = true;
            return Err(e);
        }
        self.read(line.as_bytes());
        Ok(())
    }

    /// Write back what changed. A write that fails leaves the old head, whose record the next call reads again from where
    /// it stood, or a dirty one, and the next call rebuilds the index: either way nothing of the index outruns its head.
    fn flush(&mut self) {
        if self.broken {
            let _ = std::fs::remove_file(self.dir.join("head.json"));
        } else if self.moved {
            let _ = self.write();
        }
    }

    fn write(&mut self) -> io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        put(&self.dir, "head.json", &self.head.json(&self.boot, true))?;
        if self.fresh {
            for f in std::fs::read_dir(&self.dir)? {
                let f = f?;
                if f.file_name() != "head.json" {
                    std::fs::remove_file(f.path())?;
                }
            }
        }
        for s in &self.changed.0 {
            put(&self.dir, &format!("s{}.json", hex(s.as_bytes())), &self.sessions[s].json())?;
        }
        for t in &self.changed.1 {
            if let Some(Some(x)) = self.streams.get(t) {
                put(&self.dir, &format!("t{}.json", hex(t.as_bytes())), &json!({"budget": x.budget, "n": x.n}))?;
            }
        }
        put(&self.dir, "head.json", &self.head.json(&self.boot, false))
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
struct Probe {
    /// read the record whole at every call, as before the index
    whole: bool,
    /// the record's lines parsed, and the files of the index read
    parsed: u64,
    loads: u64,
    /// the calls, and those the index answered
    calls: u64,
    indexed: u64,
}

#[cfg(test)]
thread_local! {
    static PROBE: std::cell::RefCell<Probe> = Default::default();
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

    fn locked(&self) -> io::Result<Held> {
        let lock = self.lock_file("record.lock")?;
        lock.lock()?;
        Ok(Held(lock))
    }

    /// `f` on the record as the rules read it, under `record.lock`; `sessions` and `streams` are the ones `f` reads and
    /// appends for ([`Index::load`]).
    fn with<T>(&self, sessions: &[&str], streams: &[&str], f: impl FnOnce(&mut View) -> io::Result<T>) -> io::Result<T> {
        let _held = self.locked()?;
        let mut view = self.view(sessions, streams)?;
        let out = f(&mut view);
        if let View::Index(ix) | View::Whole(_, Some(ix)) = &mut view {
            ix.flush();
        }
        out
    }

    /// The record as the rules read it: the index, given the lines appended since it last read the record; or the record
    /// read whole, when the index can't answer it or a line won't read (the whole read then fails as it always has).
    fn view(&self, sessions: &[&str], streams: &[&str]) -> io::Result<View> {
        let path = self.state.join("record.jsonl");
        #[cfg(test)]
        if PROBE.with(|p| {
            let mut p = p.borrow_mut();
            p.calls += 1;
            p.whole
        }) {
            return Ok(View::Whole(Events::read(&path)?, None));
        }
        let mut f = match File::open(&path) {
            Ok(f) => Some(f),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        let at = match &f {
            Some(f) => f.metadata().map(|m| Some(((m.dev(), m.ino()), m.len())))?,
            None => None,
        };
        let mut ix = Index::of(self.state.join("index"), &path, at);
        let mut data = Vec::new();
        if let Some(f) = f.as_mut() {
            let back = ix.head.last.len();
            f.seek(SeekFrom::Start(ix.head.offset - back as u64))?;
            f.read_to_end(&mut data)?;
            if data.starts_with(&ix.head.last) {
                data.drain(..back);
            } else {
                ix = Index::new(ix.dir, &path, ix.boot, at.map(|a| a.0));
                data.clear();
                f.seek(SeekFrom::Start(0))?;
                f.read_to_end(&mut data)?;
            }
        }
        let whole = |ix: Option<Index>| Events::read(&path).map(|ev| View::Whole(ev, ix));
        let Ok(text) = std::str::from_utf8(&data) else {
            return whole(None);
        };
        let end = text.rfind('\n').map_or(0, |k| k + 1);
        let mut lines = Vec::new();
        for line in text[..end].lines().filter(|l| !l.is_empty()) {
            match serde_json::from_str::<Value>(line) {
                Ok(e) => lines.push(e),
                Err(_) => return whole(None),
            }
        }
        #[cfg(test)]
        PROBE.with(|p| p.borrow_mut().parsed += lines.len() as u64);
        let read = lines.iter().try_for_each(|e| ix.add(e)).and_then(|()| {
            ix.read(&data[..end]);
            ix.load(sessions, streams)
        });
        if read.is_err() {
            ix.broken = true;
            ix.flush();
            return whole(None);
        }
        if !ix.head.regular || end < text.len() {
            return whole(Some(ix));
        }
        #[cfg(test)]
        PROBE.with(|p| p.borrow_mut().indexed += 1);
        Ok(View::Index(ix))
    }

    /// Every event so far, in order.
    pub fn events(&self) -> io::Result<Vec<Value>> {
        let _held = self.locked()?;
        Ok(Events::read(&self.state.join("record.jsonl"))?.list)
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
        self.with(&[], &[stream], |ev| {
            if let Some(first) = ev.first(stream)? {
                if first != budget {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput,
                        format!("stream {stream}'s budget is {}, not {}", show(first), show(budget))));
                }
            }
            if let Some(b) = budget {
                let n = ev.failures(stream)?;
                if n >= b {
                    return Ok(Err(format!("stream {stream} stopped at its failure budget ({n} of {b})")));
                }
            }
            let sid = format!("r{:04}", ev.opens());
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
        self.with(&[session], &[], |ev| ev.append(json!({"event": "finish", "session": session, "part": INNER})))
    }

    /// The recursive session stops at `part`: its one failure, recorded once (false if it had already stopped).
    pub fn stop(&self, session: &str, part: &str, why: &str) -> io::Result<bool> {
        self.with(&[session], &[], |ev| {
            if ev.stopped(session)?.is_some() {
                return Ok(false);
            }
            ev.append(json!({"event": "stop", "session": session, "part": part, "why": why}))?;
            Ok(true)
        })
    }

    /// An outer proof of the recursive session was released (the challenger's verdict accepted it).
    pub fn release(&self, session: &str, part: &str) -> io::Result<()> {
        self.with(&[session], &[], |ev| ev.append(json!({"event": "release", "session": session, "part": part})))
    }

    /// Whether an outer session of `session` at `part` may be served now: the recursive session is open, has not stopped,
    /// has a statement not yet released, its stream has not stopped at its budget, its inner session finished, and `part`
    /// is its next statement (rule 1). A request for any other statement is the recursive session's stop at the next one
    /// (never opened), then refused. A refusal is recorded (`refused`).
    pub fn admit(&self, session: &str, part: &str) -> io::Result<Result<Admitted, String>> {
        self.with(&[session], &[], |ev| {
            let why = match ev.opened(session)? {
                None => format!("recursive session {session} was never opened"),
                Some(open) => {
                    let stream = open["stream"].as_str().unwrap_or_default().to_string();
                    let spent = match ev.budget(&stream)? {
                        Some(b) => Some((ev.failures(&stream)?, b)).filter(|&(n, b)| n >= b),
                        None => None,
                    };
                    let next = ev.next(&open)?;
                    if let Some(s) = ev.stopped(session)? {
                        format!("recursive session {session} stopped at {}: the firewall serves nothing more of it",
                                s["part"].as_str().unwrap_or_default())
                    } else if next.is_none() {
                        format!("recursive session {session} released all its statements: the firewall serves nothing more of it")
                    } else if let Some((n, b)) = spent {
                        format!("stream {stream} stopped at its failure budget ({n} of {b})")
                    } else if !ev.finished(session)? {
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
        self.with(&[], &[stream], |ev| ev.failures(stream))
    }

    /// The firewall's one outer slot, held until the returned [`Slot`] drops. A session that finds it taken records `wait`,
    /// calls `waiting`, and blocks until the holder leaves it. Once it holds the slot, a `slot` event whose session has no
    /// outcome after it (its holder died) is recorded as that session's stop, and then `slot` is recorded.
    pub fn slot(&self, session: &str, part: &str, waiting: impl FnOnce()) -> io::Result<Slot> {
        let f = self.lock_file("slot.lock")?;
        match f.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                self.with(&[session], &[], |ev| ev.append(json!({"event": "wait", "session": session, "part": part})))?;
                waiting();
                f.lock()?;
            }
            Err(TryLockError::Error(e)) => return Err(e),
        }
        let held = Held(f);
        self.with(&[session], &[], |ev| {
            if let Some((seq, s, p)) = ev.orphan() {
                if ev.stopped(&s)?.is_none() {
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

    fn probe() -> Probe {
        PROBE.with(|p| *p.borrow())
    }

    /// An event appended as another process writes it (`verity_flock.firewall`'s inner side, say), outside this crate's
    /// calls: its line, with its seq.
    fn appended(rec: &Record, mut ev: Value) {
        let path = rec.state.join("record.jsonl");
        ev["seq"] = json!(std::fs::read_to_string(&path).map_or(0, |t| t.lines().filter(|l| !l.is_empty()).count()));
        let mut f = OpenOptions::new().create(true).append(true).open(path).unwrap();
        f.write_all(format!("{}\n", py_json(&ev)).as_bytes()).unwrap();
    }

    /// 300 recursive sessions, each opened and finished by another process (as the inner side does) and its three
    /// statements' outer sessions run here, every fifth stopping at its second statement so that the third is refused.
    /// Each call parses only the lines the other process appended since the index last read the record, never the whole
    /// record, and a round's work (lines parsed, files of the index read) is the same in the last round as in the first
    /// of its kind; the record read whole parses all of it.
    #[test]
    fn a_call_reads_only_what_was_appended_since_the_index_last_read() {
        let rec = scratch("history");
        let mut work = Vec::new();
        for i in 0..300 {
            let sid = format!("r{i:04}");
            appended(&rec, json!({"event": "open", "session": sid, "stream": "s", "parts": PARTS, "budget": 1_000_000}));
            appended(&rec, json!({"event": "finish", "session": sid, "part": INNER}));
            let mut fresh = 2;
            let round = probe();
            for (j, p) in PARTS.iter().enumerate() {
                let before = probe();
                let cut = i % 5 == 4 && j == 1;
                let o = outer_session(&rec, &sid, p, &fake(0), || {}, || if cut { Err("cut".to_string()) } else { Ok(()) },
                                      |_| Ok(()), |_| (true, ())).unwrap();
                assert_eq!(o.outcome, if i % 5 == 4 && j > 0 { if j == 1 { "stop" } else { "refused" } } else { "release" });
                let after = probe();
                assert_eq!(after.parsed - before.parsed, std::mem::take(&mut fresh), "round {i}, {p}");
                assert_eq!(after.indexed - before.indexed, after.calls - before.calls, "round {i}, {p}");
            }
            let end = probe();
            work.push((end.parsed - round.parsed, end.loads - round.loads, end.calls - round.calls));
        }
        for i in 6..300 {
            assert_eq!(work[i], work[i - 5], "round {i}");
        }
        assert_eq!(rec.failures("s").unwrap(), 60);
        let lines = rec.events().unwrap().len() as u64;
        PROBE.with(|p| p.borrow_mut().whole = true);
        let before = probe();
        assert_eq!(rec.failures("s").unwrap(), 60);
        assert_eq!(probe().parsed - before.parsed, lines);
        PROBE.with(|p| p.borrow_mut().whole = false);
    }

    /// A head that another boot wrote, or that a writer left dirty (it ended between the head and the files it was
    /// writing), is not read: the next call rebuilds the index from the whole record, so a file of the index it would have
    /// trusted (a stream's failures, here) doesn't count, and the call after reads nothing of the record.
    #[test]
    fn an_index_from_another_boot_or_left_mid_write_is_rebuilt() {
        let rec = scratch("rebuilt");
        let sid = opened(&rec, None, Some(5));
        assert!(rec.stop(&sid, "L0", "cut").unwrap());
        let lines = rec.events().unwrap().len() as u64;
        let index = rec.state.join("index");
        let head: Value = serde_json::from_slice(&std::fs::read(index.join("head.json")).unwrap()).unwrap();
        for (k, v) in [("boot", json!("another boot")), ("dirty", json!(true))] {
            let mut h = head.clone();
            h[k] = v;
            std::fs::write(index.join("head.json"), serde_json::to_vec(&h).unwrap()).unwrap();
            std::fs::write(index.join(format!("t{}.json", hex(b"s"))), br#"{"budget": 5, "n": 4}"#).unwrap();
            for parsed in [lines, 0] {
                let before = probe();
                assert_eq!(rec.failures("s").unwrap(), 1, "{k}");
                assert_eq!(probe().parsed - before.parsed, parsed, "{k}");
            }
        }
    }

    /// splitmix64: the differential test's fixed stream of choices.
    struct Draw(u64);

    impl Draw {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
            xs[self.below(xs.len())]
        }
    }

    /// Damage to the record, or to its index, between two calls: a blank line, a torn line, a line without its newline, a
    /// line the firewall doesn't write, the record replaced or rewritten as it was, cut by its last lines, those lines
    /// rewritten in place as others of their length, or halved; the index removed, or its head or every other file of it
    /// torn. `mark` is the record's length before a damage that a later `repair` undoes.
    fn damage(state: &Path, kind: &str, mark: &mut Option<usize>, sid: &str) {
        let log = state.join("record.jsonl");
        let data = std::fs::read(&log).unwrap_or_default();
        let add = |b: &[u8]| OpenOptions::new().create(true).append(true).open(&log).unwrap().write_all(b).unwrap();
        let index = |head: bool| -> Vec<PathBuf> {
            std::fs::read_dir(state.join("index")).into_iter().flatten().flatten().map(|f| f.path())
                .filter(|p| (p.file_name().unwrap() == "head.json") == head).collect()
        };
        match kind {
            "blank" => add(b"\n"),
            "torn" | "unterminated" | "foreign" if mark.is_none() => {
                *mark = Some(data.len());
                let n = data.split(|&b| b == b'\n').filter(|l| !l.is_empty()).count();
                let line = match kind {
                    "torn" => b"{\"event\": \"sto".to_vec(),
                    "foreign" => b"{\"event\": \"note\", \"seq\": 7}\n".to_vec(),
                    _ => py_json(&json!({"event": "finish", "part": INNER, "seq": n, "session": sid})).into_bytes(),
                };
                add(&line);
            }
            "repair" => {
                if let Some(m) = mark.take() {
                    OpenOptions::new().write(true).open(&log).unwrap().set_len(m as u64).unwrap();
                }
            }
            "replace" => {
                std::fs::write(state.join("record.tmp"), &data).unwrap();
                std::fs::rename(state.join("record.tmp"), &log).unwrap();
            }
            "rewrite" => std::fs::write(&log, &data).unwrap(),
            "cut" | "edit" | "halve" if mark.is_none() && data.iter().filter(|&&b| b == b'\n').count() > 4 => {
                let lines: Vec<&[u8]> = data.split_inclusive(|&b| b == b'\n').collect();
                let mut kept = lines[..if kind == "halve" { lines.len() / 2 } else { lines.len() - 3 }].concat();
                if kind == "edit" {
                    // each of the last lines rewritten in place as a `wait` of the same length, where one fits
                    for (k, line) in lines[lines.len() - 3..].iter().enumerate() {
                        let wait = |pad: usize| json!({"event": "wait", "part": "x".repeat(pad), "seq": k, "session": sid});
                        let pad = (line.len() - 1).checked_sub(py_json(&wait(0)).len());
                        kept.extend(pad.map_or(line.to_vec(), |p| format!("{}\n", py_json(&wait(p))).into_bytes()));
                    }
                }
                std::fs::write(&log, kept).unwrap();
            }
            "index-gone" => {
                let _ = std::fs::remove_dir_all(state.join("index"));
            }
            "index-head" | "index-entries" => {
                for p in index(kind == "index-head") {
                    std::fs::write(p, b"{\"v\": ").unwrap();
                }
            }
            _ => {}
        }
    }

    /// A fixed stream of `steps` ops on a record in `state` (each op's result, and the record's digest after it), read
    /// whole at every call if `whole`: opens on two streams, one with a budget, finishes, admits, releases, stops, outer
    /// sessions in the slot and holders that end before their outcome, events another process appends, and between them
    /// damage to the record and to the index ([`damage`]). `long` draws session names past [`KEY_MAX`] too.
    fn drive(state: &Path, whole: bool, seed: u64, steps: usize, long: bool) -> Vec<String> {
        use std::hash::{Hash, Hasher};
        let _ = std::fs::remove_dir_all(state);
        let rec = Record::new(state).unwrap();
        PROBE.with(|p| p.borrow_mut().whole = whole);
        let (mut draw, mut out, mut mark) = (Draw(seed), Vec::new(), None);
        let mut sids = vec!["r9999".to_string()];
        let name = state.display().to_string();
        let show = |r: io::Result<String>| match r {
            Ok(v) => v,
            Err(e) => format!("error {:?}: {}", e.kind(), e.to_string().replace(&name, "STATE")),
        };
        let long_sid = "r".repeat(KEY_MAX + 1);
        for _ in 0..steps {
            let mut recent: Vec<&str> = sids.iter().rev().take(4).map(String::as_str).collect();
            recent.push("r9999");
            if long {
                recent.push(&long_sid);
            }
            let sid = draw.pick(&recent).to_string();
            let part = draw.pick(&["L0", "L1", "alg", "L9"]);
            let stream = draw.pick(&["s", "t"]);
            let r = draw.below(100);
            if r < 4 {
                let kind = draw.pick(&["blank", "torn", "unterminated", "foreign", "replace", "rewrite", "cut", "edit", "halve",
                                       "index-gone", "index-head", "index-entries"]);
                damage(state, kind, &mut mark, &sid);
                out.push(format!("damage {kind}"));
            } else if r < 7 {
                damage(state, "repair", &mut mark, &sid);
                out.push("damage repair".into());
            } else {
                let op = draw.pick(&["open", "open", "finish", "admit", "admit", "admit", "release", "release", "stop",
                                     "failures", "slot", "slot", "events", "appended"]);
                let choices: [&[&str]; 3] = [&PARTS, &["L0"], &["L0", "L1"]];
                let parts = choices[draw.below(3)];
                let budget = if draw.below(20) == 0 { Some(9) } else if stream == "s" { Some(4) } else { None };
                let x = draw.below(10);
                let got = match op {
                    "open" => rec.open(stream, parts, budget, None).map(|r| {
                        if let Ok(s) = &r {
                            sids.push(s.clone());
                        }
                        format!("{r:?}")
                    }),
                    "finish" => rec.finish(&sid).map(|()| "()".into()),
                    "admit" => rec.admit(&sid, part).map(|r| format!("{r:?}")),
                    "release" => rec.release(&sid, part).map(|()| "()".into()),
                    "stop" => rec.stop(&sid, part, "cut").map(|r| format!("{r:?}")),
                    "failures" => rec.failures(stream).map(|n| n.to_string()),
                    "events" => rec.events().map(|ev| ev.len().to_string()),
                    "slot" => (|| -> io::Result<String> {
                        let held = rec.slot(&sid, part, || {})?;
                        if x < 3 {
                            drop(held);
                            return Ok("ended".to_string());
                        }
                        let a = rec.admit(&sid, part)?;
                        if a.is_ok() {
                            if x < 8 {
                                rec.release(&sid, part)?;
                            } else {
                                rec.stop(&sid, part, "the challenger's verdict rejects it")?;
                            }
                        }
                        Ok(format!("{a:?}"))
                    })(),
                    _ => {
                        let kind = draw.pick(&["open", "finish", "stop", "release", "refused", "wait", "slot"]);
                        let mut ev = json!({"event": kind, "session": sid, "part": part});
                        if kind == "open" {
                            ev = json!({"event": "open", "session": sid, "stream": stream, "parts": parts, "budget": budget});
                        }
                        appended(&rec, ev);
                        Ok(kind.to_string())
                    }
                };
                out.push(format!("{op} {}", show(got)));
            }
            let mut h = std::hash::DefaultHasher::new();
            std::fs::read(state.join("record.jsonl")).unwrap_or_default().hash(&mut h);
            out.push(format!("{:016x}", h.finish()));
        }
        PROBE.with(|p| p.borrow_mut().whole = false);
        out
    }

    /// On fixed streams of ops, with the record damaged and repaired and the index broken between them, every result,
    /// refusal and error, and the record after each op, are those of the record read whole at every call, and the index
    /// answered most of the calls (the rest are a damaged record's). With names past [`KEY_MAX`], the same.
    #[test]
    fn the_index_answers_as_the_whole_record_does() {
        let base = std::env::temp_dir().join(format!("flock-release-drive-{}", std::process::id()));
        for (seed, long) in [(20261016, false), (7, false), (11, true)] {
            let before = probe();
            let got = drive(&base.join("index"), false, seed, 1500, long);
            let after = probe();
            let want = drive(&base.join("whole"), true, seed, 1500, long);
            assert_eq!(got.len(), want.len());
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "seed {seed}, step {k}: {:?}", &got[k.saturating_sub(6)..k]);
            }
            let ops: HashSet<&str> = got.iter().filter_map(|s| s.split(' ').next()).collect();
            for op in ["open", "finish", "admit", "release", "stop", "failures", "slot", "events", "appended", "damage"] {
                assert!(ops.contains(op), "seed {seed}: no {op}");
            }
            assert!(got.iter().any(|s| s.contains("error InvalidData")), "seed {seed}: no torn record");
            if !long {
                let (calls, indexed) = (after.calls - before.calls, after.indexed - before.indexed);
                assert!(indexed * 2 > calls, "seed {seed}: the index answered {indexed} of {calls} calls");
            }
        }
        let _ = std::fs::remove_dir_all(base);
    }
}
