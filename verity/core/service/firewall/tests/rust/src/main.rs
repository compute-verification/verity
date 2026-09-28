//! `release-reference STATE_DIR`: Rust's release side (`flock_live::release`, compiled from its file in the live crate) as
//! the difftest's reference. One request a line on stdin, `{"fn": "op" | "outer", "input": …}` (`Firewall/Json.lean`
//! documents the inputs), one canonical JSON line back: `{"output": …}`, `{"reject": why}` when the record can't be read,
//! or `{"error": why}` for a request it can't take. Each request runs on a fresh state directory under STATE_DIR holding
//! the input's record, and its output carries the record as the request left it.
#[allow(dead_code)]
#[path = "../../../../../../ml/flock/live/src/release.rs"]
mod release;

use release::{outer_session, Clock, Record};
use serde_json::{json, Value};
use std::cell::Cell;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

/// The clock a test drives: `now`, and the release time the session waited for.
struct Fake {
    now: u64,
    slept: Cell<Option<u64>>,
}

impl Clock for Fake {
    fn now_ms(&self) -> u64 {
        self.now
    }
    fn sleep_until_ms(&self, t: u64) {
        self.slept.set(Some(t));
    }
}

/// `release::run_op` (the shared vectors' driver), except that a record it can't read is an error of the call, not a
/// result.
fn op(rec: &Record, op: &[Value]) -> io::Result<Value> {
    let s = |k: usize| op.get(k).and_then(Value::as_str).unwrap_or_default();
    let at = |k: usize| op.get(k).unwrap_or(&Value::Null);
    Ok(match s(0) {
        "open" => {
            let ps: Vec<&str> = at(2).as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
            match rec.open(s(1), &ps, at(3).as_u64(), (!at(4).is_null()).then_some(at(4))) {
                Ok(Ok(sid)) => json!(sid),
                Ok(Err(why)) => json!({"refused": why}),
                Err(e) if e.kind() == io::ErrorKind::InvalidInput => json!({"error": e.to_string()}),
                Err(e) => return Err(e),
            }
        }
        "finish" => {
            rec.finish(s(1))?;
            Value::Null
        }
        "stop" => json!(rec.stop(s(1), s(2), s(3))?),
        "release" => {
            rec.release(s(1), s(2))?;
            Value::Null
        }
        "admit" => match rec.admit(s(1), s(2))? {
            Ok(a) => json!({"release_ms": a.release_ms}),
            Err(why) => json!({"refused": why}),
        },
        "failures" => json!(rec.failures(s(1))?),
        "slot" => {
            drop(rec.slot(s(1), s(2), || {})?);
            Value::Null
        }
        other => return Err(io::Error::other(format!("unknown op {other}"))),
    })
}

/// A step outside the firewall as the input gives it: `"ok"`, `{"err": why}`, or `{"panic": msg}`, which panics.
fn step(v: &Value) -> Result<(), String> {
    if v == "ok" {
        return Ok(());
    }
    if let Some(why) = v["err"].as_str() {
        return Err(why.to_string());
    }
    panic!("{}", v["panic"].as_str().unwrap_or_default())
}

fn outer(rec: &Record, input: &Value) -> io::Result<Value> {
    let clock = Fake { now: input["now_ms"].as_u64().unwrap_or_default(), slept: Cell::new(None) };
    let send = |_: ()| match &input["send"] {
        Value::Bool(b) => (*b, ()),
        v => panic!("{}", v["panic"].as_str().unwrap_or_default()),
    };
    let o = outer_session(rec, input["session"].as_str().unwrap_or_default(), input["part"].as_str().unwrap_or_default(),
                          &clock, || {}, || step(&input["prove"]), |_| step(&input["ready"]), send)?;
    Ok(json!({"outcome": o.outcome, "why": o.why, "sent_at": clock.slept.get()}))
}

fn handle(dir: &Path, line: &str) -> Value {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return json!({"error": format!("request: not JSON: {e}")}),
    };
    let input = &req["input"];
    let _ = std::fs::remove_dir_all(dir);
    let rec = Record::new(dir).expect("a state directory");
    let path = dir.join("record.jsonl");
    std::fs::write(&path, input["record"].as_str().unwrap_or_default()).expect("the record written");
    let got = match req["fn"].as_str() {
        Some("op") => op(&rec, input["op"].as_array().map(Vec::as_slice).unwrap_or_default())
            .map(|r| json!({"result": r})),
        Some("outer") => outer(&rec, input),
        other => return json!({"error": format!("unknown fn: {other:?}")}),
    };
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_dir_all(dir);
    match got {
        Ok(Value::Object(mut m)) => {
            m.insert("record".into(), json!(text));
            json!({"output": m})
        }
        Ok(_) => unreachable!("an object"),
        Err(e) if e.kind() == io::ErrorKind::InvalidData => json!({"reject": e.to_string()}),
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let base = PathBuf::from(std::env::args().nth(1).expect("usage: release-reference STATE_DIR"));
    let mut out = io::stdout().lock();
    for (k, line) in io::stdin().lock().lines().enumerate() {
        let resp = handle(&base.join(k.to_string()), &line.expect("a line"));
        writeln!(out, "{}", serde_json::to_string(&resp).expect("JSON")).expect("stdout");
        out.flush().expect("stdout");
    }
}
