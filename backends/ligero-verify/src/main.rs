//! ligero-verify: the independent verifier for Candidate B's B-Ligero chain proofs.
//!
//! ~~~text
//! ligero-verify verify --system system.bin --statement X.stmt --proof X.proof [--threads T] [--soundness-bits 128]
//!                      [--coins FILE]   (interactive mode: the verifier's own r1||s1||r2||s2, 128 bytes raw or hex)
//!                      [--allow-any-system] [--json out.json] [--reps R]
//! ligero-verify batch  --system system.bin --dir DIR [--jobs N] [--threads T] [--target-bits 128] [--allow-any-system]
//!                      [--json out.json] [--reps R] [--coins FILE]
//!                      # every DIR/*.proof with its DIR/*.stmt (and DIR/*.coins when present: that proof's own step-0
//!                      # coins, as `verify --coins`); the sub-batches are verified concurrently on N jobs (default: the
//!                      # available parallelism), each exactly as `verify` does; the batch is accepted iff every
//!                      # sub-batch is AND (PROTOCOL.md 8c, "Batch verification") the union of this verifier's own
//!                      # per-proof bounds over the N sub-batches presented is <= 2^-target AND every proof's n_proofs
//!                      # is N (`batch: n_proofs mismatch`).  Agreement with DIR/manifest.json's python_verdict when present.
//! ligero-verify system-digest --system system.bin           # the two digests to pin in verify.rs
//! ~~~

mod field;
mod format;
mod hash;
mod verify;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// A coins file: the verifier's own `r1 || s1 || r2 || s2`, 128 raw bytes or 256 hex digits (+ newline).
fn parse_coins(raw: Vec<u8>) -> Result<[u8; 128], String> {
    let raw = if raw.len() == 256 || raw.len() == 257 {
        let txt = String::from_utf8_lossy(&raw).trim().to_string();
        let mut v = Vec::with_capacity(128);
        for i in 0..128 {
            let pair = txt.get(2 * i..2 * i + 2).ok_or("coins hex too short")?;
            v.push(u8::from_str_radix(pair, 16).map_err(|_| "coins hex: bad digit")?);
        }
        v
    } else {
        raw
    };
    if raw.len() != 128 {
        return Err(format!("coins must hold r1 || s1 || r2 || s2 (128 bytes, raw or hex); got {} bytes", raw.len()));
    }
    let mut out = [0u8; 128];
    out.copy_from_slice(&raw);
    Ok(out)
}

fn options(args: &[String]) -> verify::Options {
    verify::Options {
        threads: arg(args, "--threads").map(|s| s.parse().expect("--threads")).unwrap_or(1),
        soundness_bits: Some(arg(args, "--soundness-bits").map(|s| s.parse().expect("--soundness-bits")).unwrap_or(128.0)),
        allow_any_system: flag(args, "--allow-any-system"),
        own_coins: arg(args, "--coins").map(|path| {
            let raw = std::fs::read(&path).expect("--coins file");
            parse_coins(raw).unwrap_or_else(|e| panic!("--coins {path}: {e}"))
        }),
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn timing_json(t: &verify::Timing) -> String {
    format!(
        "{{\"shared\":{:.6},\"parse\":{:.6},\"soundness\":{:.6},\"statement_hash\":{:.6},\"challenge_hash\":{:.6},\"merkle\":{:.6},\"ntt\":{:.6},\"dfold\":{:.6},\"quadratic\":{:.6},\"chain\":{:.6},\"total\":{:.6}}}",
        t.shared, t.parse, t.soundness, t.statement_hash, t.challenge_hash, t.merkle, t.ntt, t.dfold, t.quadratic, t.chain, t.total
    )
}

fn statement_json(s: &Option<verify::StatementInfo>) -> String {
    match s {
        None => "null".into(),
        Some(s) => format!(
            "{{\"l\":{},\"n\":{},\"D\":{},\"t\":{},\"zk\":{},\"mode\":\"{}\",\"n_proofs\":{},\"n_vus\":{},\"steps\":{},\"digest\":{}}}",
            s.l,
            s.n,
            s.d,
            s.t,
            s.zk,
            if s.interactive { "interactive" } else { "fiat-shamir" },
            s.n_proofs,
            s.n_vus,
            s.steps,
            s.digest.as_ref().map_or("null".to_string(), |d| format!("\"{d}\""))
        ),
    }
}

fn verdict_json(v: &verify::Verdict, name: &str, python: Option<(bool, &str)>, reps: usize) -> String {
    let snd = match &v.soundness {
        Some(s) => format!(
            "{{\"per_proof_log2\":{:.3},\"union_log2\":{:.3},\"terms\":{{{}}}}}",
            s.per_proof_log2,
            s.union_log2,
            s.terms.iter().map(|(k, x)| format!("\"{k}\":{x:.3}")).collect::<Vec<_>>().join(",")
        ),
        None => "null".into(),
    };
    let py = match python {
        Some((ok, why)) => format!(
            ",\"python\":{{\"accepted\":{ok},\"reason\":{}}},\"agree\":{}",
            json_str(why),
            ok == v.accepted
        ),
        None => String::new(),
    };
    format!(
        "{{\"name\":{},\"accepted\":{},\"reason\":{},\"statement\":{},\"timing\":{},\"reps\":{},\"soundness\":{}{}}}",
        json_str(name),
        v.accepted,
        json_str(&v.reason),
        statement_json(&v.statement),
        timing_json(&v.timing),
        reps,
        snd,
        py
    )
}

/// Minimal extraction of `"python_verdict": {"accepted": ..., "reason": "..."}` per `"proof": "<name>"` entry from
/// manifest.json (no JSON dependency; the manifest is written by serialize.py with indent=1).
fn manifest_verdicts(path: &Path) -> Vec<(String, bool, String)> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut rest = text.as_str();
    while let Some(i) = rest.find("\"proof\": \"") {
        let s = &rest[i + 10..];
        let end = s.find('"').unwrap();
        let name = s[..end].to_string();
        let after = &s[end..];
        let stop = after.find("\"proof\": \"").unwrap_or(after.len());
        let block = &after[..stop];
        let acc = block.find("\"accepted\": ").map(|j| block[j + 12..].starts_with("true"));
        let why = block.find("\"reason\": \"").map(|j| {
            let r = &block[j + 11..];
            let mut o = String::new();
            let mut esc = false;
            for ch in r.chars() {
                if esc {
                    o.push(ch);
                    esc = false;
                } else if ch == '\\' {
                    esc = true;
                } else if ch == '"' {
                    break;
                } else {
                    o.push(ch);
                }
            }
            o
        });
        if let (Some(a), Some(w)) = (acc, why) {
            out.push((name, a, w));
        }
        rest = after;
    }
    out
}

fn run_one(sys: &[u8], st: &[u8], pf: &[u8], opt: &verify::Options, reps: usize) -> verify::Verdict {
    let mut best: Option<verify::Verdict> = None;
    for _ in 0..reps.max(1) {
        let v = verify::verify(sys, st, pf, opt);
        best = Some(match best {
            Some(b) if b.timing.total <= v.timing.total => b,
            _ => v,
        });
    }
    best.unwrap()
}

/// `run_one` against a `Shared` (batch mode): the fastest of `reps` verifications of this one sub-batch.
fn run_one_with(shared: &verify::Shared, st: &[u8], pf: &[u8], opt: &verify::Options, reps: usize) -> verify::Verdict {
    let mut best: Option<verify::Verdict> = None;
    for _ in 0..reps.max(1) {
        let v = verify::verify_with(shared, st, pf, opt);
        best = Some(match best {
            Some(b) if b.timing.total <= v.timing.total => b,
            _ => v,
        });
    }
    best.unwrap()
}

/// One sub-batch of a batch: its verdict, whether it was checked against its own coins file, or the operator input
/// that could not be read (exit 2 after the report, like an unreadable path on the command line).
struct BatchItem {
    name: String,
    verdict: Option<verify::Verdict>,
    own_coins: bool,
    io_error: Option<String>,
}

/// Verify `proofs` (each `X.proof` with `X.stmt`, and `X.coins` when present and `--coins` was not given) on `jobs`
/// OS threads pulling from one queue.  Every sub-batch is verified exactly as `verify` would (own statement, own
/// proof, own challenges, own checks, `--threads` inside); only the parsed system, its pinned-digest verdict and the
/// domain tables are shared, read-only.  Results land in proof order, so the report and the batch verdict do not
/// depend on which thread finished first.
fn verify_batch(shared: &verify::Shared, proofs: &[PathBuf], opt: &verify::Options, reps: usize, jobs: usize) -> Vec<BatchItem> {
    let next = AtomicUsize::new(0);
    let jobs = jobs.max(1).min(proofs.len().max(1));
    let mut slots: Vec<Option<BatchItem>> = (0..proofs.len()).map(|_| None).collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..jobs)
            .map(|_| {
                let next = &next;
                s.spawn(move || {
                    let mut mine: Vec<(usize, BatchItem)> = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= proofs.len() {
                            break;
                        }
                        let p = &proofs[i];
                        let name = p.file_name().unwrap().to_string_lossy().to_string();
                        let mut item = BatchItem { name, verdict: None, own_coins: false, io_error: None };
                        let read = |path: &Path| std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()));
                        let files = read(&p.with_extension("stmt")).and_then(|st| read(p.as_path()).map(|pf| (st, pf)));
                        let (st, pf) = match files {
                            Ok(x) => x,
                            Err(e) => {
                                item.io_error = Some(e);
                                mine.push((i, item));
                                continue;
                            }
                        };
                        let mut o = opt.clone();
                        if o.own_coins.is_none() {
                            let cp = p.with_extension("coins");
                            if cp.is_file() {
                                match read(&cp).and_then(parse_coins) {
                                    Ok(c) => o.own_coins = Some(c),
                                    Err(e) => {
                                        item.io_error = Some(format!("{}: {e}", cp.display()));
                                        mine.push((i, item));
                                        continue;
                                    }
                                }
                            }
                        }
                        item.own_coins = o.own_coins.is_some();
                        item.verdict = Some(run_one_with(shared, &st, &pf, &o, reps));
                        mine.push((i, item));
                    }
                    mine
                })
            })
            .collect();
        for h in handles {
            for (i, item) in h.join().expect("batch worker panicked") {
                slots[i] = Some(item);
            }
        }
    });
    slots.into_iter().map(|x| x.expect("every sub-batch has a slot")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let opt = options(&args);
    let reps: usize = arg(&args, "--reps").map(|s| s.parse().expect("--reps")).unwrap_or(1);
    match cmd {
        "system-digest" => {
            let sys = std::fs::read(arg(&args, "--system").expect("--system")).expect("read system");
            match format::System::parse(&sys) {
                Ok(s) => println!(
                    "{{\"sys_id\":\"{}\",\"table_digest\":\"{}\",\"m\":{},\"L\":{},\"Q\":{},\"pins\":{},\"chain\":{}}}",
                    hash::hex(&s.sys_id),
                    hash::hex(&format::System::table_digest(&sys)),
                    s.m,
                    s.linear.len(),
                    s.quadratic.len(),
                    s.pins.len(),
                    s.chain.is_some()
                ),
                Err(e) => {
                    eprintln!("system file: {e}");
                    std::process::exit(2);
                }
            }
        }
        "verify" => {
            let sys = std::fs::read(arg(&args, "--system").expect("--system")).expect("read system");
            let st = std::fs::read(arg(&args, "--statement").expect("--statement")).expect("read statement");
            let pf = std::fs::read(arg(&args, "--proof").expect("--proof")).expect("read proof");
            let v = run_one(&sys, &st, &pf, &opt, reps);
            let js = verdict_json(&v, &arg(&args, "--proof").unwrap(), None, reps);
            println!("{js}");
            if let Some(p) = arg(&args, "--json") {
                std::fs::write(p, js).expect("write json");
            }
            std::process::exit(if v.accepted { 0 } else { 1 });
        }
        "batch" => {
            let sys = std::fs::read(arg(&args, "--system").expect("--system")).expect("read system");
            let dir = PathBuf::from(arg(&args, "--dir").expect("--dir"));
            let py = manifest_verdicts(&dir.join("manifest.json"));
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("batch: cannot read {}: {e}", dir.display());
                    std::process::exit(2);
                }
            };
            let mut proofs: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().map_or(false, |x| x == "proof"))
                .filter(|p| !p.file_name().map_or(false, |n| n.to_string_lossy().starts_with("._"))) // macOS AppleDouble
                .collect();
            proofs.sort();
            if proofs.is_empty() {
                eprintln!("batch: no *.proof in {}", dir.display());
                std::process::exit(2);
            }
            let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            let jobs = arg(&args, "--jobs").map(|s| s.parse::<usize>().expect("--jobs")).unwrap_or(available).max(1).min(proofs.len());
            // The batch target (PROTOCOL.md 8c, "Batch verification"): the union of this verifier's own per-proof bounds
            // over the sub-batches presented must be <= 2^-target_bits.  It replaces the per-proof gate of `verify`
            // (a proof's own n_proofs is checked against the count presented instead); --soundness-bits is its alias here.
            let target_bits: f64 = arg(&args, "--target-bits")
                .or_else(|| arg(&args, "--soundness-bits"))
                .map(|s| s.parse().expect("--target-bits"))
                .unwrap_or(128.0);
            let opt = verify::Options { soundness_bits: None, ..opt };
            // shared, read-only, once: the parsed system + its pinned-digest verdict (the tables per (l, n) are built
            // by the first sub-batch that reaches them and reused by the rest)
            let t0 = Instant::now();
            let shared = verify::Shared::new(&sys, &opt);
            let shared_secs = t0.elapsed().as_secs_f64();
            let results = verify_batch(&shared, &proofs, &opt, reps, jobs);
            // the batch rule, inside the wall: this verifier's bound of every sub-batch (from its own statement) and the
            // n_proofs each was sized for, against the number presented -- arithmetic on N numbers
            let mut per_bits: Vec<f64> = Vec::with_capacity(results.len());
            let mut claimed: Vec<usize> = Vec::with_capacity(results.len());
            let mut names: Vec<String> = Vec::with_capacity(results.len());
            for item in &results {
                if let Some(v) = &item.verdict {
                    if let Some(s) = &v.soundness {
                        per_bits.push(s.per_proof_log2);
                        claimed.push(v.statement.as_ref().map_or(0, |st| st.n_proofs));
                        names.push(item.name.clone());
                    }
                }
            }
            let every_verdict_in = per_bits.len() == results.len() && results.iter().all(|i| i.verdict.as_ref().map_or(false, |v| v.accepted));
            let bound = if every_verdict_in { Some(verify::batch_bound(&per_bits, &claimed, target_bits, &names)) } else { None };
            let wall = t0.elapsed().as_secs_f64();
            // the report, in proof order whatever the scheduling
            let mut items = Vec::new();
            let (mut n_acc, mut n_rej, mut agree, mut disagree, mut n_coins) = (0, 0, 0, 0, 0);
            let mut total_secs = 0.0;
            let mut io_errors = 0;
            let mut first_reject: Option<String> = None;
            for item in &results {
                let name = &item.name;
                let v = match (&item.verdict, &item.io_error) {
                    (Some(v), _) => v,
                    (None, Some(e)) => {
                        io_errors += 1;
                        eprintln!("{name}: ERROR {e}");
                        continue;
                    }
                    (None, None) => unreachable!(),
                };
                total_secs += v.timing.total;
                n_coins += item.own_coins as usize;
                if v.accepted {
                    n_acc += 1
                } else {
                    n_rej += 1;
                    if first_reject.is_none() {
                        first_reject = Some(format!("{name}: {}", v.reason));
                    }
                }
                let pyv = py.iter().find(|(n, _, _)| n == name).map(|(_, a, w)| (*a, w.as_str()));
                if let Some((a, _)) = pyv {
                    if a == v.accepted {
                        agree += 1
                    } else {
                        disagree += 1
                    }
                }
                eprintln!(
                    "{name}: {} ({}) {:.3}s{}",
                    if v.accepted { "ACCEPT" } else { "reject" },
                    v.reason,
                    v.timing.total,
                    match pyv {
                        Some((a, w)) => format!("  python={} ({}){}", a, w, if a == v.accepted { "" } else { "  ** DISAGREE **" }),
                        None => String::new(),
                    }
                );
                items.push(verdict_json(v, name, pyv, reps));
            }
            // precedence: unreadable input, a rejected sub-batch (first in name order), then the batch rule
            let (batch_accepted, batch_reason) = match (&first_reject, io_errors, &bound) {
                (_, e, _) if e > 0 => (false, format!("{e} sub-batch file(s) could not be read")),
                (Some(r), _, _) => (false, r.clone()),
                (None, _, Some(b)) => (b.accepted && n_acc == proofs.len(), b.reason.clone()),
                (None, _, None) => (false, "batch: no verdict".to_string()),
            };
            // the batch accounting is reported whatever the verdict (over the sub-batches whose statements parsed)
            let batch_log2 = bound.as_ref().map(|b| b.batch_log2).unwrap_or_else(|| {
                let m = per_bits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if per_bits.is_empty() { f64::NEG_INFINITY } else { m + per_bits.iter().map(|x| (x - m).exp2()).sum::<f64>().log2() }
            });
            let f9 = |x: f64| if x.is_finite() { format!("{x:.9}") } else { "null".to_string() };
            let summary = format!(
                "{{\"dir\":{},\"jobs\":{jobs},\"available_parallelism\":{available},\"threads\":{},\"reps\":{},\"target_bits\":{target_bits},\"n\":{},\"accepted\":{n_acc},\"rejected\":{n_rej},\"own_coins\":{n_coins},\"batch_accepted\":{batch_accepted},\"batch_reason\":{},\"batch_log2\":{},\"batch_bits\":{},\"per_proof_bits\":[{}],\"n_proofs_claimed\":[{}],\"python_agree\":{agree},\"python_disagree\":{disagree},\"verify_seconds_sum\":{total_secs:.4},\"shared_seconds\":{shared_secs:.4},\"wall_seconds\":{wall:.4},\"items\":[{}]}}",
                json_str(&dir.to_string_lossy()),
                opt.threads,
                reps,
                proofs.len(),
                json_str(&batch_reason),
                f9(batch_log2),
                f9(-batch_log2),
                per_bits.iter().map(|b| f9(-b)).collect::<Vec<_>>().join(","),
                claimed.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(","),
                items.join(",")
            );
            eprintln!(
                "{} proofs: {n_acc} accepted, {n_rej} rejected ({n_coins} against their own coins); batch {} ({batch_reason}); batch bits {:.6} at target {target_bits}; python agreement {agree}/{} ({disagree} disagree); verify sum {total_secs:.3}s, wall {wall:.3}s at {jobs} jobs x {} threads",
                proofs.len(),
                if batch_accepted { "ACCEPT" } else { "reject" },
                -batch_log2,
                agree + disagree,
                opt.threads
            );
            if let Some(p) = arg(&args, "--json") {
                std::fs::write(p, &summary).expect("write json");
            } else {
                println!("{summary}");
            }
            // 0 every sub-batch accepted; 1 a sub-batch rejected; 2 a sub-batch file unreadable; 3 a verdict disagrees
            // with manifest.json's python_verdict (the agreement gate, checked first)
            std::process::exit(if disagree > 0 {
                3
            } else if io_errors > 0 {
                2
            } else if !batch_accepted {
                1
            } else {
                0
            });
        }
        _ => {
            eprintln!("usage: ligero-verify verify|batch|system-digest ... (see src/main.rs)");
            std::process::exit(2);
        }
    }
}
