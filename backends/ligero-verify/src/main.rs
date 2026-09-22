//! ligero-verify: the independent verifier for Candidate B's B-Ligero chain proofs.
//!
//! ~~~text
//! ligero-verify verify --system system.bin --statement X.stmt --proof X.proof [--threads T] [--soundness-bits 128]
//!                      [--coins FILE]   (interactive mode: the verifier's own r1||s1||r2||s2, 128 bytes raw or hex)
//!                      [--allow-any-system] [--json out.json] [--reps R]
//! ligero-verify batch  --system system.bin --dir DIR [--threads T] [--soundness-bits 128] [--allow-any-system]
//!                      [--json out.json] [--reps R]          # every DIR/*.proof with its DIR/*.stmt; agreement with
//!                                                            # DIR/manifest.json's python_verdict when present
//! ligero-verify system-digest --system system.bin           # the two digests to pin in verify.rs
//! ~~~

mod auth;
mod field;
mod format;
mod hash;
mod verify;

use std::path::{Path, PathBuf};
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn options(args: &[String]) -> verify::Options {
    verify::Options {
        threads: arg(args, "--threads").map(|s| s.parse().expect("--threads")).unwrap_or(1),
        soundness_bits: arg(args, "--soundness-bits").map(|s| s.parse().expect("--soundness-bits")).unwrap_or(128.0),
        allow_any_system: flag(args, "--allow-any-system"),
        own_coins: arg(args, "--coins").map(|path| {
            let raw = std::fs::read(path).expect("--coins file");
            let raw = if raw.len() == 256 || raw.len() == 257 {
                let txt = String::from_utf8_lossy(&raw).trim().to_string();
                (0..128).map(|i| u8::from_str_radix(&txt[2 * i..2 * i + 2], 16).expect("--coins hex")).collect::<Vec<u8>>()
            } else {
                raw
            };
            let mut out = [0u8; 128];
            assert_eq!(raw.len(), 128, "--coins must hold r1 || s1 || r2 || s2 (128 bytes, raw or hex)");
            out.copy_from_slice(&raw);
            out
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
        "{{\"parse\":{:.6},\"soundness\":{:.6},\"statement_hash\":{:.6},\"challenge_hash\":{:.6},\"merkle\":{:.6},\"ntt\":{:.6},\"dfold\":{:.6},\"quadratic\":{:.6},\"chain\":{:.6},\"auth\":{:.6},\"total\":{:.6}}}",
        t.parse, t.soundness, t.statement_hash, t.challenge_hash, t.merkle, t.ntt, t.dfold, t.quadratic, t.chain, t.auth, t.total
    )
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
        "{{\"name\":{},\"accepted\":{},\"reason\":{},\"timing\":{},\"reps\":{},\"soundness\":{}{}}}",
        json_str(name),
        v.accepted,
        json_str(&v.reason),
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
            let mut items = Vec::new();
            let (mut n_acc, mut n_rej, mut agree, mut disagree) = (0, 0, 0, 0);
            let mut total_secs = 0.0;
            let t0 = Instant::now();
            for p in &proofs {
                let name = p.file_name().unwrap().to_string_lossy().to_string();
                let stp = p.with_extension("stmt");
                let st = std::fs::read(&stp).expect("read statement");
                let pf = std::fs::read(p).expect("read proof");
                let v = run_one(&sys, &st, &pf, &opt, reps);
                total_secs += v.timing.total;
                if v.accepted {
                    n_acc += 1
                } else {
                    n_rej += 1
                }
                let pyv = py.iter().find(|(n, _, _)| *n == name).map(|(_, a, w)| (*a, w.as_str()));
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
                items.push(verdict_json(&v, &name, pyv, reps));
            }
            let wall = t0.elapsed().as_secs_f64();
            let summary = format!(
                "{{\"dir\":{},\"threads\":{},\"reps\":{},\"soundness_bits\":{},\"n\":{},\"accepted\":{n_acc},\"rejected\":{n_rej},\"python_agree\":{agree},\"python_disagree\":{disagree},\"verify_seconds_sum\":{total_secs:.4},\"wall_seconds\":{wall:.4},\"items\":[{}]}}",
                json_str(&dir.to_string_lossy()),
                opt.threads,
                reps,
                opt.soundness_bits,
                proofs.len(),
                items.join(",")
            );
            eprintln!(
                "{} proofs: {n_acc} accepted, {n_rej} rejected; python agreement {agree}/{} ({disagree} disagree); verify sum {total_secs:.3}s at {} threads",
                proofs.len(),
                agree + disagree,
                opt.threads
            );
            if let Some(p) = arg(&args, "--json") {
                std::fs::write(p, &summary).expect("write json");
            } else {
                println!("{summary}");
            }
            std::process::exit(if disagree > 0 { 3 } else { 0 });
        }
        _ => {
            eprintln!("usage: ligero-verify verify|batch|system-digest ... (see src/main.rs)");
            std::process::exit(2);
        }
    }
}
