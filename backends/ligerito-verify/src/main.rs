//! ligerito-verify: the independent verifier for the Ligerito backend.
//!
//! ~~~text
//! ligerito-verify verify --proof X.proof [--coins X.coins] [--label L] [--json out.json] [--reps R]
//!     # one PCS proof: LGRTOPV0 (statement embedded) or proto/pcs.py bytes (point in X.z); --coins = the
//!     # verifier's coins of a live session (u32 LE length + bytes each; interactive replay; default: Fiat-Shamir
//!     # diagnostic with transcript label L, default "ligerito-proto/v1")
//! ligerito-verify batch --dir DIR [--jobs N] [--json out.json] [--label L] [--target-bits B]
//!     # every DIR/*.proof (with DIR/<name>.z / <name>.coins when present) must verify and every DIR/*.proof.neg must
//!     # be rejected; with DIR/manifest.json (a list, or {"proofs": [...]}, of {"proof": name, "accepted": bool,
//!     # "reason": str}) every listed verdict must agree with ours; with --target-bits the union of the proofs' own
//!     # soundness bounds must be <= 2^-B.  Exit 0 iff accepted.
//! ligerito-verify selftest [--n N] [--kprime a,b,c] [--queries s,s,s] [--rate-log2 R|r1,r2,r3] [--emit DIR]
//!     # reference prover -> verifier round trip (+ negatives); --emit writes the proof / coins / negatives to DIR
//! ~~~

use ligerito_verify::format::{looks_like_proto, read_pcs_v0, read_point, read_proto, write_pcs_v0};
use ligerito_verify::params::Params;
use ligerito_verify::pcs::{verify, Report};
use ligerito_verify::refprover::{prove, random_point, random_vector, tamper};
use ligerito_verify::transcript::{FiatShamir, Live};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
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

struct Verdict {
    name: String,
    accepted: bool,
    reason: String,
    report: Option<Report>,
    params: Option<Params>,
    bytes: usize,
    mode: &'static str,
    seconds: f64,
}

fn verdict_json(v: &Verdict) -> String {
    let rep = match &v.report {
        Some(r) => format!(
            "{{\"coins\":{},\"soundness_log2\":{:.3},\"opened_rows\":{},\"terms_final\":{}}}",
            r.coins, r.soundness_log2, r.opened_rows, r.terms_final
        ),
        None => "null".into(),
    };
    let params = match &v.params {
        Some(p) => format!(
            "{{\"n\":{},\"kprime\":{:?},\"queries\":{:?},\"rate_log2\":{:?},\"domain_shift\":{}}}",
            p.n, p.kprime, p.queries, p.rate_log2, p.domain_shift
        ),
        None => "null".into(),
    };
    format!(
        "{{\"name\":{},\"accepted\":{},\"reason\":{},\"mode\":\"{}\",\"proof_bytes\":{},\"seconds\":{:.6},\"report\":{},\"params\":{}}}",
        json_str(&v.name),
        v.accepted,
        json_str(&v.reason),
        v.mode,
        v.bytes,
        v.seconds,
        rep,
        params
    )
}

/// Verify one proof file, `reps` times (timing), with replayed coins when a coins file is given.
/// `X.proof` / `X.proof.neg` -> `X.<ext>` when it exists.
fn sidecar(proof_path: &Path, ext: &str) -> Option<PathBuf> {
    let name = proof_path.file_name()?.to_string_lossy().to_string();
    let stem = name.strip_suffix(".proof.neg").or_else(|| name.strip_suffix(".proof"))?;
    let p = proof_path.with_file_name(format!("{stem}.{ext}"));
    if p.exists() { Some(p) } else { None }
}

/// DIR/manifest.json: a JSON list (or an object with a "proofs" list) of {"proof": name, "accepted": bool, "reason": str}
/// -- the Python verifier's verdicts, matched by file name.  Missing or unparsable manifest: no entries.
fn manifest_verdicts(dir: &Path) -> Vec<(String, bool, String)> {
    let text = match std::fs::read_to_string(dir.join("manifest.json")) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let doc = match ligerito_verify::json::parse(&text) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let list = doc.as_arr().or_else(|| doc.get("proofs").and_then(|p| p.as_arr())).unwrap_or(&[]);
    list.iter()
        .filter_map(|e| {
            let name = e.get("proof")?.as_str()?.to_string();
            let acc = e.get("accepted")?.as_bool()?;
            let why = e.get("reason").and_then(|r| r.as_str()).unwrap_or("").to_string();
            Some((name, acc, why))
        })
        .collect()
}

fn verify_file(proof_path: &Path, coins_path: Option<&Path>, label: &[u8], reps: usize) -> Verdict {
    let name = proof_path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let bytes = match std::fs::read(proof_path) {
        Ok(b) => b,
        Err(e) => {
            return Verdict { name, accepted: false, reason: format!("read: {e}"), report: None, params: None, bytes: 0, mode: "-", seconds: 0.0 }
        }
    };
    let coins_raw = match coins_path {
        Some(p) => match std::fs::read(p) {
            Ok(b) => Some(b),
            Err(e) => {
                return Verdict { name, accepted: false, reason: format!("coins: {e}"), report: None, params: None, bytes: bytes.len(), mode: "-", seconds: 0.0 }
            }
        },
        None => None,
    };
    let mode = if coins_raw.is_some() { "interactive-replay" } else { "fiat-shamir" };
    let t0 = Instant::now();
    let mut result: Result<Report, String> = Err("no reps".into());
    let mut params = None;
    for _ in 0..reps.max(1) {
        result = (|| {
            let (stmt, proof) = if looks_like_proto(&bytes) {
                let zp = sidecar(proof_path, "z").unwrap_or_else(|| proof_path.with_extension("z"));
                let zraw = std::fs::read(&zp).map_err(|e| format!("proto proof needs its point {}: {e}", zp.display()))?;
                read_proto(&bytes, read_point(&zraw)?)?
            } else {
                read_pcs_v0(&bytes)?
            };
            params = Some(proof.params.clone());
            let rep = match &coins_raw {
                Some(raw) => {
                    let mut lv = Live::parse(raw)?;
                    let rep = verify(&stmt, &proof, &mut lv)?;
                    if !lv.exhausted() {
                        return Err("coins: unused verifier coins after the protocol ended".into());
                    }
                    rep
                }
                None => verify(&stmt, &proof, &mut FiatShamir::new(label))?,
            };
            Ok(rep)
        })();
    }
    let seconds = t0.elapsed().as_secs_f64() / reps.max(1) as f64;
    match result {
        Ok(rep) => Verdict { name, accepted: true, reason: "ok".into(), report: Some(rep), params, bytes: bytes.len(), mode, seconds },
        Err(e) => Verdict { name, accepted: false, reason: e, report: None, params, bytes: bytes.len(), mode, seconds },
    }
}

fn write_json(path: Option<String>, text: &str) {
    if let Some(p) = path {
        std::fs::write(&p, text).unwrap_or_else(|e| panic!("--json {p}: {e}"));
    }
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',').filter(|x| !x.is_empty()).map(|x| x.trim().parse().expect("list of integers")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let label = arg(&args, "--label").unwrap_or_else(|| String::from_utf8_lossy(FiatShamir::LABEL).to_string());
    let reps: usize = arg(&args, "--reps").map(|s| s.parse().expect("--reps")).unwrap_or(1);
    match cmd {
        "verify" => {
            let proof = PathBuf::from(arg(&args, "--proof").expect("--proof FILE"));
            let coins = arg(&args, "--coins").map(PathBuf::from);
            let v = verify_file(&proof, coins.as_deref(), label.as_bytes(), reps);
            let js = verdict_json(&v);
            println!("{js}");
            write_json(arg(&args, "--json"), &js);
            std::process::exit(if v.accepted { 0 } else { 1 });
        }
        "batch" => {
            let dir = PathBuf::from(arg(&args, "--dir").expect("--dir DIR"));
            let jobs: usize = arg(&args, "--jobs")
                .map(|s| s.parse().expect("--jobs"))
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
            let target_bits: Option<f64> = arg(&args, "--target-bits").map(|s| s.parse().expect("--target-bits"));
            // *.proof must verify; *.proof.neg must be rejected
            let mut files: Vec<(PathBuf, bool)> = std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("--dir {}: {e}", dir.display()))
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter_map(|p| {
                    let name = p.file_name()?.to_string_lossy().to_string();
                    if name.ends_with(".proof") {
                        Some((p, true))
                    } else if name.ends_with(".proof.neg") {
                        Some((p, false))
                    } else {
                        None
                    }
                })
                .collect();
            files.sort();
            let manifest = manifest_verdicts(&dir);
            let next = AtomicUsize::new(0);
            let results: Vec<std::sync::Mutex<Option<Verdict>>> = files.iter().map(|_| std::sync::Mutex::new(None)).collect();
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                for _ in 0..jobs.max(1).min(files.len().max(1)) {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= files.len() {
                            break;
                        }
                        let p = &files[i].0;
                        let coins = sidecar(p, "coins");
                        let v = verify_file(p, coins.as_deref(), label.as_bytes(), reps);
                        *results[i].lock().unwrap() = Some(v);
                    });
                }
            });
            let verdicts: Vec<Verdict> = results.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect();
            let mut problems: Vec<String> = Vec::new();
            if verdicts.is_empty() {
                problems.push("no *.proof in the directory".into());
            }
            for (v, (_, expect)) in verdicts.iter().zip(&files) {
                if v.accepted != *expect {
                    problems.push(format!("{}: {} but expected {}", v.name, if v.accepted { "accepted" } else { "rejected" }, if *expect { "acceptance" } else { "rejection" }));
                }
            }
            let mut manifest_checked = 0usize;
            for (name, py_accepted, py_reason) in &manifest {
                if let Some(v) = verdicts.iter().find(|v| &v.name == name) {
                    manifest_checked += 1;
                    if v.accepted != *py_accepted {
                        problems.push(format!("{name}: python_verdict {py_accepted} ({py_reason}) but ours is {} ({})", v.accepted, v.reason));
                    }
                }
            }
            // union bound over the accepted proofs' own soundness figures
            let union_log2 = {
                let s: f64 = verdicts.iter().filter(|v| v.accepted).filter_map(|v| v.report.as_ref()).map(|r| r.soundness_log2.exp2()).sum();
                if s > 0.0 { s.log2() } else { f64::NEG_INFINITY }
            };
            if let Some(b) = target_bits {
                if union_log2 > -b {
                    problems.push(format!("union soundness 2^{union_log2:.2} exceeds the target 2^-{b}"));
                }
            }
            let all = problems.is_empty();
            let js = format!(
                "{{\"dir\":{},\"accepted\":{},\"n_proofs\":{},\"n_negatives\":{},\"n_accepted\":{},\"manifest_checked\":{},\"union_soundness_log2\":{},\"problems\":[{}],\"wall_seconds\":{:.6},\"proofs\":[{}]}}",
                json_str(&dir.display().to_string()),
                all,
                files.iter().filter(|f| f.1).count(),
                files.iter().filter(|f| !f.1).count(),
                verdicts.iter().filter(|v| v.accepted).count(),
                manifest_checked,
                if union_log2.is_finite() { format!("{union_log2:.3}") } else { "null".into() },
                problems.iter().map(|p| json_str(p)).collect::<Vec<_>>().join(","),
                t0.elapsed().as_secs_f64(),
                verdicts.iter().map(verdict_json).collect::<Vec<_>>().join(",")
            );
            println!("{js}");
            write_json(arg(&args, "--json"), &js);
            std::process::exit(if all { 0 } else { 1 });
        }
        "selftest" => {
            let n: usize = arg(&args, "--n").map(|s| s.parse().expect("--n")).unwrap_or(12);
            let kprime = arg(&args, "--kprime").map(|s| parse_list(&s)).unwrap_or_else(|| vec![4, 3, 2]);
            let queries = arg(&args, "--queries").map(|s| parse_list(&s)).unwrap_or_else(|| vec![16; kprime.len()]);
            // --rate-log2 R or R1,R2,... (per committed round)
            let rates: Vec<usize> = arg(&args, "--rate-log2").map(|s| parse_list(&s)).unwrap_or_else(|| vec![2]);
            let rates = if rates.len() == 1 { vec![rates[0]; kprime.len()] } else { rates };
            let p = Params::with_rates(n, kprime, queries, rates);
            p.validate().unwrap_or_else(|e| panic!("{e}"));
            let f = random_vector(n, 1);
            let z = random_point(n, 2);
            let t0 = Instant::now();
            let (stmt, proof) = prove(&p, &f, &z, &mut FiatShamir::new(label.as_bytes()));
            let t_prove = t0.elapsed().as_secs_f64();
            let n_coins = p.kprime.iter().sum::<usize>() + 2 * p.rounds() - 1;
            let coins = Live::random_coins(n_coins, 12345);
            let (stmt_live, proof_live) = prove(&p, &f, &z, &mut Live::new(coins.clone()));
            verify(&stmt_live, &proof_live, &mut Live::new(coins.clone())).unwrap_or_else(|e| panic!("selftest live: {e}"));
            let bytes = write_pcs_v0(&stmt, &proof);
            let t1 = Instant::now();
            let rep = verify(&stmt, &proof, &mut FiatShamir::new(label.as_bytes())).unwrap_or_else(|e| panic!("selftest: {e}"));
            let t_verify = t1.elapsed().as_secs_f64();
            let negs = tamper(&stmt, &proof);
            let mut rejected = 0;
            for (what, s, pf) in &negs {
                match verify(s, pf, &mut FiatShamir::new(label.as_bytes())) {
                    Ok(_) => eprintln!("NEGATIVE ACCEPTED: {what}"),
                    Err(_) => rejected += 1,
                }
            }
            let (snd, terms) = p.soundness_log2();
            println!(
                "{{\"params\":{{\"n\":{},\"kprime\":{:?},\"queries\":{:?},\"rate_log2\":{:?}}},\"proof_bytes\":{},\"coins\":{},\"t_prove\":{:.4},\"t_verify\":{:.6},\"negatives\":{},\"rejected\":{},\"soundness_log2\":{:.2},\"terms\":{{{}}}}}",
                p.n, p.kprime, p.queries, p.rate_log2, bytes.len(), rep.coins, t_prove, t_verify, negs.len(), rejected, snd,
                terms.iter().map(|(k, v)| format!("\"{k}\":{v:.2}")).collect::<Vec<_>>().join(",")
            );
            if let Some(dir) = arg(&args, "--emit") {
                let dir = PathBuf::from(dir);
                std::fs::create_dir_all(&dir).expect("--emit dir");
                std::fs::write(dir.join("honest.proof"), &bytes).unwrap();
                std::fs::write(dir.join("live.proof"), write_pcs_v0(&stmt_live, &proof_live)).unwrap();
                std::fs::write(dir.join("live.coins"), Live::serialize(&coins)).unwrap();
                std::fs::write(dir.join("proto.proof"), ligerito_verify::format::write_proto(&stmt, &proof, rep.coins)).unwrap();
                std::fs::write(dir.join("proto.z"), ligerito_verify::transcript::ext_bytes(&stmt.z)).unwrap();
                let mut written = 0;
                for (j, (what, s, pf)) in negs.iter().enumerate() {
                    let b = write_pcs_v0(s, pf);
                    if b == bytes {
                        continue; // an in-memory tamper the container cannot express (e.g. an ext entry in a base row)
                    }
                    std::fs::write(dir.join(format!("neg{j:02}.proof.neg")), &b).unwrap();
                    std::fs::write(dir.join(format!("neg{j:02}.what")), what).unwrap();
                    written += 1;
                }
                eprintln!("wrote {} (honest.proof, live.proof/.coins, proto.proof/.z + {written} negatives as *.proof.neg)", dir.display());
            }
            std::process::exit(if rejected == negs.len() { 0 } else { 1 });
        }
        _ => {
            eprintln!("usage: ligerito-verify verify --proof X.proof [--coins X.coins] | batch --dir DIR [--jobs N] | selftest [--n N --kprime a,b --queries s,s --rate-log2 R --emit DIR]");
            std::process::exit(2);
        }
    }
}
