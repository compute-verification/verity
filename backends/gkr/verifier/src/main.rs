//! Independent verifier for Candidate A's GPU proofs (`backends/gkr/gpu`, BabyBear^6 / X^6 - 31).
//!
//!   verity-gkr-verify verify    --dir D --proof P [--vus N] [--threads T] [--json OUT]   accept / reject one proof
//!   verity-gkr-verify negatives --dir D --proofs PD [--threads T] [--json OUT]           every PD/<id>.bin (D/neg/<id>) must be rejected
//!   verity-gkr-verify mutate    --dir D --proof P [--vus N] [--threads T] [--sample K] [--json OUT]
//!                                                                                        bit-flip / statement mutations of an accepted proof
//!
//! `D` is a `gpu.bb_export` directory.  The verifier reads `circuit.txt`, `epilogue.txt`, `manifest.json` (the
//! `steps` count) and `public.bin` -- never `units.bin` / `epilogue.bin` (the witness).

mod circuit;
mod field;
mod lanes;
mod ntt;
mod proof;
mod sha256;
mod transcript;
mod verify;

use circuit::{layers, read_rows, Circuit, Layer};
use std::path::{Path, PathBuf};
use std::time::Instant;
use verify::{Chain, Segment, Statement, LIGERO};

pub struct Loaded { pub circ: Circuit, pub layers: Vec<Layer> }

fn load_circuit(path: &Path) -> Loaded {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let circ = Circuit::parse(&text);
    let layers = layers(&circ);
    Loaded { circ, layers }
}

/// `"steps": N` from the export manifest (the only manifest field the verifier uses).
fn manifest_steps(root: &Path) -> usize {
    let text = std::fs::read_to_string(root.join("manifest.json")).expect("manifest.json");
    let key = "\"steps\":";
    let i = text.find(key).expect("manifest: steps") + key.len();
    let rest = text[i..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().expect("manifest: steps value")
}

fn col(c: &Circuit, name: &str) -> usize { c.col_index(name).unwrap_or_else(|| panic!("circuit has no column {name}")) }

/// The statement of a batch directory: circuits, `vus` (prefix of the public words), steps.  `d` may be a
/// `neg/<id>` subdirectory sharing its parent's manifest and circuits.
fn statement<'a>(unit: &'a Loaded, epi: &'a Loaded, d: &Path, vus: Option<usize>) -> Result<Statement<'a>, String> {
    let root = if d.join("manifest.json").is_file() { d.to_path_buf() } else { d.parent().unwrap().parent().unwrap().to_path_buf() };
    let steps = manifest_steps(&root);
    let (n, m, pub_words) = read_rows(&std::fs::read(d.join("public.bin")).map_err(|e| format!("public.bin: {e}"))?)?;
    if m != 1 { return Err("public.bin: one word per VU expected".into()); }
    let b = vus.unwrap_or(n);
    if b > n || b == 0 { return Err(format!("{n} public words, {b} requested")); }
    let uc = &unit.circ;
    let ec = &epi.circ;
    let chain = Chain {
        steps,
        col_c: vec![col(uc, "x.c_lo"), col(uc, "x.c_hi")],
        col_y: vec![col(uc, "y_lo"), col(uc, "y_hi")],
        epi_y32: vec![col(ec, "y32_lo"), col(ec, "y32_hi")],
        epi_y16: col(ec, "y16"),
        public_y16: pub_words[..b].to_vec(),
    };
    Ok(Statement {
        segs: vec![
            Segment { name: "unit".into(), circ: uc, layers: &unit.layers, units: b * steps },
            Segment { name: "epilogue".into(), circ: ec, layers: &epi.layers, units: b },
        ],
        chain,
    })
}

fn arg(args: &[String], key: &str) -> Option<String> { args.iter().position(|a| a == key).map(|i| args[i + 1].clone()) }

fn threads(args: &[String]) -> usize {
    arg(args, "--threads").map(|s| s.parse().unwrap())
        .or_else(|| std::env::var("VERITY_GKR_THREADS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

fn peak_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn cpu_model() -> Option<String> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    s.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
}

fn stats_json(res: &Result<(), String>, st: &verify::Stats, extra: &str) -> String {
    format!(
        "{{\"accepted\": {}, \"error\": {}, \"verify_seconds\": {:.6}, \"t_parse\": {:.6}, \"t_transcript\": {:.6}, \"t_functional_value\": {:.6}, \"t_merkle_proximity\": {:.6}, \"t_rows_linear_test\": {:.6}, \"t_rows_fill_cpu\": {:.6}, \"t_rows_eval_cpu\": {:.6}, \"bytes_read\": {}, \"msgs\": {}, \"slots\": {}, \"committed_elements\": {}, \"ligero_rows\": {}, \"threads\": {}, \"peak_rss_bytes\": {}, \"cpu\": {}{}}}",
        res.is_ok(), match res { Ok(()) => "null".to_string(), Err(e) => format!("\"{}\"", e.replace('"', "'")) },
        st.t_total, st.t_parse, st.t_transcript, st.t_konst, st.t_merkle, st.t_rows, st.t_rows_fill_cpu, st.t_rows_eval_cpu, st.bytes_read, st.msgs, st.slots, st.committed_elements, st.rows, st.threads,
        peak_rss_bytes().map(|v| v.to_string()).unwrap_or("null".into()), cpu_model().map(|m| format!("\"{m}\"")).unwrap_or("null".into()), extra)
}

fn cmd_verify(args: &[String]) -> i32 {
    let dir = PathBuf::from(arg(args, "--dir").expect("--dir"));
    let root = if dir.join("circuit.txt").is_file() { dir.clone() } else { dir.parent().unwrap().parent().unwrap().to_path_buf() };
    let unit = load_circuit(&root.join("circuit.txt"));
    let epi = load_circuit(&root.join("epilogue.txt"));
    let vus: Option<usize> = arg(args, "--vus").map(|s| s.parse().unwrap());
    let stmt = statement(&unit, &epi, &dir, vus).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) });
    let bytes = std::fs::read(arg(args, "--proof").expect("--proof")).expect("proof file");
    let th = threads(args);
    let (res, st) = verify::verify(&stmt, &bytes, &LIGERO, th);
    let extra = format!(", \"vus\": {}, \"units\": {}, \"steps\": {}", stmt.vus(), stmt.segs[0].units, stmt.chain.steps);
    let json = stats_json(&res, &st, &extra);
    println!("{json}");
    if let Some(out) = arg(args, "--json") { std::fs::write(out, json + "\n").expect("write json"); }
    if res.is_ok() { 0 } else { 1 }
}

fn cmd_negatives(args: &[String]) -> i32 {
    let dir = PathBuf::from(arg(args, "--dir").expect("--dir"));
    let proofs = PathBuf::from(arg(args, "--proofs").expect("--proofs"));
    let unit = load_circuit(&dir.join("circuit.txt"));
    let epi = load_circuit(&dir.join("epilogue.txt"));
    let th = threads(args);
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&proofs).expect("proofs dir").map(|e| e.unwrap().path()).filter(|p| p.extension().map(|e| e == "bin").unwrap_or(false) && p.file_stem().unwrap().to_string_lossy().parse::<u64>().is_ok()).collect();
    entries.sort_by_key(|p| p.file_stem().unwrap().to_string_lossy().parse::<u64>().unwrap_or(u64::MAX));
    let (mut rejected, mut accepted) = (0, 0);
    let mut rows = Vec::new();
    for p in &entries {
        let id = p.file_stem().unwrap().to_string_lossy().to_string();
        let stmt = statement(&unit, &epi, &dir.join("neg").join(&id), None).unwrap_or_else(|e| panic!("{id}: {e}"));
        let bytes = std::fs::read(p).unwrap();
        let (res, st) = verify::verify(&stmt, &bytes, &LIGERO, th);
        match &res {
            Ok(()) => { accepted += 1; println!("{id}: ACCEPTED (soundness bug)"); }
            Err(e) => { rejected += 1; println!("{id}: rejected: {e} ({:.3} s)", st.t_total); }
        }
        rows.push(format!("{{\"id\": \"{id}\", \"rejected\": {}, \"error\": {}, \"verify_seconds\": {:.6}}}", res.is_err(), match &res { Ok(()) => "null".to_string(), Err(e) => format!("\"{}\"", e.replace('"', "'")) }, st.t_total));
    }
    println!("negatives: {rejected} rejected, {accepted} accepted, {} proofs", entries.len());
    if let Some(out) = arg(args, "--json") {
        std::fs::write(out, format!("{{\"rejected\": {rejected}, \"accepted\": {accepted}, \"total\": {}, \"results\": [{}]}}\n", entries.len(), rows.join(", "))).unwrap();
    }
    if accepted > 0 || entries.is_empty() { 1 } else { 0 }
}

/// Deterministic xorshift for the sampled mutation campaign.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

/// Bit-flip mutations of an accepted proof, grouped by what they hit: the Merkle root, every extension message
/// (or a sample), the Ligero `w` / `q` coefficients, the opened columns (a value, a Merkle sibling), the message
/// count (truncation / extension), and the statement (a public word, the VU count).  Every mutation must be
/// rejected.  `--sample K`: at most K mutations per group (0 = exhaustive).
fn cmd_mutate(args: &[String]) -> i32 {
    let dir = PathBuf::from(arg(args, "--dir").expect("--dir"));
    let unit = load_circuit(&dir.join("circuit.txt"));
    let epi = load_circuit(&dir.join("epilogue.txt"));
    let vus: Option<usize> = arg(args, "--vus").map(|s| s.parse().unwrap());
    let stmt = statement(&unit, &epi, &dir, vus).unwrap();
    let bytes = std::fs::read(arg(args, "--proof").expect("--proof")).expect("proof file");
    let th = threads(args);
    let sample: usize = arg(args, "--sample").map(|s| s.parse().unwrap()).unwrap_or(0);
    let parallel: usize = arg(args, "--parallel").map(|s| s.parse().unwrap()).unwrap_or(1);
    let (res, _) = verify::verify(&stmt, &bytes, &LIGERO, th);
    assert!(res.is_ok(), "the unmutated proof must verify: {res:?}");
    // layout of proof.bin (gpu/README.md "Proof file")
    let n_msgs = u64::from_le_bytes(bytes[44..52].try_into().unwrap()) as usize;
    let msgs_at = 52;
    let after_msgs = msgs_at + 24 * n_msgs;
    let k = u64::from_le_bytes(bytes[after_msgs + 8..after_msgs + 16].try_into().unwrap()) as usize;
    let w_at = after_msgs + 16;
    let nq = u64::from_le_bytes(bytes[w_at + 24 * k..w_at + 24 * k + 8].try_into().unwrap()) as usize;
    let qc_at = w_at + 24 * k + 8;
    let shape_at = qc_at + 24 * nq;
    let t = u64::from_le_bytes(bytes[shape_at..shape_at + 8].try_into().unwrap()) as usize;
    let rows = u64::from_le_bytes(bytes[shape_at + 8..shape_at + 16].try_into().unwrap()) as usize;
    let depth = u64::from_le_bytes(bytes[shape_at + 16..shape_at + 24].try_into().unwrap()) as usize;
    let cols_at = shape_at + 24;
    let per_col = 4 * rows + 32 * depth;
    let mut rng = Rng(0x9e3779b97f4a7c15);
    // (group, byte offset, bit) -- flipping a bit of a canonical field element may leave it canonical or not;
    // both are mutations the verifier must reject (non-canonical: at parse; canonical: by the checks)
    let mut muts: Vec<(&str, usize, u8)> = Vec::new();
    let pick = |rng: &mut Rng, total: usize, sample: usize| -> Vec<usize> {
        if sample == 0 || sample >= total { (0..total).collect() } else { (0..sample).map(|_| rng.below(total)).collect() }
    };
    for bit in pick(&mut rng, 256, if sample == 0 { 0 } else { sample.min(32) }) { muts.push(("merkle_root", 12 + bit / 8, (bit % 8) as u8)); }
    for i in pick(&mut rng, n_msgs, sample) { let bit = rng.below(31); muts.push(("message", msgs_at + 24 * i + 4 * rng.below(6) + bit / 8, (bit % 8) as u8)); }
    for i in pick(&mut rng, k, sample) { let bit = rng.below(31); muts.push(("ligero_w", w_at + 24 * i + 4 * rng.below(6) + bit / 8, (bit % 8) as u8)); }
    for i in pick(&mut rng, nq, sample) { let bit = rng.below(31); muts.push(("ligero_qc", qc_at + 24 * i + 4 * rng.below(6) + bit / 8, (bit % 8) as u8)); }
    for oi in pick(&mut rng, t, sample) {
        let bit = rng.below(31);
        let ri = rng.below(rows);
        muts.push(("column_value", cols_at + oi * per_col + 4 * ri + bit / 8, (bit % 8) as u8));
        let di = rng.below(depth);
        let b = rng.below(256);
        muts.push(("merkle_sibling", cols_at + oi * per_col + 4 * rows + 32 * di + b / 8, (b % 8) as u8));
    }
    let n_total = muts.len() + 2 + 2;
    println!("mutations: {} bit flips + 2 count mutations + 2 statement mutations = {n_total}", muts.len());
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<Vec<(usize, bool, String, f64)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..parallel).map(|_| s.spawn(|| {
            let mut out = Vec::new();
            loop {
                let mi = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if mi >= muts.len() { break; }
                let (_, off, bit) = muts[mi];
                let mut m = bytes.clone();
                m[off] ^= 1 << bit;
                let t0 = Instant::now();
                let (r, _) = verify::verify(&stmt, &m, &LIGERO, th);
                out.push((mi, r.is_err(), r.err().unwrap_or_default(), t0.elapsed().as_secs_f64()));
            }
            out
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut per_group: std::collections::BTreeMap<&str, (usize, usize)> = std::collections::BTreeMap::new();
    let mut accepted_list = Vec::new();
    let mut errors: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut t_sum = 0.0;
    for part in results {
        for (mi, rejected, err, dt) in part {
            let g = muts[mi].0;
            let e = per_group.entry(g).or_insert((0, 0));
            e.1 += 1;
            if rejected { e.0 += 1; } else { accepted_list.push(format!("{g}@{}", muts[mi].1)); }
            let key = err.split(|c: char| c.is_ascii_digit()).next().unwrap_or("").trim().to_string();
            *errors.entry(key).or_insert(0) += 1;
            t_sum += dt;
        }
    }
    // count mutations: drop the last message / append one (the file's own count is adjusted so only the
    // statement-derived expectation can catch it)
    let mut count_ok = 0;
    for delta in [-1i64, 1] {
        let mut m = Vec::with_capacity(bytes.len() + 24);
        m.extend_from_slice(&bytes[..44]);
        m.extend_from_slice(&((n_msgs as i64 + delta) as u64).to_le_bytes());
        let keep = if delta < 0 { n_msgs - 1 } else { n_msgs };
        m.extend_from_slice(&bytes[msgs_at..msgs_at + 24 * keep]);
        if delta > 0 { m.extend_from_slice(&bytes[msgs_at..msgs_at + 24]); }
        m.extend_from_slice(&bytes[after_msgs..]);
        let (r, _) = verify::verify(&stmt, &m, &LIGERO, th);
        if r.is_err() { count_ok += 1; } else { accepted_list.push(format!("message_count{delta:+}")); }
        *errors.entry(r.err().unwrap_or_default().split(|c: char| c.is_ascii_digit()).next().unwrap_or("").trim().to_string()).or_insert(0) += 1;
    }
    per_group.insert("message_count", (count_ok, 2));
    // statement mutations: one public word changed (+1), one VU fewer claimed
    let mut stmt_ok = 0;
    {
        let mut s2 = statement(&unit, &epi, &dir, vus).unwrap();
        let v = rng.below(s2.vus());
        s2.chain.public_y16[v] = (s2.chain.public_y16[v] + 1) % field::P;
        let (r, _) = verify::verify(&s2, &bytes, &LIGERO, th);
        if r.is_err() { stmt_ok += 1; } else { accepted_list.push("public_word".into()); }
        *errors.entry(r.err().unwrap_or_default().split(|c: char| c.is_ascii_digit()).next().unwrap_or("").trim().to_string()).or_insert(0) += 1;
        if stmt.vus() > 1 {
            let s3 = statement(&unit, &epi, &dir, Some(stmt.vus() - 1)).unwrap();
            let (r, _) = verify::verify(&s3, &bytes, &LIGERO, th);
            if r.is_err() { stmt_ok += 1; } else { accepted_list.push("vu_count".into()); }
            *errors.entry(r.err().unwrap_or_default().split(|c: char| c.is_ascii_digit()).next().unwrap_or("").trim().to_string()).or_insert(0) += 1;
        } else { stmt_ok += 1; }
    }
    per_group.insert("statement", (stmt_ok, 2));
    let total_rej: usize = per_group.values().map(|v| v.0).sum();
    let total: usize = per_group.values().map(|v| v.1).sum();
    for (g, (r, n)) in &per_group { println!("  {g}: {r}/{n} rejected"); }
    println!("rejection reasons: {errors:?}");
    println!("mutations rejected: {total_rej}/{total}; mean verify {:.3} s; accepted (BUGS): {accepted_list:?}", t_sum / muts.len().max(1) as f64);
    if let Some(out) = arg(args, "--json") {
        let groups: Vec<String> = per_group.iter().map(|(g, (r, n))| format!("\"{g}\": {{\"rejected\": {r}, \"total\": {n}}}")).collect();
        let errs: Vec<String> = errors.iter().map(|(e, n)| format!("\"{}\": {n}", e.replace('"', "'"))).collect();
        std::fs::write(out, format!("{{\"rejected\": {total_rej}, \"total\": {total}, \"groups\": {{{}}}, \"reasons\": {{{}}}, \"accepted\": [{}]}}\n", groups.join(", "), errs.join(", "), accepted_list.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(", "))).unwrap();
    }
    if total_rej == total { 0 } else { 1 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("usage: verify|negatives|mutate ..."); std::process::exit(2); }
    let code = match args[1].as_str() {
        "verify" => cmd_verify(&args),
        "negatives" => cmd_negatives(&args),
        "mutate" => cmd_mutate(&args),
        _ => { eprintln!("unknown command"); 2 }
    };
    std::process::exit(code);
}
