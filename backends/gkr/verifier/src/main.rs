//! Independent verifier for Candidate A's GPU proofs (`backends/gkr/gpu`, BabyBear^6 / X^6 - 22).
//!
//!   verity-gkr-verify verify    --dir D --proof P (--relation R | --allow-any-circuit) [--vus N] [--threads T] [--json OUT] [--trace-logup]
//!                               [--require-bound] [--allow-unpinned-instances] [--require-commitment] [--allow-unpinned-commitment]
//!                                                                                        accept / reject one proof (the flag prints every LogUp level's operands)
//!   verity-gkr-verify negatives --dir D --proofs PD [--threads T] [--json OUT]           every PD/<id>.bin (D/neg/<id>) must be rejected
//!   verity-gkr-verify mutate    --dir D --proof P [--vus N] [--threads T] [--sample K] [--json OUT]
//!                                                                                        bit-flip / statement mutations of an accepted proof
//!   verity-gkr-verify circuit-digest --dir D                                             D's circuit set and the relation it is pinned for
//!
//! `D` is a `gpu.bb_export` directory.  The verifier reads `circuit.txt`, `epilogue.txt`, `manifest.json` (the
//! `steps` count) and `public.bin` -- never `units.bin` / `epilogue.bin` (the witness).  `verify` refuses circuit files
//! that are not a pinned circuit set of the relation `R` the proof is claimed for (pins.rs); `--allow-any-circuit`
//! lifts the pin and the verdict then reports `circuit_pinned = false`.
//!
//! With a `bind.txt` the statement is operands-bound (PROTOCOL 16): it also reads `instances.txt`, `x.bin`, `w.bin`,
//! requires `bind.txt` to be the unit circuit's operand columns, and batches the binding into the functional.  A bound
//! statement is claimed as `R+bound`: the circuit files must be a pinned set of `R` and the instance files the
//! `R+bound` pin of `instances.rs`; `--relation R` refuses a bound statement and `--relation R+bound` an unbound one.
//! Under `--allow-any-circuit` only, `--require-bound` refuses an unbound statement and `--allow-unpinned-instances`
//! skips the instance pin.
//!
//! With a `commitment.txt` the statement is operands-committed (commitments.rs): its public words carry each VU's
//! x row and W column digests after the `y` words, and the verifier rebuilds the scheme's trees from them and compares
//! the roots with the pinned roots of the instance set.  It is claimed as `R+sha256` / `R+blake3`, whose circuit set
//! (the hash layers binding the digests to the operand columns) is pinned under that name.  Under
//! `--allow-any-circuit` only, `--require-commitment` refuses an uncommitted statement and `--allow-unpinned-commitment`
//! skips the root pin.

mod circuit;
mod commitments;
mod field;
mod instances;
mod lanes;
mod ntt;
mod pins;
mod proof;
mod sha256;
mod sha512;
mod transcript;
mod verify;

use circuit::{layers, read_rows, Circuit, Layer};
use std::path::{Path, PathBuf};
use std::time::Instant;
use verify::{Chain, Link, Segment, Statement, LIGERO};

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

/// `"variant": "v2"` in the manifest selects the checker-v2 statement (PROTOCOL 14: `chain.txt` links, implicit tables).
fn manifest_is_v2(root: &Path) -> bool {
    let text = std::fs::read_to_string(root.join("manifest.json")).expect("manifest.json");
    text.split("\"variant\":").nth(1).map(|r| r.trim_start().starts_with("\"v2\"")).unwrap_or(false)
}

/// `chain.txt` (`gpu/run.py::read_chain`, `main.rs::read_chain`): `link <in|-> <out> <epi> <init> <zero_nonlast>`,
/// and optionally one `public <epi col> ...` line naming the epilogue columns bound to `public.bin`'s columns
/// (default `public y16`).
fn read_chain(path: &Path, uc: &Circuit, ec: &Circuit) -> Result<(Vec<Link>, Vec<usize>, [u8; 32]), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut links = Vec::new();
    let mut epi_pub: Option<Vec<usize>> = None;
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() { continue; }
        if t[0] == "public" {
            if epi_pub.is_some() || t.len() < 2 { return Err("chain.txt: one public line with at least one column".into()); }
            epi_pub = Some(t[1..].iter().map(|n| col(ec, n)).collect());
            continue;
        }
        if t[0] != "link" || t.len() != 6 { return Err(format!("chain.txt: unknown line {line:?}")); }
        let init: u64 = t[4].parse().map_err(|_| "chain.txt: init".to_string())?;
        if init >= field::P as u64 { return Err("chain.txt: init is not a canonical BabyBear element (a Goldilocks chain file?)".into()); }
        let col_c = if t[1] == "-" { None } else { Some(col(uc, t[1])) };
        links.push(Link { col_c, col_y: col(uc, t[2]), epi: col(ec, t[3]), init: init as u32, zero_nonlast: t[5] == "1" });
    }
    let epi_pub = epi_pub.unwrap_or_else(|| vec![col(ec, "y16")]);
    Ok((links, epi_pub, sha256::Sha256::digest(text.as_bytes())))
}

fn manifest_str(root: &Path, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(root.join("manifest.json")).ok()?;
    let rest = text.split(&format!("\"{key}\":")).nth(1)?.trim_start().strip_prefix('"')?;
    Some(rest[..rest.find('"')?].to_string())
}

fn hex(d: &[u8]) -> String { d.iter().map(|b| format!("{b:02x}")).collect() }

/// The operand columns an export's unit circuit must bind, as `bind.txt` text (`gpu/bind.py::derive`, re-derived here
/// from the column layout): checker v2 `one, x.s, x.e, x.M, x.z, <k x words>, <k w words>, y.s, ...` binds column
/// `5 + j` to `x[v, k i + j]` and `5 + k + j` to `w[v, k i + j]`; fp4-nvf4 binds `x.p{i} = 16 a + b` for the 64 E2M1
/// code pairs and `x.s{g} = 128 a + b` for the 4 UE4M3 scale pairs of the 68-byte step.
fn expected_bind(uc: &Circuit, root: &Path) -> Result<String, String> {
    let mut s = String::new();
    if manifest_str(root, "relation").as_deref() == Some("fp4-nvf4") {
        for i in 0..64 { s += &format!("bind x.p{i} x:68:{i}:16 w:68:{i}:1\n"); }
        for g in 0..4 { s += &format!("bind x.s{g} x:68:{}:128 w:68:{}:1\n", 64 + g, 64 + g); }
        return Ok(s);
    }
    let n = &uc.col_names;
    let ys = n.iter().position(|c| c == "y.s").ok_or("bind: the unit circuit has no y.s column")?;
    if n.len() < 5 || n[..5] != ["one", "x.s", "x.e", "x.M", "x.z"] || ys < 7 || (ys - 5) % 2 != 0 {
        return Err("bind: not a checker-v2 unit column layout".into());
    }
    let k = (ys - 5) / 2;
    for j in 0..k { s += &format!("bind {} x:{k}:{j}:1\n", n[5 + j]); }
    for j in 0..k { s += &format!("bind {} w:{k}:{j}:1\n", n[5 + k + j]); }
    Ok(s)
}

/// What an operands-bound statement names (`instances.txt`) and what this verifier found in its files.
pub struct BindInfo {
    pub relation: String, pub lo: usize, pub vus: usize, pub model: String,
    pub x_sha: String, pub w_sha: String, pub y_sha: String, pub spec_matches: bool,
    pub pin: Option<&'static instances::Pin>,
}
impl BindInfo {
    /// `None` when the statement binds the circuit's operand columns to a pinned frozen set.
    fn problem(&self, allow_unpinned: bool) -> Option<String> {
        if !self.spec_matches { return Some("bind: bind.txt is not the unit circuit's operand columns".into()); }
        if allow_unpinned { return None; }
        let Some(p) = self.pin else {
            return Some(format!("instances: {} [{}, {}) is not a pinned frozen set", self.relation, self.lo, self.lo + self.vus));
        };
        if p.model != self.model { return Some(format!("instances: {} is pinned for {}, the circuits are {}", p.relation, p.model, self.model)); }
        for (what, got, want) in [("x.bin", &self.x_sha, p.x), ("w.bin", &self.w_sha, p.w), ("public words", &self.y_sha, p.y)] {
            if got != want { return Some(format!("instances: {what} sha256 {got} is not the pinned {want}")); }
        }
        None
    }
    fn pinned(&self) -> bool { self.pin.is_some() && self.problem(false).is_none() }
    fn json(&self) -> String {
        format!("{{\"relation\": \"{}\", \"lo\": {}, \"vus\": {}, \"model\": \"{}\", \"x_sha256\": \"{}\", \"w_sha256\": \"{}\", \"y_sha256\": \"{}\", \"bind_spec_matches_circuit\": {}, \"pinned\": {}, \"pin_source\": {}}}",
            self.relation, self.lo, self.vus, self.model, self.x_sha, self.w_sha, self.y_sha, self.spec_matches, self.pinned(),
            self.pin.map(|p| format!("\"{}\"", p.source)).unwrap_or("null".into()))
    }
}

/// `bind.txt` (in `d`, else the export root), `instances.txt`, `x.bin`, `w.bin` (in `d`): the binding and what it names.
fn read_bind(d: &Path, root: &Path, uc: &Circuit, steps: usize, vus: usize, public: &[u32]) -> Result<Option<(verify::Bind, BindInfo)>, String> {
    let bp = if d.join("bind.txt").is_file() { d.join("bind.txt") } else { root.join("bind.txt") };
    if !bp.is_file() { return Ok(None); }
    let spec = std::fs::read_to_string(&bp).map_err(|e| format!("bind.txt: {e}"))?;
    let (mut cols, mut terms) = (Vec::new(), Vec::new());
    for line in spec.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() { continue; }
        if t[0] != "bind" || t.len() < 3 { return Err(format!("bind.txt: unknown line {line:?}")); }
        cols.push(uc.col_index(t[1]).ok_or_else(|| format!("bind.txt: the unit circuit has no column {}", t[1]))?);
        let mut ts = Vec::new();
        for tok in &t[2..] {
            let f: Vec<&str> = tok.split(':').collect();
            let arr = match f.first() { Some(&"x") => 0, Some(&"w") => 1, _ => return Err(format!("bind.txt: bad term {tok}")) };
            let num = |i: usize| -> Result<u64, String> { f.get(i).and_then(|s| s.parse().ok()).ok_or_else(|| format!("bind.txt: bad term {tok}")) };
            if f.len() != 4 || num(1)? == 0 { return Err(format!("bind.txt: bad term {tok}")); }
            ts.push((arr, num(1)? as usize, num(2)? as usize, (num(3)? % field::P as u64) as u32));
        }
        terms.push(ts);
    }
    let mut seen = cols.clone();
    seen.sort_unstable();
    seen.dedup();
    if seen.len() != cols.len() || cols.is_empty() { return Err("bind.txt: the bound columns must be distinct and non-empty".into()); }
    let inst = std::fs::read_to_string(d.join("instances.txt")).map_err(|e| format!("instances.txt: {e}"))?;
    let mut kv: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
    for line in inst.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if let Some((k, v)) = t.split_first() { kv.insert(k, v.to_vec()); }
    }
    let get = |k: &str, i: usize| -> Result<&str, String> { kv.get(k).and_then(|v| v.get(i).copied()).ok_or_else(|| format!("instances.txt: {k}")) };
    let parse = |s: &str| -> Result<usize, String> { s.parse().map_err(|_| format!("instances.txt: {s:?}")) };
    let (relation, lo, n_inst) = (get("relation", 0)?.to_string(), parse(get("lo", 0)?)?, parse(get("vus", 0)?)?);
    if n_inst != vus { return Err(format!("instances.txt: {n_inst} VUs, the statement has {vus}")); }
    let width = |s: &str| -> Result<usize, String> { match s { "u8" => Ok(1), "u16" => Ok(2), "u32" => Ok(4), _ => Err(format!("instances.txt: word {s}")) } };
    let (xw, words) = (width(get("x", 0)?)?, parse(get("x", 1)?)?);
    if get("w", 0)? != get("x", 0)? || parse(get("w", 1)?)? != words { return Err("instances.txt: x and w differ in shape".into()); }
    let yw = width(get("y", 0)?)?;
    let mut arrs: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    let mut dig = vec![sha256::Sha256::digest(spec.as_bytes()), sha256::Sha256::digest(inst.as_bytes())];
    for (a, name) in ["x.bin", "w.bin"].iter().enumerate() {
        let bytes = std::fs::read(d.join(name)).map_err(|e| format!("{name}: {e}"))?;
        if bytes.len() != vus * words * xw { return Err(format!("{name}: {} bytes, instances.txt says {}", bytes.len(), vus * words * xw)); }
        arrs[a] = bytes.chunks_exact(xw).map(|c| c.iter().rev().fold(0u64, |acc, &b| (acc << 8) | b as u64) as u32).collect();
        if arrs[a].iter().any(|&v| v >= field::P) { return Err(format!("{name}: a word is not a canonical BabyBear element")); }
        dig.push(sha256::Sha256::digest(&bytes));
    }
    let mut ybytes = Vec::with_capacity(public.len() * yw);
    for &y in public {
        if yw < 4 && (y as u64) >> (8 * yw) != 0 { return Err("public.bin: a public word exceeds instances.txt's y width".into()); }
        ybytes.extend_from_slice(&y.to_le_bytes()[..yw]);
    }
    for ts in &terms { for &(_, s, o, _) in ts { if s * (steps - 1) + o >= words { return Err("bind.txt: a term reads past the VU's words".into()); } } }
    let info = BindInfo {
        relation: relation.clone(), lo, vus, model: manifest_str(root, "model").unwrap_or_default(),
        x_sha: hex(&dig[2]), w_sha: hex(&dig[3]), y_sha: hex(&sha256::Sha256::digest(&ybytes)),
        spec_matches: expected_bind(uc, root).map(|e| e == spec).unwrap_or(false), pin: instances::find(&relation, lo, vus),
    };
    Ok(Some((verify::Bind { cols, terms, arrs, words, digests: dig }, info)))
}

/// `commitment.txt`: in `d`, else the export root.
fn commitment_file(d: &Path, root: &Path) -> Option<PathBuf> {
    [d.join("commitment.txt"), root.join("commitment.txt")].into_iter().find(|p| p.is_file())
}

/// `commitment.txt` and the tree check of the public words (commitments.rs).
fn read_commitment(d: &Path, root: &Path, public: &[u32]) -> Result<Option<commitments::Checked>, String> {
    let Some(p) = commitment_file(d, root) else { return Ok(None) };
    let spec = commitments::Spec::parse(&std::fs::read_to_string(&p).map_err(|e| format!("commitment.txt: {e}"))?)?;
    commitments::check(spec, public).map(Some)
}

/// The statement of a batch directory: circuits, `vus` (prefix of the public words), steps.  `d` may be a
/// `neg/<id>` subdirectory sharing its parent's manifest and circuits.
fn statement<'a>(unit: &'a Loaded, epi: &'a Loaded, d: &Path, vus: Option<usize>) -> Result<Statement<'a>, String> {
    Ok(statement_bound(unit, epi, d, vus)?.0)
}

fn statement_bound<'a>(unit: &'a Loaded, epi: &'a Loaded, d: &Path, vus: Option<usize>) -> Result<(Statement<'a>, Option<BindInfo>), String> {
    let root = if d.join("manifest.json").is_file() { d.to_path_buf() } else { d.parent().unwrap().parent().unwrap().to_path_buf() };
    let steps = manifest_steps(&root);
    let (n, m, pub_words) = read_rows(&std::fs::read(d.join("public.bin")).map_err(|e| format!("public.bin: {e}"))?)?;
    let b = vus.unwrap_or(n);
    if b > n || b == 0 { return Err(format!("{n} public words, {b} requested")); }
    let uc = &unit.circ;
    let ec = &epi.circ;
    let chain = if manifest_is_v2(&root) {
        let (links, epi_pub, spec_hash) = read_chain(&root.join("chain.txt"), uc, ec)?;
        if m != epi_pub.len() { return Err(format!("public.bin: {} words per VU expected, {m} found", epi_pub.len())); }
        Chain { steps, col_c: vec![], col_y: vec![], epi_y32: vec![], epi_y16: epi_pub[0], public_y16: pub_words[..b * m].to_vec(), links: Some(links), spec_hash: Some(spec_hash), epi_pub }
    } else {
        if m != 1 { return Err("public.bin: one word per VU expected".into()); }
        Chain {
            steps,
            col_c: vec![col(uc, "x.c_lo"), col(uc, "x.c_hi")],
            col_y: vec![col(uc, "y_lo"), col(uc, "y_hi")],
            epi_y32: vec![col(ec, "y32_lo"), col(ec, "y32_hi")],
            epi_y16: col(ec, "y16"),
            public_y16: pub_words[..b].to_vec(),
            links: None,
            spec_hash: None,
            epi_pub: vec![col(ec, "y16")],
        }
    };
    let bound = read_bind(d, &root, uc, steps, b, &chain.public_y16)?;
    let (bind, info) = match bound { Some((bd, info)) => (Some(bd), Some(info)), None => (None, None) };
    let commit = match commitment_file(d, &root) {
        Some(p) => Some(sha256::Sha256::digest(&std::fs::read(&p).map_err(|e| format!("commitment.txt: {e}"))?)),
        None => None,
    };
    Ok((Statement {
        segs: vec![
            Segment { name: "unit".into(), circ: uc, layers: &unit.layers, units: b * steps },
            Segment { name: "epilogue".into(), circ: ec, layers: &epi.layers, units: b },
        ],
        chain,
        bind,
        commit,
    }, info))
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
    let relation = arg(args, "--relation");
    let allow_any = args.iter().any(|a| a == "--allow-any-circuit");
    if relation.is_none() && !allow_any {
        eprintln!("verify: --relation {} (the relation the proof is claimed for), or --allow-any-circuit (the verdict then reports circuit_pinned = false)",
                  pins::relations().join("|"));
        return 2;
    }
    let pin = format!(", \"relation\": {}, \"circuit_pinned\": {}",
                      relation.as_deref().map_or("null".to_string(), |r| format!("\"{}\"", r.replace(['"', '\\'], "'"))), !allow_any);
    let refuse = |e: String| {
        let json = stats_json(&Err(e), &verify::Stats::default(), &pin);
        println!("{json}");
        if let Some(out) = arg(args, "--json") { std::fs::write(out, json + "\n").expect("write json"); }
        1
    };
    // `Some(true)`: claimed for a bound relation R+bound, whose circuit files are R's
    let bound_claim = relation.as_deref().filter(|_| !allow_any).map(|r| r.ends_with(instances::BOUND));
    // checked on the raw files, before any of them is parsed
    let pinned = match (relation.as_deref(), allow_any) {
        (Some(r), false) => {
            let c = pins::Circuits::read(&root, manifest_steps(&root), manifest_is_v2(&root)).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) });
            if let Err(e) = pins::check(&c, r.strip_suffix(instances::BOUND).unwrap_or(r)) { return refuse(e); }
            Some(c)
        }
        _ => None,
    };
    let unit = load_circuit(&root.join("circuit.txt"));
    let epi = load_circuit(&root.join("epilogue.txt"));
    let vus: Option<usize> = arg(args, "--vus").map(|s| s.parse().unwrap());
    let (stmt, info) = statement_bound(&unit, &epi, &dir, vus).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) });
    // statement() finds manifest.json and chain.txt by its own rule: what the transcript absorbs must be the set checked
    if let Some(c) = &pinned {
        let absorbed = pins::Circuits::of(&unit.circ.hash, &epi.circ.hash, stmt.chain.spec_hash.as_ref(), stmt.chain.steps);
        if absorbed != *c {
            return refuse(format!("the statement built from {} is not the circuit set checked in {} (it has steps {}, chain.txt sha256 {}...)",
                                  dir.display(), root.display(), absorbed.steps, absorbed.chain.as_deref().map_or("none", |h| &h[..16])));
        }
    }
    let bytes = std::fs::read(arg(args, "--proof").expect("--proof")).expect("proof file");
    let th = threads(args);
    if args.iter().any(|a| a == "--trace-logup") { verify::TRACE_LOGUP.store(true, std::sync::atomic::Ordering::Relaxed); }
    let problem = match (&info, bound_claim) {
        (None, Some(true)) => Some(format!("statement: not operands-bound (no bind.txt), the proof is claimed for {}", relation.as_deref().unwrap_or(""))),
        (Some(_), Some(false)) => Some(format!("statement: operands-bound (bind.txt); claim it as {}{}", relation.as_deref().unwrap_or(""), instances::BOUND)),
        (Some(bi), Some(true)) => bi.problem(false).or_else(|| (Some(bi.relation.as_str()) != relation.as_deref())
            .then(|| format!("instances: instances.txt names {}, the proof is claimed for {}", bi.relation, relation.as_deref().unwrap_or("")))),
        (None, _) if args.iter().any(|a| a == "--require-bound") => Some("statement: not operands-bound (no bind.txt)".to_string()),
        (None, _) => None,
        (Some(bi), None) => bi.problem(args.iter().any(|a| a == "--allow-unpinned-instances")),
    };
    let committed = match read_commitment(&dir, &root, &stmt.chain.public_y16) {
        Ok(c) => c,
        Err(e) => return refuse(e),
    };
    let commit_claim = relation.as_deref().filter(|_| !allow_any).and_then(commitments::relation_hash);
    let problem = problem.or_else(|| match (&committed, commit_claim) {
        (Some(_), _) if info.is_some() => Some("statement: both operands-bound (bind.txt) and operands-committed (commitment.txt)".into()),
        (None, Some(h)) => Some(format!("statement: not operands-committed (no commitment.txt), the proof is claimed for a {} relation", h.suffix())),
        (Some(c), None) if !allow_any => Some(format!("statement: operands-committed (commitment.txt); claim it as {}", c.spec.relation)),
        (Some(c), Some(_)) => c.problem(false).or_else(|| (Some(c.spec.relation.as_str()) != relation.as_deref())
            .then(|| format!("commitment: commitment.txt names {}, the proof is claimed for {}", c.spec.relation, relation.as_deref().unwrap_or("")))),
        (None, None) if allow_any && args.iter().any(|a| a == "--require-commitment") => Some("statement: not operands-committed (no commitment.txt)".into()),
        (None, None) => None,
        (Some(c), None) => c.problem(args.iter().any(|a| a == "--allow-unpinned-commitment")),
    });
    let (res, st) = match problem {
        Some(e) => (Err(e), verify::Stats { threads: th, ..Default::default() }),
        None => verify::verify(&stmt, &bytes, &LIGERO, th),
    };
    let extra = format!(", \"vus\": {}, \"units\": {}, \"steps\": {}{pin}, \"operands_bound\": {}, \"instances\": {}, \"operands_committed\": {}, \"commitment\": {}",
        stmt.vus(), stmt.segs[0].units, stmt.chain.steps, info.is_some() && res.is_ok(),
        info.as_ref().map(|i| i.json()).unwrap_or("null".into()), committed.is_some() && res.is_ok(),
        committed.as_ref().map(|c| c.json()).unwrap_or("null".into()));
    let json = stats_json(&res, &st, &extra);
    println!("{json}");
    if let Some(out) = arg(args, "--json") { std::fs::write(out, json + "\n").expect("write json"); }
    if res.is_ok() { 0 } else { 1 }
}

/// The circuit set of a statement directory (steps and the three files' SHA-256) and the relation it is pinned for.
fn cmd_circuit_digest(args: &[String]) -> i32 {
    let dir = PathBuf::from(arg(args, "--dir").expect("--dir"));
    match pins::Circuits::read(&dir, manifest_steps(&dir), manifest_is_v2(&dir)) {
        Ok(c) => { println!("{}", c.json()); 0 }
        Err(e) => { eprintln!("{e}"); 2 }
    }
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
    // layout of proof.bin (gpu/README.md "Proof file"): magic, version, digest size, root, message count
    const D: usize = sha512::DIGEST_BYTES;
    let count_at = proof::ROOT_AT + D;
    let n_msgs = u64::from_le_bytes(bytes[count_at..count_at + 8].try_into().unwrap()) as usize;
    let msgs_at = count_at + 8;
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
    let per_col = 4 * rows + D * depth;
    let mut rng = Rng(0x9e3779b97f4a7c15);
    // (group, byte offset, bit) -- flipping a bit of a canonical field element may leave it canonical or not;
    // both are mutations the verifier must reject (non-canonical: at parse; canonical: by the checks)
    let mut muts: Vec<(&str, usize, u8)> = Vec::new();
    let pick = |rng: &mut Rng, total: usize, sample: usize| -> Vec<usize> {
        if sample == 0 || sample >= total { (0..total).collect() } else { (0..sample).map(|_| rng.below(total)).collect() }
    };
    for bit in pick(&mut rng, 8 * D, if sample == 0 { 0 } else { sample.min(32) }) { muts.push(("merkle_root", proof::ROOT_AT + bit / 8, (bit % 8) as u8)); }
    for i in pick(&mut rng, n_msgs, sample) { let bit = rng.below(31); muts.push(("message", msgs_at + 24 * i + 4 * rng.below(6) + bit / 8, (bit % 8) as u8)); }
    for i in pick(&mut rng, k, sample) { let bit = rng.below(31); muts.push(("ligero_w", w_at + 24 * i + 4 * rng.below(6) + bit / 8, (bit % 8) as u8)); }
    for i in pick(&mut rng, nq, sample) { let bit = rng.below(31); muts.push(("ligero_qc", qc_at + 24 * i + 4 * rng.below(6) + bit / 8, (bit % 8) as u8)); }
    for oi in pick(&mut rng, t, sample) {
        let bit = rng.below(31);
        let ri = rng.below(rows);
        muts.push(("column_value", cols_at + oi * per_col + 4 * ri + bit / 8, (bit % 8) as u8));
        let di = rng.below(depth);
        let b = rng.below(8 * D);
        muts.push(("merkle_sibling", cols_at + oi * per_col + 4 * rows + D * di + b / 8, (b % 8) as u8));
    }
    let n_stmt = if stmt.bind.is_some() { 3 } else { 2 };
    println!("mutations: {} bit flips + 2 count mutations + {n_stmt} statement mutations = {}", muts.len(), muts.len() + 2 + n_stmt);
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
        m.extend_from_slice(&bytes[..count_at]);
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
        let v = rng.below(s2.chain.public_y16.len());
        s2.chain.public_y16[v] = (s2.chain.public_y16[v] + 1) % field::P;
        let (r, _) = verify::verify(&s2, &bytes, &LIGERO, th);
        if r.is_err() { stmt_ok += 1; } else { accepted_list.push("public_word".into()); }
        *errors.entry(r.err().unwrap_or_default().split(|c: char| c.is_ascii_digit()).next().unwrap_or("").trim().to_string()).or_insert(0) += 1;
        if stmt.vus() > 1 {
            // an operands-bound statement names its VU count: fewer VUs is already not the statement
            let r = match statement(&unit, &epi, &dir, Some(stmt.vus() - 1)) {
                Ok(s3) => verify::verify(&s3, &bytes, &LIGERO, th).0,
                Err(e) => Err(e),
            };
            if r.is_err() { stmt_ok += 1; } else { accepted_list.push("vu_count".into()); }
            *errors.entry(r.err().unwrap_or_default().split(|c: char| c.is_ascii_digit()).next().unwrap_or("").trim().to_string()).or_insert(0) += 1;
        } else { stmt_ok += 1; }
    }
    let mut stmt_total = 2;
    if stmt.bind.is_some() {
        // operands bound: one frozen x word changed (bit 0: the proof's witness no longer matches the statement)
        stmt_total += 1;
        let mut s4 = statement(&unit, &epi, &dir, vus).unwrap();
        let bd = s4.bind.as_mut().unwrap();
        let i = rng.below(bd.arrs[0].len());
        bd.arrs[0][i] ^= 1;
        let (r, _) = verify::verify(&s4, &bytes, &LIGERO, th);
        if r.is_err() { stmt_ok += 1; } else { accepted_list.push("operand_word".into()); }
        *errors.entry(r.err().unwrap_or_default().split(|c: char| c.is_ascii_digit()).next().unwrap_or("").trim().to_string()).or_insert(0) += 1;
    }
    per_group.insert("statement", (stmt_ok, stmt_total));
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
    if args.len() < 2 { eprintln!("usage: verify|negatives|mutate|circuit-digest ..."); std::process::exit(2); }
    let code = match args[1].as_str() {
        "verify" => cmd_verify(&args),
        "negatives" => cmd_negatives(&args),
        "mutate" => cmd_mutate(&args),
        "circuit-digest" => cmd_circuit_digest(&args),
        _ => { eprintln!("unknown command"); 2 }
    };
    std::process::exit(code);
}
