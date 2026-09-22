//! Candidate A, wave 1: prover + verifier for the WP2 transition unit, data-parallel over B units.
//!
//!   verity-gkr run --circuit C --witness W --units B [--out result.json] [--run-id ID]
//!   verity-gkr negatives --circuit C --dir DIR          every witness in DIR must be rejected
//!
//! Default mode (PCS): the input/hint columns and the lookup multiplicities are committed with the hash-based
//! Ligero-style commitment of `ligero.rs` (no masking) → NON_ZK_PROOF_DIAGNOSTIC.  `--clear`: the columns are given
//! to the verifier in the clear and only the GKR + logUp part is proved → ARITHMETIC_DIAGNOSTIC.
//! authentication = excluded in both.

mod circuit;
mod field;
mod gkr;
mod ligero;
mod logup;
mod transcript;

use circuit::{layers, read_witness, Circuit, Layer, Query};
use field::{eq_table, Ext, Fp};
use gkr::{eval_columns, LayerClaim, Reader, Strided};
use logup::TableChallenges;
use std::time::Instant;
use transcript::{Sha256, Transcript};

pub struct Instance<'a> { pub circ: &'a Circuit, pub layers: &'a [Layer], pub cols: Vec<Fp>, pub units: usize, pub ncols: usize }

fn log2ceil(n: usize) -> usize { let mut s = 0; while (1usize << s) < n { s += 1; } s }

#[derive(Default, Debug)]
pub struct Stats {
    pub t_lookup: f64, pub t_arith: f64, pub t_witness_wires: f64, pub proof_elems: usize, pub slots: u64,
    pub depth_circuit: u64, pub depth_lookup_max: u64, pub lookup_dims: Vec<(String, usize, usize)>, pub committed_elements: usize,
    pub t_commit: f64, pub t_open: f64, pub ligero_rows: usize,
}

fn table_queries<'a>(circ: &'a Circuit, name: &str) -> Vec<&'a Query> { circ.queries.iter().filter(|q| q.table == name).collect() }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode { Clear, Pcs }

pub struct Proof { pub root: [u8; 32], pub msgs: Vec<Ext>, pub bytes: Vec<u8> }
impl Proof { pub fn size(&self) -> usize { (if self.root != [0u8; 32] { 32 } else { 0 }) + self.msgs.len() * 24 + self.bytes.len() } }

pub const LIGERO: ligero::Params = ligero::Params { k: 4096, n: 16384, t: 192 };

fn start_transcript(inst: &Instance, circuit_hash: &[u8; 32], mode: Mode, root: &[u8; 32]) -> (Transcript, Vec<TableChallenges>, Vec<Ext>, Vec<Ext>) {
    let mut tr = Transcript::new(b"verity-gkr/candidate-a/v0");
    tr.absorb(circuit_hash);
    match mode {
        Mode::Clear => {
            // the witness is public in this diagnostic: bind the challenges to it
            let mut h = Sha256::new();
            for v in &inst.cols { h.update(&v.to_bytes()); }
            tr.absorb(&h.finalize());
        }
        Mode::Pcs => tr.absorb(root),
    }
    tr.absorb(&(inst.units as u64).to_le_bytes());
    let mut chs = Vec::new();
    for t in &inst.circ.tables {
        let z = tr.challenge();
        let beta = if t.cols > 1 { tr.challenge() } else { Ext::ONE };
        chs.push(TableChallenges { z, beta });
    }
    let s_a = inst.layers[0].s_out;
    let b = log2ceil(inst.units);
    let z0 = tr.challenges(s_a);
    let gamma = tr.challenges(b);
    (tr, chs, z0, gamma)
}

/// Layout of the committed vector X: W row-major (units × ncols), then every table's multiplicities.
fn x_layout(inst: &Instance) -> (Vec<usize>, usize) {
    let mut offs = Vec::new();
    let mut l = inst.units * inst.ncols;
    for t in &inst.circ.tables { offs.push(l); l += t.rows.len(); }
    (offs, l)
}

struct Functional { a: Vec<Ext>, b: Ext }

fn functional_input(inst: &Instance, l: usize, x_pt: &[Ext], c_pt: &[Ext], value: Ext) -> Functional {
    let ec = eq_table(c_pt);
    let ex = eq_table(x_pt);
    let mut a = vec![Ext::ZERO; l];
    for u in 0..inst.units { for i in 0..inst.ncols { a[u * inst.ncols + i] = ec[u] * ex[i]; } }
    Functional { a, b: value }
}

/// p̃(ρ) + λ q̃(ρ) as a functional on X plus a public constant (table rows, padding, linear-form constants).
fn functional_lookup(inst: &Instance, l: usize, off: usize, table: &circuit::Table, qs: &[&Query], ch: &TableChallenges, n: usize, c: &logup::LeafClaim) -> Functional {
    let eqt = eq_table(&c.rho);
    let q_t = qs.len();
    let n_wit = inst.units * q_t;
    let betas: Vec<Ext> = { let mut v = vec![Ext::ONE]; for k in 1..table.cols { v.push(v[k - 1] * ch.beta); } v };
    let mut a = vec![Ext::ZERO; l];
    let mut konst = Ext::ZERO;
    for u in 0..inst.units {
        for (j, qu) in qs.iter().enumerate() {
            let e = eqt[u * q_t + j];
            let le = c.lambda * e;
            konst += e; // p = 1
            let mut cst = ch.z;
            for (k, lin) in qu.cols.iter().enumerate() {
                cst -= betas[k].scale(lin.konst);
                let lb = le * betas[k];
                for t in &lin.terms { a[u * inst.ncols + t.idx as usize] -= lb.scale(t.coef); }
            }
            konst += le * cst;
        }
    }
    for (v, row) in table.rows.iter().enumerate() {
        let e = eqt[n_wit + v];
        a[off + v] -= e; // p = -m_v
        let mut comb = Ext::ZERO;
        for (k, x) in row.iter().enumerate() { comb += betas[k].scale(*x); }
        konst += c.lambda * e * (ch.z - comb);
    }
    for idx in n_wit + table.rows.len()..(1usize << n) { konst += c.lambda * eqt[idx]; } // padding (0, 1)
    Functional { a, b: c.value - konst }
}

fn batch(fs: &[Functional], l: usize, tr: &mut Transcript) -> Functional {
    let mut a = vec![Ext::ZERO; l];
    let mut b = Ext::ZERO;
    for f in fs {
        let rho = tr.challenge();
        for (x, y) in a.iter_mut().zip(&f.a) { if !y.is_zero() { *x += rho * *y; } }
        b += rho * f.b;
    }
    Functional { a, b }
}

pub fn prove(inst: &Instance, circuit_hash: &[u8; 32], mode: Mode) -> (Proof, Stats) {
    let mut st = Stats::default();
    let mut out: Vec<Ext> = Vec::new();
    let mut bytes: Vec<u8> = Vec::new();
    let (offs, l) = x_layout(inst);
    // multiplicities are challenge-independent: count them first so X can be committed before any challenge
    let mut x: Vec<Fp> = inst.cols.clone();
    let mut mults: Vec<Vec<u64>> = Vec::new();
    for t in &inst.circ.tables {
        let qs = table_queries(inst.circ, &t.name);
        let (_, _, _, m) = logup::leaves(inst.circ, t, &qs, &inst.cols, inst.ncols, inst.units, &TableChallenges { z: Ext::ZERO, beta: Ext::ONE });
        x.extend(m.iter().map(|v| Fp::new(*v)));
        mults.push(m);
    }
    debug_assert_eq!(x.len(), l);
    let tc = Instant::now();
    let committed = if mode == Mode::Pcs { Some(ligero::commit(&x, &LIGERO)) } else { None };
    let root = committed.as_ref().map(|c| c.root()).unwrap_or([0u8; 32]);
    st.t_commit = tc.elapsed().as_secs_f64();
    let (mut tr, chs, z0, gamma) = start_transcript(inst, circuit_hash, mode, &root);
    let t0 = Instant::now();
    let mut lookup_ns = Vec::new();
    for (ti, t) in inst.circ.tables.iter().enumerate() {
        let qs = table_queries(inst.circ, &t.name);
        let (p, q, n, _) = logup::leaves(inst.circ, t, &qs, &inst.cols, inst.ncols, inst.units, &chs[ti]);
        let s = logup::prove_table(p, q, n, &mut tr, &mut out);
        st.lookup_dims.push((t.name.clone(), s.n, qs.len()));
        lookup_ns.push(n);
        let d: u64 = 1 + (0..n as u64).map(|k| k + 2).sum::<u64>();
        st.depth_lookup_max = st.depth_lookup_max.max(d);
    }
    st.t_lookup = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let nw = inst.circ.nwires();
    let mut wires = vec![Fp::ZERO; inst.units * nw];
    let mut tmp = Vec::with_capacity(nw);
    for u in 0..inst.units {
        inst.circ.eval_wires(&inst.cols[u * inst.ncols..(u + 1) * inst.ncols], &mut tmp);
        wires[u * nw..(u + 1) * nw].copy_from_slice(&tmp);
    }
    st.t_witness_wires = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let b = log2ceil(inst.units);
    let mut claim = LayerClaim { w: eq_table(&z0), copy_point: gamma, value: Ext::ZERO };
    let nl = inst.layers.len();
    let depth_of: Vec<u32> = inst.circ.wires.iter().map(|w| match w { circuit::Wire::Input(_) => 0, circuit::Wire::Prod { depth, .. } => *depth }).collect();
    let mut final_claims = None;
    for (i, layer) in inst.layers.iter().enumerate() {
        let last = i == nl - 1;
        let k_in = inst.circ.max_depth - i as u32;
        let masked: Vec<Fp> = if last { vec![] } else {
            let mut m = wires.clone();
            for (j, v) in m.iter_mut().enumerate() { if depth_of[j % nw] > k_in { *v = Fp::ZERO; } }
            m
        };
        let input_wires = Strided { data: &masked, width: nw, copies: inst.units };
        let input_cols = Strided { data: &inst.cols, width: inst.ncols, copies: inst.units };
        let inp: &(dyn gkr::LayerInput + Sync) = if last { &input_cols } else { &input_wires };
        let (next, fin) = gkr::prove_layer(layer, inp, b, &claim, &mut tr, &mut out, last);
        st.depth_circuit += (b + 2 * layer.s_in + if last { 0 } else { 1 }) as u64;
        if let Some(f) = fin { final_claims = Some((f, next.copy_point.clone())); }
        claim = next;
    }
    st.t_arith = t2.elapsed().as_secs_f64();
    if let Some(cm) = &committed {
        let t3 = Instant::now();
        // the verifier's residual claims, recomputed by the prover from the transcript it produced
        let ((l_star, vl, r_star, vr), c_star) = final_claims.unwrap();
        let mut fs = vec![functional_input(inst, l, &l_star, &c_star, vl), functional_input(inst, l, &r_star, &c_star, vr)];
        // lookup claims: replay the verifier's transcript view of each table (the claims are determined by the proof)
        let mut tr2 = start_transcript(inst, circuit_hash, mode, &root).0;
        let mut rd = Reader { msgs: &out, pos: 0 };
        for (ti, t) in inst.circ.tables.iter().enumerate() {
            let qs = table_queries(inst.circ, &t.name);
            let lc = logup::verify_table(lookup_ns[ti], &mut tr2, &mut rd, &t.name).expect("prover replays its own lookup transcript");
            fs.push(functional_lookup(inst, l, offs[ti], t, &qs, &chs[ti], lookup_ns[ti], &lc));
        }
        let f = batch(&fs, l, &mut tr);
        ligero::prove_open(cm, &f.a, &LIGERO, &mut tr, &mut out, &mut bytes);
        st.t_open = t3.elapsed().as_secs_f64();
        st.ligero_rows = cm.rows;
    }
    st.proof_elems = out.len();
    st.slots = tr.slots;
    st.committed_elements = l;
    (Proof { root, msgs: out, bytes }, st)
}

/// Verifier.  In `Mode::Clear` it reads `inst.cols`; in `Mode::Pcs` it never touches them (they may be empty).
pub fn verify(inst: &Instance, circuit_hash: &[u8; 32], proof: &Proof, mode: Mode) -> Result<(), String> {
    let (mut tr, chs, z0, gamma) = start_transcript(inst, circuit_hash, mode, &proof.root);
    let mut rd = Reader { msgs: &proof.msgs, pos: 0 };
    let (offs, l) = x_layout(inst);
    let mut fs: Vec<Functional> = Vec::new();
    let mut lookup_claims = Vec::new();
    for (ti, t) in inst.circ.tables.iter().enumerate() {
        let qs = table_queries(inst.circ, &t.name);
        let n = logup::log2ceil_pub(inst.units * qs.len() + t.rows.len());
        let lc = logup::verify_table(n, &mut tr, &mut rd, &t.name)?;
        match mode {
            Mode::Clear => {
                let (p, q, n2, _) = logup::leaves(inst.circ, t, &qs, &inst.cols, inst.ncols, inst.units, &chs[ti]);
                debug_assert_eq!(n, n2);
                logup::check_leaves(&p, &q, &lc, &t.name)?;
            }
            Mode::Pcs => lookup_claims.push((ti, n, lc)),
        }
    }
    let b = log2ceil(inst.units);
    let mut claim = LayerClaim { w: eq_table(&z0), copy_point: gamma, value: Ext::ZERO };
    let nl = inst.layers.len();
    let s_c = inst.layers[nl - 1].s_in;
    for (i, layer) in inst.layers.iter().enumerate() {
        let last = i == nl - 1;
        let (next, fin) = gkr::verify_layer(layer, b, &claim, &mut tr, &mut rd, last)?;
        if let Some((l_star, vl, r_star, vr)) = fin {
            let c_star = &next.copy_point;
            match mode {
                Mode::Clear => {
                    if eval_columns(&inst.cols, inst.ncols, inst.units, s_c, &l_star, c_star) != vl { return Err("input layer: left evaluation mismatch".into()); }
                    if eval_columns(&inst.cols, inst.ncols, inst.units, s_c, &r_star, c_star) != vr { return Err("input layer: right evaluation mismatch".into()); }
                }
                Mode::Pcs => {
                    fs.push(functional_input(inst, l, &l_star, c_star, vl));
                    fs.push(functional_input(inst, l, &r_star, c_star, vr));
                }
            }
        }
        claim = next;
    }
    if mode == Mode::Pcs {
        for (ti, n, lc) in &lookup_claims {
            let t = &inst.circ.tables[*ti];
            let qs = table_queries(inst.circ, &t.name);
            fs.push(functional_lookup(inst, l, offs[*ti], t, &qs, &chs[*ti], *n, lc));
        }
        let f = batch(&fs, l, &mut tr);
        let rows = l.div_ceil(LIGERO.k);
        ligero::verify_open(&proof.root, rows, &f.a, f.b, &LIGERO, &mut tr, &mut rd, &proof.bytes)?;
    }
    if rd.pos != proof.msgs.len() { return Err("trailing proof data".into()); }
    Ok(())
}

fn load(circuit_path: &str, witness_path: &str, units: Option<usize>) -> (Circuit, Vec<Layer>, Vec<Fp>, usize, usize, [u8; 32]) {
    let text = std::fs::read_to_string(circuit_path).expect("circuit");
    let hash = Sha256::digest(text.as_bytes());
    let circ = Circuit::parse(&text);
    let ls = layers(&circ);
    let bytes = std::fs::read(witness_path).expect("witness");
    let (n, m, mut cols) = read_witness(&bytes);
    assert_eq!(m, circ.ncols);
    let units = units.unwrap_or(n);
    if units > n {
        // tile the available units to the requested batch (timing runs only; each unit is still a valid unit)
        let base = cols.clone();
        while cols.len() < units * m { let take = (units * m - cols.len()).min(base.len()); cols.extend_from_slice(&base[..take]); }
    } else { cols.truncate(units * m); }
    (circ, ls, cols, units, m, hash)
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

fn arg(args: &[String], key: &str) -> Option<String> { args.iter().position(|a| a == key).map(|i| args[i + 1].clone()) }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("usage: run|negatives ..."); std::process::exit(2); }
    match args[1].as_str() {
        "run" => {
            let units: Option<usize> = arg(&args, "--units").map(|s| s.parse().unwrap());
            let (circ, ls, cols, units, m, hash) = load(&arg(&args, "--circuit").unwrap(), &arg(&args, "--witness").unwrap(), units);
            let mode = if args.iter().any(|a| a == "--clear") { Mode::Clear } else { Mode::Pcs };
            let inst = Instance { circ: &circ, layers: &ls, cols, units, ncols: m };
            let tp = Instant::now();
            let (proof, st) = prove(&inst, &hash, mode);
            let t_prove = tp.elapsed().as_secs_f64();
            let tv = Instant::now();
            let vinst = Instance { circ: &circ, layers: &ls, cols: if mode == Mode::Pcs { vec![] } else { inst.cols.clone() }, units, ncols: m };
            let res = verify(&vinst, &hash, &proof, mode);
            let t_verify = tv.elapsed().as_secs_f64();
            let ok = res.is_ok();
            let depth = 1 + st.depth_circuit.max(st.depth_lookup_max);
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16);
            let rss = peak_rss_bytes();
            eprintln!("mode={mode:?} units={units} verify={:?} prove_s={t_prove:.3} (commit {:.3} lookup {:.3} arith {:.3} open {:.3}) verify_s={t_verify:.3} proof_bytes={} slots={} depth={depth} rows={} lookups={:?} rss={:?}", res, st.t_commit, st.t_lookup, st.t_arith, st.t_open, proof.size(), st.slots, st.ligero_rows, st.lookup_dims, rss);
            if let Some(out) = arg(&args, "--out") {
                let run_id = arg(&args, "--run-id").or_else(|| std::env::var("RESEARCH_RUN_ID").ok()).unwrap_or_else(|| "local".into());
                let commit = arg(&args, "--commit").unwrap_or_else(|| "unknown".into());
                let cpu = cpu_model().map(|m| format!("{{\"model\": \"{m}\", \"count\": {}}}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0))).unwrap_or("null".into());
                let meas = |name: &str, value: f64, unit: &str| format!("{{\"name\": \"{name}\", \"value\": {value}, \"unit\": \"{unit}\"}}");
                let mut ms = vec![
                    meas("t.total", t_prove, "s"), meas("t.lookup", st.t_lookup, "s"), meas("t.arithmetic", st.t_arith, "s"),
                    meas("t.witness", st.t_witness_wires, "s"), meas("t.encoding_commitment", st.t_commit + st.t_open, "s"),
                    meas("verifier.seconds", t_verify, "s"), meas("transcript.bytes", proof.size() as f64, "B"),
                    meas("rounds.protocol", st.slots as f64, "count"), meas("rounds.sequential_depth", depth as f64, "count"),
                    meas("census.committed_elements", st.committed_elements as f64, "count"),
                    meas("prove_seconds", t_prove, "s"), meas("verify_seconds", t_verify, "s"), meas("proof_bytes", proof.size() as f64, "B"), meas("ligero_rows", st.ligero_rows as f64, "count"),
                    meas("threads", threads as f64, "count"), meas("units", units as f64, "count"),
                    meas("columns_per_unit", m as f64, "count"),
                ];
                if let Some(r) = rss { ms.push(meas("mem.peak_host_bytes", r as f64, "B")); ms.push(meas("peak_rss_bytes", r as f64, "B")); }
                let lk: Vec<String> = st.lookup_dims.iter().map(|(n, d, q)| format!("\"{n}\": {{\"hypercube_dim\": {d}, \"queries_per_unit\": {q}}}")).collect();
                let json = format!(
                    "{{\n\"schema\": \"research/result/v0.1\",\n\"run_id\": \"{run_id}\",\n\"workload_fingerprint\": {{\n  \"profile\": \"first-campaign-target/2026-09-21\", \"domain\": \"finite\", \"K\": 16, \"B\": {units}, \"N_subbatches\": 1,\n  \"unit\": \"transition\",\n  \"proof_class\": \"{}\", \"authentication\": \"excluded\", \"zk_mode\": \"none\",\n  \"security\": {{\"target\": -128, \"trace_field\": \"goldilocks\", \"challenge_field\": \"goldilocks^3 (X^3-X-1)\", \"fiat_shamir\": \"sha256\", \"commitment\": \"{}\", \"note\": \"see report for the error terms this configuration achieves\"}},\n  \"hardware\": {{\"gpu\": null, \"cpu\": {cpu}}},\n  \"software\": {{\"backend\": {{\"name\": \"verity-gkr\", \"version\": \"0.1.0\", \"commit\": \"{commit}\"}}, \"toolchain\": {{\"rustc\": \"{}\"}}}},\n  \"preprocessing\": \"circuit export from the WP2 checker (verity_numerical.gkr_export) and witness generation by WP2 gen_step in Python; not timed here\",\n  \"retention\": \"retained\",\n  \"threads\": {threads},\n  \"lookups\": {{{}}}\n}},\n\"validation\": {{\"status\": \"{}\", \"checker\": \"verity-gkr verify\", \"detail\": \"{}\", \"evidence\": []}},\n\"measurements\": [\n{}\n]\n}}\n",
                    if mode == Mode::Pcs { "NON_ZK_PROOF_DIAGNOSTIC" } else { "ARITHMETIC_DIAGNOSTIC" },
                    if mode == Mode::Pcs { format!("ligero-style RS(k={},n={}) over Goldilocks, SHA-256 Merkle columns, t={} opened, no masking", LIGERO.k, LIGERO.n, LIGERO.t) } else { "none: columns in the clear".to_string() },
                    std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".into()), lk.join(", "), if ok { "passed" } else { "failed" },
                    match &res { Ok(()) => "verifier accepted the batch".to_string(), Err(e) => e.replace('"', "'") }, ms.join(",\n"));
                std::fs::write(&out, json).expect("write result");
            }
            if !ok { std::process::exit(1); }
        }
        "negatives" => {
            let cpath = arg(&args, "--circuit").unwrap();
            let dir = arg(&args, "--dir").unwrap();
            let mut n_ok = 0;
            let mut n_bad = 0;
            let mut entries: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().map(|e| e == "bin").unwrap_or(false)).collect();
            entries.sort();
            for p in entries {
                let (circ, ls, cols, units, m, hash) = load(&cpath, p.to_str().unwrap(), None);
                let inst = Instance { circ: &circ, layers: &ls, cols, units, ncols: m };
                let mode = if args.iter().any(|a| a == "--clear") { Mode::Clear } else { Mode::Pcs };
                let (proof, _) = prove(&inst, &hash, mode);
                match verify(&inst, &hash, &proof, mode) {
                    Ok(()) => { n_bad += 1; println!("ACCEPTED (BUG) {}", p.display()); }
                    Err(e) => { n_ok += 1; println!("rejected {} : {e}", p.file_name().unwrap().to_string_lossy()); }
                }
            }
            println!("negatives rejected: {n_ok}, wrongly accepted: {n_bad}");
            if n_bad > 0 { std::process::exit(1); }
        }
        _ => { eprintln!("unknown command"); std::process::exit(2); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fx(p: &str) -> String { format!("{}/fixtures/{p}", env!("CARGO_MANIFEST_DIR")) }

    #[test]
    fn honest_units_accepted_at_several_batch_sizes() {
        for units in [1usize, 2, 8] {
            let (circ, ls, cols, units, m, hash) = load(&fx("circuit.txt"), &fx("honest8.bin"), Some(units));
            let inst = Instance { circ: &circ, layers: &ls, cols, units, ncols: m };
            for mode in [Mode::Clear, Mode::Pcs] {
                let (proof, _) = prove(&inst, &hash, mode);
                let vinst = Instance { circ: &circ, layers: &ls, cols: if mode == Mode::Pcs { vec![] } else { inst.cols.clone() }, units, ncols: m };
                verify(&vinst, &hash, &proof, mode).unwrap();
            }
        }
    }

    #[test]
    fn every_exported_negative_is_rejected() {
        let dir = fx("negatives");
        let mut n = 0;
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().map(|e| e != "bin").unwrap_or(true) { continue; }
            let (circ, ls, cols, units, m, hash) = load(&fx("circuit.txt"), p.to_str().unwrap(), None);
            let inst = Instance { circ: &circ, layers: &ls, cols, units, ncols: m };
            for mode in [Mode::Clear, Mode::Pcs] {
                let (proof, _) = prove(&inst, &hash, mode);
                let vinst = Instance { circ: &circ, layers: &ls, cols: if mode == Mode::Pcs { vec![] } else { inst.cols.clone() }, units, ncols: m };
                assert!(verify(&vinst, &hash, &proof, mode).is_err(), "{} accepted in {mode:?}", p.display());
            }
            n += 1;
        }
        assert!(n >= 50);
    }

    #[test]
    fn tampered_proof_is_rejected() {
        let (circ, ls, cols, units, m, hash) = load(&fx("circuit.txt"), &fx("honest8.bin"), Some(2));
        let inst = Instance { circ: &circ, layers: &ls, cols, units, ncols: m };
        for mode in [Mode::Clear, Mode::Pcs] {
            let (mut proof, _) = prove(&inst, &hash, mode);
            let k = proof.msgs.len() / 2;
            proof.msgs[k] = proof.msgs[k] + Ext::ONE;
            assert!(verify(&inst, &hash, &proof, mode).is_err());
            let (mut proof, _) = prove(&inst, &hash, mode);
            if mode == Mode::Pcs { proof.bytes[5] ^= 1; assert!(verify(&inst, &hash, &proof, mode).is_err()); }
        }
    }
}
