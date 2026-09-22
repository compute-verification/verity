//! Candidate A, wave 1: prover + verifier for the WP2 transition unit, data-parallel over B units.
//!
//!   verity-gkr run --circuit C --witness W --units B [--out result.json] [--run-id ID]
//!   verity-gkr negatives --circuit C --dir DIR          every witness in DIR must be rejected
//!
//! proof_class = ARITHMETIC_DIAGNOSTIC: the input/hint columns are given to the verifier in the clear (no PCS);
//! the GKR over the checker circuit and the logUp fractional GKRs are proved and verified.  authentication = excluded.

mod circuit;
mod field;
mod gkr;
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
}

fn table_queries<'a>(circ: &'a Circuit, name: &str) -> Vec<&'a Query> { circ.queries.iter().filter(|q| q.table == name).collect() }

fn start_transcript(inst: &Instance, circuit_hash: &[u8; 32]) -> (Transcript, Vec<TableChallenges>, Vec<Ext>, Vec<Ext>) {
    let mut tr = Transcript::new(b"verity-gkr/candidate-a/v0");
    tr.absorb(circuit_hash);
    // the witness is public in this diagnostic: bind the challenges to it
    let mut h = Sha256::new();
    for v in &inst.cols { h.update(&v.to_bytes()); }
    tr.absorb(&h.finalize());
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

pub fn prove(inst: &Instance, circuit_hash: &[u8; 32]) -> (Vec<Ext>, Stats) {
    let mut st = Stats::default();
    let mut out: Vec<Ext> = Vec::new();
    let (mut tr, chs, z0, gamma) = start_transcript(inst, circuit_hash);
    let t0 = Instant::now();
    for (ti, t) in inst.circ.tables.iter().enumerate() {
        let qs = table_queries(inst.circ, &t.name);
        let (p, q, n) = logup::leaves(inst.circ, t, &qs, &inst.cols, inst.ncols, inst.units, &chs[ti]);
        let s = logup::prove_table(p, q, n, &mut tr, &mut out);
        st.lookup_dims.push((t.name.clone(), s.n, qs.len()));
        // chain: λ after q_root, then per level k: k rounds + μ + λ
        let d: u64 = 1 + (0..n as u64).map(|k| k + 2).sum::<u64>();
        st.depth_lookup_max = st.depth_lookup_max.max(d);
    }
    st.t_lookup = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    // wires of every unit
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
    for (i, layer) in inst.layers.iter().enumerate() {
        let last = i == nl - 1;
        let input_wires = Strided { data: &wires, width: nw, copies: inst.units };
        let input_cols = Strided { data: &inst.cols, width: inst.ncols, copies: inst.units };
        let inp: &(dyn gkr::LayerInput + Sync) = if last { &input_cols } else { &input_wires };
        let (next, _fin) = gkr::prove_layer(layer, inp, b, &claim, &mut tr, &mut out, last);
        st.depth_circuit += (b + 2 * layer.s_in + if last { 0 } else { 1 }) as u64;
        claim = next;
    }
    st.t_arith = t2.elapsed().as_secs_f64();
    st.proof_elems = out.len();
    st.slots = tr.slots;
    st.committed_elements = inst.units * inst.ncols;
    (out, st)
}

pub fn verify(inst: &Instance, circuit_hash: &[u8; 32], proof: &[Ext]) -> Result<(), String> {
    let (mut tr, chs, z0, gamma) = start_transcript(inst, circuit_hash);
    let mut rd = Reader { msgs: proof, pos: 0 };
    for (ti, t) in inst.circ.tables.iter().enumerate() {
        let qs = table_queries(inst.circ, &t.name);
        let (p, q, n) = logup::leaves(inst.circ, t, &qs, &inst.cols, inst.ncols, inst.units, &chs[ti]);
        logup::verify_table(&p, &q, n, &mut tr, &mut rd, &t.name)?;
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
            if eval_columns(&inst.cols, inst.ncols, inst.units, s_c, &l_star, c_star) != vl { return Err("input layer: left evaluation mismatch".into()); }
            if eval_columns(&inst.cols, inst.ncols, inst.units, s_c, &r_star, c_star) != vr { return Err("input layer: right evaluation mismatch".into()); }
        }
        claim = next;
    }
    if rd.pos != proof.len() { return Err("trailing proof data".into()); }
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
            let inst = Instance { circ: &circ, layers: &ls, cols, units, ncols: m };
            let tp = Instant::now();
            let (proof, st) = prove(&inst, &hash);
            let t_prove = tp.elapsed().as_secs_f64();
            let tv = Instant::now();
            let res = verify(&inst, &hash, &proof);
            let t_verify = tv.elapsed().as_secs_f64();
            let ok = res.is_ok();
            let depth = 1 + st.depth_circuit.max(st.depth_lookup_max);
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16);
            let rss = peak_rss_bytes();
            eprintln!("units={units} verify={:?} prove_s={t_prove:.3} verify_s={t_verify:.3} proof_bytes={} slots={} depth={depth} lookups={:?} rss={:?}", res, proof.len() * 24, st.slots, st.lookup_dims, rss);
            if let Some(out) = arg(&args, "--out") {
                let run_id = arg(&args, "--run-id").or_else(|| std::env::var("RESEARCH_RUN_ID").ok()).unwrap_or_else(|| "local".into());
                let commit = arg(&args, "--commit").unwrap_or_else(|| "unknown".into());
                let cpu = cpu_model().map(|m| format!("{{\"model\": \"{m}\", \"count\": {}}}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0))).unwrap_or("null".into());
                let meas = |name: &str, value: f64, unit: &str| format!("{{\"name\": \"{name}\", \"value\": {value}, \"unit\": \"{unit}\"}}");
                let mut ms = vec![
                    meas("t.total", t_prove, "s"), meas("t.lookup", st.t_lookup, "s"), meas("t.arithmetic", st.t_arith, "s"),
                    meas("t.witness", st.t_witness_wires, "s"),
                    meas("verifier.seconds", t_verify, "s"), meas("transcript.bytes", (proof.len() * 24) as f64, "B"),
                    meas("rounds.protocol", st.slots as f64, "count"), meas("rounds.sequential_depth", depth as f64, "count"),
                    meas("census.committed_elements", st.committed_elements as f64, "count"),
                    meas("prove_seconds", t_prove, "s"), meas("verify_seconds", t_verify, "s"), meas("proof_bytes", (proof.len() * 24) as f64, "B"),
                    meas("threads", threads as f64, "count"), meas("units", units as f64, "count"),
                    meas("columns_per_unit", m as f64, "count"),
                ];
                if let Some(r) = rss { ms.push(meas("mem.peak_host_bytes", r as f64, "B")); ms.push(meas("peak_rss_bytes", r as f64, "B")); }
                let lk: Vec<String> = st.lookup_dims.iter().map(|(n, d, q)| format!("\"{n}\": {{\"hypercube_dim\": {d}, \"queries_per_unit\": {q}}}")).collect();
                let json = format!(
                    "{{\n\"schema\": \"research/result/v0.1\",\n\"run_id\": \"{run_id}\",\n\"workload_fingerprint\": {{\n  \"profile\": \"first-campaign-target/2026-09-21\", \"domain\": \"finite\", \"K\": 16, \"B\": {units}, \"N_subbatches\": 1,\n  \"unit\": \"transition\",\n  \"proof_class\": \"ARITHMETIC_DIAGNOSTIC\", \"authentication\": \"excluded\", \"zk_mode\": \"none\",\n  \"security\": {{\"target\": -128, \"trace_field\": \"goldilocks\", \"challenge_field\": \"goldilocks^3 (X^3-X-1)\", \"fiat_shamir\": \"sha256\", \"commitment\": \"none: columns in the clear\", \"note\": \"soundness of the interactive part only; no PCS term; see report\"}},\n  \"hardware\": {{\"gpu\": null, \"cpu\": {cpu}}},\n  \"software\": {{\"backend\": {{\"name\": \"verity-gkr\", \"version\": \"0.1.0\", \"commit\": \"{commit}\"}}, \"toolchain\": {{\"rustc\": \"{}\"}}}},\n  \"preprocessing\": \"circuit export from the WP2 checker (verity_numerical.gkr_export) and witness generation by WP2 gen_step in Python; not timed here\",\n  \"retention\": \"retained\",\n  \"threads\": {threads},\n  \"lookups\": {{{}}}\n}},\n\"validation\": {{\"status\": \"{}\", \"checker\": \"verity-gkr verify\", \"detail\": \"{}\", \"evidence\": []}},\n\"measurements\": [\n{}\n]\n}}\n",
                    option_env!("RUSTC_VERSION").unwrap_or("stable"), lk.join(", "), if ok { "passed" } else { "failed" },
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
                let (proof, _) = prove(&inst, &hash);
                match verify(&inst, &hash, &proof) {
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
            let (proof, _) = prove(&inst, &hash);
            verify(&inst, &hash, &proof).unwrap();
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
            let (proof, _) = prove(&inst, &hash);
            assert!(verify(&inst, &hash, &proof).is_err(), "{} accepted", p.display());
            n += 1;
        }
        assert!(n >= 50);
    }

    #[test]
    fn tampered_proof_is_rejected() {
        let (circ, ls, cols, units, m, hash) = load(&fx("circuit.txt"), &fx("honest8.bin"), Some(2));
        let inst = Instance { circ: &circ, layers: &ls, cols, units, ncols: m };
        let (mut proof, _) = prove(&inst, &hash);
        let k = proof.len() / 2;
        proof[k] = proof[k] + Ext::ONE;
        assert!(verify(&inst, &hash, &proof).is_err());
    }
}
