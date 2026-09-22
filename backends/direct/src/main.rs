//! `direct-unit bench --units B --out DIR [--seed S] [--queries Q]`   prove + verify, write result.json
//! `direct-unit crosscheck --in units.json --out hints.json`           check Rust witness gen against WP2 data
//! `direct-unit layout`                                                 print the column/constraint census

use std::fs;
use std::time::Instant;

use p3_matrix::Matrix;
use serde_json::{json, Value};
use verity_direct::air::{self, UnitAir};
use verity_direct::program::UnitIo;
use verity_direct::stark::{self, FriParams, DEFAULT_FRI};

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn peak_rss_bytes() -> Option<u64> {
    let s = fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn cpu_model() -> Option<String> {
    let s = fs::read_to_string("/proc/cpuinfo").ok()?;
    s.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("bench") => bench(&args),
        Some("crosscheck") => crosscheck(&args),
        Some("layout") => {
            let lay = air::layout();
            let air_ = UnitAir { width: lay.width };
            println!(
                "width={} hints={} materialised={} constraints={} log_quotient_degree={}",
                lay.width,
                lay.n_hints,
                lay.n_mat,
                lay.constraint_names.len(),
                stark::log_quotient_degree(&air_)
            );
            let limbs = lay.col_names.iter().filter(|n| n.contains(".limb[")).count();
            let sels = lay.col_names.iter().filter(|n| n.contains(".sel[") && !n.contains(".max.sel[")).count();
            let invs = lay.col_names.iter().filter(|n| n.ends_with(".inv")).count();
            let slack = lay.col_names.iter().filter(|n| n.ends_with(".slack")).count();
            let mats = lay.col_names.iter().filter(|n| *n == "_mat").count();
            println!("range-limb columns={limbs} table one-hot selectors={sels} exclusion inverses={invs} slack={slack} auto-materialised={mats}");
            println!("value columns (words + hints + named materialisations)={}", lay.width - limbs - sels - invs - slack - mats);
        }
        _ => {
            eprintln!("usage: direct-unit bench|crosscheck|layout ...");
            std::process::exit(2);
        }
    }
}

fn bench(args: &[String]) {
    let units: usize = arg(args, "--units").map(|s| s.parse().unwrap()).unwrap_or(1);
    let out = arg(args, "--out").or_else(|| std::env::var("RESEARCH_RUN_DIR").ok()).unwrap_or_else(|| ".".into());
    let seed: u64 = arg(args, "--seed").map(|s| s.parse().unwrap()).unwrap_or(1);
    let mut fri: FriParams = DEFAULT_FRI;
    if let Some(q) = arg(args, "--queries") {
        fri.num_queries = q.parse().unwrap();
    }
    let min_rows = 2;
    fs::create_dir_all(&out).unwrap();

    let t0 = Instant::now();
    let ios = verity_direct::random_units(units, seed);
    let lay = air::layout();
    let air_ = UnitAir { width: lay.width };
    let trace = air::trace(&ios, min_rows);
    let t_witness = t0.elapsed().as_secs_f64();
    let rows = trace.height();
    let width = trace.width();
    let log_n = rows.trailing_zeros() as usize;
    let lqd = stark::log_quotient_degree(&air_);
    assert!(lqd <= fri.log_blowup, "quotient degree 2^{lqd} exceeds blowup 2^{}", fri.log_blowup);

    let t1 = Instant::now();
    let proof = stark::prove_trace(&air_, trace, fri);
    let t_prove = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let bytes = stark::proof_bytes(&proof);
    let t_ser = t2.elapsed().as_secs_f64();
    let t3 = Instant::now();
    let ok = stark::verify_proof(&air_, &proof, log_n, fri);
    let t_verify = t3.elapsed().as_secs_f64();
    assert!(ok.is_ok(), "verification failed: {:?}", ok.err());
    let t_total = t_witness + t_prove + t_ser;
    let threads = rayon::current_num_threads();
    let rounds = log_n + 5; // alpha, zeta, FRI batch beta, log_n fold betas, PoW, query indices (Fiat-Shamir)

    // FRI soundness (proven unique-decoding bound, conservative): per query -log2((1 + rate)/2), plus PoW.
    let rate = 1.0 / (1u64 << fri.log_blowup) as f64;
    let bits_proven = fri.num_queries as f64 * -((1.0 + rate) / 2.0).log2() + fri.proof_of_work_bits as f64;
    let bits_conj = fri.num_queries as f64 * (fri.log_blowup as f64) + fri.proof_of_work_bits as f64;

    let committed_trace = (rows * width) as u64;
    let quotient_cells = (rows * (1 << lqd) * 2) as u64; // quotient chunks over the quadratic extension
    let census_b_per_unit: u64 = 1064 + 306; // census_estimate_b: committed slots + helper columns per unit
    let run_id = std::env::var("RESEARCH_RUN_ID").unwrap_or_else(|_| format!("local-{seed}"));
    let commit = option_env!("VERITY_COMMIT").map(|s| s.to_string());

    let measurements = vec![
        json!({"name": "t.witness", "value": t_witness, "unit": "s"}),
        json!({"name": "t.joint.encoding_commitment+arithmetic", "value": t_prove, "unit": "s"}),
        json!({"name": "t.serialization", "value": t_ser, "unit": "s"}),
        json!({"name": "t.total", "value": t_total, "unit": "s"}),
        json!({"name": "verifier.seconds", "value": t_verify, "unit": "s"}),
        json!({"name": "transcript.bytes", "value": bytes, "unit": "B"}),
        json!({"name": "rounds.protocol", "value": rounds, "unit": "count"}),
        json!({"name": "rounds.sequential_depth", "value": rounds, "unit": "count"}),
        json!({"name": "mem.peak_host_bytes", "value": peak_rss_bytes().unwrap_or(0), "unit": "B"}),
        json!({"name": "census.committed_elements", "value": committed_trace + quotient_cells, "unit": "count"}),
        json!({"name": "census.witness_elements", "value": committed_trace, "unit": "count"}),
        json!({"name": "direct.units", "value": units, "unit": "count"}),
        json!({"name": "direct.rows", "value": rows, "unit": "count"}),
        json!({"name": "direct.width", "value": width, "unit": "count"}),
        json!({"name": "direct.trace_cells", "value": committed_trace, "unit": "count"}),
        json!({"name": "direct.quotient_cells", "value": quotient_cells, "unit": "count"}),
        json!({"name": "direct.census_estimate_b_cells", "value": census_b_per_unit * units as u64, "unit": "count"}),
        json!({"name": "direct.threads", "value": threads, "unit": "count"}),
        json!({"name": "direct.prove_s", "value": t_prove, "unit": "s"}),
        json!({"name": "direct.verify_s", "value": t_verify, "unit": "s"}),
        json!({"name": "direct.fri_bits_proven", "value": bits_proven, "unit": "bits"}),
        json!({"name": "direct.fri_bits_conjectured", "value": bits_conj, "unit": "bits"}),
    ];
    let result = json!({
        "schema": "research/result/v0.1",
        "run_id": run_id,
        "workload_fingerprint": {
            "profile": "first-campaign-target/2026-09-21",
            "domain": "finite",
            "K": 16,
            "B": rows,
            "B_real": units,
            "N_subbatches": 1,
            "proof_class": "NON_ZK_PROOF_DIAGNOSTIC",
            "authentication": "excluded",
            "zk_mode": "none",
            "security": {
                "target": -80,
                "fri": fri,
                "field": "goldilocks",
                "challenge_field": "goldilocks^2",
                "hash": "keccak256",
                "fri_bits_proven_bound": bits_proven,
                "fri_bits_conjectured_bound": bits_conj,
                "note": "proven bound: unique decoding radius, -log2((1+rate)/2) per query + PoW; field terms (DEEP/quotient batching over F_p^2, ~2^-128 * degree) omitted"
            },
            "hardware": {"gpu": null, "cpu": cpu_model().map(|m| json!({"model": m, "count": threads}))},
            "software": {"backend": {"name": "verity-direct (plonky3 uni-stark)", "version": "0.1.0", "commit": commit, "plonky3": "0.4.3-succinct"}, "toolchain": {"rustc": option_env!("VERITY_RUSTC").unwrap_or("unknown")}},
            "preprocessing": "none",
            "retention": "retained",
            "unit": "transition",
            "constraints": "WP2 gadget contracts as direct AIR constraints; range checks as 2-bit limbs (degree-4), fixed tables as one-hot selectors; no lookup argument",
            "note": "Plonky3 FRI/Keccak is the diagnostic PCS, not Candidate B's Ligero-style interactive PCS (PROTOCOL.md); public words are committed trace cells, not bound to any statement (authentication excluded)"
        },
        "validation": {"status": "passed", "checker": "p3_uni_stark::verify", "detail": format!("proof verified; {units} units in {rows} rows"), "evidence": []},
        "measurements": measurements,
        "measurement_files": [],
        "artifacts": []
    });
    let path = format!("{out}/result.json");
    fs::write(&path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!(
        "units={units} rows={rows} width={width} prove={t_prove:.3}s verify={t_verify:.3}s bytes={bytes} witness={t_witness:.3}s threads={threads} peak_rss={:?} -> {path}",
        peak_rss_bytes()
    );
}

fn crosscheck(args: &[String]) {
    let input = arg(args, "--in").expect("--in units.json");
    let out = arg(args, "--out").expect("--out hints.json");
    let text = fs::read_to_string(&input).unwrap();
    let units: Vec<UnitIo> = serde_json::from_str(&text).unwrap();
    let lay = air::layout();
    let mut all = Vec::with_capacity(units.len());
    let mut mismatched_y = 0usize;
    for (i, u) in units.iter().enumerate() {
        let mut ctx = verity_direct::ctx::IntCtx::new();
        ctx.record_hints = true;
        verity_direct::program::unit_program(&mut ctx, u);
        assert!(ctx.violations.is_empty(), "unit {i}: integer violations {:?}", ctx.violations);
        let row: Vec<_> = ctx.cols.iter().map(|&v| air::to_f(v)).collect();
        let failing = air::failing_constraints(&lay, &row);
        assert!(failing.is_empty(), "unit {i}: field constraints fail {:?}", failing);
        let y = air::f_to_u64(row[lay.col("pack.y")]);
        if y != u.y {
            mismatched_y += 1;
        }
        let hints: Vec<Value> = ctx.hint_values.iter().map(|(n, v)| json!([n, v])).collect();
        all.push(Value::Array(hints));
    }
    assert_eq!(mismatched_y, 0, "output words differ from WP2 on {mismatched_y} units");
    fs::write(&out, serde_json::to_string(&Value::Array(all)).unwrap()).unwrap();
    println!("crosscheck: {} units satisfy the AIR with y = WP2's y; hints written to {out}", units.len());
}
