//! `direct-unit bench --units B --out DIR [--seed S] [--queries Q]`   prove + verify, write result.json
//! `direct-unit crosscheck --in units.json --out hints.json`           check Rust witness gen against WP2 data
//! `direct-unit layout`                                                 print the column/constraint census
//!
//! The K=1536 VU chain (`ChainAir`, Goldilocks or BabyBear):
//! `direct-unit vu-bench --root fixtures/bench-instances/v1 --field goldilocks|babybear [--vus N] [--queries Q] --out DIR`
//! `direct-unit vu-neg   --root fixtures/bench-instances/v1 --field goldilocks|babybear [--out DIR]`
//! `direct-unit chain-layout --field goldilocks|babybear`

use std::fs;
use std::path::Path;
use std::time::Instant;

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_uni_stark::Val;
use serde_json::{json, Value};
use verity_direct::air::{self, ChainAir, ChainIdx, Layout, UnitAir, VuIo};
use verity_direct::ctx::FieldKind;
use verity_direct::program::{UnitIo, STEPS};
use verity_direct::stark::{self, BabyBearBackend, Backend, FriParams, GoldilocksBackend, DEFAULT_FRI};

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
        Some("vu-bench") => match field_arg(&args) {
            FieldKind::Goldilocks => vu_bench::<GoldilocksBackend>(&args),
            FieldKind::BabyBear => vu_bench::<BabyBearBackend>(&args),
        },
        Some("vu-neg") => match field_arg(&args) {
            FieldKind::Goldilocks => vu_neg::<GoldilocksBackend>(&args),
            FieldKind::BabyBear => vu_neg::<BabyBearBackend>(&args),
        },
        Some("chain-layout") => {
            let kind = field_arg(&args);
            let (air_, lay) = ChainAir::new(kind);
            let lqd = match kind {
                FieldKind::Goldilocks => stark::log_quotient_degree_of::<stark::Val64, _>(&air_),
                FieldKind::BabyBear => stark::log_quotient_degree_of::<stark::Val32, _>(&air_),
            };
            println!("field={kind:?} width={} hints={} materialised={} constraints={} log_quotient_degree={}", lay.width, lay.n_hints, lay.n_mat, lay.constraint_names.len(), lqd);
            census(&lay);
            if arg(&args, "--names").is_some() {
                for (i, n) in lay.col_names.iter().enumerate() {
                    println!("col {i:4} {n}");
                }
            }
        }
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
            census(&lay);
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
            "constraints": "WP2 gadget contracts as direct AIR constraints; range checks as 2-bit limbs (degree-4 constraint), fixed tables as one-hot selectors, products materialised above degree 5; no lookup argument",
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

// ---------------------------------------------------------------------------------------------------------------
// The K=1536 VU chain
// ---------------------------------------------------------------------------------------------------------------

fn field_arg(args: &[String]) -> FieldKind {
    match arg(args, "--field").as_deref().unwrap_or("goldilocks") {
        "goldilocks" => FieldKind::Goldilocks,
        "babybear" => FieldKind::BabyBear,
        other => panic!("--field must be goldilocks or babybear, got {other}"),
    }
}

fn census(lay: &Layout) {
    let limbs = lay.col_names.iter().filter(|n| n.contains(".limb[")).count();
    let sels = lay.col_names.iter().filter(|n| n.contains(".sel[") && !n.contains(".max.sel[")).count();
    let invs = lay.col_names.iter().filter(|n| n.ends_with(".inv")).count();
    let slack = lay.col_names.iter().filter(|n| n.ends_with(".slack")).count();
    let mats = lay.col_names.iter().filter(|n| *n == "_mat").count();
    println!("range-limb columns={limbs} table one-hot selectors={sels} exclusion inverses={invs} slack={slack} auto-materialised={mats}");
    println!("value columns (words + hints + named materialisations)={}", lay.width - limbs - sels - invs - slack - mats);
}

fn read_u16_words(path: &Path) -> Vec<u64> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u64).collect()
}

fn read_u32_words(path: &Path) -> Vec<u64> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64).collect()
}

/// A VU tier of `bench-instances/v1`: `x`, `w` (K words per instance), claimed `y`, the recorded accumulators
/// (positives only).
struct VuTier {
    vus: Vec<VuIo>,
    accs: Option<Vec<Vec<u64>>>,
    index: Value,
}

fn load_tier(root: &Path, tier: &str) -> VuTier {
    let k = 16 * STEPS as usize;
    let x = read_u16_words(&root.join(format!("{tier}.x.u16")));
    let w = read_u16_words(&root.join(format!("{tier}.w.u16")));
    let y = read_u16_words(&root.join(format!("{tier}.y.u16")));
    let n = y.len();
    assert_eq!(x.len(), n * k, "{tier}.x.u16 length");
    assert_eq!(w.len(), n * k, "{tier}.w.u16 length");
    let vus = (0..n).map(|i| VuIo { x: x[i * k..(i + 1) * k].to_vec(), w: w[i * k..(i + 1) * k].to_vec(), y16: y[i] }).collect();
    let accs = if tier == "vu-k1536" {
        let a = read_u32_words(&root.join(format!("{tier}.acc.u32")));
        assert_eq!(a.len(), n * STEPS as usize, "{tier}.acc.u32 length");
        Some((0..n).map(|i| a[i * STEPS as usize..(i + 1) * STEPS as usize].to_vec()).collect())
    } else {
        None
    };
    let index: Value = serde_json::from_str(&fs::read_to_string(root.join(format!("{tier}.index.json"))).unwrap()).unwrap();
    VuTier { vus, accs, index }
}

fn manifest_sha256(root: &Path) -> Option<String> {
    use sha2::Digest;
    let b = fs::read(root.join("manifest.json")).ok()?;
    Some(format!("{:x}", sha2::Sha256::digest(&b)))
}

fn vu_bench<B: Backend>(args: &[String])
where
    Val<B::SC>: PrimeField64 + Send + Sync,
{
    let kind = B::KIND;
    let root = Path::new(&arg(args, "--root").unwrap_or_else(|| "fixtures/bench-instances/v1".into())).to_path_buf();
    let out = arg(args, "--out").or_else(|| std::env::var("RESEARCH_RUN_DIR").ok()).unwrap_or_else(|| ".".into());
    let mut fri: FriParams = DEFAULT_FRI;
    if let Some(q) = arg(args, "--queries") {
        fri.num_queries = q.parse().unwrap();
    }
    fs::create_dir_all(&out).unwrap();

    let tier = load_tier(&root, "vu-k1536");
    let n_vus: usize = arg(args, "--vus").map(|s| s.parse().unwrap()).unwrap_or(tier.vus.len());
    let vus = &tier.vus[..n_vus];
    let recorded_accs = tier.accs.as_ref().unwrap();

    let (air_, lay) = ChainAir::new(kind);
    let idx: &ChainIdx = &air_.idx;
    let t0 = Instant::now();
    let (trace, accs): (RowMajorMatrix<Val<B::SC>>, Vec<Vec<u64>>) = air::chain_trace(kind, lay.width, idx, vus, 2);
    let t_witness = t0.elapsed().as_secs_f64();

    // the correctness gate: every intermediate accumulator and the output word against the recorded ones; the
    // claim column carries y16, so an honest trace exists only when the last-row epilogue equals the claim
    // (chain_trace panics on an integer violation)
    let mut acc_mismatch = 0usize;
    for (i, a) in accs.iter().enumerate() {
        if a != &recorded_accs[i] {
            acc_mismatch += 1;
        }
    }
    assert_eq!(acc_mismatch, 0, "{acc_mismatch} VUs disagree with the recorded accumulators");
    let claim_col = lay.col(verity_direct::program::COL_CLAIM);
    let last_col = lay.col(verity_direct::program::COL_LAST);
    let cast_col = lay.col("cast.y16");
    let steps = STEPS as usize;
    for (i, vu) in vus.iter().enumerate() {
        let r = i * steps + steps - 1;
        let row = trace.row_slice(r);
        assert_eq!(row[last_col].as_canonical_u64(), 1);
        assert_eq!(row[claim_col].as_canonical_u64(), vu.y16, "vu {i}: claim column");
        assert_eq!(row[cast_col].as_canonical_u64(), vu.y16, "vu {i}: epilogue word != recorded word");
    }

    let rows = trace.height();
    let width = trace.width();
    let log_n = rows.trailing_zeros() as usize;
    let lqd = stark::log_quotient_degree_of::<Val<B::SC>, _>(&air_);
    assert!(lqd <= fri.log_blowup, "quotient degree 2^{lqd} exceeds blowup 2^{}", fri.log_blowup);

    let t1 = Instant::now();
    let proof = stark::prove_with::<B, _>(&air_, trace, fri);
    let t_prove = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let bytes = stark::proof_bytes_of(&proof);
    let t_ser = t2.elapsed().as_secs_f64();
    let t3 = Instant::now();
    let ok = stark::verify_with::<B, _>(&air_, &proof, log_n, fri);
    let t_verify = t3.elapsed().as_secs_f64();
    assert!(ok.is_ok(), "verification failed: {:?}", ok.err());
    let t_total = t_witness + t_prove + t_ser;
    let threads = rayon::current_num_threads();
    let rounds = log_n + 5;
    let bits_proven = fri.bits_proven();
    let bits_conj = fri.bits_conjectured();
    let q128 = fri.queries_for_proven(128.0);

    let ext_deg = match kind {
        FieldKind::Goldilocks => 2,
        FieldKind::BabyBear => 4,
    };
    let committed_trace = (rows * width) as u64;
    let quotient_cells = (rows * (1 << lqd) * ext_deg) as u64;
    let field_bytes = match kind {
        FieldKind::Goldilocks => 8u64,
        FieldKind::BabyBear => 4,
    };
    let per_vu_cells = (steps * width) as u64;
    // bench.contract NATIVE_REFERENCE: A100 dense BF16 tensor-core peak, 312 TFLOP/s; 2K flop per VU
    let native_seconds_per_vu = 2.0 * 1536.0 / 312e12;
    let run_id = std::env::var("RESEARCH_RUN_ID").unwrap_or_else(|_| "local".into());
    let commit = option_env!("VERITY_COMMIT").map(|s| s.to_string());
    let field_name = match kind {
        FieldKind::Goldilocks => "goldilocks",
        FieldKind::BabyBear => "babybear",
    };

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
        json!({"name": "vu.count", "value": n_vus, "unit": "count"}),
        json!({"name": "vu.seconds_per_vu", "value": t_total / n_vus as f64, "unit": "s"}),
        json!({"name": "vu.per_second", "value": n_vus as f64 / t_total, "unit": "1/s"}),
        json!({"name": "overhead.vs_native_peak", "value": (t_total / n_vus as f64) / native_seconds_per_vu, "unit": "ratio"}),
        json!({"name": "direct.vus", "value": n_vus, "unit": "count"}),
        json!({"name": "direct.rows", "value": rows, "unit": "count"}),
        json!({"name": "direct.rows_real", "value": n_vus * steps, "unit": "count"}),
        json!({"name": "direct.width", "value": width, "unit": "count"}),
        json!({"name": "direct.trace_cells", "value": committed_trace, "unit": "count"}),
        json!({"name": "direct.quotient_cells", "value": quotient_cells, "unit": "count"}),
        json!({"name": "direct.committed_cells_per_vu", "value": per_vu_cells, "unit": "count"}),
        json!({"name": "direct.committed_bytes_per_vu", "value": per_vu_cells * field_bytes, "unit": "B"}),
        json!({"name": "direct.threads", "value": threads, "unit": "count"}),
        json!({"name": "direct.prove_s", "value": t_prove, "unit": "s"}),
        json!({"name": "direct.verify_s", "value": t_verify, "unit": "s"}),
        json!({"name": "direct.fri_bits_proven", "value": bits_proven, "unit": "bits"}),
        json!({"name": "direct.fri_bits_conjectured", "value": bits_conj, "unit": "bits"}),
        json!({"name": "direct.fri_queries_for_128_proven", "value": q128, "unit": "count"}),
    ];
    let result = json!({
        "schema": "research/result/v0.1",
        "run_id": run_id,
        "workload_fingerprint": {
            "profile": "first-campaign-target/2026-09-21",
            "domain": "finite",
            "K": 1536,
            "B": n_vus,
            "N_subbatches": 1,
            "scope": "vu",
            "instances": {"dataset": "bench-instances/v1", "tier": "vu-k1536", "range": [0, n_vus], "manifest_sha256": manifest_sha256(&root)},
            "proof_class": "NON_ZK_PROOF_DIAGNOSTIC",
            "authentication": "excluded",
            "zk_mode": "none",
            "security": {
                "target": -80,
                "fri": fri,
                "field": field_name,
                "challenge_field": B::challenge_name(),
                "challenge_bits": B::challenge_bits(),
                "hash": "keccak256",
                "fri_bits_proven_bound": bits_proven,
                "fri_bits_conjectured_bound": bits_conj,
                "queries_for_128_proven": q128,
                "note": "proven bound: unique decoding radius, -log2((1+rate)/2) per query + PoW; field terms (DEEP/quotient batching over the challenge field, ~degree/|challenge field|) omitted"
            },
            "hardware": {"gpu": null, "cpu": cpu_model().map(|m| json!({"model": m, "count": threads}))},
            "software": {"backend": {"name": "verity-direct (plonky3 uni-stark)", "version": "0.2.0", "commit": commit, "plonky3": "0.4.3-succinct"}, "toolchain": {"rustc": option_env!("VERITY_RUSTC").unwrap_or("unknown")}},
            "preprocessing": "none",
            "retention": "retained",
            "unit": "vu",
            "constraints": "WP2 gadget contracts + Epilogue as direct AIR constraints, one row per transition unit, accumulator carried by a transition constraint, VU boundary by a step counter; range checks as 2-bit limbs, fixed tables as one-hot selectors; no lookup argument",
            "note": "Plonky3 FRI/Keccak is the diagnostic PCS, not Candidate B's Ligero-style interactive PCS (PROTOCOL.md); the claimed output words are committed trace cells (the `vu.claim` column) that the last-row constraint binds to the epilogue word -- not bound to any external statement (authentication excluded)"
        },
        "validation": {"status": "passed", "checker": "p3_uni_stark::verify + recorded accumulators", "detail": format!("proof verified; {n_vus} VUs x {steps} steps in {rows} rows; every intermediate accumulator and output word equals bench-instances/v1"), "evidence": [], "index_note": tier.index.get("note").cloned().unwrap_or(Value::Null)},
        "measurements": measurements,
        "measurement_files": [],
        "artifacts": []
    });
    let path = format!("{out}/result.json");
    fs::write(&path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!(
        "field={field_name} vus={n_vus} rows={rows} width={width} witness={t_witness:.3}s prove={t_prove:.3}s total={t_total:.3}s ({:.4} s/VU) verify={t_verify:.3}s bytes={bytes} threads={threads} peak_rss={:?} bits_proven={bits_proven:.1} -> {path}",
        t_total / n_vus as f64,
        peak_rss_bytes()
    );
}

/// The 52 `vu-k1536-neg` instances: the honest witness generator must reject each claimed word (an integer-checker
/// violation on some row), and the rows it produces must violate at least one field constraint of the AIR -- so
/// no trace of this shape satisfies the AIR with the claimed word.  A negative that yields violation-free rows is a
/// soundness bug and fails the command.
fn vu_neg<B: Backend>(args: &[String])
where
    Val<B::SC>: PrimeField64 + Send + Sync,
{
    let kind = B::KIND;
    let root = Path::new(&arg(args, "--root").unwrap_or_else(|| "fixtures/bench-instances/v1".into())).to_path_buf();
    let out = arg(args, "--out").or_else(|| std::env::var("RESEARCH_RUN_DIR").ok());
    let tier = load_tier(&root, "vu-k1536-neg");
    let (air_, lay) = ChainAir::new(kind);
    let rows_index = tier.index["rows"].as_array().unwrap().clone();
    let mut report = vec![];
    let mut soundness_bugs = 0usize;
    for (i, vu) in tier.vus.iter().enumerate() {
        let (rows, _accs, viol) = air::chain_rows(kind, lay.width, &air_.idx, vu);
        let field_fail = air::chain_failing_rows::<Val<B::SC>>(kind, &air_, &lay, &rows, true);
        let rejected = !viol.is_empty();
        let unsatisfiable = !field_fail.is_empty();
        if !rejected || !unsatisfiable {
            soundness_bugs += 1;
        }
        let meta = &rows_index[i];
        println!(
            "neg {i:2} mistake={} verdict={} claimed={} correct={}: integer violations={} ({}), failing field constraints={} ({})",
            meta[1],
            meta[5],
            meta[3],
            meta[4],
            viol.len(),
            viol.first().map(|s| s.as_str()).unwrap_or("-"),
            field_fail.len(),
            field_fail.first().map(|(r, n)| format!("row {r} {n}")).unwrap_or("-".into())
        );
        report.push(json!({"id": meta[0], "mistake": meta[1], "verdict": meta[5], "claimed_y": meta[3], "correct_y": meta[4], "integer_violations": viol.len(), "first_violation": viol.first(), "failing_field_constraints": field_fail.len(), "first_failing": field_fail.first().map(|(r, n)| json!([r, n]))}));
    }
    // the positives' claim columns are honest: as a control, the first negative's base VU with its correct word
    let control = if let Some(row) = rows_index.first() {
        if let Some(correct) = row[4].as_u64() {
            let mut vu = tier.vus[0].clone();
            vu.y16 = correct;
            let (rows, _, viol) = air::chain_rows(kind, lay.width, &air_.idx, &vu);
            let field_fail = air::chain_failing_rows::<Val<B::SC>>(kind, &air_, &lay, &rows, true);
            println!("control: negative 0 with its correct word: integer violations={} field failures={}", viol.len(), field_fail.len());
            assert!(viol.is_empty() && field_fail.is_empty(), "control failed: the correct word is rejected");
            json!({"integer_violations": viol.len(), "failing_field_constraints": field_fail.len()})
        } else {
            Value::Null
        }
    } else {
        Value::Null
    };
    let summary = json!({
        "schema": "verity-direct/vu-neg/v1",
        "field": format!("{kind:?}"),
        "dataset": {"dataset": "bench-instances/v1", "tier": "vu-k1536-neg", "manifest_sha256": manifest_sha256(&root), "n": tier.vus.len()},
        "rejected_all": soundness_bugs == 0,
        "soundness_bugs": soundness_bugs,
        "control": control,
        "instances": report,
        "run_id": std::env::var("RESEARCH_RUN_ID").ok(),
    });
    if let Some(out) = out {
        fs::create_dir_all(&out).unwrap();
        fs::write(format!("{out}/vu_neg.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    }
    println!("vu-neg field={kind:?}: {} negatives, {} rejected by witness generation AND unsatisfiable in the field", tier.vus.len(), tier.vus.len() - soundness_bugs);
    assert_eq!(soundness_bugs, 0, "SOUNDNESS BUG: {soundness_bugs} negatives produce violation-free rows");
}
