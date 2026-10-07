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
use p3_uni_stark::{StarkGenericConfig, Val};
use serde_json::{json, Value};
use verity_direct::air::{self, ChainAir, ChainIdx, Layout, UnitAir, VuIo};
use verity_direct::ctx::{modp, FieldKind};
use verity_direct::program::{UnitIo, STEPS};
use verity_direct::stark::{self, BabyBearBackend, Backend, FriParams, GoldilocksBackend, DEFAULT_FRI};
use verity_direct::statement::Statement;

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
    if std::env::var_os("RUST_LOG").is_some() {
        // profiling: Plonky3's `info_span!`s (commit to trace data / compute quotient polynomial / commit to
        // quotient poly chunks / open, and the DFT + Merkle spans inside) are printed on close with `time.busy`.
        use tracing_subscriber::fmt::format::FmtSpan;
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_span_events(FmtSpan::CLOSE)
            .with_target(false)
            .with_ansi(false)
            .with_writer(std::io::stderr)
            .init();
    }
    let v2 = args.iter().any(|a| a == "--v2");
    match args.get(1).map(|s| s.as_str()) {
        Some("chain-layout") if v2 => chain_layout_v2(&args),
        Some("vu-bench") if v2 => match arg(&args, "--ext").as_deref() {
            Some("5") => vu_bench_v2::<verity_direct::stark::BabyBear5Backend>(&args),
            Some("6") => vu_bench_v2::<verity_direct::stark::BabyBear6Backend>(&args),
            _ => vu_bench_v2::<BabyBearBackend>(&args),
        },
        Some("vu-neg") if v2 => vu_neg_v2(&args),
        Some("vu-honest") if v2 => vu_honest_v2(&args),
        Some("check-rows") if v2 => check_rows_v2::<stark::Val32, stark::ChallengeBb>(&args),
        Some("bench") => bench(&args),
        Some("crosscheck") => crosscheck(&args),
        Some("vu-bench") if args.iter().any(|a| a == "--lookup") => match (field_arg(&args), arg(&args, "--ext").as_deref()) {
            (FieldKind::Goldilocks, _) => vu_bench_lookup::<GoldilocksBackend>(&args),
            (FieldKind::BabyBear, Some("5")) => vu_bench_lookup::<verity_direct::stark::BabyBear5Backend>(&args),
            (FieldKind::BabyBear, _) => vu_bench_lookup::<BabyBearBackend>(&args),
        },
        Some("vu-bench") => match field_arg(&args) {
            FieldKind::Goldilocks => vu_bench::<GoldilocksBackend>(&args),
            FieldKind::BabyBear => vu_bench::<BabyBearBackend>(&args),
        },
        Some("vu-neg") => match field_arg(&args) {
            FieldKind::Goldilocks => vu_neg::<GoldilocksBackend>(&args),
            FieldKind::BabyBear => vu_neg::<BabyBearBackend>(&args),
        },
        Some("check-rows") => match field_arg(&args) {
            FieldKind::Goldilocks => check_rows::<stark::Val64, stark::Challenge>(&args),
            FieldKind::BabyBear => check_rows::<stark::Val32, stark::ChallengeBb>(&args),
        },
        Some("chain-layout") if args.iter().any(|a| a == "--lookup") => {
            let kind = field_arg(&args);
            let d = if kind.small() { 4 } else { 2 };
            let (air_, lay) = air::LookupChainAir::new(kind, d);
            let ll = &air_.lay;
            let pre_w = ll.tables.pre_width();
            let aux_w = ll.aux_width(d);
            println!(
                "field={kind:?} lookup=logup main_width={} (incl. multiplicity) aux_width={} (helpers={} x{d} + g + S) pre_width={pre_w} committed_width={} hints={} materialised={} constraints={} queries/row={} table_rows={}",
                lay.width,
                aux_w,
                ll.n_helpers(),
                lay.width + aux_w + pre_w,
                lay.n_hints,
                lay.n_mat,
                lay.constraint_names.len(),
                ll.n_queries(),
                ll.tables.total_rows
            );
            for (n, r) in ll.tables.describe() {
                println!("  table {n:8} rows={r}");
            }
            let mut by_deg = std::collections::BTreeMap::new();
            for d in &ll.query_degs {
                *by_deg.entry(*d).or_insert(0usize) += 1;
            }
            println!("  query degrees: {by_deg:?}");
            census(&lay);
            if args.iter().any(|a| a == "--names") {
                for (i, n) in lay.col_names.iter().enumerate() {
                    println!("col {i:4} {n}");
                }
                for (i, n) in ll.query_names.iter().enumerate() {
                    println!("query {i:4} deg={} {n}", ll.query_degs[i]);
                }
            }
        }
        Some("chain-layout") => {
            let kind = field_arg(&args);
            let (air_, lay) = ChainAir::new(kind);
            let lqd = match kind {
                FieldKind::Goldilocks => stark::log_quotient_degree_of::<stark::Val64, _>(&air_),
                FieldKind::BabyBear => stark::log_quotient_degree_of::<stark::Val32, _>(&air_),
            };
            println!("field={kind:?} width={} hints={} materialised={} constraints={} log_quotient_degree={}", lay.width, lay.n_hints, lay.n_mat, lay.constraint_names.len(), lqd);
            census(&lay);
            if args.iter().any(|a| a == "--names") {
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

/// A VU tier of `bench-instances/v1`: `x`, `w` (K words per input), claimed `y`, the recorded accumulators
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
    Val<B::SC>: PrimeField64 + p3_field::TwoAdicField + Send + Sync,
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
    // the epilogue word is the linear form hi + cast.carry (hi = cast.hi over Goldilocks, the pack.y_hi limb over BabyBear)
    let cast_hi = if lay.has("cast.hi") { lay.col("cast.hi") } else { lay.col("pack.y_hi") };
    let cast_carry = lay.col("cast.carry");
    let steps = STEPS as usize;
    for (i, vu) in vus.iter().enumerate() {
        let r = i * steps + steps - 1;
        let row = trace.row_slice(r);
        assert_eq!(row[last_col].as_canonical_u64(), 1);
        assert_eq!(row[claim_col].as_canonical_u64(), vu.y16, "vu {i}: claim column");
        assert_eq!(row[cast_hi].as_canonical_u64() + row[cast_carry].as_canonical_u64(), vu.y16, "vu {i}: epilogue word != recorded word");
    }

    let rows = trace.height();
    let width = trace.width();
    let log_n = rows.trailing_zeros() as usize;
    let lqd = stark::log_quotient_degree_of::<Val<B::SC>, _>(&air_);
    assert!(lqd <= fri.log_blowup, "quotient degree 2^{lqd} exceeds blowup 2^{}", fri.log_blowup);

    // the statement (PROTOCOL.md "Statement binding"): every claimed word; the operands too with --bind-operands
    let bind_operands = args.iter().any(|a| a == "--bind-operands");
    let words: Vec<u64> = vus.iter().map(|v| v.y16).collect();
    let stmt = if bind_operands {
        Statement::with_operands(words, vus.iter().flat_map(|v| v.x.iter().copied()).collect(), vus.iter().flat_map(|v| v.w.iter().copied()).collect())
    } else {
        Statement::new(words)
    };
    assert_eq!(stmt.degree_bits(), log_n);

    // `--unbound`: the pre-fix path (no header, no statement check, degree_bits from the proof) for a same-machine
    // A/B timing of the fix ONLY; its result is labelled as such and is not a proof number.
    let unbound = args.iter().any(|a| a == "--unbound");
    // --rtt-ms R (or VERITY_RTT_MS): emulate the network; the prover sleeps R ms at every critical-path challenge.
    // rtt::begin precedes the prover's B::challenger() (inside prove_bound / prove_unbound) so that challenger is
    // the active (counting) one; the statement header is absorbed into that same challenger.
    let rtt_ms: f64 = arg(args, "--rtt-ms").or_else(|| std::env::var("VERITY_RTT_MS").ok()).map(|s| s.parse().expect("--rtt-ms takes milliseconds")).unwrap_or(0.0);
    let t1 = Instant::now();
    verity_direct::rtt::begin(rtt_ms);
    let proof = if unbound { stark::prove_unbound::<B, _>(&air_, trace, fri) } else { stark::prove_bound::<B, _>(&air_, &lay, trace, fri, &stmt) };
    let (trips, slept) = verity_direct::rtt::end();
    let t_prove_wall = t1.elapsed().as_secs_f64();
    let t_prove = t_prove_wall - slept; // compute: the wall time less the emulated waits for the verifier's coins
    let t2 = Instant::now();
    let proof_bytes = bincode::serialize(&proof).expect("serialise proof");
    let bytes = proof_bytes.len();
    let t_ser = t2.elapsed().as_secs_f64();
    let t3 = Instant::now();
    let ok = if unbound {
        stark::verify_unbound::<B, _>(&air_, &proof, log_n, fri).map_err(stark::BoundError::Inner)
    } else {
        stark::verify_bound_bytes::<B, _>(&air_, &lay, &proof_bytes, fri, &stmt)
    };
    let t_verify = t3.elapsed().as_secs_f64();
    assert!(ok.is_ok(), "verification failed: {:?}", ok.err());
    // the statement checks fire: a proof of these VUs is not a proof of any other word list or batch size
    if !unbound {
        let mut other = stmt.clone();
        other.words[0] ^= 1;
        let r = stark::verify_bound_bytes::<B, _>(&air_, &lay, &proof_bytes, fri, &other);
        assert!(matches!(r, Err(stark::BoundError::StatementColumn { .. })), "wrong-word statement not rejected: {r:?}");
        let mut bigger = stmt.clone();
        bigger.words.extend(std::iter::repeat(0).take(n_vus.max(1)));
        let r = stark::verify_bound_bytes::<B, _>(&air_, &lay, &proof_bytes, fri, &bigger);
        assert!(matches!(r, Err(stark::BoundError::DegreeBits { .. })), "wrong-B statement not rejected: {r:?}");
    }
    let binding = binding_json(unbound, bind_operands, "vu.y16", 1, rows, B::challenge_bits(), &[]);
    write_result(BenchNumbers {
        kind,
        out: &out,
        fri,
        n_vus,
        rows,
        width,
        committed_width: width,
        t_witness,
        t_prove,
        t_ser,
        t_verify,
        bytes,
        lqd,
        tier: &tier,
        root: &root,
        challenge_name: B::challenge_name(),
        challenge_bits: B::challenge_bits(),
        rounds_extra: 0,
        constraints_note: "WP2 gadget contracts + Epilogue as direct AIR constraints, one row per transition unit, accumulator carried by a transition constraint, VU boundary by a step counter; range checks as 2-bit limbs, fixed tables as one-hot selectors; no lookup argument",
        realisation: "limbs",
        extra: vec![],
        lookup_bits: None,
        ext_deg: if kind.small() { 4 } else { 2 },
        rtt_ms,
        depth: trips as usize,
        slept,
        t_prove_wall,
        binding,
    });
}

/// The `statement_binding` block of a result, its `checker` / `detail` strings: shared by both realisations.
struct Binding {
    statement_binding: Value,
    checker: &'static str,
    detail_suffix: String,
    note: &'static str,
}

fn binding_json(unbound: bool, bind_operands: bool, public_columns: &str, n_public_columns: usize, rows: usize, challenge_bits: usize, header_extra: &[&str]) -> Binding {
    if unbound {
        return Binding {
            statement_binding: json!({"bound": false, "note": "TIMING BASELINE of the pre-fix path (red-team F1/F2 forgeable): not a proof number"}),
            checker: "verify (UNBOUND baseline) + recorded accumulators",
            detail_suffix: String::new(),
            note: "TIMING BASELINE: pre-fix unbound path (no statement); Plonky3 FRI/Keccak diagnostic PCS",
        };
    }
    let mut header: Vec<&str> = vec!["domain tag", "AIR id (sha256 of field, layout, constraint names)", "field prime", "width", "challenge extension degree", "B", "K", "STEPS", "rows", "degree_bits", "FRI log_blowup/num_queries/pow_bits", "every claimed word", if bind_operands { "sha256(x), sha256(w)" } else { "operands: not bound" }];
    header.extend_from_slice(header_extra);
    let cols = if bind_operands { n_public_columns + 32 } else { n_public_columns };
    Binding {
        statement_binding: json!({
            "bound": true,
            "header_absorbed_before_first_challenge": header,
            "degree_bits": "fixed by the verifier from B; a proof with another degree_bits is rejected (DegreeBitsMismatch)",
            "public_columns_checked_at_zeta": if bind_operands { format!("{public_columns} + x.a[0..16] + x.b[0..16]") } else { public_columns.to_string() },
            "extra_soundness_term": format!("<= {cols} columns x n / |challenge field| = 2^{:.1}", (cols as f64 * rows as f64).log2() - challenge_bits as f64),
            "operands_bound": bind_operands
        }),
        checker: "verity_direct statement-bound verify (uni-stark transcript + statement header + public columns at zeta) + recorded accumulators",
        detail_suffix: " for the statement of the claimed words; wrong-word and wrong-B statements rejected".to_string(),
        note: "Plonky3 FRI/Keccak is the diagnostic PCS, not Candidate B's Ligero-style interactive PCS (PROTOCOL.md); the claimed output words are the statement: absorbed into the transcript before the first challenge and checked at zeta against the committed `vu.y16` column (PROTOCOL.md 12 'Statement binding', red-team F1-F3 fixed); operands are witness unless --bind-operands (authentication excluded)",
    }
}

/// `vu-bench --lookup`: the LogUp realisation (PROTOCOL.md section 14): three commitment rounds.
fn vu_bench_lookup<B: Backend>(args: &[String])
where
    Val<B::SC>: PrimeField64 + p3_field::TwoAdicField + Send + Sync,
    <<B::SC as StarkGenericConfig>::Pcs as p3_commit::Pcs<<B::SC as StarkGenericConfig>::Challenge, <B::SC as StarkGenericConfig>::Challenger>>::Commitment: PartialEq,
{
    use verity_direct::lookup::{aux_matrix, lookup_soundness_bits, PRE_WIDTH};
    use verity_direct::lookup_stark::{lookup_statement, prove_lookup, prove_lookup_bound, verify_lookup, verify_lookup_bound_bytes};
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
    let ext_deg = <<B::SC as StarkGenericConfig>::Challenge as p3_field::AbstractExtensionField<Val<B::SC>>>::D;

    let (air_, lay) = air::LookupChainAir::new(kind, ext_deg);
    let t0 = Instant::now();
    let (main, queries, accs): (RowMajorMatrix<Val<B::SC>>, Vec<u32>, Vec<Vec<u64>>) = air::chain_trace_lookup(&air_, vus, 2);
    let rows = main.height();
    let pre: RowMajorMatrix<Val<B::SC>> = air_.lay.tables.preprocessed(rows);
    let t_witness = t0.elapsed().as_secs_f64();

    let mut acc_mismatch = 0usize;
    for (i, a) in accs.iter().enumerate() {
        if a != &recorded_accs[i] {
            acc_mismatch += 1;
        }
    }
    assert_eq!(acc_mismatch, 0, "{acc_mismatch} VUs disagree with the recorded accumulators");
    let claim_col = lay.col(verity_direct::program::COL_CLAIM);
    let last_col = lay.col(verity_direct::program::COL_LAST);
    let cast_hi = if lay.has("cast.hi") { lay.col("cast.hi") } else { lay.col("pack.y_hi") };
    let cast_carry = lay.col("cast.carry");
    let steps = STEPS as usize;
    for (i, vu) in vus.iter().enumerate() {
        let r = i * steps + steps - 1;
        let row = main.row_slice(r);
        assert_eq!(row[last_col].as_canonical_u64(), 1);
        assert_eq!(row[claim_col].as_canonical_u64(), vu.y16, "vu {i}: claim column");
        assert_eq!(row[cast_hi].as_canonical_u64() + row[cast_carry].as_canonical_u64(), vu.y16, "vu {i}: epilogue word != recorded word");
    }

    let width = main.width();
    let aux_w = air_.lay.aux_width(ext_deg);
    let committed_width = width + aux_w + PRE_WIDTH;
    let log_n = rows.trailing_zeros() as usize;
    let lqd = p3_util::log2_ceil_usize(verity_direct::program::MAX_DEG - 1);
    assert!(lqd <= fri.log_blowup);
    let mult: Vec<Val<B::SC>> = (0..rows).map(|r| main.values[r * width + width - 1]).collect();
    let llay = air_.lay.clone();

    // the statement (PROTOCOL.md 12.5 / 14.7): every claimed word, the trace at least as tall as the tables
    let bind_operands = args.iter().any(|a| a == "--bind-operands");
    let words: Vec<u64> = vus.iter().map(|v| v.y16).collect();
    let stmt = if bind_operands {
        Statement::with_operands(words, vus.iter().flat_map(|v| v.x.iter().copied()).collect(), vus.iter().flat_map(|v| v.w.iter().copied()).collect())
    } else {
        Statement::new(words)
    };
    let stmt = lookup_statement(&air_, stmt);
    assert_eq!(stmt.degree_bits(), log_n);
    // `--unbound`: the pre-fix path (no header, degree_bits from the proof, no statement check): a same-machine
    // A/B timing of the fix ONLY; labelled as such in the result
    let unbound = args.iter().any(|a| a == "--unbound");

    // --rtt-ms R (or VERITY_RTT_MS): emulate the network; the prover sleeps R ms at every critical-path challenge.
    // rtt::begin must precede the prover's B::challenger() so that challenger is the active (counting) one.
    let rtt_ms: f64 = arg(args, "--rtt-ms").or_else(|| std::env::var("VERITY_RTT_MS").ok()).map(|s| s.parse().expect("--rtt-ms takes milliseconds")).unwrap_or(0.0);
    let t1 = Instant::now();
    verity_direct::rtt::begin(rtt_ms);
    let aux_fn = |z, beta| aux_matrix(&llay, &queries, &mult, rows, z, beta, true);
    let (proof, stage) = if unbound {
        let config = B::config(log_n, fri);
        let mut challenger = B::challenger();
        prove_lookup(&config, &air_, &mut challenger, pre, main, aux_fn)
    } else {
        prove_lookup_bound::<B>(&air_, &lay, fri, &stmt, pre, main, aux_fn, true)
    };
    let (trips, slept) = verity_direct::rtt::end();
    let t_prove_wall = t1.elapsed().as_secs_f64();
    let t_prove = t_prove_wall - slept; // compute: the wall time less the emulated waits for the verifier's coins
    let t2 = Instant::now();
    let proof_bytes = bincode::serialize(&proof).expect("serialise proof");
    let bytes = proof_bytes.len();
    let t_ser = t2.elapsed().as_secs_f64();
    let t3 = Instant::now();
    let ok = if unbound {
        let config = B::config(log_n, fri);
        let mut challenger = B::challenger();
        verify_lookup(&config, &air_, &mut challenger, &proof, air_.lay.tables.preprocessed(rows), proof.degree_bits, &[]).map_err(stark::BoundError::Inner)
    } else {
        verify_lookup_bound_bytes::<B>(&air_, &lay, &proof_bytes, fri, &stmt)
    };
    let t_verify = t3.elapsed().as_secs_f64();
    assert!(ok.is_ok(), "verification failed: {:?}", ok.err());
    // the statement checks fire: not a proof of another word list, another batch size, or another table set
    if !unbound {
        let mut other = stmt.clone();
        other.words[0] ^= 1;
        let r = verify_lookup_bound_bytes::<B>(&air_, &lay, &proof_bytes, fri, &other);
        assert!(matches!(r, Err(stark::BoundError::StatementColumn { .. })), "wrong-word statement not rejected: {r:?}");
        let mut bigger = stmt.clone();
        bigger.words.extend(std::iter::repeat(0).take((rows / steps + 1).max(1)));
        let r = verify_lookup_bound_bytes::<B>(&air_, &lay, &proof_bytes, fri, &bigger);
        assert!(matches!(r, Err(stark::BoundError::DegreeBits { .. })), "wrong-B statement not rejected: {r:?}");
        let other_air = air::LookupChainAir { width: air_.width, kind: air_.kind, idx: air_.idx.clone(), lay: verity_direct::lookup::substitute_table_row(&air_.lay, 0), ext_degree: air_.ext_degree };
        let r = verify_lookup_bound_bytes::<B>(&air_, &lay, &proof_bytes, fri, &stmt).and(verify_lookup_bound_bytes::<B>(&other_air, &lay, &proof_bytes, fri, &stmt));
        assert!(r.is_err(), "substituted-table verifier accepted the proof");
    }
    let lookup_bits = lookup_soundness_bits(rows, air_.lay.n_queries(), air_.lay.tables.total_rows, B::challenge_bits() as f64);
    println!(
        "lookup stages: commit_pre={:.2}s commit_main={:.2}s aux_build={:.2}s commit_aux={:.2}s quotient={:.2}s commit_quotient={:.2}s open={:.2}s; main_width={width} aux_width={aux_w} pre_width={PRE_WIDTH} helpers={} queries/row={} table_rows={} lookup_bits={lookup_bits:.1}",
        stage.commit_pre,
        stage.commit_main,
        stage.aux_build,
        stage.commit_aux,
        stage.quotient,
        stage.commit_quotient,
        stage.open,
        air_.lay.n_helpers(),
        air_.lay.n_queries(),
        air_.lay.tables.total_rows
    );
    let extra = vec![
        json!({"name": "direct.lookup.main_width", "value": width, "unit": "count"}),
        json!({"name": "direct.lookup.aux_width", "value": aux_w, "unit": "count"}),
        json!({"name": "direct.lookup.pre_width", "value": PRE_WIDTH, "unit": "count"}),
        json!({"name": "direct.lookup.helpers", "value": air_.lay.n_helpers(), "unit": "count"}),
        json!({"name": "direct.lookup.queries_per_row", "value": air_.lay.n_queries(), "unit": "count"}),
        json!({"name": "direct.lookup.table_rows", "value": air_.lay.tables.total_rows, "unit": "count"}),
        json!({"name": "direct.lookup.bits", "value": lookup_bits, "unit": "bits"}),
        json!({"name": "direct.stage.commit_pre_s", "value": stage.commit_pre, "unit": "s"}),
        json!({"name": "direct.stage.commit_main_s", "value": stage.commit_main, "unit": "s"}),
        json!({"name": "direct.stage.aux_build_s", "value": stage.aux_build, "unit": "s"}),
        json!({"name": "direct.stage.commit_aux_s", "value": stage.commit_aux, "unit": "s"}),
        json!({"name": "direct.stage.quotient_s", "value": stage.quotient, "unit": "s"}),
        json!({"name": "direct.stage.commit_quotient_s", "value": stage.commit_quotient, "unit": "s"}),
        json!({"name": "direct.stage.open_s", "value": stage.open, "unit": "s"}),
    ];
    let binding = binding_json(unbound, bind_operands, "vu.y16 (main matrix, round 1)", 1, rows, B::challenge_bits(), &["LogUp tag", "preprocessed width", "aux width", "aux extension degree", "sha256 of the table set (verifier's own)"]);
    write_result(BenchNumbers {
        kind,
        out: &out,
        fri,
        n_vus,
        rows,
        width,
        committed_width,
        t_witness,
        t_prove,
        t_ser,
        t_verify,
        bytes,
        lqd,
        tier: &tier,
        root: &root,
        challenge_name: B::challenge_name(),
        challenge_bits: B::challenge_bits(),
        rounds_extra: 2,
        constraints_note: "WP2 gadget contracts + Epilogue as direct AIR constraints, one row per transition unit, accumulator carried by a transition constraint, VU boundary by a step counter; range checks and fixed tables as LogUp queries against preprocessed tables (PROTOCOL.md section 14): multiplicity column in the main trace, helper / g / running-sum columns in an auxiliary round after the challenges z, beta",
        realisation: "logup",
        extra,
        lookup_bits: Some(lookup_bits),
        ext_deg,
        rtt_ms,
        depth: trips as usize,
        slept,
        t_prove_wall,
        binding,
    });
}

struct BenchNumbers<'a> {
    kind: FieldKind,
    out: &'a str,
    fri: FriParams,
    n_vus: usize,
    rows: usize,
    width: usize,
    /// Total committed width per row (main + aux + preprocessed for the lookup realisation).
    committed_width: usize,
    t_witness: f64,
    t_prove: f64,
    t_ser: f64,
    t_verify: f64,
    bytes: usize,
    lqd: usize,
    tier: &'a VuTier,
    root: &'a Path,
    challenge_name: &'static str,
    challenge_bits: usize,
    rounds_extra: usize,
    constraints_note: &'a str,
    realisation: &'a str,
    extra: Vec<Value>,
    lookup_bits: Option<f64>,
    ext_deg: usize,
    /// Emulated network RTT the prover slept at every critical-path challenge (0 = none).
    rtt_ms: f64,
    /// Measured observe -> sample boundaries of the prover's transcript (rounds.sequential_depth).
    depth: usize,
    /// Seconds spent in the emulated RTT sleeps (already removed from t_prove).
    slept: f64,
    /// Prover wall time including the sleeps.
    t_prove_wall: f64,
    /// Statement binding of this run (PROTOCOL.md 12): bound, or the labelled `--unbound` timing baseline.
    binding: Binding,
}

fn write_result(b: BenchNumbers) {
    let BenchNumbers { kind, out, fri, n_vus, rows, width, committed_width, t_witness, t_prove, t_ser, t_verify, bytes, lqd, tier, root, challenge_name, challenge_bits, rounds_extra, constraints_note, realisation, extra, lookup_bits, ext_deg, rtt_ms, depth, slept, t_prove_wall, binding } = b;
    let steps = STEPS as usize;
    let log_n = rows.trailing_zeros() as usize;
    let t_total = t_witness + t_prove + t_ser;
    let threads = rayon::current_num_threads();
    // derived coin count (rounds.protocol): alpha, zeta, FRI batch beta, log_n fold betas, PoW, query indices,
    // plus the extra commitment rounds of the realisation (z, beta for LogUp); the measured depth is `depth`
    let rounds = log_n + 5 + rounds_extra;
    let bits_proven = fri.bits_proven();
    let bits_conj = fri.bits_conjectured();
    let q128 = fri.queries_for_proven(128.0);

    let committed_trace = (rows * committed_width) as u64;
    let quotient_cells = (rows * (1 << lqd) * ext_deg) as u64;
    let field_bytes = match kind {
        FieldKind::Goldilocks => 8u64,
        FieldKind::BabyBear => 4,
    };
    let per_vu_cells = (steps * committed_width) as u64;
    // bench.contract native_reference(FIRST.name) == verity.ml.circuits.targets.target.FIRST.native_peak: the A100 SXM4 dense
    // BF16 tensor-core peak with FP32 accumulate, 312 TFLOP/s; one GemmCoordinate<K> VU is 2K flop.  This bench is
    // BF16-only (the fingerprint below names profile FIRST); a bench of another target divides by that target's own
    // Target.native_peak, never by this constant.  overhead.vs_native_peak == rate.native_peak / rate.proved.
    let native_peak_flop_per_second = 312e12;
    let native_seconds_per_vu = 2.0 * 1536.0 / native_peak_flop_per_second;
    let proved_flop_per_second = 2.0 * 1536.0 * n_vus as f64 / t_total;
    let run_id = std::env::var("RESEARCH_RUN_ID").unwrap_or_else(|_| "local".into());
    let commit = option_env!("VERITY_COMMIT").map(|s| s.to_string());
    let field_name = match kind {
        FieldKind::Goldilocks => "goldilocks",
        FieldKind::BabyBear => "babybear",
    };

    // end-to-end wall time at the RTT sweep: measured at the emulated RTT (prover wall incl. the sleeps + verifier),
    // predicted at the others as compute + depth x RTT (the honest serial number; bytes / bandwidth not included)
    let mut e2e_measured: Vec<String> = vec![];
    let mut e2e_predicted: Vec<String> = vec![];
    let mut e2e: Vec<Value> = vec![];
    for (name, rtt) in [("0", 0.0f64), ("0.1", 0.1), ("1", 1.0), ("10", 10.0), ("50", 50.0)] {
        let key = format!("t.e2e.rtt_{name}ms");
        if (rtt - rtt_ms).abs() < 1e-9 {
            e2e.push(json!({"name": key, "value": t_witness + t_prove_wall + t_ser + t_verify, "unit": "s"}));
            e2e_measured.push(key);
        } else {
            e2e.push(json!({"name": key, "value": t_total + t_verify + depth as f64 * rtt / 1000.0, "unit": "s"}));
            e2e_predicted.push(key);
        }
    }
    let mut measurements = vec![
        json!({"name": "t.witness", "value": t_witness, "unit": "s"}),
        json!({"name": "t.joint.encoding_commitment+arithmetic", "value": t_prove, "unit": "s"}),
        json!({"name": "t.serialization", "value": t_ser, "unit": "s"}),
        json!({"name": "t.total", "value": t_total, "unit": "s"}),
        json!({"name": "verifier.seconds", "value": t_verify, "unit": "s"}),
        json!({"name": "transcript.bytes", "value": bytes, "unit": "B"}),
        json!({"name": "rounds.protocol", "value": rounds, "unit": "count"}),
        json!({"name": "rounds.sequential_depth", "value": depth, "unit": "count"}),
        json!({"name": "direct.depth_formula_legacy", "value": rounds, "unit": "count"}),
        json!({"name": "direct.rtt_ms", "value": rtt_ms, "unit": "ms"}),
        json!({"name": "direct.rtt_sleep_seconds", "value": slept, "unit": "s"}),
        json!({"name": "direct.prove_wall_s", "value": t_prove_wall, "unit": "s"}),
        json!({"name": "mem.peak_host_bytes", "value": peak_rss_bytes().unwrap_or(0), "unit": "B"}),
        json!({"name": "census.committed_elements", "value": committed_trace + quotient_cells, "unit": "count"}),
        json!({"name": "census.witness_elements", "value": committed_trace, "unit": "count"}),
        json!({"name": "vu.count", "value": n_vus, "unit": "count"}),
        json!({"name": "vu.seconds_per_vu", "value": t_total / n_vus as f64, "unit": "s"}),
        json!({"name": "vu.per_second", "value": n_vus as f64 / t_total, "unit": "1/s"}),
        json!({"name": "rate.proved_flop_per_second", "value": proved_flop_per_second, "unit": "FLOP/s"}),
        json!({"name": "rate.native_peak_flop_per_second", "value": native_peak_flop_per_second, "unit": "FLOP/s"}),
        json!({"name": "overhead.vs_native_peak", "value": (t_total / n_vus as f64) / native_seconds_per_vu, "unit": "ratio"}),
        json!({"name": "direct.vus", "value": n_vus, "unit": "count"}),
        json!({"name": "direct.rows", "value": rows, "unit": "count"}),
        json!({"name": "direct.rows_real", "value": n_vus * steps, "unit": "count"}),
        json!({"name": "direct.width", "value": width, "unit": "count"}),
        json!({"name": "direct.committed_width", "value": committed_width, "unit": "count"}),
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
    measurements.extend(extra);
    measurements.extend(e2e);
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
                "challenge_field": challenge_name,
                "challenge_bits": challenge_bits,
                "hash": "keccak256",
                "fri_bits_proven_bound": bits_proven,
                "fri_bits_conjectured_bound": bits_conj,
                "queries_for_128_proven": q128,
                "lookup_bits": lookup_bits,
                "note": "proven bound: unique decoding radius, -log2((1+rate)/2) per query + PoW; field terms (DEEP/quotient batching over the challenge field, ~degree/|challenge field|) omitted; lookup_bits = the LogUp identity term at the challenge field (section 14.4), null for the limb realisation"
            },
            "hardware": {"gpu": null, "cpu": cpu_model().map(|m| json!({"model": m, "count": threads}))},
            "software": {"backend": {"name": "verity-direct (plonky3 uni-stark)", "version": "0.3.0", "commit": commit, "plonky3": "0.4.3-succinct"}, "toolchain": {"rustc": option_env!("VERITY_RUSTC").unwrap_or("unknown")}},
            "preprocessing": "none",
            "retention": "retained",
            "unit": "vu",
            "constraints": constraints_note,
            "realisation": realisation,
            "statement_binding": binding.statement_binding,
            "latency": {
                "protocol": "Fiat-Shamir as run (Plonky3); the interactive reading counts every observe -> sample boundary of the prover's transcript as one verifier message; measured with an emulated RTT: the prover sleeps one RTT at each",
                "rtt_ms": rtt_ms,
                "e2e_measured": e2e_measured,
                "e2e_predicted": e2e_predicted,
                "e2e_model": "t.total + verifier.seconds + rounds.sequential_depth x RTT (bytes / bandwidth not included)",
                "depth_note": "rounds.sequential_depth is the measured count of observe -> sample boundaries (verity_direct::rtt); direct.depth_formula_legacy is the earlier derived log_n + 5 (+ the realisation's extra commitment rounds)"
            },
            "note": binding.note
        },
        "validation": {"status": "passed", "checker": binding.checker, "detail": format!("proof verified{}; {n_vus} VUs x {steps} steps in {rows} rows; every intermediate accumulator and output word equals bench-instances/v1", binding.detail_suffix), "evidence": [], "index_note": tier.index.get("note").cloned().unwrap_or(Value::Null)},
        "measurements": measurements,
        "measurement_files": [],
        "artifacts": []
    });
    let path = format!("{out}/result.json");
    fs::write(&path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!(
        "field={field_name} vus={n_vus} rows={rows} width={width} witness={t_witness:.3}s prove={t_prove:.3}s total={t_total:.3}s ({:.4} s/VU) verify={t_verify:.3}s depth={depth} (legacy {rounds}) rtt_ms={rtt_ms} slept={slept:.3}s prove_wall={t_prove_wall:.3}s bytes={bytes} threads={threads} peak_rss={:?} bits_proven={bits_proven:.1} -> {path}",
        t_total / n_vus as f64,
        peak_rss_bytes()
    );
}

/// `check-rows --field F --in jobs.json --out report.json`: evaluate EVERY AIR constraint on supplied
/// (adversarial) traces and report the first violated one -- the verifier's view, independent of witness
/// generation.  Jobs: `{"kind": "rows", "starts_trace": bool, "rows": [RowSpec...]}` (row-local constraints on
/// every row, transition constraints between consecutive rows, boundary constraints when `starts_trace`) or
/// `{"kind": "gadget", "name", "inputs", "hints", "claimed"}` (one gadget in isolation, `program::gadget_program`).
/// A job is `accepted` iff no constraint fails.  Used by `tools/battery_adapter.py` (the red team's 83 forgeries).
fn check_rows<G: PrimeField64 + Send + Sync, EF: p3_field::ExtensionField<G>>(args: &[String]) {
    let kind = field_arg(args);
    let input = arg(args, "--in").expect("--in jobs.json");
    let out = arg(args, "--out").expect("--out report.json");
    let lookup = args.iter().any(|a| a == "--lookup");
    let text = fs::read_to_string(&input).unwrap();
    let jobs: Value = serde_json::from_str(&text).unwrap();
    let (air_, lay_limbs) = ChainAir::new(kind);
    let ext_deg = <EF as p3_field::AbstractExtensionField<G>>::D;
    let (air_l, lay_lookup) = air::LookupChainAir::new(kind, ext_deg);
    let lay = if lookup { &lay_lookup } else { &lay_limbs };
    let idx = if lookup { &air_l.idx } else { &air_.idx };
    let p = kind.prime();
    let y_hi_col = idx.y[0];
    let carry_col = lay.col("cast.carry");
    let cast_hi = if lay.has("cast.hi") { lay.col("cast.hi") } else { lay.col("pack.y_hi") };
    let mut report = vec![];
    let mut n_accepted = 0usize;
    for job in jobs["jobs"].as_array().expect("jobs") {
        let id = job["id"].as_str().unwrap_or("").to_string();
        let res = match job["kind"].as_str().unwrap_or("rows") {
            "gadget" => {
                let case: verity_direct::program::GadgetCase = serde_json::from_value(job.clone()).unwrap();
                let overrides: Vec<(String, i128)> = job["hints"].as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap() as i128)).collect()).unwrap_or_default();
                let (failing, viol) = if lookup { air::gadget_failing_lookup::<G>(kind, &air_l.lay.tables, &case, &overrides) } else { air::gadget_failing::<G>(kind, &case, &overrides) };
                let accepted = failing.is_empty();
                json!({"id": id, "kind": "gadget", "gadget": case.name, "accepted": accepted, "n_failing": failing.len(), "first_failing": failing.first(), "failing": failing.iter().take(8).collect::<Vec<_>>(), "integer_violations": viol.len()})
            }
            _ => {
                let specs: Vec<air::RowSpec> = serde_json::from_value(job["rows"].clone()).unwrap();
                let starts = job["starts_trace"].as_bool().unwrap_or(false);
                let (rows, viol, failing) = if lookup {
                    let (rows, qs, viol) = air::chain_rows_spec_lookup(kind, lay.width, idx, &air_l.lay.tables, &specs);
                    let failing = air::lookup_failing_rows::<G, EF>(&air_l, lay, &rows, &qs, starts);
                    (rows, viol, failing)
                } else {
                    let (rows, viol) = air::chain_rows_spec(kind, lay.width, idx, &specs);
                    let failing = air::chain_failing_rows::<G>(kind, &air_, lay, &rows, starts);
                    (rows, viol, failing)
                };
                let words: Vec<u64> = rows
                    .iter()
                    .map(|r| if idx.y.len() == 1 { modp(r[idx.y[0]], p) } else { (modp(r[idx.y[0]], p) << 16) | modp(r[idx.y[1]], p) })
                    .collect();
                let _ = y_hi_col;
                let y16s: Vec<u64> = rows.iter().map(|r| modp(r[cast_hi] + r[carry_col], p)).collect();
                let accepted = failing.is_empty();
                json!({"id": id, "kind": "rows", "n_rows": rows.len(), "accepted": accepted, "n_failing": failing.len(), "first_failing": failing.first().map(|(r, n)| json!([r, n])), "failing": failing.iter().take(8).map(|(r, n)| json!([r, n])).collect::<Vec<_>>(), "integer_violations": viol.len(), "words": words, "y16": y16s})
            }
        };
        if res["accepted"].as_bool().unwrap() {
            n_accepted += 1;
        }
        report.push(res);
    }
    let summary = json!({"schema": "verity-direct/check-rows/v1", "field": kind.name(), "realisation": if lookup { "logup" } else { "limbs" }, "width": lay.width, "constraints": lay.constraint_names.len(), "jobs": report.len(), "accepted": n_accepted, "results": report});
    fs::write(&out, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("check-rows field={} jobs={} accepted={} -> {out}", kind.name(), summary["jobs"], n_accepted);
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
    // --lookup: the LogUp realisation's rows and constraints (row-local, transition, helper / running-sum at
    // fixed z, beta) instead of the limb realisation's
    let lookup = args.iter().any(|a| a == "--lookup");
    let tier = load_tier(&root, "vu-k1536-neg");
    let (air_, lay_limbs) = ChainAir::new(kind);
    let ext_deg = <<B::SC as StarkGenericConfig>::Challenge as p3_field::AbstractExtensionField<Val<B::SC>>>::D;
    let (air_l, lay_lookup) = air::LookupChainAir::new(kind, ext_deg);
    let lay = if lookup { &lay_lookup } else { &lay_limbs };
    let idx = if lookup { &air_l.idx } else { &air_.idx };
    // (integer violations, failing field constraints) of one VU's honest-generator rows
    let check = |vu: &VuIo| -> (Vec<String>, Vec<(usize, String)>) {
        if lookup {
            let (rows, qs, _accs, viol) = air::chain_rows_lookup(kind, lay.width, idx, &air_l.lay.tables, vu);
            let field_fail = air::lookup_failing_rows::<Val<B::SC>, <B::SC as StarkGenericConfig>::Challenge>(&air_l, lay, &rows, &qs, true);
            (viol, field_fail)
        } else {
            let (rows, _accs, viol) = air::chain_rows(kind, lay.width, idx, vu);
            let field_fail = air::chain_failing_rows::<Val<B::SC>>(kind, &air_, lay, &rows, true);
            (viol, field_fail)
        }
    };
    let rows_index = tier.index["rows"].as_array().unwrap().clone();
    let mut report = vec![];
    let mut soundness_bugs = 0usize;
    for (i, vu) in tier.vus.iter().enumerate() {
        let (viol, field_fail) = check(vu);
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
            let (viol, field_fail) = check(&vu);
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
        "realisation": if lookup { "logup" } else { "limbs" },
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
    println!("vu-neg field={kind:?} realisation={}: {} negatives, {} rejected by witness generation AND unsatisfiable in the field", if lookup { "logup" } else { "limbs" }, tier.vus.len(), tier.vus.len() - soundness_bugs);
    assert_eq!(soundness_bugs, 0, "SOUNDNESS BUG: {soundness_bugs} negatives produce violation-free rows");
}

// ---------------------------------------------------------------------------------------------------------------
// Checker v2 on BabyBear with LogUp (PROTOCOL.md 15, V2_AIR.md)
// ---------------------------------------------------------------------------------------------------------------

use verity_direct::air_v2::{self, V2ChainAir};

fn ext_arg(args: &[String]) -> usize {
    arg(args, "--ext").map(|s| s.parse().unwrap()).unwrap_or(4)
}

fn chain_layout_v2(args: &[String]) {
    let d = ext_arg(args);
    let (air_, lay) = V2ChainAir::new(FieldKind::BabyBear, d);
    let ll = &air_.lay;
    let pre_w = ll.tables.pre_width();
    let aux_w = ll.aux_width(d);
    println!(
        "field=BabyBear air=v2 lookup=logup main_width={} (incl. multiplicity) aux_width={} (helpers={} x{d} + g + S) pre_width={pre_w} (ncols={}) committed_width={} hints={} materialised={} constraints={} queries/row={} table_rows={}",
        lay.width,
        aux_w,
        ll.n_helpers(),
        ll.tables.ncols,
        lay.width + aux_w + pre_w,
        lay.n_hints,
        lay.n_mat,
        lay.constraint_names.len(),
        ll.n_queries(),
        ll.tables.total_rows
    );
    for (n, r) in ll.tables.describe() {
        println!("  table {n:9} rows={r}");
    }
    let mut by_deg = std::collections::BTreeMap::new();
    for d in &ll.query_degs {
        *by_deg.entry(*d).or_insert(0usize) += 1;
    }
    println!("  query degrees: {by_deg:?}");
    let outs = lay.col_names.iter().filter(|n| n.contains(".out[")).count();
    let mats = lay.col_names.iter().filter(|n| *n == "_mat" || n.ends_with(".mat")).count();
    println!("  table-output columns={outs} auto-materialised={mats} other (words, state, hints, named materialisations, position)={}", lay.width - outs - mats);
    if args.iter().any(|a| a == "--names") {
        for (i, n) in lay.col_names.iter().enumerate() {
            println!("col {i:4} {n}");
        }
        for (i, n) in ll.query_names.iter().enumerate() {
            println!("query {i:4} deg={} {n}", ll.query_degs[i]);
        }
        for (i, n) in lay.constraint_names.iter().enumerate() {
            println!("constraint {i:4} {n}");
        }
    }
}

/// `vu-bench --v2 [--ext 4|5] [--vus N] [--queries Q] [--bind-operands] --out DIR`: the v2 gate.  The proof is
/// bound to the statement (every claimed word; the operands too with `--bind-operands`).
fn vu_bench_v2<B: Backend>(args: &[String])
where
    Val<B::SC>: PrimeField64 + Send + Sync + p3_field::TwoAdicField,
    <<B::SC as StarkGenericConfig>::Pcs as p3_commit::Pcs<<B::SC as StarkGenericConfig>::Challenge, <B::SC as StarkGenericConfig>::Challenger>>::Commitment: PartialEq,
{
    use verity_direct::lookup::lookup_soundness_bits_cols;
    use verity_direct::lookup_stark::proof_bytes_lookup;
    let kind = B::KIND;
    assert!(kind.small());
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
    let ext_deg = <<B::SC as StarkGenericConfig>::Challenge as p3_field::AbstractExtensionField<Val<B::SC>>>::D;
    let bind_operands = args.iter().any(|a| a == "--bind-operands");

    let (air_, lay) = V2ChainAir::new(kind, ext_deg);
    let t0 = Instant::now();
    let (main, queries, accs): (RowMajorMatrix<Val<B::SC>>, Vec<u32>, Vec<Vec<u64>>) = air_v2::trace(&air_, vus, 2);
    let rows = main.height();
    let t_witness = t0.elapsed().as_secs_f64();

    let mut acc_mismatch = 0usize;
    for (i, a) in accs.iter().enumerate() {
        if a != &recorded_accs[i] {
            acc_mismatch += 1;
            if acc_mismatch <= 3 {
                let j = a.iter().zip(&recorded_accs[i]).position(|(x, y)| x != y).unwrap();
                eprintln!("vu {i}: step {j}: state word {:#010x} != recorded {:#010x}", a[j], recorded_accs[i][j]);
            }
        }
    }
    assert_eq!(acc_mismatch, 0, "{acc_mismatch} VUs disagree with the recorded accumulators");
    let claim_col = lay.col(verity_direct::program::COL_CLAIM);
    let last_col = lay.col(verity_direct::program::COL_LAST);
    let (y_hi, carry) = (lay.col("pack.y_hi"), lay.col("cast.carry"));
    let steps = STEPS as usize;
    for (i, vu) in vus.iter().enumerate() {
        let r = i * steps + steps - 1;
        let row = main.row_slice(r);
        assert_eq!(row[last_col].as_canonical_u64(), 1);
        assert_eq!(row[claim_col].as_canonical_u64(), vu.y16, "vu {i}: claim column");
        assert_eq!(row[y_hi].as_canonical_u64() + row[carry].as_canonical_u64(), vu.y16, "vu {i}: epilogue word != recorded word");
    }
    let words: Vec<u64> = vus.iter().map(|v| v.y16).collect();
    let stmt = if bind_operands {
        Statement::with_operands(words, vus.iter().flat_map(|v| v.x.iter().copied()).collect(), vus.iter().flat_map(|v| v.w.iter().copied()).collect())
    } else {
        Statement::new(words)
    };

    let width = main.width();
    let aux_w = air_.lay.aux_width(ext_deg);
    let pre_w = air_.lay.tables.pre_width();
    let committed_width = width + aux_w + pre_w;
    let lqd = p3_util::log2_ceil_usize(verity_direct::program::MAX_DEG - 1);
    assert!(lqd <= fri.log_blowup);

    // rtt::begin must precede the prover's challenger construction so its RttChallenger is the active (counting) one.
    let rtt_ms: f64 = arg(args, "--rtt-ms").or_else(|| std::env::var("VERITY_RTT_MS").ok()).map(|s| s.parse().expect("--rtt-ms takes milliseconds")).unwrap_or(0.0);
    let t1 = Instant::now();
    verity_direct::rtt::begin(rtt_ms);
    let (proof, stage) = air_v2::prove_bound::<B>(&air_, &lay, main, &queries, fri, &stmt);
    let (trips, slept) = verity_direct::rtt::end();
    let t_prove_wall = t1.elapsed().as_secs_f64();
    let t_prove = t_prove_wall - slept; // compute: the wall time less the emulated waits for the verifier's coins
    let t2 = Instant::now();
    let bytes = proof_bytes_lookup(&proof);
    let t_ser = t2.elapsed().as_secs_f64();
    let t3 = Instant::now();
    let ok = air_v2::verify_bound::<B>(&air_, &lay, &proof, fri, &stmt);
    let t_verify = t3.elapsed().as_secs_f64();
    assert!(ok.is_ok(), "verification failed: {:?}", ok.err());
    // statement binding: a proof of these words is not a proof of other words
    let mut other = stmt.clone();
    other.words[0] ^= 1;
    assert!(air_v2::verify_bound::<B>(&air_, &lay, &proof, fri, &other).is_err(), "a proof verified against a different statement");
    let lookup_bits = lookup_soundness_bits_cols(rows, air_.lay.n_queries(), air_.lay.tables.total_rows, B::challenge_bits() as f64, air_.lay.tables.ncols);
    println!(
        "lookup stages: commit_pre={:.2}s commit_main={:.2}s aux_build={:.2}s commit_aux={:.2}s quotient={:.2}s commit_quotient={:.2}s open={:.2}s; main_width={width} aux_width={aux_w} pre_width={pre_w} helpers={} queries/row={} table_rows={} lookup_bits={lookup_bits:.1}",
        stage.commit_pre,
        stage.commit_main,
        stage.aux_build,
        stage.commit_aux,
        stage.quotient,
        stage.commit_quotient,
        stage.open,
        air_.lay.n_helpers(),
        air_.lay.n_queries(),
        air_.lay.tables.total_rows
    );
    let extra = vec![
        json!({"name": "direct.lookup.main_width", "value": width, "unit": "count"}),
        json!({"name": "direct.lookup.aux_width", "value": aux_w, "unit": "count"}),
        json!({"name": "direct.lookup.pre_width", "value": pre_w, "unit": "count"}),
        json!({"name": "direct.lookup.helpers", "value": air_.lay.n_helpers(), "unit": "count"}),
        json!({"name": "direct.lookup.queries_per_row", "value": air_.lay.n_queries(), "unit": "count"}),
        json!({"name": "direct.lookup.table_rows", "value": air_.lay.tables.total_rows, "unit": "count"}),
        json!({"name": "direct.lookup.bits", "value": lookup_bits, "unit": "bits"}),
        json!({"name": "direct.stage.commit_pre_s", "value": stage.commit_pre, "unit": "s"}),
        json!({"name": "direct.stage.commit_main_s", "value": stage.commit_main, "unit": "s"}),
        json!({"name": "direct.stage.aux_build_s", "value": stage.aux_build, "unit": "s"}),
        json!({"name": "direct.stage.commit_aux_s", "value": stage.commit_aux, "unit": "s"}),
        json!({"name": "direct.stage.quotient_s", "value": stage.quotient, "unit": "s"}),
        json!({"name": "direct.stage.commit_quotient_s", "value": stage.commit_quotient, "unit": "s"}),
        json!({"name": "direct.stage.open_s", "value": stage.open, "unit": "s"}),
        json!({"name": "direct.statement.bound_operands", "value": bind_operands as u64, "unit": "count"}),
    ];
    write_result(BenchNumbers {
        kind,
        out: &out,
        fri,
        n_vus,
        rows,
        width,
        committed_width,
        t_witness,
        t_prove,
        t_ser,
        t_verify,
        bytes,
        lqd,
        tier: &tier,
        root: &root,
        challenge_name: B::challenge_name(),
        challenge_bits: B::challenge_bits(),
        rounds_extra: 2,
        constraints_note: "checker v2 (checker_min.md) as direct AIR constraints on BabyBear: one row per transition unit, the state (s, e_hat, M) carried by a transition constraint, tuple LogUp lookups (T_OP, ALIGN4, chunked SHIFT/SSHIFT, LEAD10, LEADNORM, TNORM10, range tables) against preprocessed tables, attained as a product identity, the FP32 word as two 16-bit limbs; statement binding: header absorbed before the first challenge, degree_bits fixed by the statement, claim column checked at zeta (PROTOCOL.md 15, V2_AIR.md)",
        realisation: "logup-v2",
        extra,
        lookup_bits: Some(lookup_bits),
        ext_deg,
        rtt_ms,
        depth: trips as usize,
        slept,
        t_prove_wall,
        binding: binding_json(false, bind_operands, "vu.y16 (claim column of the v2 main matrix, round 1)", 1, rows, B::challenge_bits(), &["(v2 domain tag verity-direct/chain-air-v2/statement/v1; tables bound by the verifier's recomputed preprocessed commitment)"]),
    });
}

/// `vu-neg --v2`: every `vu-k1536-neg` instance must be rejected by witness generation AND violate a field
/// constraint (row-local evaluation of the honest-hint rows with the claimed word); every `wrong` negative with its
/// correct word is accepted (control).
fn vu_neg_v2(args: &[String]) {
    let kind = FieldKind::BabyBear;
    let root = Path::new(&arg(args, "--root").unwrap_or_else(|| "fixtures/bench-instances/v1".into())).to_path_buf();
    let out = arg(args, "--out").or_else(|| std::env::var("RESEARCH_RUN_DIR").ok());
    let tier = load_tier(&root, "vu-k1536-neg");
    let (air_, lay) = V2ChainAir::new(kind, 4);
    let rows_index = tier.index["rows"].as_array().unwrap().clone();
    let mut report = vec![];
    let mut soundness_bugs = 0usize;
    let eval = |vu: &VuIo| {
        let (rows, qs, _, viol) = air_v2::vu_rows(kind, lay.width, &air_.lay.tables, vu);
        let fail = air_v2::failing_rows::<stark::Val32, stark::ChallengeBb>(&air_, &lay, &rows, &qs, true);
        (viol, fail)
    };
    for (i, vu) in tier.vus.iter().enumerate() {
        let (viol, field_fail) = eval(vu);
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
    let mut n_control = 0usize;
    for (i, row) in rows_index.iter().enumerate() {
        if let Some(correct) = row[4].as_u64() {
            let mut vu = tier.vus[i].clone();
            vu.y16 = correct;
            let (viol, field_fail) = eval(&vu);
            assert!(viol.is_empty() && field_fail.is_empty(), "control {i} failed: the correct word is rejected: {:?} {:?}", viol.first(), field_fail.first());
            n_control += 1;
        }
    }
    println!("controls: {n_control} negatives with their correct word: accepted (no integer violation, no failing field constraint or lookup)");
    let summary = json!({
        "schema": "verity-direct/vu-neg/v2",
        "field": "BabyBear", "air": "v2",
        "dataset": {"dataset": "bench-instances/v1", "tier": "vu-k1536-neg", "manifest_sha256": manifest_sha256(&root), "n": tier.vus.len()},
        "rejected_all": soundness_bugs == 0,
        "soundness_bugs": soundness_bugs,
        "controls": {"n": n_control, "accepted": n_control},
        "instances": report,
        "run_id": std::env::var("RESEARCH_RUN_ID").ok(),
    });
    if let Some(out) = out {
        fs::create_dir_all(&out).unwrap();
        fs::write(format!("{out}/vu_neg_v2.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    }
    println!("vu-neg --v2: {} negatives, {} rejected by witness generation AND unsatisfiable in the field", tier.vus.len(), tier.vus.len() - soundness_bugs);
    assert_eq!(soundness_bugs, 0, "SOUNDNESS BUG: {soundness_bugs} negatives produce violation-free rows");
}

/// `vu-honest --v2 [--vus N] [--eval M]`: the honest witness of the bench tier agrees with the recorded
/// accumulators and (for the first `M` VUs) satisfies every row-local field constraint and every lookup.
fn vu_honest_v2(args: &[String]) {
    use p3_maybe_rayon::prelude::*;
    let kind = FieldKind::BabyBear;
    let root = Path::new(&arg(args, "--root").unwrap_or_else(|| "fixtures/bench-instances/v1".into())).to_path_buf();
    let tier = load_tier(&root, "vu-k1536");
    let n: usize = arg(args, "--vus").map(|s| s.parse().unwrap()).unwrap_or(tier.vus.len());
    let m: usize = arg(args, "--eval").map(|s| s.parse().unwrap()).unwrap_or(8.min(n));
    let (air_, lay) = V2ChainAir::new(kind, 4);
    let (y_hi, carry) = (lay.col("pack.y_hi"), lay.col("cast.carry"));
    let recorded = tier.accs.as_ref().unwrap();
    let t0 = Instant::now();
    let results: Vec<(usize, usize, usize, Vec<String>)> = tier.vus[..n]
        .par_iter()
        .enumerate()
        .map(|(i, vu)| {
            let (rows, qs, states, viol) = air_v2::vu_rows(kind, lay.width, &air_.lay.tables, vu);
            let words: Vec<u64> = rows.iter().map(|r| air_v2::row_word(&air_.idx, r, kind)).collect();
            let mism = words.iter().zip(&recorded[i]).filter(|(a, b)| a != b).count();
            if mism > 0 {
                for (j, (a, b)) in words.iter().zip(&recorded[i]).enumerate() {
                    if a != b {
                        eprintln!("vu {i} step {j}: state {:?} -> word {a:#010x} != recorded {b:#010x}", states[j]);
                    }
                }
            }
            let last = &rows[rows.len() - 1];
            let y = modp(last[y_hi] + last[carry], kind.prime());
            let ybad = (y != vu.y16) as usize;
            let mut fails = vec![];
            if i < m {
                fails = air_v2::failing_rows::<stark::Val32, stark::ChallengeBb>(&air_, &lay, &rows, &qs, true).into_iter().map(|(r, n)| format!("row {r} {n}")).collect();
            }
            (viol.len(), mism, ybad, fails)
        })
        .collect();
    let t = t0.elapsed().as_secs_f64();
    let (mut v, mut mm, mut yb, mut ff) = (0, 0, 0, 0);
    for (i, (a, b, c, f)) in results.iter().enumerate() {
        v += a;
        mm += b;
        yb += c;
        ff += f.len();
        if *a > 0 || *b > 0 || *c > 0 || !f.is_empty() {
            println!("vu {i}: violations={a} acc_mismatch_steps={b} word_bad={c} field_fails={} {:?}", f.len(), &f[..f.len().min(4)]);
        }
    }
    println!("vu-honest --v2: {n} VUs in {t:.2}s ({:.2} ms/VU witness): integer violations={v} accumulator mismatches={mm} bad words={yb} field failures (first {m} VUs)={ff}", 1e3 * t / n as f64);
    if !args.iter().any(|a| a == "--no-assert") {
        assert!(v == 0 && mm == 0 && yb == 0 && ff == 0);
    }
}

/// `check-rows --v2 --in jobs.json --out report.json`: the battery on the v2 AIR (`tools/battery_adapter.py --v2`).
fn check_rows_v2<G: PrimeField64 + Send + Sync, EF: p3_field::ExtensionField<G>>(args: &[String]) {
    let kind = FieldKind::BabyBear;
    let input = arg(args, "--in").expect("--in jobs.json");
    let out = arg(args, "--out").expect("--out report.json");
    let text = fs::read_to_string(&input).unwrap();
    let jobs: Value = serde_json::from_str(&text).unwrap();
    let ext_deg = <EF as p3_field::AbstractExtensionField<G>>::D;
    let (air_, lay) = V2ChainAir::new(kind, ext_deg);
    let p = kind.prime();
    let carry_col = lay.col("cast.carry");
    let y_hi = lay.col("pack.y_hi");
    let mut report = vec![];
    let mut n_accepted = 0usize;
    for job in jobs["jobs"].as_array().expect("jobs") {
        let id = job["id"].as_str().unwrap_or("").to_string();
        let res = match job["kind"].as_str().unwrap_or("rows") {
            "gadget" => {
                let mut case: verity_direct::program::GadgetCase = serde_json::from_value(job.clone()).unwrap();
                let honest = air_v2::gadget_honest_outputs(kind, &case);
                if job["v2_honest_claim"].as_bool().unwrap_or(false) {
                    case.claimed = honest.clone();
                }
                let overrides: Vec<(String, i128)> = job["hints"].as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap() as i128)).collect()).unwrap_or_default();
                let (failing, viol) = air_v2::gadget_failing::<G>(kind, &case, &overrides);
                let accepted = failing.is_empty();
                json!({"id": id, "kind": "gadget", "gadget": case.name, "accepted": accepted, "n_failing": failing.len(), "first_failing": failing.first(), "failing": failing.iter().take(8).collect::<Vec<_>>(), "integer_violations": viol.len(), "claimed": case.claimed, "honest_v2": honest})
            }
            _ => {
                let specs: Vec<air::RowSpec> = serde_json::from_value(job["rows"].clone()).unwrap();
                let starts = job["starts_trace"].as_bool().unwrap_or(false);
                let (rows, qs, viol) = air_v2::rows_spec(kind, lay.width, &air_.lay.tables, &specs);
                let failing = air_v2::failing_rows::<G, EF>(&air_, &lay, &rows, &qs, starts);
                let words: Vec<u64> = rows.iter().map(|r| air_v2::row_word(&air_.idx, r, kind)).collect();
                let y16s: Vec<u64> = rows.iter().map(|r| modp(r[y_hi] + r[carry_col], p)).collect();
                let accepted = failing.is_empty();
                json!({"id": id, "kind": "rows", "n_rows": rows.len(), "accepted": accepted, "n_failing": failing.len(), "first_failing": failing.first().map(|(r, n)| json!([r, n])), "failing": failing.iter().take(8).map(|(r, n)| json!([r, n])).collect::<Vec<_>>(), "integer_violations": viol.len(), "words": words, "y16": y16s})
            }
        };
        if res["accepted"].as_bool().unwrap() {
            n_accepted += 1;
        }
        report.push(res);
    }
    let summary = json!({"schema": "verity-direct/check-rows/v2", "field": kind.name(), "realisation": "logup-v2", "width": lay.width, "constraints": lay.constraint_names.len(), "jobs": report.len(), "accepted": n_accepted, "results": report});
    fs::write(&out, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("check-rows --v2 jobs={} accepted={} -> {out}", summary["jobs"], n_accepted);
}
