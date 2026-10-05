//! Red-team 3 battery against the B-Ligero LogUp realisation (`verity-direct`, PROTOCOL.md section 14):
//! adversarial `(main, aux)` pairs built with the lane's own witness generator and *adversarially chosen*
//! auxiliary columns (the helper of an out-of-table query carries the term the honest prover cannot compute,
//! `g` follows a tampered multiplicity, the running sum is re-accumulated), run through the real
//! `prove_lookup` / `verify_lookup` over BabyBear (BabyBear^4 challenges), with the full-trace constraint
//! evaluator naming what fires.  Also the two statement-binding forgeries of wave 2 re-run on the three-round
//! fork (no public values, `degree_bits` read from the proof).
//!
//!     redteam-logup-b battery --root fixtures/bench-instances/v1 --out DIR [--queries 8]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, Field, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use serde_json::{json, Value};
use verity_direct::air::{self, LookupChainAir, VuIo};
use verity_direct::ctx::FieldKind;
use verity_direct::lookup::{aux_matrix, lookup_soundness_bits, LookupLayout, TableSet, INVALID, N_COLS};
use verity_direct::lookup_stark::{prove_lookup, verify_lookup, LookupProof};
use verity_direct::program::STEPS;
use verity_direct::stark::{BabyBearBackend, Backend, FriParams, Val32};

type B = BabyBearBackend;
type SC = <B as Backend>::SC;
type EF4 = BinomialExtensionField<Val32, 4>;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn read_u16(p: &Path) -> Vec<u64> {
    std::fs::read(p).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u64).collect()
}

/// The first `n` bases of `vu-k1536-neg` with their correct words (`correct_y` of the index).
fn real_vus(root: &Path, n: usize) -> Vec<VuIo> {
    let k = 16 * STEPS as usize;
    let x = read_u16(&root.join("vu-k1536-neg.x.u16"));
    let w = read_u16(&root.join("vu-k1536-neg.w.u16"));
    let idx: Value = serde_json::from_str(&std::fs::read_to_string(root.join("vu-k1536-neg.index.json")).unwrap()).unwrap();
    let cols: Vec<String> = idx["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    let cy = cols.iter().position(|c| c == "correct_y").expect("correct_y column");
    let mut out = vec![];
    for (i, row) in idx["rows"].as_array().unwrap().iter().enumerate() {
        if out.len() == n {
            break;
        }
        if let Some(y) = row[cy].as_u64() {
            out.push(VuIo { x: x[i * k..(i + 1) * k].to_vec(), w: w[i * k..(i + 1) * k].to_vec(), y16: y });
        }
    }
    assert_eq!(out.len(), n, "not enough real VUs with a correct word");
    out
}

/// Row `r`, extension element `j` of an aux matrix with `d` base columns per element.
fn ext_at(aux: &RowMajorMatrix<Val32>, r: usize, j: usize, d: usize) -> EF4 {
    let w = aux.width();
    EF4::from_base_slice(&aux.values[r * w + j * d..r * w + (j + 1) * d])
}

fn set_ext(aux: &mut RowMajorMatrix<Val32>, r: usize, j: usize, d: usize, v: EF4) {
    let w = aux.width();
    aux.values[r * w + j * d..r * w + (j + 1) * d].copy_from_slice(v.as_base_slice());
}

/// Re-accumulate the running sum `S` from the (possibly patched) helpers and `g`.
fn reaccumulate(aux: &mut RowMajorMatrix<Val32>, lay: &LookupLayout, d: usize) {
    let h = lay.n_helpers();
    let mut s = EF4::zero();
    for r in 0..aux.height() {
        let mut row_sum = EF4::zero();
        for j in 0..h {
            row_sum += ext_at(aux, r, j, d);
        }
        row_sum -= ext_at(aux, r, h, d);
        s += row_sum;
        set_ext(aux, r, h + 1, d, s);
    }
}

/// The combined query value `v + beta^5 tag` of a bare range query on table `tag` (key only).
fn bare_query_value(v: u64, tag: u32, beta: EF4) -> EF4 {
    EF4::from_canonical_u64(v) + beta.exp_u64(N_COLS as u64) * Val32::from_canonical_u32(tag)
}

struct Case {
    name: &'static str,
    class: &'static str,
    expected: &'static str, // "accept" | "reject"
    mechanism: &'static str,
}

struct Outcome {
    verdict: &'static str,
    error: Option<String>,
    failing: Vec<(usize, String)>,
    prove_s: f64,
    verify_s: f64,
    degree_bits: usize,
}

/// An adversarial proof: `main` and `aux_fn(z, beta) -> aux` are the prover's; the verifier is configured for
/// `verifier_log_n` and hands `verify_lookup` a preprocessed matrix of `verifier_pre_rows` rows.
fn attempt(
    air_: &LookupChainAir,
    lay: &air::Layout,
    main: RowMajorMatrix<Val32>,
    pre: RowMajorMatrix<Val32>,
    fri: FriParams,
    aux_fn: impl FnOnce(EF4, EF4) -> RowMajorMatrix<Val32>,
    verifier_log_n: Option<usize>,
    verifier_pre: Option<RowMajorMatrix<Val32>>,
    verifier_tables: Option<&TableSet>,
    diagnose: bool,
) -> Outcome {
    let rows = main.height();
    let log_n = rows.trailing_zeros() as usize;
    let config = B::config(log_n, fri);
    // keep an adversarial (main, aux) copy for the evaluator: aux depends on (z, beta), so capture them
    let captured: std::cell::RefCell<Option<(EF4, EF4, RowMajorMatrix<Val32>)>> = std::cell::RefCell::new(None);
    let main_copy = if diagnose { Some(main.clone()) } else { None };
    let pre_copy = pre.clone();
    let t0 = Instant::now();
    let mut ch = B::challenger();
    let (proof, _): (LookupProof<SC>, _) = prove_lookup(&config, air_, &mut ch, pre, main, |z, beta| {
        let a = aux_fn(z, beta);
        if diagnose {
            *captured.borrow_mut() = Some((z, beta, a.clone()));
        }
        a
    });
    let prove_s = t0.elapsed().as_secs_f64();
    let vcfg = verifier_log_n.map(|l| B::config(l, fri));
    let vpre = match (verifier_pre, verifier_tables) {
        (Some(p), _) => p,
        (None, Some(t)) => t.preprocessed::<Val32>(1 << proof.degree_bits),
        (None, None) => pre_copy,
    };
    let t1 = Instant::now();
    let mut ch = B::challenger();
    let res = match &vcfg {
        Some(c) => verify_lookup(c, air_, &mut ch, &proof, vpre),
        None => verify_lookup(&config, air_, &mut ch, &proof, vpre),
    };
    let verify_s = t1.elapsed().as_secs_f64();
    let failing = match (diagnose, captured.into_inner(), main_copy) {
        (true, Some((z, beta, aux)), Some(m)) => {
            let pre_d = air_.lay.tables.preprocessed::<Val32>(m.height());
            let mut f = air::lookup_failing_trace::<Val32, EF4>(air_, lay, &m, &pre_d, &aux, z, beta);
            f.truncate(12);
            f
        }
        _ => vec![],
    };
    Outcome {
        verdict: if res.is_ok() { "ACCEPTED" } else { "REJECTED" },
        error: res.err().map(|e| format!("{e:?}")),
        failing,
        prove_s,
        verify_s,
        degree_bits: proof.degree_bits,
    }
}

fn record(out: &mut Vec<Value>, c: &Case, o: Outcome, extra: Value) {
    let forged = o.verdict == "ACCEPTED" && c.expected == "reject";
    let v = json!({
        "case": c.name, "class": c.class, "expected": c.expected, "mechanism": c.mechanism,
        "verdict": o.verdict, "forged": forged, "error": o.error,
        "evaluator_failing": o.failing.iter().map(|(r, n)| json!([r, n])).collect::<Vec<_>>(),
        "prove_s": o.prove_s, "verify_s": o.verify_s, "proof_degree_bits": o.degree_bits, "extra": extra,
    });
    println!("{}", v);
    out.push(v);
}

fn battery(root: &Path, out_dir: &Path, fri: FriParams) -> i32 {
    std::fs::create_dir_all(out_dir).unwrap();
    let kind = FieldKind::BabyBear;
    let (air_, lay) = LookupChainAir::new(kind, 4);
    let vus = real_vus(root, 2);
    let (main, queries, _accs): (RowMajorMatrix<Val32>, Vec<u32>, _) = air::chain_trace_lookup(&air_, &vus, 2);
    let rows = main.height();
    let width = main.width();
    let mc = air_.mult_col();
    let pre: RowMajorMatrix<Val32> = air_.lay.tables.preprocessed(rows);
    let llay = air_.lay.clone();
    let tables = air_.lay.tables.clone();
    let mult = |m: &RowMajorMatrix<Val32>| -> Vec<Val32> { (0..rows).map(|r| m.values[r * width + mc]).collect() };
    let mut results: Vec<Value> = vec![];
    let n_q = air_.lay.n_queries();
    println!("rows={rows} main_width={width} aux_width={} pre_width=6 queries/row={n_q} table_rows={} fri={:?}", air_.lay.aux_width(4), tables.total_rows, (fri.log_blowup, fri.num_queries, fri.proof_of_work_bits));

    // ---- 0. honest control ------------------------------------------------------------------------------------
    {
        let c = Case { name: "honest", class: "control", expected: "accept", mechanism: "-" };
        let m = mult(&main);
        let q = queries.clone();
        let l = llay.clone();
        let o = attempt(&air_, &lay, main.clone(), pre.clone(), fri, move |z, b| aux_matrix(&l, &q, &m, rows, z, b, true), None, None, None, true);
        record(&mut results, &c, o, json!({}));
    }

    // ---- 1. F1 class: no statement -----------------------------------------------------------------------------
    {
        let c = Case { name: "zero_batch_as_any_statement", class: "statement_binding", expected: "reject", mechanism: "verify_lookup takes no public values: a proof of two zero VUs is a proof of the two real VUs (of any x, w, words)" };
        let zeros = vec![air::zero_vu(), air::zero_vu()];
        let (zmain, zq, _): (RowMajorMatrix<Val32>, Vec<u32>, _) = air::chain_trace_lookup(&air_, &zeros, 2);
        let m = mult(&zmain);
        let l = llay.clone();
        let o = attempt(&air_, &lay, zmain, pre.clone(), fri, move |z, b| aux_matrix(&l, &zq, &m, rows, z, b, true), None, None, None, false);
        record(&mut results, &c, o, json!({"verifier_statement": {"x_w": "the two real VUs of vu-k1536-neg", "words": [vus[0].y16, vus[1].y16]}, "prover_trace": "two zero VUs (a = b = 0, y16 = 0)"}));
    }

    // ---- 2. F2 class: degree_bits from the proof ------------------------------------------------------------------
    {
        let c = Case { name: "degree_bits_from_proof", class: "statement_binding", expected: "reject", mechanism: "verifier configured for log_n = 19 (4096 VUs); the 2^17-row proof supplies degree_bits = 17 and the verifier builds the public table matrix at 1 << proof.degree_bits" };
        let m = mult(&main);
        let q = queries.clone();
        let l = llay.clone();
        let o = attempt(&air_, &lay, main.clone(), pre.clone(), fri, move |z, b| aux_matrix(&l, &q, &m, rows, z, b, true), Some(19), None, Some(&tables), false);
        record(&mut results, &c, o, json!({"verifier_log_n": 19, "verifier_pre_rows": 1usize << 17}));
        let c = Case { name: "degree_bits_pinned_by_pre_height", class: "statement_binding", expected: "reject", mechanism: "same, but the verifier hands verify_lookup the 2^19-row table matrix of its own configuration (the height check rejects)" };
        let m = mult(&main);
        let q = queries.clone();
        let l = llay.clone();
        let o = attempt(&air_, &lay, main.clone(), pre.clone(), fri, move |z, b| aux_matrix(&l, &q, &m, rows, z, b, true), Some(19), Some(tables.preprocessed(1 << 19)), None, false);
        record(&mut results, &c, o, json!({"verifier_log_n": 19, "verifier_pre_rows": 1usize << 19}));
    }

    // ---- 3. out-of-table query with an adversarial helper ---------------------------------------------------------
    let t16 = tables.range(16);
    let tag16 = t16.tag;
    let qi = air_.lay.query_names.iter().position(|n| n == "x.c_lo").expect("query x.c_lo");
    assert_eq!(air_.lay.query_degs[qi], 1);
    let grp = air_.lay.groups.iter().position(|g| g.contains(&qi)).unwrap();
    let col = lay.col("x.c_lo");
    let oot_row = 2usize;
    let oot_val = 70000u64;
    let forged_main = |extra_mult: Option<(usize, i64)>| -> (RowMajorMatrix<Val32>, Vec<u32>) {
        let mut m = main.clone();
        m.values[oot_row * width + col] = Val32::from_canonical_u64(oot_val);
        let mut q = queries.clone();
        let old = q[oot_row * n_q + qi];
        assert_ne!(old, INVALID);
        q[oot_row * n_q + qi] = INVALID;
        // the honest histogram loses the old query's count
        let ov = m.values[old as usize * width + mc].as_canonical_u64();
        m.values[old as usize * width + mc] = Val32::from_canonical_u64(ov - 1);
        if let Some((r, dm)) = extra_mult {
            let v = m.values[r * width + mc];
            m.values[r * width + mc] = if dm >= 0 { v + Val32::from_canonical_u64(dm as u64) } else { v - Val32::from_canonical_u64((-dm) as u64) };
        }
        (m, q)
    };
    for (name, extra, mech) in [
        ("oot_query_adversarial_helper", None, "x.c_lo := 70000 on row 2 (outside T16); the helper of its group carries 1/(z - q*) exactly, g and S re-accumulated: only the LogUp identity (sum_last) can fire"),
        ("oot_query_mult_credited", Some((t16.offset + 65535, 1i64)), "as above plus the multiplicity of table row 65535 raised by one (histogram total preserved)"),
        ("oot_query_mult_negative", Some((t16.offset + 65535, -1i64)), "as above with the multiplicity of table row 65535 lowered by one (a p-1 wrap where it was zero)"),
    ] {
        let c = Case { name, class: "logup_multiplicity", expected: "reject", mechanism: mech };
        let (fm, fq) = forged_main(extra);
        let m = mult(&fm);
        let l = llay.clone();
        let o = attempt(&air_, &lay, fm, pre.clone(), fri, move |z, b| {
            let mut a = aux_matrix(&l, &fq, &m, rows, z, b, false);
            let h = ext_at(&a, oot_row, grp, 4) + (z - bare_query_value(oot_val, tag16, b)).inverse();
            set_ext(&mut a, oot_row, grp, 4, h);
            reaccumulate(&mut a, &l, 4);
            a
        }, None, None, None, true);
        record(&mut results, &c, o, json!({"row": oot_row, "column": "x.c_lo", "value": oot_val, "helper_group": grp, "extra_mult": extra}));
    }

    // ---- 4. multiplicity on a padding table row --------------------------------------------------------------------
    {
        let c = Case { name: "padding_row_multiplicity", class: "logup_multiplicity", expected: "reject", mechanism: "m := 1 on the first padding row of P (tuple (0,..,0; tag 0)); g := 1/z there; S re-accumulated" };
        let pad_row = tables.total_rows;
        let mut fm = main.clone();
        fm.values[pad_row * width + mc] = Val32::one();
        let m = mult(&fm);
        let q = queries.clone();
        let l = llay.clone();
        let o = attempt(&air_, &lay, fm, pre.clone(), fri, move |z, b| {
            let mut a = aux_matrix(&l, &q, &m, rows, z, b, false);
            let h = l.n_helpers();
            set_ext(&mut a, pad_row, h, 4, z.inverse());
            reaccumulate(&mut a, &l, 4);
            a
        }, None, None, None, true);
        record(&mut results, &c, o, json!({"pad_row": pad_row}));
    }

    // ---- 5. table substitution: the prover's T16 has 65537 rows --------------------------------------------------------
    {
        let c = Case { name: "table_row_substitution", class: "table_binding", expected: "reject", mechanism: "prover commits a table matrix whose T16 has a 65537th row (65536) and queries it; the verifier recomputes P from the public table set and compares the commitment" };
        let mut ts: TableSet = (*tables).clone();
        let i16 = ts.tables.iter().position(|t| t.name == "T16").unwrap();
        ts.tables[i16].rows += 1;
        for t in ts.tables.iter_mut().skip(i16 + 1) {
            t.offset += 1;
        }
        ts.total_rows += 1;
        let ts = Arc::new(ts);
        let mut air2 = LookupChainAir::new(kind, 4).0;
        air2.lay.tables = ts.clone();
        let (mut fm, fq, _): (RowMajorMatrix<Val32>, Vec<u32>, _) = air::chain_trace_lookup(&air2, &vus, 2);
        // put 65536 into x.c_lo on row 2 and point its query at the new row (all honest for the prover's table)
        let mut fq = fq;
        let old = fq[oot_row * n_q + qi] as usize;
        let ov = fm.values[old * width + mc].as_canonical_u64();
        fm.values[old * width + mc] = Val32::from_canonical_u64(ov - 1);
        fm.values[oot_row * width + col] = Val32::from_canonical_u64(65536);
        let newrow = ts.tables[i16].offset + 65536;
        fq[oot_row * n_q + qi] = newrow as u32;
        let nv = fm.values[newrow * width + mc].as_canonical_u64();
        fm.values[newrow * width + mc] = Val32::from_canonical_u64(nv + 1);
        let pre2: RowMajorMatrix<Val32> = ts.preprocessed(rows);
        let m = mult(&fm);
        let l2 = air2.lay.clone();
        let o = attempt(&air2, &lay, fm, pre2, fri, move |z, b| aux_matrix(&l2, &fq, &m, rows, z, b, true), None, Some(pre.clone()), None, false);
        record(&mut results, &c, o, json!({"prover_table_rows": ts.total_rows, "verifier_table_rows": tables.total_rows}));
    }

    // ---- 6. the check-rows battery path on the adversarial row (coverage, not a forgery) --------------------------------
    let coverage = {
        let (rows_i, qs, _, viol) = air::chain_rows_lookup(kind, air_.width, &air_.idx, &tables, &vus[0]);
        assert!(viol.is_empty());
        let mut rows_i = rows_i;
        let mut qs = qs;
        rows_i[oot_row][col] = oot_val as i128;
        qs[oot_row][qi] = INVALID;
        let f = air::lookup_failing_rows::<Val32, EF4>(&air_, &lay, &rows_i, &qs, true);
        json!({"check_rows_failing": f.iter().map(|(r, n)| json!([r, n])).collect::<Vec<_>>(),
               "note": "check-rows evaluates program + helper constraints with the honest prover's helper (no term for the out-of-table query); the adversarial helper above satisfies lookup.helper exactly and the constraint that actually rejects is lookup.sum_last, which check-rows does not evaluate"})
    };

    let bits = lookup_soundness_bits(1 << 19, n_q, tables.total_rows, 124.0);
    let forged: Vec<&Value> = results.iter().filter(|r| r["forged"].as_bool() == Some(true)).collect();
    let summary = json!({
        "schema": "redteam-3/logup-b/v1", "field": "BabyBear", "challenge": "BabyBear^4", "rows": rows, "vus": 2,
        "fri": {"log_blowup": fri.log_blowup, "num_queries": fri.num_queries, "proof_of_work_bits": fri.proof_of_work_bits,
                "note": "FRI parameters do not enter the algebraic forgeries below; reduced for time"},
        "layout": {"main_width": width, "aux_width": air_.lay.aux_width(4), "queries_per_row": n_q, "helpers": air_.lay.n_helpers(), "table_rows": tables.total_rows},
        "lookup_bits_code_formula_2e19_bb4": bits,
        "coverage": coverage,
        "forged": forged.iter().map(|r| r["case"].clone()).collect::<Vec<_>>(),
        "results": results,
    });
    std::fs::write(out_dir.join("logup_b_battery.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("forged: {:?}", summary["forged"]);
    if forged.is_empty() { 0 } else { 1 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(arg(&args, "--root").unwrap_or_else(|| "fixtures/bench-instances/v1".into()));
    let out = PathBuf::from(arg(&args, "--out").or_else(|| std::env::var("RESEARCH_RUN_DIR").ok()).unwrap_or_else(|| ".".into()));
    let queries: usize = arg(&args, "--queries").map(|s| s.parse().unwrap()).unwrap_or(8);
    let fri = FriParams { log_blowup: 2, num_queries: queries, proof_of_work_bits: 1 };
    match args.get(1).map(String::as_str) {
        Some("battery") => std::process::exit(battery(&root, &out, fri)),
        _ => {
            eprintln!("usage: redteam-logup-b battery --root fixtures/bench-instances/v1 --out DIR [--queries 8]");
            std::process::exit(2);
        }
    }
}
