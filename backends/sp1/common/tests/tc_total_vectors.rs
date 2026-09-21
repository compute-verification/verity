//! The total tensor-core family `veritor.tensor-core.ampere_bf16_m16n8k16@3`
//! (lane L6, `docs/verified-backend/tc-total-semantics.md`) against the
//! cross-implementation vectors of `lean/vectors/`:
//!
//! * `tc-ampere-bf16-total.json`: `evaluations` written by
//!   `lean/scripts/export_vectors.py --family tc-total` from the Python total
//!   model (`veritor.core.silicon_total`) -- finite, special-encoding and
//!   malformed argument lists;
//! * `tc-ampere-bf16-total-hw.json`: `evaluations` whose expected outputs are
//!   A100 measurements (equal to the RTX 4090's on every entry) from
//!   `docs/data/tc-total-2026-09-07/`, plus the finite A100 golden records.
//!
//! Both files are also checked by `lake exe vectors` against the Lean
//! definition, so Python, Lean, Rust and the two devices agree entry by entry.

use serde::Deserialize;
use veritor_zk_common::tc_total::evaluate;

#[derive(Deserialize)]
struct Vectors {
    gate_set: GateSetInfo,
    evaluations: Vec<Evaluation>,
}

#[derive(Deserialize)]
struct GateSetInfo {
    id: String,
}

#[derive(Deserialize)]
struct Evaluation {
    op: String,
    args: Vec<u64>,
    out: Option<u64>,
    #[serde(default)]
    a100: Option<String>,
    #[serde(default)]
    rtx4090: Option<String>,
}

fn load(name: &str) -> Vectors {
    let path = format!(
        "{}/../../../fixtures/lean-vectors/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let data = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let vectors: Vectors = serde_json::from_str(&data).expect("vector file parses");
    assert_eq!(
        vectors.gate_set.id,
        "veritor.tensor-core.ampere_bf16_m16n8k16@3"
    );
    vectors
}

fn check(name: &str) -> (usize, usize) {
    let vectors = load(name);
    let mut rejected = 0;
    for e in &vectors.evaluations {
        let got = evaluate(&e.op, &e.args).map(u64::from);
        assert_eq!(got, e.out, "{name}: {} {:?}", e.op, e.args);
        if e.out.is_none() {
            rejected += 1;
        }
        if let Some(hw) = &e.a100 {
            assert_eq!(
                Some(u64::from_str_radix(hw, 16).unwrap()),
                got,
                "{name}: A100 column"
            );
        }
        if let Some(hw) = &e.rtx4090 {
            assert_eq!(
                Some(u64::from_str_radix(hw, 16).unwrap()),
                got,
                "{name}: RTX 4090 column"
            );
        }
    }
    (vectors.evaluations.len(), rejected)
}

#[test]
fn the_total_family_agrees_with_the_python_model() {
    let (n, rejected) = check("tc-ampere-bf16-total.json");
    assert!(n >= 1400, "{n} evaluations");
    assert_eq!(rejected, 9, "only malformed argument lists are rejected");
}

#[test]
fn the_total_family_agrees_with_the_a100_and_rtx4090_measurements() {
    let (n, rejected) = check("tc-ampere-bf16-total-hw.json");
    assert!(n >= 8000, "{n} evaluations");
    assert_eq!(rejected, 0);
}
