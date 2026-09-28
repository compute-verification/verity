//! The plan-layer vectors of `fixtures/lean-vectors/plan.json` (lane L4, Target C)
//! against the guest's `validate_plan`: every plan the file marks valid is
//! accepted and every plan it marks invalid is rejected.  The file was generated
//! by the predecessor implementation from its Python `encode_plan` (honest
//! plans) and its tampered-plan corpus (negative plans), and is frozen.
//!
//! The file is looked up at `$PLAN_VECTORS` or, failing that, at
//! `<crate>/../../../fixtures/lean-vectors/plan.json`.

use serde::Deserialize;
use veritor_zk_common::{parse_plan, parse_statement, plan_of, validate_plan};

#[derive(Deserialize)]
struct Vectors {
    statements: Vec<Statement>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Statement {
    hex: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    statement: usize,
    plan_hex: String,
    expected_valid: bool,
    honest: bool,
}

fn load() -> Vectors {
    let path = std::env::var("PLAN_VECTORS")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../fixtures/lean-vectors/plan.json").to_string());
    let data = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&data).expect("plan.json parses")
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).expect("hex"))
        .collect()
}

#[test]
fn validate_plan_agrees_with_the_plan_vectors() {
    let vectors = load();
    let statements: Vec<_> = vectors
        .statements
        .iter()
        .map(|s| parse_statement(&unhex(&s.hex)).expect("statement parses"))
        .collect();
    let mut checked = 0;
    let mut honest_checked = 0;
    let mut failures = Vec::new();
    for case in &vectors.cases {
        let statement = &statements[case.statement];
        let plan = parse_plan(&unhex(&case.plan_hex)).expect("plan parses");
        let verdict = validate_plan(statement, &plan).is_ok();
        checked += 1;
        if verdict != case.expected_valid {
            failures.push(format!(
                "{}: validate_plan {} but expected {}",
                case.name,
                if verdict { "accepts" } else { "rejects" },
                if case.expected_valid { "valid" } else { "invalid" }
            ));
        }
        if case.honest {
            // the guest's own honest plan is the Python one, list for list
            let own = plan_of(statement).expect("the honest plan exists");
            honest_checked += 1;
            if own.leaf_counts != plan.leaf_counts || own.tree_of != plan.tree_of || own.leaf_of != plan.leaf_of {
                failures.push(format!("{}: plan_of differs from the Python encode_plan", case.name));
            }
        }
    }
    println!("plan vectors: {checked} verdicts checked, {honest_checked} honest plans compared, {} disagree", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
