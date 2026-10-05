//! The planned opening pass against the reference one, on the cross-language
//! vector batches: the honest plan validates (completeness), it yields the
//! very same decoded operands and verdict as `check_openings_multi`
//! (agreement), and every class of tampered plan is rejected (soundness of
//! the validator: a plan is advice, never trust).

use serde::Deserialize;
use veritor_zk_common::{
    check_batch_multi, check_batch_planned, check_openings_multi, check_openings_planned,
    gate_set, parse_multiproof_witness, parse_plan, parse_statement, plan_of, tree_table,
    validate_plan, CheckError, Plan, PLAN_MAGIC,
};

#[derive(Deserialize)]
struct Vectors {
    batches: Vec<Batch>,
}

#[derive(Deserialize)]
struct Batch {
    name: String,
    statement: String,
    multiproof_witness: String,
    verdict: bool,
}

fn load() -> Vectors {
    let data = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors.json"))
        .expect("vectors.json");
    serde_json::from_str(&data).expect("vectors.json parses")
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).expect("hex"))
        .collect()
}

/// The plan's canonical bytes, as the Python `encode_plan` writes them.
fn encode_plan(plan: &Plan) -> Vec<u8> {
    let mut out = PLAN_MAGIC.to_vec();
    for list in [&plan.leaf_counts, &plan.tree_of, &plan.leaf_of] {
        out.extend((list.len() as u32).to_be_bytes());
        for item in list {
            out.extend(item.to_be_bytes());
        }
    }
    out
}

#[test]
fn the_honest_plan_validates_and_agrees_with_the_reference() {
    for batch in load().batches {
        let statement = parse_statement(&unhex(&batch.statement)).expect("statement parses");
        let witness =
            parse_multiproof_witness(&unhex(&batch.multiproof_witness)).expect("witness parses");
        let set = gate_set(&statement).expect("gate set");
        let plan = plan_of(&statement).expect("the honest plan exists");
        // round trip through the wire layout
        let parsed = parse_plan(&encode_plan(&plan)).expect("plan parses");
        assert_eq!(parsed, plan, "{}", batch.name);
        // completeness: the honest plan is accepted and is the canonical table
        let table = validate_plan(&statement, &plan).expect("the honest plan validates");
        let reference = tree_table(&statement).expect("reference table");
        assert_eq!(table.trees.len(), reference.trees.len(), "{}", batch.name);
        for (planned, canonical) in table.trees.iter().zip(&reference.trees) {
            assert_eq!(planned.commitment, canonical.commitment, "{}", batch.name);
            let ranks: Vec<u64> = planned.leaves.iter().map(|leaf| leaf.rank).collect();
            let expected: Vec<u64> = canonical.leaves.iter().map(|leaf| leaf.rank).collect();
            assert_eq!(ranks, expected, "{}", batch.name);
        }
        // agreement: the same operands reach the interpreter, the same verdict comes out
        let reference_values =
            check_openings_multi(&statement, &witness, &set).expect("reference authenticates");
        let planned_values = check_openings_planned(&statement, &witness, &plan, &set)
            .expect("planned pass authenticates");
        assert_eq!(planned_values, reference_values, "{}", batch.name);
        assert_eq!(
            check_batch_planned(&statement, &witness, &plan).is_ok(),
            check_batch_multi(&statement, &witness).is_ok(),
            "{}",
            batch.name
        );
        assert_eq!(
            check_batch_planned(&statement, &witness, &plan).is_ok(),
            batch.verdict,
            "{}",
            batch.name
        );
    }
}

/// Every way a plan can lie, on every vector batch that has room for it: the
/// validator must refuse each, and the refusal must be a `Plan` error (the
/// lie is caught before anything is authenticated or decoded).
#[test]
fn tampered_plans_are_rejected() {
    for batch in load().batches {
        let statement = parse_statement(&unhex(&batch.statement)).expect("statement parses");
        let witness =
            parse_multiproof_witness(&unhex(&batch.multiproof_witness)).expect("witness parses");
        let set = gate_set(&statement).expect("gate set");
        let honest = plan_of(&statement).expect("honest plan");
        let mut cases: Vec<(&str, Plan)> = Vec::new();

        // wrong leaf: point one position at another leaf of its tree (a different rank)
        if let Some(at) = (0..honest.leaf_of.len()).find(|&at| {
            let (tree, _) = locate(&statement, &honest, at);
            honest.leaf_counts[tree] >= 2
        }) {
            let mut plan = honest.clone();
            let (tree, _) = locate(&statement, &honest, at);
            plan.leaf_of[at] = (plan.leaf_of[at] + 1) % honest.leaf_counts[tree];
            cases.push(("wrong-leaf", plan));
        }
        // omission: drop the last obligation's entries
        {
            let last = statement.obligations.last().expect("an obligation");
            let mut plan = honest.clone();
            plan.leaf_of.truncate(plan.leaf_of.len() - last.positions.len());
            plan.tree_of.truncate(plan.tree_of.len() - last.commitments.len());
            cases.push(("omitted-obligation", plan));
        }
        // permutation: swap the numbering of two leaves everywhere (breaks the rank order)
        if let Some(tree) = (0..honest.leaf_counts.len()).find(|&t| honest.leaf_counts[t] >= 2) {
            let mut plan = honest.clone();
            for at in 0..plan.leaf_of.len() {
                let (t, _) = locate(&statement, &honest, at);
                if t == tree {
                    plan.leaf_of[at] = match plan.leaf_of[at] {
                        0 => 1,
                        1 => 0,
                        other => other,
                    };
                }
            }
            cases.push(("permuted-leaves", plan));
        }
        // duplication: declare one leaf too many that no position names
        {
            let mut plan = honest.clone();
            plan.leaf_counts[0] += 1;
            cases.push(("unnamed-leaf", plan));
        }
        // aliasing: two ranks on one leaf index
        if let Some(tree) = (0..honest.leaf_counts.len()).find(|&t| honest.leaf_counts[t] >= 2) {
            let mut plan = honest.clone();
            for at in 0..plan.leaf_of.len() {
                let (t, _) = locate(&statement, &honest, at);
                if t == tree && plan.leaf_of[at] == 1 {
                    plan.leaf_of[at] = 0;
                }
            }
            cases.push(("aliased-leaves", plan));
        }
        // malformed: an index outside its tree
        {
            let mut plan = honest.clone();
            plan.leaf_of[0] = u32::MAX;
            cases.push(("leaf-out-of-range", plan));
            let mut plan = honest.clone();
            plan.tree_of[0] = u32::MAX;
            cases.push(("tree-out-of-range", plan));
        }
        // tree order: number the trees in another order than first appearance
        if honest.leaf_counts.len() >= 2 {
            let mut plan = honest.clone();
            plan.leaf_counts.swap(0, 1);
            for item in &mut plan.tree_of {
                *item = match *item {
                    0 => 1,
                    1 => 0,
                    other => other,
                };
            }
            cases.push(("permuted-trees", plan));
        }
        // shape: one entry too many
        {
            let mut plan = honest.clone();
            plan.leaf_of.push(0);
            cases.push(("extra-entry", plan));
        }
        // split: a commitment naming an already-defined tree gets a fresh tree of its own
        if let Some(plan) = split_tree(&statement, &honest) {
            cases.push(("split-tree", plan));
        }

        println!(
            "{}: {} tampered plans: {:?}",
            batch.name,
            cases.len(),
            cases.iter().map(|(name, _)| *name).collect::<Vec<_>>()
        );
        for (name, plan) in cases {
            let outcome = check_openings_planned(&statement, &witness, &plan, &set);
            assert!(
                matches!(outcome, Err(CheckError::Plan(_))),
                "{}: {name} must be rejected by the validator, got {outcome:?}",
                batch.name
            );
            assert!(check_batch_planned(&statement, &witness, &plan).is_err());
        }
    }
}

/// The honest plan with the first commitment that names an already-defined
/// tree moved to a fresh tree index (its positions renumbered within the new
/// tree, ranks ascending): a well-formed table for a statement with one tree
/// more than it has, which the validator must refuse as a tree defined twice.
pub fn split_tree(statement: &veritor_zk_common::Statement, honest: &Plan) -> Option<Plan> {
    let mut plan = honest.clone();
    let mut seen = vec![false; honest.leaf_counts.len()];
    let mut tree_cursor = 0;
    let mut leaf_cursor = 0;
    for obligation in &statement.obligations {
        for c in 0..obligation.commitments.len() {
            let tree = honest.tree_of[tree_cursor + c] as usize;
            if seen[tree] {
                let fresh = plan.leaf_counts.len() as u32;
                plan.tree_of[tree_cursor + c] = fresh;
                let mut old_leaves: Vec<(u64, u32)> = obligation
                    .positions
                    .iter()
                    .enumerate()
                    .filter(|(_, position)| position.commitment as usize == c)
                    .map(|(slot, position)| (position.rank, honest.leaf_of[leaf_cursor + slot]))
                    .collect();
                old_leaves.sort();
                old_leaves.dedup();
                for (slot, position) in obligation.positions.iter().enumerate() {
                    if position.commitment as usize == c {
                        let old = honest.leaf_of[leaf_cursor + slot];
                        let new = old_leaves.iter().position(|&(_, o)| o == old).expect("own leaf");
                        plan.leaf_of[leaf_cursor + slot] = new as u32;
                    }
                }
                plan.leaf_counts.push(old_leaves.len() as u32);
                return Some(plan);
            }
            seen[tree] = true;
        }
        tree_cursor += obligation.commitments.len();
        leaf_cursor += obligation.positions.len();
    }
    None
}

/// The `(tree, obligation)` a flat leaf-entry index `at` belongs to, under the honest plan.
fn locate(
    statement: &veritor_zk_common::Statement,
    plan: &Plan,
    at: usize,
) -> (usize, usize) {
    let mut leaf_cursor = 0;
    let mut tree_cursor = 0;
    for (index, obligation) in statement.obligations.iter().enumerate() {
        if at < leaf_cursor + obligation.positions.len() {
            let position = &obligation.positions[at - leaf_cursor];
            let tree = plan.tree_of[tree_cursor + position.commitment as usize] as usize;
            return (tree, index);
        }
        leaf_cursor += obligation.positions.len();
        tree_cursor += obligation.commitments.len();
    }
    panic!("entry {at} beyond the statement");
}

#[test]
fn a_plan_needs_the_statement_shape() {
    // the empty plan is a valid encoding of no trees; it fits only the empty statement
    let empty = parse_plan(&encode_plan(&Plan {
        leaf_counts: vec![],
        tree_of: vec![],
        leaf_of: vec![],
    }))
    .expect("empty plan parses");
    for batch in load().batches {
        let statement = parse_statement(&unhex(&batch.statement)).expect("statement parses");
        assert!(matches!(
            validate_plan(&statement, &empty),
            Err(CheckError::Plan(_))
        ));
    }
    assert!(parse_plan(b"veritor/proofs/plan/tree-table/v1\0\0\0\0").is_err());
    assert!(parse_plan(b"not a plan").is_err());
}
