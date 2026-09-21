//! The Aeneas-translatable model (`zk/sp1/check-model`) against the
//! production checker, on the int8-q7 vector batches: same verdict and same
//! first error for the honest plan, for every tampered plan of `plan.rs`, and
//! for the relation pass -- so the restructuring recorded in the model's
//! module docs is tested to be semantics-preserving, not only argued.

use std::collections::BTreeMap;

#[path = "plan.rs"]
#[allow(dead_code)]
mod plan_tests;

use serde::Deserialize;
use veritor_check_model as model;
use veritor_zk_common::{
    check_openings_multi, check_relations, decode_schema, gate_set, parse_multiproof_witness,
    parse_statement, plan_of, validate_plan, Arg, CheckError, Plan, Statement,
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
}

fn load() -> Vec<Batch> {
    let data = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors.json"))
        .expect("vectors.json");
    let vectors: Vectors = serde_json::from_str(&data).expect("vectors.json parses");
    vectors
        .batches
        .into_iter()
        .filter(|batch| batch.name.starts_with("int8-q7"))
        .collect()
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).expect("hex"))
        .collect()
}

/// An injective renaming of 32-byte digests / schema strings to small integers.
#[derive(Default)]
struct Names {
    digests: BTreeMap<[u8; 32], u64>,
    schemas: BTreeMap<String, u8>,
}

impl Names {
    fn digest(&mut self, value: &[u8; 32]) -> u64 {
        let next = self.digests.len() as u64 + 1;
        *self.digests.entry(*value).or_insert(next)
    }
    fn schema(&mut self, value: &str) -> u8 {
        let next = self.schemas.len() as u8;
        *self.schemas.entry(value.to_owned()).or_insert(next)
    }
}

fn op(name: &str) -> model::Op {
    match name {
        "add32" => model::Op::Add32,
        "in" => model::Op::In,
        "mul8" => model::Op::Mul8,
        "requant_s8" => model::Op::RequantS8,
        "requant_u7" => model::Op::RequantU7,
        "weight" => model::Op::Weight,
        other => panic!("not an int8-q7 op: {other}"),
    }
}

/// The production statement in the model's representation.
fn to_model(statement: &Statement) -> model::Statement {
    let mut names = Names::default();
    let programs = statement
        .kinds
        .iter()
        .map(|kind| model::Program {
            gates: kind
                .gates
                .iter()
                .map(|gate| model::Gate {
                    op: op(&gate.op),
                    args: gate
                        .args
                        .iter()
                        .map(|arg| match *arg {
                            Arg::Port(k) => model::Arg::Port(k),
                            Arg::Local(j) => model::Arg::Local(j),
                        })
                        .collect(),
                })
                .collect(),
            outputs: kind.outputs.clone(),
        })
        .collect();
    let obligations = statement
        .obligations
        .iter()
        .map(|obligation| model::Obligation {
            program: statement
                .kinds
                .iter()
                .position(|kind| kind.kind == obligation.kind)
                .expect("the parser checked the kind") as u32,
            commitments: obligation
                .commitments
                .iter()
                .map(|c| model::Commitment {
                    domain_id: names.digest(&c.domain_id),
                    root: names.digest(&c.root),
                    count: c.count,
                })
                .collect(),
            positions: obligation
                .positions
                .iter()
                .map(|p| model::Position {
                    commitment: p.commitment,
                    rank: p.rank,
                    position: p.position,
                    schema: names.schema(&p.schema),
                })
                .collect(),
            inputs: obligation.inputs.clone(),
            outputs: obligation.outputs.clone(),
        })
        .collect();
    model::Statement {
        programs,
        obligations,
    }
}

fn to_model_plan(plan: &Plan) -> model::Plan {
    model::Plan {
        leaf_counts: plan.leaf_counts.clone(),
        tree_of: plan.tree_of.clone(),
        leaf_of: plan.leaf_of.clone(),
    }
}

/// The production validator's message for each model error variant.
fn message(error: model::PlanError) -> &'static str {
    match error {
        model::PlanError::Shorter => "plan is shorter than the statement",
        model::PlanError::Longer => "plan is longer than the statement",
        model::PlanError::TreeOutOfRange => "tree index out of range",
        model::PlanError::NotByFirstAppearance => "trees are not numbered by first appearance",
        model::PlanError::TreeDefinedTwice => "a tree is defined twice",
        model::PlanError::OtherCommitmentsTree => "a commitment is mapped to another commitment's tree",
        model::PlanError::LeafOutOfRange => "leaf index out of range",
        model::PlanError::OtherCoordinates => "a position is mapped to a leaf with other coordinates",
        model::PlanError::TreeNeverNamed => "a tree is never named by a commitment",
        model::PlanError::LeafNeverNamed => "a leaf is named by no position",
        model::PlanError::NotAscending => "leaves are not strictly ascending by rank",
    }
}

/// Production and model validators agree: both accept, or both reject with the same first error.
fn assert_same_validation(name: &str, statement: &Statement, plan: &Plan) {
    let production = validate_plan(statement, plan);
    let modelled = model::validate_plan(&to_model(statement), &to_model_plan(plan));
    match (&production, &modelled) {
        (Ok(table), Ok(model_table)) => {
            assert_eq!(table.trees.len(), model_table.trees.len(), "{name}");
            for (a, b) in table.trees.iter().zip(&model_table.trees) {
                let ranks: Vec<u64> = a.leaves.iter().map(|leaf| leaf.rank).collect();
                let model_ranks: Vec<u64> = b.leaves.iter().map(|leaf| leaf.rank).collect();
                assert_eq!(ranks, model_ranks, "{name}");
            }
        }
        (Err(CheckError::Plan(text)), Err(error)) => {
            assert_eq!(*text, message(*error), "{name}: same first error");
        }
        other => panic!("{name}: production {:?} vs model {:?}", other.0.is_ok(), other.1),
    }
}

#[test]
fn the_model_agrees_with_production_on_honest_plans_and_the_relation_pass() {
    for batch in load() {
        let statement = parse_statement(&unhex(&batch.statement)).expect("statement parses");
        let witness =
            parse_multiproof_witness(&unhex(&batch.multiproof_witness)).expect("witness parses");
        let set = gate_set(&statement).expect("gate set");
        let plan = plan_of(&statement).expect("honest plan");
        assert_same_validation(&batch.name, &statement, &plan);

        // the operands: production decodes per position; the model is handed the
        // per-tree decoded words (what `check_openings_planned` computes after
        // authentication) and must distribute them identically
        let table = validate_plan(&statement, &plan).expect("honest plan validates");
        let tree_words: Vec<Vec<u64>> = table
            .trees
            .iter()
            .zip(&witness.trees)
            .map(|(tree, opened)| {
                tree.leaves
                    .iter()
                    .zip(&opened.values)
                    .map(|(leaf, value)| decode_schema(leaf.schema, value).expect("decodes"))
                    .collect()
            })
            .collect();
        let reference = check_openings_multi(&statement, &witness, &set).expect("reference pass");
        let modelled = to_model(&statement);
        let operands = model::planned_operands(&modelled, &to_model_plan(&plan), &tree_words);
        assert_eq!(operands, reference, "{}: same operands", batch.name);

        // the relation pass: same verdict, same error class
        let production = check_relations(&statement, &reference, &set);
        let model_verdict = model::check_relations(&modelled, &operands);
        match (&production, &model_verdict) {
            (Ok(()), Ok(())) => {}
            (Err(CheckError::Relation { .. }), Err(model::CheckError::Relation)) => {}
            (Err(CheckError::Arity { .. }), Err(model::CheckError::Arity)) => {}
            (Err(CheckError::Argument { .. }), Err(model::CheckError::Argument)) => {}
            (Err(CheckError::SourceNotOpened { .. }), Err(model::CheckError::SourceNotOpened)) => {}
            other => panic!("{}: production {:?} vs model {:?}", batch.name, other.0, other.1),
        }
        assert_eq!(
            model::check_fast(&modelled, &to_model_plan(&plan), &tree_words).is_ok(),
            production.is_ok(),
            "{}",
            batch.name
        );
    }
}

#[test]
fn the_model_rejects_the_tampered_plans_with_productions_first_error() {
    for batch in load() {
        let statement = parse_statement(&unhex(&batch.statement)).expect("statement parses");
        let honest = plan_of(&statement).expect("honest plan");
        let mut cases: Vec<(&str, Plan)> = Vec::new();
        let tree_with_two = (0..honest.leaf_counts.len()).find(|&t| honest.leaf_counts[t] >= 2);
        if let Some(at) = (0..honest.leaf_of.len()).find(|&at| {
            let (tree, _) = locate(&statement, &honest, at);
            honest.leaf_counts[tree] >= 2
        }) {
            let mut plan = honest.clone();
            let (tree, _) = locate(&statement, &honest, at);
            plan.leaf_of[at] = (plan.leaf_of[at] + 1) % honest.leaf_counts[tree];
            cases.push(("wrong-leaf", plan));
        }
        {
            let last = statement.obligations.last().expect("an obligation");
            let mut plan = honest.clone();
            plan.leaf_of.truncate(plan.leaf_of.len() - last.positions.len());
            plan.tree_of.truncate(plan.tree_of.len() - last.commitments.len());
            cases.push(("omitted-obligation", plan));
        }
        if let Some(tree) = tree_with_two {
            let mut plan = honest.clone();
            for at in 0..plan.leaf_of.len() {
                if locate(&statement, &honest, at).0 == tree {
                    plan.leaf_of[at] = match plan.leaf_of[at] {
                        0 => 1,
                        1 => 0,
                        other => other,
                    };
                }
            }
            cases.push(("permuted-leaves", plan));
            let mut plan = honest.clone();
            for at in 0..plan.leaf_of.len() {
                if locate(&statement, &honest, at).0 == tree && plan.leaf_of[at] == 1 {
                    plan.leaf_of[at] = 0;
                }
            }
            cases.push(("aliased-leaves", plan));
        }
        {
            let mut plan = honest.clone();
            plan.leaf_counts[0] += 1;
            cases.push(("unnamed-leaf", plan));
            let mut plan = honest.clone();
            plan.leaf_of[0] = u32::MAX;
            cases.push(("leaf-out-of-range", plan));
            let mut plan = honest.clone();
            plan.tree_of[0] = u32::MAX;
            cases.push(("tree-out-of-range", plan));
            let mut plan = honest.clone();
            plan.leaf_of.push(0);
            cases.push(("extra-entry", plan));
        }
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
        if let Some(plan) = plan_tests::split_tree(&statement, &honest) {
            cases.push(("split-tree", plan));
        }
        for (name, plan) in cases {
            assert!(validate_plan(&statement, &plan).is_err(), "{}: {name}", batch.name);
            assert_same_validation(&format!("{}: {name}", batch.name), &statement, &plan);
        }
    }
}

/// The `(tree, obligation)` a flat leaf-entry index `at` belongs to, under the honest plan.
fn locate(statement: &Statement, plan: &Plan, at: usize) -> (usize, usize) {
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
