//! The acceptance boundary itself: `kernel::verify` from untrusted bytes.
//!
//! Everything the guest decides is decided here, so these tests exercise the
//! byte-level entry point exactly as the guest calls it -- statement bytes,
//! witness bytes, plan bytes, object-digest bytes in; committed public values
//! out -- on the cross-language vector batches and on every class of
//! malformed input: bytes that do not parse, format pairings the wire formats
//! do not support, tampered plans, and object digests of the wrong length.
//!
//! The per-phase suites (vectors.rs, plan.rs, packed.rs, ranges.rs, model.rs)
//! exercise the kernel's subroutines; this file pins the boundary: for every
//! input, `verify` commits `sha256(statement bytes) || verdict` (or the
//! 65-byte layout with a supplied `h_O`), and nothing else.

use serde::Deserialize;
use veritor_zk_common::kernel::{verify, ValidatedStatement};
use veritor_zk_common::{
    check_relations, decode_verdict, gate_set, parse_any_witness, parse_statement, plan_of,
    sha256, split_public_values, Arg, CheckError, GateOp, KindProgram, Plan, PLAN_MAGIC, UNSOUND,
};

#[derive(Deserialize)]
struct Vectors {
    batches: Vec<Batch>,
}

#[derive(Deserialize)]
struct Batch {
    name: String,
    statement: String,
    witness: String,
    multiproof_witness: String,
    #[serde(default)]
    rejected_witnesses: Vec<RejectedWitness>,
    #[serde(default)]
    rejected_multiproof_witnesses: Vec<RejectedWitness>,
    statement_digest: String,
    verdict: bool,
}

#[derive(Deserialize)]
struct RejectedWitness {
    name: String,
    witness: String,
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

/// `verify` commits the 33-byte layout with this exact digest and verdict.
fn expect(decision: &veritor_zk_common::Decision, statement_bytes: &[u8], verdict: bool) {
    let (h_o, digest, byte) = split_public_values(decision.public_values()).expect("layout");
    assert_eq!(h_o, None);
    assert_eq!(digest, sha256(statement_bytes));
    assert_eq!(decode_verdict(byte), Some((verdict, UNSOUND)));
    assert_eq!(decision.verdict(), verdict);
}

#[test]
fn the_committed_batches_decide_exactly_as_committed() {
    for batch in load().batches {
        let statement = unhex(&batch.statement);
        assert_eq!(
            sha256(&statement).to_vec(),
            unhex(&batch.statement_digest),
            "{}",
            batch.name
        );
        for witness in [unhex(&batch.witness), unhex(&batch.multiproof_witness)] {
            let decision = verify(&statement, &witness, &[], &[]);
            expect(&decision, &statement, batch.verdict);
        }
    }
}

#[test]
fn every_committed_rejected_witness_rejects_at_the_boundary() {
    let mut exercised = 0;
    for batch in load().batches {
        let statement = unhex(&batch.statement);
        for rejected in batch
            .rejected_witnesses
            .iter()
            .chain(&batch.rejected_multiproof_witnesses)
        {
            let decision = verify(&statement, &unhex(&rejected.witness), &[], &[]);
            expect(&decision, &statement, false);
            exercised += 1;
            let _ = &rejected.name;
        }
    }
    assert!(exercised > 0, "the vectors carry rejected witnesses");
}

#[test]
fn bytes_that_do_not_parse_reject_over_their_own_digest() {
    let batch = &load().batches[0];
    let statement = unhex(&batch.statement);
    let witness = unhex(&batch.witness);

    // statements: empty, wrong magic, truncated
    for bad in [
        vec![],
        b"not a statement".to_vec(),
        statement[..statement.len() - 1].to_vec(),
        statement[..8].to_vec(),
    ] {
        let decision = verify(&bad, &witness, &[], &[]);
        expect(&decision, &bad, false);
    }
    // witnesses: empty, wrong magic, truncated
    for bad in [
        vec![],
        b"not a witness".to_vec(),
        witness[..witness.len() - 1].to_vec(),
    ] {
        let decision = verify(&statement, &bad, &[], &[]);
        expect(&decision, &statement, false);
    }
    // plans: garbage bytes reject even alongside a valid pair
    let decision = verify(&statement, &witness, b"not a plan", &[]);
    expect(&decision, &statement, false);
}

#[test]
fn the_planned_path_decides_identically_and_only_with_a_multiproof_witness() {
    for batch in load().batches {
        let statement_bytes = unhex(&batch.statement);
        let statement = parse_statement(&statement_bytes).expect("statement parses");
        let plan_bytes = encode_plan(&plan_of(&statement).expect("honest plan"));

        // the honest plan with the multiproof witness: the very same decision
        let multi = unhex(&batch.multiproof_witness);
        let planned = verify(&statement_bytes, &multi, &plan_bytes, &[]);
        let reference = verify(&statement_bytes, &multi, &[], &[]);
        assert_eq!(
            planned.public_values(),
            reference.public_values(),
            "{}",
            batch.name
        );

        // a plan with a per-opening witness is a pairing the formats refuse
        let openings = unhex(&batch.witness);
        let decision = verify(&statement_bytes, &openings, &plan_bytes, &[]);
        expect(&decision, &statement_bytes, false);

        // a tampered plan rejects: swap the first two leaf indices of a tree
        // with at least two leaves (or truncate, when none has two)
        let plan = plan_of(&statement).expect("honest plan");
        let tampered = if let Some(_) = plan.leaf_counts.iter().find(|&&n| n >= 2) {
            let mut tampered = plan.clone();
            // find two adjacent leaf_of entries of the same tree and swap them
            let mut swapped = false;
            'outer: for i in 0..tampered.leaf_of.len().saturating_sub(1) {
                if tampered.leaf_of[i] != tampered.leaf_of[i + 1] {
                    tampered.leaf_of.swap(i, i + 1);
                    swapped = true;
                    break 'outer;
                }
            }
            if !swapped {
                continue;
            }
            encode_plan(&tampered)
        } else {
            plan_bytes[..plan_bytes.len() - 4].to_vec()
        };
        let decision = verify(&statement_bytes, &multi, &tampered, &[]);
        expect(&decision, &statement_bytes, false);
    }
}

#[test]
fn the_object_digest_is_carried_through_exactly_or_refused() {
    let batch = &load().batches[0];
    let statement = unhex(&batch.statement);
    let witness = unhex(&batch.witness);

    // 32 bytes: the 65-byte layout, h_O first, digest and verdict unchanged
    let h_o = [7u8; 32];
    let decision = verify(&statement, &witness, &[], &h_o);
    let (carried, digest, byte) = split_public_values(decision.public_values()).expect("layout");
    assert_eq!(carried, Some(h_o));
    assert_eq!(digest, sha256(&statement));
    assert_eq!(decode_verdict(byte), Some((batch.verdict, UNSOUND)));

    // any other length: malformed input, rejected in the 33-byte layout --
    // even when statement and witness would have been accepted
    for bad in [1usize, 31, 33, 64] {
        let decision = verify(&statement, &witness, &[], &vec![9u8; bad]);
        expect(&decision, &statement, false);
    }
}

/// DEDICATED REGRESSION for the one behavior change of the kernel refactor:
/// before, the `attribution` build's copy of `check_relations` skipped
/// `check_programs`, so a statement carrying an *unused* malformed program
/// was accepted by `attribution` builds and rejected by every sound build
/// and by the Lean reference.  There is now ONE `check_relations`, and this
/// test runs in every feature build of the pod matrix (default, `scopes`,
/// `attribution`): a wrong-arity gate in a program no obligation names must
/// reject, and the same statement without it must still accept.
#[test]
fn an_unused_malformed_program_rejects_in_every_build() {
    let batch = &load().batches[0];
    let statement_bytes = unhex(&batch.statement);
    assert!(batch.verdict, "the fixture batch accepts");
    let mut statement = parse_statement(&statement_bytes).expect("statement parses");
    let witness = parse_any_witness(&unhex(&batch.witness)).expect("witness parses");
    let set = gate_set(&statement).expect("gate set");

    // decode with the honest statement, then tack on a malformed UNUSED
    // program: a `not` (or whatever the first op is) applied with arity 0.
    let decoded = {
        let validated = ValidatedStatement::validate(&statement_bytes).expect("validates");
        let values = validated
            .authenticate(&witness, None)
            .expect("fixture authenticates");
        values.values().to_vec()
    };
    assert!(check_relations(&statement, &decoded, &set).is_ok());

    let malformed_op = statement.kinds[0].gates.last().expect("a gate").op.clone();
    statement.kinds.push(KindProgram {
        kind: [0xAB; 32],
        size: 1,
        ports: vec![],
        gates: vec![GateOp {
            op: malformed_op.clone(),
            args: vec![Arg::Local(0); 7], // no op of any family has arity 7
        }],
        outputs: vec![0],
    });
    match check_relations(&statement, &decoded, &set) {
        Err(CheckError::Arity { kind, offset }) => {
            assert_eq!(kind, [0xAB; 32]);
            assert_eq!(offset, 0);
        }
        other => panic!(
            "an unused malformed program must reject with Arity, got {other:?} \
             (the pre-kernel `attribution` build accepted it)"
        ),
    }
}

#[test]
fn the_typed_pipeline_is_only_entered_through_validation() {
    let batch = &load().batches[0];
    let statement_bytes = unhex(&batch.statement);

    // garbage never becomes a ValidatedStatement
    assert!(ValidatedStatement::validate(b"garbage").is_err());

    // a validated statement authenticates its witness and checks relations
    let validated = ValidatedStatement::validate(&statement_bytes).expect("fixture validates");
    let witness = veritor_zk_common::parse_any_witness(&unhex(&batch.witness)).expect("witness");
    let values = validated
        .authenticate(&witness, None)
        .expect("fixture authenticates");
    assert_eq!(values.values().len(), {
        let statement = parse_statement(&statement_bytes).expect("parses");
        statement.obligations.len()
    });
    assert_eq!(values.check_relations().is_ok(), batch.verdict, "{}", batch.name);
}
