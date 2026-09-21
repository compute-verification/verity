//! The packed opening pass (`check_openings_packed`, witness format 3) on a
//! hand-built statement: two words of one row leaf feed an `add`, a third word
//! of the same leaf is the opened output.  Covers what the conformance vectors
//! do not: the pin comparison against the extracted word's big-endian bytes,
//! the descriptor-agreement rule, and each rejection of the leaf check.

use veritor_zk_common::{
    check_openings_packed, check_relations, gate_set, vllm_leaf_hash, vllm_lift_hash,
    vllm_node_hash, Arg, CheckError, CommitmentRef, GateOp, GateSet, KindProgram, Obligation,
    PackedLeaf, PackedWitness, PositionRef, Statement,
};

const WORD_ARITH: &str = "veritor.word-arithmetic@2";
const LEAF_COUNT: u64 = 6; // an odd level on the way up: one lift
const RANK: u64 = 4;
const ID: &str = "p=b3eba618b1b5/r=0/t=0/ph=decode/l=0/op=layers.0.rmsnorm/port=0";

/// The row leaf: four u16le words `[a, b, a+b, filler]`.
fn row(a: u16, b: u16) -> Vec<u8> {
    let mut out = Vec::new();
    for w in [a, b, a.wrapping_add(b), 0x1234u16] {
        out.extend(w.to_le_bytes());
    }
    out
}

fn leaf_hash(value: &[u8]) -> [u8; 32] {
    vllm_leaf_hash(ID, "bfloat16", "[4]", value)
}

/// The serving side's tree over `LEAF_COUNT` leaves with `value` at `RANK`;
/// returns `(root, path of RANK)`.
fn tree(value: &[u8]) -> ([u8; 32], Vec<Option<[u8; 32]>>) {
    let leaves: Vec<[u8; 32]> = (0..LEAF_COUNT)
        .map(|i| if i == RANK { leaf_hash(value) } else { leaf_hash(&(i as u64).to_le_bytes()) })
        .collect();
    let mut levels = vec![leaves];
    while levels.last().unwrap().len() > 1 {
        let level = levels.last().unwrap();
        let next: Vec<[u8; 32]> = level
            .chunks(2)
            .map(|pair| if pair.len() == 2 { vllm_node_hash(&pair[0], &pair[1]) } else { vllm_lift_hash(&pair[0]) })
            .collect();
        levels.push(next);
    }
    let mut path = Vec::new();
    let mut cursor = RANK as usize;
    for level in &levels[..levels.len() - 1] {
        path.push(level.get(cursor ^ 1).copied());
        cursor /= 2;
    }
    (levels.last().unwrap()[0], path)
}

fn schema(index: u32) -> String {
    format!("packed/v1:verity-vllm/leaf/v1:u16le:4:{index}")
}

fn position(index: u32, expected: Option<Vec<u8>>) -> PositionRef {
    PositionRef { commitment: 0, rank: RANK, position: RANK, schema: schema(index), expected }
}

fn build(root: [u8; 32], positions: Vec<PositionRef>) -> (Statement, GateSet) {
    let set = GateSet::resolve(WORD_ARITH, 16).expect("word arithmetic");
    let kind = [7u8; 32];
    let statement = Statement {
        gate_set_id: WORD_ARITH.to_string(),
        gate_set_digest: set.digest(),
        width: 16,
        kinds: vec![KindProgram {
            kind,
            size: 1,
            ports: vec![0, 1],
            gates: vec![GateOp { op: "add".to_string(), args: vec![Arg::Port(0), Arg::Port(1)] }],
            outputs: vec![0],
        }],
        obligations: vec![Obligation {
            session: [1u8; 32],
            compiled: [2u8; 32],
            unit: 0,
            replay_unit: 0,
            kind,
            commitments: vec![CommitmentRef {
                owner_plus_two: 2,
                domain_id: [3u8; 32],
                root,
                count: LEAF_COUNT,
            }],
            positions,
            inputs: vec![0, 1],
            outputs: vec![2],
        }],
    };
    (statement, set)
}

fn witness(value: Vec<u8>, path: Vec<Option<[u8; 32]>>) -> PackedWitness {
    PackedWitness {
        trees: vec![vec![PackedLeaf {
            canonical_id: ID.to_string(),
            dtype: "bfloat16".to_string(),
            shape_text: "[4]".to_string(),
            value,
            path,
        }]],
    }
}

fn run(statement: &Statement, witness: &PackedWitness, set: &GateSet) -> Result<Vec<Vec<u64>>, CheckError> {
    let words = check_openings_packed(statement, witness, set)?;
    check_relations(statement, &words, set)?;
    Ok(words)
}

#[test]
fn three_words_of_one_leaf_authenticate_once_and_the_relation_holds() {
    let value = row(0x1234, 0xbed1);
    let (root, path) = tree(&value);
    let (statement, set) = build(root, vec![position(0, None), position(1, None), position(2, None)]);
    let words = run(&statement, &witness(value, path), &set).expect("accepted");
    assert_eq!(words, vec![vec![0x1234, 0xbed1, 0x1234u64.wrapping_add(0xbed1) & 0xffff]]);
}

#[test]
fn the_pin_is_compared_with_the_extracted_word_in_big_endian() {
    let value = row(0x1234, 0xbed1);
    let (root, path) = tree(&value);
    // the leaf stores 0xbed1 as `d1 be` (le); the pin is the word's canonical be bytes
    let (statement, set) = build(root, vec![position(0, None), position(1, Some(vec![0xbe, 0xd1])), position(2, None)]);
    run(&statement, &witness(value.clone(), path.clone()), &set).expect("pin holds");
    // the leaf's own byte order is *not* what a pin means
    let (statement, set) = build(root, vec![position(0, None), position(1, Some(vec![0xd1, 0xbe])), position(2, None)]);
    assert_eq!(
        run(&statement, &witness(value.clone(), path.clone()), &set),
        Err(CheckError::PublicInput { obligation: 0, position: 1 })
    );
    // a pin of the wrong length
    let (statement, set) = build(root, vec![position(0, None), position(1, Some(vec![0xbe])), position(2, None)]);
    assert_eq!(
        run(&statement, &witness(value, path), &set),
        Err(CheckError::PublicInput { obligation: 0, position: 1 })
    );
}

#[test]
fn a_shifted_index_changes_the_supplied_word_and_fails_the_relation() {
    let value = row(0x1234, 0xbed1);
    let (root, path) = tree(&value);
    let (statement, set) = build(root, vec![position(0, None), position(3, None), position(2, None)]);
    let error = run(&statement, &witness(value, path), &set).unwrap_err();
    assert!(matches!(error, CheckError::Relation { obligation: 0, offset: 0, .. }), "{error:?}");
}

#[test]
fn positions_of_one_leaf_must_agree_on_the_descriptor() {
    let value = row(1, 2);
    let (root, path) = tree(&value);
    // same rank, another word count: not one leaf
    let mut other = position(1, None);
    other.schema = "packed/v1:verity-vllm/leaf/v1:u16le:8:1".to_string();
    let (statement, set) = build(root, vec![position(0, None), other, position(2, None)]);
    assert!(matches!(
        run(&statement, &witness(value.clone(), path.clone()), &set),
        Err(CheckError::Opening { obligation: 0, position: 1, .. })
    ));
    // same rank, another byte order
    let mut other = position(1, None);
    other.schema = "packed/v1:verity-vllm/leaf/v1:u16be:4:1".to_string();
    let (statement, set) = build(root, vec![position(0, None), other, position(2, None)]);
    assert!(matches!(
        run(&statement, &witness(value.clone(), path.clone()), &set),
        Err(CheckError::Opening { obligation: 0, position: 1, .. })
    ));
    // a one-word schema among packed positions: a format-3 witness cannot serve it
    // (Python: MALFORMED_TRANSCRIPT)
    let mut other = position(1, None);
    other.schema = "u16".to_string();
    let (statement, set) = build(root, vec![position(0, None), other, position(2, None)]);
    assert!(matches!(run(&statement, &witness(value, path), &set), Err(CheckError::WitnessShape(_))));
}

#[test]
fn the_leaf_check_rejects_before_any_word_is_read() {
    let value = row(1, 2);
    let (root, path) = tree(&value);
    let (statement, set) = build(root, vec![position(0, None), position(1, None), position(2, None)]);
    let is_opening = |r: &Result<Vec<Vec<u64>>, CheckError>| matches!(r, Err(CheckError::Opening { obligation: 0, position: 0, .. }));

    // a tampered byte: the path no longer reaches the root
    let mut tampered = value.clone();
    tampered[0] ^= 1;
    assert!(is_opening(&run(&statement, &witness(tampered, path.clone()), &set)));
    // a leaf of the wrong length for `u16le:4`
    assert!(is_opening(&run(&statement, &witness(value[..6].to_vec(), path.clone()), &set)));
    // a path of the wrong length
    assert!(is_opening(&run(&statement, &witness(value.clone(), path[..path.len() - 1].to_vec()), &set)));
    // a lift where the shape has a sibling
    let mut bad = path.clone();
    let k = bad.iter().position(|s| s.is_some()).unwrap();
    bad[k] = None;
    assert!(is_opening(&run(&statement, &witness(value.clone(), bad), &set)));
    // another header: another leaf
    let mut w = witness(value.clone(), path.clone());
    w.trees[0][0].shape_text = "[1,4]".to_string();
    assert!(is_opening(&run(&statement, &w, &set)));
    // the witness must have one leaf per opened rank per tree
    let mut w = witness(value.clone(), path.clone());
    let duplicate = w.trees[0][0].clone();
    w.trees[0].push(duplicate);
    assert!(matches!(run(&statement, &w, &set), Err(CheckError::WitnessShape(_))));
    let w = PackedWitness { trees: vec![] };
    assert!(matches!(run(&statement, &w, &set), Err(CheckError::WitnessShape(_))));
    // a rank outside the commitment
    let mut out_of_range = statement.clone();
    for p in &mut out_of_range.obligations[0].positions {
        p.rank = LEAF_COUNT;
        p.position = LEAF_COUNT;
    }
    assert!(is_opening(&run(&out_of_range, &witness(value, path), &set)));
    assert_eq!(gate_set(&statement).expect("gate set resolves").digest(), set.digest());
}
