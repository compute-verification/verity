//! Statement format 3 (`veritor/proofs/statement/v3`): range positions on a
//! hand-built statement, the same row leaf as `packed.rs` (`[a, b, a+b, filler]`
//! as u16le words) under the serving side's framing.  Covers what the
//! conformance vectors do not: the ranged pass reads exactly the words the
//! expansion names (order, stride, aliases), a range shifted by one word fails
//! the relation as PW's `shifted-index` did, a tampered leaf is refused before
//! any word is read, the parser refuses ill-formed ranges (bounds, empty,
//! stride 0, overflow, a schema where a descriptor belongs, a pin on a range
//! -- which the wire cannot say), and a format-3 statement takes a format-3
//! witness only.  The v2 vectors are untouched (`vectors.rs`).

use veritor_zk_common::{
    check_any_statement, check_batch_ranged, check_openings_packed, check_openings_ranged,
    check_relations, expand_statement, gate_set, parse_any_statement, parse_statement_v3,
    ranged_table, vllm_leaf_hash, vllm_lift_hash, vllm_node_hash, AnyStatement, AnyWitness, Arg,
    CheckError, CommitmentRef, GateOp, GateSet, KindProgram, Obligation, Opening, PackedLeaf,
    PackedWitness, PositionEntry, PositionRange, PositionRef, Statement, Witness,
    STATEMENT_V3_MAGIC,
};

const WORD_ARITH: &str = "veritor.word-arithmetic@2";
const LEAF_COUNT: u64 = 6;
const RANK: u64 = 4;
const ID: &str = "p=b3eba618b1b5/r=0/t=0/ph=decode/l=0/op=layers.0.rmsnorm/port=0";
const DESCRIPTOR: &str = "packed/v1:verity-vllm/leaf/v1:u16le:4";

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

fn single(index: u32, expected: Option<Vec<u8>>) -> PositionEntry {
    PositionEntry::Single(PositionRef {
        commitment: 0,
        rank: RANK,
        position: RANK,
        schema: format!("{DESCRIPTOR}:{index}"),
        expected,
    })
}

fn range(start: u64, count: u64, stride: u64) -> PositionEntry {
    PositionEntry::Range(PositionRange {
        commitment: 0,
        rank: RANK,
        position: RANK,
        descriptor: DESCRIPTOR.to_string(),
        start,
        count,
        stride,
    })
}

/// `add(port 0, port 1)` opened at slot 2; `slots` ports when the entries expand to more.
fn build(root: [u8; 32], entries: Vec<PositionEntry>, slots: u32) -> (Statement<PositionEntry>, GateSet) {
    let set = GateSet::resolve(WORD_ARITH, 16).expect("word arithmetic");
    let kind = [7u8; 32];
    let statement = Statement {
        gate_set_id: WORD_ARITH.to_string(),
        gate_set_digest: set.digest(),
        width: 16,
        kinds: vec![KindProgram {
            kind,
            size: 1,
            ports: (0..slots - 1).collect(),
            gates: vec![GateOp { op: "add".to_string(), args: vec![Arg::Port(0), Arg::Port(1)] }],
            outputs: vec![0],
        }],
        obligations: vec![Obligation {
            session: [1u8; 32],
            compiled: [2u8; 32],
            unit: 0,
            replay_unit: 0,
            kind,
            commitments: vec![CommitmentRef { owner_plus_two: 2, domain_id: [3u8; 32], root, count: LEAF_COUNT }],
            positions: entries,
            inputs: (0..slots - 1).collect(),
            outputs: vec![slots - 1],
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

fn run(statement: &Statement<PositionEntry>, witness: &PackedWitness, set: &GateSet) -> Result<Vec<Vec<u64>>, CheckError> {
    let words = check_openings_ranged(statement, witness, set)?;
    check_relations(statement, &words, set)?;
    Ok(words)
}

/// The format-3 wire bytes of a statement (mirrors `wire.encode_statement_v3`).
fn encode_v3(statement: &Statement<PositionEntry>) -> Vec<u8> {
    let mut w: Vec<u8> = Vec::new();
    let u32 = |w: &mut Vec<u8>, v: u32| w.extend(v.to_be_bytes());
    let u64 = |w: &mut Vec<u8>, v: u64| w.extend(v.to_be_bytes());
    let bytes = |w: &mut Vec<u8>, b: &[u8]| {
        w.extend((b.len() as u32).to_be_bytes());
        w.extend(b);
    };
    w.extend(STATEMENT_V3_MAGIC);
    bytes(&mut w, statement.gate_set_id.as_bytes());
    w.extend(statement.gate_set_digest);
    u32(&mut w, statement.width);
    u32(&mut w, statement.kinds.len() as u32);
    for p in &statement.kinds {
        w.extend(p.kind);
        u32(&mut w, p.size);
        u32(&mut w, p.ports.len() as u32);
        for &k in &p.ports {
            u32(&mut w, k);
        }
        u32(&mut w, p.gates.len() as u32);
        for gate in &p.gates {
            bytes(&mut w, gate.op.as_bytes());
            u32(&mut w, gate.args.len() as u32);
            for arg in &gate.args {
                match arg {
                    Arg::Port(k) => {
                        w.push(0);
                        u32(&mut w, *k);
                    }
                    Arg::Local(j) => {
                        w.push(1);
                        u32(&mut w, *j);
                    }
                }
            }
        }
        u32(&mut w, p.outputs.len() as u32);
        for &o in &p.outputs {
            u32(&mut w, o);
        }
    }
    u32(&mut w, statement.obligations.len() as u32);
    for o in &statement.obligations {
        w.extend(o.session);
        w.extend(o.compiled);
        u64(&mut w, o.unit);
        u64(&mut w, o.replay_unit);
        w.extend(o.kind);
        u32(&mut w, o.commitments.len() as u32);
        for c in &o.commitments {
            u64(&mut w, c.owner_plus_two);
            w.extend(c.domain_id);
            w.extend(c.root);
            u64(&mut w, c.count);
        }
        u32(&mut w, o.positions.len() as u32);
        for entry in &o.positions {
            match entry {
                PositionEntry::Single(p) => {
                    w.push(0);
                    u32(&mut w, p.commitment);
                    u64(&mut w, p.rank);
                    u64(&mut w, p.position);
                    bytes(&mut w, p.schema.as_bytes());
                    match &p.expected {
                        None => w.push(0),
                        Some(e) => {
                            w.push(1);
                            bytes(&mut w, e);
                        }
                    }
                }
                PositionEntry::Range(r) => {
                    w.push(1);
                    u32(&mut w, r.commitment);
                    u64(&mut w, r.rank);
                    u64(&mut w, r.position);
                    bytes(&mut w, r.descriptor.as_bytes());
                    u64(&mut w, r.start);
                    u64(&mut w, r.count);
                    u64(&mut w, r.stride);
                }
            }
        }
        u32(&mut w, o.inputs.len() as u32);
        for &x in &o.inputs {
            u32(&mut w, x);
        }
        u32(&mut w, o.outputs.len() as u32);
        for &x in &o.outputs {
            u32(&mut w, x);
        }
    }
    w
}

#[test]
fn a_range_reads_the_words_its_expansion_names() {
    let value = row(0x1234, 0xbed1);
    let (root, path) = tree(&value);
    let sum = 0x1234u64.wrapping_add(0xbed1) & 0xffff;
    // `[a, b]` as one range, the output as a single: the same three words as three positions
    let (statement, set) = build(root, vec![range(0, 2, 1), single(2, None)], 3);
    let words = run(&statement, &witness(value.clone(), path.clone()), &set).expect("accepted");
    assert_eq!(words, vec![vec![0x1234, 0xbed1, sum]]);
    let expanded = expand_statement(&statement);
    assert_eq!(expanded.obligations[0].positions.len(), 3);
    assert_eq!(check_openings_packed(&expanded, &witness(value.clone(), path.clone()), &set).unwrap(), words);
    // one entry per leaf in the table, whatever the number of words
    let table = ranged_table(&statement).unwrap();
    assert_eq!(table.trees.len(), 1);
    assert_eq!(table.trees[0].leaves.len(), 1);
    assert_eq!(table.slots[0].len(), 2);
    // stride 2 and aliases: `a, a+b` by stride, `b` alone, then `a, b, a+b` again -- the
    // relation reads ports 0 and 1 (`a`, `a+b`) so it fails, but the words are the expansion's
    let (statement, set) = build(root, vec![range(0, 2, 2), single(1, None), range(0, 3, 1)], 6);
    let expanded = expand_statement(&statement);
    let schemas: Vec<&str> = expanded.obligations[0].positions.iter().map(|p| p.schema.as_str()).collect();
    assert_eq!(schemas.iter().map(|s| s.rsplit(':').next().unwrap()).collect::<Vec<_>>(), ["0", "2", "1", "0", "1", "2"]);
    let ranged = check_openings_ranged(&statement, &witness(value.clone(), path.clone()), &set).unwrap();
    assert_eq!(ranged, vec![vec![0x1234, sum, 0xbed1, 0x1234, 0xbed1, sum]]);
    assert_eq!(ranged, check_openings_packed(&expanded, &witness(value.clone(), path.clone()), &set).unwrap());
    assert!(matches!(check_relations(&statement, &ranged, &set), Err(CheckError::Relation { obligation: 0, .. })));
    // the whole leaf as one range of one word each way round
    let (statement, set) = build(root, vec![range(0, 4, 1)], 4);
    assert_eq!(
        check_openings_ranged(&statement, &witness(value, path), &set).unwrap(),
        vec![vec![0x1234, 0xbed1, sum, 0x1234]]
    );
}

#[test]
fn a_shifted_range_changes_the_supplied_words_and_fails_the_relation() {
    let value = row(0x1234, 0xbed1);
    let (root, path) = tree(&value);
    // words 1, 2 instead of 0, 1: `b + (a+b) != a+b`; PW's `shifted-index` failed the same way
    let (statement, set) = build(root, vec![range(1, 2, 1), single(2, None)], 3);
    let error = run(&statement, &witness(value.clone(), path.clone()), &set).unwrap_err();
    assert!(matches!(error, CheckError::Relation { obligation: 0, offset: 0, .. }), "{error:?}");
    // and the same shift on the wire: parses (in bounds), then fails the relation, not the opening
    let bytes = encode_v3(&statement);
    let parsed = parse_statement_v3(&bytes).expect("in bounds");
    assert_eq!(parsed, statement);
    assert!(matches!(check_batch_ranged(&parsed, &witness(value, path)), Err(CheckError::Relation { .. })));
}

#[test]
fn a_tampered_leaf_is_refused_before_any_word_is_read() {
    let value = row(1, 2);
    let (root, path) = tree(&value);
    let (statement, set) = build(root, vec![range(0, 2, 1), single(2, None)], 3);
    let is_opening = |r: &Result<Vec<Vec<u64>>, CheckError>| matches!(r, Err(CheckError::Opening { obligation: 0, position: 0, .. }));
    let mut tampered = value.clone();
    tampered[0] ^= 1;
    assert!(is_opening(&run(&statement, &witness(tampered, path.clone()), &set)));
    // a leaf of the wrong length for the descriptor
    assert!(is_opening(&run(&statement, &witness(value[..6].to_vec(), path.clone()), &set)));
    // a path of the wrong length
    assert!(is_opening(&run(&statement, &witness(value.clone(), path[..path.len() - 1].to_vec()), &set)));
    // the witness must have one leaf per opened rank per tree
    let mut w = witness(value.clone(), path.clone());
    let duplicate = w.trees[0][0].clone();
    w.trees[0].push(duplicate);
    assert!(matches!(run(&statement, &w, &set), Err(CheckError::WitnessShape(_))));
    // a range and a single of one rank must agree on the descriptor
    let mut other = single(2, None);
    if let PositionEntry::Single(p) = &mut other {
        p.schema = "packed/v1:verity-vllm/leaf/v1:u16be:4:2".to_string();
    }
    let (mixed, set) = build(root, vec![range(0, 2, 1), other], 3);
    assert!(matches!(run(&mixed, &witness(value.clone(), path.clone()), &set), Err(CheckError::Opening { .. })));
    // a rank outside the commitment
    let (beyond, set) = build(
        root,
        vec![PositionEntry::Range(PositionRange {
            commitment: 0,
            rank: LEAF_COUNT,
            position: LEAF_COUNT,
            descriptor: DESCRIPTOR.to_string(),
            start: 0,
            count: 3,
            stride: 1,
        })],
        3,
    );
    assert!(is_opening(&run(&beyond, &witness(value, path), &set)));
    assert_eq!(gate_set(&statement).unwrap().digest(), set.digest());
}

#[test]
fn a_pinned_word_is_a_single_beside_the_range() {
    let value = row(0x1234, 0xbed1);
    let (root, path) = tree(&value);
    // the pin on the output word, compared with its canonical big-endian bytes
    let sum = (0x1234u16.wrapping_add(0xbed1)).to_be_bytes().to_vec();
    let (statement, set) = build(root, vec![range(0, 2, 1), single(2, Some(sum))], 3);
    run(&statement, &witness(value.clone(), path.clone()), &set).expect("pin holds");
    let (statement, set) = build(root, vec![range(0, 2, 1), single(2, Some(vec![0, 0]))], 3);
    assert_eq!(
        run(&statement, &witness(value, path), &set),
        Err(CheckError::PublicInput { obligation: 0, position: 2 })
    );
}

#[test]
fn ill_formed_ranges_are_refused_by_the_parser() {
    let value = row(1, 2);
    let (root, _) = tree(&value);
    let (good, _) = build(root, vec![range(0, 2, 1), single(2, None)], 3);
    let bytes = encode_v3(&good);
    assert_eq!(parse_statement_v3(&bytes).unwrap(), good);
    assert!(matches!(parse_any_statement(&bytes), Ok(AnyStatement::V3(_))));
    // the v2 parser does not read v3 bytes, nor the reverse
    assert!(veritor_zk_common::parse_statement(&bytes).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(parse_statement_v3(&trailing).is_err());
    assert!(parse_statement_v3(&bytes[..bytes.len() - 1]).is_err());

    // (the kind needs two ports for its `add`; the range is refused before slots are counted)
    let rejects = |entries: Vec<PositionEntry>, slots: u32, what: &str| {
        let (statement, _) = build(root, entries, slots.max(3));
        let error = parse_statement_v3(&encode_v3(&statement)).expect_err(what);
        assert!(error.0.contains(what), "{what}: {error:?}");
    };
    // bounds: the last word `start + (count-1)*stride` must be below the leaf's 4 words
    rejects(vec![range(0, 5, 1)], 5, "reaches word 4 of a leaf of 4 words");
    rejects(vec![range(1, 4, 1)], 4, "reaches word 4");
    rejects(vec![range(2, 2, 2)], 2, "reaches word 4");
    rejects(vec![range(4, 1, 1)], 1, "reaches word 4");
    // empty, or standing still
    rejects(vec![range(0, 0, 1)], 1, "at least one");
    rejects(vec![range(0, 2, 0)], 2, "at least one");
    // overflow: checked arithmetic, refused before any comparison
    rejects(vec![range(u64::MAX, 2, 1)], 2, "overflows");
    rejects(vec![range(0, u64::MAX, u64::MAX)], 2, "overflows");
    rejects(vec![range(1, u64::MAX, 1)], 2, "reaches word 18446744073709551615"); // 2^64-1 fits
    rejects(vec![range(2, u64::MAX, 1)], 2, "overflows");
    // the descriptor has no index; a one-word schema is not a leaf descriptor
    let with = |descriptor: &str| {
        PositionEntry::Range(PositionRange {
            commitment: 0,
            rank: RANK,
            position: RANK,
            descriptor: descriptor.to_string(),
            start: 0,
            count: 2,
            stride: 1,
        })
    };
    rejects(vec![with(&format!("{DESCRIPTOR}:0"))], 2, "not a packed/v1 leaf descriptor");
    rejects(vec![with("u16")], 2, "not a packed/v1 leaf descriptor");
    rejects(vec![with("packed/v1:verity-vllm/leaf/v1:u16:4")], 2, "not a packed/v1 leaf descriptor");
    // a range naming a commitment the obligation lacks
    let (mut foreign, _) = build(root, vec![range(0, 2, 1), single(2, None)], 3);
    if let PositionEntry::Range(r) = &mut foreign.obligations[0].positions[0] {
        r.commitment = 1;
    }
    assert!(parse_statement_v3(&encode_v3(&foreign)).unwrap_err().0.contains("commitment"));
    // a pin on a range is not expressible: the range record has no `expected` field, and
    // an entry tag other than 0/1 is refused
    let mut bad_tag = bytes.clone();
    let marker = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, RANK as u8]; // range tag, commitment 0, rank
    let at = bad_tag.windows(marker.len()).position(|w| w == marker).unwrap();
    bad_tag[at] = 2;
    assert!(parse_statement_v3(&bad_tag).unwrap_err().0.contains("entry tag"));
    // the slot count is the expanded count: a kind of two ports cannot take a two-word range plus a single
    let (short, _) = build(root, vec![range(0, 2, 1), single(2, None)], 2);
    assert!(parse_statement_v3(&encode_v3(&short)).is_err());
}

#[test]
fn a_format_3_statement_takes_a_packed_witness_only() {
    let value = row(1, 2);
    let (root, path) = tree(&value);
    let (statement, _) = build(root, vec![range(0, 2, 1), single(2, None)], 3);
    let v3 = AnyStatement::V3(statement.clone());
    assert!(check_any_statement(&v3, &AnyWitness::Packed(witness(value.clone(), path))).is_ok());
    let plain = Witness { obligations: vec![vec![Opening { value: value.clone(), path: vec![] }; 3]] };
    assert!(matches!(check_any_statement(&v3, &AnyWitness::Openings(plain)), Err(CheckError::WitnessShape(_))));
}
