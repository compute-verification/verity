//! The cross-implementation vectors of the Lean specification
//! (`lean/vectors/*.json`, written by `lean/scripts/export_vectors.py` from
//! the Python reference and checked by `lake exe vectors` in `lean/`).
//!
//! Three implementations, one vector file per family: the Rust interpreter
//! must agree with the Python reference and the Lean specification on every
//! gate evaluation of `veritor.int8-q7@1` (`q7.json`, with the exhaustive
//! `mul8` table) and of `veritor.tensor-core.ampere_bf16_m16n8k16@2`
//! (`tc-ampere-bf16.json`: exact, rounding, overflow, sticky-infinity,
//! subnormal and non-finite cases; `tc-ampere-bf16-a100.json`: the words the
//! numerics lane measured on the A100), on the verdict of every single
//! obligation and on the verdict of every batch.
//!
//! The numerical check is exercised through the crate's own pipeline: each
//! vector batch is *encoded* as a canonical statement (the layout of
//! `codec.rs`; synthetic commitments and positions, since only the values and
//! the shape matter to `check_relations`), parsed with `parse_statement` -- so
//! the parser's well-formedness checks decide malformed programs -- and then
//! checked with `check_relations` on the vector's values, which stand for the
//! decoded openings.  Authentication (`check_openings*`) is out of scope of
//! these vectors, exactly as it is out of scope of the Lean `checkReference`.

use serde::Deserialize;
use veritor_zk_common::{check_relations, gate_set, parse_statement, GateSet, STATEMENT_MAGIC};

#[derive(Deserialize)]
struct Vectors {
    gate_set: GateSetInfo,
    evaluations: Vec<Evaluation>,
    /// Only the `q7` file carries the exhaustive product table.
    #[serde(default)]
    mul8_table: Vec<Vec<u64>>,
    #[serde(default)]
    programs: Vec<Program>,
    #[serde(default)]
    claims: Vec<Claim>,
    #[serde(default)]
    batches: Vec<Batch>,
}

/// The vector files, relative to `lean/vectors/`.
const FILES: &[&str] = &["q7.json", "tc-ampere-bf16.json", "tc-ampere-bf16-a100.json"];

#[derive(Deserialize)]
struct GateSetInfo {
    id: String,
    digest: String,
    width: u32,
}

#[derive(Deserialize)]
struct Evaluation {
    op: String,
    args: Vec<u64>,
    out: Option<u64>,
}

#[derive(Deserialize)]
struct Program {
    name: String,
    ports: Vec<u32>,
    gates: Vec<Gate>,
    outputs: Vec<u32>,
    well_formed: bool,
}

#[derive(Deserialize)]
struct Gate {
    op: String,
    args: Vec<(String, u32)>,
}

#[derive(Deserialize)]
struct Claim {
    name: String,
    program: usize,
    inputs: Vec<u64>,
    opened: Vec<u64>,
    accept: bool,
}

#[derive(Deserialize)]
struct Batch {
    name: String,
    programs: Vec<usize>,
    /// `(local program index, claim index)`.
    obligations: Vec<(usize, usize)>,
    accept: bool,
}

fn load(file: &str) -> Vectors {
    let path = format!("{}/../../../fixtures/lean-vectors/{file}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e} (run lean/scripts/export_vectors.py)"));
    serde_json::from_str(&text).expect("valid vector file")
}

fn unhex32(text: &str) -> [u8; 32] {
    assert_eq!(text.len(), 64, "a digest is 32 bytes of hex");
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("hex");
    }
    out
}

// -- a minimal canonical encoder (the inverse of `codec::parse_statement`) ------------

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }
    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_be_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_be_bytes());
    }
    fn digest(&mut self, digest: &[u8; 32]) {
        self.0.extend_from_slice(digest);
    }
    fn string(&mut self, text: &str) {
        self.u32(text.len() as u32);
        self.0.extend_from_slice(text.as_bytes());
    }
}

/// The kind digest of the batch's `index`-th program: strictly increasing in
/// the index, as the statement requires of its kinds.
fn kind_digest(index: usize) -> [u8; 32] {
    let mut digest = [0u8; 32];
    digest[24..].copy_from_slice(&(index as u64).to_be_bytes());
    digest
}

fn write_program(writer: &mut Writer, index: usize, program: &Program) {
    writer.digest(&kind_digest(index));
    writer.u32(program.gates.len() as u32);
    // the statement's ports are strictly increasing ordinals; the vector file
    // records the ports' widths instead, which the Rust checker learns from
    // the leaf schemas -- here only their number matters
    writer.u32(program.ports.len() as u32);
    for ordinal in 0..program.ports.len() {
        writer.u32(ordinal as u32);
    }
    writer.u32(program.gates.len() as u32);
    for gate in &program.gates {
        writer.string(&gate.op);
        writer.u32(gate.args.len() as u32);
        for (space, value) in &gate.args {
            writer.u8(match space.as_str() {
                "port" => 0,
                "local" => 1,
                other => panic!("unknown argument space {other}"),
            });
            writer.u32(*value);
        }
    }
    writer.u32(program.outputs.len() as u32);
    for offset in &program.outputs {
        writer.u32(*offset);
    }
}

/// One obligation whose positions are the claim's inputs followed by its
/// opened values, all under one synthetic commitment.
fn write_obligation(writer: &mut Writer, unit: usize, kind: usize, claim: &Claim) {
    let positions = claim.inputs.len() + claim.opened.len();
    writer.digest(&[0u8; 32]); // session
    writer.digest(&[0u8; 32]); // compiled
    writer.u64(unit as u64);
    writer.u64(0); // replay unit
    writer.digest(&kind_digest(kind));
    writer.u32(1); // one commitment
    writer.u64(2); // owner + 2
    writer.digest(&[0u8; 32]); // domain id
    writer.digest(&[0u8; 32]); // root
    writer.u64(positions as u64); // count
    writer.u32(positions as u32);
    for rank in 0..positions {
        writer.u32(0); // commitment
        writer.u64(rank as u64); // rank
        writer.u64(rank as u64); // position
        writer.string("u32"); // schema (unused: values are supplied decoded)
        writer.u8(0); // no expected value
    }
    writer.u32(claim.inputs.len() as u32);
    for slot in 0..claim.inputs.len() {
        writer.u32(slot as u32);
    }
    writer.u32(claim.opened.len() as u32);
    for slot in claim.inputs.len()..positions {
        writer.u32(slot as u32);
    }
}

/// The canonical statement of a batch of the vector file, and the decoded
/// values of its obligations.
fn encode_batch(
    vectors: &Vectors,
    programs: &[usize],
    obligations: &[(usize, usize)],
) -> (Vec<u8>, Vec<Vec<u64>>) {
    let mut writer = Writer::default();
    writer.raw(STATEMENT_MAGIC);
    writer.string(&vectors.gate_set.id);
    writer.digest(&unhex32(&vectors.gate_set.digest));
    writer.u32(vectors.gate_set.width);
    writer.u32(programs.len() as u32);
    for (index, &program) in programs.iter().enumerate() {
        write_program(&mut writer, index, &vectors.programs[program]);
    }
    writer.u32(obligations.len() as u32);
    let mut decoded = Vec::with_capacity(obligations.len());
    for (unit, &(kind, claim)) in obligations.iter().enumerate() {
        let claim = &vectors.claims[claim];
        write_obligation(&mut writer, unit, kind, claim);
        let mut values = claim.inputs.clone();
        values.extend_from_slice(&claim.opened);
        decoded.push(values);
    }
    (writer.0, decoded)
}

/// The verdict of the numerical check on a batch: the statement parses, its
/// gate set resolves to the pinned semantics, and every relation holds.
fn verdict(vectors: &Vectors, programs: &[usize], obligations: &[(usize, usize)]) -> bool {
    let (bytes, decoded) = encode_batch(vectors, programs, obligations);
    let Ok(statement) = parse_statement(&bytes) else {
        return false;
    };
    let set = gate_set(&statement).expect("the vector file names the pinned gate set");
    check_relations(&statement, &decoded, &set).is_ok()
}

fn resolve(vectors: &Vectors) -> GateSet {
    let set = GateSet::resolve(&vectors.gate_set.id, vectors.gate_set.width).expect("gate set");
    assert_eq!(set.digest(), unhex32(&vectors.gate_set.digest), "gate set digest");
    set
}

#[test]
fn evaluations_match_the_reference() {
    for file in FILES {
        let vectors = load(file);
        let set = resolve(&vectors);
        let mut rejected = 0;
        for Evaluation { op, args, out } in &vectors.evaluations {
            // `GateSet::evaluate` is never called on a source (`check_relations`
            // takes a source's value from its opening); an unknown op is rejected
            let got = match set.op(op) {
                Some(op) if !op.is_source() => set.evaluate(op, args),
                _ => None,
            };
            assert_eq!(&got, out, "{file}: {op}{args:?}");
            if out.is_none() {
                rejected += 1;
            }
        }
        eprintln!(
            "{file}: evaluations: {} checked ({rejected} rejected by the reference)",
            vectors.evaluations.len()
        );
    }
}

#[test]
fn mul8_table_matches_the_reference() {
    let vectors = load("q7.json");
    let set = resolve(&vectors);
    let mul8 = set.op("mul8").expect("mul8");
    assert_eq!(vectors.mul8_table.len(), 256);
    for (a, row) in vectors.mul8_table.iter().enumerate() {
        assert_eq!(row.len(), 256);
        for (b, &expected) in row.iter().enumerate() {
            assert_eq!(set.evaluate(mul8, &[a as u64, b as u64]), Some(expected), "mul8({a}, {b})");
        }
    }
    eprintln!("mul8_table: 65536 checked");
}

#[test]
fn well_formed_programs_parse() {
    // The parser enforces part of the Lean `WellFormed` (earlier-only and
    // in-range references, in-range strictly increasing outputs); the rest
    // (arity, opened sources, known ops) is enforced by `check_relations` per
    // obligation, so only the sound direction is a parser property.
    for file in FILES {
        let vectors = load(file);
        let mut parsed = 0;
        for (index, program) in vectors.programs.iter().enumerate() {
            let (bytes, _) = encode_batch(&vectors, &[index], &[]);
            let ok = parse_statement(&bytes).is_ok();
            if program.well_formed {
                assert!(ok, "{file}: well-formed program {} must parse", program.name);
            }
            parsed += ok as usize;
        }
        eprintln!("{file}: programs: {} checked ({parsed} parse)", vectors.programs.len());
    }
}

#[test]
fn claims_match_the_reference() {
    for file in FILES {
        let vectors = load(file);
        let mut accepted = 0;
        for (index, claim) in vectors.claims.iter().enumerate() {
            let got = verdict(&vectors, &[claim.program], &[(0, index)]);
            assert_eq!(got, claim.accept, "{file}: claim {}", claim.name);
            accepted += got as usize;
        }
        eprintln!("{file}: claims: {} checked ({accepted} accepted)", vectors.claims.len());
    }
}

#[test]
fn batches_match_the_reference() {
    for file in FILES {
        let vectors = load(file);
        for batch in &vectors.batches {
            let got = verdict(&vectors, &batch.programs, &batch.obligations);
            assert_eq!(got, batch.accept, "{file}: batch {}", batch.name);
            eprintln!(
                "{file}: batch {}: {} programs, {} obligations, verdict {got}",
                batch.name,
                batch.programs.len(),
                batch.obligations.len()
            );
        }
        eprintln!("{file}: batches: {} checked", vectors.batches.len());
    }
}
