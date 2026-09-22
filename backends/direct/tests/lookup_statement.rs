//! Statement binding of the LogUp realisation (PROTOCOL.md 14.6; the red team's F1-F3 carried from the limb
//! path): the zero batch claiming real words, a swapped / changed word, other FRI parameters, `degree_bits` 17
//! (2 VUs, table-height trace) vs 19 (4096 VUs), a substituted table row.  Pod-sized (the tables make the trace
//! 2^17 rows): `#[ignore]` locally, `cargo test --release --tests -- --include-ignored` on the pod.

use std::path::{Path, PathBuf};

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use verity_direct::air::{self, LookupChainAir, VuIo};
use verity_direct::lookup::{aux_matrix, substitute_table_row};
use verity_direct::lookup_stark::{lookup_statement, prove_lookup_bound, verify_lookup_bound, LookupProof};
use verity_direct::program::STEPS;
use verity_direct::stark::{BabyBear5Backend, BabyBearBackend, Backend, BoundError, FriParams, GoldilocksBackend};
use verity_direct::statement::Statement;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/bench-instances/v1")
}

fn read_u16(p: &Path) -> Vec<u64> {
    std::fs::read(p).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u64).collect()
}

/// Two real VUs (the negatives' operands with their correct words).
fn two_vus() -> Vec<VuIo> {
    let k = 16 * STEPS as usize;
    let x = read_u16(&root().join("vu-k1536-neg.x.u16"));
    let w = read_u16(&root().join("vu-k1536-neg.w.u16"));
    let y = read_u16(&root().join("vu-k1536-neg.y.u16"));
    (0..2).map(|i| VuIo { x: x[i * k..(i + 1) * k].to_vec(), w: w[i * k..(i + 1) * k].to_vec(), y16: y[i] - 1 }).collect()
}

const FRI: FriParams = FriParams { log_blowup: 2, num_queries: 8, proof_of_work_bits: 1 };

type Com<B> = <<<B as Backend>::SC as StarkGenericConfig>::Pcs as p3_commit::Pcs<<<B as Backend>::SC as StarkGenericConfig>::Challenge, <<B as Backend>::SC as StarkGenericConfig>::Challenger>>::Commitment;

/// Prove the LogUp trace of `vus` for the statement `stmt` (unchecked: the forgery attempts sign a trace that does
/// not match the statement).
fn prove_for<B: Backend>(air_: &LookupChainAir, lay: &air::Layout, vus: &[VuIo], stmt: &Statement) -> LookupProof<B::SC>
where
    Val<B::SC>: PrimeField64 + Send + Sync,
{
    let (main, queries, _): (RowMajorMatrix<Val<B::SC>>, Vec<u32>, _) = air::chain_trace_lookup(air_, vus, 2);
    let rows = main.height();
    let width = main.width();
    let pre: RowMajorMatrix<Val<B::SC>> = air_.lay.tables.preprocessed(rows);
    let mult: Vec<Val<B::SC>> = (0..rows).map(|r| main.values[r * width + width - 1]).collect();
    let llay = air_.lay.clone();
    let (proof, _) = prove_lookup_bound::<B>(air_, lay, FRI, stmt, pre, main, |z, beta| aux_matrix(&llay, &queries, &mult, rows, z, beta, true), false);
    proof
}

fn statement_suite<B: Backend>()
where
    Val<B::SC>: PrimeField64 + Send + Sync,
    Com<B>: PartialEq,
{
    let kind = B::KIND;
    let d = <<B::SC as StarkGenericConfig>::Challenge as p3_field::AbstractExtensionField<Val<B::SC>>>::D;
    let (air_, lay) = LookupChainAir::new(kind, d);
    let vus = two_vus();
    let real = lookup_statement(&air_, Statement::new(vus.iter().map(|v| v.y16).collect()));
    let table_bits = air_.lay.tables.total_rows.next_power_of_two().trailing_zeros() as usize;
    assert_eq!(real.degree_bits(), table_bits, "2 VUs: the tables set the height");

    // F1a: the zero batch signed as the real statement
    let zeros = vec![air::zero_vu(), air::zero_vu()];
    let zproof = prove_for::<B>(&air_, &lay, &zeros, &real);
    let r = verify_lookup_bound::<B>(&air_, &lay, &zproof, FRI, &real);
    assert!(matches!(r, Err(BoundError::StatementColumn { .. })), "zero batch accepted as the real statement: {r:?}");
    // the same zero trace is a fine proof of the zero statement
    let zero_stmt = lookup_statement(&air_, Statement::new(vec![0, 0]));
    let zproof = prove_for::<B>(&air_, &lay, &zeros, &zero_stmt);
    assert!(verify_lookup_bound::<B>(&air_, &lay, &zproof, FRI, &zero_stmt).is_ok());

    // honest proof of the real words
    let proof = prove_for::<B>(&air_, &lay, &vus, &real);
    assert!(verify_lookup_bound::<B>(&air_, &lay, &proof, FRI, &real).is_ok(), "honest proof rejected");
    // F1b: one word changed / words swapped
    let mut other = real.clone();
    other.words[0] ^= 1;
    assert!(matches!(verify_lookup_bound::<B>(&air_, &lay, &proof, FRI, &other), Err(BoundError::StatementColumn { .. })));
    let swapped = lookup_statement(&air_, Statement::new(vec![real.words[1], real.words[0]]));
    if swapped.words != real.words {
        assert!(matches!(verify_lookup_bound::<B>(&air_, &lay, &proof, FRI, &swapped), Err(BoundError::StatementColumn { .. })));
    }
    // F1c: other FRI parameters -> a different header -> every challenge differs -> the opening argument fails
    let mut fri2 = FRI;
    fri2.num_queries += 1;
    assert!(verify_lookup_bound::<B>(&air_, &lay, &proof, fri2, &real).is_err());
    // F2: the 2-VU proof for the 4096-VU verifier (degree_bits 19)
    let mut big = real.clone();
    big.words.resize(4096, 0);
    assert_eq!(big.degree_bits(), 19);
    let r = verify_lookup_bound::<B>(&air_, &lay, &proof, FRI, &big);
    assert!(matches!(r, Err(BoundError::DegreeBits { expected: 19, got }) if got == table_bits), "2-VU proof accepted for B = 4096: {r:?}");
    // F2 the other way: a statement whose rows() would be 8 bits in the limb realisation is table-height here,
    // so the proof's degree_bits must be the tables', not the words'
    let unpadded = Statement::new(real.words.clone());
    assert_eq!(unpadded.degree_bits(), 8);
    assert!(matches!(verify_lookup_bound::<B>(&air_, &lay, &proof, FRI, &unpadded), Err(BoundError::DegreeBits { expected: 8, .. })));
    // table substitution: a verifier holding a table set that differs in one entry rejects the honest proof
    let other_air = LookupChainAir { width: air_.width, kind: air_.kind, idx: air_.idx.clone(), lay: substitute_table_row(&air_.lay, 3), ext_degree: air_.ext_degree };
    let r = verify_lookup_bound::<B>(&other_air, &lay, &proof, FRI, &real);
    assert!(r.is_err(), "substituted-table verifier accepted the honest proof");
    // and a prover who proves against the substituted table cannot pass the honest verifier
    let bad_proof = prove_for::<B>(&other_air, &lay, &vus, &real);
    let r = verify_lookup_bound::<B>(&air_, &lay, &bad_proof, FRI, &real);
    assert!(r.is_err(), "proof against a substituted table accepted by the honest verifier");
}

#[test]
#[ignore]
fn lookup_statement_binding_babybear() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    statement_suite::<BabyBearBackend>();
}

#[test]
#[ignore]
fn lookup_statement_binding_babybear5() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    statement_suite::<BabyBear5Backend>();
}

#[test]
#[ignore]
fn lookup_statement_binding_goldilocks() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    statement_suite::<GoldilocksBackend>();
}

/// With `--bind-operands` the operand columns are public too (lookup realisation).
#[test]
#[ignore]
fn lookup_operand_binding_babybear() {
    if !root().join("vu-k1536-neg.x.u16").is_file() {
        eprintln!("fixture missing; skipped");
        return;
    }
    type B = BabyBearBackend;
    let (air_, lay) = LookupChainAir::new(B::KIND, 4);
    let vus = two_vus();
    let x: Vec<u64> = vus.iter().flat_map(|v| v.x.iter().copied()).collect();
    let w: Vec<u64> = vus.iter().flat_map(|v| v.w.iter().copied()).collect();
    let stmt = lookup_statement(&air_, Statement::with_operands(vus.iter().map(|v| v.y16).collect(), x.clone(), w.clone()));
    let proof = prove_for::<B>(&air_, &lay, &vus, &stmt);
    assert!(verify_lookup_bound::<B>(&air_, &lay, &proof, FRI, &stmt).is_ok());
    let mut x2 = x;
    x2[1536 + 7] ^= 1;
    let other = lookup_statement(&air_, Statement::with_operands(stmt.words.clone(), x2, w));
    let r = verify_lookup_bound::<B>(&air_, &lay, &proof, FRI, &other);
    assert!(matches!(r, Err(BoundError::StatementColumn { .. })), "{r:?}");
}

/// The header alone (no proof): the statement, table digest and lookup widths each change it.
#[test]
fn lookup_header_is_sensitive() {
    use p3_baby_bear::BabyBear;
    use verity_direct::statement::{lookup_header, table_digest};
    let (air_, _) = LookupChainAir::new(verity_direct::ctx::FieldKind::BabyBear, 4);
    let d0 = table_digest(&air_.lay.tables);
    let d1 = table_digest(&substitute_table_row(&air_.lay, 0).tables);
    assert_ne!(d0, d1);
    let h0 = lookup_header::<BabyBear>(6, 40, 4, &d0);
    assert_ne!(h0, lookup_header::<BabyBear>(6, 40, 4, &d1));
    assert_ne!(h0, lookup_header::<BabyBear>(6, 44, 4, &d0));
    assert_ne!(h0, lookup_header::<BabyBear>(6, 40, 5, &d0));
    // the lookup statement's rows are the tables', its header differs from the limb statement's
    let s = Statement::new(vec![1, 2]);
    let ls = lookup_statement(&air_, s.clone());
    assert!(ls.rows() >= air_.lay.tables.total_rows);
    assert_ne!(s.rows(), ls.rows());
}
