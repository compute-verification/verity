//! A three-stage uni-stark (PROTOCOL.md 14.3): preprocessed tables, main trace, lookup challenges `z, beta`,
//! auxiliary lookup columns, then the usual quotient / DEEP opening.  A fork of `p3_uni_stark::prove` /
//! `verify` (0.4.3) that adds the rounds; the PCS, the challenger and the FRI parameters are the section 12
//! backend's.  Still `NON_ZK_PROOF_DIAGNOSTIC`.

use std::iter;

use p3_air::{AirBuilder, BaseAir};
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{AbstractExtensionField, AbstractField, Field, PackedValue};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, PackedChallenge, PackedVal, StarkGenericConfig, Val, VerificationError};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use crate::lookup::LookupAirBuilder;

/// The AIR of the three-stage protocol: `eval` on a `LookupAirBuilder` (main + preprocessed + aux).
pub trait LookupAir<F: Field>: BaseAir<F> {
    fn preprocessed_width(&self) -> usize;
    fn aux_width(&self) -> usize;
    /// Maximum constraint degree (the quotient degree is one less).
    fn max_degree(&self) -> usize;
    fn eval_lookup<AB: LookupAirBuilder<F = F>>(&self, builder: &mut AB);
}

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<<SC as StarkGenericConfig>::Challenge, <SC as StarkGenericConfig>::Challenger>>::Commitment;
type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<<SC as StarkGenericConfig>::Challenge, <SC as StarkGenericConfig>::Challenger>>::Proof;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LookupProof<SC: StarkGenericConfig> {
    pub pre_commit: Com<SC>,
    pub main_commit: Com<SC>,
    pub aux_commit: Com<SC>,
    pub quotient_commit: Com<SC>,
    pub opened: LookupOpened<SC::Challenge>,
    pub opening_proof: PcsProof<SC>,
    pub degree_bits: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LookupOpened<C> {
    pub pre_local: Vec<C>,
    pub pre_next: Vec<C>,
    pub main_local: Vec<C>,
    pub main_next: Vec<C>,
    pub aux_local: Vec<C>,
    pub aux_next: Vec<C>,
    pub quotient_chunks: Vec<Vec<C>>,
}

/// Prover-side timings of the stages (seconds).
#[derive(Clone, Debug, Default, Serialize)]
pub struct StageTimes {
    pub commit_pre: f64,
    pub commit_main: f64,
    pub aux_build: f64,
    pub commit_aux: f64,
    pub quotient: f64,
    pub commit_quotient: f64,
    pub open: f64,
}

fn log_quotient_degree(max_degree: usize) -> usize {
    p3_util::log2_ceil_usize(max_degree.max(2) - 1)
}

/// Prove: `aux_fn(z, beta)` builds the auxiliary matrix after the lookup challenges are sampled.
pub fn prove_lookup<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    preprocessed: RowMajorMatrix<Val<SC>>,
    main: RowMajorMatrix<Val<SC>>,
    aux_fn: impl FnOnce(SC::Challenge, SC::Challenge) -> RowMajorMatrix<Val<SC>>,
) -> (LookupProof<SC>, StageTimes)
where
    SC: StarkGenericConfig,
    A: LookupAir<Val<SC>>,
{
    let mut times = StageTimes::default();
    let degree = main.height();
    assert_eq!(preprocessed.height(), degree);
    let log_degree = log2_strict_usize(degree);
    let log_quotient_degree = log_quotient_degree(air.max_degree());
    let quotient_degree = 1 << log_quotient_degree;
    let pcs = config.pcs();
    let trace_domain = pcs.natural_domain_for_degree(degree);

    let t = std::time::Instant::now();
    let (pre_commit, pre_data) = pcs.commit(vec![(trace_domain, preprocessed)]);
    times.commit_pre = t.elapsed().as_secs_f64();
    challenger.observe(pre_commit.clone());
    let t = std::time::Instant::now();
    let (main_commit, main_data) = pcs.commit(vec![(trace_domain, main)]);
    times.commit_main = t.elapsed().as_secs_f64();
    challenger.observe(main_commit.clone());
    let z: SC::Challenge = challenger.sample_ext_element();
    let beta: SC::Challenge = challenger.sample_ext_element();

    let t = std::time::Instant::now();
    let aux = aux_fn(z, beta);
    times.aux_build = t.elapsed().as_secs_f64();
    assert_eq!(aux.height(), degree);
    let t = std::time::Instant::now();
    let (aux_commit, aux_data) = pcs.commit(vec![(trace_domain, aux)]);
    times.commit_aux = t.elapsed().as_secs_f64();
    challenger.observe(aux_commit.clone());
    let alpha: SC::Challenge = challenger.sample_ext_element();

    let quotient_domain = trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));
    let pre_q = pcs.get_evaluations_on_domain(&pre_data, 0, quotient_domain);
    let main_q = pcs.get_evaluations_on_domain(&main_data, 0, quotient_domain);
    let aux_q = pcs.get_evaluations_on_domain(&aux_data, 0, quotient_domain);
    let t = std::time::Instant::now();
    let quotient_values = quotient_values::<SC, A, _, _, _>(air, trace_domain, quotient_domain, pre_q, main_q, aux_q, z, beta, alpha);
    times.quotient = t.elapsed().as_secs_f64();
    let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
    let quotient_chunks = quotient_domain.split_evals(quotient_degree, quotient_flat);
    let qc_domains = quotient_domain.split_domains(quotient_degree);
    let t = std::time::Instant::now();
    let (quotient_commit, quotient_data) = pcs.commit(qc_domains.into_iter().zip(quotient_chunks).collect::<Vec<_>>());
    times.commit_quotient = t.elapsed().as_secs_f64();
    challenger.observe(quotient_commit.clone());

    let zeta: SC::Challenge = challenger.sample();
    let zeta_next = trace_domain.next_point(zeta).unwrap();
    let t = std::time::Instant::now();
    let (opened_values, opening_proof) = {
        pcs.open(
            vec![
                (&pre_data, vec![vec![zeta, zeta_next]]),
                (&main_data, vec![vec![zeta, zeta_next]]),
                (&aux_data, vec![vec![zeta, zeta_next]]),
                (&quotient_data, (0..quotient_degree).map(|_| vec![zeta]).collect::<Vec<_>>()),
            ],
            challenger,
        )
    };
    times.open = t.elapsed().as_secs_f64();
    let opened = LookupOpened {
        pre_local: opened_values[0][0][0].clone(),
        pre_next: opened_values[0][0][1].clone(),
        main_local: opened_values[1][0][0].clone(),
        main_next: opened_values[1][0][1].clone(),
        aux_local: opened_values[2][0][0].clone(),
        aux_next: opened_values[2][0][1].clone(),
        quotient_chunks: opened_values[3].iter().map(|v| v[0].clone()).collect::<Vec<_>>(),
    };
    (LookupProof { pre_commit, main_commit, aux_commit, quotient_commit, opened, opening_proof, degree_bits: log_degree }, times)
}

#[allow(clippy::too_many_arguments)]
fn quotient_values<SC, A, M1, M2, M3>(air: &A, trace_domain: Domain<SC>, quotient_domain: Domain<SC>, pre: M1, main: M2, aux: M3, z: SC::Challenge, beta: SC::Challenge, alpha: SC::Challenge) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: LookupAir<Val<SC>>,
    M1: Matrix<Val<SC>> + Sync,
    M2: Matrix<Val<SC>> + Sync,
    M3: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);
    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_zeroifier.push(Val::<SC>::default());
    }
    let (pw, mw, aw) = (pre.width(), main.width(), aux.width());
    let zp = PackedChallenge::<SC>::from_f(z);
    let bp = PackedChallenge::<SC>::from_f(beta);
    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::<SC>::WIDTH)
        .flat_map_iter(|i_start| {
            let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;
            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_zeroifier = *PackedVal::<SC>::from_slice(&sels.inv_zeroifier[i_range.clone()]);
            let pack = |m: &dyn Fn(usize) -> Vec<PackedVal<SC>>, w: usize| RowMajorMatrix::new(iter::empty().chain(m(i_start)).chain(m(i_start + next_step)).collect::<Vec<_>>(), w);
            let pre_m = pack(&|i| pre.vertically_packed_row(i).collect::<Vec<_>>(), pw);
            let main_m = pack(&|i| main.vertically_packed_row(i).collect::<Vec<_>>(), mw);
            let aux_m = pack(&|i| aux.vertically_packed_row(i).collect::<Vec<_>>(), aw);
            let mut folder = ProverLookupFolder::<SC> { pre: pre_m, main: main_m, aux: aux_m, is_first_row, is_last_row, is_transition, z: zp, beta: bp, alpha, accumulator: PackedChallenge::<SC>::zero() };
            air.eval_lookup(&mut folder);
            let quotient = folder.accumulator * inv_zeroifier;
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                let quotient_value = (0..<SC::Challenge as AbstractExtensionField<Val<SC>>>::D).map(|coeff_idx| quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing]).collect::<Vec<_>>();
                SC::Challenge::from_base_slice(&quotient_value)
            })
        })
        .collect()
}

/// Verify; `preprocessed` is recomputed by the verifier (the tables are public) and its commitment compared.
/// Statement binding (`statement.rs`, red-team F1-F3): the caller has absorbed the header into `challenger`;
/// `expected_degree_bits` is the trace height the verifier derives from its statement (a proof of another
/// height is rejected before anything is read from it); `public` lists main columns whose values on every row
/// the verifier knows -- each is checked at `zeta` and `zeta g` against the verifier's own evaluation.
pub fn verify_lookup<SC, A>(config: &SC, air: &A, challenger: &mut SC::Challenger, proof: &LookupProof<SC>, preprocessed: RowMajorMatrix<Val<SC>>, expected_degree_bits: usize, public: &[(usize, Vec<Val<SC>>)]) -> Result<(), VerificationError>
where
    SC: StarkGenericConfig,
    A: LookupAir<Val<SC>>,
    Com<SC>: PartialEq,
    Val<SC>: p3_field::TwoAdicField + p3_field::PrimeField64,
{
    if proof.degree_bits != expected_degree_bits {
        return Err(VerificationError::InvalidProofShape);
    }
    let degree = 1usize << proof.degree_bits;
    let log_quotient_degree = log_quotient_degree(air.max_degree());
    let quotient_degree = 1 << log_quotient_degree;
    let pcs = config.pcs();
    let trace_domain = pcs.natural_domain_for_degree(degree);
    let quotient_domain = trace_domain.create_disjoint_domain(1 << (proof.degree_bits + log_quotient_degree));
    let quotient_chunks_domains = quotient_domain.split_domains(quotient_degree);

    let d = <SC::Challenge as AbstractExtensionField<Val<SC>>>::D;
    let o = &proof.opened;
    let shape_ok = o.pre_local.len() == air.preprocessed_width()
        && o.pre_next.len() == air.preprocessed_width()
        && o.main_local.len() == air.width()
        && o.main_next.len() == air.width()
        && o.aux_local.len() == air.aux_width()
        && o.aux_next.len() == air.aux_width()
        && o.quotient_chunks.len() == quotient_degree
        && o.quotient_chunks.iter().all(|qc| qc.len() == d);
    if !shape_ok {
        return Err(VerificationError::InvalidProofShape);
    }
    // the public tables
    if preprocessed.height() != degree {
        return Err(VerificationError::InvalidProofShape);
    }
    let (pre_commit, _) = pcs.commit(vec![(trace_domain, preprocessed)]);
    if pre_commit != proof.pre_commit {
        return Err(VerificationError::InvalidOpeningArgument);
    }

    challenger.observe(proof.pre_commit.clone());
    challenger.observe(proof.main_commit.clone());
    let z: SC::Challenge = challenger.sample_ext_element();
    let beta: SC::Challenge = challenger.sample_ext_element();
    challenger.observe(proof.aux_commit.clone());
    let alpha: SC::Challenge = challenger.sample_ext_element();
    challenger.observe(proof.quotient_commit.clone());
    let zeta: SC::Challenge = challenger.sample();
    let zeta_next = trace_domain.next_point(zeta).unwrap();

    // the statement's public columns at zeta and zeta g
    for (col, want) in public {
        if want.len() != degree || *col >= air.width() {
            return Err(VerificationError::InvalidProofShape);
        }
        let (w0, w1) = crate::statement::eval_pair::<Val<SC>, SC::Challenge>(want, zeta);
        if o.main_local[*col] != w0 || o.main_next[*col] != w1 {
            return Err(VerificationError::OodEvaluationMismatch);
        }
    }

    pcs.verify(
        vec![
            (proof.pre_commit.clone(), vec![(trace_domain, vec![(zeta, o.pre_local.clone()), (zeta_next, o.pre_next.clone())])]),
            (proof.main_commit.clone(), vec![(trace_domain, vec![(zeta, o.main_local.clone()), (zeta_next, o.main_next.clone())])]),
            (proof.aux_commit.clone(), vec![(trace_domain, vec![(zeta, o.aux_local.clone()), (zeta_next, o.aux_next.clone())])]),
            (proof.quotient_commit.clone(), quotient_chunks_domains.iter().zip(&o.quotient_chunks).map(|(domain, values)| (*domain, vec![(zeta, values.clone())])).collect::<Vec<_>>()),
        ],
        &proof.opening_proof,
        challenger,
    )
    .map_err(|_| VerificationError::InvalidOpeningArgument)?;

    let zps = quotient_chunks_domains
        .iter()
        .enumerate()
        .map(|(i, domain)| {
            quotient_chunks_domains
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other_domain)| other_domain.zp_at_point(zeta) * other_domain.zp_at_point(domain.first_point()).inverse())
                .product::<SC::Challenge>()
        })
        .collect::<Vec<_>>();
    let quotient = o
        .quotient_chunks
        .iter()
        .enumerate()
        .map(|(ch_i, ch)| ch.iter().enumerate().map(|(e_i, &c)| zps[ch_i] * SC::Challenge::monomial(e_i) * c).sum::<SC::Challenge>())
        .sum::<SC::Challenge>();

    let sels = trace_domain.selectors_at_point(zeta);
    fn pair<'b, C: Clone + Send + Sync>(a: &'b [C], b: &'b [C]) -> ViewPair<'b, C> {
        VerticalPair::new(RowMajorMatrixView::new_row(a), RowMajorMatrixView::new_row(b))
    }
    let mut folder = VerifierLookupFolder::<SC> {
        pre: pair(&o.pre_local, &o.pre_next),
        main: pair(&o.main_local, &o.main_next),
        aux: pair(&o.aux_local, &o.aux_next),
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        z,
        beta,
        alpha,
        accumulator: SC::Challenge::zero(),
    };
    air.eval_lookup(&mut folder);
    if folder.accumulator * sels.inv_zeroifier != quotient {
        return Err(VerificationError::OodEvaluationMismatch);
    }
    Ok(())
}

// --- folders -----------------------------------------------------------------------------------------------------

pub struct ProverLookupFolder<SC: StarkGenericConfig> {
    pub pre: RowMajorMatrix<PackedVal<SC>>,
    pub main: RowMajorMatrix<PackedVal<SC>>,
    pub aux: RowMajorMatrix<PackedVal<SC>>,
    pub is_first_row: PackedVal<SC>,
    pub is_last_row: PackedVal<SC>,
    pub is_transition: PackedVal<SC>,
    pub z: PackedChallenge<SC>,
    pub beta: PackedChallenge<SC>,
    pub alpha: SC::Challenge,
    pub accumulator: PackedChallenge<SC>,
}

impl<SC: StarkGenericConfig> AirBuilder for ProverLookupFolder<SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = RowMajorMatrix<PackedVal<SC>>;
    fn main(&self) -> Self::M {
        self.main.clone()
    }
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert_eq!(size, 2);
        self.is_transition
    }
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: PackedVal<SC> = x.into();
        self.accumulator *= PackedChallenge::<SC>::from_f(self.alpha);
        self.accumulator += x;
    }
}

impl<SC: StarkGenericConfig> LookupAirBuilder for ProverLookupFolder<SC> {
    type EF = PackedChallenge<SC>;
    fn preprocessed(&self) -> Self::M {
        self.pre.clone()
    }
    fn aux(&self) -> Self::M {
        self.aux.clone()
    }
    fn challenges(&self) -> (Self::EF, Self::EF) {
        (self.z, self.beta)
    }
    fn ext_from_base(&self, x: Self::Expr) -> Self::EF {
        PackedChallenge::<SC>::from_base(x)
    }
    fn ext_from_cols(&self, cols: &[Self::Var]) -> Self::EF {
        PackedChallenge::<SC>::from_base_slice(cols)
    }
    fn assert_zero_ext(&mut self, x: Self::EF) {
        self.accumulator *= PackedChallenge::<SC>::from_f(self.alpha);
        self.accumulator += x;
    }
    fn ext_degree(&self) -> usize {
        <SC::Challenge as AbstractExtensionField<Val<SC>>>::D
    }
}

type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

pub struct VerifierLookupFolder<'a, SC: StarkGenericConfig> {
    pub pre: ViewPair<'a, SC::Challenge>,
    pub main: ViewPair<'a, SC::Challenge>,
    pub aux: ViewPair<'a, SC::Challenge>,
    pub is_first_row: SC::Challenge,
    pub is_last_row: SC::Challenge,
    pub is_transition: SC::Challenge,
    pub z: SC::Challenge,
    pub beta: SC::Challenge,
    pub alpha: SC::Challenge,
    pub accumulator: SC::Challenge,
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierLookupFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type M = ViewPair<'a, SC::Challenge>;
    fn main(&self) -> Self::M {
        self.main
    }
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert_eq!(size, 2);
        self.is_transition
    }
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: SC::Challenge = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<'a, SC: StarkGenericConfig> LookupAirBuilder for VerifierLookupFolder<'a, SC> {
    type EF = SC::Challenge;
    fn preprocessed(&self) -> Self::M {
        self.pre
    }
    fn aux(&self) -> Self::M {
        self.aux
    }
    fn challenges(&self) -> (Self::EF, Self::EF) {
        (self.z, self.beta)
    }
    fn ext_from_base(&self, x: Self::Expr) -> Self::EF {
        x
    }
    fn ext_from_cols(&self, cols: &[Self::Var]) -> Self::EF {
        cols.iter().enumerate().map(|(i, &c)| c * SC::Challenge::monomial(i)).sum()
    }
    fn assert_zero_ext(&mut self, x: Self::EF) {
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
    fn ext_degree(&self) -> usize {
        <SC::Challenge as AbstractExtensionField<Val<SC>>>::D
    }
}

pub fn proof_bytes_lookup<SC: StarkGenericConfig>(proof: &LookupProof<SC>) -> usize {
    bincode::serialize(proof).expect("serialise proof").len()
}
