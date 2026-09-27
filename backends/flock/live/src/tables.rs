//! Multi-table statements (private-recursion's spec): several tables in one Flock union commitment, each a pinned template
//! (`A_t`, `B_t`, `C = I`) repeated over its blocks, with private glue between them (`glue`). The union proves every table's
//! relation (one zerocheck and one lincheck over `Σ_t sel_t ⊗ A_t`); the glue sumcheck runs after them and its claim joins the
//! one merged opening.
//!
//! A table's local index is its in-block bit `c` (`0..k`) and its block `j` (bits `k..k+ν`). The union lays slot `t` out
//! batch-major, so local bit `ℓ` is address bit `ℓ` (`ℓ < 7`, inside a word), `ℓ + ν` (the rest of `c`) or `7 + ℓ − k` (the block);
//! the address bits above the slot are its prefix.

use flock_core::{
    challenger::Challenger,
    lincheck::{LincheckCircuit, SparseMatrixCircuit, pack_z_lincheck},
    proof::{LinkClaim, R1csProofMergedLigerito},
    r1cs::SparseBinaryMatrix,
    verifier::{FlockVerifyError, verify_ligerito_union_glued},
};
use flock_field::F128;
use flock_prover::{
    pcs::{Commitment, PcsParams},
    prover::{UnionSlotProverInput, prove_fast_ligerito_union_glued},
    schedule::{Registry, TableType},
    union::UnionInstance,
};

use crate::glue::{self, GlueProof, Map, Region, Relation};

/// A table's template: `2^k_log` rows, rows `[0, n_in)` its inputs (glue destinations), `const_pos` the constant one, and every
/// other row below `useful` the AND of its `a` and `b` rows' XORs.
#[derive(Clone, Debug)]
pub struct Template {
    pub k_log: usize,
    pub n_in: usize,
    pub const_pos: usize,
    pub useful: usize,
    pub a: Vec<Vec<usize>>,
    pub b: Vec<Vec<usize>>,
}

impl Template {
    /// `(z, A z, B z)` of one block from its input bits.
    pub fn eval(&self, inputs: &[bool]) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
        let k = 1 << self.k_log;
        let (mut z, mut av, mut bv) = (vec![false; k], vec![false; k], vec![false; k]);
        for (i, &x) in inputs.iter().enumerate().take(self.n_in) {
            (z[i], av[i], bv[i]) = (x, x, x);
        }
        let c = self.const_pos;
        (z[c], av[c], bv[c]) = (true, true, true);
        for i in self.n_in..self.useful {
            if i == c {
                continue;
            }
            let x = self.a[i].iter().fold(false, |s, &j| s ^ z[j]);
            let y = self.b[i].iter().fold(false, |s, &j| s ^ z[j]);
            (av[i], bv[i], z[i]) = (x, y, x & y);
        }
        (z, av, bv)
    }

    fn matrix(&self, rows: &[Vec<usize>]) -> SparseBinaryMatrix {
        let k = 1 << self.k_log;
        let mut r: Vec<Vec<usize>> = (0..k).map(|i| if i < self.n_in || i == self.const_pos { vec![i] } else { Vec::new() }).collect();
        for i in self.n_in..self.useful {
            if i != self.const_pos {
                r[i] = rows[i].clone();
            }
        }
        SparseBinaryMatrix { num_rows: k, num_cols: k, rows: r.into() }
    }

    fn table_type(&self) -> TableType {
        let k = 1 << self.k_log;
        TableType {
            k_log: self.k_log,
            useful_bits: self.useful,
            a_0: self.matrix(&self.a),
            b_0: self.matrix(&self.b),
            c_0: SparseBinaryMatrix { num_rows: k, num_cols: k, rows: (0..k).map(|i| vec![i]).collect::<Vec<_>>().into() },
            const_pin: Some(self.const_pos),
            class: Default::default(),
            io_schema: Vec::new(),
        }
    }
}

/// A sub-cube of a table's local index: `fixed` bits and ordered `free` bits over `0..k_log + b` (its blocks' `2^b`); the local
/// block bits past `b` (the union's capacity `ν`) are zero.
#[derive(Clone, Debug)]
pub struct TableRegion {
    pub table: usize,
    pub fixed: Vec<(usize, bool)>,
    pub free: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Glue {
    pub dst: TableRegion,
    pub src: TableRegion,
    pub map: Map,
}

/// The statement: tables (sorted by `k_log` descending: the union's slot order), each with its block count, and the glue.
pub struct Tables {
    pub templates: Vec<Template>,
    pub counts: Vec<usize>,
    pub glue: Vec<Glue>,
    pub registry: Registry,
    pub nu: usize,
    circuits: Vec<SparseMatrixCircuit<'static>>,
    rels: Vec<Relation>,
}

fn leak<T>(x: T) -> &'static T {
    Box::leak(Box::new(x))
}

impl Tables {
    /// Load-time checks: slot order, counts within the capacity, every region inside its table, every destination an input
    /// row, no destination twice, and the relations well formed.
    pub fn new(templates: Vec<Template>, counts: Vec<usize>, glue: Vec<Glue>) -> Result<Self, String> {
        if templates.windows(2).any(|w| w[0].k_log < w[1].k_log) || templates.len() != counts.len() {
            return Err("tables: one count per table, in k_log-descending order".into());
        }
        let nu = counts.iter().map(|&n| n.max(8).next_power_of_two().trailing_zeros() as usize).max().unwrap_or(3);
        let registry = Registry::new(templates.iter().map(Template::table_type).collect(), nu);
        let circuits = templates.iter().map(|t| {
            let tt = t.table_type();
            SparseMatrixCircuit::new(leak(tt.a_0), leak(tt.b_0)).with_const_pin(Some(t.const_pos))
        }).collect();
        let mut s = Tables { templates, counts, glue, registry, nu, circuits, rels: Vec::new() };
        let m = s.registry.m_total();
        let mut dsts = std::collections::BTreeSet::new();
        let mut rels = Vec::new();
        for g in &s.glue {
            let (dst, src) = (s.region(&g.dst)?, s.region(&g.src)?);
            let t = &s.templates[g.dst.table];
            for x in 0..1usize << g.dst.free.len() {
                let local: usize = g.dst.fixed.iter().map(|&(b, v)| (v as usize) << b).sum::<usize>()
                    + g.dst.free.iter().enumerate().map(|(j, &b)| (x >> j & 1) << b).sum::<usize>();
                if local & ((1 << t.k_log) - 1) >= t.n_in {
                    return Err(format!("glue: destination row {} of table {} is not an input row", local & ((1 << t.k_log) - 1), g.dst.table));
                }
                if !dsts.insert((g.dst.table, local)) {
                    return Err("glue: a destination position is glued twice".into());
                }
            }
            rels.push(Relation { dst, src, map: g.map.clone() });
        }
        glue::check(&rels, m)?;
        s.rels = rels;
        Ok(s)
    }

    /// A table's region as a sub-cube of the union's address space.
    fn region(&self, r: &TableRegion) -> Result<Region, String> {
        let t = self.templates.get(r.table).ok_or("glue: no such table")?;
        let slot = self.registry.slots()[r.table];
        let (k, nu, m) = (t.k_log, self.nu, self.registry.m_total());
        let b = self.counts[r.table].max(1).next_power_of_two().trailing_zeros() as usize;
        let addr = |l: usize| if l < 7 { l } else if l < k { l + nu } else { 7 + l - k };
        let mut local_seen = vec![false; k + b];
        let (mut mask, mut fixed) = (0u64, 0u64);
        for &(l, v) in &r.fixed {
            if l >= k + b || std::mem::replace(&mut local_seen[l], true) {
                return Err("glue: a region's fixed bit is outside its table or repeated".into());
            }
            mask |= 1 << addr(l);
            fixed |= (v as u64) << addr(l);
        }
        let mut free = Vec::with_capacity(r.free.len());
        for &l in &r.free {
            if l >= k + b || std::mem::replace(&mut local_seen[l], true) {
                return Err("glue: a region's free bit is outside its table or repeated".into());
            }
            free.push(addr(l));
        }
        if local_seen.iter().any(|&x| !x) {
            return Err("glue: a region leaves a table bit neither fixed nor free".into());
        }
        for l in k + b..k + nu {
            mask |= 1 << addr(l);
        }
        for a in slot.m_slot..m {
            mask |= 1 << a;
            fixed |= ((slot.offset >> a) as u64 & 1) << a;
        }
        Ok(Region { mask, fixed, free })
    }

    pub fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.registry, self.counts.clone())
    }

    pub fn params(&self, profile: flock_core::pcs::ligerito::LigeritoProfile) -> PcsParams {
        let u = self.union();
        let m = u.dense_m();
        let batch = flock_core::pcs::ligerito::embedded_initial_k_or_default(m, profile);
        #[cfg(feature = "sha512")]
        let merkle_hash = flock_core::merkle::HashKind::Sha512;
        #[cfg(not(feature = "sha512"))]
        let merkle_hash = Default::default();
        PcsParams { m, log_inv_rate: profile.log_inv_rate(), log_batch_size: batch, profile, num_lanes: u.commit_lanes(batch), merkle_hash }
    }

    /// Every block's `(z, a, b)` from its inputs (`inputs[t][j]`), batch-major per slot, with its lincheck stripe.
    fn slot(&self, t: usize, blocks: &[Vec<bool>]) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let tp = &self.templates[t];
        let (k, nu) = (tp.k_log, self.nu);
        let words = 1usize << (k + nu - 7);
        let (mut z, mut a, mut b) = (vec![F128::ZERO; words], vec![F128::ZERO; words], vec![F128::ZERO; words]);
        let mut logical = vec![false; 1 << (k + nu)];
        let set = |v: &mut [F128], w: usize, bit: usize| if bit < 64 { v[w].lo |= 1 << bit } else { v[w].hi |= 1 << (bit - 64) };
        for (j, ins) in blocks.iter().enumerate() {
            let (zz, aa, bb) = tp.eval(ins);
            for c in 0..1usize << k {
                let w = ((c >> 7) << nu) + j;
                for (v, bits) in [(&mut z, &zz), (&mut a, &aa), (&mut b, &bb)] {
                    if bits[c] {
                        set(v, w, c & 127);
                    }
                }
                logical[(j << k) + c] = zz[c];
            }
        }
        (z, a, b, pack_z_lincheck(&logical, k + nu, k))
    }

    /// Table `t`'s block `j`, in-block bit `c`: its address in the union's witness.
    pub fn address(&self, t: usize, j: usize, c: usize) -> usize {
        self.registry.slots()[t].offset + ((((c >> 7) << self.nu) + j) << 7) + (c & 127)
    }

    /// The honest statement's proof from every table's blocks' inputs (`inputs[t][j]`, `counts[t]` blocks each).
    pub fn prove<Ch: Challenger>(&self, inputs: &[Vec<Vec<bool>>], profile: flock_core::pcs::ligerito::LigeritoProfile, ch: &mut Ch)
        -> (Commitment, R1csProofMergedLigerito, GlueProof) {
        self.prove_glue_over(inputs, profile, &|_| {}, ch)
    }

    /// [`Tables::prove`] with the glue sumcheck run over the committed witness as `alter` changes it (a cheating prover).
    fn prove_glue_over<Ch: Challenger>(&self, inputs: &[Vec<Vec<bool>>], profile: flock_core::pcs::ligerito::LigeritoProfile,
                                       alter: &dyn Fn(&mut Vec<F128>), ch: &mut Ch) -> (Commitment, R1csProofMergedLigerito, GlueProof) {
        let slots: Vec<UnionSlotProverInput<'_>> = (0..self.templates.len())
            .map(|t| UnionSlotProverInput::new(self.slot(t, &inputs[t]), &self.circuits[t] as &dyn LincheckCircuit))
            .collect();
        let m = self.registry.m_total();
        let mut gp = None;
        let (proof, commitment, _) = prove_fast_ligerito_union_glued(&self.union(), &self.params(profile), slots, &mut |_cm: &Commitment, z: &[F128], ch: &mut Ch| {
            let mut z = z.to_vec();
            alter(&mut z);
            let (p, rho) = glue::prove_packed(&self.rels, m, &z, ch);
            let claim = LinkClaim { point: rho, value: p.value };
            gp = Some(p);
            vec![claim]
        }, ch);
        (commitment, proof, gp.expect("the glue hook ran"))
    }

    pub fn verify<Ch: Challenger>(&self, commitment: &Commitment, proof: &R1csProofMergedLigerito, gp: &GlueProof,
                                  profile: flock_core::pcs::ligerito::LigeritoProfile, ch: &mut Ch) -> Result<(), FlockVerifyError> {
        let circs: Vec<&dyn LincheckCircuit> = self.circuits.iter().map(|c| c as &dyn LincheckCircuit).collect();
        let m = self.registry.m_total();
        verify_ligerito_union_glued(&self.union(), &circs, commitment, proof, &self.params(profile), &mut |_cm: &Commitment, ch: &mut Ch| {
            glue::verify(&self.rels, m, gp, ch).map(|(point, value)| vec![LinkClaim { point, value }])
        }, ch).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use flock_core::{challenger::FsChallenger, pcs::ligerito::LigeritoProfile};

    use super::*;

    const K: usize = 16;
    const BLOCKS: usize = 64;

    /// A `2^16`-row template (big enough for a registered Ligerito shape): inputs rows 0..31, the constant at 32, row `64 + j` =
    /// `x_j AND x_{j+s}` (j < 32), then filler gates to the end.
    fn template(s: usize) -> Template {
        let k = 1 << K;
        let (mut a, mut b) = (vec![Vec::new(); k], vec![Vec::new(); k]);
        for j in 0..32 {
            a[64 + j] = vec![j];
            b[64 + j] = vec![(j + s) % 32];
        }
        for i in 96..k {
            a[i] = vec![i - 1, i % 32];
            b[i] = vec![i - 2, 32];
        }
        Template { k_log: K, n_in: 32, const_pos: 32, useful: k, a, b }
    }

    /// Rows `base..base+32` of every block: in-block bits 0..4 and the block bits free, in-block bits 5..15 fixed to `base`.
    fn rows(t: usize, base: usize) -> TableRegion {
        TableRegion { table: t, fixed: (5..K).map(|l| (l, base >> l & 1 == 1)).collect(), free: (0..5).chain(K..K + 6).collect() }
    }

    /// Table 0 produces rows 64..95 of its 64 blocks; table 1 reads them as its inputs 0..31 (identity over rows and blocks).
    fn statement() -> Tables {
        let glue = vec![Glue { dst: rows(1, 0), src: rows(0, 64), map: Map::Id }];
        Tables::new(vec![template(1), template(3)], vec![BLOCKS, BLOCKS], glue).unwrap()
    }

    fn inputs(flip: Option<(usize, usize)>) -> Vec<Vec<Vec<bool>>> {
        let t0 = template(1);
        let mut s = 0x2545f4914f6cdd1du64;
        let p: Vec<Vec<bool>> = (0..BLOCKS).map(|_| (0..32).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s & 1 == 1 }).collect()).collect();
        let mut c: Vec<Vec<bool>> = p.iter().map(|ins| t0.eval(ins).0[64..96].to_vec()).collect();
        if let Some((j, i)) = flip {
            c[j][i] = !c[j][i];
        }
        vec![p, c]
    }

    fn run(st: &Tables, ins: &[Vec<Vec<bool>>]) -> Result<(), FlockVerifyError> {
        let (cm, proof, gp) = st.prove(ins, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test"));
        st.verify(&cm, &proof, &gp, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test"))
    }

    #[test]
    fn two_tables_with_private_glue_prove_and_verify() {
        let st = statement();
        run(&st, &inputs(None)).unwrap();
    }

    #[test]
    fn a_glued_bit_that_differs_is_refused_by_the_glue() {
        // each table's witness is honest for its own inputs; only the glue is broken
        let st = statement();
        assert_eq!(run(&st, &inputs(Some((2, 17)))), Err(FlockVerifyError::Glue));
    }

    #[test]
    fn a_glue_claim_off_the_committed_witness_is_refused_by_the_opening() {
        // the committed destination bit differs; the prover runs the glue on a copy with it repaired, so the sumcheck holds
        // and only the merged opening of z~(rho) can catch it
        let st = statement();
        let (j, i) = (2, 17);
        let at = st.address(1, j, i);
        let repair = move |z: &mut Vec<F128>| {
            let (w, t) = (at >> 7, at & 127);
            if t < 64 { z[w].lo ^= 1 << t } else { z[w].hi ^= 1 << (t - 64) }
        };
        let (cm, proof, gp) = st.prove_glue_over(&inputs(Some((j, i))), LigeritoProfile::Fast100, &repair, &mut FsChallenger::new(b"tables-test"));
        let got = st.verify(&cm, &proof, &gp, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test"));
        assert!(matches!(got, Err(FlockVerifyError::PcsOpen(_))), "{got:?}");
    }

    #[test]
    fn an_edited_glue_descriptor_is_refused() {
        let st = statement();
        let ins = inputs(None);
        let (cm, proof, gp) = st.prove(&ins, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test"));
        let mut other = statement();
        other.rels[0].map = Map::Perm((0..11).map(|j| if j < 2 { 1 - j } else { j }).collect());
        assert!(other.verify(&cm, &proof, &gp, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test")).is_err());
    }

    #[test]
    fn a_destination_that_is_not_an_input_row_is_refused_at_load() {
        let glue = vec![Glue { dst: rows(1, 64), src: rows(0, 64), map: Map::Id }];
        let err = Tables::new(vec![template(1), template(3)], vec![BLOCKS, BLOCKS], glue).err().unwrap();
        assert!(err.contains("not an input row"), "{err}");
    }
}
