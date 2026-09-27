//! Private glue between the tables of one committed witness (the multi-table statement's glue relations): each relation says
//! `z(D(x)) = z(S(π(x)))` for every `x ∈ {0,1}^f`, where `D` and `S` are aligned sub-cubes of the witness's `M`-bit address space
//! and `π` an identity, an index-bit permutation or a broadcast. One sumcheck proves every relation at once:
//!
//! - the verifier draws `γ` and `u` after the commitment; relation `g` is weighted `γ^(g+1)` and read at `u`'s first `f_g`
//!   coordinates, where its two sides are `z̃(P_D)` and `z̃(P_S)` for the points `P_D = D(u)` and `P_S = S(π(u))`;
//! - `Σ_g γ^(g+1) (z̃(P_D) − z̃(P_S)) = Σ_y z(y) W(y)` with `W(y) = Σ_g γ^(g+1) (eq(P_D, y) − eq(P_S, y))`, which is 0 for an honest
//!   witness;
//! - the sumcheck binds the address bits low first (degree 2 per round) and ends in one claim `z̃(ρ) = v`, which the caller
//!   opens with the others; the verifier evaluates `W̃(ρ)` itself in `O(#relations · M)`.
//!
//! Soundness: `#relations / |F|` from `γ`, `f_max / |F|` from `u`, `2M / |F|` from the rounds. The prover here is dense (vectors
//! of `2^M`); a sparse one follows `W`'s support.

use flock_core::challenger::Challenger;
use flock_field::F128;

pub const LABEL: &[u8] = b"verity/flock-glue/v1";

/// An aligned sub-cube of the address space: address bit `a` is `fixed`'s bit `a` where `mask` has it, else the region's free
/// bit `j` for `free[j] = a` (free bits in order; every address bit is one or the other).
#[derive(Clone, Debug)]
pub struct Region {
    pub mask: u64,
    pub fixed: u64,
    pub free: Vec<usize>,
}

/// `π`: `Id` (the source's free bit `j` is the destination's), `Perm(p)` (the source's free bit `p[j]` is the destination's free
/// bit `j`), `Broadcast(keep)` (the source's free bit `i` is the destination's free bit `keep[i]`; the other destination bits
/// read the same source value).
#[derive(Clone, Debug)]
pub enum Map {
    Id,
    Perm(Vec<usize>),
    Broadcast(Vec<usize>),
}

#[derive(Clone, Debug)]
pub struct Relation {
    pub dst: Region,
    pub src: Region,
    pub map: Map,
}

/// The prover's messages: per round `g(0)` and `g`'s leading coefficient, then the final claim `v = z̃(ρ)`.
#[derive(Clone, Debug, PartialEq)]
pub struct GlueProof {
    pub rounds: Vec<[F128; 2]>,
    pub value: F128,
}

fn bit_f(b: u64) -> F128 {
    if b & 1 == 1 { F128::ONE } else { F128::ZERO }
}

impl Region {
    fn check(&self, m: usize) -> Result<(), String> {
        let all = if m == 64 { u64::MAX } else { (1u64 << m) - 1 };
        let mut free = 0u64;
        for &a in &self.free {
            if a >= m || free >> a & 1 == 1 {
                return Err(format!("a free bit {a} is out of range or repeated"));
            }
            free |= 1 << a;
        }
        if free & self.mask != 0 || (free | self.mask) != all || self.fixed & !self.mask != 0 {
            return Err("a region's fixed and free bits must partition the address bits".into());
        }
        Ok(())
    }

    /// The point whose free coordinates are `vals` (in free-bit order) and whose fixed coordinates are the fixed bits.
    fn point(&self, m: usize, vals: &[F128]) -> Vec<F128> {
        let mut p: Vec<F128> = (0..m).map(|a| bit_f(self.fixed >> a)).collect();
        for (j, &a) in self.free.iter().enumerate() {
            p[a] = vals[j];
        }
        p
    }

    /// Every address of the sub-cube with `eq(point, address)`, the point's free coordinates `vals`.
    fn eq_entries(&self, vals: &[F128]) -> Vec<(usize, F128)> {
        let mut out = vec![(self.fixed as usize, F128::ONE)];
        for (j, &a) in self.free.iter().enumerate() {
            let (one, zero) = (vals[j], vals[j] + F128::ONE);
            out = out.into_iter().flat_map(|(addr, w)| [(addr, w * zero), (addr | 1 << a, w * one)]).collect();
        }
        out
    }
}

impl Relation {
    fn check(&self, m: usize) -> Result<(), String> {
        self.dst.check(m)?;
        self.src.check(m)?;
        let (fd, fs) = (self.dst.free.len(), self.src.free.len());
        match &self.map {
            Map::Id if fd == fs => Ok(()),
            Map::Perm(p) if p.len() == fd && fs == fd && {
                let mut s = p.clone();
                s.sort_unstable();
                s.iter().enumerate().all(|(i, &x)| i == x)
            } => Ok(()),
            Map::Broadcast(k) if k.len() == fs && k.iter().all(|&j| j < fd) => Ok(()),
            _ => Err("the map's arity does not match its regions".into()),
        }
    }

    /// The source's free coordinates at `u` (the destination's).
    fn src_vals(&self, u: &[F128]) -> Vec<F128> {
        match &self.map {
            Map::Id => u.to_vec(),
            Map::Perm(p) => {
                let mut v = vec![F128::ZERO; p.len()];
                for (j, &i) in p.iter().enumerate() {
                    v[i] = u[j];
                }
                v
            }
            Map::Broadcast(k) => k.iter().map(|&j| u[j]).collect(),
        }
    }
}

/// Every relation well formed over `m` address bits.
pub fn check(rels: &[Relation], m: usize) -> Result<(), String> {
    rels.iter().try_for_each(|r| r.check(m))
}

/// `(γ, u)`, drawn after the commitment (`u` has the widest relation's free-bit count).
fn draw<Ch: Challenger>(rels: &[Relation], ch: &mut Ch) -> (F128, Vec<F128>) {
    ch.observe_label(LABEL);
    let gamma = ch.sample_f128();
    let f = rels.iter().map(|r| r.dst.free.len()).max().unwrap_or(0);
    (gamma, (0..f).map(|_| ch.sample_f128()).collect())
}

fn round_value(e0: F128, e2: F128, claim: F128, r: F128) -> F128 {
    e0 + (claim + e2) * r + e2 * r * r
}

/// `W` on the whole address space (`2^m` entries).
fn weights(rels: &[Relation], m: usize, gamma: F128, u: &[F128]) -> Vec<F128> {
    let mut w = vec![F128::ZERO; 1 << m];
    let mut g = gamma;
    for r in rels {
        let ud = &u[..r.dst.free.len()];
        for (side, vals) in [(&r.dst, ud.to_vec()), (&r.src, r.src_vals(ud))] {
            for (addr, e) in side.eq_entries(&vals) {
                w[addr] += g * e;
            }
        }
        g *= gamma;
    }
    w
}

/// The prover: `z` is the witness's bits, one per address (`2^m`).
pub fn prove<Ch: Challenger>(rels: &[Relation], m: usize, z: &[bool], ch: &mut Ch) -> (GlueProof, Vec<F128>) {
    assert_eq!(z.len(), 1 << m);
    let (gamma, u) = draw(rels, ch);
    let mut w = weights(rels, m, gamma, &u);
    let mut zz: Vec<F128> = z.iter().map(|&b| if b { F128::ONE } else { F128::ZERO }).collect();
    let (mut rounds, mut rho) = (Vec::with_capacity(m), Vec::with_capacity(m));
    for _ in 0..m {
        let (mut e0, mut e2) = (F128::ZERO, F128::ZERO);
        for (zp, wp) in zz.chunks(2).zip(w.chunks(2)) {
            e0 += zp[0] * wp[0];
            e2 += (zp[1] + zp[0]) * (wp[1] + wp[0]);
        }
        ch.observe_f128(e0);
        ch.observe_f128(e2);
        rounds.push([e0, e2]);
        let r = ch.sample_f128();
        let fold = |v: &[F128]| -> Vec<F128> { v.chunks(2).map(|p| p[0] + (p[1] + p[0]) * r).collect() };
        zz = fold(&zz);
        w = fold(&w);
        rho.push(r);
    }
    ch.observe_f128(zz[0]);
    (GlueProof { rounds, value: zz[0] }, rho)
}

/// `W̃(ρ)` from the relations alone.
pub fn weight_at(rels: &[Relation], m: usize, gamma: F128, u: &[F128], rho: &[F128]) -> F128 {
    let eq = |p: &[F128]| p.iter().zip(rho).fold(F128::ONE, |acc, (&a, &r)| acc * (a * r + (a + F128::ONE) * (r + F128::ONE)));
    let mut g = gamma;
    let mut out = F128::ZERO;
    for r in rels {
        let ud = &u[..r.dst.free.len()];
        out += g * (eq(&r.dst.point(m, ud)) + eq(&r.src.point(m, &r.src_vals(ud))));
        g *= gamma;
    }
    out
}

/// The verifier: replays the rounds and returns the claim `(ρ, v)` the caller must open (`z̃(ρ) = v`); `Err` if the rounds do
/// not reduce `0` to `v · W̃(ρ)`.
pub fn verify<Ch: Challenger>(rels: &[Relation], m: usize, proof: &GlueProof, ch: &mut Ch) -> Result<(Vec<F128>, F128), String> {
    check(rels, m)?;
    if proof.rounds.len() != m {
        return Err("glue: one round per address bit".into());
    }
    let (gamma, u) = draw(rels, ch);
    let mut claim = F128::ZERO;
    let mut rho = Vec::with_capacity(m);
    for &[e0, e2] in &proof.rounds {
        ch.observe_f128(e0);
        ch.observe_f128(e2);
        let r = ch.sample_f128();
        claim = round_value(e0, e2, claim, r);
        rho.push(r);
    }
    ch.observe_f128(proof.value);
    if claim != proof.value * weight_at(rels, m, gamma, &u, &rho) {
        return Err("glue: the sumcheck does not reduce to the opened claim (a glued value differs)".into());
    }
    Ok((rho, proof.value))
}

#[cfg(test)]
mod tests {
    use flock_core::challenger::FsChallenger;

    use super::*;

    const M: usize = 10;

    fn region(fixed_hi: u64, lo_bits: usize, order: Vec<usize>) -> Region {
        // the low `lo_bits` address bits free (in `order`), the rest fixed to `fixed_hi`
        let mask = ((1u64 << M) - 1) & !((1u64 << lo_bits) - 1);
        Region { mask, fixed: fixed_hi << lo_bits & mask, free: order }
    }

    fn mle(z: &[bool], rho: &[F128]) -> F128 {
        let mut v: Vec<F128> = z.iter().map(|&b| if b { F128::ONE } else { F128::ZERO }).collect();
        for &r in rho {
            v = v.chunks(2).map(|p| p[0] + (p[1] + p[0]) * r).collect();
        }
        v[0]
    }

    /// Three relations over disjoint sub-cubes of 16 addresses: an identity, a bit permutation, a broadcast of 8 source bits to 16.
    fn statement() -> (Vec<Relation>, Vec<bool>) {
        let rels = vec![
            Relation { dst: region(1, 4, vec![0, 1, 2, 3]), src: region(10, 4, vec![0, 1, 2, 3]), map: Map::Id },
            Relation { dst: region(2, 4, vec![0, 1, 2, 3]), src: region(11, 4, vec![0, 1, 2, 3]), map: Map::Perm(vec![2, 0, 3, 1]) },
            Relation { dst: region(3, 4, vec![0, 1, 2, 3]), src: region(24, 3, vec![0, 1, 2]), map: Map::Broadcast(vec![1, 2, 3]) },
        ];
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut z: Vec<bool> = (0..1 << M).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s & 1 == 1 }).collect();
        for r in &rels {
            let f = r.dst.free.len();
            for x in 0..1usize << f {
                let y: usize = match &r.map {
                    Map::Id => x,
                    Map::Perm(p) => (0..f).map(|j| (x >> j & 1) << p[j]).sum(),
                    Map::Broadcast(k) => k.iter().enumerate().map(|(i, &j)| (x >> j & 1) << i).sum(),
                };
                let at = |reg: &Region, v: usize| reg.fixed as usize | reg.free.iter().enumerate().map(|(j, &a)| (v >> j & 1) << a).sum::<usize>();
                z[at(&r.dst, x)] = z[at(&r.src, y)];
            }
        }
        (rels, z)
    }

    fn run(rels: &[Relation], z: &[bool]) -> (Result<(Vec<F128>, F128), String>, Vec<F128>, GlueProof) {
        let (proof, rho) = prove(rels, M, z, &mut FsChallenger::new(b"glue-test"));
        (verify(rels, M, &proof, &mut FsChallenger::new(b"glue-test")), rho, proof)
    }

    #[test]
    fn honest_glue_reduces_to_the_witness_at_rho() {
        let (rels, z) = statement();
        check(&rels, M).unwrap();
        let (got, rho, proof) = run(&rels, &z);
        let (vr, v) = got.unwrap();
        assert_eq!(vr, rho);
        assert_eq!(v, mle(&z, &rho), "the claim to open is the witness's own z̃(ρ)");
        assert_eq!(proof.value, v);
    }

    #[test]
    fn a_flipped_glued_bit_is_refused() {
        let (rels, z) = statement();
        for at in [(1 << 4) + 5, (2 << 4) + 9, (3 << 4) + 14] {
            let mut bad = z.clone();
            bad[at] = !bad[at];
            assert!(run(&rels, &bad).0.is_err(), "flipping address {at} (a destination bit) must be refused");
        }
        let mut free = z.clone();
        free[(5 << 4) + 1] = !free[(5 << 4) + 1];
        assert!(run(&rels, &free).0.is_ok(), "an unglued bit is free");
    }

    #[test]
    fn a_false_opening_value_is_refused() {
        let (rels, z) = statement();
        let (mut proof, _) = prove(&rels, M, &z, &mut FsChallenger::new(b"glue-test"));
        proof.value += F128::ONE;
        assert!(verify(&rels, M, &proof, &mut FsChallenger::new(b"glue-test")).is_err());
    }

    #[test]
    fn malformed_relations_are_refused() {
        let (mut rels, _) = statement();
        rels[0].map = Map::Perm(vec![0, 0, 1, 2]);
        assert!(check(&rels, M).is_err());
        let (mut rels, _) = statement();
        rels[1].dst.mask &= !(1 << 9);
        assert!(check(&rels, M).is_err(), "an address bit neither fixed nor free");
    }
}
