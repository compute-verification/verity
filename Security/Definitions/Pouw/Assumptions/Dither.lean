import Definitions.Claims
import Definitions.Pouw.Dither

/-!
# The no-fold Dither protocol's work-soundness: its named premises (trusted)

The named premises of `Definitions.Pouw.Guarantees.Dither.WorkSound` besides the accelerator model (G1, G2, P1, P2). They
are the program's trust base (section 0) and nothing else:

* `SAKMatmul`: symmetric algorithmic knowledge for exact matrix products (R-ACCEL-6), on the round's words;
* `ShaSpeed`: SHA-512's implementation speed (R-ACCEL-7), on the round's commitment;
* `ServiceSound`: the proof service is sound per prover: C-Flock sound and zero knowledge, SHA-512 collision
  resistance only through each adversary's finders, live coins from the auditor's challenger;
* `ImportWithinCertifier`: an accepted period imported at most `K` bits, the network certifier's count (class L);
* `Extract`: commit-and-open costs what computing costs: an accepted run's rounds, which compute the words and their
  commitments, are pieces of the run. Collision resistance gives binding, not this.

Only `SAKMatmul`, `ShaSpeed` and `Extract` say anything about which algorithms a prover runs.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Pouw.Dither

/-- `pouw-sak-matmul` (proposed; R-ACCEL-6, Daniel 25 Oct). **Symmetric algorithmic knowledge for exact matrix
products, SAK(matmul)**, on a round's words: in the model at its device premises, on every committed instance, every
words program fixed before the round's coins (it computes the words from the first coin alone) has moment at most 1
at the prices `μ`, constant `c_mm`, rate `η` and import weight `β` against the honest words' cost `H_mm`:
`E_σ[1_pass(σ) · e^{η (c_mm − (cost_μ + β · imported)/H_mm)₊}] ≤ 1`.

At `c_mm = pub/H_mm`, `pub` the public best matmul method's cost of the round's words, it says that no program computes
the noised products' segment prefix sums (or all but a small fraction of the words the check could draw) more cheaply
at `μ` than the best public method: the words are prefix sums of an exact matrix product. The program is chosen after
the instance, so it may carry anything computed from the instance before the round (the un-noised product `A B`, say)
within (G1)'s memory; the noise is drawn after the instance, so `c_mm` must hold against that precomputation. The
public best's tensor-core cost is the ring count (BET-K-DISCHARGE's `RingKCount`), so `pub` is a model quantity.
"Alongside other work" (section 0, 4a) is read through `Extract`'s cut: the rounds the analysis cuts out of a run are
pieces of it, so the other work runs beside them. -/
def SAKMatmul (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (cmm η β Hmm : ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ (a : P.Inst) (Mw : G.Mc.Alg P.Draw P.Tr), wordsMoment G P μ cmm η β Hmm a Mw ≤ 1

/-- `pouw-sha512-speed` (proposed; R-ACCEL-7, Daniel 25 Oct: "sure"). **Nobody computes SHA-512 meaningfully faster
than the best public implementation**, on a round's commitment: in the model at its device premises, on every
committed instance, every commitment program fixed before the round's coins, beside any words program, computing the
commitment (the SHA-512 Merkle root over the leaves) of that program's words on every first coin, has moment at most 1
at the prices `μ`, constant `c_sha`, rate `η` and import weight `β` against the honest commitment's cost `H_sha`:
`E_σ[1_pass(σ) · e^{η (c_sha − (cost_μ + β · imported)/H_sha)₊}] ≤ 1`, `pass` the check on the words.

At `c_sha = shaPub/H_sha`, `shaPub` the public best implementation's cost of the round's commitment, it says that
committing to the words the check accepts takes the public best's hashing. The words are noised, so a table of a few
words' leaves passes the check only with the probability of guessing them; the moment's rate `η` prices that luck. -/
def ShaSpeed (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (csha η β Hsha : ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ (a : P.Inst) (Mw : G.Mc.Alg P.Draw P.Tr) (Mh : G.Mc.Alg P.Draw P.Root),
    (∀ d, G.Mc.eval Mh d = P.commit (G.Mc.eval Mw d)) → commitMoment G P μ csha η β Hsha a Mw Mh ≤ 1

/-- `pouw-nofold-service` (proposed; `pouw-rs-staged-service`'s form per prover, on a period of rounds). **The proof
service is sound up to its proof layer, each break bounded through the adversary's own collision finders.** The
service's seed is the period's live coins (`e`): `R` rounds, each a first coin (the noise) and a check (the words
drawn), drawn by the auditor's challenger after the round's commitments (live coins on `trusted-nebius-host`). For
every adversary in `G` that commits first and every value `r` of its own coins:
- round `w`'s binding break has probability at most `δb + βb · Pr[fb finds a collision of sha]`, its zero-knowledge
  break `δz + βz · Pr[fz …]` (C-Flock's binding and zero knowledge; SHA-512 commitments, each word a Merkle leaf);
- every run the verifier accepts is accepted by the play's game, every round's check passing on the words its words
  program computes, or breaks binding or zero knowledge in some round.

It assumes nothing of SHA-512 (`ecr/sha-512` enters only the bound, `breakBound`) and nothing of the adversary's
algorithms: a round program is any program of `G`, fixed before its round's coins. What the rounds cost beside the run
is `Extract`'s. -/
def ServiceSound (G : Accel) (S : Service) (P : RoundProto) {R : ℕ} (run : PeriodMap G S P R)
    (e : S.Seed ≃ Paths (P.Draw × P.Chk) R) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders G.Mc S R)
    (δb βb δz βz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : G.Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (G.Mc.eval M (σ, r)).1 = (G.Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb + βb * (fb Ω' M r w).Pr sha) ∧
      (∀ w, prob ((run Ω' M r).ZK w) ≤ δz + βz * (fz Ω' M r w).Pr sha) ∧
      ∀ x, S.verify (G.Mc.eval M (e.symm x, r)).1 (e.symm x) (G.Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-dither-import` (proposed; class L, the network certifier's count on `trusted-nebius-host`). **An accepted
period imported at most `K` bits.** The network certifier sits on the period's devices' links, on a machine the prover
doesn't control, and signs the bits that enter; the period's verdict reads its signed count and refuses a period past
`K`. So every run the verifier accepts read at most `K` bits from outside the period's devices. -/
def ImportWithinCertifier (G : Accel) (S : Service) (K : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : G.Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)) (ω : S.Seed × Ω'),
    S.verify (G.Mc.eval M ω).1 ω.1 (G.Mc.eval M ω).2 = true → G.ext M ω ≤ K

/-- `pouw-commit-extract` (proposed; open with Daniel: ABSORB-THM writes its discharge from SHA-512 as a random
oracle, each query priced at the public best's speed, R-ACCEL-3 and R-ACCEL-7). **Committing and opening costs what
computing costs**: in the model at its device premises, the analysis `run` cuts every adversary that commits first
into a play whose every round's commitment program computes the commitment to its words program's words
(`PeriodPlay.Computes`), and on every run the verifier accepts, the rounds' uses of each resource sum to at most the
run's, and their imported bits to at most the run's.

SHA-512's collision resistance gives binding (the opened words are the committed ones), not that the committed words
and their Merkle root were computed: a root produced some cheaper way, then opened lazily where the check draws, is not
excluded by it. This premise excludes it. The cut is at the commitments, so the rounds are disjoint stretches of the
period's devices' time (round `w`'s commitment is sent before round `w + 1`'s first coin), and the rest of the run is
its other work. -/
def Extract (G : Accel) (S : Service) (P : RoundProto) {R : ℕ} (run : PeriodMap G S P R)
    (e : S.Seed ≃ Paths (P.Draw × P.Chk) R) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 →
    ∀ (Ω' : Type) [Fintype Ω'] (M : G.Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
      (∀ σ σ' r, (G.Mc.eval M (σ, r)).1 = (G.Mc.eval M (σ', r)).1) → ∀ r : Ω', (run Ω' M r).play.Computes ∧
        ∀ x, S.verify (G.Mc.eval M (e.symm x, r)).1 (e.symm x) (G.Mc.eval M (e.symm x, r)).2 = true →
          (∀ q, (run Ω' M r).play.use x q ≤ G.use M (e.symm x, r) q) ∧ (run Ω' M r).play.ext x ≤ G.ext M (e.symm x, r)

end Pouw.Assumptions
