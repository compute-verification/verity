import Definitions.Claims
import Definitions.Pouw.Dither

/-!
# The no-fold Dither protocol's work-soundness: its named premises (trusted)

The named premises of `Definitions.Pouw.Guarantees.Dither.WorkSound` besides the accelerator model (G1, G2, P1, P2). They
are the program's trust base (section 0) and nothing else:

* `SAKMatmul`: symmetric algorithmic knowledge for exact matrix products (R-ACCEL-6), on the round's words;
* `ShaSpeed`: SHA-512's implementation speed (R-ACCEL-7), on the round's commitment: under the random oracle, each
  oracle query costs at least the pinned public implementation's (ABSORB-THM's realization premise);
* `ServiceSound`: the service's verdict accepts only periods whose rounds' checks pass, but with probability `κ_S`
  (C-Flock sound and zero knowledge, live coins from the auditor's challenger);
* `ImportWithinCertifier`: an accepted period imported at most `K` bits, the network certifier's count (class L);
* `Extract`: commit-and-open costs what computing costs: each round's work splits into a words program and a
  commitment program on those words, and an accepted round's check passes on the words but with probability `κ_x`
  (ABSORB-THM's `extract_of_ro` is its probability half, from the oracle).

Only `SAKMatmul`, `ShaSpeed` and `Extract` say anything about which algorithms a prover runs.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Pouw.Dither

/-- `pouw-sak-matmul` (proposed; R-ACCEL-6, Daniel 25 Oct). **Symmetric algorithmic knowledge for exact matrix
products, SAK(matmul)**, on a round's words: in the model at its device premises, on every committed instance, every
words program on any state fixed before the round's coins has moment at most 1 at the prices `μ`, constant `c_mm`,
rate `η` and import weight `β` against the honest words' cost `H_mm`:
`E_σ[1_pass(σ) · e^{η (c_mm − (cost_μ + β · imported)/H_mm)₊}] ≤ 1`.

The headline reads it at `c_mm = pub/H_mm`, `pub` the public best matmul method's priced cost of the round's words
(RT-LEAN L3): no program computes the noised products' segment prefix sums (or all but a small fraction of the words
the check could draw) more cheaply at `μ` than the best public method. The state is advice: anything computed before
the round (the un-noised product `A B`, the periodic part of the noise) within (G1)'s memory, so `c_mm` must hold
against that precomputation (O-IO's lookup routes). -/
def SAKMatmul (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (cmm η β Hmm : ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ (a : P.Inst) (St : Type) (s : St) (Mw : G.Mc.Alg (St × P.Draw) P.Tr),
    wordsMoment G P μ cmm η β Hmm a s Mw ≤ 1

/-- `pouw-sha512-speed` (proposed; R-ACCEL-7, Daniel 25 Oct: "sure"). **Nobody computes SHA-512 meaningfully faster
than the best public implementation**, on a round's commitment: in the model at its device premises, at every oracle
and committed instance, every commitment program that hashes the words (`Mh` on the oracle and the words, RT-LEAN L24)
and computes the commitment of every transcript, beside any words program on any state, has moment at most 1 at the
prices `μ`, constant `c_sha`, rate `η` and import weight `β` against the honest commitment's cost `H_sha`.

The headline reads it at `c_sha = shaPub/H_sha`, `shaPub` the pinned public implementation's priced cost of the
round's commitment. Under the random oracle it is ABSORB-THM's realization: each oracle query costs at least the
public implementation's, so the honest prover meets it at `c_sha = 1`. MODEL-SPEC's θ floors (RT-MODEL M12, on
`Open.sync`, `b1` and `madHi`) are what keep the model from hashing faster than the card. -/
def ShaSpeed (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (csha η β Hsha : ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ (o : P.Orc) (a : P.Inst) (St : Type) (s : St) (Mw : G.Mc.Alg (St × P.Draw) P.Tr)
    (Mh : G.Mc.Alg (P.Orc × P.Tr) P.Root), (∀ T, G.Mc.eval Mh (o, T) = P.commit o T) →
    commitMoment G P μ csha η β Hsha o a s Mw Mh ≤ 1

/-- `pouw-nofold-service` (proposed; `pouw-rs-staged-service`'s form, per prover, on a period of rounds). **The proof
service's verdict accepts only periods whose rounds' checks pass**, but with probability `κ_S` over the oracle and
the verifier's coins, for every prover: the rounds' checks run inside C-Flock's zero-knowledge proofs (sound and zero
knowledge), on live coins drawn by the auditor's challenger after each round's commitment (`trusted-nebius-host`).
It assumes nothing of the prover's algorithms. -/
def ServiceSound (G : Accel) (P : RoundProto) (k : ℕ) (verdict : Verdict G P k) (κS : ℝ) : Prop :=
  ∀ A : Prover G P k, prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) =>
    verdict A ox.1 ox.2 ∧ ¬ A.Accepted ox.1 ox.2) ≤ κS

/-- `pouw-dither-import` (proposed; class L, the network certifier's count on `trusted-nebius-host`). **An accepted
period imported at most `K` bits**: the network certifier sits on the period's devices' links, on a machine the
prover doesn't control, and signs the bits that enter; the period's verdict reads its signed count and refuses a
period past `K`. Open (RT-PROTO P2, O-IO): the per-period form from the certifier's counts with `n_max · E_out`. -/
def ImportWithinCertifier (G : Accel) (P : RoundProto) (k : ℕ) (verdict : Verdict G P k) (K : ℝ) : Prop :=
  ∀ (A : Prover G P k) (o : P.Orc) (x : Fin k → P.Draw × P.Chk), verdict A o x → A.ext o x ≤ K

/-- `pouw-commit-extract` (proposed; the Accel-model instance of ASK-16's random-oracle card, with
`StraightLineExtraction`; ABSORB-THM's `extract_of_ro` proves its probability half from the oracle, and its
realization half is `ShaSpeed`'s). **Committing and opening costs what computing costs**: in the model at its device
premises, every prover's round `w` splits into a words program `Mw w` on the round's state and dither (no oracle:
the prover's hash calls done by the public implementation and logged) and a commitment program `Mh w` that hashes
those words, whose uses and imported bits sum to at most `work w`'s, and the probability that the rounds' checks
accept while some round's check fails on `Mw w`'s words is at most `κ_x`. Collision resistance gives binding, not
that the committed words and their Merkle root were computed; this premise gives that. -/
def Extract (G : Accel) (P : RoundProto) (k : ℕ) (κx : ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ A : Prover G P k,
    ∃ (Mw : Fin k → G.Mc.Alg (A.St × P.Draw) P.Tr) (Mh : Fin k → G.Mc.Alg (P.Orc × P.Tr) P.Root),
      (∀ w o T, G.Mc.eval (Mh w) (o, T) = P.commit o T) ∧
      (∀ o x w q, G.use (Mw w) (A.wordsIn o x w) q + G.use (Mh w) (o, G.Mc.eval (Mw w) (A.wordsIn o x w)) q ≤
        G.use (A.work w) (A.workIn o x w) q) ∧
      (∀ o x w, G.ext (Mw w) (A.wordsIn o x w) + G.ext (Mh w) (o, G.Mc.eval (Mw w) (A.wordsIn o x w)) ≤
        G.ext (A.work w) (A.workIn o x w)) ∧
      prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => A.Accepted ox.1 ox.2 ∧
        ∃ w, ¬ P.ok (A.instAt ox.1 ox.2 w) (ox.2 w).1 (ox.2 w).2 (G.Mc.eval (Mw w) (A.wordsIn ox.1 ox.2 w))) ≤ κx

end Pouw.Assumptions
