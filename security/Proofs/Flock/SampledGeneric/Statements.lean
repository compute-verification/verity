module

public import Proofs.Flock.SampledGeneric.Defs

/-!
# The sampled audit as a compiler, generic in the proof system (stated in the game model)

Proof material, listed in no lock: the statement is the claims core's (`Claims.SamplingGeneric`).

Take any proof system for proof blocks (`ProofSystem`) with a knowledge extractor (`Extractor`, `Trials`), a binding
commitment to the trace (`Binding`) and a sampling design (`Audit.Law`). The sampled audit (`Audit.audit`: the prover
registers, the verifier samples proof blocks with live coins after the registration, the proof system's session proves
the sample) accepts a trace with incorrect set `F` with probability at most the sampling design's acceptance
probability `P_a(F)`, plus the knowledge error at one sampled proof block, plus the binding term `ε_bind`, which is
ecr/sha-512's collision term at the extractor's link finders:
$$\Pr[\text{accept}]\le P_a(F)+\varepsilon_{\rm ks}+\varepsilon_{\rm bind}.$$
`F` is the incorrect set of the committed trace, a function of the prover's state at registration, so it is fixed
before the verifier's first coin.

* `GenericSampledKS`: the event form, `Pr[accept ∧ E(X, S)] ≤ Pr_S[E(X, S) ∧ S ∩ F = ∅] + ε_ks + ε_bind`;
* `GenericSampledKSPa`: `Pr[accept] ≤ P_a(F) + ε_ks + ε_bind`, with `P_a(F) = L.escape F`;
* `GenericSampledKSEscape`: `Pr[accept ∧ W ⊆ F] ≤ P_a(W) + ε_ks + ε_bind` at every set `W` of proof blocks;
* `GenericSampledKSCount`: `Pr[accept ∧ |F| ≥ K] ≤ miss(K) + ε_ks + ε_bind`, the count curve;
* `GenericSampledKSJoint`: the same for an extractor in the textbook form (`JointExtractor`), with the knowledge error
  `κ` of its definition, no slack, and the committed trace the plurality of its trials;
* `GenericSampledZK`: the audit's view is `(ε_zk + ε_hide)`-close to a simulator's that knows no trace, from the proof
  system's zero knowledge and the commitment's hiding; `FlockZk.lean` gives the leakage form (`ZKBelow`,
  `audit_below`) the ZK server's theorem has.

Flock is an instance (`SampledGeneric/Flock.lean`), at a bound no worse than `zk_flock_countC_regC`'s, the count curve
`SampledKSLiveV5` runs through.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric
open scoped ENNReal

/-- **The sampled audit's knowledge soundness, generic in the proof system** (event form). For a proof system `PS`
whose extractor's two terms cover acceptance at every sampled proof block incorrect in a trace (`Extractor.Covers`), with
trials that open the registered commitment (`Trials.Opens`), carry the binding error (`Trials.Transfers`) and define the
committed trace `Xc` as a plurality of what they read (`Trials.Plurality`): at every prover `σ` whose link finders satisfy
ecr/sha-512 (`Trials.CR`), and every event `E` of the committed trace and the sample,
$$\Pr[\text{accept}\wedge E(X,S)]\le\Pr_S[E(X,S)\wedge S\cap F(X)=\emptyset]+\varepsilon_{\rm ks}(\sigma)
+\varepsilon_{\rm bind}(\sigma),$$
with `X = Xc(σ)` fixed at registration, `F(X)` its incorrect set, `ε_ks` the knowledge term at one sampled proof block
averaged over the sample (`Extractor.ksAvg`) and `ε_bind` the binding term (`Trials.bindErr`). -/
def GenericSampledKS : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : Law n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B) (Tr : Trials L PS B)
    (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) (M : ℕ),
    Ex.Covers Correct → Tr.Opens → Tr.WellFormed → Tr.Transfers Ex → Tr.Plurality Xc → 1 ≤ M →
    ∀ (σ : Strategy (audit L Reg PS.session)) (E : B.Trace → Finset (Fin n) → Prop),
      Tr.CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true ∧ E (Xc (reg σ) (cont σ)) o.2) (audit L Reg PS.session) σ ≤
        prCoin (fun ω => E (Xc (reg σ) (cont σ)) (L.draw ω) ∧
          Disjoint (L.draw ω) (incorrect B Correct (reg σ) (Xc (reg σ) (cont σ)))) +
        Ex.ksAvg L (reg σ) (cont σ) + Tr.bindErr (reg σ) (cont σ) M

/-- **The sampled audit's knowledge soundness, generic in the proof system**:
$$\Pr[\text{accept}]\le P_a(F)+\varepsilon_{\rm ks}(\sigma)+\varepsilon_{\rm bind}(\sigma),$$
`GenericSampledKS` at the sure event, with `P_a(F) = L.escape F` the sampling design's acceptance probability at the
committed trace's incorrect set. -/
def GenericSampledKSPa : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : Law n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B) (Tr : Trials L PS B)
    (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) (M : ℕ),
    Ex.Covers Correct → Tr.Opens → Tr.WellFormed → Tr.Transfers Ex → Tr.Plurality Xc → 1 ≤ M →
    ∀ σ : Strategy (audit L Reg PS.session), Tr.CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true) (audit L Reg PS.session) σ ≤
        L.escape (incorrect B Correct (reg σ) (Xc (reg σ) (cont σ))) + Ex.ksAvg L (reg σ) (cont σ) +
          Tr.bindErr (reg σ) (cont σ) M

/-- **The escape bound at a set of proof blocks, generic in the proof system**: for every set `W` of proof blocks,
`Pr[accept ∧ W ⊆ F] ≤ P_a(W) + ε_ks(σ) + ε_bind(σ)`, `GenericSampledKS` at `W ⊆ F`. -/
def GenericSampledKSEscape : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : Law n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B) (Tr : Trials L PS B)
    (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) (M : ℕ),
    Ex.Covers Correct → Tr.Opens → Tr.WellFormed → Tr.Transfers Ex → Tr.Plurality Xc → 1 ≤ M →
    ∀ (W : Finset (Fin n)) (σ : Strategy (audit L Reg PS.session)), Tr.CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true ∧ W ⊆ incorrect B Correct (reg σ) (Xc (reg σ) (cont σ)))
          (audit L Reg PS.session) σ ≤
        L.escape W + Ex.ksAvg L (reg σ) (cont σ) + Tr.bindErr (reg σ) (cont σ) M

/-- **The count curve, generic in the proof system**: `Pr[accept ∧ |F| ≥ K] ≤ miss(K) + ε_ks(σ) + ε_bind(σ)`, with
`miss(K)` the sampling design's largest acceptance probability over incorrect sets of at least `K` proof blocks. -/
def GenericSampledKSCount : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : Law n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B) (Tr : Trials L PS B)
    (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) (M : ℕ),
    Ex.Covers Correct → Tr.Opens → Tr.WellFormed → Tr.Transfers Ex → Tr.Plurality Xc → 1 ≤ M →
    ∀ (K : ℕ) (σ : Strategy (audit L Reg PS.session)), Tr.CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true ∧ K ≤ (incorrect B Correct (reg σ) (Xc (reg σ) (cont σ))).card)
          (audit L Reg PS.session) σ ≤
        L.miss K + Ex.ksAvg L (reg σ) (cont σ) + Tr.bindErr (reg σ) (cont σ) M

/-- **The textbook form**: for a knowledge extractor that is a game after the judged session (`JointExtractor`), whose
witnesses are sound for the proof blocks' statements (`JointExtractor.Sound`) and which fails to extract with
probability at most `κ` jointly with acceptance (`JointExtractor.KnowledgeSound`),
$$\Pr[\text{accept}]\le P_a(F)+\mathbb E_S\max_{u\in S}\kappa_u+\varepsilon_{\rm bind}(\sigma),$$
with `F` the incorrect set of the plurality of its trials (`Trials.plur`) and `ε_bind` with no slack
(`JointExtractor.toTrials`). -/
def GenericSampledKSJoint : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : Law n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (J : JointExtractor PS B)
    (κ : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → ℝ≥0∞) (M : ℕ),
    J.Judged → J.Sound Correct → (∀ S R o, 0 ≤ J.cost S R o) → J.KnowledgeSound κ → 1 ≤ M →
    ∀ σ : Strategy (audit L Reg PS.session), (J.toTrials L).CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true) (audit L Reg PS.session) σ ≤
        L.escape (incorrect B Correct (reg σ) ((J.toTrials L).plur (reg σ) (cont σ))) +
          avg (fun ω => ⨆ u ∈ L.draw ω, κ (L.draw ω) (reg σ) (cont σ ω) u) +
          (J.toTrials L).bindErr (reg σ) (cont σ) M

/-- **The sampled audit's zero knowledge, generic in the proof system.** For a proof system whose honest prover's view
is `ε_zk`-close to its simulator's at every statement in its relation (`Zk.ZK`), and a commitment that is
`ε_hide`-hiding (`Hider.Hiding`): against every verifier `v`, which samples (`samp`) after seeing the commitment, the
audit's view at any trace `X` whose openings are witnesses of every sample's statement is `(ε_zk + ε_hide)`-close to the
simulator's at any fixed trace `X₀` (`auditSim`), which reads nothing of `X`. -/
def GenericSampledZK : Prop :=
  ∀ (n : ℕ) (Tr Rnd Com Wit VS View : Type) [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)
    (Z : Zk (Com × Finset (Fin n)) Wit VS View) (Rel : Com × Finset (Fin n) → Wit → Prop) (εzk εh : ℝ≥0∞)
    (samp : VS → Com → Finset (Fin n)) (wit : Tr → Rnd → Finset (Fin n) → Wit),
    Z.ZK Rel (fun _ => εzk) → H.Hiding εh →
    ∀ X : Tr, (∀ r S, Rel (H.commit X r, S) (wit X r S)) →
    ∀ (X₀ : Tr) (v : VS), Close (auditView H Z samp wit X v) (auditSim H Z samp X₀ v) (εzk + εh)

end Flock.Guarantees
