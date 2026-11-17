module

public import Proofs.Flock.SampledGeneric.Defs

/-! The sampled audit as a compiler, generic in the proof system, stated in the game model; proof material, listed in no lock: the statement is the claims core's (`Claims.SamplingGeneric`). -/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric
open scoped ENNReal

def GenericSampledKS : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : SamplingDesign n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B) (Tr : Trials L PS B)
    (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) (M : ℕ),
    Ex.Covers Correct → Tr.Opens → Tr.WellFormed → Tr.Transfers Ex → Tr.Plurality Xc → 1 ≤ M →
    ∀ (σ : Strategy (audit L Reg PS.session)) (E : B.Trace → Finset (Fin n) → Prop),
      Tr.CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true ∧ E (Xc (reg σ) (cont σ)) o.2) (audit L Reg PS.session) σ ≤
        prCoin (fun ω => E (Xc (reg σ) (cont σ)) (L.draw ω) ∧
          Disjoint (L.draw ω) (incorrect B Correct (reg σ) (Xc (reg σ) (cont σ)))) +
        Ex.ksAvg L (reg σ) (cont σ) + Tr.bindErr (reg σ) (cont σ) M

def GenericSampledKSPa : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : SamplingDesign n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B) (Tr : Trials L PS B)
    (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) (M : ℕ),
    Ex.Covers Correct → Tr.Opens → Tr.WellFormed → Tr.Transfers Ex → Tr.Plurality Xc → 1 ≤ M →
    ∀ σ : Strategy (audit L Reg PS.session), Tr.CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true) (audit L Reg PS.session) σ ≤
        L.missProb (incorrect B Correct (reg σ) (Xc (reg σ) (cont σ))) + Ex.ksAvg L (reg σ) (cont σ) +
          Tr.bindErr (reg σ) (cont σ) M

def GenericSampledKSEscape : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : SamplingDesign n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B) (Tr : Trials L PS B)
    (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) (M : ℕ),
    Ex.Covers Correct → Tr.Opens → Tr.WellFormed → Tr.Transfers Ex → Tr.Plurality Xc → 1 ≤ M →
    ∀ (W : Finset (Fin n)) (σ : Strategy (audit L Reg PS.session)), Tr.CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true ∧ W ⊆ incorrect B Correct (reg σ) (Xc (reg σ) (cont σ)))
          (audit L Reg PS.session) σ ≤
        L.missProb W + Ex.ksAvg L (reg σ) (cont σ) + Tr.bindErr (reg σ) (cont σ) M

def GenericSampledKSCount : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : SamplingDesign n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B) (Tr : Trials L PS B)
    (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) (M : ℕ),
    Ex.Covers Correct → Tr.Opens → Tr.WellFormed → Tr.Transfers Ex → Tr.Plurality Xc → 1 ≤ M →
    ∀ (K : ℕ) (σ : Strategy (audit L Reg PS.session)), Tr.CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true ∧ K ≤ (incorrect B Correct (reg σ) (Xc (reg σ) (cont σ))).card)
          (audit L Reg PS.session) σ ≤
        L.miss K + Ex.ksAvg L (reg σ) (cont σ) + Tr.bindErr (reg σ) (cont σ) M

def GenericSampledKSJoint : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : SamplingDesign n) (PS : ProofSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (J : JointExtractor PS B)
    (κ : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → ℝ≥0∞) (M : ℕ),
    J.Judged → J.Sound Correct → (∀ S R o, 0 ≤ J.cost S R o) → J.KnowledgeSound κ → 1 ≤ M →
    ∀ σ : Strategy (audit L Reg PS.session), (J.toTrials L).CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true) (audit L Reg PS.session) σ ≤
        L.missProb (incorrect B Correct (reg σ) ((J.toTrials L).plur (reg σ) (cont σ))) +
          avg (fun ω => ⨆ u ∈ L.draw ω, κ (L.draw ω) (reg σ) (cont σ ω) u) +
          (J.toTrials L).bindErr (reg σ) (cont σ) M

def GenericSampledZK : Prop :=
  ∀ (n : ℕ) (Tr Rnd Com Wit VS View : Type) [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)
    (Z : Zk (Com × Finset (Fin n)) Wit VS View) (Rel : Com × Finset (Fin n) → Wit → Prop) (εzk εh : ℝ≥0∞)
    (samp : VS → Com → Finset (Fin n)) (wit : Tr → Rnd → Finset (Fin n) → Wit),
    Z.ZK Rel (fun _ => εzk) → H.Hiding εh →
    ∀ X : Tr, (∀ r S, Rel (H.commit X r, S) (wit X r S)) →
    ∀ (X₀ : Tr) (v : VS), Close (auditView H Z samp wit X v) (auditSim H Z samp X₀ v) (εzk + εh)

end Flock.Guarantees
