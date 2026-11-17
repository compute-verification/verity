module

public import Definitions.Claims.RecursionGeneric
public import Proofs.Flock.RecGeneric.Outer
public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.SampledGeneric.Statements

/-! A claims-core proof system through the sampled compiler, the one bridge between the claims core's verifiers and the sampled audit's games; definitions, no proofs. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims FlockSoundness FlockSoundness.Game

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

def sessionOf (H : Msg → Msg) (V : Verifier Coin) (x : Msg) : FlockSoundness.Game Bool :=
  .send (Prover Unit) fun P => .coin (Fin (V.k x) → Coin) fun r => .ret (decide ((run H P V x () r).out = .acc))

def proverOf {H : Msg → Msg} {V : Verifier Coin} {x : Msg} (τ : Strategy (sessionOf H V x)) : Prover Unit := τ.1

variable {n : ℕ} {Reg Dh : Type} {Hc : List UInt8 → Dh}

abbrev proofSystemOf (H : Msg → Msg) (V : Verifier Coin) (stmt : Finset (Fin n) → Reg → Msg) :
    FlockSoundness.SampledGeneric.ProofSystem n Reg :=
  ⟨fun S R => sessionOf H V (stmt S R)⟩

def trialCalls (H : Msg → Msg) (P : Prover Unit) (V : Verifier Coin) (E : Claims.Extractor Coin W V) (x : Msg)
    (r₀ : Fin (V.k x) → Coin) (c : E.C x) : ℕ :=
  (run H P V x () r₀).pLog.length + (run H P V x () r₀).vLog.length + (E.eval H P x () r₀ c).2.1

noncomputable def jointOf (H : Msg → Msg) (V : Verifier Coin) (Rl : Rel W) (E : Claims.Extractor Coin W V)
    (stmt : Finset (Fin n) → Reg → Msg) (B : FlockSoundness.SampledGeneric.Binding Reg Hc)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) :
    FlockSoundness.SampledGeneric.JointExtractor (proofSystemOf H V stmt) B where
  Wit := Option W × ℕ
  Rel S R _ w := Rl.ext (stmt S R) w.1
  reads := reads
  opens S R u w c := (w.1.map fun w' => opens S R u w' c).getD default
  G S R := .send (Prover Unit) fun P => .coin (Fin (V.k (stmt S R)) → Coin) fun r₀ =>
    .coin (E.C (stmt S R)) fun c => .ret (decide ((run H P V (stmt S R) () r₀).out = .acc),
      fun u => if u ∈ S then some ((E.eval H P (stmt S R) () r₀ c).1, trialCalls H P V E (stmt S R) r₀ c) else none)
  strat _ _ τ := ⟨proverOf τ, fun _ _ => PUnit.unit⟩
  cost _ _ o := ((Finset.univ.sup fun u => ((o.2 u).map Prod.snd).getD 0 : ℕ) : ℝ)

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric
open scoped ENNReal

/-- `E` is any rerun program and may depend on `H`; the trials' cost counts its `H` calls and its reruns', not its
computation between them, so `F` is the committed trace's incorrect set only at a counted `E`. -/
def GenericSampledKSClaims : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Verifier Coin) (Rl : Rel W)
    (E : Claims.Extractor Coin W V) (H : Claims.Msg → Claims.Msg) (err : Claims.Msg → Prover Unit → ℝ) (n : ℕ)
    (Reg Dh : Type)
    (Hc : List UInt8 → Dh) (L : SamplingDesign n) (B : Binding Reg Hc) (Correct : Reg → B.Trace → Fin n → Prop)
    (stmt : Finset (Fin n) → Reg → Claims.Msg)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) (M : ℕ),
    (∀ S R (P : Prover Unit), PrBad H V Rl E (stmt S R) P ≤ err (stmt S R) P) →
    (jointOf H V Rl E stmt B reads opens).Sound Correct → 1 ≤ M →
    ∀ σ : Strategy (audit L Reg (proofSystemOf H V stmt).session),
      ((jointOf H V Rl E stmt B reads opens).toTrials L).CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true) (audit L Reg (proofSystemOf H V stmt).session) σ ≤
        L.missProb (incorrect B Correct (reg σ)
            (((jointOf H V Rl E stmt B reads opens).toTrials L).plur (reg σ) (cont σ))) +
          avg (fun ω => ⨆ u ∈ L.draw ω,
            ENNReal.ofReal (err (stmt (L.draw ω) (reg σ)) (proverOf (x := stmt (L.draw ω) (reg σ)) (cont σ ω)))) +
          ((jointOf H V Rl E stmt B reads opens).toTrials L).bindErr (reg σ) (cont σ) M

def RecursiveSampledKS : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (I : Proto Coin W)
    (Sv : Server) (K : Claims.RecGeneric.Comp Coin) (mI : Fin I.nT → ℕ) (Mo : ℕ) (n : ℕ) (Reg Dh : Type)
    (Hc : List UInt8 → Dh) (L : SamplingDesign n) (B : Binding Reg Hc) (Correct : Reg → B.Trace → Fin n → Prop)
    (stmt : Finset (Fin n) → Reg → Claims.Msg)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) (M : ℕ),
    I.WF → I.KSAbortLin H → I.LastCoinFree H → Sv.Binds →
    (∀ S R, K.OuterKSAt H (stmt S R) ∧ 0 ≤ K.κO (stmt S R) ∧ 0 ≤ K.γO (stmt S R) ∧
      K.Contract H (Sv.IVStar I.V I.ans) (fun v => I.opens (Sv.openView v)) (stmt S R)) →
    (jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).Sound Correct → 1 ≤ M →
    ∀ σ : Strategy (audit L Reg (proofSystemOf H K.V stmt).session),
      ((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials L).CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true) (audit L Reg (proofSystemOf H K.V stmt).session) σ ≤
        L.missProb (incorrect B Correct (reg σ)
            (((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials L).plur (reg σ) (cont σ))) +
          avg (fun ω => ⨆ u ∈ L.draw ω,
            ENNReal.ofReal (K.err I Sv H (proverOf (x := stmt (L.draw ω) (reg σ)) (cont σ ω))
              (stmt (L.draw ω) (reg σ)) mI Mo)) +
          ((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials L).bindErr (reg σ) (cont σ) M

end Flock.Guarantees
