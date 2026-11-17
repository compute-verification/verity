module

public import Proofs.Flock.C3.Compose.Decoder
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.C3.Compose.Plays

/-! `SrcOK`'s decode fixed point: a run is the decode of its own steps where each stage plays on them to its
output. -/

@[expose] public section

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.InnerAccepts (firstSteps)
open FlockC3.PubRows (holoSteps_eq stagesOf firstSteps_length proverAt_some coinAt_some)
open FlockC3.HoloValues (repAt repsAt repStages_decodeRun)
open FlockC3.Sparse (sparseAt afterReps sparseStages_decodeRun)
open FlockC3.Opening (openingAt afterSparse decodeRun_op decodeRun_opened)
open FlockC3.RegLevels (ligAt tapeR lgStages_decodeRun)

/-- Outputs, not only plays: `decodeRun` copies each play's output into its run, and a later stage's game
reads it. -/
def RunPlays (e : ExecHolo) (u : C3Run e) : Prop :=
  (∀ i : Fin 2, PlaysT (repAt e u i) (u.ts i) (some (u.rep i), u.pend i)) ∧
    (∀ i : Fin 2, PlaysT (sparseAt e u i) (u.sp i) (some (u.claim i))) ∧
    PlaysT (openingAt u) u.op (some u.opened) ∧ PlaysT (ligAt e u) u.lg u.pendR

theorem c3Run_ext {e : ExecHolo} {u v : C3Run e} (h₁ : u.cap₀ = v.cap₀) (h₂ : u.pts = v.pts) (h₃ : u.ts = v.ts)
    (h₄ : u.rep = v.rep) (h₅ : u.pend = v.pend) (h₆ : u.sp = v.sp) (h₇ : u.claim = v.claim) (h₈ : u.op = v.op)
    (h₉ : u.opened = v.opened) (h₁₀ : u.lg = v.lg) (h₁₁ : u.pendR = v.pendR) (h₁₂ : u.os = v.os) : u = v := by
  cases u
  cases v
  simp only [C3Run.mk.injEq]
  exact ⟨h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉, h₁₀, h₁₁, h₁₂⟩

theorem tape_runSteps {e : ExecHolo} (u : C3Run e) : (runSteps u).drop 2 = (firstSteps (u.ts 0) ++ u.ts 1) ++
    (u.sp 0 ++ (u.sp 1 ++ (u.op ++ (u.lg ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩])))) := by
  rw [runSteps, holoSteps_eq]
  simp only [List.drop_succ_cons, List.drop_zero, stagesOf, List.append_assoc]

theorem afterReps_runSteps {e : ExecHolo} (u : C3Run e) : afterReps (runSteps u) u =
    u.sp 0 ++ (u.sp 1 ++ (u.op ++ (u.lg ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩]))) := by
  rw [afterReps, tape_runSteps, ← firstSteps_length, ← List.length_append, List.drop_left]

theorem decodeRun_runSteps {e : ExecHolo} {u : C3Run e} (h : RunPlays e u) : decodeRun e (runSteps u) = u := by
  obtain ⟨hr, hs, ho, hl⟩ := h
  have hcap : (decodeRun e (runSteps u)).cap₀ = proverAt (Cap Dig) (runSteps u)[0]? (c3Default e).cap₀ := rfl
  have hpts : (decodeRun e (runSteps u)).pts =
      coinAt (Fin 2 → Fin e.cls.mPts → GF128) (runSteps u)[1]? (c3Default e).pts := rfl
  have hR := repStages_decodeRun e (runSteps u)
  have hS := sparseStages_decodeRun e (runSteps u)
  have hop := decodeRun_op e (runSteps u)
  have hopd := decodeRun_opened e (runSteps u)
  have hL := lgStages_decodeRun e (runSteps u)
  generalize decodeRun e (runSteps u) = d at hcap hpts hR hS hop hopd hL ⊢
  rw [show (runSteps u)[0]? = some (.inl ⟨Cap Dig, u.cap₀⟩) from rfl, proverAt_some] at hcap
  rw [show (runSteps u)[1]? = some (.inr ⟨Fin 2 → Fin e.cls.mPts → GF128, u.pts⟩) from rfl, coinAt_some] at hpts
  have hra : repsAt e (runSteps u) d = (u.ts 0, (some (u.rep 0), u.pend 0), u.ts 1, (some (u.rep 1), u.pend 1)) := by
    have hg : repsAt e (runSteps u) d = repsAt e (runSteps u) u := by
      unfold repsAt repAt
      rw [hpts]
    rw [hg]
    exact FlockC3.HoloValues.Read.repsOf_of_playsT _ (hr 0) (hr 1)
      (by rw [tape_runSteps]; exact List.prefix_append _ _)
  obtain ⟨hts₀, hts₁, hp₀, hp₁, hr₀, hr₁⟩ := hR
  rw [hra] at hts₀ hts₁ hp₀ hp₁ hr₀ hr₁
  dsimp only at hts₀ hts₁ hp₀ hp₁ hr₀ hr₁
  have hts : d.ts = u.ts := funext (Fin.forall_fin_two.2 ⟨hts₀, hts₁⟩)
  have hrep : d.rep = u.rep := funext (Fin.forall_fin_two.2 ⟨hr₀ _ rfl, hr₁ _ rfl⟩)
  have har : afterReps (runSteps u) d =
      u.sp 0 ++ (u.sp 1 ++ (u.op ++ (u.lg ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩]))) := by
    unfold afterReps
    rw [hts]
    exact afterReps_runSteps u
  obtain ⟨hsp₀, hcl₀, hsp₁, hcl₁⟩ := hS
  rw [show sparseAt e d 0 = sparseAt e u 0 by unfold sparseAt; rw [hrep]] at hsp₀ hcl₀
  rw [show sparseAt e d 1 = sparseAt e u 1 by unfold sparseAt; rw [hrep]] at hsp₁ hcl₁
  have e₀ : stageOf (sparseAt e u 0) (afterReps (runSteps u) d) none = (u.sp 0, some (u.claim 0)) :=
    stageOf_of_playsT none (hs 0) (by rw [har]; exact List.prefix_append _ _)
  rw [e₀] at hsp₀ hcl₀
  dsimp only at hsp₀ hcl₀
  have hrest : (afterReps (runSteps u) d).drop (d.sp 0).length =
      u.sp 1 ++ (u.op ++ (u.lg ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩])) := by
    rw [har, hsp₀, List.drop_left]
  have e₁ : stageOf (sparseAt e u 1) ((afterReps (runSteps u) d).drop (d.sp 0).length) none =
      (u.sp 1, some (u.claim 1)) :=
    stageOf_of_playsT none (hs 1) (by rw [hrest]; exact List.prefix_append _ _)
  rw [e₁] at hsp₁ hcl₁
  dsimp only at hsp₁ hcl₁
  have hcl : d.claim = u.claim := funext (Fin.forall_fin_two.2 ⟨hcl₀ _ rfl, hcl₁ _ rfl⟩)
  have has : afterSparse d (runSteps u) = u.op ++ (u.lg ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩]) := by
    show ((afterReps (runSteps u) d).drop (d.sp 0).length).drop (d.sp 1).length = _
    rw [hrest, hsp₁, List.drop_left]
  have e₂ : stageOf (openingAt u) (afterSparse d (runSteps u)) none = (u.op, some u.opened) :=
    stageOf_of_playsT none ho (by rw [has]; exact List.prefix_append _ _)
  rw [show openingAt d = openingAt u by unfold openingAt; rw [hcl], e₂] at hop hopd
  dsimp only at hop
  simp only [Option.getD_some] at hopd
  have htr : tapeR (runSteps u) d = u.lg ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩] := by
    show (afterSparse d (runSteps u)).drop d.op.length = _
    rw [has, hop, List.drop_left]
  obtain ⟨hlg, hpR, hos⟩ := hL
  rw [show ligAt e d = ligAt e u by unfold ligAt; rw [hopd]] at hlg hpR
  have e₃ : stageOf (ligAt e u) (tapeR (runSteps u) d) PendL.reject = (u.lg, u.pendR) :=
    stageOf_of_playsT PendL.reject hl (by rw [htr]; exact List.prefix_append _ _)
  rw [e₃] at hlg hpR
  dsimp only at hlg hpR
  exact c3Run_ext hcap hpts hts hrep (funext (Fin.forall_fin_two.2 ⟨hp₀, hp₁⟩))
    (funext (Fin.forall_fin_two.2 ⟨hsp₀, hsp₁⟩)) hcl hop hopd hlg hpR
    (hos u.os (by rw [htr, hlg, List.drop_left, List.getElem?_cons_zero]))

theorem claim_of_decodeRun {e : ExecHolo} {u : C3Run e} (hfix : decodeRun e (runSteps u) = u) :
    ∀ (i : Fin 2) (x : RegClaim GF128 e.cls.shape), PlaysT (sparseAt e u i) (u.sp i) (some x) → x = u.claim i := by
  have hS := sparseStages_decodeRun e (runSteps u)
  rw [hfix] at hS
  obtain ⟨-, hc₀, -, hc₁⟩ := hS
  have har := afterReps_runSteps u
  refine Fin.forall_fin_two.2 ⟨fun x h => (hc₀ x ?_).symm, fun x h => (hc₁ x ?_).symm⟩
  · have e₀ := stageOf_of_playsT none h (l := afterReps (runSteps u) u) (by rw [har]; exact List.prefix_append _ _)
    exact congrArg Prod.snd e₀
  · have e₁ := stageOf_of_playsT none h (l := (afterReps (runSteps u) u).drop (u.sp 0).length)
      (by rw [har, List.drop_left]; exact List.prefix_append _ _)
    exact congrArg Prod.snd e₁

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

theorem srcDecode_of_plays {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (h : AtSrc zs dg N k Rw src fun _ u => RunPlays (holoAt st hp) u) :
    AtSrc zs dg N k Rw src fun _ u => decodeRun (holoAt st hp) (runSteps u) = u :=
  fun R ω t R₂ τ o hacc => decodeRun_runSteps (h R ω t R₂ τ o hacc)

end

end FlockC3.Compose
