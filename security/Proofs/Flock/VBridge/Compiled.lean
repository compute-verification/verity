module

public import Proofs.Flock.Recursive.Target
public import Proofs.Flock.Soundness.Assumptions.Recursive
public import Proofs.Flock.VBridge.Algebra.Build
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

/-! `Flock.Assumptions.VBridge` at the compiled inner protocol (`vbridge_compiled`), from hypotheses each a statement about V* or the verifier. -/

@[expose] public section

namespace FlockVBridge.Compiled

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open Flock.RecAlgebra (verifierRows structureOf residuals valOf kapSum)

structure RepData where
  w : Array F128
  comb : Array F128
  cons : ℕ → ℕ → F128
  rounds : Array (Array F128)
  extras : Array Claim

def AlgebraHolds (sh : Shape) (d : RepData) : Prop :=
  ∃ v info kappa S rs, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) ∧ structureOf sh = .ok S ∧
    d.w.size = S.nw ∧ d.comb.size = 64 ∧ residuals S d.w v = .ok rs ∧ ∀ x ∈ rs, x = 0

def AccHolds (sh : Shape) (d : RepData) : Prop :=
  ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
    ∀ r < 2, valOf sh d.w d.comb d.cons (.acc1 r) + valOf sh d.w d.comb d.cons (.acc0 r) = kapSum kappa r d.cons

section outer

variable {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}

def Accepted (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (R : Reg) (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂)
    (τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N))
    (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript) : Prop :=
  (∀ j, LiveAcceptsZKR (zs R ω t R₂ j).I (zs R ω t R₂ j).own (o j) ∧
    (zs R ω t R₂ j).Good ((zs R ω t R₂ j).layer dg N k Rw (τ j))) ∧ ¬ RegColl zs R ω t R₂ ∧
    ∀ j, Game.Outcome ((zs R ω t R₂ j).game dg N) (τ j) (o j)

noncomputable def layers (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (R : Reg) (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂)
    (τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N)) : (j : Fin S) → (Vs j).Bits :=
  fun j => (zs R ω t R₂ j).layer dg N k Rw (τ j)

def AlgebraDecodes (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape)
    (rep : Reg → L.Ω → List (Cm × I.Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData) : Prop :=
  ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
    AlgebraHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r)

def AccChains (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape)
    (rep : Reg → L.Ω → List (Cm × I.Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData) : Prop :=
  ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, AccHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r)

def StagesAccept (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) : Prop :=
  ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o →
    (∀ r, AlgebraHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) ∧
      AccHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r)) →
    I.accepts (x R ω) ((wmsg (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd))

end outer

section target

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- `Flock.Assumptions.VBridge` with the inner protocol `flockInnerCL`, from hypotheses no proof here discharges: the
algebra decoded from V*'s layers (`AlgebraDecodes`), the running sums' chain (`AccChains`), the stages run from E's
conclusions (`StagesAccept`), and V*'s files and sampling designs (`VCommits`, `VDecodesR`, `VFullR`). -/
theorem vbridge_compiled (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : FlockSoundness.Accounting.Schedule) {n : ℕ} {L : SamplingDesign n} {Reg : Type} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (flockInnerCL T CT A H E Lf cls regs sch).Coin) → Reg₂ → (j : Fin S) →
      (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (flockInnerCL T CT A H E Lf cls regs sch).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      (flockInnerCL T CT A H E Lf cls regs sch).Msg)
    (sh : Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (halg : AlgebraDecodes zs dg N k Rw sh fun _ _ _ => rep) (hacc : AccChains zs dg N k Rw sh fun _ _ _ => rep)
    (hst : StagesAccept zs dg N k Rw sh rep x wmsg)
    (hcm : VCommits zs Sk cmt) (hdec : VDecodesR wmsg (fun _ _ a b => a = b) Sk msg)
    (hfull : VFullR (flockInnerCL T CT A H E Lf cls regs sch).r Sk) :
    Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg :=
  ⟨fun R ω t R₂ hc τ o hout h => hst R ω t R₂ τ o ⟨h, hc, hout⟩ fun r =>
    ⟨halg R ω t R₂ τ o ⟨h, hc, hout⟩ r, hacc R ω t R₂ τ o ⟨h, hc, hout⟩ r⟩, hcm, hdec, hfull⟩

end target

end FlockVBridge.Compiled
