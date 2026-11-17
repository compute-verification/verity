module

public import Proofs.Flock.VBridge.Family.Commits
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

/-! The decoder's last round, which no round commitment holds, read in full from σ's leaf, the outer proof's
layers. -/

@[expose] public section

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Family (skOf wmsgOf msgOf maskOf layerOf)

variable {S : ℕ} (Vs : Fin S → VStmt)

noncomputable def skLeaf (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (r i : ℕ)
    (j : Fin S) : Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv) :=
  if i + 1 < r then skOf Vs own sel i j else Finset.univ

variable {Vs} {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}

theorem skLeaf_of_lt {r i : ℕ} (hi : i + 1 < r) (j : Fin S) : skLeaf Vs own sel r i j = skOf Vs own sel i j :=
  ite_eq_left hi

theorem maskOf_skLeaf {r i : ℕ} (hi : i + 1 < r) :
    maskOf Vs (skLeaf Vs own sel r) i = maskOf Vs (skOf Vs own sel) i :=
  funext fun _ => funext fun j => funext fun _ => by simp only [maskOf]; rw [skLeaf_of_lt hi j]

theorem vDecodesR_wmsgLeaf {I : Recursion.Inner} (dec : ℕ → ((j : Fin S) → (Vs j).Bits) → I.Msg) :
    VDecodesR (wmsgOf Vs (skLeaf Vs own sel I.r) I.r dec) (fun i _ a b => i + 1 < I.r → a = b)
      (fun i j => if i + 1 < I.r then skOf Vs own sel i j else ∅) (msgOf Vs (skLeaf Vs own sel I.r) dec) := by
  intro i hi _ Xt v hX
  simp only [wmsgOf, List.getElem?_map, List.getElem?_range hi, Option.map_some]
  refine Option.Rel.some ?_
  intro hc
  show dec i (maskOf Vs (skLeaf Vs own sel I.r) i Xt) = dec i (layerOf Vs (skLeaf Vs own sel I.r) i v)
  refine congrArg (dec i) (funext fun j => funext fun g => ?_)
  simp only [maskOf, layerOf]
  rw [skLeaf_of_lt hc j]
  split_ifs with h
  · exact hX j g (by simpa [hc] using h.1) h.2
  · rfl

end FlockC3.Compose
