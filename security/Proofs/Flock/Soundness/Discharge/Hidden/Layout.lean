module

public import Proofs.Flock.Soundness.Discharge.Hidden.Defs
public import Proofs.Flock.Soundness.Discharge.Hidden.Delta
public import Proofs.Flock.Soundness.Discharge.Hidden.Rows
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

/-! A hidden-output statement's layout (`HiddenLayout`): `HmOut.delta` appends to `HmRow.delta`'s Δ only `outPairs`, whose first columns are input bits of net slots. -/

@[expose] public section

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

theorem deltaMsgCol_rowBit {c : Circuit} {g p byte t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ht : t < 8) :
    ShaRowBit c (deltaMsgCol c g p byte t) ∨ HmRowBit c (deltaMsgCol c g p byte t) := by
  have ho := Typed.byteAt_snd (c.ports.getD p default) byte
  by_cases hk : (HmRow.byteAt (c.ports.getD p default) byte).1 < ((HmRow.rowSlots c.ports).2[p]!).size
  · refine Or.inl ⟨g, hg, p, by rw [rowSlots_spec]; exact hp, (HmRow.byteAt (c.ports.getD p default) byte).1, hk, 1,
      by omega, 64 * (byte % 128 / 8) + 56 - 8 * (byte % 128 % 8) + t, by omega, ?_⟩
    simp only [deltaMsgCol, hk, ↓reduceIte, ho]
    omega
  · refine Or.inr ⟨g, hg, p, by rw [rowSlots_spec]; exact hp, 1, by omega,
      64 * (byte % 128 / 8) + 56 - 8 * (byte % 128 % 8) + t, by omega, ?_⟩
    simp only [deltaMsgCol, hk, ↓reduceIte, ho]
    omega

theorem outPairs_inBit {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ()) :
    ∀ x ∈ outPairs c, InBit c x.1 := by
  have P := parse_facts hp
  obtain ⟨ls, hcheck⟩ := P.check
  have C := check_facts P.typed hcheck
  intro x hx
  simp only [outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hx
  obtain ⟨g, hg, j, hj, k, -, t, -, hx⟩ := hx
  have hlt := (checkRows_out hc hj).1
  have hs := deltaMsgCol_rowBit hg hlt (byte := 2 * k + t / 8) (Nat.mod_lt t (by norm_num))
  rcases hx with rfl | rfl <;> exact deltaIn_inBit P C (hs.imp id Or.inl)

end FlockSoundness.Discharge.Hidden
