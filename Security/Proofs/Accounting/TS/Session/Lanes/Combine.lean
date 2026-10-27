import Proofs.Accounting.TS.Session.Lanes.Assumptions
import Proofs.Pouw.H1Zy.Combine
import Proofs.Pouw.H1Zy.CombineBits

/-!
# The combine's part of `LeafChecks` from its Definition

`Accounting.Assumptions.TS.LeafChecks`' combine field bundles three claims: the combine circuit's input wires sit where
`decC` reads them, the circuit computes `PouwH1Combine_v1`, and that Definition's `ok = 1` means `CombineOk`. The third is
proved here for every input: the Definition on words (`Pouw.H1Zy.combineOk`, `h1zy._combine_body`) is `true` exactly
when the lanes' partials XOR, in lane order, to the receipt (`combineOk_iff_xorAll`), so a view whose `ok` is the
Definition's on its own partials and receipt meets `CombineOk` wherever `ok = 1` (`combineOk_of_def`).

What stays of the field is the first two claims, `CombineComputes`: on every wire assignment the session's evaluation
agrees with, the decoded `ok` is the Definition's on the decoded partials and receipt. `leafChecks_of_combineComputes`
gives `LeafChecks` from it and the lanes' field.

`CombineComputes` is in turn split at the combine's pieces: `CombineLowers` says the circuit computes the body on bits,
its pieces (`PouwXor32_v2`, `PouwEq32_v2`, `PouwAndBit_v2`) at their lowerings (`Pouw.H1Zy.combineBits`), and that each
piece's lowering computes its word view, and so their composition the body on words, is proved
(`Pouw.H1Zy.combineBits_bitsOf`). The two are equivalent (`combineLowers_iff`).
-/

namespace Accounting.TS.Lanes

open Accounting.TS Accounting.TS.Checked Flock.SoundAtBridge PouwWindow.Lanes

/-- `PouwH1Combine_v1`'s `ok` on a leaf's 32 lanes' partials is `true` exactly when they XOR, in lane order, to the
receipt (`xorAll`). -/
theorem combineOk_iff_xorAll {nw : ℕ} (p : Fin 32 → Fin nw → BitVec 32) (r : Fin nw → BitVec 32) :
    Pouw.H1Zy.combineOk p r = true ↔ xorAll p = r := by
  rw [Pouw.H1Zy.combineOk_iff]
  exact ⟨fun h => funext h, fun h w => congrFun h w⟩

/-- **Claim 3 of `LeafChecks` at the combine**: a combine view whose `ok` is `PouwH1Combine_v1`'s on its own partials
and receipt meets `CombineOk` where `ok = 1`. -/
theorem combineOk_of_def {nw : ℕ} (v : CombineView nw) (hdef : v.ok = Pouw.H1Zy.combineOk v.parts v.receipt)
    (hok : v.ok = true) : CombineOk v :=
  (combineOk_iff_xorAll v.parts v.receipt).1 (hdef ▸ hok)

/-- **The combine's circuit computes its Definition at its port rows** (claims 1 and 2 of `LeafChecks`' combine field):
at the combine's session `sess none`, on every wire assignment the key program's evaluation agrees with on every unit's
wires, the `ok` word `decC` reads from the port rows (`portRead`) is `PouwH1Combine_v1`'s on the partials and receipt
`decC` reads from them. -/
def CombineComputes {nw : ℕ} (sess : Option (Fin 32) → RowsSession)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) : Prop :=
  ∀ vals : Fin (sess none).Circ.N → Bool,
    (∀ u, ∀ g ∈ (sess none).P.io u, (sess none).Circ.eval vals g = vals g) →
    (decC (sess none) (portRead (sess none) vals)).ok =
      Pouw.H1Zy.combineOk (decC (sess none) (portRead (sess none) vals)).parts
        (decC (sess none) (portRead (sess none) vals)).receipt

/-- **The combine's circuit computes its pieces' composition at its port rows**: at the combine's session `sess none`, on
every wire assignment the key program's evaluation agrees with on every unit's wires, the `ok` word `decC` reads from the
port rows is `PouwH1Combine_v1`'s body on bits (`Pouw.H1Zy.combineBits`: `PouwXor32_v2`, `PouwEq32_v2` and
`PouwAndBit_v2` at their lowerings) on the bits of the partials and receipt `decC` reads. What it rests on is
circuit-check's lowering of `PouwH1Combine_v1`, which composes those three pieces. -/
def CombineLowers {nw : ℕ} (sess : Option (Fin 32) → RowsSession)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) : Prop :=
  ∀ vals : Fin (sess none).Circ.N → Bool,
    (∀ u, ∀ g ∈ (sess none).P.io u, (sess none).Circ.eval vals g = vals g) →
    (decC (sess none) (portRead (sess none) vals)).ok =
      Pouw.H1Zy.combineBits (fun l w => Pouw.H1Zy.bitsOf ((decC (sess none) (portRead (sess none) vals)).parts l w))
        (fun w => Pouw.H1Zy.bitsOf ((decC (sess none) (portRead (sess none) vals)).receipt w))

/-- **`CombineComputes` is `CombineLowers`**: the pieces' lowerings compose to the body on words
(`Pouw.H1Zy.combineBits_bitsOf`). -/
theorem combineLowers_iff {nw : ℕ} (sess : Option (Fin 32) → RowsSession)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw) :
    CombineLowers sess decC ↔ CombineComputes sess decC := by
  unfold CombineLowers CombineComputes
  simp only [Pouw.H1Zy.combineBits_bitsOf]

/-- **`CombineComputes` from `CombineLowers`.** -/
theorem combineComputes_of_lowers {nw : ℕ} {sess : Option (Fin 32) → RowsSession}
    {decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw} (h : CombineLowers sess decC) :
    CombineComputes sess decC :=
  (combineLowers_iff sess decC).1 h

/-- **`LeafChecks` from the lanes' field and `CombineComputes`**: the combine's field of
`Accounting.Assumptions.TS.LeafChecks` needs only that its circuit computes its Definition at its port rows; that the
Definition's `ok = 1` is `CombineOk` is `combineOk_of_def`. -/
theorem leafChecks_of_combineComputes {K nw : ℕ} {Pub : Type} (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (sess : Option (Fin 32) → RowsSession)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw)
    (hL : 32 ∣ K → K < 2 ^ 17 → ∀ l (vals : Fin (sess (some l)).Circ.N → Bool),
      (∀ u, ∀ g ∈ (sess (some l)).P.io u, (sess (some l)).Circ.eval vals g = vals g) →
      (decL (sess (some l)) (portRead (sess (some l)) vals)).ok = true →
      LaneOk dq part (decL (sess (some l)) (portRead (sess (some l)) vals)))
    (hC : CombineComputes sess decC) : Accounting.Assumptions.TS.LeafChecks dq part sess decL decC :=
  ⟨hL, fun vals hv hok => combineOk_of_def _ (hC vals hv) hok⟩

end Accounting.TS.Lanes
