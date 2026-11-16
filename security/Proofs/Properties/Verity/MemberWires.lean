module

public import Properties.Verity.Committed
public import Proofs.Core.Game.Prob

/-!
# The per-member wires' target gives the one-member one

`Vocab.MemberWiresSound` reads the wiring off SPOT's case and each row's root under its member's name;
`Vocab.WiresSound` reads the wiring the policy's record names and each root under the row's plain name. Under
`Vocab.WiresAtMembers` (SPOT's case names the record's wiring, and the record names no root by a plain row name)
the first gives the second (`Vocab.wiresSound_of_memberWiresSound`): the record's plain names name no root, so
`SameRow` compares nothing and the units `unheld` gives are unbound ones, which `memberUnheld` counts too.

The converse does not hold: `WiresSound` bounds the units bound in the transcript only through rows the policy's record
names by plain name, and says nothing of a wire into a member's row, which `MemberWiresHold` holds; and at a play with
no case of the policy, or one whose record names no wiring, `WiresHold` holds while `MemberWiresHold` still reads SPOT's
case.
-/

@[expose] public section

namespace Verity

open ResourceAccounting

namespace Vocab

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

/-- Where the record names no root by any of `W`'s plain row names, the units `W` doesn't hold are among the
population's units it doesn't hold at any record, ranges and members: `SameRow` compares nothing there. -/
theorem unheld_subset_memberUnheld {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {record : Lean.Json}
    {W : Spot.Wires.Wiring} (record' : Lean.Json) (rs ms : Array (String × ℕ × ℕ)) (X : (V.circuitOf θ F).Trace)
    (hno : ∀ (t port : String) (row : ℕ) (R : Hash.Digest), ¬ RecordRoot record (rowName W t port row) R) :
    V.unheld θ F c record W X ⊆ V.memberUnheld θ F c record' W rs ms X := by
  intro u hu
  simp only [unheld, Finset.mem_filter, Finset.mem_univ, true_and] at hu
  simp only [memberUnheld, Finset.mem_filter, Finset.mem_univ, true_and]
  refine Or.inl (Classical.byContradiction fun hb => hu fun t => ?_)
  unfold Properties.Spot.WiredCorrect
  refine ⟨hb, ?_⟩
  rintro ⟨e, -, -, hne⟩
  exact hne fun _ _ _ _ _ _ _ hRx => (hno _ _ _ _ hRx).elim

/-- In a play SPOT's verdict at `k` accepts, under `WiresAtMembers`, the per-member wires' claim gives the
one-member one. -/
theorem wiresHold_of_memberWiresHold {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {σ : Game.Strategy (V.game θ F c)} {o : Exchange} (hA : V.WiresAtMembers k)
    (hv : V.Verdict k .sampled o) (hm : V.MemberWiresHold k θ F c σ o) : V.WiresHold k θ F c σ o := by
  intro p r e hpc W hW
  obtain ⟨⟨sp, sc, ms, S, prog, rs, hsc, hms, hwir, hprog, hrs⟩, hno⟩ := hA o hv p r e hpc W hW
  refine le_trans ?_ (hm sp sc hsc ms W S hms hwir prog rs hprog hrs)
  exact_mod_cast Finset.card_le_card (V.unheld_subset_memberUnheld sc.record _ _ _ hno)

/-- **The per-member wires' target gives the one-member one**, under `WiresAtMembers`. -/
theorem wiresSound_of_memberWiresSound (k : Verity.Registration.Composite) (hM : V.MemberWiresSound k)
    (hA : V.WiresAtMembers k) : V.WiresSound k := fun θ F c σ =>
  (Game.prob_mono (fun _ ⟨hv, hw⟩ => ⟨hv, fun hm => hw (V.wiresHold_of_memberWiresHold hA hv hm)⟩) _ σ).trans
    (hM θ F c σ)

end Vocab

end Verity
