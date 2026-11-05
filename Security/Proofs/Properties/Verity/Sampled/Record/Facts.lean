module

public import Proofs.Properties.Verity.Sampled.Record

/-!
# The sampling record's facts at one-stage v2's session

* `drawFixed`: the draw precedes the session (`DrawFixed`): every unit's session rounds are the second stage, and the
  checked units read only the draw rounds (`Claims.Sampling.drawFixed_of_first`).
* `interiorFixed`: the interior values are committed under the session's roots, which the statement fixes, so the roots
  a run shows for them are the same at fresh second-stage coins (`InteriorFixed` at `fun _ _ _ => A.roots`).
* `refuses_of_inScope`: the verifier refuses every statement but the record's at once, so it refuses what is out of
  scope (`Refuses`) exactly where the record's statement is in scope (`InScope (partSlots A) (stmtSlots A)`).
* **v2's slots** (`blocks_partSlots`): slot `u < A.n` is unit `u`'s lowered gates (`partSlots_getElem?_unit`), and slot
  `A.n + i` the lowered program's `i`th input gate (`partSlots_getElem?_input`). `designSlots`' `P_a` is the law's on the
  population's strata at `B`'s population slots, times `all`'s at its input slots (`pa_partSlots`), and `designSlots` is
  well formed on `partSlots` where the law's rule is on the strata (`designSlots_wellFormed`).
-/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling

variable (A : Live) (o : Own) (e : Ext)

theorem record_parse_eq (x : Msg) :
    (record A o e).parse x = if x = xOf A o then some (stmtSlots A) else none := rfl

theorem record_parse : (record A o e).parse (xOf A o) = some (stmtSlots A) :=
  (record_parse_eq A o e (xOf A o)).trans (ite_eq_left rfl)

/-- **The draw precedes the session.** -/
theorem drawFixed : DrawFixed (record A o e) := by
  refine drawFixed_of_first _ (fun _ _ _ _ => subset_refl _) fun x r r' h => ?_
  have hc : @coinsAt ((V A o).k x) r = @coinsAt ((V A o).k x) r' := coinsAt_congr fun t ht => h t ht
  refine (record_checked A o e x r).trans ?_
  rw [hc]
  exact (record_checked A o e x r').symm

/-- **The interior roots are the statement's**, at every second stage. -/
theorem interiorFixed : InteriorFixed sha512Msg (record A o e) (fun _ _ _ => A.roots) :=
  fun _ _ _ _ _ _ _ => rfl

/-- **The verifier refuses what is out of scope**, where the record's statement is in scope: an accepted run is of the
record's statement, since the verifier refuses every other at once. -/
theorem refuses_of_inScope (H : Msg → Msg) (h : InScope (partSlots A) (stmtSlots A)) : Refuses H (record A o e) := by
  intro x Ω P ω r hacc
  obtain ⟨as, cs, has⟩ := Claims.Sampling.run_acc_next H P (record A o e).V x ω r hacc
  by_cases hx : x = xOf A o
  · subst hx
    exact ⟨stmtSlots A, partSlots A, record_parse A o e, rfl, h⟩
  · have hn : (record A o e).V.next x as cs = .ret .rej := ite_eq_right hx
    rw [hn] at has
    cases has

/-! ## v2's slots -/

/-- The population's class of `partSlots`: the units' lowered gates in the registration's strata. -/
noncomputable def popSlots : List (List (List ℕ)) :=
  (Flock.Bridge.nsOf A.desc A.qr).splitLengths ((List.finRange A.n).map (Flock.Bridge.unitGates A.P))

theorem classes_partSlots :
    (partSlots A).classes = [popSlots A, InputSlot.inputClass (Flock.Bridge.progOf A.Cc A.κ)] := rfl

theorem flatten_popSlots : (popSlots A).flatten = (List.finRange A.n).map (Flock.Bridge.unitGates A.P) :=
  List.flatten_splitLengths _ _ (by simp [Flock.Bridge.nsOf_sum A.hpop])

theorem strataSizes_popSlots : strataSizes (popSlots A) = Flock.Bridge.nsOf A.desc A.qr :=
  List.map_splitLengths_length _ _ (by simp [Flock.Bridge.nsOf_sum A.hpop])

/-- **`partSlots`' slots**: the units' lowered gates, then one slot per input gate of the lowered program. -/
theorem blocks_partSlots : (partSlots A).blocks =
    (List.finRange A.n).map (Flock.Bridge.unitGates A.P) ++
      (InputSlot.inputsAt (Flock.Bridge.progOf A.Cc A.κ)).map fun g => [g] := by
  rw [partSlots, InputSlot.blocks_withInputs]
  congr 1
  simp only [Claims.Sampling.Partition.blocks, List.flatten_cons, List.flatten_nil, List.append_nil]
  exact flatten_popSlots A

theorem length_blocks_partSlots :
    (partSlots A).blocks.length = A.n + (InputSlot.inputsAt (Flock.Bridge.progOf A.Cc A.κ)).length := by
  simp [blocks_partSlots]

theorem length_inputSlots : (inputSlots A).length = (InputSlot.inputsAt (Flock.Bridge.progOf A.Cc A.κ)).length := by
  simp [inputSlots]

/-- Slot `u` of `partSlots` is unit `u`'s lowered gates. -/
theorem partSlots_getElem?_unit (u : Fin A.n) :
    (partSlots A).blocks[u.val]? = some (Flock.Bridge.unitGates A.P u) := by
  rw [blocks_partSlots, List.getElem?_append_left (by simp)]
  simp

/-- Slot `A.n + i` of `partSlots` is the lowered program's `i`th input gate. -/
theorem partSlots_getElem?_input {i g : ℕ}
    (hi : (InputSlot.inputsAt (Flock.Bridge.progOf A.Cc A.κ))[i]? = some g) :
    (partSlots A).blocks[A.n + i]? = some [g] := by
  rw [blocks_partSlots, List.getElem?_append_right (by simp)]
  simp [hi]

/-- **`designSlots`' `P_a` on `partSlots`**: the law's rule on the population's strata at `B`'s population slots, times
`all`'s at its input slots. -/
theorem pa_partSlots (B : Finset ℕ) :
    (designSlots A).Pa (partSlots A) B =
      (_root_.Properties.SampledProofs.oneStageOf A.law).Pa (Flock.Bridge.nsOf A.desc A.qr) (window 0 A.n B) *
        (if window A.n (InputSlot.inputsAt (Flock.Bridge.progOf A.Cc A.κ)).length B = ∅ then 1 else 0) := by
  show classesPa [_, .all] (partSlots A).classes 0 B = _
  rw [classes_partSlots]
  simp only [classesPa, flatten_popSlots, strataSizes_popSlots, InputSlot.inputClass_flatten, List.length_map,
    List.length_finRange, zero_add, OneStage.Pa, mul_one]

/-- **`designSlots` is well formed on `partSlots`** where the law's one-stage rule is well formed on the population's
strata: the input class's rule, `all`, is well formed on every class. -/
theorem designSlots_wellFormed
    (h : (_root_.Properties.SampledProofs.oneStageOf A.law).WellFormed (Flock.Bridge.nsOf A.desc A.qr)) :
    (designSlots A).WellFormed (partSlots A) := by
  refine ⟨rfl, fun p hp => ?_⟩
  simp only [designSlots, classes_partSlots, List.zip_cons_cons, List.zip_nil_right, List.mem_cons,
    List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · show (_root_.Properties.SampledProofs.oneStageOf A.law).WellFormed (strataSizes (popSlots A))
    rw [strataSizes_popSlots]
    exact h
  · trivial

end Verity.Sampled.Record
