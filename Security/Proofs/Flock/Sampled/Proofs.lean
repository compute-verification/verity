import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fintype.BigOperators
import Proofs.Flock.Sampled.Statements

/-!
# F2's statements, proved

All but `Composition`.

* `coverage_holds`: the coverage check routes the slots bijectively onto the draw's order, which is injective, so each
  drawn unit is held by exactly one slot; and the slots, `Σ_b slots(b)` of them, number at least the draw.
* `slotBinding_holds`: a held slot off its unit's committed record opens the root at the unit's position with a record
  other than the committed one (the record's format), so the two openings give a SHA-512 collision: of the two tree
  leaves' inputs, or of the two openings' (`Flock.Registered.opensLeaf_binding`).
* `slotsSound_holds`: from the two, and the statement's slot check.
* `zeroKnowledge_holds`: F1's `hides` at the honest audit's reveal list.
* `menuComplete_holds`, `menuCompleteAt_holds`, `zeroKnowledgeAt_holds`: the honest prover fills a fitting batching's
  held slots with the trace's values (`fillValues`); so it aborts only where the draw doesn't fit; and at every other
  draw its audit's view is `zeroKnowledge_holds`'s.
-/

namespace FlockSampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

/-- The draw's `j`-th unit, for `j < |S|`. -/
theorem drawOrder_fin {N : ℕ} (S : Finset (Fin N)) (j : Fin S.card) :
    drawOrder S j = some (S.orderEmbOfFin rfl j) := by
  unfold drawOrder
  exact dite_eq_left j.isLt

theorem drawOrder_eq_some {N : ℕ} {S : Finset (Fin N)} {j : ℕ} {i : Fin N} (h : drawOrder S j = some i) :
    ∃ hj : j < S.card, S.orderEmbOfFin rfl ⟨j, hj⟩ = i := by
  unfold drawOrder at h
  split at h
  · exact ⟨‹_›, Option.some.inj h⟩
  · cases h

theorem coverage_holds : Coverage := by
  intro N T V X S β hc hB
  obtain ⟨n, route, hn, hr⟩ := hc
  refine ⟨fun i hi => ?_, ?_⟩
  · obtain ⟨j₀, rfl⟩ : ∃ j₀, S.orderEmbOfFin rfl j₀ = i := by
      have hi' : i ∈ Set.range (S.orderEmbOfFin rfl) := by
        rw [Finset.range_orderEmbOfFin]
        exact Finset.mem_coe.2 hi
      exact Set.mem_range.1 hi'
    have hheld : β.held (route ⟨j₀, lt_of_lt_of_le j₀.isLt hn⟩) = some (S.orderEmbOfFin rfl j₀) :=
      (hr _).trans (drawOrder_fin S j₀)
    refine ⟨route ⟨j₀, lt_of_lt_of_le j₀.isLt hn⟩, ⟨hheld, hB _ _ hheld⟩, ?_⟩
    rintro q ⟨hq, -⟩
    obtain ⟨hj, he⟩ := drawOrder_eq_some
      ((hr (route.symm q)).symm.trans ((congrArg β.held (route.apply_symm_apply q)).trans hq))
    have e := (S.orderEmbOfFin rfl).injective he
    exact (route.apply_symm_apply q).symm.trans (congrArg route (Fin.ext (congrArg Fin.val e)))
  · have hcard : Fintype.card β.Pos = ∑ b, (β.shape b).slots :=
      Fintype.card_sigma.trans (Finset.sum_congr rfl fun b _ => Fintype.card_fin _)
    have hsum : ∑ b, (β.shape b).slots = (β.view.map Shape.slots).sum :=
      (Fin.sum_univ_def _).trans
        (congrArg List.sum (List.map_map (g := Shape.slots) (f := β.shape) (l := List.finRange β.sessions)).symm)
    calc S.card ≤ n := hn
      _ = Fintype.card (Fin n) := (Fintype.card_fin n).symm
      _ = Fintype.card β.Pos := Fintype.card_congr route
      _ = (β.view.map Shape.slots).sum := hcard.trans hsum

theorem slotBinding_holds : SlotBinding := by
  intro N T V C X path₀ h hinj hcom hopen
  by_cases hB : Bound X h.β
  · exact Or.inl hB
  · right
    obtain ⟨p, i, hp, hne⟩ : ∃ p i, h.β.held p = some i ∧ X i ≠ some (h.β.slotTyp p, h.β.slotVal p) :=
      Classical.byContradiction fun hc => hB fun p i hp => Classical.byContradiction fun hne => hc ⟨p, i, hp, hne⟩
    obtain ⟨⟨t, v⟩, hX⟩ := Option.isSome_iff_exists.1 (hcom.full i _ _ _ (hopen p i hp))
    refine ⟨p, i, t, v, hp, hX, ?_⟩
    have hxy : C.enc (h.β.slotTyp p) (h.β.slotVal p) ≠ C.enc t v := fun e => by
      obtain ⟨e1, e2⟩ := hinj _ _ _ _ e
      exact hne (by rw [hX, e1, e2])
    unfold recordPair
    split
    · rename_i hl
      exact ⟨fun e => hxy ((ByteArray.append_right_inj _).mp e), hl⟩
    · rename_i hl
      exact Flock.Registered.opensLeaf_binding _ _ _ _ _ _ (Flock.Sha512.hash_size _) (Flock.Sha512.hash_size _) hl
        (hopen p i hp) (hcom.opens i t v hX)

theorem slotsSound_holds : SlotsSound := by
  intro N T V Ty C X path₀ S h hinj hcom hp
  rcases slotBinding_holds C X path₀ h hinj hcom hp.opens with hB | hbrk
  · left
    have hcov := (coverage_holds X S h.β hp.cover hB).1
    refine Finset.disjoint_left.2 fun i hi hw => ?_
    obtain ⟨q, ⟨hq, hX⟩, -⟩ := hcov i hi
    exact (Finset.mem_filter.1 hw).2 _ _ hX (hp.holds q i hq)
  · exact Or.inr hbrk

theorem zeroKnowledge_holds : ZeroKnowledge := by
  intro N T V Reg O session Ty com I M R S h hv hp
  exact I.hides (reveal M S (I.out R)) R h rfl hv.symm hp

/-- A batching's skeleton with each held slot's values read from the trace, `default` in the padding. -/
def fillValues {N : ℕ} {T V : Type} [Inhabited V] (β : Batching N T Unit) (X : Trace N T V) : Batching N T V where
  sessions := β.sessions
  shape := β.shape
  typ := β.typ
  idx := β.idx
  val := fun b j => ((β.idx b j).bind X).elim default Prod.snd

theorem menuComplete_holds : MenuComplete := by
  intro N T V _ M Ty X S hM hSM hF
  obtain ⟨β, hv, hc, hp, ht⟩ := hF
  refine ⟨fillValues β X, hv, fun s hs => hM s (hSM S.card s ?_), hc, fun p i hi => ?_,
    fun p i hi => (hp p i hi).2, ht⟩
  · rw [← hv]
    exact hs
  · obtain ⟨⟨w, hw⟩, -⟩ := hp p i hi
    have hi' : β.held p = some i := hi
    have e1 : (β.held p).bind X = some (β.slotTyp p, w) := by
      rw [hi']
      exact hw
    have e2 : (fillValues β X).slotVal p = w := congrArg (fun o => Option.elim o default Prod.snd) e1
    exact hw.trans (congrArg (fun u => some (β.slotTyp p, u)) e2.symm)

theorem menuCompleteAt_holds : MenuCompleteAt := by
  intro N T V _ M Ty L X δ hM hSM hR
  refine le_trans (prCoin_mono (q := fun ω => ¬ M.Fits Ty X (L.draw ω)) fun ω h hF => ?_) hR
  exact h (menuComplete_holds M Ty X (L.draw ω) hM hSM hF)

theorem zeroKnowledgeAt_holds : ZeroKnowledgeAt := by
  intro N T V Reg O _ session Ty com I M L R X path₀ δ hM hSM hcom hX hroom
  refine ⟨menuCompleteAt_holds M Ty L X δ hM hSM hroom, fun ω β hβ => ?_⟩
  obtain ⟨hv, -, hc, hB, hfit, -⟩ := hβ
  refine zeroKnowledge_holds session Ty com I M R (L.draw ω) (Hidden.honest β path₀) hv
    { cover := hc, opens := fun p i hp => ?_, fits := hfit, holds := fun p i hp => hX i _ _ (hB p i hp) }
  have e : (Hidden.honest β path₀).path p = path₀ i := congrArg (fun o => Option.elim o ByteArray.empty path₀) hp
  rw [e]
  exact hcom.opens i _ _ (hB p i hp)

end FlockSampled
