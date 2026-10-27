import Properties.NetworkRecords

/-!
# The check without the grid: the proof

An egress link-window's verdict without the grid is the full check's, or the full check's is `mismatch` and it is
accepted (`egress_noGrid`); an ingress link-window's is the same function's. So `verdictsNoGrid` is `verdicts` but for
the egress grid comparison (`verdicts_noGrid`), and `evaluateNoGrid` refuses where `evaluate` does, and a raw or
committed egress record besides (`uncertifiedEgress_iff`): `noGridMeans`, `noGridAcceptsMore`.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check (Params LinkWindow manifestSet)

namespace Properties.NetworkRecords

namespace NoGridProofs

/-- **An egress link-window's verdict without the grid** is the full check's, or the full check's is `mismatch` and it
is none: the two run the same tests, in the same order, up to the grid comparison, which is the full check's last. -/
theorem egress_noGrid (p : Params) (c : Records.Commitment) (e : Records.ByteEvidence) (eg : ByteArray) (l : String)
    (link : Records.ByteLink) (expected : Records.RawGrid) (w : Nat) :
    Records.egressVerdictNoGrid p c e l link w = Records.egressVerdict p c e eg l link expected w ∨
      Records.egressVerdict p c e eg l link expected w = some "mismatch" ∧
        Records.egressVerdictNoGrid p c e l link w = none := by
  unfold Records.egressVerdict Records.egressVerdictNoGrid
  generalize (Records.found c .egress l w).toList = R
  dsimp only
  generalize (!(Check.inSyncSet p.syncSet (link.syncs.getD w Check.noSync) &&
    (link.syncs.getD w Check.noSync).delta == (Records.unitSync e w).delta &&
    (link.syncs.getD w Check.noSync).rho == (Records.unitSync e w).rho)) = A
  match R with
  | [] => exact Or.inl rfl
  | [.certified r] =>
    dsimp only
    cases Check.screen p #[r] with
    | error v => exact Or.inl rfl
    | ok r' =>
      dsimp only [Except.map]
      cases A
      · simp only [Bool.false_eq_true, ite_false]
        split <;> simp
      · exact Or.inl rfl
  | [.raw _ _ st grid] =>
    dsimp only
    cases Records.statusViolation st with
    | some v => exact Or.inl rfl
    | none =>
      dsimp only
      cases A
      · simp only [Bool.false_eq_true, ite_false]
        split <;> simp
      · exact Or.inl rfl
  | [.committed _ _ st dg] =>
    dsimp only
    cases Records.statusViolation st with
    | some v => exact Or.inl rfl
    | none =>
      dsimp only
      cases A
      · simp only [Bool.false_eq_true, ite_false]
        split <;> simp
      · exact Or.inl rfl
  | r :: _ :: _ => cases r <;> exact Or.inl rfl

theorem gridOnly_refl (v : LinkWindow) : GridOnly v v := ⟨rfl, rfl, rfl, Or.inl rfl⟩

/-- A link-window `GridOnly` accepts every link-window it is of that is accepted. -/
theorem gridOnly_accepts {v v' : LinkWindow} (h : GridOnly v v') (hv : v.violation = none) : v'.violation = none := by
  rcases h.2.2.2 with h | ⟨-, h, -⟩
  · rw [h, hv]
  · rw [hv] at h; cases h

theorem zip_map {α β γ : Type} (R : β → γ → Prop) {L : List α} {f : α → β} {g : α → γ}
    (h : ∀ a ∈ L, R (f a) (g a)) :
    (L.map f).length = (L.map g).length ∧ ∀ x ∈ (L.map f).zip (L.map g), R x.1 x.2 := by
  induction L with
  | nil => simp
  | cons a L ih =>
    obtain ⟨-, hr⟩ := ih fun b hb => h b (List.mem_cons_of_mem _ hb)
    refine ⟨by simp, fun x hx => ?_⟩
    simp only [List.map_cons, List.zip_cons_cons, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact h a (List.mem_cons_self ..)
    · exact hr x hx

theorem zip_flatMap {α β γ : Type} (R : β → γ → Prop) {L : List α} {f : α → List β} {g : α → List γ}
    (h : ∀ a ∈ L, (f a).length = (g a).length ∧ ∀ x ∈ (f a).zip (g a), R x.1 x.2) :
    (L.flatMap f).length = (L.flatMap g).length ∧ ∀ x ∈ (L.flatMap f).zip (L.flatMap g), R x.1 x.2 := by
  induction L with
  | nil => simp
  | cons a L ih =>
    obtain ⟨hl, hr⟩ := h a (List.mem_cons_self ..)
    obtain ⟨hl', hr'⟩ := ih fun b hb => h b (List.mem_cons_of_mem _ hb)
    rw [List.flatMap_cons, List.flatMap_cons, List.zip_append hl]
    refine ⟨by simp [hl, hl'], fun x hx => ?_⟩
    rcases List.mem_append.mp hx with hx | hx
    · exact hr x hx
    · exact hr' x hx

theorem zip_append {β γ : Type} (R : β → γ → Prop) {a b : List β} {a' b' : List γ}
    (ha : a.length = a'.length ∧ ∀ x ∈ a.zip a', R x.1 x.2) (hb : b.length = b'.length ∧ ∀ x ∈ b.zip b', R x.1 x.2) :
    (a ++ b).length = (a' ++ b').length ∧ ∀ x ∈ (a ++ b).zip (a' ++ b'), R x.1 x.2 := by
  rw [List.zip_append ha.1]
  refine ⟨by simp [ha.1, hb.1], fun x hx => ?_⟩
  rcases List.mem_append.mp hx with hx | hx
  · exact ha.2 x hx
  · exact hb.2 x hx

/-- **`verdictsNoGrid` is `verdicts` but for the egress grid comparison.** -/
theorem verdicts_noGrid (p : Params) (c : Records.Commitment) (e : Records.ByteEvidence) (eg ig : ByteArray) :
    AllGridOnly (Records.verdicts p c e eg ig) (Records.verdictsNoGrid p c e ig) := by
  unfold AllGridOnly Records.verdicts Records.verdictsNoGrid
  rw [List.size_toArray, List.size_toArray, List.toList_toArray, List.toList_toArray]
  refine zip_append GridOnly (zip_flatMap GridOnly fun ⟨l, link⟩ _ => ?_)
    (zip_flatMap GridOnly fun l _ => zip_map GridOnly fun w _ => gridOnly_refl _)
  exact zip_map GridOnly fun w _ =>
    ⟨rfl, rfl, rfl, (egress_noGrid p c e eg l link _ w).imp id fun h => ⟨rfl, h⟩⟩

theorem uncertifiedEgress_iff (p : Params) (c : Records.Commitment) :
    Records.uncertifiedEgress p c = true ↔ UncertifiedEgress p c := by
  unfold Records.uncertifiedEgress UncertifiedEgress
  rw [← Array.any_toList, List.any_eq_true]
  constructor
  · rintro ⟨r, hr, h⟩
    refine ⟨r, hr, ?_⟩
    cases r <;> simp_all
  · rintro ⟨r, hr, h⟩
    refine ⟨r, hr, ?_⟩
    cases r <;> simp_all

end NoGridProofs

open NoGridProofs

/-- **What the check without the grid answers.** -/
theorem noGridMeans : NoGridMeans := by
  intro p c e vs' h
  unfold Records.evaluateNoGrid at h
  by_cases h1 : e.links.size ≠ p.egress.size
  · rw [ite_eq_left h1] at h; cases h
  rw [ite_eq_right h1] at h
  by_cases h2 : (e.links.any fun x => decide (x.syncs.size ≠ p.windows)) = true
  · rw [ite_eq_left h2] at h; cases h
  rw [ite_eq_right h2] at h
  by_cases h3 : (e.links.any fun x => x.frames.any fun x => decide (x.payload.size ≠ p.grid.frameBytes)) = true
  · rw [ite_eq_left h3] at h; cases h
  rw [ite_eq_right h3] at h
  by_cases hu : Records.uncertifiedEgress p c = true
  · rw [ite_eq_left hu] at h; cases h
  rw [ite_eq_right hu] at h
  cases hd1 : parametersDigest p.grid (some (manifestSet p.syncSet)) with
  | error _ => rw [hd1] at h; cases h
  | ok eg =>
    cases hd2 : parametersDigest p.ingressGrid none with
    | error _ => rw [hd1, hd2] at h; cases h
    | ok ig =>
      rw [hd1, hd2] at h
      cases h
      refine ⟨fun hU => hu ((uncertifiedEgress_iff p c).mpr hU), _, ?_, verdicts_noGrid p c e eg ig⟩
      unfold Records.evaluate
      rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, hd1, hd2]

/-- **The check without the grid accepts more.** -/
theorem noGridAcceptsMore : NoGridAcceptsMore := by
  intro p c e vs h
  unfold Records.evaluate at h
  by_cases h1 : e.links.size ≠ p.egress.size
  · rw [ite_eq_left h1] at h; cases h
  rw [ite_eq_right h1] at h
  by_cases h2 : (e.links.any fun x => decide (x.syncs.size ≠ p.windows)) = true
  · rw [ite_eq_left h2] at h; cases h
  rw [ite_eq_right h2] at h
  by_cases h3 : (e.links.any fun x => x.frames.any fun x => decide (x.payload.size ≠ p.grid.frameBytes)) = true
  · rw [ite_eq_left h3] at h; cases h
  rw [ite_eq_right h3] at h
  cases hd1 : parametersDigest p.grid (some (manifestSet p.syncSet)) with
  | error _ => rw [hd1] at h; cases h
  | ok eg =>
    cases hd2 : parametersDigest p.ingressGrid none with
    | error _ => rw [hd1, hd2] at h; cases h
    | ok ig =>
      rw [hd1, hd2] at h
      cases h
      unfold Records.evaluateNoGrid
      rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3]
      by_cases hu : Records.uncertifiedEgress p c = true
      · rw [ite_eq_left hu]
        exact Or.inl ⟨(uncertifiedEgress_iff p c).mp hu, _, rfl⟩
      · rw [ite_eq_right hu, hd1, hd2]
        exact Or.inr ⟨_, rfl, verdicts_noGrid p c e eg ig⟩

end Properties.NetworkRecords
