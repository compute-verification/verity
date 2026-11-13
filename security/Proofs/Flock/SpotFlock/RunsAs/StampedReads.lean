module

public import Proofs.Flock.SpotFlock.RunsAs.VerdictsWf
public import Proofs.Flock.SpotFlock.RunsAs.OwnReadsForm
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF

/-!
# The reads the verdicts report are the same before and after `stamped` names their sessions

The two-stage form runs its sessions (`Spot.OneStageZk.membersRun`, verdicts `r.out`), then `stamped` sets five keys
on each verdict (`sessions`, `public_header_sha512`, `session_tables`, and where they apply `registered_sha512` and
`unit_draw`) and hands the result to the run of record as its verdicts. The form's last step reads `r.out`; the run of
record reads the stamped verdicts. `Spot.Record.sessionRegistered` reads three keys of a verdict, `of_record`,
`accepted` and `registered` (`reads3`), none of which `stamped` sets, so it returns the same reads on both
(`inputAt_reported`): the premise `VerdictsReadSame` holds (`verdictsReadSame`).

* `sessionRegistered_congr`: `sessionRegistered` is a function of the verdicts' three reads, in order.
* `getD_setObjVals`, `reads3_stamp`: setting other keys on an object over a well-formed map leaves the three reads.
* `stamped_reads`: where `stamped` returns, on verdicts that are objects (`OutWf`), it returns verdicts with the same
  three reads, in the same order.
* `ownReads_form_of_record`: `accepted_ownReads_form` with the verdicts' premise discharged.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

private theorem bind_iff {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} :
    (x >>= f) = .ok b ↔ ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x <;> simp [bind, Except.bind]

private theorem bind_ok' {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := bind_iff.1 h

private theorem throw_ne' {α : Type} (e : String) (a : α) : (throw e : Flock.V α) ≠ .ok a := fun h => by cases h

private theorem map_ok' {ε α β : Type} {x : Except ε α} {f : α → β} {b : β} (h : f <$> x = .ok b) :
    ∃ a, x = .ok a ∧ f a = b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, Except.ok.inj h⟩

private theorem size_of_not_bne {a b : Nat} (h : ¬(a != b) = true) : a = b :=
  Decidable.byContradiction fun hne => h (bne_iff_ne.2 hne)

/-- The three reads `Spot.Record.sessionRegistered` makes of a verdict: its `of_record`, `accepted` and
`registered`. -/
def reads3 (x : String × Json) : Json × Json × Json :=
  (Spot.getD x.2 "of_record", Spot.getD x.2 "accepted", Spot.getD x.2 "registered")

theorem forIn_reads {β : Type} (f : String × Json → β → Id (ForInStep β))
    (hf : ∀ x y, reads3 x = reads3 y → f x = f y) (l₁ l₂ : List (String × Json)) (b : β)
    (h : l₁.map reads3 = l₂.map reads3) : forIn l₁ b f = forIn l₂ b f := by
  induction l₁ generalizing l₂ b with
  | nil =>
    cases l₂ with
    | nil => rfl
    | cons y l₂ => simp at h
  | cons x l₁ ih =>
    cases l₂ with
    | nil => simp at h
    | cons y l₂ =>
      simp only [List.map_cons, List.cons.injEq] at h
      rw [List.forIn_cons, List.forIn_cons, hf x y h.1]
      refine bind_congr fun s => ?_
      cases s with
      | done b' => rfl
      | yield b' => exact ih l₂ b' h.2

/-- **`sessionRegistered` is a function of the verdicts' three reads, in order**: verdicts with the same
`of_record`, `accepted` and `registered`, one for one, report the same reads. -/
theorem sessionRegistered_congr {a b : Array (String × Json)} (h : a.toList.map reads3 = b.toList.map reads3) :
    Spot.Record.sessionRegistered a = Spot.Record.sessionRegistered b := by
  have key : ∀ {β : Type} (b0 : β) (F : String × Json → β → Id (ForInStep β)),
      (∀ x y, reads3 x = reads3 y → F x = F y) → forIn a b0 F = forIn b b0 F := by
    intro β b0 F hF
    rw [← Array.forIn_toList, ← Array.forIn_toList]
    exact forIn_reads F hF _ _ _ h
  unfold Spot.Record.sessionRegistered
  try dsimp only
  rw [key]
  rintro ⟨kx, vx⟩ ⟨ky, vy⟩ hxy
  simp only [reads3, Prod.mk.injEq] at hxy
  obtain ⟨h1, h2, h3⟩ := hxy
  funext r
  simp only [h1, h2, h3]

/-- Setting keys other than `key` on an object over a well-formed map leaves its entry at `key`. -/
theorem getD_setObjVals {key : String} (F : Json → String × Json → Json)
    (hF : ∀ acc p, F acc p = acc.setObjVal! p.1 p.2) (fields : List (String × Json))
    (m : Std.TreeMap.Raw String Json compare) (hm : m.WF) (hk : ∀ p ∈ fields, p.1 ≠ key) :
    Spot.getD (fields.foldl F (.obj m)) key = Spot.getD (.obj m) key := by
  induction fields generalizing m with
  | nil => rfl
  | cons p fields ih =>
    rw [List.foldl_cons, hF, setObjVal_obj, ih (m.insert p.1 p.2) hm.insert
      (fun q hq => hk q (List.mem_cons.2 (Or.inr hq))), getD_obj, getD_obj,
      Std.TreeMap.Raw.getElem?_insert hm]
    simp only [cmp_ne (hk p (List.mem_cons.2 (Or.inl rfl))), ↓reduceIte]

/-- Setting keys other than `of_record`, `accepted` and `registered` on a verdict leaves its three reads. -/
theorem reads3_stamp {k : String} (F : Json → String × Json → Json) (hF : ∀ acc p, F acc p = acc.setObjVal! p.1 p.2)
    (fields : List (String × Json)) (m : Std.TreeMap.Raw String Json compare) (hm : m.WF)
    (hk : ∀ p ∈ fields, p.1 ≠ "of_record" ∧ p.1 ≠ "accepted" ∧ p.1 ≠ "registered") :
    reads3 (k, fields.foldl F (.obj m)) = reads3 (k, .obj m) := by
  have h1 := getD_setObjVals F hF fields m hm (key := "of_record") fun p hp => (hk p hp).1
  have h2 := getD_setObjVals F hF fields m hm (key := "accepted") fun p hp => (hk p hp).2.1
  have h3 := getD_setObjVals F hF fields m hm (key := "registered") fun p hp => (hk p hp).2.2
  show (Spot.getD (fields.foldl F (.obj m)) "of_record", Spot.getD (fields.foldl F (.obj m)) "accepted",
      Spot.getD (fields.foldl F (.obj m)) "registered") =
    (Spot.getD (.obj m) "of_record", Spot.getD (.obj m) "accepted", Spot.getD (.obj m) "registered")
  rw [h1, h2, h3]

theorem mapM_reads {α : Type} (G : α × (String × Json) → Flock.V (String × Json)) :
    ∀ (l : List (α × (String × Json))) (ys : List (String × Json)),
      l.mapM G = .ok ys → (∀ z ∈ l, ∀ y, G z = .ok y → reads3 y = reads3 z.2) →
      ys.map reads3 = (l.map Prod.snd).map reads3
  | [], ys, h, _ => by
    rw [List.mapM_nil] at h
    obtain rfl : [] = ys := Except.ok.inj h
    rfl
  | z :: l, ys, h, hG => by
    rw [List.mapM_cons] at h
    obtain ⟨y, hy, h⟩ := bind_ok' h
    obtain ⟨ys', hys, h⟩ := bind_ok' h
    obtain rfl : y :: ys' = ys := Except.ok.inj h
    rw [List.map_cons, List.map_cons, List.map_cons, hG z (List.mem_cons.2 (Or.inl rfl)) y hy,
      mapM_reads G l ys' hys fun z' hz' => hG z' (List.mem_cons.2 (Or.inr hz'))]

/-- **`stamped` leaves the three reads of each verdict**: where it returns, on verdicts that are objects over
well-formed maps, the verdicts it returns have the same `of_record`, `accepted` and `registered`, one for one. -/
theorem stamped_reads {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    {bs : Core.Blobs} {out vs : Array (String × Json)}
    (h : VStar.Recursive.SampledTwo.stamped p e bs out = .ok vs) (hw : OutWf out) :
    vs.toList.map reads3 = out.toList.map reads3 := by
  unfold VStar.Recursive.SampledTwo.stamped at h
  try dsimp only at h
  split at h
  · obtain ⟨_, hthrow, -⟩ := bind_ok' h
    cases hthrow
  rename_i hsize
  have hsz := size_of_not_bne hsize
  rw [Array.mapM_eq_mapM_toList] at h
  obtain ⟨ys, hys, rfl⟩ := map_ok' h
  have hz := congrArg Array.toList (Array.map_snd_zip (Nat.le_of_eq hsz.symm))
  rw [Array.toList_map] at hz
  refine (mapM_reads _ _ ys hys ?_).trans (congrArg (List.map reads3) hz)
  rintro ⟨⟨member, s⟩, k, v⟩ hz' y hy
  obtain ⟨m, hv, hm⟩ := hw _ (Array.of_mem_zip (Array.mem_toList_iff.1 hz')).2
  have hv' : v = .obj m := hv
  subst hv'
  try dsimp only at hy
  split at hy
  · obtain ⟨b1, -, hy⟩ := bind_ok' hy
    obtain ⟨hd, -, hy⟩ := bind_ok' hy
    obtain ⟨b2, -, hy⟩ := bind_ok' hy
    split at hy
    · try dsimp only at hy
      split at hy <;> (try simp only [pure_bind] at hy)
      all_goals
        repeat (obtain ⟨_, -, hy⟩ := bind_ok' hy)
        obtain rfl := Except.ok.inj hy
        refine reads3_stamp _ (by rintro acc ⟨f, x⟩; rfl) _ m hm ?_
        split <;> simp
    · exact absurd hy (throw_ne' _ _)
  · exact absurd hy (throw_ne' _ _)

/-- The run of record's verdicts are the ones the form hands it. -/
theorem inputOf_verdicts (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment) (ξ ξ₂ : ByteArray)
    (e : VStar.Recursive.SampledTwo.Evidence) (received₂ : String) (vs : Array (String × Json)) :
    (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).verdicts = vs := rfl

/-- **The run of record's verdicts report the reads the sessions' own verdicts report**: `inputAt`'s input carries
the verdicts `stamped` returned on `r.out`, and `sessionRegistered` reads the same of both. -/
theorem inputAt_reported {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {bs : Core.Blobs} {c₂ : Spot.OneStage.Commitment} {ξ₂ : ByteArray}
    {r : Spot.OneStageZk.Run} {i : Spot.Run.Input}
    (h : VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i)) :
    Spot.Record.sessionRegistered i.verdicts = Spot.Record.sessionRegistered r.out := by
  obtain ⟨-, -, hrun, -, vs, received₂, j₁, j₂, hst, rfl, -, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok h
  rw [inputOf_verdicts]
  exact sessionRegistered_congr (stamped_reads hst (membersRun_wf hrun))

/-- The verdicts' premise of `accepted_ownReads_form` holds. -/
theorem verdictsReadSame : VerdictsReadSame := fun _ _ _ _ _ _ _ _ _ h => inputAt_reported h

/-- **At the form's accept, each read its last step's record checks leave out is one the second-stage statement
check held**, given that the interior record the run of record reads is the record the form parsed
(`RecordReadSame`): `accepted_ownReads_form` with the verdicts' premise discharged. -/
theorem ownReads_form_of_record (hrec : RecordReadSame) {form : Bool} {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {ξ : ByteArray} {e : VStar.Recursive.SampledTwo.Evidence} {bs pub prof}
    (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (h : VStar.Recursive.SampledTwo.checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i it, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ i.interior = some it ∧
      ∀ opened v root,
        Spot.Record.ownRegistered c₂.record (Spot.Record.sessionRegistered r.out) = some opened →
        (v, root) ∈ opened → (v, root) ∉ (Spot.Record.lessOwnReads c₂.record (some opened)).getD #[] →
        ∃ x, (v, x) ∈ Spot.Record.byPort c₂.record ∧ Spot.jeq (Spot.getD x "root") root = true ∧
          (Spot.jeq (Spot.getD x "root") .null = true ∨ HeldRead i it v root) :=
  accepted_ownReads_form hrec verdictsReadSame hp h

end SpotFlock.RunsAs
