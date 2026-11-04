import Proofs.Properties.Verity.Sampled.Assembled
import Proofs.Flock.Claims.JsonNorm
import Proofs.Properties.Challenger.CanonRead
import Proofs.Properties.Challenger.Coins

/-!
# The record's draw is the session's draw file

Clause (i) of `MovesOf`: the record v2 reads in an accepted play decodes (`Exec.recordDraw`) to the session's draw file
(`Live.dj`) at the proof blocks the play's sample selects.

* **v2's refusal** (`DrawHeld`): wherever v2 drew `units` under the law `L` over `n` units (`Drew`), the units are
  ascending below `n` (`unitsAscending`) and the record's `unit_draw` prints as `UnitDraw.toJson ⟨L, n, units, #[]⟩`
  does (`recordDrawn`). It is a `Prop` about v2's code, taken as a hypothesis.
* **The record half** (`recordDraw_of_drew`): where `DrawHeld` holds, v2 drew `units`, and the session's law and
  population are `L` and `n`, the record decodes to the session's draw file at the proof blocks `units` names, at a
  session whose draw files `norm` fixes. Both are rebuilt values (`Flock.norm`) that `canonUtf8` prints alike, so they
  are equal (`eq_of_canonUtf8`): a rebuilt value is well formed (`FlockDeploy.wf_norm`), so its objects hold their keys
  in order (`sortedKeys_of_wf`), and sorted values that print alike are equivalent
  (`Properties.Challenger.CanonRead.canon_inj`). The draw file at the units' proof blocks is
  `UnitDraw.toJson ⟨L, n, units, #[]⟩` (`drawOf_filter`).
* **The draw files `norm` fixes** (`norm_printedDraw`): the printed draw file at every law but the work law. A work
  stratum's object (`WorkStratum.toJson`) lists its fields out of key order, so its tree isn't `norm`'s.
* **v2's coins fit `MAX_COINS` frames** (`coins_size_le`): v2's evidence holds its segment's openings within `LIMITS`
  (`Properties.Challenger.Limits.evidenceOf_within`), and the coins v2 reads are one frame per opening
  (`challengerStream_ok`).
-/

namespace Verity.Sampled

open Lean (Json)
open Properties.Challenger (bitsOf Draws)

/-! ## The record's draw -/

/-- **The units are ascending below `n`**: `UnitDraw.ofJson`'s check of a draw's units. -/
def unitsAscending (n : ℕ) (units : Array ℕ) : Bool :=
  (List.range units.size).all fun i => units[i]! < n && (i == 0 || units[i - 1]! < units[i]!)

/-- **The record's draw is the one v2 drew**: the units are ascending below `n`, and the record's `unit_draw`
(`Flock.Witness.recordDraw`) prints as the draw of `units` under the law `L` over `n` units does. -/
def recordDrawn (L : Flock.Draw.Law) (n : ℕ) (units : Array ℕ) (record : String) : Bool :=
  unitsAscending n units && match Flock.Witness.recordDraw record with
    | some r => match Flock.canonUtf8 r, Flock.canonUtf8 (Flock.Draw.UnitDraw.toJson ⟨L, n, units, #[]⟩) with
      | .ok a, .ok b => a == b
      | _, _ => false
    | none => false

/-- **v2 holds the record's draw to its own** in what composite `k` accepts: wherever v2 drew `units` under the law `L`
over `n` units in what the prover sent (`Drew`), the record v2 reads there is drawn so (`recordDrawn`). -/
def DrawHeld (k : Verity.Registration.Composite) : Prop :=
  ∀ s dr L n ξ units record, Drew k s dr L n ξ units → sentRecord s = some record →
    recordDrawn L n units record = true

theorem exec_recordDraw (record : String) :
    FlockSoundness.Discharge.Exec.recordDraw record = Flock.Witness.recordDraw record :=
  rfl

/-- **The units of an ascending draw** are below `n` and strictly ascending. -/
theorem unitsAscending_spec {n : ℕ} {units : Array ℕ} (h : unitsAscending n units = true) :
    (∀ x ∈ units.toList, x < n) ∧ units.toList.Pairwise (· < ·) := by
  have hi : ∀ i (hi : i < units.size), units[i] < n ∧ (i = 0 ∨ units[i - 1]! < units[i]) := fun i hi => by
    have := List.all_eq_true.1 h i (List.mem_range.2 hi)
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true, beq_iff_eq, getElem!_pos units i hi] at this
    exact this
  have step : ∀ i (h₁ : i + 1 < units.size), units[i] < units[i + 1] := fun i h₁ => by
    rcases (hi (i + 1) h₁).2 with h0 | h0
    · omega
    · simpa [getElem!_pos units i (by omega)] using h0
  have far : ∀ d i (h₁ : i + d + 1 < units.size), units[i] < units[i + d + 1] := by
    intro d
    induction d with
    | zero => exact fun i h₁ => step i h₁
    | succ d ih => exact fun i h₁ => (ih i (by omega)).trans (step (i + d + 1) h₁)
  refine ⟨fun x hx => ?_, ?_⟩
  · obtain ⟨i, hi', rfl⟩ := List.getElem_of_mem hx
    exact (hi i (by simpa using hi')).1
  · rw [List.pairwise_iff_getElem]
    intro i j h₁ h₂ hij
    obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_lt hij
    simpa using far d i (by simpa using h₂)

/-- **The verifier's own draw of the proof blocks an ascending draw names** is that draw. -/
theorem drawOf_filter {n : ℕ} {units : Array ℕ} (h : unitsAscending n units = true) (L : Flock.Draw.Law) :
    FlockSoundness.Discharge.ZkLink.drawOf L n (Finset.univ.filter fun u : Fin n => u.val ∈ units) =
      ⟨L, n, units, #[]⟩ := by
  obtain ⟨hlt, hs⟩ := unitsAscending_spec h
  have himg : (Finset.univ.filter fun u : Fin n => u.val ∈ units).image Fin.val = units.toList.toFinset := by
    ext x
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, List.mem_toFinset,
      Array.mem_toList_iff]
    constructor
    · rintro ⟨u, hu, rfl⟩
      exact hu
    · intro hx
      exact ⟨⟨x, hlt x (Array.mem_toList_iff.2 hx)⟩, hx, rfl⟩
  have hsort : units.toList.toFinset.sort (· ≤ ·) = units.toList :=
    (List.toFinset_sort (· ≤ ·) (hs.imp ne_of_lt)).2 (hs.imp le_of_lt)
  simp only [FlockSoundness.Discharge.ZkLink.drawOf, himg, hsort, Array.toArray_toList]

open Properties.Challenger.CanonRead (sortedKeys_arr sortedKeys_obj increasing_iff) in
/-- **A well-formed value's objects hold their keys in order.** -/
theorem sortedKeys_of_wf : ∀ j : Json, Claims.Json.Wf j → _root_.SampledProofs.sortedKeys j = true := by
  refine FlockDeploy.json_ind (fun _ => by simp [_root_.SampledProofs.sortedKeys])
    (fun _ _ => by simp [_root_.SampledProofs.sortedKeys]) (fun _ _ => by simp [_root_.SampledProofs.sortedKeys])
    (fun _ _ => by simp [_root_.SampledProofs.sortedKeys]) ?_ ?_
  · intro a ih hw
    cases hw with
    | arr h => exact sortedKeys_arr.2 fun x hx => ih x hx (h x hx)
  · intro m ih hw
    cases hw with
    | obj hm h =>
      refine sortedKeys_obj.2 ⟨?_, fun p hp => ?_⟩
      · rw [Std.TreeMap.Raw.toList_toArray, increasing_iff, List.pairwise_map]
        exact (Std.TreeMap.Raw.ordered_keys_toList hm).imp fun hlt => compareOfLessAndEq_eq_lt.1 hlt
      · have hp' : p ∈ m.toList := by
          rw [← Std.TreeMap.Raw.toList_toArray]
          exact Array.mem_toList_iff.2 hp
        refine ih p hp' (h p.1 p.2 ?_)
        rw [Std.TreeMap.Raw.get?_eq_getElem?]
        exact (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp'

/-- **Two values `norm` fixes that `canonUtf8` prints alike are equal.** -/
theorem eq_of_canonUtf8 {a b : Json} {s : String} (ha : Flock.norm a = a) (hb : Flock.norm b = b)
    (hsa : Flock.canonUtf8 a = .ok s) (hsb : Flock.canonUtf8 b = .ok s) : a = b := by
  have wa : Claims.Json.Wf a := by
    rw [← ha]
    exact FlockDeploy.wf_norm a
  have wb : Claims.Json.Wf b := by
    rw [← hb]
    exact FlockDeploy.wf_norm b
  exact FlockDeploy.eq_of_jsonEquiv_norm ha hb
    (Properties.Challenger.CanonRead.canon_inj (sortedKeys_of_wf a wa) (sortedKeys_of_wf b wb) hsa hsb)

/-- **The record's draw is the session's draw file at v2's units**: where v2 holds the record's draw to its own
(`DrawHeld`), v2 drew `units` under the law `L` over `n` units in what the prover sent, and the session's law and
population are `L` and `n`, the record v2 reads decodes to the session's draw file at the proof blocks `units` names,
at a session whose draw files `norm` fixes. -/
theorem recordDraw_of_drew {k : Verity.Registration.Composite} (hH : DrawHeld k) {s : Verity.Sent} {dr : Draws}
    {L : Flock.Draw.Law} {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (hd : Drew k s dr L n ξ units) (A : Live)
    (hL : A.law = L) (hn : A.n = n) (hfix : ∀ S, Flock.norm (A.dj S) = A.dj S) {record : String}
    (hr : sentRecord s = some record) :
    FlockSoundness.Discharge.Exec.recordDraw record =
      some (A.dj (Finset.univ.filter fun u : Fin A.n => u.val ∈ units)) := by
  have h := hH s dr L n ξ units record hd hr
  subst hL hn
  unfold recordDrawn at h
  rw [Bool.and_eq_true] at h
  obtain ⟨hu, h⟩ := h
  have ht : Flock.Draw.UnitDraw.toJson ⟨A.law, A.n, units, #[]⟩ =
      A.dj (Finset.univ.filter fun u : Fin A.n => u.val ∈ units) := by
    show _ = (FlockSoundness.Discharge.ZkLink.drawOf A.law A.n _).toJson
    rw [drawOf_filter hu]
  rw [ht] at h
  rw [exec_recordDraw]
  cases hw : Flock.Witness.recordDraw record with
  | none => simp [hw] at h
  | some r =>
    have hnr : Flock.norm r = r := Claims.Bridge.Custody.recordDraw_norm (record := record) (by
      rw [exec_recordDraw]
      exact hw)
    cases ha : Flock.canonUtf8 r with
    | error _ => simp [hw, ha] at h
    | ok a =>
      cases hb : Flock.canonUtf8 (A.dj (Finset.univ.filter fun u : Fin A.n => u.val ∈ units)) with
      | error _ => simp [hw, ha, hb] at h
      | ok b =>
        simp only [hw, ha, hb, beq_iff_eq] at h
        subst h
        exact congrArg some (eq_of_canonUtf8 hnr (hfix _) ha hb)

/-! ## The draw files `norm` fixes -/

theorem norm_arr_of {a : Array Json} (h : ∀ x ∈ a, Flock.norm x = x) : Flock.norm (.arr a) = .arr a := by
  rw [FlockDeploy.norm_arr]
  congr 1
  exact (Array.map_congr_left h).trans (Array.map_id' a)

theorem norm_nats (a : Array ℕ) :
    Flock.norm (.arr (a.map fun x => Lean.toJson x)) = .arr (a.map fun x => Lean.toJson x) :=
  norm_arr_of fun x hx => by
    obtain ⟨y, -, rfl⟩ := Array.mem_map.1 hx
    rfl

theorem norm_stratum (s : Flock.Draw.Stratum) : Flock.norm s.toJson = s.toJson := by
  have h : ∀ a b c : Json, Flock.norm (Json.mkObj [("template", a), ("units", b), ("k", c)]) =
      Json.mkObj [("template", Flock.norm a), ("units", Flock.norm b), ("k", Flock.norm c)] :=
    fun _ _ _ => rfl
  show Flock.norm (Json.mkObj [("template", Json.str s.template),
    ("units", Json.arr (s.units.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])),
    ("k", Lean.toJson s.k)]) = _
  rw [h]
  congr 4
  refine norm_arr_of fun x hx => ?_
  obtain ⟨⟨lo, hi⟩, -, rfl⟩ := Array.mem_map.1 hx
  exact norm_arr_of fun y hy => by
    simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hy
    rcases hy with rfl | rfl <;> rfl

/-- **`norm` fixes the printed draw file at every law but the work law.** -/
theorem norm_printedDraw {n : ℕ} {law : Flock.Draw.Law} (hw : ∀ K ws, law ≠ .work K ws) (S : Finset (Fin n)) :
    Flock.norm (printedDraw law S) = printedDraw law S := by
  have hu := norm_nats ((((S.image Fin.val).sort (· ≤ ·)).toArray))
  cases law with
  | subset k =>
    have h : ∀ a b c d : Json, Flock.norm (Json.mkObj [("law", a), ("population", b), ("k", c), ("units", d)]) =
        Json.mkObj [("law", Flock.norm a), ("population", Flock.norm b), ("k", Flock.norm c),
          ("units", Flock.norm d)] :=
      fun _ _ _ _ => rfl
    show Flock.norm (Json.mkObj [("law", "subset"), ("population", Lean.toJson n), ("k", Lean.toJson k),
      ("units", .arr (((S.image Fin.val).sort (· ≤ ·)).toArray.map fun x => Lean.toJson x))]) = _
    rw [h, hu]
    rfl
  | bernoulli a b =>
    have h : ∀ w x y z : Json, Flock.norm (Json.mkObj [("law", w), ("population", x), ("p", y), ("units", z)]) =
        Json.mkObj [("law", Flock.norm w), ("population", Flock.norm x), ("p", Flock.norm y),
          ("units", Flock.norm z)] :=
      fun _ _ _ _ => rfl
    show Flock.norm (Json.mkObj [("law", "bernoulli"), ("population", Lean.toJson n), ("p", Json.str s!"{a}/{b}"),
      ("units", .arr (((S.image Fin.val).sort (· ≤ ·)).toArray.map fun x => Lean.toJson x))]) = _
    rw [h, hu]
    rfl
  | stratified K St =>
    have h : ∀ a b c d e : Json,
        Flock.norm (Json.mkObj [("law", a), ("population", b), ("k", c), ("strata", d), ("units", e)]) =
          Json.mkObj [("law", Flock.norm a), ("population", Flock.norm b), ("k", Flock.norm c),
            ("strata", Flock.norm d), ("units", Flock.norm e)] :=
      fun _ _ _ _ _ => rfl
    show Flock.norm (Json.mkObj [("law", "stratified"), ("population", Lean.toJson n), ("k", Lean.toJson K),
      ("strata", Json.arr (St.map (·.toJson))),
      ("units", .arr (((S.image Fin.val).sort (· ≤ ·)).toArray.map fun x => Lean.toJson x))]) = _
    rw [h, hu, norm_arr_of fun x hx => by
      obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
      exact norm_stratum s]
    rfl
  | work K ws => exact absurd rfl (hw K ws)

/-- **`norm` fixes the session's draw files** at a session whose law isn't the work law. -/
theorem Live.norm_dj (A : Live) (hw : ∀ K ws, A.law ≠ .work K ws) : ∀ S, Flock.norm (A.dj S) = A.dj S :=
  norm_printedDraw hw

/-! ## v2's coins fit `MAX_COINS` frames -/

/-- **The coins v2 reads at an accepted case fit `MAX_COINS` frames**: v2's evidence holds its segment's openings
within `LIMITS`, and v2 reads one frame of `COIN_BYTES` per opening. -/
theorem coins_size_le {bs : Core.Blobs} {a b e : ByteArray} {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {ev : _root_.SampledProofs.OneStageZk.Evidence} {ξ : ByteArray}
    {dr : Draws} (h : V2Accepts bs a b e p c ev ξ dr) :
    ξ.size ≤ _root_.SampledProofs.Challenger.COIN_BYTES * _root_.SampledProofs.Challenger.MAX_COINS := by
  obtain ⟨-, -, hev, hξ, hdr, -⟩ := h
  obtain ⟨b', hb'⟩ := Binding.dispatch_evidence hev
  obtain ⟨-, -, -, -, hops, -⟩ :=
    Properties.Challenger.Limits.withinAt_spec (Properties.Challenger.Limits.evidenceOf_within hb')
  obtain ⟨raw, hraw, rfl⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  obtain ⟨cm, roots, -, hstream⟩ := drawCoins_ok hraw
  obtain ⟨l, start, openings, os, -, -, hop, -, -, hr⟩ := Properties.Challenger.Coins.stream_ok hstream
  rw [challengerStream_ok hstream, ← zipIdx_filter_length, hr, List.length_range']
  rw [hop] at hops
  exact Nat.mul_le_mul (le_refl _) hops

end Verity.Sampled
