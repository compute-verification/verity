/-
The lemmas `StreamSound` is proved from: what the stream computes on the blobs it holds is what `checkBlobs` computes
on all the blobs fed.

* `find?_sublist`, `Blobs.get_sublist`, `tablesOf_sublist`: a search of the blobs held finds what a search of all the
  blobs fed finds, where the blobs held are a sublist of those fed, their SHA-512 distinct, and the one entry the
  search can match is held;
* `verdictOf_split`, `StepKept.step_of`: a session's `step` is its kept pieces' step, its verdict split at the
  coin-tree key's freshness, the one piece that reads the sessions before it.
-/
module

public import Spot.OneStageZkStream

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk.Streamed

open Lean (Json)
open Flock (V)

/-- Two entries of a list whose first parts are distinct are one where their first parts are. -/
theorem key_unique {α β : Type} : ∀ {l : List (α × β)}, (l.map (·.1)).Nodup →
    ∀ {x y : α × β}, x ∈ l → y ∈ l → x.1 = y.1 → x = y
  | [], _, _, _, hx, _, _ => by cases hx
  | a :: l, hn, x, y, hx, hy, h => by
    rw [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.1 hx with hxa | hx' <;> rcases List.mem_cons.1 hy with hya | hy'
    · rw [hxa, hya]
    · subst hxa
      exact absurd (by rw [h]; exact List.mem_map_of_mem (f := fun z : α × β => z.1) hy') hn.1
    · subst hya
      exact absurd (by rw [← h]; exact List.mem_map_of_mem (f := fun z : α × β => z.1) hx') hn.1
    · exact key_unique hn.2 hx' hy' h

/-- **A search of a sublist finds what a search of the list finds**, where at most one entry of the list matches and
every entry that matches is in the sublist. -/
theorem find?_sublist {α : Type} {p : α → Bool} {l₁ l₂ : List α} (hs : l₁.Sublist l₂)
    (hu : ∀ x ∈ l₂, ∀ y ∈ l₂, p x = true → p y = true → x = y) (hin : ∀ x ∈ l₂, p x = true → x ∈ l₁) :
    l₁.find? p = l₂.find? p := by
  cases h₂ : l₂.find? p with
  | none =>
    rw [List.find?_eq_none] at h₂ ⊢
    exact fun x hx => h₂ x (hs.subset hx)
  | some x =>
    have hx := List.mem_of_find?_eq_some h₂
    have hp := List.find?_some h₂
    cases h₁ : l₁.find? p with
    | none => exact absurd hp (List.find?_eq_none.mp h₁ x (hin x hx hp))
    | some y =>
      exact congrArg some (hu y (hs.subset (List.mem_of_find?_eq_some h₁)) x hx (List.find?_some h₁) hp)

/-- `==` on bytes is equality. -/
theorem bytes_beq {a b : ByteArray} : (a == b) = true ↔ a = b := by
  constructor
  · intro h
    exact ByteArray.ext (eq_of_beq (show (a.data == b.data) = true from h))
  · rintro rfl
    exact (show (a.data == a.data) = true from beq_self_eq_true _)

/-- **The blob held under a digest is the blob fed under it**, where the blobs held are a sublist of those fed, their
SHA-512 distinct, and the entry fed under the digest, if any, is held; a refusal of the digest is the same refusal. -/
theorem Blobs.get_sublist {held full : Core.Blobs} (hs : held.entries.toList.Sublist full.entries.toList)
    (hn : (full.entries.toList.map (·.1)).Nodup) {digest what : String}
    (hin : ∀ x ∈ full.entries.toList, Flock.hexOf x.1 = digest → x ∈ held.entries.toList) :
    held.get digest what = full.get digest what := by
  unfold Core.Blobs.get
  cases hu : Flock.unhex digest with
  | error e => rfl
  | ok d =>
    show (if (d.size != 64 || Flock.hexOf d != digest) = true then _ else _) =
      (if (d.size != 64 || Flock.hexOf d != digest) = true then _ else _)
    by_cases hc : (d.size != 64 || Flock.hexOf d != digest) = true
    · rw [if_pos hc, if_pos hc]
      rfl
    · rw [if_neg hc, if_neg hc]
      have hd : Flock.hexOf d = digest := by
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
        exact hc.2
      have hf : held.entries.find? (·.1 == d) = full.entries.find? (·.1 == d) := by
        rw [← Array.find?_toList, ← Array.find?_toList]
        refine find?_sublist hs (fun x hx y hy hpx hpy => key_unique hn hx hy ?_) (fun x hx hpx => hin x hx ?_)
        · exact (bytes_beq.mp hpx).trans (bytes_beq.mp hpy).symm
        · rw [bytes_beq.mp hpx, hd]
      rw [hf]

theorem filterMap_congr' {α β : Type} {f g : α → Option β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.filterMap f = l.filterMap g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.filterMap_cons, List.filterMap_cons, h a List.mem_cons_self,
      filterMap_congr' fun x hx => h x (List.mem_cons_of_mem _ hx)]

/-- Two digests with one hex are one. -/
theorem hexOf_inj {a b : ByteArray} (h : Flock.hexOf a = Flock.hexOf b) : a = b := by
  have ha := Flock.Hex.unhex_hexOf a
  rw [h, Flock.Hex.unhex_hexOf b] at ha
  exact (Except.ok.inj ha).symm

/-- **The pinned tables held are the pinned tables fed**, where the blobs held are a sublist of those fed, their
SHA-512 distinct, and every blob fed under a pinned table's SHA-512 is held. -/
theorem tablesOf_sublist {held full : Core.Blobs} (hs : held.entries.toList.Sublist full.entries.toList)
    (hn : (full.entries.toList.map (·.1)).Nodup)
    (hin : ∀ x ∈ full.entries.toList, (∃ t ∈ Flock.MUFU_TABLES, Flock.hexOf x.1 = t.sha512) →
      x ∈ held.entries.toList) :
    tablesOf held = tablesOf full := by
  unfold tablesOf
  refine filterMap_congr' fun t ht => ?_
  rw [← Array.find?_toList, ← Array.find?_toList]
  refine congrArg _ (find?_sublist hs (fun x hx y hy hpx hpy => ?_) (fun x hx hpx => ?_))
  · rcases x with ⟨dx, bx⟩
    rcases y with ⟨dy, by_⟩
    dsimp only at hpx hpy
    exact key_unique hn hx hy (hexOf_inj ((beq_iff_eq.mp hpx).trans (beq_iff_eq.mp hpy).symm))
  · rcases x with ⟨dx, bx⟩
    dsimp only at hpx
    exact hin _ hx ⟨t, ht, beq_iff_eq.mp hpx⟩

/-- **`verdictOf` is the session's proof files, its key's freshness, then the rest** (`restOf`), the files read once
for the rest's proofs: the split `StepKept` keeps. -/
theorem verdictOf_split (I : Flock.Witness.Inputs) (scope ksFiles population : V Unit) (coinKeys : Array String)
    (record : String) (draw : Option Json) (st : SetUp) (files : V (Array ByteArray)) (coinKey : Option String) :
    verdictOf I scope ksFiles population coinKeys record draw st files coinKey =
      ((files >>= fun _ => pure ()) >>= fun _ =>
        (match coinKey with | some key => Flock.CoinTree.checkFresh coinKeys key | none => pure () : V Unit) >>=
          fun _ => files >>= restOf I scope ksFiles population record draw st) := by
  cases files <;> cases coinKey <;> rfl

/-- **A session's `step` is the step on what is kept of it**, after any run: the verdict's pieces made once, on the
blobs held, and only the coin-tree key's freshness read against the run. -/
theorem StepKept.step_of (a : StepAt) (bs : Core.Blobs) (s : RunFiles) (r : Run) :
    (StepKept.of a bs s).step r = a.step bs r s := by
  unfold StepKept.of StepAt.step OneStageZk.step
  cases hg : bs.get s.record s!"session {s.name}'s record" with
  | error e => rfl
  | ok rb =>
    simp only [ok_bind]
    cases hu : String.fromUTF8? rb with
    | none => rfl
    | some record =>
      dsimp only
      cases hd : drawFitsWhy WORD (Flock.Witness.recordDraw record) with
      | some what => rfl
      | none =>
        simp only [verdictOf_split]
        rfl

/-- **What the plan's distinctness check holds where it passes**: no digest is one `seen` holds, their hex is
distinct, and none is a pinned table's. -/
theorem distinctFrom_ok : ∀ {seen : Std.HashSet String} {l : List ByteArray}, distinctFrom seen l = .ok () →
    (∀ b ∈ l, seen.contains (Flock.hexOf b) = false) ∧ (l.map Flock.hexOf).Nodup ∧ ∀ b ∈ l, b ∉ pinned
  | _, [], _ => ⟨by simp, List.nodup_nil, by simp⟩
  | seen, b :: bs, h => by
    unfold distinctFrom at h
    by_cases hp : pinned.any (fun t => decide (t = b)) = true
    · rw [if_pos hp] at h
      cases h
    rw [if_neg hp] at h
    by_cases hs : seen.contains (Flock.hexOf b) = true
    · rw [if_pos hs] at h
      cases h
    rw [if_neg hs] at h
    obtain ⟨hseen, hnd, hpin⟩ := distinctFrom_ok h
    have hne : ∀ b' ∈ bs, Flock.hexOf b ≠ Flock.hexOf b' ∧ seen.contains (Flock.hexOf b') = false := by
      intro b' hb'
      have := hseen b' hb'
      rw [Std.HashSet.contains_insert, Bool.or_eq_false_iff] at this
      exact ⟨fun he => by simp [he] at this, this.2⟩
    refine ⟨?_, ?_, ?_⟩
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · simpa using hs
      · exact (hne x hx).2
    · rw [List.map_cons, List.nodup_cons]
      refine ⟨fun hm => ?_, hnd⟩
      obtain ⟨b', hb', he⟩ := List.mem_map.1 hm
      exact (hne b' hb').1 he.symm
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · intro hm
        exact hp (List.any_eq_true.2 ⟨x, hm, by simp⟩)
      · exact hpin x hx

/-- Digests whose hex is distinct are distinct. -/
theorem nodup_of_hex {l : List ByteArray} (h : (l.map Flock.hexOf).Nodup) : l.Nodup :=
  List.Pairwise.of_map Flock.hexOf (fun _ _ hne he => hne (congrArg Flock.hexOf he)) h

end Spot.OneStageZk.Streamed
