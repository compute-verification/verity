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
    rcases List.mem_cons.1 hx with rfl | hx <;> rcases List.mem_cons.1 hy with rfl | hy
    · rfl
    · exact absurd (by rw [h]; exact List.mem_map_of_mem (f := fun z : α × β => z.1) hy) hn.1
    · exact absurd (by rw [← h]; exact List.mem_map_of_mem (f := fun z : α × β => z.1) hx) hn.1
    · exact key_unique hn.2 hx hy h

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
    · rw [if_neg hc, if_neg hc]
      have hd : Flock.hexOf d = digest := by
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
        exact hc.2
      have hf : held.entries.find? (·.1 == d) = full.entries.find? (·.1 == d) := by
        rw [← Array.find?_toList, ← Array.find?_toList]
        refine find?_sublist hs (fun x hx y hy hpx hpy => key_unique hn hx hy ?_) (fun x hx hpx => hin x hx ?_)
        · exact (beq_iff_eq.mp hpx).trans (beq_iff_eq.mp hpy).symm
        · rw [beq_iff_eq.mp hpx, hd]
      rw [hf]

/-- Two digests with one hex are one. -/
theorem hexOf_inj {a b : ByteArray} (h : Flock.hexOf a = Flock.hexOf b) : a = b := by
  have ha := Flock.unhex_hexOf a
  rw [h, Flock.unhex_hexOf b] at ha
  exact (Except.ok.inj ha).symm

/-- **The pinned tables held are the pinned tables fed**, where the blobs held are a sublist of those fed, their
SHA-512 distinct, and every blob fed under a pinned table's SHA-512 is held. -/
theorem tablesOf_sublist {held full : Core.Blobs} (hs : held.entries.toList.Sublist full.entries.toList)
    (hn : (full.entries.toList.map (·.1)).Nodup)
    (hin : ∀ x ∈ full.entries.toList, (∃ t ∈ Flock.MUFU_TABLES, Flock.hexOf x.1 = t.sha512) →
      x ∈ held.entries.toList) :
    tablesOf held = tablesOf full := by
  unfold tablesOf
  congr 1
  funext t
  rw [← Array.find?_toList, ← Array.find?_toList]
  by_cases ht : t ∈ Flock.MUFU_TABLES
  · rw [find?_sublist hs (fun x hx y hy hpx hpy => key_unique hn hx hy (hexOf_inj
      ((beq_iff_eq.mp hpx).trans (beq_iff_eq.mp hpy).symm))) (fun x hx hpx => hin x hx ⟨t, ht, beq_iff_eq.mp hpx⟩)]
  · rfl

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
    cases hu : String.fromUTF8? rb with
    | none => rfl
    | some record =>
      cases hd : drawFitsWhy WORD (Flock.Witness.recordDraw record) with
      | some what => rfl
      | none =>
        simp only [verdictOf_split]
        rfl

end Spot.OneStageZk.Streamed
