module

public import Proofs.Flock.Encodings.Relink
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric

/-!
# Toys for the registered encodings' statements

The instances the registered encodings' witnesses (`EncodingsGeneric`) read, each at every hash; none makes a call.

* `repCode`: the one-column repetition code of a bit, whose column check passes `true` only. Its distance is one, its
  radius zero, and its column check binds (`repCode_meets`), so its scheme binds (`repCode_binds`, by `codeBinds`).
  Its openings of `true` pass, as `trueCm`'s do, so the one pair `pair₀` links it to the record `trueCm` at position `0`
  (`enc₀`, `linked₀`), and the toy statement's opening of its input is in the sampling relation under their sum at
  `δ = 0` (`relS₀`).
* `badCode`: a code with no columns, so its openings pass whatever value they claim (`badCode_fails`); the sum of two
  commitments whose openings all pass doesn't bind (`byPos_fails`); the record's and encoding's openings of different
  values at a linked position under `accCm` make no call (`linkColl_fails`); and where the record rejects every opening,
  the relinked toy trace is outside the sampling relation, with no call (`relink_fails`).
-/

@[expose] public section

namespace Claims.Witness.Encodings

open Claims Claims.Sampling Claims.Encodings
open Claims.Witness.Sampling (accCm rejCm toyStmt toyPart toyWit toy_faulty toy_Pa rejCm_faulty)
open Claims.Witness.SamplingGeneric (trueCm trueCm_binds)

/-! ## The repetition code and its link to the record -/

/-- The one-column repetition code of a bit: an opening's message is whether its data is empty, its one column that bit,
which its check passes when `true`. -/
def repCode : CodeScheme Bool Bool Unit where
  len _ := 1
  agree _ := 1
  encode _ m _ := m
  bit _ m _ := some m
  parse d := some (d.isEmpty, [(0, d.isEmpty, ())])
  colOK _ _ c _ := .ret c

theorem repCode_distance : repCode.Distance fun _ => 1 := by
  intro roots m m' h g
  by_contra hne
  have hm : repCode.encode roots m 0 ≠ repCode.encode roots m' 0 := fun e => hne (by rw [show m = m' from e])
  have hmem : 0 ∈ (Finset.range (repCode.len roots)).filter fun i =>
      repCode.encode roots m i ≠ repCode.encode roots m' i :=
    Finset.mem_filter.2 ⟨Finset.mem_range.2 Nat.one_pos, hm⟩
  have hpos := Finset.card_pos.2 ⟨0, hmem⟩
  have h1 : _ < 1 := h
  omega

theorem repCode_colBinds (H : Msg → Msg) : repCode.ColBinds H := by
  intro roots i c c' pf pf' hne hc hc'
  have h1 : c = true := hc
  have h2 : c' = true := hc'
  exact absurd (h1.trans h2.symm) hne

/-- **`repCode` meets `CodeBinds`' premises**: distance one, radius zero, a binding column check. -/
theorem repCode_meets (H : Msg → Msg) :
    repCode.Distance (fun _ => 1) ∧ (∀ roots, 2 * (repCode.len roots - repCode.agree roots) < 1) ∧
      repCode.ColBinds H :=
  ⟨repCode_distance, fun _ => by simp [repCode], repCode_colBinds H⟩

theorem repCode_binds (H : Msg → Msg) : repCode.scheme.Binds H :=
  codeBinds Bool Bool Unit H repCode (fun _ => 1) repCode_distance (fun _ => by simp [repCode]) (repCode_colBinds H)

/-- The roots encode position `0`. -/
def enc₀ : List Msg → ℕ → Bool := fun _ g => g == 0

/-- The link's one pair, at position `0`: the record's opening of `true` and the encoding's. -/
def pair₀ : Opening Bool × Opening Bool := (⟨0, true, []⟩, ⟨0, true, []⟩)

theorem linkedAt₀ (H : Msg → Msg) : linkedAt H trueCm repCode.scheme [] [pair₀] 0 = some true := rfl

theorem linked₀ (H : Msg → Msg) : Linked H trueCm repCode.scheme [] (fun g => enc₀ [] g = true) [pair₀] := by
  intro g hg
  obtain rfl : g = 0 := by simpa [enc₀] using hg
  rw [linkedAt₀]
  exact Option.some_ne_none _

/-- The toy statement's opening of its input is in the sampling relation under the sum, at `δ = 0`. -/
theorem relS₀ (H : Msg → Msg) : RelS H (byPos enc₀ repCode.scheme trueCm) 0 toyStmt toyPart toyWit := by
  have hval : toyWit.val H (byPos enc₀ repCode.scheme trueCm) toyStmt 0 = some true := rfl
  have hf : faulty H (byPos enc₀ repCode.scheme trueCm) toyStmt toyPart toyWit = ∅ := by
    rw [toy_faulty, hval]
    rfl
  show faulty H (byPos enc₀ repCode.scheme trueCm) toyStmt toyPart toyWit ∈ toyStmt.design.M toyPart 0
  rw [hf]
  show (0 : ℝ) < toyStmt.design.Pa toyPart ∅
  rw [toy_Pa]
  simp [window]

/-- **`RelinkR`'s premises hold together** at the repetition code, the record `trueCm`, the pair `pair₀` and the toy
trace. -/
theorem relink_meets (H : Msg → Msg) :
    repCode.scheme.Binds H ∧ trueCm.Binds H ∧
      Linked H trueCm repCode.scheme toyStmt.roots (fun g => enc₀ toyStmt.roots g = true) [pair₀] ∧
      RelS H (byPos enc₀ repCode.scheme trueCm) 0 toyStmt toyPart toyWit :=
  ⟨repCode_binds H, trueCm_binds (Coin := Bool) H, linked₀ H, relS₀ H⟩

/-! ## Where the conclusions fail -/

/-- A code with no columns: an opening's message is whether its data is empty, and its bit that message. -/
def badCode : CodeScheme Bool Bool Unit where
  len _ := 0
  agree _ := 0
  encode _ m _ := m
  bit _ m _ := some m
  parse d := some (d.isEmpty, [])
  colOK _ _ c _ := .ret c

theorem badCode_fails (H : Msg → Msg) : ¬ badCode.scheme.Binds H := fun h => by
  obtain ⟨p, hp, _⟩ := h [] ⟨0, true, []⟩ ⟨0, false, [0]⟩ rfl (by decide) rfl rfl
  have h0 : ((badCode.scheme.opens [] ⟨0, true, []⟩).eval H).2 ++
      ((badCode.scheme.opens [] ⟨0, false, [0]⟩).eval H).2 = [] := rfl
  rw [h0] at hp
  exact List.not_mem_nil hp

theorem byPos_fails (H : Msg → Msg) : ¬ (byPos (fun _ _ => true) accCm accCm).Binds H := fun h => by
  obtain ⟨p, hp, _⟩ := h [] ⟨0, true, []⟩ ⟨0, false, []⟩ rfl (by decide) rfl rfl
  have h0 : (((byPos (fun _ _ => true) accCm accCm).opens [] ⟨0, true, []⟩).eval H).2 ++
      (((byPos (fun _ _ => true) accCm accCm).opens [] ⟨0, false, []⟩).eval H).2 = [] := rfl
  rw [h0] at hp
  exact List.not_mem_nil hp

theorem linkColl_fails (H : Msg → Msg) :
    ¬ LogColl (((accCm.opens [] ⟨0, true, []⟩).eval H).2 ++ ((accCm.opens [] ⟨0, false, []⟩).eval H).2 ++
      linkLog H accCm accCm [] []) := by
  rintro ⟨p, hp, _⟩
  exact List.not_mem_nil hp

theorem relink_fails (H : Msg → Msg) :
    ¬ (RelS H rejCm 0 toyStmt toyPart (relink (fun _ _ => false) toyStmt.roots toyWit []) ∨
      LogColl ((toyWit.vals.flatMap fun o =>
          (((byPos (fun _ _ => false) rejCm rejCm).opens toyStmt.roots o).eval H).2) ++
        linkLog H rejCm rejCm toyStmt.roots [])) := by
  rintro (h | ⟨p, hp, _⟩)
  · have h' : (0 : ℝ) < toyStmt.design.Pa toyPart (faulty H rejCm toyStmt toyPart _) := h
    rw [rejCm_faulty, toy_Pa, ite_eq_right (by decide)] at h'
    exact lt_irrefl _ h'
  · exact List.not_mem_nil hp

end Claims.Witness.Encodings
