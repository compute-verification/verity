import FlockSoundness.Discharge.PrivateCircuit.UnitEval

/-!
# Evaluated rows: equations, instances, stacked copies

* `rowsEval_input`, `rowsEval_one`, `rowsEval_comp`: `rowsEval`'s equations;
* `localEval_rows`: a unit that is an instance of `R` computes `rowsEval R` on the values at its input and constant
  gates; `mem_wrong_iff_rows`: so it is wrong exactly when a committed computed gate misses it;
* `rowsEval_stackCol`: copy `u`'s columns of `k` stacked copies hold the template's values on the copy's inputs;
  `mem_wrong_iff_stack1`: so a unit of one copy is wrong exactly where a returned output misses the template's value;
* `ne_claim_iff`: claimed outputs read off returned bits miss `r` exactly where a bit misses `r`'s (`0` past `r`).
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Audit

section Eval

variable (R : FlockSoundness.Rows) (z : Fin R.nIn → Bool) (one : Bool)

theorem rowsEval_input (c : Fin R.w) (h : c.val < R.nIn) : rowsEval R z one c = z ⟨c.val, h⟩ := by
  rw [rowsEval]
  simp only [h, ↓reduceDIte]

theorem rowsEval_one : rowsEval R z one R.one = one := by
  rw [rowsEval]
  simp [FlockSoundness.Rows.one]

theorem rowsEval_comp (i : Fin R.nComp) :
    rowsEval R z one (R.comp i) = (xorSum (R.a i) (rowsEval R z one) && xorSum (R.b i) (rowsEval R z one)) := by
  have h1 : ¬ (R.comp i).val < R.nIn := by simp [FlockSoundness.Rows.comp]
  have h2 : (R.comp i).val < R.nIn + R.nComp := by simp [FlockSoundness.Rows.comp]
  rw [rowsEval]
  simp only [h1, h2, ↓reduceDIte]
  have hi : (⟨(R.comp i).val - R.nIn, by omega⟩ : Fin R.nComp) = i :=
    Fin.ext (by simp [FlockSoundness.Rows.comp])
  rw [hi]
  have key : ∀ d ∈ R.a i ++ R.b i,
      (if _hd : d.val < (R.comp i).val then rowsEval R z one d else one) = rowsEval R z one d := by
    intro d hd
    rcases R.topo i d hd with h | h
    · have h' : d.val < (R.comp i).val := by simpa [FlockSoundness.Rows.comp] using h
      simp only [h', ↓reduceDIte]
    · have e : d = R.one := Fin.ext h
      subst e
      have h' : ¬ R.one.val < (R.comp i).val := by simp [FlockSoundness.Rows.comp, FlockSoundness.Rows.one]
      simp only [h', ↓reduceDIte, rowsEval_one]
  congr 1
  · exact xorSum_congr fun d hd => key d (List.mem_append_left _ hd)
  · exact xorSum_congr fun d hd => key d (List.mem_append_right _ hd)

end Eval

section Instance

variable {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : FlockSoundness.Rows}
  (I : P.IsRowsUnit u R) (X : Fin C.N → Bool)

/-- **An instance computes its rows**: each of the unit's computed gates, evaluated in order with every other gate read
from `X`, is `rowsEval R` on the values `X` holds at the unit's input gates and its constant's gate. -/
theorem localEval_rows (i : Fin R.nComp) :
    P.localEval X u (I.wire (R.comp i)) =
      rowsEval R (fun j => X (I.wire ⟨j.val, by have := j.isLt; show _ < _ + _ + 1; omega⟩)) (X (I.wire R.one))
        (R.comp i) := by
  suffices key : ∀ k (i : Fin R.nComp), i.val = k → P.localEval X u (I.wire (R.comp i)) =
      rowsEval R (fun j => X (I.wire ⟨j.val, by have := j.isLt; show _ < _ + _ + 1; omega⟩)) (X (I.wire R.one))
        (R.comp i) from key _ i rfl
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
  intro i hik
  have hcol : ∀ c ∈ R.a i ++ R.b i, P.localEval X u (I.wire c) =
      rowsEval R (fun j => X (I.wire ⟨j.val, by have := j.isLt; show _ < _ + _ + 1; omega⟩)) (X (I.wire R.one)) c := by
    intro c hc
    by_cases hcomp : ∃ i', c = R.comp i'
    · obtain ⟨i', rfl⟩ := hcomp
      have hlt : i'.val < i.val := by
        rcases R.topo i _ hc with h | h
        · simp only [FlockSoundness.Rows.comp] at h; omega
        · simp only [FlockSoundness.Rows.comp] at h; have := i'.isLt; omega
      exact ih i'.val (hik ▸ hlt) i' rfl
    · push Not at hcomp
      rw [P.localEval_of_not_mem X (I.outside c hcomp)]
      by_cases hin : c.val < R.nIn
      · rw [rowsEval_input R _ _ c hin]
      · have e : c = R.one := by
          apply Fin.ext
          rcases R.topo i c hc with h | h
          · exact absurd (Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega) :
              c = R.comp ⟨c.val - R.nIn, by omega⟩) (hcomp _)
          · exact h
        subst e
        rw [rowsEval_one]
  rw [P.localEval_of_mem X (I.mem i), I.op i, rowsEval_comp]
  simp only [Op.apply, xorSum_map]
  congr 1
  · exact xorSum_congr fun c hc => hcol c (List.mem_append_left _ hc)
  · exact xorSum_congr fun c hc => hcol c (List.mem_append_right _ hc)

/-- **An instance is wrong exactly where a committed computed gate misses its rows' values** on the values at its
input gates and its constant's gate. -/
theorem mem_wrong_iff_rows :
    u ∈ P.wrong X ↔ ∃ i : Fin R.nComp, I.wire (R.comp i) ∈ P.committed ∧
      X (I.wire (R.comp i)) ≠
        rowsEval R (fun j => X (I.wire ⟨j.val, by have := j.isLt; show _ < _ + _ + 1; omega⟩)) (X (I.wire R.one))
          (R.comp i) := by
  rw [P.mem_wrong]
  unfold Audit.Partition.Correct
  constructor
  · intro h
    push Not at h
    obtain ⟨g, hg, hgu, hne⟩ := h
    obtain ⟨i, rfl⟩ := I.onto g hgu
    exact ⟨i, hg, by rwa [← localEval_rows]⟩
  · rintro ⟨i, hg, hne⟩ h
    exact hne (by rw [← localEval_rows]; exact h _ hg (I.mem i))

end Instance

section Stack

variable (R : FlockSoundness.Rows) (k : ℕ) (zs : Fin (R.stack k).nIn → Bool) (one : Bool)

theorem stackCol_congr {u₁ u₂ : ℕ} (h₁ : u₁ < k) (h₂ : u₂ < k) (e : u₁ = u₂) (c : Fin R.w) :
    R.stackCol k h₁ c = R.stackCol k h₂ c := by
  subst e; rfl

theorem rowsEval_input' (R : FlockSoundness.Rows) (z : Fin R.nIn → Bool) (one : Bool) (c : Fin R.w) (j : Fin R.nIn)
    (h : c.val = j.val) : rowsEval R z one c = z j := by
  rw [rowsEval_input R z one c (h ▸ j.isLt)]
  exact congrArg z (Fin.ext h)

theorem rowsEval_comp' (R : FlockSoundness.Rows) (z : Fin R.nIn → Bool) (one : Bool) (c : Fin R.w) (i : Fin R.nComp)
    (h : c.val = R.nIn + i.val) :
    rowsEval R z one c = (xorSum (R.a i) (rowsEval R z one) && xorSum (R.b i) (rowsEval R z one)) := by
  rw [show c = R.comp i from Fin.ext h, rowsEval_comp]

/-- **A copy's constant row holds the constant.** -/
theorem rowsEval_stack_one {u : ℕ} (hu : u < k) :
    rowsEval (R.stack k) zs one (R.stackCol k hu R.one) = one := by
  have hlt : u * (R.nComp + 1) + 0 < (R.stack k).nComp :=
    FlockSoundness.Rows.mul_add_lt hu (show 0 < R.nComp + 1 by omega)
  have hm : (u * (R.nComp + 1) + 0) % (R.nComp + 1) = 0 := by simp
  have ha : (R.stack k).a ⟨_, hlt⟩ = [(R.stack k).one] := by
    simp only [FlockSoundness.Rows.stack, FlockSoundness.Rows.stackRow, hm, ↓reduceDIte]; rfl
  have hb : (R.stack k).b ⟨_, hlt⟩ = [(R.stack k).one] := by
    simp only [FlockSoundness.Rows.stack, FlockSoundness.Rows.stackRow, hm, ↓reduceDIte]; rfl
  refine (rowsEval_comp' (R.stack k) zs one (R.stackCol k hu R.one) ⟨_, hlt⟩ (by
    simp only [FlockSoundness.Rows.stackCol, FlockSoundness.Rows.one, show ¬ R.nIn + R.nComp < R.nIn by omega,
      lt_irrefl, ↓reduceDIte]; rfl)).trans ?_
  rw [ha, hb]
  simp [xorSum, rowsEval_one]

/-- **Copy `u` of `k` stacked copies holds the template's values on the copy's inputs**, at each of its columns. -/
theorem rowsEval_stackCol {u : ℕ} (hu : u < k) (c : Fin R.w) :
    rowsEval (R.stack k) zs one (R.stackCol k hu c) =
      rowsEval R (fun j => zs ⟨u * R.nIn + j.val, FlockSoundness.Rows.mul_add_lt hu j.isLt⟩) one c := by
  suffices key : ∀ m (c : Fin R.w), c.val = m → rowsEval (R.stack k) zs one (R.stackCol k hu c) =
      rowsEval R (fun j => zs ⟨u * R.nIn + j.val, FlockSoundness.Rows.mul_add_lt hu j.isLt⟩) one c from key _ c rfl
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro c hcm
  have hcw := c.isLt
  simp only [FlockSoundness.Rows.w] at hcw
  by_cases h1 : c.val < R.nIn
  · refine (rowsEval_input' (R.stack k) zs one (R.stackCol k hu c)
      ⟨u * R.nIn + c.val, FlockSoundness.Rows.mul_add_lt hu h1⟩ (by simp [FlockSoundness.Rows.stackCol, h1])).trans ?_
    rw [rowsEval_input R _ _ c h1]
  by_cases h2 : c.val < R.nIn + R.nComp
  · set i0 : Fin R.nComp := ⟨c.val - R.nIn, by omega⟩ with hi0
    have hlt : u * (R.nComp + 1) + (1 + (c.val - R.nIn)) < (R.stack k).nComp :=
      FlockSoundness.Rows.mul_add_lt hu (show 1 + (c.val - R.nIn) < R.nComp + 1 by omega)
    obtain ⟨hd, hm⟩ := FlockSoundness.div_mod_copy (u := u) (show 1 + (c.val - R.nIn) < R.nComp + 1 by omega)
    have hm0 : ¬ (u * (R.nComp + 1) + (1 + (c.val - R.nIn))) % (R.nComp + 1) = 0 := by rw [hm]; omega
    have hi : (⟨(u * (R.nComp + 1) + (1 + (c.val - R.nIn))) % (R.nComp + 1) - 1, by omega⟩ : Fin R.nComp) = i0 :=
      Fin.ext (by simp only [hm, hi0]; omega)
    have hrow : ∀ f : Fin R.nComp → List (Fin R.w), R.stackRow k f ⟨_, hlt⟩ = (f i0).map (R.stackCol k hu) := by
      intro f
      simp only [FlockSoundness.Rows.stackRow, hm0, ↓reduceDIte]
      rw [hi]
      exact List.map_congr_left fun d _ => stackCol_congr R k _ hu hd d
    have hcol : ∀ d ∈ R.a i0 ++ R.b i0, rowsEval (R.stack k) zs one (R.stackCol k hu d) =
        rowsEval R (fun j => zs ⟨u * R.nIn + j.val, FlockSoundness.Rows.mul_add_lt hu j.isLt⟩) one d := by
      intro d hd
      rcases R.topo i0 d hd with h | h
      · exact ih d.val (by simp only [hi0] at h; omega) d rfl
      · have e1 : d = R.one := Fin.ext h
        subst e1
        rw [rowsEval_stack_one, rowsEval_one]
    refine (rowsEval_comp' (R.stack k) zs one (R.stackCol k hu c) ⟨_, hlt⟩
      (by simp [FlockSoundness.Rows.stackCol, FlockSoundness.Rows.stack, h1, h2])).trans ?_
    rw [rowsEval_comp' R _ _ c i0 (by simp only [hi0]; omega)]
    show (xorSum (R.stackRow k R.a ⟨_, hlt⟩) _ && xorSum (R.stackRow k R.b ⟨_, hlt⟩) _) = _
    rw [hrow, hrow]
    exact congrArg₂ (· && ·)
      ((xorSum_map (R.a i0) (R.stackCol k hu) (rowsEval (R.stack k) zs one)).trans
        (xorSum_congr fun d hd => hcol d (List.mem_append_left _ hd)))
      ((xorSum_map (R.b i0) (R.stackCol k hu) (rowsEval (R.stack k) zs one)).trans
        (xorSum_congr fun d hd => hcol d (List.mem_append_right _ hd)))
  · have e : c = R.one := Fin.ext (by simp only [FlockSoundness.Rows.one]; omega)
    subst e
    rw [rowsEval_stack_one, rowsEval_one]

section One

theorem stackCol_comp_val (R : FlockSoundness.Rows) {k u : ℕ} (hu : u < k) (c : Fin R.w) (h1 : R.nIn ≤ c.val)
    (h2 : c.val < R.nIn + R.nComp) :
    (R.stackCol k hu c).val = k * R.nIn + (u * (R.nComp + 1) + (1 + (c.val - R.nIn))) := by
  unfold FlockSoundness.Rows.stackCol
  simp only [show ¬ c.val < R.nIn by omega, h2, ↓reduceDIte]

/-- **A unit of one copy of `R0` is wrong exactly where a returned output misses the template's value**, when its
committed computed gates are the copy's columns `col k` (stacked: `pos k`) of the returned outputs `k < nB`, its input
gates hold `z` and its constant's gate holds 1. -/
theorem mem_wrong_iff_stack1 {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    {R R0 : FlockSoundness.Rows} (I : P.IsRowsUnit u R) (hR : R = R0.stack 1) (X : Fin C.N → Bool)
    (z : Fin R0.nIn → Bool) (hz : ∀ (j : Fin R0.nIn) (x : Fin R.w), x.val = j.val → X (I.wire x) = z j)
    (hone : ∀ x : Fin R.w, x.val = R.nIn + R.nComp → X (I.wire x) = true)
    (g : ℕ → Fin C.N) (hg : ∀ x (h : x < R.w), g x = I.wire ⟨x, h⟩)
    (nB : ℕ) (col pos : ℕ → ℕ) (hcol : ∀ k < nB, R0.nIn ≤ col k ∧ col k < R0.nIn + R0.nComp)
    (hpos : ∀ k < nB, pos k = 1 * R0.nIn + (0 * (R0.nComp + 1) + (1 + (col k - R0.nIn))))
    (hcom : ∀ i : Fin R.nComp, I.wire (R.comp i) ∈ P.committed ↔ ∃ k < nB, R.nIn + i.val = pos k) :
    u ∈ P.wrong X ↔ ∃ k < nB, X (g (pos k)) ≠ rowsEvalN R0 z true (col k) := by
  subst hR
  have h0 : 0 < 1 := Nat.one_pos
  have hval : ∀ (x : Fin (R0.stack 1).w) (c : Fin R0.w), x.val = (R0.stackCol 1 h0 c).val →
      rowsEval (R0.stack 1) (fun j => X (I.wire ⟨j.val, by have := j.isLt; show _ < _ + _ + 1; omega⟩))
        (X (I.wire (R0.stack 1).one)) x = rowsEval R0 z true c := by
    intro x c hx
    have e : x = R0.stackCol 1 h0 c := Fin.ext hx
    refine (congrArg (rowsEval (R0.stack 1) _ _) e).trans ((rowsEval_stackCol R0 1 _ _ h0 c).trans ?_)
    rw [hone _ rfl]
    congr 1
    funext j
    exact hz j _ (by simp)
  rw [mem_wrong_iff_rows I]
  constructor
  · rintro ⟨i, hi, hne⟩
    obtain ⟨k, hk, hik⟩ := (hcom i).1 hi
    obtain ⟨h1, h2⟩ := hcol k hk
    refine ⟨k, hk, ?_⟩
    have hpk := hpos k hk
    have hw : pos k < (R0.stack 1).w := by
      have := ((R0.stack 1).comp i).isLt
      simp only [FlockSoundness.Rows.comp] at this
      omega
    rw [hg _ hw, show (⟨pos k, hw⟩ : Fin (R0.stack 1).w) = (R0.stack 1).comp i from
      Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)]
    unfold rowsEvalN
    simp only [show col k < R0.w by simp only [FlockSoundness.Rows.w]; omega, ↓reduceDIte]
    rwa [← hval ((R0.stack 1).comp i) ⟨col k, by simp only [FlockSoundness.Rows.w]; omega⟩ (by
      rw [stackCol_comp_val R0 h0 _ h1 h2]
      simp only [FlockSoundness.Rows.comp]
      exact hik.trans hpk)]
  · rintro ⟨k, hk, hne⟩
    obtain ⟨h1, h2⟩ := hcol k hk
    have hpk := hpos k hk
    have hlt : pos k - (R0.stack 1).nIn < (R0.stack 1).nComp := by
      simp only [FlockSoundness.Rows.stack]; omega
    set i : Fin (R0.stack 1).nComp := ⟨_, hlt⟩
    have hci : ((R0.stack 1).comp i).val = pos k := by
      simp only [FlockSoundness.Rows.comp, i, FlockSoundness.Rows.stack]; omega
    refine ⟨i, (hcom i).2 ⟨k, hk, by rw [← hci]; rfl⟩, ?_⟩
    have hw : pos k < (R0.stack 1).w := hci ▸ ((R0.stack 1).comp i).isLt
    rw [hg _ hw, show (⟨pos k, hw⟩ : Fin (R0.stack 1).w) = (R0.stack 1).comp i from Fin.ext hci.symm] at hne
    unfold rowsEvalN at hne
    simp only [show col k < R0.w by simp only [FlockSoundness.Rows.w]; omega, ↓reduceDIte] at hne
    rwa [hval ((R0.stack 1).comp i) ⟨col k, by simp only [FlockSoundness.Rows.w]; omega⟩ (by
      rw [hci, stackCol_comp_val R0 h0 _ h1 h2]
      exact hpk)]

end One

end Stack

section Claim

/-- **Claimed outputs read from returned bits**: `r` is the claim `ys` makes of `m` outputs (its first `m` bits when the
rest are `0`, all of it otherwise) exactly when each bit of `ys` is `r`'s, and `0` past `m`. -/
theorem eq_claim_iff (r ys : List Bool) (m : ℕ) (hr : r.length = m) (hm : m ≤ ys.length) :
    r = (if (ys.drop m).all (fun b => !b) then ys.take m else ys) ↔
      ∀ k, ∀ hk : k < ys.length, ys[k] = if h : k < m then r[k]'(hr ▸ h) else false := by
  split_ifs with hall
  · rw [List.all_eq_true] at hall
    constructor
    · rintro rfl k hk
      split_ifs with h
      · simp
      · have := hall ys[k] (by
          rw [List.mem_iff_getElem]
          exact ⟨k - m, by simp; omega, by simp; congr 1; omega⟩)
        simpa using this
    · intro H
      apply List.ext_getElem (by simp; omega)
      intro k h1 h2
      have := H k (by omega)
      simp only [show k < m by omega, ↓reduceDIte] at this
      simp [this]
  · constructor
    · intro h
      exfalso
      apply hall
      have hl : ys.length = m := h ▸ hr
      rw [List.drop_eq_nil_of_le (by omega)]
      rfl
    · intro H
      exfalso
      apply hall
      rw [List.all_eq_true]
      intro x hx
      obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hx
      have := H (m + k) (by simp at hk; omega)
      simp only [show ¬ m + k < m by omega, ↓reduceDIte] at this
      simp [List.getElem_drop, this]

/-- `eq_claim_iff`, negated. -/
theorem ne_claim_iff (r ys : List Bool) (m : ℕ) (hr : r.length = m) (hm : m ≤ ys.length) :
    r ≠ (if (ys.drop m).all (fun b => !b) then ys.take m else ys) ↔
      ∃ k, ∃ hk : k < ys.length, ys[k] ≠ if h : k < m then r[k]'(hr ▸ h) else false := by
  rw [Ne, eq_claim_iff r ys m hr hm]
  push Not
  rfl

end Claim

end FlockSoundness.Discharge.PrivateCircuit
