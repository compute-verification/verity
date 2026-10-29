import Flock.Tmpl

/-!
# What `Tmpl.wellFormed` gives the template fold's proof

`parse` refuses a template that is not `Tmpl.wellFormed`, a total check on the description. Here it gives the hypotheses
of `fold_eq_foldRef_layout` that read the template: each call's binding is as long as its callee's inputs
(`wf_bound`), its callee is earlier and its copies inside the caller (`wf_call`), columns and calls below `2^64`
(`wf_small`), the root a Definition with no inputs and a constant (`wf_root`), and every row at one of its Definition's
columns (`wf_fit`).
-/

open Flock Flock.Tmpl

namespace FlockSoundness.TmplFold

theorem flatMap_range_length (n m : Nat) (f : Nat → List Nat) (hf : ∀ r, (f r).length = m) :
    ((List.range n).flatMap f).length = n * m := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ, List.flatMap_append, ih, hf, Nat.succ_mul]

theorem bound_length (c : Call) (g : Nat → Nat) (j : Nat) :
    (c.bound g j).length = c.segs.foldl (fun n s => n + s.reps * s.len) 0 := by
  unfold Call.bound
  rw [← Array.foldl_toList]
  suffices ∀ (l : List Seg) (a : Nat), a + (l.flatMap fun s => (List.range s.reps).flatMap fun r =>
      (List.range s.len).map fun t => g (s.at_ + r * s.rstride + j * s.step + t)).length =
      l.foldl (fun n s => n + s.reps * s.len) a by simpa using this c.segs.toList 0
  intro l
  induction l with
  | nil => simp
  | cons s l ih =>
    intro a
    rw [List.flatMap_cons, List.length_append, List.foldl_cons, ← ih,
      flatMap_range_length s.reps s.len _ fun r => by simp]
    omega

theorem getD_mem {α : Type} [Inhabited α] {a : Array α} {i : Nat} (h : i < a.size) :
    a.getD i default ∈ a.toList := by
  simp only [Array.getD, h, ↓reduceDIte]
  exact Array.getElem_mem_toList h

theorem getD_out {α : Type} [Inhabited α] {a : Array α} {i : Nat} (h : ¬ i < a.size) :
    a.getD i default = default := by
  simp only [Array.getD, h, ↓reduceDIte]

theorem wf_tmpl {t : Tmpl} (h : t.wellFormed = true) :
    t.root < t.defs.size ∧ (t.defs.getD t.root default).ins = 0 ∧ 0 < (t.defs.getD t.root default).cols ∧
      ∀ d < t.defs.size, (t.defs.getD d default).wellFormed t.defs d = true := by
  unfold Tmpl.wellFormed at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true, List.mem_range] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  exact ⟨h1, h2, h3, h4⟩

theorem wf_def {t : Tmpl} (h : t.wellFormed = true) {d : Nat} (hd : d < t.defs.size) :
    (t.defs.getD d default).cols < 2 ^ 64 ∧ (t.defs.getD d default).calls.size < 2 ^ 64 ∧
      (∀ r ∈ (t.defs.getD d default).rows.toList, r.1 < (t.defs.getD d default).cols) ∧
      ∀ c ∈ (t.defs.getD d default).calls.toList, c.def_ < d ∧
        c.base + c.count * (t.defs.getD c.def_ default).cols ≤ (t.defs.getD d default).cols ∧
        c.segs.foldl (fun n s => n + s.reps * s.len) 0 = (t.defs.getD c.def_ default).ins := by
  have h1 := (wf_tmpl h).2.2.2 d hd
  unfold Def.wellFormed at h1
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true] at h1
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h1
  exact ⟨h1, h2, h3, fun c hc => ⟨(h4 c hc).1.1, (h4 c hc).1.2, (h4 c hc).2⟩⟩

theorem wf_bound {t : Tmpl} (h : t.wellFormed = true) (d i j : Nat) (hi : i < (t.defs.getD d default).calls.size) :
    (((t.defs.getD d default).calls.getD i default).bound id j).length =
      (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).ins := by
  by_cases hd : d < t.defs.size
  · rw [bound_length]
    exact ((wf_def h hd).2.2.2 _ (getD_mem hi)).2.2
  · rw [getD_out hd] at hi
    exact absurd hi (Nat.not_lt_zero _)

theorem wf_call {t : Tmpl} (h : t.wellFormed = true) : ∀ d < t.defs.size, ∀ i < (t.defs.getD d default).calls.size,
    ((t.defs.getD d default).calls.getD i default).def_ < d ∧
      ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
        (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols ≤ (t.defs.getD d default).cols :=
  fun _ hd _ hi => have hc := (wf_def h hd).2.2.2 _ (getD_mem hi); ⟨hc.1, hc.2.1⟩

theorem wf_small {t : Tmpl} (h : t.wellFormed = true) :
    ∀ d < t.defs.size, (t.defs.getD d default).cols < 2 ^ 64 ∧ (t.defs.getD d default).calls.size < 2 ^ 64 :=
  fun _ hd => ⟨(wf_def h hd).1, (wf_def h hd).2.1⟩

theorem wf_fit {t : Tmpl} (h : t.wellFormed = true) :
    ∀ d, ∀ r ∈ (t.defs.getD d default).rows.toList, r.1 < (t.defs.getD d default).cols := by
  intro d r hr
  by_cases hd : d < t.defs.size
  · exact (wf_def h hd).2.2.1 r hr
  · rw [getD_out hd] at hr
    exact absurd hr (by simp [show (default : Def).rows = #[] from rfl])

end FlockSoundness.TmplFold
