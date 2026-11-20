module

public import Proofs.Properties.Flock.Circuit.LowerAsm
public import Proofs.Properties.Flock.Circuit.Chain
public import Proofs.Properties.Flock.DecodeBoolean
public import Proofs.Properties.Poub.Remote.Sort

/-!
# `LowerEval`, proved

The root `lower` writes holds a copy of the lowered Definition that satisfies its equations (`lower_root`); lowered
from an empty state, that template Definition computes the lifted Definition at its return columns (`lowerDef_spec`),
which the output port bits read.
-/

@[expose] public section

namespace Flock.Chain

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue Flock.Chain.LowerDef
open Flock.Program hiding Form

/-- **`LowerEval`**: the template `lower` writes computes, at its output port bits, `Prog.evalBool` of the lifted
Definition on its input port bits. -/
theorem lowerEval : LowerEval := by
  intro P d inP outP t n Q e hq hl h hu hB z hs hone
  obtain ⟨hwf, -, -, -, -⟩ := Flock.Lower.lower_checked hl
  obtain ⟨Q', e0, defn, e', rets, s, base, hq', hdf, hrun, hpre, hH, hout⟩ := lower_root hl h hu hB hs hone
  have hqq := hq.symm.trans hq'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hqq
  obtain ⟨hq1, hq2⟩ := hqq
  subst Q' e0
  have he : e < Q.defs.size := by
    rcases Nat.lt_or_ge e Q.defs.size with h' | h'
    · exact h'
    · rw [Array.getElem?_eq_none h'] at hdf; cases hdf
  have hdfe : Q.defs.getD e default = defn := by simp [Array.getD_eq_getD_getElem?, hdf]
  have hdfe' : Q.defs[e]! = defn := by rw [getElem!_eq_getD, hdfe]
  have hWF : ∀ e < t.defs.size, (t.defs.getD e default).wellFormed t.defs e = true := by
    unfold Tmpl.wellFormed at hwf
    simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at hwf
    exact hwf.2
  obtain ⟨-, hc⟩ := lowerDef_spec hWF _ e {} s (e', rets) he hrun
  obtain ⟨hS, he', hsz, -⟩ := hc ⟨fun f r hr => by simp at hr, fun f r hr => by simp at hr⟩ hpre
  have hlen : ((List.range defn.paramLeaves).map z).length = (t.defs.getD e' default).ins := by
    rw [hS.2.1, hdfe']; simp
  obtain ⟨l, hl1, hl2, hl3⟩ := hS.2.2 (2 * Q.defs.size + 2) (by omega) t.defs.size _ _ hlen hH
  have hv : Q.evalBool e ((List.range (Q.defs.getD e default).paramLeaves).map fun i => z (Flock.Lower.inBit t i)) =
      l := by
    unfold Prog.evalBool
    rw [hdfe]
    show (evalDef? Q (2 * Q.defs.size + 2) e ((List.range defn.paramLeaves).map z)).getD [] = l
    rw [hl1]
    rfl
  show _ ∧ _
  rw [hv, hdfe]
  refine ⟨by rw [hl2, hsz, hdfe'], fun k hk => ?_⟩
  have hk' : k < rets.size := by rw [hsz, hdfe']; exact hk
  have hb := hS.1 k hk'
  rw [cols_eq hpre he'] at hb
  rw [hout k hk hb]
  exact hl3 k hk'

/-- `doorCol.go` at rows of whole u16 words lays bit `i` at `base + i`. -/
theorem doorCol_go_whole : ∀ (rs : List (String × ℕ)), (∀ r ∈ rs, r.2 % 16 = 0) → ∀ i base,
    doorCol.go rs i base = base + i
  | [], _, i, base => rfl
  | (nm, b) :: r, h, i, base => by
    have hb : b % 16 = 0 := h (nm, b) (List.mem_cons_self ..)
    have ih := doorCol_go_whole r (fun x hx => h x (List.mem_cons_of_mem _ hx))
    unfold doorCol.go
    split
    · rfl
    · rw [ih]; omega

/-- At rows of whole u16 words the door's file lays each bit at its own index. -/
theorem doorCol_whole {rs : Array (String × ℕ)} (h : ∀ r ∈ rs.toList, r.2 % 16 = 0) (i : ℕ) : doorCol rs i = i := by
  unfold doorCol; rw [doorCol_go_whole _ h]; omega

/-- **`AdmitWhole`**: every input row and return field `admitted` gives is whole u16 words (its clause `whole-words`). -/
def AdmitWhole : Prop :=
  ∀ {partition : Lean.Json} {cuts : List (String × List (String × ℕ))} {S : Program.Prog} {a : Spot.OneStageZk.Admit},
    Spot.OneStageZk.admitted partition cuts S = .ok a →
    (∀ r ∈ a.ins.toList, r.2 % 16 = 0) ∧ (∀ r ∈ a.fields.toList, r.2 % 16 = 0)

/-- **`AdmitLeaves`**: the input rows and return fields `admitted` gives hold exactly the lifted template's parameter
and return leaves (its clauses `cut-rows` and `leaves-are-bits`). -/
def AdmitLeaves : Prop :=
  ∀ {partition : Lean.Json} {cuts : List (String × List (String × ℕ))} {S : Program.Prog} {a : Spot.OneStageZk.Admit}
    {Q : Program.Prog} {e : ℕ},
    Spot.OneStageZk.admitted partition cuts S = .ok a → Flock.Lift.lift S a.d = .ok (Q, e) →
    bitsOf a.ins = (Q.defs.getD e default).paramLeaves ∧ bitsOf a.fields = (Q.defs.getD e default).outLeaves

/-- **`LowerEvalDoor`** from `LowerEval` at `blockTmpl`'s lowering, `AdmitWhole` and `AdmitLeaves`. -/
theorem lowerEvalDoor_of (hW : AdmitWhole) (hL : AdmitLeaves) : LowerEvalDoor := by
  intro bytes P units partition cuts S a t n Q e _ _ _ ha ht hq hu hU hb z hs hone
  obtain ⟨h1, h2⟩ := lowerEval hq ht hu hU hb hs hone
  obtain ⟨hw1, hw2⟩ := hW ha
  obtain ⟨hb1, hb2⟩ := hL ha hq
  simp only [doorCol_whole hw1, doorCol_whole hw2, hb1, hb2]
  exact ⟨h1, h2⟩


theorem notAdmitted_ne {α : Type} {c d : String} {b : α} : ¬ (Spot.OneStageZk.notAdmitted c d : V α) = .ok b := by
  intro h; cases h

theorem notAdmitted_bind_ne {α β : Type} {c d : String} {f : α → V β} {b : β} :
    ¬ ((Spot.OneStageZk.notAdmitted c d : V α) >>= f) = .ok b := by
  intro h; cases h

set_option hygiene false in
local macro "na_close" : tactic => `(tactic| any_goals (first
  | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done)))


theorem none_of_forall {α : Type} {o : Option α} (h : ∀ a, o = some a → False) : o = none := by
  cases o with
  | none => rfl
  | some a => exact absurd rfl (h a)

theorem none_of_forall2 {α β : Type} {o : Option (α × β)} (h : ∀ a b, o = some (a, b) → False) : o = none := by
  cases o with
  | none => rfl
  | some p => exact absurd rfl (h p.1 p.2)

theorem forIn_unit_all {α : Type} {l : List α} {f : α → PUnit.{1} → V (ForInStep PUnit.{1})} {Q : α → Prop}
    (h : forIn l PUnit.unit f = .ok PUnit.unit)
    (hf : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit ∧ Q a) : ∀ a ∈ l, Q a := by
  induction l with
  | nil => intro a ha; cases ha
  | cons b bs ih =>
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := Flock.HmNets.bind_ok h
    obtain ⟨rfl, hb⟩ := hf b s hs
    intro a ha
    rcases List.mem_cons.1 ha with rfl | ha
    · exact hb
    · exact ih h a ha

/-- `leavesOff`'s totals: the parameters' bits are the lift's parameter leaves, the return's its output leaves. -/
theorem leavesOff_none {fn l : Defn} (h : Spot.OneStageZk.leavesOff fn l = none) :
    fn.params.foldl (fun a x => a + x.2.widths.foldl (· + ·) 0) 0 = l.paramLeaves ∧
      fn.ret.widths.foldl (· + ·) 0 = l.outLeaves := by
  unfold Spot.OneStageZk.leavesOff at h
  dsimp only at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · rename_i hne
        simpa [Bool.or_eq_true, bne_iff_ne] using hne

theorem ite_ok {α ε : Type} {c : Prop} [Decidable c] {x y : Except ε α} {a : α}
    (h : (if c then x else y) = .ok a) : (c ∧ x = .ok a) ∨ (¬ c ∧ y = .ok a) := by
  by_cases hc : c
  · exact .inl ⟨hc, by rwa [if_pos hc] at h⟩
  · exact .inr ⟨hc, by rwa [if_neg hc] at h⟩

/-- What `admitted` checked of the template it admits: its Definition `fn` and lift `l`, `leaves-are-bits`, the rows
and fields it returns, and `whole-words`. -/
theorem admitted_walk {partition : Lean.Json} {cuts : List (String × List (String × ℕ))} {S : Program.Prog}
    {a : Spot.OneStageZk.Admit} (h : Spot.OneStageZk.admitted partition cuts S = .ok a) :
    ∃ (fn : Defn) (L : Program.Prog) (dl : ℕ) (l : Defn),
      S.defs[a.d]? = some fn ∧ Flock.Lift.lift S a.d = .ok (L, dl) ∧ L.defs[dl]? = some l ∧
      Spot.OneStageZk.leavesOff fn l = none ∧
      a.ins = Spot.OneStageZk.cutRows ((cuts.lookup a.key).getD [])
        (fn.params.map fun x => (x.1, x.2.widths.foldl (· + ·) 0)) ∧
      a.fields = (fieldsOf fn.ret).toArray.mapIdx (fun j f => (s!"o{j}", f.widths.foldl (· + ·) 0)) ∧
      (∀ r ∈ a.ins, (r.2 == 0 || r.2 % 16 != 0) = false) ∧
      (∀ r ∈ a.fields, (r.2 == 0 || r.2 % 16 != 0) = false) ∧
      (∀ x ∈ (cuts.lookup a.key).getD [], ∃ n t, fn.params.find? (·.1 == x.1) = some (n, t) ∧
        ¬ (decide (x.2 < 2) || (Widths.homo t).isNone || t.leaves % x.2 != 0) = true) ∧
      (∀ n ∈ a.ins.map (·.1) ++ a.fields.map (·.1),
        ¬ decide (((a.ins.map (·.1) ++ a.fields.map (·.1)).filter (· == n)).size > 1) = true) ∧
      S.index[a.key]? = some a.d := by
  unfold Spot.OneStageZk.admitted at h
  split at h
  na_close
  split at h
  na_close
  obtain ⟨called, -, h⟩ := Flock.HmNets.bind_ok h
  try dsimp only at h
  generalize Array.qsort called (fun x1 x2 => decide (x1 < x2)) = keys at h
  rcases keys with ⟨_ | ⟨key, _ | ⟨k2, rest⟩⟩⟩
  · dsimp only at h
    exact absurd h notAdmitted_ne
  swap
  · dsimp only at h
    exact absurd h notAdmitted_ne
  try dsimp only at h
  rcases ite_ok h with ⟨-, h⟩ | ⟨-, h⟩
  · exact absurd h notAdmitted_bind_ne
  split at h
  na_close
  split at h
  na_close
  try dsimp only at h
  split at h
  na_close
  try dsimp only at h
  split at h
  na_close
  try dsimp only at h
  split at h
  na_close
  rename_i r _
  obtain ⟨L, dl⟩ := r
  rw [pure_bind] at h
  try dsimp only at h
  split at h
  na_close
  try dsimp only at h
  obtain ⟨⟨⟩, hc, h⟩ := Flock.HmNets.bind_ok h
  try dsimp only at h
  split at h
  na_close
  try dsimp only at h
  obtain ⟨⟨⟩, hw, h⟩ := Flock.HmNets.bind_ok h
  split at h
  na_close
  split at h
  na_close
  cases h
  simp only [List.forIn_cons, List.forIn_nil] at hw
  obtain ⟨s1, h, hw⟩ := Flock.HmNets.bind_ok hw
  split at h
  na_close
  cases h
  obtain ⟨s2, h, -⟩ := Flock.HmNets.bind_ok hw
  split at h
  na_close
  refine ⟨_, _, _, _, by assumption, by assumption, by assumption, none_of_forall (by assumption), rfl, rfl, ?_, ?_,
    forIn_unit_all hc fun x s hs => ?_, ?_, by assumption⟩
  · intro r hr
    have := Array.find?_eq_none.1 (none_of_forall2 (by assumption)) r hr
    simpa using this
  · intro r hr
    have := Array.find?_eq_none.1 (none_of_forall2 (by assumption)) r hr
    simpa using this
  · split at hs
    · rcases ite_ok hs with ⟨-, hs⟩ | ⟨hc', hs⟩
      · exact absurd hs notAdmitted_bind_ne
      · cases hs
        exact ⟨rfl, _, _, by assumption, hc'⟩
    · exact absurd hs notAdmitted_bind_ne
  · intro n hn
    have := Array.find?_eq_none.1 (none_of_forall (by assumption)) n hn
    simpa using this

/-- **`AdmitWhole`**, from `admitted_walk`'s `whole-words`. -/
theorem admitWhole : AdmitWhole := by
  intro partition cuts S a ha
  obtain ⟨_, _, _, _, -, -, -, -, -, -, hi, hf, -, -, -⟩ := admitted_walk ha
  refine ⟨fun r hr => ?_, fun r hr => ?_⟩
  · have := hi r (Array.mem_def.2 hr)
    simp at this
    omega
  · have := hf r (Array.mem_def.2 hr)
    simp at this
    omega

/-- **`HomoBits`**: a type of one leaf width (`Widths.homo`) has that width at each of its leaves, so its bits are its
leaves times the width. -/
def HomoBits : Prop :=
  ∀ (t : Ty) (w : ℕ), Widths.homo t = some w → t.widths.foldl (· + ·) 0 = t.leaves * w

theorem hb_foldl_append_size {α : Type} (l : List α) (g : α → Array ℕ) (a : Array ℕ) :
    (l.foldl (fun a t => a ++ g t) a).size = a.size + (l.map fun t => (g t).size).sum := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => simp only [List.foldl_cons, ih, Array.size_append, List.map_cons, List.sum_cons]; omega

theorem hb_foldl_add {α : Type} (l : List α) (g : α → ℕ) (a : ℕ) :
    l.foldl (fun a t => a + g t) a = a + (l.map g).sum := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => simp only [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

theorem hb_mem_foldl_append {α : Type} (l : List α) (g : α → Array ℕ) (a : Array ℕ) (x : ℕ)
    (h : x ∈ l.foldl (fun a t => a ++ g t) a) : x ∈ a ∨ ∃ t ∈ l, x ∈ g t := by
  induction l generalizing a with
  | nil => exact .inl h
  | cons y l ih =>
    rcases ih _ h with h | ⟨t, ht, hx⟩
    · rcases Array.mem_append.1 h with h | h
      · exact .inl h
      · exact .inr ⟨y, List.mem_cons_self .., h⟩
    · exact .inr ⟨t, List.mem_cons_of_mem _ ht, hx⟩

theorem hb_sizeOf_toList {α : Type} [SizeOf α] (fs : Array α) : sizeOf fs = 1 + sizeOf fs.toList := by
  cases fs; simp

theorem hb_lt_mem {α : Type} [SizeOf α] {fs : Array α} {f : α} (h : f ∈ fs.toList) : sizeOf f < 1 + sizeOf fs := by
  have := List.sizeOf_lt_of_mem h; have := hb_sizeOf_toList fs; omega

theorem hb_lt_mem2 {fs : Array (String × Ty)} {f : String × Ty} (h : f ∈ fs.toList) : sizeOf f.2 < 1 + sizeOf fs := by
  have := hb_lt_mem h; obtain ⟨k, t⟩ := f; simp only [Prod.mk.sizeOf_spec] at *; omega

theorem hb_widths_size : ∀ t : Ty, t.widths.size = t.leaves
  | .value _ => by simp [Ty.widths, Ty.leaves]
  | .array n e => by
    have ih := hb_widths_size e
    rw [Ty.widths, Ty.leaves]
    simp only [← Array.foldl_toList, hb_foldl_append_size, ih, Array.size_empty, Array.toList_range, List.map_const',
      List.sum_replicate, smul_eq_mul, List.length_range]
    omega
  | .tuple fs => by
    rw [Ty.widths, Ty.leaves, ← Array.foldl_toList, ← Array.foldl_toList, hb_foldl_append_size, hb_foldl_add]
    simp only [Array.size_empty, Nat.zero_add]
    congr 1
    exact List.map_congr_left fun f hf => hb_widths_size f
  | .record fs => by
    rw [Ty.widths, Ty.leaves, ← Array.foldl_toList, ← Array.foldl_toList, hb_foldl_append_size, hb_foldl_add]
    simp only [Array.size_empty, Nat.zero_add]
    congr 1
    exact List.map_congr_left fun f hf => hb_widths_size f.2
termination_by t => t
decreasing_by
  all_goals simp_wf
  all_goals first
    | omega
    | (have := hb_lt_mem (by assumption); omega)
    | (have := hb_lt_mem2 (by assumption); omega)

theorem hb_tuple_fold (T : ℕ) (l : List Ty) (acc : List ℕ) (o : ℕ) (hT : o + (l.map Ty.leaves).sum ≤ T) :
    (∀ x ∈ acc, x ∈ (l.foldl (fun (acc : List ℕ × ℕ) f =>
      (if acc.2 + f.leaves ≤ 0 || T ≤ acc.2 then acc.1
        else acc.1 ++ Widths.rangeWidths f (0 - acc.2) (min T (acc.2 + f.leaves) - acc.2), acc.2 + f.leaves))
      (acc, o)).1) ∧
    ∀ f ∈ l, 0 < f.leaves → ∀ x ∈ Widths.rangeWidths f 0 f.leaves, x ∈ (l.foldl (fun (acc : List ℕ × ℕ) f =>
      (if acc.2 + f.leaves ≤ 0 || T ≤ acc.2 then acc.1
        else acc.1 ++ Widths.rangeWidths f (0 - acc.2) (min T (acc.2 + f.leaves) - acc.2), acc.2 + f.leaves))
      (acc, o)).1 := by
  induction l generalizing acc o with
  | nil => exact ⟨fun x hx => hx, fun f hf => by simp at hf⟩
  | cons g l ih =>
    simp only [List.map_cons, List.sum_cons] at hT
    have ih' := ih (if o + g.leaves ≤ 0 || T ≤ o then acc
      else acc ++ Widths.rangeWidths g (0 - o) (min T (o + g.leaves) - o)) (o + g.leaves) (by omega)
    simp only [List.foldl_cons]
    have hsub : ∀ x ∈ acc, x ∈ (if o + g.leaves ≤ 0 || T ≤ o then acc
        else acc ++ Widths.rangeWidths g (0 - o) (min T (o + g.leaves) - o)) := by
      intro x hx; split
      · exact hx
      · exact List.mem_append_left _ hx
    refine ⟨fun x hx => ih'.1 x (hsub x hx), fun f hf hpos x hx => ?_⟩
    rcases List.mem_cons.1 hf with rfl | hf
    · apply ih'.1
      have hc : ¬ ((decide (o + f.leaves ≤ 0) || decide (T ≤ o)) = true) := by
        simp only [Bool.or_eq_true, decide_eq_true_eq, not_or]; omega
      rw [if_neg hc]
      apply List.mem_append_right
      have h1 : 0 - o = 0 := Nat.zero_sub o
      have h2 : min T (o + f.leaves) - o = f.leaves := by omega
      rw [h1, h2]; exact hx
    · exact ih'.2 f hf hpos x hx

theorem hb_record_fold (T : ℕ) (l : List (String × Ty)) (acc : List ℕ) (o : ℕ)
    (hT : o + (l.map fun f => f.2.leaves).sum ≤ T) :
    (∀ x ∈ acc, x ∈ (l.foldl (fun (acc : List ℕ × ℕ) f =>
      (if acc.2 + f.2.leaves ≤ 0 || T ≤ acc.2 then acc.1
        else acc.1 ++ Widths.rangeWidths f.2 (0 - acc.2) (min T (acc.2 + f.2.leaves) - acc.2), acc.2 + f.2.leaves))
      (acc, o)).1) ∧
    ∀ f ∈ l, 0 < f.2.leaves → ∀ x ∈ Widths.rangeWidths f.2 0 f.2.leaves, x ∈ (l.foldl (fun (acc : List ℕ × ℕ) f =>
      (if acc.2 + f.2.leaves ≤ 0 || T ≤ acc.2 then acc.1
        else acc.1 ++ Widths.rangeWidths f.2 (0 - acc.2) (min T (acc.2 + f.2.leaves) - acc.2), acc.2 + f.2.leaves))
      (acc, o)).1 := by
  induction l generalizing acc o with
  | nil => exact ⟨fun x hx => hx, fun f hf => by simp at hf⟩
  | cons g l ih =>
    simp only [List.map_cons, List.sum_cons] at hT
    have ih' := ih (if o + g.2.leaves ≤ 0 || T ≤ o then acc
      else acc ++ Widths.rangeWidths g.2 (0 - o) (min T (o + g.2.leaves) - o)) (o + g.2.leaves) (by omega)
    simp only [List.foldl_cons]
    have hsub : ∀ x ∈ acc, x ∈ (if o + g.2.leaves ≤ 0 || T ≤ o then acc
        else acc ++ Widths.rangeWidths g.2 (0 - o) (min T (o + g.2.leaves) - o)) := by
      intro x hx; split
      · exact hx
      · exact List.mem_append_left _ hx
    refine ⟨fun x hx => ih'.1 x (hsub x hx), fun f hf hpos x hx => ?_⟩
    rcases List.mem_cons.1 hf with rfl | hf
    · apply ih'.1
      have hc : ¬ ((decide (o + f.2.leaves ≤ 0) || decide (T ≤ o)) = true) := by
        simp only [Bool.or_eq_true, decide_eq_true_eq, not_or]; omega
      rw [if_neg hc]
      apply List.mem_append_right
      have h1 : 0 - o = 0 := Nat.zero_sub o
      have h2 : min T (o + f.2.leaves) - o = f.2.leaves := by omega
      rw [h1, h2]; exact hx
    · exact ih'.2 f hf hpos x hx

theorem hb_mem_range : ∀ (t : Ty) (x : ℕ), x ∈ t.widths → x ∈ Widths.rangeWidths t 0 t.leaves
  | .value w, x, hx => by
    simp only [Ty.widths, Array.mem_singleton] at hx
    subst hx; simp [Widths.rangeWidths, Ty.leaves]
  | .array n e, x, hx => by
    rw [Ty.widths] at hx
    simp only [← Array.foldl_toList] at hx
    rcases hb_mem_foldl_append _ _ _ _ hx with h | ⟨_, hi, hx⟩
    · simp at h
    have hn : 0 < n := by simp at hi; omega
    have hL : 0 < e.leaves := by
      rw [← hb_widths_size e]; exact Array.size_pos_of_mem hx
    have ih := hb_mem_range e x hx
    rw [Widths.rangeWidths, Ty.leaves]
    simp only [Nat.min_self]
    have h0 : ¬ (e.leaves == 0 || n * e.leaves ≤ 0) = true := by
      simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or]
      exact ⟨by omega, by have := Nat.mul_pos hn hL; omega⟩
    rw [if_neg h0]
    simp only [Nat.zero_div, Nat.zero_mul, Nat.sub_zero]
    split
    · rename_i hab
      have hb : (n * e.leaves - 1) / e.leaves = 0 := by simpa using (beq_iff_eq.1 hab).symm
      have h1 : n = 1 := by
        rw [Nat.div_eq_zero_iff] at hb
        rcases hb with hb | hb
        · omega
        · have : n * e.leaves < 2 * e.leaves := by omega
          have := Nat.lt_of_mul_lt_mul_right this
          omega
      subst h1; simpa using ih
    · exact List.mem_eraseDups.2 (List.mem_append_left _ (List.mem_append_left _ ih))
  | .tuple fs, x, hx => by
    rw [Ty.widths] at hx
    simp only [← Array.foldl_toList] at hx
    rcases hb_mem_foldl_append _ _ _ _ hx with h | ⟨f, hf, hx⟩
    · simp at h
    have hpos : 0 < f.leaves := by rw [← hb_widths_size f]; exact Array.size_pos_of_mem hx
    have ih := hb_mem_range f x hx
    rw [Widths.rangeWidths]
    apply List.mem_eraseDups.2
    rw [← Array.foldl_toList]
    have hT : 0 + (fs.toList.map Ty.leaves).sum ≤ (Ty.tuple fs).leaves := by
      rw [Ty.leaves, ← Array.foldl_toList, hb_foldl_add] <;> omega
    exact (hb_tuple_fold _ fs.toList [] 0 hT).2 f hf hpos x ih
  | .record fs, x, hx => by
    rw [Ty.widths] at hx
    simp only [← Array.foldl_toList] at hx
    rcases hb_mem_foldl_append _ _ _ _ hx with h | ⟨f, hf, hx⟩
    · simp at h
    have hpos : 0 < f.2.leaves := by rw [← hb_widths_size f.2]; exact Array.size_pos_of_mem hx
    have ih := hb_mem_range f.2 x hx
    rw [Widths.rangeWidths]
    apply List.mem_eraseDups.2
    rw [← Array.foldl_toList]
    have hT : 0 + (fs.toList.map fun f => f.2.leaves).sum ≤ (Ty.record fs).leaves := by
      rw [Ty.leaves, ← Array.foldl_toList, hb_foldl_add] <;> omega
    exact (hb_record_fold _ fs.toList [] 0 hT).2 f hf hpos x ih
termination_by t => t
decreasing_by
  all_goals simp_wf
  all_goals first
    | omega
    | (have := hb_lt_mem (by assumption); omega)
    | (have := hb_lt_mem2 (by assumption); omega)

theorem hb_foldl_const (l : List ℕ) (w c : ℕ) (h : ∀ x ∈ l, x = w) : l.foldl (· + ·) c = c + l.length * w := by
  induction l generalizing c with
  | nil => simp
  | cons y l ih =>
    simp only [List.foldl_cons, List.length_cons]
    rw [ih (c + y) fun x hx => h x (List.mem_cons_of_mem _ hx), h y (List.mem_cons_self ..)]
    rw [Nat.succ_mul]; omega

/-- `HomoBits` holds. -/
theorem homoBits : HomoBits := by
  intro t w h
  unfold Widths.homo at h
  split at h
  · rename_i w' hw
    have hw' : w' = w := by simpa using h
    have hall : ∀ x ∈ t.widths.toList, x = w' := fun x hx => by
      have := hb_mem_range t x (Array.mem_toList_iff.1 hx)
      rw [hw] at this; simpa using this
    rw [← Array.foldl_toList, hb_foldl_const _ w' 0 hall, Array.length_toList, hb_widths_size, hw']; omega
  · cases h

theorem foldl_snd_eq (l : List (String × ℕ)) (a : ℕ) :
    l.foldl (fun x y => x + y.2) a = a + (l.map Prod.snd).sum := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => simp only [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

theorem bitsOf_eq (rs : Array (String × ℕ)) : bitsOf rs = (rs.toList.map Prod.snd).sum := by
  unfold bitsOf
  rw [← Array.foldl_toList, foldl_snd_eq]
  simp

theorem cutRows_toList (cs : List (String × ℕ)) (ins : Array (String × ℕ)) :
    (Spot.OneStageZk.cutRows cs ins).toList = ins.toList.flatMap fun x => (Spot.OneStageZk.cutRows cs #[x]).toList := by
  unfold Spot.OneStageZk.cutRows
  simp [Array.toList_flatMap]

theorem cutRows_one (cs : List (String × ℕ)) (n : String) (b : ℕ) :
    (Spot.OneStageZk.cutRows cs #[(n, b)]).toList = match cs.lookup n with
      | some k => if k ≤ 1 then [(n, b)] else (List.range k).map fun j => (s!"{n}#{j}", b / k)
      | none => [(n, b)] := by
  unfold Spot.OneStageZk.cutRows
  cases hk : cs.lookup n with
  | none => simp [hk]
  | some k => by_cases k1 : k ≤ 1 <;> simp [hk, k1]

theorem bitsOf_cutRows (cs : List (String × ℕ)) (ins : Array (String × ℕ))
    (h : ∀ x ∈ ins.toList, ∀ k, cs.lookup x.1 = some k → 2 ≤ k → k ∣ x.2) :
    bitsOf (Spot.OneStageZk.cutRows cs ins) = bitsOf ins := by
  rw [bitsOf_eq, bitsOf_eq, cutRows_toList]
  generalize ins.toList = l at h ⊢
  induction l with
  | nil => simp
  | cons x l ih =>
    rw [List.flatMap_cons, List.map_append, List.sum_append, List.map_cons, List.sum_cons,
      ih fun y hy => h y (List.mem_cons_of_mem _ hy)]
    obtain ⟨n, b⟩ := x
    have hd := h (n, b) List.mem_cons_self
    rw [cutRows_one]
    congr 1
    cases hk : cs.lookup n with
    | none => simp
    | some k =>
      by_cases k1 : k ≤ 1
      · simp [k1]
      · simp only [k1, if_false]
        simp [Function.comp_def, List.sum_replicate, Nat.mul_div_cancel' (hd k hk (by omega))]

theorem sum_foldl_add (arr : Array ℕ) : arr.foldl (· + ·) 0 = arr.toList.sum := by
  rw [← Array.foldl_toList]
  suffices h : ∀ (l : List ℕ) (a : ℕ), l.foldl (· + ·) a = a + l.sum by simpa using h arr.toList 0
  intro l
  induction l with
  | nil => simp
  | cons x l ih => intro a; simp only [List.foldl_cons, ih, List.sum_cons]; omega

theorem foldl_append_toList {β : Type} (fs : List β) (g : β → Array ℕ) (acc : Array ℕ) :
    (fs.foldl (fun a t => a ++ g t) acc).toList = acc.toList ++ fs.flatMap fun t => (g t).toList := by
  induction fs generalizing acc with
  | nil => simp
  | cons x l ih => simp [ih]

theorem sum_map_flatMap {β : Type} (l : List β) (g : β → Array ℕ) :
    (l.map fun x => (g x).toList.sum).sum = (l.flatMap fun x => (g x).toList).sum := by
  induction l with
  | nil => simp
  | cons x l ih => simp_all [List.flatMap_cons, List.sum_append]

/-- The return fields' bits are the return's. -/
theorem fields_bits (ret : Ty) :
    ((fieldsOf ret).map fun f => f.widths.foldl (· + ·) 0).sum = ret.widths.foldl (· + ·) 0 := by
  cases ret with
  | value w => simp [fieldsOf]
  | array n e => simp [fieldsOf]
  | tuple fs =>
    simp only [fieldsOf, sum_foldl_add, Ty.widths]
    rw [← Array.foldl_toList, foldl_append_toList]
    simpa using sum_map_flatMap fs.toList Ty.widths
  | record fs =>
    simp only [fieldsOf, sum_foldl_add, Ty.widths]
    rw [← Array.foldl_toList, foldl_append_toList]
    simpa [Function.comp_def] using sum_map_flatMap fs.toList fun x => x.2.widths

theorem bitsOf_fields (ret : Ty) :
    bitsOf ((fieldsOf ret).toArray.mapIdx fun j f => (s!"o{j}", f.widths.foldl (· + ·) 0)) =
      ret.widths.foldl (· + ·) 0 := by
  rw [bitsOf_eq, ← fields_bits]
  congr 1
  apply List.ext_getElem <;> simp

theorem lookup_mem {cs : List (String × ℕ)} {n : String} {k : ℕ} (h : cs.lookup n = some k) : (n, k) ∈ cs := by
  induction cs with
  | nil => simp at h
  | cons x l ih =>
    obtain ⟨m, j⟩ := x
    by_cases hm : n = m
    · subst hm; simp [List.lookup] at h; subst h; exact List.mem_cons_self
    · have : (n == m) = false := by simpa using hm
      simp only [List.lookup, this] at h
      exact List.mem_cons_of_mem _ (ih h)

theorem find?_of_nodup {ps : List (String × Ty)} {p : String × Ty} (hn : (ps.map Prod.fst).Nodup) (hp : p ∈ ps) :
    ps.find? (·.1 == p.1) = some p := by
  induction ps with
  | nil => cases hp
  | cons q qs ih =>
    rw [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.1 hp with rfl | hp
    · simp
    · have hq : (q.1 == p.1) = false := by
        have : q.1 ≠ p.1 := fun e => hn.1 (e ▸ List.mem_map_of_mem hp)
        simpa using this
      rw [List.find?_cons, hq]
      exact ih hn.2 hp

theorem count_flatMap_ge {β : Type} (l : List β) (g : β → List String) (a : String) (P : β → Bool)
    (h : ∀ x ∈ l, P x = true → a ∈ g x) : (l.filter P).length ≤ (l.flatMap g).count a := by
  induction l with
  | nil => simp
  | cons x l ih =>
    rw [List.flatMap_cons, List.count_append]
    have ih' := ih fun y hy => h y (List.mem_cons_of_mem _ hy)
    by_cases hx : P x = true
    · rw [List.filter_cons_of_pos hx, List.length_cons]
      have := List.count_pos_iff.2 (h x List.mem_cons_self hx)
      omega
    · rw [List.filter_cons_of_neg hx]
      omega

theorem count_fst_le (ps : List (String × Ty)) (n : String) :
    (ps.map Prod.fst).count n ≤ (ps.filter fun x => x.1 == n).length := by
  induction ps with
  | nil => simp
  | cons x l ih =>
    by_cases hx : x.1 = n
    · simp [hx]; omega
    · simp [hx]; omega

/-- **`AdmitLeaves`**, from `admitted_walk` and `homoBits`. -/
theorem admitLeaves : AdmitLeaves := by
  intro partition cuts S a Q e ha hq
  obtain ⟨fn, L, dl, l, -, hlift, hl, hoff, hins, hfields, -, -, hcut, hnames, -⟩ := admitted_walk ha
  rw [hlift] at hq
  cases hq
  have hg : Q.defs.getD e default = l := by
    rw [Array.getD_eq_getD_getElem?, hl]; rfl
  rw [hg]
  obtain ⟨hp, hr⟩ := leavesOff_none hoff
  have hnd : (fn.params.toList.map Prod.fst).Nodup := by
    rw [List.nodup_iff_count_le_one]
    intro n
    by_contra hc
    have h2 := count_fst_le fn.params.toList n
    let cs := (cuts.lookup a.key).getD []
    let r : String := match cs.lookup n with
      | some k => if k ≤ 1 then n else s!"{n}#{(0 : ℕ)}"
      | none => n
    have hrow : ∀ x ∈ fn.params.toList.map (fun p => (p.1, p.2.widths.foldl (· + ·) 0)), (x.1 == n) = true →
        r ∈ ((Spot.OneStageZk.cutRows cs #[x]).toList.map Prod.fst) := by
      intro x _ hx
      have hx : x.1 = n := by simpa using hx
      obtain ⟨m, b⟩ := x
      simp only at hx
      subst hx
      rw [cutRows_one]
      simp only [r]
      cases hk : cs.lookup m with
      | none => simp
      | some k =>
        by_cases k1 : k ≤ 1
        · simp [k1]
        · simp only [k1, if_false]
          simp
          omega
    have hge := count_flatMap_ge (fn.params.toList.map fun p => (p.1, p.2.widths.foldl (· + ·) 0))
      (fun x => (Spot.OneStageZk.cutRows cs #[x]).toList.map Prod.fst) r (fun x => x.1 == n) hrow
    have hlen : ((fn.params.toList.map fun p => (p.1, p.2.widths.foldl (· + ·) 0)).filter fun x => x.1 == n).length =
        (fn.params.toList.filter fun x => x.1 == n).length := by
      rw [List.filter_map, List.length_map]; rfl
    have hin : 2 ≤ (a.ins.toList.map fun x => x.1).count r := by
      rw [hins, cutRows_toList, List.map_flatMap, Array.toList_map]
      exact le_trans (by omega) hge
    rw [List.count_eq_length_filter] at hin
    have hall : 2 ≤ ((a.ins.map (·.1) ++ a.fields.map (·.1)).filter (· == r)).size := by
      rw [← Array.length_toList, Array.toList_filter, Array.toList_append, List.filter_append, List.length_append,
        Array.toList_map, Array.toList_map]
      omega
    have hmem : r ∈ a.ins.map (·.1) ++ a.fields.map (·.1) := by
      rw [Array.mem_def, Array.toList_append, Array.toList_map]
      exact List.mem_append_left _ (List.count_pos_iff.1 (by rw [List.count_eq_length_filter]; omega))
    exact hnames r hmem (decide_eq_true (Nat.lt_of_lt_of_le (by decide) hall))
  constructor
  · rw [hins, bitsOf_cutRows, ← hp]
    · rw [bitsOf, Array.foldl_map]
    · intro x hx k hk hk2
      rw [Array.toList_map, List.mem_map] at hx
      obtain ⟨p, hp', rfl⟩ := hx
      obtain ⟨n', t', hf, hc⟩ := hcut _ (lookup_mem hk)
      rw [← Array.find?_toList, find?_of_nodup hnd hp'] at hf
      have ht : p.2 = t' := congrArg Prod.snd (Option.some.inj hf)
      rw [← ht] at hc
      obtain ⟨w, hw⟩ : ∃ w, Widths.homo p.2 = some w := by
        cases h : Widths.homo p.2 with
        | none => simp [h] at hc
        | some w => exact ⟨w, rfl⟩
      simp [hw] at hc
      have hk' : p.2.leaves % k = 0 := by omega
      simp only
      rw [homoBits _ _ hw]
      exact Dvd.dvd.mul_right (Nat.dvd_of_mod_eq_zero hk') w
  · rw [hfields, bitsOf_fields, hr]

/-- **`LowerEvalDoor`**. -/
theorem lowerEvalDoor : LowerEvalDoor :=
  lowerEvalDoor_of admitWhole admitLeaves

/-! ## `restrict` keeps C's Boolean Definitions -/

/-- What `restrict` keeps of C in the block's Program `S`: C's Definitions, as a prefix of `S`'s; its Boolean map; each
entry of its index; and a primitive `S` adds has a key C's index lacks. -/
def RShape (P S : Prog) : Prop :=
  P.defs.size ≤ S.defs.size ∧ (∀ i < P.defs.size, S.defs[i]! = P.defs[i]!) ∧ S.boolean = P.boolean ∧
    (∀ (k : String) (i : ℕ), P.index[k]? = some i → S.index[k]? = some i) ∧
    ∀ i < S.defs.size, P.defs.size ≤ i → (S.defs[i]!).prim = true → P.index[(S.defs[i]!).key]? = none

/-- **`RestrictShape`**: `restrict` keeps C as `RShape` says. -/
def RestrictShape : Prop :=
  ∀ (P : Prog) (units : Array ℕ) (S : Prog), Spot.Restrict.restrict P units = .ok S → RShape P S

theorem bodyOf_eq {P S : Prog} (hm : P.BoolMapOk) (hb : S.boolean = P.boolean)
    (hi : ∀ (k : String) (i : ℕ), P.index[k]? = some i → S.index[k]? = some i) (k : String) :
    S.bodyOf? k = P.bodyOf? k := by
  unfold Prog.bodyOf?
  rw [hb]
  cases hv : P.boolean[k]? with
  | none => rfl
  | some v =>
    obtain ⟨w, hw, -, hkey, hword⟩ := hm.2 k v hv
    obtain ⟨b, -, hbo, -⟩ := hm.1 w hw (by rw [hkey]; exact hword)
    rw [hkey] at hbo
    unfold Prog.bodyOf? at hbo
    rw [hv] at hbo
    have hbo' : P.index[v]? = some b := hbo
    show S.index[v]? = P.index[v]?
    rw [hbo', hi v b hbo']

theorem boolDef_lift {P S : Prog} (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!) :
    ∀ {d : ℕ}, P.BoolDef d → d < P.defs.size → S.BoolDef d := by
  intro d h
  induction h with
  | prim hp hbp =>
    intro hd
    exact .prim (by rw [hpre _ hd]; exact hp) (by rw [hpre _ hd]; exact hbp)
  | @comp d hpf hprimc hlt _ ih =>
    intro hd
    have hfn : ∀ nd ∈ (P.defs[d]!).body.nodes, nd.fn < P.defs.size := by
      intro nd hnd
      by_contra hc
      have h0 : (P.defs[nd.fn]!).prim = false := by
        rw [getElem!_neg P.defs nd.fn hc]; rfl
      have := hlt nd hnd h0
      omega
    refine .comp (by rw [hpre _ hd]; exact hpf) (fun nd hnd hp => ?_) (fun nd hnd hp => ?_) (fun nd hnd hp => ?_)
    all_goals rw [hpre _ hd] at hnd; rw [hpre _ (hfn nd hnd)] at hp
    · rw [hpre _ (hfn nd hnd)]
      exact hprimc nd hnd hp
    · exact hlt nd hnd hp
    · exact ih nd hnd hp (hfn nd hnd)

open FlockSoundness.DecodeBoolean (decode_facts word_ids gateOf_none bool_none fuel_stable) in
/-- **`RestrictBodyAgrees` for a decoded C**, on `RestrictShape`: the block's Program of a Program `decode` accepts
whose root has no recut keeps its word primitives evaluating as their Boolean Definitions. -/
theorem restrictBodyAgrees_decode (hR : RestrictShape) :
    ∀ (bytes : ByteArray) (P : Prog) (units : Array ℕ) (S : Prog), decode bytes = .ok P →
      (P.defs[P.root]!).body.recut = #[] → Spot.Restrict.restrict P units = .ok S → BodyAgrees S := by
  intro bytes P units S h hr hs p hp b hprim hb
  obtain ⟨hle, hpre, hbool, hidx, hnew⟩ := hR P units S hs
  obtain ⟨⟨hmap, hkeys⟩, hids⟩ := decode_facts bytes P h hr
  have hbo := bodyOf_eq ⟨hmap, hkeys⟩ hbool hidx
  have hbP := hb
  rw [hbo] at hbP
  by_cases hpP : p < P.defs.size
  swap
  · exfalso
    have hn := hnew p hp (by omega) hprim
    unfold Prog.bodyOf? at hbP
    cases hv : P.boolean[(S.defs[p]!).key]? with
    | none => rw [hv] at hbP; cases hbP
    | some v =>
      obtain ⟨w, -, hiw, -⟩ := hkeys _ v hv
      rw [hn] at hiw
      cases hiw
  rw [hpre p hpP] at hbP
  have hprimP : (P.defs[p]!).prim = true := by rw [← hpre p hpP]; exact hprim
  have hidxP := (InferenceOnlyVerifier.Properties.SecurityProofs.DecodeKeys.decode_keys bytes P h).2 p hpP
  have hb0 := hbP
  unfold Prog.bodyOf? at hb0
  cases hv : P.boolean[(P.defs[p]!).key]? with
  | none => rw [hv] at hb0; cases hb0
  | some v =>
    obtain ⟨w, -, hiw, -, hword⟩ := hkeys _ _ hv
    rw [hidxP] at hiw
    obtain rfl := Option.some.inj hiw
    obtain ⟨b', hb', hbo', hbody⟩ := hmap p hpP hword
    have hbb : b' = b := Option.some.inj (hbo'.symm.trans hbP)
    rw [hbb] at hb' hbody
    obtain ⟨hid, hmis, -⟩ := hids p hpP hprimP
    obtain ⟨hin, hbp⟩ := word_ids hword
    have hg : Flock.Lower.gateOf? (P.defs[p]!).id = none := by
      rw [hid]
      exact gateOf_none hbp hmis hin
    have hdS : S.defs[p]? = some (P.defs[p]!) := by
      rw [← hpre p hpP, getElem!_pos S.defs p hp]
      exact getElem?_pos S.defs p hp
    have hBS : ∀ k, isBooleanPrim k = true → S.bodyOf? k = none := fun k hk => by
      rw [hbo]
      exact bool_none hkeys k hk
    have hbS : S.bodyOf? (P.defs[p]!).key = some b := by rw [hbo]; exact hbP
    funext x
    unfold Prog.evalBool
    rw [show 2 * S.defs.size + 2 = (2 * S.defs.size + 1) + 1 from rfl,
      FlockSoundness.EvalBool.evalDef?_body S hdS hprimP hg hbS,
      fuel_stable S hBS (boolDef_lift hpre hbody.1 hb') (by omega : b < 2 * S.defs.size + 1) x]

/-! ## `RestrictShape`, proved -/

/-- `restrict`'s Definitions and index while it adds the input primitives: `RShape` but the Boolean map. -/
def RInv (P : Prog) (defs : Array Defn) (index : Std.HashMap String ℕ) : Prop :=
  P.defs.size ≤ defs.size ∧ (∀ i < P.defs.size, defs[i]! = P.defs[i]!) ∧
    (∀ (k : String) (i : ℕ), P.index[k]? = some i → index[k]? = some i) ∧
    ∀ i < defs.size, P.defs.size ≤ i → (defs[i]!).prim = true → P.index[(defs[i]!).key]? = none

theorem rinv_init (P : Prog) : RInv P P.defs P.index :=
  ⟨Nat.le_refl _, fun _ _ => rfl, fun _ _ h => h, fun _ h1 h2 => absurd h1 (by omega)⟩

theorem push_get!_lt {defs : Array Defn} {d : Defn} {i : ℕ} (h : i < defs.size) : (defs.push d)[i]! = defs[i]! := by
  rw [getElem!_pos (defs.push d) i (by simp; omega), getElem!_pos defs i h, Array.getElem_push_lt h]

theorem push_get!_eq {defs : Array Defn} {d : Defn} : (defs.push d)[defs.size]! = d := by
  rw [getElem!_pos (defs.push d) defs.size (by simp), Array.getElem_push_eq]

theorem rinv_push {P : Prog} {defs : Array Defn} {index : Std.HashMap String ℕ} {d : Defn} {n : ℕ}
    (h : RInv P defs index) (hk : index[d.key]? = none) : RInv P (defs.push d) (index.insert d.key n) := by
  obtain ⟨hle, hpre, hidx, hnew⟩ := h
  refine ⟨by simp; omega, fun i hi => by rw [push_get!_lt (by omega)]; exact hpre i hi, fun k i hki => ?_,
    fun i hi hPi hp => ?_⟩
  · rw [Std.HashMap.getElem?_insert]
    split
    · rename_i he
      have := hidx k i hki
      rw [← eq_of_beq he, hk] at this
      cases this
    · exact hidx k i hki
  · simp only [Array.size_push] at hi
    by_cases hlt : i < defs.size
    · rw [push_get!_lt hlt] at hp ⊢
      exact hnew i hlt hPi hp
    · obtain rfl : i = defs.size := by omega
      rw [push_get!_eq]
      cases hP : P.index[d.key]? with
      | none => rfl
      | some j => rw [hidx _ j hP] at hk; cases hk

theorem rshape_of {P S : Prog} {defs : Array Defn} {index : Std.HashMap String ℕ} {blk : Defn}
    (h : RInv P defs index) (hc : ¬ index.contains blk.key = true)
    (hd : S.defs = defs.push blk) (hi : S.index = index.insert blk.key defs.size) (hb : S.boolean = P.boolean) :
    RShape P S := by
  have hk : index[blk.key]? = none := by
    exact Std.HashMap.getElem?_eq_none (by simpa using hc)
  obtain ⟨hle, hpre, hidx, hnew⟩ := rinv_push (n := defs.size) h hk
  refine ⟨hd ▸ hle, hd ▸ hpre, hb, hi ▸ hidx, hd ▸ fun i hi' hPi hpi => hnew i hi' hPi hpi⟩

section RestrictTriple
open Std.Do
set_option mvcgen.warning false

theorem exc_spec {ε α : Type} (x : Except ε α) : ⦃⌜True⌝⦄ x ⦃⇓? _ => ⌜True⌝⦄ := by
  cases x <;> exact fun _ => True.intro

theorem exc_throw_spec {ε α : Type} (e : ε) (Q : PostCond α (.except ε .pure)) :
    ⦃Q.2.1 e⦄ (MonadExceptOf.throw e : Except ε α) ⦃Q⦄ := fun h => h

theorem exc_of_triple {ε α : Type} {x : Except ε α} {Q : α → Prop} {r : α}
    (hT : ⦃⌜True⌝⦄ x ⦃⇓? a => ⌜Q a⌝⦄) (h : x = .ok r) : Q r := by
  subst h; exact hT True.intro

theorem exc_wp_true {ε α : Type} (x : Except ε α) (Q : α → Assertion (.except ε .pure)) (hQ : ∀ a, (Q a).down) :
    (wp⟦x⟧ (Q, ExceptConds.true)).down := by
  cases x with
  | error e => exact True.intro
  | ok a => exact hQ a

theorem front_spec (P : Prog) (us : Array ℕ) : ⦃⌜True⌝⦄ Spot.BlockFront.front P us ⦃⇓? _ => ⌜True⌝⦄ := exc_spec _
theorem members_spec (P : Prog) (us : Array ℕ) : ⦃⌜True⌝⦄ Spot.BlockFront.members P us ⦃⇓? _ => ⌜True⌝⦄ :=
  exc_spec _
theorem spell_spec (P : Prog) r rs : ⦃⌜True⌝⦄ Spot.Restrict.spell P r rs ⦃⇓? _ => ⌜True⌝⦄ := exc_spec _
theorem resolve_spec (P : Prog) sc r : ⦃⌜True⌝⦄ Flock.Program.resolve P sc r ⦃⇓? _ => ⌜True⌝⦄ := exc_spec _

theorem restrict_triple (P : Prog) (units : Array ℕ) :
    ⦃⌜True⌝⦄ Spot.Restrict.restrict P units ⦃⇓? S => ⌜RShape P S⌝⦄ := by
  unfold Spot.Restrict.restrict
  mvcgen [front_spec, members_spec, spell_spec, resolve_spec, exc_throw_spec, -Spec.throw_Except] invariants
  · ⇓? ⟨_, defs, index, _, _⟩ => ⌜RInv P defs index⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  all_goals first
    | trivial
    | exact rinv_init P
    | (exact rinv_push (by assumption) (by assumption))
    | (exact rshape_of (by assumption) (by assumption) rfl rfl rfl)
    | exact exc_wp_true _ _ (fun r => by cases r <;> exact True.intro)

end RestrictTriple

/-- **`restrict` keeps C** (`RestrictShape`). -/
theorem restrictShape : RestrictShape := fun P units _ h => exc_of_triple (restrict_triple P units) h

/-- **`RestrictBodyAgrees` for a decoded C**: the block's Program of a Program `decode` accepts whose root has no recut
keeps its word primitives evaluating as their Boolean Definitions. -/
theorem restrictBodyAgreesDecoded :
    ∀ (bytes : ByteArray) (P : Prog) (units : Array ℕ) (S : Prog), decode bytes = .ok P →
      (P.defs[P.root]!).body.recut = #[] → Spot.Restrict.restrict P units = .ok S → BodyAgrees S :=
  restrictBodyAgrees_decode restrictShape

/-! ## `BlockEvalDoor`, on its four open premises -/

/-- Unit `g`'s gates in C: the gates of its member scope (`Prog.child` of the root at its node and member). -/
def InUnit (P : Prog) (k m h : ℕ) : Prop :=
  (P.child P.rootScope k m).off ≤ h ∧
    h < (P.child P.rootScope k m).off + (((P.defs[(P.child P.rootScope k m).d]?).map (·.gates)).getD 0)

/-- **`FrontUnitRanges`** (open): each unit of a block `front` accepts is located (`unitAt`), and its gates are in the
front's ranges. -/
def FrontUnitRanges : Prop :=
  ∀ (P : Prog) (units : Array ℕ) (fr : Spot.BlockFront.Front), Spot.BlockFront.front P units = .ok fr →
    ∀ g ∈ units, ∃ k m, Spot.BlockFront.unitAt P g = .ok (k, m) ∧
      ∀ h, InUnit P k m h → Spot.BlockFront.inside fr.ranges h = true

/-- **`UnitComputes`** (open): in a trace of C whose every computed gate of unit `g` is its operation on the trace,
`g`'s return leaves hold, bit for bit, `Prog.evalBool` on `g`'s argument bits of the lift to bits (`Flock.Lift.lift`)
of the template `admitted` gives, where that template is `g`'s callee. -/
def UnitComputes : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (L : Prog) (e : ℕ),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.OneStageZk.admittedBlock P units a.key = .ok () → Flock.Lift.lift S a.d = .ok (L, e) →
    ∀ g ∈ units, ∀ k m, Spot.BlockFront.unitAt P g = .ok (k, m) → (P.child P.rootScope k m).d = a.d →
    ∀ Y : (Flock.OfProgram.ofProgram P).Trace,
      (∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
        Y h = (Flock.OfProgram.ofProgram P).op h Y) →
        outBits Y g = Prog.evalBool L e (argBits Y g)

/-- **`AdmitCallee`**: every unit of a block whose C_S `admitted` admits, and whose units `admittedBlock` admits in C,
calls the template `admitted` gives. -/
def AdmitCallee : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit),
    decode bytes = .ok P → Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.OneStageZk.admittedBlock P units a.key = .ok () →
    ∀ g ∈ units, ∀ k m, Spot.BlockFront.unitAt P g = .ok (k, m) → (P.child P.rootScope k m).d = a.d

section AdmitTriples
open Std.Do
set_option mvcgen.warning false

theorem ac_mapM_mem {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α) (r : List β), l.mapM f = .ok r → ∀ x ∈ l, ∃ e ∈ r, f x = .ok e := by
  intro l
  induction l with
  | nil => intro _ _ x hx; simp at hx
  | cons y l ih =>
    intro r h x hx
    rw [List.mapM_cons] at h
    obtain ⟨e, he, h⟩ := Flock.HmNets.bind_ok h
    obtain ⟨es, hes, h⟩ := Flock.HmNets.bind_ok h
    change Except.ok (e :: es) = Except.ok r at h
    cases h
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨e, List.mem_cons_self .., he⟩
    · obtain ⟨e', he', hf⟩ := ih es hes x hx
      exact ⟨e', List.mem_cons_of_mem _ he', hf⟩

theorem ac_array_mapM_mem {α β ε : Type} (f : α → Except ε β) (xs : Array α) (ys : Array β)
    (h : xs.mapM f = .ok ys) : ∀ x ∈ xs, ∃ e ∈ ys, f x = .ok e := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hm : xs.toList.mapM f with
  | error e => rw [hm] at h; cases h
  | ok r =>
    rw [hm] at h
    change Except.ok r.toArray = Except.ok ys at h
    cases h
    intro x hx
    obtain ⟨e, he, hf⟩ := ac_mapM_mem f xs.toList r hm x (Array.mem_toList_iff.2 hx)
    exact ⟨e, List.mem_toArray.2 he, hf⟩

theorem ac_mapM_spec {α β : Type} (f : α → V β) (xs : Array α) :
    ⦃⌜True⌝⦄ xs.mapM f ⦃⇓? ys => ⌜∀ x ∈ xs, ∃ e ∈ ys, f x = .ok e⌝⦄ := by
  intro _
  cases h : xs.mapM f with
  | error e => exact True.intro
  | ok ys => exact ac_array_mapM_mem f xs ys h

theorem operandsW_spec (P : Prog) x m : ⦃⌜True⌝⦄ Spot.BlockFront.operandsW P x m ⦃⇓? _ => ⌜True⌝⦄ := exc_spec _

/-- An entry `(u, k, m, lo, hi)` of `admittedBlock`'s table of units passes `one-callee`: root node `k` calls `key`. -/
def CalleeOk (P : Prog) (key : String) (e : ℕ × ℕ × ℕ × ℕ × ℕ) : Prop :=
  ∃ (x : Node) (f : Defn), (P.defs[P.root]!).body.nodes[e.2.1]? = some x ∧ P.defs[x.fn]? = some f ∧ f.key = key

theorem ac_step {P : Prog} {key : String} {root : Defn} {pref : List (ℕ × ℕ × ℕ × ℕ × ℕ)}
    {cur : ℕ × ℕ × ℕ × ℕ × ℕ} {x : Node} {f : Defn} (hr : P.defs[P.root]? = some root)
    (h1 : root.body.nodes[cur.2.1]? = some x) (h2 : P.defs[x.fn]? = some f) (h3 : ¬ (f.key != key) = true)
    (hinv : ∀ e ∈ pref, CalleeOk P key e) : ∀ e, e ∈ pref ∨ e = cur → CalleeOk P key e := by
  intro e he
  rcases he with he | rfl
  · exact hinv e he
  · have hroot : P.defs[P.root]! = root := by simp [getElem!_def, hr]
    exact ⟨x, f, by rw [hroot]; exact h1, h2, by simpa using h3⟩

theorem ac_lam {P : Prog} {u : ℕ} {e : ℕ × ℕ × ℕ × ℕ × ℕ} {k m : ℕ}
    (h : (do
      let (k, m) ← Spot.BlockFront.unitAt P u
      let sc := P.child P.rootScope k m
      let some d := P.defs[sc.d]? | Spot.OneStageZk.notAdmitted "one-callee" s!"unit {u}'s callee is no Definition of C"
      pure (u, k, m, sc.off, sc.off + d.gates) : V (ℕ × ℕ × ℕ × ℕ × ℕ)) = .ok e)
    (hk : Spot.BlockFront.unitAt P u = .ok (k, m)) : e.2.1 = k := by
  rw [hk] at h
  dsimp only [bind, Except.bind] at h
  split at h
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    rw [← h]
  · exact absurd h notAdmitted_ne

theorem ac_final {P : Prog} {units : Array ℕ} {key : String} {us : Array (ℕ × ℕ × ℕ × ℕ × ℕ)}
    {lam : ℕ → V (ℕ × ℕ × ℕ × ℕ × ℕ)}
    (hlam : ∀ u e k m, lam u = .ok e → Spot.BlockFront.unitAt P u = .ok (k, m) → e.2.1 = k)
    (hmap : ∀ x ∈ units, ∃ e ∈ us, lam x = .ok e) (hinv : ∀ e ∈ us.toList, CalleeOk P key e) :
    ∀ u ∈ units, ∀ k m, Spot.BlockFront.unitAt P u = .ok (k, m) →
      ∃ (x : Node) (f : Defn), (P.defs[P.root]!).body.nodes[k]? = some x ∧ P.defs[x.fn]? = some f ∧ f.key = key := by
  intro u hu k m hk
  obtain ⟨e, he, hf⟩ := hmap u hu
  have := hlam u e k m hf hk
  obtain ⟨x, f, h1, h2, h3⟩ := hinv e (Array.mem_toList_iff.2 he)
  exact ⟨x, f, this ▸ h1, h2, h3⟩

/-- `admittedBlock`'s `one-callee`: where it accepts, every unit's root node calls `key`. -/
theorem admittedBlock_triple (P : Prog) (units : Array ℕ) (key : String) :
    ⦃⌜True⌝⦄ Spot.OneStageZk.admittedBlock P units key
    ⦃⇓? _ => ⌜∀ u ∈ units, ∀ k m, Spot.BlockFront.unitAt P u = .ok (k, m) →
      ∃ (x : Node) (f : Defn), (P.defs[P.root]!).body.nodes[k]? = some x ∧ P.defs[x.fn]? = some f ∧ f.key = key⌝⦄ := by
  unfold Spot.OneStageZk.admittedBlock
  mvcgen [ac_mapM_spec, operandsW_spec, Spot.OneStageZk.notAdmitted, exc_throw_spec, -Spec.throw_Except] invariants
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? ⟨xs, _⟩ => ⌜∀ e ∈ xs.prefix, CalleeOk P key e⌝
  · ⇓? _ => ⌜True⌝
  all_goals first
    | trivial
    | (intro e he; simp at he; done)
    | ((try simp only [List.mem_append, List.mem_singleton]);
        exact ac_step (by assumption) (by assumption) (by assumption) (by assumption) (by assumption))
    | (exact ac_final (fun u e k m h hk => ac_lam h hk) (by assumption) (by assumption))
    | exact exc_wp_true _ _ (fun r => by cases r <;> exact True.intro)

theorem admittedBlock_callee {P : Prog} {units : Array ℕ} {key : String}
    (h : Spot.OneStageZk.admittedBlock P units key = .ok ()) :
    ∀ u ∈ units, ∀ k m, Spot.BlockFront.unitAt P u = .ok (k, m) →
      ∃ (x : Node) (f : Defn), (P.defs[P.root]!).body.nodes[k]? = some x ∧ P.defs[x.fn]? = some f ∧ f.key = key :=
  exc_of_triple (admittedBlock_triple P units key) h

theorem admitted_index {partition : Lean.Json} {cuts : List (String × List (String × ℕ))} {S : Prog}
    {a : Spot.OneStageZk.Admit} (h : Spot.OneStageZk.admitted partition cuts S = .ok a) : S.index[a.key]? = some a.d := by
  obtain ⟨_, _, _, _, -, -, -, -, -, -, -, -, -, -, hi⟩ := admitted_walk h
  exact hi

end AdmitTriples

section FrontTriples
open Std.Do
set_option mvcgen.warning false

theorem exc_self_spec {ε α : Type} (x : Except ε α) : ⦃⌜True⌝⦄ x ⦃⇓? r => ⌜x = .ok r⌝⦄ := by
  cases x with
  | error e => exact fun _ => True.intro
  | ok a => exact fun _ => rfl

theorem unitAt_self_spec (P : Prog) u :
    ⦃⌜True⌝⦄ Spot.BlockFront.unitAt P u ⦃⇓? r => ⌜Spot.BlockFront.unitAt P u = .ok r⌝⦄ := exc_self_spec _
theorem portsOf_spec xs : ⦃⌜True⌝⦄ Spot.BlockFront.portsOf xs ⦃⇓? _ => ⌜True⌝⦄ := exc_spec _
theorem frontLift_spec (P : Prog) st : ⦃⌜True⌝⦄ Spot.BlockFront.lift P st ⦃⇓? _ => ⌜True⌝⦄ := exc_spec _

/-- Unit `u` is located, at root node `k`'s member `m`, and its gates' range is in `S`. -/
def MemOk (P : Prog) (S : Array (ℕ × ℕ)) (u : ℕ) : Prop :=
  ∃ k m, Spot.BlockFront.unitAt P u = .ok (k, m) ∧
    ((P.child P.rootScope k m).off, (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).gates) ∈ S

theorem memOk_push {P : Prog} {S : Array (ℕ × ℕ)} {u : ℕ} (r : ℕ × ℕ) (h : MemOk P S u) : MemOk P (S.push r) u := by
  obtain ⟨k, m, hk, hm⟩ := h
  exact ⟨k, m, hk, Array.mem_push.2 (.inl hm)⟩

theorem mem_step {P : Prog} {S : Array (ℕ × ℕ)} {pref : List ℕ} {cur k m : ℕ}
    (hk : Spot.BlockFront.unitAt P cur = .ok (k, m)) (hinv : ∀ u ∈ pref, MemOk P S u) :
    ∀ u, u ∈ pref ∨ u = cur → MemOk P (S.push ((P.child P.rootScope k m).off,
      (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).gates)) u := by
  intro u hu
  rcases hu with hu | rfl
  · exact memOk_push _ (hinv u hu)
  · exact ⟨k, m, hk, Array.mem_push.2 (.inr rfl)⟩

theorem mem_final {P : Prog} {S : Array (ℕ × ℕ)} {units : Array ℕ} (h : ∀ u ∈ units.toList, MemOk P S u) :
    ∀ u ∈ units, MemOk P S u := fun u hu => h u (Array.mem_toList_iff.2 hu)

/-- `members` locates every unit and puts its range in its table. -/
theorem members_triple (P : Prog) (units : Array ℕ) :
    ⦃⌜True⌝⦄ Spot.BlockFront.members P units ⦃⇓? r => ⌜∀ u ∈ units, MemOk P r.2 u⌝⦄ := by
  unfold Spot.BlockFront.members
  mvcgen [unitAt_self_spec, exc_throw_spec, -Spec.throw_Except] invariants
  · ⇓? ⟨xs, S, _⟩ => ⌜∀ u ∈ xs.prefix, MemOk P S u⌝
  all_goals first
    | trivial
    | (intro u hu; simp at hu; done)
    | ((try simp only [List.mem_append, List.mem_singleton]); exact mem_step (by assumption) (by assumption))
    | exact mem_final (by assumption)
    | exact exc_wp_true _ _ (fun r => by cases r <;> exact True.intro)

theorem memOk_qsort {P : Prog} {S : Array (ℕ × ℕ)} {u : ℕ} (lt : ℕ × ℕ → ℕ × ℕ → Bool) (h : MemOk P S u) :
    MemOk P (S.qsort lt) u := by
  obtain ⟨k, m, hk, hm⟩ := h
  exact ⟨k, m, hk, Array.mem_toList_iff.1
    ((Properties.Poub.qsort_toList_perm S lt).mem_iff.2 (Array.mem_toList_iff.2 hm))⟩

/-- `front`'s ranges hold every unit's, each unit located. -/
theorem front_triple (P : Prog) (units : Array ℕ) :
    ⦃⌜True⌝⦄ Spot.BlockFront.front P units ⦃⇓? fr => ⌜∀ u ∈ units, MemOk P fr.ranges u⌝⦄ := by
  unfold Spot.BlockFront.front
  mvcgen [members_triple, operandsW_spec, resolve_spec, portsOf_spec, frontLift_spec, exc_throw_spec,
    -Spec.throw_Except] invariants
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  all_goals first
    | trivial
    | exact fun u hu => memOk_qsort _ ((by assumption : ∀ u ∈ units, MemOk P _ u) u hu)
    | exact exc_wp_true _ _ (fun r => by cases r <;> exact True.intro)

end FrontTriples

/-- **`FrontUnitRanges`** holds. -/
theorem frontUnitRanges : FrontUnitRanges := by
  intro P units fr hf g hg
  obtain ⟨k, m, hk, hm⟩ := exc_of_triple (front_triple P units) hf g hg
  refine ⟨k, m, hk, fun h ⟨h1, h2⟩ => ?_⟩
  have h2' : h < (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).gates := by
    cases hd : P.defs[(P.child P.rootScope k m).d]? with
    | none => rw [hd] at h2; simp at h2; omega
    | some D =>
      rw [hd] at h2
      have : P.defs[(P.child P.rootScope k m).d]! = D := by simp [getElem!_def, hd]
      rw [this]; simpa using h2
  unfold Spot.BlockFront.inside
  rw [Array.any_eq_true']
  exact ⟨_, hm, by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨h1, h2'⟩⟩

theorem child_d (P : Prog) (k m : ℕ) :
    (P.child P.rootScope k m).d = ((P.defs[P.root]!).body.nodes[k]!).fn := rfl

/-- **`AdmitCallee`** holds. -/
theorem admitCallee : AdmitCallee := by
  intro bytes P partition cuts units S a hd hs ha hab g hg k m hu
  obtain ⟨x, f, h1, h2, h3⟩ := admittedBlock_callee hab g hg k m hu
  have hfn : x.fn < P.defs.size := (Array.getElem?_eq_some_iff.1 h2).1
  have hf : P.defs[x.fn]! = f := by simp [getElem!_def, h2]
  have hP := (InferenceOnlyVerifier.Properties.SecurityProofs.DecodeKeys.decode_keys bytes P hd).2 x.fn hfn
  rw [hf, h3] at hP
  have hS := (restrictShape P units S hs).2.2.2.1 a.key x.fn hP
  rw [admitted_index ha] at hS
  rw [child_d]
  generalize P.defs[P.root]! = R at h1 ⊢
  rw [show R.body.nodes[k]! = x by simp [getElem!_def, h1]]
  exact (Option.some.inj hS).symm

/-- Port `q`'s bits in `X`, as `portBits` lays it. -/
def portWord {P : Prog} (X : (Flock.OfProgram.ofProgram P).Trace) (q : Spot.BlockFront.Port) : List Bool :=
  portBits X #[q]

/-- Unit `g`'s arguments in `X`, one word per parameter leaf. -/
def argWords {P : Prog} (X : (Flock.OfProgram.ofProgram P).Trace) (g : ℕ) : List (List Bool) :=
  (unitArgs P g).toList.map (portWord X)

/-- Unit `g`'s return leaves in `X`, one word each. -/
def outWords {P : Prog} (X : (Flock.OfProgram.ofProgram P).Trace) (g : ℕ) : List (List Bool) :=
  (unitOuts P g).toList.map (portWord X)

theorem flatMap_eq_flatten_single {α β : Type} (f : α → List β) (l : List α) :
    l.flatMap f = (l.map fun a => [a].flatMap f).flatten := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.flatMap_cons, ih]

theorem portBits_eq {P : Prog} (X : (Flock.OfProgram.ofProgram P).Trace) (ports : Array Spot.BlockFront.Port) :
    portBits X ports = (ports.toList.map (portWord X)).flatten :=
  flatMap_eq_flatten_single _ _

/-- A Definition's parameter leaves' widths, in order. -/
def leafWidths (d : Defn) : List ℕ := d.params.toList.flatMap fun p => p.2.widths.toList

/-- The walk's wires as words: a leaf's bits, a §9 gate on its operands' first bits. -/
def wordWires : Flock.Program.Wires Option (List Bool) :=
  ⟨fun g xs => some [Flock.Program.gateVal g (xs.map (·.headD false))], fun o => o⟩

/-- Definition `d` of `P` on the words `x`, one per parameter leaf, at most `fuel` calls deep: a primitive's one
return leaf is `Prog.evalBool` of it on `x`'s bits, a composite the walk (`walkStep`) on words. -/
def wordDef? (P : Prog) : ℕ → ℕ → List (List Bool) → Option (List (List Bool))
  | 0, _, _ => none
  | fuel + 1, d, x => do
    let defn ← P.defs[d]?
    if defn.prim then some [Prog.evalBool P d x.flatten]
    else Flock.Program.walkStep wordWires P (wordDef? P fuel) d x

/-- **Definition `d`'s value on words** `x`, one per parameter leaf: its return leaves' words, `[]` where the walk
fails. -/
def wordEval (P : Prog) (d : ℕ) (x : List (List Bool)) : List (List Bool) :=
  (wordDef? P (2 * P.defs.size + 2) d x).getD []

theorem portWord_length {P : Prog} (X : (Flock.OfProgram.ofProgram P).Trace) (q : Spot.BlockFront.Port) :
    (portWord X q).length = leafWidth P q := by
  by_cases h : q.gate < (Flock.OfProgram.ofProgram P).N <;> simp [portWord, portBits, leafWidth, h]

theorem portBits_length {P : Prog} (X : (Flock.OfProgram.ofProgram P).Trace) (ports : Array Spot.BlockFront.Port) :
    (portBits X ports).length = (ports.toList.map (leafWidth P)).sum := by
  simp [portBits_eq, List.length_flatten, Function.comp_def, portWord_length]

/-- **`UnitWidths`** (open): unit `g`'s arguments, as `portBits` lays them, have its callee's parameter leaf widths, and
its return leaves its callee's return widths, for every trace alike. -/
def UnitWidths : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ (g k m : ℕ), Spot.BlockFront.unitAt P g = .ok (k, m) →
      (unitArgs P g).toList.map (leafWidth P) = leafWidths (P.defs[(P.child P.rootScope k m).d]!) ∧
        (unitOuts P g).toList.map (leafWidth P) = (P.defs[(P.child P.rootScope k m).d]!).ret.widths.toList

/-- **`UnitWordComputes`** (open): in a trace of C whose every computed gate of unit `g` is its operation on the trace,
`g`'s return leaves' words are its callee's value on its arguments' words (`wordEval`). -/
def UnitWordComputes : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    ∀ (g k m : ℕ), Spot.BlockFront.unitAt P g = .ok (k, m) → ∀ Y : (Flock.OfProgram.ofProgram P).Trace,
      (∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
        Y h = (Flock.OfProgram.ofProgram P).op h Y) →
        outWords Y g = wordEval P (P.child P.rootScope k m).d (argWords Y g)

/-- **`LiftWordEval`** (open): the lift to bits (`Flock.Lift.lift`) of the template `admitted` gives, a Definition of C
taken in the block's Program, evaluated on bits (`Prog.evalBool`), is C's value of it on words (`wordEval`), bit for
bit, on words of its parameter leaf widths. -/
def LiftWordEval : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (L : Prog) (e : ℕ),
    decode bytes = .ok P → Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    a.d < P.defs.size → Flock.Lift.lift S a.d = .ok (L, e) →
    ∀ x : List (List Bool), x.map List.length = leafWidths (P.defs[a.d]!) →
      Prog.evalBool L e x.flatten = (wordEval P a.d x).flatten

/-- **`UnitComputes`**, on `UnitWidths`, `UnitWordComputes` and `LiftWordEval`. -/
theorem unitComputes_of (hWd : UnitWidths) (hW : UnitWordComputes) (hL : LiftWordEval) : UnitComputes := by
  intro bytes P partition cuts units S a L e hd hr hs ha hab hl g hg k m hu hda Y hY
  have ho := hW bytes P hd hr g k m hu Y hY
  have hw : (argWords Y g).map List.length = leafWidths (P.defs[(P.child P.rootScope k m).d]!) := by
    rw [← (hWd bytes P hd hr g k m hu).1]
    simp [argWords, Function.comp_def, portWord_length]
  obtain ⟨x, f, h1, h2, -⟩ := admittedBlock_callee hab g hg k m hu
  have hdx : (P.child P.rootScope k m).d = x.fn := by
    rw [child_d]
    generalize P.defs[P.root]! = R at h1 ⊢
    simp [getElem!_def, h1]
  have hlt : a.d < P.defs.size := by
    rw [← hda, hdx]
    exact (Array.getElem?_eq_some_iff.1 h2).1
  rw [hda] at hw ho
  unfold outBits argBits
  rw [portBits_eq, portBits_eq]
  change (outWords Y g).flatten = Prog.evalBool L e (argWords Y g).flatten
  rw [ho, hL bytes P partition cuts units S a L e hd hs ha hlt hl _ hw]

/-- **`BlockEvalDoor`**, on `UnitComputes`. -/
theorem blockEvalDoor_of (hU : UnitComputes) : BlockEvalDoor := by
  intro bytes P partition cuts units S a fr L e hd hr hs ha hab hf hl blocks u hB X g hg Y
  obtain ⟨k, m, hu, hin⟩ := frontUnitRanges P units fr hf g hg
  have hY : ∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
      Y h = (Flock.OfProgram.ofProgram P).op h Y :=
    fun h h1 h2 => Verity.Partition.localEval_of_mem _ X ((hB h).2 ⟨hin h.val h1, h2⟩)
  exact (hU bytes P partition cuts units S a L e hd hr hs ha hab hl g hg k m hu
    (admitCallee bytes P partition cuts units S a hd hs ha hab g hg k m hu) Y hY).symm

/-- **`DoorLengths`**, on `UnitWidths`. -/
theorem doorLengths_of (hWd : UnitWidths) : DoorLengths := by
  intro bytes P partition cuts units S a fr hd hr hs ha hab hf X g hg
  obtain ⟨k, m, hu, -⟩ := frontUnitRanges P units fr hf g hg
  have hda := admitCallee bytes P partition cuts units S a hd hs ha hab g hg k m hu
  obtain ⟨hwa, hwo⟩ := hWd bytes P hd hr g k m hu
  obtain ⟨x, f, h1, h2, -⟩ := admittedBlock_callee hab g hg k m hu
  have hdx : (P.child P.rootScope k m).d = x.fn := by
    rw [child_d]
    generalize P.defs[P.root]! = R at h1 ⊢
    simp [getElem!_def, h1]
  have hlt : a.d < P.defs.size := by
    rw [← hda, hdx]
    exact (Array.getElem?_eq_some_iff.1 h2).1
  rw [hda] at hwa hwo
  obtain ⟨-, hpre, -⟩ := restrictShape P units S hs
  obtain ⟨fn, L, dl, l, hfn, hlift, hl, hoff, -⟩ := admitted_walk ha
  obtain ⟨hpi, hpo⟩ := admitLeaves ha hlift
  obtain ⟨hp, hr'⟩ := leavesOff_none hoff
  have hg' : L.defs.getD dl default = l := by
    rw [Array.getD_eq_getD_getElem?, hl]; rfl
  have hfnP : P.defs[a.d]! = fn := by
    rw [← hpre _ hlt]
    simp [getElem!_def, hfn]
  rw [hfnP] at hwa hwo
  have key : ∀ (ps : List (String × Ty)) (acc : ℕ),
      ps.foldl (fun a x => a + x.2.widths.foldl (· + ·) 0) acc = acc + (ps.flatMap fun p => p.2.widths.toList).sum := by
    intro ps
    induction ps with
    | nil => simp
    | cons x ps ih =>
      intro acc
      simp only [List.foldl_cons]
      rw [ih]
      simp only [List.flatMap_cons, List.sum_append, sum_foldl_add]
      omega
  unfold argBits outBits
  rw [portBits_length, portBits_length, hwa, hwo, hpi, hpo, hg', ← hp, ← hr', sum_foldl_add, ← Array.foldl_toList,
    key]
  simp [leafWidths]

/-! ## The lift's bit layout, which `LiftWordEval` reads -/

/-- **`BitAtSpec`**: word leaf `j` of `t` starts in `Ty.bitType t` at the sum of the widths before it, with its own
width (`Flock.Lift.bitAt`). -/
def BitAtSpec : Prop :=
  ∀ (t : Ty) (j : ℕ), j < t.widths.size → Flock.Lift.bitAt t j = ((t.widths.toList.take j).sum, t.widths[j]!)

/-- **`LiftSeqRefs`**: `liftSeq` puts in place of each word reference of a body the bits it occupies, `at_`'s run of
them, in order, wherever `at_` agrees with `homo`'s one width. -/
def LiftSeqRefs : Prop :=
  ∀ (at_ : Bool → ℕ → ℕ → ℕ × ℕ) (homo : Bool → ℕ → Option ℕ) (s : Seq),
    (∀ p i w, homo p i = some w → ∀ j, at_ p i j = (j * w, w)) →
    (Flock.Lift.liftSeq at_ homo s).toArray.toList = s.toArray.toList.flatMap fun r =>
      (List.range (at_ r.p r.i r.j).2).map fun t => ({ p := r.p, i := r.i, j := (at_ r.p r.i r.j).1 + t } : Ref)

/-- **`BitTypeWidths`**: `Ty.bitType t` has a one-bit leaf for each bit of each of `t`'s leaves, in leaf order. -/
def BitTypeWidths : Prop :=
  ∀ t : Ty, (Ty.bitType t).widths.toList = t.widths.toList.flatMap fun w => List.replicate w 1

/-- **`BitTypeAxis`**: a batch argument's axis of `t` is the same axis of `Ty.bitType t`, of the same length, the leaves
below it the bits of the word leaves below it. -/
def BitTypeAxis : Prop :=
  ∀ (t : Ty) (ax n L : ℕ), t.axisDims ax = some (n, L) →
    ∃ e : Ty, e.leaves = L ∧ (Ty.bitType t).axisDims ax = some (n, (Ty.bitType e).leaves) ∧
      (Ty.bitType e).leaves = (e.widths.foldl (· + ·) 0)

/-- **`ScanCarryBits`**: a scan node's lift keeps its sequences (`nxs`), and each sequence argument's length; its carries
are its carries' bits. -/
def ScanCarryBits : Prop :=
  ∀ (d : Defn) (fnOf : ℕ → ℕ) (primOf : ℕ → Bool) (gatesOf : ℕ → ℕ) (k : ℕ) (x : Node),
    d.body.nodes[k]? = some x → x.form = .scan →
    ∃ y, (Flock.Lift.liftDef d fnOf primOf gatesOf).body.nodes[k]? = some y ∧ y.nxs = x.nxs ∧ y.members = x.members ∧
      y.args.size = x.args.size ∧ ∀ i < x.args.size, (y.args[i]!).ty = Ty.bitType (x.args[i]!).ty

/-- **`RestrictWordEval`**: C's Definitions keep their value on words in the block's Program `restrict P units`, for a
C `decode` accepts whose root has no recut. -/
def RestrictWordEval : Prop :=
  ∀ (bytes : ByteArray) (P S : Prog) (units : Array ℕ), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → ∀ d < P.defs.size, ∀ x, wordEval S d x = wordEval P d x

theorem widths_homo_all {t : Ty} {w : ℕ} (h : Widths.homo t = some w) : ∀ x ∈ t.widths.toList, x = w := by
  intro x hx
  have hr := hb_mem_range t x (Array.mem_def.2 hx)
  unfold Widths.homo at h
  split at h
  · rename_i hw
    cases h
    rw [hw] at hr
    simpa using hr
  · cases h

theorem prefix_sums (l : List ℕ) : ∀ (acc : Array ℕ), 0 < acc.size →
    (l.foldl (fun a w => a.push (a.back! + w)) acc).toList =
      acc.toList ++ (List.range l.length).map fun i => acc.back! + (l.take (i + 1)).sum := by
  induction l with
  | nil => intro acc _; simp
  | cons y l ih =>
    intro acc _
    rw [List.foldl_cons, ih _ (by simp)]
    have hb : (acc.push (acc.back! + y)).back! = acc.back! + y := by simp [Array.back!]
    rw [hb, List.length_cons, List.range_succ_eq_map]
    simp [List.take_succ_cons, Nat.add_assoc]

theorem bitAtSpec : BitAtSpec := by
  intro t j hj
  unfold Flock.Lift.bitAt
  cases hw : Widths.homo t with
  | some w =>
    simp only
    have hall := widths_homo_all hw
    have hjw : t.widths[j]! = w := by
      rw [getElem!_pos t.widths j hj]
      exact hall _ (Array.getElem_mem_toList hj)
    rw [hjw]
    have hs : ∀ (l : List ℕ), (∀ x ∈ l, x = w) → (l.take j).sum = (l.take j).length * w := by
      intro l hl
      have : ∀ x ∈ l.take j, x = w := fun x hx => hl x (List.mem_of_mem_take hx)
      rw [List.sum_eq_card_nsmul _ w this, smul_eq_mul]
    rw [hs _ hall, List.length_take, Array.length_toList, Nat.min_eq_left (Nat.le_of_lt hj)]
  | none =>
    simp only
    have hp := prefix_sums t.widths.toList #[0] (by simp)
    rw [Array.foldl_toList] at hp
    congr 1
    rw [getElem!_def, ← Array.getElem?_toList, hp]
    cases j with
    | zero => simp
    | succ j =>
      have hj' : j < t.widths.toList.length := by simp; omega
      simp [Array.back!, List.getElem?_range, show j < t.widths.size by omega]

theorem flatMap_range_one : ∀ n : ℕ, (List.range n).flatMap (fun _ => [1]) = List.replicate n 1
  | 0 => rfl
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, flatMap_range_one n, List.replicate_succ']
    simp

theorem bitTypeWidths_go : ∀ t : Ty,
    (Ty.bitType t).widths.toList = t.widths.toList.flatMap fun w => List.replicate w 1
  | .value w => by
    by_cases h : w = 1
    · subst h; simp [Ty.bitType, Ty.widths]
    · have hb : (w == 1) = false := by simpa using h
      simp only [Ty.bitType, hb, Bool.false_eq_true, ↓reduceIte, Ty.widths]
      rw [← Array.foldl_toList, foldl_append_toList]
      simp [flatMap_range_one]
  | .array n e => by
    have ih := bitTypeWidths_go e
    simp only [Ty.bitType, Ty.widths]
    rw [← Array.foldl_toList, foldl_append_toList, ← Array.foldl_toList, foldl_append_toList]
    simp only [Array.toList_empty, List.nil_append, ih, List.flatMap_assoc]
  | .tuple fs => by
    have ih : ∀ f ∈ fs.toList, (Ty.bitType f).widths.toList = f.widths.toList.flatMap fun w => List.replicate w 1 :=
      fun f hf => bitTypeWidths_go f
    simp only [Ty.bitType, Ty.widths, Array.attach_map_val]
    rw [← Array.foldl_toList, foldl_append_toList, ← Array.foldl_toList, foldl_append_toList]
    simp only [Array.toList_empty, List.nil_append, Array.toList_map, List.flatMap_map, List.flatMap_assoc]
    exact List.flatMap_congr ih
  | .record fs => by
    have ih : ∀ f ∈ fs.toList,
        (Ty.bitType f.2).widths.toList = f.2.widths.toList.flatMap fun w => List.replicate w 1 :=
      fun f hf => bitTypeWidths_go f.2
    simp only [Ty.bitType, Ty.widths]
    rw [Array.attach_map_val fs fun p => (p.1, Ty.bitType p.2)]
    rw [← Array.foldl_toList, foldl_append_toList, ← Array.foldl_toList, foldl_append_toList]
    simp only [Array.toList_empty, List.nil_append, Array.toList_map, List.flatMap_map, List.flatMap_assoc]
    exact List.flatMap_congr ih
termination_by t => t
decreasing_by
  all_goals simp_wf
  all_goals first
    | omega
    | (have := hb_lt_mem (by assumption); omega)
    | (have := hb_lt_mem2 (by assumption); omega)

theorem bitTypeWidths : BitTypeWidths := bitTypeWidths_go

theorem sum_flatMap_replicate (l : List ℕ) : (l.flatMap fun w => List.replicate w 1).length = l.sum := by
  induction l with
  | nil => rfl
  | cons w l ih => simp [List.flatMap_cons, ih]

/-- `Ty.bitType t` has as many leaves as `t` has bits. -/
theorem bitType_leaves (t : Ty) : (Ty.bitType t).leaves = t.widths.foldl (· + ·) 0 := by
  rw [← hb_widths_size, ← Array.length_toList, bitTypeWidths, sum_flatMap_replicate, sum_foldl_add]

theorem bitTypeAxis : BitTypeAxis := by
  intro t ax
  induction ax generalizing t with
  | zero =>
    intro n L h
    cases t with
    | array m e =>
      simp only [Ty.axisDims, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨e, rfl, by simp [Ty.bitType, Ty.axisDims], bitType_leaves e⟩
    | _ => simp [Ty.axisDims] at h
  | succ ax ih =>
    intro n L h
    cases t with
    | array m e =>
      simp only [Ty.axisDims] at h
      obtain ⟨e', h1, h2, h3⟩ := ih e n L h
      exact ⟨e', h1, by simpa [Ty.bitType, Ty.axisDims] using h2, h3⟩
    | _ => simp [Ty.axisDims] at h

/-- A sequence's references (`Seq.toArray`), part by part. -/
theorem seq_toArray_toList (s : Seq) :
    s.toArray.toList = s.parts.toList.flatMap fun q => (List.range q.count).map q.get := by
  unfold Seq.toArray
  rw [← Array.foldl_toList]
  suffices h : ∀ (ps : List Part) (acc : Array Ref),
      (ps.foldl (fun acc q => (Array.range q.count).foldl (fun acc k => acc.push (q.get k)) acc) acc).toList =
        acc.toList ++ ps.flatMap fun q => (List.range q.count).map q.get by
    simpa using h s.parts.toList #[]
  intro ps
  induction ps with
  | nil => simp
  | cons q ps ih =>
    intro acc
    rw [List.foldl_cons, ih, List.flatMap_cons, ← List.append_assoc]
    congr 1
    rw [← Array.foldl_toList]
    suffices h' : ∀ (ks : List ℕ) (a : Array Ref), (ks.foldl (fun acc k => acc.push (q.get k)) a).toList =
        a.toList ++ ks.map q.get by simpa using h' _ acc
    intro ks
    induction ks with
    | nil => simp
    | cons k ks ihk => intro a; simp [ihk]

theorem range_mul_map {β : Type} (g : ℕ → β) (w : ℕ) : ∀ c : ℕ,
    (List.range (c * w)).map g = (List.range c).flatMap fun k => (List.range w).map fun t => g (k * w + t)
  | 0 => by simp
  | c + 1 => by
    rw [Nat.succ_mul, List.range_add, List.map_append, range_mul_map g w c, List.range_succ, List.flatMap_append]
    simp [Function.comp_def]

theorem ofParts_parts (ps : Array Part) : (Seq.ofParts ps).parts = ps := rfl

theorem explicit_refs (A : Array Ref) : (List.range (Part.explicit A).count).map (Part.explicit A).get = A.toList := by
  apply List.ext_getElem
  · simp [Part.count]
  · intro n h1 h2
    simp only [Part.count, List.length_map, List.length_range] at h1
    simp [Flock.Program.Part.get, getElem!_pos A n h1]

theorem foldl_append_of {β : Type} {f : Array Ref → β → Array Ref} {g : β → Array Ref}
    (h : ∀ a k, f a k = a ++ g k) (l : List β) (acc : Array Ref) :
    (l.foldl f acc).toList = acc.toList ++ l.flatMap fun k => (g k).toList := by
  induction l generalizing acc with
  | nil => simp
  | cons k l ih => rw [List.foldl_cons, ih, h, List.flatMap_cons]; simp

theorem liftSeqRefs : LiftSeqRefs := by
  intro at_ homo s hh
  let G : Ref → List Ref := fun r =>
    (List.range (at_ r.p r.i r.j).2).map fun t => ({ p := r.p, i := r.i, j := (at_ r.p r.i r.j).1 + t } : Ref)
  have hG : ∀ q : Part, ((Array.range q.count).foldl (fun a k => a ++
      (Array.range (at_ (q.get k).p (q.get k).i (q.get k).j).2).map fun t =>
        ({ p := (q.get k).p, i := (q.get k).i, j := (at_ (q.get k).p (q.get k).i (q.get k).j).1 + t } : Ref)) #[]).toList =
      ((List.range q.count).map q.get).flatMap G := by
    intro q
    rw [← Array.foldl_toList, foldl_append_of (fun a k => rfl)]
    simp [G, List.flatMap_map]
  unfold Flock.Lift.liftSeq
  rw [seq_toArray_toList, seq_toArray_toList, ofParts_parts, Array.toList_map, List.flatMap_map, List.flatMap_assoc]
  apply List.flatMap_congr
  intro q _
  rcases q with ⟨p, i, b, c⟩ | ⟨p, i, b, inner, os, is, c⟩ | ⟨it⟩ | ⟨run, lanes⟩
  · dsimp only
    cases hw : homo p i with
    | none =>
      dsimp only
      rw [explicit_refs]
      exact hG _
    | some w =>
      dsimp only
      simp only [Part.count]
      rw [range_mul_map, List.flatMap_map]
      apply List.flatMap_congr
      intro k _
      show List.map (fun t => (⟨p, i, b * w + (k * w + t)⟩ : Ref)) (List.range w) =
        List.map (fun t => ({ p := p, i := i, j := (at_ p i (b + k)).1 + t } : Ref)) (List.range (at_ p i (b + k)).2)
      rw [hh p i w hw (b + k)]
      apply List.map_congr_left
      intro t _
      simp [Nat.add_mul, Nat.add_assoc]
  all_goals
    dsimp only
    rw [explicit_refs]
    exact hG _

/-! ## `RestrictWordEval`, proved -/

/-- The walk's one call reads Definition `d`, the Definitions below it and, for a primitive with no gate, its Boolean
Definition: two Programs and two recursive calls that agree there give the same value. -/
theorem walkStep_congr {W : Type} [Inhabited W] (I : Flock.Program.Wires Option W) (hI : ∀ {α : Type} (o : Option α), I.lift o = o)
    {P Q : Prog} {ev ev' : ℕ → List W → Option (List W)} {d : ℕ} (x : List W)
    (hd : P.defs[d]? = Q.defs[d]?) (hlow : ∀ i < d, P.defs[i]? = Q.defs[i]?)
    (hev : ∀ e < d, ∀ ins, ev e ins = ev' e ins)
    (hprim : ∀ defn, P.defs[d]? = some defn → defn.prim = true → P.bodyOf? defn.key = Q.bodyOf? defn.key ∧
      ∀ b, P.bodyOf? defn.key = some b → ∀ ins, ev b ins = ev' b ins) :
    Flock.Program.walkStep I P ev d x = Flock.Program.walkStep I Q ev' d x := by
  unfold Flock.Program.walkStep
  rw [← hd]
  cases hD : P.defs[d]? with
  | none => simp [hI]
  | some defn =>
    simp only [hI, Option.bind_eq_bind, Option.bind_some]
    by_cases hp : defn.prim = true
    · obtain ⟨hb, hbe⟩ := hprim defn hD hp
      simp only [hp, ↓reduceIte, hb]
      split
      · rfl
      · cases hB : Q.bodyOf? defn.key with
        | none => rfl
        | some b =>
          have := hbe b (by rw [hb, hB])
          simp [this]
    · simp only [hp, Bool.false_eq_true, ↓reduceIte]
      congr 1
      congr 1
      funext outs nd
      by_cases h : d ≤ nd.fn
      · simp only [h, ↓reduceIte]
        rfl
      · simp only [h, ↓reduceIte, hlow nd.fn (by omega), hev nd.fn (by omega)]

theorem get?_of_pre {P S : Prog} (hle : P.defs.size ≤ S.defs.size) (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!)
    {i : ℕ} (hi : i < P.defs.size) : S.defs[i]? = P.defs[i]? := by
  have h1 : i < S.defs.size := by omega
  rw [getElem?_pos S.defs i h1, getElem?_pos P.defs i hi]
  congr 1
  have := hpre i hi
  rwa [getElem!_pos S.defs i h1, getElem!_pos P.defs i hi] at this

theorem body_lt {P : Prog} (hm : P.BoolMapOk) {k : String} {b : ℕ} (h : P.bodyOf? k = some b) : b < P.defs.size := by
  have h0 := h
  unfold Prog.bodyOf? at h0
  cases hv : P.boolean[k]? with
  | none => rw [hv] at h0; cases h0
  | some v =>
    obtain ⟨w, hw, -, hkey, hword⟩ := hm.2 k v hv
    obtain ⟨b', hb', hbo, -⟩ := hm.1 w hw (by rw [hkey]; exact hword)
    rw [hkey] at hbo
    obtain rfl : b' = b := Option.some.inj (hbo.symm.trans h)
    exact hb'

/-- The walk on bits in `S` is the walk in `P` at `P`'s Definitions, where `S` keeps them and their Boolean
Definitions. -/
theorem evalDef_restrict {P S : Prog} (hle : P.defs.size ≤ S.defs.size)
    (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!) (hbody : ∀ k, S.bodyOf? k = P.bodyOf? k)
    (hblt : ∀ k b, P.bodyOf? k = some b → b < P.defs.size) :
    ∀ f e ins, e < P.defs.size → Flock.Program.evalDef? S f e ins = Flock.Program.evalDef? P f e ins
  | 0, _, _, _ => rfl
  | f + 1, e, ins, he => by
    unfold Flock.Program.evalDef?
    apply walkStep_congr _ (fun _ => rfl) _ (get?_of_pre hle hpre he)
      (fun i hi => get?_of_pre hle hpre (by omega))
      (fun e' he' ins' => evalDef_restrict hle hpre hbody hblt f e' ins' (by omega))
    intro defn _ _
    refine ⟨hbody _, fun b hb ins' => evalDef_restrict hle hpre hbody hblt f b ins' ?_⟩
    rw [hbody] at hb
    exact hblt _ _ hb

theorem evalDef_fuel_iter (P : Prog) (hB : ∀ k, isBooleanPrim k = true → P.bodyOf? k = none) {b : ℕ}
    (hb : P.BoolDef b) (x : List Bool) :
    ∀ k, Flock.Program.evalDef? P (b + 1 + k) b x = Flock.Program.evalDef? P (b + 1) b x
  | 0 => rfl
  | k + 1 => by
    rw [show b + 1 + (k + 1) = (b + 1 + k) + 1 from by omega,
      FlockSoundness.DecodeBoolean.fuel_stable P hB hb (by omega) x, evalDef_fuel_iter P hB hb x k]

theorem evalDef_fuel_at (P : Prog) (hB : ∀ k, isBooleanPrim k = true → P.bodyOf? k = none) {b m : ℕ}
    (hb : P.BoolDef b) (hm : b < m) (x : List Bool) :
    Flock.Program.evalDef? P m b x = Flock.Program.evalDef? P (b + 1) b x := by
  obtain ⟨k, rfl⟩ : ∃ k, m = b + 1 + k := ⟨m - (b + 1), by omega⟩
  exact evalDef_fuel_iter P hB hb x k

/-- `restrict` keeps every primitive of C evaluating as in C. -/
theorem evalBool_restrict {bytes : ByteArray} {P S : Prog} {units : Array ℕ} (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[]) (hs : Spot.Restrict.restrict P units = .ok S) {p : ℕ}
    (hp : p < P.defs.size) (hprim : (P.defs[p]!).prim = true) (y : List Bool) :
    Prog.evalBool S p y = Prog.evalBool P p y := by
  obtain ⟨hle, hpre, hbool, hidx, -⟩ := restrictShape P units S hs
  obtain ⟨⟨hmap, hkeys⟩, -⟩ := FlockSoundness.DecodeBoolean.decode_facts bytes P h hr
  have hm : P.BoolMapOk := ⟨hmap, hkeys⟩
  have hbody := bodyOf_eq hm hbool hidx
  have hB := FlockSoundness.DecodeBoolean.bool_none hkeys
  have hBS : ∀ k, isBooleanPrim k = true → S.bodyOf? k = none := fun k hk => by rw [hbody]; exact hB k hk
  have hdp : S.defs[p]? = some (P.defs[p]!) := by
    rw [get?_of_pre hle hpre hp, getElem!_pos P.defs p hp]; exact getElem?_pos P.defs p hp
  have hdP : P.defs[p]? = some (P.defs[p]!) := by rw [getElem!_pos P.defs p hp]; exact getElem?_pos P.defs p hp
  unfold Prog.evalBool
  rw [show 2 * S.defs.size + 2 = (2 * S.defs.size + 1) + 1 from rfl,
    show 2 * P.defs.size + 2 = (2 * P.defs.size + 1) + 1 from rfl]
  unfold Flock.Program.evalDef?
  unfold Flock.Program.walkStep
  rw [hdp, hdP]
  simp only [Flock.Program.Wires.bool, Option.pure_def, Option.bind_eq_bind, Option.bind_some, hprim, ↓reduceIte]
  split
  · rfl
  · rw [hbody]
    cases hb : P.bodyOf? (P.defs[p]!).key with
    | none => rfl
    | some b =>
      have hbl := body_lt hm hb
      have hbd : P.BoolDef b := by
        have h0 := hb
        unfold Prog.bodyOf? at h0
        cases hv : P.boolean[(P.defs[p]!).key]? with
        | none => rw [hv] at h0; cases h0
        | some v =>
          obtain ⟨w, hw, -, hkey, hword⟩ := hkeys _ v hv
          obtain ⟨b', -, hbo, hbody', -⟩ := hmap w hw (by rw [hkey]; exact hword)
          rw [hkey] at hbo
          obtain rfl : b' = b := Option.some.inj (hbo.symm.trans hb)
          exact hbody'
      have hbdS : S.BoolDef b := boolDef_lift hpre hbd hbl
      simp only [Option.bind_some]
      rw [evalDef_fuel_at S hBS hbdS (by omega) y, evalDef_fuel_at P hB hbd (by omega) y,
        evalDef_restrict hle hpre hbody (fun k b' h' => body_lt hm h') (b + 1) b y hbl]

theorem wordDef_restrict {P S : Prog} (hle : P.defs.size ≤ S.defs.size)
    (hpre : ∀ i < P.defs.size, S.defs[i]! = P.defs[i]!)
    (hpe : ∀ p < P.defs.size, (P.defs[p]!).prim = true → ∀ y, Prog.evalBool S p y = Prog.evalBool P p y) :
    ∀ f d x, d < P.defs.size → wordDef? S f d x = wordDef? P f d x
  | 0, _, _, _ => rfl
  | f + 1, d, x, hd => by
    unfold wordDef?
    rw [get?_of_pre hle hpre hd]
    cases hD : P.defs[d]? with
    | none => rfl
    | some defn =>
      have hdn : P.defs[d]! = defn := by rw [getElem!_def, hD]
      simp only [Option.bind_eq_bind, Option.bind_some]
      by_cases hp : defn.prim = true
      · simp only [hp, ↓reduceIte, hpe d hd (by rw [hdn]; exact hp)]
      · simp only [hp, Bool.false_eq_true, ↓reduceIte]
        apply walkStep_congr _ (fun _ => rfl) _ (get?_of_pre hle hpre hd)
          (fun i hi => get?_of_pre hle hpre (by omega))
          (fun e he ins => wordDef_restrict hle hpre hpe f e ins (by omega))
        intro defn' h' hp'
        rw [get?_of_pre hle hpre hd, hD] at h'
        cases h'
        exact absurd hp' hp

theorem wordDef_fuel (P : Prog) : ∀ d f x, d < f → wordDef? P f d x = wordDef? P (d + 1) d x := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d ih =>
    intro f x hf
    obtain ⟨f', rfl⟩ : ∃ f', f = f' + 1 := ⟨f - 1, by omega⟩
    unfold wordDef?
    cases hD : P.defs[d]? with
    | none => rfl
    | some defn =>
      simp only [Option.bind_eq_bind, Option.bind_some]
      by_cases hp : defn.prim = true
      · simp only [hp, ↓reduceIte]
      · simp only [hp, Bool.false_eq_true, ↓reduceIte]
        apply walkStep_congr _ (fun _ => rfl) _ rfl (fun _ _ => rfl)
          (fun e he ins => by rw [ih e he f' ins (by omega), ih e he d ins he])
        intro defn' h' hp'
        rw [hD] at h'
        cases h'
        exact absurd hp' hp

theorem restrictWordEval : RestrictWordEval := by
  intro bytes P S units h hr hs d hd x
  obtain ⟨hle, hpre, -⟩ := restrictShape P units S hs
  unfold wordEval
  rw [wordDef_fuel S d _ x (by omega), wordDef_fuel P d _ x (by omega),
    wordDef_restrict hle hpre (fun p hp hprim y => evalBool_restrict h hr hs hp hprim y) (d + 1) d x hd]

end Flock.Chain

