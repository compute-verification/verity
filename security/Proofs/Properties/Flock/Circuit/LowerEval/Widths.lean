module

public import Proofs.Properties.Flock.Circuit.LowerAsm
public import Proofs.Properties.Flock.Circuit.Chain
public import Proofs.Properties.Flock.DecodeBoolean
public import Proofs.Properties.Poub.Remote.Sort
public import Proofs.Properties.InferenceOnly.OperandWidths
public import Proofs.Core.Expand

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
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
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
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  try dsimp only at h
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  try dsimp only at h
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  try dsimp only at h
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  rename_i r _
  obtain ⟨L, dl⟩ := r
  rw [pure_bind] at h
  try dsimp only at h
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  try dsimp only at h
  obtain ⟨⟨⟩, hc, h⟩ := Flock.HmNets.bind_ok h
  try dsimp only at h
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  try dsimp only at h
  obtain ⟨⟨⟩, hw, h⟩ := Flock.HmNets.bind_ok h
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  cases h
  simp only [List.forIn_cons, List.forIn_nil] at hw
  obtain ⟨s1, h, hw⟩ := Flock.HmNets.bind_ok hw
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
  cases h
  obtain ⟨s2, h, -⟩ := Flock.HmNets.bind_ok hw
  split at h
  any_goals (first | exact absurd h notAdmitted_ne | exact absurd h notAdmitted_bind_ne | (exfalso; cases h; done))
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

/-- **`RestrictBodyAgrees`**. -/
theorem restrictBodyAgrees : RestrictBodyAgrees := restrictBodyAgreesDecoded

/-! ## `BlockEvalDoor`, on its four open premises -/

/-- Unit `g`'s gates in C: the gates of its member scope (`Prog.child` of the root at its node and member). -/
def InUnit (P : Prog) (k m h : ℕ) : Prop :=
  (P.child P.rootScope k m).off ≤ h ∧
    h < (P.child P.rootScope k m).off + (((P.defs[(P.child P.rootScope k m).d]?).map (·.gates)).getD 0)

/-- **`FrontUnitRanges`**: each unit of a block `front` accepts is located (`unitAt`), and its gates are in the
front's ranges. -/
def FrontUnitRanges : Prop :=
  ∀ (P : Prog) (units : Array ℕ) (fr : Spot.BlockFront.Front), Spot.BlockFront.front P units = .ok fr →
    ∀ g ∈ units, ∃ k m, Spot.BlockFront.unitAt P g = .ok (k, m) ∧
      ∀ h, InUnit P k m h → Spot.BlockFront.inside fr.ranges h = true

section GateDefs
open Flock.Lower Flock.Chain.LowerDef

/-- Definition `d` of `P` evaluates through §9 gates alone (`gateOf?`): a primitive that is one, or a composite each of
whose nodes calls such a Definition placed before it. An input primitive is none. -/
inductive GateDef (P : Prog) : ℕ → Prop
  | prim {d : ℕ} : (P.defs[d]!).prim = true → (gateOf? (P.defs[d]!).id).isSome = true → GateDef P d
  | comp {d : ℕ} : (P.defs[d]!).prim = false → (∀ nd ∈ (P.defs[d]!).body.nodes, nd.fn < d) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, GateDef P nd.fn) → GateDef P d

theorem foldlM_isSome {α β : Type} (f : β → α → Option β) :
    ∀ (l : List α) (init : β), (∀ b, ∀ a ∈ l, (f b a).isSome = true) → (l.foldlM f init).isSome = true
  | [], _, _ => rfl
  | a :: l, init, h => by
    obtain ⟨c, hc⟩ := Option.isSome_iff_exists.1 (h init a (by simp))
    rw [List.foldlM_cons, hc]
    exact foldlM_isSome f l c fun b a' ha' => h b a' (by simp [ha'])

theorem mapM_isSome {α β : Type} [Inhabited β] (f : α → Option β) (l : List α) (h : ∀ a ∈ l, (f a).isSome = true) :
    (l.mapM f).isSome = true := by
  rw [Flock.Chain.LowerDef.mapM_some (g := fun a => (f a).getD default) fun a ha => ?_]
  · rfl
  · obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 (h a ha)
    simp [hv]

theorem bind_isSome {α β : Type} {a : Option α} {f : α → Option β} (ha : a.isSome = true)
    (hf : ∀ v, (f v).isSome = true) : (a.bind f).isSome = true := by
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 ha
  rw [hv]
  exact hf v

/-- **A Definition through gates alone evaluates on every input**, at fuel one past its place. -/
theorem gateDef_total (P : Prog) : ∀ d, d < P.defs.size → GateDef P d → ∀ x : List Bool,
    (evalDef? P (d + 1) d x).isSome = true := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d ih =>
  intro hd hg x
  have hdc : P.defs[d]? = some (P.defs[d]!) := by
    rw [getElem!_pos P.defs d hd]; exact Array.getElem?_eq_getElem hd
  cases hg with
  | prim hp hgt =>
    obtain ⟨g, hg⟩ := Option.isSome_iff_exists.1 hgt
    rw [evalDef?_succ]
    simp [walkStep, Wires.bool, hdc, hp, hg]
  | comp hp hlt hnd =>
    rw [evalDef?_succ, walkStep_eq hdc hp]
    have hf := foldlM_isSome (walkNode P (evalDef? P d) d x (paramStarts (P.defs[d]!).params))
      (P.defs[d]!).body.nodes.toList #[] fun outs nd hn => by
        have hn' : nd ∈ (P.defs[d]!).body.nodes := Array.mem_def.2 hn
        have hfd := hlt nd hn'
        have hcal : P.defs[nd.fn]? = some (P.defs[nd.fn]!) := by
          rw [getElem!_pos P.defs nd.fn (by omega)]; exact Array.getElem?_eq_getElem (by omega)
        obtain ⟨G, hG⟩ : ∃ G : List Bool → List Bool, ∀ ins, evalDef? P d nd.fn ins = some (G ins) := by
          refine ⟨fun ins => (evalDef? P d nd.fn ins).getD [], fun ins => ?_⟩
          obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 (ih nd.fn hfd (by omega) (hnd nd hn') ins)
          show evalDef? P d nd.fn ins = some ((evalDef? P d nd.fn ins).getD [])
          rw [FlockSoundness.EvalBool.evalDef?_mono P hfd hv]
          rfl
        unfold walkNode
        simp only [Nat.not_le.2 hfd, ite_false, hcal, hG]
        cases nd.form <;> simp
        all_goals first
          | exact bind_isSome (mapM_isSome _ _ fun _ _ => rfl) fun _ => rfl
          | exact bind_isSome (foldlM_isSome _ _ _ fun _ _ _ => rfl) fun _ => rfl
    obtain ⟨o, ho⟩ := Option.isSome_iff_exists.1 hf
    rw [ho]
    rfl

theorem gateDef_done {P : Prog} {f : ℕ} {s s' : St} {r : ℕ × Array ℕ} (h : (gateDef P f).run s = .ok (r, s')) :
    s'.done = s.done := by
  unfold gateDef at h
  obtain ⟨s1, s2, h1, h2⟩ := st_bind_ok h
  have h1' : (Except.ok (s, s) : Except String (St × St)) = .ok (s1, s2) := h1
  simp only [Except.ok.injEq, Prod.mk.injEq] at h1'
  obtain ⟨h1a, h1b⟩ := h1'
  subst s1 s2
  split at h2
  · rename_i r0 hr0
    have h2' : (Except.ok (r0, s) : Except String ((ℕ × Array ℕ) × St)) = .ok (r, s') := h2
    simp only [Except.ok.injEq, Prod.mk.injEq] at h2'
    obtain ⟨-, h2b⟩ := h2'
    subst s'
    rfl
  dsimp only at h2
  split at h2
  case h_2 => exact absurd h2 (st_throw _ _ _)
  rename_i g hg
  obtain ⟨a, s3, h3, h4⟩ := st_bind_ok h2
  have h3' : (StateT.lift (({} : Work).op g ((Array.range (P.defs[f]!).paramLeaves).map Flock.Lower.Form.input)) :
      M (Work × Flock.Lower.Form)).run s = .ok (a, s3) := h3
  obtain ⟨a', -, hq⟩ := st_lift_ok h3'
  simp only [Prod.mk.injEq] at hq
  obtain ⟨hqa, hqb⟩ := hq
  subst a s3
  obtain ⟨r1, s4, h5, h6⟩ := st_bind_ok h4
  rw [emit_run] at h5
  simp only [Except.ok.injEq, Prod.mk.injEq] at h5
  obtain ⟨h5a, h5b⟩ := h5
  subst r1 s4
  obtain ⟨u, s5, h7, h8⟩ := st_bind_ok h6
  rw [modify_run] at h7
  simp only [Except.ok.injEq, Prod.mk.injEq] at h7
  obtain ⟨-, h7b⟩ := h7
  subst s5
  have h8' : (Except.ok (_, _) : Except String ((ℕ × Array ℕ) × St)) = .ok (r, s') := h8
  simp only [Except.ok.injEq, Prod.mk.injEq] at h8'
  obtain ⟨-, h8b⟩ := h8'
  subst s'
  rfl

/-- `lowerDef`'s cache of lowered Definitions holds Definitions through gates alone. -/
def GateCache (P : Prog) (s : St) : Prop := ∀ f r, s.done[f]? = some r → GateDef P f

theorem nodes_gates {P : Prog} {low : ℕ → M (ℕ × Array ℕ)} {defn : Defn} {d : ℕ} {st : Array ℕ}
    (hlow : ∀ f (s s1 : St) (r : ℕ × Array ℕ), (low f).run s = .ok (r, s1) → GateCache P s →
      GateDef P f ∧ GateCache P s1) :
    ∀ (L : List Node) (w w' : Work) (s s' : St), (L.foldlM (nodeStep P low defn d st) w).run s = .ok (w', s') →
      GateCache P s → (∀ x ∈ L, x.fn < d ∧ GateDef P x.fn) ∧ GateCache P s'
  | [], w, w', s, s', h, hI => by
    have h' : (Except.ok (w, s) : Except String (Work × St)) = .ok (w', s') := h
    simp only [Except.ok.injEq, Prod.mk.injEq] at h'
    obtain ⟨-, rfl⟩ := h'
    exact ⟨fun x hx => absurd hx (by simp), hI⟩
  | x :: L, w, w', s, s', h, hI => by
    rw [List.foldlM_cons] at h
    obtain ⟨w1, s1, h1, h2⟩ := st_bind_ok h
    obtain ⟨hfn, -, hc⟩ := nodeStep_cases h1
    have hx : GateDef P x.fn ∧ GateCache P s1 := by
      rcases hc with ⟨-, -, e, rets, s2, hl, rfl, -⟩ | ⟨-, hpr, g, hg, hb⟩ | ⟨-, -, e, rets, s2, hl, rfl, -⟩
      · exact hlow _ _ _ _ hl hI
      · refine ⟨.prim hpr (by rw [hg]; rfl), ?_⟩
        rcases hb with ⟨-, e, rets, s2, hgd, rfl, -⟩ | ⟨-, rfl, -⟩
        · intro f r hr
          rw [gateDef_done hgd] at hr
          exact hI f r hr
        · exact hI
      · exact hlow _ _ _ _ hl hI
    obtain ⟨hL, hI2⟩ := nodes_gates hlow L w1 w' s1 s' h2 hx.2
    refine ⟨fun y hy => ?_, hI2⟩
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨hfn, hx.1⟩
    · exact hL y hy

/-- **What `lowerDef` lowers evaluates through gates alone**: it refuses a node whose callee is a primitive that is no
§9 gate (an input primitive), and lowers each composite callee first. -/
theorem lowerDef_gates {P : Prog} :
    ∀ (fuel d : ℕ) (s s' : St) (r : ℕ × Array ℕ), (lowerDef P fuel d).run s = .ok (r, s') → GateCache P s →
      GateDef P d ∧ GateCache P s'
  | 0, d, s, s', r, h, _ => by
    rw [lowerDef] at h
    exact absurd h (st_throw _ _ _)
  | fuel + 1, d, s, s', r, h, hI => by
    rw [lowerDef] at h
    obtain ⟨s1, s2, h1, h2⟩ := st_bind_ok h
    have h1' : (Except.ok (s, s) : Except String (St × St)) = .ok (s1, s2) := h1
    simp only [Except.ok.injEq, Prod.mk.injEq] at h1'
    obtain ⟨h1a, h1b⟩ := h1'
    subst s1 s2
    split at h2
    · rename_i r0 hr0
      have h2' : (Except.ok (r0, s) : Except String ((ℕ × Array ℕ) × St)) = .ok (r, s') := h2
      simp only [Except.ok.injEq, Prod.mk.injEq] at h2'
      obtain ⟨-, rfl⟩ := h2'
      exact ⟨hI d r0 hr0, hI⟩
    dsimp only at h2
    split at h2
    · first | exact absurd h2 (st_throw_bind _ _ _ _) | exact absurd h2 (st_throw _ _ _)
    rename_i hpr
    obtain ⟨u, s1, hb, h2⟩ := st_bind_ok h2
    obtain ⟨_, -, hq⟩ := st_lift_ok hb
    simp only [Prod.mk.injEq] at hq
    obtain ⟨-, hq⟩ := hq
    subst s1
    split at h2
    · first | exact absurd h2 (st_throw_bind _ _ _ _) | exact absurd h2 (st_throw _ _ _)
    split at h2
    · first | exact absurd h2 (st_throw_bind _ _ _ _) | exact absurd h2 (st_throw _ _ _)
    split at h2
    · first | exact absurd h2 (st_throw_bind _ _ _ _) | exact absurd h2 (st_throw _ _ _)
    have hpr' : (P.defs[d]!).prim = false := by simpa using hpr
    obtain ⟨w, s3, h3, h4⟩ := st_bind_ok h2
    obtain ⟨hN, hI3⟩ := nodes_gates (fun f s s1 r h hI => lowerDef_gates fuel f s s1 r h hI) _ _ _ _ _ h3 hI
    have hG : GateDef P d := .comp hpr' (fun nd hn => (hN nd (Array.mem_toList_iff.2 hn)).1)
      (fun nd hn => (hN nd (Array.mem_toList_iff.2 hn)).2)
    obtain ⟨wr, s4, h5, h6⟩ := st_bind_ok h4
    obtain ⟨wr', -, hq⟩ := st_lift_ok h5
    simp only [Prod.mk.injEq] at hq
    obtain ⟨hqa, hqb⟩ := hq
    subst wr s4
    obtain ⟨w2, rets⟩ := wr'
    dsimp only at h6
    obtain ⟨r1, s5, h7, h8⟩ := st_bind_ok h6
    rw [emit_run] at h7
    simp only [Except.ok.injEq, Prod.mk.injEq] at h7
    obtain ⟨h7a, h7b⟩ := h7
    subst r1 s5
    obtain ⟨u, s6, h9, h10⟩ := st_bind_ok h8
    rw [modify_run] at h9
    simp only [Except.ok.injEq, Prod.mk.injEq] at h9
    obtain ⟨-, h9b⟩ := h9
    subst s6
    have h10' : (Except.ok (_, _) : Except String ((ℕ × Array ℕ) × St)) = .ok (r, s') := h10
    simp only [Except.ok.injEq, Prod.mk.injEq] at h10'
    obtain ⟨-, rfl⟩ := h10'
    refine ⟨hG, fun f r' hr' => ?_⟩
    simp only [Std.HashMap.getElem?_insert] at hr'
    split at hr'
    · rename_i hff
      simp only [beq_iff_eq] at hff
      subst hff
      exact hG
    · exact hI3 f r' hr'

/-- **The lowering reaches gates alone**: where `lower` writes a template of Definition `d` of `P`, the lift to bits
of `d` evaluates through §9 gates alone. -/
theorem lower_gates {P : Prog} {d : ℕ} {inP outP : Array ℕ} {t : Tmpl.Tmpl}
    (hl : Flock.Lower.lower P d inP outP = .ok t) :
    ∃ Q e, Flock.Lift.lift P d = .ok (Q, e) ∧ GateDef Q e := by
  unfold Flock.Lower.lower at hl
  obtain ⟨t', hb, -⟩ := Flock.TmplOrder.bind_ok hl
  unfold Flock.Lower.build at hb
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at hb
  split at hb
  · cases hb
  cases hq : Flock.Lift.lift P d with
  | error err => simp [hq] at hb
  | ok qe =>
  obtain ⟨Q, e⟩ := qe
  simp only [hq] at hb
  cases hdf : Q.defs[e]? with
  | none => simp [hdf] at hb
  | some defn =>
  simp only [hdf] at hb
  split at hb
  · cases hb
  split at hb
  · cases hb
  cases hrun : StateT.run (Flock.Lower.lowerDef Q (Q.defs.size + 1) e) {} with
  | error err => simp [hrun] at hb
  | ok v =>
  obtain ⟨⟨e', rets⟩, s⟩ := v
  exact ⟨Q, e, rfl, (lowerDef_gates _ _ _ _ _ hrun fun f r hr => by simp at hr).1⟩

theorem gateDef_prefix {P L : Prog} (hpre : ∀ i < P.defs.size, L.defs[i]! = P.defs[i]!) {b : ℕ}
    (h : GateDef L b) : b < P.defs.size → GateDef P b := by
  induction h with
  | prim hp hg => intro hb; exact .prim (by rwa [← hpre _ hb]) (by rwa [← hpre _ hb])
  | @comp d hp hlt _ ih =>
    intro hb
    rw [hpre _ hb] at hp hlt ih
    exact .comp hp hlt fun nd hn => ih nd hn (by have := hlt nd hn; omega)

/-- Each Boolean Definition a node of Definition `d` of `P` leads to (`Prog.bodyOf?`) evaluates through gates alone. -/
def GateBodies (P : Prog) (d : ℕ) : Prop :=
  ∀ nd ∈ (P.defs[d]!).body.nodes, ∀ b, P.bodyOf? (P.defs[nd.fn]!).key = some b → GateDef P b

/-- The Definitions reached from `d` through composites: `d`, and each callee of a reached composite. -/
inductive Reach (S : Prog) (d : ℕ) : ℕ → Prop
  | root : Reach S d d
  | callee {y : ℕ} {nd : Node} : Reach S d y → (S.defs[y]!).prim = false → nd ∈ (S.defs[y]!).body.nodes →
      Reach S d nd.fn

/-- Each Boolean Definition a node of a composite reached from `d` in `P` leads to evaluates through gates alone
(`GateBodies` at every composite `Reach P d` walks). Its `Reach` walks `P`, the decoded C; for a Program that keeps C's
Definitions, as the restricted Program the lift walks does, the two walks from a `d` placed in C reach the same
Definitions: `reach_restrict` (C's walk within the keeping Program's) and `reach_of_restrict` (the keeping Program's within
C's). -/
def GateBodiesDeep (P : Prog) (d : ℕ) : Prop :=
  ∀ y, Reach P d y → (P.defs[y]!).prim = false → GateBodies P y

theorem gateBodies_of_deep {P : Prog} {d : ℕ} (hp : (P.defs[d]!).prim = false) (h : GateBodiesDeep P d) :
    GateBodies P d :=
  h d .root hp

end GateDefs

/-- **`UnitComputes`** (open): in a trace of C whose every computed gate of unit `g` is its operation on the trace,
`g`'s return leaves hold, bit for bit, `Prog.evalBool` on `g`'s argument bits of the lift to bits (`Flock.Lift.lift`)
of the template `admitted` gives, where that template is `g`'s callee and the block's lowering (`blockTmpl`) succeeds. -/
def UnitComputes : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (L : Prog) (e : ℕ) (t : Tmpl.Tmpl),
    decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    Spot.OneStageZk.admittedBlock P units a.key = .ok () → Flock.Lift.lift S a.d = .ok (L, e) →
    Spot.OneStageZk.blockTmpl S a = .ok t →
    (∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h) →
    (∀ g ∈ units, UnitResolves P g) →
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

/-- An entry `(u, m, x, f, lo, hi)` of `admittedBlock`'s table of units passes `one-callee`: its callee `f` is `key`. -/
def CalleeOk (key : String) (e : ℕ × ℕ × Node × Defn × ℕ × ℕ) : Prop := e.2.2.2.1.key = key

theorem ac_step {key : String} {pref : List (ℕ × ℕ × Node × Defn × ℕ × ℕ)} {cur : ℕ × ℕ × Node × Defn × ℕ × ℕ}
    (h3 : ¬ (cur.2.2.2.1.key != key) = true) (hinv : ∀ e ∈ pref, CalleeOk key e) :
    ∀ e, e ∈ pref ∨ e = cur → CalleeOk key e := by
  intro e he
  rcases he with he | rfl
  · exact hinv e he
  · simpa [CalleeOk] using h3

theorem ac_lam {P : Prog} {root : Defn} (hroot : P.defs[P.root]? = some root) {u : ℕ}
    {e : ℕ × ℕ × Node × Defn × ℕ × ℕ} {k m : ℕ}
    (h : (do
      let (k, m) ← Spot.BlockFront.unitAt P u
      let some x := root.body.nodes[k]? | Spot.OneStageZk.notAdmitted "one-callee" s!"unit {u} is no node of C's root"
      let some f := P.defs[x.fn]? | Spot.OneStageZk.notAdmitted "one-callee" s!"unit {u}'s callee is no Definition of C"
      let off := (P.child P.rootScope k m).off
      pure (u, m, x, f, off, off + f.gates) : V (ℕ × ℕ × Node × Defn × ℕ × ℕ)) = .ok e)
    (hk : Spot.BlockFront.unitAt P u = .ok (k, m)) :
    (P.defs[P.root]!).body.nodes[k]? = some e.2.2.1 ∧ P.defs[e.2.2.1.fn]? = some e.2.2.2.1 := by
  have hr : P.defs[P.root]! = root := by simp [getElem!_def, hroot]
  rw [hk] at h
  dsimp only [bind, Except.bind] at h
  split at h
  · rename_i x hx
    split at h
    · rename_i f hf
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨by rw [hr]; exact hx, hf⟩
    · exact absurd h notAdmitted_ne
  · exact absurd h notAdmitted_ne

theorem ac_final {P : Prog} {units : Array ℕ} {key : String} {us : Array (ℕ × ℕ × Node × Defn × ℕ × ℕ)}
    {lam : ℕ → V (ℕ × ℕ × Node × Defn × ℕ × ℕ)}
    (hlam : ∀ u e k m, lam u = .ok e → Spot.BlockFront.unitAt P u = .ok (k, m) →
      (P.defs[P.root]!).body.nodes[k]? = some e.2.2.1 ∧ P.defs[e.2.2.1.fn]? = some e.2.2.2.1)
    (hmap : ∀ x ∈ units, ∃ e ∈ us, lam x = .ok e) (hinv : ∀ e ∈ us.toList, CalleeOk key e) :
    ∀ u ∈ units, ∀ k m, Spot.BlockFront.unitAt P u = .ok (k, m) →
      ∃ (x : Node) (f : Defn), (P.defs[P.root]!).body.nodes[k]? = some x ∧ P.defs[x.fn]? = some f ∧ f.key = key := by
  intro u hu k m hk
  obtain ⟨e, he, hf⟩ := hmap u hu
  obtain ⟨h1, h2⟩ := hlam u e k m hf hk
  exact ⟨e.2.2.1, e.2.2.2.1, h1, h2, hinv e (Array.mem_toList_iff.2 he)⟩

/-- `admittedBlock`'s `one-callee`: where it accepts, every unit's root node calls `key`. -/
theorem admittedBlock_triple (P : Prog) (units : Array ℕ) (key : String) :
    ⦃⌜True⌝⦄ Spot.OneStageZk.admittedBlock P units key
    ⦃⇓? _ => ⌜∀ u ∈ units, ∀ k m, Spot.BlockFront.unitAt P u = .ok (k, m) →
      ∃ (x : Node) (f : Defn), (P.defs[P.root]!).body.nodes[k]? = some x ∧ P.defs[x.fn]? = some f ∧ f.key = key⌝⦄ := by
  unfold Spot.OneStageZk.admittedBlock
  mvcgen [ac_mapM_spec, operandsW_spec, Spot.OneStageZk.notAdmitted, exc_throw_spec, -Spec.throw_Except] invariants
  · ⇓? _ => ⌜True⌝
  · ⇓? _ => ⌜True⌝
  · ⇓? ⟨xs, _⟩ => ⌜∀ e ∈ xs.prefix, CalleeOk key e⌝
  · ⇓? _ => ⌜True⌝
  all_goals first
    | trivial
    | (intro e he; simp at he; done)
    | ((try simp only [List.mem_append, List.mem_singleton]);
        exact ac_step (by assumption) (by assumption))
    | (exact ac_final (fun u e k m h hk => ac_lam (by assumption) h hk) (by assumption) (by assumption))
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

/-- **`UnitWidths`**: unit `g`'s arguments, as `portBits` lays them, have its callee's parameter leaf widths, and its
return leaves its callee's return widths, for every trace alike, where every gate of C is located (`GateOk`) and `g`
resolves (`UnitResolves`). Without gates located it is false: a gate `Prog.gate` cannot locate has width 0. -/
def UnitWidths : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    (∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h) →
    ∀ (g k m : ℕ), Spot.BlockFront.unitAt P g = .ok (k, m) → UnitResolves P g →
      (unitArgs P g).toList.map (leafWidth P) = leafWidths (P.defs[(P.child P.rootScope k m).d]!) ∧
        (unitOuts P g).toList.map (leafWidth P) = (P.defs[(P.child P.rootScope k m).d]!).ret.widths.toList

/-- **`UnitWordComputes`** (open): in a trace of C whose every computed gate of unit `g` is its operation on the trace,
`g`'s return leaves' words are its callee's value on its arguments' words (`wordEval`), where C's gates
are located, `g` resolves, as `UnitWidths` reads, and each Boolean Definition the nodes of a composite reached from its
callee lead to evaluates through gates alone (`GateBodiesDeep`). -/
def UnitWordComputes : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → (P.defs[P.root]!).body.recut = #[] →
    (∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h) →
    ∀ (g k m : ℕ), Spot.BlockFront.unitAt P g = .ok (k, m) → UnitResolves P g →
      GateBodiesDeep P (P.child P.rootScope k m).d →
      ∀ Y : (Flock.OfProgram.ofProgram P).Trace,
      (∀ h : Fin (Flock.OfProgram.ofProgram P).N, InUnit P k m h.val → Flock.OfProgram.inputAt P h = false →
        Y h = (Flock.OfProgram.ofProgram P).op h Y) →
        outWords Y g = wordEval P (P.child P.rootScope k m).d (argWords Y g)

/-- **`LiftWordEval`** (open): the lift to bits (`Flock.Lift.lift`) of the template `admitted` gives, a Definition of C
taken in the block's Program, evaluated on bits (`Prog.evalBool`), is C's value of it on words (`wordEval`), bit for
bit, on words of its parameter leaf widths, where each Boolean Definition the nodes of a composite reached from it lead
to evaluates through gates alone (`GateBodiesDeep`). -/
def LiftWordEval : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (partition : Lean.Json) (cuts : List (String × List (String × ℕ)))
    (units : Array ℕ) (S : Prog) (a : Spot.OneStageZk.Admit) (L : Prog) (e : ℕ),
    decode bytes = .ok P → Spot.Restrict.restrict P units = .ok S → Spot.OneStageZk.admitted partition cuts S = .ok a →
    a.d < P.defs.size → Flock.Lift.lift S a.d = .ok (L, e) → GateBodiesDeep P a.d →
    ∀ x : List (List Bool), x.map List.length = leafWidths (P.defs[a.d]!) →
      Prog.evalBool L e x.flatten = (wordEval P a.d x).flatten

section UnitWidthsProof
open Std.Do
open InferenceOnlyVerifier.Properties.SecurityProofs
set_option mvcgen.warning false

/-- Root node `k` is a call or a batch, and `m` is one of its members. -/
def UnitShape (P : Prog) (k m : ℕ) : Prop :=
  (((P.defs[P.root]!).body.nodes[k]!).form = Form.call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = Form.batch) ∧
    m < ((P.defs[P.root]!).body.nodes[k]!).members

theorem unitAt_triple (P : Prog) (u : ℕ) :
    ⦃⌜True⌝⦄ Spot.BlockFront.unitAt P u ⦃⇓? r => ⌜UnitShape P r.1 r.2⌝⦄ := by
  unfold Spot.BlockFront.unitAt
  mvcgen [exc_throw_spec, -Spec.throw_Except] invariants
  · Invariant.withEarlyReturnNewDo (onContinue := fun _ at_ => ⌜at_ ≤ u⌝)
      (onReturn := fun (r : ℕ × ℕ) _ => ⌜UnitShape P r.1 r.2⌝) (onExcept := ExceptConds.true)
  case vc1.step.h_1.isTrue =>
    rename_i pref cur suff h2 b at_ x jp hf h1 h
    have hb : b.snd ≤ u := by rcases h1 with ⟨_, h⟩ | ⟨_, _, h, _⟩; exact h; cases h
    have e : at_ = b.snd := rfl
    have hm : Node.members x = 1 := by unfold Node.members; rw [hf]; rfl
    refine Or.inr ⟨_, rfl, trivial, Or.inl hf, ?_⟩
    show u - at_ < Node.members x
    omega
  case vc3.step.h_2.isTrue =>
    rename_i pref cur suff h2 b at_ x jp hf h1 h
    have hb : b.snd ≤ u := by rcases h1 with ⟨_, h⟩ | ⟨_, _, h, _⟩; exact h; cases h
    have e : at_ = b.snd := rfl
    have hm : Node.members x = x.n := by unfold Node.members; rw [hf]; rfl
    refine Or.inr ⟨_, rfl, trivial, Or.inr hf, ?_⟩
    show u - at_ < Node.members x
    omega
  case vc2.step.h_1.isFalse =>
    rename_i pref cur suff h2 b at_ x jp hf h1 h
    have e : at_ = b.snd := rfl
    exact Or.inl ⟨trivial, by omega⟩
  case vc4.step.h_2.isFalse =>
    rename_i pref cur suff h2 b at_ x jp hf h1 h
    have e : at_ = b.snd := rfl
    exact Or.inl ⟨trivial, by omega⟩
  case vc6.pre => exact Or.inl ⟨trivial, Nat.zero_le u⟩
  case vc7.post.success.h_1 =>
    rename_i r a hr h
    rcases h with ⟨h, _⟩ | ⟨a', h', _, hs⟩
    · rw [hr] at h; cases h
    · rw [hr] at h'; cases h'; exact hs

/-- A unit `unitAt` locates is a member of a call or a batch of the root. -/
theorem unitAt_shape {P : Prog} {u k m : ℕ} (h : Spot.BlockFront.unitAt P u = .ok (k, m)) : UnitShape P k m := by
  have hT := unitAt_triple P u
  revert hT
  generalize Spot.BlockFront.unitAt P u = x at h ⊢
  intro hT
  subst h
  exact hT True.intro

theorem forall₂_zip_map {α : Type} (f : α → ℕ → ℕ) {l : List α} {ws : List ℕ}
    (h : List.Forall₂ (fun a w => f a w = w) l ws) : (l.zip ws).map (fun p => f p.1 p.2) = ws := by
  induction h with
  | nil => rfl
  | cons h _ ih => simp only [List.zip_cons_cons, List.map_cons, h, ih]

theorem forall₂_map_eq {α β : Type} (f : α → ℕ) (g : β → ℕ) {l : List α} {r : List β}
    (h : List.Forall₂ (fun a b => g b = f a) l r) : r.map g = l.map f := by
  induction h with
  | nil => rfl
  | cons h _ ih => simp only [List.map_cons, h, ih]

theorem mapM_forall₂ {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α), (∀ a ∈ l, ∃ b, f a = .ok b) →
      ∃ r, l.mapM f = .ok r ∧ List.Forall₂ (fun a b => f a = .ok b) l r := by
  intro l
  induction l with
  | nil => intro _; exact ⟨[], rfl, .nil⟩
  | cons a l ih =>
    intro hf
    obtain ⟨b, hb⟩ := hf a (List.mem_cons_self ..)
    obtain ⟨r, hr, h2⟩ := ih fun a ha => hf a (List.mem_cons_of_mem _ ha)
    refine ⟨b :: r, ?_, .cons hb h2⟩
    rw [List.mapM_cons, hb, hr]
    rfl

theorem forall₂_mem {α β : Type} {R : α → β → Prop} {l : List α} {r : List β} (h : List.Forall₂ R l r) :
    List.Forall₂ (fun a b => a ∈ l ∧ R a b) l r := by
  induction h with
  | nil => exact .nil
  | cons hab _ ih =>
    exact .cons ⟨List.mem_cons_self .., hab⟩ (ih.imp fun _ _ ⟨hm, h⟩ => ⟨List.mem_cons_of_mem _ hm, h⟩)

theorem flatten_widths (ps : Array (String × Ty)) :
    ((ps.map fun p => p.2.widths).flatten).toList = ps.toList.flatMap fun p => p.2.widths.toList := by
  simp only [Array.toList_flatten, Array.toList_map, List.map_map]
  rfl

theorem operandsW_ok {P : Prog} {x : Node} {m : ℕ} {ops : Array (ℕ × ℕ)}
    (h : Spot.BlockFront.operandsW P x m = .ok ops) :
    ∃ raw, operands P P.rootScope x m = .ok raw ∧
      ops = raw.zip (((P.defs[x.fn]!).params.map fun p => p.2.widths).flatten) := by
  unfold Spot.BlockFront.operandsW at h
  cases hop : operands P P.rootScope x m with
  | error e => simp [hop, bind, Except.bind] at h
  | ok raw =>
    refine ⟨raw, rfl, ?_⟩
    simp [hop, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    split at h
    · first | exact (Except.ok.inj h).symm | exact h.symm
    · cases h

/-- The width `leafWidth` gives a gate of C `Prog.gate` locates at a primitive returning `Value<w>`. -/
theorem leafWidth_of_gate {P : Prog} (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P,
      InferenceOnlyVerifier.Properties.GateOk P h) (q w : ℕ)
    (hq : q < (Flock.OfProgram.ofProgram P).N → ∀ p o s, P.gate q = .ok (p, o, s) → (P.defs[p]!).ret = .value w) :
    leafWidth P { gate := q, width := w } = w := by
  unfold leafWidth
  split
  · rename_i hlt
    obtain ⟨p, o, w', ⟨⟨s, hg⟩, -, -⟩, -⟩ := hG q hlt
    show Flock.OfProgram.widthAt P q = w
    simp [Flock.OfProgram.widthAt, Flock.OfProgram.gateAt, hg, hq hlt p o s hg]
  · rfl

/-- **`UnitWidths`, the arguments**: each operand of unit `g` that `operandsW` gives is as wide in C as its callee's
parameter leaf. -/
theorem unitArgs_widths {bytes : ByteArray} {P : Prog} (hd : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[])
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    {g k m : ℕ} (hu : Spot.BlockFront.unitAt P g = .ok (k, m)) (hR : UnitResolves P g) :
    (unitArgs P g).toList.map (leafWidth P) = leafWidths (P.defs[(P.child P.rootScope k m).d]!) := by
  obtain ⟨k', m', x, hu', hx0, ⟨ops, hops⟩, -⟩ := hR
  rw [hu] at hu'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hu'
  obtain ⟨rfl, rfl⟩ := hu'
  have hx : (P.defs[P.root]!).body.nodes[k]? = some x := by rw [Array.getElem!_eq_getD]; exact hx0
  have hk : k < (P.defs[P.root]!).body.nodes.size := (Array.getElem?_eq_some_iff.1 hx).1
  have hxk : (P.defs[P.root]!).body.nodes[k]! = x := by
    rw [getElem!_pos _ k hk]; exact (Array.getElem?_eq_some_iff.1 hx).2
  have hx0' : (P.defs[P.root]?.getD default).body.nodes[k]? = some x := by
    rw [← Array.getD_eq_getD_getElem?]; exact hx0
  have hua : unitArgs P g = ops.map fun (q, w) => ({ gate := q, width := w } : Spot.BlockFront.Port) := by
    simp [unitArgs, hu, hx0', hops, bind, Except.bind]
  obtain ⟨raw, hraw, rfl⟩ := operandsW_ok hops
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_full bytes P hd hr
  have hD : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hrt, hrp, -⟩ := BodiesOk.decode_root bytes P hd hr
  have hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false := ⟨hrt, hrp⟩
  have hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc _ => BodiesOk.decoded_bodyOk bytes P hd hr sc.d
  have hL := OperandWidths.decoded_laid bytes P hd hr
  have hW := LeafWidths.decoded_wide bytes P hd hr
  obtain ⟨hsh, hmem⟩ := unitAt_shape hu
  rw [hxk] at hsh hmem
  have hxk' : (P.defs[P.rootScope.d]!).body.nodes[k]! = x := hxk
  have hns : x.form ≠ .scan := by rcases hsh with h | h <;> rw [h] <;> exact fun h => nomatch h
  have hw := OperandWidths.node_operand_widths P hroot hB hL hD hW P.rootScope ⟨rfl, rfl, rfl⟩ k hk m
    (by rw [hxk']; exact hmem) (by rw [hxk']; exact hns) raw (by rw [hxk']; exact hraw)
  rw [hxk'] at hw
  rw [hua, child_d, hxk]
  unfold leafWidths
  rw [← flatten_widths] at hw ⊢
  simp only [Array.toList_map, Array.toList_zip, List.map_map]
  exact forall₂_zip_map (fun q w => leafWidth P { gate := q, width := w })
    (hw.imp fun a w haw => leafWidth_of_gate hG a w fun _ => haw)

/-- **`UnitWidths`, the return leaves**: each return leaf of unit `g` resolves to a gate of C as wide as its callee's
return leaf. -/
theorem unitOuts_widths {bytes : ByteArray} {P : Prog} (hd : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[])
    (hG : ∀ h < InferenceOnlyVerifier.Properties.gates P, InferenceOnlyVerifier.Properties.GateOk P h)
    {g k m : ℕ} (hu : Spot.BlockFront.unitAt P g = .ok (k, m)) (hR : UnitResolves P g) :
    (unitOuts P g).toList.map (leafWidth P) = (P.defs[(P.child P.rootScope k m).d]!).ret.widths.toList := by
  obtain ⟨k', m', x, hu', hx0, -, hres⟩ := hR
  rw [hu] at hu'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hu'
  obtain ⟨rfl, rfl⟩ := hu'
  have hx : (P.defs[P.root]!).body.nodes[k]? = some x := by rw [Array.getElem!_eq_getD]; exact hx0
  have hk : k < (P.defs[P.root]!).body.nodes.size := (Array.getElem?_eq_some_iff.1 hx).1
  have hxk : (P.defs[P.root]!).body.nodes[k]! = x := by
    rw [getElem!_pos _ k hk]; exact (Array.getElem?_eq_some_iff.1 hx).2
  have hc : P.defs.getD x.fn default = P.defs[x.fn]! := Array.getElem!_eq_getD.symm
  rw [hc] at hres
  rw [child_d, hxk]
  generalize hcd : P.defs[x.fn]! = c at hres
  let F : ℕ → V Spot.BlockFront.Port := fun l => do
    let q ← resolve P P.rootScope { p := false, i := k, j := m * c.ret.leaves + l }
    pure ({ gate := q, width := c.ret.widths.getD l 0 } : Spot.BlockFront.Port)
  obtain ⟨r, hrm, h2⟩ := mapM_forall₂ F (List.range c.ret.leaves) fun l hl => by
    obtain ⟨q, hq⟩ := hres l (List.mem_range.1 hl)
    exact ⟨_, by simp only [F, hq]; rfl⟩
  have huo : unitOuts P g = r.toArray := by
    unfold unitOuts
    simp only [hu, hx0, hc, hcd, bind, Except.bind]
    rw [Array.mapM_eq_mapM_toList, Array.toList_range]
    change (match (Functor.map List.toArray (List.mapM F (List.range c.ret.leaves)) : V _) with
      | .ok ps => ps | .error _ => #[]) = _
    rw [hrm]
    rfl
  rw [huo, List.toList_toArray]
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_full bytes P hd hr
  have hD : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hrt, hrp, -⟩ := BodiesOk.decode_root bytes P hd hr
  have hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false := ⟨hrt, hrp⟩
  have hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d := fun sc _ => BodiesOk.decoded_bodyOk bytes P hd hr sc.d
  have hL := OperandWidths.decoded_laid bytes P hd hr
  have hW := LeafWidths.decoded_wide bytes P hd hr
  obtain ⟨hsh, hmem⟩ := unitAt_shape hu
  rw [hxk] at hsh hmem
  obtain ⟨-, hnd, -⟩ := (hD P.root hrt).2.1 k hk
  obtain ⟨-, -, -, -, hshape, -⟩ := hnd
  rw [hxk, hcd] at hshape
  have hxk' : (P.defs[P.rootScope.d]!).body.nodes[k]! = x := hxk
  have hwsz : c.ret.widths.size = c.ret.leaves := by rw [← Array.length_toList, Flock.Sigma.widths_length]
  have hrw : c.ret.widths.toList = (List.range c.ret.leaves).map fun l => c.ret.widths.getD l 0 := by
    rw [← hwsz]
    apply List.ext_getElem (by simp)
    intro t h1 _
    simp only [Array.length_toList] at h1
    simp [Array.getD, h1]
  rw [hrw]
  refine forall₂_map_eq _ _ ((forall₂_mem h2).imp fun l b hlF => ?_)
  obtain ⟨hl, hF⟩ := hlF
  have hl := List.mem_range.1 hl
  cases hq : resolve P P.rootScope { p := false, i := k, j := m * c.ret.leaves + l } with
  | error e => simp [F, hq, bind, Except.bind] at hF
  | ok q =>
    simp only [F, hq, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hF
    subst hF
    have hty : LeafWidths.tyOf P P.rootScope { p := false, i := k, j := m * c.ret.leaves + l } = x.out := by
      simp only [LeafWidths.tyOf, Bool.false_eq_true, ↓reduceIte]
      rw [← hxk]
      rfl
    have hmod : (m * c.ret.leaves + l) % c.ret.leaves = l := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]
    have hfin : ∀ w, Widths.wAt x.out (m * c.ret.leaves + l) = w → w = c.ret.widths.getD l 0 →
        leafWidth P { gate := q, width := c.ret.widths.getD l 0 } = c.ret.widths.getD l 0 := by
      intro w hwa hw
      exact leafWidth_of_gate hG q _ fun _ p o s hg => by
        have hRef : Resolve.RefOk P P.rootScope.d (P.defs[P.rootScope.d]!).body.nodes.size
            { p := false, i := k, j := m * c.ret.leaves + l } := by
          refine ⟨fun h => absurd h Bool.false_ne_true, fun _ => ⟨hk, ?_⟩⟩
          show m * c.ret.leaves + l < ((P.defs[P.rootScope.d]!).body.nodes[k]!).out.leaves
          rw [hxk']
          rcases hsh with hf | hf
          · simp only [hf, BodiesOk.Shape] at hshape
            have h1 : x.members = 1 := by unfold Node.members; rw [hf]; rfl
            rw [hshape.1]
            have : m = 0 := by omega
            subst this
            omega
          · simp only [hf, BodiesOk.Shape] at hshape
            have h1 : x.members = x.n := by unfold Node.members; rw [hf]; rfl
            rw [hshape.1, BodiesOk.leaves_array]
            have := Nat.mul_le_mul_right c.ret.leaves (show m + 1 ≤ x.n by omega)
            rw [Nat.add_mul, Nat.one_mul] at this
            omega
        rw [LeafWidths.resolve_wide P hroot hB hL hD hW P.rootScope ⟨rfl, rfl, rfl⟩ _ hRef q hq p o s hg, hty, hwa, hw]
    rcases hsh with hf | hf
    · simp only [hf, BodiesOk.Shape] at hshape
      have h1 : x.members = 1 := by unfold Node.members; rw [hf]; rfl
      have h0 : m = 0 := by omega
      refine hfin _ rfl ?_
      rw [hshape.1, h0, Nat.zero_mul, Nat.zero_add, LeafWidths.wAt_widths _ _ hl, Array.getElem!_eq_getD]
      rfl
    · simp only [hf, BodiesOk.Shape] at hshape
      refine hfin _ rfl ?_
      rw [hshape.1, LeafWidths.wAt_array _ _ _ (by omega), hmod, LeafWidths.wAt_widths _ _ hl,
        Array.getElem!_eq_getD]
      rfl

/-- **The gate of a member of a primitive node inside a unit**: in member `m` of root node `k`, which calls a
composite, the gate of member `m'` of its node `k'`, which calls a primitive of one gate, is located at that node, in
the unit's activation, with the member's operands. -/
theorem unit_gate_at {P : Prog} (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    {k m : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size) (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    {k' m' : ℕ} (hk' : k' < (P.defs[(P.child P.rootScope k m).d]!).body.nodes.size)
    (hm' : m' < ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[k']!).members)
    (hp' : (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[k']!).fn]!).prim = true)
    (hg1 : (P.defs[((P.defs[(P.child P.rootScope k m).d]!).body.nodes[k']!).fn]!).gates = 1)
    {p : ℕ} {ops : Array ℕ} {sc' : Scope}
    (h : P.gate ((P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[k']! + m') =
      .ok (p, ops, sc')) :
    p = ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[k']!).fn ∧ sc' = P.child P.rootScope k m ∧
      operands P (P.child P.rootScope k m) ((P.defs[(P.child P.rootScope k m).d]!).body.nodes[k']!) m' = .ok ops := by
  have hgood : Resolve.Good P (P.child P.rootScope k m) :=
    Resolve.good_child P (parent := P.rootScope) ⟨rfl, rfl, rfl⟩ hk rfl hc hm rfl
  obtain ⟨hd, hpc⟩ := Resolve.good_d P hroot hB _ hgood
  obtain ⟨-, hK, hnodes⟩ := hL _ hd hpc
  have hstep := hnodes k' hk'
  rw [hg1, Nat.mul_one] at hstep
  have hend := Resolve.offsets_mono P _ (hL _ hd) hpc (k' + 1) _ (by omega) (Nat.le_refl _)
  have hoff : (P.child P.rootScope k m).off = P.rootScope.off + (P.defs[P.rootScope.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[P.rootScope.d]!).body.nodes[k]!).fn]!).gates := rfl
  have hgd : (P.defs[(P.child P.rootScope k m).d]!).gates =
      (P.defs[((P.defs[P.rootScope.d]!).body.nodes[k]!).fn]!).gates := rfl
  rw [Prog.gate_of (Prog.gate_recut h)] at h
  by_cases hge : (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).body.offsets[k']! + m' ≥
      (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  obtain ⟨n, hn⟩ : ∃ n, Flock.Steps.FUEL = n + 2 := ⟨Flock.Steps.FUEL - 2, by unfold Flock.Steps.FUEL; omega⟩
  rw [hn, Flock.Steps.run.eq_2] at h
  rcases hs1 : gateStep P ((P.child P.rootScope k m).off +
      (P.defs[(P.child P.rootScope k m).d]!).body.offsets[k']! + m') P.rootScope with e | x
  · simp [hs1, bind, Except.bind] at h
  have hx := Resolve.gateStep_member P hL P.rootScope hroot.1 hroot.2 hk hm hc (by omega) (by omega) hs1
  subst hx
  simp only [hs1, bind, Except.bind] at h
  rw [Flock.Steps.run.eq_2] at h
  rcases hs2 : gateStep P ((P.child P.rootScope k m).off +
      (P.defs[(P.child P.rootScope k m).d]!).body.offsets[k']! + m') (P.child P.rootScope k m) with e | x
  · simp [hs2, bind, Except.bind] at h
  obtain ⟨ops', rfl, hops⟩ := Resolve.gateStep_at_prim P hL (P.child P.rootScope k m) hd hpc hk' hm' hp'
    (by rw [hg1, Nat.mul_one]) (by rw [hg1, Nat.mul_one]; omega) hs2
  simp only [hs2, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl, rfl⟩ := h
  exact ⟨rfl, rfl, hops⟩

/-- A step of resolution to another reference keeps the gate it resolves to. -/
theorem resolve_step_up {P : Prog} (hrc : (P.defs[P.root]!).body.recut = #[]) {sc sc' : Scope} {r r' : Ref}
    {g : ℕ} (hs : resolveStep P (sc, r) = .ok (.inr (sc', r'))) (h : resolve P sc r = .ok g) :
    resolve P sc' r' = .ok g := by
  rw [Flock.Program.resolve_eq hrc] at h ⊢
  obtain ⟨n, hn⟩ : ∃ n, Flock.Steps.FUEL = n + 1 := ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩
  rw [hn, Flock.Steps.run.eq_2] at h
  rw [hn]
  simp only [hs, bind, Except.bind] at h
  exact Flock.Steps.run_succ h

/-- A step of resolution that ends at a gate is the resolution. -/
theorem resolve_step_done {P : Prog} (hrc : (P.defs[P.root]!).body.recut = #[]) {sc : Scope} {r : Ref} {g : ℕ}
    (hs : resolveStep P (sc, r) = .ok (.inl g)) : resolve P sc r = .ok g := by
  rw [Flock.Program.resolve_eq hrc]
  obtain ⟨n, hn⟩ : ∃ n, Flock.Steps.FUEL = n + 1 := ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩
  rw [hn, Flock.Steps.run.eq_2]
  simp only [hs, bind, Except.bind]
  rfl

/-- **A unit's parameter leaf** resolves to what its root node's argument reference does: leaf `j` of argument `i` of
member `m`, the member's reference for a batch and the sequence's for a call. -/
theorem unit_param_resolve {P : Prog} (hrc : (P.defs[P.root]!).body.recut = #[]) {k m i j g : ℕ}
    (hf : ((P.defs[P.root]!).body.nodes[k]!).form = .call ∨ ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    (h : resolve P (P.child P.rootScope k m) ⟨true, i, j⟩ = .ok g) :
    resolve P P.rootScope (OperandWidths.refOf ((P.defs[P.root]!).body.nodes[k]!) m i j) = .ok g := by
  refine resolve_step_up hrc ?_ h
  rcases hf with hf | hf
  · have hb : (((P.defs[P.root]!).body.nodes[k]!).form == .batch) = false := by rw [hf]; rfl
    simp only [OperandWidths.refOf, hb, Bool.false_eq_true, ite_false]
    exact Resolve.rs_up P P.rootScope _ _ k m i j (Or.inr hf)
  · have hb : (((P.defs[P.root]!).body.nodes[k]!).form == .batch) = true := by rw [hf]; rfl
    simp only [OperandWidths.refOf, hb, ite_true]
    exact Resolve.rs_up_batch P P.rootScope _ _ k m i j hf

/-- A leaf of a primitive node resolves to the node's gate. -/
theorem resolve_prim_node {P : Prog} (hrc : (P.defs[P.root]!).body.recut = #[]) (sc : Scope) {k j : ℕ}
    (hf : ((P.defs[sc.d]!).body.nodes[k]!).form = .prim) :
    resolve P sc ⟨false, k, j⟩ = .ok (sc.off + (P.defs[sc.d]!).body.offsets[k]!) :=
  resolve_step_done hrc (Resolve.rs_prim P sc k j hf)

/-- Leaf `j` of a batch node of a primitive of one return leaf resolves to the gate of member `j`. -/
theorem resolve_batch_prim {P : Prog} (hrc : (P.defs[P.root]!).body.recut = #[]) (sc : Scope) {k j : ℕ}
    (hf : ((P.defs[sc.d]!).body.nodes[k]!).form = .batch)
    (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (h1 : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).outLeaves = 1) :
    resolve P sc ⟨false, k, j⟩ = .ok (sc.off + (P.defs[sc.d]!).body.offsets[k]! + j) := by
  refine resolve_step_done hrc ?_
  rw [Resolve.rs_batch P sc k j hf]
  simp only [hp, h1, Nat.div_one, ite_true]

/-- **A unit's return leaf** resolves to what leaf `l` of its Definition's return does in the unit's activation. -/
theorem unit_ret_resolve {P : Prog} (hrc : (P.defs[P.root]!).body.recut = #[]) {k m l q : ℕ}
    (hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false)
    (hf : (((P.defs[P.root]!).body.nodes[k]!).form = .call ∧ m = 0) ∨
      ((P.defs[P.root]!).body.nodes[k]!).form = .batch)
    (hl : l < (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves)
    (h : resolve P P.rootScope
      ⟨false, k, m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves + l⟩ = .ok q) :
    resolve P (P.child P.rootScope k m) ((P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).body.ret.get l) =
      .ok q := by
  refine resolve_step_up hrc ?_ h
  rcases hf with ⟨hf, rfl⟩ | hf
  · rw [Resolve.rs_call P P.rootScope k _ hf, Nat.zero_mul, Nat.zero_add]
    simp only [Flock.Program.Prog.child, Nat.zero_mul, Nat.add_zero]
    try rfl
  · have hdiv : (m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves + l) /
        (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves = m := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hl, Nat.zero_add]
    have hmod : (m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves + l) %
        (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves = l := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]
    have hc' : (P.defs[((P.defs[P.rootScope.d]!).body.nodes[k]!).fn]!).prim = false := hc
    have hdiv' : (m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves + l) /
        (P.defs[((P.defs[P.rootScope.d]!).body.nodes[k]!).fn]!).outLeaves = m := hdiv
    have hmod' : (m * (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).outLeaves + l) %
        (P.defs[((P.defs[P.rootScope.d]!).body.nodes[k]!).fn]!).outLeaves = l := hmod
    rw [Resolve.rs_batch P P.rootScope k _ hf]
    simp only [hc', Bool.false_eq_true, ite_false]
    rw [hdiv', hmod']
    rfl

theorem ofFn_getD_eq {n : ℕ} (l : List Bool) (h : l.length = n) :
    List.ofFn (fun i : Fin n => l.getD i false) = l := by
  subst h
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2]

theorem portWord_lt {P : Prog} (Y : (Flock.OfProgram.ofProgram P).Trace) {q w : ℕ}
    (hq : q < (Flock.OfProgram.ofProgram P).N) :
    portWord Y { gate := q, width := w } =
      List.ofFn fun i : Fin ((Flock.OfProgram.ofProgram P).w ⟨q, hq⟩) => (Y ⟨q, hq⟩).getLsbD i := by
  simp [portWord, portBits, hq]

/-- The bits a computed gate's operation reads, where its operands are all below it: its operands' words in order. -/
theorem opBits_eq {P : Prog} (Y : (Flock.OfProgram.ofProgram P).Trace) {G : ℕ} (hG : G < (Flock.OfProgram.ofProgram P).N)
    {p : ℕ} {ops : Array ℕ} (hga : Flock.OfProgram.gateAt P G = some (p, ops))
    (hin : Flock.OfProgram.inputAt P G = false) (hlt : ∀ a ∈ ops.toList, a < G) :
    Flock.OfProgram.opBits (Flock.OfProgram.opsBelow P ⟨G, hG⟩) Y =
      (ops.toList.map fun a => portWord Y { gate := a, width := 0 }).flatten := by
  unfold Flock.OfProgram.opsBelow
  simp only [hin, Bool.false_eq_true, ite_false, hga]
  generalize ops.toList = l at hlt ⊢
  induction l with
  | nil => rfl
  | cons a l ih =>
    have ha : a < G := hlt a (List.mem_cons_self ..)
    rw [List.filterMap_cons, dif_pos ha]
    simp only [Flock.OfProgram.opBits, List.flatMap_cons, List.map_cons, List.flatten_cons] at ih ⊢
    rw [ih fun b hb => hlt b (List.mem_cons_of_mem _ hb), portWord_lt Y (Nat.lt_trans ha hG)]
    rfl

/-- **A computed gate's word** is its primitive's value on its operands' words, where its operands are below it and
the value is as wide as the gate. -/
theorem word_at_computed {P : Prog} (Y : (Flock.OfProgram.ofProgram P).Trace) {G w : ℕ}
    (hG : G < (Flock.OfProgram.ofProgram P).N) {p : ℕ} {ops : Array ℕ}
    (hga : Flock.OfProgram.gateAt P G = some (p, ops)) (hin : Flock.OfProgram.inputAt P G = false)
    (hlt : ∀ a ∈ ops.toList, a < G) (hY : Y ⟨G, hG⟩ = (Flock.OfProgram.ofProgram P).op ⟨G, hG⟩ Y)
    (hlen : (Prog.evalBool P p (ops.toList.map fun a => portWord Y { gate := a, width := 0 }).flatten).length =
      Flock.OfProgram.widthAt P G) :
    portWord Y { gate := G, width := w } =
      Prog.evalBool P p (ops.toList.map fun a => portWord Y { gate := a, width := 0 }).flatten := by
  have hop : (Flock.OfProgram.ofProgram P).op ⟨G, hG⟩ Y = Verity.ofBits (fun i => (Prog.evalBool P p
      (Flock.OfProgram.opBits (Flock.OfProgram.opsBelow P ⟨G, hG⟩) Y)).getD i false) := by
    simp only [Flock.OfProgram.ofProgram, Flock.OfProgram.ofProgramWith, hga]
  rw [portWord_lt Y hG, hY, hop]
  simp only [Verity.getLsbD_ofBits]
  rw [opBits_eq Y hG hga hin hlt]
  exact ofFn_getD_eq _ hlen

end UnitWidthsProof

end Flock.Chain
