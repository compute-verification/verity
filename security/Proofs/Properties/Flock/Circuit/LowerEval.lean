module

public import Proofs.Properties.Flock.Circuit.LowerAsm
public import Proofs.Properties.Flock.Circuit.Chain

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

/-- `admitted`'s `let #[key] := keys` at a one-key array, by `delta`: Lean generates no equations for an array-literal
pattern. -/
theorem match11_one {β : Sort _} (key : String) (h1 : String → β) (h2 : Array String → β) :
    Spot.OneStageZk.admitted.match_11 (fun _ => β) ⟨[key]⟩ h1 h2 = h1 key := by
  delta Spot.OneStageZk.admitted.match_11; rfl

/-- `admitted`'s `let #[key] := keys` at any other array. -/
theorem match11_ne {β : Sort _} (x : Array String) (h1 : String → β) (h2 : Array String → β) (hx : x.size ≠ 1) :
    Spot.OneStageZk.admitted.match_11 (fun _ => β) x h1 h2 = h2 x := by
  delta Spot.OneStageZk.admitted.match_11; rw [dif_neg hx]

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
        ¬ decide (((a.ins.map (·.1) ++ a.fields.map (·.1)).filter (· == n)).size > 1) = true) := by
  unfold Spot.OneStageZk.admitted at h
  split at h
  na_close
  split at h
  na_close
  obtain ⟨called, -, h⟩ := Flock.HmNets.bind_ok h
  try dsimp only at h
  generalize Array.qsort called (fun x1 x2 => decide (x1 < x2)) = keys at h
  rcases keys with ⟨_ | ⟨key, _ | ⟨k2, rest⟩⟩⟩
  · rw [match11_ne _ _ _ (by decide)] at h
    exact absurd h notAdmitted_ne
  swap
  · rw [match11_ne _ _ _ (by simp)] at h
    exact absurd h notAdmitted_ne
  rw [match11_one] at h
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
    forIn_unit_all hc fun x s hs => ?_, ?_⟩
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
  obtain ⟨_, _, _, _, -, -, -, -, -, -, hi, hf, -, -⟩ := admitted_walk ha
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

/-- **`AdmitLeaves`**, from `admitted_walk` and `HomoBits`. -/
theorem admitLeaves (hH : HomoBits) : AdmitLeaves := by
  intro partition cuts S a Q e ha hq
  obtain ⟨fn, L, dl, l, -, hlift, hl, hoff, hins, hfields, -, -, hcut, hnames⟩ := admitted_walk ha
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
          first | omega | exact ⟨0, by omega, by first | rfl | simp⟩
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
      rw [hH _ _ hw]
      exact Dvd.dvd.mul_right (Nat.dvd_of_mod_eq_zero hk') w
  · rw [hfields, bitsOf_fields, hr]

/-- **`LowerEvalDoor`**, on `HomoBits`. -/
theorem lowerEvalDoor (hH : HomoBits) : LowerEvalDoor :=
  lowerEvalDoor_of admitWhole (admitLeaves hH)

end Flock.Chain
