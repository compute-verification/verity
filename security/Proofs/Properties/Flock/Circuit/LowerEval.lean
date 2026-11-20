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
      (∀ r ∈ a.fields, (r.2 == 0 || r.2 % 16 != 0) = false) := by
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
  obtain ⟨⟨⟩, -, h⟩ := Flock.HmNets.bind_ok h
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
  refine ⟨_, _, _, _, by assumption, by assumption, by assumption, none_of_forall (by assumption), rfl, rfl, ?_, ?_⟩
  · intro r hr
    have := Array.find?_eq_none.1 (none_of_forall2 (by assumption)) r hr
    simpa using this
  · intro r hr
    have := Array.find?_eq_none.1 (none_of_forall2 (by assumption)) r hr
    simpa using this


end Flock.Chain
