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
  intro partition cuts S a t n Q e ha ht hq hu hU hb z hs hone
  obtain ⟨h1, h2⟩ := lowerEval hq ht hu hU hb hs hone
  obtain ⟨hw1, hw2⟩ := hW ha
  obtain ⟨hb1, hb2⟩ := hL ha hq
  simp only [doorCol_whole hw1, doorCol_whole hw2, hb1, hb2]
  exact ⟨h1, h2⟩

end Flock.Chain
