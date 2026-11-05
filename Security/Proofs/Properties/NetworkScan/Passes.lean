import NetworkCheck.Circuit
import Proofs.Properties.Nci.BodiesOk.Order
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# When `scanProg`'s checks over hash sets pass

Two of `scanProg`'s checks fold a hash set, which no proof evaluates (a hash set indexes through `USize`, whose width the
kernel doesn't know). What makes each pass, stated so a proof decides it:

* `keysUnique_of_nodup`: `keysUnique` accepts a Program whose Definitions' keys are distinct, the converse of
  `Properties.NetworkPeriod.keysUnique_nodup`.
* `orderViolation_none`: `Prog.orderViolation` finds nothing in a Program whose composites' nodes read only parameters
  and earlier nodes (`ordered`, a `Bool`), when its walk fits the fuel: the root, then each composite's nodes once
  (`walkW`, summed by `sum_walkW_le`). It is the converse of `BodiesOk.order_closed`, over the same loops: each inner one
  keeps its verdict empty element by element (`forIn_mem_none`), and the walk reads one queue entry a step
  (`WalkInv`).
-/

namespace Properties.NetworkScan

open Flock.Program
open NciCheck.Properties.SecurityProofs.BodiesOk (partNames forIn_id_inv)

theorem keysUnique_foldl_of : ∀ (l : List Defn) (seen : Std.HashMap String Unit) (ok : Bool), ok = true →
    (∀ d ∈ l, seen.contains d.key = false) → (l.map (·.key)).Nodup →
    (l.foldl (fun (acc : Std.HashMap String Unit × Bool) d =>
      (acc.1.insert d.key (), acc.2 && !acc.1.contains d.key)) (seen, ok)).2 = true
  | [], _, _, hok, _, _ => hok
  | d :: l, seen, ok, hok, hl, hnd => by
    rw [List.foldl_cons]
    rw [List.map_cons, List.nodup_cons] at hnd
    refine keysUnique_foldl_of l _ _ ?_ (fun d' hd' => ?_) hnd.2
    · simp [hok, hl d (List.mem_cons.2 (Or.inl rfl))]
    · rw [Std.HashMap.contains_insert, Bool.or_eq_false_iff]
      refine ⟨beq_eq_false_iff_ne.2 fun he => hnd.1 (List.mem_map.2 ⟨d', hd', he.symm⟩),
        hl d' (List.mem_cons.2 (Or.inr hd'))⟩

/-- **`keysUnique` accepts distinct keys**: a Program whose Definitions' keys are distinct passes `keysUnique`. -/
theorem keysUnique_of_nodup {P : Prog} (h : (P.defs.toList.map (·.key)).Nodup) :
    NetworkCheck.Period.keysUnique P = true := by
  unfold NetworkCheck.Period.keysUnique
  rw [← Array.foldl_toList]
  exact keysUnique_foldl_of _ _ _ rfl (fun _ _ => by simp) h

/-- Node `k` of `body` reads only parameters and earlier nodes, as `Prog.orderViolation` checks it. -/
def nodeOrdered (body : Body) (k : ℕ) : Bool :=
  (body.nodes[k]!).args.toList.all fun a => a.seq.parts.toList.all fun q => (partNames q).toList.all fun nm =>
    if nm.1 then decide (nm.2 < body.params.size) else decide (nm.2 < k)

/-- Every composite's nodes read only parameters and earlier nodes. -/
def ordered (P : Prog) : Bool :=
  (List.range P.defs.size).all fun d => (P.defs[d]!).prim ||
    (List.range (P.defs[d]!).body.nodes.size).all fun k => nodeOrdered (P.defs[d]!).body k

/-- The entries `Prog.orderViolation` queues for Definition `d`: a composite's nodes'. -/
def walkW (P : Prog) (d : ℕ) : ℕ := if (P.defs[d]!).prim then 0 else (P.defs[d]!).body.nodes.size

theorem walkW_out (P : Prog) {d : ℕ} (hd : P.defs.size ≤ d) : walkW P d = 0 := by
  rw [walkW, getElem!_neg P.defs d (by omega)]
  rfl

/-- The Definitions any walk sees queue at most every Definition's entries. -/
theorem sum_walkW_le (P : Prog) (S : Finset ℕ) :
    ∑ d ∈ S, walkW P d ≤ ∑ d ∈ Finset.range P.defs.size, walkW P d := by
  calc ∑ d ∈ S, walkW P d = ∑ d ∈ S.filter (· < P.defs.size), walkW P d :=
        (Finset.sum_subset (Finset.filter_subset _ S) fun d hd hn =>
          walkW_out P (not_lt.1 fun h => hn (Finset.mem_filter.2 ⟨hd, h⟩))).symm
    _ ≤ ∑ d ∈ Finset.range P.defs.size, walkW P d :=
        Finset.sum_le_sum_of_subset fun d hd => Finset.mem_range.2 (Finset.mem_filter.1 hd).2

/-- A `for` loop in `Id` whose state carries an optional verdict ends with none when each element it reads, from a
verdict of none, continues with none. -/
theorem forIn_mem_none {α γ δ : Type} (f : α → Option γ × δ → Id (ForInStep (Option γ × δ))) (Q : α → Prop)
    (hf : ∀ x b, Q x → b.1 = none → match (f x b).run with | .yield b' => b'.1 = none | .done _ => False)
    (l : List α) (hl : ∀ x ∈ l, Q x) (b : Option γ × δ) (hb : b.1 = none) : (forIn l b f).run.1 = none :=
  (forIn_id_inv f (fun _ => False) l (fun _ b => b.1 = none) b hb fun i hi b hI =>
    hf l[i] b (hl _ (List.getElem_mem hi)) hI).elim id False.elim

/-- `Prog.orderViolation`'s walk so far: no verdict, the queue read up to `at_`, and the queue the root and the entries
of each Definition seen. -/
def WalkInv (P : Prog) (b : Option (Option Lean.Json) × Std.HashSet ℕ × Array ℕ × ℕ) : Prop :=
  b.1 = none ∧ b.2.2.2 ≤ b.2.2.1.size ∧
    ∃ S : Finset ℕ, (∀ d, b.2.1.contains d = true ↔ d ∈ S) ∧ b.2.2.1.size = 1 + ∑ d ∈ S, walkW P d

/-- **`Prog.orderViolation` passes an ordered Program**: one whose composites' nodes read only parameters and earlier
nodes, when the root and every composite's nodes number under the fuel. -/
theorem orderViolation_none (P : Prog) (hord : ordered P = true)
    (hfuel : 2 + ∑ d ∈ Finset.range P.defs.size, walkW P d ≤ Flock.Steps.FUEL) : P.orderViolation = none := by
  have hok : ∀ d, (P.defs[d]!).prim = false → ∀ k < (P.defs[d]!).body.nodes.size,
      nodeOrdered (P.defs[d]!).body k = true := by
    intro d hp k hk
    by_cases hd : d < P.defs.size
    · have h1 := List.all_eq_true.1 hord d (List.mem_range.2 hd)
      simp only [hp, Bool.false_or] at h1
      exact List.all_eq_true.1 h1 k (List.mem_range.2 hk)
    · rw [getElem!_neg P.defs d hd] at hk
      exact absurd hk (Nat.not_lt_zero _)
  unfold Prog.orderViolation
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  generalize hR : forIn (m := Id) (List.range' _ _ _) _ _ = R
  have key : (WalkInv P R.run ∧ R.run.2.2.2 = Flock.Steps.FUEL) ∨
      (R.run.1 = none ∧ R.run.2.2.1.size ≤ R.run.2.2.2) := by
    rw [← hR]
    refine (forIn_id_inv _ (fun b => b.1 = none ∧ b.2.2.1.size ≤ b.2.2.2) _ (fun i b => WalkInv P b ∧ b.2.2.2 = i) _
      ⟨⟨rfl, by simp, ∅, fun d => by simp, by simp⟩, rfl⟩ ?_).imp
      (fun h => ⟨h.1, by rw [h.2]; simp [Std.Legacy.Range.size]⟩) id
    rintro i - ⟨r, seen, todo, at_⟩ ⟨⟨hr, hle, S, hS, hsz⟩, hat⟩
    dsimp only at hr hle hS hsz hat ⊢
    subst hr
    by_cases hA : at_ ≥ todo.size
    · rw [ite_eq_left hA]
      exact ⟨rfl, hA⟩
    rw [ite_eq_right hA]
    have hlt : at_ < todo.size := by omega
    by_cases hB : seen.contains todo[at_]! = true
    · rw [ite_eq_left hB]
      exact ⟨⟨rfl, show at_ + 1 ≤ todo.size by omega, S, hS, hsz⟩, show at_ + 1 = i + 1 by omega⟩
    rw [ite_eq_right hB]
    have hdS : todo[at_]! ∉ S := fun h => hB ((hS _).2 h)
    have hS' : ∀ d, (seen.insert todo[at_]!).contains d = true ↔ d ∈ insert todo[at_]! S := by
      intro d
      rw [Std.HashSet.contains_insert, Bool.or_eq_true_iff, beq_iff_eq, hS d, Finset.mem_insert]
      exact or_congr eq_comm Iff.rfl
    by_cases hC : P.defs[todo[at_]!]!.prim = true
    · rw [ite_eq_left hC]
      refine ⟨⟨rfl, show at_ + 1 ≤ todo.size by omega, insert todo[at_]! S, hS', ?_⟩,
        show at_ + 1 = i + 1 by omega⟩
      rw [Finset.sum_insert hdS, walkW, ite_eq_left hC, Nat.zero_add]
      exact hsz
    rw [ite_eq_right hC]
    generalize hN : forIn (m := Id) (_ : Std.Legacy.Range) (none, todo) _ = N
    have hC' : (P.defs[todo[at_]!]!).prim = false := by simpa using hC
    generalize hd : todo[at_]! = d at hN hC' hS' hdS ⊢
    have hW : walkW P d = (P.defs[d]!).body.nodes.size := by simp [walkW, hC']
    have hokd := hok d hC'
    generalize hbody : (P.defs[d]!).body = body at hN hW hokd
    have hnode : N.run.1 = none ∧ N.run.2.size = todo.size + body.nodes.size := by
      rw [← hN, Std.Legacy.Range.forIn_eq_forIn_range']
      refine (forIn_id_inv _ (fun _ => False) _
        (fun k (b : Option (Option Lean.Json) × Array ℕ) => b.1 = none ∧ b.2.size = todo.size + k) (none, todo)
        ⟨rfl, rfl⟩ ?_).elim (fun h => ⟨h.1, by rw [h.2]; simp [Std.Legacy.Range.size]⟩) False.elim
      rintro k hk ⟨r, t⟩ ⟨hr, ht⟩
      dsimp only at hr ht
      subst hr
      simp only [List.length_range', Std.Legacy.Range.size] at hk
      simp (config := { zeta := false }) only [List.getElem_range', Nat.one_mul, Nat.zero_add]
      have hk' : k < body.nodes.size := by simpa [Std.Legacy.Range.size] using hk
      generalize hA : forIn (m := Id) (_ : Std.Legacy.Range) (none, ()) _ = A
      have hA1 : A.run.1 = none := by
        rw [← hA, Std.Legacy.Range.forIn_eq_forIn_range']
        refine (forIn_id_inv _ (fun _ => False) _ (fun _ (b : Option (Option Lean.Json) × Unit) => b.1 = none)
          (none, ()) rfl ?_).elim id False.elim
        rintro a ha ⟨r, u⟩ hr
        dsimp only at hr
        subst hr
        simp only [List.length_range', Std.Legacy.Range.size] at ha
        simp (config := { zeta := false }) only [List.getElem_range', Nat.one_mul, Nat.zero_add]
        have ha' : a < (body.nodes[k]!).args.size := by simpa [Std.Legacy.Range.size] using ha
        have hargs : (((body.nodes[k]!).args[a]!).seq.parts.toList.all fun q => (partNames q).toList.all fun nm =>
            if nm.1 then decide (nm.2 < body.params.size) else decide (nm.2 < k)) = true := by
          refine List.all_eq_true.1 (hokd k hk') _ ?_
          rw [getElem!_pos _ a ha']
          exact Array.mem_def.1 (Array.getElem_mem ha')
        generalize hQ : forIn (m := Id) (_ : Array Part) (none, ()) _ = Q
        have hQ1 : Q.run.1 = none := by
          rw [← hQ, ← Array.forIn_toList]
          refine forIn_mem_none _ (fun q => ((partNames q).toList.all fun nm =>
            if nm.1 then decide (nm.2 < body.params.size) else decide (nm.2 < k)) = true) ?_ _
            (List.all_eq_true.1 hargs) _ rfl
          rintro q ⟨r, u⟩ hq hr
          dsimp only at hr
          subst hr
          generalize hM : forIn (m := Id) (_ : Array (Bool × ℕ)) (none, ()) _ = M
          have hM1 : M.run.1 = none := by
            rw [← hM, ← Array.forIn_toList]
            refine forIn_mem_none _ (fun nm => (if nm.1 then decide (nm.2 < body.params.size)
              else decide (nm.2 < k)) = true) ?_ _ (List.all_eq_true.1 hq) _ rfl
            rintro ⟨_ | _, n⟩ ⟨r, u⟩ hn hr
            all_goals
              dsimp only at hr
              subst hr
              simp at hn
              simp [hn]
          simp only [Id.run_bind, hM1, Id.run_pure]
        simp only [Id.run_bind, hQ1, Id.run_pure]
      simp only [Id.run_bind, hA1, Id.run_pure]
      exact ⟨trivial, by rw [Array.size_push, ht]; omega⟩
    obtain ⟨h1, h2⟩ := hnode
    simp only [Id.run_bind, h1, Id.run_pure]
    refine ⟨⟨rfl, show at_ + 1 ≤ N.run.2.size by omega, insert d S, hS', ?_⟩, show at_ + 1 = i + 1 by omega⟩
    show N.run.2.size = 1 + ∑ x ∈ insert d S, walkW P x
    rw [h2, Finset.sum_insert hdS, hW, hsz]
    omega
  rcases key with ⟨⟨-, hle, S, -, hsz⟩, hat⟩ | ⟨h1, h2⟩
  · have := sum_walkW_le P S
    omega
  · simp only [Id.run_bind, h1]
    split
    · exfalso
      omega
    · rfl

end Properties.NetworkScan
