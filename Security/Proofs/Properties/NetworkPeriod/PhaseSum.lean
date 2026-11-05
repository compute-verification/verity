import Definitions.NetworkCertifier.Tied
import Proofs.Properties.NetworkCertifier.Slots

/-!
# A phase's free inputs number at most its statements' own

A rows-form phase's joint Program (`NetTiming.Phase.prog`) lays its statements' Programs side by side, so each of its
gates is gate `g − offset j` of the statement `j` that holds it (`locate_some`), and an input gate there is an input
gate of that statement (`input_of_sideBySide`). A free input of the phase outside its registered inputs is then a free
input of its statement outside that statement's own registered inputs, where each tied audit's layout has no ingress
before the horizon (the phase's ingress is its scan's): the phase's wires include each statement's own
(`Phase.ownWires`), and its registered inputs each statement's (`Phase.reg`). So the phase's count is at most its
statements' counts summed (`card_le`).
-/

namespace Properties.NetworkPeriod.PhaseSum

open Claims.Sampling
open NetTiming (Phase Ties BoundaryGates freeInputs sideBySide shiftGate)
open Properties.NetworkCertifier.Counts (bigU mem_bigU card_bigU_le sum_range_eq)

theorem shiftGate_input {o : ℕ} {x : BGate} (h : shiftGate o x = .input) : x = .input := by
  cases x with
  | input => rfl
  | const b => simp [shiftGate] at h
  | not i => simp [shiftGate] at h
  | and ws => simp [shiftGate] at h
  | xor ws => simp [shiftGate] at h

/-- **An input gate of Programs side by side**, at Program `j`'s offset, is an input gate of Program `j`. -/
theorem input_of_sideBySide : ∀ (Cs : List Prog) (j h : ℕ) (C : Prog), Cs[j]? = some C →
    (sideBySide Cs).gates[((Cs.take j).map (·.gates.length)).sum + h]? = some .input → h < C.gates.length →
      C.gates[h]? = some .input
  | [], _, _, _, hC, _, _ => by simp at hC
  | C' :: Cs, 0, h, C, hC, hg, hh => by
    have e : C' = C := by simpa using hC
    subst e
    have eg : (sideBySide (C' :: Cs)).gates = C'.gates ++ (sideBySide Cs).gates.map (shiftGate C'.gates.length) :=
      rfl
    rw [List.take_zero, List.map_nil, List.sum_nil, Nat.zero_add, eg, List.getElem?_append_left hh] at hg
    exact hg
  | C' :: Cs, j + 1, h, C, hC, hg, hh => by
    have hC' : Cs[j]? = some C := by simpa using hC
    have eg : (sideBySide (C' :: Cs)).gates = C'.gates ++ (sideBySide Cs).gates.map (shiftGate C'.gates.length) :=
      rfl
    rw [List.take_succ_cons, List.map_cons, List.sum_cons, eg, Nat.add_assoc,
      List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left, List.getElem?_map] at hg
    obtain ⟨x, hx, hxi⟩ := Option.map_eq_some_iff.1 hg
    rw [shiftGate_input hxi] at hx
    exact input_of_sideBySide Cs j h C hC' hx hh

theorem sideBySide_length : ∀ Cs : List Prog, (sideBySide Cs).gates.length = (Cs.map (·.gates.length)).sum
  | [] => rfl
  | C :: Cs => by
    show (C.gates ++ (sideBySide Cs).gates.map (shiftGate C.gates.length)).length = _
    rw [List.length_append, List.length_map, sideBySide_length Cs, List.map_cons, List.sum_cons]

theorem offset_zero (ph : Phase) : ph.offset 0 = 0 := by
  simp [Phase.offset]

theorem offset_succ (ph : Phase) {j : ℕ} (hj : j < ph.stmts.length) :
    ph.offset (j + 1) = ph.offset j + ph.stmts[j].prog.gates.length := by
  unfold Phase.offset
  rw [List.take_succ_eq_append_getElem hj, List.map_append, List.sum_append, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero]

theorem prog_length (ph : Phase) : ph.prog.gates.length = ph.offset ph.stmts.length := by
  unfold Phase.prog Phase.offset
  rw [sideBySide_length, List.take_length, List.map_map]
  try rfl

theorem exists_offset (o : ℕ → ℕ) (h0 : o 0 = 0) {g : ℕ} : ∀ n, g < o n → ∃ j < n, o j ≤ g ∧ g < o (j + 1)
  | 0, h => by omega
  | n + 1, h => by
    by_cases hn : g < o n
    · obtain ⟨j, hj, h1, h2⟩ := exists_offset o h0 n hn
      exact ⟨j, by omega, h1, h2⟩
    · exact ⟨n, by omega, by omega, h⟩

/-- **The statement that holds a gate**: gate `g` of the joint Program is gate `g − offset j` of statement `j`. -/
theorem locate_some {ph : Phase} {g : ℕ} (hg : g < ph.prog.gates.length) :
    ∃ j s, ph.locate g = some (j, g - ph.offset j) ∧ ph.stmts[j]? = some s ∧ ph.offset j ≤ g ∧
      g - ph.offset j < s.prog.gates.length := by
  rw [prog_length] at hg
  obtain ⟨j0, hj0, h1, h2⟩ := exists_offset ph.offset (offset_zero ph) _ hg
  cases hl : ph.locate g with
  | none =>
    unfold Phase.locate at hl
    rw [List.findSome?_eq_none_iff] at hl
    have h := hl j0 (List.mem_range.2 hj0)
    rw [ite_eq_left (And.intro h1 h2)] at h
    cases h
  | some x =>
    unfold Phase.locate at hl
    obtain ⟨j, hj, hf⟩ := List.exists_of_findSome?_eq_some hl
    have hjn := List.mem_range.1 hj
    by_cases hc : ph.offset j ≤ g ∧ g < ph.offset (j + 1)
    · rw [ite_eq_left hc] at hf
      injection hf with hf
      subst hf
      refine ⟨j, ph.stmts[j], rfl, List.getElem?_eq_getElem hjn, hc.1, ?_⟩
      have := offset_succ ph hjn
      omega
    · rw [ite_eq_right hc] at hf
      cases hf

theorem orElse_eq_none {α : Type} {a : Option α} {b : Unit → Option α} (h : a.orElse b = none) : a = none := by
  cases a with
  | none => rfl
  | some x =>
    rw [Option.orElse_some] at h
    cases h

/-- Statement `j`'s free inputs outside its registered inputs, at its offset in the joint Program. -/
def stmtFree {L S : ℕ} (ph : Phase) (lay : AuditStatement → BoundaryGates L S) (k : ℕ)
    (reg : AuditStatement → Finset ℕ) (j : ℕ) : Finset ℕ :=
  ((ph.stmts[j]?).map fun s => (freeInputs s.prog (lay s) k \ reg s).image (· + ph.offset j)).getD ∅

/-- **A phase's free inputs number at most its statements'**: the joint Program's free inputs over the horizon `k`
outside the phase's registered inputs number at most the sum over its statements of each one's free inputs outside its
own registered inputs, where each tied audit's layout has no ingress before `k`. -/
theorem card_le {L S : ℕ} (ph : Phase) (lay : AuditStatement → BoundaryGates L S) (t : Ties) (k : ℕ)
    (reg : AuditStatement → Finset ℕ) (hin : ∀ s ∈ ph.tied, ∀ i < k, (lay s).ingress i = []) :
    (freeInputs ph.prog (ph.layout lay t) k \ ph.reg reg).card ≤
      (ph.stmts.map fun s => (freeInputs s.prog (lay s) k \ reg s).card).sum := by
  have hsub : freeInputs ph.prog (ph.layout lay t) k \ ph.reg reg ⊆
      bigU ((List.range ph.stmts.length).map (stmtFree ph lay k reg)) := by
    intro g hg
    obtain ⟨hgF, hgR⟩ := Finset.mem_sdiff.1 hg
    obtain ⟨hglt, hgin, hging, hgw⟩ := Finset.mem_filter.1 hgF
    obtain ⟨j, s, hloc, hs, hoff, hlt⟩ := locate_some (Finset.mem_range.1 hglt)
    obtain ⟨hjn, -⟩ := List.getElem?_eq_some_iff.1 hs
    have hCs : ((ph.stmts.map (·.prog)).take j).map (·.gates.length) =
        (ph.stmts.take j).map (·.prog.gates.length) := by
      rw [← List.map_take, List.map_map]
      try rfl
    have hsin : s.prog.gates[g - ph.offset j]? = some .input := by
      refine input_of_sideBySide (ph.stmts.map (·.prog)) j (g - ph.offset j) s.prog (by simp [hs]) ?_ hlt
      rw [hCs]
      show ph.prog.gates[ph.offset j + (g - ph.offset j)]? = some .input
      rw [Nat.add_sub_of_le hoff]
      exact hgin
    have hsing : (List.range k).any (fun i => ((lay s).ingress i).contains (g - ph.offset j)) = false := by
      rcases j with _ | j'
      · have hs0 : ph.scan = s := by simpa [Phase.stmts] using hs
        subst hs0
        rw [offset_zero, Nat.sub_zero]
        exact hging
      · have hst : s ∈ ph.tied := by
          have h' : ph.tied[j']? = some s := by simpa [Phase.stmts] using hs
          exact List.mem_of_getElem? h'
        rw [List.any_eq_false]
        intro i hi
        rw [hin s hst i (List.mem_range.1 hi)]
        simp
    have hsw : (lay s).wires (g - ph.offset j) = none := by
      have hown : ph.ownWires lay g = none :=
        orElse_eq_none (a := ph.ownWires lay g) (b := fun _ => t.state g <|> t.prog g) hgw
      have e : ph.ownWires lay g = ((lay s).wires (g - ph.offset j)).map (· + ph.offset j) := by
        unfold Phase.ownWires
        rw [hloc, Option.bind_some]
        show ((ph.stmts[j]?).bind fun s => ((lay s).wires (g - ph.offset j)).map (· + ph.offset j)) = _
        rw [hs, Option.bind_some]
      rw [e] at hown
      exact Option.map_eq_none_iff.1 hown
    have hsr : g - ph.offset j ∉ reg s := by
      classical
      exact fun hr => hgR (Finset.mem_filter.2 ⟨hglt, j, g - ph.offset j, s, hloc, hs, hr⟩)
    refine mem_bigU.2 ⟨stmtFree ph lay k reg j, List.mem_map.2 ⟨j, List.mem_range.2 hjn, rfl⟩, ?_⟩
    unfold stmtFree
    rw [hs, Option.map_some, Option.getD_some]
    exact Finset.mem_image.2 ⟨g - ph.offset j, Finset.mem_sdiff.2 ⟨Finset.mem_filter.2
      ⟨Finset.mem_range.2 hlt, hsin, hsing, hsw⟩, hsr⟩, Nat.sub_add_cancel hoff⟩
  refine (Finset.card_le_card hsub).trans ((card_bigU_le _).trans ?_)
  rw [List.map_map, ← sum_range_eq (fun s => (freeInputs s.prog (lay s) k \ reg s).card) ph.stmts
    (fun j => ((ph.stmts[j]?).map fun s => (freeInputs s.prog (lay s) k \ reg s).card).getD 0)
    fun i hi => by rw [List.getElem?_eq_getElem hi, Option.map_some, Option.getD_some]]
  refine List.sum_le_sum fun j _ => ?_
  show (stmtFree ph lay k reg j).card ≤
    ((ph.stmts[j]?).map fun s => (freeInputs s.prog (lay s) k \ reg s).card).getD 0
  unfold stmtFree
  cases ph.stmts[j]? with
  | none => simp
  | some s =>
    simp only [Option.map_some, Option.getD_some]
    exact Finset.card_image_le

end Properties.NetworkPeriod.PhaseSum
