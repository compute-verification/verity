import Definitions.Nci
import Definitions.Nci.Transfers

/-!
# NCI from the bits on the wire and the credited gates

For a division `P` of a circuit into machines, what `R_b` reads from outside itself is its transfers in and the input
gates it reads, `in(R_b) = transfersInto b ∪ inputsRead b` (`into_eq`), two disjoint sets, and a transfer gate is
computed, so never in a baseline. So `novelReads d R_b` is the width of `(transfersInto b ∪ inputsRead b) \ θ`
(`novelReads_eq`): the bits on the wire into `b`, plus the width of the novel input gates `b` reads itself
(`novelReads_eq_add`). In the Boolean profile, where every gate is one bit (`∀ g, d.D.w g = 1`, which
`FlockSoundness.Audit.Circuit.toCircuit` meets, `boolean_w`), these are numbers of gates (`novelReads_eq_card`,
`novelReads_eq_card_add`).

So NCI compliance is a ratio of two certified numbers. Machine `b`'s IU condition, `Complies`'s condition on one IU, holds
exactly when `size R_b ≤ Γ` and `|transfersInto b| + |inputsRead b \ θ| ≤ α · size R_b` (`iu_iff_counts`), with
`|transfersInto b|` network accounting's certified count of the bits into `b` and `size R_b` compute credit's total for
`b`. When every novel input reaches `b` through another machine, it is `|transfersInto b| ≤ α · size R_b`
(`iu_iff_wire`). That is the form a server needs among clients and the auditor's warden, which needn't comply as IUs: the
warden reads every bit it relays. When each machine is one IU and every machine meets its condition, the whole
declaration complies (`complies_of_counts`).

In the subcircuits' terms (`machine`), the transfer gates from `a` to `b` are `(machine a).inner ∩ (machine b).ins`
less the constants `a` computes (`inner_inter_ins`), and `in(R_b)` is the union over `a ≠ b` of those, together with the
gates with no predecessors that `b` reads (`into_eq_subcircuits`): the same decomposition, with the constants on the
inputs' side.
-/

namespace Nci.SecurityProofs.Transfers

open Verity Finset

section Machines

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- A machine sends itself nothing. -/
theorem transfers_self (b : Fin n) : transfers P b b = ∅ := by
  ext g
  simp only [transfers, Circuit.into, mem_inter, mem_filter, mem_univ, true_and, notMem_empty, iff_false, not_and]
  exact fun h => h.1

/-- **Transfers in NCI's words**: for `a ≠ b`, the gates of `R_a` that `R_b` reads are `out(R_a) ∩ in(R_b)`. -/
theorem transfers_eq_exits_inter_into {a b : Fin n} (hab : a ≠ b) :
    transfers P a b = C.exits (P.gates a) ∩ C.into (P.gates b) := by
  ext g
  simp only [transfers, Circuit.exits, Circuit.into, mem_inter, mem_filter, mem_univ, true_and]
  constructor
  · rintro ⟨⟨hgb, h, hh, hgh⟩, hga⟩
    refine ⟨⟨hga, Or.inr ⟨h, fun hha => ?_, hgh⟩⟩, hgb, h, hh, hgh⟩
    simp only [Partition.gates, mem_filter, mem_univ, true_and] at hh hha
    exact hab (Option.some_injective _ (hha.symm.trans hh))
  · rintro ⟨⟨hga, -⟩, hin⟩
    exact ⟨hin, hga⟩

/-- What `b` receives over the wire is the computed gates of `in(R_b)`. -/
theorem mem_transfersInto {b : Fin n} {g : Fin C.N} :
    g ∈ transfersInto P b ↔ g ∈ C.into (P.gates b) ∧ C.isInput g = false := by
  constructor
  · intro h
    obtain ⟨a, -, hg⟩ := mem_biUnion.1 h
    rw [transfers, mem_inter] at hg
    refine ⟨hg.1, ?_⟩
    cases hi : C.isInput g
    · rfl
    · have hn := (P.input_iff g).2 hi
      simp [Partition.gates, hn] at hg
  · rintro ⟨hin, hi⟩
    have hu : P.unit g ≠ none := fun h => by simp [(P.input_iff g).1 h] at hi
    obtain ⟨a, ha⟩ := Option.ne_none_iff_exists'.1 hu
    have hga : g ∈ P.gates a := by simp [Partition.gates, ha]
    refine mem_biUnion.2 ⟨a, mem_filter.2 ⟨mem_univ _, ?_⟩, mem_inter.2 ⟨hin, hga⟩⟩
    rintro rfl
    simp only [Circuit.into, mem_filter] at hin
    exact hin.2.1 hga

/-- **What `R_b` reads from outside itself** is what reaches it over the wire and the input gates it reads:
`in(R_b) = transfersInto b ∪ inputsRead b`. -/
theorem into_eq (b : Fin n) : C.into (P.gates b) = transfersInto P b ∪ inputsRead P b := by
  ext g
  rw [mem_union, mem_transfersInto, inputsRead, mem_filter]
  cases C.isInput g <;> simp

theorem disjoint_transfersInto_inputsRead (b : Fin n) : Disjoint (transfersInto P b) (inputsRead P b) := by
  rw [disjoint_left]
  intro g h1 h2
  rw [inputsRead, mem_filter] at h2
  exact Bool.false_ne_true (((mem_transfersInto P).1 h1).2.symm.trans h2.2)

/-- A machine's gates as a subcircuit compute its gates that have predecessors. -/
theorem machine_inner (a : Fin n) : (machine P a).inner = P.gates a \ Covering.sources C := by
  ext g
  simp only [Covering.Subcircuit.inner, machine, mem_sdiff, mem_union, mem_inter, not_or, not_and]
  constructor
  · rintro ⟨hg | hg, hin, hs⟩
    · exact ⟨hg, hs hg⟩
    · exact absurd hg hin
  · rintro ⟨hg, hs⟩
    refine ⟨Or.inl hg, fun h => ?_, fun _ => hs⟩
    simp only [Circuit.into, mem_filter] at h
    exact h.2.1 hg

/-- **Transfers in the subcircuits' terms**: the gates `machine a` computes that `machine b` takes as inputs are the
transfer gates from `a` to `b` less the constants (gates with no predecessors), which a subcircuit holds as inputs. -/
theorem inner_inter_ins (a b : Fin n) :
    (machine P a).inner ∩ (machine P b).ins = transfers P a b \ Covering.sources C := by
  rw [machine_inner]
  ext g
  simp only [machine, transfers, mem_inter, mem_sdiff, mem_union]
  constructor
  · rintro ⟨⟨hga, hs⟩, hin | ⟨-, hs'⟩⟩
    · exact ⟨⟨hin, hga⟩, hs⟩
    · exact absurd hs' hs
  · rintro ⟨⟨hin, hga⟩, hs⟩
    exact ⟨⟨hga, hs⟩, Or.inl hin⟩

/-- **`in(R_b)` in the subcircuits' terms**: the union over `a ≠ b` of `(machine a).inner ∩ (machine b).ins`, and the
gates with no predecessors that `R_b` reads (input gates of `C`, and constants another machine computes). -/
theorem into_eq_subcircuits (b : Fin n) :
    C.into (P.gates b) = ((univ.filter (· ≠ b)).biUnion fun a => (machine P a).inner ∩ (machine P b).ins) ∪
      (C.into (P.gates b) ∩ Covering.sources C) := by
  ext g
  simp only [inner_inter_ins, mem_union, mem_biUnion, mem_inter, mem_sdiff]
  constructor
  · intro hg
    by_cases hs : g ∈ Covering.sources C
    · exact Or.inr ⟨hg, hs⟩
    · refine Or.inl ?_
      have hi : C.isInput g = false := by
        cases h : C.isInput g
        · rfl
        · exact absurd (show g ∈ Covering.sources C from mem_filter.2 ⟨mem_univ _, C.input_reads g h⟩) hs
      obtain ⟨a, ha, hga⟩ := mem_biUnion.1 ((mem_transfersInto P).2 ⟨hg, hi⟩)
      exact ⟨a, ha, hga, hs⟩
  · rintro (⟨a, -, hga, -⟩ | ⟨hg, -⟩)
    · exact (mem_inter.1 hga).1
    · exact hg

end Machines

/-- In the Boolean profile a set's width is its number of gates. -/
theorem width_eq_card {C : Circuit} (hw : ∀ g, C.w g = 1) (S : Finset (Fin C.N)) : C.width S = S.card := by
  simp [Circuit.width, hw]

/-- A Boolean circuit's gates are one bit wide: the hypothesis of the Boolean forms below. -/
theorem boolean_w (B : FlockSoundness.Audit.Circuit) (g : Fin B.toCircuit.N) : B.toCircuit.w g = 1 := rfl

section Registered

variable (d : Registered) {n : ℕ} (P : Partition d.D n)

/-- Nothing on the wire is baseline: a transfer gate is computed. -/
theorem disjoint_transfersInto_θ (b : Fin n) : Disjoint (transfersInto P b) d.θ := by
  rw [disjoint_left]
  intro g h1 h2
  exact Bool.false_ne_true (((mem_transfersInto P).1 h1).2.symm.trans (d.θ_input g h2))

/-- **Novel reads from transfers**: `novelReads d R_b = width((transfersInto b ∪ inputsRead b) \ θ)`. -/
theorem novelReads_eq (b : Fin n) :
    novelReads d (P.gates b) = d.D.width ((transfersInto P b ∪ inputsRead P b) \ d.θ) := by
  rw [novelReads, into_eq]

/-- **Novel reads are the bits on the wire and the novel inputs read**: `novelReads d R_b` is the width of
`transfersInto b` plus the width of `inputsRead b \ θ`. -/
theorem novelReads_eq_add (b : Fin n) :
    novelReads d (P.gates b) = d.D.width (transfersInto P b) + d.D.width (inputsRead P b \ d.θ) := by
  rw [novelReads_eq, union_sdiff_distrib, sdiff_eq_self_of_disjoint (disjoint_transfersInto_θ d P b)]
  unfold Circuit.width
  exact sum_union (disjoint_of_subset_right sdiff_subset (disjoint_transfersInto_inputsRead P b))

/-- **Novel reads in the Boolean profile**: `novelReads d R_b = |(transfersInto b ∪ inputsRead b) \ θ|`. -/
theorem novelReads_eq_card (hw : ∀ g, d.D.w g = 1) (b : Fin n) :
    novelReads d (P.gates b) = ((transfersInto P b ∪ inputsRead P b) \ d.θ).card := by
  rw [novelReads_eq, width_eq_card hw]

/-- **Novel reads in the Boolean profile, as two counts**: `novelReads d R_b = |transfersInto b| + |inputsRead b \ θ|`,
the bits on the wire into `b` and the novel input gates `b` reads. -/
theorem novelReads_eq_card_add (hw : ∀ g, d.D.w g = 1) (b : Fin n) :
    novelReads d (P.gates b) = (transfersInto P b).card + (inputsRead P b \ d.θ).card := by
  rw [novelReads_eq_add, width_eq_card hw, width_eq_card hw]

/-- **One machine's IU condition from its counts** (the Boolean profile): `Complies`'s condition on the IU `R_b`
(at most `Γ` gates, at most `α` bits of novel information per gate) holds exactly when `size R_b ≤ Γ` and
`|transfersInto b| + |inputsRead b \ θ| ≤ α · size R_b`: network accounting's certified bits into `b`, plus the novel
input gates `b` reads itself, against compute credit's total for `b`. It is stated for one machine because the others
need not comply as IUs: a server's session is one IU, while the clients and the auditor's warden, which reads every bit
it relays, are not IUs of the server's workload. -/
theorem iu_iff_counts (Γ : ℕ) (α : ℝ) (hw : ∀ g, d.D.w g = 1) (b : Fin n) :
    (d.D.size (P.gates b) ≤ Γ ∧ (novelReads d (P.gates b) : ℝ) ≤ α * d.D.size (P.gates b)) ↔
      (d.D.size (P.gates b) ≤ Γ ∧
        ((transfersInto P b).card + (inputsRead P b \ d.θ).card : ℝ) ≤ α * d.D.size (P.gates b)) := by
  rw [novelReads_eq_card_add d P hw, Nat.cast_add]

/-- **NCI compliance is a ratio of two certified numbers** (the Boolean profile): when every input gate `R_b` reads
outside the baseline reaches it through another machine instead (a client holds the prompt), `R_b`'s IU condition is
`size R_b ≤ Γ` and `|transfersInto b| ≤ α · size R_b`, the bits on the wire into `b` against the gates credited to `b`. -/
theorem iu_iff_wire (Γ : ℕ) (α : ℝ) (hw : ∀ g, d.D.w g = 1) (b : Fin n) (hθ : inputsRead P b ⊆ d.θ) :
    (d.D.size (P.gates b) ≤ Γ ∧ (novelReads d (P.gates b) : ℝ) ≤ α * d.D.size (P.gates b)) ↔
      (d.D.size (P.gates b) ≤ Γ ∧ ((transfersInto P b).card : ℝ) ≤ α * d.D.size (P.gates b)) := by
  rw [iu_iff_counts d P Γ α hw, sdiff_eq_empty_iff_subset.2 hθ, card_empty, Nat.cast_zero, add_zero]

/-- **Compliance from the certified counts** (the Boolean profile): when the machines are the IU division, one IU per
machine (a whole-node session), and every machine `b` has at most `Γ` credited gates and
`|transfersInto b| + |inputsRead b \ θ| ≤ α · size R_b`, the declaration complies at `(Γ, α)`. The left side is
`|(transfersInto b ∪ inputsRead b) \ θ|`, the two sets being disjoint and the wire outside the baseline. -/
theorem complies_of_counts (Γ : ℕ) (α : ℝ) (hw : ∀ g, d.D.w g = 1)
    (h : ∀ b, d.D.size (P.gates b) ≤ Γ ∧
      ((transfersInto P b).card + (inputsRead P b \ d.θ).card : ℝ) ≤ α * d.D.size (P.gates b)) :
    Complies Γ α d :=
  ⟨n, P, fun b => (iu_iff_counts d P Γ α hw b).2 (h b)⟩

end Registered

end Nci.SecurityProofs.Transfers
