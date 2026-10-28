import Definitions.Nci.Guarantees.Capability.Fresh
import Proofs.Core.Formalism
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Local bounds: proofs of the effective fresh input

* `BlockDescriptionAdvice`: Lemma 1 (`Verity.SecurityProofs.BlockDescription`) at the evaluator `e ++ a`, of at most
  `aBlk + ℓa` bits.
* `FaultyInjection`: the cut circuit's second copy computes the **held run** (`heldRun`), `C` run with the injected gates
  held at `X`'s values (`cut_eval`). On every committed wire `X` is that run (`heldRun_committed`), by the induction behind
  `Partition.cone_induction`: a correct proof block's committed gates hold what its gates compute, and a faulty one's
  are held. The cut keeps each inference block's gates and adds to its unregistered inputs only the fresh inputs of the
  injected gates inside it (`cut_isCK`), at most `m · w` (`card_injected_inter_le`).
-/

namespace Nci.Capability

open Verity Finset

section Cut

variable {C : Circuit} (S : Finset (Fin C.N))

@[simp] theorem cut_isInput_castAdd (i : Fin C.N) : (cutCircuit C S).isInput (Fin.castAdd C.N i) = true := by
  simp only [Fin.addCases_left]

@[simp] theorem cut_isInput_natAdd (g : Fin C.N) : (cutCircuit C S).isInput (Fin.natAdd C.N g) = C.isInput g := by
  simp only [Fin.addCases_right]

theorem cut_reads_natAdd (g : Fin C.N) : (cutCircuit C S).reads (Fin.natAdd C.N g) =
    if g ∈ S ∧ C.isInput g = false then {Fin.castAdd C.N g} else (C.reads g).map (Fin.natAddEmb C.N) := by
  simp only [cutCircuit, Fin.addCases_right]

theorem cut_op_natAdd (g : Fin C.N) (v : (cutCircuit C S).Transcript) :
    (cutCircuit C S).op (Fin.natAdd C.N g) v =
      if g ∈ S ∧ C.isInput g = false then v (Fin.castAdd C.N g)
      else (C.op g fun a => (v (Fin.natAdd C.N a)).setWidth (C.w a)).setWidth 1 := by
  simp only [cutCircuit, Fin.addCases_right]

theorem cutInputs_castAdd (X : C.Transcript) (i : Fin C.N) :
    cutInputs S X (Fin.castAdd C.N i) = (X i).setWidth 1 := by
  simp only [cutInputs, Fin.addCases_left]

theorem cutInputs_natAdd (X : C.Transcript) (g : Fin C.N) :
    cutInputs S X (Fin.natAdd C.N g) = (X g).setWidth 1 := by
  simp only [cutInputs, Fin.addCases_right]

/-- **The held run**: `C` run with its inputs and `S`'s gates held at `X`'s values, every other gate computed. -/
def heldRun (X : C.Transcript) : C.Transcript :=
  C.run fun g => if C.isInput g = true ∨ g ∈ S then some (X g) else none

/-- The cut circuit's second copy, at `X` on both copies, evaluates to the held run. -/
theorem cut_eval (hB : C.Boolean) (X : C.Transcript) (g : Fin C.N) :
    (cutCircuit C S).eval (cutInputs S X) (Fin.natAdd C.N g) = (heldRun S X g).setWidth 1 := by
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  cases hi : C.isInput g
  · rw [Circuit.eval_op _ _ ((cut_isInput_natAdd S g).trans hi), cut_op_natAdd]
    by_cases hS : g ∈ S
    · have hc : (g ∈ S ∧ C.isInput g = false) = True := by simp [hS, hi]
      rw [ite_eq_left_of_eq_true _ _ hc, Circuit.eval_input _ _ (cut_isInput_castAdd S g), cutInputs_castAdd,
        heldRun, C.run_fixed (b := X g) (by simp [hS])]
    · have hc : (g ∈ S ∧ C.isInput g = false) = False := by simp [hS]
      rw [ite_eq_right_of_eq_false _ _ hc, heldRun, C.run_free (by simp [hS, hi])]
      congr 1
      refine C.op_congr g _ _ fun a ha => ?_
      rw [ih a (C.reads_lt g a ha), BitVec.setWidth_setWidth_of_le _ (by rw [hB.one_bit a]), BitVec.setWidth_eq]
      rfl
  · rw [Circuit.eval_input _ _ ((cut_isInput_natAdd S g).trans hi), cutInputs_natAdd, heldRun,
      C.run_fixed (b := X g) (by simp [hi])]

theorem castAdd_ne_natAdd (i g : Fin C.N) : Fin.castAdd C.N i ≠ Fin.natAdd C.N g := by
  intro h
  have := congrArg Fin.val h
  simp only [Fin.val_castAdd, Fin.val_natAdd] at this
  omega

variable {n : ℕ}

theorem cut_gates (P : Partition C n) (u : Fin n) :
    (cutPartition P S).gates u = (P.gates u).map (Fin.natAddEmb C.N) := by
  ext j
  rw [Partition.mem_gates, mem_map]
  refine Fin.addCases (fun i => ?_) (fun g => ?_) j
  · simp only [cutPartition, Fin.addCases_left, Fin.natAddEmb_apply]
    constructor
    · intro h
      cases h
    · rintro ⟨g, -, h⟩
      exact absurd h.symm (castAdd_ne_natAdd i g)
  · simp only [cutPartition, Fin.addCases_right, Fin.natAddEmb_apply]
    constructor
    · exact fun h => ⟨g, P.mem_gates.2 h, rfl⟩
    · rintro ⟨g', hg', he⟩
      rw [(Fin.natAdd_inj _).1 he] at hg'
      exact P.mem_gates.1 hg'

end Cut

theorem cut_unregistered {n : ℕ} (d : Declaration) (S : Finset (Fin d.D.N)) (P : Partition d.D n) (u : Fin n) :
    (cutDecl d S).unregistered ((cutPartition P S).gates u) ⊆
      (d.unregistered (P.gates u)).map (Fin.natAddEmb d.D.N) ∪ (S ∩ P.gates u).map (Fin.castAddEmb d.D.N) := by
  intro j hj
  simp only [Declaration.unregistered, mem_sdiff, Circuit.into, mem_filter, mem_univ, true_and] at hj
  obtain ⟨⟨hjS, g', hg', hjr⟩, hjθ⟩ := hj
  rw [cut_gates] at hg' hjS
  obtain ⟨g, hg, rfl⟩ := mem_map.1 hg'
  simp only [Fin.natAddEmb_apply, Fin.addCases_right] at hjr
  split_ifs at hjr with hc
  · rw [mem_singleton.1 hjr]
    exact mem_union_right _ (mem_map.2 ⟨g, mem_inter.2 ⟨hc.1, hg⟩, rfl⟩)
  · obtain ⟨a, ha, rfl⟩ := mem_map.1 hjr
    refine mem_union_left _ (mem_map.2 ⟨a, ?_, rfl⟩)
    simp only [Declaration.unregistered, mem_sdiff, Circuit.into, mem_filter, mem_univ, true_and]
    exact ⟨⟨fun haS => hjS (mem_map.2 ⟨a, haS, rfl⟩), g, hg, ha⟩, fun haθ => hjθ (mem_map.2 ⟨a, haθ, rfl⟩)⟩

theorem card_cut_unregistered {n : ℕ} (d : Declaration) (S : Finset (Fin d.D.N)) (P : Partition d.D n) (u : Fin n) :
    ((cutDecl d S).unregistered ((cutPartition P S).gates u)).card ≤
      (d.unregistered (P.gates u)).card + (S ∩ P.gates u).card :=
  (card_le_card (cut_unregistered d S P u)).trans ((card_union_le _ _).trans (by rw [card_map, card_map]))

/-- The cut of a `(c, k)`-partition at gates of which each block holds at most `k'` is a `(c, k + k')`-partition. -/
theorem cut_isCK (R : Registration) (S : Finset (Fin R.d.D.N)) {c k k' : ℕ} (hCK : R.P.IsCK R.d c k)
    (hS : ∀ u, (S ∩ R.P.gates u).card ≤ k') : (cutReg R S).P.IsCK (cutReg R S).d c (k + k') where
  boolean := by
    refine ⟨fun _ => rfl, fun j => ?_⟩
    refine Fin.addCases (fun i => ?_) (fun g => ?_) j
    · intro h
      rw [cut_isInput_castAdd] at h
      cases h
    · intro h
      rw [cut_isInput_natAdd] at h
      rw [cut_reads_natAdd]
      split_ifs
      · exact singleton_nonempty _
      · exact (hCK.boolean.fanin_pos g h).map
  card u := by rw [cut_gates, card_map, hCK.card u]
  reads u := (card_cut_unregistered R.d S R.P u).trans (Nat.add_le_add (hCK.reads u) (hS u))

section Injected

variable {C : Circuit} {n : ℕ}

theorem mem_injected {Q : Partition C n} {X : C.Transcript} {g : Fin C.N} :
    g ∈ injected Q X ↔ ∃ v ∈ Q.wrong X, g ∈ C.exits (Q.gates v) := by
  simp [injected]

theorem unit_of_mem_injected {Q : Partition C n} {X : C.Transcript} {g : Fin C.N} (h : g ∈ injected Q X) :
    ∃ v ∈ Q.wrong X, Q.unit g = some v := by
  obtain ⟨v, hv, hg⟩ := mem_injected.1 h
  exact ⟨v, hv, Q.mem_gates.1 (mem_filter.1 hg).1⟩

/-- The induction behind `heldRun_committed`, as `Partition.cone_induction`'s: on every gate, a correct proof block's
local evaluation of its own gates is the held run, and so is every committed wire. -/
theorem heldRun_induction (Q : Partition C n) (X : C.Transcript) :
    ∀ m (g : Fin C.N), g.val = m →
      (∀ v, Q.unit g = some v → v ∉ Q.wrong X → Q.localEval X v g = heldRun (injected Q X) X g) ∧
      (g ∈ Q.committed → X g = heldRun (injected Q X) X g) := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm
  have hA : ∀ v, Q.unit g = some v → v ∉ Q.wrong X → Q.localEval X v g = heldRun (injected Q X) X g := by
    intro v hv hvF
    have hni : C.isInput g = false := by
      cases h : C.isInput g
      · rfl
      · exact absurd ((Q.input_iff g).2 h) (by simp [hv])
    have hnS : g ∉ injected Q X := fun hS => by
      obtain ⟨v', hv'F, hv'⟩ := unit_of_mem_injected hS
      rw [hv] at hv'
      exact hvF (Option.some_injective _ hv' ▸ hv'F)
    rw [Q.localEval_of_mem X hv, heldRun, C.run_free (by simp [hni, hnS])]
    refine C.op_congr g _ _ fun x hx => ?_
    obtain ⟨ihA, ihB⟩ := ih x.val (hgm ▸ C.reads_lt g x hx) x rfl
    by_cases hxv : Q.unit x = some v
    · exact ihA v hxv hvF
    · rw [Q.localEval_of_not_mem X hxv]
      exact ihB (Q.mem_committed_of_read hx (by rw [hv]; exact Ne.symm hxv))
  refine ⟨hA, fun hg => ?_⟩
  cases hi : C.isInput g
  · obtain ⟨v, hv⟩ := Option.ne_none_iff_exists'.1 fun h => by simp [(Q.input_iff g).1 h] at hi
    by_cases hvF : v ∈ Q.wrong X
    · have hS : g ∈ injected Q X := mem_injected.2
        ⟨v, hvF, by rw [← Q.committed_inter_gates]; exact mem_inter.2 ⟨hg, Q.mem_gates.2 hv⟩⟩
      rw [heldRun, C.run_fixed (b := X g) (by simp [hS])]
    · have hc : Q.Correct X v := by
        by_contra h
        exact hvF (Q.mem_wrong.2 h)
      rw [hc g hg hv]
      exact hA v hv hvF
  · rw [heldRun, C.run_fixed (b := X g) (by simp [hi])]

/-- **Only the faulty proof blocks inject**: every committed wire of the trace `X` holds the held run at what they
inject. -/
theorem heldRun_committed (Q : Partition C n) (X : C.Transcript) :
    ∀ g ∈ Q.committed, X g = heldRun (injected Q X) X g :=
  fun g hg => (heldRun_induction Q X g.val g rfl).2 hg

theorem width_eq_card (hB : C.Boolean) (T : Finset (Fin C.N)) : C.width T = T.card := by
  simp [Circuit.width, hB.one_bit]

/-- An inference block holds at most `m · w` injected gates when at most `m` faulty proof blocks, of at most `w` output
bits each, have outputs inside it. -/
theorem card_injected_inter_le (hB : C.Boolean) {n' : ℕ} (Q : Partition C n) (X : C.Transcript)
    (P : Partition C n') (u : Fin n') (m w : ℕ) (hw : ∀ v ∈ Q.wrong X, C.width (C.exits (Q.gates v)) ≤ w)
    (hm : ((Q.wrong X).filter fun v => ¬ Disjoint (C.exits (Q.gates v)) (P.gates u)).card ≤ m) :
    (injected Q X ∩ P.gates u).card ≤ m * w := by
  classical
  set F' := (Q.wrong X).filter fun v => ¬ Disjoint (C.exits (Q.gates v)) (P.gates u)
  have hsub : injected Q X ∩ P.gates u ⊆ F'.biUnion fun v => C.exits (Q.gates v) := by
    intro g hg
    obtain ⟨hgi, hgu⟩ := mem_inter.1 hg
    obtain ⟨v, hv, hgv⟩ := mem_injected.1 hgi
    exact mem_biUnion.2 ⟨v, mem_filter.2 ⟨hv, not_disjoint_iff.2 ⟨g, hgv, hgu⟩⟩, hgv⟩
  calc (injected Q X ∩ P.gates u).card ≤ (F'.biUnion fun v => C.exits (Q.gates v)).card := card_le_card hsub
    _ ≤ ∑ v ∈ F', (C.exits (Q.gates v)).card := card_biUnion_le
    _ ≤ ∑ _v ∈ F', w := sum_le_sum fun v hv => (width_eq_card hB _) ▸ hw v (mem_filter.1 hv).1
    _ = F'.card * w := by rw [sum_const, smul_eq_mul]
    _ ≤ m * w := Nat.mul_le_mul_right w hm

end Injected

end Nci.Capability

namespace Nci.SecurityProofs

open Verity Finset Nci.Capability

theorem BlockDescriptionAdvice : Guarantees.BlockDescriptionAdvice := by
  intro build ℓa c k ρ M r t aBlk e y he hE a ha hCK x hx u
  have h := Verity.SecurityProofs.BlockDescription (build a) c k hCK ρ M r t (aBlk + ℓa) (e ++ a) (y a)
    (by rw [List.length_append, ha]; omega) (hE a ha) x hx u
  rwa [← Nat.add_assoc] at h

theorem FaultyInjection : Guarantees.FaultyInjection := by
  intro R c k hCK n Q X m w hX hw hm
  refine ⟨cut_isCK R _ hCK fun u => card_injected_inter_le hCK.boolean Q X R.P u m w hw (hm u), ?_,
    fun g hg => ?_⟩
  · intro j hj
    obtain ⟨g, hg, rfl⟩ := mem_map.1 hj
    rw [Fin.natAddEmb_apply, cutInputs_natAdd, hX g hg]
    exact (cutInputs_natAdd _ R.val g).symm
  · rw [cut_eval _ hCK.boolean X g, ← heldRun_committed Q X g hg]

end Nci.SecurityProofs
