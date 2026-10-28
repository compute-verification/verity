import Definitions.Core.Guarantees.Expand
import Definitions.Core.Boolean
import Proofs.Core.Circuit

/-!
# Proofs of the bit-level expansion's guarantees

* `getLsbD_ofBits`, `ofBits_getLsbD`: `ofBits` assembles a value from its bits;
* `Circuit.card_filter_unbit`: the expansion's gates whose gate satisfies `p` are the bits of the gates satisfying `p`;
* `FlockSoundness.Audit.Circuit.toCircuit_boolean`: C-Flock's circuits are Boolean in the draft's sense when no gate is
  an empty row, the one operation that reads nothing.
-/

namespace Verity

open Finset

theorem getLsbD_ofBits {n : ℕ} (f : Fin n → Bool) (i : Fin n) : (ofBits f).getLsbD i = f i := by
  unfold ofBits
  rw [BitVec.getLsbD_cast, BitVec.getLsbD_ofBoolListLE]
  simp

theorem ofBits_getLsbD {n : ℕ} (x : BitVec n) : ofBits (fun i : Fin n => x.getLsbD i) = x := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  exact getLsbD_ofBits (fun i : Fin n => x.getLsbD i) ⟨i, hi⟩

theorem ofBool_getLsbD_zero (x : BitVec 1) : BitVec.ofBool (x.getLsbD 0) = x := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  obtain rfl : i = 0 := by omega
  simp

/-- Counting a filter on `Fin (Σ n)` by the first component. -/
theorem card_filter_fst {m : ℕ} {n : Fin m → ℕ} (p : Fin m → Prop) [DecidablePred p] :
    (univ.filter fun x : Fin (∑ i, n i) => p (finSigmaFinEquiv.symm x).1).card = ∑ i ∈ univ.filter p, n i := by
  rw [Finset.card_equiv (finSigmaFinEquiv.symm) (t := (univ.filter p).sigma fun _ => univ)
    (fun x => by simp), Finset.card_sigma]
  simp

namespace Circuit

variable (C : Circuit)

theorem card_filter_unbit (p : Fin C.N → Prop) [DecidablePred p] :
    (univ.filter fun b => p (C.unbit b).1).card = ∑ g ∈ univ.filter p, C.w g :=
  card_filter_fst p

theorem expandT_bit (X : C.Transcript) (g : Fin C.N) (i : Fin (C.w g)) :
    C.expandT X (C.bit g i) = BitVec.ofBool ((X g).getLsbD i) := by
  unfold expandT
  rw [unbit_bit]

theorem expand_isInput (b : Fin C.expand.N) : C.expand.isInput b = C.isInput (C.unbit b).1 := rfl

theorem mem_expand_reads {b b' : Fin C.expand.N} (hi : C.isInput (C.unbit b).1 = false) :
    b' ∈ C.expand.reads b ↔ (C.unbit b').1 ∈ C.reads (C.unbit b).1 := by
  show b' ∈ (if C.isInput (C.unbit b).1 then ∅ else _) ↔ _
  rw [hi]
  simp only [Bool.false_eq_true, ↓reduceIte]
  exact ⟨fun h => (mem_filter.1 h).2, fun h => mem_filter.2 ⟨mem_univ _, h⟩⟩

theorem expand_eval (a : C.expand.Transcript) : C.expand.eval a = C.expandT (C.eval (C.collapse a)) := by
  funext b
  induction b using WellFoundedLT.induction with
  | _ b ih => ?_
  cases hi : C.isInput (C.unbit b).1
  · rw [C.expand.eval_op a hi]
    show BitVec.ofBool _ = BitVec.ofBool ((C.eval (C.collapse a) (C.unbit b).1).getLsbD (C.unbit b).2)
    rw [C.eval_op _ hi]
    congr 2
    refine C.op_congr _ _ _ fun x hx => ?_
    have hlt : ∀ j, C.bit x j < b := fun j => by
      rw [← C.bit_unbit b]
      exact C.bit_lt (C.reads_lt _ _ hx) _ _
    conv_rhs => rw [← ofBits_getLsbD (C.eval (C.collapse a) x)]
    congr 1
    funext j
    rw [ih _ (hlt j), expandT_bit]
    simp
  · rw [C.expand.eval_input a hi]
    show a b = BitVec.ofBool ((C.eval (C.collapse a) (C.unbit b).1).getLsbD (C.unbit b).2)
    rw [C.eval_input _ hi, collapse, getLsbD_ofBits, C.bit_unbit b, ofBool_getLsbD_zero]

theorem expand_bits (X : C.Transcript) (l : List (Fin C.N)) :
    C.expand.bits (C.expandT X) (l.flatMap fun g => List.ofFn (C.bit g)) = C.bits X l := by
  induction l with
  | nil => rfl
  | cons g l ih =>
    simp only [List.flatMap_cons, bits] at ih ⊢
    rw [List.flatMap_append, ih]
    congr 1
    show (List.ofFn (C.bit g)).flatMap (fun b => List.ofFn fun i : Fin 1 => (C.expandT X b).getLsbD i) = _
    have hs : ∀ l : List (Fin (∑ g, C.w g)),
        (l.flatMap fun b => List.ofFn fun i : Fin 1 => (C.expandT X b).getLsbD i) =
          l.map fun b => (C.expandT X b).getLsbD 0 := by
      intro l
      induction l with
      | nil => rfl
      | cons b l ih =>
        rw [List.flatMap_cons, List.map_cons, ih]
        simp [List.ofFn_succ]
    rw [hs, List.map_ofFn]
    congr 1
    funext i
    simp [expandT_bit]

theorem expand_boolean (h : C.BitFaninPos) : C.expand.Boolean where
  one_bit _ := rfl
  fanin_pos b hb := by
    obtain ⟨a, ha, hw⟩ := h _ hb (Fin.pos (C.unbit b).2)
    exact ⟨C.bit a ⟨0, hw⟩, (C.mem_expand_reads hb).2 (by rw [unbit_bit]; exact ha)⟩

end Circuit

theorem Partition.expand_gates {C : Circuit} {B : ℕ} (P : Partition C B) (u : Fin B) :
    P.expand.gates u = univ.filter fun b => (C.unbit b).1 ∈ P.gates u := by
  ext b
  simp only [Partition.gates, mem_filter, mem_univ, true_and]
  rfl

theorem Partition.isInput_of_mem_gates {C : Circuit} {B : ℕ} (P : Partition C B) {u : Fin B} {g : Fin C.N}
    (h : g ∈ P.gates u) : C.isInput g = false := by
  cases hi : C.isInput g
  · rfl
  · rw [← P.input_iff] at hi
    simp [Partition.gates, hi] at h

theorem sum_univ_filter_mem {α : Type*} [Fintype α] [DecidableEq α] (S : Finset α) (f : α → ℕ) :
    ∑ g ∈ univ.filter (· ∈ S), f g = ∑ g ∈ S, f g := by
  rw [filter_mem_eq_inter, univ_inter]

end Verity

namespace Verity.SecurityProofs

open Finset

theorem ExpandEval : Guarantees.ExpandEval := fun C a => C.expand_eval a

theorem ExpandBits : Guarantees.ExpandBits := fun C X l => C.expand_bits X l

theorem ExpandBoolean : Guarantees.ExpandBoolean := fun C h => C.expand_boolean h

theorem ExpandSize : Guarantees.ExpandSize := by
  intro d B P
  refine ⟨d.D.card_filter_unbit (fun g => d.D.isInput g = false), fun u => ⟨?_, ?_⟩⟩
  · rw [Circuit.size, P.expand_gates, d.D.card_filter_unbit (· ∈ P.gates u), sum_univ_filter_mem]
    rfl
  · classical
    calc (d.expand.unregistered (P.expand.gates u)).card
        ≤ (univ.filter fun b => (d.D.unbit b).1 ∈ d.unregistered (P.gates u)).card := by
          refine card_le_card fun b' hb' => ?_
          simp only [RegisteredCircuit.unregistered, mem_sdiff, Circuit.into, mem_filter, mem_univ, true_and]
            at hb' ⊢
          obtain ⟨⟨hout, b, hb, hr⟩, hθ⟩ := hb'
          rw [P.expand_gates, mem_filter] at hb
          have hi := P.isInput_of_mem_gates hb.2
          refine ⟨⟨fun h => hout ?_, (d.D.unbit b).1, hb.2, (d.D.mem_expand_reads hi).1 hr⟩, fun h => hθ ?_⟩
          · rw [P.expand_gates, mem_filter]
            exact ⟨mem_univ _, h⟩
          · exact h
      _ = d.D.width (d.unregistered (P.gates u)) := by
          rw [d.D.card_filter_unbit (· ∈ d.unregistered (P.gates u)), sum_univ_filter_mem]
          rfl

end Verity.SecurityProofs

namespace FlockSoundness.Audit.Circuit

/-- C-Flock's circuits are Boolean in the draft's sense when no gate is an empty row. -/
theorem toCircuit_boolean (B : Circuit) (h : ∀ g, B.op g ≠ .row [] []) : B.toCircuit.Boolean where
  one_bit _ := rfl
  fanin_pos g hg := by
    show ((B.op g).args.toFinset).Nonempty
    rw [List.toFinset_nonempty_iff]
    have hg' : (B.op g).isInput = false := hg
    have := h g
    cases hop : B.op g <;> simp_all [Op.args, Op.isInput]

end FlockSoundness.Audit.Circuit
