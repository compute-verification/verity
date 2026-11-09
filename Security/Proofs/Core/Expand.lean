module

public import Definitions.Core.Guarantees.Expand
public import Definitions.Core.Boolean
public import Proofs.Core.Circuit

/-!
# Proofs of the bit-level expansion's guarantees

* `getLsbD_ofBits`, `ofBits_getLsbD`: `ofBits` assembles a value from its bits;
* `Circuit.constInputs_bitFaninPos`: with its constants as inputs every circuit has positive fan-in in bits, so its
  expansion is Boolean, and `RegisteredCircuit.constInputs_unregistered`: registering them adds no unregistered read;
* `Circuit.isConst_false`: a circuit of positive fan-in in bits whose computed gates are at least a bit wide has no
  constant, so `Partition.ofConstInputs` and `RegisteredCircuit.constInputs_unregistered_eq` (and their expansion
  forms) carry a partition of it with its constants registered back to it, unchanged;
* `Circuit.card_filter_unbit`: the expansion's gates whose gate satisfies `p` are the bits of the gates satisfying `p`;
* `Verity.Boolean.Circuit.toCircuit_boolean`: Flock's circuits are Boolean in the draft's sense when no gate is
  an empty row, the one operation that reads nothing.
-/

@[expose] public section

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

theorem expandT_bit (X : C.Trace) (g : Fin C.N) (i : Fin (C.w g)) :
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

theorem expand_eval (a : C.expand.Trace) : C.expand.eval a = C.expandT (C.eval (C.collapse a)) := by
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

theorem expand_bits (X : C.Trace) (l : List (Fin C.N)) :
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

theorem isConst_eq_true {g : Fin C.N} :
    C.isConst g = true ↔ C.isInput g = false ∧ ∀ a ∈ C.reads g, C.w a = 0 := by
  simp [isConst]

theorem constInputs_reads {g : Fin C.N} (hc : C.isConst g = false) : C.constInputs.reads g = C.reads g := by
  show (if C.isConst g = true then ∅ else C.reads g) = C.reads g
  simp [hc]

/-- With its constants as inputs, every computed gate of positive width reads a gate of positive width: a computed
gate that reads none is a constant. -/
theorem constInputs_bitFaninPos : C.constInputs.BitFaninPos := by
  intro g hg _
  have hg' : (C.isInput g || C.isConst g) = false := hg
  simp only [Bool.or_eq_false_iff] at hg'
  have hne : ∃ a ∈ C.reads g, C.w a ≠ 0 := Classical.byContradiction fun hne => by
    have h : ∀ a ∈ C.reads g, C.w a = 0 := fun a ha => Classical.byContradiction fun hw => hne ⟨a, ha, hw⟩
    have := (C.isConst_eq_true).2 ⟨hg'.1, h⟩
    rw [hg'.2] at this
    exact Bool.false_ne_true this
  obtain ⟨a, ha, hwa⟩ := hne
  exact ⟨a, by rw [C.constInputs_reads hg'.2]; exact ha, Nat.pos_of_ne_zero hwa⟩

theorem constInputs_boolean (h : ∀ g, C.w g = 1) : C.constInputs.Boolean where
  one_bit g := h g
  fanin_pos g hg := by
    have h1 := h g
    obtain ⟨a, ha, -⟩ := C.constInputs_bitFaninPos g hg (by show 0 < C.w g; omega)
    exact ⟨a, ha⟩

theorem constInputs_isInput {g : Fin C.N} (hc : C.isConst g = false) : C.constInputs.isInput g = C.isInput g := by
  show (C.isInput g || C.isConst g) = C.isInput g
  rw [hc, Bool.or_false]

theorem constInputs_expand_isInput (hc : ∀ g, C.isConst g = false) (b : Fin (∑ g, C.w g)) :
    C.constInputs.expand.isInput b = C.expand.isInput b := by
  show C.constInputs.isInput (C.unbit b).1 = C.isInput (C.unbit b).1
  exact C.constInputs_isInput (hc _)

theorem constInputs_expand_reads (hc : ∀ g, C.isConst g = false) (b : Fin (∑ g, C.w g)) :
    C.constInputs.expand.reads b = C.expand.reads b := by
  show (if C.constInputs.isInput (C.unbit b).1 = true then ∅ else
      univ.filter fun b' => (C.unbit b').1 ∈ C.constInputs.reads (C.unbit b).1) =
    (if C.isInput (C.unbit b).1 = true then ∅ else univ.filter fun b' => (C.unbit b').1 ∈ C.reads (C.unbit b).1)
  rw [C.constInputs_isInput (hc _), C.constInputs_reads (hc _)]

/-- A circuit with positive fan-in in bits whose computed gates are at least a bit wide has no constant. -/
theorem isConst_false (hf : C.BitFaninPos) (hw : ∀ g, C.isInput g = false → 1 ≤ C.w g) (g : Fin C.N) :
    C.isConst g = false := by
  cases hc : C.isConst g
  · rfl
  · obtain ⟨hi, h0⟩ := (C.isConst_eq_true).1 hc
    obtain ⟨a, ha, hpos⟩ := hf g hi (Nat.lt_of_lt_of_le Nat.one_pos (hw g hi))
    exact absurd (h0 a ha) (Nat.pos_iff_ne_zero.1 hpos)

theorem Boolean.bitFaninPos {D : Circuit} (hB : D.Boolean) : D.BitFaninPos := fun g hg _ => by
  obtain ⟨a, ha⟩ := hB.fanin_pos g hg
  exact ⟨a, ha, Nat.lt_of_lt_of_le Nat.one_pos (hB.one_bit a).ge⟩

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

/-- `P` with `C`'s constants as inputs (`Circuit.constInputs`): each constant leaves its unit. -/
def Partition.constInputs {C : Circuit} {n : ℕ} (P : Partition C n) : Partition C.constInputs n where
  unit g := if C.isConst g then none else P.unit g
  input_iff g := by
    show (if C.isConst g = true then none else P.unit g) = none ↔ (C.isInput g || C.isConst g) = true
    cases C.isConst g
    · simp [P.input_iff]
    · simp

theorem Partition.mem_constInputs_gates {C : Circuit} {n : ℕ} (P : Partition C n) {u : Fin n} {g : Fin C.N} :
    g ∈ P.constInputs.gates u ↔ C.isConst g = false ∧ g ∈ P.gates u := by
  simp only [Partition.gates, mem_filter, mem_univ, true_and]
  show (if C.isConst g = true then none else P.unit g) = some u ↔ C.isConst g = false ∧ P.unit g = some u
  cases C.isConst g <;> simp

/-- With its constants registered, a unit reads no unregistered gate its unit in `P` doesn't: a constant is
registered, and every other gate reads what it read. -/
theorem RegisteredCircuit.constInputs_unregistered (d : RegisteredCircuit) {n : ℕ} (P : Partition d.D n) (u : Fin n) :
    d.constInputs.unregistered (P.constInputs.gates u) ⊆ d.unregistered (P.gates u) := by
  intro a ha
  have ha' : a ∈ d.D.constInputs.into (P.constInputs.gates u) \
      (d.θ ∪ univ.filter fun g => d.D.isConst g = true) := ha
  obtain ⟨hin, hout⟩ := mem_sdiff.1 ha'
  obtain ⟨haS, g, hg, hag⟩ := (mem_filter.1 hin).2
  have haθ : a ∉ d.θ := fun h => hout (mem_union_left _ h)
  have hac : d.D.isConst a = false := by
    cases hc : d.D.isConst a
    · rfl
    · exact absurd (mem_union_right _ (mem_filter.2 ⟨mem_univ a, hc⟩)) hout
  rw [P.mem_constInputs_gates] at haS hg
  rw [d.D.constInputs_reads hg.1] at hag
  show a ∈ d.D.into (P.gates u) \ d.θ
  exact mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, fun haS' => haS ⟨hac, haS'⟩, g, hg.2, hag⟩, haθ⟩

/-- On a circuit with no constant, a partition of it with its constants as inputs, as a partition of it: the same
units. -/
def Partition.ofConstInputs {C : Circuit} {n : ℕ} (hc : ∀ g, C.isConst g = false) (P : Partition C.constInputs n) :
    Partition C n where
  unit := P.unit
  input_iff g := (P.input_iff g).trans (by rw [C.constInputs_isInput (hc g)])

/-- On a circuit with no constant, a partition of its expansion with its constants as inputs, as a partition of its
expansion: the same units. -/
def Partition.ofConstInputsExpand {C : Circuit} {n : ℕ} (hc : ∀ g, C.isConst g = false)
    (P : Partition C.constInputs.expand n) : Partition C.expand n where
  unit := P.unit
  input_iff b := (P.input_iff b).trans (by rw [C.constInputs_expand_isInput hc b])

/-- On a declaration with no constant, registering its constants changes no set's unregistered reads. -/
theorem RegisteredCircuit.constInputs_unregistered_eq (d : RegisteredCircuit) (hc : ∀ g, d.D.isConst g = false)
    (S : Finset (Fin d.D.N)) : d.constInputs.unregistered S = d.unregistered S := by
  show d.D.constInputs.into S \ (d.θ ∪ univ.filter fun g => d.D.isConst g = true) = d.D.into S \ d.θ
  have hθ : ∀ a, a ∉ d.θ → a ∉ (d.θ ∪ univ.filter fun g => d.D.isConst g = true) := fun a ha h =>
    (mem_union.1 h).elim ha fun h' => by simp [hc a] at h'
  ext a
  constructor
  · intro ha
    obtain ⟨hin, hout⟩ := mem_sdiff.1 ha
    obtain ⟨haS, g, hg, hag⟩ := (mem_filter.1 hin).2
    rw [d.D.constInputs_reads (hc g)] at hag
    exact mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, haS, g, hg, hag⟩, fun h => hout (mem_union_left _ h)⟩
  · intro ha
    obtain ⟨hin, hout⟩ := mem_sdiff.1 ha
    obtain ⟨haS, g, hg, hag⟩ := (mem_filter.1 hin).2
    rw [← d.D.constInputs_reads (hc g)] at hag
    exact mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, haS, g, hg, hag⟩, hθ a hout⟩

/-- On a declaration with no constant, registering its constants changes no set's unregistered reads in its
expansion. -/
theorem RegisteredCircuit.constInputs_expand_unregistered_eq (d : RegisteredCircuit) (hc : ∀ g, d.D.isConst g = false)
    (S : Finset (Fin d.D.expand.N)) : d.constInputs.expand.unregistered S = d.expand.unregistered S := by
  show d.D.constInputs.expand.into S \
      univ.filter (fun b => (d.D.unbit b).1 ∈ (d.θ ∪ univ.filter fun g => d.D.isConst g = true)) =
    d.D.expand.into S \ univ.filter fun b => (d.D.unbit b).1 ∈ d.θ
  have hθ : ∀ b, (d.D.unbit b).1 ∈ (d.θ ∪ univ.filter fun g => d.D.isConst g = true) ↔ (d.D.unbit b).1 ∈ d.θ :=
    fun b => ⟨fun h => (mem_union.1 h).elim id fun h' => by simp [hc] at h', mem_union_left _⟩
  ext a
  constructor
  · intro ha
    obtain ⟨hin, hout⟩ := mem_sdiff.1 ha
    obtain ⟨haS, g, hg, hag⟩ := (mem_filter.1 hin).2
    rw [d.D.constInputs_expand_reads hc] at hag
    refine mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, haS, g, hg, hag⟩, fun h => hout ?_⟩
    exact mem_filter.2 ⟨mem_univ a, (hθ a).2 (mem_filter.1 h).2⟩
  · intro ha
    obtain ⟨hin, hout⟩ := mem_sdiff.1 ha
    obtain ⟨haS, g, hg, hag⟩ := (mem_filter.1 hin).2
    rw [← d.D.constInputs_expand_reads hc] at hag
    refine mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ a, haS, g, hg, hag⟩, fun h => hout ?_⟩
    exact mem_filter.2 ⟨mem_univ a, (hθ a).1 (mem_filter.1 h).2⟩

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
          · exact mem_filter.2 ⟨mem_univ _, h⟩
      _ = d.D.width (d.unregistered (P.gates u)) := by
          rw [d.D.card_filter_unbit (· ∈ d.unregistered (P.gates u)), sum_univ_filter_mem]
          rfl

end Verity.SecurityProofs

namespace Verity.Boolean.Circuit

/-- Flock's circuits are Boolean in the draft's sense when no gate is an empty row. -/
theorem toCircuit_boolean (B : Circuit) (h : ∀ g, B.op g ≠ .row [] []) : B.toCircuit.Boolean where
  one_bit _ := rfl
  fanin_pos g hg := by
    show ((B.op g).args.toFinset).Nonempty
    rw [List.toFinset_nonempty_iff]
    have hg' : (B.op g).isInput = false := hg
    have := h g
    cases hop : B.op g <;> simp_all [Op.args, Op.isInput]

end Verity.Boolean.Circuit
