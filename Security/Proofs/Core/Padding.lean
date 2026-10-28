import Definitions.Core.Guarantees.Formalism
import Proofs.Core.Formalism
import Proofs.Core.Expand
import Mathlib.Tactic.Linarith

/-!
# Proofs of padding and of Theorem 3 at the serving blocks

* `Pad.unregistered`: a padded proof block reads the unregistered gates its proof block read;
* `PadBuild`: the padding of a Boolean declaration's proof blocks, `n u` gates into proof block `u`. `d`'s gates come
  first, then one registered input gate `z` (the constant the padding reads), then the padding gates, each reading `z`
  and computing `0`. `Padding` pads the chosen proof blocks to `c` gates and the others by none.
-/

namespace Verity

open Finset

/-- A padded proof block reads the unregistered gates its proof block read. -/
theorem Pad.unregistered {d d' : Registered} {B : ℕ} {P : Partition d.D B} {P' : Partition d'.D B}
    (X : Pad d P d' P') (u : Fin B) :
    d'.unregistered (P'.gates u) = (d.unregistered (P.gates u)).map ⟨X.M.emb, X.M.emb_inj⟩ := by
  ext a'
  simp only [Registered.unregistered, mem_sdiff, Circuit.into, mem_filter, mem_univ, true_and, mem_map,
    Function.Embedding.coeFn_mk]
  constructor
  · rintro ⟨⟨hout, g', hg', hr⟩, hθ⟩
    by_cases hrange : g' ∈ Set.range X.M.emb
    · obtain ⟨g, rfl⟩ := hrange
      rw [P'.mem_gates, X.unit_emb, ← P.mem_gates] at hg'
      rw [X.M.reads_emb g (P.isInput_of_mem_gates hg'), mem_map] at hr
      obtain ⟨a, ha, rfl⟩ := hr
      exact ⟨a, ⟨⟨fun h => hout (P'.mem_gates.2 ((X.unit_emb a).trans (P.mem_gates.1 h))), g, hg', ha⟩,
        fun h => hθ ((X.θ_emb a).2 h)⟩, rfl⟩
    · exact absurd (X.pad_reads g' hrange a' hr) hθ
  · rintro ⟨a, ⟨⟨hout, g, hg, ha⟩, hθ⟩, rfl⟩
    refine ⟨⟨fun h => hout (P.mem_gates.2 ((X.unit_emb a).symm.trans (P'.mem_gates.1 h))), X.M.emb g,
      P'.mem_gates.2 ((X.unit_emb g).trans (P.mem_gates.1 hg)), ?_⟩, fun h => hθ ((X.θ_emb a).1 h)⟩
    · rw [X.M.reads_emb g (P.isInput_of_mem_gates hg)]
      exact mem_map_of_mem _ ha

namespace PadBuild

variable {d : Registered} {B : ℕ} (P : Partition d.D B) (n : Fin B → ℕ)

/-- The number of padding gates: `n u` in proof block `u`. -/
abbrev T {d : Registered} {B : ℕ} (_P : Partition d.D B) (n : Fin B → ℕ) : ℕ := ∑ u : Fin B, n u

/-- `d`'s gate `g`. -/
@[reducible] def emb (g : Fin d.D.N) : Fin (d.D.N + 1 + T P n) := ⟨g, by have := g.2; omega⟩

/-- The registered input gate the padding reads. -/
@[reducible] def z : Fin (d.D.N + 1 + T P n) := ⟨d.D.N, by omega⟩

/-- Padding gate `p`. -/
@[reducible] def pad (p : Fin (T P n)) : Fin (d.D.N + 1 + T P n) := ⟨d.D.N + 1 + p, by have := p.2; omega⟩

theorem emb_inj : Function.Injective (emb P n) := fun _ _ h => Fin.ext (congrArg Fin.val h :)

/-- `emb` as an embedding. -/
@[reducible] def embE : Fin d.D.N ↪ Fin (d.D.N + 1 + T P n) := ⟨emb P n, emb_inj P n⟩

theorem pad_inj : Function.Injective (pad P n) := fun a b h => by
  have := congrArg Fin.val h
  simp only at this
  exact Fin.ext (by omega)

/-- `pad` as an embedding. -/
def padE : Fin (T P n) ↪ Fin (d.D.N + 1 + T P n) := ⟨pad P n, pad_inj P n⟩

theorem slot (b : Fin (d.D.N + 1 + T P n)) : (∃ g, b = emb P n g) ∨ b = z P n ∨ ∃ p, b = pad P n p := by
  have := b.2
  by_cases h : b.val < d.D.N
  · exact Or.inl ⟨⟨b, h⟩, Fin.ext rfl⟩
  · by_cases h' : b.val = d.D.N
    · exact Or.inr (Or.inl (Fin.ext h'))
    · exact Or.inr (Or.inr ⟨⟨b - (d.D.N + 1), by omega⟩, Fin.ext (by simp only; omega)⟩)

theorem emb_ne_z (g : Fin d.D.N) : emb P n g ≠ z P n := fun h => by
  have := congrArg Fin.val h
  simp only at this
  omega

theorem emb_ne_pad (g : Fin d.D.N) (p : Fin (T P n)) : emb P n g ≠ pad P n p := fun h => by
  have := congrArg Fin.val h
  simp only at this
  omega

theorem z_ne_pad (p : Fin (T P n)) : z P n ≠ pad P n p := fun h => by
  have := congrArg Fin.val h
  simp only at this
  omega

/-- Which gates are inputs: `d`'s inputs and `z`. -/
def isIn (b : Fin (d.D.N + 1 + T P n)) : Bool :=
  if h : b.val < d.D.N then d.D.isInput ⟨b, h⟩ else decide (b.val = d.D.N)

/-- What each gate reads: `d`'s gates their wires, the padding `z`. -/
def rd (b : Fin (d.D.N + 1 + T P n)) : Finset (Fin (d.D.N + 1 + T P n)) :=
  if h : b.val < d.D.N then (d.D.reads ⟨b, h⟩).map (embE P n) else if b.val = d.D.N then ∅ else {z P n}

/-- The proof blocks: `d`'s gates in theirs, `z` in none, padding gate `p` in the proof block of its index. -/
def unitF (b : Fin (d.D.N + 1 + T P n)) : Option (Fin B) :=
  if h : b.val < d.D.N then P.unit ⟨b, h⟩
  else if _ : b.val = d.D.N then none
  else some (finSigmaFinEquiv.symm (⟨b.val - (d.D.N + 1), by have := b.2; omega⟩ : Fin (T P n))).1

theorem isIn_emb (g : Fin d.D.N) : isIn P n (emb P n g) = d.D.isInput g := by
  rw [isIn, dif_pos (show (emb P n g).val < d.D.N from g.2)]

theorem isIn_z : isIn P n (z P n) = true := by
  rw [isIn, dif_neg (show ¬ (z P n).val < d.D.N from lt_irrefl _)]
  simp

theorem isIn_pad (p : Fin (T P n)) : isIn P n (pad P n p) = false := by
  rw [isIn, dif_neg (show ¬ (pad P n p).val < d.D.N by simp only; omega), decide_eq_false_iff_not]
  simp only
  omega

theorem rd_emb (g : Fin d.D.N) : rd P n (emb P n g) = (d.D.reads g).map (embE P n) := by
  rw [rd, dif_pos (show (emb P n g).val < d.D.N from g.2)]

theorem rd_z : rd P n (z P n) = ∅ := by
  rw [rd, dif_neg (show ¬ (z P n).val < d.D.N from lt_irrefl _), if_pos (show (z P n).val = d.D.N from rfl)]

theorem rd_pad (p : Fin (T P n)) : rd P n (pad P n p) = {z P n} := by
  rw [rd, dif_neg (show ¬ (pad P n p).val < d.D.N by simp only; omega),
    if_neg (show ¬ (pad P n p).val = d.D.N by simp only; omega)]

theorem unitF_emb (g : Fin d.D.N) : unitF P n (emb P n g) = P.unit g := by
  rw [unitF, dif_pos (show (emb P n g).val < d.D.N from g.2)]

theorem unitF_z : unitF P n (z P n) = none := by
  rw [unitF, dif_neg (show ¬ (z P n).val < d.D.N from lt_irrefl _), dif_pos (show (z P n).val = d.D.N from rfl)]

theorem unitF_pad (p : Fin (T P n)) : unitF P n (pad P n p) = some (finSigmaFinEquiv.symm p).1 := by
  rw [unitF, dif_neg (show ¬ (pad P n p).val < d.D.N by simp only; omega),
    dif_neg (show ¬ (pad P n p).val = d.D.N by simp only; omega)]
  congr 3
  exact Fin.ext (by simp)

variable (hB : d.D.Boolean)

/-- The padded circuit. -/
@[reducible] def circuit : Circuit where
  N := d.D.N + 1 + T P n
  w _ := 1
  isInput := isIn P n
  reads := rd P n
  reads_lt b a ha := by
    rcases slot P n b with ⟨g, rfl⟩ | rfl | ⟨p, rfl⟩
    · rw [rd_emb, mem_map] at ha
      obtain ⟨a, ha, rfl⟩ := ha
      exact (d.D.reads_lt g a ha :)
    · simp [rd_z] at ha
    · rw [rd_pad, mem_singleton] at ha
      subst ha
      show d.D.N < d.D.N + 1 + p
      omega
  input_reads b hb := by
    rcases slot P n b with ⟨g, rfl⟩ | rfl | ⟨p, rfl⟩
    · rw [isIn_emb] at hb
      rw [rd_emb, d.D.input_reads g hb, map_empty]
    · exact rd_z P n
    · simp [isIn_pad] at hb
  op b v := if h : b.val < d.D.N then
      (d.D.op ⟨b, h⟩ fun a => (v (emb P n a)).cast (hB.one_bit a).symm).cast (hB.one_bit _)
    else 0
  op_congr b v v' h := by
    by_cases hb : b.val < d.D.N
    · rw [dif_pos hb, dif_pos hb]
      congr 1
      refine d.D.op_congr _ _ _ fun a ha => ?_
      rw [h (emb P n a) (by rw [rd, dif_pos hb]; exact mem_map_of_mem _ ha)]
    · rw [dif_neg hb, dif_neg hb]
  outputs := d.D.outputs.map (embE P n)

/-- The padded declaration: `d`'s registered gates and `z`. -/
@[reducible] def decl : Registered :=
  .ofCircuit (circuit P n hB) (d.θ.map (embE P n) ∪ {z P n}) fun b hb => by
    rcases mem_union.1 hb with hb | hb
    · obtain ⟨g, hg, rfl⟩ := mem_map.1 hb
      show isIn P n (emb P n g) = true
      rw [isIn_emb]
      exact d.θ_input g hg
    · rw [mem_singleton.1 hb]
      exact isIn_z P n

/-- The padded partition. -/
def part : Partition (decl P n hB).D B where
  unit := unitF P n
  input_iff b := by
    show unitF P n b = none ↔ isIn P n b = true
    rcases slot P n b with ⟨g, rfl⟩ | rfl | ⟨p, rfl⟩
    · rw [unitF_emb, isIn_emb]
      exact P.input_iff g
    · simp [unitF_z, isIn_z]
    · simp [unitF_pad, isIn_pad]

/-- The padding is a `Pad`. -/
def pad_ : Pad d P (decl P n hB) (part P n hB) where
  M :=
    { emb := emb P n
      emb_inj := emb_inj P n
      emb_w g := (hB.one_bit g).symm
      isInput_emb g := isIn_emb P n g
      reads_emb g _ := rd_emb P n g
      op_emb g _ v := by
        show (((if h : (emb P n g).val < d.D.N then _ else 0) : BitVec 1)).cast _ = _
        rw [dif_pos (show (emb P n g).val < d.D.N from g.2)]
        simp only [BitVec.cast_cast, BitVec.cast_eq]
      outputs_emb o ho := mem_map_of_mem (embE P n) ho }
  θ_emb g := by
    show emb P n g ∈ d.θ.map (embE P n) ∪ {z P n} ↔ _
    rw [mem_union, mem_singleton, or_iff_left (emb_ne_z P n g)]
    exact mem_map' (embE P n)
  unit_emb g := unitF_emb P n g
  pad_input g' hg' hi := by
    show g' ∈ d.θ.map (embE P n) ∪ {z P n}
    rcases slot P n g' with ⟨g, rfl⟩ | rfl | ⟨p, rfl⟩
    · exact absurd ⟨g, rfl⟩ hg'
    · exact mem_union_right _ (mem_singleton_self _)
    · exact absurd hi (by simp [isIn_pad])
  pad_reads g' hg' a ha := by
    show a ∈ d.θ.map (embE P n) ∪ {z P n}
    change a ∈ rd P n g' at ha
    rcases slot P n g' with ⟨g, rfl⟩ | rfl | ⟨p, rfl⟩
    · exact absurd ⟨g, rfl⟩ hg'
    · simp [rd_z] at ha
    · rw [rd_pad, mem_singleton] at ha
      exact mem_union_right _ (mem_singleton.2 ha)

theorem gates_part (u : Fin B) :
    (part P n hB).gates u =
      (P.gates u).map (embE P n) ∪ (univ.filter fun p : Fin (T P n) => (finSigmaFinEquiv.symm p).1 = u).map
        (padE P n) := by
  ext b
  rw [Partition.mem_gates, mem_union, mem_map, mem_map]
  change unitF P n b = some u ↔ _
  rcases slot P n b with ⟨g, rfl⟩ | rfl | ⟨p, rfl⟩
  · rw [unitF_emb, ← P.mem_gates]
    constructor
    · exact fun h => Or.inl ⟨g, h, rfl⟩
    · rintro (⟨g', hg', he⟩ | ⟨p, _, he⟩)
      · rwa [(emb_inj P n he :)] at hg'
      · exact absurd he.symm (emb_ne_pad P n g p)
  · rw [unitF_z]
    simp only [reduceCtorEq, false_iff, not_or, not_exists, not_and]
    exact ⟨fun g _ he => emb_ne_z P n g he, fun p _ he => z_ne_pad P n p he.symm⟩
  · rw [unitF_pad, Option.some_inj]
    constructor
    · exact fun h => Or.inr ⟨p, mem_filter.2 ⟨mem_univ _, h⟩, rfl⟩
    · rintro (⟨g, _, he⟩ | ⟨p', hp', he⟩)
      · exact absurd he (emb_ne_pad P n g p)
      · rw [← (pad_inj P n he :)]
        exact (mem_filter.1 hp').2

theorem card_gates_part (u : Fin B) : ((part P n hB).gates u).card = (P.gates u).card + n u := by
  rw [gates_part, card_union_of_disjoint, card_map, card_map]
  · rw [card_filter_fst (n := n) (· = u)]
    simp only [sum_filter, sum_ite_eq', mem_univ, if_true]
  · rw [disjoint_left]
    rintro b hb hb'
    obtain ⟨g, _, rfl⟩ := mem_map.1 hb
    obtain ⟨p, _, he⟩ := mem_map.1 hb'
    exact emb_ne_pad P n g p he.symm

end PadBuild

end Verity

namespace Verity.SecurityProofs

open Finset

theorem Padding : Guarantees.Padding := by
  classical
  intro d B c P S hB hc
  let n : Fin B → ℕ := fun u => if u ∈ S then c - (P.gates u).card else 0
  refine ⟨PadBuild.decl P n hB, PadBuild.part P n hB, ⟨PadBuild.pad_ P n hB⟩, ?_, fun u => ⟨?_, ?_⟩, fun u => ?_⟩
  · exact
      { one_bit := fun _ => rfl
        fanin_pos := fun b hb => by
          change PadBuild.isIn P n b = false at hb
          change (PadBuild.rd P n b).Nonempty
          rcases PadBuild.slot P n b with ⟨g, rfl⟩ | rfl | ⟨p, rfl⟩
          · rw [PadBuild.isIn_emb] at hb
            rw [PadBuild.rd_emb]
            exact (hB.fanin_pos g hb).map
          · simp [PadBuild.isIn_z] at hb
          · rw [PadBuild.rd_pad]
            exact singleton_nonempty _ }
  · intro hu
    rw [PadBuild.card_gates_part]
    have := hc u hu
    simp only [n, if_pos hu]
    omega
  · intro hu
    rw [PadBuild.card_gates_part]
    simp [n, hu]
  · rw [(PadBuild.pad_ P n hB).unregistered u, card_map]

theorem ServingBlockSum : Guarantees.ServingBlockSum := by
  intro C B P S V ω qS qI qDark hV hS hI hD
  rw [P.sum_blocks hV ω, ← sum_sdiff (subset_univ S)]
  have : ∑ u ∈ S, ∑ v ∈ V ∩ P.gates u, ω v ≤ ∑ u ∈ S, min (qS u) (qI u) :=
    sum_le_sum fun u hu => le_min (hS u hu) (hI u hu)
  linarith

theorem InferenceCap : Guarantees.InferenceCap := by
  classical
  intro C nf nc Pf Pc r V U ω q hV hq s
  have hsplit : ∑ v ∈ V ∩ Pc.gates s, ω v =
      ∑ v ∈ (V ∩ Pc.gates s) \ U, ω v + ∑ v ∈ (V ∩ U) ∩ Pc.gates s, ω v := by
    have hsub : (V ∩ U) ∩ Pc.gates s ⊆ V ∩ Pc.gates s := fun v hv => by
      simp only [mem_inter] at hv ⊢
      exact ⟨hv.1.1, hv.2⟩
    rw [← sum_sdiff hsub, show (V ∩ Pc.gates s) \ ((V ∩ U) ∩ Pc.gates s) = (V ∩ Pc.gates s) \ U by
      ext v
      simp only [mem_sdiff, mem_inter]
      tauto]
  have hfine : (V ∩ Pc.gates s) \ U =
      (univ.filter fun i => r.parent i = s).biUnion fun i => (V \ U) ∩ Pf.gates i := by
    ext v
    simp only [mem_sdiff, mem_inter, mem_biUnion, mem_filter, mem_univ, true_and, Partition.mem_gates]
    constructor
    · rintro ⟨⟨hv, hs⟩, hu⟩
      have hi : C.isInput v = false := by simpa [Circuit.computed] using hV hv
      obtain ⟨i, hi'⟩ : ∃ i, Pf.unit v = some i :=
        Option.ne_none_iff_exists'.1 fun h => by simp [(Pf.input_iff v).1 h] at hi
      refine ⟨i, ?_, ⟨hv, hu⟩, hi'⟩
      have := r.sub v
      rw [hs, hi', Option.map_some] at this
      exact (Option.some_injective _ this).symm
    · rintro ⟨i, hi, ⟨hv, hu⟩, hi'⟩
      refine ⟨⟨hv, ?_⟩, hu⟩
      rw [r.sub v, hi', Option.map_some, hi]
  have hdisj : Set.PairwiseDisjoint ((univ.filter fun i => r.parent i = s) : Set (Fin nf))
      (fun i => (V \ U) ∩ Pf.gates i) := by
    intro i _ j _ hij
    refine disjoint_left.2 fun v hvi hvj => hij ?_
    rw [mem_inter, Partition.mem_gates] at hvi hvj
    exact Option.some_injective _ (hvi.2.symm.trans hvj.2)
  rw [hsplit, hfine, sum_biUnion hdisj]
  have : ∑ i ∈ univ.filter (fun i => r.parent i = s), ∑ v ∈ (V \ U) ∩ Pf.gates i, ω v ≤
      ((univ.filter fun i => r.parent i = s).card : ℝ) * q := by
    rw [← nsmul_eq_mul]
    exact sum_le_card_nsmul _ _ _ fun i _ => hq i
  linarith

end Verity.SecurityProofs
