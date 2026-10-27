import Proofs.InferenceOnly.Ldl

/-!
# Local description length: the serving-scale attack in miniature, and `HeldRate`'s witnesses

One explicit circuit (the held-layers attack, at one bit a value):
two tokens' entries `x₀ x₁` and three layers' weights `w₀ w₁ w₂` (gates 0–4, one bit each, nothing registered), and
one block that takes both tokens through the three layers, each layer's product `a ∧ w` (gates 5–10). The block reads
5 bits; its 6 counted products read 6 activations, 4 of them computed inside it from the other two and the weights.

* `activationRate_recompute`: for every decoder that evaluates blocks, `ActivationRate 1 0` fails there, since Lemma 1
  caps the tuple's description at 5 bits and the premise asks for 6. This is the attack: a block holding the layers
  computes the activations it is charged for.
* `heldRate_honest`, `pairBound_honest`: at `βw = βe = 1`, `ℓ = 0` and the identity decoder, `HeldRate` holds there
  (3 weights and 2 entries, 5 bits), and so does every other hypothesis of `PairBound`, over a nonempty count.
* `pairBound_nontrivial`: at `βw = βe = 10` every hypothesis of `PairBound` but `HeldRate` holds and its conclusion
  fails, so the premise carries the bound.
* `heldRate_nontrivial`: `HeldRate 1 1 0` fails for the decoder that prints the tuple on the empty program.
-/

namespace InferenceOnly.Ldl.SecurityProofs.Witness

open Verity Finset
open scoped ENNReal

/-- Each computed gate's activation operand: a token's entry at the first layer, the previous layer's product after. -/
def actOf : Fin 11 → Fin 11 := ![0, 0, 0, 0, 0, 0, 1, 5, 6, 7, 8]

/-- Each computed gate's weight operand, its layer's weight. -/
def wtOf : Fin 11 → Fin 11 := ![0, 0, 0, 0, 0, 2, 2, 3, 3, 4, 4]

/-- Each computed gate's token. -/
def srcOf : Fin 11 → Fin 2 := ![0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1]

/-- The circuit: gates 0–4 are inputs, gate `g ≥ 5` is `actOf g ∧ wtOf g`. -/
def circuit : Circuit where
  N := 11
  w := fun _ => 1
  isInput := fun g => decide (g.val < 5)
  reads := fun g => if g.val < 5 then ∅ else {actOf g, wtOf g}
  reads_lt := by decide
  input_reads := fun g hg => by simpa using hg
  op := fun g v => if g.val < 5 then 0 else v (actOf g) &&& v (wtOf g)
  op_congr := fun g v v' h => by
    by_cases hg : g.val < 5
    · simp [hg]
    · simp only [hg, ite_false]
      rw [h (actOf g) (by simp [hg]), h (wtOf g) (by simp [hg])]
  outputs := {9, 10}

/-- Nothing registered. -/
def decl : Declaration := ⟨circuit, ∅, by simp⟩

/-- One block holding every computed gate. -/
def part : Partition decl.D 1 where
  unit := fun g => if g.val < 5 then none else some 0
  input_iff := fun g => by by_cases hg : g.val < 5 <;> simp [decl, circuit, hg]

def reg : Registration decl 1 := ⟨fun _ => 0, part⟩

/-- Every computed gate is a counted product, its own representative. -/
def prods : Products decl.D (Fin 11) (Fin 11) where
  J := univ.filter fun g => ¬ g.val < 5
  rep := id
  rep_inj := Set.injOn_id _
  rep_op := fun p hp => by simpa [decl, circuit] using hp
  act := actOf

/-- The block's tuple for `HeldRate`: the values of the weights and entries it reads, gates 0–4. -/
def tuple : Fin 1 → decl.D.Transcript → List Bool :=
  fun _ X => List.ofFn fun i : Fin 5 => (X (Fin.castLE (by decide) i)).getLsbD 0

/-- The decoder that outputs its program. -/
def idDec : Decoder := fun p _ => some p

/-- The decoder that outputs `y` on every program, the empty one included. -/
def constDec (y : List Bool) : Decoder := fun _ _ => some y

theorem LDL_idDec (t : ℕ) (y : List Bool) : LDL idDec t y = y.length :=
  le_antisymm (LDL_le rfl) (le_iInf₂ fun p hp => by cases Option.some.inj hp; exact le_rfl)

theorem LDL_constDec (t : ℕ) (y : List Bool) : LDL (constDec y) t y = 0 := by
  simpa using LDL_le (D := constDec y) (t := t) (y := y) (p := []) rfl

theorem gates_eq : reg.P.gates 0 = univ.filter fun g => ¬ g.val < 5 := by decide

theorem inBlock_eq : prods.inBlock reg.P 0 = prods.J := by decide

theorem into_eq : decl.D.into (reg.P.gates 0) \ decl.θ = univ.filter fun g => g.val < 5 := by decide

theorem novelReads_eq : novelReads decl (reg.P.gates 0) = 5 := by
  unfold novelReads
  rw [into_eq]
  decide

theorem size_computed : decl.D.size decl.D.computed = 6 := by decide

theorem acts_card : (prods.acts reg.P 0).card = 6 := by decide

theorem held_card : ((prods.inBlock reg.P 0).image wtOf).card = 3 := by decide

theorem entries_card : ((prods.inBlock reg.P 0).image srcOf).card = 2 := by decide

theorem J_card : prods.J.card = 6 := by decide

theorem pairs_inj : Set.InjOn (fun p => (wtOf p, srcOf p)) prods.J :=
  Finset.card_image_iff.1 (by decide)

theorem ckPartition : CKPartition decl reg.P 6 5 := by
  refine ⟨fun u => ?_, fun u => ?_⟩
  · rw [Subsingleton.elim u 0, gates_eq]
    decide
  · rw [Subsingleton.elim u 0, novelReads_eq]

theorem consistent (X : decl.D.Transcript) : reg.Consistent X := fun g hg => absurd hg (Finset.notMem_empty g)

theorem evaluates : Assumptions.EvaluatesBlocks idDec 0 reg tuple 0 := by
  refine ⟨tuple, fun u X _ => ⟨fun X' _ hX' => ?_, ?_, ?_⟩⟩
  · simp only [tuple]
    congr 1
    funext i
    rw [hX' _ (by rw [Subsingleton.elim u 0, into_eq]; simp [Fin.castLE])]
  · rw [Subsingleton.elim u 0, novelReads_eq]
    simp [tuple]
  · simp only [idDec, tuple]
    congr 2
    funext i
    rw [reg.P.localEval_of_not_mem X (u := u) (g := Fin.castLE (by decide) i) (by simp [reg, part, Fin.castLE])]

theorem heldRate_honest (inp : decl.D.Transcript) :
    Assumptions.HeldRate 1 1 0 0 idDec reg prods wtOf srcOf tuple (decl.D.eval inp) := fun u => by
  rw [Subsingleton.elim u 0, held_card, entries_card, LDL_idDec]
  simp [tuple]
  norm_num

/-- The attack: no decoder that evaluates blocks meets `ActivationRate 1 0` here. -/
theorem activationRate_recompute (D : Decoder) (t : ℕ) (a : Fin 1 → decl.D.Transcript → List Bool)
    (inp : decl.D.Transcript) (hev : Assumptions.EvaluatesBlocks D t reg a 0) :
    ¬ Assumptions.ActivationRate 1 0 t D reg prods a (decl.D.eval inp) := fun hrate => by
  have h1 := BlockDescription decl 1 5 reg D t a 0 inp ckPartition.2 (consistent _) hev 0
  have h2 : ENNReal.ofReal (1 * (prods.acts reg.P 0).card - 0) ≤ ((5 + Nat.clog 2 1 + 0 : ℕ) : ℝ≥0∞) :=
    (hrate 0).trans (by exact_mod_cast ENat.toENNReal_le.2 h1)
  rw [acts_card, ENNReal.ofReal_le_natCast] at h2
  norm_num at h2

theorem pairBound_honest (inp : decl.D.Transcript) :
    CKPartition decl reg.P 6 5 ∧ reg.Consistent (decl.D.eval inp) ∧
      Set.InjOn (fun p => (wtOf p, srcOf p)) prods.J ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      Assumptions.EvaluatesBlocks idDec 0 reg tuple 0 ∧
      Assumptions.HeldRate 1 1 0 0 idDec reg prods wtOf srcOf tuple (decl.D.eval inp) ∧ prods.J.Nonempty :=
  ⟨ckPartition, consistent _, pairs_inj, one_pos, one_pos, one_pos, evaluates, heldRate_honest inp,
    Finset.card_pos.1 (by rw [J_card]; norm_num)⟩

theorem pairBound_nontrivial (inp : decl.D.Transcript) :
    CKPartition decl reg.P 6 5 ∧ reg.Consistent (decl.D.eval inp) ∧
      Set.InjOn (fun p => (wtOf p, srcOf p)) prods.J ∧ (0 : ℝ) < 10 ∧ (0 : ℝ) < 10 ∧ (0 : ℝ) < 1 ∧
      Assumptions.EvaluatesBlocks idDec 0 reg tuple 0 ∧
      ¬ (4 * ((6 : ℕ) : ℝ) * 10 * 10 * prods.J.card / (1 * (((5 : ℕ) : ℝ) + Nat.clog 2 1 + ((0 : ℕ) : ℝ) + 0) ^ 2) ≤
        decl.D.size decl.D.computed / 1) := by
  refine ⟨ckPartition, consistent _, pairs_inj, by norm_num, by norm_num, one_pos, evaluates, ?_⟩
  rw [J_card, size_computed]
  norm_num

theorem heldRate_nontrivial (inp : decl.D.Transcript) :
    ¬ Assumptions.HeldRate 1 1 0 0 (constDec (tuple 0 (decl.D.eval inp))) reg prods wtOf srcOf tuple
      (decl.D.eval inp) := fun h => by
  have := h 0
  rw [held_card, entries_card, LDL_constDec] at this
  norm_num at this

end InferenceOnly.Ldl.SecurityProofs.Witness
