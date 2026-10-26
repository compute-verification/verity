import Proofs.Accounting.ServingBlocks
import Mathlib.Data.Fin.VecNotation

/-!
# The serving-block bound: honest and nontrivial lemmas

Each guarantee of `Accounting.Guarantees.ServingBlocks` has two lemmas here. `<Guarantee>_honest`: its premises hold
together at `Honest`, a small instance of a serving-block layout. `<Guarantee>_nontrivial`: at an explicit circuit every
premise but one named premise holds, that one fails, and the counted weight is above the bound's right-hand side, so
the guarantee is false without it. The `example`s at the end apply each guarantee to its lemmas, so the lemmas are the
premises as the guarantee states them.

`Honest` is one server, leaf `0`, wiped at tick `1` of `[0, 2)`, and one dark leaf, leaf `1`. Gate 0 is a registered
input on the server, gate 1 copies it at tick 0, gate 2 negates gate 1 at tick 1 and gate 3 is a constant at the dark
leaf. Its two counted operations, each of weight 1, are gate 2's, computed by gates 1 and 2 and so split across the wipe
(`Honest.straddles`: the operation `OperationBodies` allows and the boundary term pays for), and gate 3's. The serving
blocks are `{1}` and `{2}`, each counted with itself as its one inference block, and the dark leaves hold `{3}`.
`q(c, k) = c` is a per-block cap at every size, as each operation's body holds its representative and weighs 1. It rests
on no run and no `art:`. It is not the deployed partition at the onsite reference numbers, whose premises are Layer 2's
envelopes with their probabilities.

The counter-examples, each on one server:
* `Counter`: one constant gate computing one operation of weight 1, with the cap `q(c, 0) = 0` and `q(c, k) = c` for
  `k > 0`. A cap at `k = 0` says a block reading nothing computes nothing, which the constant refutes. Without
  `PerBlockCap` at the serving block's size `(1, 0)` `ServingBlockBound` gives `0`, and without it at the inference
  size `(1, 0)` so does `ServingBlockBoundRemote`. Without `Alignment`, with no blocks and the gate certified,
  `UncertifiedElementary` gives `0`.
* `Straddle`: two constants, gate 1 reading gate 0 after the wipe, and one operation computed by both. No block of one
  gate computes it, so the cap is `0`, and only the boundary term `ω_max μ k_S` covers it: without `Overlap`, at
  `μ = 0`, `ServingBlockBound` gives `0`.
* `Detached`: `Straddle` with gate 1 reading nothing, so the operation takes in a gate it never reads. Without
  `OperationBodies` the serving block after the wipe reads nothing (`k_S = 0`) yet is assigned the operation, and
  `ServingBlockBound` gives `0`.
-/

namespace Accounting.SecurityProofs.ServingBlocks

open Verity Finset Accounting.ServingBlocks

variable {L σ : Type*}

theorem mem_block {A : Placed L} {X : σ → Finset L} {t : ℕ} {κ : σ → Cuts t} {S : Block κ} {g : Fin A.N} :
    g ∈ block A X κ S ↔
      A.isInput g = false ∧ A.x g ∈ X S.1 ∧ (κ S.1).τ S.2.castSucc ≤ A.t g ∧ A.t g < (κ S.1).τ S.2.succ := by
  simp only [block, SecurityProofs.Actual.mem_region, Set.mem_Ico]

namespace Counted

variable {C : Circuit} (J : Counted C)

/-- With every operation's body holding its representative and every weight at most 1, a set of gates computes at most
its number of gates wholly inside itself. -/
theorem inside_le_card (hv : ∀ v ∈ J.V, v ∈ J.body v) (hω : ∀ v ∈ J.V, J.ω v ≤ 1) (I : Finset (Fin C.N)) :
    J.inside I ≤ I.card := by
  unfold Counted.inside
  refine (Finset.sum_le_card_nsmul _ _ 1 fun v h => hω v (Finset.mem_of_mem_filter v h)).trans ?_
  rw [nsmul_one, Nat.cast_le]
  refine Finset.card_le_card fun v h => ?_
  rw [Finset.mem_filter] at h
  exact h.2 (hv v h.1)

/-- With every weight at most 1, what the blocks `B` leave of `R` is at most `R`'s number of gates. -/
theorem outside_le_card (hω : ∀ v ∈ J.V, J.ω v ≤ 1) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) : J.outside R B ≤ R.card := by
  unfold Counted.outside
  refine (Finset.sum_le_card_nsmul _ _ 1 fun v h => hω v (Finset.mem_of_mem_filter v h)).trans ?_
  rw [nsmul_one, Nat.cast_le]
  refine Finset.card_le_card fun v h => ?_
  rw [Finset.mem_filter] at h
  exact h.2.1

end Counted

/-- The cuts of `[0, J)` at every tick: the windows `[j, j + 1)`. -/
abbrev ticks (J : ℕ) : Cuts J := ⟨J, fun i => i.val, fun _ _ h => h, rfl, rfl⟩

namespace Honest

/-- Gate 0 is an input, gate 1 copies it, gate 2 negates gate 1 and gate 3 is a constant. -/
abbrev C : Circuit where
  N := 4
  w _ := 1
  isInput := ![true, false, false, false]
  reads := ![∅, {0}, {1}, ∅]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 1 then v 0 else if g = 2 then ~~~ v 1 else 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · simpa using hv 0 (by decide)
    · simpa using congrArg (~~~ ·) (hv 1 (by decide))
    · rfl
  outputs := {2, 3}

/-- Gates 0 to 2 on the server, leaf 0, gate 2 after its wipe at tick 1; gate 3 at the dark leaf, leaf 1. -/
abbrev A : Placed (Fin 2) where
  toCircuit := C
  x := ![0, 0, 0, 1]
  t := ![0, 0, 1, 0]
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p => p.2.1) h)

abbrev θ : Finset (Fin A.N) := {0}

abbrev dark : Finset (Fin 2) := {1}

abbrev X : Unit → Finset (Fin 2) := fun _ => {0}

abbrev κ : Unit → Cuts 2 := fun _ => ticks 2

/-- Gate 2's operation, computed by gates 1 and 2, and gate 3's, each of weight 1. -/
abbrev J : Counted A.toCircuit where
  V := {2, 3}
  body := ![∅, ∅, {1, 2}, {3}]
  ω _ := 1
  V_computed := by decide
  ω_nonneg _ _ := zero_le_one

noncomputable abbrev q : ℕ → ℕ → ℝ := fun c _ => c

noncomputable abbrev inf : (S : Block κ) → Fin 1 → Finset (Fin A.N) := fun S _ => block A X κ S

theorem block_zero : block A X κ ⟨(), 0⟩ = {1} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem block_one : block A X κ ⟨(), 1⟩ = {2} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem dark_eq : A.region dark Set.univ = {3} := by
  ext g
  rw [SecurityProofs.Actual.mem_region]
  simp only [Set.mem_univ, and_true]
  revert g
  decide

theorem forall_block {P : Block κ → Prop} (h0 : P ⟨(), 0⟩) (h1 : P ⟨(), 1⟩) : ∀ S, P S := by
  rintro ⟨⟨⟩, j⟩
  fin_cases j
  · exact h0
  · exact h1

theorem body_self : ∀ v ∈ J.V, v ∈ J.body v := by decide

theorem ω_le : ∀ v ∈ J.V, J.ω v ≤ 1 := fun _ _ => le_rfl

theorem cap (c k : ℕ) : PerBlockCap A.toCircuit θ J q c k := fun I _ hc _ =>
  (Counted.inside_le_card J body_self ω_le I).trans (by exact_mod_cast hc)

theorem blocks : ∀ S, A.size (block A X κ S) ≤ 1 ∧ A.width (A.into (block A X κ S) \ θ) ≤ 1 :=
  forall_block (by rw [block_zero]; decide) (by rw [block_one]; decide)

theorem card_block : ∀ S, (block A X κ S).card = 1 :=
  forall_block (by rw [block_zero]; rfl) (by rw [block_one]; rfl)

/-- Gate 2's operation is assigned to the serving block after the wipe and computed wholly inside none of its blocks. -/
theorem straddles : J.outside (block A X κ ⟨(), 1⟩) (inf ⟨(), 1⟩) = 1 := by
  simp only [inf, block_one, Counted.outside, Finset.sum_const, nsmul_eq_mul, mul_one, Nat.cast_eq_one]
  decide

end Honest

namespace Counter

/-- One computed gate, a constant. -/
abbrev C : Circuit where
  N := 1
  w _ := 1
  isInput _ := false
  reads _ := ∅
  reads_lt _ _ h := absurd h (Finset.notMem_empty _)
  input_reads _ h := absurd h (by decide)
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {0}

abbrev A : Placed Unit where
  toCircuit := C
  x _ := ()
  t _ := 0
  cell _ := 0
  cell_inj a b _ := Subsingleton.elim a b

abbrev θ : Finset (Fin A.N) := ∅

abbrev dark : Finset Unit := ∅

abbrev X : Unit → Finset Unit := fun _ => {()}

abbrev κ : Unit → Cuts 1 := fun _ => ticks 1

/-- The gate's operation, of weight 1. -/
abbrev J : Counted A.toCircuit where
  V := {0}
  body v := {v}
  ω _ := 1
  V_computed := by decide
  ω_nonneg _ _ := zero_le_one

/-- A cap at `k = 0` says a block reading nothing computes nothing. -/
noncomputable abbrev q : ℕ → ℕ → ℝ := fun c k => if k = 0 then 0 else c

noncomputable abbrev inf : (S : Block κ) → Fin 1 → Finset (Fin A.N) := fun S _ => block A X κ S

theorem block_eq (S : Block κ) : block A X κ S = {0} := by
  obtain ⟨⟨⟩, j⟩ := S
  fin_cases j
  ext g
  rw [mem_block]
  revert g
  decide

theorem dark_eq : A.region dark Set.univ = ∅ := by
  ext g
  rw [SecurityProofs.Actual.mem_region]
  simp

theorem total_eq : J.total = 1 := by simp [Counted.total]

theorem cap_one (c : ℕ) (hc : 1 ≤ c) : PerBlockCap A.toCircuit θ J q c 1 := fun I _ hI _ =>
  (Counted.inside_le_card J (by decide) (fun _ _ => le_rfl) I).trans (by
    have : (I.card : ℝ) ≤ 1 := by exact_mod_cast (Finset.card_le_univ I).trans (by simp)
    simp only [q, one_ne_zero, ↓reduceIte]
    exact this.trans (by exact_mod_cast hc))

theorem not_cap_zero : ¬ PerBlockCap A.toCircuit θ J q 1 0 := fun h => by
  have := h {0} (by decide) (by decide) (by decide)
  simp [Counted.inside, q] at this

theorem blocks (S : Block κ) : A.size (block A X κ S) ≤ 1 ∧ A.width (A.into (block A X κ S) \ θ) ≤ 0 := by
  rw [block_eq]
  decide

theorem inside_block (S : Block κ) : J.outside (block A X κ S) (inf S) = 0 := by
  simp only [inf, block_eq, Counted.outside, Finset.sum_const, nsmul_eq_mul, mul_one, Nat.cast_eq_zero]
  decide

theorem outside_dark : J.outside (A.region dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin A.N)) = 0 := by
  simp only [dark_eq, Counted.outside]
  simp

end Counter

namespace Straddle

/-- Two constants, gate 1 reading gate 0. -/
abbrev C : Circuit where
  N := 2
  w _ := 1
  isInput _ := false
  reads := ![∅, {0}]
  reads_lt := by decide
  input_reads _ h := absurd h (by decide)
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {1}

/-- Both gates on one server, gate 1 after its wipe at tick 1. -/
abbrev A : Placed Unit where
  toCircuit := C
  x _ := ()
  t := ![0, 1]
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p => p.2.1) h)

abbrev θ : Finset (Fin A.N) := ∅

abbrev dark : Finset Unit := ∅

abbrev X : Unit → Finset Unit := fun _ => {()}

abbrev κ : Unit → Cuts 2 := fun _ => ticks 2

/-- Gate 1's operation, computed by both gates and so split across the wipe, of weight 1. -/
abbrev J : Counted A.toCircuit where
  V := {1}
  body := ![∅, {0, 1}]
  ω _ := 1
  V_computed := by decide
  ω_nonneg _ _ := zero_le_one

/-- No block of one gate computes the operation wholly. -/
noncomputable abbrev q : ℕ → ℕ → ℝ := fun _ _ => 0

noncomputable abbrev inf : (S : Block κ) → Fin 1 → Finset (Fin A.N) := fun S _ => block A X κ S

theorem block_zero : block A X κ ⟨(), 0⟩ = {0} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem block_one : block A X κ ⟨(), 1⟩ = {1} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem dark_eq : A.region dark Set.univ = ∅ := by
  ext g
  rw [SecurityProofs.Actual.mem_region]
  simp

theorem forall_block {P : Block κ → Prop} (h0 : P ⟨(), 0⟩) (h1 : P ⟨(), 1⟩) : ∀ S, P S := by
  rintro ⟨⟨⟩, j⟩
  fin_cases j
  · exact h0
  · exact h1

theorem total_eq : J.total = 1 := by simp [Counted.total]

theorem cap (k : ℕ) : PerBlockCap A.toCircuit θ J q 1 k := fun I _ hc _ => by
  unfold Counted.inside
  refine le_of_eq (Finset.sum_eq_zero fun v hv => ?_)
  obtain ⟨hv, hsub⟩ := Finset.mem_filter.1 hv
  obtain rfl := Finset.mem_singleton.1 hv
  exact absurd ((Finset.card_le_card hsub).trans hc) (by decide)

theorem blocks : ∀ S, A.size (block A X κ S) ≤ 1 ∧ A.width (A.into (block A X κ S) \ θ) ≤ 1 :=
  forall_block (by rw [block_zero]; decide) (by rw [block_one]; decide)

theorem outside_le : ∀ S, J.outside (block A X κ S) (inf S) ≤ 1 := fun S =>
  (Counted.outside_le_card _ (fun _ _ => le_rfl) _ _).trans (by exact_mod_cast (blocks S).1)

theorem outside_dark : J.outside (A.region dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin A.N)) = 0 := by
  simp only [dark_eq, Counted.outside]
  simp

end Straddle

namespace Detached

/-- Two constants, reading nothing. -/
abbrev C : Circuit where
  N := 2
  w _ := 1
  isInput _ := false
  reads _ := ∅
  reads_lt _ _ h := absurd h (Finset.notMem_empty _)
  input_reads _ h := absurd h (by decide)
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {1}

/-- Both gates on one server, gate 1 after its wipe at tick 1. -/
abbrev A : Placed Unit where
  toCircuit := C
  x _ := ()
  t := ![0, 1]
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p => p.2.1) h)

abbrev θ : Finset (Fin A.N) := ∅

abbrev dark : Finset Unit := ∅

abbrev X : Unit → Finset Unit := fun _ => {()}

abbrev κ : Unit → Cuts 2 := fun _ => ticks 2

/-- An operation represented by gate 1 whose body takes in gate 0, which no gate of it reads, of weight 1. -/
abbrev J : Counted A.toCircuit where
  V := {1}
  body := ![∅, {0, 1}]
  ω _ := 1
  V_computed := by decide
  ω_nonneg _ _ := zero_le_one

/-- No block of one gate computes the operation wholly. -/
noncomputable abbrev q : ℕ → ℕ → ℝ := fun _ _ => 0

noncomputable abbrev inf : (S : Block κ) → Fin 1 → Finset (Fin A.N) := fun S _ => block A X κ S

theorem block_zero : block A X κ ⟨(), 0⟩ = {0} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem block_one : block A X κ ⟨(), 1⟩ = {1} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem dark_eq : A.region dark Set.univ = ∅ := by
  ext g
  rw [SecurityProofs.Actual.mem_region]
  simp

theorem forall_block {P : Block κ → Prop} (h0 : P ⟨(), 0⟩) (h1 : P ⟨(), 1⟩) : ∀ S, P S := by
  rintro ⟨⟨⟩, j⟩
  fin_cases j
  · exact h0
  · exact h1

theorem total_eq : J.total = 1 := by simp [Counted.total]

theorem cap (k : ℕ) : PerBlockCap A.toCircuit θ J q 1 k := fun I _ hc _ => by
  unfold Counted.inside
  refine le_of_eq (Finset.sum_eq_zero fun v hv => ?_)
  obtain ⟨hv, hsub⟩ := Finset.mem_filter.1 hv
  obtain rfl := Finset.mem_singleton.1 hv
  exact absurd ((Finset.card_le_card hsub).trans hc) (by decide)

theorem blocks : ∀ S, A.size (block A X κ S) ≤ 1 ∧ A.width (A.into (block A X κ S) \ θ) ≤ 0 :=
  forall_block (by rw [block_zero]; decide) (by rw [block_one]; decide)

theorem outside_le : ∀ S, J.outside (block A X κ S) (inf S) ≤ 1 := fun S =>
  (Counted.outside_le_card _ (fun _ _ => le_rfl) _ _).trans (by exact_mod_cast (blocks S).1)

theorem outside_dark : J.outside (A.region dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin A.N)) = 0 := by
  simp only [dark_eq, Counted.outside]
  simp

end Detached

/-- **The serving-block bound's premises hold** at `Honest`, at `(c_S, k_S) = (c, k) = (1, 1)`, one inference block per
serving block (the block itself), `ω_max = μ = 1`, `q_U(S) = 1` and `q_U(L_dark) = 1`. -/
theorem ServingBlockBound_honest :
    ServingBlocksPartition Honest.A Honest.θ Honest.J Honest.dark Honest.X Honest.κ (fun _ => 1) (fun _ => 1) ∧
    (∀ _ : Block Honest.κ, PerBlockCap Honest.A.toCircuit Honest.θ Honest.J Honest.q 1 1) ∧
    PerBlockCap Honest.A.toCircuit Honest.θ Honest.J Honest.q 1 1 ∧
    InferenceBlocks Honest.A.toCircuit Honest.θ (fun _ : Block Honest.κ => 1) Honest.inf 1 1 ∧
    OperationBodies Honest.A.toCircuit Honest.θ Honest.J ∧ Overlap Honest.J 1 ∧ (∀ g, 1 ≤ Honest.A.w g) ∧
    (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Honest.J.V, Honest.J.ω v ≤ 1) ∧
    (∀ S, UncertifiedCap Honest.J (block Honest.A Honest.X Honest.κ S) (Honest.inf S) 1) ∧
    UncertifiedCap Honest.J (Honest.A.region Honest.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Honest.A.N)) 1 := by
  refine ⟨⟨?_, Honest.blocks⟩, fun _ => Honest.cap 1 1, Honest.cap 1 1,
    fun S _ => ⟨block_subset _ _ _ S, Honest.blocks S⟩, ?_, ?_, fun _ => le_rfl, zero_le_one, Honest.ω_le,
    fun S => ?_, ?_⟩
  · unfold ServingBlocksCover; decide
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · exact (Counted.outside_le_card _ Honest.ω_le _ _).trans (by rw [Honest.card_block]; norm_num)
  · exact (Counted.outside_le_card _ Honest.ω_le _ _).trans (by rw [Honest.dark_eq]; norm_num)

/-- **The serving-block bound is false without the per-block cap at the serving blocks' size**: at `Counter`, with
`(c_S, k_S) = (1, 0)`, `(c, k) = (1, 1)`, one inference block, `ω_max = μ = 1` and `q_U = 0`, every other premise holds,
the cap at `(1, 0)` fails, and the counted weight `1` is above the right-hand side `0`. -/
theorem ServingBlockBound_nontrivial :
    ServingBlocksPartition Counter.A Counter.θ Counter.J Counter.dark Counter.X Counter.κ (fun _ => 1) (fun _ => 0) ∧
    ¬ (∀ _ : Block Counter.κ, PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 0) ∧
    PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 1 ∧
    InferenceBlocks Counter.A.toCircuit Counter.θ (fun _ : Block Counter.κ => 1) Counter.inf 1 1 ∧
    OperationBodies Counter.A.toCircuit Counter.θ Counter.J ∧ Overlap Counter.J 1 ∧ (∀ g, 1 ≤ Counter.A.w g) ∧
    (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Counter.J.V, Counter.J.ω v ≤ 1) ∧
    (∀ S, UncertifiedCap Counter.J (block Counter.A Counter.X Counter.κ S) (Counter.inf S) 0) ∧
    UncertifiedCap Counter.J (Counter.A.region Counter.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N))
      0 ∧
    ∑ _S : Block Counter.κ, min (Counter.q 1 0 + 1 * (1 : ℕ) * (0 : ℕ)) ((1 : ℕ) * Counter.q 1 1 + 0) + 0 <
      Counter.J.total := by
  refine ⟨⟨?_, Counter.blocks⟩, fun h => Counter.not_cap_zero (h ⟨(), 0⟩), Counter.cap_one 1 le_rfl,
    fun S _ => ⟨block_subset _ _ _ S, (Counter.blocks S).1, (Counter.blocks S).2.trans (Nat.zero_le 1)⟩, ?_, ?_,
    fun _ => le_rfl, zero_le_one, fun _ _ => le_rfl, fun S => (Counter.inside_block S).le, Counter.outside_dark.le, ?_⟩
  · unfold ServingBlocksCover; decide
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · rw [Counter.total_eq]
    simp [Counter.q]

/-- **The serving-block bound is false without operation bodies**: at `Detached`, whose operation takes in gate 0 though
no gate of it reads gate 0, with `(c_S, k_S) = (c, k) = (1, 0)`, one inference block, `ω_max = μ = 1`, `q_U(S) = 1` and
`q_U(L_dark) = 0`, every other premise holds and the counted weight `1` is above the right-hand side `0`: the serving
block after the wipe reads nothing, yet the operation is assigned to it. -/
theorem ServingBlockBound_nontrivial_bodies :
    ServingBlocksPartition Detached.A Detached.θ Detached.J Detached.dark Detached.X Detached.κ (fun _ => 1)
      (fun _ => 0) ∧
    (∀ _ : Block Detached.κ, PerBlockCap Detached.A.toCircuit Detached.θ Detached.J Detached.q 1 0) ∧
    PerBlockCap Detached.A.toCircuit Detached.θ Detached.J Detached.q 1 0 ∧
    InferenceBlocks Detached.A.toCircuit Detached.θ (fun _ : Block Detached.κ => 1) Detached.inf 1 0 ∧
    ¬ OperationBodies Detached.A.toCircuit Detached.θ Detached.J ∧ Overlap Detached.J 1 ∧
    (∀ g, 1 ≤ Detached.A.w g) ∧ (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Detached.J.V, Detached.J.ω v ≤ 1) ∧
    (∀ S, UncertifiedCap Detached.J (block Detached.A Detached.X Detached.κ S) (Detached.inf S) 1) ∧
    UncertifiedCap Detached.J (Detached.A.region Detached.dark Set.univ)
      (Fin.elim0 : Fin 0 → Finset (Fin Detached.A.N)) 0 ∧
    ∑ _S : Block Detached.κ, min (Detached.q 1 0 + 1 * (1 : ℕ) * (0 : ℕ)) ((1 : ℕ) * Detached.q 1 0 + 1) + 0 <
      Detached.J.total := by
  refine ⟨⟨?_, Detached.blocks⟩, fun _ => Detached.cap 0, Detached.cap 0,
    fun S _ => ⟨block_subset _ _ _ S, Detached.blocks S⟩, ?_, ?_, fun _ => le_rfl, zero_le_one, fun _ _ => le_rfl,
    Detached.outside_le, Detached.outside_dark.le, ?_⟩
  · unfold ServingBlocksCover; decide
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · rw [Detached.total_eq]
    simp [Detached.q]

/-- **The serving-block bound is false without overlap, and so without its boundary term**: at `Straddle`, whose
operation is split across the wipe, with `(c_S, k_S) = (c, k) = (1, 1)`, one inference block, `ω_max = 1`, `μ = 0`,
`q_U(S) = 1` and `q_U(L_dark) = 0`, every other premise holds, no gate is in at most `0` bodies, and the counted weight
`1` is above the right-hand side `0`. At `μ = 0` the boundary term `ω_max μ k_S` vanishes: the bound without it is
false where every premise holds at `μ = 1`. -/
theorem ServingBlockBound_nontrivial_overlap :
    ServingBlocksPartition Straddle.A Straddle.θ Straddle.J Straddle.dark Straddle.X Straddle.κ (fun _ => 1)
      (fun _ => 1) ∧
    (∀ _ : Block Straddle.κ, PerBlockCap Straddle.A.toCircuit Straddle.θ Straddle.J Straddle.q 1 1) ∧
    PerBlockCap Straddle.A.toCircuit Straddle.θ Straddle.J Straddle.q 1 1 ∧
    InferenceBlocks Straddle.A.toCircuit Straddle.θ (fun _ : Block Straddle.κ => 1) Straddle.inf 1 1 ∧
    OperationBodies Straddle.A.toCircuit Straddle.θ Straddle.J ∧ Overlap Straddle.J 1 ∧ ¬ Overlap Straddle.J 0 ∧
    (∀ g, 1 ≤ Straddle.A.w g) ∧ (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Straddle.J.V, Straddle.J.ω v ≤ 1) ∧
    (∀ S, UncertifiedCap Straddle.J (block Straddle.A Straddle.X Straddle.κ S) (Straddle.inf S) 1) ∧
    UncertifiedCap Straddle.J (Straddle.A.region Straddle.dark Set.univ)
      (Fin.elim0 : Fin 0 → Finset (Fin Straddle.A.N)) 0 ∧
    ∑ _S : Block Straddle.κ, min (Straddle.q 1 1 + 1 * (0 : ℕ) * (1 : ℕ)) ((1 : ℕ) * Straddle.q 1 1 + 1) + 0 <
      Straddle.J.total := by
  refine ⟨⟨?_, Straddle.blocks⟩, fun _ => Straddle.cap 1, Straddle.cap 1,
    fun S _ => ⟨block_subset _ _ _ S, Straddle.blocks S⟩, ?_, ?_, ?_, fun _ => le_rfl, zero_le_one,
    fun _ _ => le_rfl, Straddle.outside_le, Straddle.outside_dark.le, ?_⟩
  · unfold ServingBlocksCover; decide
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold Overlap; decide
  · rw [Straddle.total_eq]
    simp [Straddle.q]

/-- **The remote bound's premises hold** at `Honest`, at `(c, k) = (1, 1)`, one inference block per serving block,
`q_U(S) = 1` and `q_U(L_dark) = 1`. -/
theorem ServingBlockBoundRemote_honest :
    ServingBlocksCover Honest.A Honest.J Honest.dark Honest.X 2 ∧
    PerBlockCap Honest.A.toCircuit Honest.θ Honest.J Honest.q 1 1 ∧
    InferenceBlocks Honest.A.toCircuit Honest.θ (fun _ : Block Honest.κ => 1) Honest.inf 1 1 ∧
    (∀ S, UncertifiedCap Honest.J (block Honest.A Honest.X Honest.κ S) (Honest.inf S) 1) ∧
    UncertifiedCap Honest.J (Honest.A.region Honest.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Honest.A.N)) 1 := by
  obtain ⟨⟨hcov, -⟩, -, hq, hinf, -, -, -, -, -, hU, hD⟩ := ServingBlockBound_honest
  exact ⟨hcov, hq, hinf, hU, hD⟩

/-- **The remote bound is false without the per-block cap at the inference size**: at `Counter`, with `(c, k) = (1, 0)`,
one inference block and `q_U = 0`, every other premise holds, the cap at `(1, 0)` fails, and the counted weight `1` is
above the right-hand side `0`. -/
theorem ServingBlockBoundRemote_nontrivial :
    ServingBlocksCover Counter.A Counter.J Counter.dark Counter.X 1 ∧
    ¬ PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 0 ∧
    InferenceBlocks Counter.A.toCircuit Counter.θ (fun _ : Block Counter.κ => 1) Counter.inf 1 0 ∧
    (∀ S, UncertifiedCap Counter.J (block Counter.A Counter.X Counter.κ S) (Counter.inf S) 0) ∧
    UncertifiedCap Counter.J (Counter.A.region Counter.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N))
      0 ∧
    ∑ _S : Block Counter.κ, ((1 : ℕ) * Counter.q 1 0 + 0) + 0 < Counter.J.total := by
  obtain ⟨⟨hcov, -⟩, -, -, -, -, -, -, -, -, hU, hD, -⟩ := ServingBlockBound_nontrivial
  refine ⟨hcov, Counter.not_cap_zero, fun S _ => ⟨block_subset _ _ _ S, Counter.blocks S⟩, hU, hD, ?_⟩
  rw [Counter.total_eq]
  simp [Counter.q]

/-- **The elementary uncertified cap's premises hold** at `Honest`: at each serving block with its inference block and
every gate certified, and at the dark leaves with no blocks and none certified, at `ω_max = 1`. -/
theorem UncertifiedElementary_honest :
    (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Honest.J.V, Honest.J.ω v ≤ 1) ∧
    (∀ S, Alignment (block Honest.A Honest.X Honest.κ S) (Honest.inf S) (block Honest.A Honest.X Honest.κ S)) ∧
    Alignment (Honest.A.region Honest.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Honest.A.N)) ∅ :=
  ⟨zero_le_one, Honest.ω_le, fun _ _ hg _ => ⟨0, hg⟩, fun _ _ h => absurd h (Finset.notMem_empty _)⟩

/-- **The elementary uncertified cap is false without alignment**: at `Counter`'s gate, with no blocks and the gate
certified, at `ω_max = 1`, the cap `ω_max |R \ W| + split(R)` is `0` and the operation's weight `1` is above it. -/
theorem UncertifiedElementary_nontrivial :
    (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Counter.J.V, Counter.J.ω v ≤ 1) ∧
    ¬ Alignment ({0} : Finset (Fin Counter.A.N)) (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N)) {0} ∧
    ¬ UncertifiedCap Counter.J {0} (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N))
      (1 * (({0} : Finset (Fin Counter.A.N)) \ {0}).card + Counter.J.split {0} Fin.elim0) := by
  refine ⟨zero_le_one, fun _ _ => le_rfl, fun h => ?_, fun h => ?_⟩
  · obtain ⟨i, _⟩ := h 0 (Finset.mem_singleton_self 0) (Finset.mem_singleton_self 0)
    exact i.elim0
  · simp [UncertifiedCap, Counted.outside, Counted.split] at h

/-! Each lemma is its guarantee's premises as the guarantee states them: the guarantee applies at the honest instance,
and at a counter-example the dropped premise would give the bound the lemma's last conjunct refutes. -/

example : Honest.J.total ≤
    ∑ _S : Block Honest.κ, min (Honest.q 1 1 + 1 * (1 : ℕ) * (1 : ℕ)) ((1 : ℕ) * Honest.q 1 1 + 1) + 1 := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := ServingBlockBound_honest
  exact ServingBlockBound Honest.A Honest.θ Honest.J Honest.q 1 1 Honest.dark Honest.X 2 Honest.κ (fun _ => 1)
    (fun _ => 1) (fun _ => 1) Honest.inf 1 1 (fun _ => 1) 1 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11

example (h2 : ∀ _ : Block Counter.κ, PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 0) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, h7, h8, h9, h10, h11, hlt⟩ := ServingBlockBound_nontrivial
  exact not_le.2 hlt (ServingBlockBound Counter.A Counter.θ Counter.J Counter.q 1 1 Counter.dark Counter.X 1 Counter.κ
    (fun _ => 1) (fun _ => 0) (fun _ => 1) Counter.inf 1 1 (fun _ => 0) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11)

example (h5 : OperationBodies Detached.A.toCircuit Detached.θ Detached.J) : False := by
  obtain ⟨h1, h2, h3, h4, -, h6, h7, h8, h9, h10, h11, hlt⟩ := ServingBlockBound_nontrivial_bodies
  exact not_le.2 hlt (ServingBlockBound Detached.A Detached.θ Detached.J Detached.q 1 0 Detached.dark Detached.X 2
    Detached.κ (fun _ => 1) (fun _ => 0) (fun _ => 1) Detached.inf 1 1 (fun _ => 1) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    h11)

example (h6 : Overlap Straddle.J 0) : False := by
  obtain ⟨h1, h2, h3, h4, h5, -, -, h7, h8, h9, h10, h11, hlt⟩ := ServingBlockBound_nontrivial_overlap
  exact not_le.2 hlt (ServingBlockBound Straddle.A Straddle.θ Straddle.J Straddle.q 1 1 Straddle.dark Straddle.X 2
    Straddle.κ (fun _ => 1) (fun _ => 1) (fun _ => 1) Straddle.inf 1 0 (fun _ => 1) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    h11)

example : Honest.J.total ≤ ∑ _S : Block Honest.κ, ((1 : ℕ) * Honest.q 1 1 + 1) + 1 := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := ServingBlockBoundRemote_honest
  exact ServingBlockBoundRemote Honest.A Honest.θ Honest.J Honest.q 1 1 Honest.dark Honest.X 2 Honest.κ (fun _ => 1)
    Honest.inf (fun _ => 1) 1 h1 h2 h3 h4 h5

example (h2 : PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 0) : False := by
  obtain ⟨h1, -, h3, h4, h5, hlt⟩ := ServingBlockBoundRemote_nontrivial
  exact not_le.2 hlt (ServingBlockBoundRemote Counter.A Counter.θ Counter.J Counter.q 1 0 Counter.dark Counter.X 1
    Counter.κ (fun _ => 1) Counter.inf (fun _ => 0) 0 h1 h2 h3 h4 h5)

example (S : Block Honest.κ) :
    UncertifiedCap Honest.J (block Honest.A Honest.X Honest.κ S) (Honest.inf S)
      (1 * (block Honest.A Honest.X Honest.κ S \ block Honest.A Honest.X Honest.κ S).card +
        Honest.J.split (block Honest.A Honest.X Honest.κ S) (Honest.inf S)) := by
  obtain ⟨h1, h2, h3, -⟩ := UncertifiedElementary_honest
  exact UncertifiedElementary _ Honest.J _ _ _ 1 h1 h2 (h3 S)

example (h3 : Alignment ({0} : Finset (Fin Counter.A.N)) (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N)) {0}) :
    False := by
  obtain ⟨h1, h2, -, hcap⟩ := UncertifiedElementary_nontrivial
  exact hcap (UncertifiedElementary _ Counter.J _ _ _ 1 h1 h2 h3)

end Accounting.SecurityProofs.ServingBlocks
