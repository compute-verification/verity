import Definitions.Accounting.Guarantees.FixedMachine
import Proofs.Accounting.Actual.Split
import Proofs.Core.Formalism
import Proofs.Core.Unroll

/-!
# Proofs: a fixed machine's proof blocks are described within their budgets

* `card_unregistered_le`: a proof block that is the region of a location over the run reads at most its initial budget
  plus its ingress in unregistered bits: an unregistered gate it reads is placed outside the location, and counted in the
  ingress, or inside it, where it is an input gate and so initial information;
* `BlockDescription`: Lemma 1's program for the proof block, at most `|e| + ⌈log₂ B⌉ + D + in_X([0, ∞))` bits;
* `GateCount`: an isomorphism keeps the size, and `H` run `T` steps has size `T · |H|`;
* the witnesses at `FixedMachine.Instances`: the honest machine's premises, and the hard-coding circuit, whose string
  needs a one-bit program and has a budget of `0` bits.
-/

namespace Accounting.SecurityProofs.FixedMachine

open Verity Finset Accounting.FixedMachine

variable {L : Type*}

open Classical in
/-- **A proof block reads its initial budget plus its ingress**: when proof block `u` of `R` is, through `φ`, the region
of `X` over the run, with `X`'s initial information `a` within `D` bits, and `A` holds one bit a gate, its unregistered
inputs are at most `D + in_X([0, ∞))`. -/
theorem card_unregistered_le (R : Registration) (A : Placed L) (φ : Circuit.Iso R.d.D A.toCircuit) (X : Finset L)
    (a : Finset (Fin A.N)) (D : ℕ) (u : Fin R.B) (hw : ∀ g, A.w g = 1)
    (hblk : (R.P.gates u).map φ.e.toEmbedding = A.region X (Set.Ici 0))
    (hbud : InitialBudget A (R.d.θ.map φ.e.toEmbedding) a X D) :
    (R.d.unregistered (R.P.gates u)).card ≤ D + A.ingress X (Set.Ici 0) := by
  obtain ⟨_, hin, hD⟩ := hbud
  have key : (R.d.unregistered (R.P.gates u)).map φ.e.toEmbedding ⊆
      (((A.into (A.region X (Set.Ici 0))).filter fun g => A.x g ∈ X) ∩ a) ∪
        ((A.into (A.region X (Set.Ici 0))).filter fun g => A.x g ∉ X) := by
    intro g hg
    obtain ⟨b, hb, rfl⟩ := mem_map.1 hg
    rw [Declaration.unregistered, mem_sdiff] at hb
    have hinto := φ.map_mem_into hb.1
    rw [hblk] at hinto
    by_cases hx : A.x (φ.e b) ∈ X
    · refine mem_union_left _ (mem_inter.2 ⟨mem_filter.2 ⟨hinto, hx⟩, ?_⟩)
      have hinput : A.isInput (φ.e b) = true := by
        have hout := (mem_filter.1 hinto).2.1
        cases h : A.isInput (φ.e b)
        · exact absurd ((Actual.mem_region A).2 ⟨h, hx, Set.mem_Ici.2 (Nat.zero_le _)⟩) hout
        · rfl
      rcases hin _ hinput hx with h | h
      · exact absurd ((mem_map' φ.e.toEmbedding).1 h) hb.2
      · exact h
    · exact mem_union_right _ (mem_filter.2 ⟨hinto, hx⟩)
  have hS : A.stateIn X 0 a + A.ingress X (Set.Ici 0) =
      (((A.into (A.region X (Set.Ici 0))).filter fun g => A.x g ∈ X) ∩ a).card +
        ((A.into (A.region X (Set.Ici 0))).filter fun g => A.x g ∉ X).card := by
    rw [Placed.stateIn, Placed.ingress, Circuit.width_eq_card hw, Circuit.width_eq_card hw]
  calc (R.d.unregistered (R.P.gates u)).card
      = ((R.d.unregistered (R.P.gates u)).map φ.e.toEmbedding).card := (card_map _).symm
    _ ≤ _ := card_le_card key
    _ ≤ _ := card_union_le _ _
    _ ≤ D + A.ingress X (Set.Ici 0) := by omega

theorem BlockDescription : Guarantees.FixedMachine.BlockDescription := by
  intro H S T R hR hH ρ M r t a_blk e y he hev L A φ X a D hblk hbud x hx u
  have hB : R.d.D.Boolean := by
    rw [hR]
    exact Circuit.unroll_boolean H S T hH
  have hw : ∀ g, A.w g = 1 := fun g => by rw [φ.w_symm, hB.one_bit]
  have hrun := hev u (R.d.D.eval (φ.pull x))
  rw [R.recompute_eval hx u] at hrun
  unfold BlockDescribed
  rw [φ.pull_eval]
  refine (iInf₂_le (blockProgram R e (R.d.D.eval (φ.pull x)) u) hrun).trans ?_
  rw [length_blockProgram R hB]
  have := card_unregistered_le R A φ (X u) (a u) (D u) u hw (hblk u) (hbud u)
  exact_mod_cast (by omega : e.length + Nat.clog 2 R.B + (R.d.unregistered (R.P.gates u)).card ≤
    D u + A.ingress (X u) (Set.Ici 0) + Nat.clog 2 R.B + a_blk)

theorem GateCount : Guarantees.FixedMachine.GateCount := by
  rintro H S T L A ⟨φ⟩
  rw [φ.size_computed, Circuit.size_unroll]

theorem R₀_op_one (v : R₀.d.D.Transcript) : R₀.d.D.op ⟨1, by decide⟩ v = ~~~ (v ⟨0, by decide⟩) := rfl

theorem R₀_op_two (v : R₀.d.D.Transcript) : R₀.d.D.op ⟨2, by decide⟩ v = ~~~ (v ⟨1, by decide⟩) := rfl

theorem BlockDescription_honest : Guarantees.FixedMachine.BlockDescription_honest := by
  refine ⟨⟨fun _ => rfl, by decide⟩, ?_, ⟨φ₀⟩, ?_, ?_, ?_⟩
  · rintro ⟨u, hu⟩ X
    obtain rfl : u = 0 := by have : u < 1 := hu; omega
    have hunreg : R₀.d.unregistered (R₀.P.gates ⟨0, by decide⟩) = {⟨0, by decide⟩} := by decide
    have hidx : indexCode (⟨0, hu⟩ : Fin R₀.B) = [] := List.eq_nil_of_length_eq_zero (by simp [indexCode, R₀])
    have hprog : blockProgram R₀ [] X ⟨0, hu⟩ = [(X ⟨0, by decide⟩).getLsbD 0] := by
      rw [blockProgram, hunreg, hidx, sort_singleton]
      rfl
    have h1 : R₀.recompute X ⟨0, hu⟩ ⟨1, by decide⟩ = ~~~ (X ⟨0, by decide⟩) := by
      rw [Registration.recompute, Circuit.run_free _ rfl, R₀_op_one, Circuit.run_fixed _ rfl]
    have h2 : R₀.recompute X ⟨0, hu⟩ ⟨2, by decide⟩ = X ⟨0, by decide⟩ := by
      rw [Registration.recompute, Circuit.run_free _ rfl, R₀_op_two, ← Registration.recompute, h1]
      exact BitVec.not_not
    rw [hprog]
    show some [(X ⟨0, by decide⟩).getLsbD 0] = some [(R₀.recompute X ⟨0, hu⟩ ⟨2, by decide⟩).getLsbD 0]
    rw [h2]
    rfl
  · rintro ⟨u, hu⟩
    obtain rfl : u = 0 := by have : u < 1 := hu; omega
    ext g
    rw [mem_map, Actual.mem_region]
    constructor
    · rintro ⟨b, hb, rfl⟩
      rw [Partition.mem_gates] at hb
      refine ⟨?_, mem_univ _, Set.mem_Ici.2 (Nat.zero_le _)⟩
      show C₀.isInput b = false
      revert hb
      show (if C₀.isInput b then none else some (⟨0, hu⟩ : Fin R₀.B)) = some ⟨0, hu⟩ → C₀.isInput b = false
      cases C₀.isInput b <;> simp
    · rintro ⟨hg, -, -⟩
      refine ⟨φ₀.e.symm g, ?_, φ₀.e.apply_symm_apply g⟩
      rw [Partition.mem_gates]
      have hg' : C₀.isInput (φ₀.e.symm g) = false := hg
      show (if C₀.isInput (φ₀.e.symm g) then none else some (⟨0, hu⟩ : Fin R₀.B)) = some ⟨0, hu⟩
      simp [hg']
  · intro u
    refine ⟨?_, ?_, ?_⟩
    · intro g hg
      rw [mem_singleton.1 hg]
      rfl
    · intro g hg hx
      clear hx
      right
      show g ∈ ({⟨0, by decide⟩} : Finset (Fin A₀.N))
      revert hg
      revert g
      decide
    · rw [Placed.stateIn, Circuit.width_eq_card (fun _ => rfl)]
      exact (card_le_card inter_subset_right).trans (card_singleton _).le
  · intro x g hg
    exact absurd hg (notMem_empty g)

theorem BlockDescription_nontrivial : Guarantees.FixedMachine.BlockDescription_nontrivial := by
  unfold Guarantees.FixedMachine.BlockDescription_nontrivial BlockDescribed
  have hin : Hard.ingress univ (Set.Ici 0) = 0 := by
    simp [Placed.ingress, Circuit.width]
  rw [hin, Nat.clog_one_right]
  have h1 : (1 : ℕ∞) ≤ LDL M₀ 0 () hardString := by
    refine le_iInf₂ fun p hp => ?_
    have hp1 : p.length = 1 := by
      by_contra hc
      simp [M₀, hc] at hp
    simp [hp1]
  intro h
  have := h1.trans h
  norm_num at this

theorem Hard_notFixed : Guarantees.FixedMachine.Hard_notFixed := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rintro T ⟨φ⟩
    have hN : 1 + 1 * T = 2 := by
      have := Fintype.card_congr φ.e
      simp only [Fintype.card_fin] at this
      exact this
    have hT : T = 1 := by omega
    subst hT
    have hg : (H₀.unroll S₀ 1).isInput ⟨1, by decide⟩ = false := rfl
    have he : φ.e ⟨1, by decide⟩ = ⟨1, by decide⟩ := by
      have hi := φ.isInput_e ⟨1, by decide⟩
      rw [hg] at hi
      revert hi
      generalize φ.e ⟨1, by decide⟩ = g
      revert g
      decide
    let ones : Hard.Transcript := fun _ => (1 : BitVec 1)
    have h := congrArg BitVec.toNat (φ.op_e ⟨1, by decide⟩ hg ones)
    rw [BitVec.toNat_cast, he] at h
    exact absurd h (by
      show (1 : ℕ) ≠ (~~~ (1 : BitVec 1)).toNat
      decide)
  · intro g hg
    exact absurd hg (notMem_empty g)
  · intro g hg hx
    clear hx
    left
    revert hg
    revert g
    decide
  · simp [Placed.stateIn, Circuit.width]

end Accounting.SecurityProofs.FixedMachine
