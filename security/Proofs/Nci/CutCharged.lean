module

public import Proofs.Nci.CutWitness

/-!
# Blocks charged whole: what they hold, `HECharged`, and that it says something

A set of gates holds whole at most its own gates over the fewest gates of a MAC, MACs being disjoint
(`card_inside_mul_le`): what a block charged whole holds.

`HECharged` is the count over a set `G` of blocks with (H) at each block's own reads (`bound_on_rows`), at one `β_x`.
Each block of `G` reads at most `k` gates outside `θ`, and (H) at `k` holds at what the block reads (`hardness_mono`),
so the count applies with `S / (β_w β_x)` for its `Q`. Its two other terms are the statement's: the MACs a block
outside `G` holds, matmul by matmul, are the MACs each of whose gates is in the block (`card_whole_eq`), which is
`Work.inside`; and the pairs of a block of `G` and an output with a wire in it are at most `O` (`outOn_le`). MACs are
disjoint, so the MACs a block outside `G` holds whole take `γ` of its at most `c` gates each (`card_inside_mul_le`):
the second bound.

`HECharged_honest` and `HECharged_nontrivial` are at the toy `Nci.Cut.Toy`. With a block for each MAC, the first under
the cap `k = 2` and the second charged whole, `HECharged`'s premises hold together. With one block holding both MACs
and no block charged, the block reads three gates, and at the total `T = 0` the bound is `0`, below `N = 2`.
-/

@[expose] public section

namespace Nci.Cut.SecurityProofs

open Verity Finset

/-- A set `A` of gates holds whole at most `|A| / γ` MACs of at least `γ` gates each: MACs are disjoint. -/
theorem card_inside_mul_le {C : Circuit} {ι : Type} [Fintype ι] (W : Work C ι) (A : Finset (Fin C.N)) {γ : ℕ}
    (hγ : ∀ s, γ ≤ (W.mac s).card) : (W.inside A).card * γ ≤ A.card :=
  calc (W.inside A).card * γ = ∑ _s ∈ W.inside A, γ := by rw [sum_const, smul_eq_mul]
    _ ≤ ∑ s ∈ W.inside A, (W.mac s).card := sum_le_sum fun s _ => hγ s
    _ = ((W.inside A).biUnion W.mac).card := (card_biUnion fun s _ t _ hst => W.mac_disjoint s t hst).symm
    _ ≤ A.card := card_le_card (biUnion_subset.2 fun _ hs => (mem_filter.1 hs).2)

/-- (H) at a read cap holds at every smaller one. -/
theorem hardness_mono {ρ V : Type} [Fintype ρ] [DecidableEq V] {d : RegisteredCircuit}
    {A : ρ → Finset (Finset (Fin d.D.N))} {val : Finset (Fin d.D.N) → V} {β : ρ → ℝ} {k k' : ℕ}
    (h : Assumptions.Hardness d A val β k) (hk : k' ≤ k) : Assumptions.Hardness d A val β k' :=
  fun S hS => h S (hS.trans hk)

/-- **`HECharged`.** The count over a set of blocks with (H) at each block's own reads (`bound_on_rows`), each block
of `G` at what it reads and all of them at one `β_x`: (H) at `k` holds at what each block of `G` reads
(`hardness_mono`), and `S / (β_w β_x)` is at least the sum over `G` of the squares of what each block reads over
`β_w β_x`. The MACs a block outside `G` holds are the MACs each of whose gates is in it (`card_whole_eq`), the pairs of
a block of `G` and an output with a wire in it are at most `O` (`outOn_le`), and the MACs a block outside `G` holds
whole take `γ` of its at most `c` gates each. -/
theorem HECharged : Guarantees.HECharged := by
  intro ι V _ _ _ d W val k m c γ T S P G βw βx b hw hx hb hval hshare hH hE hk hT hS
  have hq : 0 < βw * βx := mul_pos hw hx
  have hQ : ∑ u ∈ G, ((d.unregistered (P.gates u)).card : ℝ) ^ 2 / (βw * βx) ≤ S / (βw * βx) := by
    rw [← sum_div]
    exact div_le_div_of_nonneg_right (by exact_mod_cast hS) hq.le
  have hrows := bound_on_rows (W := W) (val := val) P G (fun u => (d.unregistered (P.gates u)).card) (fun _ => βx) hw
    (fun _ _ => hx) hb hval hshare (fun u hu => hardness_mono hH (hk u hu)) hE (fun _ _ => le_rfl) hT hQ
  have hwhole : ∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card = ∑ u ∈ Gᶜ, (W.inside (P.gates u)).card :=
    sum_congr rfl fun u _ => (card_whole_eq P W u).symm
  have hout : √((S : ℝ) / (βw * βx) * outOn P W G) ≤ √((S : ℝ) / (βw * βx) * W.O) :=
    Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (Nat.cast_le.2 (outOn_le P W G))
      (div_nonneg (Nat.cast_nonneg S) hq.le))
  have e1 : (S : ℝ) / (βw * βx) * W.O = S * W.O / (βw * βx) := by ring
  have e2 : (S : ℝ) / (βw * βx) * T / b = S * T / (βw * βx * b) := by ring
  rw [e1] at hout
  rw [e2, hwhole] at hrows
  have hfirst : (W.N : ℝ) ≤
      ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) + workHEOn T S (βw * βx) b W.O := by
    unfold workHEOn
    linarith
  refine ⟨hfirst, fun hγ hc => ?_⟩
  have hheld : (γ : ℝ) * ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) ≤ (Gᶜ.card : ℝ) * c := by
    have h : γ * ∑ u ∈ Gᶜ, (W.inside (P.gates u)).card ≤ Gᶜ.card * c :=
      calc γ * ∑ u ∈ Gᶜ, (W.inside (P.gates u)).card = ∑ u ∈ Gᶜ, (W.inside (P.gates u)).card * γ := by
            rw [mul_sum]
            exact sum_congr rfl fun _ _ => mul_comm _ _
        _ ≤ ∑ u ∈ Gᶜ, (P.gates u).card := sum_le_sum fun u _ => card_inside_mul_le W (P.gates u) hγ
        _ ≤ ∑ _u ∈ Gᶜ, c := sum_le_sum fun u hu => hc u hu
        _ = Gᶜ.card * c := by rw [sum_const, smul_eq_mul]
    exact_mod_cast h
  calc (γ : ℝ) * W.N
      ≤ γ * (((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) + workHEOn T S (βw * βx) b W.O) :=
        mul_le_mul_of_nonneg_left hfirst (Nat.cast_nonneg γ)
    _ = γ * ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) + γ * workHEOn T S (βw * βx) b W.O :=
        mul_add _ _ _
    _ ≤ (Gᶜ.card : ℝ) * c + γ * workHEOn T S (βw * βx) b W.O := add_le_add hheld le_rfl

theorem toy_card_P2 : ∀ u, (Toy.P2.gates u).card ≤ 1 := by decide

theorem toy_card_mac : ∀ s, 1 ≤ (Toy.W.mac s).card := by decide

theorem HECharged_honest : Guarantees.HECharged_honest := by
  have h0 := toy_reads_P2 0
  have h1 := toy_reads_P2 1
  refine ⟨toy_values, toy_noSharedPair, toy_hardness 2, toy_exports 1, fun u _ => toy_reads_P2 u, ?_, ?_,
    toy_card_mac, fun u _ => toy_card_P2 u⟩
  · rw [Fin.sum_univ_two]
    omega
  · rw [sum_singleton]
    exact (Nat.pow_le_pow_left h0 2).trans (by norm_num)

theorem HECharged_nontrivial : Guarantees.HECharged_nontrivial := by
  refine ⟨one_pos, toy_values, toy_noSharedPair, toy_hardness 0, toy_exports 1, fun h => ?_, ?_⟩
  · have h3 := toy_reads_P1
    rw [Fin.sum_univ_one] at h
    omega
  · rw [toy_N, compl_univ, sum_empty]
    norm_num [workHEOn]

end Nci.Cut.SecurityProofs
