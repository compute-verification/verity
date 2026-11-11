module

public import Proofs.Nci.CutWitness

/-!
# Blocks charged whole: what they hold, what `HECharged` follows from, and that it says something

A set of gates holds whole at most its own gates over the fewest gates of a MAC, MACs being disjoint
(`card_inside_mul_le`): what a block charged whole holds.

`HECharged` follows from the cut's counts over a set of blocks (`CountsOn`, `HECharged_of_countsOn`). `CountsOn` is
`counts` with three changes: `V`, `F` and `E` count over the blocks of a set `G` alone; each block of `G` enters `Π` at
what it reads itself, so `V² ≤ (∑_G Π(reads)) F`; and the split MACs, and the gates leaving a block of `G`, are
counted against what all the blocks read between them, `T`, with the MACs a block outside `G` holds whole counted
apart. No theorem here proves `CountsOn`. From it, (H) at what a block of `G` reads gives `Π(r) ≤ r² / (4 β_w β_x)` at
that block's own `r` and its own `β_x` (`onePremise`), and (E) gives `b E ≤ T` (`real_step_charged`): the count with (H)
at each block's own reads (`countsOn_rows`). `HECharged` is that count at one `β_x`, (H) at the cap holding at every
smaller one (`hardness_mono`).

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

/-- **The cut's counts over a set of blocks.** For a partition whose blocks read at most `T` gates outside `θ` between
them and a set `G` of its blocks: `V` MACs the blocks of `G` hold whole, `F` pairs of a block of `G` and an output it
feeds, and `E` pairs of a block of `G` and an output in its `E(block)`. A split MAC is an entry of some block, so it
is inside `T`, and a MAC a block outside `G` holds whole is counted apart. Each block of `G` enters `Π` at what it
reads itself. -/
def CountsOn : Prop :=
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι] (d : RegisteredCircuit) (W : Work d.D ι) (m T : ℕ) (P : Partition d.D m)
    (G : Finset (Fin m)), ∑ u, (d.unregistered (P.gates u)).card ≤ T →
    ∃ V F E : ℕ, W.N ≤ T + V + ∑ u ∈ Gᶜ, (W.inside (P.gates u)).card ∧
      V ^ 2 ≤ (∑ u ∈ G, piMax d W (d.unregistered (P.gates u)).card) * F ∧ F ≤ W.O + E ∧
      W.bY * (E : ℝ) ≤ (T : ℝ) ∧ (E = 0 ∨ 0 < W.bY)

/-- The counts' last step over a set of blocks: `V² ≤ (S / (4 q)) (O + E)` and `b E ≤ T` give
`V ≤ ½ √(S O / q) + ½ √(S T / (q b))`. -/
theorem real_step_charged {V S T O E q b : ℝ} (hS : 0 ≤ S) (hO : 0 ≤ O) (hE : 0 ≤ E) (hq : 0 < q) (hb : 0 < b)
    (hV : V ^ 2 ≤ S / (4 * q) * (O + E)) (hbE : b * E ≤ T) :
    V ≤ √(S * O / q) / 2 + √(S * T / (q * b)) / 2 := by
  have hs : 0 ≤ S / (4 * q) := div_nonneg hS (mul_pos four_pos hq).le
  have hE' : E ≤ T / b := (le_div_iff₀ hb).2 ((mul_comm E b).trans_le hbE)
  have h2 : √(S / (4 * q) * O) = √(S * O / q) / 2 := by
    rw [show S / (4 * q) * O = (1 / 2) ^ 2 * (S * O / q) by ring, sqrt_scaled (by norm_num : (0 : ℝ) ≤ 1 / 2)]
    ring
  have h3 : √(S / (4 * q) * (T / b)) = √(S * T / (q * b)) / 2 := by
    rw [show S / (4 * q) * (T / b) = (1 / 2) ^ 2 * (S * T / (q * b)) by ring,
      sqrt_scaled (by norm_num : (0 : ℝ) ≤ 1 / 2)]
    ring
  calc V ≤ √(S / (4 * q) * (O + E)) := Real.le_sqrt_of_sq_le hV
    _ = √(S / (4 * q) * O + S / (4 * q) * E) := by rw [mul_add]
    _ ≤ √(S / (4 * q) * O) + √(S / (4 * q) * E) := sqrt_add_le (mul_nonneg hs hO) (mul_nonneg hs hE)
    _ ≤ √(S / (4 * q) * O) + √(S / (4 * q) * (T / b)) :=
        add_le_add le_rfl (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hE' hs))
    _ = √(S * O / q) / 2 + √(S * T / (q * b)) / 2 := by rw [h2, h3]

/-- **The count over a set of blocks with (H) at each block's own reads.** Each block `u` of `G` takes (H) at what it
reads itself, at its own `β_x u`, so `Π(reads u) ≤ reads u ² / (4 β_w β_x u)`, and with `Q` at least the sum over `G`
of `reads u ² / (β_w β_x u)`: `N ≤ (the MACs the blocks outside G hold whole) + T + ½ √(Q O) + ½ √(Q T / b)`. No
block's cap is another's: a block that reads more lowers the `β_x` it is asked at, and no other block's. -/
theorem countsOn_rows (hC : CountsOn) {ι V : Type} [Fintype ι] [DecidableEq ι] [DecidableEq V]
    {d : RegisteredCircuit} (W : Work d.D ι) (val : Finset (Fin d.D.N) → V) {m T : ℕ} (P : Partition d.D m)
    (G : Finset (Fin m)) {βw b Q : ℝ} (βx : Fin m → ℝ) (hw : 0 < βw) (hx : ∀ u ∈ G, 0 < βx u) (hb : 0 < b)
    (hval : ValuesInjective W val) (hshare : NoSharedPair W val)
    (hH : ∀ u ∈ G, Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx u] (d.unregistered (P.gates u)).card)
    (hE : Assumptions.Exports W b) (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T)
    (hQ : ∑ u ∈ G, ((d.unregistered (P.gates u)).card : ℝ) ^ 2 / (βw * βx u) ≤ Q) :
    (W.N : ℝ) ≤ ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) +
      ((T : ℝ) + √(Q * W.O) / 2 + √(Q * T / b) / 2) := by
  obtain ⟨Vc, F, E, hN, hV, hF, hbE, hb0⟩ := hC ι d W m T P G hT
  have hQ0 : 0 ≤ Q :=
    (sum_nonneg fun u hu => div_nonneg (sq_nonneg _) (mul_pos hw (hx u hu)).le).trans hQ
  have hpi : ((∑ u ∈ G, piMax d W (d.unregistered (P.gates u)).card : ℕ) : ℝ) ≤ Q / (4 * 1) :=
    calc ((∑ u ∈ G, piMax d W (d.unregistered (P.gates u)).card : ℕ) : ℝ)
        = ∑ u ∈ G, (piMax d W (d.unregistered (P.gates u)).card : ℝ) := Nat.cast_sum _ _
      _ ≤ ∑ u ∈ G, ((d.unregistered (P.gates u)).card : ℝ) ^ 2 / (βw * βx u) / 4 :=
          sum_le_sum fun u hu =>
            ((onePremise hw (hx u hu) hval hshare (hH u hu)).1).trans_eq (by ring)
      _ = (∑ u ∈ G, ((d.unregistered (P.gates u)).card : ℝ) ^ 2 / (βw * βx u)) / 4 := (sum_div _ _ _).symm
      _ ≤ Q / 4 := div_le_div_of_nonneg_right hQ (by norm_num)
      _ = Q / (4 * 1) := by ring
  have h0 : Vc ^ 2 ≤ (∑ u ∈ G, piMax d W (d.unregistered (P.gates u)).card) * (W.O + E) :=
    hV.trans (Nat.mul_le_mul le_rfl hF)
  have hV' : (Vc : ℝ) ^ 2 ≤ Q / (4 * 1) * (W.O + E) :=
    calc (Vc : ℝ) ^ 2 = ((Vc ^ 2 : ℕ) : ℝ) := (Nat.cast_pow Vc 2).symm
      _ ≤ (((∑ u ∈ G, piMax d W (d.unregistered (P.gates u)).card) * (W.O + E) : ℕ) : ℝ) := Nat.cast_le.2 h0
      _ = ((∑ u ∈ G, piMax d W (d.unregistered (P.gates u)).card : ℕ) : ℝ) * ((W.O + E : ℕ) : ℝ) :=
          Nat.cast_mul _ _
      _ ≤ Q / (4 * 1) * ((W.O + E : ℕ) : ℝ) := mul_le_mul_of_nonneg_right hpi (Nat.cast_nonneg _)
      _ = Q / (4 * 1) * (W.O + E) := by rw [Nat.cast_add]
  have hbE' : b * (E : ℝ) ≤ T := by
    by_cases h : W.fed.Nonempty
    · exact (mul_le_mul_of_nonneg_right (le_bY W hE h) (Nat.cast_nonneg E)).trans hbE
    · have hY : W.bY = 0 := by rw [Work.bY, dif_neg h]
      rcases hb0 with h0 | h0
      · rw [h0, Nat.cast_zero, mul_zero]
        exact Nat.cast_nonneg T
      · rw [hY] at h0
        exact absurd h0 (lt_irrefl _)
  have hstep := real_step_charged hQ0 (Nat.cast_nonneg W.O) (Nat.cast_nonneg E) one_pos hb hV' hbE'
  rw [div_one, one_mul] at hstep
  have hN' : (W.N : ℝ) ≤ T + Vc + ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) := by exact_mod_cast hN
  linarith

/-- **`HECharged` from the counts over a set of blocks.** (H) at the cap `k` holds at what each block of `G` reads
(`hardness_mono`), so the count with (H) at each block's own reads applies at one `β_x` and `Q = S / (β_w β_x)`; and
the MACs a block outside `G` holds whole take `γ` of its at most `c` gates each. -/
theorem HECharged_of_countsOn (hC : CountsOn) : Guarantees.HECharged := by
  intro ι V _ _ _ d W val k m c γ T S P G βw βx b hw hx hb hval hshare hH hE hk hT hS
  have hq : 0 < βw * βx := mul_pos hw hx
  have hQ : ∑ u ∈ G, ((d.unregistered (P.gates u)).card : ℝ) ^ 2 / (βw * βx) ≤ S / (βw * βx) := by
    rw [← sum_div]
    exact div_le_div_of_nonneg_right (by exact_mod_cast hS) hq.le
  have hrows := countsOn_rows hC W val P G (fun _ => βx) hw (fun _ _ => hx) hb hval hshare
    (fun u hu => hardness_mono hH (hk u hu)) hE hT hQ
  have e1 : (S : ℝ) / (βw * βx) * W.O = S * W.O / (βw * βx) := by ring
  have e2 : (S : ℝ) / (βw * βx) * T / b = S * T / (βw * βx * b) := by ring
  rw [e1, e2] at hrows
  have hfirst : (W.N : ℝ) ≤
      ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) + workHEOn T S (βw * βx) b W.O := by
    unfold workHEOn
    exact hrows
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
