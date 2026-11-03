import Definitions.Nci.Guarantees.Products.Sum
import Proofs.Nci.Products.Whole

/-!
# Products made whole through chains, several in a block: the cap and its bound on `ρ` summed over a set

For a set `G` of a whole-making system's products and a partition `P` into `m` inference blocks, block `u` with gates
`I_u` and reads `U_u = in(I_u) \ θ`, `T_qu` the MACs of product `q`'s cells that `I_u` holds whole (`Standard.cellWhole`),
and `start_qij` the sets its partial sums received at their width give:

1. **Values had** (`System.card_hadValues_le`): `b |hadValues G I_u| ≤ k (1 + E)`. The paid values are disjoint, with
   `b` bits of `U_u` each, and the made ones number at most `e_j k / b` for each of `G`'s classes `j`
   (`System.sum_fanout_le`).
2. **Several products in a block** (`sum_card_le_shared`): triples `T_q` with `Σ_q |π₁₂ T_q| |π₂₃ T_q| ≤ c H²`,
   `b H ≤ K₁` and `b_C Σ_q |π₁₃ T_q| ≤ K₂ + b_C s` have `Σ_q |T_q| ≤ √c (K₁ / b) (√(K₂ / b_C) + √s)`: Loomis–Whitney on
   each, `|T_q| ≤ √(|π₁₂| |π₂₃|) √|π₁₃|`, then Cauchy–Schwarz over `q`.
3. **A block's whole cells** (`System.sum_card_cellWhole_le`): 2 at `T_qu`, with P-share on `G` and `H` the values
   `I_u` has, `K₁ = k (1 + E)` by 1, and `K₂ = k`, `s = s_u = Σ_q Σ_ij |I_u ∩ start_qij| / b_C` by the partial sums
   received (`Standard.received_sums`), `I_u` reading at most `k` of `G`'s MACs' gates, the MACs being apart
   (`System.sum_card_macs_le`).
4. **The cap** (`SecurityProofs.SumCap`): 3 summed over blocks, with `Σ_u √s_u ≤ √(m Σ_u s_u)` and `Σ_u s_u ≤ O`, each
   start held once (`sum_sum_card_inter_le`), and the split cells, `w Σ_u Σ_q |U_u ∩ macs_q| ≤ m w k`
   (`Standard.card_le_cellWhole_add`).
5. **The bound** (`SecurityProofs.SumBound`): the cap inverted by `chainBlocks_le`.

P-share on a set within one class is P-share at that class's `c_j` (`System.sharedOn_of_shared`).
-/

namespace Nci.Products

open Finset
open Verity (Declaration Partition)
open Nci.Formalism
open scoped ENNReal

variable {d : Declaration}

/-- `√(a + b) ≤ √a + √b` for `a, b ≥ 0`. -/
theorem sqrt_add_le_add {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : √(a + b) ≤ √a + √b := by
  rw [Real.sqrt_le_left (by positivity)]
  nlinarith [Real.sq_sqrt ha, Real.sq_sqrt hb, mul_nonneg (Real.sqrt_nonneg a) (Real.sqrt_nonneg b)]

/-- **Several products in a block**: triples `T_q` over `q ∈ G` with `Σ_q |π₁₂ T_q| |π₂₃ T_q| ≤ c H²`, `b H ≤ K₁` and
`b_C Σ_q |π₁₃ T_q| ≤ K₂ + b_C s` have `Σ_q |T_q| ≤ √c (K₁ / b) (√(K₂ / b_C) + √s)`. -/
theorem sum_card_le_shared {ι : Type*} {α β γ : ι → Type*} [∀ q, DecidableEq (α q)] [∀ q, DecidableEq (β q)]
    [∀ q, DecidableEq (γ q)] (G : Finset ι) (T : (q : ι) → Finset (α q × β q × γ q)) {c H K₁ K₂ s b bC : ℝ}
    (hH : 0 ≤ H) (hb : 0 < b) (hC : 0 < bC) (hK₂ : 0 ≤ K₂) (hs : 0 ≤ s)
    (hxy : ∑ q ∈ G, (((T q).image fun x => (x.1, x.2.1)).card : ℝ) * ((T q).image fun x => (x.2.1, x.2.2)).card ≤
      c * H ^ 2)
    (hHK : b * H ≤ K₁) (hz : bC * ∑ q ∈ G, (((T q).image fun x => (x.1, x.2.2)).card : ℝ) ≤ K₂ + bC * s) :
    ∑ q ∈ G, ((T q).card : ℝ) ≤ √c * (K₁ / b) * (√(K₂ / bC) + √s) := by
  have hHb : H ≤ K₁ / b := by
    rw [le_div_iff₀ hb]
    linarith
  have hz' : ∑ q ∈ G, (((T q).image fun x => (x.1, x.2.2)).card : ℝ) ≤ K₂ / bC + s := by
    rw [div_add' _ _ _ hC.ne', le_div_iff₀ hC]
    linarith
  have h1 : √(∑ q ∈ G, (((T q).image fun x => (x.1, x.2.1)).card : ℝ) * ((T q).image fun x => (x.2.1, x.2.2)).card) ≤
      √c * H :=
    calc _ ≤ √(c * H ^ 2) := Real.sqrt_le_sqrt hxy
      _ = √c * H := by rw [Real.sqrt_mul' c (sq_nonneg H), Real.sqrt_sq hH]
  have h2 : √(∑ q ∈ G, (((T q).image fun x => (x.1, x.2.2)).card : ℝ)) ≤ √(K₂ / bC) + √s :=
    (Real.sqrt_le_sqrt hz').trans (sqrt_add_le_add (div_nonneg hK₂ hC.le) hs)
  calc ∑ q ∈ G, ((T q).card : ℝ)
      ≤ ∑ q ∈ G, √((((T q).image fun x => (x.1, x.2.1)).card : ℝ) * ((T q).image fun x => (x.2.1, x.2.2)).card) *
          √(((T q).image fun x => (x.1, x.2.2)).card : ℝ) :=
        sum_le_sum fun q _ => by
          rw [← Real.sqrt_mul (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))]
          exact Verity.Combinatorics.card_le_sqrt_proj (T q)
    _ ≤ √(∑ q ∈ G, (((T q).image fun x => (x.1, x.2.1)).card : ℝ) * ((T q).image fun x => (x.2.1, x.2.2)).card) *
          √(∑ q ∈ G, (((T q).image fun x => (x.1, x.2.2)).card : ℝ)) :=
        Real.sum_sqrt_mul_sqrt_le G (fun _ => mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
          (fun _ => Nat.cast_nonneg _)
    _ ≤ √c * H * (√(K₂ / bC) + √s) := mul_le_mul h1 h2 (Real.sqrt_nonneg _) (mul_nonneg (Real.sqrt_nonneg _) hH)
    _ ≤ √c * (K₁ / b) * (√(K₂ / bC) + √s) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hHb (Real.sqrt_nonneg _))
          (add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))

namespace System

variable {ι κ : Type} [Fintype ι] [DecidableEq ι] (sys : System d ι κ)

omit [Fintype ι] [DecidableEq ι] in
/-- **P-share on a set within one class** is P-share at that class's `c_j`. -/
theorem sharedOn_of_shared {c k : ℕ} {cs : κ → ℝ} (hsh : sys.Shared c k cs) {G : Finset ι} {j : κ}
    (hG : ∀ p ∈ G, sys.cls p = j) : sys.SharedOn c k (cs j) G :=
  fun I hI => hsh I hI j G hG

omit [Fintype ι] [DecidableEq ι] in
/-- An inference block reads at most `k` gates of a set's MACs, the set's products being apart. -/
theorem sum_card_macs_le {G : Finset ι} (hap : sys.Apart G) {c k : ℕ} {I : Finset (Fin d.D.N)}
    (hI : InferenceBlock d c k I) : ∑ q ∈ G, (d.unregistered I ∩ (sys.prod q).macs).card ≤ k := by
  rw [← card_biUnion fun p hp q hq h => (hap p hp q hq h).mono inter_subset_right inter_subset_right]
  exact (card_le_card (biUnion_subset.2 fun _ _ => inter_subset_left)).trans hI.2.2

/-- **Values had**: an inference block has at most `k (1 + E) / b` of the values of a set `G`'s products, `E` the
whole-making factors of `G`'s classes summed, the paid ones at their width from its `k` bits and the made ones from
the output entries whose whole chains it holds. -/
theorem card_hadValues_le [Fintype κ] [DecidableEq κ] {c k b : ℕ} {cs : κ → ℝ} {A : κ → κ → ℝ} {μ : κ → ℝ}
    (hS : ∀ p, (sys.prod p).Wiring) (hV : sys.Values) (hsh : sys.Shared c k cs) (hW : sys.WidthExtended c k b)
    (hM : sys.Matched A) (hb : 0 < b) (hcs : ∀ j, 0 ≤ cs j) (hA : ∀ i j, 0 ≤ A i j) (hμ : sys.PostFixed k b cs A μ)
    {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I) (G : Finset ι) :
    (b : ℝ) * (sys.hadValues G I).card ≤ k * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) := by
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  have hmemH : ∀ v ∈ sys.hadValues G I, ∃ p ∈ G, v ∈ (sys.prod p).hadCones I := fun v hv => mem_biUnion.1 hv
  rw [← card_filter_add_card_filter_not (s := sys.hadValues G I) (fun v => b ≤ (d.unregistered I ∩ v).card),
    Nat.cast_add, mul_add]
  have hpaid : b * ((sys.hadValues G I).filter fun v => b ≤ (d.unregistered I ∩ v).card).card ≤ k := by
    refine (mul_card_le_card_inter (fun v hv w hw h => ?_) fun v hv => (mem_filter.1 hv).2).trans
      ((card_le_card inter_subset_left).trans hI.2.2)
    obtain ⟨p, -, hvp⟩ := hmemH v (mem_filter.1 hv).1
    obtain ⟨p', -, hwp⟩ := hmemH w (mem_filter.1 hw).1
    exact sys.disjoint_of_ne hV ((sys.prod p).hadCones_subset_values I hvp)
      ((sys.prod p').hadCones_subset_values I hwp) h
  have hsub : ((sys.hadValues G I).filter fun v => ¬ b ≤ (d.unregistered I ∩ v).card) ⊆
      (G.image sys.cls).biUnion fun j =>
        ((sys.hadValues G I).filter fun v => ¬ b ≤ (d.unregistered I ∩ v).card).filter
          fun v => v ∈ sys.classValues j := fun v hv => by
    obtain ⟨p, hpG, hvp⟩ := hmemH v (mem_filter.1 hv).1
    exact mem_biUnion.2 ⟨sys.cls p, mem_image_of_mem _ hpG, mem_filter.2 ⟨hv,
      mem_biUnion.2 ⟨p, mem_filter.2 ⟨mem_univ _, rfl⟩, (sys.prod p).hadCones_subset_values I hvp⟩⟩⟩
  have hmade : ((((sys.hadValues G I).filter fun v => ¬ b ≤ (d.unregistered I ∩ v).card).card : ℕ) : ℝ) ≤
      (k : ℝ) / b * ∑ j ∈ G.image sys.cls, wholeFactor A μ j := by
    calc _ ≤ ((∑ j ∈ G.image sys.cls, (((sys.hadValues G I).filter fun v => ¬ b ≤ (d.unregistered I ∩ v).card).filter
            fun v => v ∈ sys.classValues j).card : ℕ) : ℝ) := by
          exact_mod_cast (card_le_card hsub).trans card_biUnion_le
      _ = ∑ j ∈ G.image sys.cls, ((((sys.hadValues G I).filter fun v => ¬ b ≤ (d.unregistered I ∩ v).card).filter
            fun v => v ∈ sys.classValues j).card : ℝ) := Nat.cast_sum _ _
      _ ≤ ∑ j ∈ G.image sys.cls, (k : ℝ) / b * wholeFactor A μ j := sum_le_sum fun j _ => by
          refine (sys.card_le_sum_fanout hM j _ fun v hv => ?_).trans
            (sys.sum_fanout_le hS hV hsh hW hM hb hcs hA hμ hI j)
          obtain ⟨hvV, hvj⟩ := mem_filter.1 hv
          obtain ⟨hvH, hvb⟩ := mem_filter.1 hvV
          obtain ⟨p, -, hvp⟩ := hmemH v hvH
          obtain ⟨x, hxs, hx⟩ := (hW I hI p v hvp).resolve_left hvb
          exact ⟨hvj, x, hxs, mem_sigma.2 ⟨mem_univ _, mem_filter.2 ⟨mem_univ _, hx⟩⟩⟩
      _ = (k : ℝ) / b * ∑ j ∈ G.image sys.cls, wholeFactor A μ j := (mul_sum _ _ _).symm
  have hp : (b : ℝ) * ((((sys.hadValues G I).filter fun v => b ≤ (d.unregistered I ∩ v).card).card : ℕ) : ℝ) ≤ k := by
    exact_mod_cast hpaid
  have hbk : (b : ℝ) * ((k : ℝ) / b * ∑ j ∈ G.image sys.cls, wholeFactor A μ j) =
      k * ∑ j ∈ G.image sys.cls, wholeFactor A μ j := by
    field_simp
  linarith [mul_le_mul_of_nonneg_left hmade hb'.le]

/-- **A block's whole cells**: in block `u` of a partition into inference blocks, the MACs of a set `G`'s cells that it
holds whole number at most `√c_G (k (1 + E) / b) (√(k / b_C) + √s_u)`, `E` the whole-making factors of `G`'s classes
summed and `s_u` the gates of `G`'s starts it holds over `b_C`. -/
theorem sum_card_cellWhole_le [Fintype κ] [DecidableEq κ] {c k b bC : ℕ} {cs : κ → ℝ} {A : κ → κ → ℝ}
    {μ : κ → ℝ} (hS : ∀ p, (sys.prod p).Wiring) (hV : sys.Values) (hsh : sys.Shared c k cs)
    (hW : sys.WidthExtended c k b) (hM : sys.Matched A) (hb : 0 < b) (hC : 0 < bC) (hcs : ∀ j, 0 ≤ cs j)
    (hA : ∀ i j, 0 ≤ A i j) (hμ : sys.PostFixed k b cs A μ) {G : Finset ι} {cG : ℝ}
    (hsO : sys.SharedOn c k cG G) (hap : sys.Apart G)
    (cell : (q : ι) → Fin (sys.ni q) → Fin (sys.K (sys.cls q)) → Fin (sys.nj q) → ℕ)
    (start : (q : ι) → Fin (sys.ni q) → Fin (sys.nj q) → Finset (Fin d.D.N))
    (hread : ∀ q ∈ G, ∀ I, InferenceBlock d c k I → ∀ i l j,
      (sys.prod q).cellGates (cell q) (cell q i l j) ⊆ I →
        bC ≤ (d.unregistered I ∩ (sys.prod q).entry i j).card + (I ∩ start q i j).card)
    {m : ℕ} {P : Partition d.D m} (hP : P.IsCK d c k) (u : Fin m) :
    ∑ q ∈ G, (((sys.prod q).cellWhole (cell q) (P.gates u)).card : ℝ) ≤
      √cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) *
        (√((k : ℝ) / bC) + √(((∑ q ∈ G, ∑ e : Fin (sys.ni q) × Fin (sys.nj q),
          (P.gates u ∩ start q e.1 e.2).card : ℕ) : ℝ) / bC)) := by
  have hI := inferenceBlock_gates hP u
  have hbpos : (0 : ℝ) < b := by exact_mod_cast hb
  have hCpos : (0 : ℝ) < bC := by exact_mod_cast hC
  have hxy : ∑ q ∈ G, ((((sys.prod q).cellWhole (cell q) (P.gates u)).image fun x => (x.1, x.2.1)).card : ℝ) *
      (((sys.prod q).cellWhole (cell q) (P.gates u)).image fun x => (x.2.1, x.2.2)).card ≤
      cG * ((sys.hadValues G (P.gates u)).card : ℝ) ^ 2 := by
    refine le_trans ?_ (hsO _ hI)
    push_cast
    refine sum_le_sum fun q _ => mul_le_mul ?_ ?_ (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    · exact_mod_cast card_le_card ((image_subset_image ((sys.prod q).cellWhole_subset_whole (cell q) _)).trans
        ((sys.prod q).image_whole_subset_hadA (hS q) _))
    · exact_mod_cast card_le_card ((image_subset_image ((sys.prod q).cellWhole_subset_whole (cell q) _)).trans
        ((sys.prod q).image_whole_subset_hadB (hS q) _))
  have hz : (bC : ℝ) * ∑ q ∈ G,
      ((((sys.prod q).cellWhole (cell q) (P.gates u)).image fun x => (x.1, x.2.2)).card : ℝ) ≤
      k + bC * (((∑ q ∈ G, ∑ e : Fin (sys.ni q) × Fin (sys.nj q),
        (P.gates u ∩ start q e.1 e.2).card : ℕ) : ℝ) / bC) := by
    rw [mul_div_cancel₀ _ hCpos.ne', mul_sum]
    have h1 : ∑ q ∈ G, bC * (((sys.prod q).cellWhole (cell q) (P.gates u)).image fun x => (x.1, x.2.2)).card ≤
        ∑ q ∈ G, (d.unregistered (P.gates u) ∩ (sys.prod q).macs).card +
          ∑ q ∈ G, ∑ e : Fin (sys.ni q) × Fin (sys.nj q), (P.gates u ∩ start q e.1 e.2).card := by
      rw [← sum_add_distrib]
      exact sum_le_sum fun q hq => (sys.prod q).received_sums (cell q) (hS q) (hread q hq) hP u
    have h2 := h1.trans (Nat.add_le_add_right (sys.sum_card_macs_le hap hI) _)
    exact_mod_cast h2
  exact sum_card_le_shared G (fun q => (sys.prod q).cellWhole (cell q) (P.gates u)) (Nat.cast_nonneg _) hbpos hCpos
    (Nat.cast_nonneg _) (div_nonneg (Nat.cast_nonneg _) hCpos.le) hxy
    (sys.card_hadValues_le hS hV hsh hW hM hb hcs hA hμ hI G) hz

end System

namespace SecurityProofs

theorem SumCap : Guarantees.SumCap := by
  intro d ι κ _ _ _ _ sys c k b bC cs A μ G cG cell w m P hS hV hsh hW hM hsO hap hc hR hP hb hC hcs hA hμ
  have hR' : ∀ q, ∃ start : Fin (sys.ni q) → Fin (sys.nj q) → Finset (Fin d.D.N), (∀ i j, (start i j).card ≤ bC) ∧
      (q ∈ G → ∀ I, InferenceBlock d c k I → ∀ i l j, (sys.prod q).cellGates (cell q) (cell q i l j) ⊆ I →
        bC ≤ (d.unregistered I ∩ (sys.prod q).entry i j).card + (I ∩ start i j).card) := fun q => by
    by_cases hq : q ∈ G
    · obtain ⟨start, hcard, hread⟩ := hR q hq
      exact ⟨start, hcard, fun _ => hread⟩
    · exact ⟨fun _ _ => ∅, fun _ _ => by simp, fun h => absurd h hq⟩
  choose start hcard hread using hR'
  have hCpos : (0 : ℝ) < bC := by exact_mod_cast hC
  have hbpos : (0 : ℝ) < b := by exact_mod_cast hb
  have hE : 0 ≤ ∑ j ∈ G.image sys.cls, wholeFactor A μ j :=
    sum_nonneg fun j _ => sum_nonneg fun i _ => mul_nonneg (hA _ _) (sys.nonneg_of_postFixed hcs hμ i)
  have hL : 0 ≤ √cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) :=
    mul_nonneg (Real.sqrt_nonneg _) (div_nonneg (mul_nonneg (Nat.cast_nonneg _) (by linarith)) hbpos.le)
  have hstarts : ∑ u, ∑ q ∈ G, ∑ e : Fin (sys.ni q) × Fin (sys.nj q), (P.gates u ∩ start q e.1 e.2).card ≤
      (∑ q ∈ G, sys.ni q * sys.nj q) * bC := by
    rw [sum_comm, sum_mul]
    exact sum_le_sum fun q _ => sum_sum_card_inter_le (hcard q) P
  have hO : ∑ u, ((∑ q ∈ G, ∑ e : Fin (sys.ni q) × Fin (sys.nj q), (P.gates u ∩ start q e.1 e.2).card : ℕ) : ℝ) /
      bC ≤ ((∑ q ∈ G, sys.ni q * sys.nj q : ℕ) : ℝ) := by
    rw [← sum_div, div_le_iff₀ hCpos]
    exact_mod_cast hstarts
  have hsq : ∑ u, √(((∑ q ∈ G, ∑ e : Fin (sys.ni q) × Fin (sys.nj q), (P.gates u ∩ start q e.1 e.2).card : ℕ) : ℝ) /
      bC) ≤ √((m : ℝ) * (∑ q ∈ G, sys.ni q * sys.nj q : ℕ)) := by
    have h := Real.sum_sqrt_mul_sqrt_le univ (f := fun _ : Fin m => (1 : ℝ))
      (g := fun u => ((∑ q ∈ G, ∑ e : Fin (sys.ni q) × Fin (sys.nj q), (P.gates u ∩ start q e.1 e.2).card : ℕ) : ℝ) /
        bC) (fun _ => zero_le_one) (fun _ => div_nonneg (Nat.cast_nonneg _) hCpos.le)
    simp only [Real.sqrt_one, one_mul, sum_const, card_univ, nsmul_eq_mul, mul_one, Fintype.card_fin] at h
    refine h.trans ?_
    rw [← Real.sqrt_mul (Nat.cast_nonneg _)]
    exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hO (Nat.cast_nonneg _))
  have hT : ∑ u, ∑ q ∈ G, (((sys.prod q).cellWhole (cell q) (P.gates u)).card : ℝ) ≤
      m * (√cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) * √((k : ℝ) / bC)) +
        √cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) *
          √((m : ℝ) * (∑ q ∈ G, sys.ni q * sys.nj q : ℕ)) :=
    calc _ ≤ ∑ u, √cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) *
          (√((k : ℝ) / bC) + √(((∑ q ∈ G, ∑ e : Fin (sys.ni q) × Fin (sys.nj q),
            (P.gates u ∩ start q e.1 e.2).card : ℕ) : ℝ) / bC)) :=
          sum_le_sum fun u _ => sys.sum_card_cellWhole_le hS hV hsh hW hM hb hC hcs hA hμ hsO hap cell start
            (fun q hq => hread q hq) hP u
      _ = m * (√cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) * √((k : ℝ) / bC)) +
          √cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) *
            ∑ u, √(((∑ q ∈ G, ∑ e : Fin (sys.ni q) × Fin (sys.nj q), (P.gates u ∩ start q e.1 e.2).card : ℕ) : ℝ) /
              bC) := by
          rw [← mul_sum, sum_add_distrib, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring
      _ ≤ _ := add_le_add le_rfl (mul_le_mul_of_nonneg_left hsq hL)
  have hsplit : ∑ u, ∑ q ∈ G, ((d.unregistered (P.gates u) ∩ (sys.prod q).macs).card : ℝ) ≤ m * k :=
    calc _ ≤ ∑ _u : Fin m, (k : ℝ) := sum_le_sum fun u _ => by
          exact_mod_cast sys.sum_card_macs_le hap (inferenceBlock_gates hP u)
      _ = m * k := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hN : ((∑ q ∈ G, sys.ni q * sys.K (sys.cls q) * sys.nj q : ℕ) : ℝ) ≤
      ∑ u, ∑ q ∈ G, (((sys.prod q).cellWhole (cell q) (P.gates u)).card : ℝ) +
        w * ∑ u, ∑ q ∈ G, ((d.unregistered (P.gates u) ∩ (sys.prod q).macs).card : ℝ) := by
    rw [sum_comm (s := univ) (t := G), sum_comm (s := univ) (t := G), mul_sum, ← sum_add_distrib, Nat.cast_sum]
    exact sum_le_sum fun q hq => by
      exact_mod_cast (sys.prod q).card_le_cellWhole_add (cell q) (hS q) (hc q hq) P
  calc _ ≤ _ := hN
    _ ≤ m * (√cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) * √((k : ℝ) / bC)) +
          √cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) *
            √((m : ℝ) * (∑ q ∈ G, sys.ni q * sys.nj q : ℕ)) + w * (m * k) :=
        add_le_add hT (mul_le_mul_of_nonneg_left hsplit (Nat.cast_nonneg _))
    _ = _ := by ring

theorem SumBound : Guarantees.SumBound := by
  intro d ι κ _ _ _ _ sys c k b bC cs A μ G cG cell w hS hV hsh hW hM hsO hap hc hR hb hC hcs hA hμ m P hP
  have hcap := SumCap d ι κ sys c k b bC cs A μ G cG cell w m P hS hV hsh hW hM hsO hap hc hR hP hb hC hcs hA hμ
  have hbpos : (0 : ℝ) < b := by exact_mod_cast hb
  have hE : 0 ≤ ∑ j ∈ G.image sys.cls, wholeFactor A μ j :=
    sum_nonneg fun j _ => sum_nonneg fun i _ => mul_nonneg (hA _ _) (sys.nonneg_of_postFixed hcs hμ i)
  have hL : 0 ≤ √cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) :=
    mul_nonneg (Real.sqrt_nonneg _) (div_nonneg (mul_nonneg (Nat.cast_nonneg _) (by linarith)) hbpos.le)
  have hm : chainBlocks ((∑ q ∈ G, sys.ni q * sys.K (sys.cls q) * sys.nj q : ℕ) : ℝ)
      (√cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) * √((k : ℝ) / bC) + w * k)
      (√cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) *
        √((∑ q ∈ G, sys.ni q * sys.nj q : ℕ) : ℝ)) ≤ m := by
    refine chainBlocks_le (Nat.cast_nonneg _) (add_nonneg (mul_nonneg hL (Real.sqrt_nonneg _)) (by positivity))
      (mul_nonneg hL (Real.sqrt_nonneg _)) (Nat.cast_nonneg _) ?_
    rw [Real.sqrt_mul (Nat.cast_nonneg (α := ℝ) m) ((∑ q ∈ G, sys.ni q * sys.nj q : ℕ) : ℝ)] at hcap
    linarith
  unfold sumBound partitionSlowdown
  refine ENNReal.div_le_div_right ?_ _
  rw [show ((m * c : ℕ) : ℝ≥0∞) = (m : ℝ≥0∞) * c from Nat.cast_mul m c, ← ENNReal.ofReal_natCast m]
  gcongr

end SecurityProofs

end Nci.Products
