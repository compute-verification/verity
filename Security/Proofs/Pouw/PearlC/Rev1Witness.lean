import Proofs.Pouw.PearlC.DeviceRev1Gamma

/-!
# rev1's H100 rows are satisfiable (the cheap-binding lane's file)

Staged; nothing pinned. The pins proposed are `ttOutPearlCH100Rev1Witness`, `ttOutTilePearlCH100Rev1Witness` and
`pearlCTilesDevRev1_total` (rev1's tiles cover each unit's words once, so `GγSampled`'s W is the layout's rev1 `W_ref`).

Each shows the named row (`TTOutPearlCH100Rev1`, `TTOutTilePearlCH100Rev1`) holding beside an honest reference whose cost
fits rev1's `W_ref` (unit by unit, or tile by tile), on an in-domain workload whose one unit or tile is within the cap and
has positive rev1 credit. The cost models are `witnessCM` and `witnessTileCM` at rev1's protocol and tiles; they show
satisfiability only.
-/

namespace Pouw.PearlC

open Finset Assumptions Pouw.Fp8Atom

/-- rev1's credit at the H100 record is non-negative at every shape: the peel alone credits `128·m·n`, more than the
first add's `32·m·n`. -/
theorem creditDevRev1_h100_nonneg (s : Shape) : 0 ≤ creditDevRev1 devH100 s := by
  rw [creditDevRev1_h100]
  simp only [creditOfRev1, firstAddOf, creditOf, Params.pi, Costs.adopted]
  norm_num
  have hm : (0 : ℚ) ≤ s.m := by positivity
  have hn : (0 : ℚ) ≤ s.n := by positivity
  have hk : (0 : ℚ) ≤ s.k := by positivity
  have h1 : (0 : ℚ) ≤ s.m * s.k * s.n := by positivity
  have h2 : (0 : ℚ) ≤ s.m * s.k := by positivity
  have h3 : (0 : ℚ) ≤ s.m * s.n := by positivity
  nlinarith

theorem creditDevRev1_le_wref_h100 (s : Shape) : creditDevRev1 devH100 s ≤ wrefDevRev1 devH100 s := by
  unfold wrefDevRev1
  have : (0 : ℚ) ≤ (devH100.prices.costs.qa : ℚ) * s.m * s.k := by positivity
  linarith

/-- On a layout of nonempty units, rev1's audit tiles' `W_ref` adds up to the layout's `∑_u wrefDevRev1`. -/
theorem pearlCTilesDevRev1_total {Q R S : Type} (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (L : Layout)
    (hpos : ∀ u < L.N, 0 < (L.shape u).m ∧ 0 < (L.shape u).n) :
    tileWrefTotal (pearlCTilesDevRev1 d sem ρ) L = ∑ u ∈ range L.N, (wrefDevRev1 d (L.shape u) : ℝ) := by
  let F : ℕ × ℕ × ℕ → ℝ := fun b => ((wrefDevRev1 d (L.shape b.1) * tileShare (L.shape b.1)
    (Ico (b.2.1 * 64) (min ((b.2.1 + 1) * 64) (L.shape b.1).m))
    (Ico (b.2.2 * 64) (min ((b.2.2 + 1) * 64) (L.shape b.1).n)) : ℚ) : ℝ)
  have hdef : tileWrefTotal (pearlCTilesDevRev1 d sem ρ) L =
      ∑ g ∈ range (gridTiles 64 64 L).length, F ((gridTiles 64 64 L).getD g (0, 0, 0)) := rfl
  rw [hdef, sum_range_getD, gridTiles, sum_map_flatMap, sum_map_range]
  refine sum_congr rfl fun u hu => ?_
  obtain ⟨hm, hn⟩ := hpos u (mem_range.mp hu)
  rw [sum_map_flatMap, sum_map_range]
  have inner : ∀ bi, (((List.range (((L.shape u).n + 64 - 1) / 64)).map fun bj => (u, bi, bj)).map F).sum
      = ∑ bj ∈ range (((L.shape u).n + 64 - 1) / 64), F (u, bi, bj) := fun bi => by
    rw [List.map_map]
    exact sum_map_range _ _
  rw [sum_congr rfl fun bi _ => inner bi]
  rw [← unit_blocks (wrefDevRev1 d (L.shape u)) _ _ hm hn, Rat.cast_sum]
  refine sum_congr rfl fun bi _ => ?_
  rw [Rat.cast_sum]
  rfl

/-- A word of at most one promotion group has no later promotion into a +0 total. -/
theorem laterZeroPromos_eq_zero_of_le (d : PearlCDevice) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n) (hk : k / (32 * d.G) ≤ 1) :
    laterZeroPromos d A B i j = 0 := by
  unfold laterZeroPromos
  split_ifs
  · rfl
  · rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    intro g hg
    rw [Finset.mem_Ico] at hg
    omega

theorem ttOutPearlCH100Rev1Witness :
    AdmitsRef (witnessCM (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)))
        (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) (pearlCDomainDev devH100) (fun _ => 0) ∧
      TTOutPearlCH100Rev1 (witnessCM (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400))) witnessSem (1 / 400) ∧
      witnessWorkload.InDomain (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) (pearlCDomainDev devH100) ∧
      witnessWorkload.WeightsOK (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) ∧
      (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)).capOK (fun _ => ()) () witnessWorkload 0 (witnessAct 0) ∧
      0 < ∑ u ∈ range witnessWorkload.layout.N,
        (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)).credit (fun _ => ()) () witnessWorkload u (witnessAct u) := by
  set P := pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)
  have hzero0 : debitDev devH100 (witnessSem.unitDebitDev devH100 (fun _ => ()) () witnessWorkload 0 (witnessAct 0)) = 0 := by
    rw [unitDebitDev_h100, debitDev_h100]
    obtain ⟨ha, he⟩ := witness_debit_zero (fun _ => ()) ()
    unfold debitOf
    rw [ha, he]
    simp
  have hzero : witnessSem.unitDebitRev1 devH100 (fun _ => ()) () witnessWorkload 0 (witnessAct 0) = 0 := by
    unfold PearlCSem.unitDebitRev1
    rw [hzero0]
    simp (disch := decide) only [laterZeroPromos_eq_zero_of_le, Nat.cast_zero, sum_const_zero, mul_zero,
      add_zero]
  have hdom := ttOutPearlCWitness.2.2.1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨true, fun U _ _ act _ H s => ⟨by simp [witnessCM], ?_, by simp [witnessCM]⟩⟩
    show (if true = true then ∑ u ∈ range U.layout.N, max (P.credit H s U u (act u)) 0 else 0) ≤ _
    simp only [↓reduceIte]
    apply sum_le_sum
    intro u _
    show max (((creditDevRev1 devH100 (U.layout.shape u) -
        witnessSem.unitDebitRev1 devH100 H s U u (act u) : ℚ) : ℝ)) 0 ≤
      (wrefDevRev1 devH100 (U.layout.shape u) : ℝ)
    have h0 := unitDebitRev1_nonneg witnessSem devH100 (by norm_num [devH100, Prices.h100]) H s U u (act u)
    have h1 := creditDevRev1_le_wref_h100 (U.layout.shape u)
    have h2 := creditDevRev1_h100_nonneg (U.layout.shape u)
    apply max_le
    · exact_mod_cast (by linarith)
    · exact_mod_cast (by linarith)
  · refine ⟨fun prog U act σ s => ⟨QTree.done ((witnessCM P).run prog (fun _ => ()) U act σ s), fun H => ?_⟩, ?_⟩
    · have hH : H = fun _ => () := funext fun _ => Subsingleton.elim _ _
      subst hH
      exact ⟨rfl, rfl⟩
    · intro U _ T hT q _ A hA
      rw [Pouw.pr_eq_zero]
      · unfold εPearlC
        positivity
      intro x hx
      have hsum := witness_sum_le P U x.1 x.2 A.online A.act (A.pre.eval x.1)
      have hcost := (hA x.1 x.2).1
      have hT' : T ≤ T / (1 - 1 / 400) := by
        rw [le_div_iff₀ (by norm_num)]
        nlinarith
      unfold Adversary.transcript at hx
      linarith
  · rw [pearlCDomainDev_h100]
    exact hdom
  · exact ttOutPearlCWitness.2.2.2.1
  · show witnessSem.unitDebitRev1 devH100 (fun _ => ()) () witnessWorkload 0 (witnessAct 0) ≤
      1 / 400 * creditDevRev1 devH100 (witnessWorkload.layout.shape 0)
    rw [hzero]
    have := creditDevRev1_h100_nonneg (witnessWorkload.layout.shape 0)
    positivity
  · rw [show range witnessWorkload.layout.N = range 1 from rfl, sum_range_one]
    show (0 : ℝ) < ((creditDevRev1 devH100 (witnessWorkload.layout.shape 0) -
      witnessSem.unitDebitRev1 devH100 (fun _ => ()) () witnessWorkload 0 (witnessAct 0) : ℚ) : ℝ)
    rw [hzero, creditDevRev1_h100]
    norm_num [creditOfRev1, firstAddOf, creditOf, Params.pi, Costs.adopted, witnessWorkload]

theorem ttOutTilePearlCH100Rev1Witness :
    AdmitsRefTile (witnessTileCM (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400))
        (pearlCTilesDevRev1 devH100 witnessSem (1 / 400)))
      (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) (pearlCTilesDevRev1 devH100 witnessSem (1 / 400))
      (pearlCDomainDev devH100) (fun _ => 0) ∧
    TTOutTilePearlCH100Rev1 (witnessTileCM (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400))
        (pearlCTilesDevRev1 devH100 witnessSem (1 / 400))) witnessSem (1 / 400) ∧
    witnessWorkload.InDomain (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) (pearlCDomainDev devH100) ∧
    witnessWorkload.WeightsOK (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) ∧
    (pearlCTilesDevRev1 devH100 witnessSem (1 / 400)).capOK (fun _ => ()) () witnessWorkload 0 (witnessAct 0) ∧
    0 < ∑ g ∈ range ((pearlCTilesDevRev1 devH100 witnessSem (1 / 400)).tiling witnessWorkload.layout).NT,
      (pearlCTilesDevRev1 devH100 witnessSem (1 / 400)).credit (fun _ => ()) () witnessWorkload g (witnessAct 0) := by
  set P := pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)
  set TR := pearlCTilesDevRev1 devH100 witnessSem (1 / 400)
  have hNT : (TR.tiling witnessWorkload.layout).NT = 1 := by
    show (gridTiles 64 64 witnessWorkload.layout).length = 1
    rw [witness_gridTiles]; rfl
  have hu : (TR.tiling witnessWorkload.layout).unit 0 = 0 := by
    show ((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).1 = 0
    rw [witness_gridTiles]; rfl
  have hr : (TR.tiling witnessWorkload.layout).rows 0 = Ico 0 1 := by
    show Ico (((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.1 * 64)
      (min ((((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.1 + 1) * 64)
        (witnessWorkload.layout.shape ((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).1).m) = Ico 0 1
    rw [witness_gridTiles]; rfl
  have hc : (TR.tiling witnessWorkload.layout).cols 0 = Ico 0 1 := by
    show Ico (((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.2 * 64)
      (min ((((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.2 + 1) * 64)
        (witnessWorkload.layout.shape ((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).1).n) = Ico 0 1
    rw [witness_gridTiles]; rfl
  have hdom := ttOutPearlCWitness.2.2.1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨true, fun U _ _ act _ H s => ⟨by simp [witnessTileCM], ?_, by simp [witnessTileCM]⟩⟩
    show (if true = true then ∑ g ∈ range (TR.tiling U.layout).NT,
        max (TR.credit H s U g (act ((TR.tiling U.layout).unit g))) 0 else 0) ≤ tileWrefTotal TR U.layout
    simp only [↓reduceIte, tileWrefTotal]
    apply sum_le_sum
    intro g _
    simp only [TR, pearlCTilesDevRev1]
    set sh := U.layout.shape ((auditTiling 64 64 U.layout).unit g)
    set rows := (auditTiling 64 64 U.layout).rows g
    set cols := (auditTiling 64 64 U.layout).cols g
    set pr := witnessSem.passRows sh (act ((auditTiling 64 64 U.layout).unit g)) rows
    have h1 := creditDevRev1_le_wref_h100 sh
    have h2 := creditDevRev1_h100_nonneg sh
    have h3 := tileShare_nonneg sh pr cols
    have h4 := tileShare_mono sh (passRows_subset witnessSem sh (act ((auditTiling 64 64 U.layout).unit g)) rows) cols
    have h5 : 0 ≤ wrefDevRev1 devH100 sh := le_trans h2 h1
    have key : (1 - 1 / 400) * creditDevRev1 devH100 sh * tileShare sh pr cols ≤
        wrefDevRev1 devH100 sh * tileShare sh rows cols :=
      calc (1 - 1 / 400) * creditDevRev1 devH100 sh * tileShare sh pr cols
          ≤ wrefDevRev1 devH100 sh * tileShare sh pr cols := mul_le_mul_of_nonneg_right (by linarith) h3
        _ ≤ wrefDevRev1 devH100 sh * tileShare sh rows cols := mul_le_mul_of_nonneg_left h4 h5
    apply max_le
    · exact_mod_cast key
    · exact_mod_cast mul_nonneg h5 (tileShare_nonneg sh rows cols)
  · refine ⟨fun prog U act σ s => ⟨QTree.done ((witnessTileCM P TR).run prog (fun _ => ()) U act σ s),
      fun H => ?_⟩, ?_⟩
    · have hH : H = fun _ => () := funext fun _ => Subsingleton.elim _ _
      subst hH
      exact ⟨rfl, rfl⟩
    · intro U _ T hT q _ A hA
      rw [Pouw.pr_eq_zero]
      · unfold εPearlC
        positivity
      intro x hx
      have hsum := witnessTile_sum_le P TR U x.1 x.2 A.online A.act (A.pre.eval x.1)
      have hcost := (hA x.1 x.2).1
      have hT' : T ≤ T / (1 - 1 / 400) := by
        rw [le_div_iff₀ (by norm_num)]
        nlinarith
      unfold Adversary.transcript at hx
      linarith
  · rw [pearlCDomainDev_h100]
    exact hdom
  · exact ttOutPearlCWitness.2.2.2.1
  · show witnessSem.tileDebitRev1 devH100 (fun _ => ()) () witnessWorkload ((TR.tiling witnessWorkload.layout).unit 0)
        (witnessSem.passRows (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0)) (witnessAct 0)
          ((TR.tiling witnessWorkload.layout).rows 0)) ((TR.tiling witnessWorkload.layout).cols 0) (witnessAct 0) ≤
      1 / 400 * creditDevRev1 devH100 (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0)) *
        tileShare (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0))
          (witnessSem.passRows (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0)) (witnessAct 0)
            ((TR.tiling witnessWorkload.layout).rows 0)) ((TR.tiling witnessWorkload.layout).cols 0)
    rw [hu, hr, hc, witness_passRows]
    unfold PearlCSem.tileDebitRev1
    rw [tileDebitDev_h100, witness_tileDebit_zero]
    simp (disch := decide) only [laterZeroPromos_eq_zero_of_le, Nat.cast_zero, sum_const_zero, mul_zero,
      add_zero]
    have h2 := creditDevRev1_h100_nonneg (witnessWorkload.layout.shape 0)
    have h3 := tileShare_nonneg (witnessWorkload.layout.shape 0) (Ico 0 1) (Ico 0 1)
    positivity
  · rw [hNT, sum_range_one]
    show (0 : ℝ) < (((1 - 1 / 400) * creditDevRev1 devH100
        (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0)) *
        tileShare (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0))
          (witnessSem.passRows (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0))
            (witnessAct 0) ((TR.tiling witnessWorkload.layout).rows 0))
          ((TR.tiling witnessWorkload.layout).cols 0) : ℚ) : ℝ)
    rw [hu, hr, hc, witness_passRows, creditDevRev1_h100]
    norm_num [creditOfRev1, firstAddOf, creditOf, tileShare, Params.pi, Costs.adopted, witnessWorkload]

end Pouw.PearlC
