import Proofs.Pouw.PearlC.DeviceRev1Gamma

/-!
# rev1's H100 rows are satisfiable

Staged; nothing pinned. The pins proposed are `ttOutPearlCH100Rev1Witness`, `ttOutTilePearlCH100Rev1Witness` and
`pearlCTilesDevRev1_total` (rev1's tiles cover each unit's words once, so `GγSampled`'s W is the layout's rev1 `W_ref`).

Each shows the named row (`TTOutPearlCH100Rev1`, `TTOutTilePearlCH100Rev1`) holding beside an honest reference whose cost
fits rev1's `W_ref` (unit by unit, or tile by tile), on an in-domain workload (`wideWorkload`, one `1 × 1024 × 128` unit)
whose unit, or first tile, is within the cap and has positive rev1 credit. The cost models are `witnessCM` and `witnessTileCM` at rev1's protocol and tiles; they show
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
theorem pearlCTilesDevRev1_total {Q R S : Type} (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (L : UnitLayout)
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

/-! ## A workload in the device domain

`witnessWorkload`'s one column is outside `pearlCDomainDev`, which needs `64 ≤ n` and `64 ∣ n`; the rows below are
witnessed on one unit of shape `1 × 1024 × 128`, every word `1.0`. Its flags, debit and rules are `witnessWorkload`'s,
column by column. -/

/-- One unit of shape `1 × 1024 × 128` in the device domain, every word `1.0`. -/
def wideWorkload : Workload := ⟨⟨1, fun _ => ⟨1, 1024, 128⟩, fun _ => 0⟩, fun _ _ _ => 1065353216⟩

theorem wide_rowA (H : Unit → Unit) (s : Unit) (i : Fin (wideWorkload.layout.shape 0).m) :
    (witnessSem.unitChain H s wideWorkload 0 (witnessAct 0)).1 i = fun _ => e4m3 1 := by
  funext l
  simp [PearlCSem.unitChain, witnessSem, FormChain.quantA, FormChain.quant, FormChain.elem, FormChain.exact, wordsVal,
    witnessAct, wideWorkload, wordQ_one, codeVal_zero]

theorem wide_rowB (H : Unit → Unit) (s : Unit) (j : Fin (wideWorkload.layout.shape 0).n) :
    (witnessSem.unitChain H s wideWorkload 0 (witnessAct 0)).2 j = fun _ => e4m3 1 := by
  funext l
  simp [PearlCSem.unitChain, witnessSem, FormChain.quantB, FormChain.quant, FormChain.elem, FormChain.exact, wordsVal,
    wideWorkload, wordQ_one, codeVal_zero]

theorem wide_unitFlags (H : Unit → Unit) (s : Unit) (i : Fin (wideWorkload.layout.shape 0).m)
    (j : Fin (wideWorkload.layout.shape 0).n) (t : ℕ) (ht : t < 4) :
    Skip.unitFlags (witnessSem.unitChain H s wideWorkload 0 (witnessAct 0)).1
      (witnessSem.unitChain H s wideWorkload 0 (witnessAct 0)).2 i j t = false := by
  unfold Skip.unitFlags
  rw [wide_rowA, wide_rowB]
  exact witness_flags t ht

open Classical in
theorem wide_debit_zero (H : Unit → Unit) (s : Unit) :
    (witnessSem.unitDebit H s wideWorkload 0 (witnessAct 0)).atoms = 0 ∧
      (witnessSem.unitDebit H s wideWorkload 0 (witnessAct 0)).elems = 0 := by
  constructor
  · unfold PearlCSem.unitDebit
    dsimp only
    apply sum_eq_zero
    intro i _
    apply sum_eq_zero
    intro j _
    rw [card_eq_zero, filter_eq_empty_iff]
    intro t ht
    have ht' : t < 4 := by simpa [wideWorkload] using mem_range.mp ht
    rw [wide_unitFlags H s i j t ht']
    simp
  · unfold PearlCSem.unitDebit
    dsimp only
    rw [card_eq_zero, filter_eq_empty_iff]
    intro e _
    rintro (hd | hn)
    · exact witness_saltLive _ _ _ (by simp [wordsVal, witnessAct, wordQ_one]) hd
    apply hn
    unfold Skip.needed
    have hl : (e.2 : ℕ) / 32 < 4 := by
      have := e.2.isLt
      simp only [wideWorkload] at this
      omega
    exact ⟨⟨0, by simp [wideWorkload]⟩, wide_unitFlags H s e.1 _ _ hl⟩

open Classical in
/-- The wide workload's first tile (row 0, columns `[0, 64)`) has zero debit. -/
theorem wide_tileDebit_zero (H : Unit → Unit) (s : Unit) :
    witnessSem.tileDebit Costs.adopted H s wideWorkload 0 (Ico 0 1) (Ico 0 64) (witnessAct 0) = 0 := by
  unfold PearlCSem.tileDebit
  dsimp only
  have ha : ∑ e ∈ (univ : Finset (Fin (wideWorkload.layout.shape 0).m × Fin (wideWorkload.layout.shape 0).n)).filter
      (fun e => (e.1 : ℕ) ∈ Ico 0 1 ∧ (e.2 : ℕ) ∈ Ico 0 64),
      ((range ((wideWorkload.layout.shape 0).k / 32)).filter fun t =>
        Skip.unitFlags (witnessSem.unitChain H s wideWorkload 0 (witnessAct 0)).1
          (witnessSem.unitChain H s wideWorkload 0 (witnessAct 0)).2 e.1 e.2 t = true).card = 0 := by
    apply sum_eq_zero
    intro e _
    rw [card_eq_zero, filter_eq_empty_iff]
    intro t ht
    have ht' : t < 4 := by simpa [wideWorkload] using mem_range.mp ht
    rw [wide_unitFlags H s e.1 e.2 t ht']
    simp
  have he : ((univ : Finset (Fin (wideWorkload.layout.shape 0).m × Fin (wideWorkload.layout.shape 0).k)).filter
      fun e => (e.1 : ℕ) ∈ Ico 0 1 ∧
        (SaltDead.saltDead witnessSem.alpha witnessSem.beta (wideWorkload.layout.shape 0).k
            (wordsVal (wideWorkload.layout.shape 0).m (wideWorkload.layout.shape 0).k (witnessAct 0) e.1) e.2 ∨
          ∀ j : Fin (wideWorkload.layout.shape 0).n, (j : ℕ) ∈ Ico 0 64 →
            Skip.unitFlags (witnessSem.unitChain H s wideWorkload 0 (witnessAct 0)).1
              (witnessSem.unitChain H s wideWorkload 0 (witnessAct 0)).2 e.1 j ((e.2 : ℕ) / 32) = true)).card = 0 := by
    rw [card_eq_zero, filter_eq_empty_iff]
    rintro e _ ⟨_, hd | hall⟩
    · exact witness_saltLive _ _ _ (by simp [wordsVal, witnessAct, wordQ_one]) hd
    have hl : (e.2 : ℕ) / 32 < 4 := by
      have := e.2.isLt
      simp only [wideWorkload] at this
      omega
    have h0 := hall ⟨0, by simp [wideWorkload]⟩ (by simp)
    rw [wide_unitFlags H s e.1 _ _ hl] at h0
    exact Bool.false_ne_true h0
  rw [ha, he]
  simp [debitOf]

theorem wide_passRows :
    witnessSem.passRows (wideWorkload.layout.shape 0) (witnessAct 0) (Ico 0 1) = Ico 0 1 := by
  classical
  unfold PearlCSem.passRows
  apply filter_true_of_mem
  intro i hi
  have hi' : i < (wideWorkload.layout.shape 0).m := by simpa [wideWorkload] using hi
  exact witnessSem.rowOK_of_operandOK (witness_operandOK 1) hi'

theorem wide_inDomain {Q R S : Type} (P : Protocol Q R S) : wideWorkload.InDomain P (pearlCDomainDev devH100) := by
  refine ⟨by norm_num [wideWorkload], ?_⟩
  intro u _
  refine ⟨by norm_num [wideWorkload], ⟨by norm_num [wideWorkload], by norm_num [wideWorkload]⟩,
    by norm_num [wideWorkload], by norm_num [wideWorkload], Or.inr ?_, by norm_num [wideWorkload]⟩
  show 32 * 4 ∣ 128
  norm_num

theorem wide_weightsOK {ρ : ℚ} : wideWorkload.WeightsOK (pearlCProtocolDevRev1 devH100 witnessSem ρ) :=
  fun _ _ j hj => ⟨fun l hl => (witness_operandOK 1024).1 j hj l hl, fun hj' => (witness_operandOK 1024).2 ⟨j, hj'⟩⟩

theorem wide_gridTiles_length : (gridTiles 64 64 wideWorkload.layout).length = 16 := by decide

theorem wide_gridTiles_zero : (gridTiles 64 64 wideWorkload.layout).getD 0 (0, 0, 0) = (0, 0, 0) := by decide

theorem ttOutPearlCH100Rev1Witness :
    AdmitsRef (witnessCM (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)))
        (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) (pearlCDomainDev devH100) (fun _ => 0) ∧
      TTOutPearlCH100Rev1 (witnessCM (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400))) witnessSem (1 / 400) ∧
      wideWorkload.InDomain (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) (pearlCDomainDev devH100) ∧
      wideWorkload.WeightsOK (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) ∧
      (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)).capOK (fun _ => ()) () wideWorkload 0 (witnessAct 0) ∧
      0 < ∑ u ∈ range wideWorkload.layout.N,
        (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)).credit (fun _ => ()) () wideWorkload u (witnessAct u) := by
  set P := pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)
  have hzero0 : debitDev devH100 (witnessSem.unitDebitDev devH100 (fun _ => ()) () wideWorkload 0 (witnessAct 0)) = 0 := by
    rw [unitDebitDev_h100, debitDev_h100]
    obtain ⟨ha, he⟩ := wide_debit_zero (fun _ => ()) ()
    unfold debitOf
    rw [ha, he]
    simp
  have hzero : witnessSem.unitDebitRev1 devH100 (fun _ => ()) () wideWorkload 0 (witnessAct 0) = 0 := by
    unfold PearlCSem.unitDebitRev1
    rw [hzero0]
    simp (disch := decide) only [laterZeroPromos_eq_zero_of_le, Nat.cast_zero, sum_const_zero, mul_zero,
      add_zero]
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
  · exact wide_inDomain P
  · exact wide_weightsOK
  · show witnessSem.unitDebitRev1 devH100 (fun _ => ()) () wideWorkload 0 (witnessAct 0) ≤
      1 / 400 * creditDevRev1 devH100 (wideWorkload.layout.shape 0)
    rw [hzero]
    have := creditDevRev1_h100_nonneg (wideWorkload.layout.shape 0)
    positivity
  · rw [show range wideWorkload.layout.N = range 1 from rfl, sum_range_one]
    show (0 : ℝ) < ((creditDevRev1 devH100 (wideWorkload.layout.shape 0) -
      witnessSem.unitDebitRev1 devH100 (fun _ => ()) () wideWorkload 0 (witnessAct 0) : ℚ) : ℝ)
    rw [hzero, creditDevRev1_h100]
    norm_num [creditOfRev1, firstAddOf, creditOf, Params.pi, Costs.adopted, wideWorkload]

theorem ttOutTilePearlCH100Rev1Witness :
    AdmitsRefTile (witnessTileCM (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400))
        (pearlCTilesDevRev1 devH100 witnessSem (1 / 400)))
      (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) (pearlCTilesDevRev1 devH100 witnessSem (1 / 400))
      (pearlCDomainDev devH100) (fun _ => 0) ∧
    TTOutTilePearlCH100Rev1 (witnessTileCM (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400))
        (pearlCTilesDevRev1 devH100 witnessSem (1 / 400))) witnessSem (1 / 400) ∧
    wideWorkload.InDomain (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) (pearlCDomainDev devH100) ∧
    wideWorkload.WeightsOK (pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)) ∧
    (pearlCTilesDevRev1 devH100 witnessSem (1 / 400)).capOK (fun _ => ()) () wideWorkload 0 (witnessAct 0) ∧
    0 < ∑ g ∈ range ((pearlCTilesDevRev1 devH100 witnessSem (1 / 400)).tiling wideWorkload.layout).NT,
      (pearlCTilesDevRev1 devH100 witnessSem (1 / 400)).credit (fun _ => ()) () wideWorkload g (witnessAct 0) := by
  set P := pearlCProtocolDevRev1 devH100 witnessSem (1 / 400)
  set TR := pearlCTilesDevRev1 devH100 witnessSem (1 / 400)
  have hNT : (TR.tiling wideWorkload.layout).NT = 16 := wide_gridTiles_length
  have hu : (TR.tiling wideWorkload.layout).unit 0 = 0 := by
    show ((gridTiles 64 64 wideWorkload.layout).getD 0 (0, 0, 0)).1 = 0
    rw [wide_gridTiles_zero]
  have hr : (TR.tiling wideWorkload.layout).rows 0 = Ico 0 1 := by
    show Ico (((gridTiles 64 64 wideWorkload.layout).getD 0 (0, 0, 0)).2.1 * 64)
      (min ((((gridTiles 64 64 wideWorkload.layout).getD 0 (0, 0, 0)).2.1 + 1) * 64)
        (wideWorkload.layout.shape ((gridTiles 64 64 wideWorkload.layout).getD 0 (0, 0, 0)).1).m) = Ico 0 1
    rw [wide_gridTiles_zero] <;> rfl
  have hc : (TR.tiling wideWorkload.layout).cols 0 = Ico 0 64 := by
    show Ico (((gridTiles 64 64 wideWorkload.layout).getD 0 (0, 0, 0)).2.2 * 64)
      (min ((((gridTiles 64 64 wideWorkload.layout).getD 0 (0, 0, 0)).2.2 + 1) * 64)
        (wideWorkload.layout.shape ((gridTiles 64 64 wideWorkload.layout).getD 0 (0, 0, 0)).1).n) = Ico 0 64
    rw [wide_gridTiles_zero] <;> rfl
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
  · exact wide_inDomain P
  · exact wide_weightsOK
  · show witnessSem.tileDebitRev1 devH100 (fun _ => ()) () wideWorkload ((TR.tiling wideWorkload.layout).unit 0)
        (witnessSem.passRows (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit 0)) (witnessAct 0)
          ((TR.tiling wideWorkload.layout).rows 0)) ((TR.tiling wideWorkload.layout).cols 0) (witnessAct 0) ≤
      1 / 400 * creditDevRev1 devH100 (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit 0)) *
        tileShare (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit 0))
          (witnessSem.passRows (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit 0)) (witnessAct 0)
            ((TR.tiling wideWorkload.layout).rows 0)) ((TR.tiling wideWorkload.layout).cols 0)
    rw [hu, hr, hc, wide_passRows]
    unfold PearlCSem.tileDebitRev1
    rw [tileDebitDev_h100, wide_tileDebit_zero]
    simp (disch := decide) only [laterZeroPromos_eq_zero_of_le, Nat.cast_zero, sum_const_zero, mul_zero,
      add_zero]
    have h2 := creditDevRev1_h100_nonneg (wideWorkload.layout.shape 0)
    have h3 := tileShare_nonneg (wideWorkload.layout.shape 0) (Ico 0 1) (Ico 0 64)
    positivity
  · rw [hNT]
    refine sum_pos' (fun g _ => ?_) ⟨0, by simp, ?_⟩
    · show (0 : ℝ) ≤ (((1 - 1 / 400) * creditDevRev1 devH100
          (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit g)) *
          tileShare (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit g))
            (witnessSem.passRows (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit g))
              (witnessAct 0) ((TR.tiling wideWorkload.layout).rows g))
            ((TR.tiling wideWorkload.layout).cols g) : ℚ) : ℝ)
      exact_mod_cast mul_nonneg (mul_nonneg (by norm_num) (creditDevRev1_h100_nonneg _)) (tileShare_nonneg _ _ _)
    show (0 : ℝ) < (((1 - 1 / 400) * creditDevRev1 devH100
        (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit 0)) *
        tileShare (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit 0))
          (witnessSem.passRows (wideWorkload.layout.shape ((TR.tiling wideWorkload.layout).unit 0))
            (witnessAct 0) ((TR.tiling wideWorkload.layout).rows 0))
          ((TR.tiling wideWorkload.layout).cols 0) : ℚ) : ℝ)
    rw [hu, hr, hc, wide_passRows, creditDevRev1_h100]
    norm_num [creditOfRev1, firstAddOf, creditOf, tileShare, Params.pi, Costs.adopted, wideWorkload]

end Pouw.PearlC
