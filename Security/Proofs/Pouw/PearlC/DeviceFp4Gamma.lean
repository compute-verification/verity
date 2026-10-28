import Proofs.Pouw.PearlC.TileGamma
import Definitions.Pouw.Assumptions.PearlC.TTOutFp4

/-!
# γ at an FP4 record, and Pearl-C4's headline on sm_120 (staged, nothing pinned)

* **γ at any FP4 record,** on the one-shape domain at `sem` (D-SK included, `Fp4Sem.SaltKeyedB`) where
  `wrefFp4 = ω·(1 − ρ)·creditFp4`: `G_γ` from TT_OUT
  (`pearlCGammaFp4At`) and `GγSampled` from TT_OUT per tile (`pearlCSampledFp4At`), both at `γ = 1 − (399/400)/ω`. These
  are `pearlCGammaDevAt` and `pearlCSampledDevAt` on `Fp4Device`.
* **Pearl-C4's headline on sm_120**, cap `1/400`, forming credited, at `Fp4Prices.sm120` (`fs = 105.299`, `qa =
  18.846`: block scale on `lut256`, FADD at `1047/125`): `9875013901/1376663200000` (0.71732%) at 8192³ and
  `5473473967/895794400000` (0.61102%) at 16384³, per unit (`pearlCGammaFp4Sm120_8192`, `_16384`) and per audit tile
  (`pearlCSampledFp4Sm120_8192`, `_16384`). They are `fp4-delta/README.md`'s `lut256` values at 8.376. Per `m·k`, the
  credit is `n + 32 + 256·n/k + fs` and `W_ref` adds `qa`. At `fs = 97.76` they were 0.71761% and 0.61112%.
* **The sub-grid flags read the ticket's chain:** on rows `cTilde` accepts, the firings they replay fold to `cTilde`'s
  chain (`fires_fold_eq_cTilde`).
* **Sanity checks**, by the kernel: the forming's E2M1 cast (`e2m1Mag_ties`, `e2m1Mag_grid`, `e2m1Word_examples`), the
  sub-grid test (`subgridAt_examples`), D-24's pair rule on the reference's vector (`rowWin24_groups`), F2's three
  readings (`int8Saving_readings`), D-NF on β's sign and byte (`dnf_rejects`, `dnf_admits`, `rowRules4_admit`), the
  headline shapes inside the domain (`pearlCShapeFp4_headline`), and F2's pinned `c_L` table: 0.6612, 0.5790,
  0.5069 and 0.4083 at the four cubes (`cL_8192` to `cL_65536`), F2's threshold at 8,192³ (`cL_f2_threshold`), the
  corner, rectangular shapes, shapes between and past the table's points (`cL_corner`, `cL_rectangular`, `cL_between`,
  `cL_beyond`), and the table's shape and fall in each dimension (`cLTable_dims`, `cLTable_falls`).
* **Neither the base-split fix nor its restage moves γ.** F1′, F2 and R1 are debit terms
  (`Fp4Sem.tileDebit`), and the γ statements hold at any debit within the cap. They take TT_OUT on the domain
  `pearlCDomainFp4`, now bounded to the grant's shapes, and read only its one-shape clause, so the instances are
  unchanged.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

theorem Fp4Sem.passRows_subset {Q R S : Type} (sem : Fp4Sem Q R S) (sh : Shape) (act : Codes) (rows : Finset ℕ) :
    sem.passRows sh act rows ⊆ rows := by
  classical
  unfold Fp4Sem.passRows
  exact filter_subset _ _

/-- TT_OUT at an FP4 record gives `G_γ` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefFp4 = ω·(1 − ρ)·creditFp4`: a unit within the cap is credited at least `(1 − ρ)·creditFp4`. -/
theorem pearlCGammaFp4At {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hω : wrefFp4 d s = ω * ((1 - ρ) * creditFp4 d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutFp4 CM d sem ρ) :
    Gγ CM (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4At sem s) (γ : ℝ) εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4At sem s) → ∀ (H : Q → R) (s' : S),
      ∀ u < U.layout.N, ∀ act : Codes, (pearlCProtocolFp4 d sem ρ).actOK (U.layout.shape u) act →
        (pearlCProtocolFp4 d sem ρ).capOK H s' U u act →
          (pearlCProtocolFp4 d sem ρ).Wref U.layout u ≤ (ω : ℝ) * (pearlCProtocolFp4 d sem ρ).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hsh : U.layout.shape u = s := hU.2.2 u hu
    have hc : sem.unitDebit d H s' U u act ≤ ρ * creditFp4 d (U.layout.shape u) := hcap
    show (wrefFp4 d (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditFp4 d (U.layout.shape u) - sem.unitDebit d H s' U u act : ℚ) : ℝ)
    rw [hsh] at hc ⊢
    rw [hω]
    have : ω * ((1 - ρ) * creditFp4 d s) ≤ ω * (creditFp4 d s - sem.unitDebit d H s' U u act) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast this
  have key := gammaFromTTOut CM (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4At sem s) (1 / 400) (ω : ℝ) εPearlC
    (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- TT_OUT per tile at an FP4 record gives `GγSampled` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefFp4 = ω·(1 − ρ)·creditFp4 ≥ 0`. -/
theorem pearlCSampledFp4At {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω) (hw0 : 0 ≤ wrefFp4 d s)
    (hω : wrefFp4 d s = ω * ((1 - ρ) * creditFp4 d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutTileFp4 CM d sem ρ) :
    GγSampled CM (pearlCProtocolFp4 d sem ρ) (pearlCTilesFp4 d sem ρ) (pearlCDomainFp4At sem s) (γ : ℝ) εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4At sem s) → ∀ (H : Q → R) (s' : S),
      ∀ g < ((pearlCTilesFp4 d sem ρ).tiling U.layout).NT, ∀ act : Codes, (pearlCTilesFp4 d sem ρ).capOK H s' U g act →
        (pearlCTilesFp4 d sem ρ).Wcred U.layout g act ≤ (ω : ℝ) * (pearlCTilesFp4 d sem ρ).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesFp4]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4At sem s) →
      ∀ g < ((pearlCTilesFp4 d sem ρ).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesFp4 d sem ρ).Wcred U.layout g act ≤ (pearlCTilesFp4 d sem ρ).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesFp4]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (Fp4Sem.passRows_subset sem _ _ _) _) hw0
  have key := gammaSampled CM (pearlCProtocolFp4 d sem ρ) (pearlCTilesFp4 d sem ρ) (pearlCDomainFp4At sem s) (1 / 400)
    (ω : ℝ) εPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc (ttOutTile_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- **Pearl-C4's headline on sm_120, 8192³**, worst case under the cap `1/400`: `γ = 9875013901/1376663200000` (0.71732%). -/
theorem pearlCGammaFp4Sm120_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120 CM sem (1 / 400)) :
    Gγ CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
      ((9875013901 / 1376663200000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaFp4At CM devFp4Sm120 sem _ _ (3441658000 / 3425534301) _ (by norm_num)
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

/-- **Pearl-C4's headline on sm_120, 16384³**, worst case under the cap `1/400`: `γ = 5473473967/895794400000` (0.61102%). -/
theorem pearlCGammaFp4Sm120_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120 CM sem (1 / 400)) :
    Gγ CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh16384)
      ((5473473967 / 895794400000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaFp4At CM devFp4Sm120 sem _ _ (2239486000 / 2231380767) _ (by norm_num)
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

/-- **Pearl-C4's headline on sm_120 per audit tile, 8192³**: `GγSampled` at `γ = 9875013901/1376663200000`. -/
theorem pearlCSampledFp4Sm120_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120 CM sem (1 / 400)) :
    GγSampled CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
      (pearlCDomainFp4At sem sh8192) ((9875013901 / 1376663200000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledFp4At CM devFp4Sm120 sem _ _ (3441658000 / 3425534301) _ (by norm_num)
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh8192])
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

/-- **Pearl-C4's headline on sm_120 per audit tile, 16384³**: `GγSampled` at `γ = 5473473967/895794400000`. -/
theorem pearlCSampledFp4Sm120_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120 CM sem (1 / 400)) :
    GγSampled CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
      (pearlCDomainFp4At sem sh16384) ((5473473967 / 895794400000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledFp4At CM devFp4Sm120 sem _ _ (2239486000 / 2231380767) _ (by norm_num)
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh16384])
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

/-! ## The flags read the ticket's chain -/

theorem foldl_fireStep_none (p : Fp4.Fp4Atom) (xs : List Fire4) : xs.foldl (fireStep p) none = none := by
  induction xs with
  | nil => rfl
  | cons x xs ih => exact ih

theorem chainN_eq_fold (p : Fp4.Fp4Atom) : ∀ (n c : ℕ) (a b : List Fp4.Nib) (sa sb : List (Fin 256)),
    p.chainN n c a b sa sb = ((List.range n).map fun t =>
      ((a.drop (t * p.k)).take p.k, (b.drop (t * p.k)).take p.k,
        (sa.drop (t * p.scales)).take p.scales, (sb.drop (t * p.scales)).take p.scales)).foldl (fireStep p) (some c)
  | 0, c, a, b, sa, sb => rfl
  | n + 1, c, a, b, sa, sb => by
    rw [List.range_succ_eq_map, List.map_cons, List.foldl_cons, List.map_map]
    have e : ((fun t => ((a.drop (t * p.k)).take p.k, (b.drop (t * p.k)).take p.k,
        (sa.drop (t * p.scales)).take p.scales, (sb.drop (t * p.scales)).take p.scales)) ∘ Nat.succ) =
        fun t => (((a.drop p.k).drop (t * p.k)).take p.k, ((b.drop p.k).drop (t * p.k)).take p.k,
          ((sa.drop p.scales).drop (t * p.scales)).take p.scales,
          ((sb.drop p.scales).drop (t * p.scales)).take p.scales) := by
      funext t
      simp [List.drop_drop, Nat.add_mul, Nat.add_comm]
    rw [e]
    simp only [Fp4.Fp4Atom.chainN, zero_mul, List.drop_zero]
    show (p.step c (a.take p.k) (b.take p.k) (sa.take p.scales) (sb.take p.scales)).bind _ =
      List.foldl (fireStep p) (p.step c (a.take p.k) (b.take p.k) (sa.take p.scales) (sb.take p.scales)) _
    cases p.step c (a.take p.k) (b.take p.k) (sa.take p.scales) (sb.take p.scales) with
    | none => exact (foldl_fireStep_none p _).symm
    | some d => exact chainN_eq_fold p n d _ _ _ _

/-- On rows `cTilde` accepts, the fold of `fireStep` over `fires` from `+0` is `cTilde`'s chain, so the sub-grid flags
replay the ticket's firings. -/
theorem fires_fold_eq_cTilde (p : Fp4.Fp4Atom) (a b : Row4)
    (h : b.1.length = a.1.length ∧ a.1.length % p.k = 0 ∧ a.2.length = a.1.length / p.k * p.scales ∧
      b.2.length = a.1.length / p.k * p.scales) :
    (fires p a b).foldl (fireStep p) (some 0) = Fp4.cTilde p a.1 b.1 a.2 b.2 := by
  unfold Fp4.cTilde Fp4.Fp4Atom.chain fires
  rw [ite_eq_left_of_eq_true _ _ (eq_true h), chainN_eq_fold]

/-! ## Sanity checks on the E2M1 cast -/

/-- Ties go to the even code: `0.25 ↦ 0`, `0.75 ↦ 1.0`, `1.25 ↦ 1.0`, `1.75 ↦ 2.0`, `2.5 ↦ 2.0`, `3.5 ↦ 4.0`,
`5 ↦ 4.0`. -/
theorem e2m1Mag_ties : (e2m1Mids.map e2m1Mag) = [0, 2, 2, 4, 4, 6, 6] := by decide +kernel

/-- The grid's magnitudes are fixed, and the cast saturates at 6. -/
theorem e2m1Mag_grid : ([0, 1 / 2, 1, 3 / 2, 2, 3, 4, 6, 7, 100] : List ℚ).map e2m1Mag = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7] := by
  decide +kernel

/-- The cast on FP32 words keeps the sign bit: `1.0 ↦ 0x2`, `−6.0 ↦ 0xF`, `7.0 ↦ 0x7` (saturated), `−0 ↦ 0x8`. -/
theorem e2m1Word_examples :
    [e2m1Word 0x3F800000, e2m1Word 0xC0C00000, e2m1Word 0x40E00000, e2m1Word 0x80000000] = [2, 15, 7, 8] := by
  decide +kernel

/-- A block of `−0`, `1.0`, thirteen `+0` and `−0` (`test_nvfp4_blocks_keep_the_sign_of_minus_zero`): scale `0x23`, and
the `−0` words keep their sign (`0x8`) where the `+0` words give `0x0`. -/
theorem nvf4Block_signedZero :
    nvf4Block ([0x80000000, 0x3F800000] ++ List.replicate 13 0 ++ [0x80000000]) =
      ([8, 7] ++ List.replicate 13 0 ++ [8], 0x23) := by
  decide +kernel

/-- Under an accumulator of `2⁴⁰` the firing's grid is `2⁵`. A block at the scale floor `2⁻⁹` on both operands is then
sub-grid (its largest dot is `2304·2⁻²⁰`), and a block at scale `1.0` is not (`576 ≥ 2⁵`). -/
theorem subgridAt_examples :
    subgridAt Fp4.sm120Nvf4 0x53800000 (List.replicate 64 2, List.replicate 64 2, [1, 0x38, 0x38, 0x38],
      [1, 0x38, 0x38, 0x38]) 0 = true ∧
    subgridAt Fp4.sm120Nvf4 0x53800000 (List.replicate 64 2, List.replicate 64 2, [1, 0x38, 0x38, 0x38],
      [1, 0x38, 0x38, 0x38]) 1 = false := by
  decide +kernel

/-- D-24's window test is the hardware's pair rule. The reference's vector (`two_four_windows([0x1, 0x9, 0, 0] * 32 +
[0x1] * 128)`) reads `[True, False]` under both rules: `0x9` is `−0.5`, nonzero. The rules differ on the other two:
`[2, 2, 2, 2, 0, 0, 0, 0]` repeated passes (each 8-chunk has two nonzero pairs), though the reference's 4-group rule
fails it; `[2, 0, 2, 0, 2, 0, 0, 0]` repeated fails (three nonzero pairs), though the 4-group rule passes it. -/
theorem rowWin24_groups :
    let r := (some ((List.replicate 32 ([1, 9, 0, 0] : List Fp4.Nib)).flatten ++ List.replicate 128 1, []) : Option Row4)
    (rowWin24 r 0 = true ∧ rowWin24 r 1 = false) ∧
    rowWin24 (some ((List.replicate 16 ([2, 2, 2, 2, 0, 0, 0, 0] : List Fp4.Nib)).flatten, [])) 0 = true ∧
    rowWin24 (some ((List.replicate 16 ([2, 0, 2, 0, 2, 0, 0, 0] : List Fp4.Nib)).flatten, [])) 0 = false := by
  dsimp only
  decide +kernel

/-- F2's three readings (`test_f2_reads_flatness_per_row_and_per_block_column_class`, at 16 blocks and 4 rows a
side, in a unit of `2^24 × 2^18`). At one byte the route saves its whole `1 − c_L(256) = 0.2627`. With row factors
(alternate rows 8 bytes up) only the per-row reading sees it, and saves the same. Two classes a byte apart, each a
sub-GEMM over half of `k`, save `1 − c_L(128) = 0.0734` by the class split, where the tile's and the rows' modal shares
(1/2) read nothing. Spread bytes, shares `7/32`, read nothing. -/
theorem int8Saving_readings :
    let flat : List (List ℤ) := List.replicate 4 (List.replicate 16 0x50)
    let shifted : List (List ℤ) := (List.range 4).map fun r => List.replicate 16 (0x50 + 8 * ((r % 2 : ℕ) : ℤ))
    let two : List (List ℤ) := List.replicate 4 ((List.range 16).map fun c => 0x50 + ((c % 2 : ℕ) : ℤ))
    let spread : List (List ℤ) :=
      (List.range 4).map fun r => (List.range 16).map fun c => 0x50 + (((c * (r + 1) + c / 3) % 6 : ℕ) : ℤ)
    int8Saving (2 ^ 24) (2 ^ 18) 16 flat flat = 2627 / 10000 ∧
      int8Saving (2 ^ 24) (2 ^ 18) 16 shifted flat = 2627 / 10000 ∧
      int8Saving (2 ^ 24) (2 ^ 18) 16 two two = 734 / 10000 ∧ int8Saving (2 ^ 24) (2 ^ 18) 16 spread spread = 0 := by
  dsimp only
  decide +kernel

/-- **The headline shapes are in the domain** (`pearlCShapeFp4`), so the γ instances at 8192³ and 16384³ are not vacuous
on shape grounds. -/
theorem pearlCShapeFp4_headline : pearlCShapeFp4 sh8192 ∧ pearlCShapeFp4 sh16384 := by
  unfold pearlCShapeFp4 sh8192 sh16384
  decide +kernel

/-- **D-NF rejects a negative β** (`pearl_c4`'s `test_dnf_rejects_a_negative_beta_whose_byte_would_be_zero_noise`).
`β = −1/4096` casts to `0x80`, whose significand under sm_120's low-7-bit decode is `0`, a zero scale; `β = −1` casts
to `0xB8`, which decodes as `+1.0`. Both bytes pass `0x08 ≤ a_E` alone. In ℚ the FP32 `−0.0` is `0`, whose byte `0x00`
fails `0x08 ≤ a_E`, as does `2⁻⁷`'s `0x04`. -/
theorem dnf_rejects :
    ((e4m3 (-1 / 4096)).val = 0x80 ∧ (Fp4.scaleDyadic .ue4m3 (e4m3 (-1 / 4096))).map Prod.fst = some 0) ∧
    ((e4m3 (-1)).val = 0xB8 ∧ Fp4.scaleDyadic .ue4m3 (e4m3 (-1)) = Fp4.scaleDyadic .ue4m3 (e4m3 1)) ∧
    ∀ (k : ℕ) (x : Fin k → ℚ), ¬ RowAdmit4 (fun _ _ => -1 / 4096) k x ∧ ¬ RowAdmit4 (fun _ _ => -1) k x ∧
      ¬ RowAdmit4 (fun _ _ => -0) k x ∧ ¬ RowAdmit4 (fun _ _ => 2 ^ (-7 : ℤ)) k x ∧ ¬ RowAdmit4 (fun _ _ => 0) k x := by
  refine ⟨by decide +kernel, by decide +kernel, fun k x => ?_⟩
  simp only [RowAdmit4]
  decide +kernel

/-- **D-NF admits `2⁻⁶` through `448` and beyond** (bytes `0x08`, `0x7E`, and `0x7E` again by saturation at `10³⁰`). -/
theorem dnf_admits : ∀ (k : ℕ) (x : Fin k → ℚ), RowAdmit4 (fun _ _ => 2 ^ (-6 : ℤ)) k x ∧
    RowAdmit4 (fun _ _ => 448) k x ∧ RowAdmit4 (fun _ _ => 10 ^ 30) k x := by
  intro k x
  simp only [RowAdmit4]
  decide +kernel

/-- `RhoDFp4`'s row premise implies admission. -/
theorem rowRules4_admit {beta : (k : ℕ) → (Fin k → ℚ) → ℚ} {k : ℕ} {x : Fin k → ℚ} (h : RowRules4 beta k x) :
    RowAdmit4 beta k x :=
  ⟨h.1, h.2.1, h.2.2.1⟩

/-- F2's `c_L` is 0.6612 at 8,192³ (Strassen–Winograd's formula gave 0.808). -/
theorem cL_8192 : cL 8192 8192 8192 = 6612 / 10000 := by decide +kernel

/-- F2's `c_L` is 0.5790 at 16,384³ (Strassen–Winograd: 0.707). -/
theorem cL_16384 : cL 16384 16384 16384 = 5790 / 10000 := by decide +kernel

/-- F2's `c_L` is 0.5069 at 32,768³ (Strassen–Winograd: 0.619). -/
theorem cL_32768 : cL 32768 32768 32768 = 5069 / 10000 := by decide +kernel

/-- F2's `c_L` is 0.4083 at 65,536³ (Strassen–Winograd: 0.542). -/
theorem cL_65536 : cL 65536 65536 65536 = 4083 / 10000 := by decide +kernel

/-- At 8,192³ F2 is positive only past `f_A + f_B = 2 − (1 − c_L)/4 = 1.9153`. -/
theorem cL_f2_threshold : 2 - (1 - cL 8192 8192 8192) / 4 = 19153 / 10000 := by decide +kernel

/-- The table's corner, `m = 2^24`, `k = 2^16`, `n = 2^18`, reads its least entry, 0.2697 (Strassen–Winograd: 0.421). -/
theorem cL_corner : cL (2 ^ 24) (2 ^ 16) (2 ^ 18) = 2697 / 10000 := by decide +kernel

/-- Rectangular shapes read their own entries: 0.8081 at `(2^16, 8192, 1024)` and at its transpose, 1.1302 on a
128-wide weight (no route pays), and 2 at one block of `k` (no level). -/
theorem cL_rectangular : cL (2 ^ 16) 8192 1024 = 8081 / 10000 ∧ cL 1024 8192 (2 ^ 16) = 8081 / 10000 ∧
    cL (2 ^ 24) 8192 128 = 11302 / 10000 ∧ cL (2 ^ 24) 16 (2 ^ 18) = 2 := by
  decide +kernel

/-- A shape between tabled points reads the next tabled point up in each dimension, whose `c_L` is the lower. -/
theorem cL_between : cL 3000 5000 700 = cL 8192 8192 1024 ∧ cL 20000 9000 100 = cL 32768 16384 128 ∧
    cL 64 128 16 = cL 128 128 128 := by
  decide +kernel

/-- Past the table `c_L` is `0`, where `pearl_c4.int8_depth_cost` raises. -/
theorem cL_beyond : cL (2 ^ 25) 8192 8192 = 0 ∧ cL 8192 8192 (2 ^ 19) = 0 ∧ cL 8192 (2 ^ 17) 8192 = 0 := by
  decide +kernel

/-- The grid is strictly increasing in each dimension, and the table is 9 × 13 × 7 over it. -/
theorem cLTable_dims : cLMs.Pairwise (· < ·) ∧ cLKs.Pairwise (· < ·) ∧ cLNs.Pairwise (· < ·) ∧
    cLMs.length = 9 ∧ cLKs.length = 13 ∧ cLNs.length = 7 ∧ cLTable.length = 9 ∧
    ∀ s ∈ cLTable, s.length = 13 ∧ ∀ r ∈ s, r.length = 7 := by
  decide +kernel

/-- The table falls in each of `m`, `k` and `n`. -/
theorem cLTable_falls : ∀ i < 9, ∀ j < 13, ∀ l < 7,
    cLEntry (min (i + 1) 8) j l ≤ cLEntry i j l ∧ cLEntry i (min (j + 1) 12) l ≤ cLEntry i j l ∧
      cLEntry i j (min (l + 1) 6) ≤ cLEntry i j l := by
  decide +kernel

end Pouw.PearlC
