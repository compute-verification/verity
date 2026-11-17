module

public import Proofs.Pouw.PearlC.TileGap
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4
public import Definitions.Pouw.PearlC.NoiseSha512
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
public import Proofs.Pouw.PearlC.Records

/-! g at an FP4 record, and Pearl-C4's headline on sm_120 (staged, nothing pinned). -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

theorem Fp4Sem.passRows_subset {Q R S : Type} (sem : Fp4Sem Q R S) (sh : Shape) (act : Codes) (rows : Finset ℕ) :
    sem.passRows sh act rows ⊆ rows := by
  classical
  unfold Fp4Sem.passRows
  exact filter_subset _ _

theorem pearlCGapFp4At {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) (s : Shape) (ω g : ℚ) (hω0 : 0 < ω)
    (hω : wrefFp4 d s = ω * ((1 - ρ) * creditFp4 d s)) (hg : g = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutFp4 CM d sem ρ) :
    GGap CM (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4At sem s) (g : ℝ) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4At sem s) → ∀ (H : Q → R) (s' : S),
      ∀ u < U.layout.N, ∀ act : Codes, (pearlCProtocolFp4 d sem ρ).actOK (U.layout.shape u) act →
        (pearlCProtocolFp4 d sem ρ).inertBoundOK H s' U u act →
          (pearlCProtocolFp4 d sem ρ).Wref U.layout u ≤ (ω : ℝ) * (pearlCProtocolFp4 d sem ρ).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hsh : U.layout.shape u = s := hU.2.2 u hu
    have hc : sem.unitInertWork d H s' U u act ≤ ρ * creditFp4 d (U.layout.shape u) := hcap
    show (wrefFp4 d (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditFp4 d (U.layout.shape u) - sem.unitInertWork d H s' U u act : ℚ) : ℝ)
    rw [hsh] at hc ⊢
    rw [hω]
    have : ω * ((1 - ρ) * creditFp4 d s) ≤ ω * (creditFp4 d s - sem.unitInertWork d H s' U u act) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast this
  have key := gapFromTTOut CM (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4At sem s) (1 / 400) (ω : ℝ) δPearlC
    (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) hTT)
  have e : (g : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hg]; push_cast; ring
  rw [e]
  exact key

theorem pearlCSampledFp4At {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω) (hw0 : 0 ≤ wrefFp4 d s)
    (hω : wrefFp4 d s = ω * ((1 - ρ) * creditFp4 d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutTileFp4 CM d sem ρ) :
    GGapSampled CM (pearlCProtocolFp4 d sem ρ) (pearlCTilesFp4 d sem ρ) (pearlCDomainFp4At sem s) (γ : ℝ) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4At sem s) → ∀ (H : Q → R) (s' : S),
      ∀ g < ((pearlCTilesFp4 d sem ρ).tiling U.layout).NT, ∀ act : Codes, (pearlCTilesFp4 d sem ρ).inertBoundOK H s' U g act →
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
  have key := gapSampled CM (pearlCProtocolFp4 d sem ρ) (pearlCTilesFp4 d sem ρ) (pearlCDomainFp4At sem s) (1 / 400)
    (ω : ℝ) δPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc (ttOutTile_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

theorem pearlCGapFp4Sm120_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120 CM sem (1 / 400)) :
    GGap CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
      ((9875013901 / 1376663200000 : ℚ) : ℝ) δPearlC :=
  pearlCGapFp4At CM devFp4Sm120 sem _ _ (3441658000 / 3425534301) _ (by norm_num)
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

theorem pearlCGapFp4Sm120_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120 CM sem (1 / 400)) :
    GGap CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh16384)
      ((5473473967 / 895794400000 : ℚ) : ℝ) δPearlC :=
  pearlCGapFp4At CM devFp4Sm120 sem _ _ (2239486000 / 2231380767) _ (by norm_num)
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

theorem pearlCSampledFp4Sm120_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120 CM sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
      (pearlCDomainFp4At sem sh8192) ((9875013901 / 1376663200000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledFp4At CM devFp4Sm120 sem _ _ (3441658000 / 3425534301) _ (by norm_num)
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh8192])
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

theorem pearlCSampledFp4Sm120_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120 CM sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
      (pearlCDomainFp4At sem sh16384) ((5473473967 / 895794400000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledFp4At CM devFp4Sm120 sem _ _ (2239486000 / 2231380767) _ (by norm_num)
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh16384])
    (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

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

theorem fires_fold_eq_cTilde (p : Fp4.Fp4Atom) (a b : Row4)
    (h : b.1.length = a.1.length ∧ a.1.length % p.k = 0 ∧ a.2.length = a.1.length / p.k * p.scales ∧
      b.2.length = a.1.length / p.k * p.scales) :
    (fires p a b).foldl (fireStep p) (some 0) = Fp4.cTilde p a.1 b.1 a.2 b.2 := by
  unfold Fp4.cTilde Fp4.Fp4Atom.chain fires
  rw [ite_eq_left_of_eq_true _ _ (eq_true h), chainN_eq_fold]

theorem e2m1Mag_ties : (e2m1Mids.map e2m1Mag) = [0, 2, 2, 4, 4, 6, 6] := by decide +kernel

theorem e2m1Mag_grid : ([0, 1 / 2, 1, 3 / 2, 2, 3, 4, 6, 7, 100] : List ℚ).map e2m1Mag = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7] := by
  decide +kernel

theorem e2m1Word_examples :
    [e2m1Word 0x3F800000, e2m1Word 0xC0C00000, e2m1Word 0x40E00000, e2m1Word 0x80000000] = [2, 15, 7, 8] := by
  decide +kernel

theorem nvf4Block_signedZero :
    nvf4Block ([0x80000000, 0x3F800000] ++ List.replicate 13 0 ++ [0x80000000]) =
      ([8, 7] ++ List.replicate 13 0 ++ [8], 0x23) := by
  decide +kernel

theorem subgridAt_examples :
    subgridAt Fp4.sm120Nvf4 0x53800000 (List.replicate 64 2, List.replicate 64 2, [1, 0x38, 0x38, 0x38],
      [1, 0x38, 0x38, 0x38]) 0 = true ∧
    subgridAt Fp4.sm120Nvf4 0x53800000 (List.replicate 64 2, List.replicate 64 2, [1, 0x38, 0x38, 0x38],
      [1, 0x38, 0x38, 0x38]) 1 = false := by
  decide +kernel

theorem rowWin24_groups :
    let r := (some ((List.replicate 32 ([1, 9, 0, 0] : List Fp4.Nib)).flatten ++ List.replicate 128 1, []) : Option Row4)
    (rowWin24 r 0 = true ∧ rowWin24 r 1 = false) ∧
    rowWin24 (some ((List.replicate 16 ([2, 2, 2, 2, 0, 0, 0, 0] : List Fp4.Nib)).flatten, [])) 0 = true ∧
    rowWin24 (some ((List.replicate 16 ([2, 0, 2, 0, 2, 0, 0, 0] : List Fp4.Nib)).flatten, [])) 0 = false := by
  dsimp only
  decide +kernel

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

theorem pearlCShapeFp4_headline : pearlCShapeFp4 sh8192 ∧ pearlCShapeFp4 sh16384 := by
  unfold pearlCShapeFp4 sh8192 sh16384
  decide +kernel

theorem sha512Fp4Sem_saltKeyedB {Q R S : Type} (base : Fp4Sem Q R S) {salt : S → List ℕ}
    (hs : Function.Injective salt) (rootA : List ℕ → ℕ → Codes → List ℕ) (rootB : ℕ → List ℕ) :
    (sha512Fp4Sem base salt rootA rootB).SaltKeyedB :=
  ⟨salt, rootB, hs, fun _ _ _ _ _ => rfl⟩

theorem salt32_bytes_injective : Function.Injective NoiseSha512.Salt32.bytes := by
  intro a b h
  have h' : (fun i => (a i).val) = fun i => (b i).val := List.ofFn_inj.mp h
  funext i
  exact Fin.ext (congrFun h' i)

theorem sha512Fp4Sem_inDomain {Q R : Type} (d : Fp4Device) (base : Fp4Sem Q R NoiseSha512.Salt32) (ρ : ℚ)
    (rootA : List ℕ → ℕ → Codes → List ℕ) (rootB : ℕ → List ℕ) (s : Shape) (hs : pearlCShapeFp4 s) (N : ℕ) (hN : N ≤ 2 ^ 64)
    (fixedId : ℕ → ℕ) (fixed : ℕ → Codes) :
    (⟨⟨N, fun _ => s, fixedId⟩, fixed⟩ : Workload).InDomain
      (pearlCProtocolFp4 d (sha512Fp4Sem base NoiseSha512.Salt32.bytes rootA rootB) ρ)
      (pearlCDomainFp4At (sha512Fp4Sem base NoiseSha512.Salt32.bytes rootA rootB) s) :=
  ⟨hN, ⟨sha512Fp4Sem_saltKeyedB base salt32_bytes_injective rootA rootB, fun _ _ => hs⟩, fun _ _ => rfl⟩

theorem sha512Fp4Sem_inDomain_headline {Q R : Type} (base : Fp4Sem Q R NoiseSha512.Salt32)
    (rootA : List ℕ → ℕ → Codes → List ℕ) (rootB : ℕ → List ℕ) (fixed : ℕ → Codes) :
    (⟨⟨1, fun _ => sh8192, fun _ => 0⟩, fixed⟩ : Workload).InDomain
        (pearlCProtocolFp4 devFp4Sm120 (sha512Fp4Sem base NoiseSha512.Salt32.bytes rootA rootB) (1 / 400))
        (pearlCDomainFp4At (sha512Fp4Sem base NoiseSha512.Salt32.bytes rootA rootB) sh8192) ∧
      (⟨⟨1, fun _ => sh16384, fun _ => 0⟩, fixed⟩ : Workload).InDomain
        (pearlCProtocolFp4 devFp4Sm120 (sha512Fp4Sem base NoiseSha512.Salt32.bytes rootA rootB) (1 / 400))
        (pearlCDomainFp4At (sha512Fp4Sem base NoiseSha512.Salt32.bytes rootA rootB) sh16384) :=
  ⟨sha512Fp4Sem_inDomain _ base _ rootA rootB _ pearlCShapeFp4_headline.1 1 (by norm_num) _ fixed,
    sha512Fp4Sem_inDomain _ base _ rootA rootB _ pearlCShapeFp4_headline.2 1 (by norm_num) _ fixed⟩

theorem dnf_rejects :
    ((e4m3 (-1 / 4096)).val = 0x80 ∧ (Fp4.scaleDyadic .ue4m3 (e4m3 (-1 / 4096))).map Prod.fst = some 0) ∧
    ((e4m3 (-1)).val = 0xB8 ∧ Fp4.scaleDyadic .ue4m3 (e4m3 (-1)) = Fp4.scaleDyadic .ue4m3 (e4m3 1)) ∧
    ∀ (k : ℕ) (x : Fin k → ℚ), ¬ RowAdmit4 (fun _ _ => -1 / 4096) k x ∧ ¬ RowAdmit4 (fun _ _ => -1) k x ∧
      ¬ RowAdmit4 (fun _ _ => -0) k x ∧ ¬ RowAdmit4 (fun _ _ => 2 ^ (-7 : ℤ)) k x ∧ ¬ RowAdmit4 (fun _ _ => 0) k x := by
  refine ⟨by decide +kernel, by decide +kernel, fun k x => ?_⟩
  simp only [RowAdmit4]
  decide +kernel

theorem dnf_admits : ∀ (k : ℕ) (x : Fin k → ℚ), RowAdmit4 (fun _ _ => 2 ^ (-6 : ℤ)) k x ∧
    RowAdmit4 (fun _ _ => 448) k x ∧ RowAdmit4 (fun _ _ => 10 ^ 30) k x := by
  intro k x
  simp only [RowAdmit4]
  decide +kernel

theorem rowRules4_admit {beta : (k : ℕ) → (Fin k → ℚ) → ℚ} {k : ℕ} {x : Fin k → ℚ} (h : RowRules4 beta k x) :
    RowAdmit4 beta k x :=
  ⟨h.1, h.2.1, h.2.2.1⟩

theorem cL_8192 : cL 8192 8192 8192 = 6612 / 10000 := by decide +kernel

theorem cL_16384 : cL 16384 16384 16384 = 5790 / 10000 := by decide +kernel

theorem cL_32768 : cL 32768 32768 32768 = 5069 / 10000 := by decide +kernel

theorem cL_65536 : cL 65536 65536 65536 = 4083 / 10000 := by decide +kernel

theorem cL_f2_threshold : 2 - (1 - cL 8192 8192 8192) / 4 = 19153 / 10000 := by decide +kernel

theorem cL_corner : cL (2 ^ 24) (2 ^ 16) (2 ^ 18) = 2697 / 10000 := by decide +kernel

theorem cL_rectangular : cL (2 ^ 16) 8192 1024 = 8081 / 10000 ∧ cL 1024 8192 (2 ^ 16) = 8081 / 10000 ∧
    cL (2 ^ 24) 8192 128 = 11302 / 10000 ∧ cL (2 ^ 24) 16 (2 ^ 18) = 2 := by
  decide +kernel

theorem cL_between : cL 3000 5000 700 = cL 8192 8192 1024 ∧ cL 20000 9000 100 = cL 32768 16384 128 ∧
    cL 64 128 16 = cL 128 128 128 := by
  decide +kernel

theorem cL_beyond : cL (2 ^ 25) 8192 8192 = 0 ∧ cL 8192 8192 (2 ^ 19) = 0 ∧ cL 8192 (2 ^ 17) 8192 = 0 := by
  decide +kernel

theorem cLTable_dims : cLMs.Pairwise (· < ·) ∧ cLKs.Pairwise (· < ·) ∧ cLNs.Pairwise (· < ·) ∧
    cLMs.length = 9 ∧ cLKs.length = 13 ∧ cLNs.length = 7 ∧ cLTable.length = 9 ∧
    ∀ s ∈ cLTable, s.length = 13 ∧ ∀ r ∈ s, r.length = 7 := by
  decide +kernel

theorem cLTable_falls : ∀ i < 9, ∀ j < 13, ∀ l < 7,
    cLEntry (min (i + 1) 8) j l ≤ cLEntry i j l ∧ cLEntry i (min (j + 1) 12) l ≤ cLEntry i j l ∧
      cLEntry i j (min (l + 1) 6) ≤ cLEntry i j l := by
  decide +kernel

end Pouw.PearlC
