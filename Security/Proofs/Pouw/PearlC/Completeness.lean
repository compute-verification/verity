import Definitions.Pouw.PearlC.Completeness
import Definitions.Pouw.PearlC.DevicePricesLoop
import Definitions.Pouw.Assumptions.PearlC.HonestCap
import Proofs.Pouw.PearlC.HonestWords
import Proofs.Pouw.Copy

/-!
# The sampled audit's completeness from the honest tiles' measured cap rate

* **The honest words are proved.** `honestWords_devRev1K` gives every reference value an FP32 word, so the honest
  prover's committed words are the reference's (`wordOf_spec`). The workload's weights pass the rules (completeness's
  hypothesis `Workload.WeightsOK`), so a drawn tile of a committed unit is good exactly when its replayed debit is
  within the cap.
* **The cap is assumed at its measured rate.** `HonestTileCapRate` bounds one draw's failure by `ε`, an upper confidence
  limit on the measured rate; the union bound over the `t` draws gives `t·ε`.
* **The served models** at v1's record under rev1 at the cap 1/1,000, the honest cast at 8.72: Llama-3.1-8B's and
  Qwen3-8B's served layouts (`pearlCDomainDevShapes`), whose units are at most `2^16` deep.
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom Pouw.PearlC.Assumptions Pouw.PearlC.HonestWords

variable {Q R S : Type}

/-- `wordOf v` is a word of `v` when `v` has one. -/
theorem wordOf_spec {v : Option ℚ} (h : ∃ w, wordQ w = v) : wordQ (wordOf v) = v := by
  unfold wordOf
  split_ifs
  exact Classical.choose_spec h

/-- An event on one of `t` independent draws has at most that draw's probability. -/
theorem pr_coord_le {α β T : Type} [Fintype α] [Fintype β] [Fintype T] {t : ℕ} (i : Fin t) (P : α × β × T → Prop) :
    pr (fun x : α × β × (Fin t → T) => P (x.1, x.2.1, x.2.2 i)) ≤ pr P := by
  classical
  let e : (α × β × T) × ({j : Fin t // j ≠ i} → T) ≃ α × β × (Fin t → T) :=
    (Equiv.prodAssoc α (β × T) _).trans <| (Equiv.refl α).prodCongr <|
      (Equiv.prodAssoc β T _).trans <| (Equiv.refl β).prodCongr (Equiv.piSplitAt i fun _ => T).symm
  have he : ∀ y, P ((e y).1, (e y).2.1, (e y).2.2 i) = P y.1 := by
    rintro ⟨⟨a, b, c⟩, f⟩
    simp [e, Equiv.piSplitAt_symm_apply]
  rw [← pr_equiv e]
  simp only [he]
  exact pr_fst_le P

/-- A tile of a committed unit of the honest transcript is good when its weight rows pass the rules and its debit is
within the cap, given honest words. -/
theorem mem_goodTiles_honest (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S)
    (act : ℕ → Codes) {g : ℕ} (hg : g < (TR.tiling U.layout).NT) (hu : (TR.tiling U.layout).unit g < U.layout.N)
    (hW : ∀ i j : ℕ, (∃ wd, wordQ wd = P.ticket H s U ((TR.tiling U.layout).unit g) (act ((TR.tiling U.layout).unit g)) i j) ∧
      ∃ wd, wordQ wd = P.useful H s U ((TR.tiling U.layout).unit g) (act ((TR.tiling U.layout).unit g)) i j)
    (hB : ∀ j ∈ (TR.tiling U.layout).cols g,
      P.weightOK (U.layout.shape ((TR.tiling U.layout).unit g)) (U.weight (U.layout.wid ((TR.tiling U.layout).unit g)))
        j)
    (hcap : TR.capOK H s U g (act ((TR.tiling U.layout).unit g))) :
    g ∈ goodTiles P TR U H s (honestTranscript P H s U act) := by
  classical
  unfold goodTiles
  rw [mem_filter]
  refine ⟨mem_range.2 hg, ⟨act ((TR.tiling U.layout).unit g),
    fun i j => wordOf (P.ticket H s U ((TR.tiling U.layout).unit g) (act ((TR.tiling U.layout).unit g)) i j),
    fun i j => wordOf (P.useful H s U ((TR.tiling U.layout).unit g) (act ((TR.tiling U.layout).unit g)) i j)⟩, ?_, hcap,
    hB, ?_⟩
  · unfold honestTranscript
    exact ite_eq_left_iff.2 fun h => absurd hu h
  · intro a _ _ b _
    exact ⟨wordOf_spec (hW a b).1, wordOf_spec (hW a b).2⟩

/-- A tile of a unit past the layout is blank in the honest transcript. -/
theorem mem_blankTiles_honest (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S)
    (act : ℕ → Codes) {g : ℕ} (hg : g < (TR.tiling U.layout).NT) (hu : ¬ (TR.tiling U.layout).unit g < U.layout.N) :
    g ∈ blankTiles TR U (honestTranscript P H s U act) := by
  classical
  unfold blankTiles
  rw [mem_filter]
  refine ⟨mem_range.2 hg, ?_⟩
  unfold honestTranscript
  exact ite_eq_right_iff.2 fun h => absurd h hu

/-- **Completeness from the honest words and the measured cap rate.** If every reference value of a unit of an
in-domain workload is an FP32 word's and every tile's columns are columns of its unit, `HonestTileCapRate` at `ε` gives
`CompleteSampled` at `ε` per draw. -/
theorem completeSampled_of_rate [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : UnitLayout → Prop) (Served : Workload → (ℕ → Codes) → Prop) (ε : ℝ)
    (hW : ∀ U : Workload, U.InDomain P D → ∀ u < U.layout.N, ∀ (H : Q → R) (s : S) (act : Codes) (i j : ℕ),
      (∃ wd, wordQ wd = P.ticket H s U u act i j) ∧ ∃ wd, wordQ wd = P.useful H s U u act i j)
    (hcols : ∀ (L : UnitLayout) (g : ℕ), ∀ j ∈ (TR.tiling L).cols g, j < (L.shape ((TR.tiling L).unit g)).n)
    (hR : HonestTileCapRate P TR D Served ε) :
    CompleteSampled P TR D Served ε := by
  classical
  intro U hU hWU act hS w κ hκ hw t
  let F : Fin t → (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) → Prop := fun i x =>
    (TR.tiling U.layout).unit (x.2.2 i).1 < U.layout.N ∧ ¬ TR.capOK x.1 x.2.1 U (x.2.2 i).1 (act ((TR.tiling U.layout).unit (x.2.2 i).1))
  have hsub : ∀ x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT),
      ¬ AuditAccepts (goodTiles P TR U x.1 x.2.1 (honestTranscript P x.1 x.2.1 U act) ∪
        blankTiles TR U (honestTranscript P x.1 x.2.1 U act)) x.2.2 → ∃ i ∈ (univ : Finset (Fin t)), F i x := by
    intro x hx
    obtain ⟨i, hi⟩ := not_forall.1 hx
    refine ⟨i, mem_univ _, ?_⟩
    have hg : ((x.2.2 i).1 : ℕ) < (TR.tiling U.layout).NT := (x.2.2 i).1.isLt
    by_cases hu : (TR.tiling U.layout).unit (x.2.2 i).1 < U.layout.N
    · exact ⟨hu, fun hcap => hi (mem_union_left _
        (mem_goodTiles_honest P TR U x.1 x.2.1 act hg hu (fun a b => hW U hU _ hu _ _ _ a b)
          (fun j hj => hWU _ hu j (hcols _ _ j hj)) hcap))⟩
    · exact absurd (mem_union_right _ (mem_blankTiles_honest P TR U x.1 x.2.1 act hg hu)) hi
  calc pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          ¬ AuditAccepts (goodTiles P TR U x.1 x.2.1 (honestTranscript P x.1 x.2.1 U act) ∪
            blankTiles TR U (honestTranscript P x.1 x.2.1 U act)) x.2.2)
      ≤ pr (fun x => ∃ i ∈ (univ : Finset (Fin t)), F i x) := pr_mono hsub
    _ ≤ ∑ i : Fin t, pr (F i) := Proofs.pr_exists_le_sum _ _
    _ ≤ ∑ _i : Fin t, ε := sum_le_sum fun i _ =>
        (pr_coord_le i (fun y : (Q → R) × S × Ticket w (TR.tiling U.layout).NT =>
          (TR.tiling U.layout).unit y.2.2.1 < U.layout.N ∧ ¬ TR.capOK y.1 y.2.1 U y.2.2.1 (act ((TR.tiling U.layout).unit y.2.2.1)))).trans
          (hR U hU hWU act hS w κ hκ hw)
    _ = t * ε := by simp

/-- **The sampled audit's completeness at a device record on a step `⟨[32], w, z⟩` with `w ≥ 24`**, under rev1 with the
honest kernel's `W_ref`, on any domain within `pearlCDomainDev` (so every unit is at most `2^16` deep): the measured
cap rate `HonestTileCapRate` at `ε` gives `CompleteSampled` at `ε` per draw. The honest words are proved
(`honestWords_devRev1K`); the cap rate is the one assumption. -/
theorem completeSampledDevRev1K [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {w : ℕ} (hw : 24 ≤ w) (z : ℤ)
    (G : ℕ) (pr : Prices) (sem : PearlCSem Q R S) (ρ c : ℚ) (D : UnitLayout → Prop)
    (hD : ∀ L, D L → pearlCDomainDev (devAt ⟨[32], w, z⟩ G pr) L) (Served : Workload → (ℕ → Codes) → Prop)
    (ε : ℝ)
    (hR : HonestTileCapRate (pearlCProtocolDevRev1K (devAt ⟨[32], w, z⟩ G pr) sem ρ c)
      (pearlCTilesDevRev1K (devAt ⟨[32], w, z⟩ G pr) sem ρ c) D Served ε) :
    CompleteSampled (pearlCProtocolDevRev1K (devAt ⟨[32], w, z⟩ G pr) sem ρ c)
      (pearlCTilesDevRev1K (devAt ⟨[32], w, z⟩ G pr) sem ρ c) D Served ε := by
  refine completeSampled_of_rate _ _ D Served ε ?_
    (fun _ _ _ hj => lt_of_lt_of_le (Finset.mem_Ico.1 hj).2 (min_le_right _ _)) hR
  intro U hU u hu H s act i j
  have hk : (U.layout.shape u).k ≤ 2 ^ 16 := ((hD _ hU.2) u hu).2.2.2.2.2
  exact honestWords_devRev1K hw z G pr sem ρ c H s U u act
    (lt_of_le_of_lt (Nat.div_le_self _ _) (lt_of_le_of_lt hk (by norm_num))) i j

/-- **The sampled audit's completeness at v1's record under rev1 at the cap 1/1,000 on Llama-3.1-8B's served layout**,
the honest cast at 8.72: on the population `Served`, the measured cap rate at `ε` gives `CompleteSampled` at `ε`
per draw. -/
theorem pearlCCompleteSm120v1LoopCast8p72Rev1Cap1000_llama31_8b [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (sem : PearlCSem Q R S) (Served : Workload → (ℕ → Codes) → Prop) (ε : ℝ)
    (hR : HonestTileCapRate (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears) Served ε) :
    CompleteSampled (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears) Served ε :=
  completeSampledDevRev1K (w := 26) (by norm_num) (-133) 4 Prices.sm120Loop sem _ _ _ (fun _ hL => hL.1) Served ε hR

/-- **The sampled audit's completeness at v1's record under rev1 at the cap 1/1,000 on Qwen3-8B's served layout**, the
honest cast at 8.72: on the population `Served`, the measured cap rate at `ε` gives `CompleteSampled` at `ε` per
draw. -/
theorem pearlCCompleteSm120v1LoopCast8p72Rev1Cap1000_qwen3_8b [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (sem : PearlCSem Q R S) (Served : Workload → (ℕ → Codes) → Prop) (ε : ℝ)
    (hR : HonestTileCapRate (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) qwen3_8bLinears) Served ε) :
    CompleteSampled (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) qwen3_8bLinears) Served ε :=
  completeSampledDevRev1K (w := 26) (by norm_num) (-133) 4 Prices.sm120Loop sem _ _ _ (fun _ hL => hL.1) Served ε hR

end Pouw.PearlC
