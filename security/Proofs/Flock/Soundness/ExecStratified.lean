module

public import Proofs.Flock.Soundness.ExecSample
public import Proofs.Flock.Soundness.Audit.Stratified
public import Proofs.Flock.Soundness.E2E

/-! The executable stratified sample escapes no more often than `SamplingDesign.stratified`
(`stratified_exec_escape_le`), and the e2e theorems at its sampling design. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace SamplingDesign

variable {n m : ℕ}

/-- The executable strata `S` describe `σ`'s: stratum `s` samples `k s` of its units, has `σ`'s stratum's size, and its
`j`-th unit, for `j` below that size, runs over `σ`'s stratum `s`. -/
def ExecStrata (σ : Fin n → Fin m) (k : Fin m → ℕ) (S : Fin m → Flock.Draw.Stratum) : Prop :=
  ∀ s, (S s).k = k s ∧ (S s).size = (stratum σ s).card ∧
    (range (S s).size).image (S s).unit = (stratum σ s).map Fin.valEmbedding

/-- The live sample extends a stream that runs out and reruns instead (`execOS`). -/
def execStratified (S : Fin m → Flock.Draw.Stratum) (L : Fin m → ℕ) : SamplingDesign n where
  Ω := (s : Fin m) → Fin (L s) → Fin 256
  draw ω := match Flock.Draw.stratified (Array.ofFn S) (Array.ofFn fun s => streamOf (ω s)) with
    | some D => univ.filter fun u => u.val ∈ D
    | none => univ

theorem forIn_append (g : ℕ → Option (Array ℕ)) (u : ℕ → ℕ → ℕ) (F : ℕ → Array ℕ → Option (ForInStep (Array ℕ)))
    (hF : ∀ i out, F i out = (g i).bind fun l => some (ForInStep.yield (out ++ l.map (u i)))) :
    ∀ (l : List ℕ) (out res : Array ℕ), forIn l out F = some res →
      (∀ x ∈ out, x ∈ res) ∧ ∀ i ∈ l, ∃ li, g i = some li ∧ ∀ j ∈ li, u i j ∈ res := by
  intro l
  induction l with
  | nil =>
    intro out res h
    obtain rfl : out = res := by simpa using h
    exact ⟨fun x hx => hx, fun i hi => absurd hi (List.not_mem_nil)⟩
  | cons i is ih =>
    intro out res h
    rw [List.forIn_cons, hF] at h
    cases hg : g i with
    | none => rw [hg] at h; exact absurd h (by simp)
    | some li =>
      rw [hg] at h
      simp only [Option.bind_some] at h
      obtain ⟨hout, his⟩ := ih _ res h
      refine ⟨fun x hx => hout x (Array.mem_append_left _ hx), fun i' hi' => ?_⟩
      rcases List.mem_cons.1 hi' with rfl | hi'
      · exact ⟨li, hg, fun j hj => hout _ (Array.mem_append_right _ (Array.mem_map_of_mem hj))⟩
      · exact his i' hi'

theorem stratified_units (strata : Array Flock.Draw.Stratum) (streams : Array ByteArray) (D : Array ℕ)
    (h : Flock.Draw.stratified strata streams = some D) :
    ∀ i < strata.size, ∃ li, Flock.Draw.subset strata[i]!.k strata[i]!.size (streams.getD i ByteArray.empty) = some li ∧
      ∀ j ∈ li, strata[i]!.unit j ∈ D := by
  unfold Flock.Draw.stratified at h
  dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize] at h
  obtain ⟨res, hres, hD⟩ := Option.bind_eq_some_iff.1 h
  have key := forIn_append (fun i => Flock.Draw.subset strata[i]!.k strata[i]!.size (streams.getD i ByteArray.empty))
    (fun i => strata[i]!.unit) _ (fun i out => rfl) _ _ _ hres
  intro i hi
  obtain ⟨li, hli, hmem⟩ := key.2 i (by
    show i ∈ List.range' 0 strata.size 1
    rw [List.mem_range'_1]; omega)
  refine ⟨li, hli, fun j hj => ?_⟩
  have hD' : res.qsort (fun x1 x2 => decide (x1 < x2)) = D := Option.some.inj hD
  rw [← hD', SampleExec.mem_qsort]
  exact hmem j hj

/-- Its probability space is independent uniform streams; that the challenger's coins are uniform is A3
(`Assumptions.UniformRandomBytes`), which `sampleOS_stratified_escape_le` takes for the live sample. -/
theorem stratified_exec_escape_le (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S) (L : Fin m → ℕ) (B : Finset (Fin n)) :
    prCoin (fun f : (s : Fin m) → Fin (L s) → Fin 256 =>
        ∃ D, Flock.Draw.stratified (Array.ofFn S) (Array.ofFn fun s => streamOf (f s)) = some D ∧
          ∀ u ∈ B, u.val ∉ D) ≤
      (stratified σ k hk).missProb B := by
  classical
  let Bl : (s : Fin m) → Finset (Fin (S s).size) := fun s =>
    univ.filter fun j => ∃ u ∈ B, (S s).unit j.val = u.val
  have hkS : ∀ s, (S s).k ≤ (S s).size := fun s => by rw [(hS s).1, (hS s).2.1]; exact hk s
  calc prCoin (fun f : (s : Fin m) → Fin (L s) → Fin 256 =>
          ∃ D, Flock.Draw.stratified (Array.ofFn S) (Array.ofFn fun s => streamOf (f s)) = some D ∧
            ∀ u ∈ B, u.val ∉ D)
      ≤ prCoin (fun f : (s : Fin m) → Fin (L s) → Fin 256 => ∀ s ∈ (univ : Finset (Fin m)),
          (fun s (g : Fin (L s) → Fin 256) => ∃ D, Flock.Draw.subset (S s).k (S s).size (streamOf g) = some D ∧
            ∀ j ∈ Bl s, j.val ∉ D) s (f s)) := by
        refine prCoin_mono fun f ⟨D, hD, hmiss⟩ s _ => ?_
        obtain ⟨li, hli, hmem⟩ := stratified_units _ _ D hD s.val (by simp)
        have hSs : (Array.ofFn S)[s.val]! = S s := by
          rw [getElem!_pos (Array.ofFn S) s.val (by simp)]; simp
        have hfs : (Array.ofFn fun s => streamOf (f s)).getD s.val ByteArray.empty = streamOf (f s) := by
          simp [Array.getD_eq_getD_getElem?]
        rw [hSs, hfs] at hli
        rw [hSs] at hmem
        refine ⟨li, hli, fun j hj hjli => ?_⟩
        obtain ⟨u, hu, hue⟩ := (Finset.mem_filter.1 hj).2
        exact hmiss u hu (hue ▸ hmem _ hjli)
    _ = ∏ s, prCoin (fun g : Fin (L s) → Fin 256 =>
          ∃ D, Flock.Draw.subset (S s).k (S s).size (streamOf g) = some D ∧ ∀ j ∈ Bl s, j.val ∉ D) :=
        prCoin_pi_forall (Ω := fun s => Fin (L s) → Fin 256) univ (fun s (g : Fin (L s) → Fin 256) =>
          ∃ D, Flock.Draw.subset (S s).k (S s).size (streamOf g) = some D ∧ ∀ j ∈ Bl s, j.val ∉ D)
    _ ≤ ∏ s, (subset (S s).size (S s).k (hkS s)).missProb (Bl s) :=
        Finset.prod_le_prod fun s _ => subset_exec_escape_le _ _ _ (hkS s) (Bl s)
    _ ≤ (stratified σ k hk).missProb B := by
        rw [stratified_escape]
        refine Finset.prod_le_prod fun s _ => ?_
        rw [subset_escape]
        have hcard : (B.filter (σ · = s)).card ≤ (Bl s).card := by
          have hsub : (B.filter (σ · = s)).map Fin.valEmbedding ⊆ (Bl s).image fun j => (S s).unit j.val := by
            intro x hx
            obtain ⟨u, hu, rfl⟩ := Finset.mem_map.1 hx
            have hus : u.val ∈ (stratum σ s).map Fin.valEmbedding :=
              Finset.mem_map_of_mem _ (mem_stratum.2 (Finset.mem_filter.1 hu).2)
            rw [← (hS s).2.2] at hus
            obtain ⟨j, hj, hje⟩ := Finset.mem_image.1 hus
            refine Finset.mem_image.2 ⟨⟨j, Finset.mem_range.1 hj⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _,
              u, (Finset.mem_filter.1 hu).1, hje⟩, hje⟩
          calc (B.filter (σ · = s)).card = ((B.filter (σ · = s)).map Fin.valEmbedding).card := (Finset.card_map _).symm
            _ ≤ ((Bl s).image fun j => (S s).unit j.val).card := Finset.card_le_card hsub
            _ ≤ (Bl s).card := Finset.card_image_le
        have hnum : ((S s).size - (Bl s).card).choose (S s).k ≤
            ((stratum σ s).card - (B.filter (σ · = s)).card).choose (k s) := by
          rw [(hS s).1]
          calc ((S s).size - (Bl s).card).choose (k s) ≤ ((S s).size - (B.filter (σ · = s)).card).choose (k s) :=
                Nat.choose_le_choose _ (Nat.sub_le_sub_left hcard _)
            _ = _ := by rw [(hS s).2.1]
        have hden : (((S s).size.choose (S s).k : ℕ) : ℝ≥0∞) = (((stratum σ s).card.choose (k s) : ℕ) : ℝ≥0∞) := by
          rw [(hS s).2.1, (hS s).1]
        rw [hden]
        exact ENNReal.div_le_div_right (by exact_mod_cast hnum) _

theorem execStratified_escape_le (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S) (L : Fin m → ℕ) (B : Finset (Fin n)) :
    (execStratified (n := n) S L).missProb B ≤ (stratified σ k hk).missProb B := by
  classical
  rcases B.eq_empty_or_nonempty with rfl | ⟨b, hb⟩
  · have h1 : (stratified σ k hk).missProb ∅ = 1 := by
      rw [stratified_escape]
      refine Finset.prod_eq_one fun s _ => ?_
      simp only [Finset.filter_empty, Finset.card_empty, Nat.sub_zero]
      exact ENNReal.div_self (by exact_mod_cast (Nat.choose_pos (hk s)).ne') (ENNReal.natCast_ne_top _)
    rw [h1]
    exact SampleExec.prCoin_le_one _
  · refine le_trans (prCoin_mono fun ω hd => ?_) (stratified_exec_escape_le σ k hk S hS L B)
    change Disjoint (match Flock.Draw.stratified (Array.ofFn S) (Array.ofFn fun s => streamOf (ω s)) with
      | some D => univ.filter fun u => u.val ∈ D
      | none => univ) B at hd
    split at hd
    · rename_i D hD
      exact ⟨D, hD, fun u hu huD => Finset.disjoint_left.1 hd (Finset.mem_filter.2 ⟨Finset.mem_univ _, huD⟩) hu⟩
    · exact absurd hb (Finset.disjoint_left.1 hd (Finset.mem_univ b))

theorem execStratified_miss_le (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S) (L : Fin m → ℕ) (K : ℕ) :
    (execStratified (n := n) S L).miss K ≤ (stratified σ k hk).miss K :=
  iSup₂_mono fun B _ => execStratified_escape_le σ k hk S hS L B

end SamplingDesign

local notation "𝔽" => FlockLevel3.GF128

namespace Partition

open Model Accounting Extract LinkFinder

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Ls : Fin m → ℕ)

theorem flock_e2e_sampled_exec
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := SamplingDesign.execStratified St Ls) plan tab vb k Rw M ρ t' :=
  P.flock_e2e_sampled H E plan tab hExec dp ones hConst zeros hZero vb hk hRw hM hρ hr ht σ hCR

theorem flock_e2e_count_exec (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (hS : SamplingDesign.ExecStrata σs ks St)
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      (SamplingDesign.stratified σs ks hks).miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        P.linkBoundE (L := SamplingDesign.execStratified St Ls) plan tab vb k Rw M ρ t' := by
  refine (P.flock_e2e_count H E plan tab hExec dp ones hConst zeros hZero vb hk hRw hM hρ hr ht K
    σ hCR).trans ?_
  gcongr
  exact SamplingDesign.execStratified_miss_le σs ks hks St hS Ls K

end Partition

end FlockSoundness.Audit
