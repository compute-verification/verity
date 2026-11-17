module

public import Proofs.Flock.Soundness.E2E
public import Proofs.Flock.Soundness.ExecStratified
public import Proofs.Flock.Soundness.Binding.Weights
public import Proofs.Flock.Soundness.StrictCR

/-! The end-to-end theorem with the value binding of `hm96-sha512/row/v1` rows, whose one gap is `HmRowComputes`; none of its forms takes A3. -/

@[expose] public section

namespace FlockSoundness.Binding

open Game Model Accounting Extract LinkFinder Refine Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

theorem prob_split_accept {n : ℕ} (G : Game (Bool × Finset (Fin n))) (S : Strategy G)
    (A B : Bool × Finset (Fin n) → Prop) :
    prob (fun o => A o ∧ B o) G S ≤
      prob (fun o => (o.1 = true ∧ A o) ∧ B o) G S + prob (fun o => A o ∧ o.1 = false) G S :=
  (prob_mono (p := fun o => A o ∧ B o) (q := fun o => ((o.1 = true ∧ A o) ∧ B o) ∨ (A o ∧ o.1 = false))
    (fun o h => (Bool.eq_false_or_eq_true o.1).elim (fun h1 => Or.inl ⟨⟨h1, h.1⟩, h.2⟩)
      fun h1 => Or.inr ⟨h.1, h1⟩) G S).trans (prob_or_le _ _ G S)

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n}

/-- It takes no A3. `tr`, the serving roots' rewinding, is a named hypothesis: the refinement from the
executable's openings to the model's (`ExecRefines`) is open. -/
theorem flock_e2e_count_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' + ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((P.flock_e2e_count H E plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht K σ hCR).trans
      (add_le_add (add_le_add le_rfl (StrictCR.ksAvgBE_le_strict H E plan k (reg σ) (cont σ) hKS)) le_rfl))
    (tr.le hT))

/-- It reads the sampling design only through `ε_ks` and `ε_bind`, so it holds as stated at the live sample, with no
A3. -/
theorem flock_e2e_sampled_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan L k qF qS (reg σ) + P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((P.flock_e2e_sampled H E plan tab (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1) dp ones
      hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht σ hCR).trans
      (add_le_add (StrictCR.ksAvgBE_le_strict H E plan k (reg σ) (cont σ) hKS) le_rfl))
    (tr.le hT))

variable {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Ls : Fin m → ℕ)

/-- It takes no A3: `SamplingDesign.execStratified`'s outcomes are uniform byte tapes. -/
theorem flock_e2e_sampled_exec_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan ((SamplingDesign.execStratified St Ls).draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan (SamplingDesign.execStratified St Ls) k qF qS (reg σ) +
        P.linkBoundE (L := SamplingDesign.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((P.flock_e2e_sampled_exec H E plan tab St Ls (execAccept := fun o => o.1 = true ∧ execAccept o) (fun _ h => h.1)
      dp ones hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht σ hCR).trans
      (add_le_add (StrictCR.ksAvgBE_le_strict H E plan k (reg σ) (cont σ) hKS) le_rfl))
    (tr.le hT))

/-- It takes no A3: `SamplingDesign.execStratified`'s outcomes are uniform byte tapes. -/
theorem flock_e2e_count_exec_hm96 (σs : Fin n → Fin m) (ks : Fin m → ℕ)
    (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card) (hS : SamplingDesign.ExecStrata σs ks St)
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan ((SamplingDesign.execStratified St Ls).draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      (SamplingDesign.stratified σs ks hks).miss K + StrictCR.ksAvgStrict plan (SamplingDesign.execStratified St Ls) k qF qS (reg σ) +
        P.linkBoundE (L := SamplingDesign.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) :=
  (prob_split_accept _ σ execAccept _).trans (add_le_add
    ((P.flock_e2e_count_exec H E plan tab St Ls σs ks hks hS (execAccept := fun o => o.1 = true ∧ execAccept o)
      (fun _ h => h.1) dp ones hConst zeros hZero (rs.binding hHm) hk hRw hM hρ hr ht K σ hCR).trans
      (add_le_add (add_le_add le_rfl (StrictCR.ksAvgBE_le_strict H E plan k (reg σ) (cont σ) hKS)) le_rfl))
    (tr.le hT))

end FlockSoundness.Binding
