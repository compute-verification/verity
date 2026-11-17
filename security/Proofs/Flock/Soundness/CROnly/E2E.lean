module

public import Proofs.Flock.Soundness.Headline
public import Proofs.Flock.Soundness.CROnly.Defs
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Binding.E2E
import Proofs.Flock.Soundness.CROnly.Link

/-! The end-to-end theorems under `cr/sha-512` alone, each about the non-ZK session (`batchedSession`) and none about `--zk`. -/

@[expose] public section

namespace FlockSoundness.CROnly

open Game Model Accounting Extract LinkFinder Refine Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

section Generic

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n}

theorem linkBoundE_at_cube {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : P.ValueBinding plan tab decode Hc) {k Rw M : ℕ} {ρ t' q : ℝ} (hq : 0 < q)
    (hcube : q ^ 3 = t' * 2 ^ (512 : ℝ)) :
    P.linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) =
      ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (P.Ωc plan tab vb L c) : ℝ) / Fintype.card L.Ω) *
        (3 * (1 + k) * (q ^ 2 / 2 ^ (512 : ℝ)) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) := by
  unfold linkBoundE
  rw [strictRunCost_at_cube k hq hcube]

theorem flock_e2e_count
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        P.linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  P.flock_e2e_count H E plan tab hExec dp ones hConst zeros hZero vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le K σ
    (linkCR_of_strict P H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

theorem flock_e2e_sampled
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  P.flock_e2e_sampled H E plan tab hExec dp ones hConst zeros hZero vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le σ
    (linkCR_of_strict P H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

theorem flock_e2e_count_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Binding.Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : Binding.HmRows lay)
    (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  Binding.flock_e2e_count_hm96 P H E plan tab dp ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr
    (strictRunCost_pos ht hq).le K σ (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs)
    hKS tr hT

theorem flock_e2e_sampled_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Binding.Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : Binding.HmRows lay)
    (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  Binding.flock_e2e_sampled_hm96 P H E plan tab dp ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr
    (strictRunCost_pos ht hq).le σ (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs)
    hKS tr hT

variable {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Ls : Fin m → ℕ)

theorem flock_e2e_sampled_exec_hm96
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Binding.Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : Binding.HmRows lay)
    (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan ((SamplingDesign.execStratified St Ls).draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan (SamplingDesign.execStratified St Ls) k qF qS (reg σ) +
        P.linkBoundE (L := SamplingDesign.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  Binding.flock_e2e_sampled_exec_hm96 P H E plan tab St Ls dp ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr
    (strictRunCost_pos ht hq).le σ (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs)
    hKS tr hT

theorem flock_e2e_count_exec_hm96 (σs : Fin n → Fin m) (ks : Fin m → ℕ)
    (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card) (hS : SamplingDesign.ExecStrata σs ks St)
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Binding.Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : Binding.HmRows lay)
    (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan ((SamplingDesign.execStratified St Ls).draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (SamplingDesign.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      (SamplingDesign.stratified σs ks hks).miss K + StrictCR.ksAvgStrict plan (SamplingDesign.execStratified St Ls) k qF qS (reg σ) +
        P.linkBoundE (L := SamplingDesign.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  Binding.flock_e2e_count_exec_hm96 P H E plan tab St Ls σs ks hks hS dp ones hConst zeros hZero lay rs hHm hk hRw
    hM hρ hr (strictRunCost_pos ht hq).le K σ
    (linkCR_of_strict P H E plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs) hKS tr hT

end Generic

section Prog

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

section Session

variable {D : Type} [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {L : SamplingDesign n}

theorem prog_e2e_count
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.partition outs) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros ((p.partition outs).Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  p.flock_e2e_count outs plan tab H E hExec pp ones hConst zeros hZero vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le
    K σ (linkCR_of_strict (p.partition outs) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

theorem prog_e2e_sampled
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCRs : LinkCRStrict (p.partition outs) H E plan tab vb (reg σ) (cont σ) k Rw M t' q) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.partition outs).incorrect
          (Xpub ones zeros ((p.partition outs).Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) +
        (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ (strictRunCost t' q) :=
  p.flock_e2e_sampled outs plan tab H E hExec pp ones hConst zeros hZero vb hk hRw hM hρ hr (strictRunCost_pos ht hq).le
    σ (linkCR_of_strict (p.partition outs) H E plan tab vb (reg σ) (cont σ) ht hq hCRs)

end Session

/-- Every cryptographic hypothesis is an instance of `cr/sha-512`, for finders built from the prover bounded; A3
(`hA3`) is among the headline's other hypotheses. -/
theorem flock_headline
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (K : ℕ)
    (σ : Strategy (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSession H512 enc512 plan)))
    (hCRs : LinkCRStrict (p.partition outs) H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t' q)
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H512 enc512
      (plan ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ)) j
      (batchStrat H512 enc512 plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H512
      (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ
      fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H512 qT) :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros ((p.partition outs).Xplur H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSession H512 enc512 plan)) σ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict plan (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        (p.partition outs).linkBoundE (L := SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) plan tab
          (rs.binding hHm) k Rw M ρ (strictRunCost t' q) +
        ENNReal.ofReal (tr.bound qT) :=
  p.flock_headline outs plan tab σs ks hks St hS Kd pop Lb src hA3 pp ones hConst zeros hZero lay rs hHm hk hRw hM hρ
    hr (strictRunCost_pos ht hq).le K σ
    (linkCR_of_strict (p.partition outs) H512 enc512 plan tab (rs.binding hHm) (reg σ) (cont σ) ht hq hCRs) hKS tr hT

end Prog

end FlockSoundness.CROnly
