module

public import Proofs.Flock.Soundness.Audit.FlockPublic
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Game

/-! The end-to-end theorem as a skeleton whose every gap (`hExec`, `dp`, `hConst`, `hZero`, `vb`) is a named hypothesis,
at the model's acceptance, not the executable verifier's verdict. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

namespace Partition

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n}

/-- **Each sampled proof block's place in its table, for the rows the verifier derives** (named: 1d, 1e, S4). `unitRows u` is
unit `u`'s rows as the verifier derives them: `Rows.compose` of `derive` on its types (1d). -/
structure DerivedPlaces (unitRows : Fin n → FlockSoundness.Rows) where
  place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u
  rows : ∀ S R u (h : u ∈ S), (place S R u h).R = unitRows u

/-- The end-to-end theorems don't take it: they are about the rows' circuit. -/
def RowsL1 {CB : Verity.Boolean.Circuit} (PB : Partition CB n) (proj : (Fin C.N → Bool) → Fin CB.N → Bool) (ones : Finset (Fin C.N)) :
    Prop :=
  ∀ vals, (∀ g ∈ ones, vals g = true) → PB.incorrect (proj vals) ⊆ P.incorrect vals

/-- Its gaps are named hypotheses (`hExec`, `dp`, `hConst`, `hZero`, `vb`); `SHA512CRExpected` for `σ`'s link finders
(`hCR`) is its assumption. -/
theorem flock_e2e_count
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, h.2⟩) _ σ).trans
    (P.flock_batched_count_public H E plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht K σ hCR)

theorem flock_e2e_sampled
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, h.2⟩) _ σ).trans
    (P.flock_batched_sampled_public H E plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht σ hCR)

end Partition

open Partition (batchedSession ksAvgBE)

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **Each sampled proof block's place in its table, for the program's circuit** (named). Unit `u` is an instance of its rows
`p.rows u` (`inst`; `Prog.isRowsUnit` gives one), and the statement of the table it is sampled into holds those rows in
block `o` at the columns `col`:
* `placed`: the statement's level-0 matrices hold `p.rows u` there, up to other matrices every satisfying witness
  satisfies (`PlacedSat`, as at a pinned output's row). From the executable's accept step,
  `placement_of_setupH` gives it for the rows `Rows.ofNet` reads off setupH's net; that those rows are the program's
  unit's (W6) is open;
* `aliased`: the unit's input columns on one gate carry one value in every satisfying witness (`Layout.Aliased`, which
  `unitPlace_of_setupH` takes as a hypothesis too). -/
structure ProgPlaces where
  inst : ∀ u, (p.partition outs).IsRowsUnit u (p.rows u)
  o : ∀ S R u, u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl)
  col : ∀ S R u, u ∈ S → Fin (p.rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog)
  placed : ∀ S R u (h : u ∈ S), PlacedSat (Partition.tabOf plan tab S R u).S (p.rows u) (col S R u h)
  aliased : ∀ S R u (h : u ∈ S), ∀ c c', (inst u).wire c = (inst u).wire c' → c.val < (p.rows u).nIn →
    c'.val < (p.rows u).nIn → ∀ z : Witness (Partition.tabOf plan tab S R u).S.m,
      (Partition.tabOf plan tab S R u).S.Satisfies z →
      (Partition.tabOf plan tab S R u).S.block z (o S R u h) (col S R u h c) =
        (Partition.tabOf plan tab S R u).S.block z (o S R u h) (col S R u h c')

variable {p outs plan tab}

/-- The places, as `DerivedPlaces` for the program's rows. -/
def ProgPlaces.derived (pp : ProgPlaces p outs plan tab) : (p.partition outs).DerivedPlaces plan tab p.rows where
  place S R u h := ⟨p.rows u, pp.inst u, pp.o S R u h, pp.col S R u h, pp.placed S R u h, pp.aliased S R u h⟩
  rows _ _ _ _ := rfl

variable (p outs plan tab) {D : Type} [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {L : SamplingDesign n}

theorem _root_.FlockSoundness.Prog.flock_e2e_count
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.partition outs).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).incorrect
          (Xpub ones zeros ((p.partition outs).Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (p.partition outs).flock_e2e_count H E plan tab hExec pp.derived ones hConst zeros hZero vb hk hRw hM hρ hr ht K σ hCR

theorem _root_.FlockSoundness.Prog.flock_e2e_sampled
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.partition outs).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.partition outs).incorrect
          (Xpub ones zeros ((p.partition outs).Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (p.partition outs).flock_e2e_sampled H E plan tab hExec pp.derived ones hConst zeros hZero vb hk hRw hM hρ hr ht σ hCR

end FlockSoundness.Audit
