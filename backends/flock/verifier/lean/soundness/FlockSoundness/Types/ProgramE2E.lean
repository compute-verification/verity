import FlockSoundness.Types.ProgramPlaces
import FlockSoundness.Audit.FlockPublic

/-!
# The end-to-end theorem on a program of units, with `hL1` discharged (verified-lowering phase 1, S4d)

`flock_e2e_count` and `flock_e2e_drawn` (`E2E.lean`) on a program of units (`Types.Program`):
- the committed circuit is the program's rows circuit, `p.C`, and "wrong" is measured on its Boolean circuit, `p.CB`,
  read off the rows through `p.mapG`;
- `hL1` is `UProg.rowsL1`, the committed values are 1 on the constant's gate `one` and the plurality elsewhere (`Xpub`),
  and `hConst` says that only a unit's constant column sits on `one`.

The other gaps stay named, as in `E2E.lean`: `hExec`, `dp` and `vb`. The `_classes` forms build `dp` from each drawn
unit's table class (`UProg.derivedPlaces_of_classes`), what 1d and 1e give for an accepted statement, and discharge
`hConst` (`UProg.constCols_of_classes`): each unit's constant column is wired to `one` (`pairs_oneGate`). A program has no
zero of its own, so `zeros` is empty here.
-/

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

namespace Partition

variable {kIn one : ℕ} (hone : one < kIn) {outs : List ℕ} (p : FlockSoundness.UProg kIn one outs) {Reg : Type} {D : Type}
  [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
  {L : Law p.n}

/-- **The end-to-end count curve on a program of units.** `flock_e2e_count` with `hL1` discharged by `UProg.rowsL1`:
if `execAccept` holds, the probability that at least `K` units are wrong on the program's Boolean circuit, at the
committed values read through `mapG`, is at most `miss_L(K) + ε_ks(σ) + δ_link`, when only a unit's constant column sits
on the constant's gate. -/
theorem _root_.FlockSoundness.UProg.flock_e2e_count
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin p.n → FlockSoundness.Rows} (dp : (p.P hone).DerivedPlaces plan tab unitRows)
    (hConst : (p.P hone).ConstCols plan tab dp.place {p.oneOf hone})
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab ((p.P hone).placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ ((p.PB hone).wrong
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (p.P hone).flock_e2e_count H E plan tab hExec dp (p.PB hone) (fun vals => vals ∘ p.mapG hone) {p.oneOf hone}
    (p.rowsL1 hone) hConst ∅ (fun _ _ _ _ _ h => absurd h (Finset.notMem_empty _)) vb hk hRw hM hρ hr ht K σ hCR

/-- **The end-to-end drawn-unit bound on a program of units.** `flock_e2e_drawn` with `hL1` discharged by
`UProg.rowsL1`: if `execAccept` holds, the probability that a drawn unit is wrong on the program's Boolean circuit is at
most `ε_ks(σ) + δ_link`, when only a unit's constant column sits on the constant's gate. -/
theorem _root_.FlockSoundness.UProg.flock_e2e_drawn
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin p.n → FlockSoundness.Rows} (dp : (p.P hone).DerivedPlaces plan tab unitRows)
    (hConst : (p.P hone).ConstCols plan tab dp.place {p.oneOf hone})
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab ((p.P hone).placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.PB hone).wrong
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (p.P hone).flock_e2e_drawn H E plan tab hExec dp (p.PB hone) (fun vals => vals ∘ p.mapG hone) {p.oneOf hone}
    (p.rowsL1 hone) hConst ∅ (fun _ _ _ _ _ h => absurd h (Finset.notMem_empty _)) vb hk hRw hM hρ hr ht σ hCR

/-- **Only a unit's constant column sits on `one`**, at the places built from the tables' classes: each unit's instance
wires its constant column to `one` (`pairs_oneGate`). -/
theorem _root_.FlockSoundness.UProg.constCols_of_classes
    (cls : ∀ S R u, u ∈ S → TableClass (tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u)) :
    (p.P hone).ConstCols plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place
      {p.oneOf hone} :=
  (p.P hone).constCols_singleton plan tab _ _ fun _ _ u _ => Fin.ext (p.pairs_oneGate hone u)

/-- **The end-to-end count curve on a program of units, from its tables' classes.** `UProg.flock_e2e_count` with `dp`
built from each drawn unit's table class: unit `u`, drawn in `S` under `R`, sits at slot `slot` of block `blk` of its
table's class `cls`, whose derivation inputs it reads, and the slot's inputs that read one of its sources copy one
position, or forced-zero rows (`hcp`). -/
theorem _root_.FlockSoundness.UProg.flock_e2e_count_classes
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ ((p.PB hone).wrong
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  UProg.flock_e2e_count hone p H E plan tab hExec (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp)
    (p.constCols_of_classes hone plan tab cls slot blk hrd hcp) vb hk hRw hM hρ hr ht K σ hCR

/-- **The end-to-end drawn-unit bound on a program of units, from its tables' classes.** `UProg.flock_e2e_drawn` with
`dp` built from each drawn unit's table class, as in `UProg.flock_e2e_count_classes`. -/
theorem _root_.FlockSoundness.UProg.flock_e2e_drawn_classes
    {execAccept : Bool × Finset (Fin p.n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (cls : ∀ S R u, u ∈ S → TableClass (tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (blk : ∀ S R u, u ∈ S → Fin (2 ^ (tabOf plan tab S R u).S.nbl))
    (hrd : ∀ S R u (h : u ∈ S), (p.spec hone u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.P hone).ValueBinding plan tab
      ((p.P hone).placeDecoder plan tab (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp).place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : (p.P hone).LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.PB hone).wrong
          (Xpub {p.oneOf hone} ∅ ((p.P hone).Xplur H E plan tab vb (reg σ) (cont σ) k Rw) ∘ p.mapG hone)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.P hone).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  UProg.flock_e2e_drawn hone p H E plan tab hExec (p.derivedPlaces_of_classes hone plan tab cls slot blk hrd hcp)
    (p.constCols_of_classes hone plan tab cls slot blk hrd hcp) vb hk hRw hM hρ hr ht σ hCR

end Partition

end FlockSoundness.Audit
