import Proofs.Flock.Soundness.Audit.FlockPublic

/-!
# The end-to-end theorem: a skeleton, every gap a named hypothesis

**The statement.** Except with the stated ε, whenever the audit accepts, the integrity profile holds for the committed
circuit and values, 1 on the constants and 0 on the zero, which are public:
* `flock_e2e_count`: the count curve, `Pr[accept ∧ at least K units wrong] ≤ miss_L(K) + ε_ks(σ) + δ_link`;
* `flock_e2e_drawn`: `Pr[accept ∧ a drawn unit wrong] ≤ ε_ks(σ) + δ_link`.

"Wrong" is measured on the audit's circuit `C`, at the committed values: the circuit the statement's rows make, each unit
its template's rows (`Prog.circuit`: the circuit is chosen on the untrusted side, and the lowering does
no optimizing). `Prog.flock_e2e_count` and `Prog.flock_e2e_drawn` state it there, with each drawn unit's place from its
table's statement (`ProgPlaces`). What a correct unit computes is the certificate's: rows that are `GateRows.rows` of a
gate circuit `G` compute `G`, so a correct unit's committed values compute `G` (`IsRowsUnit.computes_of_cert`,
`Certificate.lean`). The only cryptographic assumption is `SHA512CRExpected`, for the link finders built from the prover
the theorem bounds (`hCR : LinkCR … (reg σ) (cont σ) …`), not from every prover at once. `ε_ks(σ)` is the prover's
collision advantages; under `SHA512CRStrict` it is at most `StrictCR.ksAvgStrict` (`StrictCR.ksAvgBE_le_strict`), which
the `_hm96` forms state (`Binding/E2E.lean`).

**What the statement carries today.**
* "Accept" is `execAccept`, any event on the audit's outcome that implies the model's acceptance (`hExec`). So as pinned,
  these are the model-level theorems. The executable's verdict, `flock-verify` on the run's transcript, is not an event
  on that outcome. It enters when `hExec` is discharged, and that changes the statement's shape.
* "Rows derived by the verifier" is not in the generic statement: `unitRows` appears only in `dp`. In the `Prog` forms the
  units' rows are the program's (`Prog.rows`), and `ProgPlaces.placed` says the table's statement holds them.

**The composition.** It composes `flock_batched_count_public` and `flock_batched_drawn_public`, which are the compiled
audit, the link theorem and knowledge soundness. Each piece not yet proved is a named hypothesis (`ASSUMPTIONS.md`,
The headline, gives each one's kind):
* `hExec`: the executable verifier's acceptance implies the model's, which the verifier's refinement theorem gives;
* `dp : DerivedPlaces`: each drawn unit's place in its table, for the rows the verifier derives. For the program's
  circuit it is `ProgPlaces`: `placed`, the table's statement holds the unit's rows (`placement_of_setupH` for the rows
  `Rows.ofNet` reads off setupH's net, and W6, that those are the program's), and `aliased` (`Layout.Aliased`);
* `hConst : ConstCols`: at each drawn unit's place, only the constant's column sits on a gate of `ones`. No unit's input
  column reads a constant, a fact of the circuit's wiring. With it the committed values carry 1 on `ones`: the
  committed transcript is public there (`Xpub`, `FlockPublic.lean`), since every satisfying message decodes the constant
  to 1;
* `hZero : ZeroCols`: every column on a gate of `zeros`, the statement's zero, carries 0 in every satisfying witness.
  These are the forced-zero rows, which the verifier's Δ copies to the slot inputs that read the zero. The
  committed transcript is 0 there, so the profile is about the zero padding at 0. This is the zero's binding;
* `vb : ValueBinding`: the binding of wires to registered commit strings, already named on `main`.
-/

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- **Each drawn unit's place in its table, for the rows the verifier derives** (named: 1d, 1e, S4). `unitRows u` is
unit `u`'s rows as the verifier derives them: `Rows.compose` of `derive` on its types (1d). -/
structure DerivedPlaces (unitRows : Fin n → FlockSoundness.Rows) where
  place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u
  rows : ∀ S R u (h : u ∈ S), (place S R u h).R = unitRows u

/-- **A program of units' Boolean circuit refines to its rows**: at values that carry 1 on `ones`, the gates the units'
constants read, every unit wrong on the program's Boolean circuit `CB` (partition `PB`, the same units) is wrong on its
rows, the audit's circuit. `proj` reads the Boolean circuit's wires off the rows' columns. The end-to-end theorems don't
take it: they are about the rows' circuit. `UProg.rowsL1` proves it for every program of units over circuit types, which
carries `UProg.flock_e2e_count` and `UProg.flock_e2e_drawn` to the Boolean circuit. -/
def RowsL1 {CB : Circuit} (PB : Partition CB n) (proj : (Fin C.N → Bool) → Fin CB.N → Bool) (ones : Finset (Fin C.N)) :
    Prop :=
  ∀ vals, (∀ g ∈ ones, vals g = true) → PB.wrong (proj vals) ⊆ P.wrong vals

/-- **The end-to-end count curve (skeleton).** If `execAccept` holds (an event that implies the model's acceptance), the
probability that at least `K` units of the committed circuit are wrong at the committed values is at most
`miss_L(K) + ε_ks(σ) + δ_link`. The committed values are 1 on `ones`, 0 on `zeros` and the plurality elsewhere
(`Xpub`). The gaps are named hypotheses: `hExec`, `dp`, `hConst`, `hZero` and `vb`;
`SHA512CRExpected` for `σ`'s link finders (`hCR`) is the assumption of record. -/
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
        K ≤ (P.wrong (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, h.2⟩) _ σ).trans
    (P.flock_batched_count_public H E plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht K σ hCR)

/-- **The end-to-end drawn-unit bound (skeleton).** If `execAccept` holds (an event that implies the model's
acceptance), the probability that a drawn unit of the committed circuit is wrong at the committed values (`Xpub`, as in
`flock_e2e_count`) is at most `ε_ks(σ) + δ_link`. -/
theorem flock_e2e_drawn
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
        ¬ Disjoint (P.wrong (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, h.2⟩) _ σ).trans
    (P.flock_batched_drawn_public H E plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht σ hCR)

end Partition

/-! ## On the circuit of the statement's rows -/

open Partition (batchedSession ksAvgBE)

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **Each drawn unit's place in its table, for the program's circuit** (named). Unit `u` is an instance of its rows
`p.rows u` (`inst`; `Prog.isRowsUnit` gives one), and the statement of the table it is drawn into holds those rows in
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

variable (p outs plan tab) {D : Type} [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {L : Law n}

/-- **The end-to-end count curve on the circuit of the statement's rows.** `flock_e2e_count` at `C = p.circuit outs`,
each unit its template's rows, with each drawn unit's place from `ProgPlaces`: if `execAccept` holds, the probability
that at least `K` units of the program are wrong at the committed values is at most `miss_L(K) + ε_ks(σ) + δ_link`. -/
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
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros ((p.partition outs).Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (p.partition outs).flock_e2e_count H E plan tab hExec pp.derived ones hConst zeros hZero vb hk hRw hM hρ hr ht K σ hCR

/-- **The end-to-end drawn-unit bound on the circuit of the statement's rows.** `flock_e2e_drawn` at
`C = p.circuit outs`, as in `Prog.flock_e2e_count`. -/
theorem _root_.FlockSoundness.Prog.flock_e2e_drawn
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
        ¬ Disjoint ((p.partition outs).wrong
          (Xpub ones zeros ((p.partition outs).Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (p.partition outs).flock_e2e_drawn H E plan tab hExec pp.derived ones hConst zeros hZero vb hk hRw hM hρ hr ht σ hCR

end FlockSoundness.Audit
