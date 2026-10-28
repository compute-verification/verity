import Proofs.Flock.Soundness.Discharge.Exec.FlockPublicL
import Proofs.Flock.Soundness.E2E

/-!
# The end-to-end theorem: a skeleton, every gap a named hypothesis

**The statement.** Except with the stated ε, whenever the audit accepts, the integrity profile holds for the committed
circuit and values, 1 on the constants and 0 on the zero, which are public:
* `flock_e2e_countL`: the count curve, `Pr[accept ∧ at least K units wrong] ≤ miss_L(K) + ε_ks(σ) + δ_link`;
* `flock_e2e_drawnL`: `Pr[accept ∧ a drawn unit wrong] ≤ ε_ks(σ) + δ_link`.

"Wrong" is measured on the audit's circuit `C`, at the committed values: the circuit the statement's rows make, each unit
its template's rows (`Prog.circuit`: the circuit is chosen on the untrusted side, and the lowering does
no optimizing). `Prog.flock_e2e_count` and `Prog.flock_e2e_drawn` state it there, with each drawn unit's place from its
table's statement (`ProgPlaces`). What a correct unit computes is the certificate's: rows that are `GateRows.rows` of a
gate circuit `G` compute `G`, so a correct unit's committed values compute `G` (`IsRowsUnit.computes_of_cert`,
`Certificate.lean`). The only cryptographic assumption is `SHA512CRExpected`, for the link finders built from the prover
the theorem bounds (`hCR : LinkCRL … (reg σ) (cont σ) …`), not from every prover at once. `ε_ks(σ)` is the prover's
collision advantages; under `SHA512CRStrict` it is at most `StrictCR.ksAvgStrict` (`StrictCR.ksAvgBE_le_strict`), which
the `_hm96` forms state (`Binding/E2E.lean`).

**What the statement carries today.**
* "Accept" is `execAccept`, any event on the audit's outcome that implies the Lean protocol's acceptance (`hExec`). So
  as pinned, these are theorems about the protocol as Lean defines it. The executable's verdict, `flock-verify` on the
  run's transcript, is not an event on that outcome. It enters when `hExec` is discharged, and that changes the
  statement's shape.
* "Rows derived by the verifier" is not in the generic statement: `unitRows` appears only in `dp`. In the `Prog` forms the
  units' rows are the program's (`Prog.rows`), and `ProgPlaces.placed` says the table's statement holds them.

**The composition.** It composes `flock_batched_count_publicL` and `flock_batched_drawn_publicL`, which are the compiled
audit, the link theorem and knowledge soundness, all on `main`. Each piece not yet proved is a named hypothesis. The
checklist `assumption-notes/e2e-checklist.md` gives each one's owner and how it is discharged:
* `hExec`: the executable verifier's acceptance implies the Lean protocol's. This is the refinement lane's theorem;
* `dp : DerivedPlaces`: each drawn unit's place in its table, for the rows the verifier derives. For the program's
  circuit it is `ProgPlaces`: `placed`, the table's statement holds the unit's rows (`placement_of_setupH` for the rows
  `Rows.ofNet` reads off setupH's net, and W6, that those are the program's), and `aliased` (`Layout.Aliased`);
* `hConst : ConstCols`: at each drawn unit's place, only the constant's column sits on a gate of `ones`. No unit's input
  column reads a constant, a fact of the circuit's wiring (S4). With it the committed values carry 1 on `ones`: the
  committed transcript is public there (`Xpub`, `FlockPublic.lean`), since every satisfying message decodes the constant
  to 1;
* `hZero : ZeroCols`: every column on a gate of `zeros`, the statement's zero, carries 0 in every satisfying witness.
  These are the forced-zero rows, which the verifier's Δ copies to the slot inputs that read the zero (1d, 1e, S4). The
  committed transcript is 0 there, so the profile is about the zero padding at 0. This is the zero's binding;
* `vb : ValueBinding`: the binding of wires to registered commit strings, already named on `main`.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

section

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- **The end-to-end count curve (skeleton).** If `execAccept` holds (an event that implies the Lean protocol's
acceptance), the probability that at least `K` units of the committed circuit are wrong at the committed values is at
most `miss_L(K) + ε_ks(σ) + δ_link`. The committed values are 1 on `ones`, 0 on `zeros` and the plurality elsewhere
(`Xpub`). The gaps are named hypotheses: `hExec`, `dp`, `hConst`, `hZero` and `vb` (`assumption-notes/e2e-checklist.md`);
`SHA512CRExpected` for `σ`'s link finders (`hCR`) is the assumption of record. -/
theorem flock_e2e_countL
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (Xpub ones zeros (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, h.2⟩) _ σ).trans
    (flock_batched_count_publicL P H E Lf plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht K σ hCR)

/-- **The end-to-end drawn-unit bound (skeleton).** If `execAccept` holds (an event that implies the Lean protocol's
acceptance), the probability that a drawn unit of the committed circuit is wrong at the committed values (`Xpub`, as in
`flock_e2e_countL`) is at most `ε_ks(σ) + δ_link`. -/
theorem flock_e2e_drawnL
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, h.2⟩) _ σ).trans
    (flock_batched_drawn_publicL P H E Lf plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht σ hCR)

end

/-! ## On the circuit of the statement's rows -/

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

variable {D Sl : Type} [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {L : Law n}

/-- **The end-to-end count curve on the circuit of the statement's rows.** `flock_e2e_countL` at `C = p.circuit outs`,
each unit its template's rows, with each drawn unit's place from `ProgPlaces`: if `execAccept` holds, the probability
that at least `K` units of the program are wrong at the committed values is at most `miss_L(K) + ε_ks(σ) + δ_link`. -/
theorem _root_.FlockSoundness.Prog.flock_e2e_countL
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL (p.partition outs) H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros (XplurL (p.partition outs) H E Lf plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBEL H E Lf plan k (reg σ) (cont σ) +
        (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  FlockSoundness.Discharge.Exec.flock_e2e_countL (p.partition outs) H E Lf plan tab hExec pp.derived ones hConst zeros hZero vb hk hRw hM hρ hr ht K σ hCR

/-- **The end-to-end drawn-unit bound on the circuit of the statement's rows.** `flock_e2e_drawnL` at
`C = p.circuit outs`, as in `Prog.flock_e2e_count`. -/
theorem _root_.FlockSoundness.Prog.flock_e2e_drawnL
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    (pp : ProgPlaces p outs plan tab)
    (ones : Finset (Fin N)) (hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones)
    (zeros : Finset (Fin N)) (hZero : (p.partition outs).ZeroCols plan tab pp.derived.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : (p.partition outs).ValueBinding plan tab ((p.partition outs).placeDecoder plan tab pp.derived.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL (p.partition outs) H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint ((p.partition outs).wrong
          (Xpub ones zeros (XplurL (p.partition outs) H E Lf plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBEL H E Lf plan k (reg σ) (cont σ) + (p.partition outs).linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  FlockSoundness.Discharge.Exec.flock_e2e_drawnL (p.partition outs) H E Lf plan tab hExec pp.derived ones hConst zeros hZero vb hk hRw hM hρ hr ht σ hCR

end FlockSoundness.Discharge.Exec
