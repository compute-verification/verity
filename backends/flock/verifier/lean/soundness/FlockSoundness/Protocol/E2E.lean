import FlockSoundness.Protocol.Audit.FlockPublic

/-!
# The end-to-end theorem: a skeleton, every gap a named hypothesis

**The statement.** Except with the stated ε, whenever the audit accepts, the integrity profile holds for the committed
circuit and values, 1 on the constants and 0 on the zero, which are public:
* `flock_e2e_count`: the count curve, `Pr[accept ∧ at least K units wrong] ≤ miss_L(K) + ε_ks(σ) + δ_link`;
* `flock_e2e_drawn`: `Pr[accept ∧ a drawn unit wrong] ≤ ε_ks(σ) + δ_link`.

"Wrong" is measured on the audit's circuit `C`, at the committed values: the circuit the statement's rows make, each unit
its template's rows (`Prog.circuit`; Daniel, Sep 30: the circuit is chosen on the untrusted side, and the lowering does
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
audit, the link theorem and knowledge soundness, all on `main`. Each piece not yet proved is a named hypothesis. The
checklist `assumptions/e2e-checklist.md` gives each one's owner and how it is discharged:
* `hExec`: the executable verifier's acceptance implies the model's. This is the refinement lane's theorem;
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

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

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

end Partition

/-! ## On the circuit of the statement's rows -/

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **Each drawn unit's place in its table, for the program's circuit** (named). Unit `u` is an instance of its rows
`p.rows u` (`inst`; `Prog.isRowsUnit` gives one), and the statement of the table it is drawn into holds those rows in
block `o` at the columns `col`:
* `placed`: the statement's level-0 matrices hold `p.rows u` there. From the executable's accept step,
  `placement_of_setupH` gives it for the rows `Rows.ofNet` reads off setupH's net; that those rows are the program's
  unit's (W6) is open;
* `aliased`: the unit's input columns on one gate carry one value in every satisfying witness (`Layout.Aliased`, which
  `unitPlace_of_setupH` takes as a hypothesis too). -/
structure ProgPlaces where
  inst : ∀ u, (p.partition outs).IsRowsUnit u (p.rows u)
  o : ∀ S R u, u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl)
  col : ∀ S R u, u ∈ S → Fin (p.rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog)
  placed : ∀ S R u (h : u ∈ S), Placement (Partition.tabOf plan tab S R u).S (p.rows u) (col S R u h)
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

end FlockSoundness.Audit
