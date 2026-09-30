import FlockSoundness.Audit.FlockPublic

/-!
# The end-to-end theorem: a skeleton, every gap a named hypothesis

**The statement.** Except with the stated ε, whenever the audit accepts, the integrity profile holds for the committed
circuit and values, 1 on the constants and 0 on the zero, which are public:
* `flock_e2e_count`: the count curve, `Pr[accept ∧ at least K units wrong] ≤ miss_L(K) + ε_ks(σ) + δ_link`;
* `flock_e2e_drawn`: `Pr[accept ∧ a drawn unit wrong] ≤ ε_ks(σ) + δ_link`.

"Wrong" is measured on the program's Boolean circuit `CB`, as its circuit types flatten it, at the committed values. The
only cryptographic assumption is A2, SHA-512 expected-time collision resistance, for the link finders built from the
prover the theorem bounds (`hCR : LinkCR … (reg σ) (cont σ) …`), not from every prover at once.

**What the statement carries today.**
* "Accept" is `execAccept`, any event on the audit's outcome that implies the model's acceptance (`hExec`). So as pinned,
  these are the model-level theorems. The executable's verdict, `flock-verify` on the run's transcript, is not an event
  on that outcome. It enters when `hExec` is discharged, and that changes the statement's shape.
* "Rows derived by the verifier" is not in the statement either: `unitRows` appears in neither the conclusion nor
  `RowsL1`. S4's discharge of `dp` and `hL1` carries it.

**The composition.** It composes `flock_batched_count_placed` and `flock_batched_drawn_placed`, which are the compiled
audit, the link theorem and knowledge soundness, all on `main`. Each piece not yet proved is a named hypothesis. The
checklist `assumptions/e2e-checklist.md` gives each one's owner and how it is discharged:
* `hExec`: the executable verifier's acceptance implies the model's. This is the refinement lane's theorem;
* `dp : DerivedPlaces`: each drawn unit's place in its table, for the rows the verifier derives. This is 1d
  (`placement_compose`), 1e (the verifier folds `derive`'s rows) and S4 (the partition's units from the circuit types);
* `hL1 : RowsL1`: at values that carry 1 on the gates the units' constants read (`ones`), a unit wrong on the Boolean
  circuit is wrong on its rows. This is L1: `compose_sound` (#205, flat types; `Types.Dag.unit_sound`, type DAGs), with
  `flatten_eval` and W6 (S4). Both assume the constant at 1: at a constant of 0 a unit's rows hold with every output 0;
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

/-- **L1 for the partition's units** (named: `compose_sound` #205 and `Types.Dag.unit_sound`, `flatten_eval` and W6 in
S4). At values that carry 1 on `ones`, the gates the units' constants read, every unit wrong on the program's Boolean
circuit `CB` (partition `PB`, the same units) is wrong on its rows, the audit's circuit. `proj` reads the Boolean circuit's
wires off the rows' columns. -/
def RowsL1 {CB : Circuit} (PB : Partition CB n) (proj : (Fin C.N → Bool) → Fin CB.N → Bool) (ones : Finset (Fin C.N)) :
    Prop :=
  ∀ vals, (∀ g ∈ ones, vals g = true) → PB.wrong (proj vals) ⊆ P.wrong vals

/-- **The end-to-end count curve (skeleton).** If `execAccept` holds (an event that implies the model's acceptance), the
probability that at least `K` units of the committed circuit are wrong at the committed values is at most
`miss_L(K) + ε_ks(σ) + δ_link`. The committed values are 1 on `ones`, 0 on `zeros` and the plurality elsewhere
(`Xpub`). The gaps are named hypotheses: `hExec`, `dp`, `hL1`, `hConst`, `hZero` and `vb`
(`assumptions/e2e-checklist.md`); A2 for `σ`'s link finders (`hCR`) is the assumption of record. -/
theorem flock_e2e_count
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    {CB : Circuit} (PB : Partition CB n) (proj : (Fin C.N → Bool) → Fin CB.N → Bool) (ones : Finset (Fin C.N))
    (hL1 : P.RowsL1 PB proj ones) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        K ≤ (PB.wrong (proj (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw)))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, h.2.trans (Finset.card_le_card (hL1 _ fun _ hg => Xpub_of_mem_ones hg))⟩) _
    σ).trans
    (P.flock_batched_count_public H E plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht K σ hCR)

/-- **The end-to-end drawn-unit bound (skeleton).** If `execAccept` holds (an event that implies the model's
acceptance), the probability that a drawn unit of the committed circuit is wrong at the committed values (`Xpub`, as in
`flock_e2e_count`) is at most `ε_ks(σ) + δ_link`. -/
theorem flock_e2e_drawn
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    {CB : Circuit} (PB : Partition CB n) (proj : (Fin C.N → Bool) → Fin CB.N → Bool) (ones : Finset (Fin C.N))
    (hL1 : P.RowsL1 PB proj ones) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    {Dh : Type} {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab (P.placeDecoder plan tab dp.place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (PB.wrong (proj (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw)))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (prob_mono (fun o h => ⟨hExec o h.1, fun hd => h.2 (Finset.disjoint_of_subset_left
    (hL1 _ fun _ hg => Xpub_of_mem_ones hg) hd)⟩) _ σ).trans
    (P.flock_batched_drawn_public H E plan tab vb ones zeros (P.pubDecodes_of_place plan tab dp.place hConst hZero)
      (P.loweringSoundB_of_place plan tab dp.place) hk hRw hM hρ hr ht σ hCR)

end Partition

end FlockSoundness.Audit
