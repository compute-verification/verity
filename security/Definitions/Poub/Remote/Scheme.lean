module

public import Definitions.Core.Scheme
public import Definitions.Poub.Remote.Run

/-!
# PoUB's route `Paid` as a `Verity.Scheme`

A run of PoUB (`ResourceAccounting.Remote.Run`) read as a scheme, its route `Run.Paid` the scheme's charge exactly:

* the blocks are the run's receipts, `K` of them at most, each costing a word's `b` bits;
* `over` is the receipts committed, `committed · b` bits, which is what the verifier credits (it checks
  `committed · b ≥ C`); `under` is what the run paid for words, `Run.paid`: its reads, state held at the pass
  boundaries and bits rebuilt, inside the node and outside it;
* the incorrect blocks are the committed receipts that are not correct on the committed store. A `Run` counts them
  (`committed`, `correct`) and does not name them, so the scheme takes the set `bad` with `Run.Counted`, the named
  premise that its count covers the committed receipts that are not correct;
* the charges are the `P` passes' guessing slack, `P · λ`;
* the break event is `Run.Unpaid`, an accepted run whose correct receipts' words exceed what it paid plus the slack;
  `Run.Paid δ` bounds exactly its probability, so the scheme's own term is `Paid`'s `δ` (in the deployed game the hash
  term `n·P·G·(N/2^h + C(q,2)/2^h + 2^N/2^λ)`, `RemoteRunRoutes`).

The charge inequality then holds by arithmetic: an accepted run outside the break event pays for its correct receipts'
words, and every other committed receipt is in `bad` at `b` bits. The run states no condition on the verifier's log:
that the pass roots are recorded before the window's coins is in the game that builds the run's coins `Ω`, so `logOK`
is every run.

`PaidBoundsBreaks` and `PaidGetsBack` are the links from today's route to the instance, stated and not yet proved.
Today's statements (`Run.Paid`, `RemoteRunRoutes`, `RemoteRunConservationDeployed`) stay where they are.
-/

@[expose] public section

namespace ResourceAccounting.Remote

open scoped ENNReal

namespace Run

variable {τ : Type} [Fintype τ] {N : Node τ} {a : Audit} {Ω : Type} (r : Run N a Ω)

/-- **The incorrect receipts are counted** (a named premise): in every accepted run, the committed receipts are at
most the correct ones plus the receipts in `bad ω`, so `bad` names at least the committed receipts that are not
correct. -/
def Counted {K : ℕ} (bad : Ω → Finset (Fin K)) : Prop :=
  ∀ ω, r.accepts ω → r.committed ω ≤ r.correct ω + (bad ω).card

/-- **The run breaks `Paid`**: it is accepted and its correct receipts' words exceed what it paid for them plus the
`P` passes' slack `λ`. -/
def Unpaid (ω : Ω) : Prop :=
  r.accepts ω ∧ ¬ r.correct ω * a.b ≤ r.paid ω + a.P * a.lam

/-- **PoUB's route `Paid` as a scheme**: the receipts its blocks, `b` bits each; `over` the committed receipts' bits,
`under` the run's payments, the charges the passes' slack; `bad` its incorrect receipts, counted by `Counted`; the
break event `Unpaid`, whose bound `δ` is the scheme's own term. -/
def scheme {K : ℕ} (bad : Ω → Finset (Fin K)) (hbad : r.Counted bad) (δ : ℝ≥0∞) : Verity.Scheme where
  Block := Fin K
  cost _ := a.b
  charges := a.P * a.lam
  Run := Ω
  accepts := r.accepts
  over ω := r.committed ω * a.b
  under := r.paid
  logOK _ := True
  incorrect := bad
  breaks := r.Unpaid
  δscheme := δ
  charge ω hacc _ hnb := by
    have hp : r.correct ω * a.b ≤ r.paid ω + a.P * a.lam := by
      by_contra h
      exact hnb ⟨hacc, h⟩
    have hc : r.committed ω * a.b ≤ (r.correct ω + (bad ω).card) * a.b := Nat.mul_le_mul_right _ (hbad ω hacc)
    rw [Finset.sum_const, smul_eq_mul, add_mul] at *
    linarith

variable [Fintype Ω]

/-- The run's probability in the scheme's form: `Pouw.pr`, the uniform probability over the coins, in `ℝ≥0∞`. -/
noncomputable def prE (p : Ω → Prop) : ℝ≥0∞ := ENNReal.ofReal (Pouw.pr p)

end Run

/-- **`Paid` bounds the scheme's break event** (stated, not proved): a run that meets `Paid` at `δ` breaks it with
probability at most `δ`, the scheme's own term. -/
def PaidBoundsBreaks : Prop :=
  ∀ {τ : Type} [Fintype τ] {N : Node τ} {a : Audit} {Ω : Type} [Fintype Ω] (r : Run N a Ω) (δ : ℝ),
    r.Paid δ → Run.prE r.Unpaid ≤ ENNReal.ofReal δ

/-- **Today's route gives the instance's `GetsBack`** (stated, not proved): under the accounting layer's theorem, a
run that meets `Paid` at `δ`, with its incorrect receipts counted by `bad` and the audit's bound at those receipts,
gets back at every cutoff `δ'`: an accepted run's committed bits exceed its payments by more than the missable cost
plus `P · λ` with probability at most `δ' + ε + δ`. -/
def PaidGetsBack : Prop :=
  ∀ {τ : Type} [Fintype τ] {N : Node τ} {a : Audit} {Ω : Type} [Fintype Ω] (r : Run N a Ω) {K : ℕ}
    (bad : Ω → Finset (Fin K)) (hbad : r.Counted bad) (L : Verity.SamplingDesign (Fin K)) (δ : ℝ) (ε : ℝ≥0∞),
    Verity.Scheme.AccountingLayer → r.Paid δ → Verity.AuditBound L (Run.prE (Ω := Ω)) r.accepts bad ε →
      ∀ δ', (r.scheme bad hbad (ENNReal.ofReal δ)).GetsBack (fun _ => L) (Run.prE (Ω := Ω)) δ' ε

end ResourceAccounting.Remote
