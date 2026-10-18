import Models.Accounting.Guarantees.TS.Stages
import Models.Accounting.Guarantees.Sheet
import Models.Accounting.Assumptions.Compose

/-!
# PoUW's credit on an arbitrary declared circuit (trusted)

THEOREM's composed statement (note:20261018T0346Z-draft-theorem-compose), each `def G : Prop`, proved (or a `sorry`
stub) as `Accounting.SecurityProofs.Compose.G` in `Proofs.Accounting.Compose`. The circuit is any `Declared` one, its
credited subcircuits compiled to their kinds' templates (`Scheme`), and nothing about its form is a hypothesis: the
premises are the data sheet's A1–A7, FOLD-OPT and FILL-OPT, the templates' noise (`TemplateNoise`), (I8) and, deployed,
the seeds' grinding (`SeedGrind`).

* `WorkWeighted`: a uniform ticket draws each unit with probability its work weight.
* `StageLuckOfTemplates`: units each lucky at most their template's receipt luck, all at most `p`, give `StageLuck p`.
* `TemplateLocal`: the local lemma, one per template and uniform over them: a unit whose skip and receipt read off its
  `b` fresh bits at fibre `Z` is lucky at most `Z·C(b, ⌊b/2⌋)/2^b` (`HMu.ReceiptLuckOfCoins` at its stage).
* `CompilerFloor`: Lemma C (`CompilerC`) against the floor credit, (2.7♭) with `ε_sample`.
* `StageIdealFloor`: `StageIdeal` against the floor credit, (2.7♭) itself at work-weighted draws.
* `CircuitCreditLive`: the theorem with live per-subcircuit coins.
* `CircuitCredit`: the theorem as deployed, with SHA-512 seeds, at `T·μ^m` more.

`ε = ε_♭ + ε_μ + ε_Λ` is NOTATION's, at the circuit's floor `♭ = Σ_j ♭_j`: `ε_Λ = κ·ln(1/δ_s)/(ln(1/p)·S)` is all that
sampling and luck cost (`ε_sample = 0` at work-weighted draws), and needs `p ≥ 1/e` (`StageBudget`'s range).
-/

namespace Accounting.Guarantees.Compose

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **A uniform ticket is a work-weighted draw.** Each unit's work weight is `≥ 0` and at most the probability that a
uniform ticket names it. -/
def WorkWeighted : Prop :=
  ∀ {κ K : Type} (Sc : Scheme κ K) (Cd : Declared κ) (u : UnitOf Sc Cd),
    0 ≤ workWeight Sc Cd u ∧ workWeight Sc Cd u ≤ prob (fun t : TicketOf Sc Cd => ticketUnit Sc Cd t = u)

/-- **Per-stage luck from the templates.** If every unit is lucky at most its template's receipt luck in every round,
and every unit's receipt luck is at most `p`, the window's per-stage coins are lucky at `p` (`StageLuck`). -/
def StageLuckOfTemplates : Prop :=
  ∀ {κ K C D : Type} [Fintype C] {M s n : ℕ} (Sc : Scheme κ K) (Cd : Declared κ)
    (W : StageWindow C (UnitOf Sc Cd) D M s n) (p : ℝ),
    (∀ u, luckOf Sc Cd u ≤ p) → TemplateLuck Sc Cd W → StageLuck p (stageOf Sc Cd) W

/-- **The local lemma, for every template.** A window whose units' skips and receipts read off their own fresh noise
(`TemplateNoise`) has every unit lucky at most `fibreOf u · C(b, ⌊b/2⌋)/2^b`, `b = bitsOf u`, at its stage, whatever
the past. -/
def TemplateLocal : Prop :=
  ∀ {κ K C D : Type} [Fintype C] {M s n : ℕ} (Sc : Scheme κ K) (Cd : Declared κ)
    (W : StageWindow C (UnitOf Sc Cd) D M s n),
    TemplateNoise Sc Cd W → TemplateLuck Sc Cd W

/-- **Lemma C against the floor credit.** As `CompilerC`, against the credit `T_cred = B_♭/f_max` of any labelled
program ♭ in place of honest's time: outside an event of probability at most `δ`, every accepted run satisfies
`T(M)/T_cred ≥ 1 − ε^♭_skip − ε^♭_cert − ε^♭_route − ε^♭_mand − ε^♭_jl − ε^♭_sample`. -/
def CompilerFloor : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (Fl : Labelled K) (Ω : Type) [Fintype Ω] (S : ℕ) (P : Play Ω S K T L U)
    (lam : R → ℝ) (Λ p δ δbind δzk : ℝ) (nπ : ℕ),
    IsCert lam → Z.Wf Fl → 0 < S → 0 < p → p < 1 → 0 < Λ →
    0 < δ - S * (δbind + nπ * δzk) → (∀ x, (P.run x).Split) →
    (∀ x, FacetCeilings Z.G (P.run x).run) → (∀ x, WordsWritten Z (P.run x)) → (∀ x, ScalarRoute Z (P.run x)) →
    (∀ x, SideUnits Z (P.run x)) → (∀ x, NoSum (P.run x)) → (∀ x, FoldOpt Z (P.run x)) →
    (∀ x, FillOpt Z (P.run x)) → RoundGame.BudgetAt Λ P.game → ProofsSound P Z.Nw δbind δzk nπ →
    prob (fun x => P.accepts x ∧ (P.run x).run.time / Z.credit Fl <
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
        Z.epsJl Fl lam p (δ - S * (δbind + nπ * δzk)) S -
        Z.epsSample Fl lam Λ p (δ - S * (δbind + nπ * δzk)) S) ≤ δ

/-- **D-ideal under per-stage coins, against the floor credit.** As `StageIdeal`, against the credit `T_cred = B_♭/f_max`
of any labelled program ♭: with work-weighted draws, outside an event of probability at most `δ`, every accepted run
satisfies `T(M)/T_cred ≥ 1 − ε^♭_skip − ε^♭_cert − ε^♭_route − ε^♭_mand − ε^♭_jl`, with no `ε_sample`. -/
def StageIdealFloor : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X] {C V D : Type} [Fintype C] [Fintype V] [Fintype D] {M s : ℕ}
    (Z : Model K R T L U F X) (Fl : Labelled K) (S : ℕ) (P : Play (Paths C M × (Fin s → D)) S K T L U)
    (W : StageWindow C V D M s S) (stg : V → ℕ) (tk : D → V) (w : V → ℝ) (lam : R → ℝ) (p δ δbind δzk : ℝ)
    (nπ : ℕ),
    IsCert lam → Z.Wf Fl → 0 < S → 2 ≤ s → Real.exp (-1) ≤ p → p < 1 →
    0 < δ - S * (δbind + nπ * δzk) → (∀ x, (P.run x).Split) →
    (∀ x, FacetCeilings Z.G (P.run x).run) → (∀ x, WordsWritten Z (P.run x)) → (∀ x, ScalarRoute Z (P.run x)) →
    (∀ x, SideUnits Z (P.run x)) → (∀ x, NoSum (P.run x)) → (∀ x, FoldOpt Z (P.run x)) →
    (∀ x, FillOpt Z (P.run x)) → P.game = W.game stg tk w → (∀ u, 0 ≤ w u) →
    (∀ u, w u ≤ prob (fun t => tk t = u)) → StageLuck p stg W → ProofsSound P Z.Nw δbind δzk nπ →
    prob (fun x => P.accepts x ∧ (P.run x).run.time / Z.credit Fl <
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
        Z.epsJl Fl lam p (δ - S * (δbind + nπ * δzk)) S) ≤ δ

/-- **The credit theorem, live coins.** For every declared circuit `Cd` compiled by scheme `Sc` and run on the device
of sheet `Sh`, a window of `S` rounds, each one run of the compiled circuit whose subcircuits' noise coins are released
in order after their operands are committed, then `s ≥ 2` work-weighted draws of whole units, each proved with its
noise's generation: under A1–A7, FOLD-OPT, FILL-OPT, the templates' noise and (I8) at `n_π = s`, the window is accepted
with time per round below `(1 − ε)·T_cred` with probability at most `δ_s + S·(δ_bind + s·δ_ZK)`, where
`T_cred = B(Σ_j ♭_j)/f_max` and `ε = ε_♭ + ε_μ + ε_Λ` at the circuit's floor. -/
def CircuitCreditLive : Prop :=
  ∀ {κ K P T L U F X C : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] [Fintype C] {s S : ℕ}
    (Sc : Scheme κ K) (Cd : Declared κ) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X)
    (Pl : Play (Paths C Cd.N × (Fin s → TicketOf Sc Cd)) S K T L U)
    (W : StageWindow C (UnitOf Sc Cd) (TicketOf Sc Cd) Cd.N s S) (lam : SheetRow P → ℝ) (p δs δbind δzk : ℝ),
    Fits Sc Cd Sh Z → Scope Sc Cd Z lam S s p δs → PlayOf Sc Cd Pl W → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) → (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOpt Z (Pl.run x)) → (∀ x, FillOpt Z (Pl.run x)) → TemplateNoise Sc Cd W →
    ProofsSound Pl Z.Nw δbind δzk s →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time < (1 - epsOf Sc Cd Z lam p δs S) * creditOf Sc Cd Z) ≤
      δs + S * (δbind + s * δzk)

/-- **The credit theorem, as deployed.** As `CircuitCreditLive` for the live play `Pl`, and a deployed play `Pd` whose
seeds are SHA-512 of the operands' roots, which grinds them at most `T·μ^m` past `Pl` at the theorem's threshold
(`SeedGrind`): `Pd` is accepted with time per round below `(1 − ε)·T_cred` with probability at most
`δ_s + S·(δ_bind + s·δ_ZK) + T·μ^m`. -/
def CircuitCredit : Prop :=
  ∀ {κ K P T L U F X C Ωd : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] [Fintype C] [Fintype Ωd] {s S : ℕ}
    (Sc : Scheme κ K) (Cd : Declared κ) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X)
    (Pl : Play (Paths C Cd.N × (Fin s → TicketOf Sc Cd)) S K T L U)
    (W : StageWindow C (UnitOf Sc Cd) (TicketOf Sc Cd) Cd.N s S) (Pd : Play Ωd S K T L U)
    (lam : SheetRow P → ℝ) (p δs δbind δzk μ : ℝ) (Tq m : ℕ),
    Fits Sc Cd Sh Z → Scope Sc Cd Z lam S s p δs → PlayOf Sc Cd Pl W → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) → (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOpt Z (Pl.run x)) → (∀ x, FillOpt Z (Pl.run x)) → TemplateNoise Sc Cd W →
    ProofsSound Pl Z.Nw δbind δzk s →
    SeedGrind Pd Pl ((1 - epsOf Sc Cd Z lam p δs S) * creditOf Sc Cd Z) Tq μ m →
    prob (fun x => Pd.accepts x ∧ (Pd.run x).run.time < (1 - epsOf Sc Cd Z lam p δs S) * creditOf Sc Cd Z) ≤
      δs + S * (δbind + s * δzk) + Tq * μ ^ m

end Accounting.Guarantees.Compose
