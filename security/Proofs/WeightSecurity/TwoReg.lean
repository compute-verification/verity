module

public import Definitions.Claims.SamplingTwoReg
public import Proofs.WeightSecurity.Two

/-!
# A period of two-stage windows at the committed traces

The period over windows each audited by SPOT's two-stage verifier, in the form `period_undetected` takes. A window's
record reads the committed traces (`Claims.SamplingTwoReg`): its sample holds an incorrect proof block when it holds one
of the committed incorrect set (`CompiledTwo.FregTwo`), and it counts the replay blocks wrong at the committed boundary
trace (`wrongAtBoundaryAt` at `Compiled.regVal`), which the statement and prover its strategy sends fix before the run's
coins. `twoRec` counts at the extracted trace instead, which the extractor outputs after the coins, so its windows'
misses don't multiply and `two_undetected` adds a cap per window.

* `regCount`: that count at a window's statement and prover, `0` out of the two-stage scope.
* `twoRegRec`, `twoRegRun`: the record, and the deployment's run over `T` windows recorded by it.
* `PeriodTwoReg`: the run leaves `M` incorrect replay blocks undetected with probability at most
  `Σ_t ε_t + (1 − r₁·r₂)^M`, each `ε_t` bounding window `t`'s `ε_reg,2` at every strategy the run plays it with. It is
  `period_undetected` at `twoRegRec`: `hacc` is `CompiledTwo.KSRegTwo`'s first conjunct and `hstage` is
  `CompiledTwo.MissTwo`.
* `CompilerPeriodTwoReg`: `PeriodTwoReg` at every run of window verifiers that meet `CompilerKSTwo`'s premises, at the
  two stages' rates.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open FlockSoundness.Game hiding Msg
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

open Classical in
/-- **The incorrect replay blocks a window fixes**: at the statement `x` and the prover `P` its strategy sends, the replay
blocks wrong at the committed boundary trace (`wrongAtBoundaryAt` at `Compiled.regVal`); `0` on a statement out of the
two-stage scope. -/
noncomputable def regCount (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) (P : Prover Unit) : ℕ :=
  match L.parse x with
  | none => 0
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d =>
      if InScopeTwo Pt d s then (wrongAtBoundaryAt s Pt d.part (L.regVal H P x ())).card else 0
    | _, _ => 0

/-- **A two-stage window's record at the committed traces**: accepted when the run accepts; sampled incorrect when its
sample holds a proof block of the committed incorrect set; incorrect, the count `n` its strategy fixes; and the
release. -/
noncomputable def twoRegRec (H : Msg → Msg) (L : CompiledTwo Coin A) (n : ℕ)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) : Served O V Tick Adv where
  accepted := b.2.accepted H
  sampledIncorrect := ∃ u ∈ L.checked b.2.1 b.2.2.2, u ∈ L.FregTwo H b.2.2.1 b.2.1 () b.2.2.2
  incorrect := n
  egress := b.1.1
  tick := b.1.2.1
  adv := b.1.2.2

/-- **The deployment's run over `T` two-stage windows** at the datacenter's strategy `s`, each window recorded by
`twoRegRec` at the count its strategy fixes (`regCount`). -/
noncomputable def twoRegRun (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (T : ℕ)
    (s : Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)) :
    Verity.Game (List (Served O V Tick Adv)) :=
  periodRun (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => twoRegRec H (Lv t) (regCount H (Lv t) τ.2.1.1 τ.2.1.2)) 0 T s

/-- **The period at the committed traces**: over `T` two-stage windows, at every strategy whose windows' committed-form
errors (`CompiledTwo.εregTwo` at the statement and prover each sends) are at most `ε_t`, the run leaves `M` incorrect
replay blocks undetected with probability at most `Σ_t ε_t + (1 − r₁·r₂)^M`. -/
def PeriodTwoReg (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (r₁ r₂ : ℝ) : Prop :=
  ∀ (T : ℕ) (ε : ℕ → ℝ) (s : Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)),
    (∀ t τ, PeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ →
      (Lv t).εregTwo H τ.2.1.1 τ.2.1.2 ≤ ε t) →
    ∀ M : ℕ, Verity.Game.value (Served.Undetected M) (twoRegRun H Lv T s) ≤
      ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) + ENNReal.ofReal ((1 - r₁ * r₂) ^ M)

/-- **The period bound from the windows' verifiers**: when every window's verifier meets `CompilerKSTwo`'s premises and
every statement in its two-stage scope has a first stage that catches each replay block at rate `r₁` and a second stage
that samples each cell at rate `r₂`, `PeriodTwoReg` holds (`period_undetected` at `twoRegRec`, from `CompilerKSRegTwo`
and `CompilerMissTwo`). -/
def CompilerPeriodTwoReg : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (O V Tick Adv : Type)
    (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (r₁ r₂ : ℝ), A.KS H →
    (∀ t, (Lv t).commit.Binds H ∧ (Lv t).Sessions H) → 0 ≤ r₁ → r₁ ≤ 1 → 0 ≤ r₂ → r₂ ≤ 1 →
    (∀ t x s Pt d, (Lv t).parse x = some s → (Lv t).partition s = some Pt → (Lv t).two s = some d →
      InScopeTwo Pt d s → d.FirstRate r₁ ∧ d.SecondRate Pt r₂) →
    PeriodTwoReg (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv r₁ r₂

end WeightSecurity.SecurityProofs
