module

public import Proofs.Flock.SamplingGeneric.TwoRegFirstHit
public import Proofs.Flock.SamplingGeneric.TwoRegX
public import Proofs.Flock.SamplingGeneric.TwoRegMiss
public import Proofs.Flock.SamplingGeneric.TwoCap
import Proofs.Flock.Claims.Prob

/-!
# The two-stage committed form at the first-hit error

`CompiledTwo.KSRegTwoX` bounds a two-stage run that accepts with a sampled proof block incorrect at the boundary trace
read at the statement by `ε_reg,2,X`, which has two square roots over the program's gates. `CompiledTwo.KSFirstHitTwoX`
is the same three statements (the event, the threshold form and the headline) at `ε_first,2,X`, which has none:

`ε_first,2,X(c_R, c_F) = κ_s + √(k·γ_s·Pr[B]) + (2·c_R + 4·c_F)/2^256`,

a function of two costs in calls of `H`: `c_R`, a run's, and `c_F`, a first-hit finder's. The form fixes neither cost.
It takes them with what makes them costs, three bounds on masses of one prover's runs and their extractions, each a
named `Prop`, so that a guarantee that takes them names them in its record:

* `BconfXLe`, `Pr[B_conf,X] ≤ 2·c_R/2^256`: two runs' extractions open a gate two ways under the statement's roots;
* `OffRegXLe`, `Pr[acc ∧ off the boundary trace] ≤ 2·c_F/2^256`: an accepted run's extractions open a gate away from
  the boundary trace read at the statement (`Compiled.OffRegX`);
* `OffIregLe`, `Pr[acc ∧ off a committed interior] ≤ 2·c_F/2^256`: they open one away from its replay block's committed
  interior (`CompiledTwo.OffIreg`).

What it concludes at one statement, prover and two costs is `FirstHitTwoXAt`. So the first-hit term is a parameter: a
lemma that bounds the two off-trace masses by a finder's expected calls over `2^256` gives `c_F`, whatever the finder
counts by, and nothing here reads a finder. `ksFirstHitTwoX` proves the form
at every compiled two-stage verifier whose proof system is knowledge sound, whose commitment binds and that meets the
two-stage contract; it takes no premise on `H`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition InScopeTwo footprintAt wrongAllAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The first-hit error at the boundary trace read at the statement**, a function of two costs in calls of `H`:
`κ_s + √(k·γ_s·Pr[B]) + (2·c_R + 4·c_F)/2^256`, `c_R` a run's and `c_F` a first-hit finder's. -/
noncomputable def εfirstTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (cR cF : ℝ) : ℝ :=
  L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (2 * cR + 4 * cF) / 2 ^ (256 : ℝ)

/-- **`B_conf,X` finds a collision with probability at most `2·c_R/2^256`**: two runs' extractions open a gate two ways
under the statement's roots that often at most. The first of the three bounds the form takes, the one that makes
`c_R` a cost. A named `Prop`, so a statement that takes it names it. -/
def BconfXLe (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (cR : ℝ) : Prop :=
  (Compiled.BconfX H L.toCompiled Ω P x).Pr H ≤ 2 * cR / 2 ^ (256 : ℝ)

/-- **An accepted run is off the boundary trace read at the statement with probability at most `2·c_F/2^256`**: its
extractions open a gate away from that trace (`Compiled.OffRegX`), over the prover's tape, the run's coins and the
extractor's. The second bound the form takes. -/
def OffRegXLe (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (cF : ℝ) : Prop :=
  Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
    L.OffRegX H P x q.1 q.2.1 q.2.2) ≤ 2 * cF / 2 ^ (256 : ℝ)

/-- **An accepted run is off a committed interior with probability at most `2·c_F/2^256`**: its extractions open a gate
away from the committed interior of the replay block of `Pρ` that holds it (`CompiledTwo.OffIreg`), over the prover's
tape, the run's coins and the extractor's. The third bound the form takes. -/
def OffIregLe (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (Pρ : Partition) (cF : ℝ) : Prop :=
  Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
    L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤ 2 * cF / 2 ^ (256 : ℝ)

/-- **What the committed form at the first-hit error concludes**, at a statement `x` that parses to `s` with the
partition `Pt` and the two-stage design `d`, one prover and two costs, with `ε = εfirstTwoX` at `c_R` and `c_F`:

* `Pr[acc ∧ a sampled proof block is in FregTwoX] ≤ ε`;
* at every `δ ≥ 0`, `Pr[acc ∧ no interiors put the footprint at the boundary trace read at the statement in M̄_δ]`
  is at most `δ + ε`;
* `Pr[acc] ≤ p̄ + ε` at every `p̄` that bounds `P̄_a` at that footprint with any interiors. -/
def FirstHitTwoXAt (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) (s : AuditStatement) (Pt : Partition)
    (d : Sampling.Design.Two) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (cR cF : ℝ) : Prop :=
  Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤ L.εfirstTwoX H x P cR cF ∧
    (∀ δ : ℝ, 0 ≤ δ →
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∀ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part (L.regValX H P x) ι ∉ d.M Pt δ) ≤
        δ + L.εfirstTwoX H x P cR cF) ∧
    ∀ pb : ℝ, (∀ ι, d.Pa Pt (footprintAt s Pt d.part (L.regValX H P x) ι) ≤ pb) →
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc) ≤
        pb + L.εfirstTwoX H x P cR cF

/-- **The two-stage committed form at the first-hit error.** At a statement in scope, every prover and every two costs
`c_R` and `c_F`: if `Pr[B_conf,X] ≤ 2·c_R/2^256` (`BconfXLe`), and an accepted run's extractions open a gate away from
the boundary trace read at the statement with probability at most `2·c_F/2^256` (`OffRegXLe`), and away from its
replay block's committed interior with probability at most `2·c_F/2^256` (`OffIregLe`), then the three statements of
`FirstHitTwoXAt` hold at `ε = εfirstTwoX` at `c_R` and `c_F`. -/
def KSFirstHitTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  ∀ x s Pt d, L.Scope x → L.parse x = some s → L.partition s = some Pt → L.two s = some d →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (cR cF : ℝ),
      L.BconfXLe H x P cR → L.OffRegXLe H x P cF → L.OffIregLe H x P d.part cF →
        L.FirstHitTwoXAt H x s Pt d P cR cF

/-- **A compiled two-stage verifier meets the committed form at the first-hit error**: where its proof system is
knowledge sound, its commitment binds and it meets the two-stage contract, with no premise on `H`. The event is
`eventTwoX_le_of_off` at the two off-trace bounds and `Pr[B_conf,X]` at its own; the threshold form and the headline
follow from the event as they do at `ε_reg,2,X` (`threshold_of_event`, `headline_of_event`). -/
theorem ksFirstHitTwoX {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) : L.KSFirstHitTwoX H := by
  intro x s Pt d hx hs hPt hd Ω _ _ P cR cF hR hB hN
  have hin : InScopeTwo Pt d s := L.inScopeTwo_of hx hs hPt hd
  have hev : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤ L.εfirstTwoX H x P cR cF := by
    have hR' : (Compiled.BconfX H L.toCompiled Ω P x).Pr H ≤ 2 * cR / 2 ^ (256 : ℝ) := hR
    have h := eventTwoX_le_of_off hA hbind hS hs hPt hd hin P (bB := 2 * cF / 2 ^ (256 : ℝ))
      (bN := 2 * cF / 2 ^ (256 : ℝ)) hB hN
    have e : (2 * cR + 4 * cF) / 2 ^ (256 : ℝ) =
        2 * cR / 2 ^ (256 : ℝ) + 2 * cF / 2 ^ (256 : ℝ) + 2 * cF / 2 ^ (256 : ℝ) := by ring
    unfold εfirstTwoX
    linarith
  have hev' := hev
  simp only [FregTwoX_eq H L P hs hPt hd] at hev'
  unfold FirstHitTwoXAt
  refine ⟨hev, fun δ hδ => ?_, fun pb hpb => ?_⟩
  · exact threshold_of_event H L hS hs hPt hd hin P (fun _ => L.regValX H P x) hev' hδ
  · exact (headline_of_event H L hS hs hPt hd hin P (fun _ => L.regValX H P x) hev' (fun _ => pb)
      fun _ ι => hpb ι).trans_eq (by rw [Ex_const])

end CompiledTwo

end Claims.SamplingGeneric
