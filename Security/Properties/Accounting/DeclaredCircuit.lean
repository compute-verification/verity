import Definitions.Accounting.Guarantees.DeclaredCircuit
import Properties.RemoteAudit.Schedule

/-!
# A declared circuit's space and bandwidth at the committed steering cap

`Accounting.Guarantees.DeclaredCircuitRequirement` (`G`) takes `Accounting.Assumptions.SteeredLayout`: each pass's layout
reads the codec and the append coins only through an `a`-bit summary. `ScheduledRequirement` is `G` with that premise
replaced by what the verifier checks: the schedule check (`RemoteAudit.Schedule.check`) at the steering cap the session
committed before its first coin (`audit.Params.schedule_cap`, `audit.Ledger.steering`) accepts each pass of the
session, under the named gap between the check and the model's layout, `Properties.RemoteAudit.ScheduleLayoutGap`. It is
stated here because it reads the verifier's code, which the spec doesn't; its theorem,
`Properties.DeclaredCircuit.SecurityProofs.scheduledRequirement` in
`Security/Proofs/Accounting/DeclaredCircuit/Scheduled.lean`, is `G`'s argument on the coins the check accepts, with the
layout steered there by `Properties.RemoteAudit.steered_of_checked`.
-/

namespace Properties.DeclaredCircuit

open Accounting
open Pous (pr)
open scoped ENNReal

/-- **A declared circuit's held and moved bits from an accepted stacked audit, at the committed steering cap.**

`G` (`Accounting.Guarantees.DeclaredCircuitRequirement`: the audit, the prover system, the circuit, the requirement and
the bound as it states them) with its premise `SteeredLayout layout steer` replaced by the named gap `ScheduleLayoutGap`
and the schedule check's acceptance at the cap `cap`, which is the event's first conjunct. `accepted c y`: at the prefix
`c` of the session's coins before a pass and the codec and append coins `y`, the check at `cap` over `blocks` registered
blocks accepts some schedule for a session whose appends are `appended c y` and whose pass `idx c` sent the runs
`sent c y`. The gap: the pass's layout is the verifier's positions `pos c` of those appends and runs. `cap` is the one
cap the session committed before its first coin, over which the check refuses a schedule, so on the coins the
check accepts the layout reads `y` only through the accepted schedule's `cap + 1` padded bits
(`Properties.RemoteAudit.steered_of_checked`).

On a session the check refuses at some pass the ledger credits nothing after the seed, so no shortfall against its
credit is counted there: the event, an accepted audit whose reads fall short of its credit, asks that the check accept
every pass of the session. The bound is `G`'s over the accepted sessions; a prover may run any layout on the coins the
check refuses.

The pass devices are told `steer c y`, any `a` bits, as in `G`. The layout's summary is the schedule's, which they are
told only through `steer`, so the union in `G`'s query term runs over both summaries: `2^(a + (cap + 1))` in place of
`G`'s `2^a`. At `a = 0`, devices told nothing past the session's coins before their pass, it is `G`'s bound at
`a = cap + 1`; devices told the schedule are told its bits in `steer`, at `a = cap + 1`, and pay them twice.

Frozen (no-function, hash): `G`'s game, its premises but `SteeredLayout`, `ScheduleLayoutGap` and
`RemoteAudit.Schedule.check`'s acceptance at the committed cap at every pass as the event's first conjunct. -/
def ScheduledRequirement : Prop :=
  ∀ (n P N ρ S q G Xr Lr qr τ k a L E es B εnum εden T M : ℕ) (ε δ : ℝ≥0∞) {ιW : Type} (Wc : Gates.IntDag ιW)
    (caps : Gates.Caps) (Aud : Gates.Audit) {Ω Ca Ch : Type} [Fintype Ω] [Fintype Ca]
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → RateSketch.Sector))
    (choose : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (layout : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Fin N ↪ Fin T × Fin M)
    (steer : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Pous.Bits a)
    (joined : RateSketch.StackedCoins Ω Ca T n P N ρ G → Fin T → ℕ)
    (A : RateSketch.SketchSummarySession a n P N 128 ρ 512 S (Lr * τ) q 16 G 2773)
    (blocks cap : ℕ) (idx : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → ℕ)
    (pos : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Array (ℕ × ℕ) → Array (ℕ × ℕ) → Fin N ↪ Fin T × Fin M)
    (appended sent : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Array (ℕ × ℕ))
    {V K Dv : Type} [DecidableEq V] [DecidableEq K] (W : Requirement.Wiring V)
    (p : RateSketch.StackedCoins Ω Ca T n P N ρ G → Requirement.Placement Dv V)
    (ℓ : RateSketch.StackedCoins Ω Ca T n P N ρ G → Requirement.Layout V)
    (grp : RateSketch.StackedCoins Ω Ca T n P N ρ G → V → K)
    (bind : RateSketch.StackedCoins Ω Ca T n P N ρ G → Fin n → Fin P → ℕ)
    (σ : RateSketch.StackedCoins Ω Ca T n P N ρ G → Requirement.Store)
    (plan : RateSketch.StackedCoins Ω Ca T n P N ρ G → Fin n → Fin P → Finset ℕ)
    (addr : RateSketch.StackedCoins Ω Ca T n P N ρ G → ℕ → Fin T × Fin M),
    7 ≤ n → 0 < P → 0 < N → 0 < E →
    Assumptions.SketchRank 16 128 12 ε →
    Assumptions.KeyBinding T M δ enc →
    Assumptions.SquaringFloor Gates.p1024 es εnum εden → Assumptions.ChainSum Gates.p1024 3 B es εnum εden →
    E ≤ 3 * Gates.p1024.squarings * es → εden ≤ 2 ^ 64 * εnum →
    (∀ s w η j pre, Assumptions.SketchHashBudget (((A.session s).window w η).device j pre) (qr * τ)) →
    Assumptions.CircuitCarriers (fun c y => A.session (steer c y)) (RateSketch.steeredStore enc choose layout) layout
      joined (Xr * τ) L B εnum εden Wc Aud →
    Properties.RemoteAudit.ScheduleLayoutGap pos appended sent layout →
    RateSketch.LifetimeCapped layout joined L →
    Gates.Restricted B Wc → Gates.Property caps Aud Wc → (∀ v < Aud.windows, Aud.reads v ≤ P * N * 4096) →
    (∀ x, (p x).Deadlines (bind x) τ ∧ (∀ w j w' j', w < w' ∨ w = w' ∧ j < j' → bind x w j < bind x w' j') ∧
      ∀ w j,
        let e := (p x).toExecution
        Requirement.StoreMatches W e (ℓ x) (σ x) 32768 (bind x w j) ∧
        Requirement.PlanMatches W e (ℓ x) (plan x w j) 32768 (bind x w j) ∧
        Requirement.OneBlockEach (ℓ x) (grp x)
          (Requirement.reads W e (bind x w j) \ Requirement.creditedReads W e (ℓ x) (bind x w j)) ∧
        Set.InjOn (addr x) ↑(plan x w j) ∧
        ∀ b ∈ plan x w j, addr x b ∈ Set.range (layout (RateSketch.SketchPrefix.of x.2 w j) x.1)) →
    let RL := ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      fun I => ∑ v ∈ I, caps.runBytes v
    let C := 8 * RL + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    let γ := RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768)
    let owned := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (t : ℕ) =>
      (Requirement.blocks (ℓ x) (Requirement.creditedReads W (p x).toExecution (ℓ x) t)).filter
        fun b => ((addr x b).1 : ℕ) ≠ 0
    let credit := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (t : ℕ) =>
      let e := (p x).toExecution
      max (Requirement.bitsAt W (ℓ x) (Requirement.credited W e (ℓ x) t) (owned x t))
        (Requirement.bits W ((Requirement.reads W e t).filter fun v => ((addr x ((ℓ x).block v)).1 : ℕ) ≠ 0) -
          ((Requirement.reads W e t \ Requirement.creditedReads W e (ℓ x) t).image (grp x)).card * 32768)
    let tiles := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (w : Fin n) (j : Fin P) =>
      Finset.univ.filter fun i : Fin N => layout (RateSketch.SketchPrefix.of x.2 w j) x.1 i ∈
        (owned x (bind x w j)).image (addr x)
    let store := RateSketch.programmed (RateSketch.steeredStore enc choose layout)
    let win := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (w : Fin n) =>
      A.win w x.2 fun j => steer (RateSketch.SketchPrefix.of x.2 w j) x.1
    let accepted := fun (c : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773) (y : Ω × (Fin T → Ca)) =>
      ∃ sc passes b, RemoteAudit.Schedule.check sc blocks cap passes (appended c y) = .ok b ∧
        (passes[idx c]?).map Prod.snd = some (sent c y)
    pr (fun x : RateSketch.StackedCoins Ω Ca T n P N ρ G =>
      (∀ w j, accepted (RateSketch.SketchPrefix.of x.2 w j) x.1) ∧
      (∀ w t, (x.2 w).2.2 t ∈ (win x w).correctOn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) : ℕ)) : ℝ) <
            ((∑ w, ∑ j, credit x (bind x w j) : ℕ) : ℝ) - (0.0015 + γ) * (n * P * N * 32768) ∨
          ∃ w, (((win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) : ℕ) : ℝ) <
            ((∑ j, credit x (bind x w j) : ℕ) : ℝ) - (0.01 + γ) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (n * P : ℝ≥0∞) * (2 ^ (a + (cap + 1)) * (((qr * τ : ℕ) : ℝ≥0∞) / 2 ^ 96) ^ (k + 1) + δ)

end Properties.DeclaredCircuit
