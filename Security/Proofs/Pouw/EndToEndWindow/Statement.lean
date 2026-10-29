import Definitions.Pouw.Game.Params
import Definitions.Pouw.Assumptions.Game
import Definitions.Pouw.Assumptions.NCP.Assumptions
import Proofs.Flock.Soundness.Audit.Extraction
import Proofs.Flock.Soundness.Audit.RegDraw
import Proofs.Flock.Soundness.Audit.Window

/-!
# PoUW's work bound at the window audit (trusted)

`EndToEndWindow` is `Pouw.Guarantees.EndToEnd` with the window audit in place of its ideal audit, at the service's draw
and decision 107's units, for the window's share `ε` (`0 < ε ≤ 1`, the `ε` of `window.sampling`: main serves 0.1% at
`δ = 2^-40` and 1% at `δ = 2^-128`). Its cost `T` is the cost of the prover that is audited:

* **One prover, costed.** A prover is a program of a prover model (`WindowWork.ProverModel`). Its play is its whole
  strategy in the window audit, the registration and then the sessions, and the model charges its online phase, from
  the salt's release to the registration's receipt, at most `T`. `q` counts every oracle call: preprocessing's, the
  registration's, the commitments' (`ProverModel.RegQueries` ties the count to the program's queries), and the
  sessions' (`ProverModel.sessCalls`).
* **PoUW's adversary is extracted from the registration** (`WindowWork.StraightLineExtraction`, the one random-oracle
  premise): from every prover within `(T, q)`, an adversary of PoUW's game within `(T + T_ext, q + q_ext)` whose
  transcript is the transcript the registration's commitments bind (`Ccom`) on every run received before the draw
  coin's cutoff, except with probability `ε_id(q, N)` over the oracle and the salt. The extraction reads none of the
  sessions' queries, and a session that opens the registered roots to anything but `Ccom` is the link term inside
  `WindowBound`'s `β`. `q` counts the sessions' calls too (`ProverModel.sessCalls`), so that `β`'s state terms, which
  a session state holding a collision of `H` makes about 1, can be bounded on average over the oracle by the counted
  queries on the random-oracle claim, as `ε_id` is. That bound holds only where the sessions are one oracle program's
  run at `H`, counted by that program's calls. `sessCalls` is a model field, not yet tied to an oracle program (see
  `ProverModel`), so no card reads a number for `E[β]` from it until that tie is stated.
* **The acceptance is the service's at the draw coin's cutoff** (`WindowWork.AcceptsBy`): the sessions accept and the
  challenger received the registration before it released the epoch's draw coin, in its record's order (a window
  received after the draw coin is refused, `late-registration`). The bound is that order, not a time. The wall-clock
  deadline (`Check.registration_deadline`, counted from the salt coin's release) is the challenger's, which waits on
  its record's `released_at`; what it refuses as late is refused on top of the cutoff, so the service accepts only
  runs inside the event. The extraction is claimed only on runs received before the cutoff.
* **The work step** is `WindowWork.TileWork γ_u`, at decision 107's coordinate units (`coordDomain`).
* **The window audit's guarantee is a hypothesis** (`WindowWork.WindowBound`): at each oracle and salt, the audit accepts
  the prover's play while the incorrect work of the registration's transcript is at least the share `ε` of `W` with
  probability at most `β(H, s)`, a function of the oracle, the salt and the prover. The conclusion averages it.

It is a statement with no proof yet; its proof is to be
`Pouw.SecurityProofs.EndToEndWindow : Pouw.Guarantees.EndToEndWindow`.

**Units.** Under decision 107 a PoUW unit is one output coordinate, shape `(1, k, 1)`, a `K`-chunked s8 dot with s32
running sums, and one template instance of the window (`WindowWork.coordinates`, `WindowWork.coordDomain`). There
`WindowWork.Realizes` holds by construction (`WindowWork.realizes_of_coordinates`, proved from
`WindowWork.CoordinateInstances`). Larger units can't be audit units: an `r × c` tile breaks the 16-bit output rule, and
a GEMM unit spans audit units the closures don't tie together.

**The work step's instance.** `TileWork` is stated at `coordDomain` only, so TT's instances, whose carded domains
(`Shape.InDefaultDomain`, `m1Domain`, `NCP.leadDomain`) all have `16 ∣ m`, `m > 0` and `n ≥ 2^10` and hold no coordinate
layout, are no instances of it. `WindowWork.tileWork_metered` exhibits one: the metered model (`meteredModel P γ`), a
device that pays `(1 − γ)·W_ref(u)` for every coordinate it commits, satisfies `TileWork` at `γ_u = γ` with error `0`,
for every protocol and every `γ ≤ 1`, on a domain that holds a workload (`coordDomain_inhabited`). Its programs reach
the oracle only through counted queries (`meteredModel_queries`). The deployed device's instance is EPS1's: compute
accounting's per-unit credit at its `ε` (`Accounting.Guarantees.Compose.CircuitCredit`, on `cursor/compose-stmt-ea24`, not
on main), through a bridge from its play to PoUW's cost model, which is PoUW's gap (3) and is stated when that module
lands.

**The window's law** is the service's (decision 102, `window.py`): one uniform `K′`-subset of the `n` padded units
(`subset:K′`, `Law.subset`), the `N_real` real tiles padded with fillers to `pad(N_real + 1)`, drawn after the receipt,
each drawn unit proved alone (no closures). `WindowWork.WindowBoundOfService` discharges `WindowBound` there, its first
term the subset law's own escape, which is 0 where `K′ = N`. PoUW's PROTOCOL.md (Lifecycle, step 5) describes
`Sampled(t)`, `t` independent work-weighted draws with replacement, not the code's draw.

Gap (6)'s `Pouw.Guarantees.Window.WindowEps` (branch `cursor/pouw-window-eps-stmt2-c7a9`) bounds the same event in
gap (6)'s copy of the game (`Pouw.Window`), with the window's knowledge and link terms composed from its calls'. The
bridge that makes it a discharge of `WindowBound` is named there (`Pouw.Window.FlockBridge`).

**Pearl-C is out of scope**: it has its own game and premises (`TTOutTile`, `TileProofSound`,
`Pouw.Guarantees.PearlCSampledByDeadline`).

C-Flock's audit definitions are still in the proofs package, so this module imports them, as `Flock.Guarantees.EndToEnd`'s
statement does C-Flock's; `Models.lean`, which the `security` package builds without `security_proofs`, does not import
it. `TileWork` and `StraightLineExtraction` are named premises awaiting their card.
-/

namespace Pouw.WindowWork

open Finset FlockSoundness.Audit
open scoped ENNReal

variable {Q R S : Type}

/-- **The incorrect work** of a transcript `τ` at the oracle `H` and the salt `s`: the declared work of the units that
are not correct in it, `∑_{u < N, u ∉ K(τ)} W_ref(u)`. -/
noncomputable def incorrectWork (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Pouw.Transcript) : ℝ :=
  ∑ u ∈ range U.layout.N \ correctSet P U H s τ, (P.Wref U.layout u : ℝ)

/-- **Coordinate units** (decision 107): every unit is one output coordinate, shape `(1, k, 1)`: one row of `A`
against one column of `B`, a `K`-chunked s8 dot with s32 running sums. -/
def coordinates (L : Layout) : Prop := ∀ u < L.N, (L.shape u).m = 1 ∧ (L.shape u).n = 1

/-- **Decision 107's domain**: coordinate units, each a dot of positive length `k` (a whole dot, or a `K`-chunk of
2048 committing its s32 running sum). -/
def coordDomain (L : Layout) : Prop := coordinates L ∧ ∀ u < L.N, 0 < (L.shape u).k

/-- **The work step at coordinate units, TileWork(γ_u)** (a named premise, open), for a cost model `CM`, a protocol `P`,
a gap `γ_u` and an error `η(q, N)`, on decision 107's domain `coordDomain`. A correct unit was paid for: the prover
spent at least `(1 − γ_u)·W_ref(u)` on it. A cost model prices a whole run, not a unit, so it is stated summed over the
window, in `G_γ`'s game (the same adversaries, budgets and oracle-call bounds). For every adversary within `(T, q)`,
$$\Pr_{H,s}\Big[T < (1-\gamma_u)\sum_{u \in K} W_{ref}(u)\Big] \le \eta(q, N),$$
`K` being the correct units (`correctSet`: checked values and useful output correct). It is `G_γ` at `γ = γ_u` with
the gap multiplied out, so it needs no `γ_u < 1`: at `γ_u ≥ 1` its event is empty.

It is stated at coordinates only, so no tile-unit premise (TT, TT_NCP) instantiates it: none of their carded domains
holds a coordinate layout. Its instance in Lean is the metered model (`tileWork_metered`, at `γ_u = γ` with error `0`,
on a domain that holds a workload, `coordDomain_inhabited`). The deployed device's is EPS1's, from compute accounting's
per-unit credit (`Accounting.Guarantees.Compose.CircuitCredit`) through gap (3)'s bridge to PoUW's cost model. -/
def TileWork [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (γu : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain coordDomain → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM,
    A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T < (1 - γu) * ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), (P.Wref U.layout u : ℝ))
      ≤ η q U.layout.N

/-- **The metered model at `γ`**, `TileWork`'s instance: its programs are adaptive oracle programs (one for each
workload, activation input, state and salt), a run outputs what its program returns, makes the program's queries, and
costs `(1 − γ)·W_ref(u)` for every unit `u < N` it commits. It is a device that pays each committed coordinate's
declared work at the discount `γ` and can't commit one for less. -/
noncomputable def meteredModel (P : Protocol Q R S) (γ : ℝ) : CostModel Q R S where
  Prog := Workload → (ℕ → Pouw.Mat) → Pouw.State → S → QTree Q R Pouw.Transcript
  run p H U act st s := (p U act st s).eval H
  cost p H U act st s := (1 - γ) * ∑ u ∈ (range U.layout.N).filter (fun u => ((p U act st s).eval H u).isSome),
    (P.Wref U.layout u : ℝ)
  calls p H U act st s := (p U act st s).calls H

/-- The metered model's programs reach the oracle only through counted queries (the form of `CostModel.Queries`). -/
theorem meteredModel_queries (P : Protocol Q R S) (γ : ℝ) : (meteredModel P γ).Queries :=
  fun p U act st s => ⟨p U act st s, fun _ => ⟨rfl, rfl⟩⟩

/-- **`TileWork` holds at the metered model**, at `γ_u = γ` with error `0`, for every protocol and every `γ ≤ 1`: a
correct unit is a committed one, so the work of the correct units at the discount is at most what the run costs. -/
theorem tileWork_metered [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) {γ : ℝ}
    (hγ : γ ≤ 1) : TileWork (meteredModel P γ) P γ fun _ _ => 0 := by
  intro U _ T _ q _ A hA
  have hempty : ∀ x : (Q → R) × S,
      ¬ T < (1 - γ) * ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), (P.Wref U.layout u : ℝ) := by
    intro x hlt
    have hsub : correctSet P U x.1 x.2 (A.transcript U x.1 x.2) ⊆
        (range U.layout.N).filter (fun u => ((A.online U A.act (A.pre.eval x.1) x.2).eval x.1 u).isSome) := by
      intro u hu
      simp only [correctSet, Finset.mem_filter] at hu
      obtain ⟨hu, c, hc, -⟩ := hu
      refine Finset.mem_filter.2 ⟨hu, ?_⟩
      change (A.transcript U x.1 x.2 u).isSome
      rw [hc]
      rfl
    have hle := Finset.sum_le_sum_of_subset_of_nonneg hsub fun u _ _ => (Nat.cast_nonneg (P.Wref U.layout u) : (0 : ℝ) ≤ _)
    have hcost := (hA x.1 x.2).1
    have : (1 - γ) * ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), (P.Wref U.layout u : ℝ) ≤ T :=
      (mul_le_mul_of_nonneg_left hle (sub_nonneg.2 hγ)).trans hcost
    exact absurd hlt (not_lt.2 this)
  simp only [pr, hempty, Finset.filter_false, Finset.card_empty, Nat.cast_zero, zero_div, le_refl]

/-- **Decision 107's domain holds a workload**: one coordinate unit `(1, 4096, 1)` with zero weights. -/
theorem coordDomain_inhabited : ∃ U : Workload, U.InDomain coordDomain ∧ U.layout.N = 1 := by
  refine ⟨⟨⟨1, fun _ => ⟨1, 4096, 1⟩, fun _ => 0⟩, fun _ _ _ => 0⟩, ⟨by norm_num, ⟨fun _ _ => ⟨rfl, rfl⟩,
    fun _ _ => by norm_num⟩, fun _ _ _ _ _ _ => ⟨by norm_num, by norm_num⟩⟩, rfl⟩

/-- **A window prover's model**: its programs, and for each a whole strategy in the window audit at every oracle and
salt (`play`: the registration, then the sessions on every draw), with what the model charges for it. `cost` is the
work of its online phase, from the salt's release to the registration's receipt, in PoUW's cost units; `received` and
`cutoff` are places in the challenger's record of the epoch's coins, which logs each root's receipt and then its coin's
release (`LiveRecord.log`; that the challenger keeps that order is `trusted-host`'s): the registration's
receipt, and the release of the epoch's draw coin (`pouw/draw`); `calls` is the registration's oracle calls, the
commitments', which `RegQueries` ties to the queries of the program that computes it; `sessCalls` is the sessions'
oracle calls, the most over every draw and every coin of the sessions. The extraction reads none of the sessions'
queries, but `Bounded` counts them: a session state that holds a collision of `H` makes the window's state terms about
1, and only a count of the queries that found it bounds them on average over the oracle. The model fixes all five, as it
fixes the cost; a prover only picks a program.
A session's strategy is a strategy of the session at `H`, whose type depends on `H`, so no oracle program's output is
one. `sessCalls` is free: the model picks a separate session play at each oracle and its count, and nothing ties the
count to the play's queries. So a bound on the sessions' state terms at counted queries (the random-oracle claim's,
through `Pouw.Window.CollCounted`) holds only for a model whose sessions are one oracle program's run at `H`,
counted by that program's calls:
* at a true count it is false: a model whose session play at each `H` holds a hardcoded collision of `H` counts 0
  session calls, and its `β` is about 1 at every `H`;
* at a count that isn't a count of calls it is vacuous: a model can report a count large enough that the bound is at
  least 1 exactly where its play holds a collision.
`EndToEndWindow` is true at any `sessCalls`, since `β` is its hypothesis; what is open is a number for `E[β]`. Before
the card states one, `sessCalls` is tied to the play, with `CollCounted`'s restatement and one definition for both: the
sessions as an oracle strategy of the window's game, one `QTree` per prover move with `H` only in the verdict, and
`sessCalls` its calls (as `RegQueries` does for the registration). The window audit is `auditReg Lr (session H s)`:
`n` audit units, the law `Lr R` at the registration `R`, and the session at the oracle `H` and the salt `s`. -/
structure ProverModel (Q R S : Type) {n : ℕ} (Reg : Type) (Lr : Reg → Law n)
    (session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool) where
  Prog : Type
  play : Prog → (H : Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → (s : S) →
    FlockSoundness.Game.Strategy (auditReg Lr (session H s))
  cost : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℝ
  received : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℕ
  cutoff : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℕ
  calls : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℕ
  sessCalls : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℕ

section Prover

variable {n : ℕ} {Reg : Type} {Lr : Reg → Law n}
  {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}

/-- **The registration is an oracle program's output, and `calls` counts its queries** (the form of
`CostModel.Queries`, for the registration): on fixed inputs, a play's registration is the result of an adaptive oracle
program, and the model's `calls` is exactly that program's queries. The sessions' strategies, whose type depends on `H`,
are counted by the model's `sessCalls` (`Prover.Bounded`), not tied to an oracle program here. -/
def ProverModel.RegQueries (PM : ProverModel Q R S Reg Lr session) : Prop :=
  ∀ (prog : PM.Prog) (U : Workload) (act : ℕ → Pouw.Mat) (st : Pouw.State) (s : S), ∃ t : QTree Q R Reg,
    ∀ H : Q → R, (PM.play prog H U act st s).1 = t.eval H ∧ t.calls H = PM.calls prog H U act st s

/-- **A window prover**, chosen after the workload, as PoUW's `Adversary`: a preprocessing oracle program (never
reading the salt; counted queries), a program of the prover model, and the activation input it reads. -/
structure Prover (PM : ProverModel Q R S Reg Lr session) where
  pre : QTree Q R Pouw.State
  online : PM.Prog
  act : ℕ → Pouw.Mat

/-- **The prover's play** at the oracle `H` and the salt `s`: the strategy the window audit runs against. -/
def Prover.play {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (H : Q → R) (s : S) :
    FlockSoundness.Game.Strategy (auditReg Lr (session H s)) :=
  PM.play B.online H U B.act (B.pre.eval H) s

/-- The place of the prover's registration's receipt in the challenger's record. -/
def Prover.received {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (H : Q → R) (s : S) : ℕ :=
  PM.received B.online H U B.act (B.pre.eval H) s

/-- **The draw coin's cutoff**: the place of the epoch's draw coin's release in the challenger's record. -/
def Prover.cutoff {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (H : Q → R) (s : S) : ℕ :=
  PM.cutoff B.online H U B.act (B.pre.eval H) s

/-- The online phase costs at most `T`, and preprocessing, the registration and the sessions together make at most `q`
oracle calls, for every oracle and salt (surely). -/
def Prover.Bounded {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), PM.cost B.online H U B.act (B.pre.eval H) s ≤ T ∧
    B.pre.calls H + PM.calls B.online H U B.act (B.pre.eval H) s +
      PM.sessCalls B.online H U B.act (B.pre.eval H) s ≤ q

/-- **The service's acceptance at the draw coin's cutoff**, for an outcome `o` of the window audit against `B`'s play
at `H` and `s`: the sessions accept (`o.1`), and the challenger received the registration before it released the
epoch's draw coin (`received < cutoff`, in its record's order). A window received after the draw coin is refused
(`late-registration`), whatever it holds: its registration may depend on the draw, which the window audit's game, the
registration and then the draw coin, doesn't model. The bound is an order, not a time. The wall-clock deadline
(`Check.registration_deadline`, counted from the salt coin's release) is the challenger's, which waits on its record's
`released_at` and refuses a registration past it too, so the service accepts only runs this event holds of. -/
def AcceptsBy {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (H : Q → R) (s : S)
    (o : Bool × Finset (Fin n)) : Prop :=
  o.1 = true ∧ B.received U H s < B.cutoff U H s

end Prover

/-- **Straight-line extraction of the registration's transcript** (a named premise, open; the random-oracle model,
claim id `random-oracle`), for a prover model `PM`, PoUW's cost model `CM`, a domain `D`, the transcript
`Ccom H s R` that the registration `R`'s commitments bind at the oracle and the salt (the analysis's committed transcript,
read at the registration alone), the extractor's cost `T_ext` and calls `q_ext`, and an error `ε_id(q, N)`. For every
in-domain workload and every prover `B` within `(T, q)` there is an adversary `A` of PoUW's game within
`(T + T_ext, q + q_ext)` whose transcript is the one `B`'s registration binds, on every run whose registration the
challenger received before the draw coin's cutoff (`AcceptsBy`'s order), except with probability `ε_id(q, N)` over the
oracle and the salt:
$$\Pr_{H,s}\big[\mathrm{received}_B(H, s) < \mathrm{cutoff}_B(H, s) \wedge
  \mathrm{Ccom}(H, s, \mathrm{reg}\,B.\mathrm{play}(H, s)) \ne A.\mathrm{transcript}(H, s)\big]
  \le \varepsilon_{id}(q, N).$$

The witness is `B`'s online phase run in `CM` with a straight-line extractor: it reads the query log of the online phase
up to the registration, the commitments' oracle calls that `calls` counts, and decodes each registered root's tree from
the queries whose answers are its nodes, a lookup at cost `T_ext`, with no rewinding (a rewinding extractor's cost can't
sit inside `T`). It reads no session query: `Ccom` is a function of the registration, so what the sessions query, and
what they open the registered roots to, can't change the transcript it must match. A session that opens a root to
anything but `Ccom` is a link failure, charged in `WindowBound`'s `β` (`δ_link` at the prover's state). Its
random-oracle content is the probability: over a uniform `H`, which stands for SHA-512 domain-separated for commitments
and the noise, the registration's transcript differs from the decoded one only if the `q` queries before the
registration hold a collision or the registration holds a node that none of them answered, so
`ε_id(q, N) ≈ (q² + q·N)/2^512` is the expected instance. Its other content is a cost-model fact: `CM` runs `PM`'s
online programs at their cost. It is claimed only on runs received before the cutoff, the runs the service accepts: a
registration received after the draw coin may depend on the draw, which the window audit's game, the registration and
then the draw coin, doesn't model.

It is stated for models whose registration is an oracle program's output with its queries counted
(`ProverModel.RegQueries`, its antecedent); the sessions' calls are counted in `q` too (`Prover.Bounded`), which can only
shrink the provers it quantifies over. It is the shape of relative security's
premises, an adversary that commits first with random-oracle commitments (`Assumptions.ServiceSoundStaged`,
`Extraction`), and it is filed under the random-oracle claim with the seeded windows' premise
(`Accounting.Assumptions.Compose.SeedExtractedAt`), so that the random oracle is ruled on once. -/
def StraightLineExtraction [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {n : ℕ} {Reg : Type}
    {Lr : Reg → Law n} {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}
    (PM : ProverModel Q R S Reg Lr session) (CM : CostModel Q R S) (D : Layout → Prop)
    (Ccom : (Q → R) → S → Reg → Pouw.Transcript) (Text : ℝ) (qext : ℕ) (εid : ℕ → ℕ → ℝ) : Prop :=
  PM.RegQueries → ∀ U : Workload, U.InDomain D → ∀ (T : ℝ) (q : ℕ) (B : Prover PM), B.Bounded U T q →
    ∃ A : Adversary CM, A.Bounded U (T + Text) (q + qext) ∧
      pr (fun x : (Q → R) × S =>
        B.received U x.1 x.2 < B.cutoff U x.1 x.2 ∧ Ccom x.1 x.2 (B.play U x.1 x.2).1 ≠ A.transcript U x.1 x.2) ≤
        εid q U.layout.N

/-- **The window audit's bound on a play** (a hypothesis of `EndToEndWindow`, discharged below), for the window's share
`ε`, the registration's transcript `Ccom`, a play `σ'` and a bound `β(H, s)`: at every oracle and salt, the audit
accepts `σ'` while the incorrect work of its registration's transcript is at least `ε·W`, `W = ∑_{u < N} W_ref(u)`, with
probability at most `β(H, s)`. `β` may depend on the oracle, the salt and the prover: no constant bounds it, since a
play that holds a collision of `H` passes with extraction failing. On average over the oracle it is bounded by the
counted queries (`Prover.Bounded`) only where the sessions are one oracle program's run at `H`, counted by that
program's calls, which `ProverModel.sessCalls` doesn't yet tie (see `ProverModel`). `W` holds PoUW's units only, so a
filler, which is no PoUW unit, adds nothing to it.

Its discharges:
* `WindowBoundOfService`, at the service's law;
* gap (6)'s `Pouw.Guarantees.Window.WindowEps` at `𝓑 = harmSets W_ref ε ∧ CopyClosed`, with the service's
  sessions as its calls at `η = 0` (`ServiceEscapes`), gives it at `β(H, s)` its right side at the play, through the
  bridge to C-Flock's game in `Proofs.Pouw.Window.FlockBridge` (`Pouw.Window.prob_ofFlock`, `CallAnalysis.ofFlock`, and
  its premise `FlockKnowledgeSound`; branch `cursor/pouw-window-eps-stmt2-c7a9`), with the deployed wrong sets
  copy-closed (`Pouw.Window.WrongCopyClosed`) and `Realizes` at them. -/
def WindowBound (P : Protocol Q R S) (U : Workload) (ε : ℝ) {n : ℕ} {Reg : Type} {Lr : Reg → Law n}
    {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}
    (Ccom : (Q → R) → S → Reg → Pouw.Transcript)
    (σ' : (H : Q → R) → (s : S) → FlockSoundness.Game.Strategy (auditReg Lr (session H s)))
    (β : (Q → R) → S → ℝ≥0∞) : Prop :=
  ∀ (H : Q → R) (s : S), FlockSoundness.Game.prob
      (fun o => o.1 = true ∧ ε * totalWref P U.layout ≤ incorrectWork P U H s (Ccom H s (σ' H s).1))
      (auditReg Lr (session H s)) (σ' H s) ≤ β H s

/-- **The window realizes PoUW's units** at the oracle `H` and the salt `s`, for the wrong units `wrong` of an
analysis's transcripts and the map `tx` from them to PoUW transcripts: at every transcript `X`, the incorrect work of
`tx X` is at most the window's unsound work at `X`'s wrong units,
$$\sum_{u < N,\ u \notin K(tx\,X)} W_{ref}(u) \le \mathrm{unsoundWork}_w(cl, \mathrm{wrong}\,X).$$
At the service's draw the closures are empty (`fun _ => ∅`, each drawn unit proved alone), and the unsound work is the
wrong units' own. At coordinate units it holds by construction (`realizes_of_coordinates`). -/
def Realizes (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) {n m : ℕ} (σ : Fin n → Fin m) (w : Fin m → ℕ)
    (cl : Fin n → Finset (Fin n)) {Tr : Type} (wrong : Tr → Finset (Fin n)) (tx : Tr → Pouw.Transcript) : Prop :=
  ∀ X : Tr, incorrectWork P U H s (tx X) ≤ (Law.unsoundWork σ w cl (wrong X) : ℝ)

/-- **Each PoUW unit is one template instance** (what the compiler gives at decision 107's coordinate units), at the
oracle `H` and the salt `s`, for the audit units' strata `σ` and work `w`, the wrong units `wrong` of the analysis's
transcripts, the map `tx` to PoUW transcripts and the instance `a u` of each PoUW unit `u`:
* distinct PoUW units are distinct instances;
* an instance's work is at least its unit's `W_ref` (the work table at coordinates);
* a unit not correct in `tx X` is a wrong instance in `X`: the instance's statement is the unit's `Correct` (int7
  activations, the checked value equal to `Y` at the noise from its seed, the useful output `A·B`), so a unit that fails
  it has committed outputs that aren't its instance's function of its inputs.
Audit units that are no PoUW unit's instance, the fillers among them, are unconstrained. -/
def CoordinateInstances (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) {n m : ℕ} (σ : Fin n → Fin m)
    (w : Fin m → ℕ) {Tr : Type} (wrong : Tr → Finset (Fin n)) (tx : Tr → Pouw.Transcript) (a : ℕ → Fin n) : Prop :=
  (∀ u < U.layout.N, ∀ u' < U.layout.N, a u = a u' → u = u') ∧
    (∀ u < U.layout.N, P.Wref U.layout u ≤ w (σ (a u))) ∧
    ∀ (X : Tr) (u : ℕ), u < U.layout.N → u ∉ correctSet P U H s (tx X) → a u ∈ wrong X

/-- **`Realizes` by construction at coordinate units** (proved, `realizes_of_coordinates`): if each PoUW unit is one
template instance, the window realizes PoUW's units, whatever the closures. The incorrect units' `W_ref` sum to at most
their instances' work, which is in the wrong units' work, which is in the unsound work. -/
def RealizesOfCoordinates : Prop :=
  ∀ (Q R S : Type) (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (n m : ℕ) (σ : Fin n → Fin m)
    (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (Tr : Type) (wrong : Tr → Finset (Fin n))
    (tx : Tr → Pouw.Transcript) (a : ℕ → Fin n),
    CoordinateInstances P U H s σ w wrong tx a → Realizes P U H s σ w cl wrong tx

theorem realizes_of_coordinates : RealizesOfCoordinates := by
  intro Q R S P U H s n m σ w cl Tr wrong tx a ⟨hinj, hw, hwrong⟩ X
  set D := range U.layout.N \ correctSet P U H s (tx X)
  have hD : ∀ u ∈ D, u < U.layout.N := fun u hu => Finset.mem_range.1 (Finset.mem_sdiff.1 hu).1
  have h1 : ∑ u ∈ D, P.Wref U.layout u ≤ ∑ u ∈ D, w (σ (a u)) := Finset.sum_le_sum fun u hu => hw u (hD u hu)
  have h2 : ∑ v ∈ D.image a, w (σ v) = ∑ u ∈ D, w (σ (a u)) :=
    Finset.sum_image fun u hu u' hu' h => hinj u (hD u hu) u' (hD u' hu') h
  have h3 : D.image a ⊆ wrong X := by
    intro v hv
    obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hv
    exact hwrong X u (hD u hu) (Finset.mem_sdiff.1 hu).2
  have h4 : ∑ v ∈ D.image a, w (σ v) ≤ Law.workOf σ w (wrong X) := Finset.sum_le_sum_of_subset h3
  have h5 := Law.workOf_le_unsoundWork σ w cl (wrong X)
  have hN : ∑ u ∈ D, P.Wref U.layout u ≤ Law.unsoundWork σ w cl (wrong X) := h1.trans (h2 ▸ h4.trans h5)
  unfold incorrectWork
  exact_mod_cast hN

/-- **`WindowBound` at the service's law** (statement, no proof yet; to be proved as `windowBound_of_service`). The
service (decision 102, `window.py`) draws one uniform `K′`-subset of the window's `N` units after the receipt
(`subset:K′`), `N` the real units padded to the grid with fillers (`pad(N_real + 1)`), each proved alone (no closures).
If every audit unit does at most `w_max` (fillers included), the window realizes PoUW's units, the window's one analysis
at each oracle and salt has knowledge and link terms at the prover's state, and its committed transcript is the
registration's (`tx ∘ committed` reads the registration alone, `Ccom`), then `WindowBound` holds at the share `ε` at
$$\beta(H, s) = \mathrm{miss}_{K'}(b) + \varepsilon_{ks}(H, s, R, \tau) + \delta_{link}(H, s, R, \tau),\qquad
  b = \lceil \varepsilon W / w_{max} \rceil,$$
`(R, τ)` the play's registration and state. A wrong set holding `ε·W` of incorrect work holds at least `b` units, and
the first term is the subset law's own escape of `b` units (`Law.miss`), `C(N − b, K′)/C(N, K′)` for `b ≤ N` by
`FlockSoundness.Audit.Law.subset_miss` and `0` past it, the exact hypergeometric miss, so it needs no work table (over
main's served levels it is at most `2^-40.6` at `(ε, δ) = (0.1%, 2^-40)` and `2^-128.5` at `(1%, 2^-128)`,
the recheck's probe 2):
* where `K′ = N` (`k_prime`'s answer when no `K < N` reaches `δ`), every unit is drawn and the first term is `0`
  (`b ≥ 1`);
* where `K′ < N`, it is at most the binomial `(1 − ε·W/(N·w_max))^{K′}`, which `k_prime` makes at most its `δ`, since
  `W` is at least the credited work.
A session that opens the registered roots to anything but `Ccom` is in `δ_link`.

The service's draw is one law for the whole window; the window's `ε_ks` and `δ_link` composed from its sessions' are
gap (6)'s (`WindowEps`). -/
def WindowBoundOfService : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (U : Workload)
    (ε : ℝ), 0 < ε → ε ≤ 1 →
    ∀ (N K : ℕ) (hK : K ≤ N) (m : ℕ) (σ : Fin N → Fin m) (w : Fin m → ℕ) (wmax : ℕ), (∀ t, w t ≤ wmax) →
    ∀ (Reg : Type) (session : (Q → R) → S → Finset (Fin N) → Reg → FlockSoundness.Game Bool)
      (An : (H : Q → R) → (s : S) → ExtractionAnalysis (Law.subset N K hK) Reg (session H s))
      (tx : (H : Q → R) → (s : S) → (An H s).Tr → Pouw.Transcript) (Ccom : (Q → R) → S → Reg → Pouw.Transcript),
    (∀ H s R' τ, tx H s ((An H s).committed R' τ) = Ccom H s R') →
    (∀ H s, Realizes P U H s σ w (fun _ => ∅) (An H s).wrong (tx H s)) →
    ∀ εks δlink : (H : Q → R) → (s : S) → (R' : Reg) → Cont (Law.subset N K hK) Reg (session H s) R' → ℝ≥0∞,
    (∀ H s, (An H s).KnowledgeSound (εks H s)) → (∀ H s, (An H s).LinkSound (δlink H s)) →
    ∀ σ' : (H : Q → R) → (s : S) →
      FlockSoundness.Game.Strategy (auditReg (fun _ : Reg => Law.subset N K hK) (session H s)),
      WindowBound P U ε Ccom σ' fun H s =>
        (Law.subset N K hK).miss ⌈ε * totalWref P U.layout / (wmax : ℝ)⌉₊ +
          εks H s (reg (atReg (σ' H s))) (cont (atReg (σ' H s))) +
          δlink H s (reg (atReg (σ' H s))) (cont (atReg (σ' H s)))

end Pouw.WindowWork

namespace Pouw.Guarantees

open Finset FlockSoundness.Audit
open scoped ENNReal

/-- **PoUW's work bound at the window audit** (statement only, no proof yet), at the window's share `ε`
(`0 < ε ≤ 1`). A prover that passes the window audit, its registration received before the epoch's draw coin, spent at
least a `(1 − γ_u)(1 − ε)` share of the window's declared work between the salt's release and the registration's
receipt, less the extractor's cost, except with probability `E[β] + ε_id(q, N) + η(q + q_ext, N)`.

**What it bounds.** The prover `B` is a program of the prover model `PM` (`WindowWork.Prover`), and the audit runs
against its play. Its work `T` bounds the cost of its online phase, from the salt's release to the registration's
receipt, in the cost model's units, which the model fixes, not the prover; an accepted registration is received before
the draw coin's release, so that is work between the salt coin and the draw coin. `q` counts its oracle calls, the
sessions' among them. Its work is compared with the window's declared work `W = ∑_{u < N} W_ref(u)` in `CM`'s units,
`W_ref(u)` being the honest reference's cost of unit `u` without hash calls, on decision 107's coordinate units
(`WindowWork.coordDomain`). Averaged over the oracle `H` and the salt `s`, the service accepts the play at the draw
coin's cutoff (`WindowWork.AcceptsBy`) while `T + T_ext < (1 − γ_u)(1 − ε)·W` with probability at most
$$\mathbb{E}_{H,s}\Pr[\text{accept before the draw coin} \wedge T + T_{ext} < (1-\gamma_u)(1-\varepsilon)\,W] \le
  \mathbb{E}_{H,s}\,\beta(H, s) + \varepsilon_{id}(q, N) + \eta(q + q_{ext}, N).$$
The challenger's wall-clock deadline (`Check.registration_deadline`) only refuses more, so the bound holds of what the
service accepts at any deadline.
The audit is `FlockSoundness.Audit.auditReg` at the service's law, one uniform `K`-subset of the `n` padded units
(`Law.subset n K`, `subset:K′` in `window.py`), with any sessions. `ε` is the window's, the `ε` its `K′` was sized at
(`window.sampling`), so the statement covers each grid main serves, `(0.1%, 2^-40)` and `(1%, 2^-128)`.

**What it assumes**, each a hypothesis:
* the work step `WindowWork.TileWork γ_u` with error `η` at coordinate units (a named premise; its instance in Lean is
  the metered model, `WindowWork.tileWork_metered`; the deployed device's is EPS1's, gap (3));
* `WindowWork.StraightLineExtraction` at `(T_ext, q_ext, ε_id)` (the named random-oracle premise): from `B`, a PoUW
  adversary within `(T + T_ext, q + q_ext)` whose transcript is the registration's (`Ccom`) on every run received
  before the draw coin's cutoff, except with probability `ε_id`; with its antecedent, that the model's registrations
  are oracle programs' outputs with their queries counted (`ProverModel.RegQueries`);
* `WindowWork.WindowBound` at `ε` for `B`'s play at `β(H, s)` (discharged by `WindowBoundOfService`, or gap (6)'s
  `WindowEps` through its bridge, each from `WindowWork.Realizes`, which holds by construction at coordinate units).

**What it composes**, at every oracle and salt: either the registration was received after the draw coin, which the
service refuses; or the registration's transcript isn't the extracted adversary's (on average at most `ε_id`); or the
work step's event holds for that adversary at budget `T + T_ext` (on average at most `η`); or the units correct in the
registration's transcript are worth at most `(T + T_ext)/(1 − γ_u) < (1 − ε)·W`, so its incorrect work exceeds `ε·W`,
and the audit accepts with that at most `β(H, s)`. At `γ_u ≥ 1` the event is empty. -/
def EndToEndWindow : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (γu : ℝ) (η : ℕ → ℕ → ℝ),
    WindowWork.TileWork CM P γu η →
    -- the window's share, the `ε` its draw was sized at
    ∀ ε : ℝ, 0 < ε → ε ≤ 1 →
    -- the window audit at the service's draw: `n` padded units, `K` of them drawn uniformly, the session at each
    -- oracle and salt, and the transcript a registration's commitments bind, read as PoUW's transcript
    ∀ (n K : ℕ) (hK : K ≤ n) (Reg : Type)
      (session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool)
      (Ccom : (Q → R) → S → Reg → Transcript)
    -- the prover model and the extraction of PoUW's adversary from its provers
      (PM : WindowWork.ProverModel Q R S Reg (fun _ : Reg => Law.subset n K hK) session) (Text : ℝ) (qext : ℕ)
      (εid : ℕ → ℕ → ℝ),
    PM.RegQueries → 0 ≤ Text → WindowWork.StraightLineExtraction PM CM WindowWork.coordDomain Ccom Text qext εid →
    ∀ U : Workload, U.InDomain WindowWork.coordDomain → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q + qext ≤ 2 ^ 64 →
    ∀ B : WindowWork.Prover PM, B.Bounded U T q →
    -- the window audit's bound on the prover's play, at each oracle and salt
    ∀ β : (Q → R) → S → ℝ≥0∞, WindowWork.WindowBound P U ε Ccom (B.play U) β →
      FlockSoundness.Game.avg (fun x : (Q → R) × S =>
        FlockSoundness.Game.prob
          (fun o => WindowWork.AcceptsBy B U x.1 x.2 o ∧
            T + Text < (1 - γu) * (1 - ε) * totalWref P U.layout)
          (auditReg (fun _ : Reg => Law.subset n K hK) (session x.1 x.2)) (B.play U x.1 x.2))
        ≤ FlockSoundness.Game.avg (fun x : (Q → R) × S => β x.1 x.2) + ENNReal.ofReal (εid q U.layout.N) +
          ENNReal.ofReal (η (q + qext) U.layout.N)

end Pouw.Guarantees
