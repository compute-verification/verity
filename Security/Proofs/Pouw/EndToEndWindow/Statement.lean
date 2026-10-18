import Definitions.Pouw.Game.Params
import Models.Pouw.Assumptions.Game
import Models.Pouw.Assumptions.NCP.Assumptions
import Proofs.Flock.Soundness.Audit.Extraction
import Proofs.Flock.Soundness.Audit.RegDraw
import Proofs.Flock.Soundness.Audit.Window

/-!
# PoUW's work bound at the window audit (trusted)

`EndToEndWindow` is `Pouw.Guarantees.EndToEnd` with the window audit in place of its ideal audit. Its cost `T` is the
cost of the prover that is audited:

* **One prover, costed.** A prover is a program of a prover model (`WindowWork.ProverModel`). Its play is its whole
  strategy in the window audit, the registration and then the sessions, and the model charges its online phase, from
  the salt to the registration, at most `T`, with at most `q` oracle calls in all. The audited strategy is that play.
* **PoUW's adversary is extracted from it** (`WindowWork.StraightLineExtraction`, the one random-oracle premise): from
  every prover within `(T, q)`, an adversary of PoUW's game within `(T + T_ext, q + q_ext)` whose transcript is the
  analysis's committed transcript of the prover's play, except with probability `ε_id(q, N)` over the oracle and the salt.
* **The work step** is a parameter, `WindowWork.TileWork γ_u`.
* **The window audit's guarantee is a hypothesis** (`WindowWork.WindowBound`): at each oracle and salt, the audit accepts
  the prover's play while its incorrect work is at least 0.1% of `W` with probability at most `β(H, s)`, a function of
  the oracle, the salt and the prover. The conclusion averages it.

It is a statement with no proof yet; its proof is to be
`Pouw.SecurityProofs.EndToEndWindow : Pouw.Guarantees.EndToEndWindow`.

**Units.** Under decision 107 a PoUW unit is one output coordinate, shape `(1, k, 1)`, a `K`-chunked s8 dot with s32
running sums, and one template instance of the window (`WindowWork.coordinates`). There `WindowWork.Realizes` holds by
construction (`WindowWork.RealizesOfCoordinates`, a counting lemma from `WindowWork.CoordinateInstances`). Larger units
can't be audit units: an `r × c` tile breaks the 16-bit output rule, and a GEMM unit spans audit units the closures
don't tie together.

**The work step's instances.** TT's (`WindowWork.TileWorkOfTT`, `WindowWork.TileWorkOfTTNCP`) are vacuous at
coordinates: every carded TT domain (`Shape.InDefaultDomain`, `m1Domain`, `NCP.leadDomain`) has `16 ∣ m`, `m > 0` and
`n ≥ 2^10`, so none holds a layout with a coordinate unit. EPS1's comes from compute accounting's tensor–scalar model,
`Accounting.Guarantees.Compose.CircuitCredit`, through a bridge from its play to PoUW's cost model, which is PoUW's gap
(3) and isn't stated here.

**The window's law.** `EndToEndWindow` takes any law at each registration. Two discharges of `WindowBound` are stated:
* `WindowWork.WindowBoundOfService`: the service's law (decision 102, `window.py`), one uniform `K′`-subset of the `N`
  padded units with fillers, sized by `k_prime`'s share form, with no work table;
* `WindowWork.WindowBoundOfRecord`: a law within `η` of C-Flock's law of record, with the work table, from
  `FlockSoundness.Audit.extraction_audit_window_split_of_record_of_le_slack`.

Gap (6)'s `Pouw.Guarantees.Window.WindowEps` (branch `cursor/pouw-window-eps-stmt-c7a9`) bounds the same event in
gap (6)'s copy of the game (`Pouw.Window`), with the window's knowledge and link terms composed from its calls'. The
bridge that makes it a discharge of `WindowBound` is named in `WindowBound`'s docstring and isn't stated here: gap (6)'s
definitions aren't on main.

**Deadline.** Compute accounting's cost model `Accounting.Actual.costModel` charges the prover's work in
`[tₛ + ℓ, t_d − ℓ)`, so at that model `T` is already the work between the salt and the deadline, and the theorem needs
no separate deadline form. **Pearl-C is out of scope**: it has its own game and premises (`TTOutTile`,
`TileProofSound`, `Pouw.Guarantees.PearlCSampledByDeadline`).

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

/-- **The work step, TileWork(γ_u)** (a named premise, open), for a cost model `CM`, a protocol `P`, a domain `D`, a
gap `γ_u` and an error `η(q, N)`. A correct unit was paid for: the prover spent at least `(1 − γ_u)·W_ref(u)` on it.
A cost model prices a whole run, not a unit, so it is stated summed over the window, in `G_γ`'s game (the same
adversaries, budgets and oracle-call bounds). For every adversary within `(T, q)`,
$$\Pr_{H,s}\Big[T < (1-\gamma_u)\sum_{u \in K} W_{ref}(u)\Big] \le \eta(q, N),$$
`K` being the correct units (`correctSet`: checked values and useful output correct). It is `G_γ` at `γ = γ_u` with
the gap multiplied out, so it needs no `γ_u < 1`: at `γ_u ≥ 1` its event is empty.

Its instances: NCP's and Pearl's from TT_NCP and TT (`TileWorkOfTTNCP`, `TileWorkOfTT`), both conjectures, vacuous at
coordinate units (no carded TT domain holds one); EPS1's from `Accounting.Guarantees.Compose.CircuitCredit`, PoUW's
gap (3). -/
def TileWork [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : Layout → Prop) (γu : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T < (1 - γu) * ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), (P.Wref U.layout u : ℝ))
      ≤ η q U.layout.N

/-- **`TileWork` from TT** (statement, no proof yet; to be proved as `tileWork_of_TT`): Pearl's instance, `G_γ`'s
reduction (`Pinned.GammaFromTT`) read as the work step. If `W_ref(u) ≤ Ω_max·m·k·n` on the domain and TT(γ₀) holds
with error `ε`, then `TileWork` holds at `γ_u = 1 − (1 − γ₀)/Ω_max` with error `ε`. Vacuous at coordinate units: no
carded TT domain holds a layout with one. -/
def TileWorkOfTT : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε →
    TileWork CM P D (reductionGamma γ₀ Ωmax) ε

/-- **`TileWork` from TT_NCP** (statement, no proof yet; to be proved as `tileWork_of_TTNCP`): NCP's instance, at
the credit `3·m·k·n` (`GamePinned.GammaFromTTNCP` read as the work step). Vacuous at coordinate units, as
`TileWorkOfTT`. -/
def TileWorkOfTTNCP : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * (3 * ((U.layout.shape u).Wmm : ℝ))) →
    NCP.Assumptions.TTNCP CM P D γ₀ ε →
    TileWork CM P D (reductionGamma γ₀ Ωmax) ε

/-- **A window prover's model**: its programs, and for each a whole strategy in the window audit at every oracle and
salt (`play`: the registration, then the sessions on every draw), with what the model charges for it. `cost` is the
work of its online phase, from the salt to the registration, in PoUW's cost units (at compute accounting's model, the
work between the salt and the deadline); `calls` is every oracle call of the play, the registration's and the sessions',
worst case over the verifier's coins. The model fixes both; a prover only picks a program. The window audit is
`auditReg Lr (session H s)`: `n` audit units, the law `Lr R` at the registration `R`, and the session at the oracle
`H` and the salt `s`. -/
structure ProverModel (Q R S : Type) {n : ℕ} (Reg : Type) (Lr : Reg → Law n)
    (session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool) where
  Prog : Type
  play : Prog → (H : Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → (s : S) →
    FlockSoundness.Game.Strategy (auditReg Lr (session H s))
  cost : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℝ
  calls : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℕ

section Prover

variable {n : ℕ} {Reg : Type} {Lr : Reg → Law n}
  {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}

/-- **The registration reaches `H` only through counted queries** (red-team B6's form, `CostModel.Queries`, for the
registration): on fixed inputs, a play's registration is the result of an adaptive oracle program whose queries are
among the play's counted calls. The sessions' strategies, whose type depends on `H`, can't be put in this form. -/
def ProverModel.RegQueries (PM : ProverModel Q R S Reg Lr session) : Prop :=
  ∀ (prog : PM.Prog) (U : Workload) (act : ℕ → Pouw.Mat) (st : Pouw.State) (s : S), ∃ t : QTree Q R Reg,
    ∀ H : Q → R, (PM.play prog H U act st s).1 = t.eval H ∧ t.calls H ≤ PM.calls prog H U act st s

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

/-- The online phase costs at most `T`, and preprocessing and the play together make at most `q` oracle calls, for
every oracle and salt (surely). -/
def Prover.Bounded {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), PM.cost B.online H U B.act (B.pre.eval H) s ≤ T ∧
    B.pre.calls H + PM.calls B.online H U B.act (B.pre.eval H) s ≤ q

end Prover

/-- **Straight-line extraction of the committed transcript** (a named premise, open; the random-oracle model, claim id
`random-oracle`), for a prover model `PM`, PoUW's cost model `CM`, a domain `D`, the analysis's committed transcript of
a play `committed` (read as PoUW's transcript), the extractor's cost `T_ext` and calls `q_ext`, and an error
`ε_id(q, N)`. For every in-domain workload and every prover `B` within `(T, q)` there is an adversary `A` of PoUW's
game within `(T + T_ext, q + q_ext)` whose transcript is the committed transcript of `B`'s play except with
probability `ε_id(q, N)` over the oracle and the salt:
$$\Pr_{H,s}\big[\mathrm{committed}(B.\mathrm{play}(H, s)) \ne A.\mathrm{transcript}(H, s)\big] \le \varepsilon_{id}(q, N).$$

The witness is `B`'s online phase run in `CM` with a straight-line extractor: it reads the online phase's query log and
decodes each registered root's tree from the queries whose answers are its nodes, a lookup at cost `T_ext`, with no
rewinding (a rewinding extractor's cost can't sit inside `T`). Its random-oracle content is the probability: over a
uniform `H`, which stands for SHA-512 domain-separated for commitments and the noise, the committed transcript differs
from the decoded one only if the play's `q` queries hold a collision or open a node that no query before the
registration answered, so `ε_id(q, N) ≈ (q² + q·N)/2^512` is the expected instance. Its other content is a cost-model
fact: `CM` runs `PM`'s online programs at their cost (at one machine, such as compute accounting's model, the identity).

It is false for a model whose plays read `H` other than through their counted calls: such a play can open a registered
root to a second preimage at every oracle. So it is stated for models whose registration is an oracle program's output
(`ProverModel.RegQueries`, its antecedent), and claimed for models whose sessions are too, which the audit's strategies,
functions of `H`, can't express in Lean. It is the shape of relative security's premises, an
adversary that commits first with random-oracle commitments (`Assumptions.ServiceSoundStaged`, `Extraction`), and it is
carded on the random-oracle card with gap (7)'s grinding (`Accounting.Assumptions.Compose.SeedGrind`), so that Daniel
rules on the random oracle once. -/
def StraightLineExtraction [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {n : ℕ} {Reg : Type}
    {Lr : Reg → Law n} {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}
    (PM : ProverModel Q R S Reg Lr session) (CM : CostModel Q R S) (D : Layout → Prop)
    (committed : (H : Q → R) → (s : S) → FlockSoundness.Game.Strategy (auditReg Lr (session H s)) → Pouw.Transcript)
    (Text : ℝ) (qext : ℕ) (εid : ℕ → ℕ → ℝ) : Prop :=
  PM.RegQueries → ∀ U : Workload, U.InDomain D → ∀ (T : ℝ) (q : ℕ) (B : Prover PM), B.Bounded U T q →
    ∃ A : Adversary CM, A.Bounded U (T + Text) (q + qext) ∧
      pr (fun x : (Q → R) × S => committed x.1 x.2 (B.play U x.1 x.2) ≠ A.transcript U x.1 x.2) ≤ εid q U.layout.N

/-- **The window audit's bound on a play** (a hypothesis of `EndToEndWindow`, discharged below), for the committed
transcript `committed`, a play `σ'` and a bound `β(H, s)`: at every oracle and salt, the audit accepts `σ'` while the
incorrect work of its committed transcript is at least 0.1% of `W = ∑_{u < N} W_ref(u)` with probability at most
`β(H, s)`. `β` may depend on the oracle, the salt and the prover: no constant bounds it, since a play that holds a
collision of `H` passes with extraction failing. `W` holds PoUW's units only, so a filler, which is no PoUW unit, adds
nothing to it.

Its discharges:
* `WindowBoundOfService`, at the service's law;
* `WindowBoundOfRecord`, at a law within `η` of C-Flock's law of record;
* gap (6)'s `Pouw.Guarantees.Window.WindowEps` at `𝓑 = harmSets W_ref (1/1000) ∧ CopyClosed`, with the service's
  sessions as its calls at `η = 0` (`ServiceEscapes`), gives it at `β(H, s)` its right side at the play, through a
  bridge not stated here: a translation of `Pouw.Window`'s games, strategies, `auditReg`, `reg` and `cont` into
  `FlockSoundness`'s that keeps `prob` (the two games are copies of one inductive, with the same constructors and the
  same `prob`, so it is a lemma, not a premise); its units `WUnit` (a session and a tile) as `Fin n`; and `harmSets`
  read on copy-closed wrong sets, where a tile's copies scale both sides alike, with `Realizes` at them. -/
def WindowBound (P : Protocol Q R S) (U : Workload) {n : ℕ} {Reg : Type} {Lr : Reg → Law n}
    {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}
    (committed : (H : Q → R) → (s : S) → FlockSoundness.Game.Strategy (auditReg Lr (session H s)) → Pouw.Transcript)
    (σ' : (H : Q → R) → (s : S) → FlockSoundness.Game.Strategy (auditReg Lr (session H s)))
    (β : (Q → R) → S → ℝ≥0∞) : Prop :=
  ∀ (H : Q → R) (s : S), FlockSoundness.Game.prob
      (fun o => o.1 = true ∧ totalWref P U.layout ≤ 1000 * incorrectWork P U H s (committed H s (σ' H s)))
      (auditReg Lr (session H s)) (σ' H s) ≤ β H s

/-- **The window realizes PoUW's units** at the oracle `H` and the salt `s`, for the wrong units `wrong` of an
analysis's transcripts and the map `tx` from them to PoUW transcripts: at every transcript `X`, the incorrect work of
`tx X` is at most the window's unsound work at `X`'s wrong units,
$$\sum_{u < N,\ u \notin K(tx\,X)} W_{ref}(u) \le \mathrm{unsoundWork}_w(cl, \mathrm{wrong}\,X).$$
At coordinate units it holds by construction (`RealizesOfCoordinates`). -/
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

/-- **`Realizes` by construction at coordinate units** (statement, no proof yet; a counting lemma, to be proved as
`realizes_of_coordinates`): if each PoUW unit is one template instance, the window realizes PoUW's units, whatever the
closures. The incorrect units' `W_ref` sum to at most their instances' work, which is in the wrong units' work, which is
in the unsound work. -/
def RealizesOfCoordinates : Prop :=
  ∀ (Q R S : Type) (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (n m : ℕ) (σ : Fin n → Fin m)
    (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (Tr : Type) (wrong : Tr → Finset (Fin n))
    (tx : Tr → Pouw.Transcript) (a : ℕ → Fin n),
    CoordinateInstances P U H s σ w wrong tx a → Realizes P U H s σ w cl wrong tx

/-- **`WindowBound` at the service's law** (statement, no proof yet; to be proved as `windowBound_of_service`). The
service (decision 102, `window.py`) draws one uniform `K′`-subset of the window's `N` units after the receipt, `N` the
real units padded to the grid with fillers (`pad(N_real + 1)`), each proved alone (no closures). If every audit unit
does at most `w_max` (fillers included), the window realizes PoUW's units, and the window's one analysis at each oracle
and salt has knowledge and link terms at the prover's state, then `WindowBound` holds at
$$\beta(H, s) = \big(1 - W/(1000\,N\,w_{max})\big)^{K'} + \varepsilon_{ks}(H, s, R, \tau) + \delta_{link}(H, s, R, \tau),$$
`(R, τ)` the play's registration and state. A wrong set holding `W/1000` of incorrect work holds at least
`W/(1000·w_max)` units, which the draw misses with probability at most the first term. That is `k_prime`'s share form,
so it needs no work table, and `W ≥` the credited work makes the first term at most `k_prime`'s `δ` at `ε = 1/1000`.

The service's draw is one law for the whole window; the window's `ε_ks` and `δ_link` composed from its sessions' are
gap (6)'s (`WindowEps`). -/
def WindowBoundOfService : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (U : Workload)
    (N K : ℕ) (hK : K ≤ N) (m : ℕ) (σ : Fin N → Fin m) (w : Fin m → ℕ) (wmax : ℕ), (∀ t, w t ≤ wmax) →
    ∀ (Reg : Type) (session : (Q → R) → S → Finset (Fin N) → Reg → FlockSoundness.Game Bool)
      (An : (H : Q → R) → (s : S) → ExtractionAnalysis (Law.subset N K hK) Reg (session H s))
      (tx : (H : Q → R) → (s : S) → (An H s).Tr → Pouw.Transcript),
    (∀ H s, Realizes P U H s σ w (fun _ => ∅) (An H s).wrong (tx H s)) →
    ∀ εks δlink : (H : Q → R) → (s : S) → (R' : Reg) → Cont (Law.subset N K hK) Reg (session H s) R' → ℝ≥0∞,
    (∀ H s, (An H s).KnowledgeSound (εks H s)) → (∀ H s, (An H s).LinkSound (δlink H s)) →
    ∀ σ' : (H : Q → R) → (s : S) →
      FlockSoundness.Game.Strategy (auditReg (fun _ : Reg => Law.subset N K hK) (session H s)),
      WindowBound P U (fun H s τ => tx H s ((An H s).committedOf (atReg τ))) σ' fun H s =>
        ENNReal.ofReal ((1 - totalWref P U.layout / (1000 * (N : ℝ) * (wmax : ℝ))) ^ K) +
          εks H s (reg (atReg (σ' H s))) (cont (atReg (σ' H s))) +
          δlink H s (reg (atReg (σ' H s))) (cont (atReg (σ' H s)))

/-- **`WindowBound` at a law within `η` of C-Flock's law of record** (statement, no proof yet; to be proved as
`windowBound_of_record` from `FlockSoundness.Audit.extraction_audit_window_split_of_record_of_le_slack`). For the law
of record's sizing (#364: every call drawn by its own work law at its share of `K = 27,713`, y's floors), a law `L`
whose escapes are within `η` of it, drawn with closures `cl`, the work table (the audit units' work, fillers included,
is at most `W`), `Realizes`, and the window's one analysis at each oracle and salt with knowledge and link terms at the
prover's state, `WindowBound` holds at `β(H, s) = 2^-40 + η + ε_ks(H, s, R, τ) + δ_link(H, s, R, τ)`. The work table is
what the event needs: PoUW's `W/1000` is at least the window's 0.1% share only when the window's work is at most `W`. -/
def WindowBoundOfRecord : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (U : Workload)
    (n m C : ℕ) (σ : Fin n → Fin m) (call : Fin m → Fin C) (w v f : Fin m → ℕ) (cl : Fin n → Finset (Fin n)),
    0 < Law.totalWork σ w → (∀ t, 0 < w t → Law.callWork σ call w (call t) = w t * (Law.stratum σ t).card) →
    0 < Law.totalWork σ v → (∀ t, 27713 * v t * (Law.stratum σ t).card ≤ f t * Law.totalWork σ v) →
    (Law.totalWork σ w : ℝ) ≤ totalWref P U.layout →
    ∀ (L : Law n) (η : ℝ≥0∞), (∀ B, L.escape B ≤
      (Law.stratified σ (Law.windowK σ call w f 27713) (Law.windowK_le σ call w f 27713)).escape B + η) →
    ∀ (Reg : Type) (session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool)
      (An : (H : Q → R) → (s : S) → ExtractionAnalysis (L.closure cl) Reg (session H s))
      (tx : (H : Q → R) → (s : S) → (An H s).Tr → Pouw.Transcript),
    (∀ H s, Realizes P U H s σ w cl (An H s).wrong (tx H s)) →
    ∀ εks δlink : (H : Q → R) → (s : S) → (R' : Reg) → Cont (L.closure cl) Reg (session H s) R' → ℝ≥0∞,
    (∀ H s, (An H s).KnowledgeSound (εks H s)) → (∀ H s, (An H s).LinkSound (δlink H s)) →
    ∀ σ' : (H : Q → R) → (s : S) → FlockSoundness.Game.Strategy (auditReg (fun _ : Reg => L.closure cl) (session H s)),
      WindowBound P U (fun H s τ => tx H s ((An H s).committedOf (atReg τ))) σ' fun H s =>
        ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks H s (reg (atReg (σ' H s))) (cont (atReg (σ' H s))) +
          δlink H s (reg (atReg (σ' H s))) (cont (atReg (σ' H s)))

end Pouw.WindowWork

namespace Pouw.Guarantees

open Finset FlockSoundness.Audit
open scoped ENNReal

/-- **PoUW's work bound at the window audit** (statement only, no proof yet). A prover that passes the window audit
spent at least a `(1 − γ_u)(1 − 1/1000)` share of the window's declared work, less the extractor's cost, except with
probability `E[β] + ε_id(q, N) + ε(q + q_ext, N)`.

**What it bounds.** The prover `B` is a program of the prover model `PM` (`WindowWork.Prover`), and the audit runs
against its play. Its work `T` bounds the cost of its online phase, from the salt to the registration, in the cost
model's units, which the model fixes, not the prover. It is compared with the window's declared work
`W = ∑_{u < N} W_ref(u)` in `CM`'s units, `W_ref(u)` being the honest reference's cost of unit `u` without hash calls.
Averaged over the oracle `H` and the salt `s`, the audit accepts the play while `T + T_ext < (1 − γ_u)(1 − 1/1000)·W`
with probability at most
$$\mathbb{E}_{H,s}\Pr[\text{accept} \wedge T + T_{ext} < (1-\gamma_u)(1-10^{-3})\,W] \le \mathbb{E}_{H,s}\,\beta(H, s) +
  \varepsilon_{id}(q, N) + \varepsilon(q + q_{ext}, N).$$
The audit is `FlockSoundness.Audit.auditReg Lr (session H s)`, with any law `Lr R` at each registration `R`, so it holds
the service's draw and the law of record alike.

**What it assumes**, each a hypothesis:
* the work step `WindowWork.TileWork γ_u` with error `ε` (a named premise; TT's instances are vacuous at coordinate
  units, EPS1's is gap (3));
* `WindowWork.StraightLineExtraction` at `(T_ext, q_ext, ε_id)` (the named random-oracle premise): from `B`, a PoUW
  adversary within `(T + T_ext, q + q_ext)` whose transcript is the committed transcript of `B`'s play except with
  probability `ε_id`; with its antecedent, that the model's registrations are oracle programs' outputs
  (`ProverModel.RegQueries`);
* `WindowWork.WindowBound` for `B`'s play at `β(H, s)` (discharged by `WindowBoundOfService`, `WindowBoundOfRecord`, or
  gap (6)'s `WindowEps` through its bridge, each from `WindowWork.Realizes`, which holds by construction at coordinate
  units).

**What it composes**, at every oracle and salt: either the committed transcript isn't the extracted adversary's (on
average at most `ε_id`); or the work step's event holds for that adversary at budget `T + T_ext` (on average at most
`ε`); or the units correct in the committed transcript are worth at most `(T + T_ext)/(1 − γ_u) < (1 − 1/1000)·W`, so its
incorrect work exceeds `W/1000`, and the audit accepts with that at most `β(H, s)`. At `γ_u ≥ 1` the event is empty. -/
def EndToEndWindow : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γu : ℝ) (ε : ℕ → ℕ → ℝ),
    WindowWork.TileWork CM P D γu ε →
    -- the window audit: `n` units, the law at each registration, the session at each oracle and salt, and the
    -- analysis's committed transcript of a play, read as PoUW's transcript
    ∀ (n : ℕ) (Reg : Type) (Lr : Reg → Law n)
      (session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool)
      (committed : (H : Q → R) → (s : S) → FlockSoundness.Game.Strategy (auditReg Lr (session H s)) → Transcript)
    -- the prover model, and the extraction of PoUW's adversary from its provers
      (PM : WindowWork.ProverModel Q R S Reg Lr session) (Text : ℝ) (qext : ℕ) (εid : ℕ → ℕ → ℝ),
    PM.RegQueries → 0 ≤ Text → WindowWork.StraightLineExtraction PM CM D committed Text qext εid →
    ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q + qext ≤ 2 ^ 64 →
    ∀ B : WindowWork.Prover PM, B.Bounded U T q →
    -- the window audit's bound on the prover's play, at each oracle and salt
    ∀ β : (Q → R) → S → ℝ≥0∞, WindowWork.WindowBound P U committed (B.play U) β →
      FlockSoundness.Game.avg (fun x : (Q → R) × S =>
        FlockSoundness.Game.prob
          (fun o => o.1 = true ∧ T + Text < (1 - γu) * (1 - 1 / 1000) * totalWref P U.layout)
          (auditReg Lr (session x.1 x.2)) (B.play U x.1 x.2))
        ≤ FlockSoundness.Game.avg (fun x : (Q → R) × S => β x.1 x.2) + ENNReal.ofReal (εid q U.layout.N) +
          ENNReal.ofReal (ε (q + qext) U.layout.N)

end Pouw.Guarantees
