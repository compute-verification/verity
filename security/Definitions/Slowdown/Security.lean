module

public import Definitions.Registration.Physical
public import Definitions.Slowdown.Fit

/-!
# The security argument's three sets (trusted)

Security is `𝓗 ∩ 𝒜 ∩ ℬ = ∅`: no computation is physically feasible, consistent with what the audits account for, and
prohibited. Each set is a predicate on a computation `M` alone, at public parameters, and none reads another. `𝒜`
and `ℬ` read the frame and nothing of the hardware, and `𝓗` the hardware and nothing of the frame, so what a run
establishes, `M ∈ 𝒜`, stands before a hardware model or a training bound is chosen:

* **The frame** (`Frame`), public: the developer's servers, the run's ticks cut into windows (one server, one
  interval between two cuts), and the windows the memory and bandwidth audits don't cover (`missed`). It holds no
  hardware; `Frame.On` says its servers are locations of one.
* **A computation** (`Computation`): everything the machines did, as a run (`ResourceAccounting.Actual.Run`: the
  actual circuit, its gates' values and what each leaf holds at each tick) with what each gate uses of its location's
  compute and the link each wire runs on, the registered inputs, and the gates of the declared circuit (`declared`),
  which the audits' amounts are about.
* **`H hw`**, the feasible computations, four clauses of the hardware alone: the latency between a wire's two gates,
  the compute rate, memory on what the run holds, and the links.
* **`A F x_s x_b x_w y_b`**, what the audits account for: at each window they cover, the window's server holds at
  least `x_s` bits of the declared circuit's gates at the cut that opens it, and during the window the links deliver
  to it at least `x_b` bits of them and at most `y_b` bits in all; at each window they miss, the declared circuit's
  gates do at least `x_w` work.
* **`B F tb sh Y`**, the prohibited computations: those holding a reference training algorithm of more than `Y`
  counted multiplications that meets the premises of the training bound `tb` (`Trains`) and shares at most `sh` with
  the declared circuit.
* **A training bound** (`Training`): premises on a reference training algorithm, and the count they give for windows
  that each read at most `S` bits of it. It is the one place a route to a bound on training enters: the fit bound is
  `Training.fit` at a recipe, and another route is another value, proved to hold (`Training.Holds`) once.

`Slowdown.Guarantees.RunAccounted` is what a run establishes: the verifier accepts a run whose computation is outside
`𝒜` with probability at most the audits' errors together. `Slowdown.Guarantees.Secure` is the emptiness,
`M ∈ 𝓗 ∧ M ∈ 𝒜 → M ∉ ℬ`, at any hardware the frame's servers are locations of, any training bound that holds and
`Y` its count (`Frame.bound`) on what the hardware's limits less the amounts leave (`Frame.unaccounted`).
`Slowdown.Guarantees.RunSecure` is the two together.
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.Security

open Verity ResourceAccounting Finset

/-- **The frame**, public: the developer's `n` pairwise disjoint servers, the ticks `[0, t)` cut into windows, and
the windows `missed` that the memory and bandwidth audits don't cover. Window `(i, j)` is the server `i` over the
ticks between cuts `j` and `j + 1`. -/
structure Frame (L : Type*) where
  n : ℕ
  server : Fin n → Finset L
  disjoint : ∀ i j, i ≠ j → Disjoint (server i) (server j)
  t : ℕ
  cuts : Cuts t
  missed : Finset (Fin (n * cuts.J))

namespace Frame

/-- **The frame of one server** `X`. -/
abbrev single {L : Type*} (X : Finset L) (t : ℕ) (cuts : Cuts t) (missed : Finset (Fin (1 * cuts.J))) : Frame L where
  n := 1
  server _ := X
  disjoint i j h := absurd (Subsingleton.elim i j) h
  t := t
  cuts := cuts
  missed := missed

variable {L E : Type*} (F : Frame L) (hw : Hardware L E)

/-- **The frame is on the hardware** `hw`: each of its servers is a location of `hw`, one its limits are stated at. -/
def On : Prop := ∀ i, F.server i ∈ hw.units

/-- The ticks of window `j`: `[τ j, τ (j + 1))`. -/
def span (j : Fin F.cuts.J) : Set ℕ := Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)

/-- The length of window `j`, in ticks. -/
def len (j : Fin F.cuts.J) : ℕ := F.cuts.τ j.succ - F.cuts.τ j.castSucc

/-- `ω`: the longest window, in ticks. -/
def ω : ℕ := univ.sup F.len

/-- `m`: the most bits a server carries across a tick, on the hardware `hw`. -/
def m : ℕ := univ.sup fun i => hw.M (F.server i)

/-- `b`: the most bits a server's links carry a tick, on the hardware `hw`. -/
noncomputable def b [Fintype E] : ℕ := univ.sup fun i => linkCapacity hw.ends hw.β (F.server i)

/-- `Φ`: the most work a server does a tick, on the hardware `hw`. -/
def Φ : ℕ := univ.sup fun i => hw.compute (F.server i)

end Frame

/-- **A computation**: the run of everything the machines did (its actual circuit `run.A`, its gates' values and what
each leaf holds at each tick), what each of its gates uses of its location's compute and the link each wire runs on
(`usage`), the input gates registered before the run, and the gates of the declared circuit. -/
structure Computation (L E : Type*) where
  run : Actual.Run L
  usage : Usage run.A E
  registered : Finset (Fin run.A.N)
  registered_input : ∀ a ∈ registered, run.A.isInput a = true
  declared : Finset (Fin run.A.N)

namespace Computation

variable {L E : Type*} [DecidableEq L] (M : Computation L E) (F : Frame L)

/-- The gates of window `(i, j)`: the computed gates placed on server `i` during window `j`'s ticks. -/
noncomputable def window (i : Fin F.n) (j : Fin F.cuts.J) : Finset (Fin M.run.A.N) :=
  M.run.A.region (F.server i) (F.span j)

/-- The bits of `G` server `i` holds at the cut that opens window `j`: the part in `G` of its state there, what its
leaves hold and what is in flight between them (`Actual.Run.state`). A held bit counts whether or not a gate reads it
again. -/
noncomputable def holds (i : Fin F.n) (j : Fin F.cuts.J) (G : Finset (Fin M.run.A.N)) : ℕ :=
  M.run.A.width (M.run.state (F.server i) (F.cuts.τ j.castSucc) ∩ G)

/-- The bits of `G` server `i` carries across the cut that opens window `j`: those of its state there that its gates
read at the cut or later. -/
noncomputable def carries (i : Fin F.n) (j : Fin F.cuts.J) (G : Finset (Fin M.run.A.N)) : ℕ :=
  M.run.A.stateIn (F.server i) (F.cuts.τ j.castSucc) G

/-- The bits of `G` window `(i, j)` reads from outside its server. -/
noncomputable def reads (i : Fin F.n) (j : Fin F.cuts.J) (G : Finset (Fin M.run.A.N)) : ℕ :=
  M.run.A.width (((M.run.A.into (M.window F i j)).filter fun g => M.run.A.x g ∉ F.server i) ∩ G)

/-- The bits of `G` the links deliver to server `i` during window `j`: at each of the window's ticks, the bits of `G`
the server's gates placed at that tick read from outside it. A bit read at two ticks is delivered twice. -/
noncomputable def delivered (i : Fin F.n) (j : Fin F.cuts.J) (G : Finset (Fin M.run.A.N)) : ℕ :=
  ∑ t ∈ Finset.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ),
    M.run.A.width (((M.run.A.into (M.run.A.region (F.server i) (Set.Ico t (t + 1)))).filter
      fun g => M.run.A.x g ∉ F.server i) ∩ G)

/-- What the developer's servers did over `[0, t)`, as a circuit whose inputs are what it reads, with the registered
inputs its baseline (`Placed.declaration`). -/
noncomputable def circuit : RegisteredCircuit :=
  M.run.A.declaration (univ.biUnion F.server) F.t M.registered M.registered_input

end Computation

variable {L E : Type*}

/-- **`𝓗`, the feasible computations** on the hardware `hw`, four clauses: a gate reads only gates placed no later
than the latency between their leaves allows (`Placed.Causal`); the gates at a location over an interval of ticks use
at most the location's compute a tick (`Placed.ComputeRate`); each location's state, what its leaves hold and what is
in flight between them, is within the location's memory at every tick (`Actual.Run.Memory`); and each link delivers at
most its capacity a tick (`Placed.Links`). `Guarantees.Secure` reads of the first clause
only that a gate reads no gate placed after it; the latency is what a timed audit's deadline rests on. It reads the
hardware's locations, their latencies, compute and memory, and its links with their capacities, and nothing of the
audits' frame. -/
def H (hw : Hardware L E) : Set (Computation L E) :=
  {M | M.run.A.Causal hw.ℓ ∧ M.run.A.ComputeRate hw.units M.usage.π hw.compute ∧
    M.run.Memory hw.units hw.M ∧ M.run.A.Links hw.ends hw.β M.usage.link}

section Clauses

variable [DecidableEq L] (F : Frame L) (i : Fin F.n) (j : Fin F.cuts.J)

/-- **Memory's clause of `𝒜`** at window `(i, j)`: the window's server holds at least `x_s` bits of the declared
circuit's gates at the cut that opens it (`Computation.holds`). -/
def HeldAtLeast (x_s : ℕ) : Set (Computation L E) := {M | x_s ≤ M.holds F i j M.declared}

/-- **Bandwidth's clause of `𝒜`** at window `(i, j)`: during the window the links deliver to its server at least
`x_b` bits of the declared circuit's gates, a bit counted at each tick it is read (`Computation.delivered`). -/
def DeliveredAtLeast (x_b : ℕ) : Set (Computation L E) := {M | x_b ≤ M.delivered F i j M.declared}

/-- **The count's clause of `𝒜`** at window `(i, j)`: during the window the links deliver to its server at most
`y_b` bits in all (the network attester's count; every feasible computation is delivered at most what the server's
links carry). -/
def DeliveredAtMost (y_b : ℕ) : Set (Computation L E) := {M | M.delivered F i j univ ≤ y_b}

/-- **Work's clause of `𝒜`** at window `(i, j)`: the declared circuit's gates do at least `x_w` work in the
window. -/
def WorkAtLeast (x_w : ℕ) : Set (Computation L E) :=
  {M | x_w ≤ M.run.A.work M.usage.π (M.window F i j ∩ M.declared)}

end Clauses

/-- **`𝒜`, the computations consistent with what the audits account for**, at the amounts `x_s`, `x_b`, `x_w` and
`y_b`: at each window the memory and bandwidth audits cover, memory's clause, bandwidth's and the count's
(`HeldAtLeast`, `DeliveredAtLeast`, `DeliveredAtMost`), and at each window they miss, work's (`WorkAtLeast`). Each
clause is what one audit's guarantee concludes of an accepted run's computation, and none reads a hardware model. -/
def A [DecidableEq L] (F : Frame L) (x_s x_b x_w y_b : ℕ) : Set (Computation L E) :=
  {M | ∀ i j,
    (finProdFinEquiv (i, j) ∉ F.missed →
      M ∈ HeldAtLeast F i j x_s ∧ M ∈ DeliveredAtLeast F i j x_b ∧ M ∈ DeliveredAtMost F i j y_b) ∧
    (finProdFinEquiv (i, j) ∈ F.missed → M ∈ WorkAtLeast F i j x_w)}

/-- **What a reference training algorithm shares with the declared circuit** in one window, at most: `o_s` bits
carried, `o_b` bits read and `o_w` counted multiplications. All three are zero when no gate is both the declared
circuit's and the algorithm's. -/
structure Shared where
  o_s : ℕ
  o_b : ℕ
  o_w : ℕ

/-- **A training bound**: premises on a reference training algorithm and the count they give. The algorithm is a
circuit `c` threaded by `T` and divided into `k` windows by a partition, each thread `u` started by the window
`start u`, with a set `I` of windows left out. `Prem` is the premises, which read nothing of the hardware or the
audits; `bound k S` is the most counted multiplications `k` windows hold when each reads at most `S` unregistered
input bits; and `Valid S` is the `S` at which the bound is defined. `Secure` is stated at any training bound that
holds (`Training.Holds`), so a route to a bound on training is one value of this type with that proof. -/
structure Training where
  Prem : ∀ {c : RegisteredCircuit} (T : Threading c) {k : ℕ}, Partition c.D k → Finset (Fin k) →
    (Fin T.m → Fin k) → Prop
  Valid : ℝ → Prop
  bound : ℕ → ℝ → ℝ
  bound_zero : ∀ S, bound 0 S = 0

open Classical in
/-- **The training bound holds**: for every circuit `c` threaded by `T`, every division `Q` of it into `k`
windows and every set `I` of windows left out, where the premises hold and each window outside `I` reads at most `S`
unregistered input bits, at an `S` the bound is defined at, the counted multiplications outside `I`'s windows number
at most `bound (k − |I|) S`. It is a statement about circuits alone. -/
def Training.Holds (tb : Training) : Prop :=
  ∀ {c : RegisteredCircuit} (T : Threading c) {k : ℕ} (Q : Partition c.D k) (I : Finset (Fin k))
    (start : Fin T.m → Fin k) (S : ℝ), tb.Valid S → tb.Prem T Q I start →
    (∀ j ∉ I, (freshInput c (Q.gates j) : ℝ) ≤ S) →
    ((T.M.filter fun g => ∀ j ∈ I, g ∉ Q.gates j).card : ℝ) ≤ tb.bound (k - I.card) S

/-- **A recipe**: the parameters of the fit bound's premises. `Nstar` is the fewest weights that reach the target
loss, `b_w` the bits a window reads for each weight it uses, `w_e` the bits it reads for each thread it works on but
didn't start, and `reuse` the most times a thread uses one weight. -/
structure Recipe where
  Nstar : ℕ
  b_w : ℝ
  w_e : ℝ
  reuse : ℕ

open Classical in
/-- **The fit bound**, a training bound at the recipe `p`. Its premises:

* a thread uses one weight at most `p.reuse` times;
* F3, the loss premise (`Assumptions.Reach` counted outside `I`): each thread started in a window outside `I` that
  works there uses at least `p.Nstar` distinct weights outside `I`'s windows;
* F1 and F4 (`Assumptions.Fits` at the windows outside `I`): each window outside `I` reads at least `p.b_w`
  unregistered input bits for each weight it uses and `p.w_e` for each thread it works on but didn't start.

The second and third are empirical premises about training, with `Assumptions.Reach`'s and `Assumptions.Fits`'s
failure cases. The count is `reuse · N* S² / (4 w_e (N* b_w − S))` a window, defined where `b_w ≥ 0`, `w_e > 0` and
`0 ≤ S < N* b_w`. -/
noncomputable abbrev Training.fit (p : Recipe) : Training where
  Prem := fun {c} T {_} Q I start => T.reuse ≤ p.reuse ∧
    (∀ u, start u ∉ I → (T.threadMults (Q.gates (start u)) u).Nonempty →
      p.Nstar ≤ (T.threadWeights (univ.filter fun g => ∀ j ∈ I, g ∉ Q.gates j) u).card) ∧
    ∀ j ∉ I, p.b_w * (T.weightsUsed (Q.gates j)).card + p.w_e * (T.entered Q start j).card ≤
      freshInput c (Q.gates j)
  Valid S := 0 ≤ p.b_w ∧ 0 < p.w_e ∧ 0 ≤ S ∧ S < p.Nstar * p.b_w
  bound k S := k * p.reuse * p.Nstar * S ^ 2 / (4 * p.w_e * (p.Nstar * p.b_w - S))
  bound_zero S := by simp

section Trains

variable [DecidableEq L] (F : Frame L) (M : Computation L E)

/-- `Q` divides `c` by the frame's windows: its unit `(i, j)` is the gates `emb` sends into window `(i, j)`. -/
def Divides {c : RegisteredCircuit} (emb : Containment c.D (M.circuit F).D)
    (Q : Partition c.D (F.n * F.cuts.J)) : Prop :=
  ∀ i j g, g ∈ Q.gates (finProdFinEquiv (i, j)) ↔ emb.emb g ∈ M.window F i j

open Classical in
/-- **The reference training algorithm `c` trains inside `M`** at the training bound `tb`, sharing at most `sh` with
the declared circuit: `c`, threaded by `T`, is contained (`emb`) in what the developer's servers did, each thread `u`
started by the window `start u`, and

* `load`: each counted multiplication uses at least one unit of its location's compute;
* `prem`: the training bound's premises hold of `c` divided by the frame's windows, with the missed windows left out;
* `shared_carried`, `shared_read`, `shared_work`: of `c`'s novel gates, those outside its registered inputs, the
  declared circuit's gates hold at most `sh.o_s` bits carried into a covered window and `sh.o_b` bits it reads from
  outside its server, and at most `sh.o_w` of `c`'s counted multiplications in a missed window are the declared
  circuit's gates. -/
structure Trains (tb : Training) (sh : Shared) (c : RegisteredCircuit) (emb : Containment c.D (M.circuit F).D)
    (T : Threading c) (start : Fin T.m → Fin (F.n * F.cuts.J)) : Prop where
  load : ∀ g ∈ T.M, 1 ≤ M.usage.π (emb.emb g)
  prem : ∀ Q : Partition c.D (F.n * F.cuts.J), Divides F M emb Q → tb.Prem T Q F.missed start
  shared_carried : ∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
    M.carries F i j (M.declared ∩ (c.θᶜ).map ⟨emb.emb, emb.emb_inj⟩) ≤ sh.o_s
  shared_read : ∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
    M.reads F i j (M.declared ∩ (c.θᶜ).map ⟨emb.emb, emb.emb_inj⟩) ≤ sh.o_b
  shared_work : ∀ i j, finProdFinEquiv (i, j) ∈ F.missed →
    (T.M.filter fun g => emb.emb g ∈ M.window F i j ∩ M.declared).card ≤ sh.o_w

/-- **`ℬ`, the prohibited computations** at the training bound `tb`, the shared amounts `sh` and the count `Y`:
those in which some reference training algorithm trains (`Trains`) with more than `Y` counted multiplications. -/
def B (tb : Training) (sh : Shared) (Y : ℝ) : Set (Computation L E) :=
  {M | ∃ (c : RegisteredCircuit) (emb : Containment c.D (M.circuit F).D) (T : Threading c)
    (start : Fin T.m → Fin (F.n * F.cuts.J)), Trains F M tb sh c emb T start ∧ Y < T.M.card}

end Trains

namespace Frame

variable [Fintype E] (F : Frame L) (hw : Hardware L E)

/-- **The bits of a reference training algorithm a covered window can read** on the hardware `hw`: what the memory
limit leaves beside the `x_s` bits accounted for, what the smaller of the links' capacity and the `y_b` bits that
entered leaves beside the `x_b` bits accounted for, and what the algorithm shares with the declared circuit. -/
noncomputable def unaccounted (x_s x_b y_b : ℕ) (sh : Shared) : ℕ :=
  (F.m hw - x_s + sh.o_s) + (min (F.b hw * F.ω) y_b - x_b + sh.o_b)

/-- **The bound on training** on the hardware `hw`: the counted multiplications a computation can hold of a reference
training algorithm. The covered windows hold at most the training bound's count at `S' = unaccounted` bits read a window
(`tb.bound`), and each missed window the work its server's limit leaves over the window's ticks beside the `x_w`
accounted for, with the multiplications shared with the declared circuit. -/
noncomputable def bound (x_s x_b x_w y_b : ℕ) (tb : Training) (sh : Shared) : ℝ :=
  tb.bound (F.n * F.cuts.J - F.missed.card) (F.unaccounted hw x_s x_b y_b sh : ℕ) +
    (∑ u ∈ F.missed, (F.Φ hw * F.len (finProdFinEquiv.symm u).2 - x_w + sh.o_w) : ℕ)

/-- **The bound on training is defined and at most `Y`**: where the memory and bandwidth audits cover a window, the
training bound is defined at what a covered window can read (`tb.Valid`); and the bound is at most `Y`. Where they
miss every window the bound is the missed windows' sum alone, and nothing is asked of the training bound. -/
def BoundAtMost (x_s x_b x_w y_b : ℕ) (tb : Training) (sh : Shared) (Y : ℝ) : Prop :=
  (F.missed ≠ univ → tb.Valid (F.unaccounted hw x_s x_b y_b sh : ℕ)) ∧ F.bound hw x_s x_b x_w y_b tb sh ≤ Y

end Frame

end Slowdown.Security
