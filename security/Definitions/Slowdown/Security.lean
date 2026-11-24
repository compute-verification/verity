module

public import Definitions.Registration.Physical
public import Definitions.Slowdown.Fit

/-!
# The security argument's three sets (trusted)

Security is `𝓗 ∩ 𝒜 ∩ ℬ = ∅`: no computation is physically feasible, consistent with what the audits account for, and
prohibited. Each set is a predicate on a computation `M` alone, at public parameters, and none reads another:

* **The frame** (`Frame`), public: the hardware with its limits, the developer's servers, the run's ticks cut into
  windows (one server, one interval between two cuts), and the windows the memory and bandwidth audits don't cover
  (`missed`).
* **A computation** (`Computation`): everything the machines did, as an actual circuit (`ResourceAccounting.Placed`)
  with what each gate uses of its location's compute and the link each wire runs on, the registered inputs, and the
  gates of the declared circuit (`declared`), which the audits' amounts are about.
* **`H F`**, the feasible computations: `ResourceAccounting.Feasible` on the frame's hardware.
* **`A F x_s x_b x_w y_b`**, what the audits account for: at each window they cover, the declared circuit's gates hold
  at least `x_s` bits on the window's server at the cut that opens it, and the window reads at least `x_b` bits of
  them, and at most `y_b` bits in all, from outside its server; at each window they miss, the declared circuit's gates
  do at least `x_w` work.
* **`B F p Y`**, the prohibited computations: those holding a reference training algorithm of more than `Y` counted
  multiplications that meets the recipe `p`'s premises (`Trains`).

`Slowdown.SecurityProofs.secure` is the emptiness, at `Y` the fit bound (`Frame.bound`) on what the limits less the
amounts leave (`Frame.unaccounted`).
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.Security

open Verity ResourceAccounting Finset

/-- **The frame**, public: the hardware `hw`, the developer's `n` pairwise disjoint servers, each a location of the
hardware, the ticks `[0, t)` cut into windows, and the windows `missed` that the memory and bandwidth audits don't
cover. Window `(i, j)` is the server `i` over the ticks between cuts `j` and `j + 1`. -/
structure Frame (L E : Type*) where
  hw : Hardware L E
  n : ℕ
  server : Fin n → Finset L
  disjoint : ∀ i j, i ≠ j → Disjoint (server i) (server j)
  unit : ∀ i, server i ∈ hw.units
  t : ℕ
  cuts : Cuts t
  missed : Finset (Fin (n * cuts.J))

namespace Frame

/-- **The frame of one server** `X`, a location of the hardware. -/
abbrev single {L E : Type*} (hw : Hardware L E) (X : Finset L) (hX : X ∈ hw.units) (t : ℕ) (cuts : Cuts t)
    (missed : Finset (Fin (1 * cuts.J))) : Frame L E where
  hw := hw
  n := 1
  server _ := X
  disjoint i j h := absurd (Subsingleton.elim i j) h
  unit _ := hX
  t := t
  cuts := cuts
  missed := missed

variable {L E : Type*} (F : Frame L E)

/-- The ticks of window `j`: `[τ j, τ (j + 1))`. -/
def span (j : Fin F.cuts.J) : Set ℕ := Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)

/-- `ω`: the longest window, in ticks. -/
def ω : ℕ := univ.sup fun j : Fin F.cuts.J => F.cuts.τ j.succ - F.cuts.τ j.castSucc

/-- `m`: the most bits a server carries across a tick. -/
def m : ℕ := univ.sup fun i => F.hw.M (F.server i)

/-- `b`: the most bits a server's links carry a tick. -/
noncomputable def b [Fintype E] : ℕ := univ.sup fun i => linkCapacity F.hw.ends F.hw.β (F.server i)

/-- `Φ`: the most work a server does a tick. -/
def Φ : ℕ := univ.sup fun i => F.hw.compute (F.server i)

end Frame

/-- **A computation**: the actual circuit `run` of everything the machines did, what each of its gates uses of its
location's compute and the link each wire runs on (`usage`), the input gates registered before the run, and the gates
of the declared circuit. -/
structure Computation (L E : Type*) where
  run : Placed L
  usage : Usage run E
  registered : Finset (Fin run.N)
  registered_input : ∀ a ∈ registered, run.isInput a = true
  declared : Finset (Fin run.N)

namespace Computation

variable {L E : Type*} [DecidableEq L] (M : Computation L E) (F : Frame L E)

/-- The gates of window `(i, j)`: the computed gates placed on server `i` during window `j`'s ticks. -/
noncomputable def window (i : Fin F.n) (j : Fin F.cuts.J) : Finset (Fin M.run.N) :=
  M.run.region (F.server i) (F.span j)

/-- The bits of `G` server `i` carries across the cut that opens window `j`. -/
noncomputable def carries (i : Fin F.n) (j : Fin F.cuts.J) (G : Finset (Fin M.run.N)) : ℕ :=
  M.run.stateIn (F.server i) (F.cuts.τ j.castSucc) G

/-- The bits of `G` window `(i, j)` reads from outside its server. -/
noncomputable def reads (i : Fin F.n) (j : Fin F.cuts.J) (G : Finset (Fin M.run.N)) : ℕ :=
  M.run.width (((M.run.into (M.window F i j)).filter fun g => M.run.x g ∉ F.server i) ∩ G)

/-- What the developer's servers did over `[0, t)`, as a circuit whose inputs are what it reads, with the registered
inputs its baseline (`Placed.declaration`). -/
noncomputable def circuit : RegisteredCircuit :=
  M.run.declaration (univ.biUnion F.server) F.t M.registered M.registered_input

end Computation

variable {L E : Type*}

/-- **`𝓗`, the feasible computations** on the frame's hardware: causality, the compute rate, memory and the links
(`ResourceAccounting.Feasible`). -/
def H (F : Frame L E) : Set (Computation L E) := {M | Feasible F.hw M.run M.usage}

/-- **`𝒜`, the computations consistent with what the audits account for**, at the amounts `x_s`, `x_b`, `x_w` and
`y_b`. At each window the memory and bandwidth audits cover, the declared circuit's gates hold at least `x_s` bits on
the window's server at the cut that opens it, and the window reads at least `x_b` bits of them from outside its
server and at most `y_b` bits in all from there (the network attester's count; every feasible computation reads at
most what the server's links carry). At each window they miss, the declared circuit's gates do at least `x_w` work. -/
def A [DecidableEq L] (F : Frame L E) (x_s x_b x_w y_b : ℕ) : Set (Computation L E) :=
  {M | ∀ i j,
    (finProdFinEquiv (i, j) ∉ F.missed →
      x_s ≤ M.carries F i j M.declared ∧ x_b ≤ M.reads F i j M.declared ∧
        M.run.ingress (F.server i) (F.span j) ≤ y_b) ∧
    (finProdFinEquiv (i, j) ∈ F.missed → x_w ≤ M.run.work M.usage.π (M.window F i j ∩ M.declared))}

/-- **A recipe**: the parameters of the premises on a reference training algorithm. `Nstar` is the fewest weights that
reach the target loss, `b_w` the bits a window reads for each weight it uses, `w_e` the bits it reads for each thread it
works on but didn't start, `reuse` the most times a thread uses one weight, and `o_s`, `o_b` and `o_w` the most the
algorithm shares with the declared circuit in one window: bits carried, bits read and counted multiplications. -/
structure Recipe where
  Nstar : ℕ
  b_w : ℝ
  w_e : ℝ
  reuse : ℕ
  o_s : ℕ
  o_b : ℕ
  o_w : ℕ

section Trains

variable [DecidableEq L] (F : Frame L E) (M : Computation L E)

/-- `Q` divides `c` by the frame's windows: its unit `(i, j)` is the gates `emb` sends into window `(i, j)`. -/
def Divides {c : RegisteredCircuit} (emb : Containment c.D (M.circuit F).D)
    (Q : Partition c.D (F.n * F.cuts.J)) : Prop :=
  ∀ i j g, g ∈ Q.gates (finProdFinEquiv (i, j)) ↔ emb.emb g ∈ M.window F i j

open Classical in
/-- **The reference training algorithm `c` trains inside `M`** at the recipe `p`: `c`, threaded by `T`, is contained
(`emb`) in what the developer's servers did, each thread `u` started by the window `start u`, and

* `reuse`: a thread uses one weight at most `p.reuse` times;
* `load`: each counted multiplication uses at least one unit of its location's compute;
* `reach` (F3, the loss premise, `Assumptions.Reach` counted in the covered windows): each thread started in a covered
  window that works there uses at least `p.Nstar` distinct weights inside the covered windows;
* `fits` (F1 and F4, `Assumptions.Fits` at the covered windows): each covered window reads at least `p.b_w` fresh input
  bits for each weight it uses and `p.w_e` for each thread it works on but didn't start;
* `shared_carried`, `shared_read`, `shared_work`: of `c`'s novel gates, those outside its registered inputs, the
  declared circuit's gates hold at most `p.o_s` bits carried into a covered window and `p.o_b` bits it reads from
  outside its server, and at most `p.o_w` of `c`'s counted multiplications in a missed window are the declared
  circuit's gates.

`reach` and `fits` are empirical premises about training, with `Assumptions.Reach`'s and `Assumptions.Fits`'s failure
cases. The three `shared` premises are zero when no gate is both the declared circuit's and `c`'s. -/
structure Trains (p : Recipe) (c : RegisteredCircuit) (emb : Containment c.D (M.circuit F).D) (T : Threading c)
    (start : Fin T.m → Fin (F.n * F.cuts.J)) : Prop where
  reuse : T.reuse ≤ p.reuse
  load : ∀ g ∈ T.M, 1 ≤ M.usage.π (emb.emb g)
  reach : ∀ Q : Partition c.D (F.n * F.cuts.J), Divides F M emb Q → ∀ u, start u ∉ F.missed →
    (T.threadMults (Q.gates (start u)) u).Nonempty →
      p.Nstar ≤ (T.threadWeights (univ.filter fun g => ∀ j ∈ F.missed, g ∉ Q.gates j) u).card
  fits : ∀ Q : Partition c.D (F.n * F.cuts.J), Divides F M emb Q → ∀ j ∉ F.missed,
    p.b_w * (T.weightsUsed (Q.gates j)).card + p.w_e * (T.entered Q start j).card ≤ freshInput c (Q.gates j)
  shared_carried : ∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
    M.carries F i j (M.declared ∩ (c.θᶜ).map ⟨emb.emb, emb.emb_inj⟩) ≤ p.o_s
  shared_read : ∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
    M.reads F i j (M.declared ∩ (c.θᶜ).map ⟨emb.emb, emb.emb_inj⟩) ≤ p.o_b
  shared_work : ∀ i j, finProdFinEquiv (i, j) ∈ F.missed →
    (T.M.filter fun g => emb.emb g ∈ M.window F i j ∩ M.declared).card ≤ p.o_w

/-- **`ℬ`, the prohibited computations** at the recipe `p` and the count `Y`: those in which some reference training
algorithm trains (`Trains`) with more than `Y` counted multiplications. -/
def B (p : Recipe) (Y : ℝ) : Set (Computation L E) :=
  {M | ∃ (c : RegisteredCircuit) (emb : Containment c.D (M.circuit F).D) (T : Threading c)
    (start : Fin T.m → Fin (F.n * F.cuts.J)), Trains F M p c emb T start ∧ Y < T.M.card}

end Trains

namespace Frame

variable [Fintype E] (F : Frame L E)

/-- **The bits of a reference training algorithm a covered window can read**: what the memory limit leaves beside the
`x_s` bits accounted for, what the smaller of the links' capacity and the `y_b` bits that entered leaves beside the
`x_b` bits accounted for, and what the algorithm shares with the declared circuit. -/
noncomputable def unaccounted (x_s x_b y_b : ℕ) (p : Recipe) : ℕ :=
  (F.m - x_s + p.o_s) + (min (F.b * F.ω) y_b - x_b + p.o_b)

/-- **The fit bound**: the counted multiplications a computation can hold of a reference training algorithm, when a
covered window reads at most `S' = unaccounted` bits of it: `n J · reuse · N* S'² / (4 w_e (N* b_w − S'))` over the
windows, and for each missed window the work its server's limit leaves beside the `x_w` accounted for, with the
multiplications shared with the declared circuit. -/
noncomputable def bound (x_s x_b x_w y_b : ℕ) (p : Recipe) : ℝ :=
  (F.n * F.cuts.J : ℕ) * p.reuse * p.Nstar * (F.unaccounted x_s x_b y_b p : ℝ) ^ 2 /
      (4 * p.w_e * (p.Nstar * p.b_w - (F.unaccounted x_s x_b y_b p : ℝ))) +
    (F.missed.card * (F.Φ * F.ω - x_w + p.o_w) : ℕ)

end Frame

end Slowdown.Security
