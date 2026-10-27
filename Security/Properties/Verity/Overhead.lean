import Properties.Verity.Soundness
import Definitions.Nci.Ldl
import Definitions.Accounting.Confidence
import Mathlib.Algebra.Order.Floor.Defs

/-!
# The end-to-end overhead guarantee (serving route)

The training lane's overhead definitions (`note:training-20261027T0206Z-report-overhead-definitions`, locked 26 Oct) and
the corrected theorem of `note:training-20261027T0134Z-draft-e2e-overhead-proof` §9, as one corollary of
`Vocab.Soundness`:

* a **task** `𝒢` (`Overhead.Task`): a draw of an input `x` and the set `𝒴` of its acceptable outputs;
* **success** (`Overhead.Succeeds`): the output a run holds at positions fixed before `x` lies in `𝒴`;
* **cost** is server-time (`Overhead.serverTime`): each server is occupied from tick `0` to one past its last computed
  gate, and the costs of the servers add. A computation's cost is its server-time on its worst input (`Overhead.cost`);
* the **threshold** `T_𝒢(η; 𝒜)` (`Overhead.threshold`): the least cost of a computation whose probability of succeeding
  inside the class `𝒜` exceeds `η`, and the **overhead** (`Overhead.overhead`) the ratio of two thresholds.

**One experiment** (`Overhead.prob`): the task's coin is drawn first and the prover sees `x`. Then the deployment's `B`
server-windows are verified in order (`Overhead.windows`). In each window the prover commits, adaptively, to the
parameters, the committed files, the commitment and a strategy of `Vocab.game` at them (`Overhead.Window`), and that
game is played at that strategy (`Overhead.follow`). Each window's outcome carries the strategy, so its claims
(`Vocab.Holds`) and errors (`Vocab.error`, which carry the collision terms) are read on the window that was actually
played. The actual run (`Accounting.Actual.Run`) and its counted products (`Nci.Ldl.Products`) are read off the input and
the windows played (`Overhead.Deployment`).

`Overhead` says: for every strategy and every per-window error budget `ε`, the probability that the verifier of record
accepts every window, the run succeeds and its server-time is below `T_min = ω (⌈(N − q_D)/q⌉ − n)` is at most
`Σ_b ε_b`, plus the probability that the prover commits to a window whose soundness and wipe errors exceed its budget,
plus the coverage exception `η`. Here `n` is the number of servers, `ω` the window length and `q` the per-window cap.
`− n` is F9's off-by-one, once per server: a server's last window may be partly idle.

Its premises are named:

* `Vocab.Soundness`, the one theorem (`Verity.Soundness` gives it from the five targets);
* `Overhead.Deployment.WipeSound`, the wipe's per-window target, until `Properties.Verity.Wipe` states it;
* `Overhead.Deployment.ServingCap`, the per-serving-block cap. On a window whose claims hold and whose wipe left at most
  its survivor bound, at most `q` counted products are materialized in it. That is the per-block summand of the
  serving-block bound (`Accounting.Guarantees.ServingBlocks.ServingBlockBound`, #4834, which #4869's
  `ServingBlockSum` may fold in), with `q(c_S, k_S)` priced by `Nci.Ldl.Assumptions.ActivationRate` through
  `Nci.Ldl.Guarantees.FanOutBound`'s step;
* `Phys`, the physical premises on every run: LEAN-70's `Accounting.FixedMachine H S T` and `Accounting.InitialBudget`,
  taken as one predicate until they reach this base (LEAN-70 states them over #4869);
* `Overhead.Deployment.DarkShare`, the dark fleet's share: at most `q_D` counted products are materialized outside
  every monitored server-window;
* `Overhead.Deployment.Materialization`, coverage in materialization form: for every strategy, success with fewer than
  `N` counted products materialized has probability at most `η`.

These definitions sit on main's `Verity.Circuit` and `Accounting.Placed`. LEAN-63 (#4819, #4869) moves them onto its
one circuit type and partition.
-/

namespace Verity

open Accounting
open scoped ENNReal

namespace Overhead

/-- **A static task** `𝒢` over inputs `In` and outputs `Y`: one uniform coin, drawn up front, and the input `x` and the
set `𝒴` of its acceptable outputs that it gives (`draw ω = (x, 𝒴)`). -/
structure Task (In Y : Type) where
  /-- the coin -/
  Ω : Type
  [fintype : Fintype Ω]
  [nonempty : Nonempty Ω]
  /-- the input and its acceptable outputs, at each value of the coin -/
  draw : Ω → In × Set Y

attribute [instance] Task.fintype Task.nonempty

/-- The task as a game: its one coin, then the pair it draws. -/
def Task.game {In Y : Type} (G : Task In Y) : Game (In × Set Y) := .coin G.Ω fun ω => .ret (G.draw ω)

/-- **The probability over the task** `Pr_{(x, 𝒴) ∼ 𝒢}[E x 𝒴]`. -/
noncomputable def Task.prob {In Y : Type} (G : Task In Y) (E : In → Set Y → Prop) : ℝ≥0∞ :=
  Game.prob (fun p => E p.1 p.2) G.game fun _ => PUnit.unit

/-- **The output at positions** `pos` (each a leaf, a cell and a tick) in the run `R` is `y`: each position holds a gate
whose value's bits are `y i`. -/
def Reads {L : Type} {m : ℕ} (R : Actual.Run L) (pos : Fin m → L × ℕ × ℕ) (y : Fin m → List Bool) : Prop :=
  ∀ i, ∃ g, (R.A.x g, R.A.cell g, R.A.t g) = pos i ∧ Actual.bits (R.A.w g) (R.val g).toNat = y i

/-- **Success**: the output the run `R` holds at the positions `pos`, fixed before the input is drawn, lies in `𝒴`. -/
def Succeeds {L : Type} {m : ℕ} (R : Actual.Run L) (pos : Fin m → L × ℕ × ℕ) (𝒴 : Set (Fin m → List Bool)) :
    Prop :=
  ∃ y ∈ 𝒴, Reads R pos y

open Classical in
/-- **A location's busy time** in the actual circuit `A`: one past the last tick at which a leaf of `Y` computes a gate,
and `0` when none does. -/
noncomputable def busy {L : Type} (A : Placed L) (Y : Finset L) : ℕ :=
  (A.computed.filter fun g => A.x g ∈ Y).sup fun g => A.t g + 1

/-- **Server-time**: the servers' busy times summed, each server `X s` occupied from tick `0` to one past its last
computed gate. -/
noncomputable def serverTime {L Srv : Type} [Fintype Srv] (X : Srv → Finset L) (A : Placed L) : ℕ :=
  ∑ s, busy A (X s)

/-- **A class** `𝒜` of computations over inputs `In`: the pairs of a computation (a run on each input) and an input
that are allowed. -/
abbrev Class (In L : Type) := Set ((In → Actual.Run L) × In)

/-- **The success probability** of the computation `C`, with outputs at `pos`, in the class `𝒜` on the task `𝒢`:
`Pr_{(x, 𝒴) ∼ 𝒢}[(C, x) ∈ 𝒜 ∧ C(x) ∈ 𝒴]`. -/
noncomputable def success {In L : Type} {m : ℕ} (G : Task In (Fin m → List Bool)) (𝒜 : Class In L)
    (C : In → Actual.Run L) (pos : Fin m → L × ℕ × ℕ) : ℝ≥0∞ :=
  G.prob fun x 𝒴 => (C, x) ∈ 𝒜 ∧ Succeeds (C x) pos 𝒴

/-- **The cost** of a computation: its server-time on its worst input. -/
noncomputable def cost {In L Srv : Type} [Fintype Srv] (X : Srv → Finset L) (C : In → Actual.Run L) : ℝ≥0∞ :=
  ⨆ x, (serverTime X (C x).A : ℝ≥0∞)

/-- **The threshold** `T_𝒢(η; 𝒜)`: the least cost of a computation, with its output positions, whose success
probability in `𝒜` exceeds `η`; `⊤` when none's does. -/
noncomputable def threshold {In L Srv : Type} [Fintype Srv] {m : ℕ} (G : Task In (Fin m → List Bool))
    (𝒜 : Class In L) (X : Srv → Finset L) (η : ℝ≥0∞) : ℝ≥0∞ :=
  ⨅ (C : In → Actual.Run L) (pos : Fin m → L × ℕ × ℕ) (_ : η < success G 𝒜 C pos), cost X C

/-- **The overhead** of the class `𝒜'` over the physical class `𝒜_phys` on `𝒢` at `η`: `T_𝒢(η; 𝒜') / T_𝒢(η; 𝒜_phys)`,
read where the denominator is in `(0, ∞)`. -/
noncomputable def overhead {In L Srv : Type} [Fintype Srv] {m : ℕ} (G : Task In (Fin m → List Bool))
    (𝒜 𝒜phys : Class In L) (X : Srv → Finset L) (η : ℝ≥0∞) : ℝ≥0∞ :=
  threshold G 𝒜 X η / threshold G 𝒜phys X η

/-- **A game played at a strategy**: the strategy's messages sent, the verifier's coins still drawn. -/
def follow {β : Type} : (g : Game β) → Game.Strategy g → Game β
  | .ret b, _ => .ret b
  | .send _ k, s => follow (k s.1) s.2
  | @Game.coin _ C iF iN k, s => @Game.coin _ C iF iN fun c => follow (k c) (s c)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- **A server-window's play, as the prover commits to it**: the parameters, the committed files, the commitment, and
a strategy of the soundness game at them. -/
structure Window (V : Vocab d α) where
  θ : ByteArray
  F : ByteArray
  c : ByteArray
  σ : Game.Strategy (V.game θ F c)

/-- **`n` server-windows verified in order**: in each the prover commits to a window, chosen after every earlier
window's coins, and `Vocab.game` is played at it. The outcome lists each window with its play's outcome, first window
first. -/
def windows (V : Vocab d α) : ℕ → Game (List (Window V × Outcome))
  | 0 => .ret []
  | n + 1 => .send (Window V) fun w =>
      (follow (V.game w.θ w.F w.c) w.σ).bind fun o => (windows V n).map ((w, o) :: ·)

/-- **The experiment's probability** at the strategy `τ`, which sees the task's input `x`: the task's coin, then the
`B` windows, and the event `E x 𝒴 ws` on the input, its acceptable outputs and the windows played. -/
noncomputable def prob (V : Vocab d α) {In Y : Type} (G : Task In Y) {B : ℕ} (τ : In → Game.Strategy (windows V B))
    (E : In → Set Y → List (Window V × Outcome) → Prop) : ℝ≥0∞ :=
  ∑ ω : G.Ω, (Fintype.card G.Ω : ℝ≥0∞)⁻¹ * Game.prob (E (G.draw ω).1 (G.draw ω).2) (windows V B) (τ (G.draw ω).1)

/-- **The acceptance `acc` holds at every one of the `B` windows**, each on a file with the registered digest. -/
def AcceptedBy (V : Vocab d α) (acc : Sent → Prop) (B : ℕ) (ws : List (Window V × Outcome)) : Prop :=
  ws.length = B ∧ ∀ e ∈ ws, acc e.2.sent ∧ V.Registered e.2.params e.2.file

/-- **A clean window**: every claim holds on its play (`Vocab.Holds` at the strategy committed to), and the wipe that
opened it left at most its survivor bound at fraction `f` (`Accounting.WipeLive.Survives`). -/
def Clean (V : Vocab d α) (wipe : Window V → WipeLive d Outcome) (f : ℝ) (e : Window V × Outcome) : Prop :=
  (∀ a, V.Holds e.1.θ e.1.F e.1.c e.1.σ a e.2) ∧ (wipe e.1).Survives f e.2

/-- **A window's error**: the subprotocols' errors at its strategy (`Vocab.error`, with their collision terms) and the
wipe's. -/
noncomputable def windowError (V : Vocab d α) (εWipe : Window V → ℝ≥0∞) (w : Window V) : ℝ≥0∞ :=
  ∑ a, V.error a w.θ w.F w.c w.σ + εWipe w

/-- **Gate `g` is in server-window `i`**: placed on its server `X s` during its ticks `[ω j, ω (j + 1))`, for
`(s, j) = win i`. -/
def InWindow {L Srv : Type} {B H : ℕ} (X : Srv → Finset L) (win : Fin B ≃ Srv × Fin H) (ω : ℕ) (A : Placed L)
    (i : Fin B) (g : Fin A.N) : Prop :=
  A.x g ∈ X (win i).1 ∧ ω * (win i).2 ≤ A.t g ∧ A.t g < ω * ((win i).2 + 1)

/-- **The serving route's deployment**, read off the experiment: the actual run on the input `x` with the windows `ws`
played, its counted product occurrences, the servers' locations, the order `win` in which the `B = n H` server-windows of
`ω` ticks are verified, and the wipe that opens each window. -/
structure Deployment (V : Vocab d α) (In L Srv ιP Act : Type) [Fintype Srv] (B : ℕ) where
  run : In → List (Window V × Outcome) → Actual.Run L
  prods : (x : In) → (ws : List (Window V × Outcome)) → Nci.Ldl.Products (run x ws).A.toCircuit ιP Act
  X : Srv → Finset L
  H : ℕ
  win : Fin B ≃ Srv × Fin H
  ω : ℕ
  wipe : Window V → WipeLive d Outcome

namespace Deployment

variable {V : Vocab d α} {In L Srv ιP Act : Type} [Fintype Srv] {B : ℕ} (D : Deployment V In L Srv ιP Act B)

open Classical in
/-- The counted products materialized in window `i`: their representative gates are in it. -/
noncomputable def inWindow (x : In) (ws : List (Window V × Outcome)) (i : Fin B) : Finset ιP :=
  (D.prods x ws).J.filter fun p => InWindow D.X D.win D.ω (D.run x ws).A i ((D.prods x ws).rep p)

open Classical in
/-- The counted products materialized outside every monitored server-window. -/
noncomputable def dark (x : In) (ws : List (Window V × Outcome)) : Finset ιP :=
  (D.prods x ws).J.filter fun p => ∀ i, ¬ InWindow D.X D.win D.ω (D.run x ws).A i ((D.prods x ws).rep p)

/-- PREMISE. **The wipe's target, per window**, at the acceptance `acc`: at every window, the probability that `acc`
holds on a registered file while the wipe's survivor bound at `f` fails is at most `εWipe`. The wipe's statement joins
in `Properties.Verity.Wipe`; this is its shape there. -/
def WipeSound (acc : Sent → Prop) (f : ℝ) (εWipe : Window V → ℝ≥0∞) : Prop :=
  ∀ w : Window V, Game.prob (fun o => acc o.sent ∧ V.Registered o.params o.file ∧ ¬ (D.wipe w).Survives f o)
    (V.game w.θ w.F w.c) w.σ ≤ εWipe w

/-- PREMISE. **The per-serving-block cap** `q`: on a run meeting the physical premises `Phys`, at most `q` counted
products are materialized in a window whose claims hold and whose wipe left at most its survivor bound. It is the
per-block summand of #4834's serving-block bound, `min {q(c_S, k_S) + ω_max μ k_S, n_S q(c, k) + q_U(S)}`. There
`q(c_S, k_S)` is the activation-priced cap `e (k_S + ⌈log₂ B⌉ + a_blk + Λ) / β` (`Nci.Ldl.Assumptions.ActivationRate`
through `Nci.Ldl.Guarantees.FanOutBound`'s step). `k_S` is what survives the wipe plus the certified ingress, and the
window's claims bound it. -/
def ServingCap (f : ℝ) (Phys : Actual.Run L → Prop) (q : ℝ) : Prop :=
  ∀ x ws (i : Fin B) e, ws[(i : ℕ)]? = some e → V.Registered e.2.params e.2.file → Clean V D.wipe f e →
    Phys (D.run x ws) → ((D.inWindow x ws i).card : ℝ) ≤ q

/-- PREMISE. **The dark share** `q_D`: at most `q_D` counted products are materialized outside every monitored
server-window (the dark fleet, 0.1% in the mainline; Daniel, 26 Oct). -/
def DarkShare (qD : ℝ) : Prop :=
  ∀ x ws, ((D.dark x ws).card : ℝ) ≤ qD

/-- PREMISE. **Coverage, materialization form**: for every strategy and output positions, the run succeeds with fewer
than `N` counted products materialized with probability at most `η`. -/
def Materialization {m : ℕ} (G : Task In (Fin m → List Bool)) (N : ℕ) (η : ℝ≥0∞) : Prop :=
  ∀ (pos : Fin m → L × ℕ × ℕ) (τ : In → Game.Strategy (windows V B)),
    prob V G τ (fun x 𝒴 ws => Succeeds (D.run x ws) pos 𝒴 ∧ (D.prods x ws).J.card < N) ≤ η

end Deployment

/-- **The least server-time of a run that materializes `N` counted products**, at a per-window cap `q`, a dark share
`q_D`, `n` servers and windows of `ω` ticks: `ω (⌈(N − q_D)/q⌉ − n)`. Each server's last window may be partly idle,
hence one window less per server. -/
noncomputable def minServerTime (ω N : ℕ) (q qD : ℝ) (n : ℕ) : ℝ :=
  ω * (((⌈((N : ℝ) - qD) / q⌉ : ℤ) : ℝ) - n)

end Overhead

open Overhead

/-- **The overhead guarantee at the acceptance `acc`.** On a deployment where `acc` meets soundness
(`Vocab.SoundnessFor`), the wipe its per-window target (`WipeSound`), every run the physical premises `Phys`, and the
serving-block cap, the dark share and materialization coverage hold, for every output positions, strategy and per-window
error budget `ε`:

`Pr[acc at every window ∧ success ∧ server-time < T_min] ≤ Σ_b ε_b + Pr[some window's error exceeds its budget] + η`.

A window's error is its subprotocols' errors at the strategy committed to, with their collision terms, plus the wipe's
(`windowError`). -/
def OverheadFor (acc : Sent → Prop) : Prop :=
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) {In L Srv ιP Act : Type} [Fintype Srv]
    {m B : ℕ} (G : Task In (Fin m → List Bool)) (D : Deployment V In L Srv ιP Act B) (f : ℝ)
    (εWipe : Window V → ℝ≥0∞) (Phys : Actual.Run L → Prop) (q qD : ℝ) (N : ℕ) (η : ℝ≥0∞),
    0 < D.ω → 0 < q →
    V.SoundnessFor acc → D.WipeSound acc f εWipe → (∀ x ws, Phys (D.run x ws)) → D.ServingCap f Phys q →
    D.DarkShare qD → D.Materialization G N η →
    ∀ (pos : Fin m → L × ℕ × ℕ) (τ : In → Game.Strategy (windows V B)) (ε : Fin B → ℝ≥0∞),
      prob V G τ (fun x 𝒴 ws => AcceptedBy V acc B ws ∧ Succeeds (D.run x ws) pos 𝒴 ∧
          (serverTime D.X (D.run x ws).A : ℝ) < minServerTime D.ω N q qD (Fintype.card Srv)) ≤
        ∑ i, ε i + prob V G τ (fun _ _ ws => ∃ i : Fin B, ∃ e, ws[(i : ℕ)]? = some e ∧ ε i < windowError V εWipe e.1) +
          η

/-- **The overhead guarantee at the verifier of record** (`Accepts`): from `Vocab.Soundness`, for every strategy, the
probability that the verifier of record accepts every server-window, the run succeeds and its server-time is below
`T_min` is at most the windows' error budgets summed, plus the probability that a window's soundness and wipe errors
exceed its budget, plus the coverage exception. -/
def Overhead : Prop := OverheadFor Accepts

/-- **The overhead guarantee needs the verifier** (rule 3, through `Verity.soundness_of_subprotocols_nontrivial_of`).
Take a window at which a claim fails, at a registered file, with probability above the subprotocols' summed errors.
There, soundness for the verifier that accepts everything fails. And at a one-window, one-server deployment whose
other premises hold, `OverheadFor`'s conclusion fails at that verifier. -/
def Overhead_nontrivial : Prop :=
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) (θ F c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)), V.Registered θ F → ∀ a : Audit,
    ∑ b, V.error b θ F c σ < Game.prob (fun o => ¬ V.Holds θ F c σ a o) (V.game θ F c) σ →
    ¬ V.SoundnessFor (fun _ => True) ∧
    ∃ (G : Task Unit (Fin 1 → List Bool)) (D : Deployment V Unit Unit Unit Unit Unit 1) (f : ℝ)
      (εWipe : Window V → ℝ≥0∞) (q qD : ℝ) (N : ℕ) (η : ℝ≥0∞) (pos : Fin 1 → Unit × ℕ × ℕ)
      (τ : Unit → Game.Strategy (windows V 1)) (ε : Fin 1 → ℝ≥0∞),
      0 < D.ω ∧ 0 < q ∧ D.WipeSound (fun _ => True) f εWipe ∧ D.ServingCap f (fun _ => True) q ∧
      D.DarkShare qD ∧ D.Materialization G N η ∧
      ¬ prob V G τ (fun x 𝒴 ws => AcceptedBy V (fun _ => True) 1 ws ∧ Succeeds (D.run x ws) pos 𝒴 ∧
          (serverTime D.X (D.run x ws).A : ℝ) < minServerTime D.ω N q qD (Fintype.card Unit)) ≤
        ∑ i, ε i +
          prob V G τ (fun _ _ ws => ∃ i : Fin 1, ∃ e, ws[(i : ℕ)]? = some e ∧ ε i < windowError V εWipe e.1) + η

/-- **The overhead guarantee's premises hold where it says something** (rule 3, honest). At a deployment, every premise
of `Overhead` holds and some strategy's run is accepted at every window and succeeds with probability `1`. Its instance
is the toy deployment, once soundness's instance of `Vocab` exists (soundness is staffing it). Until then this is stated
and not proved. -/
def Overhead_honest : Prop :=
  ∃ (ι : Type) (d : Device ι) (α : Type) (_ : Fintype α) (V : Vocab d α) (In L Srv ιP Act : Type) (_ : Fintype Srv)
    (m B : ℕ) (G : Task In (Fin m → List Bool)) (D : Deployment V In L Srv ιP Act B) (f : ℝ)
    (εWipe : Window V → ℝ≥0∞) (Phys : Actual.Run L → Prop) (q qD : ℝ) (N : ℕ) (η : ℝ≥0∞),
    0 < D.ω ∧ 0 < q ∧ V.Soundness ∧ D.WipeSound Accepts f εWipe ∧ (∀ x ws, Phys (D.run x ws)) ∧
    D.ServingCap f Phys q ∧ D.DarkShare qD ∧ D.Materialization G N η ∧
    ∃ (pos : Fin m → L × ℕ × ℕ) (τ : In → Game.Strategy (windows V B)),
      prob V G τ (fun x 𝒴 ws => AcceptedBy V Accepts B ws ∧ Succeeds (D.run x ws) pos 𝒴) = 1

end Verity
