import Properties.Verity.Soundness
import Definitions.Accounting.Overhead
import Definitions.Nci.Ldl
import Mathlib.Algebra.Order.Floor.Defs

/-!
# The end-to-end overhead guarantee (serving route)

The end-to-end theorem, as one corollary of `Vocab.Soundness`, read in the task, success, server-time and threshold of
`Accounting.Overhead` (`Task`, `Succeeds`, `serverTime`, `threshold`, `overhead`).

**One experiment** (`Overhead.prob`): the task's coin is drawn first and the prover sees `x`. Then the deployment's `B`
server-windows are verified in order (`Overhead.windows`). In each window the prover commits, adaptively, to the
parameters, the committed files, the commitment and a strategy of `Vocab.game` at them (`Overhead.Window`), and that
game is played at that strategy (`Overhead.follow`). Each window's outcome carries the strategy, so its claims
(`Vocab.Holds`) and errors (`Vocab.error`, which carry the collision terms) are read on the window that was actually
played. The actual run (`Accounting.Actual.Run`) and its counted products (`Nci.Ldl.Products`) are read off the input and
the windows played (`Overhead.Deployment`).

`Overhead` says (`Overhead.Bound`): for every strategy and every per-window error budget `ε`, the probability that the
verifier of record accepts every window, the run succeeds and its server-time is below
`T_min = ω (⌈(N − q_D)/q⌉ − n)` is at most `Σ_b ε_b`, plus the probability that the prover commits to a window whose
soundness and wipe errors exceed its budget, plus the coverage exception `η`. Here `n` is the number of servers, `ω` the
window length and `q` the per-window cap. `− n` is one window per server: a server's last window may be partly idle.

Its premises are named:

* `Vocab.Soundness`, the one theorem (`Verity.Soundness` gives it from the five targets);
* `Overhead.Deployment.WipeSound`, the wipe's per-window target, until `Properties.Verity.Wipe` states it;
* `Overhead.Deployment.ServingCap`, the per-serving-block cap. On a window whose claims hold and whose wipe left at most
  its survivor bound, at most `q` counted products are materialized in it. That is the per-block summand of the
  serving-block bound, with `q(c_S, k_S)` priced by `Nci.Ldl.Guarantees.PairBound` under `Nci.Ldl.Assumptions.HeldRate`;
* `Phys`, the physical premises on every run (the fleet's fixed-machine dynamics and its chosen-state budget), taken as
  one predicate until they are stated;
* `Overhead.Deployment.DarkShare`, the dark fleet's share: at most `q_D` counted products are materialized outside
  every monitored server-window;
* `Overhead.Deployment.Materialization`, coverage in materialization form: for every strategy, success with fewer than
  `N` counted products materialized has probability at most `η`.

`Overhead_nontrivial` is closed, at an explicit instance: the toy vocabulary `Overhead.toyVocab` (a device with nothing,
every subprotocol's error `0`, and PoUW's claim false on every play, since no gate is computed and its certificate asks
for one), the one-window deployment `Overhead.oneDeployment`, whose run succeeds exactly when that claim fails, and the
verifier that accepts everything. There `Bound` fails. The proofs show beside it that every other premise of
`OverheadFor` holds at that instance and that soundness for that verifier fails.

These definitions sit on `Verity.Circuit` and `Accounting.Placed`.
-/

namespace Verity

open Accounting Accounting.Overhead
open scoped ENNReal

namespace Overhead

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
per-block summand of the serving-block bound, `min {q(c_S, k_S) + ω_max μ k_S, n_S q(c, k) + q_U(S)}`. There
`q(c_S, k_S)` is priced by `Nci.Ldl.Guarantees.PairBound` under `Nci.Ldl.Assumptions.HeldRate`: each (weight, entry) pair
counted once, a block holds at most `h² / (4 β_w β_e)` counted products, `h = k_S + ⌈log₂ B⌉ + a_blk + ℓ`.
`Nci.Ldl.Assumptions.ActivationRate`, and `Nci.Ldl.Guarantees.FanOutBound` through it, don't price it: a serving block
recomputes its activations from the layers it holds. `k_S` is what survives the wipe plus the certified ingress, and the
window's claims bound it. -/
def ServingCap (f : ℝ) (Phys : Actual.Run L → Prop) (q : ℝ) : Prop :=
  ∀ x ws (i : Fin B) e, ws[(i : ℕ)]? = some e → V.Registered e.2.params e.2.file → Clean V D.wipe f e →
    Phys (D.run x ws) → ((D.inWindow x ws i).card : ℝ) ≤ q

/-- PREMISE. **The dark share** `q_D`: at most `q_D` counted products are materialized outside every monitored
server-window (the dark fleet, 0.1% in the mainline). -/
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

/-- **The overhead bound at the acceptance `acc`**, at the output positions `pos`, the strategy `τ` and the per-window
error budget `ε`:

`Pr[acc at every window ∧ success ∧ server-time < T_min] ≤ Σ_b ε_b + Pr[some window's error exceeds its budget] + η`.

A window's error is its subprotocols' errors at the strategy committed to, with their collision terms, plus the wipe's
(`windowError`). -/
def Bound (V : Vocab d α) {In L Srv ιP Act : Type} [Fintype Srv] {m B : ℕ} (G : Task In (Fin m → List Bool))
    (D : Deployment V In L Srv ιP Act B) (acc : Sent → Prop) (εWipe : Window V → ℝ≥0∞) (q qD : ℝ) (N : ℕ)
    (η : ℝ≥0∞) (pos : Fin m → L × ℕ × ℕ) (τ : In → Game.Strategy (windows V B)) (ε : Fin B → ℝ≥0∞) : Prop :=
  prob V G τ (fun x 𝒴 ws => AcceptedBy V acc B ws ∧ Succeeds (D.run x ws) pos 𝒴 ∧
      (serverTime D.X (D.run x ws).A : ℝ) < minServerTime D.ω N q qD (Fintype.card Srv)) ≤
    ∑ i, ε i + prob V G τ (fun _ _ ws => ∃ i : Fin B, ∃ e, ws[(i : ℕ)]? = some e ∧ ε i < windowError V εWipe e.1) + η

/-! ### The nontrivial instance: one window, one server, the toy vocabulary -/

/-- The strategy of a game whose messages `follow` has fixed, followed by `f b` at `t b`: there are no messages left to
send before `f`. -/
def followBind {β γ : Type} (f : β → Game γ) (t : (b : β) → Game.Strategy (f b)) :
    (g : Game β) → (s : Game.Strategy g) → Game.Strategy ((follow g s).bind f)
  | .ret b, _ => t b
  | .send _ k, s => followBind f t (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => fun c => followBind f t (k c) (s c)

/-- Commit to the window `w` and play its game. -/
def oneStrategy (V : Vocab d α) (w : Window V) : Game.Strategy (windows V 1) :=
  ⟨w, followBind (fun o => (windows V 0).map ((w, o) :: ·)) (fun _ => PUnit.unit) (V.game w.θ w.F w.c) w.σ⟩

/-- One input, whose acceptable outputs are those with `[true]` at the one position. -/
def oneTask : Task Unit (Fin 1 → List Bool) := ⟨Unit, fun _ => ((), {y | y 0 = [true]})⟩

/-- One computed gate of width one, at leaf `()`, cell `0` and tick `0`, whose value is `b`. -/
def oneCircuit (b : Bool) : Placed Unit where
  N := 1
  w _ := 1
  isInput _ := false
  reads _ := ∅
  reads_lt _ _ h := by simp at h
  input_reads _ _ := rfl
  op _ _ := BitVec.ofBool b
  op_congr _ _ _ _ := rfl
  outputs := ∅
  x _ := ()
  t _ := 0
  cell _ := 0
  cell_inj g h _ := Fin.ext (by have := g.isLt; have := h.isLt; omega)

/-- `oneCircuit b`'s run: its gate's value is `b`, and nothing is held. -/
def oneRun (b : Bool) : Actual.Run Unit where
  A := oneCircuit b
  val _ := BitVec.ofBool b
  faithful _ _ := rfl
  held _ _ := ∅
  holding := ⟨fun _ _ _ h => by simp at h, fun _ h => absurd h Bool.false_ne_true,
    fun _ _ h => absurd h (Finset.notMem_empty _)⟩

/-- One counted product, at the gate, when `b`. -/
def oneProds (b : Bool) : Nci.Ldl.Products (oneRun b).A.toCircuit Unit Unit where
  J := if b then {()} else ∅
  rep _ := ⟨0, Nat.one_pos⟩
  rep_inj := by intro p _ q _ _; exact Subsingleton.elim p q
  rep_op _ _ := rfl
  act _ := ()

/-- A wipe that leaves nothing. -/
def oneWipe (d : Device ι) : WipeLive d Outcome where
  fill := 0
  fill_le := Nat.zero_le _
  accepts _ := True
  kept _ := 0

/-- The first window's claim `a` fails on its play. -/
def oneBad (V : Vocab d α) (a : Audit) (ws : List (Window V × Outcome)) : Prop :=
  ∃ e, ws[0]? = some e ∧ ¬ V.Holds e.1.θ e.1.F e.1.c e.1.σ a e.2

open Classical in
/-- One server and one window of two ticks; the run's gate holds `1`, and is a counted product, exactly when the
window's claim `a` fails. -/
noncomputable def oneDeployment (V : Vocab d α) (a : Audit) : Deployment V Unit Unit Unit Unit Unit 1 where
  run _ ws := oneRun (decide (oneBad V a ws))
  prods _ ws := oneProds (decide (oneBad V a ws))
  X _ := Finset.univ
  H := 1
  win := (Equiv.punitProd (Fin 1)).symm
  ω := 2
  wipe _ := oneWipe d

/-- The toy device: no on-chip state, no device memory, no ops, no side channels. -/
def toyDevice : Device Empty := ⟨0, 0, fun _ => 0, 0, fun _ => 0⟩

/-- The toy world: one account, an empty window, an idle schedule, and every network claim. -/
def toyWorld : World toyDevice Unit where
  acct _ := ()
  win := ⟨0, 0⟩
  schedule _ := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => 0⟩
  fits _ := ⟨fun _ => by simp [toyDevice], by simp [toyDevice], fun i => i.elim, by simp [toyDevice],
    by simp [toyDevice]⟩
  commWithin _ := True

/-- The toy statement tags: every string empty, the identity `null`. -/
def toyTags : Flock.Tags where
  name := ""
  statement := ""
  fileFormat := ""
  blockKeyword := ""
  publicFormat := ""
  publicCircuitKey := ""
  sigmaTag := ""
  domainPrefix := ""
  table := ""
  merkleLeaf := ""
  identity := .null
  coinScheme := ""

/-- The toy finder: one coin, no output, no SHA-512 calls. -/
def toyFinder : Claims.Finder where
  S := Unit
  out _ := none
  cost _ := 0

/-- **The toy vocabulary.** Every file reads as the empty one, the parameters register the empty file's digest, the
device has no rounds, no gate is computed, the finders find nothing, and the product publishes `δ = 0`, no loss and the
PoUW certificate `W = 1`: every subprotocol's error is `0`, and PoUW's claim fails on every play. -/
noncomputable def toyVocab : Vocab toyDevice Unit where
  read _ := ⟨toyTags, ByteArray.empty, [], ByteArray.empty⟩
  place _ _ _ _ := none
  programDigest _ := some (sha512 ByteArray.empty.data.toList)
  registered _ := ByteArray.empty
  rounds _ := []
  coinsRead _ _ := []
  extract _ _ _ _ _ := 0
  finder _ _ _ _ _ := toyFinder
  cost _ _ _ _ := 0
  callsMul _ _ := 0
  callsAdd _ _ := 0
  finder_calls _ _ _ _ _ _ := Nat.zero_le _
  computed _ _ _ _ _ := ∅
  world _ _ _ _ _ := toyWorld
  honest _ _ _ := ⟨ByteArray.empty, PUnit.unit⟩
  certOf _ := ⟨0, 1, 0, 0, 0⟩
  crBound _ _ _ := 0
  delta _ _ _ := 0
  loss _ _ _ := 0

/-- The toy window: the empty parameters, files and commitment, at the honest strategy. -/
noncomputable def toyWindow : Window toyVocab :=
  ⟨ByteArray.empty, ByteArray.empty, ByteArray.empty, toyVocab.honest _ _ _⟩

end Overhead

open Overhead

/-- **The overhead guarantee at the acceptance `acc`.** On a deployment where `acc` meets soundness
(`Vocab.SoundnessFor`), the wipe its per-window target (`WipeSound`), every run the physical premises `Phys`, and the
serving-block cap, the dark share and materialization coverage hold, the overhead bound (`Bound`) holds at every output
positions, strategy and per-window error budget `ε`. -/
def OverheadFor (acc : Sent → Prop) : Prop :=
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α) {In L Srv ιP Act : Type} [Fintype Srv]
    {m B : ℕ} (G : Task In (Fin m → List Bool)) (D : Deployment V In L Srv ιP Act B) (f : ℝ)
    (εWipe : Window V → ℝ≥0∞) (Phys : Actual.Run L → Prop) (q qD : ℝ) (N : ℕ) (η : ℝ≥0∞),
    0 < D.ω → 0 < q →
    V.SoundnessFor acc → D.WipeSound acc f εWipe → (∀ x ws, Phys (D.run x ws)) → D.ServingCap f Phys q →
    D.DarkShare qD → D.Materialization G N η →
    ∀ (pos : Fin m → L × ℕ × ℕ) (τ : In → Game.Strategy (windows V B)) (ε : Fin B → ℝ≥0∞),
      Bound V G D acc εWipe q qD N η pos τ ε

/-- **The overhead guarantee at the verifier of record** (`Accepts`): from `Vocab.Soundness`, for every strategy, the
probability that the verifier of record accepts every server-window, the run succeeds and its server-time is below
`T_min` is at most the windows' error budgets summed, plus the probability that a window's soundness and wipe errors
exceed its budget, plus the coverage exception. -/
def Overhead : Prop := OverheadFor Accepts

/-- **The overhead guarantee needs the verifier.** At the toy vocabulary, the one-window deployment on PoUW's claim, a
cap of `1/2` per window, no dark share, `N = 1`, no coverage exception and no error budget, the overhead bound fails for
the verifier that accepts everything: its window is accepted, PoUW's claim fails on it, and the run succeeds in less
than `T_min = 2` ticks of server-time with probability `1`. -/
def Overhead_nontrivial : Prop :=
  ¬ Bound toyVocab oneTask (oneDeployment toyVocab .pouw) (fun _ => True) (fun _ => 0) (1 / 2) 0 1 0
      (fun _ => ((), 0, 0)) (fun _ => oneStrategy toyVocab toyWindow) (fun _ => 0)

/-- **The overhead guarantee's premises hold where it says something** (honest). At a deployment, every premise of
`Overhead` holds and some strategy's run is accepted at every window and succeeds with probability `1`. Its instance is
the deployment at the vocabulary of the verifier of record, which accepts an honest run; it is stated and not proved. -/
def Overhead_honest : Prop :=
  ∃ (ι : Type) (d : Device ι) (α : Type) (_ : Fintype α) (V : Vocab d α) (In L Srv ιP Act : Type) (_ : Fintype Srv)
    (m B : ℕ) (G : Task In (Fin m → List Bool)) (D : Deployment V In L Srv ιP Act B) (f : ℝ)
    (εWipe : Window V → ℝ≥0∞) (Phys : Actual.Run L → Prop) (q qD : ℝ) (N : ℕ) (η : ℝ≥0∞),
    0 < D.ω ∧ 0 < q ∧ V.Soundness ∧ D.WipeSound Accepts f εWipe ∧ (∀ x ws, Phys (D.run x ws)) ∧
    D.ServingCap f Phys q ∧ D.DarkShare qD ∧ D.Materialization G N η ∧
    ∃ (pos : Fin m → L × ℕ × ℕ) (τ : In → Game.Strategy (windows V B)),
      prob V G τ (fun x 𝒴 ws => AcceptedBy V Accepts B ws ∧ Succeeds (D.run x ws) pos 𝒴) = 1

end Verity
