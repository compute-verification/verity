import Definitions.Accounting.Remote.Node
import Definitions.Pouw.Basic.Prob

/-!
# A remote audit's run, and the routes its guarantees compose (trusted)

A run of the remote audit on node `N` with parameters `a`, over the audit's coins `Ω` (the verifier's seeds σ_t, its
samples and the oracle), uniform (`Pouw.pr`). Each field is a quantity of the console's actual-circuit model (v2.9) at
each outcome, over the audited interval `I = [t_0, t_P)`, where `t_j` is the tick of pass `j`'s oracle query
`H(σ_t, root_{j−1})`, so that by the oracle's Merkle chain pass `j`'s correct receipts are computed in `[t_j, t_{j+1})`:

* `read ω m`: the bits tier `m`'s port carried to the node's compute (`ingressFrom X_N m I`); `readOut ω`, the bits
  memory tiers outside the node carried to any compute.
* `state ω`: `Σ_j (content of X_N at t_j + 1)`, the node's state at each pass boundary with its length bit
  (`LengthCode`); `stateOut ω`, the same for compute outside the node.
* `rebuilt ω`: the bits of correct receipts' words the node's compute computed rather than received, a partial
  completion of `x` of a word's bits counting `x`; `rebuiltOut ω`, the same outside the node.
* `rebuildWork ω`, `pouwWork ω`: the node compute's priced work at π_r over `I` outside PoUW's account, and PoUW's
  account's (together at most `π_r(A[X_N, I])`).
* `inBits ω`, `outBits ω`: the bits of the signals that crossed the node's boundary links, inward and outward;
  `inBits` counts whatever crosses into `X_N`, wherever it lands (an L2 or a DDIO landing is counted too).
* `zOut ω`: the bits of bound `z` (the prior PoUW tiles' noised outputs) determined outside the node other than over its
  boundary; `workOut ω`, the outside compute's priced work at PoUW's pricing over `I`.
* `other ω`: Mbw's `q`, the bits of data independent of the store, the seeds and the oracle that the node's compute
  determines from its memory tiers over `I`.

A route is a bound the run meets except with probability `δ` over the coins when the audit accepts (`Run.Holds`); a
model row that holds on every outcome is a route at `δ = 0`. The guarantees (`Models.Accounting.Guarantees.Remote`)
compose them by a union bound; each lane proves a route for the runs its game or the actual circuit realizes:

* `Sampled ε`: correct receipts cover `(1 − ε) · C` (μ_sample); the verifier checks `committed · b ≥ C` and samples.
* `Paid`: each correct receipt's word was paid for, by a read, a held bit or a rebuild, at whatever location computed
  it (conservation of bits; lane ra-remote-lean's two-tier conservation and chained passes).
* `PaidJoint`: `Paid` with Mbw's independent data counted against the same payment (Mbw at every location).
* `CoLocated`: the same paid at the node: a word read, held or rebuilt by the node, carried in over its boundary, or
  computed outside from a `z` carried out (`b / z_b` per bit) or computed there (lane ra-bound-lean). Only the
  statement without Identity takes it.
* `ChipState`: `state ≤ P · (M_N + 1)` (the Memory row and `LengthCode`; B3's charge at each `t_j`).
* `Rebuild`: `rebuilt · c_word ≤ b · rebuildWork`: rebuilding `x` bits takes at least `x · c_word / b` work at π_r
  (WorkHardness in partial-completion form; lane ra-partial-lean).
* `Fit W_r`: the rebuild fits at π_r beside `W_r`, `W_r + rebuildWork ≤ Φ_r · T`. At `W_r = 0` it is the Compute row
  alone (`δ = 0`); at `W_r > 0` it needs W (`Assumptions.ComputeAdditive`) and `Certified W_r`.
* `Certified W_r`: PoUW's account did at least `W_r` at π_r (PoUW's per-run guarantee at π_r, lane ra-per-run's
  `ComputeSoundPerRun`).
* `Links`: the boundary's signals are within `β_in · T` and `β_out · T` (the Links row).
* `Ports`: each tier's port is within `β_m · T` (the Links row at each tier's port).

`Rebuild` and `Fit` are the rebuild's route as a fit at π_r. Sequential speed's rebuild clause, that no word is rebuilt
within a pass, is the other route; each guarantee names the route it takes.
-/

namespace Accounting.Remote

/-- A run of the remote audit on node `N` with parameters `a` over coins `Ω`. -/
structure Run {τ : Type} (N : Node τ) (a : Audit) (Ω : Type) where
  /-- the verifier's verdict -/
  accepts : Ω → Prop
  /-- the receipts the prover committed to, over all passes; the verifier checks `committed · b ≥ C` -/
  committed : Ω → ℕ
  /-- the correct receipts: receipt `j` equals `H(σ, j, sketch(E_j ⊙ expand(z_{j−1})))` -/
  correct : Ω → ℕ
  /-- `ingressFrom X_N m I`: the bits tier `m`'s port carried to the node's compute -/
  read : Ω → τ → ℕ
  /-- the bits memory tiers outside the node carried to any compute over `I` -/
  readOut : Ω → ℕ
  /-- `Σ_j (content of X_N at t_j + 1)` -/
  state : Ω → ℕ
  /-- `Σ_j (content of the compute outside the node at t_j + 1)` -/
  stateOut : Ω → ℕ
  /-- the bits of correct receipts' words the node's compute rebuilt (partial completions by their bits) -/
  rebuilt : Ω → ℕ
  /-- the bits of correct receipts' words compute outside the node rebuilt -/
  rebuiltOut : Ω → ℕ
  /-- the node compute's priced work at the rebuild pricing outside PoUW's account (`π_r(A[X_N, I])` less PoUW's) -/
  rebuildWork : Ω → ℕ
  /-- PoUW's account's priced work at the rebuild pricing over `I` -/
  pouwWork : Ω → ℕ
  /-- the bits of signals inward over the node's boundary links over `I` -/
  inBits : Ω → ℕ
  /-- the bits of signals outward over the node's boundary links over `I` -/
  outBits : Ω → ℕ
  /-- the bits of bound `z` determined outside the node other than over its boundary -/
  zOut : Ω → ℕ
  /-- the outside compute's priced work at PoUW's pricing over `I` -/
  workOut : Ω → ℕ
  /-- Mbw's `q`: bits of data independent of the store, seeds and oracle that the node determines from its tiers -/
  other : Ω → ℕ

namespace Run

variable {τ : Type} [Fintype τ] {N : Node τ} {a : Audit} {Ω : Type} (r : Run N a Ω)

/-- The bits the node's tiers carried to its compute over `I`. -/
def reads (ω : Ω) : ℕ := ∑ m, r.read ω m

/-- Every payment for correct receipts' words, at every location: reads of memory tiers, state held at the pass
boundaries and bits rebuilt, inside the node and outside it. -/
def paid (ω : Ω) : ℕ :=
  r.reads ω + r.readOut ω + r.state ω + r.stateOut ω + r.rebuilt ω + r.rebuiltOut ω

/-- The payments that the node makes or that cross its boundary: its reads, state and rebuilt bits, the bits carried
in, and the bound `z` carried out or determined outside, each unlocking `b / z_b` bits outside. -/
noncomputable def paidNode (ω : Ω) : ℝ :=
  r.reads ω + r.state ω + r.rebuilt ω + r.inBits ω + (r.outBits ω + r.zOut ω : ℝ) * a.b / a.zb

variable [Fintype Ω]

/-- The run meets `P` except with probability at most `δ` over the coins on which the audit accepts. -/
def Holds (δ : ℝ) (P : Ω → Prop) : Prop := Pouw.pr (fun ω => r.accepts ω ∧ ¬ P ω) ≤ δ

/-- **Sampled** (μ_sample): the correct receipts cover `(1 − ε) · C`. -/
def Sampled (ε δ : ℝ) : Prop := r.Holds δ fun ω => (1 - ε) * N.declared a.T ≤ (r.correct ω * a.b : ℝ)

/-- **Paid** (conservation of bits, at no location): each correct receipt's word was read, held or rebuilt somewhere,
up to `λ` bits a pass. -/
def Paid (δ : ℝ) : Prop := r.Holds δ fun ω => r.correct ω * a.b ≤ r.paid ω + a.P * a.lam

/-- **CoLocated** (the node's payment): each correct receipt's word was paid at the node or across its boundary. -/
def CoLocated (δ : ℝ) : Prop := r.Holds δ fun ω => (r.correct ω * a.b : ℝ) ≤ r.paidNode ω + a.P * a.lam

/-- **PaidJoint** (Mbw at every location): the independent data the node determines and the correct receipts' words
share the payments at every location, up to `λ` bits a pass. -/
def PaidJoint (δ : ℝ) : Prop := r.Holds δ fun ω => r.other ω + r.correct ω * a.b ≤ r.paid ω + a.P * a.lam

/-- **ChipState** (the Memory row, with the length bit): the node's state at the pass boundaries is at most
`P · (M_N + 1)`. -/
def ChipState (δ : ℝ) : Prop := r.Holds δ fun ω => r.state ω ≤ a.P * (N.chip + 1)

/-- **Rebuild** (WorkHardness, partial completion): `x` rebuilt bits cost the node `x · c_word / b` at π_r. -/
def Rebuild (δ : ℝ) : Prop := r.Holds δ fun ω => r.rebuilt ω * a.cWord ≤ a.b * r.rebuildWork ω

/-- **Fit** (the rebuild as a fit at π_r): the rebuild's work fits in what the Compute row's `Φ_r · T` leaves beside
`W_r`. At `W_r = 0` it is the Compute row alone; subtracting `W_r` needs W and `Certified W_r`. -/
def Fit (Wr δ : ℝ) : Prop := r.Holds δ fun ω => Wr + r.rebuildWork ω ≤ N.rebuild * a.T

/-- **Certified** (PoUW's per-run guarantee at π_r): PoUW's account did at least `W_r` at the rebuild pricing. -/
def Certified (Wr δ : ℝ) : Prop := r.Holds δ fun ω => Wr ≤ r.pouwWork ω

/-- **Links** (the Links row): the signals over the node's boundary are within its links' capacity. -/
def Links (δ : ℝ) : Prop := r.Holds δ fun ω => r.inBits ω ≤ N.linkIn * a.T ∧ r.outBits ω ≤ N.linkOut * a.T

/-- **Ports** (the Links row at each tier's port): each tier's reads are within its calibrated ceiling. -/
def Ports (δ : ℝ) : Prop := r.Holds δ fun ω => ∀ m, r.read ω m ≤ N.ceil m * a.T

end Run

end Accounting.Remote
