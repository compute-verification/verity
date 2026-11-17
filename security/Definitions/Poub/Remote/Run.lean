module

public import Definitions.Poub.Remote.Node
public import Definitions.Pouw.Basic.Prob

/-! A run of PoUB on a node, as the actual-circuit model's quantities over the audited interval, and the routes its
guarantees compose. -/

@[expose] public section

namespace ResourceAccounting.Remote

/-- A run of PoUB on node `N` with parameters `a` over coins `Ω`. -/
structure Run {τ : Type} (N : Node τ) (a : Audit) (Ω : Type) where
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

def ChipState (δ : ℝ) : Prop := r.Holds δ fun ω => r.state ω ≤ a.P * (N.chip + 1)

/-- **Rebuild** (WorkHardness, partial completion): `x` rebuilt bits cost the node `x · c_word / b` at π_r. -/
def Rebuild (δ : ℝ) : Prop := r.Holds δ fun ω => r.rebuilt ω * a.cWord ≤ a.b * r.rebuildWork ω

/-- **Fit** (the rebuild as a fit at π_r): the rebuild's work fits in what the Compute row's `Φ_r · T` leaves beside
`W_r`. At `W_r = 0` it is the Compute row alone; subtracting `W_r` needs W and `Certified W_r`. -/
def Fit (Wr δ : ℝ) : Prop := r.Holds δ fun ω => Wr + r.rebuildWork ω ≤ N.rebuild * a.T

/-- **Certified** (PoUW's per-run guarantee at π_r): PoUW's account did at least `W_r` at the rebuild pricing. -/
def Certified (Wr δ : ℝ) : Prop := r.Holds δ fun ω => Wr ≤ r.pouwWork ω

def Links (δ : ℝ) : Prop := r.Holds δ fun ω => r.inBits ω ≤ N.linkIn * a.T ∧ r.outBits ω ≤ N.linkOut * a.T

def Ports (δ : ℝ) : Prop := r.Holds δ fun ω => ∀ m, r.read ω m ≤ N.ceil m * a.T

end Run

end ResourceAccounting.Remote
