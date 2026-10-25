import Definitions.Accounting.HMu

/-!
# A Program's coin flow: the Build's two coin rules, on a graph (trusted)

The vocabulary of `Definitions.Accounting.Guarantees.CoinFlow`. Under one σ shared by every matmul of a batch, a declared
Program could unwind its own noise (read a coin again and cancel what it added) or use one noise twice, and then a
skipped value's coins are in the cheater's view (`Accounting.HMuProofs.readsCoin`, `twoCopies`). The Build refuses
both (`verity/ml/circuits/ports.py`, `E_COIN_FLOW` and `E_COIN_REUSE`). This is its rules on a model of the Program's
dataflow, at the granularity the Build sees: one node per primitive instance of the flattened Program.

* `Program m C V`: `m` nodes in order over the run's coins `C` (uniform bits) and values `V`. Node `n` reads the values
  of the nodes `reads n` and the coins `coins n`, and `op n` computes its value from exactly those: its domain is what
  it reads, so nothing else can reach it. `noise n`: it is a noise injection (it adds noise to an operand it takes).
  `fold n`: it is a receipt fold (its value is a sink).
* `eval σ n`: node `n`'s value under the coins `σ`.
* `Desc k g`: `g` reads `k`'s value through some chain of reads, `k` itself included: `k`'s forward cone.
* The rules, and `admits`, the Build's verdict on them:
  * `Topological`: a node reads only earlier nodes (the IR's node order).
  * `CoinFlowRule`, rule (a): a node that reads a coin is a noise injection or a receipt fold.
  * `OneUse`, rule (b): a coin a noise injection reads is read by no other node, and its value, the noised operand, is
    read by at most one node: no two copies, and no noise shared across rows, matmuls or batch elements.
  * `SinkConfined`: a fold's value is read only by folds (a sink reaches only a receipt port or the outputs).
* `noiseCoins k`: the coins node `k` makes fresh, its coins when it is a noise injection (else none).
* `View k`: what a view of `k`'s fresh coins may hold: every value outside `k`'s forward cone, and every other coin.
  `view k σ` is that view under `σ`.
-/

namespace Accounting.CoinFlow

/-- A Program's dataflow: `m` nodes in order over coins `C` and values `V`, each computing its value from the nodes and
coins it reads. -/
structure Program (m : ℕ) (C V : Type) where
  reads : Fin m → Finset (Fin m)
  coins : Fin m → Finset C
  noise : Fin m → Bool
  fold : Fin m → Bool
  op : (n : Fin m) → ({k // k ∈ reads n} → V) → ({c // c ∈ coins n} → Bool) → V

namespace Program

variable {m : ℕ} {C V : Type}

/-- Node `n`'s value under the coins `σ`: its `op` on the values of the nodes it reads and on the coins it reads. A read
of a node at or after `n`, which `admits` refuses, reads `default`. -/
def eval [Inhabited V] (P : Program m C V) (σ : C → Bool) (n : Fin m) : V :=
  P.op n (fun k => if _h : k.1 < n then P.eval σ k.1 else default) (fun c => σ c.1)
termination_by n.val
decreasing_by assumption

/-- `g` reads `k`'s value through some chain of reads (`k` itself included): `g` is in `k`'s forward cone. -/
def Desc (P : Program m C V) (k g : Fin m) : Prop :=
  Relation.ReflTransGen (fun x y => x ∈ P.reads y) k g

/-- A node reads only earlier nodes. -/
def Topological (P : Program m C V) : Prop :=
  ∀ n, ∀ k ∈ P.reads n, k < n

/-- Rule (a): a node that reads a coin is a noise injection or a receipt fold. -/
def CoinFlowRule (P : Program m C V) : Prop :=
  ∀ n, (P.coins n).Nonempty → P.noise n = true ∨ P.fold n = true

/-- Rule (b): a noise injection's coins are read by no other node, and its value by at most one node. -/
def OneUse (P : Program m C V) : Prop :=
  ∀ n, P.noise n = true →
    (∀ c ∈ P.coins n, ∀ n', c ∈ P.coins n' → n' = n) ∧ ∀ r r', n ∈ P.reads r → n ∈ P.reads r' → r = r'

/-- A receipt fold's value (a sink) is read only by receipt folds. -/
def SinkConfined (P : Program m C V) : Prop :=
  ∀ n, ∀ k ∈ P.reads n, P.fold k = true → P.fold n = true

instance instDecidableTopological (P : Program m C V) : Decidable P.Topological := by
  unfold Topological; infer_instance

instance instDecidableCoinFlowRule (P : Program m C V) : Decidable P.CoinFlowRule := by
  unfold CoinFlowRule; infer_instance

instance instDecidableOneUse [DecidableEq C] (P : Program m C V) : Decidable P.OneUse := by
  unfold OneUse; infer_instance

instance instDecidableSinkConfined (P : Program m C V) : Decidable P.SinkConfined := by
  unfold SinkConfined; infer_instance

/-- The Build's verdict: the nodes are in order, and the coin rules and sink confinement hold. -/
def admits [DecidableEq C] (P : Program m C V) : Bool :=
  decide (P.Topological ∧ P.CoinFlowRule ∧ P.OneUse ∧ P.SinkConfined)

/-- The coins node `k` makes fresh: its coins when it is a noise injection, else none. -/
def noiseCoins (P : Program m C V) (k : Fin m) : Finset C :=
  if P.noise k = true then P.coins k else ∅

/-- What a view of node `k`'s fresh coins may hold: the value of every node outside `k`'s forward cone, and every coin
`k` doesn't make fresh. -/
def View (P : Program m C V) (k : Fin m) : Type :=
  ({g // ¬P.Desc k g} → V) × ({c // c ∉ P.noiseCoins k} → Bool)

/-- That view under the coins `σ`. -/
def view [Inhabited V] (P : Program m C V) (k : Fin m) (σ : C → Bool) : P.View k :=
  (fun g => P.eval σ g.1, fun c => σ c.1)

end Program

end Accounting.CoinFlow
