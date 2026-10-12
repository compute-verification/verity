import Definitions.Accounting.BoundReceipt
import Models.Pouw.Assumptions.Game
import Models.Pouw.Assumptions.NCP.RouteUAssumptions

/-!
# Receipts bound to the matmul: co-location and the link relay (trusted)

The game is `Security/Definitions/Accounting/BoundReceipt.lean`; proofs are in
`Security/Proofs/Accounting/BoundReceipt/`. No named assumption of its own: the generic-mix model (the mix a random
oracle, the tile outputs uniform) is the game, and the claims registry carries it to the deployed receipt
(`random-oracle` for the mix and the sketch inside it,
`rate-codec-work` for `c_enc`, `bandwidth-instantiation` for the store), with the ideal tile's work half stated in
PoUW's game (`TileUnpredictable`), from PoUW's named assumptions.

* `Colocation`: a receipt is right only where it was made. At every store and tile outputs, the probability over the
  mix that the GPU or the rest commits block `j`'s right receipt without having queried the mix at both `E_j` and
  `z_j` is at most `2/2^h`.
* `BoundReceiptLink`: relaying a block costs its tile output out or its contents in. An accepted node whose GPU's
  compute is within `F`, whose rest's tile work is within `F_z` and whose GPU's HBM footprint is short of `RateAudit`'s
  floor at `k − m` blocks, with `m` blocks the rest's link and tile work can't cover, has probability at most
  `((k − 1)/N)^s + 2s/2^h + C(q, k − m)/2^λ + C(q_z, m + 1)/2^λ`.
* `BoundReceiptShare`: the link's share. At `m = covered`, the link bits each way stand in for at most
  `|E|/min(|z|, |E|)` times their number of the store's bits, so `μ_link ≤ (link capacity/ΣR)·|E|/min(|z|, |E|)`.
* `TileUnpredictableOfTT` and `TileUnpredictableOfNCPU`: PoUW's `TT` and `TT_NCP_U` each give `TileUnpredictable`.
-/

namespace Accounting.Guarantees

open Accounting Accounting.Rate Accounting.BoundReceipt
open Pous (Bits pr)
open scoped ENNReal

/-- **Co-location: a receipt bound to the matmul is right only at a location that held both of its inputs.** For every
node (a GPU that holds the tile outputs and reads the store from HBM, and a rest of the node that holds the store, the
answers between them ignoring the mix), every block `j`, every store `E` and every value `z` of the tile outputs: the
probability over a fresh mix with `h`-bit receipts that the GPU's committed receipt for `j` is `Mix(j, E_j, z_j)`
while the GPU never queried the mix at `E_j` under `z`, or the rest's is while the rest never queried it at `z_j` under
`E`, is at most `2/2^h`.

**The reading.** A query at `(j, E_j, z_j)` is the location holding, inside the window, the block's contents and its
tile output at once: the GPU's chip named the contents (from HBM, the link or a rebuild) and the rest's named the tile
output (from the link or tile work). So, but for a guess, a receipt is produced only where both inputs met, and
receipts that leave a location are those it made. The bound is per block and holds at every store and tile output,
since the mix alone makes the receipt unpredictable; the verifier taking each block from either location costs the
factor `2`.

Proof: `Accounting.SecurityProofs.Colocation`: off its queries, a location's receipt is the same for every value of the
mix at the receipt's point (`Chip.run_congr`), so at most one value of the mix there matches it.

Frozen (no-function, hash): a bound in the random-oracle model over abstract chip programs, which no Lean function of
the remote protocol's code computes. -/
def Colocation : Prop :=
  ∀ (N a b h S L q Lz qz : ℕ) (A : Node N a b h S L q Lz qz) (j : Fin N) (E : Store N b) (z : Store N a),
    pr (fun G : Mix N a b h =>
        (A.gpuReceipts E z G j = G (j, E j, z j) ∧ ¬ A.MadeAtGpu E z G j) ∨
          (A.restReceipts E z G j = G (j, E j, z j) ∧ ¬ A.MadeAtRest E z G j))
      ≤ 2 / 2 ^ h

/-- **The link relay: receipts bound to the matmul make every relayed block cost its tile output or its contents.**
For every node whose GPU has on-chip state of `S` bits, takes in at most `L` bits from the rest and makes at most `q`
mix queries, and whose rest takes in at most `L_z` bits from the GPU and makes at most `q_z` mix queries, over a
uniform store of `N` blocks of `b` bits, uniform tile outputs of `a` bits, a fresh mix with `h`-bit receipts and `s`
sampled blocks; for every compute budget `F` of the GPU at `c_tag` per mix query and `c_enc` per rebuilt block, every
tile-work budget `F_z` of the rest at `w` per tile, and every `m` that the rest's link and tile work can't stretch to
`m + 1` blocks,

  `(L_z + λ)·w + F_z·a < (m + 1)·a·w`:

the probability that the verifier accepts, the GPU's compute is within `F`, the rest's tile work is within `F_z`, and
the GPU's HBM footprint falls short of `RateAudit`'s floor at `k − m` blocks with no certified stream,

  `footprint · c_enc < (k − m)·b·c_enc − (S + L + λ)·c_enc − (F − (k − m)·c_tag)·b`,

is at most

  `((k − 1)/N)^s + 2s/2^h + C(q, k − m)/2^λ + C(q_z, m + 1)/2^λ`.

**The reading.** By co-location an accepted block was made at the GPU or at the rest. The rest has the whole store and
none of the tile outputs, so each block it makes costs it `a = |z|` bits from the GPU or a tile's work `w`: it makes at
most `m` (compressing the tile outputs, `C(q_z, m + 1)/2^λ`). The GPU then made at least `k − m`, and `RateAudit`'s
compression (`C(q, k − m)/2^λ`) leaves it short of those blocks' `b = |E|` bits by no more than its state, the `L` bits
that came in from the rest and its rebuilds: a block whose contents came over the link cost `b` bits of `L`. So each
relayed block costs `min(|z|, |E|)` bits on a link, and the GPU's HBM held the rest (`BoundReceiptShare`). Against
`RateAudit`, binding the receipts to `z` takes the rest's mix out of the floor: without it, the rest makes every
receipt from the store it holds, and nothing crosses a link. In bits the event's complement is the remote statement's
`CoLocated` route (lane ra-remote-theorem) in this game: `k·b ≤ footprint + S + rebuilt·b + L + (L_z + λ + z_out)·b/a
+ λ`, the GPU's reads, its state and rebuilds, the bits in at `1:1` and the bits out and the tile outputs the rest
made by tile work (`z_out = F_z·a/w`) at `|E|/|z|`.

**What the game takes.** Each block's receipt binds a tile output of its own (the game's `z` is one independent value
per block), so one shipped tile output unlocks one block; a binding shared by `n` receipts unlocks `n`. The mix is a
random oracle on the whole of `E_j` and `z_j`, so a location makes a receipt only holding both; the deployed receipt
hashes an inner-product sketch of `E_j` against a matrix expanded from `z_j`, for which lane ra-bound-receipt proves
`½·min(|z|, |E|)` bits per relayed block, by Lindsey's lemma: the deployed bound is up to `2×` weaker than this game's.
The rest's tile work stands for every way it gets a tile output without the link, the activations' route among them:
its information is free here and only its work counts. The node is fixed before the mix is drawn: its state and memory
are functions of the store and the tile outputs, never of the mix. Over a window's passes that is a mix derived afresh
each pass from the chained seed; a mix (or a sketch matrix) fixed for the window lets a location keep, from one pass,
a digest of each block that makes the next pass's receipts without it (red team F29), which this game doesn't cover.

Proof: `Accounting.SecurityProofs.BoundReceiptLink`, from `RateAudit`'s compression on each location (the GPU's view at
each tile output and mix, the rest's at each store and mix), with the GPU's mix at the tile outputs and the rest's at
the store.

Frozen (no-function, hash): as `Colocation`, a bound in the random-oracle model over abstract chip programs. -/
def BoundReceiptLink : Prop :=
  ∀ (N a b h S L q Lz qz s k m F cTag cEnc w Fz lam : ℕ) (A : Node N a b h S L q Lz qz),
    (Lz + lam) * w + Fz * a < (m + 1) * a * w →
    pr (fun x : Store N b × BoundReceipt.Coins N a b h s =>
        A.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
          A.tileWork w x.1 x.2.1 x.2.2.1 ≤ Fz ∧
          A.footprint x.1 x.2.1 x.2.2.1 * cEnc < floor b 0 S L F cTag cEnc (k - m) lam)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + 2 * (s : ℝ≥0∞) / 2 ^ h + (q.choose (k - m) : ℝ≥0∞) / 2 ^ lam +
        (qz.choose (m + 1) : ℝ≥0∞) / 2 ^ lam

/-- **The link's share of a bound-receipt audit: inbound `1:1`, outbound at `|E|/|z|`, so
`μ_link ≤ (link capacity/ΣR)·|E|/min(|z|, |E|)`.** For tile outputs of `a > 0` bits, blocks of `b` bits and tile work
`w > 0` per tile, and every `L`, `L_z`, `F_z` and `λ`: the blocks `m = covered a w L_z F_z λ` satisfy
`BoundReceiptLink`'s hypothesis, and the store's bits the links stand in for at that `m`, the `L` bits in and `m`
blocks of `b` bits, are at most the bits in plus `|E|/|z|` times the bits out, the slack and the tile outputs the
rest's tile work made,

  `(L + m·b)·a·w ≤ L·a·w + b·((L_z + λ)·w + F_z·a)`,

so at most `|E|/min(|z|, |E|)` times the links' bits each way with the slack, plus the tile work's,

  `(L + m·b)·min(a, b)·w ≤ b·((L + L_z + λ)·w + F_z·a)`.

**The reading.** Divide the first by `a·w`: `L + m·b ≤ L + (L_z + λ + z_out)·b/a` with `z_out = F_z·a/w`, the
remote statement's `inBits + (outBits + zOut)·b/z_b` (`CoLocated`'s link part, lane ra-remote-theorem) at `z_b = a`.
Divide the second by `min(a, b)·w` and by the store's bits the window reads, `k·b`, which at a window `T` is `ΣR·T`:
the share of the floor the links take is `μ_link ≤ ((L + L_z)/T)/ΣR · |E|/min(|z|, |E|)` plus `λ`'s and the rest's
tile work's `F_z/(w·k)`, with `(L + L_z)/T` the link's capacity, both directions. At `|z| ≥ |E|` a relayed block costs
its contents, and the link's share is `RateAudit`'s `c_free/|E|`; at tile binding, `|z|/|E| = 4m/k` for a batch of `m`
rows and a reduction of `k`, it is `k/(4m)` times that on the outbound side.

Proof: `Accounting.SecurityProofs.BoundReceiptShare`, by division.

Frozen (no-function): an arithmetic fact about the covered blocks, about no function of the remote protocol's code. -/
def BoundReceiptShare : Prop :=
  ∀ a b w L Lz Fz lam : ℕ, 0 < a → 0 < w →
    (Lz + lam) * w + Fz * a < (covered a w Lz Fz lam + 1) * a * w ∧
      (L + covered a w Lz Fz lam * b) * a * w ≤ L * a * w + b * ((Lz + lam) * w + Fz * a) ∧
      (L + covered a w Lz Fz lam * b) * min a b * w ≤ b * ((L + Lz + lam) * w + Fz * a)

/-- **A tile's output is unpredictable below its work**, in PoUW's game (`Pouw.Assumptions.TT`'s), for a cost model
`CM`, a protocol `P`, a credit per unit shape, a domain `D`, a target `γ₀` and an error `ε(q, N)`: for every in-domain
workload, every adversary within compute `T` and `q` oracle calls, and every unit `u` whose credit is more than
`T/(1 − γ₀)`, the probability over the oracle and the salt that `u`'s checked values are correct is at most
`ε(q, N)`.

It is the work half of the ideal tile `BoundReceipt`'s game reads: a location that didn't do tile `u`'s work, within a
budget below its credit, doesn't have its noised output. The game's tile outputs are uniform and the rest gets one only
over the link or by tile work at `w`; this reads `w` as a unit's credit, `(1 − γ₀)` of it, and a uniform `z_j` as one
whose checked values an adversary below the credit gets right with probability `ε`. It is not a new assumption:
`TileUnpredictableOfTT` and `TileUnpredictableOfNCPU` derive it from PoUW's named assumptions. -/
def TileUnpredictable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : Pouw.CostModel Q R S)
    (P : Pouw.Protocol Q R S) (credit : Pouw.Shape → ℝ) (D : Pouw.Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Pouw.Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Pouw.Adversary CM,
    A.Bounded U T q → ∀ u : ℕ, T / (1 - γ₀) < credit (U.layout.shape u) →
      Pouw.pr (fun x : (Q → R) × S => u ∈ Pouw.checkedSet P U x.1 x.2 (A.transcript U x.1 x.2)) ≤ ε q U.layout.N

/-- **PoUW's `TT` gives `TileUnpredictable` at the unit's work `m·k·n`.** For every cost model, protocol, domain,
target and error: `Pouw.Assumptions.TT CM P D γ₀ ε` implies `TileUnpredictable CM P Wmm D γ₀ ε`. A checked unit whose
work is above `T/(1 − γ₀)` puts the checked units' total work above it.

Proof: `Accounting.SecurityProofs.TileUnpredictableOfTT`.

Frozen (conjecture, no-function, hash): it carries the named open conjecture TT, over PoUW's game with a uniform oracle,
to one unit, about no Lean function of the protocol's code. -/
def TileUnpredictableOfTT : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : Pouw.CostModel Q R S)
    (P : Pouw.Protocol Q R S) (D : Pouw.Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ),
    Pouw.Assumptions.TT CM P D γ₀ ε → TileUnpredictable CM P (fun sh => (sh.Wmm : ℝ)) D γ₀ ε

/-- **NCP's `TT_NCP_U` gives `TileUnpredictable` on route U at the stacked product's work `3·m·k·n`.** For every bias
`β`, cost model, NCP parameters, domain, target and error: `Pouw.NCP.Assumptions.TTNCP_U β CM P D γ₀ ε` implies
`TileUnpredictable CM (P.toProtocolU β) (3·Wmm) D γ₀ ε`.

Proof: `Accounting.SecurityProofs.TileUnpredictableOfNCPU`.

Frozen (conjecture, no-function, hash): as `TileUnpredictableOfTT`, for NCP's named conjecture on route U. -/
def TileUnpredictableOfNCPU : Prop :=
  ∀ (β : ℤ) (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : Pouw.CostModel Q R S)
    (P : Pouw.NCP.Params Q R S) (D : Pouw.Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ),
    Pouw.NCP.Assumptions.TTNCP_U β CM P D γ₀ ε →
      TileUnpredictable CM (P.toProtocolU β) (fun sh => 3 * (sh.Wmm : ℝ)) D γ₀ ε

end Accounting.Guarantees
