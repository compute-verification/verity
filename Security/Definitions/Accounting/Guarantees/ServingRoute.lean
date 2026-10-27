import Definitions.Accounting.Assumptions.ServingRoute
import Definitions.Nci.Assumptions.Ldl

/-!
# The serving route, end to end

The serving-block route of the end-to-end overhead guarantee (note:training-20261027T0134Z-draft-e2e-overhead-proof
§9), the route whose bound bites at the onsite reference (98.7 at 5 GB/server, set by `fanout-priced` at the serving
cut; §7.4):

* `ServingPerBlockCap` (the per-block cap at serving scale): the activation premise (`Nci.Ldl.Assumptions.ActivationRate`,
  `β` bits per activation coordinate with deficit `Λ`) and Lemma 1 (`Nci.Ldl.Assumptions.EvaluatesBlocks`), on every
  registration whose blocks fit the serving budget `(c_S, k_S)`, give `PerBlockCap` at `(c_S, k_S)` with
  `q(c_S, k_S) = a_out (k_S + ℓ + a_blk + Λ)/β`: `FanOutBound`'s per-block step at the serving scale, the index costing
  `ℓ ≥ ⌈log₂(N + 1)⌉` bits.
* `ServingEndToEnd` (Steps 1 to 7 on the serving route): in one experiment, for every strategy,
  `Pr[accept ∧ success ∧ windows < windowsMin N] ≤ δ + δ_b + η`, from the joint guarantee's bound `δ` (in
  `security_properties`, `Verity.OnsiteServingEndToEnd` reads it off `Verity.Soundness`), the serving bridge
  (`Accounting.Assumptions.ServingBridge`, `δ_b`) and coverage (`Accounting.Assumptions.ServingCoverage`, `η`), through
  `ServingBlocks.ServingBlockBound` at the uniform budget `(c_S, k_S)` and `Overhead.EndToEnd`.

**The activation premise at serving scale is contested.** A serving block is one server between two wipes, a
server-day's computation: such a block can replay training steps from data it reads, so its activations need not cost
`β` bits each beyond its `k_S` bits of reads (note:training-20261024T2336Z-report-tvi-findings, E1's P0). Lemma 1's
step is sound at any scale; whether `ActivationRate` holds at `(c_S, k_S)` is the open question, and the route's number
rests on it.
-/

namespace Accounting.Guarantees.ServingRoute

open Verity Accounting.ServingBlocks Accounting.ServingRoute
open scoped ENNReal

/-- **The per-block cap at serving scale.** For a declaration `d` with counted operations `J` (each body holding its
representative, each weight at most one) that are product occurrences of `M` (`J.V ⊆ rep(M.J)`) with activation fan-out
at most `a_out = e`, and its run `X = d.D.eval inp`: suppose every registration of `d` consistent with `X` whose blocks
have at most `c_S ≥ 1` gates and read at most `k_S` unregistered bits (`ReadCap`) has a decoder `D n R` that evaluates
its blocks at overhead `a_blk` (`EvaluatesBlocks`, Lemma 1's premise) and under which its blocks' activations cost `β`
bits each with deficit `Λ` (`ActivationRate`). If each operation gate reads at most `k_S` unregistered bits and
`N < 2^ℓ`, then `q` is a per-block cap at `(c_S, k_S)` (`PerBlockCap`) whenever `q(c_S, k_S) ≥ e (k_S + ℓ + a_blk + Λ)/β`.

The block `I` is read as block `0` of the registration that puts every other operation gate in a block of its own, so
both premises are asked of every registration within the budget, not of one fixed partition.

The premise at serving scale is contested: a server-day block can replay training steps from data
(note:training-20261024T2336Z-report-tvi-findings, E1's P0). -/
def ServingPerBlockCap : Prop :=
  ∀ (d : Declaration) (J : Counted d.D) {ι α : Type} [DecidableEq α] (M : Nci.Ldl.Products d.D ι α) (e : ℕ)
    (D : (n : ℕ) → Nci.Ldl.Registration d n → Nci.Ldl.Decoder) (t aBlk ℓ : ℕ)
    (y : Finset (Fin d.D.N) → d.D.Transcript → List Bool) (β Λ : ℝ) (inp : d.D.Transcript) (cS kS : ℕ)
    (q : ℕ → ℕ → ℝ),
    (∀ v ∈ J.V, v ∈ J.body v) → (∀ v ∈ J.V, J.ω v ≤ 1) → J.V ⊆ M.J.image M.rep → M.FanOut e →
    1 ≤ cS → (∀ g ∈ d.D.computed, Nci.novelReads d {g} ≤ kS) → d.D.N < 2 ^ ℓ → 0 < β →
    (∀ n (R : Nci.Ldl.Registration d n), R.Consistent (d.D.eval inp) → (∀ u, d.D.size (R.P.gates u) ≤ cS) →
      Nci.Ldl.ReadCap d R.P kS → Nci.Ldl.Assumptions.EvaluatesBlocks (D n R) t R (fun u => y (R.P.gates u)) aBlk) →
    (∀ n (R : Nci.Ldl.Registration d n), R.Consistent (d.D.eval inp) → (∀ u, d.D.size (R.P.gates u) ≤ cS) →
      Nci.Ldl.ReadCap d R.P kS →
      Nci.Ldl.Assumptions.ActivationRate β Λ t (D n R) R M (fun u => y (R.P.gates u)) (d.D.eval inp)) →
    e * (kS + ℓ + aBlk + Λ) / β ≤ q cS kS →
      PerBlockCap d.D d.θ J q cS kS

/-- **The serving route, end to end.** In one experiment, the game `G` played at a strategy `s` and read on its run
`o`, with the play `play s o` (`Accounting.ServingRoute.Play`): the verifier accepts (`Accept`), every claim holds
(`Claims`) and the computation succeeds (`Success`). Suppose that each server's compute over a window fits the serving
budget (`compute (X s) · ω ≤ c_S`), that `q_S > 0`, and that at every strategy

* `Pr[Accept ∧ ¬ Claims] ≤ δ s` (the joint guarantee);
* `Pr[Claims ∧ ¬ Class] ≤ δ_b s` (`ServingBridge`);
* `Pr[Class ∧ Success ∧ Σ_v ω_v < N] ≤ η s` (`ServingCoverage`).

Then at every strategy, the probability that the verifier accepts and the computation succeeds on fewer server-windows
than `windowsMin N = (N − n_dark)/(q(c_S, k_S) + μ k_S)` is at most `δ s + δ_b s + η s`. No independence is used. -/
def ServingEndToEnd : Prop :=
  ∀ {O L σ E : Type} [Fintype σ] [Fintype E] (G : Game O) (P : Params L σ E)
    (play : Game.Strategy G → O → Play L σ) (Accept Claims Success : Game.Strategy G → O → Prop) (N : ℝ)
    (δ δb η : Game.Strategy G → ℝ≥0∞),
    (∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS) → 0 < P.qS →
    (∀ s, Game.prob (fun o => Accept s o ∧ ¬ Claims s o) G s ≤ δ s) →
    Assumptions.ServingBridge G P play Claims δb → Assumptions.ServingCoverage G P play Success N η →
    ∀ s, Game.prob (fun o => Accept s o ∧ Success s o ∧ ((play s o).windows : ℝ) < P.windowsMin N) G s ≤
      δ s + δb s + η s

end Accounting.Guarantees.ServingRoute
