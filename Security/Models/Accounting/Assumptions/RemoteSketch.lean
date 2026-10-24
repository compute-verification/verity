import Definitions.Accounting.Remote.Held
import Definitions.Accounting.Remote.Sketch

/-!
# What deriving the receipts' sketch bound assumes (trusted)

`Accounting.Guarantees.Remote.SketchBoundSession` derives `Accounting.Remote.Held.Assumptions.SketchBound` on a
trace over a session's outcomes (`Accounting.Remote.Sketch.Outcome`) from the session (`Accounting.Remote.Sketch.Session`)
and these premises, each a named `Prop` taken as a hypothesis:

* `Accounting.Assumptions.SketchHashRandom A`: the random-oracle premise (`random-oracle`). At every prover coin and
  store, the keys SHA-512 gives at the session's candidate seeds are uniform: each family of keys is equally likely.
* `KeyBudget A qtot`: across pass `j`'s candidate seeds the receipt-hash queries total at most `qtot j`, the hashes the
  prover computes in the pass, on the node or off it (`SketchHashBudget`'s budget, per pass of a remote session).
* `Receipted A T`: a receipt the trace counts correct (`Trace.correct`) is one the kept candidate's device got right
  at the key SHA-512 gives at its seed (`Session.correctAt`): the trace's verdict on a receipt is the model's.
* `Present A T`: a block the kept candidate's device obtained whole in pass `j`, held on chip or read in from node
  memory, a link or a rebuild, was in node memory or on a chip at some tick of the pass (`Trace.present`). The trace
  counts whole blocks, the device sectors: the premise reads a block present when all its sectors were obtained in the
  pass, not necessarily at one tick.

The signs' rank tail is `Accounting.Assumptions.SketchRank`, evaluated outside Lean (art:d3c891e4).
-/

namespace Accounting.Assumptions

open Accounting.Remote.Sketch

variable {β Hs : Type} {P N m h S L q ks : ℕ} {G : Fin P → ℕ}

open Classical in
/-- `random-oracle` (the random-oracle card, 9cd57182, ASK-16: PoUW only, each use a named premise; its use for the
memory sketch is asked of Daniel, 24 Oct). **The keys SHA-512 gives at the session's candidate seeds are
uniform**: at every prover coin `b` and registered store `D`, every family `κ` of keys (each candidate seed's signs and
receipt hash) is `A.keys b D x` for the same number of SHA-512's outputs `x`, a `1/|Keys|` share of them:
$$\Pr_x\big[\mathrm{keys}(b, D, x) = \kappa\big] = 1/|\mathrm{Keys}|.$$

It is the one random-oracle step `SketchBoundSession` takes: the bound is derived over the model's coins
(`Accounting.Remote.Sketch.Coins`, each candidate's key uniform and independent of the prover's coins, the store and
the other keys), and this premise carries it to the trace's outcomes, whose keys are SHA-512's. The signs are `S_i`,
which `remote_bandwidth.receipts` expands from the seed by SHA-512, and the receipt hash is SHA-512 keyed by the seed
(`SKETCH_RECEIPT_DOMAIN ‖ frame(seed_i)`) read as a function of the block's index and sketch words.

SHA-512 is one fixed function, and for it the premise is false in Lean: a fixed function's outputs are one family,
not a uniform one (with `Hs` a point, `keys b D` takes one value). So it is a premise, read in the random-oracle model,
and not a lemma. There `Hs` is SHA-512 drawn as a uniform function and `keys b D x` its outputs at the seeds the
prover's run tries, and it holds when those seeds are distinct: fresh queries of a uniform function are uniform and
independent, chosen adaptively or not. It fails for a run that tries one seed under two candidates (their keys are
equal, not independent): the model gives each candidate its own key, and `timing.sketch_failure` charges each
candidate a compression of its own. -/
def SketchHashRandom [Fintype Hs] (A : Session β Hs P N m h S L q ks G) : Prop :=
  ∀ (b : β) (D : Accounting.RateSketch.Data N m 0 Accounting.RateSketch.Sector) (κ : Keys P N m h ks G),
    (Finset.univ.filter fun x : Hs => A.keys b D x = κ).card * Fintype.card (Keys P N m h ks G) =
      Fintype.card Hs

end Accounting.Assumptions

namespace Accounting.Remote.Sketch.Assumptions

open Accounting.RateSketch Accounting.Remote.Held Accounting.Remote.Sketch

variable {β Hs : Type} {P N m h S L q ks : ℕ} {G : Fin P → ℕ}

/-- **The per-pass hash budget**: at every prover coin, earlier keys, candidate keys and store, pass `j`'s candidates'
receipt-hash queries total at most `qtot j`. Grinding a seed reads no block, so the prover may compute its hashes off
the node as well as on it: `Σ_j qtot j` is at most the SHA-512 compressions the prover can make over the session,
`Q = F·C`, the node's `C` times the outside-compute factor `F` (`timing.sketch_failure`'s `compressions` and
`outside`). -/
def KeyBudget (A : Session β Hs P N m h S L q ks G) (qtot : Fin P → ℕ) : Prop :=
  ∀ b j η (κ : Fin (G j) → Key N m h ks) (D : Data N m 0 Sector),
    ∑ c, (((A.device b j η c).prog (κ c).1 D).queries (κ c).2 D).length ≤ qtot j

/-- **Receipted**: on every accepting outcome, each block whose receipt the trace counts correct in credited pass `j`
is one the kept candidate's device tagged right at the key SHA-512 gives at its seed. -/
def Receipted (A : Session β Hs P N m h S L q ks G) (T : Trace (Outcome β Hs N m)) : Prop :=
  ∀ ω, T.accepts ω → ∀ j : Fin P, ∀ b ∈ T.correct ω (j : ℕ),
    ∃ t ∈ A.correctAt ω j (A.kept ω j), (t : ℕ) = b

/-- **Present**: on every accepting outcome, a block the kept candidate's device of pass `j` obtained whole was in node
memory or on a chip at some tick of the pass. -/
def Present (A : Session β Hs P N m h S L q ks G) (T : Trace (Outcome β Hs N m)) : Prop :=
  ∀ ω, T.accepts ω → ∀ j : Fin P, ∀ t : Fin N, A.Whole ω j (A.kept ω j) t →
    ∀ b : Fin T.blocks, (b : ℕ) = t → ∃ τ, T.tick ω (j : ℕ) ≤ τ ∧ τ < T.tick ω ((j : ℕ) + 1) ∧ b ∈ T.present ω τ

end Accounting.Remote.Sketch.Assumptions
