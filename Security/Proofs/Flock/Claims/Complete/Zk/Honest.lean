module

public import Proofs.Flock.Claims.Complete.Zk.Session
public import Proofs.Flock.Claims.Complete.Honest
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# Flock's honest prover at the deployed masked verifier (`honestPZk`)

The prover of `CompleteWith H (deployZk rel) honestPZk (δcZk H)`, against B's masked record
(`FlockDeploy.V true`): `honestP`'s program (`PI`, the sends `asks`, SHA-512 the oracle `Hashes.pi`, `feed`) on the
masked session (`sessionPZk`), with its tape `TapeZk`. It reads table `i`'s bit `a` at `i·2^m + a` (`tracesOf`), the
relation's only at one table; the masked record's reference prover (`(deployZk rel).P`) is `ZkTables.honestPR`, which
runs `ZkTables.honestP` (each table on its share, `honestPZk` at one table) at the tape's mask words.

* **Its program on a statement** (`progZk`): on the session `setupOf true x` sets up, each table's trace from the
  witness (`tracesOf`, the mask slot's words the witness's), the honest masked session, and the last message
  `Final.encode ⟨x, record, proofs⟩`.
* **The prover** (`honestPZk`): at the history `hist`, the program fed the verifier's replies so far.
-/

@[expose] public section

namespace FlockComplete

open Flock FlockDeploy

/-- **The honest masked prover's program** on the statement `x`, the witness `w` and the tape `ω`; the empty message
where the statement sets up no session or the prover cannot run it, which the verifier refuses. -/
def progZk (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk) : PI Claims.Msg :=
  match setupOf true x with
  | none => .ret []
  | some (_, spec, sts) => do
    match ← (sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (tracesOf sts w) ω).run with
    | .ok (record, proofs) => pure (Final.encode ⟨x, record, proofs⟩)
    | .error _ => pure []

/-- **Flock's honest prover** at the masked verifier: at each history, its program fed the verifier's replies. -/
def honestPZk (w : Claims.Flock.Wit) : Claims.Prover TapeZk :=
  ⟨fun x hist ω => feed (progZk x w ω) (hist.map Prod.snd)⟩

end FlockComplete
