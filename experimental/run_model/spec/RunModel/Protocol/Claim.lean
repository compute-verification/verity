import Verity.Protocol.Partition
import RunModel.Protocol.Run

/-!
# What an accepting run claims (trusted)

A profile claims that at most `threshold` proof units of the developer's committed computation are wrong, except with
probability `δ`. The computation is core's: the registered program is a `Verity.Circuit` and its proof units are a
`Verity.Partition`'s. That `A.program` is that circuit's digest is the descriptor's canonical format, not modeled here.

The committed computation is what the developer's registration binds: a transcript fixed by the developer alone, before
any coin (C-Flock's `Analysis.committed`). Against an unbounded developer, salted hashes bind only through an extractor
and `cr/sha-512` on finders built from the developer; for this run the extractor reads V*'s witness off the outer
session (`ZkOuter.layer`, decoded by V*'s `wmsg`), which C-Flock's soundness package defines, so an `Extraction` is a
parameter of the statement here. The statement means what it says only for an extraction built from the developer: a
`committed` that ignores the developer, or a finder that hard-codes a collision (so its hypothesis is false), makes it
vacuous. Its definition is trusted text, not yet written where this package can import it.
-/

namespace RunModel

open Run
open scoped ENNReal

structure Claim (A : Agreement) where
  C : Verity.Circuit
  P : Verity.Partition C A.population

structure Extraction (A : Agreement) (cl : Claim A) where
  /-- the developer's committed transcript, from the developer alone -/
  committed : ℕ → Adversary → cl.C.Transcript
  /-- the finders built from the developer that a binding failure reduces to (`zk_session_recursive_full`'s `hCRo` and
  `hCR`) -/
  finders : ℕ → Adversary → List Finder

/-- The profile's δ. -/
noncomputable def delta (A : Agreement) : ℝ≥0∞ := (A.deltaNum : ℝ≥0∞) / A.deltaDen

/-- **The agreement's δ covers its law**: for every set of more than `threshold` units, the coin server's draw on
uniform bytes (`Flock.Draw.drawOn`, the bytes `CoinServer.drawNeed` counts) proves none of them with probability at
most `δ`. A draw that runs out of bytes halts the coin server, and so rejects. -/
def LawDelta (A : Agreement) : Prop :=
  ∀ W : Finset ℕ, W ⊆ Finset.range A.population → A.threshold < W.card →
    (fresh (CoinServer.drawNeed A.law A.population)).toOuterMeasure
        {r | ∃ d, Flock.Draw.drawOn A.law A.population r = some d ∧ ∀ u ∈ d.proved, u ∉ W} ≤ delta A

/-- The composition's terms beyond the draw's and `cr/sha-512`'s. -/
structure ErrorTerms where
  /-- the inner session accepts a false inner statement (C-Flock's `InnerSound`) -/
  inner : ℝ≥0∞
  /-- V*'s outer session accepts with a layer that does not hold the inner messages (`ZkOuter.bound`, the stages'
  slack) -/
  outer : ℝ≥0∞
  /-- the coin tree's root and key tell the developer something of a round's coins before it commits (hm96 hiding) -/
  tree : ℝ≥0∞

def ErrorTerms.total (ε : ErrorTerms) : ℝ≥0∞ := ε.inner + ε.outer + ε.tree

/-- The run issued a profile while more than `threshold` units of `X` are wrong. -/
def Violated (A : Agreement) (cl : Claim A) (X : cl.C.Transcript) (w : World (Setting.sound A)) : Prop :=
  w.accepted.isSome ∧ A.threshold < (cl.P.wrong X).card

end RunModel
