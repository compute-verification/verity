import VStarRecursive
import Properties.SampledProofs

/-!
# Sampled proofs' recursive audit: what its Lean verdict means (`VStar.Recursive.checkBlobs`)

The property of the executable recursive audit (`verity/core/service/vstar/recursive/`, `verity/sampled-proofs/recursive/v1`),
stated against `VStar.Recursive.checkBlobs` as it runs, for every input of the types its decoders produce and every set of
blobs. It takes no premise.

`AcceptMeans`: what the check accepts is a commitment, evidence and blobs for which

* **V*'s compiled check accepted every part** (`PartsAccept`): the inner session's inputs read off the blobs
  (`VStar.Recursive.innerHeld`) at a **dense** inner shape (`Dense`: no private inner circuits, no RecSparse phase, no R),
  the form `RecDeploy.DenseAt` reads; one part per V* statement at that shape (`Flock.RecVStar.statements`), in order;
  and each part's `flock-verify verify --zk` verdict at its recursive inputs accepted (`VStar.Recursive.partVerdict`);
* **the units** the recursive session proved are the inner public file's units at the inner record's draw, and they are
  the evidence's `proved_units` (`UnitsOfInner`);
* **one-stage's check** accepted the evidence with that one verdict of record (`OneStage.check`), so
  `Properties.SampledProofs.AcceptMeans` reads off it the registration, the draw and the published bounds.
-/

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs.Recursive

open VStar.Recursive (Own Proofs Params Evidence InnerHeld)

/-- **The dense form**: the inner circuits are public, and there is no RecSparse phase and no R (`Form.Dense`'s
`holo = false ∧ sparse = none`, with R refused by `rec-inner` at such a shape). -/
def Dense (i : Flock.RecVStar.Inner) : Prop := i.holo = false ∧ i.sparse = none ∧ i.stacked = none

/-- **V*'s compiled check accepted every part**, at the inner session held `h` and V*'s statements `ss` at its dense
shape: one part of the verifier's and one session of the evidence's per statement, and each part's verdict `.ok ()`. -/
def PartsAccept (o : Own) (pr : Proofs) (bs : Core.Blobs) (h : InnerHeld) (ss : Array VStar.Stmt) : Prop :=
  VStar.Recursive.innerHeld o pr bs = .ok h ∧ Dense h.shape ∧ Flock.RecVStar.statements h.shape = .ok ss ∧
    o.parts.size = ss.size ∧ pr.parts.size = ss.size ∧
    ∀ (j : Nat) (ho : j < o.parts.size) (hp : j < pr.parts.size),
      VStar.Recursive.partVerdict h j o.parts[j] pr.parts[j] bs = .ok (.ok ())

/-- **The proved units are the inner session's**: the inner public file's units at the inner record's draw
(`OneStageZk.provedUnits`), ascending, are the evidence's `proved_units`. -/
def UnitsOfInner (o : Own) (pr : Proofs) (bs : Core.Blobs) (proved : Array Nat) : Prop :=
  ∃ pubB recB us, bs.get o.pub "the inner public file" = .ok pubB ∧
    bs.get pr.record "the inner session's record" = .ok recB ∧
    OneStageZk.provedUnits pubB ((String.fromUTF8? recB).getD "") = .ok us ∧ us.qsort (· < ·) = proved

/-- **What the recursive audit accepts.** -/
def AcceptMeans : Prop :=
  ∀ (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) (pub : OneStage.Public)
    (prof : Core.Integrity),
    VStar.Recursive.checkBlobs p c e bs = .accept pub prof →
      (∃ h ss, PartsAccept p.own e.proofs bs h ss) ∧ UnitsOfInner p.own e.proofs bs e.base.provedUnits ∧
      ∃ session vs, e.base.sessions[0]? = some session ∧
        VStar.Recursive.verdicts p.own session e.proofs (some e.base.provedUnits) bs = .ok vs ∧
        OneStage.check p.base c { e.base with verdicts := vs } = .accept pub prof

/-- **What integrity accepts at the recursive form** is what the recursive audit accepts, and at the one-stage form what
one-stage v2's audit accepts. -/
def IntegrityMeans : Prop :=
  (∀ (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs),
    VStar.Recursive.Integrity.checkBlobs (.inr p) c (.inr e) bs = VStar.Recursive.checkBlobs p c e bs) ∧
  (∀ (p : OneStageZk.Params) (c : OneStage.Commitment) (e : OneStageZk.Evidence) (bs : Core.Blobs),
    VStar.Recursive.Integrity.checkBlobs (.inl p) c (.inl e) bs = OneStageZk.checkBlobs p c e bs) ∧
  (∀ p c e bs pub prof, VStar.Recursive.Integrity.checkBlobs p c e bs = .accept pub prof →
    (p.isLeft ∧ e.isLeft) ∨ (p.isRight ∧ e.isRight))

end Properties.SampledProofs.Recursive
