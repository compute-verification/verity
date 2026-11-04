import Properties.SampledProofs
import SampledProofs.Bits
import SampledProofs.OneStageZk

/-!
# Sampled proofs' registered values at one-stage v2: R7's roots are the roots the sessions read them at

One-stage v2 (`SampledProofs.OneStageZk.checkBlobsAt`) verifies each session with the verifier's own registered file
(`OneStageZk.readsOf`, `--registered`): a session's statement reads every registered row by a port of that file, and
each row it reads opens the port's root (`Flock.Registered.check`). Then v1's check holds the record's registered values
to the verifier's own (`Properties.SampledProofs.R7`), and the roots the verdicts report opening them against to the
record's roots (`Properties.SampledProofs.UnitsProved`). Together, R7's roots are the ports' roots.

* `HeldRoot v value r`: R7 holds the record's entry for `value` to the root `r`, a root the verifier fixed before the
  record: the earlier output's root, else the root it pinned, else the root the log first registered it under with the
  domain it takes now (`heldFor`).
* `HeldCovered`, **the pinned half**: what one-stage v2 accepts has the verifier hold registered reads in which, for
  every value R7 holds to a root `r`, a port reads that value at `r`. So does every port reading it under a name no
  other port has: a verdict reports one port per name (`Flock.Registered.finding`). A value R7 holds is one of the
  verifier's own entries, so the verifier expects values registered (`ExpectsV2`) wherever one is held.
* `GatesHeld`, the pinned half at the gates: on the same acceptance, a gate `SampledProofs.Bits.rowBit` places in a
  registered row, on a circuit every port of which is a port of the verifier's registered reads
  (`Flock.Registered.checkComplete`), reads a port of them named after that row, and that port, where no other port has
  its name, is at the root R7 holds its value to.

They take no premise: they are properties of the check's own functions.
-/

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs

open OneStage (Commitment Public)

/-- **R7 holds the record's entry for `value` to the root `r`** (lowercase hex), a root the verifier fixed before the
record: `value` is one of its own entries once its output values take their earlier records' domains
(`Registration.applyOutputs`), and the first root R7 holds that entry to (`heldFor`, which names none for a value the
verifier fills itself) is `r`: the earlier output's root, else the root it pinned, else the root the log first
registered it under with the domain it takes now (`Registration.heldRoots`). -/
def HeldRoot (v : Registration.Verifier) (value r : String) : Prop :=
  ∃ mine mine' held', Registration.ownEntries v = .ok mine ∧
    (Registration.applyOutputs #[] mine (v.pinned ++ Registration.heldRoots v.log (Registration.ownEntries v))
      v.outputs v.log).2 = (mine', held') ∧
    (∃ want ∈ mine', want.value = value) ∧ (heldFor v.supplied held' value).find? (·.1 == value) = some (value, r)

/-- **The pinned half of R7's coverage**: what one-stage v2 accepts at any coins holds every value R7 holds to a root
`r` (`HeldRoot`) to the verifier's registered reads `own` (`OneStageZk.readsOf` of its registered file): a port of
`own` reads the value at `r`, and so does every port of `own` reading it under a name no other port of `own` has. -/
def HeldCovered : Prop :=
  ∀ (p : OneStageZk.Params) (c : Commitment) (ξ : ByteArray) (e : OneStageZk.Evidence) (bs : Core.Blobs)
    (pub : Public) (prof : Core.Integrity),
    OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof →
      ∀ value r, HeldRoot p.base.verifier value r →
        ∃ rd own, OneStageZk.readsOf p.own.registered = .ok rd ∧ rd.registered = some own ∧
          (∃ q ∈ own, q.value = value ∧ Flock.hexOf q.root = r) ∧
          ∀ q ∈ own, q.value = value → (∀ q' ∈ own, q'.name = q.name → q' = q) → Flock.hexOf q.root = r

/-- **The pinned half at the gates**: what one-stage v2 accepts at any coins holds every gate `Bits` places in a
registered row to R7's roots. With `own` the verifier's registered reads,
a gate `g` that `Bits.rowBit` places at a commitment position `rp`, on a circuit every port of which is a port of `own`
(`Flock.Registered.checkComplete`, which one-stage v2 runs on each session's statements), reads a port of `own` named
`rp.row`; and each port of `own` of that name, where no other port of `own` has it, is at the root R7 holds the port's
value to (`HeldRoot`). -/
def GatesHeld : Prop :=
  ∀ (p : OneStageZk.Params) (cm : Commitment) (ξ : ByteArray) (e : OneStageZk.Evidence) (bs : Core.Blobs)
    (pub : Public) (prof : Core.Integrity),
    OneStageZk.checkBlobsAt p cm ξ e bs = .accept pub prof →
      ∀ rd own, OneStageZk.readsOf p.own.registered = .ok rd → rd.registered = some own →
        ∀ (c : Flock.Circuit) (n g : Nat) (rp : Bits.RowPos), Flock.Registered.checkComplete c own = .ok () →
          Bits.rowBit c n g = some rp → (∃ q ∈ own, q.name = rp.row) ∧
            ∀ q ∈ own, q.name = rp.row → (∀ q' ∈ own, q'.name = q.name → q' = q) →
              ∀ r, HeldRoot p.base.verifier q.value r → Flock.hexOf q.root = r

end Properties.SampledProofs
