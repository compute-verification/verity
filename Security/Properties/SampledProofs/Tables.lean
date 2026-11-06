module

public import SampledProofs.OneStageZk

/-!
# v2 holds the session's lookup tables to the registration's (`SampledProofs.OneStageZk.tablesHeld`)

The parameters of the one-stage audit's v2 (`SampledProofs.OneStageZk`) register its circuit and public files by their
SHA-512s and no lookup table, so the session v2 sets up reads the registration's tables, none, only where the blobs hold
no pinned MUFU table (`OneStageZk.tablesOf`, the tables `inputsOf` hands `flock-verify`). v2 refuses blobs holding one
before it sets up a session (`tablesHeld`, the first statement of `run`), so wherever its check accepts, the session's
table set is the registration's.

* `TablesRefused`: blobs `tablesHeld` accepts hold no pinned table, and so do blobs on which v2's check at any coins
  (`checkBlobsAt`) accepts.

It takes no premise.
-/

@[expose] public section

open SampledProofs

namespace Properties.SampledProofs

/-- **The blobs hold no pinned MUFU table**: the lookup tables the session reads off them (`OneStageZk.tablesOf`) are
the registration's, none. -/
def HoldsNoTable (bs : Core.Blobs) : Prop := OneStageZk.tablesOf bs = []

/-- **v2 refuses blobs holding a lookup table**: blobs its refusal (`tablesHeld`) accepts hold none, and so do blobs its
check accepts at any coins. -/
def TablesRefused : Prop :=
  (∀ bs : Core.Blobs, OneStageZk.tablesHeld bs = .ok () → HoldsNoTable bs) ∧
    ∀ (p : OneStageZk.Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : OneStageZk.Evidence) (bs : Core.Blobs)
      (pub : OneStage.Public) (prof : Core.Integrity),
      OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof → HoldsNoTable bs

end Properties.SampledProofs
