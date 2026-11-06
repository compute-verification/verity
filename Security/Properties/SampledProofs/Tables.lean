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

/-- Blobs as the deployment hands v2: a circuit file and a public file, each under a SHA-512 that is no pinned table's. -/
def honestBlobs : Core.Blobs :=
  ⟨#[(⟨Array.replicate 64 0x11⟩, "circuit file".toUTF8), (⟨Array.replicate 64 0x22⟩, "public file".toUTF8)]⟩

/-- `rcp`'s pinned SHA-512 (`Flock.MUFU_TABLES`), as bytes. -/
def rcpDigest : ByteArray :=
  ⟨#[0x8a, 0x31, 0x7a, 0xdb, 0xcc, 0x07, 0x8b, 0xf7, 0x0a, 0xe0, 0x19, 0x59, 0xb6, 0x79, 0xd4, 0x9f,
     0x97, 0xcd, 0x9c, 0x8a, 0x93, 0xd6, 0xf3, 0x9b, 0x73, 0x25, 0xc9, 0x3d, 0x0f, 0xd5, 0x50, 0xfa,
     0x43, 0x96, 0xa4, 0xc3, 0xa0, 0x56, 0x8d, 0x23, 0x87, 0xc2, 0xff, 0x2f, 0x33, 0xeb, 0x76, 0x45,
     0x3b, 0x33, 0xa7, 0xb8, 0x89, 0x0d, 0x97, 0x75, 0x2d, 0x86, 0xcc, 0x1d, 0x97, 0x1a, 0xea, 0xcc]⟩

/-- Blobs holding a file under `rcp`'s pinned SHA-512. -/
def tableBlobs : Core.Blobs := ⟨#[(rcpDigest, ByteArray.empty)]⟩

end Properties.SampledProofs
