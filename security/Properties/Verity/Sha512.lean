module

public import Definitions.Core.Hash
public import Flock.Hash

/-!
# SHA-512, the compiled one, in the hash area's terms

`Verity.sha512` is `Flock.Sha512.hash` as a 64-byte digest of a message, the hash every statement about the compiled
verifier's SHA-512 reads. It sits in a module of its own, importing only the hash area and Flock's hash, so a property
reads it without importing the composites' game (`Properties.Verity.Game`) and everything that imports.
-/

@[expose] public section

namespace Verity

/-- SHA-512, the compiled one (`Flock.Sha512.hash`), as a 64-byte digest. -/
def sha512 (m : Hash.Msg) : Hash.Digest := fun i => (Flock.Sha512.hash ⟨m.toArray⟩).data.getD i.val 0

/-- SHA-512, the compiled one, as the message-to-message hash `Claims.Finder.Pr` and `ecr/sha-512`
(`Claims.ExpectedCR`) read, and the one the sampled commitments' path checks open under. -/
def sha512Msg (m : Hash.Msg) : Hash.Msg := (Flock.Sha512.hash ⟨m.toArray⟩).data.toList

end Verity
