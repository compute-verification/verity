module

public import Definitions.Core.Hash
public import Flock.Hash

/-! `Verity.sha512`, the compiled SHA-512 as a 64-byte digest, in a module of its own so that a property reads it
without importing the composites' game. -/

@[expose] public section

namespace Verity

/-- SHA-512, the compiled one (`Flock.Sha512.hash`), as a 64-byte digest. -/
def sha512 (m : Hash.Msg) : Hash.Digest := fun i => (Flock.Sha512.hash ⟨m.toArray⟩).data.getD i.val 0

end Verity
