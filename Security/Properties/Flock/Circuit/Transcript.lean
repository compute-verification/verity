import Properties.Flock.Circuit.CircuitOf
import Properties.Flock.Circuit.Opened

/-!
# The opened values as the parsed circuit's transcript

`Verity.Circuit.ofBits xs`: a transcript of bits as core's. `transcriptOf D hU n κ val`: the wire values opened at
`val` (`openedOf`, at the constants `κ`) as a transcript of the parsed file's circuit (`circuitOf D hU n`).

Moved unchanged from `Proofs.Flock.Bridge.CircuitOf`; the lemmas about them stay there.
-/

namespace Verity

variable {C : FlockSoundness.Audit.Circuit}

/-- A transcript of bits as core's. -/
def Circuit.ofBits (xs : Fin C.N → Bool) : (Circuit.ofAudit C).Transcript := fun g => BitVec.ofBool (xs g)

end Verity

namespace Flock.Bridge

/-- **The transcript the witness opens**, at `circuitOf`'s wires: `openedOf`'s bits. -/
noncomputable def transcriptOf (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)
    (κ : Fin (keyCircuit D hU n).N → Option Bool) (val : ℕ → Option Bool) : (circuitOf D hU n).Transcript :=
  Verity.Circuit.ofBits (openedOf (keyCircuit D hU n) κ val)

end Flock.Bridge
