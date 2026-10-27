import Properties.Verity.Game
import Properties.Flock.Circuit.Transcript

/-!
# The deployment's extractor

`extractOf read rounds vals` has `Vocab.extract`'s type, so a vocabulary can take it as its extractor: at files that
parse to `r` (`Flock.Bridge.parsedOf`), the transcript of the wire values opened at `vals` (`Flock.Bridge.transcriptOf`,
at the parse's constants `parsedConsts`); the empty transcript at files that don't parse (`transcriptAt`). `vals` is
what the session's extractor opens at the strategy the play induces; the proofs instantiate it at C-Flock's session.
-/

namespace Verity

/-- A parse's transcript: `T` at the parse, the empty transcript at none. -/
noncomputable def transcriptAt : (o : Option Flock.Bridge.Parsed) →
    (∀ r, o = some r → (Flock.Bridge.circuitOf r.D r.hU r.n).Transcript) → (Flock.Bridge.Parsed.circuit o).Transcript
  | some r, T => T r rfl
  | none, _ => fun _ => 0

/-- **The deployment's extractor**, from the reading of the files (`read`), the device's rounds, and the values opened
at each parse (`vals`): at files that parse to `r`, the transcript of the wire values `vals` opens, at the parse's
constants; the empty transcript at files that don't parse. -/
noncomputable def extractOf (read : ByteArray → FileRead) (rounds : ByteArray → List ℕ)
    (vals : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (play (rounds θ) θ F c)) (r : Flock.Bridge.Parsed),
      Flock.Bridge.parsedOf (read F).tags (read F).circuitFile (read F).tables (read F).publicFile = some r →
      ℕ → Option Bool)
    (θ F c : ByteArray) (σ : Game.Strategy (play (rounds θ) θ F c)) : (read F).circuit.Transcript :=
  transcriptAt (Flock.Bridge.parsedOf (read F).tags (read F).circuitFile (read F).tables (read F).publicFile)
    fun r hr => Flock.Bridge.transcriptOf r.D r.hU r.n (fun g => parsedConsts (some r) g) (vals θ F c σ r hr)

end Verity
