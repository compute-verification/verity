module

public import Properties.Verity.Game
public import Properties.InferenceOnly.RecordRows

/-!
# The rows check's clause, on the rows check's own case

The clause in rows of the inference-only policy, as the rows check's guarantee states it: every case of the rows check
(`InferenceOnly.RowsProtocol.protocol`) in what the prover sent, decoded, is `Properties.Remote.RowsHeld` on the blobs
sent (`RowsClause`). It sits beside `PolicyCase`, v0's case, and reads the rows check's compiled protocol by name, not
`rowsCheck`: until `rowsCheck` is that protocol, nothing says a composite's verifier has such a case, and the clause
may hold of a play by having no case to read.

It is the clause of the case as sent. That a wired row holds its writer's committed bits (`Vocab.WiresSound`) and that
`circuits` are SPOT's files are no part of it.
-/

@[expose] public section

namespace Verity

open Core

/-- **The rows check's case in what the prover sent composite `k`**, decoded: the three byte strings decode under the
composition's formats, one of the subprotocols' cases is the rows check's (`InferenceOnly.RowsProtocol.protocol` on
the blobs sent), and under the parameters `k` names for it that case decodes to the rows check's parameters `p`, its
record `c` and its evidence `e`. -/
def RowsCase (k : Verity.Registration.Composite) (s : Sent) (p : InferenceOnly.RowsProtocol.Params)
    (c : InferenceOnlyVerifier.Protocol.Commitment) (e : InferenceOnly.RowsProtocol.Evidence) : Prop :=
  ∃ ps cm ev, Decoded s ps cm ev ∧
    ∃ x ∈ Properties.Composition.triples (s.components k) ps ev,
      x.1 = InferenceOnly.RowsProtocol.protocol.onBlobs (Blobs.ofArray s.blobs) ∧
      (canonicalJson x.2.1.params.toUTF8 >>= InferenceOnly.RowsProtocol.paramsOf) = .ok p ∧
      InferenceOnlyVerifier.Protocol.commitmentOf x.2.2.commitment = .ok c ∧
      InferenceOnly.RowsProtocol.evidenceOf x.2.2.evidence = .ok e

/-- **The inference-only policy's clause in rows at composite `k`**: every rows case in what the prover sent is
`Properties.Remote.RowsHeld` on the blobs sent. -/
def RowsClause (k : Verity.Registration.Composite) (s : Sent) : Prop :=
  ∀ p c e, RowsCase k s p c e → Properties.Remote.RowsHeld p c e (Blobs.ofArray s.blobs)

end Verity
