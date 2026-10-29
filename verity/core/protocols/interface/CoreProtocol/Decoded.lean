/-
`Protocol.answer` as decoding followed by a check of the decoded values.

* `decode`: the part of `answer` that reads bytes. The parameters go through `canonicalJson` and the protocol's
  `params`, the commitment and the evidence through its own decoders. Each may run `Lean.Json.parse`, which is
  `partial` and so opaque to a proof.
* `answerOf`: the rest of `answer`, which reads only decoded values. It gives the first refusal in the order parameters,
  commitment, evidence, else the check's outcome.
* `answer_decode`: `answer` is `answerOf` of `decode`. So what a proof evaluates on decoded values holds of `answer` on
  any bytes that decode to them.
-/
import CoreProtocol.Protocol

namespace Core

open Flock (V)

/-- The three inputs as the protocol `P`'s decoders read them, each decoded or refused. -/
structure Protocol.Decoded (P : Protocol) where
  params : V P.Params
  commitment : V P.Commitment
  evidence : V P.Evidence

/-- `answer`'s decoding of its three byte strings. -/
def Protocol.decode (P : Protocol) (params commitment evidence : ByteArray) : P.Decoded :=
  ⟨canonicalJson params >>= P.params, P.commitment commitment, P.evidence evidence⟩

/-- `answer` on decoded inputs: a refusal of the parameters, then of the commitment, then of the evidence (with the
commitment's digest), else the check's outcome. -/
def Protocol.answerOf (P : Protocol) (d : P.Decoded) : Answer :=
  let refuse (reason : String) (digest : Option ByteArray) : Answer :=
    { protocol := P.id, verdict := .reject, reason := some reason, digest, published := none, profile := none }
  match d.params with
  | .error e => refuse s!"params: {e}" none
  | .ok p =>
    match d.commitment with
    | .error e => refuse s!"commitment: {e}" none
    | .ok c =>
      let dg := P.digest c
      match d.evidence with
      | .error e => refuse s!"evidence: {e}" (some dg)
      | .ok ev =>
        match P.check p c ev with
        | .accept pub prof => { protocol := P.id, verdict := .accept, reason := none, digest := some dg,
                                published := some (P.publicJson pub), profile := some (P.profileJson prof) }
        | .reject r => refuse r (some dg)
        | .undecided r => { protocol := P.id, verdict := .undecided, reason := some r, digest := some dg,
                            published := none, profile := none }

/-- **`answer` is decoding followed by a check of the decoded values.** -/
theorem Protocol.answer_decode (P : Protocol) (params commitment evidence : ByteArray) :
    P.answer params commitment evidence = P.answerOf (P.decode params commitment evidence) := rfl

/-- On decoded inputs the check accepts, `answerOf` accepts, with the commitment's digest and the check's public
outputs and profile. -/
theorem Protocol.answerOf_accept (P : Protocol) {p : P.Params} {c : P.Commitment} {e : P.Evidence} {pub : P.Public}
    {prof : P.Profile} (h : P.check p c e = .accept pub prof) :
    P.answerOf ⟨.ok p, .ok c, .ok e⟩ =
      { protocol := P.id, verdict := .accept, reason := none, digest := some (P.digest c),
        published := some (P.publicJson pub), profile := some (P.profileJson prof) } := by
  simp only [Protocol.answerOf, h]

end Core
