/-
A protocol whose draw reads live coins (`Live`): its check is a check at the coins (`checkAt`), run on the coins it reads
from its evidence (`coins`), the challenger's signed record of them under the key its parameters pin.

* `Source`: where the coins come from, as the verifier holds it: the challenger's key, the session its parameters
  designate (a window's open root, or the id the auditor designated for PoUS's session), and the digest the coins were
  released after (the commitment's);
* `Live P`: `P`'s check factors through the coins (`factors`): `check` is `checkAt` at what `coins` reads, and refuses
  with its reason when nothing reads;
* `Protocol.answerAt`: `Protocol.answer` with `checkAt` at coins `ξ` in place of `check`, the verdict the protocol
  gives when its challenger drew `ξ`.

A soundness argument reasons about `answerAt` at the coins the challenger drew. That `answer` accepts only what
`answerAt` accepts at the coins its evidence's record shows follows from `factors`; that those are the coins the device
drew, or the run carries a break, is each reader's property (`Security/Properties`). A protocol without live coins
needs none of this: `Protocol` and `Protocol.answer` are unchanged.
-/
import CoreProtocol.Protocol

namespace Core

open Flock (V)

/-- Where a draw's live coins come from, as the verifier holds it. -/
structure Source where
  /-- The challenger's LMS public key (`I ‖ T[1]`), the verifier's own, never the prover's. -/
  key : ByteArray
  /-- The session the parameters designate: the window's open root, or the session id the auditor designated. -/
  session : ByteArray
  /-- The digest the coins were released after: the commitment's. -/
  after : ByteArray

/-- `P`'s check reads its draw's live coins and decides at them: `check p c e` is `checkAt p c ξ e` at the coins `ξ`
that `coins p c e` reads, and a refusal with its reason when it reads none. -/
structure Live (P : Protocol) where
  Coins : Type
  source : P.Params → P.Commitment → P.Evidence → Source
  coins : P.Params → P.Commitment → P.Evidence → V Coins
  checkAt : P.Params → P.Commitment → Coins → P.Evidence → Outcome P.Public P.Profile
  factors : ∀ p c e, P.check p c e = match coins p c e with
    | .ok ξ => checkAt p c ξ e
    | .error r => .reject r

/-- `P` deciding at the coins `ξ`: its check is `checkAt` at `ξ`, whatever the evidence's record shows. -/
def Live.at {P : Protocol} (L : Live P) (ξ : L.Coins) : Protocol :=
  { P with check := fun p c e => L.checkAt p c ξ e }

/-- `Protocol.answer` with `checkAt` at the coins `ξ` in place of `check`. -/
def Protocol.answerAt (P : Protocol) (L : Live P) (params commitment : ByteArray) (ξ : L.Coins)
    (evidence : ByteArray) : Answer :=
  (L.at ξ).answer params commitment evidence

end Core
