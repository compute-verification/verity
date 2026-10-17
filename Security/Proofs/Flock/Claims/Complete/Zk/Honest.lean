import Proofs.Flock.Claims.Complete.Zk.Session
import Proofs.Flock.Claims.Complete.Honest

/-!
# Flock's honest prover at the deployed masked verifier (`honestPZk`)

The masked record's reference prover (`(deployZk rel).P`), the prover of `Claims.Flock.Complete H (deployZk rel)`
(at its error `δcZk H`), against B's masked record (`FlockDeploy.V true`): `honestP`'s program (`PI`, the sends
`asks`, SHA-512 the oracle `Hashes.pi`, `feed`) on the masked session (`sessionPZk`), with its tape `TapeZk`.

* **Its program on a statement** (`progZk`): on the session `setupOf true x` sets up, each table's trace from the
  witness (`tracesOf`, the mask slot's words the witness's), the honest masked session, and the last message
  `Final.encode ⟨x, record, proofs⟩`.
* **The prover** (`honestPZk`): at the history `hist`, the program fed the verifier's replies so far.
-/

namespace FlockComplete

open Flock FlockDeploy

/-- **The honest masked prover's program** on the statement `x`, the witness `w` and the tape `ω`; the empty message
where the statement sets up no session or the prover cannot run it, which the verifier refuses. -/
def progZk (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk) : PI Claims.Msg :=
  match setupOf true x with
  | none => .ret []
  | some (_, spec, sts) => do
    match ← (sessionPZk asks Hashes.pi spec (tabsOf sts) (tracesOf sts w) ω).run with
    | .ok (record, proofs) => pure (Final.encode ⟨x, record, proofs⟩)
    | .error _ => pure []

/-- **Flock's honest prover** at the masked verifier: at each history, its program fed the verifier's replies. -/
def honestPZk (w : Claims.Flock.Wit) : Claims.Prover TapeZk :=
  ⟨fun x hist ω => feed (progZk x w ω) (hist.map Prod.snd)⟩

end FlockComplete
