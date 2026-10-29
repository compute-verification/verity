import PousCheck.Wire
import PousCheck.Certificate

/-!
# The timed audit's check

Given a session's record (its frames, the verifier's measured latencies, its secret, beacons, every pair it has used, the
root it built from `W` and the `W` it holds), `check` gives the `FINAL` frame the verifier sends and what the session
establishes. It accepts exactly when

* the setup is the store the verifier holds: a private setup's declared root is the verifier's own (`privateRoot`, never
  the declared one taken on trust), and a public setup's blocks hash to the root and decode to the committed `W`
  (`Wire.setupFailure`);
* the audit's `(epoch, audit id)` pair is not one the verifier has used (`Verifier.seen`);
* the challenge key is derived from the verifier's secret, bound to the epoch, the root and the salt
  (`Challenge.auditKey`, or round `j`'s beacon in mode 1), and the audit id is the one derived for the counter;
* every challenge carries the derived index and nonce, round by round in order, and every response is for the open
  round (`Wire.rounds`);
* every path verifies to the root (`Merkle.verifyPath`) and every block is canonical;
* every latency is at most `Δ + RTT`, and, when asked, every block decodes (flag bit 0) to the held `W`'s payload
  (flag bit 1);
* and a certificate covers `k` and `Δ + RTT` (`Certificate.gate`),

and then gives the StorageProfile's fields (`Certificate.profileJson`). A session too malformed to name an audit has no
`FINAL`, and `check` gives the fault; inputs the verifier can't judge with (a private setup without its root, flag bit 1
without `W`, an elapsed time outside u64) are refused (`Wire.Stop.refused`).
-/

namespace PousCheck.Audit

open Lean (Json)

/-- The session's `FINAL` frame and its assessment (`Certificate.assess`: verdict, `covered`, `accepted` and profile). -/
def check (frames : Array ByteArray) (elapsed : Array Nat) (v : Wire.Verifier) : Except Wire.Stop (ByteArray × Json) := do
  let final ← Wire.verifyAudit frames elapsed v
  pure (final, match Wire.parseSetup frames with
    | .ok (setup, _) => Certificate.assess setup.params final
    | .error _ => Json.mkObj [("verdict", .bool false), ("accepted", .bool false)])

end PousCheck.Audit
