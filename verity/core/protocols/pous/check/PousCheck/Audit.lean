import PousCheck.Wire
import PousCheck.Certificate

/-!
# The timed audit's check

Given a session's record (its frames, the verifier's measured latencies, its secret, beacons and seen pairs), `check`
gives the `FINAL` frame the verifier sends and what the session establishes. It accepts exactly when

* the challenge key is derived from the verifier's secret as §5 specifies, bound to the epoch, the root and the salt
  (`Challenge.auditKey`, or round `j`'s beacon in mode 1), and the audit id is the one derived for the counter;
* every challenge carries the derived index and nonce, round by round in order, and every response is for the open
  round (`Wire.rounds`);
* every path verifies to the root (`Merkle.verifyPath`) and every block is canonical;
* every latency is at most `Δ + RTT`, and, when asked, every block decodes;
* and a certificate covers `k` and `Δ + RTT` (`Certificate.gate`),

and then gives the StorageProfile's fields (`Certificate.profileJson`). A session too malformed to name an audit has no
`FINAL`, and `check` gives the fault.
-/

namespace PousCheck.Audit

open Lean (Json)

/-- The session's `FINAL` frame and its assessment (`Certificate.assess`: verdict, `covered`, `accepted` and profile). -/
def check (frames : Array ByteArray) (elapsed : Array Nat) (v : Wire.Verifier) : Wire.W (ByteArray × Json) := do
  let final ← Wire.verifyAudit frames elapsed v
  pure (final, match Wire.parseSetup frames with
    | .ok (setup, _) => Certificate.assess setup.params final
    | .error _ => Json.mkObj [("verdict", .bool false), ("accepted", .bool false)])

end PousCheck.Audit
