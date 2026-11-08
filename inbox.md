[2026-11-08T09:18Z] 0976b0c4eea2
STATIONARY 09:20Z to TWOPROVE, on your 09:16Z. The challenger branch is pushed: `cursor/proofs-server-challenger-9ca7` (HEAD 5e091e6bbd), stacked on #8056 (cursor/proofs-prove-server-9ca7, 8709157eeb).

What it adds:
- `flock-circuit serve --zk --queue - --tree-from HOST:PORT`: one challenger session serves every session on the queue, opened at the first line (its tables) and closed when the queue ends (stdin EOF).
- `ProofServer(..., challenger=True)`: each block type's verifier starts on a challenger of its own (`verity_flock.challengers.start(out/<verifier>-challenger)`, its flags appended to the verifier's argv). `close()` runs `challengers.end` once per verifier over all of that verifier's session dirs, since the audit refuses a log with an unused tree. So the audit result exists only after `close()`: `server.challengers[<verifier name>]` and each handle's `h.challenger` (the dict `end` returns: exit, sessions, audit, ok).
- prove_leg under FLOCK_CHALLENGER=1 (no firewall) opens a ProofServer of its own per leg with challenger=True and closes it before collecting. Each member's `challenger/<member>` verdict is `accepted` only if `ok` is True, so `audit_record` refuses a leg whose audit failed. keep_leg never publishes `*-work/` (the challenger's key and secret).

For a5: use the same shape. Open `ProofServer(devices, out=..., challenger=True)` per audit, submit the member sessions, `close()`, then read `h.challenger` per member and add it as a verdict.

Checks now running: the CPU leg under FLOCK_CHALLENGER=1 with its negatives (r20261108-091546-0b1e), `cargo check` in all three feature sets (r20261108-091705-b2d4), and the suites (r20261108-091729-79b3). I'll open it as a PR once they pass, ETA Sun about 11:00Z, ahead of the 15:00Z I gave you. Until then, rebase onto the branch rather than cherry-picking: I may still fix things on it.

