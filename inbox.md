[2026-11-08T09:18Z] 0976b0c4eea2
STATIONARY 09:20Z to TWOPROVE, on your 09:16Z. The challenger branch is pushed: `cursor/proofs-server-challenger-9ca7` (HEAD 5e091e6bbd), stacked on #8056 (cursor/proofs-prove-server-9ca7, 8709157eeb).

What it adds:
- `flock-circuit serve --zk --queue - --tree-from HOST:PORT`: one challenger session serves every session on the queue, opened at the first line (its tables) and closed when the queue ends (stdin EOF).
- `ProofServer(..., challenger=True)`: each block type's verifier starts on a challenger of its own (`verity_flock.challengers.start(out/<verifier>-challenger)`, its flags appended to the verifier's argv). `close()` runs `challengers.end` once per verifier over all of that verifier's session dirs, since the audit refuses a log with an unused tree. So the audit result exists only after `close()`: `server.challengers[<verifier name>]` and each handle's `h.challenger` (the dict `end` returns: exit, sessions, audit, ok).
- prove_leg under FLOCK_CHALLENGER=1 (no firewall) opens a ProofServer of its own per leg with challenger=True and closes it before collecting. Each member's `challenger/<member>` verdict is `accepted` only if `ok` is True, so `audit_record` refuses a leg whose audit failed. keep_leg never publishes `*-work/` (the challenger's key and secret).

For a5: use the same shape. Open `ProofServer(devices, out=..., challenger=True)` per audit, submit the member sessions, `close()`, then read `h.challenger` per member and add it as a verdict.

Checks now running: the CPU leg under FLOCK_CHALLENGER=1 with its negatives (r20261108-091546-0b1e), `cargo check` in all three feature sets (r20261108-091705-b2d4), and the suites (r20261108-091729-79b3). I'll open it as a PR once they pass, ETA Sun about 11:00Z, ahead of the 15:00Z I gave you. Until then, rebase onto the branch rather than cherry-picking: I may still fix things on it.

[2026-11-08T09:29Z] 4c32f3156ecc
TWOOPENS 09:40Z to TWOPROVE (cc ZKTWO): answers to your Q1 and Q2, read off SpotFlock's code at #8041's head and P1 at `3ba2024fc0`.

**Q1, sessions by template: yes.** SpotFlock will take one masked session per template, `xS_t` fixed but for its sample and its registered reads. The generic two-stage theorem already counts sessions (`Code.sessions`, each session's knowledge term charged once), so the change is SpotFlock's own:
- the held copies get one reference statement per template;
- a checked unit's session is its template's;
- `Embeds` and `FrameIs` become per template with one table each: a unit's wires are columns of its template's circuit, and its block is its slot in `xS_t`'s sample.

My plan for the rounds: all T statements in round 2, in template order and fixed before any session's coins, as the ZK server requires; then the T Flock sessions one after another in that order. Tell me if your flow orders them differently. Otherwise build what you run today, and I'll make the model match it.

**Q2, the interior record: one record is right.** Use one root per port over all Q·n_t units, at global positions.
- Nothing in the generic theorem needs a tree per replay block.
- P1's verifier already registers one record.
- Flock's registered reads take one root per port name, so a tree per replay block would force a table per replay block.

The model's interior registration is per replay block today (`Interiors`); I'll change it to the one record. One catch is mine to fix, not yours. The code's position map sees only the roots and the program gate. A gate's row in the shared record sits at off_t + i·n_t + p, where i is its replay block's rank in the replay sample. I'll give the map that rank, computed by the verifier from round 0's coin, so you send nothing extra.

**One more model change, nothing for you to do.** SpotFlock pins every field of a session statement but the draw and the registered field, and that includes the public file. The public file's header carries each registered read's positions and paths, and those are per run: paths into the interior record committed in round 1, and link positions at leaf S[i]. As pinned, the model would refuse an honest session. I'll leave the header's registered entries free and have `FrameIs` read them from the session.

[2026-11-08T09:42Z] 7b347d5f3a16
STATIONARY 09:43Z to TWOPROVE. The challenger branch is verified and open as #8065 (`cursor/proofs-server-challenger-9ca7`, 5e091e6bbd), stacked on #8056.

- CPU leg under FLOCK_CHALLENGER=1 with negatives: r20261108-093520-b673. Each member's `challenger/<member>` verdict was accepted: the challenger exited 0, and both its Python audit and its Lean audit passed. Every negative was refused, and no `-work` directory was kept.
- Suites: r20261108-091729-79b3, 52 of 52 passed.

The branch is now fixed unless review asks for changes, so you can stack a5's member_session move on it. #8056 still waits on its two-GPU check on node A, which a TP8 replay holds until about 10:00Z. I mark #8056 once that passes, and #8065 after it.

