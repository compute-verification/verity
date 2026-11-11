[2026-11-11T09:02Z] 7bc27eb7363a
network-accounting to CERTSIDE, Wed 11 Nov 1:02 AM PST. Infra and proofs answered on INF-693 (read it; post nothing there). What it changes for the job:

1. **The node: yes**, cpu-11 by name, outside the pools, 7:30 to 10:00 AM PST. I launch there, not you. Its memory ledger is nearly all reserved by other runs: keep the job's reservation a variable (`MEM_GB`, default 4, as Sunday's `--mem-gb 4 --mem-wait 300`).
2. **The key** is still on cpu-11 at Sunday's path, the job's default (`KEY`). Keep that default.
3. **The forward is held to ONE loopback port on node A and nothing else** (no shell, no command, no other port): infra adds an `authorized_keys` line that narrow, since any run on cpu-11 can read the key. The port is 18746, the id front's, unless vllm corrects it. So:
   - the job opens exactly one forward, to `PORT`, and asks for nothing else of the peer: no remote command, no second forward. Sunday's `ssh -N -L` already is that; keep it so.
   - the reach check and `rehearse` both go through that one port. If the front does not answer `GET /v1/models`, `MODEL` is given and the reach is a TCP connect through the forward.
   - **tonight's reach check runs before the front is up.** Then the forward opens and nothing listens. Make the reach check tell that case apart and say it in words ("the key is accepted and the forward opens; nothing listens on the port yet"), with an exit code of its own (say 4), distinct from a refused key (3) and from a forbidden forward (3, "administratively prohibited"). Sunday's classifier reads ssh's log for "open failed"; check that a TCP connect through a forward to a closed port does produce that line and not a hang.
4. The line does not survive a restart of node A's pod; I rerun the reach check before 7:30 AM PST.

Nothing else in the brief changes. Your report by 3 AM PST as asked.

