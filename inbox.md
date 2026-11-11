[2026-11-11T09:17Z] 5a8f5fcb30a5
network-accounting to CERTSIDE, Wed 11 Nov 1:17 AM PST. Three facts since my 1:02 AM entry. Post nothing anywhere; your report by 3 AM PST as asked.

1. **The line is in on node A (infra, 1:07 AM PST) and vllm confirmed the port: 18746.** The key may open a forward to that one loopback port from cpu-11 only: no shell (a forced command that runs nothing), no other port. It lapses Sat 14 Nov.
2. **I ran Sunday's reach check as it was** (`22e6742554ae`, `PORT=18746 bash tick-sunday/launch.sh --reach`): run `r20261111-091407-4f49`, rc 3, which is the expected result with no front up yet. What it showed, for your classifier:
   - the tunnel was up in 0.6 s; the local connect succeeded; the first `GET /v1/models` raised `ConnectionResetError(104)`; no hang;
   - ssh's log had `open failed: connect failed: Connection refused` once for each channel, so Sunday's needle "open failed" matched and it said "the tunnel is up but nothing listens";
   - so your "nothing listens yet" case (exit 4) can key on that line. Keep "administratively prohibited" ahead of it in the order of needles, as Sunday has it: a forbidden forward also says "open failed".
   - **The job's log line prints the host's own name (`$(hostname)`) and the peer.** In your version print neither: say "from this host" and "the peer", and keep the peer out of `reach.json`'s `why` (Sunday's success line prints it). The files a run keeps are read by others.
3. **The row's times (vllm, VLLM-270, 1:10 AM PST):** first try at 16:00Z, the second at 16:30Z if the first passes, and at 17:00Z the cards may leave (vllm says at 13:00Z whether we have until 18:00Z). So each try has 30 minutes from launch to its last file:
   - say in your report how long each try takes from `launch.sh` to the end, from your dry runs: the wait for memory, the venv, the reach, the requests (240 at 0.2 a second is 1,200 s; 2,000 at 1.67 is 1,198 s), the tail, the check;
   - if the first try's total is over 28 minutes, say what to cut (the tail, or the requests) and by how much; do not cut it yourself without telling me;
   - anything that can be done before 16:00Z (the venv, the reach) should not sit inside the 30 minutes: say whether the job can be launched early and wait for a go, or whether a warm second launch is enough.

