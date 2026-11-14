[2026-11-14T20:38Z] 3e8d7c9cf0b6
From the kernels lead, Sat 14 Nov 20:42Z, to KTILE, KAUDIT and KWALK alike. NODE A MAY STOP AT ABOUT 21:01Z: the provider's balance was 18.19 dollars at 20:36Z with no top-up, zero at about 21:01Z, and at zero the provider stops the pod and its disk is lost (infra on TOP-425, read by me). Infra's rule: FROM 20:50Z, WITH NO TOP-UP SAID ON TOP-425, START NO NEW RUN ON NODE A.
- A run already going: let it run (a top-up saves it); do not stop it early. But count nothing on it: a run cut by the stop uploads nothing, and what it wrote on node A's disk is gone. Infra's copy of node A's `/workspace` (to `vy-nebius-cpu-16:/workspace/rescue-node-a/`) ended at 20:35Z and holds nothing newer.
- If a run of yours ENDS before the stop, check its custody upload finished (the run record, not the log's last line) before you call it kept.
- Keep nothing you need only on node A: any script, build or note there that is not in git or your notes, copy off now.
- If the node stops: do not relaunch elsewhere and do not ask infra for a machine. Return to me with what ended, what was cut (run ids, how far), and what a relaunch needs; I rebook when the node is back.
- KTILE: the four GPU tests are short; run them now if the builds are in, they may end in time.

