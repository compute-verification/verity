[2026-11-17T20:07Z] 72a8c45450a1
KAUDIT, from the kernels lead, Tue 17 Nov 20:07Z. Read your 19:40Z note: the GLM base point `r20261117-190857-1ba2` failed `no_throttling` (2 throttles, 2.06 s). My word on it:
1. That point is void by its own check and stays in the record as failed; say so in the return with its run id. Timing the point again is not a rerun to get a pass: a void point is no measurement, and the metric needs a valid base.
2. So after the GLM walk and the two 1B points, time the GLM base point (FIRST=false, one prover) ONCE more on the same card and cores, inside our row (to 22:30Z; from 22:00Z KYARM runs on card 6 with cores 72-81, do not touch those). Before it, look for the cause: what else ran on the node's cores while it was throttled (proofs' runs on cores 72-81 and 92-95, vllm's on 40-71), the cgroup's CPU limit and throttle counters, whether the walk and 1B points show any throttling. Put what you find in the note. If the second base point also fails the check, stop: do not time a third; report both, and the GLM member stays without a value at this commit.
3. Whatever happens, run the metric's command at the end and report what each member reads.
Proofs' lead says card 6 and its cores are released to kernels the minute its second run ends (about 20:50Z); that does not change your card.

