[2026-11-20T02:49Z] 18cf12c40cc2
From the proofs lead 6:50 PM PST: lean's (A) (14058d161c9) moved the recursion setup file's generic lemmas to Proofs.Flock.Claims.DeployReads (namespace FlockDeploy.BaseSetup). Your Session mend's body_bind_ok / body_bind_yield join them there at the merge (EDGES resolves); do not touch Session further. EDGES builds a scratch merge of tip + (A) + your branch + the refusals branch tonight and posts what is red: use that list.

[2026-11-20T02:51Z] 179b3d495339
From the proofs lead 6:52 PM PST: infra reopened a pool for us: vy-nebius-cpu-3, proofs pool, 14 slots, 385 GB free. Send your runs there now: --on vy-nebius-cpu-3 --pool proofs (or --on auto). cpu-1 has 22 GB free of 755: do not wait on it.

