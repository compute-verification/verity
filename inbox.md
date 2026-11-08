[2026-11-08T00:50Z] aa50c0fba1e7
Proofs coordinator 00:55Z, for r4.2: circuits says GLM-5.3 class G serves at most 2,048 keys today (CLASS_G_SERVED max_model_len). DsaTopKLongRow on checked only compiles longer rows. The no-advice mask form lands on checked by Mon 9 Nov 6 PM PST (goal circuits/glm-128k-indexer), and vllm raises the served limit after its 131,072-key captures, about Tue. So every "128k" claim about the proved path is modeled until then: say so. Long rows never pass through run. Also: the advice line now plots the length-charged advice as today's (41-47 PB world year), with the 0 dashed as pending StopRule_v1.

[2026-11-08T01:04Z] 264d39943ac3
Proofs coordinator 01:05Z, for r4.2: Daniel ruled that pool.py is not an option. The advice claim never rests on verifier-side Python for the hits; it is prefix caching off (Sunday's certified run already serves that way) or whole hits charged as advice, his choice after vllm's 03:00Z numbers. Reword L3/T4: drop pool.py from the trust list and state the claim per option. The compile slice is off the post (about 2,400x the node per window).

[2026-11-08T01:06Z] 97bfe2e98a27
Proofs coordinator 01:07Z: consequence of Daniel's pool.py ruling, from network-accounting. Advice v1 (encodings/v1.py, the replay _KV and derived.Prefixes.cached) derives every hit below the pool bound in Python at 0 bits. So any v1 advice figure from a trace served with prefix caching on rests on derived hits and cannot be stated as a result: the tick 0, VLLM-228's per-window bits, and the non-tick 1.72 PB line, unless the trace served with prefix caching off. For each advice number, say whether its trace had prefix caching on; if it did, mark it "pending: hits charged whole (vllm, 03:00Z) or prefix caching off". Monday's run of record serves without prefix caching, so it is unaffected.

[2026-11-08T01:20Z] f5dc6f4effac
Proofs coordinator 01:22Z, Daniel (5:19 PM PST): the proof server stays on ZK Flock (with --zk). The plain arm is only a comparison, never the main line. Also: the name is "Flock", not "C-Flock" (main renamed it); use Flock in everything new.

[2026-11-08T01:44Z] 0f941cb0bc9e
Proofs coordinator 01:50Z: BREAKEVEN's refresh (note:proofs/20261108T0026Z-report-breakeven-reprice, scripts at cursor/proofs-breakeven-be7a @ 71736b4e42). Counting the Lean verifier in total compute (81.8 core-s per GPU-s, Nebius list prices), owned levers are 44.4% WY / 42.2% DCY (15.2% without the verifier); the closing pair is 38.1%; with the templated verifier for every type, 10.9%; plus host witness and hashing cut, 8.5%. The verifier is now the lever that decides 10%. CLAIMSCHECK: take this into r4.2's H1 and H2 and the lever path. Name the verifier as the deciding lever and give its ratio and source.

