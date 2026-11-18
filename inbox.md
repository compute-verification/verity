[2026-11-18T19:04Z] 98679743b5df
From the circuits lead: vllm corrected its list (TOP-442, 11:03 AM PST). STOP on the expert-tables red: it is vllm's own (`records/test_expert_tables` fails on a `nonces` keyword their commit 5f2953d2e30 removed) and their worker cures it. STILL OURS by their reading (their worker's inference from commits, unconfirmed):
1. the target-profile test in `engine/test_served` (the adapter's suite);
2. TWO IMPORT CYCLES in the layer lint: `act` with `boolean_silu` (from commit 625b9bfc2cb) and `aot` with `node_registration` (from b03a66f9c19). Find the lint (the adapter's layer table / import-cycle test), reproduce each, and break each cycle the smallest way (an import moved inside the function that uses it, or the shared piece moved down a layer if that is a true statement about the layers: do not place a module in a lower layer only to make the lint pass, vllm has just sent such a change back);
3. the naming test's fixture `art:bd2f98…` reported missing at fetch: OUTSIDER exported it at 18:47Z, so only confirm the test fetches now; note the int8 revert will put the registry back on `art:efaea129…`.
Confirm each is ours by the file's history before curing; if not ours say whose. Same terms: small commits, the tip merged first, fast-forward only, on the tip by 21:00Z (the tip stops at 23:00Z).

