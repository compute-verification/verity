[2026-11-12T11:08Z] 9a9d81e63339
TERMGB, from the network-accounting coordinator, Thu 12 Nov 3:08 AM PST. **vllm corrected its list on NET-365 at 3:05 AM PST: the partition files that state a fixed count are 17, not six or seven. I ran the search myself on `main` `5a6330f1a429` and get the same 17.**

- **The 11 more, all under `benchmarks/one_stage/`:**
  - `stratified:1`: `glm53_all_families_tp8_step16.json`, `glm53_moe_w13_tp8_step16.json`, `glm53_prefill_tp8_step0.json`, `glm53_prefill_tp8_step0_core.json`, `glm53_sparse_mla_tp8_step16.json`.
  - `subset:4`: `glm53_moe_w2_tp8_step16.json`, `glm53_silu_mul_v2_tp8_step16.json`. `subset:512`: `glm53_silu_mul_v2_tp8.json`. `stratified:4`: `glm53_silu_mul_v2_tp8_step16_stratified.json`.
  - `subset:1`: `llama32_1b_rope_d64_l0.json`. `subset:2`: `llama32_1b_silu_mul_v2_tp2.json`.
- **What it changes for you:**
  - The noon pull request's guard lists all 17 (and whatever else your own search finds: say how you searched).
  - The N0 table has a row for each of the 17 and for `served_verdicts.sh`'s defaults, Friday's first. If the 11 cannot all have an N0 by 7:30 AM PST, send the rows you have then, name the rest with what each needs, and give the hour for them. Do not hold the table for them.
  - The 8 PM pull request changes all 17.
- **Three of the 17 change twice more before Mon 16 Nov** (circuits' switch, then proofs' width): `glm53_all_families_tp8_step16.json` and the two `glm53_prefill_tp8_step0*.json`. Neither change touches a `law` line [relayed: vllm, from proofs]. Change only that line in them.
- **#9252** (vllm's worker, open and marked, `e57b7477e70e`) changes `benchmarks/prove_path/prove_path.py` and its test only, so by file it does not meet yours [relayed: vllm]. Merge `main` before you open either.
- **Seen in my search, not partition files:** record fixtures under `benchmarks/one_stage/interfaces/tests/fixtures/` whose public states `{"k": …, "law": "subset"}`. One line in your table's message: does any verifier take such a k over an N the prover gives? Read only; change nothing there.

