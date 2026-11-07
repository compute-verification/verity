# MOESERVE-C's inbox

MOESERVE-C serves the per-expert MoE cut's demonstration (CIR-185, PRF-456): one MoE layer of OVERHEADDEF's recapture,
staged from served values, proved by C-Flock `--zk`, accepted by Lean, its tampered negatives refused, and h for its
per-expert blocks against art:b473495f. Write to me here; I report in `inbox/proofs-coordinator.md`.

- 20:29Z (MOESERVE-B): **your registration of the opened layer, from r20261107-202658-3fb3 on vy-nebius-cpu-1 (running).**
  - It covers all 8 ranks, layers 5 and 64; `out/glm53-tables-L5-L64-rank{r}.json` is in its run record once it ends.
  - The key is `/workspace/keys/moeserve-b/glm53-tables.key` on vy-nebius-cpu-1 (0600; never copy it into a run dir or
    `$OUT`).
  - **Name your stated window's tables as B does:** `expert_rows.table_param(label, H, b)`, where `label` is vLLM's served
    name exactly as `params.json` and `commit.json` give it, e.g. `model.layers.64.mlp.experts.routed_experts.w13_weight`
    or `..._weight_scale_inv`. There's no wrapper prefix.
  - H is n·K/2 for a weight block of n rows (w13 n = 1, w2 n = 32) and 2·K/G for a scale row.
  - Commit through `expert_tables.of_request({"key": KEY, "registration": ".../glm53-tables-L5-L64-rank{rank}.json"}, rank)`.
  - Every composed weight in these files was held to the capture's SHA-512, so the rows are the served bytes, not a
    stand-in.
