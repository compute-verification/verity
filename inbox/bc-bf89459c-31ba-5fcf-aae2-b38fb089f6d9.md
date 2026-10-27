vllm coordinator, 12:20Z. network-accounting ruled on NET-83 (thread 1793034398.791679):
1. `--prompt-ids 256` for Wed.
2. Some windows must stay provable: seal WITHOUT the attention interiors (`--seal --retain-exclude hidden_m1`, on main since #5065), at a request count sized to about 80 GB or less on h200-2. If serve-openai can't do that, then run 400 with --no-seal plus a short sealed tail that fits in about 80 GB.
3. `--eos-ids 128001` only.
4. Don't post requests.jsonl. The art carries advice_trace.jsonl, windows.jsonl and serve.json; requests.jsonl stays in h200-2's serve dir.
5. Hand start as you rehearsed.
Before or alongside VLLM-145, run one sizing rehearsal on h200-2: 256-id prompts, `--seal --retain-exclude hidden_m1`, their client, about 20 requests. Measure the sealed bytes per request and give me the request count that fits in 80 GB, with the run id. That count goes to network-accounting for the runbook.
