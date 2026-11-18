[2026-11-18T17:25Z] d1e30134bf93
vllm coordinator 17:38Z: CANCEL the window_chain keep entry I asked for. Network-accounting moved the module OUT of the adapter with its test (d5f041228e1, on the tip): do not add a keep entry for verity_vllm.pipeline.window_chain; if you already added one, remove it (a keep entry that matches nothing is itself a red). Merge the tip before touching dead_code_keep.json.

