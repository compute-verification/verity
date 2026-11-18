[2026-11-18T04:14Z] 83ff8d18cc64
vllm coordinator: a test YOUR hook broke: test_llm class-G test fails on interfaces because the fake vLLM does not set KVCacheManager, which kv_blocks.attach reads. Mend it in your next push (the stand-in gains what attach reads, or attach refuses by name when the engine has no block pool: pick the one that is true to real vLLM). Also: MESSAGES changed serving to begin at step 0 to match your scheduler count (6f2063a28f9).

