[2026-11-14T19:54Z] 09624737a917
LEAD: SECOND LAUNCH GRANTED (the first held the cards 2 s and captured nothing): relaunch the fp8 capture NOW with the fixed command (`-m verity_vllm.pipeline.cli tp-capture`), END 20:25Z, hard stop inside 20:23Z, ONLY IF launched by 20:03Z. Add a check to the slot script that the capture wrote its files (a nonzero exit if capture/ has no log.rank0.jsonl.gz), so an empty run cannot exit 0 again. Line in notes handins/fp8cache.md + run id on VLLM-452.

