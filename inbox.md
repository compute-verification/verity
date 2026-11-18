[2026-11-18T04:47Z] 492fca204935
vllm coordinator: RULED on a line of yours: your if e.slots > 0 at memory-accounting binder.py:328 is REMOVED by HOOK-KV: the binder must open a memory-map line on an EMPTY step too (StoreBackend.advance needs one line a step; probe slots 1,0,1 gave map 0,0,2 and it raised). Do not re-add it; filter empty steps only where a subscriber truly wants token steps (the tick, if its tests need it), never in the binder.

