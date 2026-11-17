[2026-11-17T23:08Z] 0db9a04c2828
PARTY checkpoint: r20261117-213840-4b25 at #10730 400f5d97c268. verity-cli passed (304). verity-vllm is still running in its parallel workers, with nothing red so far. The failing test was red at #10663 and #10672 only (r20261117-213155-f01b, r20261117-213154-1006) and green at #10698, #10730 and 9b69. It is green at the new #10663 de89da8edde1 (r20261117-213840-ffc7). I return when verity-vllm ends, or by 3:40 PM with what I have.

