[2026-11-11T02:08Z] b857601001da
TRN-92 ONSITE, Tue ~6:15 PM PST. Draft PR #8780 is open, unmarked: https://github.com/compute-verification/verity/pull/8780 head acd22b726d, branch cursor/training-onsite-bound-of-record-890e, base main 2b869ed35d (benchmarks/rho is unchanged on main since; on a scratch merge with a930252c49 the default output is the same bytes).

RESULT (ran at head): periods.py's default prints reference-onsite-h200 2.99, 3.03, 3.04 a week, month, year, one datacenter and the world, set by min-cut at the declared block, equal to remote's rows to 1e-9. bounds.lock_only=false gives main's output byte for byte (886). Six touched suites pass (rho 608, nci 137, circuit-properties 17, training_slowdown 188 on my VM; repository 279, verity-graph 181 on a node, run r20261111-020103-9428); the two lints pass.

THE 18 as you ruled, in the body's three groups. Differences from what I told you before, all in the body:
1. threaded now sets THREE rows, not two: onsite-h200-frontier, remote-h200-frontier, onsite-h200-kimi-sft. It cites Nci.Guarantees.ThreadedThrottled, Nci.Guarantees.ReferenceThreaded (proved), Nci.Assumptions.WeightsIncompressible, ThetaRegistered, ReferenceLayered. None of the five open deletions (#8490, #8502, #8504, #8593, #8595) removes any: checked in each head's locks, Lean and classes.py.
2. Factors are for the family that set the figure: onsite-h200-frontier 2.95 (frozen-third; 2.88 was full), onsite-h200-llama405b 26.78 (26.55 was full).
3. onsite-h200-kimi-rlvr's sentence: the adversary picks the family with the least bound; a rank-1 LoRA was held at 2.62 by fanout-priced alone and every lock bound gives it 1, so it is now the pick, under lora-xs-r2's 1.11.
4. Not figures, but they change: 20 more scenarios' combined standing in rho.md goes contested -> assumed (reference-remote-h200 among them; periods.py's remote rows are unchanged), three elementary scenarios' status goes draft -> conditional, reference-onsite-h200-today's range top goes 867 -> 41/50.4/53.9, and the best attack over the lower bound for onsite goes 1.10 -> 320 (973 over 3.04).
5. periods.py --bounds CLASS naming a draft class now prints 1 and its header says why and which setting admits it.

OUTSIDE benchmarks/rho, not edited, yours to route: verity/core/protocols/remote/README.md "Their status" says cover sets reference-remote-h200-elementary's 1 "contested via output-premise" (with the rule: conditional, assumed). No tree file outside rho states 886 as onsite's. Training's two metric rows in TOP-389's comments quote "886 in every period, 2.98 at the corner" and "onsite 1.10 (973 over 886)"; that second metric's stated limit is "at most 2", and it becomes 320.

PAIR's line is in the body as PAIR's arithmetic (read in its note, not ported, not a test).

NOT DONE: the crossing of the move; main has no move commit yet. I restack when you tell me it is on main. Nothing else waits on you.

