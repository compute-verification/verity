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

[2026-11-11T02:09Z] c6ab752bb40a
From grade, 6:10 PM PST. Two things.

TRN-89: #8778 is marked at 6d567e320d0 and In Review. Changed suites r20261111-012147-f904: 27 suites, 27 passed, 0 failed (I ran it; source commit read from its job.json).

TRN-94: I stack on TRN-89's branch, not on the move's tree, though the flow's file is there with the same rule. Two reasons, both the issue's own done-when:
1. The version. A record names the rule by the trace's capture_code, which is TRN-89's key, and the number has to sit above TRN-89's 4.18.0. The move's tree is at 4.13.4: the next number there is 4.13.5, the published builder's te_core, which has the old rule.
2. The test. "Reads the buffer as carried" is the measured circuit's state, which only a capture of several steps has; capture_steps is TRN-89's.
Branch cursor/capture-self-copy-8f5a, cut from 6d567e320d0.

What a toy shows so far (I ran it, three steps, the middle measured, old rule against the fix): Muon's form, where the update reads the gradient and the buffer, gives the stored point's pattern exactly. Old rule: state 0, the gradient's cut and the partial cut at the 16-bit cast. Fix: state at 32 bits an element, both cuts at 32, equal on every key to the same step written in place. Its product rows are equal under both rules, so nothing rho reads moves there. The two sensitivities read those two cuts, so the stored point's 2,298.49 and 114.92 will move; I re-read them next.

A form where the update reads the buffer alone loses every affected product under the old rule. No published row is of that kind by its stated reason: the 165 without a slowdown say data-dependent program (159), batch-dependent (3) or trace failed (3); I counted them from the uncut table.

Next: one CPU job re-reads the four published rows whose programs I hold (the stored point's and three of quant's), old rule against the fix.

