[2026-11-11T01:24Z] c02d0c0929dc
TRN-92 ONSITE (bc-a312d8c9-549e-5c1c-a94c-9e1d65f3890e), to your 5:21 PM message. The scan under rule A is NOT clean: I stop on (ii); tonight's draft carries (i) only, with (ii) as its open question, unless you say otherwise. No tree change for (ii). Run by me at main 2b869ed35db5, scratch: scenario._refuted patched to also refuse a class one of whose own cites starts with draft: (fanout-priced, fanout-split, accumulator, output-premise); all 41 scenarios, week/month/year, one datacenter and the world of each location.

DEFAULT SET, clean: reference-onsite-h200 886 -> 2.99/3.03/3.04 (2.98725, 3.02850, 3.04022), set by min-cut, datacenter and world alike; reference-remote-h200, -sampled and reference-onsite-h200-today unchanged.

18 MOVE under the rule; the flip moved 13.
Same 13, same figures as the flip (week/month/year, set by min-cut unless said): reference-onsite-h200 886; -per-device 606; onsite-w63 982; onsite-h200-kimi 892; onsite-h100-kimi 892; onsite-b200-kimi 866; onsite-h200-kimi-5gb 728; onsite-h100-kimi-5gb 728; onsite-h200-rack4-kimi 738: all -> 2.99/3.03/3.04. onsite-h200-llama405b 480 -> 2.30/2.33/2.34. reference-onsite-h200-elementary 1.204 -> 1 (cover). onsite-no-pouw-h200-kimi 892 -> 1 (cover). onsite-h200-frontier 799 -> 2.98652/3.02774/3.03946, set by threaded (the flip gave 2.98709/3.02833/3.04005 by output-premise, which the rule also drops).
5 MORE, which the flip left alone, each set at the declared block (2e7 bits) by a draft class today:
- remote-h200-frontier (REMOTE): 2.98709/3.02833/3.04005 by output-premise -> 2.98652/3.02774/3.03946 by threaded. Same to three figures.
- remote-h200-kimi-rl (REMOTE): 2.98641/3.02763/3.03935 by fanout-priced -> 2.98466/3.02584/3.03754 by min-cut. The week prints 2.99 -> 2.98.
- onsite-h200-kimi-rl: same figures as remote-h200-kimi-rl, fanout-priced -> min-cut.
- onsite-h200-kimi-sft: 2.98650/3.02773/3.03944 by output-premise -> 2.90643/2.94546/2.95655 by threaded.
- onsite-h200-kimi-rlvr: 1.11 -> 1, set by covering both times.
No unmonitored scenario moves. Datacenter and world agree in every row.

If you want A anyway, the two remote rows and the three onsite ones go in the PR's table as the rule's own moves (none is the finding's); or the rule can be narrower. Say which, and I code it with the switch and its test.

THE TWO SENTENCES (three, with w63):
- onsite-w63: its serving block is 4.8e10 bits, below one layer's update (6.56e10 bits), where the held-layers construction finishes nothing, so the draft bound is unbroken there; it moves 982 -> 2.99/3.03/3.04 by the rule (a draft bound sets no published figure), not by the finding.
- reference-onsite-h200-elementary (1.204 -> 1): it counts an uncertified or dark gate at one counted operation (no straddling premise), under which the inference cut bounds nothing, as on remote where the same scenario already prints 1; only the serving cut held it at 1.2, and min-cut states no beta past 8e8 bits, so it bounds no serving block (3.2e11 bits).
- onsite-no-pouw-h200-kimi (892 -> 1): with no PoUW none of a server's time is certified, so only the wipe's serving cut can bound it, at a block of 4.7e11 bits; min-cut states no beta past 8e8 bits, and no other bound in a lock limits a block of that size.
One thing those two show, for the page: the bound of record's onsite 2.99/3.03/3.04 is set at the declared block (2e7 bits) by the inference cut, the same figures as reference-remote-h200; under it the wipe's serving cut adds nothing.

(i) is pushed: branch cursor/training-onsite-bound-of-record-890e, head 29c5bdde21 (held_layers, the 1.935 pinned, the size from which it works). I add the test that it finishes nothing at w63's size next, run the suite and the two lints, and open the draft.

