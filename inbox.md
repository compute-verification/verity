[2026-11-10T20:50Z] 56b89986e94f
TRN-80 TIE, final (Tue 10 Nov 12:50 PM PST). All four items are done and both PRs are marked and admitted. What is left waits on #8589, #8703 and #8700 landing. Nothing needed from you now.

1. #8591 (the beta(k) step): head 3bd2ed9a166a71870f43e77a8bfaa309585e3a64, marked, admitted. Suites r20261110-175551-89c2: 7 suites passed.

2. #8703 (items 2 to 4): https://github.com/compute-verification/verity/pull/8703, head 9310a8a736eb8f11b0db9338d72cece5ecc9fb79, base main, stacked by merge on #8591 only, marked and admitted. Suites r20261110-200644-f63c at that head: 11 suites, 11 passed, 0 failed. The records gate took r20261110-180637-c614. The kernel replay was not run (fast path).

3. THE FUNCTION: rho.min_cut.bound_charged(c, k, n=None, *, blocks, untied, untied_macs, w=inf, most=inf, word=32, N, O, size, mac_gates=None, beta_w=16, beta_x=None, b=16, rows=False) -> Charged. Beside it: best_w(c, k, n=None, *, blocks, untied, untied_macs, most=inf, word=32, **work) and capped(slowdown, uncredited). Charged gives slowdown, gates (the pooled total), share, reading (the premise's name, which heads its column) and rests_on. w is an input; untied is a parameter and its bound a named premise; a charged block stays in at its whole work.

4. THE NUMBERS AS THE FUNCTION GIVES THEM, served class (4, 32, 4), tied k = 146,080, cap at s = 0.2315% in brackets:
- no untied word: 37,653.4 (427.1), bound's own number, whatever w and the block count. One execution phase's worth of words under one cap: 6.3. Every block charged: 1.
- one node, by blocks (best w; charged): 2,379 (335,684; 0) 3,677.1 (386.6). 1e4 (335,684; 0) 7,883.9 (409.5). 54,300 (223,789; 2) 12,564.7 (417.6). 1e6 (223,789; 2) 28,795.0 (425.6). 25,165,824 (20,344; 32) 35,550.2 (426.8). 1e10 (20,344; 32) 37,647.8 (427.1).
- datacenter week / month / year (1.651e15 / 7.077e15 / 8.61e16 blocks, w 20,435, 410,653 charged): 37,653.0 / 37,653.3 / 37,653.4, each (427.1). World (6.191e18 / 2.653e19 / 3.228e20 blocks, w 20,435, 1,539,785,918 charged): the same three. These rows are SCALED from the census and the one class.
- the other two readings on the same rows: one beta_x at the top cap, no block charged: 13,529.4 (418.6) from 1e10 blocks up. A beta_x a block, no threshold: 37,653.4 (427.1) for every datacenter and world row.
- pooled, datacenter week: the blocks hold 2.281e28 gates, training's at most 6.059e23 (0.002656%); at the top cap 1.686e24 (0.007391%). The gates are every two-input gate of the gate table, not AND gates alone.
- the brief's appendix table is NOT reproduced: its columns are another form (w = 0, w = 1,582). The tests pin only what agrees with the Lean form, as the issue says. 1,000 after the cap needs s under 0.0973%.

5. LEAN: Nci.Cut.Guarantees.HECharged is STATED, with its two witnesses (HECharged_honest, HECharged_nontrivial). It is NOT proved outright: HECharged_of_countsOn proves it from the hypothesis CountsOn, which nothing in my file proves. Its proof needs #8700 on main (bound_on_rows or bound_on_caps_he, with card_whole_eq and outOn_le); then CountsOn and countsOn_rows go. Lean try r20261110-173559-bc18: exit 0, axiom probe clean on 9 theorems.

6. WHERE HE'S COUNT ENTERS THE PERIOD'S COMPOSITE: Nci.Guarantees.DampenerRemotePeriod (Security/Definitions/Nci/Guarantees/RemotePeriod.lean) reads the sum over days of Cut.workHE (n i) k (beta_w * beta_x) b (Wk i).O inside efficiency. It reads workHE at ONE k, not HECharged or workHEOn. So nothing in Lean carries the charged form into the composite yet: the charged blocks' and the feed's MACs would go to its cheating term, and the uncredited share is its unregisteredShare. That join is not stated.

7. THE LOCK: against base e7b03e4a6c89, as JSON, three shape.frozen entries listed unstated (HECharged, HECharged_honest, HECharged_nontrivial). The regenerated record also puts three entries of other guarantees in key order. Nothing else.

8. EVERY RUN: r20261110-160400-fb92 (cancelled). r20261110-160607-71e3 (suites passed, head moved). r20261110-164641-5533 (Lean try passed). r20261110-165459-5057 (Lean try failed). r20261110-170436-52a9 (Lean try passed). r20261110-173559-bc18 (Lean try passed). r20261110-174351-489d (cancelled). r20261110-175551-89c2 (#8591, 7 passed). r20261110-180637-c614 (lock records, AUDIT PASS). r20261110-185604-c0e9 (10 of 11, rho failed one test, fixed). r20261110-193829-cf67 (cancelled before a slot). r20261110-200644-f63c (11 of 11 passed).

9. OPEN, AND WHOSE:
- the command's second print (tied.toml, --uncredited-share with no default, the pooled total in gates): MINE, once #8589 and #8703 are on main. Neither is yet; main is 5482988e3496. I tell you if #8589 is not on main by Fri 13 Nov 10 AM PST.
- HECharged's proof and the two deletions: MINE, once #8700 is on main. #8700 is lean's (UNTIED), open, not marked.
- PoUW's period bound on the untied words, an untied word's bits, and the uncredited share: compute-accounting's (CMP-380). No Lean statement; the function names them as premises.
- the join from the audit's partition to the set of blocks under w: owed by the guarantee that reads the bound; nobody states it. Yours to place.
- (H) and (E): conjectures, checked on no circuit. (H) is false of a work that reads a registered embedding or weight; the docstrings say so, no code refuses it.
- the allowance of uncredited words in the policy's k: held for CMP-380; nothing built on it.

Hand-off note: lanes/training/20261110T2048Z-checkpoint-trn80-tie-handoff.md (state, commands, traps, the body's generator). No timer set.

