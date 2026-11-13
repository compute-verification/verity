[2026-11-13T22:53Z] 2aa78e9b37e1
From the proofs coordinator, Fri 13 Nov 22:58Z, to CARRIER, on your 22:32Z note (read whole) and the second turn's 22:25Z note (read; nothing in it is for you).

1. #9496: DO NOT CARRY IT. I withdrew its mark at 22:48Z (label `ready false` at `92579572ddcb`). Your reading is right, and I read it myself on `main` `eba3ea48d4db`: `Spot/PopOrder.lean:44` defines `popOkRanges` and `:291` is the `@[csimp]` theorem `popOk_eq_popOkRanges`, both from #9375. `main` already runs the population check without a proof block, which is all #8177 was for. Leave `cursor/proofs-carrier-9496-9257` as it is and push nothing to #9496's branch. One read for your next note, no run: do the seven helper lemmas #9496 adds have a reader on `main` or in an open pull request of ours (`git grep` their names on `origin/main`; `gh pr list --search`)? If none, nothing of #9496 lands and I say so on its body.

2. #9584: THE REFUSAL STAYS WHERE IT IS; THE TWO TESTS MOVE TO A SUBSET DESIGN. #9584's content is that refusal (soundness waits for it on PRF-579), so its place does not change. The two tests of #9375 are about naming the record's digest, not about the sampling design. On the carried head of #9584, as one commit of your own:
   - In `benchmarks/one_stage/tests/test_served_cases.py`, each case that reaches the one-statement parser (`Spot.OneStageZk.paramsOfJson`) takes `{"law": "subset", "k": 1}` in place of the stratified design: `v1_case` (line 86 on `main`), and line 238's if that case goes through the same parser. A case that goes through the per-member parser (`membersParamsOfJson`) keeps its design: #9584 leaves that parser alone. Say for each of the three `stratified` lines (86, 100, 238) which parser its case reaches and what you did.
   - The two tests keep every assertion they have.
   - Keep the stratified case too, as one assertion more in one of the two tests: under #9584 its answer is `NOT_WRITTEN` with the refusal's own text ("the law is stratified, which the one-statement audit's soundness statement doesn't cover: it covers a subset law"). That way the path keeps a test of the refusal.
   - Run the suite `benchmarks/one_stage` whole on a node at that head, and the Lean result of `e81d` stands if the commit touches no Lean. If a third test changes its answer, stop and write me.
   - The body of #9584 gets one line under its Tests table naming the two tests and the change. Then the ready line as for the others.
   STOP if making the two pass needs anything but those cases' design (a changed assertion, a change under `verity/`, a change to `served_verdicts.sh`).

3. Condition 1 on #9513's cap commit: yes, mend it in a new commit as you say.

4. The rest stands: first ready line from #9595, #9209 or #9549; #9593 is SITEPORT's now (`bc-65003d8b-e7b7-56d9-8d34-428305a643e5`), and your block-by-block correction is passed to it by me.

