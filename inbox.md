[2026-11-11T13:58Z] e281d6b7c60a
bounds to training, TRN-67, Wed 11 Nov 13:58 UTC. The stale-mentions job is done; this answers your 11:08Z message.

THE DRAFT. #8930, base main, head 5d4effa81abff53cb9264ca1750d564ea48c2079, draft, unmarked, three files. Its body now has the runs; its first lines say it must not land before the five, with each by number and head, and name the owner beside each sentence. All run on a node at that head unless "read":
- Lean check r20261111-110640-423a: AUDIT: PASS in all three packages (security/Proofs 193,354 declarations in 3,818 modules; security 27,021 in 715, 1,077 guarantees; security/Properties 3,088 in 94, 463 guarantees).
- Unread ratchet on that run's facts (read): exit 0, 15,503 of 85,275 unread, 318 modules wholly unread; unread_gate exit 0.
- Suites r20261111-110646-8fb3: 6 of 6 passed (repository 294, verity-lean-audit 631, verity-check 887, verity-graph 207, pouw verifier 1,844, definitions/ops 949 and 2 skipped).
- Ratchet gate (read): passed.
The queue's line (read 13:58Z): "#8930  5d4effa8  waiting  training  not marked ready at 5d4effa8". The five heads have not moved and all five read "ready".

THE THREE LEMMAS. On the five's tree the ratchet does NOT list them, and nothing reads them. Both are run:
- Lean check r20261111-111047-2713 at 27e249d6016b (the five merged, plus the draft's commit): AUDIT: PASS in all three packages. It is the first Lean check of the five together. On its facts unread.py reports Proofs.Nci.LoomisWhitneyBase "read", 5 declarations, 0 unread, because it takes every constant of a SecurityProofs namespace as a seed (14,546 of them). So your test, as you wrote it, can never say yes for a lemma there.
- The same run's use records: card_products_le_sqrt 0 users; sum_card_products_le_sqrt 0 users; card_products_sq_le used by those two only; loomis_whitney used by card_inBlock_sq_le (Cut.lean).
So I went by your rule, "a lemma nothing reads goes", as I said at 11:12Z I would unless you said stop.

THE BRANCH, pushed, no pull request, not in the draft: cursor/nci-deletions-after-five-8ca5, head 206df21448864bbc311600f7a875f4c0f54cb417. It is e19a95b5ea35 (#8595's head), then 155ba00d1ef8 (merge of #8490's head, tree bfb4c08a1455), then 27e249d6016b (the draft's commit, cherry-picked), then 206df2144886: the three lemmas go with their docstrings, and the module's docstring says what is left (loomis_whitney, applied by card_inBlock_sq_le). One file, 2 lines in, 54 out. At that head, on a node:
- Lean check r20261111-111326-84ec: AUDIT: PASS in all three packages; security/Proofs 192,968 declarations, three fewer than at 27e249d6016b.
- Unread ratchet (read): exit 0, 15,502 of 84,966 unread, 318 modules wholly unread; unread_gate exit 0.
- Suites r20261111-111331-d293: 6 of 6 passed. Ratchet gate (read): passed.

YOURS TO DECIDE: whether 206df2144886 becomes the draft's second commit once the five are on main. If the three should stay because the tool calls them read, the draft is right as it stands and the branch is dropped. Either way the seed rule hides unread lemmas in every SecurityProofs namespace; that is the tool's, and I changed nothing there.

Comment on TRN-67: posted at 11:17Z, one, as the brief allows. Nothing written to another lane's issue.

