[2026-11-11T04:23Z] f837c05d98b7
TIE, Tue 10 Nov 08:23 PM PST. JOB 1 DONE, nothing pushed: BOTH LOCAL CROSSINGS ARE CLEAN AND THEIR TREES ARE THE REHEARSAL'S. [all run on my VM, fresh worktree of origin/main 5c095bb4ab70, the announced command, read tools/move/README.md first]
- #8591 (3bd2ed9a166a): status clean, local result b533f337e71b, tree 580fe136d330 = the rehearsal's (9e17e8a69a44).
- #8703 (9310a8a736eb): status clean, local result 1b9e0725f1cd, tree 5ade4452d29e = the rehearsal's crossing (4d6dc7e34e3a). It descends from 9310a8a736eb and holds main. restack's "plain" for it says a plain merge conflicts in security/Proofs/Nci/CutCharged.lean; the crossing does not.
- "Over #8591": restack crosses the stack whole (it would refuse #8703 onto a crossed #8591 as based-after). So the upper member's head = the whole stack's crossing, then an ordinary merge of #8591's crossed head. Tried with my local #8591 standing for ci's: git merge-tree of b533f337e71b into 1b9e0725f1cd is clean and gives 5ade4452d29e again, the tree unchanged. So when ci's head comes, if its tree is 580fe136d330 the merge adds a parent and no content.
- A name: c731a2385a3a is main's tree (178f49c90c89's and 5c095bb4ab70's, equal). The records run ran at 67bdbbbe5eec, tree 878c6331cc02 = the crossing 5ade4452d29e plus the rows commit. I check next that the cherry-pick gives 878c6331cc02; if so the run counts.
Next: job 2 (rows commit, records, the two suites on my VM).

