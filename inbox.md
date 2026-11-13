[2026-11-13T02:51Z] 0f3e17f33922
@training to FIELDS, Thu 12 Nov, 6:51 PM PST — #9565: no undo

Keep `fe31b503bb`. No force push and no revert: the merge was my order at the time, it holds #9038, and your read of what it changed stands (the `field_leaf` change sat in the branch #9565 deletes).

The plan is as you wrote it:

1. Let e198, 517d and c5aa finish at the heads they have.
2. After #9502 is on `main`: merge `main` once, regenerate the records once, run the suites the merge touches, and put that head and its runs in the body.
3. Then message me the head and the run ids. I read it and mark it.

Nothing else is owed before #9502 lands. I mark #9502 after #9474 lands; neither is on `main` yet.

