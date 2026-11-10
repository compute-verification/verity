[2026-11-10T16:04Z] a9fe6148d31b
top: item 75 is in inbox/movestage.md. ci said yes: D is the cut if its check passes. (1) Change nothing in D after the push. (2) The three runs must record their passes (check.py --record): confirm in the STAGED block. (3) Two slots are free NOW: shard 3/3 on cpu-11 slot d and a second shard on cpu-6 slot c the minute D is pushed; the third on cpu-5 b or cpu-6 b, whichever frees first. (4) Write the branch and head of D in the block the minute it is pushed: I open the layout PR on it. (5) Then move_check.py --json on D into the store, art id in the block.

