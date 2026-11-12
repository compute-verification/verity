[2026-11-12T09:37Z] 3ad71e2f95cb
bounds to training, TRN-67, Thu 12 Nov 1:37 AM PST. The head moved, so my two earlier run ids are dead: `main` went to `33faab2be81c` (a batch of about thirty pull requests) after my first merge at `19fb19a21e40`, so I merged it again and cancelled both runs three minutes in.

- Head now `6cb6b430df2f19d455772574d5059e8a50acc383`, pushed: `5d4effa81abf`, merge of `19fb19a21e40`, cherry-pick `2558ba3cdb7d`, merge of `33faab2be81c`. Neither merge had a conflict (RAN).
- On that tree (READ): still nothing reads the three lemmas; the scans find no new stale sentence in the batch's 8,278 added lines; #8994 dropped the lock's `unwitnessed` entry of `Nci.SecurityProofs.UpdatedMatMulReads`, one of the three things I had found and left.
- Ratchet gate `33faab2be81c..6cb6b430df2f`: 0 failed (RAN, 1:35 AM PST).
- Live at that head: Lean records fast path `r20261112-093542-658f`, suites (slow tests too) `r20261112-093636-1b11`.

I merge `main` again only if a later batch touches the three files, the lemmas' readers or `security`'s locks. No reply needed.

