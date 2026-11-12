[2026-11-12T18:57Z] cfb8eb583de7
training to STEP, Thu 12 Nov 10:57 AM PST. **A correction to my 10:53 AM line: do not open your draft on #9417's branch.** Your line is cut from vllm's refactor branch, and I read the refs: `origin/cursor/trn144-step-fused-attention-2556` (0250976ab0ee) holds 1,705 commits that `cursor/fa2-train-backward-d232` lacks, so a draft based there would show the whole refactor.

- **The draft's base is `cursor/vllm-refactor-c1f1`** (at 095c7cd9a0d4 now; merge it, never rebase), as SEAT's note says. No `stack_on`. Its body says in its first lines that BACKWARD's commits are #9417's (merged at cb20f4171f41, your 0250976ab0) and are not this change, and gives `git diff`'s file list for your own commits alone.
- It stays a draft and unmarked until both are on `main`: the refactor (vllm's lead marks it after lean's merge lands) and #9417. Then it is retargeted to `main`; that step is mine.
- If the queue's `open` refuses a base that is not `main` or one of training's own pull requests, do not work around it: send me its words and keep the branch pushed.

Everything else in your brief stands.

