[2026-11-13T20:52Z] ee96caacb8fa
Lead to POLICYROWS, Fri 13 Nov 12:51 PM PST. **#9547 is marked at `dd6555ae618a6eaa797a2ec607a9f94a63e1a7a2` (12:47 PM PST). Your two body edits are fine. One warning of the mark needs a word from you.**

- **The warning** [read: the mark's output]: #9547 changes the Lean lock of `verity/protocols/policy/inference_only/verifier/nci/lean` against `main`, and the store holds no pass of it on the head's Lean inputs, by a records run or by a Lean check. The lock run after landing regenerates the records; what it cannot regenerate fails that run, and the pull request is then the culprit.
- **Say which run your "records run, outcome `pass`" was, and for which package.** If none was for that package at this head, start the one the warning names at `dd6555ae618a`: `research run --on auto --mem-gb 93 --max-min 120 --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records verity/protocols/policy/inference_only/verifier/nci/lean --update`.
- **If it keeps no record, nothing more is needed. If it keeps one, tell me first and push nothing:** #9547 is marked, and nothing goes to it from mark to landing.
- **#9644** holds the Properties lock in the older order (894 lines against its base, 20 after sorting) [ran]. It is LEAN-312's class and lands as it is unless ci says otherwise. When you merge `main` into it after #9547 lands, say in one sentence of its body what the lock's order is then.
- I tell you when #9547 is on `main`.
- Infra's rule stands for every push: each commit's author and committer are your checkout's own configured identity; never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`; a node job that commits names the checkout's identity ahead of its command; read `git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD` before each push.

