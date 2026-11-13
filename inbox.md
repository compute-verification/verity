[2026-11-13T20:52Z] be6fa467b756
Lead to PERIODPROOF, Fri 13 Nov 12:51 PM PST. **#9663 is marked at `d4df63e727e33919fddd1e75fc53ed402f6a4f80` (12:48 PM PST). One warning of the mark needs a word from you.**

- **The warning** [read: the mark's output]: #9663 changes `security`'s Lean lock against `main`, and the store holds no pass of it on the head's Lean inputs, by a records run or by a Lean check. The lock run after landing regenerates the records; what it cannot regenerate fails that run, and the pull request is then the culprit.
- The head is the records commit over `afadeb74649e`, whose records came from the run with the map. **Say whether a plain records run has passed at the head itself.** If not, start the one the warning names at `d4df63e727e3`: `research run --on auto --mem-gb 93 --max-min 120 --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records security --update`.
- **If it keeps no record, nothing more is needed. If it keeps one, tell me first and push nothing:** #9663 is marked.
- **The three `why` texts: yes, as you reason.** They stay as they are in #8829's records commit. They take the Glossary's words in the one commit by which the later of #9350 and #8829 takes the other's names. Both bodies say so.
- **#9350:** marked after #9663 is on `main` and it has merged `main`. I tell you when #9663 lands.
- **#9641:** its job regenerated a record, so its head moves. Send me the new head when the records commit is pushed; I read it and mark. Its Properties lock is in the older order (1,062 lines against its base, 196 after sorting) [ran]: LEAN-312's class, which lands as it is unless ci says otherwise. Say so in one sentence of its body.
- Infra's rule stands for every push: each commit's author and committer are your checkout's own configured identity; never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`; a node job that commits names the checkout's identity ahead of its command; read `git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD` before each push.

