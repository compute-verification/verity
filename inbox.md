[2026-11-10T01:24Z] e45f0fa54d5a
Follow-up on NET-283 (https://linear.app/verity-research/issue/NET-283). Done when Sunday's export holds the network certifier and the certifier's tests pass in the exported tree. Three decisions are made; two jobs follow.

Decisions
- vllm confirmed its lines (its two comments on NET-283, Mon 5:05 PM PST and Tue 01:30Z). 37 files of the integration are public, `serve_remote.py` among them with no cut in `serve_openai.py`; 22 stay private. The edit goes into #8468.
- The two `[[needed]]` entries stay: `verity/core/primitives/world` and `Security/Proofs/Properties/Hash`. Neither is a withheld protocol's, and holding them would cost the device's library.
- The clock openings are split out of `Period.lean`: yes. I checked the region (main's lines 1284 to 1403): it is byte-identical on `origin/main`, `origin/remote` and `origin/cursor/onsite-staging-c107`, and it names nothing of NCI's.
- Not decided, and not yours: whether the period check itself stays withheld with NCI. That is top's, for Daniel. Until he rules, it stays held as #8468 has it.

Job 1: #8468, today
1. In `tools/verity/check/public_boundary.toml`, `[private]`: delete the group `served_window`, replace `protocol_tests` with the text in vllm's 5:05 PM comment, and add its `served_records` line. Copy vllm's text; don't rewrite it.
2. Run the boundary tool. vllm measured `8071 public files, 0 edges` and `public_boundary: 0 edges, 0 new` on a scratch copy; yours must agree. If it doesn't, stop and tell me the difference.
3. One new commit on `cursor/net283-certifier-public-ebaa`, pushed. No force-push, no amend.
4. Update #8468's body: the integration's lines are now in it (37 public, 22 private, with vllm's reasons in one line each group), and the file counts. Remove "the integration's lines (theirs to move)" from "Not in this PR".
5. Comment on NET-283 with `--worker PUBLICCERT`: the new head and the tool's two lines. I review and mark it; you don't.

Job 2: the split, its own PR, opened by Wed 11 Nov noon PST
1. A new `verity/core/protocols/network/lean/NetworkCheck/Clocks.lean` holding the clock openings and only the helpers they need (`exactly`, `hexAt`, `certificateOf`, `clockDigest`, and whatever else the build asks for). Same namespace, every name unchanged, no import of `NciCheck`. `Period.lean` imports it. `RecordsMain.lean` imports it instead of `Period`.
2. The smallest diff that does this: moved text is moved verbatim, in its order. No rename, no reformatting, no other cleanup in `Period.lean`.
3. Before you build anything, check each helper you move is also identical on `origin/remote` and `origin/cursor/onsite-staging-c107`. A helper that differs there stays in `Period.lean` if the clocks can do without it, or you stop and tell me which one and how it differs.
4. Then `git merge-tree` your head against `origin/remote` and against `origin/cursor/onsite-staging-c107`, and report each conflict with its file and lines. vllm's label `remote-cleared` gates this PR, and I ask for it, so I need that result first.
5. Build and check through `research run`, never on this VM: the network package's Lean build, its difftest (the `clocks` port especially), the Lean check for the packages whose inputs changed, and the lock's records if the check asks for them (names are unchanged, so say what moved if anything does).
6. The boundary lines that release `RecordsMain.lean`, `Clocks.lean`, `network/tests/test_{lean_audit,network_records_protocol}.py` and whatever else then needs no NCI: in this PR if the queue takes a PR based on #8468's branch (the team skill's "Landing"), else a follow-up once #8468 is on `checked`. Say which you chose.
7. Then the export's `--build` run on the result: `network-records` builds in a tree without NCI, and the certifier's tests that take `lean_audit.py`'s or `service.py`'s verdicts pass there. Give the run id, and the passed, failed and skipped counts.
8. Open the PR, body by the template, `Refs NET-283`. Comment the PR and the run ids on NET-283. I review; proofs reviews; I mark.

Also report
- What run `r20261109-231803-0a62` showed for the certifier's own suites in the exported tree at f7b94f9172: passed, failed, skipped, and which tests can't run without `network-records`.

Standing rules, some new since your brief
- The shell hook that delivers my messages is dead on this VM. Each hour and before each commit run `python3 tools/agent-guard/inbox.py pull bc-f5a07e61-623f-5049-bd26-29e31fa5e22a --repo <your worktree>` and read `/tmp/agent-inbox/bc-f5a07e61-623f-5049-bd26-29e31fa5e22a.md`.
- No Slack. You never mark, eject or close a PR.
- `/opt/cursor/artifacts/code-tour-author-handoff.md` is shared by every worker here and is not yours. Write `/opt/cursor/artifacts/code-tour-author-handoff-NET-283-PUBLICCERT.md`: `<decision>`, `<risk>`, `<verified>`.
- Delete no fetched run copy under `~/.research/runs` unless `research data preserved RUN` says preserved.
- Stamps and times come from `date`, never from memory.

Report back, short: #8468's new head and the tool's lines; the split PR's number and head; the two merge-tree results; each run id with its verdict line; anything you stopped on.

[2026-11-10T01:39Z] bc005cff0144
From network-accounting, Mon 9 Nov 5:39 PM PST. An addition to your step 2 brief on NET-283 (the brief in your inbox stands otherwise).

top ruled on NET-283 at 5:30 PM PST: Sunday's tree has the certifier without its period check, as #8468 has it. The move that makes the period check public is NET-288, after `remote`, and is not yours. Daniel can still choose on Wed 11 Nov to publish the policy instead; if he does I tell you.

One addition to job 2 (the clocks split's PR), before you open it:
- `verity/core/protocols/certifier/README.md` names the period check twice (`NetworkCheck.Period`'s `wired`, near lines 247 and 265 on main). Where it first names it, add one sentence: the period check, the step from a period's records to the network bound, is not in the public tree, because it reads the inference-only policy's layout check, which is withheld.
- It must be true in both trees, so write "the public tree", not "this tree". No issue id, date or name in it.
- Include the README in your two merge-tree results (against `origin/remote` and `origin/cursor/onsite-staging-c107`).

