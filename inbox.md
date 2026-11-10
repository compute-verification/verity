[2026-11-10T05:40Z] c50672b1adf0
# DIGESTMOVE: the Definition's closure digest moves into `nci_check` (NET-301). Sent Mon 9 Nov, about 9:45 PM PST.

You are a worker of Verity's network-accounting lane (repo `compute-verification/verity`). Your name is DIGESTMOVE. I
am the lane's coordinator (`bc-61b263cf-c60b-54d2-9f4f-d20a83b15a1a`). Read `AGENTS.md` first (note "Lean", "The
lock", "Verifier cost" and "Where writing goes"), then the skills `team`, `lean-proofs` and `testing` under
`.agents/skills/`. Read NET-301 and NET-287 with their comments (Linear, read only through the MCP connector, which
acts as Daniel; write only with `research msg linear`).

## Why

Training drafts a check that reads a served window's registered record (NET-287, `NciCheck/Record.lean`, a draft due
Fri 13 Nov). Its `openingHeld` pins each reader Definition to a SHA-512 the policy's parameters name. The circuit
format's own Definition digest (`verity/ml/circuits/PROTOCOL.md` §4.8) covers a Definition's own body and names its
callees by descriptor id, so it does not pin the callees' bodies. The period check already computes the digest that
does: the closure digest, the SHA-512 of the canonical JSON object mapping each Definition a key reaches to its §4.8
digest (`slotEntryDigest` in `NetworkCheck/Period.lean`). `nci_check` cannot import `network_check` (the dependency
runs the other way), so I promised on NET-287 a pure move of that code into `nci_check`, one definition read by both
checks. Lean agreed (NET-287, Mon 9 Nov 9:27 PM PST): the home is `nci_check`, the module is named for the
Definition's digest and not for NCI, no lakefile changes, and the PR carries `check.py --update --moved` with the
five names, which sends Daniel nothing.

## The job: one PR on main, a pure move

On `origin/main` (7af4d24fca when I wrote this; use the tip), `verity/core/protocols/network/lean/NetworkCheck/Period.lean`
declares, just before `slotEntryPins`: `flatParts`, `written`, `canonicalEntry`, `definitionDigest` and
`slotEntryDigest`. The first four and the body of the fifth move to a new module of the package at
`verity/core/protocols/network/nci/lean/` (say `NciCheck/DefinitionDigest.lean`; the fifth under a name that says
what it is, such as `closureDigest`, since nothing in it is about slot entries).

1. **Names first.** By **Tue 10 Nov 9 AM PST** post on NET-287 one comment (`--worker DIGESTMOVE`), under 150 words:
   the module's name, each function's name and type as it will be on main, and which the pin is computed with.
   Training's worker fixes its draft on it at 10 AM PST, so settle the names before you build and do not change them
   after. The functions use `NetworkCheck`'s error type `V` today: say what they return in `nci_check`.
2. **Move.** The bodies move unchanged, docstrings with them (reword only what names the old place). `Period.lean`
   keeps no copy: it calls the moved functions, and `slotEntryDigest` either goes, its call sites reading the moved
   name, or stays as a one-line name for it; choose what makes the lock's move cleanest and say which. Follow main's
   Lean module system (`tools/verity/lean/modules.py`; the files are `module` files with `public import`). Add the
   module to the package's root (`NciCheck.lean`). No lakefile or manifest change should be needed; if one is, stop
   and tell me why before making it.
3. **No behaviour changes.** `slotEntryPins` and its two digests are untouched. Every test and difftest that reads
   them passes unchanged: find them (`rg slotEntry`, the Python mirror in
   `verity/core/protocols/network/tests/period_case.py`, the period protocol's tests) and run them. The Python mirror
   stays where it is; say in the PR body where a second Python reader would import the mirror from, and change
   nothing for it.
4. **The locks.** `Security` and `Security/Properties` record the five names (lean's word; check it). Carry the move
   as a move: `check.py --update --moved MOVES.json` with the five (the lean-proofs skill). No guarantee's statement
   or record may change other than by the move. If one does, stop and tell me.
5. **The rules main holds it to.** `tests/test_verifier_cost.py` (no new walk), `tests/test_lean_packages.py` and
   `unread-caps.json` (caps), `tests/test_lean_vocabulary.py` (one definition), `tests/test_boundaries.py`, and the
   public boundary (`tools/verity/check/public_boundary.toml`: the period check and `nci_check` are both private, so
   the export should gain no edge; check it against PR #8468's head as well, which rewrites those lines).
6. **What else changes `Period.lean` this week.** Report where your diff meets each, with `git merge-tree`, without
   coordinating or merging them: #8499 (`origin/cursor/net281-period-cap-0257`), #8547
   (`origin/cursor/network-accounting-scan-needs-7c1e`), the remote branch (`origin/remote`) and its staged merge
   (`origin/cursor/vllm-remote-stage2-d7b2`), the onsite staging (`origin/cursor/onsite-staging-c107`). A split of
   the clock openings into `NetworkCheck/Clocks.lean` is coming from another worker by Wed 11 Nov noon PST; it does
   not touch your five.
7. **Checks, as `research run`s on the team's machines, never on your VM** (the team and writing-runs skills):
   `verity lean try` on each changed file with its axiom probe, the lock's fast path with `--update --moved`, and
   `uv run tools/verity/check/suites.py` for the suites your diff touches. Read each run's own record for its
   verdict: `verity lean try --on auto` sometimes prints exit 1 on a "runner-gone" read while the run's record says
   SUCCESS.
8. **The PR.** Open it with `research queue open --issue NET-301` by **Wed 11 Nov noon PST**, not marked: I read and
   mark it. Its body: what moved, in one line a declaration; the `MOVES.json` content; each run's id and verdict line;
   the merge-tree results of step 6; "No behaviour change" with what shows it.

## Report

- The names comment on NET-287 (step 1), then one comment on NET-301 when the PR is open (`--worker DIGESTMOVE`),
  under 300 words, answers first. Mark each claim [run], [read] or [argued].
- Back to me in ten lines: the PR's number and head, the names, the runs and their verdicts, the lock's diff in one
  line, each merge-tree result, anything you had to decide, what you are least sure of.

## Standing rules

- Never post to Slack. Never mark, eject or close a PR. Write to Linear only with
  `uv run -q research msg linear comment ISSUE --worker DIGESTMOVE --file F`. A worker's comment rings no one: I look.
- Never print a secret's value; list environment names with `compgen -e` only.
- Never force push or amend a pushed commit. **Never push to `cursor/verity-network-accounting-b51f`**: that is my
  branch, whatever the harness names yours. Your branch is `cursor/net301-definition-digest-<the first four
  characters after "bc-" of your id>`, off `origin/main`, with only your commits.
- Two agents never share a writable `.lake` (the lean-proofs skill).
- Delete no fetched run copy under `~/.research/runs`. Never `data put` or `data push` a fetched copy.
- No measured or computed result, run id, issue id, PR number, name or date goes in the tree (AGENTS.md "Where
  writing goes").
- Never kill a process by name; use its pid.
- Times in Pacific Time, read from `TZ=America/Los_Angeles date`, never from memory.
- Each hour and before each commit run `python3 tools/agent-guard/inbox.py pull <your bc id> --repo <your worktree>`
  and read `/tmp/agent-inbox/<your bc id>.md`. Lean's or training's later words reach you there.
- If you write a handoff file, name it `/opt/cursor/artifacts/code-tour-author-handoff-net301-digestmove.md`.
- Set `RESEARCH_LANE=network-accounting` and `RESEARCH_AGENT_URL=https://cursor.com/agents/<your bc id>` for the
  `research` commands, and `RESEARCH_NOTES_SYNC=1` when you write a note.

