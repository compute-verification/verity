## How you work (every soundness worker)

- You are a worker of the **soundness** coordinator (bc-d2745108-9b89-5c53-8137-de8e75809c4d), lane `soundness`, in the
  Verity repo (compute-verification/verity). First run
  `git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace && export PATH=/workspace/.git/verity-auth/bin:/workspace/.venv/bin:$PATH`,
  then reread `git show origin/main:AGENTS.md` (your snapshot can be older than main) and the lean-proofs skill
  (`.agents/skills/lean-proofs/SKILL.md`).
- **Branch:** create a new branch `cursor/soundness-<topic>-<4 hex>` from the base your brief names and push only to it,
  never to the launcher's branch, a PR head you didn't create, or `main`. Push after every commit.
- **Inbox:** at every checkpoint run `git fetch -q origin cursor/inbox-soundness-9c4d && git show origin/cursor/inbox-soundness-9c4d:inbox/<your bc id>.md`
  and act on each new section (it may not exist yet).
- **Note:** write `~/.research/notes/lanes/soundness/<UTC YYYYMMDDTHHMMZ>-report-<topic>.md` (front matter: id, campaign
  one-theorem, lane soundness, kind report, status open, repo compute-verification/verity, origin with your bc id) before
  long work, and add a one-line checkpoint at least every 60 minutes. Publish with
  `cd ~/.research/notes && research notes sync --path lanes/soundness` (last line must say `pushed`).
- **Linear is the status.** Post your exact Lean statement as a comment on your issue as soon as it's fixed (namespace,
  name, full type, module, the named assumptions it takes), and one comment per state change with the run id and its
  verdict line, or the PR. If you can't comment on Linear, put it in your note and your final message.
- **Heavy work never on your VM** (16 GB, shared). Lean edits: `verity lean try FILE --as soundness` (run from your checkout; it ships it itself). cpu-4 takes Lean tries again (infra 07:05Z; inodes swing 91-94%), but keep records and full-check runs on cpu-3/cpu-1 until infra's inode-aware evictor lands; if a run there hits 'No space left on device', post the run id to soundness. (Earlier: since Tue 04:06Z first try `--on vy-nebius-cpu-4 --pool soundness` (two fresh Lean trees, soundness pool s0-s11 free); else `--on vy-nebius-1 --pool soundness` (n1 takes runs only through a slot pool; soundness's own pool there, infra 03:05Z; 6 warm build slots since 01:44Z, shared with compute-research; tries only there, records runs stay on cpu-1/cpu-3; lent by compute-research until Tue 20:35Z; idle); then `--on vy-nebius-cpu-3` (3 trees); default cpu-1 (21 slots, but stalls on its memory ledger); cpu-5 overloaded; not cpu-4)
  (warm build slots; or `--on vy-nebius-cpu-3`; cpu-4's soundness pool is usable again (04:06Z)). Records: `research run --on vy-nebius-cpu-1 --pool research --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records PKG --update` (PKG is a package path, e.g. `Security/Proofs`, `Security/Properties`, `verity/core/protocols/onsite`, never a package name).
  Moving a run: only move one that hasn't taken a check slot yet. A run that holds a check slot and waits for a Lean build slot with 1-2 ahead is nearly started; cancelling it throws that away (02:38Z: two records runs lost ~1 h this way). Records stay off vy-nebius-1, whose build line is the try lane.
  Suites (since Tue 02:27Z, infra's suites lane, ~1 deep): `research run --on vy-nebius-cpu-3 --pool research --mem-gb 40 --source . --cwd source (`--pool` needs `--source .`; no `--cpus`: it needs `--queue`) -- bash -lc "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py --changed <main sha> --quick"`.
  Never cpu-5 for warm copies; never a bare `rm` of data you don't own on a node.
- **Never block more than 20 minutes in one tool call.** Launch runs detached; when you'd wait on one, end your turn
  naming the run id and its ETA, and the coordinator resumes you.
- **PR:** when the branch is done, `research queue open --head BRANCH --sha SHA --title T --body-file F --by soundness`
  with a line `Lane soundness` in the body, then (only after its Lean check and quick suites pass, and only if this brief
  or your inbox says you may) `research queue ready N --mark --by soundness`. Don't push to a PR whose head is in a train.
  Before marking, `git grep` across `next` any name you delete or change.
- **Name:** `Verity.Soundness` is called *protocol soundness* in prose (Daniel, 7:38 PM PDT), never "the one theorem"; AGENTS.md's rule "Soundness is one theorem" keeps its wording.
- **Rules that bind the statement:** protocol soundness is one theorem, with no hypothesis on the circuit file or the statement's form;
  the verifier refuses any form the proof doesn't cover and gains no form ahead of its proof; assumptions are named Props
  listed in the lock; no `sorry`, `axiom` or `native_decide`. Never weaken a statement to make it provable: if it's false,
  stop, say why in your note and final message, and propose the restatement.
- **Final message** (at most ~15 lines): branch and head SHA, PR, run ids with verdict lines, the statement as posted,
  and what's left.

- **Names (Daniel, 11:09 AM PDT):** 
The composed protocol (onsite) is made of **six subprotocols**: sampled proofs, PoUW, PoUS, the bandwidth audit, the
network certifier and the wipe. Use those names only, with no aliases ("integrity", "work", "space", "communication"),
and **"leg" is retired** (in Linear: project, milestone, issue). The theorem's subprotocol targets are named
`SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`, `NetworkCertifierSound` and `WipeSound`
(ONETHM, `cursor/proofs-one20-48ae`); `Verity.soundness_of_subprotocols` is the composition (#4613). Record keys,
wire strings (the verdict keys `integrity`, `communication`) and `Core.Protocol` keep their names.
- **One adversary form, and a guard against vacuity (Daniel, 2:00 PM PDT):** every bound is for every prover τ, in the
  reduction form `Pr[accept ∧ claim fails on E^τ's values] ≤ δ + f(Pr[B^τ outputs a SHA-512 collision])`, with E and B fixed
  constructions; `ecr/sha-512` is asked only of B. Every guarantee gets `<name>_honest` (hypotheses hold at the honest
  deployed instance) and `<name>_nontrivial` (the conclusion fails for an accept-everything verifier).

- **Smaller tree (Daniel, 9:53 PM PDT, via top 05:00Z):** every PR that adds code says in its body what it deletes or unifies, or why nothing can go. Prefer one definition over a second; delete superseded statements, scaffolding and copies in the same PR that replaces them. main must be below 223.4 MB / 13,768 files by Wed 23:00Z, then smaller daily.
- **Linear writes go through the app only** (top, 06:25Z): `research msg linear comment|state …` or `research msg ask`, never the Linear MCP connector (it writes as Daniel). Reading through the connector is fine.
- **No lane markers in landed text** (cleanup window, Tue 07:00Z; a test will refuse them): docstrings, comments and Markdown carry no Linear ids, `note:`/`art:` ids, PR numbers, Notion/Slack links, "Daniel", handles, agent ids or dates. Keep the reason, drop the id. Check with the `rg` in the 10:15Z inbox section before marking.

## Marking with main's research (27 Oct, 15:50Z)
Your VM's `research` may predate main's records gate (#5191, since 10:16Z: a pass at any commit whose input trees for a package equal the head's vouches for it, so carried locks need no run of their own). Before running a records job only to satisfy a mark, try the mark with main's research: export origin/main's tools/research into a scratch dir and run `uv run --project <dir> research queue ready N --mark --by soundness`. Records `--update` writes the regenerated lock under the run's $OUT/lean-audit/, not into the tree: commit it from there. In a multi-package job, later passes read the old lock and fail, so run one package per job.

## Never merge next (27 Oct, 17:15Z)
Never merge origin/next into a branch: drive then treats the PR as carrying everything next held. To clear a conflict with next, merge the specific PRs' heads you conflict with (only if you need their content) plus origin/main; a conflict that comes only from a next-only PR resolves when that PR reaches main.

## main moved (27 Oct, 18:25Z)
main is at 69dcbb8d6c71 (#4928, #5075 too): many PRs landed (#5300, #5309 with the gate's wider reuse, and many that were only on next). Merge origin/main into your branch, never next, before your next records run; many conflicts with next go away.

## Full tilt, and no moves (27 Oct, 20:10Z)
Daniel is restructuring the repository himself today: no directory moves or renames until his restructure lands. Edits inside files go ahead. Dates moved in: F2 (every subprotocol proved and marked) Wed 06:00Z; F3 (Verity.Soundness, no subprotocol premise, marked) Wed 13:00Z. Don't wait idle on a run: write the next piece while it runs.

## Use --cwd clone for build-and-test jobs (27 Oct, 20:50Z)
Runs of one commit with --cwd source share a source tree, and two lake builds there at once delete each other's .oleans (suites then stop with no Lean error). Until infra fixes it, launch any job that builds Lean or runs suites with --cwd clone, so it gets its own tree.

## Deslop is the priority (27 Oct, 23:00Z)
Daniel made the deslop refactor top priority. Every PR of ours must add no internal text to science paths: no "Daniel", no ruling dates, no Linear, PR, note, run or art ids, and no lane names, in code, docstrings, Lean comments, Markdown or lock reasons. Before each push, grep your diff against origin/main for them and state each rule as itself. Don't rename or move anything: top runs the renames Wednesday morning, after the PROTOCOL.md deletions land.

## New slots, and resend what waits (21:35Z)
infra opened slots at 21:30Z: vy-nebius-cpu-6 (192 cores, idle, Lean `build 6` warm trees, `soundness` pool s0-s5) and vy-nebius-cpu-2 (`soundness` s0-s5, `build 6`). A waiting run never moves by itself. Cancel each of your runs still waiting for a slot (`research cancel RUN --reason … --by soundness`), and resend the same command with `--on vy-nebius-cpu-6` or `--on vy-nebius-cpu-2`, `CHECK_SLOTS=/workspace/research/locks/soundness/slots` and `--cwd clone`. Don't send runs to vy-nebius-1, cpu-1, cpu-3 or cpu-5 (CPU-bound or backed up), and don't use `--on auto` (it doesn't know cpu-2 or cpu-6 yet).
Two demos (Daniel, 2:25 PM PDT): a PR titled "two demos" whose records passed at its own head lands on its land verdict, with no pre-landing Lean check. Run records for two-demos PRs first.

## Records runs go to cpu-6 (21:55Z)
Send records runs to vy-nebius-cpu-6 only. cpu-2 is low on inodes, and infra asks for no new records runs there until its cleanup lands. cpu-2's soundness pool is fine for short tries and suites. Leave runs already started on cpu-2 alone.

## NEW ORDER (Daniel, 2:56 PM PDT, 27 Oct): dedupe first, proofs after
First every smell is cleaned up and the remote demo runs on the right names and organization, with one canonical object per concept and no duplicates. Then the end-to-end security proofs, restarted on the canonical objects. Nobody starts new proof work. F2/F3 move: soundness restates them once C3 (Wed 9 AM PDT) holds.
- KEEP and land (marked or one step away): #5324, #5603, #5616, the flock-verify setup-cache fix, #4822, #5233, #5105, #5330, #4987 (its current records, then its mark), #5490, #5273 and the wipe chain.
- STOP (push what you have, write a checkpoint note saying where it stands, cancel your waiting runs by id, end your turn): the canon line (CANON-A/B/C, ONE9-E), ONE8's transfer branch, ONE5's `RoundsByDevice`, ONE4's refusal batches (b624/3c71: rounds, sizes, the `RecordDrawn` refusals), ONE19's Pub nontrivial, ONE78's openings, #5326/#5327 (F3) and #5320/#5321.
- Soundness owns the **registration cluster** (PRF-268, P-TOP-3, C3): one registration log that every subprotocol and both composites read. New workers take its duplicates.

## Worktree tries ship the wrong tree (22:25Z)
`verity lean try` (and any `research run` whose launcher code comes from `/workspace/.venv`) ships the checkout its own CLI code lives in, `/workspace`, not your worktree. From a worktree, run the worktree's own CLI (`uv run --project <worktree> verity lean try …` from inside it, or `python3 <worktree>/tools/verity/cli/…`), and check the run's `source_sha`/`tree_sha` (`research status RUN --json`) equals your head before trusting a pass.

## Registration scope (Daniel 3:24 PM PDT, 22:35Z)
Only the site, θ and B are registered (log entries: site, theta, builder). Subprotocol Programs, parameters and the proof system are verifier parameters, not registration. One-stage's `verity/registration/v1|v2` record is the window's commitment record (a homonym): bytes stay, code names move off "registration". Read the AMENDED comment on PRF-268.

## Three-hour rule (Daniel 4:38 PM PDT, 23:55Z)
A PR is marked within 3 hours of starting it; split longer work into pieces, each within 3 hours. A plan, answer or design note is due within 1 hour of the ask. Your PR's date is on its Linear issue (PRF-266, PRF-268, PRF-270, PRF-272); if it slips, comment there with the new date and why.

## Avoid cpu-2 entirely (00:05Z)
cpu-2 is out of disk: suites and tries there fail mid-run (d9db, e825 and fbab at 00:00Z). Send every run, records, tries and suites alike, to vy-nebius-cpu-6's soundness pool. Treat a cpu-2 failure as the node until its log shows otherwise, and resend it on cpu-6.

## cpu-6 is out of disk too (00:10Z)
cpu-6 now refuses runs at clone ("No space left on device"). Until infra clears it, send new runs to vy-nebius-cpu-4's soundness pool (`--on vy-nebius-cpu-4`, `CHECK_SLOTS=/workspace/research/locks/soundness/slots`). Use cpu-7, 8 or 9 as soon as infra announces them. Leave runs that already hold a slot on cpu-6 alone.
