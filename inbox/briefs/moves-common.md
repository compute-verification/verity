# The layout's moves: what every mover of top's reads first (TOP-398), Thu 12 Nov 3:08 PM PST

You are a worker of `top`, lane `top`, issue TOP-398. This file is the part of your brief you share with the other
movers. Your launch message names your pieces.

## What Daniel ruled today, and the date

Daniel settled the shape of the repository with top today (Thu 12 Nov), and at 3:01 PM PST asked for it to be
implemented soon, so that he can share the repository with people. Top's target: **Monday 16 Nov 9 AM PST, the root
and the top level of `verity/` read as the tree below.** Every piece that only changes paths runs on **Sunday 15 Nov
evening**, with lean's renames, as one staged run of certified moves (`tools/move/stage.py`, several maps in order).
Each map must be on `main` by **Sat 14 Nov 10 PM PST**. So your pull request, holding the maps and the hand edits a
move needs first, is open and unmarked by **Fri 13 Nov 10 AM PST** (earlier is better; top reads it whole, then marks
it). What changes inside a file comes after Sunday, on the new paths.

    (root)   verity/ hardware/ adapters/ demos/ security/ benchmarks/ tools/ tests/ infra/  + the root's files
    verity/
      party.py      what a party is
      channel.py    what two parties exchange messages over
      device.py     the registered hardware (was to be site.py; Daniel ruled the name device.py at 3:01 PM PST)
      protocol.py   what a protocol declares
      config.py     reads a configuration, the file `verity run CONFIG` takes (a .toml in demos/)
      errors.py
      runtime/      runs parties over channels (pipes, framing, handshake)
      protocols/    flock spot recursion pouw pous poub certifier inference_only compose; experimental/NAME/; interface/
      circuits/     the circuit language and the circuits a protocol names
      commitments/
    hardware/       data sheets only (TOML)
    adapters/       torch/, and vllm/ on it with the served kernels' circuits

Not in the tree: `verity/configurations/`, `verity/configuration/`, `verity/wire/`, `verity/hardware/` (code),
`verity/definitions/`, `verity/difftest/`, `verity/composite/`, `verity/claims/`, `verity/settings.py`, `fixtures/`.
Only the file's name changed to `device.py`: the Python class `Site` and the Lean's `Site` keep their names.

## Read first

1. `git fetch -q origin main && git show origin/main:AGENTS.md` (your snapshot can be older than `main`), and
   `tools/move/README.md` whole on `origin/main`: maps, `stage.py`, `layout.py`, `move_check.py`, `restack.py`.
2. The sizing note of this morning, which tried most of these moves and says what each needs first:
   `git fetch -q origin cursor/verity-top-4db4 && git show origin/cursor/verity-top-4db4:inbox/briefs/layout-rest-sized.md`.
   Its counts are from `main` this morning. Where it says `verity/site.py`, read `verity/device.py`. Where it says
   `verity/wire/` (a trial name), read: `wire/__init__.py` to `verity/channel.py`, `wire/handshake.py` to
   `verity/runtime/handshake.py`.

## Rules

- **Branch:** a new branch from `origin/main`, named in your launch message. Push only to it, never to the
  launcher's branch, a pull request's head or `main`, and never with force.
- **Commit and push before you end any turn.** When you are resumed you start on a new machine, and everything
  unpushed is gone: files, edits, local branches, local commits, `/tmp`.
- **Lean's large merge (#9375, branch `cursor/lean-remote-merge-onmain-fc8d`) lands tonight** and changes many files
  your moves touch. Check whether it is on `main` when you start (`gh pr view 9375 --json state,mergedAt`). If it is
  not, write the maps now (a map does not depend on the base), and run your trials on a merge of `origin/main` and
  that branch, in a scratch worktree. Open your pull request from `main` once the merge is on it. If it is still not on
  `main` by Fri 13 Nov 6 AM PST, open the pull request from `main` as it is and say so in the body.
- **Your VM does editing only.** `stage.py`, `layout.py` over the whole tree, `uv lock`, any suite, any Lean build
  run on a node: `research run --on auto --cwd source -- bash -lc 'CMD'` (or `--cwd clone` for a job that writes into
  the tree; read `.agents/skills/writing-runs/SKILL.md`). Lines on the nodes are long today (waits of an hour are
  common). Never wait on a run for more than 20 minutes: end your turn naming the run id and what you do when it ends,
  and top resumes you.
- **A move changes paths only.** `move_check.py` must certify each map's commit as a pure move, and a second run of
  the map must change nothing. What a move cannot rewrite (a table the mover leaves, a unit list in
  `tests/test_boundaries.py`, a row of `public_boundary.toml`, a link the mover refuses) goes into the hand-written
  commits of your pull request, which list the new place beside the old one so that the tree passes before and after
  the move. Add no exception to a rule to make a move pass. If a move needs one, stop on that piece and put it in
  your report.
- **The pull request holds maps and hand edits, not the moved tree.** The move itself is staged on Sunday from the
  maps on `main`. In the body, for each map: files moved, files rewritten, the certificate, which suites ran on the
  moved tree and their result (run ids), the files that conflict with the newest `origin/queue/*` branch, and the
  order the maps must run in.
- **Open it yourself, unmarked:** `research queue open --head BRANCH --sha SHA --title T --body-file F --by top`,
  with a line `Lane top` in the body and a section "For top": every judgment call, every place with no clear
  destination, every rule a move would break. If the command fails on your VM, report the branch and the SHA.
- **Checkpoints:** before the long part and each hour after, push, and write one line to the note your launch message
  names (`~/.research/notes/lanes/top/...`, with YAML front-matter: id, campaign, lane: top, kind: draft, status,
  repo, origin), then `research notes sync --path lanes/top`. At each checkpoint read your inbox file, named in your
  launch message, on `origin/cursor/verity-top-4db4` (a missing file means no message).
- Message nobody on Slack or Linear. What you need from another lead goes in your final message to top.

## Your final message, to top

The branch, the head, the pull request's number; for each map, whether it certifies and on which base; the hand
edits; the run ids and their results; the order for Sunday; what is left for after Sunday, by file; and every fact
in your brief or in the sizing note that the tree contradicts.
