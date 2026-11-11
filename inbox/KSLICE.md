# Inbox: KSLICE (HASH-104)

## Tue 10 Nov 15:31Z: which machines your read-only runs may touch before 16:45Z

top's staged check runs its shards on `vy-nebius-cpu-6` and `vy-nebius-cpu-11` until about 16:40Z, and its audit on
`vy-nebius-cpu-5` until 16:44Z; its reds go to owners at 16:40Z and their fixes are due before the cut, so those
cores come first (top and infra on TOP-394, 15:29Z and 15:30Z). Your greps of run directories are light, but:

- **Before 16:45Z, launch nothing on cpu-5, cpu-6 or cpu-11.** Search the other machines first (cpu-1, cpu-2, cpu-3,
  cpu-8, cpu-12, and `vy-spot-glm53`), and those three after 16:45Z.
- One run a machine, a single `grep` process with `nice -n 19`, no `find` over anything but the runs directory.

## Tue 10 Nov 17:50Z: #8666 and #8669 are marked; the layout does not land today, so the rest waits

- The lead marked #8666 (9eb81cffe9) and #8669 (b0ad346b9f) at 17:47Z after reading their suites' lines (…adf4:
  `repository` 282, `verity-graph` 206; …e0b7: `repository` 282). #8666's body now names the live goal
  (`kernels/code-at-overhaul-bar`). Push nothing to either.
- top, TOP-394 17:39Z: Daniel wants the target tree changed first, so the layout is not on `main` today and the map
  changes once more. Piece 3 and the archive pull request (34 paths) both need the layout's `archive/`, so they
  wait for the new map's date. No restack. Do not delete or rewrite the hand-row files or the two lists in the notes.
- Nothing is asked of you now. top posts an architecture document for each directory's owner to correct; if the
  lead resumes you, it is to read that draft against your slice list.

## Tue 10 Nov 18:06Z: one piece of HASH-104 for today, on `main` as it is: #7762's six comment edits, as a new pull request

The launch freeze is lifted and nothing waits for the layout, so this piece goes now.

- **What:** #7762 (HASH-72's remainder; head 3956003c1c; 16 lines added, 17 removed, comments and docstrings only)
  was never marked and now conflicts on `main`; drive closes it on Sat 14 Nov as stale. Its body lists each edit:
  a comment keeps its decision and the reason, without measured rates or spans (AGENTS.md, "Where writing goes").
- **Do:** a new branch from `origin/main` (`cursor/kernels-comments-<hex>`). Make the same edits wherever the site
  still exists, in `benchmarks/hash/dither/dither_a.py`, `dither_so.py`, `sbench.py`,
  `benchmarks/torch_gpu/attn_bf16/attn.py` and `s8_linear/bench.py`. **Leave `torch_gpu/units/bench.py` out:** it is
  in the archive set. No code line changes: `git diff` must show comments and docstrings only.
- **Do not widen it.** If one of those five files states another measured number the same way, list it on
  HASH-104; do not edit it in this pull request.
- **Check:** ruff on the touched files; the suites the change touches at the head in one named run
  (`suites.py --changed` from the merge base, or the suites by name), and read its verdict lines.
- **Open:** `research queue open --issue HASH-104`; the body says it replaces #7762 and ends `Refs HASH-104` and
  `Goal kernels/code-at-overhaul-bar`. Mark nothing: the lead marks, and asks top to close #7762.
- **Report on HASH-104 by 19:30Z:** the pull request, its head, the run and its verdict lines, and for each of
  #7762's six sites one of: applied, already gone, changed (with the line), left out.

## Tue 10 Nov 18:14Z: the 18:06Z piece is closed: #7762 was already on `main`

Your finding stands (the lead checked `60168d4fb9` on `main`). No pull request is wanted, and the lead asked top to
close #7762 as landed. Nothing is asked of you now; the sections of 17:50Z still say what waits for the layout.

## Wed 11 Nov 00:25Z: HASH-104's dead files are DELETED, not archived. Prepare that pull request tonight, on the staged tree

Daniel, 00:10Z (top's announcement): `archive/` is deleted, by a pull request of top's worker on the moved tree
after the move lands (about 03:45Z). The lead told top (TOP-398, 00:22Z): kernels' 34 paths are deleted in one pull
request of kernels', which also rewrites the prose that names `archive/` in the files that stay; top's worker
leaves kernels' directories alone. Read `inbox/RULES.md` ("Agents, today") first.

**Base:** `cursor/top-layout-staged-e15c` (`50f940c4f285`; if top posts a later staged head on TOP-398, that one).
A new branch from it, `cursor/kernels-dead-files-<hex>`. No pull request tonight: it opens on the lead's word, once
the move is on `main` and the branch sits on that commit.

**Delete:** the 34 paths of `kernels-archive.txt` that the staged tree still has outside `archive/` (the lead
checked: 34 present; the other three, `commit_slots/{overlap.py,overlap.sh,green_probe.py}`, are in `archive/` and
go with top's). Before each directory or file goes, on the staged tree:
- nothing that stays loads it: imports, paths in scripts and data files, a suite's inputs, workspace members in the
  root `pyproject.toml` and `uv.lock`, the SASS gate's pins (`benchmarks/pouw/harness/sass_pins.json:151` names
  `units.cu`'s build, and `benchmarks/pouw/tests/test_sass_gate.py:510` a fixture built from it: those are
  compute-accounting's files, so keep them working and change only what names a path that will be gone, or list it
  for their owner on HASH-104);
- the store shows no run of it since its last change, other than KMOVE's rehearsals of the move (HASH-102,
  HASH-105, HASH-108). One that has such a run is kept, with the run, and listed.
- Then every line that names a deleted path, in the same change: READMEs (`benchmarks/README.md`,
  `benchmarks/torch_gpu/README.md`, `s8_linear/README.md`), lint allowlists, `tests/test_boundaries.py`'s lists,
  the registry, the kernel-engineering skill if it names one.

**Rewrite, in files that stay:** the lines that name `archive/` (the lead counted 8 in 5 files once `units/` and
`realw.py` are gone: `torch_gpu/README.md:37`, `attn_bf16/README.md:58`, `commit_slots/README.md:58,70` and its
line 72 on `green_probe.py`, `p2_stage/README.md:20`, `p2_stage/served_weights.py:5,6,606`). Each says what is true
with no demo in the tree: the shape or the caller that exists, never a path that is gone.
- **Leave `p2_stage/README.md` alone:** #8662 (KP2's) edits it. List its line on HASH-104 for KP2's resume.
- **Keep `torch_gpu/README.md`'s edit to the lines it needs:** #8669 (marked) adds its Interface section to the same
  file, and whichever lands second takes the conflict. Its Interface table has no row that names a deleted path?
  Check, and say on HASH-104 if it does.
- `e2e_h200.<name>` in a comment or docstring of a file that stays (`attn_bf16/attn.py`, `commit_slots/slots_gpu.py`
  and its `tests/test_gpu.py`) names a module that will not exist: rewrite those too, comments only.

**Check:** ruff on what you touch; `suites.py --changed` from the staged commit, or the suites by name (at least
the root suite, `benchmarks/pouw`, and each `torch_gpu` and `hash` suite left), in one named run on
`--on vy-nebius-cpu-12` (`inbox/RULES.md`, "Runs"). Read its verdict lines yourself.

**Report on HASH-104 by 03:00Z, then end your turn:** the branch and its head; each of the 34 as deleted or kept
with its reason; the lines rewritten; what you left for an owner; the run and its verdict lines. A note in
`lanes/kernels/` that a fresh worker could open the pull request from (the base it sits on, and the one command to
put it on the moved `main`: `git rebase --onto origin/main <staged commit>`). Mark nothing, open nothing.

## Wed 11 Nov 00:30Z: three facts for the 00:25Z piece (read when you next read this file; none changes the work)

- **The tree that lands is not `50f940c4f285`.** top's rehearsal is `cursor/top-layout-staged-bf25` (`b2b4d3ea55cd`),
  "the final staging" is posted on TOP-398 at about 00:25Z and the fixed commit at 01:05Z. The lead compared kernels'
  three directories on `50f940c4f285` and `b2b4d3ea55cd`: no byte differs, the 34 are all there, and the same 8
  lines name `archive/`. So a branch from either carries over; say in your report which commit yours sits on.
- **top's deleting branch exists:** `cursor/top-archive-out-e5ee` (`66314c5829`). Its last commit puts
  `benchmarks/torch_gpu` back to the base's text, "left to kernels' own pull request", so those lines are yours. It
  edits the root's lists (the lint allowlists, `tests/test_boundaries.py`, the registry, `benchmarks/README.md`,
  `AGENTS.md`) for the archive's entries. Your change edits other rows of the same files, and goes on `main` after
  it: before the lead's word to open, the branch is put on top's landed commit. Do not copy its edits into yours.
- **The gate:** the pull request opens on top's word that Daniel has seen the list of unused code (the Goals page
  says so). Tonight is preparation only.
