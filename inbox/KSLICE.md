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
