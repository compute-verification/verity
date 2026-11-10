# RESTACKALL: before the move lands, say for every open pull request whether it restacks onto the moved tree by itself.

**Why.** Tonight one pull request moves about 5,300 files into the layout Daniel signed off (Linear TOP-398; `lanes/top/20261110T2131Z-report-stage2.md` in the notes is the stager's record). The minute it lands, every open head (about 270) must `git merge BASE` and then run `tools/move/restack.py`. top wants the list by owner **before** the landing, so each lead knows which of its heads need hands.

**What to do.**
1. Read `tools/move/README.md`, `tools/move/restack.py` and lean's earlier count (Linear LEAN-260 and TOP-394: this morning lean measured, on the old map, 50 of 265 heads stopping with its drivers and 81 without). Check `python3 -m research notes approaches` and lean's notes: **if a tool that does this over all heads exists, run it; do not write a second one.**
2. The moved tree to measure against, until the stager posts the final one on TOP-398: staging A, `origin/cursor/top-layout-staged-e15c` at `50f940c4f28531bc33f23bbb73a2970d5820caf1`, built on `main` `2b869ed35db5` with the map `fea5e8c8a511` (the map's later heads change four `NEW` rows only). The map is on `origin/cursor/vllm-map2-9974`.
3. For every open pull request against `main` (`gh pr list --state open --base main --limit 500 --json number,headRefOid,headRefName,author,body`): in a scratch clone **on a node** (`research run --on auto --cwd clone`, batched; this is hours of git, never on your VM), merge `main`, restack onto the moved tree, and record: clean; clean after restack's drivers; stops (the files, and the kind: content, add/add, modify/delete, a file the map splits). Push nothing to anyone's branch. Keep each restacked result as a commit id in the run's output (not a branch on origin).
4. Owner of a pull request: the `Lane NAME` line of its body, else `unknown`.

**Deliver.**
- A table by owner: heads, clean, stops, and each stop's files. A second table: the 20 files most heads stop on (those are where one hand rule would free the most heads; say which rule).
- As an artifact in the evidence store (`research data put --kind restack-forecast --file F --preserve`), cited by its `art:` id in your note.
- **First table by 02:00Z Wed (6 PM PST) in your final message**; then end your turn. top resumes you with the final moved tree's commit to rerun on, and posts the announcement.

**Not yours:** changing `restack.py`, the map, or anyone's branch. If `restack.py` itself fails on the moved tree, that is the finding: say it at once in your note and final message, with the command and its output.
