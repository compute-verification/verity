#!/usr/bin/env python3
"""redo_child_mend.py PR PARTIAL OTHER BASE PHASE_A_MEND FILE...: phase A's mend of PR again, for the merge restack.py
stopped on: PARTIAL (the head after the last script step) with OTHER (what it is merged onto), on the merge base BASE
(the move's side, as restack.py pins it). The mend is made only if git's own merge of the two stops on exactly FILE...
and phase A's mended tree differs from that merge in nothing else. Prints the commit; the message's body is on stdin."""
import subprocess, sys

REPO = "/agent/repos/verity"


def git(*a, input=None, ok=(0,)):
    p = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, input=input)
    if p.returncode not in ok:
        raise SystemExit(f"git {' '.join(a)}: {p.stderr.strip()[:500]}")
    return p.stdout.strip()


pr, partial, other, base, mend_a, *files = sys.argv[1:]
partial, other, base = (git("rev-parse", x + "^{commit}") for x in (partial, other, base))
out = git("merge-tree", "--write-tree", "--name-only", "--no-messages", f"--merge-base={base}", partial, other, ok=(0, 1)).splitlines()
tree, stopped = out[0], sorted(x for x in out[1:] if x)
want = git("rev-parse", mend_a + "^{tree}")
differs = sorted(x for x in git("diff", "--name-only", tree, want).splitlines() if x)
print(f"  git's merge of {partial[:12]} with {other[:12]} on {base[:12]} stops on: {stopped}", file=sys.stderr)
print(f"  phase A's mended tree differs from that merge in: {differs}", file=sys.stderr)
if stopped != sorted(files) or not set(differs) <= set(files):
    raise SystemExit(f"#{pr}: not phase A's merge; no mend made")
title = f"Merge {other[:12]} (the layout move) into {partial}, restacked\n\n"
print(git("commit-tree", want, "-p", partial, "-p", other, "-F", "-", input=title + sys.stdin.read()))
