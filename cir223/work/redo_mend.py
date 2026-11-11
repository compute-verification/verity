#!/usr/bin/env python3
"""redo_mend.py PR PARTIAL: phase A's mend of PR again on the partial head PARTIAL that this run's restack.py left. The
inputs of the merge must be those of phase A (the partial head's tree, and the other side's: the second move's base for a
stop at `base`, main for a stop at `moved`); then the mended tree is phase A's, committed on PARTIAL. Prints the commit."""
import subprocess, sys

REPO = "/agent/repos/verity"
MAIN = "5c095bb4ab7012ba359e0a85f4ecf19892b99629"
STAGED = "178f49c90c89734ea185f01496a7b6fbbd90b88c"
B2 = "2eb7b4aa735f38910780c25bd0ebb66a2c1af577"
M1 = "6581bcf197d3c87051ae45bad0a2fb778d1f6e2e"
M2 = "c1ca25a247486ec597a815f46f0adacfd9ef2063"
#: PR: (phase A's partial head, phase A's mend, the step it stops at, the body of the message)
MENDS = {
    "8430": ("81cdc0e333ca", "70c81dfe0c07", "base",
             "tools/verity/graph/tests/test_circuit_binding.py, mended by hand: this branch's side of both hunks (it deletes the\n"
             "read-set rows and rewrites the cases the base's commit 028691fdb0 edits), and the toy tree's member stays tools/config\n"
             "on the 12 lines where the first move's script wrote verity/configuration/config into this branch's new lines, as\n"
             "028691fdb0 did for the old ones.\n"),
    "8463": ("080b3519ad9c", "8cb95bb3b36c", "base",
             "tools/verity/circuit_check/pyproject.toml, mended by hand: both sides' additions to the end of [tool.verity.tests]\n"
             "inputs, the base's twelve rows first and then this branch's tests/lint/allowlists/superseded_versions.json with its\n"
             "comment. The paths list both sides also changed was merged by the repository's own driver, with both sides' rows.\n"),
    "8464": ("13df38a03e5e", "fa2b3f29b5d0", "base",
             "tools/verity/graph/tests/test_circuit_binding.py, mended by hand: both sides of the one hunk, this branch's three\n"
             "stacks/s rows of the toy tree and then the base's row, whose toy member is tools/config (028691fdb0). On the three\n"
             "lines of this branch's new cases where the first move's script wrote verity/configuration/config (DECLARED and two\n"
             "cases of the read-beyond test), tools/config again, as the base's commit has it for the toy tree.\n"),
    "8694": ("0823deaf3535", "cf77e456d1ca", "moved",
             "benchmarks/served_units/prove_time.sh and window.py, mended by hand: this branch's side (KEY_PATHS, the refusal of a\n"
             "missing path, the pipeline's status), with KEY_PATHS naming what main's find names: verity/circuits\n"
             "verity/definitions verity/commitments verity/protocols in place of verity/ml, which the moved tree does not hold.\n"
             "benchmarks/served_units/tests/test_names.py: this branch's side (the repo fixture reads FILES), FILES with main's\n"
             "stand-in tree (a file under each of the four directories), and the unmatched-quote file under verity/circuits.\n"),
    "8466": ("aecc19bdf275", "261ff9e20575", "moved",
             "benchmarks/tc_probe_fp4/tool.py, mended by hand: this branch's deletion of TC_LUT_FP4, TC_STRASSEN_FP4 and\n"
             "TC_RECHECK_FP4 kept; the four tools that stay read verity.definitions, as main's side has them.\n"),
}


def git(*a, input=None):
    p = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, input=input)
    if p.returncode:
        raise SystemExit(f"git {' '.join(a)}: {p.stderr.strip()[:500]}")
    return p.stdout.strip()


def tree(c):
    return git("rev-parse", c + "^{tree}")


pr, partial = sys.argv[1:]
partial = git("rev-parse", partial + "^{commit}")
pa, ma, step, body = MENDS[pr]
other_a = B2 if step == "base" else STAGED
other = B2 if step == "base" else MAIN
base = M1 if step == "base" else M2
checks = {
    "the partial head's tree is phase A's": tree(partial) == tree(pa),
    "the other side's tree is phase A's": tree(other) == tree(other_a),
    "phase A's mend has phase A's partial head and other side as parents": git("rev-parse", ma + "^1", ma + "^2").split() == [git("rev-parse", pa), git("rev-parse", other_a)],
    "the merge base is the move's side as in phase A": git("merge-base", partial, other) == git("merge-base", pa, other_a) if step == "base" else (subprocess.run(["git", "merge-base", "--is-ancestor", base, partial], cwd=REPO).returncode != 0),
}
for k, v in checks.items():
    print(f"  {k}: {v}", file=sys.stderr)
if not all(checks.values()):
    raise SystemExit(f"#{pr}: the merge's inputs are not phase A's; no mend made")
if step == "base":
    title = f"Merge {B2[:12]}, the layout move's base, into {partial}\n\n"
else:
    title = f"Merge {MAIN[:12]} (the layout move) into {partial}, restacked\n\n"
print(git("commit-tree", tree(ma), "-p", partial, "-p", other, "-F", "-", input=title + body))
