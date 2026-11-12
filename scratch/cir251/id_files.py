"""For each of the 26 templates the switch renames (the long row's 23 units past 4 MiB and the DSA norm's 3, as the size
record names them today), every file under git on main that names its id exactly (as written, or JSON-escaped), and every
file that names its family (``Base_vN``) without the full id; each with its owner (`research msg owners`, read separately).
Run from the repository root; prints JSON."""
import json
import subprocess
import sys

rec = json.load(open("adapters/vllm/tools/description_sizes.json"))["templates"]
FAMILIES = ("DsaMaskPositions_v1", "DsaPositionsMerge_v1", "DsaTopKDigitPick_v1", "DsaTopKMaskBlock_v1", "DsaTopKMaskWhole_v1",
            "DsaRMSNormSumSq_v2", "DsaRMSNormApply_v2")
SIZE = 4 * 2**20
templates = sorted(t for t, e in rec.items() if t.split("{")[0] in FAMILIES and e["bytes"] > SIZE)
assert len(templates) == 26, len(templates)


def grep(s: str) -> set[str]:
    r = subprocess.run(["git", "grep", "-l", "-F", "-e", s, "--", ".", ":!archive"], capture_output=True, text=True)
    return set(r.stdout.split())


out = {"templates": {}, "families": {}}
for t in templates:
    forms = {t, json.dumps(t)[1:-1]}
    if '"f64"' in t:
        forms.add(t.replace('{"f64":"0x1.4f8b588e368f1p-17"}', "1e-05"))
    files = set().union(*(grep(f) for f in forms))
    out["templates"][t] = sorted(files)
for fam in FAMILIES:
    out["families"][fam] = sorted(grep(fam))
json.dump(out, sys.stdout, indent=1)
