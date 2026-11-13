"""Compare two circuit-check runs' report lines, sidecar by sidecar: each target's verdict line (status, failures,
unchecked, q_call) and each summary line, with timings, run dirs and report paths dropped.

  python3 compare_cc.py OLD_OUT NEW_OUT
"""
import re
import sys
from pathlib import Path

TIMING = re.compile(r"\s*\([0-9.]+s[^)]*\)\s*$")
SECONDS = re.compile(r"\s*\([0-9.]+s\)")
RUN = re.compile(r"/workspace/research/runs/r[0-9-]+[0-9a-f]+/out/")


def lines(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = SECONDS.sub("", RUN.sub("OUT/", TIMING.sub("", line))).rstrip()
        m = re.match(r"^(ok|unchecked|failed|FAIL\w*)\s+definition\s+(\S+)\s+(.*)$", line)
        if m:
            out["target " + m.group(2)] = f"{m.group(1)} {m.group(3)}"
        elif line.startswith("circuit-check: ") and not any(t in line for t in (" wall ", "reused (", "checks start")):
            out["summary " + line.split(":")[1].split()[0]] = line
        elif re.match(r"^unchecked [a-z-]+/[a-z-]+: ", line):
            out["warn " + line.split(":")[0]] = line
    return out


def main(old: str, new: str) -> int:
    differ = 0
    for s in ("sampled_accept", "topp_stages", "boolean_topp_stages"):
        a, b = lines(Path(old) / f"cc-{s}.log"), lines(Path(new) / f"cc-{s}.log")
        keys = sorted(set(a) | set(b))
        d = [k for k in keys if a.get(k) != b.get(k)]
        differ += len(d)
        print(f"{s}: {len(keys)} lines compared, {len(d)} differ")
        for k in d:
            print(f"  {k}\n    old: {a.get(k)}\n    new: {b.get(k)}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
