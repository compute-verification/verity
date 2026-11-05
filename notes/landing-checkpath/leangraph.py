"""Import graph of the tree's Lean files at a commit, and per-landing cones (changed files and everything importing them)."""
import json, re, subprocess, sys, collections
from pathlib import Path

REPO = Path("/agent/repos/verity")


def git(*a):
    return subprocess.check_output(["git", "-C", str(REPO), *a], text=True)


def graph(commit):
    files = [f for f in git("ls-tree", "-r", "--name-only", commit).splitlines() if f.endswith(".lean") and "/.lake/" not in f]
    by_suffix = collections.defaultdict(list)
    for f in files:
        parts = f[:-5].split("/")
        for i in range(len(parts)):
            by_suffix[".".join(parts[i:])].append(f)
    imp = {}
    rx = re.compile(r"^\s*(?:public\s+|private\s+)?(?:meta\s+)?import\s+(?:all\s+)?([A-Za-z0-9_.']+)", re.M)
    blob = git("cat-file", "--batch-all-objects", "--batch-check") if False else None
    # read contents in one batch
    p = subprocess.run(["git", "-C", str(REPO), "cat-file", "--batch"], input="".join(f"{commit}:{f}\n" for f in files).encode(),
                       capture_output=True, check=True)
    out, pos = p.stdout, 0
    for f in files:
        hdr_end = out.index(b"\n", pos)
        size = int(out[pos:hdr_end].split()[2])
        body = out[hdr_end + 1: hdr_end + 1 + size].decode("utf-8", "replace")
        pos = hdr_end + 1 + size + 1
        deps = []
        for m in rx.findall(body[:20000]):
            cands = by_suffix.get(m, [])
            if not cands:
                continue
            pkg = f.split("/")[0]
            best = sorted(cands, key=lambda c: (c.split("/")[0] != pkg, -len(c)))[0] if len(cands) > 1 else cands[0]
            # prefer the candidate sharing the longest directory prefix
            if len(cands) > 1:
                best = max(cands, key=lambda c: len(_common(c, f)))
            deps.append(best)
        imp[f] = deps
    return imp


def _common(a, b):
    i = 0
    for x, y in zip(a.split("/"), b.split("/")):
        if x != y:
            break
        i += 1
    return a.split("/")[:i]


if __name__ == "__main__":
    commit = sys.argv[1]
    g = graph(commit)
    Path(sys.argv[2]).write_text(json.dumps(g))
    print(len(g), sum(len(v) for v in g.values()))
