"""resolve_check.py's own `check` (main's copy), fed the in-memory trees of sim/analyze instead of two commits."""
import pickle, sys
sys.path.insert(0, "move_main")
import resolve_check as R

def run(pkl, pairs=((3, 4), (4, 5), (5, 6), (0, 1), (1, 2), (2, 3))):
    snaps, origins = pickle.load(open(pkl, "rb"))
    SNAP = {f"s{i}": s for i, s in enumerate(snaps)}
    R.texts = lambda root, commit, paths: {p: SNAP[commit][p] for p in paths if p in SNAP[commit]}
    R.lean_files = lambda root, commit: [p for p in SNAP[commit] if p.endswith(".lean")]
    R.git = lambda root, *args: "\0".join(SNAP[args[-1]].keys())
    out = {}
    for a, b in pairs:
        inv_a = {o: p for p, o in origins[a].items()}
        ren, changed = {}, set()
        for p, o in origins[b].items():
            q = inv_a[o]
            if q != p:
                ren[q] = p
            if q != p or snaps[a].get(q) != snaps[b].get(p):
                changed.add(p)
        R.changes = lambda root, base, head, ren=ren, changed=changed: (ren, changed)
        issues, compared, files = R.check(None, f"s{a}", f"s{b}")
        out[(a, b)] = issues
        print(f"stage {b}: {compared} of {files} files compared; {len(issues)} issue(s)")
        for i in issues:
            print("   ", R.said(i))
    return out

if __name__ == "__main__":
    run(sys.argv[1])
