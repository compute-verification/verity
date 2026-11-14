import io, os, re, subprocess, tokenize, sys
A, B = "/tmp/kit/tipA", "/tmp/kit/tipB"
moves = dict(l.rstrip("\n").split("\t")[::-1] for l in open("/tmp/kit/m2_moves.tsv"))
def is_test(p): return "/tests/" in p or os.path.basename(p).startswith("test_") or p.startswith("tests/")
def corpus(root):
    fs = subprocess.run(["git", "ls-files", "*.py", "*.lean"], cwd=root, capture_output=True, text=True).stdout.split()
    if root == B:
        fs = sorted(set(fs) | set(subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True).stdout.split()))
    out = []
    for f in fs:
        if is_test(f) or f.startswith("tools/move/") or not f.endswith((".py", ".lean")): continue
        try: t = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        except FileNotFoundError: continue
        if len(t) < 2_000_000 and re.search(r"(?i)draw|drew|sampl", t): out.append(t)
    return "\n".join(out)
CA, CB = corpus(A), corpus(B)
W = re.compile(r"(?i)\b(?:draw|drawn|draws|drew|sample|sampled|samples|sampling)\b")
changed = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=AMR", "--", "*.py"], cwd=B, capture_output=True, text=True).stdout.split()
def lits(src):
    out = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.STRING or t.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
            out.append((t.start[0], t.string))
    return out
def frag(s):
    s = re.sub(r"^[rbfu]*(\"\"\"|'''|\"|')", "", s, flags=re.I); s = re.sub(r"(\"\"\"|'''|\"|')$", "", s)
    s = s.replace("\\\\", "\\")
    parts = re.split(r"\{[^}]*\}|\\[A-Za-z]|[\\^$*+?()\[\]|]", s)
    parts = [p for p in parts if W.search(p)]
    return max(parts, key=len) if parts else ""
for f in changed:
    if not is_test(f): continue
    old = lits(subprocess.run(["git", "show", f"HEAD:{moves.get(f, f)}"], cwd=B, capture_output=True, text=True).stdout)
    new = lits(open(os.path.join(B, f), encoding="utf-8").read())
    if len(old) != len(new): print("MISMATCH tokens", f); continue
    for (l, so), (_, sn) in zip(old, new):
        if not W.search(sn): continue
        fo, fn = frag(so), frag(sn)
        if len(fn) < 14: continue
        inA_old, inB_new, inB_old = fo in CA, fn in CB, fo in CB
        # was the old text real code text on A, and does the new one stand on B?
        if inA_old and not inB_new:
            print(f"{f}:{l}\t{'renamed' if so != sn else 'kept'}\ttest={fn[:90]!r}\tcode-still-old={inB_old}")
