"""laycheck.py MAP...: files on the tree that still name a layout map's old paths (or their module names)."""
import subprocess
import sys
import tomllib

for mp in sys.argv[1:]:
    m = tomllib.load(open(mp, "rb"))
    olds = list(m.get("paths", {}))
    pats = []
    for o in olds:
        if o.endswith("/"):
            o = o[:-1]
        pats.append(o)
        if o.endswith(".py"):
            mod = o[:-3].replace("/", ".")
            for root in ("adapters.vllm.", "adapters.torch.", "tools.config.", "tools.difftest.python."):
                if mod.startswith(root):
                    mod = mod[len(root):]
            pats.append(mod)
    if not pats:
        print(mp, "no paths")
        continue
    args = ["git", "grep", "-c", "-F"] + sum([["-e", p] for p in pats], []) + [
        "--", ":!tools/move", ":!**/moves.json", ":!**/CHANGELOG.md", ":!**/lean-audit.json"]
    out = subprocess.run(args, capture_output=True, text=True).stdout.strip()
    print(f"{mp}: {len(olds)} paths; files still naming one: {len(out.splitlines()) if out else 0}")
    for line in out.splitlines()[:15]:
        print("  ", line)
