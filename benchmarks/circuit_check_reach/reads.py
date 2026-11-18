import sys
from pathlib import Path
from circuit_check import cache as CC
ents = [p for p in Path(sys.argv[1]).glob("*/*/*.json")]
sets = []
for p in ents:
    e = CC.read(p)
    r = e["reads"]
    files = set(r.get("files", {})) | set(r.get("modules", {}))
    print(e["key"].get("target") if isinstance(e["key"], dict) else "?", {k: len(v) for k, v in r.items() if hasattr(v, "__len__")})
    sets.append((set(r.get("files", {})), set(r.get("modules", {}))))
fi = set.intersection(*[s[0] for s in sets]); mi = set.intersection(*[s[1] for s in sets])
print("common files", len(fi)); print("\n".join(sorted(fi)))
print("common modules", len(mi)); print(" ".join(sorted(mi)))
