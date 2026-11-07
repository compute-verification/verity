"""List each file under OUT whose JSON (or JSON lines) holds a key that can carry a per-request count, a prompt length or
an advice length, printing only the file, the key path and the value's type, never a value."""
import json
import sys
from pathlib import Path

WORDS = ("request", "prompt", "prefill", "users", "advice_bits", "population", "alpha", "gamma", "tokens", "slots",
         "lengths", "scan", "answers")


def walk(x, path, hits):
    if isinstance(x, dict):
        for k, v in x.items():
            p = f"{path}.{k}" if path else str(k)
            if any(w in str(k).lower() for w in WORDS):
                hits.add((p if not p[0].isdigit() else "*", type(v).__name__))
            walk(v, p, hits)
    elif isinstance(x, list):
        for v in x[:50]:
            walk(v, f"{path}[]", hits)


out = Path(sys.argv[1])
for f in sorted(p for p in out.rglob("*") if p.is_file()):
    rel = f.relative_to(out)
    if any(w in str(rel).lower() for w in ("scan/", "answers", "requests", "prompts", "endpoint/")):
        print(f"PATH {rel}")
    if f.suffix not in (".json", ".jsonl") or f.stat().st_size > 64 << 20:
        continue
    hits: set = set()
    try:
        docs = [json.loads(f.read_text())] if f.suffix == ".json" else [json.loads(l) for l in f.open() if l.strip()]
    except Exception as e:
        print(f"UNREAD {rel}: {type(e).__name__}")
        continue
    for d in docs[:2000]:
        walk(d, "", hits)
    if hits:
        print(f"KEYS {rel}: " + ", ".join(f"{p}:{t}" for p, t in sorted(hits))[:1500])
