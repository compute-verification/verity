import json
import sys

sys.path.insert(0, "adapters/vllm/tools")
import description_sizes as DS  # noqa: E402

main = json.load(open(sys.argv[1]))["templates"]
new = json.load(open(DS.RECORD))["templates"]
restored = 0
for t, e in new.items():
    m = main.get(t)
    if m is not None and m["bytes"] == e["bytes"] and m.get("largest") == e.get("largest"):
        restored += m != e
        new[t] = m
DS.write(new)
rewritten = sorted(t for t in new if t in main and new[t] != main[t])
print(json.dumps({"restored": restored, "entries": len(new), "in": sorted(set(new) - set(main)),
                  "out": sorted(set(main) - set(new)), "rewritten": rewritten,
                  "rewritten_same_bytes_largest": [t for t in rewritten if new[t]["bytes"] == main[t]["bytes"]
                                                   and new[t].get("largest") == main[t].get("largest")]}, indent=1))
