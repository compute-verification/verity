"""same_entries.py BEFORE AFTER: whether two copies of the list hold the same entries, and in the same order."""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
key = lambda d: sorted((e["file"], e["detail"]) for e in d["entries"])
print("after --prune: the same entries" if key(a) == key(b) else "after --prune: the entries differ",
      "| the same order" if a["entries"] == b["entries"] else "| in another order", "| held", b["held"])
