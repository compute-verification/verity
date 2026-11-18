# scratch (lean/WIRED): the circuit record's registered section, as circuits' descriptor_record wrote it
import collections, gzip, json, sys
rec = json.loads(gzip.open(sys.argv[1], "rt").read())
reg = rec.get("registered", {})
print("record keys:", sorted(rec), "registered keys:", sorted(reg))
print("params:", len(reg.get("params", {})), "tables:", len(reg.get("tables", [])), "reads:", len(reg.get("reads", [])))
by = collections.Counter((r.get("writer"), r.get("reader"), r.get("port")) for r in reg.get("reads", []))
print("reads by (parameter, reader, port), first 12:", by.most_common(12))
print("first read:", json.dumps((reg.get("reads") or [None])[0])[:600])
print("params (first 12):", list(reg.get("params", {}).items())[:12])
