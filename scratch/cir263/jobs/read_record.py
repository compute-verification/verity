"""Print what a CIR-263 capture's sealed record says: python3 read_record.py RUN_DIR NAME (RUN_DIR as `research fetch` leaves it,
NAME one of eh_proj, eh_norm, indexer_logits).  It recomputes nothing but the digest's presence; every figure is the record's."""
import json
import sys
from pathlib import Path

run, name = Path(sys.argv[1]), sys.argv[2]
path = run / "out" / name / f"glm53_mtp_{name}_capture.json"
if not path.exists():
    sys.exit(f"no record at {path}")
d = json.loads(path.read_text())
print(f"record {path.name}: schema {d['schema']} ok={d['ok']} digest={d['digest']}")
print("environment", json.dumps(d.get("environment")))
print("gate", json.dumps({k: d["gate"].get(k) for k in ("ok", "name", "driver", "compute_cap", "sms")}))
if d.get("errors"):
    print("errors", json.dumps(d["errors"])[:2000])
if name == "eh_norm":
    print("served", json.dumps(d["served"]))
    print("triton", d.get("triton"), "launches", json.dumps(d["launches"]))
    p = d["ptx"]
    print("ptx ok", p["ok"], "specialisations", len(p["specialisations"]),
          [(s["code_sha256"], s["reqntid"], s["sass"]["counts"]) for s in p["specialisations"]], "want", p["want"]["code_sha256"])
    print("bind", json.dumps(d["bind"]))
    for key, t in d["summary"]["cases"].items():
        print("case", key, json.dumps(t))
    print("refuted", json.dumps(d["summary"]["refuted"]))
    tot = {k: sum(t[k] for t in d["summary"]["cases"].values()) for k in ("rows", "words", "mismatch", "reference", "twin_mismatch")}
    print("totals", json.dumps(tot))
elif name == "eh_proj":
    print("switch", json.dumps(d.get("switch"))[:600])
    print("selection", json.dumps(d["selection"]), "scan", json.dumps(d["scan"])[:900])
    print("routes", json.dumps(d["routes"]))
    print("expected", d["expected"], "launched", [x["M"] for x in d["launches"]])
    print("bind", json.dumps(d["bind"]))
    for key, c in d["cases"].items():
        print("case", key, "ok", c["ok"], "split", c["split"], "launch_ok", c["launch_ok"], "route_ok", c["route_ok"],
              json.dumps(c["totals"]), "hypothesis", json.dumps(c["hypothesis"]), "first", json.dumps(c["first_mismatch"]),
              "kernels", [k[:48] for k in c["kernels"]][:3])
    print("totals", json.dumps(d["totals"]))
    split = [c for c in d["cases"].values() if c["split"]]
    print("split launches", len(split), "rows", sum(c["totals"]["rows"] for c in split), "words", sum(c["totals"]["words"] for c in split),
          "mismatch", sum(c["totals"]["mismatch"] for c in split))
else:
    print("served", json.dumps(d["served"]))
    print("selection", json.dumps(d["selection"]))
    for x in d["launches"]:
        print("launch", json.dumps({k: v for k, v in x.items() if k != "lengths"}), "lengths", str(x.get("lengths"))[:120])
    print("bind", json.dumps(d["bind"]))
    for key, t in d["cases"].items():
        print("case", key, json.dumps(t))
    tot = {k: sum(t[k] for t in d["cases"].values()) for k in ("words", "mismatch", "unresolved", "reference", "twin_mismatch")}
    print("totals", json.dumps(tot))
