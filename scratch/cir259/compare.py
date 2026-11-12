"""compare.py T OUT.json: T/B (bound), T/U (unbound) and, where staged, T/R (rebound in place) of one type at one N: which
lines of circuit.txt differ, META's differing keys, stage.json's figures, and inst.bin's and pub.bin's header and body."""
import hashlib
import json
import sys
from pathlib import Path

T, dst = Path(sys.argv[1]), Path(sys.argv[2])
KEYS = ("n", "definition", "ands", "k_log", "unit_log", "unit_nnz", "vus_per_block", "class_sha512", "circuit_sha512", "circuit_bytes",
        "inst_bytes", "pub_bytes", "statement", "lookups", "nets", "slot_types", "row_schemas", "net_and_limits", "maxrss_gb", "seconds")


def record(d: Path) -> dict:
    s = json.loads((d / "stage.json").read_text())
    return {k: s.get(k) for k in KEYS} | {"bind": s.get("bind")}


def lines(a: Path, b: Path) -> dict:
    """Line by line where the heads differ in length, else by blocks: the differing lines' numbers (the first 20), both
    files' line 1 and META."""
    out: dict = {"bytes": [a.stat().st_size, b.stat().st_size]}
    with a.open("rb") as fa, b.open("rb") as fb:
        heads = [(fa.readline(), fa.readline()), (fb.readline(), fb.readline())]
        out["line1_same"] = heads[0][0] == heads[1][0]
        out["line2_same"] = heads[0][1] == heads[1][1]
        metas = [json.loads(h[1][5:]) if h[1].startswith(b"META ") else None for h in heads]
        if all(m is not None for m in metas):
            ks = sorted(set(metas[0]) | set(metas[1]))
            out["meta_keys"] = len(ks)
            out["meta_differs"] = {k: [json.dumps(m.get(k), sort_keys=True)[:420] if k in m else None for m in metas]
                                   for k in ks if metas[0].get(k) != metas[1].get(k)}
        same, n = True, 0
        while True:
            ca, cb = fa.read(1 << 24), fb.read(1 << 24)
            if ca != cb:
                same = False
                break
            if not ca:
                break
            n += ca.count(b"\n")
        out["rest_same"] = same
        out["rest_lines"] = n if same else None
    if not same:
        diff, total = [], 0
        with a.open("rb") as fa, b.open("rb") as fb:
            for total, (la, lb) in enumerate(zip(fa, fb), 1):
                if la != lb and len(diff) < 20:
                    diff.append(total)
        out["differing_lines_first20"] = diff
        out["lines_compared"] = total
    else:
        out["differing_lines"] = [k for k, s in ((1, out["line1_same"]), (2, out["line2_same"])) if not s]
    return out


def binfile(a: Path, b: Path) -> dict:
    """A header line (JSON) then a body: the header's differing keys and whether the bodies are the same bytes."""
    out: dict = {"bytes": [a.stat().st_size, b.stat().st_size]}
    hs, bodies = [], []
    for f in (a, b):
        with f.open("rb") as fh:
            head = fh.readline()
            h = hashlib.sha512()
            while chunk := fh.read(1 << 24):
                h.update(chunk)
        try:
            hs.append(json.loads(head))
        except ValueError:
            hs.append({"_raw": hashlib.sha512(head).hexdigest()})
        bodies.append(h.hexdigest())
    ks = sorted(set(hs[0]) | set(hs[1]))
    out["header_keys"] = ks
    out["header_differs"] = {k: [json.dumps(h.get(k), sort_keys=True)[:300] for h in hs] for k in ks if hs[0].get(k) != hs[1].get(k)}
    out["body_same"] = bodies[0] == bodies[1]
    out["body_sha512"] = bodies[0][:32]
    return out


res: dict = {"bound": record(T / "B"), "unbound": record(T / "U")}
b, u = res["bound"], res["unbound"]
res["same_figures"] = {k: b[k] == u[k] for k in KEYS if k not in ("maxrss_gb", "seconds")}
res["circuit_bound_vs_unbound"] = lines(T / "B/circuit.txt", T / "U/circuit.txt")
res["inst_bound_vs_unbound"] = binfile(T / "B/inst.bin", T / "U/inst.bin")
res["pub_bound_vs_unbound"] = binfile(T / "B/pub.bin", T / "U/pub.bin")
for f in ("tables", "unit.tmpl", "unit.layout"):
    pa, pb = T / "B" / f, T / "U" / f
    if pa.exists() or pb.exists():
        def tree(p: Path) -> dict:
            fs = sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p] if p.exists() else []
            return {str(q.relative_to(p.parent)): hashlib.sha512(q.read_bytes()).hexdigest()[:32] for q in fs}
        res[f"{f}_same"] = tree(pa) == tree(pb)
        res[f"{f}_files"] = len(tree(pa))
if (T / "R/stage.json").exists():
    r = res["rebound"] = record(T / "R")
    res["rebound_same_n"] = r["n"] == b["n"]
    if r["n"] == b["n"]:
        res["rebound_pin_is_bound_pin"] = r["circuit_sha512"] == b["circuit_sha512"]
        res["rebound_circuit_is_bound_circuit"] = lines(T / "R/circuit.txt", T / "B/circuit.txt")
        res["rebound_files_same"] = {f: (T / "R" / f).read_bytes() == (T / "B" / f).read_bytes()
                                     for f in ("partition.json", "program.json", "pin.txt", "pub.bin", "inst.bin")}
    res["rebound_pin_is_file_sha512"] = r["circuit_sha512"] == hashlib.file_digest((T / "R/circuit.txt").open("rb"), "sha512").hexdigest()
res["bound_pin_is_file_sha512"] = b["circuit_sha512"] == hashlib.file_digest((T / "B/circuit.txt").open("rb"), "sha512").hexdigest()
dst.write_text(json.dumps(res, indent=1, sort_keys=True))
c = res["circuit_bound_vs_unbound"]
print("COMPARE", json.dumps({
    "n": b["n"], "ands": b["ands"], "k_log": b["k_log"], "circuit_bytes": c["bytes"], "same_figures": res["same_figures"],
    "circuit_differing_lines": c.get("differing_lines", c.get("differing_lines_first20")), "rest_lines": c.get("rest_lines"),
    "meta_differs": sorted(c.get("meta_differs", {})), "inst": [res["inst_bound_vs_unbound"]["body_same"], sorted(res["inst_bound_vs_unbound"]["header_differs"])],
    "pub": [res["pub_bound_vs_unbound"]["body_same"], sorted(res["pub_bound_vs_unbound"]["header_differs"])],
    "rebound": {k: res[k] for k in ("rebound_same_n", "rebound_pin_is_bound_pin", "rebound_files_same", "rebound_pin_is_file_sha512") if k in res},
    "bound_pin_is_file_sha512": res["bound_pin_is_file_sha512"]}))
