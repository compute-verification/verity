"""A probe, not for landing: what the record's `registered` check (`Record.checks`) and R1-R8 (`Registration.check`)
answer on variants of the honest round's own record, verdicts and parameters, by the verifier of record
(`V.audit_record`, `V.check`: the Lean's and Python's answers, compared).

`scan_zk.audit` calls `run` under SC_PROBE_REGISTERED once the honest round's audit record is made; it writes
``probe-registered.json`` beside the round's ``audit.json`` and changes nothing the round reads.

* **A, the record's check.** The shared trees (the scan's state, prog and view and the warden's rows, each a root of
  the round's registration) moved from `roots` to `registered`, under the value names the members' registered files
  read them by; then the same record with one member's session left out, with a value in both lists, with only one
  value listed, and today's record with a session's value named as a root's port.
* **B, R1-R8.** The moved record under today's parameters and under parameters that name `prog` registered; a record
  that registers the parameter `prog` at the tree's root and domain, alone, as an output of the block round's record
  (its o0 port), with another root, with `prog` also tied, and with no log; and the state and view trees registered
  as the parameters that read them.
"""
from __future__ import annotations

import json
from pathlib import Path

OWN = "committed/"
SHARED = ("scan/", "rows/")
OTHER_ROOT = "ab" * 64


def _files(members: list[dict]) -> dict[str, dict]:
    return {m["name"]: json.loads(Path(m["registered_file"]).read_text())["ports"] for m in members}


def _shared(files: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str], dict[str, list[str]]]:
    """The values the members' files read that are not a member's own rows, each as a record's entry, its row format,
    and the members that read it."""
    out, schema, readers = {}, {}, {}
    for name, ports in files.items():
        for e in ports.values():
            if e["value"].startswith(OWN):
                continue
            entry = {k: e[k] for k in ("value", "root", "leaves", "words", "domain")}
            if out.setdefault(e["value"], entry) != entry:
                raise SystemExit(f"probe: members read {e['value']} as two trees")
            schema[e["value"]] = e.get("row_schema")
            if name not in readers.setdefault(e["value"], []):
                readers[e["value"]].append(name)
    return out, schema, readers


def _audit(V, R, rec, receipt, draw, proved, n_inst, statement, verdicts, delta, sessions, served, roots) -> dict:
    r = V.audit_record(rec, receipt if receipt is not None else R.receipt(rec, "probe_registered.py"), draw,
                       rec["law"], proved, n_inst, statement, verdicts, delta, sessions=sessions, served=served,
                       session_roots=roots)
    c = r["checks"]
    return {"accepted": r["accepted"], "checks": {k: v["ok"] for k, v in c.items()},
            "registered": c.get("registered", {}).get("detail"), "verifiers": r.get("verifiers")}


def _check(V, rec, pg, law, scheme, ports, **kw) -> dict:
    try:
        v = V.check(rec, pg.program, pg.query, law, scheme, ports, **kw).to_json()
    except Exception as e:  # noqa: BLE001  a probe reports whatever the parameters' parse raises
        return {"raised": f"{type(e).__name__}: {e}"[:700]}
    return {k: v.get(k) for k in ("ok", "codes", "verifiers", "rests_on_period_check")}


def run(M, pg, members, rec, receipt, draw, proved, n_inst, statement, verdicts, delta, sessions, served,
        session_roots, want, law, table, leaf_layer, win, out: Path) -> dict:
    V, R, scheme = M.V, M.R, M.a0.SCHEME
    files = _files(members)
    vals, schema, readers = _shared(files)
    by_root = {e["root"]: v for v, e in vals.items()}
    shared = [r for r in rec["roots"] if r["port"].startswith(SHARED)]
    moved_from = [{"port": r["port"], "value": by_root.get(r["root"]),
                   "leaves_equal": r["leaves"] == vals.get(by_root.get(r["root"]), {}).get("leaves"),
                   "domain_equal": r["domain"] == vals.get(by_root.get(r["root"]), {}).get("domain")} for r in shared]
    own_roots = [r for r in rec["roots"] if not r["port"].startswith(SHARED)]
    moved = {**rec, "format": R.FORMAT_V2, "roots": own_roots, "registered": [vals[v] for v in sorted(vals)]}

    def audit(rec_, verdicts_=verdicts, proved_=proved, n_=n_inst, served_=served, roots_=session_roots, own=False):
        return _audit(V, R, rec_, receipt if own else None, draw, proved_, n_, statement, verdicts_, delta, sessions,
                      served_, roots_)

    def less(m):
        mine = [u for u in proved if m["base"] <= u < m["base"] + m["n"]]
        return dict(verdicts_={k: v for k, v in verdicts.items() if k.split("/", 1)[-1] != m["name"]},
                    proved_=[u for u in proved if u not in mine], n_=n_inst - len(mine),
                    served_=[u for u in served if u not in mine],
                    roots_={p: r for p, r in session_roots.items() if not p.startswith(f"{m['at']}/")})

    def renamed(old, new):
        return {k: {**v, "registered": {p: ({**e, "value": new} if e["value"] == old else e)
                                        for p, e in v["registered"].items()}}
                if isinstance(v.get("registered"), dict) else v for k, v in verdicts.items()}

    drawn = sorted({k.split("/", 1)[-1] for k, v in verdicts.items() if v.get("of_record")})
    a = {"today": audit(rec, own=True), "moved": audit(moved)}
    for m in members:
        if m["name"] in drawn:
            a[f"moved, {m['name']}'s session left out"] = audit(moved, **less(m))
    one = sorted(vals)[0]
    both = {"port": one, "root": vals[one]["root"], "leaves": vals[one]["leaves"], "domain": vals[one]["domain"]}
    a[f"moved, {one} also a root at the same root"] = audit({**moved, "roots": own_roots + [both]})
    a[f"moved, {one} also a root at another root"] = audit({**moved, "roots": own_roots + [{**both, "root": OTHER_ROOT}]})
    a[f"moved, only {one} listed"] = audit({**moved, "registered": [vals[one]]})
    port = next((r["port"] for r in shared if by_root.get(r["root"])), None)
    if port is not None:
        a[f"today, the sessions' {by_root[rec_root(rec, port)]} named {port}"] = audit(
            rec, verdicts_=renamed(by_root[rec_root(rec, port)], port), own=True)

    res = {"members_drawn": drawn, "members": [m["name"] for m in members],
           "values": {v: {"readers": readers[v], "leaves": e["leaves"], "words": e["words"]} for v, e in vals.items()},
           "moved_from": moved_from, "record": a}
    (out / "probe-registered.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    b: dict = {}
    less_shared = {p: w for p, w in want.items() if not p.startswith(SHARED)}
    b["today"] = _check(V, rec, pg, law, scheme, want, leaf_layer=leaf_layer, work_table=table, tied=pg.tied or None)
    b["moved, today's parameters"] = _check(V, moved, pg, law, scheme, less_shared, leaf_layer=leaf_layer,
                                            work_table=table, tied=pg.tied or None)
    try:
        names = [n for n, _t in pg.program.fn.signature()[0]]
    except Exception as e:  # noqa: BLE001
        names = [f"no signature: {e}"]
    tree = {k: next((vals[v] for v in vals if v.endswith(f"scan-{k}/v1")), None) for k in ("prog", "state", "view")}

    def param(name, kind, **kw):
        """A record that registers the parameter `name` at `kind`'s tree (its root, leaves, words and domain), the
        tree no longer a root, and R1-R8 on it under parameters that name `name` registered."""
        t = tree[kind]
        entry = {**t, "value": name, **kw.pop("entry", {})}
        ports = {p: w for p, w in want.items() if p != f"scan/{kind}"}
        try:
            r = R.record(pg.program, pg.query, law, scheme, [x for x in rec["roots"] if x["port"] != f"scan/{kind}"],
                         win, leaf_layer=leaf_layer, work_table=table, previous=kw.pop("previous", None),
                         registered=[entry])
        except Exception as e:  # noqa: BLE001
            return {"raised": f"record: {type(e).__name__}: {e}"[:700]}, None
        return _check(V, r, pg, law, scheme, ports, leaf_layer=leaf_layer, work_table=table,
                      registered={name: t["words"]}, schemas={name: schema[t["value"]]}, **kw), r

    if tree["prog"] is not None and "prog" in names:
        w = tree["prog"]["words"]
        b["moved, parameters name prog registered"] = _check(
            V, moved, pg, law, scheme, less_shared, leaf_layer=leaf_layer, work_table=table, registered={"prog": w},
            schemas={"prog": schema[tree["prog"]["value"]]})
        b["prog registered at the tree's root and domain"], rec_p = param("prog", "prog")
        b["the same, with a window root"], _ = param("prog", "prog", window_root=OTHER_ROOT)
        if rec_p is not None:
            a["prog registered, the sessions' reads as they are"] = audit(rec_p)
            a["prog registered, the sessions' reads named prog"] = audit(
                rec_p, verdicts_=renamed(tree["prog"]["value"], "prog"))
        block = out.parent / "block" / "registration.json"
        if block.exists():
            brec = json.loads(block.read_text())
            d = R.digest(brec)
            o0 = next((r for r in brec["roots"] if r["port"].endswith("/c0/o0")), None)
            b["the block's o0"] = None if o0 is None else {
                "port": o0["port"], "root_equal": o0["root"] == tree["prog"]["root"],
                "leaves_equal": o0["leaves"] == tree["prog"]["leaves"],
                "domain_equal": o0["domain"] == tree["prog"]["domain"]}
            if o0 is not None:
                given = {"record": d, "port": o0["port"]}
                b["prog an output of the block's record"], _ = param("prog", "prog", previous=d, log=[brec],
                                                                     outputs={"prog": given})
                b["prog the previous record's port"], _ = param("prog", "prog", previous=d, log=[brec],
                                                                outputs={"prog": {"port": o0["port"]}})
                b["prog an output, at another root"], _ = param("prog", "prog", previous=d, log=[brec],
                                                                outputs={"prog": given}, entry={"root": OTHER_ROOT})
                b["prog an output and tied"], _ = param("prog", "prog", previous=d, log=[brec], outputs={"prog": given},
                                                        tied=("prog",))
                b["prog an output, the verifier's log empty"], _ = param("prog", "prog", outputs={"prog": given})
    for kind, suffix in (("state", ".sin"), ("view", ".state")):
        name = next((n for n in names if n.endswith(suffix)), None)
        if name is not None and tree[kind] is not None:
            b[f"{kind}'s tree registered as the parameter {name}"], _ = param(name, kind, tied=pg.tied or None)
    res = {**res, "parameters": names, "record": a, "registration": b}
    (out / "probe-registered.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    return res


def rec_root(rec: dict, port: str) -> str:
    return next(r["root"] for r in rec["roots"] if r["port"] == port)
