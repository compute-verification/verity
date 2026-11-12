"""cover_runs.py RUNS_DIR WORK LINEAR_DIR OUT.json: the runs CIR-271's block table read, and the runs named that it did not.

RUNS_DIR is `research fetch`'s directory; WORK holds the files fetched from run records (`r3496/`, `df04/`); LINEAR_DIR the
issues' comments as the Linear connector returned them (``ISSUE.json``, ``ISSUE-2.json`` for a second page). Each read run's
facts come from its own files; a named run is any run id in a comment that is not read.
"""
import json
import re
import sys
from pathlib import Path

RUNS, WORK, LIN, OUT = (Path(a) for a in sys.argv[1:5])
J = lambda p: json.loads(Path(p).read_text())  # noqa: E731
HERE = Path(__file__).resolve().parent
RID = re.compile(r"r2026111\d-\d{6}-[0-9a-f]{4}")

read = []
for k in J(HERE / "cover_ksteady.json"):
    read.append({"run": k["run"], "issue": "HASH-98", "lean": "accepted" if k["accepted"] else "refused",
                 "what": f"{k['word']} as {k['definition']}, {k['blocks']} units, Lean 3 of 3 {k['verdicts']}, {k['label']}",
                 "at_list_binding": True})
    read.append({"run": k["archive"], "issue": "HASH-98", "lean": None,
                 "what": f"the proof archive {k['run']} verified: staged rc {k['stage']['rc']} at {k['stage']['peak_gib']} GiB, "
                         f"proved on {k['prove']['device']} at {k['prove']['peak_gib']} GiB", "at_list_binding": True})
for name, run in (("2196", "r20261112-195705-2196"), ("df04", "r20261112-191810-df04")):
    p = RUNS / run / "out" / "second" / "summary.json" if name == "2196" else WORK / "df04" / "out" / "second" / "summary.json"
    s = J(p)
    a = s["audit"]["audit"]
    circ = sorted(set(re.findall(r"Greedy\w+\{[^}]*\}", json.dumps(s["audit"]["circuits"]))))
    read.append({"run": run, "issue": "NET-390", "lean": f"{sum(1 for v in a['verdicts'].values() if v)} of {len(a['verdicts'])} member verdicts true",
                 "what": f"{len(circ)} circuits ({circ[0]} … {circ[-1]}); audit accepted {a['accepted']}, outcome "
                         f"{s['outcome']['symbol']} on {s['outcome']['statements']}", "at_list_binding": False})
for u in ("positions", "whole"):
    d = J(WORK / "r3496" / "out" / "lean" / u / "lean-verify.json")
    read.append({"run": "r20261112-191822-3496", "issue": "CIR-251", "lean": "accepted" if d["accepted"] else "refused",
                 "what": f"{d['block']['definition']}, {d['block']['blocks']} unit, k_log {d['block']['k_log']}: {d['label']}; "
                         f"Program held {d['statement']['bound']['program_sha512'] == d['statement']['program']['sha512']}",
                 "at_list_binding": False})
fd = WORK / "86b4" / "out" / "second"
fst, fau, fout = J(fd / "statements.json"), J(fd / "audit" / "audit.json"), J(fd / "summary.json")["outcome"]
read.append({"run": "r20261112-191841-86b4", "issue": "CIR-271", "lean": f"{sum(1 for v in fau['verdicts'].values() if v)} of "
             f"{len(fau['verdicts'])} member verdicts true; audit accepted {fau['accepted']}, its Lean failed "
             f"{fau['verifiers']['lean']['failed']}",
             "what": f"{len(fst)} statements ({sorted({s['template'].split('{')[0] for s in fst})}), "
                     f"{fst[0]['header']['instances']} instances each, {fst[0]['draw']['k']} drawn; outcome {fout['symbol']} on {fout['statements']}",
             "at_list_binding": False})
read.append({"run": "r20261112-155548-66b2", "issue": "CIR-251", "lean": None,
             "what": "accepted by the proof server's session verifier only; no Lean verdict in its files", "at_list_binding": False})
acc = J(Path(sys.argv[5])) if len(sys.argv) > 5 else []
for r in sorted({a["run"] for a in acc}):
    hits = [a for a in acc if a["run"] == r]
    read.append({"run": r, "issue": "CIR-259", "lean": ", ".join(sorted({a["verdict"] for a in hits})),
                 "what": f"{len(hits)} statements (Part 1)", "at_list_binding": True})

st = J(RUNS / "r20261112-183158-1477" / "status.json")
read.append({"run": "r20261112-183158-1477", "issue": "CIR-259", "lean": None,
             "what": f"{st.get('state')} when fetched (its status written {st.get('updated_utc')}): "
                     "no staging step or verdict in its job.log yet", "at_list_binding": True})
au = J(RUNS / "r20261112-134838-1b6c" / "out" / "sparse-mla-units" / "summary.json")["rank0"]
read.append({"run": "r20261112-134838-1b6c", "issue": "CIR-259", "lean": f"audit accepted {au['audit']['accepted']}, "
             f"{sum(1 for v in au['audit']['verdicts'].values() if v)} of {len(au['audit']['verdicts'])} member verdicts true",
             "what": f"the audit of record (Part 1): {len(au['draw_counts'])} templates, one unit each", "at_list_binding": False})
seen = {x["run"] for x in read} | {x["run"][:16] for x in read}
named = {}
for f in sorted(LIN.glob("*.json")):
    issue = f.stem.removesuffix("-2")
    d = J(f)
    for c in d["comments"]:
        for m in RID.finditer(c["body"]):
            if m.group(0) not in seen:
                named.setdefault(issue, [])
                if m.group(0) not in named[issue]:
                    named[issue].append(m.group(0))
pages = {f.stem: {"comments": len(J(f)["comments"]), "more": J(f)["hasNextPage"],
                  "from": min(c["createdAt"] for c in J(f)["comments"])} for f in sorted(LIN.glob("*.json"))}
OUT.write_text(json.dumps({"read": read, "named_not_read": named, "pages": pages}, indent=1) + "\n")
print(len(read), "read;", {k: len(v) for k, v in named.items()}, "named not read")
