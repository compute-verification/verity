"""The audit of the loop's PoUW window (`pouw`) and PoUW's answer on it.

    python -m verity_torch_loop.pouw_audit --window DIR --circuit SHA512 --run HEX --m0-pythonpath PATH --bin flock-circuit \\
        --lean flock-verify --firewall flock-firewall --out DIR [--forged | --late] [--pouw-check BIN] [--skip-negatives]

The audit is one-stage's from the records alone (`examples/prove-path/records_audit.py`, `a5.audit`): the registration
checked by Lean and Python, the draw on the verifier's live coins read after the receipt, C-Flock ``--zk`` behind the Lean
firewall on the drawn tiles, Lean's verdicts and the audit record. What differs is because the registration's
window is PoUW's ``{context, kind, public}``, not one-stage's provenance:

* the verifier's window comes from ``circuit.json`` held to ``provenance.json`` (the window one-stage would name): its
  SHA-512 is the provenance's ``circuit`` and the circuit the verifier registered (``--circuit``), and the provenance's run
  is the registration's context and the run the loop published (``--run``) (`verifier_window`, `source`);
* the template is the pinned H1 tile, resolved by `pouw.resolve` and composed by C-Flock's `UnitStager` on the record's
  statement (`records_audit.composer`);
* a member's public header names its set by the window's run, which a PoUW window states as its context
  (`flock_input_files`), and the window's operands (``activations``) are held to domains binding that context
  (`committed_activations`);
* the tiles' ``a`` and ``b`` rows are registered reads of those operands: each drawn member's ``--registered`` file comes
  from the record and the registration's entries alone (`operands.held`, records_audit's ``--registered``, in place of
  its `window_pins`, which reads a one-stage window's ``run``), and Lean's
  registered reads open every row against them.

Then `answer` makes the proof service's answer for the registration in PoUW's evidence form (`core_check.evidence`:
public ``{"draw": ...}``, the profile at the verifier's δ), written to ``DIR/answer.json``, and `core_check.answer` (and,
given ``--pouw-check``, Lean's) decides the window at the verifier's parameters (`pouw`'s ε, δ, check item and range, the
published run as the context; the requests' max_tokens as registered), written to ``--out``'s
``pouw.json``: verdict, reason, W (the credited MACs), ε, δ, K′. ``--forged``: the window holds tampered tiles, and the
audit passes when Lean refused them. A live window (``salt.json`` beside it, `pouw.SALT`) is checked with its coins live
and the salt session's ``live`` params, its answer carrying the session's ``public.salt``; ``--late``: its session was
opened on other operands (`serve`'s ``--pouw-late``), and the audit passes when the tiles' proofs are accepted and PoUW
refuses the salt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from verity.core.protocols.pouw import core_check as CC, operands as OP, window as Wd
from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R
from verity.ml.windows import interface as IF

from . import pouw as PW

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "prove-path"))

import records_audit as RA  # noqa: E402

from verity.ml.windows import flock_inputs as FI  # noqa: E402

_source, _input_files, _activations = RA.source, FI.flock_input_files, RA.committed_activations


def flock_input_files(window_dir, k, circuit, out_dir, *, write=("pub", "inst"), files=None):
    """`flock_inputs.flock_input_files`, its header's ``set`` named by the run as for one-stage's window: a PoUW
    window's run is its context."""
    d = Path(window_dir)
    win, rec = files if files is not None else (json.loads((d / "window.json").read_text()),
                                                json.loads((d / "registration.json").read_text()))
    w = rec.get("window") or {}
    if "run" not in w and "context" in w:
        rec = {**rec, "window": {**w, "run": w["context"]}}
    return _input_files(window_dir, k, circuit, out_dir, write=write, files=(win, rec))


def _provenance(wdir: Path) -> tuple[dict, dict]:
    rec = json.loads((wdir / "registration.json").read_text())
    prov = json.loads((wdir / "provenance.json").read_text())
    return rec, prov


def verifier_window(window_dir: Path, law: Mapping[str, Any], template_sets: Path | None) -> tuple[Any, dict, dict]:
    """`records_audit.verifier_window` for PoUW's registration (module docstring)."""
    if template_sets:
        raise ValueError("the loop's PoUW window has no template sets")
    d = Path(window_dir)
    rec, prov = _provenance(d)
    body = (d / "circuit.json").read_bytes()
    if hashlib.sha512(body).hexdigest() != prov.get("circuit"):
        raise ValueError("circuit.json is not the circuit record the window's provenance names")
    if (rec.get("window") or {}).get("context") != prov.get("run"):
        raise ValueError("the registration's context is not the provenance's run")
    doc = json.loads(body)
    if (doc.get("program"), doc.get("partition"), doc.get("run")) != (rec.get("program"), rec.get("partition"), prov["run"]):
        raise ValueError("the circuit record names another program, partition or run than its registration")
    mine = IF.window_of_record(doc, law, PW.resolve)
    return mine, doc, {t: (RA.composer(PW.resolve(t), st), PW.resolve(t)) for t, st in doc["statements"].items()}


def source(w: Mapping[str, Any] | None, held: Mapping[str, Any]) -> dict[str, Any]:
    """`records_audit.source` on the window one-stage would name: a PoUW window's provenance when its context is the
    provenance's run; any other window as given (so a registration naming another circuit or run is refused)."""
    w = dict(w or {})
    if sorted(set(w) - set(CC.WINDOW_VALUES)) == ["context", "kind", "public"]:
        prov = held["provenance"]
        v = _source({**prov, "kind": w["kind"]} if w["context"] == prov.get("run") else {}, held)
        return v
    return _source(w, held)


def committed_activations(w: Mapping[str, Any]) -> dict[str, Any] | None:
    """`records_audit.committed_activations` with a PoUW window's run read as its context."""
    w = dict(w)
    if "run" not in w and "context" in w:
        w["run"] = w["context"]
    return _activations(w)


def answer(audit_out: Path, rec: Mapping[str, Any], delta: Fraction, salt: Mapping[str, Any] | None = None
           ) -> dict[str, Any]:
    """The proof service's answer for ``rec`` in PoUW's evidence form, from the audit records_audit wrote under
    ``audit_out``: on acceptance public ``{"draw": d}`` and the profile at ``delta``, and a live window's ``salt`` (its
    salt session's answer, `window_salt.Issued.answer`)."""
    commitment = R.digest(rec)

    def refuse(why: str) -> dict[str, Any]:
        return {"commitment": commitment, "profile": None, "protocol": R.PROTOCOL, "public": None, "reason": why,
                "verdict": "reject"}

    summary = json.loads((audit_out / "summary.json").read_text())
    if summary.get("refused"):
        return refuse("refused: " + json.dumps(summary["refused"], sort_keys=True, default=str))
    before = summary.get("checks_before_draw") or {}
    if before and not all(before.values()):
        return refuse("before the draw: " + ", ".join(sorted(c for c, v in before.items() if not v)))
    f = audit_out / "window" / "audit.json"
    if not f.exists():
        return refuse("no audit record: " + json.dumps((summary.get("audit") or {}).get("outcome"), sort_keys=True, default=str))
    a = json.loads(f.read_text())
    if a.get("accepted") is not True:
        return refuse("audit record: not accepted")
    if not (summary.get("fixed_before_draw") or {}).get("ok"):
        return refuse("not fixed before the draw")
    draw = a["draw"]
    profile = A.profile(a["registration"], draw, float(delta)).to_json()
    profile["delta"] = str(delta)
    public = {"draw": {"k": draw["k"], "law": draw["law"], "population": draw["population"], "units": sorted(draw["units"])},
              **({"salt": dict(salt)} if salt is not None else {})}
    return {"commitment": commitment, "profile": profile, "protocol": R.PROTOCOL, "public": public, "reason": None,
            "verdict": "accept"}


def keep(src: Path, dst: Path) -> None:
    """The audit's records from ``src`` into ``dst``, as `examples/prove-path/run.sh`'s ``keep``: never a member's
    circuit (its SHA-512 is in the records), a session's archive, an instance file, a private file or the prover's copy of
    a member's files."""
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if not f.is_file() or {"archive", "prover"} & set(rel.parts[:-1]) or f.name == "circuit.txt" \
                or f.name.endswith("-private.bin") or f.name.startswith("inst-") or re.match(r"m[0-9]+-inst-", f.name):
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)


_records_audit = RA.main


def in_scratch(argv: list[str]) -> int:
    """`records_audit.main` with the audit's tree (each member's circuit, the sessions' archives, the prover's files,
    tens of GiB for an H1 window) in a temporary directory, removed after, and only its records (`keep`) in the
    ``--out`` it was given. The directory is under ``$TMPDIR``, else ``/tmp``."""
    argv = list(argv)
    i = argv.index("--out") + 1
    out = Path(argv[i])
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pouw-audit-", ignore_cleanup_errors=True) as work:
        argv[i] = work
        try:
            return _records_audit(argv)
        finally:
            keep(Path(work), out)


def lean_line(exe: str, params: Any, commitment: bytes, evidence: bytes) -> str:
    """`pouw-check check PARAMS COMMITMENT EVIDENCE`'s outcome line."""
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, data in (("params", CC.line(params).encode()), ("commitment", commitment), ("evidence", evidence)):
            (Path(d) / name).write_bytes(data)
            files.append(str(Path(d) / name))
        p = subprocess.run([exe, "check", *files], capture_output=True, text=True, check=False)
    if p.returncode not in (0, 1, 2):
        raise RuntimeError(f"pouw-check exited {p.returncode}: {p.stderr.strip()}")
    return p.stdout.strip()


def decide(params: Mapping[str, Any], rec: Mapping[str, Any], ans: Mapping[str, Any], pouw_check: str | None) -> dict:
    """`core_check.answer` (and Lean's, which must agree) on the window, with W, ε, δ and K′ as the registration states
    them."""
    commitment, evidence = CC.line(rec).encode(), CC.line(ans).encode()
    got = CC.answer(CC.line(params).encode(), commitment, evidence)
    line = CC.line(got)
    out = {"python": got}
    if pouw_check:
        lean = lean_line(pouw_check, params, commitment, evidence)
        out["lean"] = lean
        out["agree"] = lean == line
    items = rec["window"]["public"]
    s = items["sampling"]
    out |= {"verdict": got["verdict"], "reason": got["reason"], "W": items["work"]["credited"],
            "tiles": {"credited": items["work"]["credited_tiles"], "n": items["work"]["n"]},
            "eps": s["eps"], "delta": s["delta"], "k": s["law"]["k"]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for f in ("--window", "--circuit", "--run", "--m0-pythonpath", "--bin", "--lean", "--firewall", "--out"):
        ap.add_argument(f, required=True)
    ap.add_argument("--delta-audit", default="0.5", help="the one-stage audit's own δ (records_audit's --delta)")
    ap.add_argument("--forged", action="store_true")
    ap.add_argument("--late", action="store_true", help="the salt session names other operands: PoUW must refuse the salt")
    ap.add_argument("--pouw-check", help="a built Lean pouw-check, which must agree with core_check")
    ap.add_argument("--skip-negatives", action="store_true", help="records_audit's --skip-negatives: a run of several "
                    "periods runs the negatives in its first")
    a = ap.parse_args(argv)
    wdir, out = Path(a.window), Path(a.out)
    rec, prov = _provenance(wdir)
    law = rec["law"]
    held_prov = {"provenance": prov}
    RA.main = in_scratch
    RA.verifier_window = verifier_window
    FI.flock_input_files = flock_input_files
    RA.source = lambda w, held: source(w, {**held, **held_prov})
    RA.committed_activations = committed_activations
    RA.pinned = lambda doc: OP.held({"context": rec["window"]["context"], OP.KEY: doc})
    # the window's own values are its operands, which --registered pins (`operands.held`, under the window's context)
    RA.window_pins = lambda w, circuit, resolve=None: ({}, {})
    out.mkdir(parents=True, exist_ok=True)
    (out / "operands.json").write_text(json.dumps(rec["window"].get(OP.KEY), sort_keys=True))
    rc = RA.main(["audit", "--window", str(wdir), "--law", f"{law['law']}:{law['k']}", "--circuit", a.circuit,
                  "--run", a.run, "--m0-pythonpath", a.m0_pythonpath, "--bin", a.bin, "--lean", a.lean,
                  "--firewall", a.firewall, "--out", str(out / "audit"), "--delta", a.delta_audit,
                  "--registered", str(out / "operands.json"),
                  *(["--forged"] if a.forged else []), *(["--skip-negatives"] if a.skip_negatives else [])])
    items = rec["window"]["public"]
    salt = json.loads((wdir / PW.SALT).read_text()) if (wdir / PW.SALT).exists() else None
    params = {"check": PW.check_item(live=salt is not None), "context": a.run, "delta": str(PW.DELTA), "eps": str(PW.EPS),
              "grid_bits": Wd.GRID_BITS, "length": {"max_tokens": items["length"]["max_tokens"], "stop": Wd.STOP_FIXED},
              "query": CC.TILES[PW.tile().id].query, "range": list(PW.RANGE),
              **({CC.LIVE_PARAM: salt["live"]} if salt is not None else {})}
    ans = answer(out / "audit", rec, PW.DELTA, salt["answer"] if salt is not None else None)
    (wdir / "answer.json").write_text(CC.line(ans) + "\n")
    got = decide(params, rec, ans, a.pouw_check)
    got |= {"audit_rc": rc, "forged": a.forged, "live": salt is not None, "params": params}
    (out / "pouw.json").write_text(json.dumps(got, indent=1, sort_keys=True, default=str) + "\n")
    print("POUW " + json.dumps({k: got.get(k) for k in ("verdict", "reason", "W", "tiles", "eps", "delta", "k", "agree",
                                                       "audit_rc", "forged", "live")}, sort_keys=True, default=str))
    if a.forged:
        return 0 if rc == 0 and got["verdict"] != "accept" else 1
    if a.late:
        return 0 if rc == 0 and got["verdict"] != "accept" and str(got["reason"]).startswith("salt") \
            and got.get("agree", True) else 1
    return 0 if got["verdict"] == "accept" and got.get("agree", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
