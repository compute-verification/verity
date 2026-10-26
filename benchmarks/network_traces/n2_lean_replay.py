"""Every recorded N2 proving run's link-window verdicts (`service.LinkWindows.verify`) through the Lean
(`NetworkCheck.LinkWindow`, `LinkWindows.lean_verdict`) beside the Python (`LinkWindows.python_verdict`), window by
window, with negative controls: the replay the 8 Oct switch asks for before `service.OF_RECORD` names the Lean.

    python n2_lean_replay.py --out DIR RUN_DIR ...

A RUN_DIR is an `n2_zk.sh` run's record tree on its node: `out/n2zk/summary.json`, each member's
`out/n2zk/<member>/verifier/` (what the verifier received, `n2_zk.RECEIVED`) and certifier-serve's runs under
`out/runs/{honest,extra,late}`. Each member's link is rebuilt from its run as `n2_zk.link` builds it (honest's and
tampered's from honest's). Per window, the verifier's inputs are read back from its files: the registration and the
draw, the certificates' public fields, the declared syncs, the statement's public file, the ``pub`` and ``o0`` rows'
salts and the output word, and the session files' SHA-512s. The findings of the verifier's own run of C-Flock's Lean
verifier (`proofs.verify`, already Lean) are the run's: the summary row's `proof` (refused, or each session accepted
on the summary's registered Program and the drawn unit where `proved`), since the run keeps no circuit file to run it
again; a member whose summary has no rows (a run that raised after proving it) takes each session as proved and is
counted apart (``findings: "assumed"``). The run's structure is checked, not assumed: the link's registered Program,
partition and gate ids as this tree derives them, against the summary's. A run made on a branch whose sidecar pinned a
size class this tree pins none of keeps that pin (`run_pins`, each listed as ``pins_from_run``), since its gate ids
name it; a pin the tree has is never replaced.

Each window's two verdicts (accept bit, public outputs, digest) are compared with each other and with the outcome
recorded (`outcome.json`'s ``network-audit`` verdict), whose digest binds the registration and the sessions. The
negative controls mutate one input of every window and run both again: the output word's verdict bit, a bit of the
``pub`` salt, a bit of the egress certificate's signature, the certificates swapped, the window's sync outside Σ_sync,
the findings refused, of another Program or of another unit, and a bit of the public file's ``o0`` commitment. A
control passes when both verdicts agree in every field and reject. Writes `replay.json` (every window and control) and
prints one line a member and a total; exit 1 unless every window agrees with itself and its record and every control
passes.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n2_zk as n2  # noqa: E402
from verity.core.protocols.certifier import advised as AD, head_check as HC, served as SV, service as SVC  # noqa: E402
from verity.core.protocols.certifier.certificate import Certificate, Direction  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync  # noqa: E402
from verity.core.protocols.sampled_proofs import service as S  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P, proofs as PF, registration as R  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage.proofs import Finding  # noqa: E402

#: each member's certifier-serve run under `out/runs`
RUN_OF = {"honest": "honest", "tampered": "honest", "extra": "extra", "late": "late"}
FIELDS = ("accepted", "public", "digest")
CONTROLS = ("word", "salt", "signature", "swapped", "sync", "refused", "program", "units", "commitment")


def _certificate(doc: dict) -> Certificate:
    return Certificate(bytes.fromhex(doc["key_id"]), doc["link"], (Direction.EGRESS, Direction.INGRESS)[doc["direction"]],
                       doc["window"], bytes.fromhex(doc["digest"]), bytes.fromhex(doc["signature"]))


def _flip(b: bytes, i: int = -1) -> bytes:
    out = bytearray(b)
    out[i] ^= 1
    return bytes(out)


def _same(a: dict | None, b: dict | None) -> bool:
    return a is not None and b is not None and all(a[k] == b[k] for k in FIELDS)


class Window:
    """One recorded link-window: the verifier's inputs read back from its files under `vdir` with prefix `pre`."""

    def __init__(self, vdir: Path, pre: str, row: dict | None, program: str):
        def doc(name):
            return json.loads((vdir / f"{pre}{name}").read_text())

        self.record, self.outcome = doc("registration.json"), doc("outcome.json")
        self.window = self.record["window"]["window"]
        reg = S.Registration(self.record, {}, R.digest(self.record), self.outcome["order"])
        self.draw = S.Draw(reg, {}, doc("draw.json"))
        self.certificates = [_certificate(c) for c in doc("certificates.json")]
        self.syncs = [ClockSync(d, Fraction(r)) for d, r in doc("syncs.json")]
        delivered = doc("delivered.json")
        self.public = vdir / f"{pre}pub-1.bin"
        self.salts = {k: bytes.fromhex(v) for k, v in delivered["salts"].items()}
        self.output = delivered["output"]
        files = sorted(vdir.glob(f"{pre}session-*-session.json"))
        self.sessions = [PF.file_sha512(f) for f in files]
        self.read = {f.name: PF.file_sha512(f) for f in [vdir / f"{pre}{n}" for n in (
            "registration.json", "draw.json", "certificates.json", "syncs.json", "delivered.json", "outcome.json",
            "pub-1.bin")] + files}
        self.recorded = (self.outcome.get("audit") or {}).get("verdicts", {}).get(SVC.VERIFIER)
        units = tuple(self.draw.units)
        if row is None:
            self.source = "assumed"
            self.findings = [Finding(s, True, "", program, {}, units) for s in self.sessions]
        else:
            self.source = "summary"
            proof = row["proof"]
            self.findings = None if proof["refused"] is not None else [
                Finding(s, True, "", program, {}, units) if proof["proved"] else Finding(s, False, "", None, {}, ())
                for s in self.sessions]

    def delivered(self, public: Path | None = None, salts: dict | None = None, output: int | None = None):
        return SVC.Delivered(PF.Proof(Path("/dev/null"), public or self.public, ()), salts or self.salts,
                             self.output if output is None else output)

    def verdicts(self, lw: SVC.LinkWindows, **edit) -> tuple[dict, dict | None]:
        args = (self.draw, edit.get("certificates", self.certificates), edit.get("syncs", self.syncs),
                edit.get("delivered", self.delivered()), edit.get("findings", self.findings), self.sessions)
        return lw.python_verdict(*args), lw.lean_verdict(*args)

    def controls(self, lw: SVC.LinkWindows, scratch: Path) -> dict[str, dict]:
        w, c = self.window, self.certificates
        outside = ClockSync(max(lw.sync_set.deltas) + 1, lw.sync_set.rhos[0])
        cut = scratch / f"{self.public.name}.flipped"
        cut.write_bytes(_flip(self.public.read_bytes()))
        edits = {
            "word": {"delivered": self.delivered(output=self.output ^ 1)},
            "salt": {"delivered": self.delivered(salts=self.salts | {"pub": _flip(self.salts["pub"])})},
            "signature": {"certificates": [dataclasses.replace(c[0], signature=_flip(c[0].signature)), c[1]]},
            "swapped": {"certificates": [c[1], c[0]]},
            "sync": {"syncs": [outside if k == w else s for k, s in enumerate(self.syncs)]},
            "refused": {"findings": None},
            "program": {"findings": [dataclasses.replace(f, program="0" * 128) for f in self.findings or []]},
            "units": {"findings": [dataclasses.replace(f, units=(1,)) for f in self.findings or []]},
            "commitment": {"delivered": self.delivered(public=cut)},
        }
        out = {}
        for name in CONTROLS:
            py, ln = self.verdicts(lw, **edits[name])
            out[name] = {"python": py["accepted"], "lean": None if ln is None else ln["accepted"],
                         "passed": _same(py, ln) and not py["accepted"]}
        return out


def run_pins(gates: list[str]) -> list[str]:
    """The run's own pins: each recorded gate's circuit digest for a size class this tree pins none of (a run on a branch
    whose sidecar pinned it), added to this process's pins before any gate is built. A pin the tree has is never
    replaced, so a recorded digest other than the tree's leaves the structure the run's not. The size classes added."""
    sidecars = {"NetworkHeadCheck_": HC.PINS, "NetworkServedCounts_": SV.PINS, "NetworkAdvisedCounts_": AD.PINS}
    added = []
    for g in gates:
        m = HC.GATE.fullmatch(g)
        path = next((p for k, p in sidecars.items() if m and m["definition"].startswith(k)), None)
        if path is None or m["digest"] == HC.UNPINNED or m["definition"] in HC._pins(path):
            continue
        HC._pins(path)[m["definition"]] = m["digest"]
        added.append(m["definition"])
    if added:
        HC.gate.cache_clear()
        SV.gate.cache_clear()
        AD.gate.cache_clear()
    return added


def _unpin(added: list[str]) -> None:
    for path in (HC.PINS, SV.PINS, AD.PINS):
        for fid in added:
            HC._pins(path).pop(fid, None)
    HC.gate.cache_clear()
    SV.gate.cache_clear()
    AD.gate.cache_clear()


def replay_run(run: Path, scratch: Path) -> dict:
    """Every member and window of one recorded run, with its controls, under the run's pins (`run_pins`)."""
    base = run / "out" / "n2zk"
    summary = json.loads((base / "summary.json").read_text())
    link = summary["link"]
    added = run_pins(link["gates"])
    try:
        return _replay(run, base, summary, link, added, scratch)
    finally:
        _unpin(added)


def _replay(run: Path, base: Path, summary: dict, link: dict, added: list[str], scratch: Path) -> dict:
    out = {"run": run.name, "link": {k: link.get(k) for k in ("name", "unit", "program", "partition", "gates")},
           "pins_from_run": added, "members": {}}
    for name in sorted(p.parent.name for p in base.glob("*/verifier")):
        lw, _ = n2.link(run / "out" / "runs" / RUN_OF[name])
        structure = {"program": lw.partition["program"] == link["program"],
                     "partition": P.digest(lw.partition) == link["partition"],
                     "gates": [g.id for g in lw.gates()] == link["gates"], "kind": lw.kind, "by": SVC.OF_RECORD[lw.kind]}
        rows = {r["window"]: r for r in (summary.get("members", {}).get(name, {}).get("windows") or [])}
        vdir, windows = base / name / "verifier", []
        for reg in sorted(vdir.glob("w*-registration.json")):
            pre = reg.name[:len("w000-")]
            x = Window(vdir, pre, rows.get(int(pre[1:4])), link["program"])
            py, ln = x.verdicts(lw)
            rec = x.recorded or {}
            windows.append({
                "window": x.window, "findings": x.source, "python": py, "lean": ln,
                "recorded": {k: rec.get(k) for k in FIELDS + ("sessions",)},
                "registration": x.outcome.get("registration") == x.draw.registration.digest,
                "agree": _same(py, ln), "as_recorded": _same(py, rec) and rec.get("sessions") == x.sessions,
                "controls": x.controls(lw, scratch), "read": x.read})
        ok = all(w["agree"] and w["as_recorded"] and w["registration"]
                 and all(c["passed"] for c in w["controls"].values()) for w in windows)
        out["members"][name] = {"structure": structure, "windows": windows, "ok": ok and bool(windows),
                                "accepted": [int(w["python"]["accepted"]) for w in windows]}
    return out


def _task(run: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        try:
            return replay_run(Path(run), Path(d))
        except Exception as e:  # a run missing or unreadable is reported, not skipped
            return {"run": Path(run).name, "error": f"{type(e).__name__}: {e}", "members": {}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="n2_lean_replay.py", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.jobs <= 1:
        results = [_task(r) for r in a.runs]
    else:
        # each run in a fresh process, so one run's pins (`run_pins`) never reach another's
        with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=1) as pool:
            results = list(pool.map(_task, a.runs))
    total = {"runs": len(results), "windows": 0, "agree": 0, "as_recorded": 0, "assumed": 0, "controls": 0,
             "controls_passed": 0, "accepted": 0, "rejected": 0}
    for r in results:
        if "error" in r:
            print(f"N2LEAN {r['run']}: ERROR {r['error']}", flush=True)
        for name, m in r["members"].items():
            ws = m["windows"]
            cs = [c for w in ws for c in w["controls"].values()]
            total["windows"] += len(ws)
            total["agree"] += sum(w["agree"] for w in ws)
            total["as_recorded"] += sum(w["as_recorded"] for w in ws)
            total["assumed"] += sum(w["findings"] == "assumed" for w in ws)
            total["accepted"] += sum(m["accepted"])
            total["rejected"] += len(ws) - sum(m["accepted"])
            total["controls"] += len(cs)
            total["controls_passed"] += sum(c["passed"] for c in cs)
            print(f"N2LEAN {r['run']} {name} ({m['structure']['kind']}, by {m['structure']['by']}): "
                  f"{len(ws)} windows, verdicts {''.join(map(str, m['accepted']))}, "
                  f"Lean = Python {sum(w['agree'] for w in ws)}/{len(ws)}, "
                  f"as recorded {sum(w['as_recorded'] for w in ws)}/{len(ws)}, "
                  f"controls {sum(c['passed'] for c in cs)}/{len(cs)}, structure {m['structure']}, "
                  f"pins from the run {r['pins_from_run'] or 'none'}; "
                  f"{'ok' if m['ok'] else 'FAILED'}", flush=True)
    ok = (all("error" not in r and r["members"] and all(m["ok"] for m in r["members"].values()) for r in results))
    total["ok"] = ok
    (a.out / "replay.json").write_text(json.dumps({"total": total, "runs": results}, indent=1, sort_keys=True,
                                                  default=str))
    print(f"N2LEAN total: {json.dumps(total, sort_keys=True)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
