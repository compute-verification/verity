"""Catches the compiled `network-rows` and the device's Python check deciding one network attester run differently,
which neither one's own tests can: each holds itself to fixtures it made, never to the other's verdict on the same
signed records.

The Python verdict is `relay_check.check`'s of the statement `network-rows` checks: each window's requests', responses'
and fails' records bound to its certified record (its ``windows``) and `network_attester.audit` over them (its
``audit_windows``). The runs are `relay_runs._run`'s at the scan's w63k4 schedule, in the id egress form, each forged
case a copy of the honest run edited, or its records republished under a key of their own. Two cases are refused by
design and asserted as such: a schedule with a `max_tokens_cap`, which `network-rows` refuses at its params, and an
egress cell that isn't an id-form cell, which only the Python audit reads; a run in the bytes egress form is outside
what `network-rows` checks, its schedule's digest binding the id form."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import verity_difftest as D
from tests.verity.protocols.resource_accounting.network.attester.relay_runs import SESSION, _run
from verity.protocols.resource_accounting.network.attester import network_attester_serve as ws
from verity.protocols.resource_accounting.network.verifier import blob_files as bf, io_tie as IT
from verity.protocols.resource_accounting.network.verifier import network_attester_rows as WR, period_check
from verity.protocols.resource_accounting.network.verifier import record_files as RF, relay_check as RC
from verity.protocols.resource_accounting.network.verifier.network_attester import (
    DATA, END, W63K4, NetworkAttesterParams, cell, read_cell)

LEAN = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "resource_accounting" / "network" / "verifier" / "lean"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]

ROWS = "verity/network-rows/v1"
TOKENS = 2
CAPPED = "params: schedule: the rows audit reads no max_tokens_cap, so a run under one isn't audited as rows"


def served(out: Path, params: NetworkAttesterParams = W63K4, windows: int = 2) -> Path:
    """A run of `lanes` id requests admitted together, each answered with `TOKENS` ids, over ``windows`` windows."""
    out.mkdir(parents=True)
    requests = [ws.id_request(model="m", user=f"user-{j}", prompt=[j + 1], max_tokens=TOKENS,
                              frame_bytes=params.cell_bytes, offset=0) for j in range(params.lanes)]
    responses = [ws.id_response(list(range(1, TOKENS + 1)), params.payload)] * params.lanes
    _run(out, out.parent / f"{out.name}-key", requests, responses, windows=windows, params=params,
         id_payload=params.payload)
    return out


def case(run: Path) -> dict:
    dep = json.loads((run / "deployment.json").read_text())
    return RC.rows_audit_case(run, dep, NetworkAttesterParams(**dep["params"]))


def python(run: Path) -> list[str]:
    """The Python check's refusals of the statement `network-rows` checks; none is its accept."""
    r = RC.check(run)
    return ([f"{w['record']} window {w['window']} isn't certified" for w in r["windows"] if not w["certified"]]
            + [f"{w['record']} window {w['window']}: {w['status']}" for w in r["windows"] if w["certified"] and w["status"]]
            + [f"audit: {p}" for a in r["audit_windows"] for p in a["problems"]])


def lean(exe: str, run: Path, d: Path) -> dict:
    got = period_check.lean_check(exe, case(run), d)
    assert got.get("protocol") == ROWS and got.get("verdict") in ("accept", "reject"), got
    return got


def copied(honest: Path, out: Path) -> Path:
    shutil.copytree(honest, out)
    return out


def edit_deployment(run: Path, change) -> Path:
    dep = json.loads((run / "deployment.json").read_text())
    (run / "deployment.json").write_text(json.dumps(change(dep)))
    return run


def republished(honest: Path, out: Path, change=lambda name, r: r) -> Path:
    """The honest run's records, each as ``change`` gives it, kept and attested by `Publisher` under a key of their own,
    beside the honest deployment with that key; the wire's rows stay the honest records'."""
    dep = json.loads((honest / "deployment.json").read_text())
    params = NetworkAttesterParams(**dep["params"])
    out.mkdir()
    signer = ws.ck.signer(out.parent / f"{out.name}-key", 4)
    records = RC.read_records(honest, params)
    by = {(r.link, r.window): r for rs in records.values() for r in rs}
    pub = ws.Publisher(out, ws.Issuer(signer, SESSION), params,
                       lambda link, d, r, *a: WR.Cells.of(link, d, by[r.link, r.window], *a))
    for w in range(dep["windows"]):
        for name in RC.CERTIFIED:
            (r,) = [r for r in records[name] if r.window == w]
            pub.keep(change(name, r), params.fails_grid if name == RC.FAILS else params.grid)
    (out / "deployment.json").write_text(json.dumps(dep | {"key": RF.public_json(signer.public_key())}))
    shutil.copyfile(honest / "client.json", out / "client.json")
    return out


def responses0(edit):
    """A `republished` change: window 0's responses record with its grid (rows of cells) edited in place."""
    def change(name, r):
        if (name, r.window) != (RC.EGRESS, 0):
            return r
        grid = [list(row) for row in r.grid]
        edit(grid)
        return replace(r, grid=tuple(map(tuple, grid)))
    return change


def _first_data(grid) -> tuple[int, int]:
    return next((b, k) for b, row in enumerate(grid) for k, c in enumerate(row)
                if c is not None and read_cell(c, W63K4.cell_bytes)[0] == DATA)


def recut(grid) -> None:
    b, k = _first_data(grid)
    _, o, payload = read_cell(grid[b][k], W63K4.cell_bytes)
    grid[b][k] = cell(END, o, payload, W63K4.cell_bytes)


def reclock(grid) -> None:
    b, _ = _first_data(grid)
    grid[b - 1], grid[b] = grid[b], grid[b - 1]


def reorder(grid) -> None:
    r = W63K4.slots
    b, _ = _first_data(grid)
    grid[b][:r], grid[b][r:2 * r] = grid[b][r:2 * r], grid[b][:r]


def not_id(grid) -> None:
    b, k = _first_data(grid)
    c = bytearray(grid[b][k])
    c[IT.ID_AT + IT.ID_DIGITS - 1] = ord("x")
    grid[b][k] = bytes(c)


def drop_record(run: Path, name: str, window: int) -> Path:
    path = run / f"{name}.records"
    blobs = bf.read_blobs(path)
    path.unlink()
    for blob in blobs:
        if RF.unpack_record(blob, W63K4.cell_bytes).window != window:
            bf.write_blobs(path, blob)
    return run


def drop_certified(run: Path, link: str, window: int) -> Path:
    path = run / "certified.jsonl"
    lines = [x for x in path.read_text().splitlines()
             if (json.loads(x).get("head", {}).get("link"), json.loads(x).get("head", {}).get("window")) != (link, window)]
    path.write_text("".join(x + "\n" for x in lines))
    return run


def forge_certificate(run: Path) -> Path:
    path = run / "certified.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    sig = lines[0]["certificate"]["signature"]
    lines[0]["certificate"]["signature"] = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return run


def change_record(run: Path) -> Path:
    """Window 0's responses record with a byte of a `DATA` cell flipped, its certified record left as signed."""
    params = W63K4
    path = run / "responses.records"
    records = [RF.unpack_record(b, params.cell_bytes) for b in bf.read_blobs(path)]
    path.unlink()
    for r in records:
        if r.window == 0:
            grid = [list(row) for row in r.grid]
            b, k = _first_data(grid)
            grid[b][k] = grid[b][k][:-1] + bytes([grid[b][k][-1] ^ 1])
            r = replace(r, grid=tuple(map(tuple, grid)))
        bf.write_blobs(path, RF.pack_record(r))
    return run


def test_network_rows_and_the_python_check_decide_each_run_alike(tmp_path):
    """Each case's two verdicts: the honest run, the run republished under another key and the deployment naming one
    window fewer accepted by both; a forged certificate, a record changed under its head, a grid re-cut, re-clocked
    or reordered and signed, another session and another key refused by both, and a record or a certified record
    missing or a window more than the run holds refused by the Python check and by the case's builder before
    `network-rows`.
    Every disagreement is reported with both verdicts. By design: a run under a
    `max_tokens_cap` the Python check accepts and `network-rows` refuses by name; a run whose egress cell isn't
    id-form the Python audit refuses and `network-rows` accepts; and a bytes-form run the Python check accepts, the
    case's builder refuses by name and `network-rows`, handed it as an id-form run's, refuses. The relay check, running it, names no `lean-disagrees` on any of them, and its `lean` is
    accepted on exactly those `network-rows` accepts and not covered on the capped and bytes-form runs."""
    exe = str(D.build(LEAN, "network-rows"))
    honest = served(tmp_path / "honest")
    resigned = republished(honest, tmp_path / "resigned")
    other_key = json.loads((resigned / "deployment.json").read_text())["key"]
    runs = {
        "honest": (honest, True),
        "resigned": (resigned, True),
        "one window fewer": (edit_deployment(copied(honest, tmp_path / "fewer"), lambda d: d | {"windows": 1}), True),
        "forged certificate": (forge_certificate(copied(honest, tmp_path / "forged")), False),
        "record changed under its head": (change_record(copied(honest, tmp_path / "changed")), False),
        "re-cut": (republished(honest, tmp_path / "re-cut", responses0(recut)), False),
        "re-clocked": (republished(honest, tmp_path / "re-clocked", responses0(reclock)), False),
        "reordered": (republished(honest, tmp_path / "reordered", responses0(reorder)), False),
        "another session": (edit_deployment(copied(honest, tmp_path / "session"), lambda d: d | {"session": "11" * 32}),
                            False),
        "another key": (edit_deployment(copied(honest, tmp_path / "key"), lambda d: d | {"key": other_key}), False),
    }
    verdicts = {}
    for name, (run, _) in runs.items():
        got = lean(exe, run, tmp_path / f"case-{name.replace(' ', '-')}")
        verdicts[name] = (python(run), got["verdict"], got["reason"])
    disagree = {n: v for n, v in verdicts.items() if (not v[0]) != (v[1] == "accept")}
    assert not disagree, "python's refusals and network-rows' verdict differ: " + json.dumps(disagree, indent=1)
    wrong = {n: verdicts[n] for n, (_, accept) in runs.items() if (verdicts[n][1] == "accept") != accept}
    assert not wrong, "both checks agree on a verdict the case isn't: " + json.dumps(wrong, indent=1)

    for run, why in ((drop_record(copied(honest, tmp_path / "no-record"), RC.FAILS, 1), "fails window 1 has no record"),
                     (drop_certified(copied(honest, tmp_path / "no-cert"), RC.LINK, 1),
                      "requests window 1 has no certified record"),
                     (edit_deployment(copied(honest, tmp_path / "more"), lambda d: d | {"windows": 3}),
                      "requests window 2 has no record")):
        assert python(run), run
        with pytest.raises(ValueError, match=f"^{why}$"):
            case(run)

    capped = served(tmp_path / "capped", replace(W63K4, max_tokens_cap=4))
    assert python(capped) == []
    got = lean(exe, capped, tmp_path / "case-capped")
    assert (got["verdict"], got["reason"]) == ("reject", CAPPED), got

    not_id_run = republished(honest, tmp_path / "not-id", responses0(not_id))
    assert any(p.startswith("audit: ") for p in python(not_id_run)), python(not_id_run)
    got = lean(exe, not_id_run, tmp_path / "case-not-id")
    assert (got["verdict"], got["reason"]) == ("accept", None), got

    bytes_run = served(tmp_path / "bytes", replace(W63K4, egress="bytes"))
    assert python(bytes_run) == []
    with pytest.raises(ValueError, match="^the rows audit's schedule is the id egress form's, not 'bytes''s$"):
        case(bytes_run)
    dep = json.loads((bytes_run / "deployment.json").read_text())
    as_id = RC.rows_audit_case(bytes_run, dep, replace(NetworkAttesterParams(**dep["params"]), egress="id"))
    got = period_check.lean_check(exe, as_id, tmp_path / "case-bytes")
    assert (got["verdict"], got["reason"]) == ("reject", "requests window 0: the certified head isn't the record's"), got

    accepted = {"honest", "resigned", "one window fewer", "not id"}
    for name, run in [*((n, r) for n, (r, _) in runs.items()), ("not id", not_id_run), ("capped", capped),
                      ("bytes", bytes_run)]:
        result = RC.check(run, rows_exe=Path(exe))
        assert not [p for p in result["problems"] if p.startswith("lean-disagrees")], (name, result["problems"])
        assert (result["lean"] == "accepted") == (name in accepted), (name, result["lean"])
        assert result["lean"].startswith("not-covered: ") == (name in ("capped", "bytes")), (name, result["lean"])
