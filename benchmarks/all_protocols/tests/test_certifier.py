"""The communication leg: the network certifier over the period, on a link that carries the torch loop's inputs and
outputs.

* the requests and step outputs the leg reads back from the loop's window are the loop's own;
* the endpoint answers every completion with the loop's outputs, one canonical event per 32 bytes as certifier-serve's
  probe counts them, and the gate's calls (``/v1/models``, ``/tokenize``) with the loop's model and requests;
* a link-window decided in the clear is ``undecided``; one decided by a proof whose audit accepted is one-stage's accept
  for its registration, one unit drawn ``subset:1``; a refused proof, or none delivered, rejects;
* the month's count is ``HorizonLeakAccepted``'s terms, the faulty units paid once at each δ;
* a period on the virtual certifier (its link, certificates and clock certificates real), each link-window registered as
  the prover registers it and its outcome standing in for the stream's (whose verifiers of record are Lean, which this
  suite doesn't build), the case decided by the period check's Python mirror: decided in the clear, the leg's answer is
  ``undecided`` and it keeps only public files; decided by proofs, the period accepts with ``K`` (the compute audit's free
  inputs, faulty sets and bits, the schedules and cells); and a forged certificate rejects.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

from verity.core.protocols.certifier import service as SVC
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity_all_protocols import table as T

L = T.load("certifier")
ROOT = Path(__file__).resolve().parents[3]


def _mirror():
    """The period check's Python mirror and generator (the network suite's difftest), as the composition's difftest loads
    it."""
    path = ROOT / "verity/core/protocols/network/tests/test_network_period_protocol.py"
    spec = importlib.util.spec_from_file_location("network_period_difftest", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


NP = _mirror()


def window(tmp_path: Path, requests=(3, 7), steps: int = 1) -> tuple[Path, object]:
    """A served period's window, as `verity_torch_loop.serve` writes it, and the loop that served it."""
    from verity_torch_loop import loop as TL

    loop = TL.TorchLoop(TL.Model(), list(requests), steps, run_id="test")
    win = loop.window(D.parse_law("subset:2"), 8)
    d = tmp_path / "p0"
    win.write(win.close(), d, {"loop": "torch-loop"}, registrar=R)
    return d, loop


def test_the_loop_io_is_the_loops(tmp_path):
    from verity_torch_loop import loop as TL

    d, loop = window(tmp_path, (3, 7), 2)
    tokens, outputs = L.loop_io(d)
    assert tokens == [3, 7]
    assert outputs == [[TL.bits(loop.outputs[r, s].to(torch.bfloat16)).astype("<u2").tobytes() for s in range(2)]
                       for r in range(2)]
    assert L.free_bits(d) == loop.units * (1024 + 32 + 1024 + 32)


def test_the_endpoint_serves_the_loop_one_event_per_32_bytes():
    cs = L.cs
    outputs = [[bytes(range(256))], [bytes(256)]]
    texts = L.transcript(outputs)
    assert len(texts) == 16 and all(len(t) == 64 for t in texts)
    assert bytes.fromhex("".join(texts)) == bytes(range(256)) + bytes(256)
    body = cs.request_body(L.MODEL, len(texts), prompt=[3, 7])

    async def go():
        server = await asyncio.start_server(L.endpoint([3, 7], texts), "127.0.0.1", 0)
        url = "http://%s:%d" % server.sockets[0].getsockname()[:2]
        probed = await asyncio.to_thread(cs.probe, url, body, 2, clock=lambda: 0.0)
        model = await asyncio.to_thread(cs.served_model, url)
        tokens = await asyncio.to_thread(cs.prompt_tokens, url, model, 2)
        server.close()
        return probed, model, tokens

    probed, model, tokens = asyncio.run(go())
    assert [r["events"] for r in probed["responses"]] == [16, 16]
    assert model == L.MODEL and tokens == [3, 7]


REG = {"format": R.FORMAT, "protocol": R.PROTOCOL, "program": "aa" * 64, "partition": "bb" * 64, "population": 1,
       "law": {"law": "subset", "k": 1}}


def body(decided_by: str | None, accepted: bool = True, symbol: str = "accept", reg: dict = REG) -> dict:
    """An outcome body as the link's stream records it, its audit by the verifier of record ``decided_by``."""
    if decided_by is None:
        return {"symbol": symbol, "audit": None}
    return {"symbol": symbol, "audit": {
        "accepted": accepted, "registration": reg, "draw": {"law": "subset", "k": 1, "population": 1, "units": [0]},
        "proved_units": [0], "checks": {"proof": {"ok": accepted}},
        "wrong_units": {"delta": 2.0 ** -40, "of": 1, "bound": 0} if accepted else None,
        "verdicts": {SVC.VERIFIER: {"decided_by": decided_by}}}}


def test_a_window_is_accepted_only_on_a_proof():
    a = L.window_answer(REG, body("proof"))
    assert T.check_answer(a) == a and a["verdict"] == "accept" and a["commitment"] == R.digest(REG)
    assert a["profile"]["delta"] == "1/1099511627776"
    assert [(lv["name"], lv["size"], lv["draw"]) for lv in a["profile"]["levels"]] == [("unit", 1, {"subset": 1})]
    assert a["public"] == {"bound": 0, "drawn": [0], "per_template": None, "proved": [0]}
    clear = L.window_answer(REG, body("clear"))
    assert clear["verdict"] == "undecided" and "not by a proof" in clear["reason"]
    refused = L.window_answer(REG, body("proof", accepted=False, symbol="reject"))
    assert refused["verdict"] == "reject" and "proof" in refused["reason"]
    late = L.window_answer(REG, body(None, symbol="late"))
    assert late["verdict"] == "reject" and "late" in late["reason"]
    assert all(T.check_answer(x) == x for x in (clear, refused, late))


def test_the_month_pays_the_faulty_units_once():
    params = {"audits": [{"free": 1000, "steps": 120}], "cells": [7440], "jitter": 1, "windows": 2,
              "sync_set": {"deltas": list(range(16)), "rhos": [[31, 1]]}}
    audit = {"profile": {"levels": [{"size": 1024, "draw": {"subset": 2}}]}}
    m = L.month(params, [audit], 3600)
    assert m["periods"] == 720 and m["q"] == "1/512" and m["units"] == 720 * 1024
    assert m["terms"] == {"schedules": 720 * 2 * 4, "cells": 720 * 13, "free": 720 * 1000, "jitter": 0}
    for d, fmax in (("1/100", math.ceil(512 * math.log(100))), ("1/1000000", math.ceil(512 * math.log(10 ** 6)))):
        row = m["at"][d]
        assert row["Fmax"] == fmax
        assert row["faulty"] == math.ceil(fmax * (16 + math.log2(720 * 1024 + 1)))
        assert row["K"] == row["faulty"] + sum(m["terms"].values())


def decided(run: Path, decided_by: str) -> list[tuple[dict, dict]]:
    """Each link-window's registration as the prover makes it (`LinkWindows.registration`, chained by digest) and an
    accepted outcome by ``decided_by``."""
    rd = L.NZ.link_windows(run)
    out, previous = [], None
    for x in rd.windows:
        reg = rd.lw.registration(x.opening, previous)
        previous = R.digest(reg)
        out.append((reg, body(decided_by, reg=reg)))
    return out


def test_a_period_on_the_virtual_certifier(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "decided_in_the_clear", lambda run: decided(run, "clear"))
    monkeypatch.setenv("ALLP_CERTIFIER_PROVE", "0")
    monkeypatch.delenv("ALLP_CERTIFIER_RUN", raising=False)
    d, loop = window(tmp_path)
    reg = json.loads((d / "registration.json").read_text())
    (d / "period.json").write_text(json.dumps({"period": 0, "tools": {}}))
    (d / "answers").mkdir()
    integrity = NP._accepted(bytes.fromhex(R.digest(reg)), reg["program"], reg["partition"], loop.units, 2, "1/2", 100)
    (d / "answers" / "sampled-proofs.json").write_text(json.dumps(integrity))
    out = tmp_path / "out"
    out.mkdir()

    a = L.leg(d, "test-node", out, check=NP.reference)
    assert a["protocol"] == L.PROTOCOL and a["verdict"] == "undecided"
    assert a["reason"] == "window 0: service: decided by clear, not by a proof"
    detail = json.loads((out / "certifier.detail.json").read_text())
    assert [w["decided_by"] for w in detail["windows"]] == ["clear", "clear"]
    assert detail["loop"] == {"requests": 2, "steps": 1, "bytes": 2 * 256}
    assert sorted(f.name for f in (out / "link").iterdir()) == sorted(L.PUBLIC_LINK)
    assert json.loads((out / "certifier.month.json").read_text())["period_verdict"] == "undecided"
    assert json.loads((out / "certifier.case.json").read_text())["params"]["windows"] == 2

    run = d.parent / "p0-certifier" / "link" / "honest"
    windows = decided(run, "proof")
    case = L.case_of(run, d, windows, [L.window_answer(r, b) for r, b in windows])
    got = NP.reference(case)
    assert got["verdict"] == "accept", got["reason"]
    pub, params = got["public"], case["params"]
    assert params["audits"] == [{"free": loop.units * 2112, "steps": 2 * 60}] and params["windows"] == 2
    (t,) = pub["audits"]
    assert t["free"] == loop.units * 2112 and t["faulty_bits"] == 16 * 100 and t["wrong_units"] == 100
    assert t["faulty_sets"] == NP.clog2(NP.at_most(loop.units, 100))
    assert pub["schedules"] == NP.clog2(16 ** 2) and pub["cells"] == NP.clog2(params["cells"][0] + 1)
    assert pub["K"] == pub["schedules"] + pub["cells"] + t["free"] + t["faulty_sets"] + t["faulty_bits"] + t["jitter"]

    evidence = json.loads(bytes.fromhex(case["evidence"]))
    sig = evidence["windows"][0]["certificates"][0]["signature"]
    evidence["windows"][0]["certificates"][0]["signature"] = ("0" if sig[0] != "0" else "1") + sig[1:]
    forged = {**case, "evidence": NP.D.canonical(evidence).encode().hex()}
    assert NP.reference(forged)["verdict"] == "reject"
    assert NP.reference(forged)["reason"] == "window 0: certificates"
