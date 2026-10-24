"""The chain's stages (`chain.py`) on fakes: the advice read in each of its forms (I1 v0's JSON and bytes, and a prove-path
stand-in's from before v0, upgraded with each kept output released in its step's bucket); its bits per link-window, v0's
summing to I1's total and the residual's giving back every step from the config alone, and with the certified I/O the
whole advice (bound), or nothing when the I/O certifies other requests; a prove path's output adopted, held to the given
advice, with a forged arm that passes only when Lean refused its unit; and the report on certifier-serve's virtual runs with
every stage (NEG's registry among them), then on a varied rehearsal's advice profile, where N2 gives no verdict and the
rehearsal's own check is the verdict."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("network_e2e", HERE / "e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2e = _load()
chain, I1 = e2e.chain, e2e.chain.I1
T = 4
CONFIG = {"block_size": 16, "num_gpu_blocks": 1000, "max_num_batched_tokens": 64, "max_model_len": 128,
          "eos_token_id": 9, "stop_token_ids": []}


def _advice():
    """q0 (4 prompt tokens, 2 outputs, length) and q1 (3, 1, eos): both prefilled whole in step 0, then q0 decodes
    once; their three outputs released in windows 0, 0 and 1 of T buckets."""
    return I1.Advice((I1.RequestAdvice("q0", 4, 2, "length"), I1.RequestAdvice("q1", 3, 1, "eos")),
                     (I1.StepAdvice((I1.Row(0, "prefill", 0, 4), I1.Row(1, "prefill", 0, 3))),
                      I1.StepAdvice((I1.Row(0, "decode", 4, 1),))), (1, 1, T + 2))


def _io(users=("q0", "q1")) -> list[dict]:
    """The certified I/O of `_advice`'s requests as a varied run's check writes it (``io.json``), in ingress order."""
    return [{"user": users[0], "prompt": [1, 2, 3, 4], "max_tokens": 2, "output": [5, 6]},
            {"user": users[1], "prompt": [1, 2, 3], "max_tokens": 4, "output": [9]}]


def _standin() -> dict:
    """A prove-path stand-in's advice as PROVE-PATH's first runs wrote it: no schema, no sources, no release."""
    return {"requests": [{"id": "r0", "prompt_len": 4, "output_len": 3, "finish": "length"}], "release": [],
            "steps": [{"rows": [{"request": 0, "kind": "prefill", "start": 0, "n": 4, "cache_hit": 0,
                                 "preempted": False}]}]
            + [{"rows": [{"request": 0, "kind": "decode", "start": 4 + j, "n": 1, "cache_hit": 0, "preempted": False}]}
               for j in range(2)]}


def test_the_advice_is_read_as_v0_json_as_its_bytes_and_as_an_upgraded_standin(tmp_path):
    adv = _advice()
    (tmp_path / "a.json").write_text(json.dumps(I1.to_json(adv)))
    (tmp_path / "a.bin").write_bytes(I1.encode(adv))
    (tmp_path / "s.json").write_text(json.dumps(_standin()))
    for name, form in (("a.json", "I1 v0 JSON"), ("a.bin", "I1 v0 bytes")):
        got, info = chain.advice(tmp_path / name)
        assert got == adv and info["form"] == form and info["bits"] == I1.total_bits(adv), name
        assert info["v0_sha512"] == hashlib.sha512(I1.encode(adv)).hexdigest()
    got, info = chain.advice(tmp_path / "s.json")
    assert got.release == (0, 1, 2) and all(r.sources == () for st in got.steps for r in st.rows)
    assert "upgraded" in info["form"] and info["file_sha512"] == hashlib.sha512((tmp_path / "s.json").read_bytes()).hexdigest()
    chunked = _standin()
    chunked["steps"][0]["rows"][0]["n"] = 2
    chunked["steps"].insert(1, {"rows": [{"request": 0, "kind": "prefill", "start": 2, "n": 2, "cache_hit": 0,
                                          "preempted": False}]})
    (tmp_path / "c.json").write_text(json.dumps(chunked))
    with pytest.raises(I1.MalformedAdvice, match="release buckets"):
        chain.advice(tmp_path / "c.json")


def test_the_bits_per_window_are_v0s_and_the_residuals_and_the_certified_io_binds_and_rebuilds_the_advice():
    adv = _advice()
    unbound = chain.bits(adv, T, 3, CONFIG, None)
    assert unbound["bound"] is None and unbound["public_fields"] is None and unbound["rebuilt"] is None
    assert "the certified tokens" in unbound["why"] and unbound["windows"][0]["derived_bits"] is None
    b = chain.bits(adv, T, 3, CONFIG, _io())
    rows = {x["window"]: x for x in b["windows"]}
    assert sorted(rows) == [0, 1, 2] and sum(x["v0_bits"] for x in rows.values()) == I1.total_bits(adv) == b["v0_total"]
    assert {w: x["v0_bits"] for w, x in rows.items() if x["v0_bits"]} == I1.bits_by_window(adv, T)
    assert b["bound"] is True and b["rebuilt"] is True and b["public_fields"] is True and b["why"] is None
    assert 0 < rows[0]["derived_bits"] < rows[0]["v0_bits"] and rows[0]["derived_shaped_bits"] <= rows[0]["derived_bits"]
    assert rows[0]["forced_bits"] <= rows[0]["derived_bits"] and rows[2]["v0_bits"] == 0
    swapped = chain.bits(adv, T, 3, CONFIG, _io()[::-1])
    assert swapped["bound"] is True and swapped["rebuilt"] is True
    other = chain.bits(adv, T, 3, CONFIG, _io(("q0", "q9")))
    assert other["bound"] is False and other["rebuilt"] is None and "aren't the 2 certified ones" in other["bound_why"]
    wrong = _io()
    wrong[0]["output"] = [5, 6, 7]
    assert chain.bits(adv, T, 3, CONFIG, wrong)["rebuilt"] is False
    none = chain.bits(adv, T, 1, None, None)
    assert none["windows"][0]["derived_bits"] is None and "--serving-config" in none["why"]
    assert [x["in_run"] for x in none["windows"]] == [True, False]


def _arm(d: Path, name: str, advice: dict, *, ok=True, forged=False, negatives=None) -> None:
    rank = {"ok": ok, "audit": {"accepted": not forged, "wrong_units": None if forged else {"bound": 3}},
            "population": 10, "draw": [1, 2], "seconds": 1.5, "source": "stand-in",
            "negatives": negatives or {"a": True, "b": {"refused": True}}}
    if forged:
        rank["forged"] = {"ok": ok}
    (d / name).mkdir(parents=True)
    (d / name / "summary.json").write_text(json.dumps({"ok": ok, "forged": forged, "world": 1, "advice": advice,
                                                       "ranks": {"0": rank}}))


def test_a_prove_paths_output_is_adopted_and_held_to_the_given_advice(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps(_standin()))
    _, info = chain.advice(tmp_path / "s.json")
    d = tmp_path / "pp"
    _arm(d, "glm", {"digest": info["file_sha512"], "bits": 8 * 100})
    _arm(d, "glm-forged", {"digest": info["file_sha512"]}, forged=True)
    (d / "advice").mkdir()
    got = chain.prove_path(d, info)
    assert got["ok"] is True and got["committed_is_the_advice_file"] is True and sorted(got["arms"]) == ["glm", "glm-forged"]
    assert got["bound"] is False and got["bound_why"] == "the --advice file isn't these runs' advice"
    assert chain.prove_path(d, info, True)["bound_why"].startswith("the Commit it proves isn't held to these runs'")
    assert got["arms"]["glm"]["ranks"]["0"] == {"ok": True, "accepted": True, "population": 10, "drawn": 2,
                                                "wrong_units_bound": 3, "negatives_refused": True,
                                                "forged_refused": None, "seconds": 1.5, "source": "stand-in"}
    assert got["arms"]["glm-forged"]["ranks"]["0"]["forged_refused"] is True
    _, other = chain.advice(tmp_path / "s.json")
    assert chain.prove_path(d, other | {"file_sha512": "0"})["ok"] is False
    v0 = tmp_path / "v0"
    _arm(v0, "honest", {"root": "r", "bits": info["bits"]}, negatives={"a": {"refused": False}})
    assert chain.prove_path(v0, info)["committed_is_the_advice_file"] is True and chain.prove_path(v0, info)["ok"] is True
    assert chain.prove_path(v0, info)["arms"]["honest"]["ranks"]["0"]["negatives_refused"] is False
    assert chain.prove_path(v0, info)["arms"]["honest"]["advice_held_by"].startswith("its bits only")
    assert got["arms"]["glm"]["advice_held_by"] == "the file's SHA-512"
    shutil.rmtree(d / "glm")
    assert chain.prove_path(d, info)["ok"] is False
    with pytest.raises(SystemExit, match="no prove-path arm"):
        chain.prove_path(tmp_path, info)


def _args(src, out, **kw) -> SimpleNamespace:
    return SimpleNamespace(**{"sources": [str(src)], "upstream": None, "vllm": None, "out": str(out), "work": None,
                              "bin": None, "lean": None, "delta": 2.0 ** -20, "no_prove": True, "proofs": None,
                              "advice": None, "serving_config": None, "prove_path": None, "no_negatives": False, **kw})


@pytest.fixture(scope="module")
def virtual(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    e2e.n2_zk.virtual(runs, requests=8, tokens=4, strategies=("honest", "late", "extra"))
    return runs


def test_the_report_carries_every_stage_on_the_planned_profile(virtual, tmp_path):
    (tmp_path / "s.json").write_text(json.dumps(_standin()))
    (tmp_path / "config.json").write_text(json.dumps({"config": CONFIG}))
    _, info = chain.advice(tmp_path / "s.json")
    pp = tmp_path / "pp" / "out"
    _arm(pp, "glm", {"digest": info["file_sha512"]})
    _arm(pp, "glm-forged", {"digest": info["file_sha512"]}, forged=True)
    with tarfile.open(tmp_path / "pp.tgz", "w:gz") as tar:
        tar.add(pp, arcname="out")
    out = tmp_path / "out"
    r = e2e.run(_args(virtual, out, advice=str(tmp_path / "s.json"), serving_config=str(tmp_path / "config.json"),
                      prove_path=str(tmp_path / "pp.tgz")), [], clock=lambda: 0.0)
    assert r["format"] == e2e.FORMAT and r["claim"] == e2e.CLAIM
    assert r["ok"] is False and r["why_not"] == ["the advice isn't these runs'",
                                                 "the advice wasn't rebuilt from the certified I/O"]
    c = r["chain"]
    assert c["advice"]["form"].startswith("a prove-path stand-in's") and c["bits"]["public_fields"] is None
    assert c["bits"]["bound"] is None and c["prove_path"]["dir"] == str(tmp_path / "pp.tgz") and c["prove_path"]["ok"]
    rows = c["negatives"]["rows"]
    assert c["negatives"]["ok"] and all(rows[k]["ok"] is True for k in ("late", "extra", "head"))
    assert rows["advice-length"]["ran"] is False and rows["wrong-token"]["ran"] is False
    md = (out / "report.md").read_text()
    for section in ("## Advice", "## Advice bits per link-window", "## Prove path", "## Negatives (NEG's registry)",
                    "## What's still stubbed"):
        assert section in md, section
    assert "| glm-forged | True | 0 | False | 10 | 2 | — | True | True | 1.5 | stand-in |" in md
    assert any(x.startswith("Advice form: the stand-in's pre-v0 JSON") for x in r["stubbed"])
    assert any(x.startswith("Auditor host") for x in r["stubbed"])
    summary = pp / "glm" / "summary.json"
    summary.write_text(json.dumps(json.loads(summary.read_text()) | {"ok": False}))
    r = e2e.run(_args(virtual, tmp_path / "out2", advice=str(tmp_path / "s.json"), prove_path=str(pp)), [],
                clock=lambda: 0.0)
    assert r["chain"]["prove_path"]["ok"] is False and r["chain"]["bits"]["config"] is None
    assert "the prove path failed" not in r["why_not"]


#: what the check finds of each negative on the advice profile: late's window faulted (the audit's status), extra's and
#: drop's response breaking the certified I/O's finish rule
OWN = {"late": {"audit": "status:coverage-never-established"}, "extra": {"io": "longer than max_tokens"},
       "drop": {"io": "short of max_tokens without EOS"}}


def _varied(d: Path, virtual: Path, verdicts: dict, found: dict | None = None) -> Path:
    """A varied rehearsal's runs as certifier-serve's check leaves them on the advice profile: its plan, the
    calibration (the virtual run's), the check's verdict with its certified I/O and what it found of a refused run
    (``found``, by default `OWN`'s), the shim's request count, and honest's ``io.json``."""
    found = OWN | (found or {})
    for s, v in verdicts.items():
        (d / s).mkdir(parents=True)
        (d / s / "certified.jsonl").write_text("")
        (d / s / "server.json").write_text(json.dumps({"requests": 2}))
        shutil.copy(virtual / "honest" / "calibration.json", d / s / "calibration.json")
        (d / s / "plan.json").write_text(json.dumps({"windows": 3, "occupancy": {"kind": "advice"}, "live_model": "m",
                                                     "varied": {"requests": 2, "max_tokens": 4}}))
        f = found.get(s, {}) if v == "refused" else {}
        (d / s / "check.json").write_text(json.dumps({
            "verdict": v, "network_verdict": {"accepted": None}, "audit_accepted": "audit" not in f,
            "salt_accepted": True, "certified_accepted": True,
            "rows": [{"window": 0, "audit": {"egress": {"raw": f.get("audit"), "committed": f.get("audit")}}}],
            "io": {"accepted": "io" not in f, "requests": 2, "tokens": 3, "finish": {"eos": 1, "length": 1},
                   "violations": [[1, f["io"]]] if "io" in f else []}}))
    (d / "honest" / "io.json").write_text(json.dumps({"windows": 3, "requests": _io()}))
    return d


def test_on_the_advice_profile_the_rehearsals_check_is_the_verdict_and_the_advice_is_bound_by_the_certified_io(
        virtual, tmp_path):
    runs = _varied(tmp_path / "varied", virtual, {"honest": "accepted", "late": "refused", "extra": "refused",
                                                  "drop": "refused", "token": "accepted"})
    (tmp_path / "a.json").write_text(json.dumps(I1.to_json(_advice())))
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    args = _args(runs, tmp_path / "out", no_prove=False, advice=str(tmp_path / "a.json"),
                 serving_config=str(tmp_path / "config.json"))
    r = e2e.run(args, [], clock=lambda: 0.0)
    assert r["ok"] is True and r["profile"] == "advice" and r["claim"] == e2e.CLAIM_ADVICE
    assert list(r["runs"]) == ["honest", "late", "extra", "drop", "token"] and r["link_windows"] == []
    assert r["pending"] == {"token": {"waits": "IO-TIE", "check": "accepted", "n2": None, "ran": True}}
    assert r["runs"]["honest"]["network"]["accepted"] is None and r["proofs"]["ran"] is False
    assert "N2-ADVICE" in r["proofs"]["why"] and r["chain"]["negatives"]["ran"] is False
    assert sorted(r["negatives"]) == ["drop", "extra", "late"]
    assert r["negatives"]["late"]["by"] == ["the rehearsal's own check (audit.status:coverage-never-established)"]
    assert r["negatives"]["drop"]["by"] == ["the rehearsal's own check (io:short of max_tokens without EOS)"]
    assert r["ran"] == ["the rehearsal's own check", "the advice, these runs' and rebuilt from their certified I/O"]
    b = r["chain"]["bits"]
    assert b["bound"] is True and b["rebuilt"] is True and [x["window"] for x in b["windows"]] == [0, 1, 2]
    md = (tmp_path / "out" / "report.md").read_text()
    assert "| honest | accepted (I/O True) | — | — |" in md and "No N2 verdict on the advice profile" in md
    assert "- **token** (NEG's, pending IO-TIE, which alone refuses it here): the rehearsal's check accepted, N2 —" in md
    assert any(x.startswith("N2 on the advice profile") for x in r["stubbed"])
    at = e2e.where({"honest": {"dir": runs / "honest", "gate": None}}, "vy-nebius-1:18080")
    assert at["where"].startswith("off the prover's node: better than the loopback, though not yet the auditor host")
    assert any(x.startswith("Auditor host: the certifier ran on another node") for x in chain.stubs(at, True))
    (tmp_path / "peers").write_text("# machine: ssh\nvy-nebius-2 -i k research@10.0.0.2\n")
    with pytest.raises(SystemExit, match="no vy-nebius-1 in"):
        e2e.forward("vy-nebius-1:18080", tmp_path / "forward.log", tmp_path / "peers")
    assert not any(x.startswith("Advice: ") for x in r["stubbed"])
    _varied(tmp_path / "bad", virtual, {"honest": "accepted", "late": "accepted", "extra": "refused"})
    r = e2e.run(_args(tmp_path / "bad", tmp_path / "out2"), [], clock=lambda: 0.0)
    assert r["ok"] is False and r["negatives"]["late"]["refused"] is False
    idle = _varied(tmp_path / "idle", virtual, {"honest": "accepted", "late": "refused", "extra": "refused",
                                               "drop": "refused"})
    for s in ("extra", "drop"):
        (idle / s / "server.json").write_text(json.dumps({"requests": 0}))
    r = e2e.run(_args(idle, tmp_path / "out3"), [], clock=lambda: 0.0)
    idle_negatives = [r["negatives"][s] for s in ("extra", "drop")]
    assert r["ok"] is False and not any(n["refused"] or n["ran"] for n in idle_negatives)
    assert r["negatives"]["late"]["refused"] is True and r["negatives"]["late"]["ran"] is True
    assert "- **drop**: NOT refused: not run, its shim received no request" in (tmp_path / "out3" / "report.md").read_text()
