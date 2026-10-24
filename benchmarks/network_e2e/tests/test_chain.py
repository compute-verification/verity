"""The chain's stages (`chain.py`) on fakes: the advice read in each of its forms (I1 v0's JSON and bytes, and a prove-path
stand-in's from before v0, upgraded with each kept output released in its step's bucket); its bits per link-window, v0's
summing to I1's total and the residual's giving back every step from the config alone, and with the certified I/O the
whole advice (bound), or nothing when the I/O certifies other requests; a prove path's output adopted, held to the given
advice, with a forged arm that passes only when Lean refused its unit; and the report on certifier-serve's virtual runs with
every stage (NEG's registry among them, with the I/O tie's verdict given the sampled ids), then on a varied rehearsal's
advice profile, where N2 gives no verdict and the rehearsal's own check is the verdict, and the I/O tie on each run
decides token and holds the honest run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
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
                                                "forged_refused": None, "seconds": 1.5, "source": "stand-in",
                                                "per_unit": {}}
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


def test_the_prove_paths_time_is_per_drawn_unit_and_beside_the_endpoints_advice_it_is_the_fixed_workloads(tmp_path):
    rank = {"draw_counts": {"S": 2}, "timing": {"S": {"lean_verify_s": 47.0, "timing": {
        "prove_total_s": 428.9, "units": 16384, "proof_bytes": [830330, 830330]}}}}
    assert chain.per_unit(rank) == {"S": {"units": 2, "proof_bytes": 1660660, "prove_s_per_unit": 214.45,
                                          "lean_verify_s_per_unit": 23.5}}
    _arm(tmp_path / "pp", "llama", {"root": "r", "bits": 2695})
    _arm(tmp_path / "pp", "llama-forged", {"root": "r", "bits": 2695}, forged=True)
    got = chain.prove_path(tmp_path / "pp", {"served": "r-ep", "bits": 100, "file_sha512": None, "v0_sha512": None},
                           True)
    assert got["ok"] is True and got["committed_is_the_advice_file"] is None and got["bound"] is False
    assert got["bound_why"].startswith("from the fixed workload, not this run")


EP = HERE / "tests" / "data" / "endpoint"


def _endpoint(tmp_path: Path) -> Path:
    shutil.copytree(EP, tmp_path / "ep")
    return tmp_path / "ep"


def test_the_endpoints_own_advice_is_the_runs_by_its_commit_windows_each_opened_and_rebuilt(tmp_path):
    """vllm's own-advice endpoint's first three Commit windows (r20261024-094558-6d6a's records, trimmed: its scheduler
    trace, its windows.jsonl and `verity-vllm advice`'s registrations and openings, as the endpoint committed them
    mid-serve) and the certified I/O of their nine requests: bound and rebuilt, each window opening its registered root;
    not bound when a window falls outside the run's link-windows, holds a request the run didn't certify, opens another
    root or names another Commit, or has no advice; not rebuilt when a certified output breaks the stop rule."""
    io = json.loads((EP / "io.json").read_text())["requests"]
    first = chain._first_steps(chain._jsonl(EP / "advice_trace.jsonl"))
    dep = {"origin_unix": first[0] - 0.05, "windows": 2, "window_s": 6.0}
    info, b = chain.served_advice(EP, EP / "advice", dep, io, 60, [128001])
    assert b["bound"] is True and b["rebuilt"] is True and b["public_fields"] is True and b["why"] is None, b["bound_why"]
    regs = [json.loads((EP / f"advice/w{w}/registration.json").read_text()) for w in range(3)]
    assert [x["window"] for x in b["commit_windows"]] == [0, 1, 2] and all(x["opened"] for x in b["commit_windows"])
    assert [x["commit"] for x in b["commit_windows"]] == [r["commit"] for r in regs]
    assert info["requests"] == len(io) == 9 and info["served"] == b["served"] == "r20261024-094558-6d6a"
    assert info["bits"] == b["v0_total"] == sum(r["binding"]["advice"]["bits"] for r in regs)
    assert sum(x["v0_bits"] for x in b["windows"]) == b["v0_total"] and [x["in_run"] for x in b["windows"]] == [True, True]
    assert all(0 < x["derived_bits"] < x["v0_bits"] for x in b["commit_windows"])
    assert b["config"] | {"num_gpu_blocks": 0} == {
        "block_size": 16, "num_gpu_blocks": 0, "max_num_batched_tokens": 2048, "max_model_len": 512,
        "eos_token_id": 128001, "stop_token_ids": [], "scheduler": "async", "long_prefill_token_threshold": 0,
        "max_num_seqs": 8, "enable_prefix_caching": False, "row_order": "batch"}
    late = chain.served_advice(EP, EP / "advice", dep | {"origin_unix": first[0] + 1.2}, io, 60, [128001])[1]
    assert late["bound"] is False and [x["window"] for x in late["commit_windows"]] == [1, 2]
    assert "1 certified requests are in none of the run's Commit windows" in late["bound_why"]
    other = chain.served_advice(EP, EP / "advice", dep, io[1:], 60, [128001])[1]
    assert other["bound"] is False and other["rebuilt"] is False
    assert "Commit window 0 holds 1 requests that aren't the run's certified ones" in other["bound_why"]
    broken = [dict(r) for r in io]
    broken[1]["output"] = [128001, *broken[1]["output"][1:]]
    eos = chain.served_advice(EP, EP / "advice", dep, broken, 60, [128001])[1]
    assert eos["bound"] is True and eos["rebuilt"] is False and "break the stop rule" in eos["why"]
    ep = _endpoint(tmp_path)
    opening = json.loads((ep / "advice/w1/private/opening.json").read_text())
    opening["salts"][0] = "00" * (len(opening["salts"][0]) // 2)
    (ep / "advice/w1/private/opening.json").write_text(json.dumps(opening))
    lines = (ep / "windows.jsonl").read_text().splitlines()
    (ep / "windows.jsonl").write_text("\n".join(lines[:2] + [lines[2].replace(regs[2]["commit"]["run_root"], "0" * 64)]))
    bad = chain.served_advice(ep, ep / "advice", dep, io, 60, [128001])[1]
    assert bad["bound"] is False and [x["opened"] for x in bad["commit_windows"]] == [True, False, True]
    assert "Commit window 1's opening doesn't open its registered root" in bad["bound_why"]
    assert "Commit window 2's registration names another Commit" in bad["bound_why"]
    index = json.loads((ep / "advice/windows.json").read_text())
    del index["windows"]["0"]
    (ep / "advice/windows.json").write_text(json.dumps(index | {"refused": {"0": "a request it holds only part of"}}))
    gone = chain.served_advice(ep, ep / "advice", dep, io, 60, [128001])[1]
    assert "Commit window 0 has no advice (a request it holds only part of)" in gone["bound_why"]
    assert gone["commit_windows"][0] == {"window": 0, "refused": "a request it holds only part of"}


def _args(src, out, **kw) -> SimpleNamespace:
    return SimpleNamespace(**{"sources": [str(src)], "upstream": None, "vllm": None, "out": str(out), "work": None,
                              "bin": None, "lean": None, "delta": 2.0 ** -20, "no_prove": True, "proofs": None,
                              "advice": None, "serving_config": None, "prove_path": None, "ids": None,
                              "no_negatives": False, **kw})


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
    assert "| glm-forged | True | 0 | False | 10 | 2 | — | True | True | 1.5 | — | — | — | stand-in |" in md
    assert any(x.startswith("Advice form: the stand-in's pre-v0 JSON") for x in r["stubbed"])
    assert any(x.startswith("Auditor host") for x in r["stubbed"])
    summary = pp / "glm" / "summary.json"
    summary.write_text(json.dumps(json.loads(summary.read_text()) | {"ok": False}))
    r = e2e.run(_args(virtual, tmp_path / "out2", advice=str(tmp_path / "s.json"), prove_path=str(pp)), [],
                clock=lambda: 0.0)
    assert r["chain"]["prove_path"]["ok"] is False and r["chain"]["bits"]["config"] is None
    assert "the prove path failed" not in r["why_not"]


def test_given_the_sampled_ids_each_negative_the_tie_reads_carries_its_verdict(virtual, tmp_path):
    """With --advice and --ids, every negative the I/O tie reads carries the tie's verdict, in the report's table and
    in `stubbed`; a row the tie ran on passes only when the tie refuses it by its part. Without --ids no row has one,
    and `stubbed` says so. Until IO-TIE lands, each such row says why it didn't run."""
    NG = chain.NG
    (tmp_path / "a.json").write_text(json.dumps(I1.to_json(_advice())))
    (tmp_path / "ids.jsonl").write_text("".join(json.dumps({"user": u, "output_ids": x}) + "\n"
                                                for u, x in (("q0", [5, 6]), ("q1", [9]))))
    r = e2e.run(_args(virtual, tmp_path / "out", advice=str(tmp_path / "a.json"), ids=str(tmp_path / "ids.jsonl")), [],
                clock=lambda: 0.0)
    rows = r["chain"]["negatives"]["rows"]
    tied = {k: x["tie"] for k, x in rows.items() if "tie" in x}
    assert set(tied) == set(NG.TIE_PARTS)
    md = (tmp_path / "out" / "report.md").read_text()
    assert "| I/O tie |" in md
    if NG.io_tie_module() is None:
        assert all(not t["ran"] and "IO-TIE" in t["why"] for t in tied.values())
        assert any(x.startswith("I/O tie: no I/O tie yet") for x in r["stubbed"])
    else:
        ran = {k: t for k, t in tied.items() if t["ran"]}
        assert {"frame", "token", "drop", "wrong-token"} <= set(ran) and all(t["ok"] for t in ran.values()), tied
        assert all(rows[k]["ok"] is True for k in ran)
        assert f"`{NG.TIE}.content`" in md and any(x.startswith("I/O tie: IO-TIE's reference") for x in r["stubbed"])
    r = e2e.run(_args(virtual, tmp_path / "out2", advice=str(tmp_path / "a.json")), [], clock=lambda: 0.0)
    assert all("tie" not in x for x in r["chain"]["negatives"]["rows"].values())
    assert any(x.startswith("I/O tie: no sampled ids given") for x in r["stubbed"])


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
    assert r["chain"]["tie"]["ran"] is False and "token" not in r["negatives"]
    assert r["pending"] == {"token": {"waits": "IO-TIE", "check": "accepted", "n2": None, "ran": True,
                                      "why": r["chain"]["tie"]["why"]}}
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
    assert "--endpoint-run" in next(x for x in r["stubbed"] if x.startswith("Advice: none given"))
    assert r["ok"] is False and not any(n["refused"] or n["ran"] for n in idle_negatives)
    assert r["negatives"]["late"]["refused"] is True and r["negatives"]["late"]["ran"] is True
    assert "- **drop**: NOT refused: not run, its shim received no request" in (tmp_path / "out3" / "report.md").read_text()


def test_on_the_advice_profile_the_endpoints_own_advice_for_the_runs_commit_windows_is_the_reports(
        virtual, tmp_path, monkeypatch):
    runs = _varied(tmp_path / "varied", virtual, {"honest": "accepted", "late": "refused", "extra": "refused",
                                                  "drop": "refused"})
    first = chain._first_steps(chain._jsonl(EP / "advice_trace.jsonl"))
    honest = runs / "honest"
    (honest / "io.json").write_text((EP / "io.json").read_text())
    (honest / "deployment.json").write_text(json.dumps({"origin_unix": first[0] - 0.05, "windows": 3, "window_s": 6.0}))
    plan = json.loads((honest / "plan.json").read_text())
    plan["varied"]["eos_token_ids"] = [128001]
    (honest / "plan.json").write_text(json.dumps(plan))
    out = tmp_path / "out"
    shutil.copytree(EP / "advice", out / "endpoint" / "advice")
    r = e2e.run(_args(runs, out, endpoint_run=str(EP)), [], clock=lambda: 0.0)
    assert r["ok"] is True, r["why_not"]
    assert "the advice, these runs' and rebuilt from their certified I/O" in r["ran"]
    assert sorted(p.name for p in (out / "endpoint").iterdir()) == sorted(["advice", *chain.ENDPOINT])
    b = r["chain"]["bits"]
    assert b["bound"] is True and b["rebuilt"] is True and len(b["commit_windows"]) == 3
    md = (out / "report.md").read_text()
    assert "The endpoint's own (r20261024-094558-6d6a)" in md and "| 1 | r20261024-094558-6d6a/w1 |" in md
    assert any(x.startswith("Derived bits: the serving config is read off the endpoint's records") for x in r["stubbed"])
    monkeypatch.setattr(e2e, "ADVICE", [sys.executable, "-c", "import sys; print('unknown command'); sys.exit(2)"])
    r = e2e.run(_args(runs, tmp_path / "out2", endpoint_run=str(EP)), [], clock=lambda: 0.0)
    assert r["ok"] is False and "the advice isn't these runs'" in r["why_not"]
    assert r["chain"]["bits"]["bound_why"].startswith("verity-vllm advice made no index (exit 2: unknown command)")
    assert (tmp_path / "out2" / "endpoint" / "advice.log").read_text() == "unknown command\n"
    r = e2e.run(_args(runs, tmp_path / "out3", endpoint_run=str(tmp_path / "nowhere")), [], clock=lambda: 0.0)
    assert r["ok"] is False and "not copied" in r["chain"]["bits"]["bound_why"]


def test_each_runs_sampled_ids_are_the_endpoints_for_the_requests_that_arrived_in_its_link_windows(tmp_path):
    """Every strategy of a rehearsal sends the same users, so the I/O tie takes a run's sampled ids from the endpoint's
    records for the requests that arrived in its own link-windows: each `arrive` of the scheduler trace in the run's
    span (from `SLACK_S` before its origin), matched to requests.jsonl by engine id. A request with no user is no
    run's, and a user arriving twice in one span is refused."""
    arrive = {r["request"]: r["t"] for r in chain._jsonl(EP / "advice_trace.jsonl") if r["kind"] == "arrive"}
    rows = {r["user"]: r for r in chain._jsonl(EP / "requests.jsonl")}
    t0, last = arrive[rows["drive-0"]["engine_id"]], max(arrive.values())
    first = chain.served_ids(EP, {"origin_unix": t0 - 0.2, "windows": 1, "window_s": 0.4})
    assert sorted(first) == ["drive-0", "drive-1", "drive-2", "drive-3"] and min(arrive.values()) < t0 - 0.2
    assert all(first[u] == rows[u]["streamed_ids"] and len(first[u]) == 16 for u in first)
    every = chain.served_ids(EP, {"origin_unix": t0, "windows": 2, "window_s": 6.0})
    assert sorted(every) == [f"drive-{i}" for i in range(8)]
    assert chain.served_ids(EP, {"origin_unix": last + chain.SLACK_S + 0.5, "windows": 1, "window_s": 6.0}) == {}
    ep = _endpoint(tmp_path)
    lines = (ep / "requests.jsonl").read_text().splitlines()
    (ep / "requests.jsonl").write_text("\n".join(x.replace('"drive-1"', '"drive-0"') for x in lines))
    with pytest.raises(ValueError, match="drive-0 reached the endpoint twice"):
        chain.served_ids(ep, {"origin_unix": t0, "windows": 2, "window_s": 6.0})


TOKEN = HERE / "tests" / "data" / "token"


def test_a_token_runs_changed_ids_fall_in_the_link_windows_its_certified_egress_carries_them(tmp_path):
    """r20261024-111424-88c9's token run (its ingress and egress records, plan, calibration and the shim's server.json,
    not its certified heads): the shim changed varied.00000's first three outputs and varied.00001's first twenty, which
    its certified egress carries in link-windows 0 and 1, each output placed as the tie places it (its stream's next id
    event). An output the egress doesn't carry falls in none, and a run whose shim logged no change has none."""
    server = json.loads((TOKEN / "server.json").read_text())
    assert sorted({u for u, _ in server["tampered"]}) == ["varied.00000", "varied.00001"]
    assert len(server["tampered"]) == 23 and chain.changed_windows(TOKEN) == [0, 1]
    run = tmp_path / "token"
    shutil.copytree(TOKEN, run)
    for changed, windows in (([["varied.00000", 0]], [0]), ([["varied.00001", 19]], [1]),
                             ([["varied.00000", 999], ["nobody", 0]], [])):
        (run / "server.json").write_text(json.dumps(server | {"tampered": changed}))
        assert chain.changed_windows(run) == windows, changed
    (run / "server.json").write_text(json.dumps({k: v for k, v in server.items() if k != "tampered"}))
    assert chain.changed_windows(run) is None


def _ties(**edits) -> dict:
    """`negatives.tie_run`'s row on each run of `_varied`, as the tie gives it on 88c9's: honest accepted, token refused
    by content at link-window 1, drop by content for the output it left out, late and extra accepted; ``edits`` by run."""
    rows = {"honest": {}, "late": {}, "extra": {}, "token": {"failing": {1: ["content"]}},
            "drop": {"failing": {-1: ["content"]}, "unplaced": 1}}
    out = {}
    for name, x in rows.items():
        x = x | edits.get(name, {})
        codes = sorted({f"IoTieWindow.{p}" for parts in x.get("failing", {}).values() for p in parts})
        out[name] = {"verdict": "refused" if codes else "accepted", "codes": codes, "failing": {}, "unplaced": 0,
                     "held": {"egress": 3, "ingress": 3}, "missing": {}, "outputs": 3, "placed": 3} | x
    return out


def test_on_the_advice_profile_the_ties_verdict_refuses_token_where_it_changed_ids_and_holds_the_honest_run(
        virtual, tmp_path, monkeypatch):
    """With IO-TIE in the tree and each request's sampled ids (here --ids), the I/O tie runs on every run of the
    advice profile: token is a negative it alone refuses, by content at exactly the link-windows its changed ids fall
    in (refused elsewhere, it isn't); drop's refusal by content is listed beside its own code; the honest run must be
    accepted for the report to be OK. Without IO-TIE, token stays pending and says why. `negatives.tie_run` is stubbed:
    NEG's suite runs it on certifier-serve's varied runs."""
    NG = chain.NG
    runs = _varied(tmp_path / "varied", virtual, {"honest": "accepted", "late": "refused", "extra": "refused",
                                                  "drop": "refused", "token": "accepted"})
    (tmp_path / "ids.jsonl").write_text("".join(json.dumps({"user": u, "output_ids": [5]}) + "\n" for u in ("q0", "q1")))
    ties, seen = _ties(), []
    monkeypatch.setattr(NG, "io_tie_module", lambda: object())
    monkeypatch.setattr(NG, "tie_run", lambda run, ids: seen.append(ids) or ties[run.name])
    monkeypatch.setattr(chain, "changed_windows", lambda run: [1] if run.name == "token" else None)
    r = e2e.run(_args(runs, tmp_path / "out", ids=str(tmp_path / "ids.jsonl")), [], clock=lambda: 0.0)
    assert r["ok"] is True, r["why_not"]
    assert seen == [{"q0": [5], "q1": [5]}] * 5 and r["pending"] == {}
    assert "the I/O tie (IO-TIE's reference in the clear, on stand-ins)" in r["ran"]
    token, drop = r["negatives"]["token"], r["negatives"]["drop"]
    assert token["refused"] is True and token["by"] == [
        "the I/O tie (IoTieWindow.content at link-windows [1], the link-windows its changed ids fall in)"]
    assert drop["by"] == ["the rehearsal's own check (io:short of max_tokens without EOS)",
                          "the I/O tie (IoTieWindow.content, 1 committed outputs no certified cell carries)"]
    assert r["negatives"]["late"]["by"] == ["the rehearsal's own check (audit.status:coverage-never-established)"]
    assert r["chain"]["tie"]["runs"]["late"]["ok"] is None
    md = (tmp_path / "out" / "report.md").read_text()
    assert "## I/O tie" in md and "| token | refused | True | `IoTieWindow.content` | 1: content | [1] | 2 |" in md
    assert "| honest | accepted | True | — | — | — | 2 | 3 | 3 | 0 | 3/3 | — |" in md
    assert "- **token**: refused by the I/O tie (IoTieWindow.content at link-windows [1]" in md
    assert any(x.startswith("I/O tie: IO-TIE's reference in the clear on NEG's stand-ins") and str(tmp_path) in x
               for x in r["stubbed"])

    ties.update(_ties(token={"failing": {0: ["content"]}}))
    r = e2e.run(_args(runs, tmp_path / "out2", ids=str(tmp_path / "ids.jsonl")), [], clock=lambda: 0.0)
    assert r["ok"] is False and r["why_not"] == ["token not refused"] and r["negatives"]["token"]["by"] == []
    assert ("- **token**: NOT refused by nothing (not proved); the I/O tie refused (IoTieWindow.content at link-windows "
            "[0]), its changed ids in link-windows [1]") in (tmp_path / "out2" / "report.md").read_text()

    ties.update(_ties(honest={"failing": {2: ["port"]}}))
    r = e2e.run(_args(runs, tmp_path / "out3", ids=str(tmp_path / "ids.jsonl")), [], clock=lambda: 0.0)
    assert r["ok"] is False and r["why_not"] == ["the I/O tie didn't accept the honest run"]

    monkeypatch.setattr(NG, "io_tie_module", lambda: None)
    r = e2e.run(_args(runs, tmp_path / "out4", ids=str(tmp_path / "ids.jsonl")), [], clock=lambda: 0.0)
    assert r["ok"] is True and "token" not in r["negatives"]
    assert r["pending"]["token"]["why"] == NG.TIE_MISSING["io-tie"]
    assert "the I/O tie (IO-TIE's reference in the clear, on stand-ins)" in r["not_run"]
    md = (tmp_path / "out4" / "report.md").read_text()
    assert f"; the I/O tie not run ({NG.TIE_MISSING['io-tie']})" in md and "## I/O tie\n\nNot run: " in md
    assert any(x.startswith("I/O tie: not run (") for x in r["stubbed"])
