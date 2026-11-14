"""The chain's stages (`chain.py`) on fakes: the advice read in each of its forms (I1 v0's JSON and bytes, and a prove-path
stand-in's from before v0, upgraded with each kept output released in its step's bucket); its bits per link-window, v0's
summing to I1's total and the residual's giving back every step from the config alone, and with the certified I/O the
whole advice (bound), or nothing when the I/O certifies other requests; a prove path's output adopted, held to the given
advice, with a forged arm that passes only when Lean refused its unit; and the endpoint's own advice, bound to the
runs by its Commit windows."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(HERE))
import chain  # noqa: E402

I1 = chain.I1
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
    """A prove-path stand-in's advice as the prove path's first runs wrote it: no schema, no sources, no release."""
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


def test_the_prove_paths_time_is_per_sampled_block_and_beside_the_endpoints_advice_it_is_the_fixed_workloads(tmp_path):
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


def _reg(w: int) -> dict:
    return json.loads((EP / f"advice/w{w}/registration.json").read_text())


def _apart(d: Path, w: int, adv=None) -> str:
    """The endpoint fixture's Commit window ``w``'s advice (or ``adv``) registered apart into ``d``, as the lane
    registered 8eb1's w212: the report's scheme, compiler and Commit, under salts of its own. Returns its root."""
    reg = _reg(w)
    opening = json.loads((EP / f"advice/w{w}/private/opening.json").read_text())
    data = I1.encode(adv or I1.from_json(json.loads((EP / f"advice/w{w}/private/advice.json").read_text())))
    n, k = reg["row_bits"] // 8, len(bytes.fromhex(opening["salts"][0]))
    rows = [data[i:i + n].ljust(n, b"\0") for i in range(0, len(data), n)]
    salts = [(hashlib.sha512(b"apart %d %d" % (w, i)).digest() * 4)[:k] for i in range(len(rows))]
    binding = hashlib.sha512(chain.canonical_json({"scheme": reg["scheme"],
                                                   "compiler": reg["binding"]["compiler"]})).digest()
    root = chain.AC.commit(rows, reg["row_bits"], binding, salts).root.hex()
    reg["binding"]["advice"]["root"] = root
    (d / "private").mkdir(parents=True)
    (d / "registration.json").write_text(json.dumps(reg))
    (d / "private" / "opening.json").write_text(json.dumps(opening | {"advice": data.hex(),
                                                                      "salts": [s.hex() for s in salts]}))
    return root


def _serve_arm(d: Path, name: str, w: int, bits: int, *, forged=False, root=None, codes=(), advice_root=None,
               before_draw=True) -> None:
    """A `prove_path.py audit --serve-run` arm of the endpoint fixture's Commit window ``w``: its summary's ``serve``
    names the window, its run id and the root it found sealed while serving (``codes`` its refusals), its committed
    advice the report's own registration of ``w`` unless ``advice_root`` names another, and its rank's
    `advice_before_draw` (the verifier's receipt of the registration before any coin) ``before_draw``, absent when None."""
    c = next(x for x in chain._jsonl(EP / "windows.jsonl") if int(x["window"]) == w)
    _arm(d, name, {}, forged=forged)
    s = json.loads((d / name / "summary.json").read_text())
    adv = {"root": advice_root or _reg(w)["binding"]["advice"]["root"], "bits": bits}
    if before_draw is not None:
        s["ranks"]["0"]["advice_before_draw"] = {"ok": before_draw, "codes": [] if before_draw else ["after-the-draw"]}
    s |= {"run_root": root or c["run_root"], "advice": adv,
          "serve": {"ok": not codes, "codes": list(codes), "window": w,
                    "detail": {"window": w, "run_id": c["run_id"], "published_root": root or c["run_root"],
                               "advice": adv}}}
    (d / name / "summary.json").write_text(json.dumps(s))


def test_a_serve_run_audit_is_bound_to_the_runs_by_the_endpoints_commit_window_it_proves(tmp_path):
    """The prove path's `--serve-run` audits of the endpoint's own Commit windows (one of them a window of the honest
    run whose advice was registered apart): bound to these
    runs when every arm's Commit is the endpoint's (its windows.jsonl's run id and root) and sealed while serving, every
    arm's advice is held by content to the report's advice of its window (its root registered under that Commit, in the
    report's own advice or apart, and the registration's opening opening it to the report's advice), and an honest arm
    proves one of the honest run's windows. Not bound otherwise, each reason named. Who registered each window's advice
    is the job's to say; when is the registration's own time against the window's close in the endpoint's trace."""
    ep = _endpoint(tmp_path)
    io = json.loads((ep / "io.json").read_text())["requests"]
    recs = chain._jsonl(ep / "advice_trace.jsonl")
    first, closed = chain._first_steps(recs), chain.closes(recs)
    assert (closed[0], closed[1]) == (1792835186.298694, 1792835187.042822) and sorted(closed) == [0, 1, 2]
    info, b = chain.served_advice(ep, ep / "advice", {"origin_unix": first[0] + 1.2, "windows": 2, "window_s": 6.0},
                                  io, 60, [128001])
    row = {x["window"]: x for x in b["commit_windows"]}
    commits = {int(x["window"]): x for x in chain._jsonl(ep / "windows.jsonl")}
    assert sorted(row) == [1, 2]
    lane = tmp_path / "lane-w1"
    apart = _apart(lane, 1)
    assert apart != row[1]["root"]
    os.utime(lane / "registration.json", (closed[1] + 1503.6,) * 2)
    os.utime(ep / "advice/w0/registration.json", (closed[0] - 0.5,) * 2)
    w1, w0 = tmp_path / "w1", tmp_path / "w0"
    for d, w in ((w1, 1), (w0, 0)):
        _serve_arm(d, "served", w, row[1]["v0_bits"] if w == 1 else 7, advice_root=apart if w == 1 else None)
        _serve_arm(d, "served-forged", w, row[1]["v0_bits"] if w == 1 else 7, forged=True,
                   advice_root=apart if w == 1 else None)
    registered = {"1": {"by": "the lane", "registration": str(lane), "ref": "r-reg",
                        "first_draw": chain._utc(closed[1] + 1503.6 + 852.6)},
                  "0": {"by": "vllm"}}
    got = chain.prove_path([w1, w0], info, b["bound"], b, ep, registered)
    assert got["bound"] is True and got["bound_why"] is None and got["ok"] is True and got["tied"] is True
    assert sorted(got["arms"]) == ["w0 served", "w0 served-forged", "w1 served", "w1 served-forged"]
    assert got["arms"]["w1 served"]["commit"] == {
        "window": 1, "run_id": commits[1]["run_id"], "run_root": commits[1]["run_root"], "sealed_while_serving": True,
        "the_endpoints": True, "of_the_honest_run": True, "advice_root": apart, "advice_bits": row[1]["v0_bits"],
        "the_reports_root": False, "registration": str(lane), "the_registered_root": True, "names_the_commit": True,
        "opens": True, "tied": True, "tied_why": None,
        "registered": {"by": "the lane", "ref": "r-reg", "first_draw": registered["1"]["first_draw"],
                       "at": chain._utc(closed[1] + 1503.6), "file_at": chain._utc(closed[1] + 1503.6),
                       "attested": None, "closed": chain._utc(closed[1]), "after_close_s": 1503.6,
                       "next_closed": chain._utc(closed[2]), "before_next_close_s": -1502.9, "session_end": None,
                       "before_session_end_s": None, "before_draw": True, "at_serve_time": False}}
    w0c = got["arms"]["w0 served"]["commit"]
    assert w0c["the_endpoints"] is True and w0c["of_the_honest_run"] is False and w0c["the_reports_root"] is None
    assert w0c["tied"] is True and w0c["registration"] == str(ep / "advice" / "w0")
    assert w0c["registered"]["at_serve_time"] is True and w0c["registered"]["after_close_s"] == -0.5
    assert w0c["registered"]["before_next_close_s"] == 1.2 and w0c["registered"]["before_draw"] is True
    md = "\n".join(chain.markdown({"advice": info, "bits": b, "prove_path": got}))
    assert "\n## Prove path\n" in md and "fixed workload" not in md and "Bound to these runs: True (every arm's" in md
    assert (f"| w1 served | 1 | {commits[1]['run_id']} | {commits[1]['run_root'][:16]}… | True | True | True | "
            f"{apart[:16]}… | {row[1]['v0_bits']} | False | True | True | True | after serving, by the lane (r-reg), "
            "1503.6 s after its Commit window closed, 1502.9 s after the next window closed, 852.6 s before the first "
            "draw |") in md
    assert f"| w0 served | 0 | {commits[0]['run_id']} |" in md
    assert md.count("| — | True | True | True | by vllm at serve time, 0.5 s before its Commit window closed, before "
                    "its draw and 1.2 s before the next window closed |") == 2
    stub = [x for x in chain.stubs({"chain": {"advice": info, "bits": b, "prove_path": got}}, True)
            if x.startswith("Prove path")]
    assert stub == [f"Prove path: the served run's own Commit windows (w0, w1), adopted from {w1}, {w0}; w1's advice "
                    "was registered after serving, by the lane (r-reg), 1503.6 s after its Commit window closed, "
                    "1502.9 s after the next window closed, 852.6 s before the first draw, not by the endpoint while "
                    "serving; the advice w1's arms committed is a registration of its own, held to the report's advice "
                    "by its opening, which the report reads in the clear."]

    def why(*dirs, given=registered) -> str | None:
        p = chain.prove_path(list(dirs), info, b["bound"], b, ep, given)
        assert p["bound"] is (p["bound_why"] is None) and p["tied"] in (None, p["tied_why"] is None)
        return p["bound_why"]
    assert why(w0) == "no honest arm proves one of the honest run's Commit windows"
    _serve_arm(tmp_path / "other", "served", 1, row[1]["v0_bits"], root="0" * 128, advice_root=apart)
    assert why(tmp_path / "other") == "w1 served: Commit window 1 isn't the endpoint's (its windows.jsonl)"
    _serve_arm(tmp_path / "unsealed", "served", 1, row[1]["v0_bits"], codes=["published-root-of-another-commit"],
               advice_root=apart)
    assert why(tmp_path / "unsealed") == "w1 served: its run root isn't the one sealed while serving"
    _serve_arm(tmp_path / "own", "served", 1, row[1]["v0_bits"] - 1)
    assert why(tmp_path / "own", given={}) is None
    assert why(tmp_path / "own") == f"w1 served: its advice root isn't the one registered at {lane}"
    _serve_arm(tmp_path / "count", "served", 1, row[1]["v0_bits"], advice_root="a" * 128)
    assert why(tmp_path / "count", given={}) == (
        f"w1 served: its advice root isn't the one registered at {ep / 'advice/w1'}")
    adv = I1.from_json(json.loads((EP / "advice/w1/private/advice.json").read_text()))
    forged = _apart(tmp_path / "forged-w1", 1, dataclasses.replace(adv, release=tuple(r + 1 for r in adv.release)))
    _serve_arm(tmp_path / "content", "served", 1, row[1]["v0_bits"], advice_root=forged)
    assert why(tmp_path / "content", given={"1": {"registration": str(tmp_path / "forged-w1")}}) == (
        f"w1 served: the registration's opening at {tmp_path / 'forged-w1'} doesn't open its root to the report's "
        "advice of Commit window 1")
    moved = json.loads((lane / "registration.json").read_text()) | {"commit": _reg(2)["commit"]}
    shutil.copytree(lane, tmp_path / "moved")
    (tmp_path / "moved" / "registration.json").write_text(json.dumps(moved))
    assert why(w1, given={"1": {"registration": str(tmp_path / "moved")}}) == "; ".join(
        f"w1 {a}: the registration at {tmp_path / 'moved'} doesn't name Commit window 1's Commit (the endpoint's "
        "windows.jsonl)" for a in ("served", "served-forged"))
    assert why(w1, given={"1": {"registration": str(tmp_path / "nowhere")}}).startswith(
        f"w1 served: no registration of Commit window 1's advice to hold it to at {tmp_path / 'nowhere'} (")
    _arm(tmp_path / "fixed", "llama", {"root": "r", "bits": 2695})
    assert why(w1, tmp_path / "fixed").startswith("from the fixed workload, not this run")
    shutil.rmtree(ep / "advice/w0")
    assert why(w1, w0) == "; ".join(f"w0 {a}: the report has no advice of Commit window 0 to hold it to"
                                    for a in ("served", "served-forged"))


def test_at_serve_time_is_before_its_sample_and_no_later_than_the_next_windows_close(tmp_path):
    """A Commit window's advice was registered at serve time when it was registered before its sample (every rank's
    `advice_before_draw`, the verifier's receipt of the registration before any coin) and no later than the next
    window's close in the endpoint's trace; after serving when either fails, and unknown
    while either is. Its lag after its own window's close is always printed, in seconds."""
    ep = _endpoint(tmp_path)
    io = json.loads((ep / "io.json").read_text())["requests"]
    recs = chain._jsonl(ep / "advice_trace.jsonl")
    first, closed = chain._first_steps(recs), chain.closes(recs)
    info, b = chain.served_advice(ep, ep / "advice", {"origin_unix": first[0] + 1.2, "windows": 2, "window_s": 6.0},
                                  io, 60, [128001])
    bits = {x["window"]: x["v0_bits"] for x in b["commit_windows"]}

    def registered(name: str, w: int, at: float, *receipts: bool | None) -> tuple[bool | None, str]:
        os.utime(ep / f"advice/w{w}/registration.json", (at,) * 2)
        _serve_arm(tmp_path / name, "served", w, bits[w], before_draw=receipts[0])
        s = json.loads((tmp_path / name / "served" / "summary.json").read_text())
        for i, r in enumerate(receipts[1:], 1):
            s["ranks"][str(i)] = s["ranks"]["0"] | {"advice_before_draw": {"ok": r}}
        (tmp_path / name / "served" / "summary.json").write_text(json.dumps(s))
        p = chain.prove_path([tmp_path / name], info, b["bound"], b, ep, {str(w): {"by": "vllm"}})
        g = p["arms"][f"w{w} served"]["commit"]["registered"]
        assert g["after_close_s"] == round(at - closed[w], 1)
        return g["at_serve_time"], chain._registered(g)

    assert round(closed[2] - closed[1], 1) == 0.7
    assert registered("in-time", 1, closed[1] + 0.5, True, True) == (
        True, "by vllm at serve time, 0.5 s after its Commit window closed, before its draw and 0.2 s before the next "
              "window closed")
    assert registered("before-close", 1, closed[1] - 3.0, True) == (
        True, "by vllm at serve time, 3.0 s before its Commit window closed, before its draw and 3.7 s before the next "
              "window closed")
    assert registered("after-the-draw", 1, closed[1] + 0.5, True, False) == (
        False, "after serving, by vllm, 0.5 s after its Commit window closed, after its draw")
    assert registered("after-the-draw-one-missing", 1, closed[1] + 0.5, False, None) == (
        False, "after serving, by vllm, 0.5 s after its Commit window closed, after its draw")
    assert registered("after-the-next", 1, closed[2] + 3.0, True) == (
        False, "after serving, by vllm, 3.7 s after its Commit window closed, 3.0 s after the next window closed")
    assert registered("just-after-the-next", 1, closed[2] + 0.01, True)[0] is False
    assert registered("no-receipt", 1, closed[1] + 0.5, True, None) == (
        None, f"by vllm at {chain._utc(closed[1] + 0.5)}, 0.5 s after its Commit window closed; whether at serve time "
              "is unknown: no draw receipt orders it against its draw")
    assert registered("last", 2, closed[2] + 0.5, True) == (
        None, f"by vllm at {chain._utc(closed[2] + 0.5)}, 0.5 s after its Commit window closed; whether at serve time "
              "is unknown: the next window's close isn't in the endpoint's records")


def test_the_serving_runs_record_times_the_registration_and_its_end_bounds_the_last_window(tmp_path):
    """Ahead of a registration's file time, the serving run's own record (`serve_record`, ``record`` in
    --prove-path-registered): its advice loop stamps each window's registered root once the registration is in place,
    to the second, so the rule reads it as registered no later than a second past the stamp; a stamp of another root
    isn't this registration's. At the stamp's resolution a registration its file puts just before the next close is
    unknown, not at serve time. The session's last window, with no next close in the endpoint's records, is bounded by
    the session's end when the record says it ended with this window the last its loop logged, and stays unknown while
    it serves or when the loop logged a later window."""
    ep = _endpoint(tmp_path)
    io = json.loads((ep / "io.json").read_text())["requests"]
    recs = chain._jsonl(ep / "advice_trace.jsonl")
    first, closed = chain._first_steps(recs), chain.closes(recs)
    info, b = chain.served_advice(ep, ep / "advice", {"origin_unix": first[0] + 1.2, "windows": 2, "window_s": 6.0},
                                  io, 60, [128001])
    bits = {x["window"]: x["v0_bits"] for x in b["commit_windows"]}
    root = {w: json.loads((ep / f"advice/w{w}/registration.json").read_text())["binding"]["advice"]["root"]
            for w in (1, 2)}
    rec = tmp_path / "record"
    (rec / "out").mkdir(parents=True)
    c1, c2, s2 = closed[1], closed[2], float(int(closed[2]))

    def stamp(t: float) -> str:
        return chain._utc(t)[11:19] + "Z"

    def registered(name: str, w: int, file_at: float, stamps: list, ended: float | None = None):
        (rec / "out" / "advice_windows.jsonl").write_text("".join(
            json.dumps({"window": k, "at": stamp(t), "rc": 0, "root": r, "bits": 1, "refused": None}) + "\n"
            for k, t, r in stamps))
        (rec / "status.json").write_text(json.dumps({"ended_utc": None if ended is None else chain._utc(ended)}))
        os.utime(ep / f"advice/w{w}/registration.json", (file_at,) * 2)
        _serve_arm(tmp_path / name, "served", w, bits[w], before_draw=True)
        p = chain.prove_path([tmp_path / name], info, b["bound"], b, ep, {str(w): {"by": "vllm", "record": str(rec)}})
        g = p["arms"][f"w{w} served"]["commit"]["registered"]
        assert g["file_at"] == chain._utc(file_at)
        return g["at_serve_time"], g, chain._registered(g)

    ok, g, label = registered("stamped", 1, c2 + 5.0, [(1, s2 - 1.0, root[1])])
    assert ok is True and g["at"] == chain._utc(s2)
    assert g["attested"] == {"stamp": stamp(s2 - 1.0), "record": str(rec), "within": True}
    assert label == (f"by vllm at serve time, no later than {chain._lag(round(s2 - c1, 1))}, before its draw and at "
                     f"least {round(c2 - s2, 1)} s before the next window closed (its run record's stamp, "
                     f"{stamp(s2 - 1.0)} to the second)")
    ok, g, label = registered("to-the-second", 1, c2 - 0.1, [(1, c2, root[1])])
    assert ok is None and g["at"] == chain._utc(s2 + 1.0) and label == (
        f"by vllm, no later than {chain._utc(s2 + 1.0)} ({chain._lag(round(s2 + 1.0 - c1, 1))}); "
        f"whether at serve time is unknown: its run record's stamp, {stamp(c2)} to the second, puts it up to "
        f"{abs(round(c2 - s2 - 1.0, 1))} s after the next window closed, though its file's time, {chain._utc(c2 - 0.1)}, "
        "is before")
    assert registered("both-late", 1, c2 + 3.0, [(1, c2 + 3.0, root[1])])[0] is False
    ok, g, _ = registered("another-root", 1, c1 + 0.5, [(1, c2 + 3.0, root[2])])
    assert ok is True and g["attested"] is None and g["at"] == chain._utc(c1 + 0.5)

    last = [(1, c1 + 0.5, root[1]), (2, c2 + 2.0, root[2])]
    assert registered("serving", 2, c2 + 1.5, last)[0] is None
    ok, g, label = registered("ended", 2, c2 + 1.5, last, ended=c2 + 60.0)
    end = float(int(c2 + 2.0)) + 1.0
    assert ok is True and g["session_end"] == chain._utc(c2 + 60.0) and g["next_closed"] is None
    assert label.endswith(f"before its draw and at least {round(c2 + 60.0 - end, 1)} s before its session ended, with "
                          f"no window after it (its run record's stamp, {stamp(c2 + 2.0)} to the second)")
    assert registered("later-window", 2, c2 + 1.5, last + [(3, c2 + 9.0, root[1])], ended=c2 + 60.0)[0] is None
    ok, _, label = registered("after-the-end", 2, c2 + 5.0, [(1, c1 + 0.5, root[1]), (2, c2 + 5.0, root[2])],
                              ended=c2 + 1.0)
    assert ok is False and "s after its session ended, with no window after it" in label


def test_the_endpoints_own_advice_is_the_runs_by_its_commit_windows_each_opened_and_rebuilt(tmp_path):
    """The own-advice endpoint's first three Commit windows (its served run's records, trimmed: its scheduler
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


CARRY = HERE / "tests" / "data" / "endpoint_carry"


def _reregister(d: Path, data: bytes) -> None:
    """Commit window ``d``'s registration and opening (a `verity-vllm advice` window directory) remade for ``data``
    under salts of its own, as a prover would register a window advice of its choosing."""
    reg = json.loads((d / "registration.json").read_text())
    opening = json.loads((d / "private" / "opening.json").read_text())
    n, k = reg["row_bits"] // 8, len(bytes.fromhex(opening["salts"][0]))
    rows = [data[i:i + n].ljust(n, b"\0") for i in range(0, len(data), n)]
    salts = [(hashlib.sha512(b"lied %d" % i).digest() * 4)[:k] for i in range(len(rows))]
    binding = hashlib.sha512(chain.canonical_json({"scheme": reg["scheme"],
                                                   "compiler": reg["binding"]["compiler"]})).digest()
    reg["binding"]["advice"]["root"] = chain.AC.commit(rows, reg["row_bits"], binding, salts).root.hex()
    (d / "registration.json").write_text(json.dumps(reg))
    (d / "private" / "opening.json").write_text(json.dumps(opening | {"advice": data.hex(),
                                                                      "salts": [s.hex() for s in salts]}))


def test_commit_windows_that_carry_a_request_across_their_close_bind_as_one_advice_that_charges_the_carry_once(
        tmp_path):
    """The gate endpoint's Commit windows 113 to 116, trimmed: 114 ends three
    requests with outputs still to keep and 115 carries them in (window advice v1). Bound and rebuilt from the certified
    I/O of their nine requests, 114 and 115 joined into one advice (`window_chain.join`), which holds each carried
    request's record and state once where 115's string repeats them; the windows that carry nothing are charged their
    own registered bits. Refused: 115 opening with a state 114 didn't leave, registered under its own salts so that it
    opens, 115 without 114 (a carry from a window that isn't the run's) and the run ending at 114 (a window left owing
    its carry)."""
    io = json.loads((CARRY / "io.json").read_text())["requests"]
    first = chain._first_steps(chain._jsonl(CARRY / "advice_trace.jsonl"))
    dep = {"origin_unix": first[113] - 0.05, "windows": 2, "window_s": 6.0}
    info, b = chain.served_advice(CARRY, CARRY / "advice", dep, io, 60, [128001])
    assert b["bound"] is True and b["rebuilt"] is True and b["public_fields"] is True, b["bound_why"] or b["why"]
    rows = {x["window"]: x for x in b["commit_windows"]}
    assert sorted(rows) == [113, 114, 115, 116] and all(x["opened"] for x in rows.values())
    assert b["chains"] == [[114, 115]] and rows[114]["chain"] == rows[115]["chain"] == [114, 115]
    assert [rows[w]["carried"] for w in sorted(rows)] == [0, 0, 3, 0] and info["requests"] == len(io) == 9
    regs = {w: json.loads((CARRY / f"advice/w{w}/registration.json").read_text()) for w in rows}
    assert all(rows[w]["committed_bits"] == regs[w]["binding"]["advice"]["bits"] for w in rows)
    assert rows[113]["v0_bits"] == rows[113]["committed_bits"] and rows[116]["v0_bits"] == rows[116]["committed_bits"]
    carried = [chain.WA.from_json(json.loads((CARRY / f"advice/w{w}/private/advice.json").read_text()))
               for w in (114, 115)]
    committed = rows[114]["committed_bits"] + rows[115]["committed_bits"]
    assert rows[115]["v0_bits"] == 0 and 0 < rows[114]["v0_bits"] < committed
    lag = rows[115]["bucket"] - rows[114]["bucket"]
    joined = chain.WC.join(list(zip((114, 115), carried)), lambda k, rid, i, bucket: bucket + lag * k)
    assert rows[114]["v0_bits"] == chain.I1.total_bits(joined) and len(joined.requests) == 3 + 4
    assert b["v0_total"] == info["bits"] == sum(x["v0_bits"] for x in b["windows"]) \
        == sum(x["v0_bits"] for x in rows.values())
    assert all(0 < x["derived_bits"] < x["v0_bits"] for w, x in rows.items() if w != 115)
    md = "\n".join(chain.markdown({"advice": info, "bits": b}))
    assert "(windows 114–115)" in md and "| requests admitted | carried in |" in md

    ep = tmp_path / "ep"
    shutil.copytree(CARRY, ep)
    a = carried[1]
    lied = chain.WA.Advice(a.requests, a.steps, a.release,
                           (dataclasses.replace(a.carried[0], C=a.carried[0].C + 1), *a.carried[1:]))
    (ep / "advice/w115/private/advice.json").write_text(json.dumps(chain.WA.to_json(lied)))
    _reregister(ep / "advice/w115", chain.WA.encode(lied))
    bad = chain.served_advice(ep, ep / "advice", dep, io, 60, [128001])[1]
    assert bad["bound"] is False and all(x["opened"] for x in bad["commit_windows"])
    assert "Commit window 115 doesn't open where window 114 closed" in bad["bound_why"]
    assert {x["window"]: x["bound"] for x in bad["commit_windows"]} == {113: True, 114: False, 115: False, 116: True}
    index = json.loads((CARRY / "advice/windows.json").read_text())
    del index["windows"]["114"]
    (ep / "advice/windows.json").write_text(json.dumps(index))
    shutil.copytree(CARRY / "advice/w115", ep / "advice/w115", dirs_exist_ok=True)
    alone = chain.served_advice(ep, ep / "advice", dep, io, 60, [128001])[1]
    assert alone["bound"] is False
    assert "Commit window 115 carries 3 requests in from window 114, which isn't among the windows" \
        in alone["bound_why"]
    short = chain.served_advice(CARRY, CARRY / "advice", dep | {"window_s": 1.2}, io, 60, [128001])[1]
    assert [x["window"] for x in short["commit_windows"]] == [113, 114] and short["bound"] is False
    assert "Commit window 114 ends request" in short["bound_why"]
    assert "with outputs still to keep, and window 115 isn't among the windows" in short["bound_why"]
