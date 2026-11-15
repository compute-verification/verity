"""`verity-vllm advice --bind` (`advice_bind`): the bits per window, v0's summing to I1's total and the residual's giving back
every step from the config alone, and with the certified I/O the whole advice (bound), or nothing when the I/O certifies other
requests; the endpoint's own advice, bound to a run by its Commit windows, a chain of windows that carry requests across
their close joined into one advice; and the command, which reads the run on stdin, writes nothing and prints the result or
the error that refused the records."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from verity.circuits.codec import canonical_json
from verity.commitments import advice as AC
from verity.protocols.resource_accounting.network.prover import window_chain as WC
from verity.protocols.resource_accounting.network.verifier import advice as I1

from verity_vllm.pipeline import advice_bind as AB
from verity_vllm.pipeline import window_advice as WA

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "benchmarks" / "network_e2e" / "tests" / "data"
EP, CARRY = DATA / "endpoint", DATA / "endpoint_carry"
T = 4
CONFIG = {"block_size": 16, "num_gpu_blocks": 1000, "max_num_batched_tokens": 64, "max_model_len": 128,
          "eos_token_id": 9, "stop_token_ids": []}


def _advice() -> I1.Advice:
    """q0 (4 prompt tokens, 2 outputs, length) and q1 (3, 1, eos): both prefilled whole in step 0, then q0 decodes
    once; their three outputs released in windows 0, 0 and 1 of T buckets."""
    return I1.Advice((I1.RequestAdvice("q0", 4, 2, "length"), I1.RequestAdvice("q1", 3, 1, "eos")),
                     (I1.StepAdvice((I1.Row(0, "prefill", 0, 4), I1.Row(1, "prefill", 0, 3))),
                      I1.StepAdvice((I1.Row(0, "decode", 4, 1),))), (1, 1, T + 2))


def _io(users: tuple[str, str] = ("q0", "q1")) -> list[dict]:
    """The certified I/O of `_advice`'s requests, in ingress order."""
    return [{"user": users[0], "prompt": [1, 2, 3, 4], "max_tokens": 2, "output": [5, 6]},
            {"user": users[1], "prompt": [1, 2, 3], "max_tokens": 4, "output": [9]}]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_the_bits_per_window_are_v0s_and_the_residuals_and_the_certified_io_binds_and_rebuilds_the_advice():
    adv = _advice()
    unbound = AB.bits(adv, T, 3, CONFIG, None)
    assert unbound["bound"] is None and unbound["public_fields"] is None and unbound["rebuilt"] is None
    assert "the certified tokens" in unbound["why"] and unbound["windows"][0]["derived_bits"] is None
    b = AB.bits(adv, T, 3, CONFIG, _io())
    rows = {x["window"]: x for x in b["windows"]}
    assert sorted(rows) == [0, 1, 2] and sum(x["v0_bits"] for x in rows.values()) == I1.total_bits(adv) == b["v0_total"]
    assert {w: x["v0_bits"] for w, x in rows.items() if x["v0_bits"]} == I1.bits_by_window(adv, T)
    assert b["bound"] is True and b["rebuilt"] is True and b["public_fields"] is True and b["why"] is None
    assert 0 < rows[0]["derived_bits"] < rows[0]["v0_bits"] and rows[0]["derived_shaped_bits"] <= rows[0]["derived_bits"]
    assert rows[0]["forced_bits"] <= rows[0]["derived_bits"] and rows[2]["v0_bits"] == 0
    swapped = AB.bits(adv, T, 3, CONFIG, _io()[::-1])
    assert swapped["bound"] is True and swapped["rebuilt"] is True
    other = AB.bits(adv, T, 3, CONFIG, _io(("q0", "q9")))
    assert other["bound"] is False and other["rebuilt"] is None and "aren't the 2 certified ones" in other["bound_why"]
    wrong = _io()
    wrong[0]["output"] = [5, 6, 7]
    assert AB.bits(adv, T, 3, CONFIG, wrong)["rebuilt"] is False
    none = AB.bits(adv, T, 1, None, None)
    assert none["windows"][0]["derived_bits"] is None and "--serving-config" in none["why"]
    assert [x["in_run"] for x in none["windows"]] == [True, False]


def test_the_endpoints_own_advice_is_the_runs_by_its_commit_windows_each_opened_and_rebuilt(tmp_path):
    """The own-advice endpoint's first three Commit windows (its served run's records, trimmed: its scheduler
    trace, its windows.jsonl and `verity-vllm advice`'s registrations and openings, as the endpoint committed them
    mid-serve) and the certified I/O of their nine requests: bound and rebuilt, each window opening its registered root;
    not bound when a window falls outside the run's execution phases, holds a request the run didn't certify, opens another
    root or names another Commit, or has no advice; not rebuilt when a certified output breaks the stop rule."""
    io = json.loads((EP / "io.json").read_text())["requests"]
    first = AB.first_steps(_jsonl(EP / "advice_trace.jsonl"))
    dep = {"origin_unix": first[0] - 0.05, "windows": 2, "window_s": 6.0}
    info, b = AB.served_advice(EP, EP / "advice", dep, io, 60, [128001])
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
    late = AB.served_advice(EP, EP / "advice", dep | {"origin_unix": first[0] + 1.2}, io, 60, [128001])[1]
    assert late["bound"] is False and [x["window"] for x in late["commit_windows"]] == [1, 2]
    assert "1 certified requests are in none of the run's Commit windows" in late["bound_why"]
    other = AB.served_advice(EP, EP / "advice", dep, io[1:], 60, [128001])[1]
    assert other["bound"] is False and other["rebuilt"] is False
    assert "Commit window 0 holds 1 requests that aren't the run's certified ones" in other["bound_why"]
    broken = [dict(r) for r in io]
    broken[1]["output"] = [128001, *broken[1]["output"][1:]]
    eos = AB.served_advice(EP, EP / "advice", dep, broken, 60, [128001])[1]
    assert eos["bound"] is True and eos["rebuilt"] is False and "break the stop rule" in eos["why"]
    ep = tmp_path / "ep"
    shutil.copytree(EP, ep)
    opening = json.loads((ep / "advice/w1/private/opening.json").read_text())
    opening["salts"][0] = "00" * (len(opening["salts"][0]) // 2)
    (ep / "advice/w1/private/opening.json").write_text(json.dumps(opening))
    lines = (ep / "windows.jsonl").read_text().splitlines()
    (ep / "windows.jsonl").write_text("\n".join(lines[:2] + [lines[2].replace(regs[2]["commit"]["run_root"], "0" * 64)]))
    bad = AB.served_advice(ep, ep / "advice", dep, io, 60, [128001])[1]
    assert bad["bound"] is False and [x["opened"] for x in bad["commit_windows"]] == [True, False, True]
    assert "Commit window 1's opening doesn't open its registered root" in bad["bound_why"]
    assert "Commit window 2's registration names another Commit" in bad["bound_why"]
    index = json.loads((ep / "advice/windows.json").read_text())
    del index["windows"]["0"]
    (ep / "advice/windows.json").write_text(json.dumps(index | {"refused": {"0": "a request it holds only part of"}}))
    gone = AB.served_advice(ep, ep / "advice", dep, io, 60, [128001])[1]
    assert "Commit window 0 has no advice (a request it holds only part of)" in gone["bound_why"]
    assert gone["commit_windows"][0] == {"window": 0, "refused": "a request it holds only part of"}


def _reregister(d: Path, data: bytes) -> None:
    """Commit window ``d``'s registration and opening (a `verity-vllm advice` window directory) remade for ``data``
    under salts of its own, as a prover would register a window advice of its choosing."""
    reg = json.loads((d / "registration.json").read_text())
    opening = json.loads((d / "private" / "opening.json").read_text())
    n, k = reg["row_bits"] // 8, len(bytes.fromhex(opening["salts"][0]))
    rows = [data[i:i + n].ljust(n, b"\0") for i in range(0, len(data), n)]
    salts = [(hashlib.sha512(b"lied %d" % i).digest() * 4)[:k] for i in range(len(rows))]
    binding = hashlib.sha512(canonical_json({"scheme": reg["scheme"], "compiler": reg["binding"]["compiler"]})).digest()
    reg["binding"]["advice"]["root"] = AC.commit(rows, reg["row_bits"], binding, salts).root.hex()
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
    first = AB.first_steps(_jsonl(CARRY / "advice_trace.jsonl"))
    dep = {"origin_unix": first[113] - 0.05, "windows": 2, "window_s": 6.0}
    info, b = AB.served_advice(CARRY, CARRY / "advice", dep, io, 60, [128001])
    assert b["bound"] is True and b["rebuilt"] is True and b["public_fields"] is True, b["bound_why"] or b["why"]
    rows = {x["window"]: x for x in b["commit_windows"]}
    assert sorted(rows) == [113, 114, 115, 116] and all(x["opened"] for x in rows.values())
    assert b["chains"] == [[114, 115]] and rows[114]["chain"] == rows[115]["chain"] == [114, 115]
    assert [rows[w]["carried"] for w in sorted(rows)] == [0, 0, 3, 0] and info["requests"] == len(io) == 9
    regs = {w: json.loads((CARRY / f"advice/w{w}/registration.json").read_text()) for w in rows}
    assert all(rows[w]["committed_bits"] == regs[w]["binding"]["advice"]["bits"] for w in rows)
    assert rows[113]["v0_bits"] == rows[113]["committed_bits"] and rows[116]["v0_bits"] == rows[116]["committed_bits"]
    carried = [WA.from_json(json.loads((CARRY / f"advice/w{w}/private/advice.json").read_text())) for w in (114, 115)]
    committed = rows[114]["committed_bits"] + rows[115]["committed_bits"]
    assert rows[115]["v0_bits"] == 0 and 0 < rows[114]["v0_bits"] < committed
    lag = rows[115]["bucket"] - rows[114]["bucket"]
    joined = WC.join(list(zip((114, 115), carried)), lambda k, rid, i, bucket: bucket + lag * k, WA)
    assert rows[114]["v0_bits"] == I1.total_bits(joined) and len(joined.requests) == 3 + 4
    assert b["v0_total"] == info["bits"] == sum(x["v0_bits"] for x in b["windows"]) \
        == sum(x["v0_bits"] for x in rows.values())
    assert all(0 < x["derived_bits"] < x["v0_bits"] for w, x in rows.items() if w != 115)

    ep = tmp_path / "ep"
    shutil.copytree(CARRY, ep)
    a = carried[1]
    lied = WA.Advice(a.requests, a.steps, a.release,
                     (dataclasses.replace(a.carried[0], C=a.carried[0].C + 1), *a.carried[1:]))
    (ep / "advice/w115/private/advice.json").write_text(json.dumps(WA.to_json(lied)))
    _reregister(ep / "advice/w115", WA.encode(lied))
    bad = AB.served_advice(ep, ep / "advice", dep, io, 60, [128001])[1]
    assert bad["bound"] is False and all(x["opened"] for x in bad["commit_windows"])
    assert "Commit window 115 doesn't open where window 114 closed" in bad["bound_why"]
    assert {x["window"]: x["bound"] for x in bad["commit_windows"]} == {113: True, 114: False, 115: False, 116: True}
    index = json.loads((CARRY / "advice/windows.json").read_text())
    del index["windows"]["114"]
    (ep / "advice/windows.json").write_text(json.dumps(index))
    shutil.copytree(CARRY / "advice/w115", ep / "advice/w115", dirs_exist_ok=True)
    alone = AB.served_advice(ep, ep / "advice", dep, io, 60, [128001])[1]
    assert alone["bound"] is False
    assert "Commit window 115 carries 3 requests in from window 114, which isn't among the windows" \
        in alone["bound_why"]
    short = AB.served_advice(CARRY, CARRY / "advice", dep | {"window_s": 1.2}, io, 60, [128001])[1]
    assert [x["window"] for x in short["commit_windows"]] == [113, 114] and short["bound"] is False
    assert "Commit window 114 ends request" in short["bound_why"]
    assert "with outputs still to keep, and window 115 isn't among the windows" in short["bound_why"]


def _bind(ep: Path, run: dict, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "advice", str(ep), "--bind", "-", *flags],
                          input=json.dumps(run), capture_output=True, text=True, check=False)


def test_bind_reads_the_run_on_stdin_writes_nothing_and_prints_what_served_advice_returns_or_the_error(tmp_path):
    """The command's document is `served_advice`'s two, under ``advice`` and ``bits``, exactly (the round trip through JSON
    loses nothing a caller records); the advice directory is left as it was; a record that can't be read is the repr of
    its error under ``error``, exit 1; and ``--bind`` without ``--eos`` is a usage error."""
    ep = tmp_path / "ep"
    shutil.copytree(EP, ep)
    io = json.loads((ep / "io.json").read_text())["requests"]
    first = AB.first_steps(_jsonl(ep / "advice_trace.jsonl"))
    dep = {"origin_unix": first[0] - 0.05, "windows": 2, "window_s": 6.0}
    before = {p: p.read_bytes() for p in sorted(ep.rglob("*")) if p.is_file()}
    p = _bind(ep, {"deployment": dep, "io": io}, "--eos", "128001", "--T", "60")
    assert p.returncode == 0, p.stderr
    info, b = AB.served_advice(ep, ep / "advice", dep, io, 60, [128001])
    assert p.stdout == json.dumps({"advice": info, "bits": b}) + "\n"
    assert {q: q.read_bytes() for q in sorted(ep.rglob("*")) if q.is_file()} == before
    (ep / "advice" / "windows.json").unlink()
    gone = _bind(ep, {"deployment": dep, "io": io}, "--eos", "128001", "--T", "60")
    assert gone.returncode == 1
    assert json.loads(gone.stdout) == {"error": "FileNotFoundError(2, 'No such file or directory')"}
    assert _bind(ep, {"deployment": dep, "io": io}).returncode == 2
