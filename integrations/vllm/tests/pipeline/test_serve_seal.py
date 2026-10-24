"""A served window sealed as a deferred Commit's row dir (`serve_seal.seal_window`) and opened as PROVE-PATH opens it
(`deferred_rows.of_record`), and the registered advice compiler (`advice_windows.Capped`, PROVE-PATH's, with the cap ℓ).  CPU only."""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_seal as ss

ROWS = {0: [("runner-a", 0, 3, 0, 3), ("runner-b", 3, 5, 0, 2), ("warmup", 5, 6, 0, 1)], 1: [("runner-a", 0, 1, 3, 3)]}
NAMES = {"runner-a": "alice", "runner-b": "cmpl-2"}


def _ledgers() -> dict[str, sc.Ledger]:
    return {"alice": sc.Ledger("cmpl-1", "e1", [1, 2, 3], {"params": {"max_tokens": 4}}, "alice", streamed=[7, 9]),
            "cmpl-2": sc.Ledger("cmpl-2", "e2", [4, 5], {"params": {"max_tokens": 2}}, None, streamed=[8])}


def test_a_windows_requests_are_named_by_user_with_their_rows_and_first_step():
    leds = _ledgers()
    reqs, rows = ss.window_requests(ROWS, NAMES.get, leds.get)
    assert [q["request_id"] for q in reqs] == ["alice", "cmpl-2"] and [q["index"] for q in reqs] == [0, 1]
    assert reqs[0] | {"ingress": None} == {"request_id": "alice", "index": 0, "arrive_step": 0, "prompt_token_ids": [1, 2, 3], "max_tokens": 4,
                                           "http_id": "cmpl-1", "engine_id": "e1", "ingress": None}
    assert rows == {0: [("alice", 0, 3), ("cmpl-2", 3, 5)], 1: [("alice", 0, 1)]}, "a row of no served request is in no request's rows"


def _sealed(tmp_path):
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from verity_vllm.commit import step_rows

    com, rc = commit_steps({0: [("model.embed_tokens", bytes(range(256)) * 2)], 1: [("runner.sampler", b"\x01\x02\x03\x04")]},
                           commit_scheme=step_rows.NAME)
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    leds = _ledgers()
    reqs, rows = ss.window_requests(ROWS, NAMES.get, leds.get)
    d = tmp_path / "windows" / "w5"
    s = ss.seal_window(com, rc, d, rec={"window": 5, "run_id": "t/w5", "steps": len(rc.steps), "passed": True}, reqs=reqs,
                       request_rows=rows, tokens={k: v.streamed for k, v in leds.items()}, sampling={}, label="serve-openai:t",
                       code="c0de", engine_of_record={"vllm": "test"})
    return rc, d, s


def test_a_sealed_rows_sha512_window_opens_as_a_deferred_commits_row_dir(tmp_path):
    from verity_vllm.pipeline import deferred_rows as DR

    rc, d, s = _sealed(tmp_path)
    assert s["commit_scheme"] == "rows-sha512" and len(rc.run_root) == DR.SERVED_ROOT and "error" not in s
    v = json.loads((d / "commit" / "verdict.json").read_text())
    assert v["run_root"] == rc.run_root.hex() and v["replay_deferred"]["bundles"][0]["manifest_sha256"] == s["manifest_sha256"]
    r = DR.of_record(s["bundle"], str(d))
    assert r.run.run_root == rc.run_root and r.ctx["served_window"]["window"] == 5 and r.ctx["code"] == "c0de"
    assert r.programs == {} and r.arrivals == {"alice": 0, "cmpl-2": 0} and r.ctx["tokens"] == {"alice": [7, 9], "cmpl-2": [8]}
    assert sorted(p.name for p in (d / "commit").iterdir()) == ["binding_map_p0.json", "replay_bundle_p0", "runs.jsonl", "verdict.json"]


def _other_root(d):
    p = d / "commit" / "runs.jsonl"
    row = json.loads(p.read_text())
    row["commit"]["run_root"] = "00" * 64
    p.write_text(json.dumps(row) + "\n")


@pytest.mark.parametrize("tamper,why", [
    (_other_root, "run root of record"),
    (lambda d: (d / "commit" / "binding_map_p0.json").write_text("{}"), "binding map"),
    (lambda d: (d / "commit" / "verdict.json").write_text(json.dumps({"replay_deferred": {"bundles": []}})), "names no bundle"),
])
def test_a_sealed_window_whose_record_is_not_its_bundles_is_refused(tmp_path, tamper, why):
    from verity_vllm.pipeline import deferred_rows as DR
    from verity_vllm.pipeline.cpu_replay import Refused

    _rc, d, s = _sealed(tmp_path)
    tamper(d)
    with pytest.raises(Refused, match=why):
        DR.of_record(s["bundle"], str(d))


# ---------------------------------------------------------------------------------------------------------------- the advice's cap

def test_the_registered_compiler_is_prove_paths_capped_compiler():
    """registration.json's compiler digest is the one PROVE-PATH's capped re-commit binds: `advice_windows.Capped`'s formula."""
    import hashlib

    from verity.ml.circuits.codec import canonical_json
    from verity_vllm.pipeline import advice_windows as AW
    from verity_vllm.predict import steps as ST

    assert ADV.CAP_BITS == AW.CAP_BITS == ST.CAP_BITS == 1 << 23 and AW.CAPPED == "verity/prove-path/capped-compiler/v0"
    c = ADV.registered_compiler("inner-compiler", "cd" * 64, cap_bits=4096)
    want = hashlib.sha512(canonical_json({"name": "verity/prove-path/capped-compiler/v0", "cap_bits": 4096,
                                          "compiler": {"name": "inner-compiler", "digest": "cd" * 64}})).hexdigest()
    assert isinstance(c, AW.Capped) and c.name == AW.CAPPED and c.digest == want
    assert c.digest == "2f470d2b6aadc53b707b0bd65820da4c55583eb18667eb3dce128aabb9dddab855473d52af758a630a2958e3555f7cafbc68e9238c0188cd9f8ef551abca2a9a"
    assert ADV.registered_compiler("inner-compiler", "cd" * 64, cap_bits=8192).digest != want
    assert ADV.registered_compiler("inner-compiler").cap_bits == ADV.CAP_BITS


@pytest.mark.parametrize("cap", [0, 7, True, "8"])
def test_a_cap_that_is_not_a_bit_count_is_refused(cap):
    with pytest.raises(ValueError, match="whole number of bits"):
        ADV.registered_compiler("c", digest="ab" * 64, cap_bits=cap)
