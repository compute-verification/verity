"""Each message the adapter writes or reads through its declaration (`verity_vllm.messages`) keeps the bytes it had
before, and its reader reads what it read before."""

from __future__ import annotations

import json

import pytest

from verity.channel import Refused
from verity_vllm import messages as M
from verity_vllm.pipeline import advice as A


def test_every_declared_message_is_a_type():
    assert {"RunTotals", "Run", "Request", "ServingConfiguration", "CommittedRows", "PoUSStore"} <= M.TYPES.types


def test_a_type_no_protocol_declares_is_refused():
    with pytest.raises(Refused, match="unknown-type"):
        M.body("Summary", b"{}")


def test_run_totals_bytes_unchanged(tmp_path):
    got = {"final": True, "served": 3, "stream_errors": {"n": 1, "first": ["engine: é"]}, "engine_steps": 7,
           "commit": None, "remote_session": None}
    (tmp_path / "before").mkdir()
    (tmp_path / "after").mkdir()
    (tmp_path / "before" / "summary.json").write_text(json.dumps(got, indent=1))
    M.write(tmp_path / "after" / "summary.json", "RunTotals", json.dumps(got, indent=1).encode())
    assert (tmp_path / "after" / "summary.json").read_bytes() == (tmp_path / "before" / "summary.json").read_bytes()
    assert A.stopped(tmp_path / "after") is True
    (tmp_path / "after" / "summary.json").write_text(json.dumps({**got, "final": False}))
    assert A.stopped(tmp_path / "after") is False
    assert A.stopped(tmp_path / "none") is False


def test_serving_configuration_bytes_unchanged(tmp_path):
    record = {"engine": "vllm", "revision": "0.0", "flags": ["--seal"], "phi": {"layers": 2}, "path": tmp_path,
              "clocks": {"perf_counter": 1.5, "time": 2.0}}
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps(record, indent=1, default=str))
    M.write(after, "ServingConfiguration", json.dumps(record, indent=1, default=str).encode())
    assert after.read_bytes() == before.read_bytes()
    assert json.loads(M.read(after, "ServingConfiguration")) == json.loads(before.read_text())


def test_committed_rows_bytes_unchanged(tmp_path):
    doc = {"window": 2, "run_root": "ab" * 64, "statics": {"width": 4}, "ids": [[1, 2], [3]],
           "session": [[None, [[0, 1]]]], "salts": [bytes(range(16)).hex()]}
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps(doc))
    M.write(after, "CommittedRows", json.dumps(doc).encode())
    assert after.read_bytes() == before.read_bytes()


@pytest.mark.parametrize("name", ["windows.jsonl", "seals.jsonl"])
def test_run_lines_bytes_unchanged(tmp_path, name):
    recs = [{"window": 0, "run_id": "r/w0", "run_root": "cd" * 64, "steps": 3, "requests": [{"engine_id": "a"}],
             "passed": True}, {"window": 1, "run_id": "r/w1", "steps": 0, "empty": True, "note": "é"}]
    (tmp_path / "before").mkdir()
    (tmp_path / "after").mkdir()
    for rec in recs:
        with open(tmp_path / "before" / name, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        M.append(tmp_path / "after" / name, "Run", json.dumps(rec).encode())
    assert (tmp_path / "after" / name).read_bytes() == (tmp_path / "before" / name).read_bytes()
    old = [json.loads(line) for line in (tmp_path / "before" / name).read_text().splitlines() if line.strip()]
    assert [json.loads(line) for line in M.lines(tmp_path / "after" / name, "Run")] == old == recs


def test_request_lines_bytes_unchanged(tmp_path):
    recs = [{"http_id": "h0", "engine_id": "e0", "user": "user-0", "prompt_ids": [1, 2], "streamed_ids": [3],
             "finish": {"reason": "length"}, "check": {"passed": True}},
            {"http_id": "h1", "engine_id": "e1", "user": "é", "prompt_ids": [4], "streamed_ids": []}]
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    with open(before / "requests.jsonl", "a") as fh, open(before / "request_users.jsonl", "a") as fu:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
            fu.write(json.dumps({"engine_id": rec["engine_id"], "user": rec["user"]}) + "\n")
    with open(after / "requests.jsonl", "ab") as fh, open(after / "request_users.jsonl", "ab") as fu:
        for rec in recs:
            fh.write(M.line("Request", json.dumps(rec).encode()))
            fu.write(M.line("Request", json.dumps({"engine_id": rec["engine_id"], "user": rec["user"]}).encode()))
    for name in ("requests.jsonl", "request_users.jsonl"):
        assert (after / name).read_bytes() == (before / name).read_bytes()
    assert A.io_record(after / "requests.jsonl") == A.io_record(before / "requests.jsonl")


def test_advice_bytes_unchanged(tmp_path):
    public = {"compiler": {"name": "i1", "digest": "ef" * 64}, "binding": {"advice": {"root": "01" * 64, "bits": 9}}}
    data = bytes([0x49, 0x31, 0, 7, 255])
    doc = {"requests": [{"id": "e0", "rows": [[0, 4]]}], "note": "é"}
    texts = {"registration.json": json.dumps(public, indent=1) + "\n", "windows.json": json.dumps({"windows": {}}, indent=1)
             + "\n", "advice.json": json.dumps(doc, separators=(",", ":")) + "\n", "opening.json": json.dumps(doc) + "\n"}
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "advice.bin").write_bytes(data)
    M.write(after / "advice.bin", "Advice", data)
    for name, text in texts.items():
        (before / name).write_text(text)
        M.write(after / name, "Advice", text.encode())
    for name in ("advice.bin", *texts):
        assert (after / name).read_bytes() == (before / name).read_bytes()
        assert M.read(after / name, "Advice") == (before / name).read_bytes()


def test_pearl_pass_bytes_unchanged(tmp_path):
    coins = {"source": "live", "seeds": ["0a" * 32], "note": "é"}
    doc = {"schema": "pearl-served/v1", "window": 0, "passes": [{"pass": 0, "units": 3}], "coins": coins, "ok": True}
    texts = {"coins.json": json.dumps(coins) + "\n", "served.json": json.dumps(doc, indent=1) + "\n"}
    for name, text in texts.items():
        (tmp_path / f"before-{name}").write_text(text)
        M.write(tmp_path / f"after-{name}", "PearlPass", text.encode())
        assert (tmp_path / f"after-{name}").read_bytes() == (tmp_path / f"before-{name}").read_bytes()


def test_h1_registration_bytes_unchanged(tmp_path):
    w = {"schema": "h1-registration/v1", "window": 0, "prev": None, "roots": {"advice": "02" * 64}, "note": "é"}
    lines = [{"kind": "declaration", "digest": "03" * 64}, {"window": 0, "kind": "registration",
                                                             "file": "w0/registration.json", "digest": "04" * 64}]
    (tmp_path / "before.json").write_text(json.dumps(w, indent=1) + "\n")
    M.write(tmp_path / "after.json", "H1Registration", (json.dumps(w, indent=1) + "\n").encode())
    assert (tmp_path / "after.json").read_bytes() == (tmp_path / "before.json").read_bytes()
    assert json.loads(M.read(tmp_path / "after.json", "H1Registration")) == w
    for line in lines:
        with open(tmp_path / "before.jsonl", "a") as fh:
            fh.write(json.dumps(line) + "\n")
        M.append(tmp_path / "after.jsonl", "H1Registration", json.dumps(line).encode())
    assert (tmp_path / "after.jsonl").read_bytes() == (tmp_path / "before.jsonl").read_bytes()
    assert [json.loads(x) for x in M.lines(tmp_path / "after.jsonl", "H1Registration")] == lines


def test_registered_weights_reading_unchanged(tmp_path):
    reg = {"schema": "verity-vllm/weights-registration/v1", "roots": {"model.embed": "05" * 64}, "note": "é"}
    (tmp_path / "weights_registration.json").write_text(json.dumps(reg, indent=1))
    path = tmp_path / "weights_registration.json"
    assert json.loads(M.read(path, "RegisteredWeights")) == json.loads(path.read_text()) == reg


def test_request_program_bytes_unchanged(tmp_path):
    docs = {"compiled.json": {"refused": "the advice's carry is refused: é"},
            "manifest.json": {"programs": [{"request": "e0", "descriptor": "descriptor.json.gz"}]}}
    for name, doc in docs.items():
        (tmp_path / f"before-{name}").write_text(json.dumps(doc, indent=1) + "\n")
        M.write(tmp_path / f"after-{name}", "RequestProgram", (json.dumps(doc, indent=1) + "\n").encode())
        assert (tmp_path / f"after-{name}").read_bytes() == (tmp_path / f"before-{name}").read_bytes()
