"""The read API over definitions: identities, order, sample, shuffle, splits, the refusals, and the one generator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import input_datasets as D



def _define(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **files: str) -> None:
    (defs := tmp_path / "definitions").mkdir()
    for name, text in files.items():
        (defs / f"{name}.toml").write_text(text)
    monkeypatch.setattr(D, "DEFINITIONS", defs)
    monkeypatch.setattr(D, "CACHE", tmp_path)


GENERATED = """kind = "completion-requests"
[generated]
generator = "fixed-window-completions"
version = 1
seed = {seed}
[generated.parameters]
model = "m"
count = {count}
window_steps = 2
prompt_tokens = [4, 9]
heads = [[1, 10, 11], [1, 20, 21, 22]]
tail = [7, 8]
filler = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
[splits]
head = [0, 2]
rest = [2, {count}]
"""


def test_an_identity_is_stable_and_changes_with_any_parameter(monkeypatch, tmp_path):
    _define(monkeypatch, tmp_path, a=GENERATED.format(seed=1, count=5), b=GENERATED.format(seed=2, count=5),
            c=GENERATED.format(seed=1, count=6), d=GENERATED.format(seed=1, count=5).replace('"m"', '"n"'))
    ids = {n: D.open(n).identity for n in "abcd"}
    assert D.open("a").identity == ids["a"]
    assert len(set(ids.values())) == 4
    assert ids["a"].startswith("generated:fixed-window-completions/v1:seed=1:parameters-sha256=")


def test_iteration_sample_and_shuffle_are_deterministic(monkeypatch, tmp_path):
    _define(monkeypatch, tmp_path, a=GENERATED.format(seed=1, count=8))
    ds = D.open("a")
    records = list(ds)
    assert records == list(D.open("a")) and len(ds) == 8
    s, t = ds.sample(3, seed=5), ds.shuffled(seed=5)
    assert list(s) == list(D.open("a").sample(3, seed=5)) and list(t) == list(D.open("a").shuffled(seed=5))
    assert [records.index(r) for r in s] == sorted(records.index(r) for r in s)
    assert sorted(map(json.dumps, t)) == sorted(map(json.dumps, records)) and list(t) != records
    assert s.identity == f"{ds.identity}|sample:n=3,seed=5" and t.identity == f"{ds.identity}|shuffle:seed=5"
    assert ds.sample(3, seed=6).identity != s.identity and list(ds.sample(3, seed=6)) != list(s)


def test_a_split_is_the_range_its_definition_names(monkeypatch, tmp_path):
    _define(monkeypatch, tmp_path, a=GENERATED.format(seed=1, count=5))
    ds = D.open("a")
    assert sorted(ds.splits) == ["head", "rest"]
    assert list(D.open("a", split="head")) == list(ds)[0:2] and list(ds.splits["rest"]) == list(ds)[2:5]
    assert D.open("a", split="rest").identity == f"{ds.identity}|split:rest=2:5"


def test_an_unknown_name_an_unknown_split_and_a_changed_byte_are_refused_by_name(monkeypatch, tmp_path):
    (tmp_path / "r.jsonl").write_bytes(b'{"x": 1}\n')
    held, digest = (hashlib.sha256(b).hexdigest() for b in (b'{"x": 1}\n', b'{"x": 2}\n'))
    _define(monkeypatch, tmp_path, a=GENERATED.format(seed=1, count=5),
            r=f'kind = "k"\n[recorded]\nsource = "art:00"\npath = "r.jsonl"\nsha256 = "{digest}"\nformat = "jsonl"\n')
    with pytest.raises(D.Refused) as e:
        D.open("nope")
    assert (e.value.name, str(e.value)) == ("unknown-dataset", f"no input dataset 'nope': each is a definition NAME.toml under {D.DEFINITIONS}")
    with pytest.raises(D.Refused) as e:
        D.open("a", split="tail")
    assert (e.value.name, str(e.value)) == ("unknown-split", "input dataset 'a' has no split 'tail' (its splits: head, rest)")
    with pytest.raises(D.Refused) as e:
        list(D.open("r"))
    assert (e.value.name, str(e.value)) == ("digest", f"input dataset 'r': r.jsonl has sha256 {held}, not its definition's "
                                                      f"{digest}; fetch it again with `research data fetch-fixtures r.jsonl`")


def test_the_recorded_workload_reads_as_its_requests():
    """Through the fixture's fetch: the runner fetches the file, an entry of artifacts.json under the suite's inputs."""
    ds = D.open("qwen05_32x16_1req")
    [record] = list(ds)
    assert ds.kind == "completion-requests"
    assert ds.identity == "recorded:sha256=ea93af44cb5d59bb7da850d088e0cc01ba8c11332282c4d4f06a08ce341a261a:json:requests"
    assert (record["request_id"], record["prompt_len"], len(record["prompt_token_ids"])) == ("728a5686-cfb6-58db-a1a8-170c12d608a6", 32, 32)


def test_the_one_window_load_is_one_request_of_bos_alone_and_one_step():
    ds = D.open("llama32_1b_fixed_1_1_1")
    [record] = list(ds)
    assert ds.kind == "completion-requests"
    assert (record["model"], record["prompt"], record["max_tokens"], record["temperature"]) == ("unsloth/Llama-3.2-1B", [128000], 1, 0.0)


def test_a_timed_view_pairs_each_record_with_its_time_and_is_timed_once(monkeypatch, tmp_path):
    _define(monkeypatch, tmp_path, a=GENERATED.format(seed=1, count=5))
    ds = D.open("a")
    timed = ds.timed("rate", seed=3, rate=2.0)
    pairs = list(timed)
    assert pairs == list(D.open("a").timed("rate", seed=3, rate=2.0)) and len(timed) == 5 and timed.kind == ds.kind
    assert [r for _, r in pairs] == list(ds) and [t for t, _ in pairs] == sorted(t for t, _ in pairs) and pairs[0][0] > 0
    assert [t for t, _ in ds.timed("at-once")] == [0.0] * 5
    assert timed.identity == f"{ds.identity}|timed:rate,rate=2.0,seed=3"
    assert len({ds.timed("at-once").identity, ds.timed("at-once", seed=9).identity}) == 1
    assert len({timed.identity, ds.timed("rate", seed=4, rate=2.0).identity, ds.timed("rate", seed=3, rate=3.0).identity,
                ds.timed("at-once").identity}) == 4
    for view in (ds.splits["head"], ds.sample(2, seed=1), ds.shuffled(seed=1)):
        assert len(view.timed("rate", seed=3, rate=2.0)) == len(view)
    for again in (timed, timed.sample(2, seed=1)):
        with pytest.raises(D.Refused) as e:
            again.timed("at-once")
        assert (e.value.name, str(e.value)) == ("timed-twice", f"{again.identity} is timed already: time the dataset it was timed from")
