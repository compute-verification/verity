"""The post's configuration (`record.json`) and a serve's stamp against it (`record.py`). The fixtures are two serves'
stored `meta.json`, each reduced to its configuration fields: one at the post's context length with no network
certifier (the serve tax's `boundary` arm), one under the certified tick at a shorter context with prefix caching off
(another harness, which records neither its decoding nor an execution phase)."""
import hashlib
import json
from pathlib import Path

import pytest
import record

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SERVE = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20]


def meta(name: str) -> dict:
    return json.loads((FIXTURES / f"meta.{name}.json").read_text())


def test_the_record_has_the_22_switches_and_the_posts_value_of_each():
    rows = record.switches()
    assert [row["n"] for row in rows] == list(range(1, 23))
    assert all(row["name"] and row["post"] is not None for row in rows)
    assert [row["n"] for row in rows if row["run"] == "serve"] == SERVE
    assert [row["n"] for row in rows if row["run"] == "proving"] == [18, 19, 21]
    assert [row["n"] for row in rows if row["run"] == "planner"] == [3, 22]
    assert {row["read"] for row in rows} - {None} == set(record.READERS)


@pytest.mark.parametrize("name,at,off", [
    ("context-131072", [1, 2, 4, 5, 7, 10], [6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 20]),
    ("tick-40960", [1, 2, 5], [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20]),
])
def test_a_stored_serves_off_list(name, at, off):
    made = record.stamp(meta(name))
    assert made["off"] == off
    assert sorted(set(SERVE) - set(off)) == at
    assert list(made) == ["digest", "observed", "off"]
    assert made["digest"] == hashlib.sha512(record.RECORD.read_bytes()).hexdigest()


def test_what_each_serve_shows():
    assert record.stamp(meta("context-131072"))["observed"] == {
        "1": "aca966e4e02791568aa6a4ced368624b3d897f42", "2": "checkpoint", "4": 131072, "5": 8,
        "6": {"commit": "d9105ea80"}, "7": "on", "8": "greedy", "10": 600.0, "20": {"tick": False}}
    assert record.stamp(meta("tick-40960"))["observed"] == {
        "1": "aca966e4e02791568aa6a4ced368624b3d897f42", "2": "checkpoint", "4": 40960, "5": 8,
        "6": {"commit": "d9105ea80"}, "7": "off", "20": {"tick": True}}


def test_a_switch_shown_only_in_part_or_by_no_reader_stays_off():
    """A serve with every switch its `meta.json` can show at the post's value still has off the two it shows in part
    (the engine unmodified, the certifier in front of it) and the seven nothing in it gives."""
    m = meta("context-131072")
    m.update(sampling="seeded", plugin=["record", "boundary_rows"], tick={"bucket_us": 250000})
    made = record.stamp(m)
    assert made["off"] == [6, 9, 11, 12, 13, 14, 15, 16, 20]
    assert made["observed"]["8"] == "sampled" and made["observed"]["17"] == "two-stage"
    assert made["observed"]["20"] == {"tick": True}


@pytest.mark.parametrize("change,switch,value", [
    (lambda m: m["server_argv"].extend(["--dtype", "bfloat16"]), 2, "bfloat16"),
    (lambda m: m["server_argv"].append("--max-model-len=4096"), 4, 4096),
    (lambda m: m["server_argv"].remove("--tensor-parallel-size"), 5, None),
    (lambda m: m["versions"].update(vllm="0.28.1"), 6, None),
    (lambda m: m["server_argv"].append("--no-enable-prefix-caching"), 7, "off"),
    (lambda m: m["server_argv"].extend(["--no-enable-prefix-caching", "--enable-prefix-caching"]), 7, "on"),
    (lambda m: m.pop("sampling"), 8, None),
    (lambda m: m.update(sampling={"requests": 651}), 8, None),
    (lambda m: m.update(window_s=3600.0), 10, 3600.0),
    (lambda m: m.pop("tick"), 20, None),
])
def test_a_changed_serve_changes_its_switch(change, switch, value):
    m = meta("context-131072")
    before = record.stamp(m)
    change(m)
    made = record.stamp(m)
    assert made["observed"].get(str(switch)) == value
    assert (switch in made["off"]) == (value != record.switches()[switch - 1]["post"])
    assert {k: v for k, v in made["observed"].items() if k != str(switch)} == \
        {k: v for k, v in before["observed"].items() if k != str(switch)}


def test_the_checkpoint_is_read_from_the_model_path_where_no_model_is_recorded():
    m = meta("context-131072")
    del m["model"]
    assert record.stamp(m)["observed"]["1"] == "aca966e4e02791568aa6a4ced368624b3d897f42"
    m["server_argv"][m["server_argv"].index("--model") + 1] = "/models/glm"
    made = record.stamp(m)
    assert "1" not in made["observed"] and 1 in made["off"]


def test_a_stamp_holds_no_request_count_or_time():
    """What a serve records about its requests, its load and its clock changes nothing in the stamp and is nowhere in
    it; nor is the machine, the run or a path."""
    m = meta("context-131072")
    plain = record.stamp(m)
    m.update(loads=[{"requests": 651, "wall_s": 1496.25, "t0_unix": 1794167899.5}], rates=[0.4375], startup_s=612.5,
             host="node-a", run_id="run-7", checkpoint="/models/glm", server_stop="stopped at 19:58:19",
             schedule=[{"at": 1794167899.5, "requests": 3000}], env={"TRACE_DIR": "/runs/run-7/out"})
    made = record.stamp(m)
    assert made == plain
    text = json.dumps({"observed": made["observed"], "off": made["off"]})
    for private in ("651", "1496", "1794167899", "0.4375", "612.5", "node-a", "run-7", "/models", "/workspace", "52719"):
        assert private not in text


@pytest.mark.parametrize("change", [
    lambda m: m.pop("server_argv"),
    lambda m: m.update(server_argv=["stand_in.py", "--port", "8000"]),
    lambda m: m.update(server_argv="-m vllm.entrypoints.openai.api_server"),
])
def test_it_refuses_a_record_that_is_no_serves(change):
    m = meta("tick-40960")
    change(m)
    with pytest.raises(ValueError, match="not a serve's meta.json"):
        record.stamp(m)
    with pytest.raises(ValueError, match="not a serve's meta.json"):
        record.stamp([m])


@pytest.mark.parametrize("change,message", [
    (lambda rows: rows.pop(4), "row 5 needs n = 5"),
    (lambda rows: rows[2].pop("post"), "row 3 needs n = 3"),
    (lambda rows: rows[0].update(run="audit"), "row 1 needs n = 1"),
    (lambda rows: rows[3].update(read="clock"), "switch 4 reads 'clock'"),
    (lambda rows: rows[17].update(read="engine"), "switch 18 reads 'engine'"),
])
def test_the_record_is_checked(change, message):
    rows = json.loads(record.RECORD.read_bytes())["switches"]
    change(rows)
    with pytest.raises(ValueError, match=message):
        record.switches(json.dumps({"switches": rows}).encode())


def test_the_command_prints_the_off_list_and_writes_the_stamp(tmp_path, capsys):
    record.main(["stamp", str(FIXTURES / "meta.context-131072.json"), "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "at the post's value: 1, 2, 4, 5, 7, 10\n" in out
    assert "off: 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 20\n" in out
    assert json.loads((tmp_path / "out" / "record.json").read_text()) == record.stamp(meta("context-131072"))
    (tmp_path / "meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="not a serve's meta.json"):
        record.main(["stamp", str(tmp_path / "meta.json")])


def test_the_command_never_writes_a_stamp_over_the_record():
    before = record.RECORD.read_bytes()
    with pytest.raises(SystemExit, match="the stamp would replace record.json"):
        record.main(["stamp", str(FIXTURES / "meta.context-131072.json"), "--out", str(record.HERE)])
    assert record.RECORD.read_bytes() == before
