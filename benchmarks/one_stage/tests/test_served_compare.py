"""served_compare.py: the served compare's verdicts (PHI, BUILD, CUT), each failing where its file can't be read, and the
line the record names. The lowering itself (`lowered_facts`) is stubbed: a real one reads the row's workload file and
lowers the 1B's window circuit twice."""
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_compare as SC  # noqa: E402

SCHEME = "pearl-c-h100-v1-h3"
PROGRAM, MANIFEST = "ab" * 64, "cd" * 64
PARAMS = {"pearl-a/0/0": 2048, "pearl-y/0/0": 128, "weights/model.layers.0.mlp.down_proj": 4096}
FIELDS = {"model": ["m", "m"], "revision": ["r", "r"], "hf_config": ["c", "c"], "execution": ["bi-eager", "bi-eager"],
          "default_seed": [7, 7], "target": ["device_name"]}


def _facts(digest: str = "ee" * 64) -> dict:
    return {"templates": [["LinearARow_v2", 4]], "params": PARAMS, "calls": [[2048, 8192, 64]], "digest": digest}


def _w(tmp_path: Path, case: str) -> Path:
    a = {"value": "pearl-a/0/0", "leaves": 3 if case == "size" else 4, "words": 256,
         "word_bits": 16 if case == "bits" else 32}
    y = {"value": "pearl-y/0/0", "leaves": 1, "words": 64, "word_bits": 32}
    registered = [a, *([] if case == "missing" else [y]), *([{**a, "value": "pearl-a/1/0"}] if case == "extra" else [])]
    (w := tmp_path / "w.json").write_text(json.dumps(
        {"window": {"scheme": "pearl-c-h100-v1" if case == "scheme" else SCHEME}, "registered": registered}))
    return w


@pytest.mark.parametrize("case", ["exact", "missing", "size", "extra", "scheme", "bits"])
def test_cut_holds_w_to_exactly_the_circuits_pearl_values_at_their_sizes(tmp_path, case):
    """CUT: W registers exactly the circuit's Pearl+ values (never its weights), each at its size in the circuit (two u16
    words an FP32 word), under the scheme, in 32-bit words; a W that can't be read fails, naming it."""
    v = SC.cut(_w(tmp_path, case), PARAMS, SCHEME)
    assert v["ok"] is (case == "exact") and v["circuit_values"] == 2
    assert v["differ"] == {"exact": [], "missing": ["pearl-y/0/0"], "size": ["pearl-a/0/0"], "extra": ["pearl-a/1/0"],
                           "scheme": [], "bits": []}[case]
    with pytest.raises(ValueError, match=re.escape(f"{tmp_path / 'gone.json'}: FileNotFoundError")):
        SC.cut(tmp_path / "gone.json", PARAMS, SCHEME)


def test_build_holds_pred_to_the_rows_build_of_record(tmp_path):
    """BUILD: predict row's predicted.json names the row's Build of record, its Program and manifest digests."""
    (tmp_path / "predicted.json").write_text(json.dumps({"program_digest": PROGRAM, "manifest_digest": MANIFEST}))
    assert SC.build(tmp_path, f"{PROGRAM}:{MANIFEST}")["ok"] is True
    assert SC.build(tmp_path, f"{PROGRAM}:{'ef' * 64}") == {"ok": False, "names": [PROGRAM, MANIFEST],
                                                           "of_record": [PROGRAM, "ef" * 64]}
    with pytest.raises(ValueError, match="is PROGRAM_DIGEST:MANIFEST_DIGEST"):
        SC.build(tmp_path, PROGRAM)


@pytest.mark.parametrize("differs", [None, "revision", "circuit"])
def test_phi_takes_every_field_and_the_circuit_and_names_a_target_the_circuit_doesnt_read(differs):
    """PHI: every φ field equal and the two circuits' facts one digest; a target profile field that differs is named,
    since the circuit's digest already says whether the circuit reads it."""
    fields = {**FIELDS, **({"revision": ["r", "r2"]} if differs == "revision" else {})}
    v = SC.phi(fields, _facts(), _facts("ff" * 64 if differs == "circuit" else "ee" * 64))
    assert v["ok"] is (differs is None) and v["differ"] == ([] if differs is None else [differs])
    assert v["target_differs"] == ["device_name"] and (v["templates"], v["values"]) == (1, 3)


def _main(tmp_path: Path, monkeypatch, declared: str = "ee" * 64, config: bool = True) -> tuple[int, Path]:
    monkeypatch.setattr(SC, "lowered_facts", lambda serve, row, cfg, cls, scheme: (FIELDS, _facts(), _facts(declared)))
    (serve := tmp_path / "serve").mkdir()
    (serve / "serve.json").write_text(json.dumps({"model": "m", "revision": "r"}))
    (pred := tmp_path / "predicted").mkdir()
    cfg = tmp_path / ("config.json" if config else "gone.json")
    if config:
        cfg.write_text("{}")
    (pred / "predicted.json").write_text(json.dumps({"program_digest": PROGRAM, "manifest_digest": MANIFEST,
                                                     "config": str(cfg)}))
    (w := serve / "pearlplus" / "w7" / "registration.json").parent.mkdir(parents=True)
    w.write_text(_w(tmp_path, "exact").read_text())
    out = tmp_path / "out"
    rc = SC.main(["--serve", str(serve), "--row", "row-1", "--class", "4,32,4", "--pearl-tables", SCHEME, "--pred",
                  str(pred), "--of-record", f"{PROGRAM}:{MANIFEST}", "--window-record", str(w), "--out", str(out)])
    return rc, out


def test_the_compare_says_each_verdict_and_ends_on_the_line_the_record_names(tmp_path, monkeypatch, capsys):
    """Each verdict's line, then the SERVED-COMPARE line the record names, kept in served_compare.json beside the two
    circuits' facts."""
    rc, out = _main(tmp_path, monkeypatch)
    lines = capsys.readouterr().out.splitlines()
    assert rc == 0 and [x.split(":")[0] for x in lines] == ["PHI", "BUILD", "CUT", "SERVED-COMPARE"]
    assert all(x.endswith(": PASS") for x in lines[:3]), lines
    doc = json.loads((out / "served_compare.json").read_text())
    assert lines[-1] == doc["line"] == ("SERVED-COMPARE: EQUAL (the serve against row row-1's Build of record: PHI, BUILD "
                                        "and CUT PASS; B-OF-RECORD is the seal's)") and doc["ok"] is True
    assert json.loads((out / "circuit_facts.json").read_text()) == {"served": _facts(), "declared": _facts()}


def test_a_circuit_the_row_lowers_otherwise_is_not_equal_and_cut_holds_w_to_the_served_one(tmp_path, monkeypatch,
                                                                                            capsys):
    """A row whose declared φ lowers another circuit fails PHI, and CUT still holds W to the circuit the serve lowers."""
    rc, _out = _main(tmp_path, monkeypatch, declared="ff" * 64)
    lines = capsys.readouterr().out.splitlines()
    assert rc == 1 and "is not row row-1's declared φ, differing in circuit (" in lines[0], lines
    assert lines[0].endswith(": FAIL") and lines[2].endswith(": PASS"), lines
    assert lines[-1].startswith("SERVED-COMPARE: NOT EQUAL (the serve against row row-1's Build of record: PHI FAIL; ")


def test_a_pred_naming_no_config_here_fails_phi_and_cut_naming_why(tmp_path, monkeypatch, capsys):
    """A predicted.json whose checkpoint config isn't on this node lowers nothing: PHI says so, naming both, and CUT has
    no circuit to hold W to."""
    rc, _out = _main(tmp_path, monkeypatch, config=False)
    lines = capsys.readouterr().out.splitlines()
    assert rc == 1 and lines[0] == (f"PHI: can't be reached (ValueError: {tmp_path / 'predicted'}/predicted.json names no "
                                    f"checkpoint config on this node ({tmp_path / 'gone.json'})): FAIL")
    assert lines[2] == "CUT: can't be reached (PHI lowered no circuit to hold W to): FAIL"
    assert "PHI, CUT FAIL" in lines[-1]
