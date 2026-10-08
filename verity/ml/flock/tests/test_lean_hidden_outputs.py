"""Hidden outputs (PROTOCOL.md §16.13): the Lean verifier reads M0's hidden-output statements, `verity/flock-circuit` and
`verity/flock-circuit/types`, as the Rust prover writes them, on its recorded CPU selftest sessions: flat RoPE (the unit
returns its output), flat GEMM (a tail stage returns it, one BF16 word in a 64-word row), typed RoPE (a flat class) and typed
GEMM (a template). Each public file holds only roots, hiding leaves and structure.

It is the one test in `check` of the default statements against the prover's sessions: check's agreement step replays only
upstream's pinned builds, none of which has hidden outputs. So it catches a verifier that disagrees with the prover on the
output rows (their Δ copies, regions, public file or identity), which no statement-only test sees. Only the cases about the
public statement are verified (the honest session, which pins the statement digest and Σ to the prover's, and the false
statements, each under its own public file): a whole set takes minutes. `vectors.json`'s `live.sets` carry every case of
the same records (`ci.py --live`).

The sets are a fetched fixture (`fixtures/artifacts.json`), which `suites.py` fetches before the suite runs and fails the suite
when it cannot, so a missing set fails this test rather than skipping it."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: the four sets (`flat/` and `typed/`, each `rope-head/` and `gemm-coordinate/`: the statement's `stage/` files and CASES'
#: `records-cpu/`), recorded by 60-circuit.sh MODE=selftest GPU=0 RECORD=1 SELF_N=4 (TYPED=1 for `typed/`) at 455a01049
#: (`art:dae0b537`, the same bytes)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hidden-outputs"
SETS = {("flat", "rope-head"): "verity/flock-circuit@f5df3bbc+seed-injection",
        ("flat", "gemm-coordinate"): "verity/flock-circuit@f5df3bbc+seed-injection",
        ("typed", "rope-head"): "verity/flock-circuit/types@f5df3bbc+seed-injection",
        ("typed", "gemm-coordinate"): "verity/flock-circuit/types@f5df3bbc+seed-injection"}
CASES = {"honest": True, "output_claim_false": False, "output_row_committed_false": False, "digest_claim_false": False}

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def store():
    assert all((FIXTURE / f / n / "stage" / "circuit.txt").is_file() for f, n in SETS), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/hidden-outputs`")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return FIXTURE


@pytest.mark.parametrize("form,name", list(SETS))
def test_the_public_file_holds_no_output_word(store, form, name):
    d = store / form / name
    meta = json.loads((d / "stage" / "circuit.txt").read_text().split("\n", 2)[1][5:])
    head, body = (d / "stage" / "pub-4.bin").read_bytes().split(b"\n", 1)
    head = json.loads(head)
    assert meta["input_ports"] + len(meta["out_ports"]) == len(meta["ports"]) and "content_digest" not in head
    assert set(head["frame_v3"]["schemas"].values()) == {"hm96-sha512/row/v1"} and len(body) == 4 * len(meta["ports"]) * 128


@pytest.mark.parametrize("form,name", list(SETS))
def test_it_gives_the_provers_sessions_their_verdicts(store, form, name, tmp_path):
    d = store / form / name
    subprocess.run([sys.executable, str(VERIFIER / "selftest_records.py"), str(d / "records-cpu"), str(tmp_path)], check=True,
                   capture_output=True)
    got = {}
    for case in CASES:
        s = tmp_path / case
        live = json.loads((s / "case.json").read_text())["live"]
        assert live["accepted"] == CASES[case], live
        pub = s / "pub.bin" if (s / "pub.bin").exists() else d / "stage" / "pub-4.bin"
        assert (s / "pub.bin").exists() == (case != "honest")
        tables = ["--tables", str(d / "stage" / "tables")] if (d / "stage" / "tables").exists() else []
        p = subprocess.run([str(BIN), "verify", "--statement", SETS[form, name], "--circuit", str(d / "stage" / "circuit.txt"),
                            "--public", str(pub), *tables, "--session", str(s)], capture_output=True, text=True)
        [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
        got[case] = v
    assert {c: v["accepted"] for c, v in got.items()} == CASES, {c: v["why"] for c, v in got.items()}
    # each false statement is refused by the proof against its own public file, not by the session parameters
    assert all(not v["why"].startswith("S2") for c, v in got.items() if c != "honest"), {c: v["why"] for c, v in got.items()}


def test_a_tail_with_no_unit_is_refused(store, tmp_path):
    """`units_per_vu = 0` is refused as the circuit is read (PROTOCOL.md §16.10, D8; the Rust side is `circuit.rs`'s
    `a_tail_with_no_unit_is_refused`). Flat GEMM's tail returns the outputs, so its unit count need not be a power of two;
    the rest is kept consistent: no unit or stage slot, no leaf map or cut, no wire but the rows'."""
    d = store / "flat" / "gemm-coordinate" / "stage"
    head, meta, rest = (d / "circuit.txt").read_text().split("\n", 2)
    m = json.loads(meta[len("META "):])
    stages = [n for n in m["per_vu"] if n.startswith("stage")]
    assert m["units_per_vu"] > 0 and m["out_net"] in stages
    m["units_per_vu"] = 0
    for r in m["ranges"]:
        if r["net"] in ["unit", *stages]:
            r["count"] = 0
    m["per_vu"].update(dict.fromkeys(["unit", *stages], 0))
    m.update(leaves_in=[], leaves_out=[], leaf_cuts=[], wires=[w for w in m["wires"] if w[0] in ("sha512x3", "hm96")])
    bad = tmp_path / "circuit.txt"
    bad.write_text(head + "\nMETA " + json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n" + rest)
    p = subprocess.run([str(BIN), "statement", "--statement", SETS["flat", "gemm-coordinate"], "--circuit", str(bad),
                        "--public", str(d / "pub-4.bin")], capture_output=True, text=True)
    assert p.returncode == 1 and "units_per_vu is 0: a VU of no unit" in p.stdout, (p.stdout, p.stderr)


def test_a_block_with_no_vu_is_refused(store, tmp_path):
    """`vus_per_block = 0` is refused as the circuit is read (PROTOCOL.md §16.10, D9; the Rust side is `circuit.rs`'s
    `a_block_with_no_vu_is_refused`). Flat GEMM's tail returns the outputs, so its unit count need not be a power of two;
    the rest is kept consistent: every range has no slot."""
    d = store / "flat" / "gemm-coordinate" / "stage"
    head, meta, rest = (d / "circuit.txt").read_text().split("\n", 2)
    m = json.loads(meta[len("META "):])
    assert m["vus_per_block"] > 0 and m["out_net"] != "unit"
    m["vus_per_block"] = 0
    for r in m["ranges"]:
        if r["type"] == "unit":
            r["count"] = 0
    bad = tmp_path / "circuit.txt"
    bad.write_text(head + "\nMETA " + json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n" + rest)
    p = subprocess.run([str(BIN), "statement", "--statement", SETS["flat", "gemm-coordinate"], "--circuit", str(bad),
                        "--public", str(d / "pub-4.bin")], capture_output=True, text=True)
    assert p.returncode == 1 and "vus_per_block is 0: a block of no VU" in p.stdout, (p.stdout, p.stderr)
