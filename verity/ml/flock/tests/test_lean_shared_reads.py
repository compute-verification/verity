"""Shared reads (META `shared_reads`, PROTOCOL.md §16.10): the Lean verifier (`Flock.HmIn.checkReads`) and its Python mirror
(`circuit.check_shared_reads`, which `circuit.write` runs) refuse the same files with the same refusal, and pass the same
ones. A shared read is form (B)'s one writer a window: input row `port` is one table row per group of `group` instances,
instance i reading table row ⌊i/G⌋, and its public input `position` is i mod G (seed `instance`, derivation `mod:G`).

Each case is `test_lean_public_inputs.py`'s fetched set (4 RoPE instances with public input port `cs`, input row `x`, output
row `out`) restaged as a shared-row file whose `x` table and refs the case names, with META naming the shared read and
`cs` as its position. It catches a verifier or a writer that lets a shared row's windows go to the wrong instances (a ref
off its group, a table of the wrong size, a position that is not the group index), or that the two disagree on, which no
statement-only test of either sees. Each case sets up the statement once (seconds)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "public-inputs"
STATEMENT = "verity/flock-circuit@f5df3bbc+seed-injection"
N, BC = 4, 128

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


@pytest.fixture(scope="module")
def store():
    assert (FIXTURE / "stage" / "circuit.txt").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/{FIXTURE.name}`")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return FIXTURE


def _split(f: Path) -> tuple[dict, bytes]:
    head, body = f.read_bytes().split(b"\n", 1)
    return json.loads(head), body


def _join(head: dict, body: bytes, f: Path) -> Path:
    f.write_bytes(json.dumps(head, sort_keys=True).encode() + b"\n" + body)
    return f


def _stage(d: Path, tmp: Path, reads, group: int = 1, rows=None, refs=None, *, seed: str = "instance",
           derivation: str | None = None, public: bool = True, tables: bool = True):
    """The set restaged: META naming `reads` (absent: no key) with `cs`'s seed and derivation, a shared-row public file whose
    `x` table is the instances' rows `rows` and whose refs are `refs` (`tables=False`: the per-instance file), and the
    verifier's own file naming the same ports. Returns the circuit, the public file, the own file, and the mirror's
    arguments (META, the tables' counts and refs)."""
    from verity_flock import circuit as FC
    rows = list(range(N)) if rows is None else rows
    refs = list(range(N)) if refs is None else refs
    head, meta_line, rest = (d / "stage" / "circuit.txt").read_text().split("\n", 2)
    meta = json.loads(meta_line[5:])
    words = 2 * N * sum(p["words"] for p in meta["public_inputs"])
    if reads is not None:
        meta["shared_reads"] = reads
    meta["public_inputs"][0].update(seed=seed, derivation=derivation or f"mod:{group}")
    if not public:
        meta.pop("public_inputs")
        meta.pop("leaves_public")
    text = head + "\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n" + rest
    (tmp / "circuit.txt").write_text(text)
    h, body = _split(d / "stage" / "pub-4.bin")
    h["circuit_sha512"] = FC.digest(text)
    h["units"]["classes"] = [FC.class_digest(text)] * N
    bcs, tail = body[:-words], body[-words:]
    assert len(bcs) == 2 * N * BC

    def bc(i: int, p: int) -> bytes:
        return bcs[(2 * i + p) * BC:(2 * i + p + 1) * BC]
    if public:
        h["public_inputs"] = meta["public_inputs"]
    else:
        h.pop("public_inputs")
        tail = b""
    if tables:
        h["shared_rows"] = {"x": len(rows), "out": N}
        body = (b"".join(bc(i, 0) for i in rows) + b"".join(bc(i, 1) for i in range(N))
                + np.array([[r, i] for i, r in enumerate(refs)], dtype="<u4").tobytes() + tail)
    else:
        body = bcs + tail
    pub = _join(h, body, tmp / "pub.bin")
    own = None
    if public:
        oh, ob = _split(d / "stage" / "public-inputs-4.bin")
        oh["ports"] = meta["public_inputs"]
        own = _join(oh, ob, tmp / "own.bin")
    mirror = (meta, {"x": len(rows)} if tables else None, {"x": np.array(refs)} if tables else None)
    return tmp / "circuit.txt", pub, own, mirror


def _statement(circuit: Path, pub: Path, own: Path | None) -> subprocess.CompletedProcess:
    flags = ["--public-inputs", str(own)] if own else []
    return subprocess.run([str(BIN), "statement", "--statement", STATEMENT, "--circuit", str(circuit), "--public", str(pub),
                           *flags], capture_output=True, text=True, check=False)


def _mirror(meta: dict, counts, refs) -> str | None:
    from verity_flock import circuit as FC
    try:
        FC.check_shared_reads(meta, N, counts, refs)
    except ValueError as e:
        return str(e)
    return None


def _read(group: int, port: str = "x", position: str = "cs") -> list[dict]:
    return [{"port": port, "group": group, "position": position}]


#: each refusal: the case's `_stage` arguments, and the refusal both give
REFUSALS = {
    "ref_off_its_group": ((_read(2), 2, [0, 2], [0, 1, 0, 1]), {},
                          "shared read x: instance 1 reads table row 1, not its group's 0"),
    "last_ref_off_its_group": ((_read(2), 2, [0, 2], [0, 0, 1, 0]), {},
                               "shared read x: instance 3 reads table row 0, not its group's 1"),
    "table_a_row_each": ((_read(2), 2, [0, 1, 2, 3], [0, 0, 1, 1]), {},
                         "shared read x: its table is not one row per group of 2 (2 rows)"),
    "instances_not_whole_groups": ((_read(3), 3, [0, 3], [0, 0, 0, 1]), {},
                                   "shared read x: 4 instances are not whole groups of 3"),
    "position_another_modulus": ((_read(2), 2, [0, 2], [0, 0, 1, 1]), {"derivation": "mod:4"},
                                 ("shared read x: its position cs is not the instance index mod 2 (seed instance, "
                                  "derivation mod:2)")),
    "position_another_seed": ((_read(2), 2, [0, 2], [0, 0, 1, 1]), {"seed": "rope/positions"},
                              ("shared read x: its position cs is not the instance index mod 2 (seed instance, "
                               "derivation mod:2)")),
    "position_not_public": ((_read(2, position="j"), 2, [0, 2], [0, 0, 1, 1]), {},
                            "shared read x: its position j is not a public input"),
    "port_an_output_row": ((_read(1, port="out"), 1), {}, "shared read out: not an input row"),
    "port_the_public_input": ((_read(1, port="cs"), 1), {}, "shared read cs: not an input row"),
    "port_named_twice": ((_read(1) * 2, 1), {}, "shared read x: named twice"),
    "group_zero": ((_read(0), 1), {},
                   "META: a shared read names no port, position, or group (a positive count of instances)"),
    "group_not_a_count": (([{"port": "x", "group": "2", "position": "cs"}], 1), {},
                          "META: a shared read names no port, position, or group (a positive count of instances)"),
    "another_key": (([{"port": "x", "group": 1, "position": "cs", "window": 0}], 1), {},
                    "META: a shared read names no port, position, or group (a positive count of instances)"),
    "not_an_object": ((["x"], 1), {}, "META: a shared read is not an object"),
    "not_a_list": (({"x": 1}, 1), {}, "META shared_reads: not a list"),
    "no_port": (([], 1), {}, "META: shared_reads names no port"),
    "per_instance_file": ((_read(1), 1), {"tables": False},
                          "shared reads: the public file is not a shared-row file (header shared_rows)"),
    "without_public_inputs": ((_read(1), 1), {"public": False},
                              "META: shared_reads without public_inputs (a shared read's position is a public input)"),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_both_refuse_what_is_not_one_writer_a_window(store, name, tmp_path):
    args, kw, why = REFUSALS[name]
    circuit, pub, own, mirror = _stage(store, tmp_path, *args, **kw)
    p = _statement(circuit, pub, own)
    assert p.returncode == 1 and p.stdout.strip() == f"setup: {why}", (p.stdout, p.stderr)
    assert _mirror(*mirror) == why


def test_both_pass_one_writer_a_window(store, tmp_path):
    """A group of one (each instance its own table row) sets up in full, under the header's roots; at groups of two the
    check passes in both, and the verifier goes on to the next check, the `x` table's root (the header's is over the
    instances' rows), which the restaged file doesn't meet."""
    circuit, pub, own, mirror = _stage(store, tmp_path, _read(1), 1)
    p = _statement(circuit, pub, own)
    assert p.returncode == 0 and json.loads(p.stdout)["regions"] == 3, (p.stdout, p.stderr)
    assert _mirror(*mirror) is None
    (tmp_path / "two").mkdir()
    circuit, pub, own, mirror = _stage(store, tmp_path / "two", _read(2), 2, [0, 2], [0, 0, 1, 1])
    p = _statement(circuit, pub, own)
    assert p.returncode == 1 and p.stdout.startswith("setup: row x: "), (p.stdout, p.stderr)
    assert _mirror(*mirror) is None
