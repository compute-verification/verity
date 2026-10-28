"""Item 1e: the Lean verifier reads a typed statement (`verity/flock-circuit/types`, `verity_flock.typed_statement`) by
deriving its unit's rows itself (`Flock.Typed`, `Derive.derive`). The rows it derives are the old statement's unit section
byte for byte, the file it then parses is the old statement's in every section, and the digest and class it binds are the
typed file's own. A typed file that carries unit rows, or whose types, layouts or unit keys don't check, is refused."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from verity.ml.circuits.codec import canonical_json

pytest.importorskip("verity_numerical")
pytest.importorskip("verity.ml.definitions.registry.prims")
from verity_flock import circuit as FC  # noqa: E402
from verity_flock import typed_statement as TS  # noqa: E402
from verity_numerical.bench import lowerings  # noqa: E402
from verity_numerical.bench import templates as TM  # noqa: E402

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
BINDING_KEYS = ("partition", "program_sha512", "program_digests")

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def rope(tmp_path_factory):
    """RoPE's old statement and its typed statement, and the verifier built."""
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    sub = TM.subcircuit("rope-head", D=64)
    low = lowerings.registry("C-Flock").module("rope-head").lowering(sub)
    d = tmp_path_factory.mktemp("typed")
    (d / "old.circuit").write_text(FC.compose(low, sub))
    (d / "typed.circuit").write_text(TS.compose(low, sub).text)
    return d


def _expand(path: Path, out: Path | None = None) -> dict:
    p = subprocess.run([str(BIN), "typed-expand", str(path), *(["--out", str(out)] if out else [])], capture_output=True, text=True)
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith("TYPED "))
    return json.loads(line.split(" ", 1)[1])


def test_the_verifier_derives_the_units_rows(rope):
    r = _expand(rope / "typed.circuit", rope / "expanded.circuit")
    assert r["ok"], r
    h0, m0, s0 = TS.sections((rope / "old.circuit").read_text())
    h1, m1, s1 = TS.sections((rope / "typed.circuit").read_text())
    h2, m2, s2 = TS.sections((rope / "expanded.circuit").read_text())
    # every section the verifier parses is the old statement's, the derived unit's rows included; head and META the typed's
    assert s2 == s0 and [n for n, _ in s2] == ["unit", "sha512x3", "hm96"]
    assert (h2, m2) == (h1, m1)
    assert r["nets"] == ["unit", "sha512x3", "hm96"] and r["unit_useful"] == dict(s0)["unit"].count("\n") - 1
    # the digest and class it binds are the typed file's
    typed = (rope / "typed.circuit").read_bytes()
    assert r["sha512"] == hashlib.sha512(typed).hexdigest()
    body = typed.split(b"\n", 2)[2]
    unbound = {k: v for k, v in m1.items() if k not in BINDING_KEYS}
    assert r["class"] == hashlib.sha512(h1.encode() + b"\nMETA " + canonical_json(unbound) + b"\n" + body).hexdigest()


def _with_meta(rope: Path, name: str, edit) -> Path:
    head, meta, secs = TS.sections((rope / "typed.circuit").read_text())
    edit(meta)
    text = head + "\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n" + \
        "".join(f"CIRCUIT {n} {s.count(chr(10))}\n{s}" for n, s in secs)
    (rope / name).write_text(text)
    return rope / name


def _set(key, value):
    return lambda m: m.__setitem__(key, value)


def _rekey(m):
    (d, c), = m["types"].items()
    m["types"] = {d[:-1] + ("0" if d[-1] != "0" else "1"): c}


def _layout_digest(m):
    m["layouts"][0]["sha512"] = "0" * 128


@pytest.mark.parametrize("name,edit,error", [
    ("policy", _set("policy", "calls/v2"), "policy"),
    ("unit-in", _set("unit_in", [[0, {"const": 1, "bind": "run"}]]), "unit_in"),
    ("unit-type", _set("unit_type", "0" * 128), "unit_type"),
    ("type-digest", _rekey, "not its content's digest"),
    ("layout-digest", _layout_digest, "sha512 is not its record's digest"),
    ("unit-layout", _set("unit_layout", 1), "layouts break the rule"),
])
def test_a_typed_statement_that_does_not_check_is_refused(rope, name, edit, error):
    r = _expand(_with_meta(rope, f"{name}.circuit", edit))
    assert not r["ok"] and error in r["error"], r


def test_a_typed_statement_carries_no_unit_rows(rope):
    head, meta, secs = TS.sections((rope / "old.circuit").read_text())
    typed_head, typed_meta, _ = TS.sections((rope / "typed.circuit").read_text())
    text = typed_head + "\nMETA " + json.dumps(typed_meta, sort_keys=True, separators=(",", ":")) + "\n" + \
        "".join(f"CIRCUIT {n} {s.count(chr(10))}\n{s}" for n, s in secs)
    (rope / "with-rows.circuit").write_text(text)
    r = _expand(rope / "with-rows.circuit")
    assert not r["ok"] and "carries no unit rows" in r["error"], r


@pytest.mark.parametrize("name,what", [("sha512x3", "three SHA-512 compressions"), ("hm96", "hm96-sha512/row/v1 commit")])
def test_a_typed_statement_with_other_row_nets_is_refused(rope, name, what):
    """A typed file's `sha512x3` and `hm96` sections are read as the old statement's are, so one AND row changed in either
    (its first row that is neither an input's self row nor empty) is refused: they must be the verifier's (`Flock.HmNets`)."""
    lines = (rope / "typed.circuit").read_text().split("\n")
    at = next(k for k, line in enumerate(lines) if line.startswith(f"CIRCUIT {name} ")) + 2
    q = next(q for q, line in enumerate(lines[at:]) if line not in (f"1 {q} 1 {q}", "0 0"))
    lines[at + q] = "1 0 1 1"
    (rope / f"rows-{name}.circuit").write_text("\n".join(lines))
    r = _expand(rope / f"rows-{name}.circuit")
    assert not r["ok"] and f"circuit {name}: not the verifier's {what} (row {q} differs)" in r["error"], r


#: a typed RoPE (`rope-head/d64/neox-bf16`, a flat class, 4 instances) staged by `typed_statement.stage`, and every CPU
#: selftest case of its flock-circuit (seed-injection build at 8505c540), recorded
RUST_SESSIONS = "art:48f286b4"


@pytest.mark.slow
def test_it_refuses_the_rust_provers_sessions(tmp_path):
    """A typed statement is a form the soundness proof does not cover (§16.15): on `flock-circuit`'s own recorded sessions
    for a flat class, the statement sets up and the verifier refuses every case, the 3 honest ones for that reason, once
    every check has passed, and every other case for the check that catches it. That build has no zk patch, so its plain
    files are upstream's form, without the `zk_level0 = None` tag the deployed prover writes, and are read under
    `--upstream-plain` (PROTOCOL.md §8.1)."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifier"))
    import ci  # noqa: PLC0415

    try:
        root = ci.fetch(f"{RUST_SESSIONS}/stage", tmp_path / "store").parent
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "verifier" / "selftest_records.py"), str(root / "records"), str(tmp_path / "s")],
                   check=True, capture_output=True)
    sessions = sorted(p for p in (tmp_path / "s").iterdir() if (p / "case.json").exists())
    statement = "verity/flock-circuit/types@210d32e1+seed-injection"
    p = subprocess.run([str(BIN), "verify", "--statement", statement,
                        "--circuit", str(root / "stage" / "circuit.txt"), "--public", str(root / "stage" / "pub-4.bin"),
                        "--upstream-plain", "--session", *map(str, sessions)], capture_output=True, text=True)
    got = {Path(v["session"]).name: v for v in (json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT "))}
    want = {s.name: json.loads((s / "case.json").read_text())["expect"] == "accept" for s in sessions}
    refused = f"refused: {statement}, a typed statement: a statement form the soundness proof does not cover"
    assert not any(got[n]["accepted"] for n in want)
    assert {n: got[n]["why"] == refused for n in want} == want, {n: got[n]["why"] for n in want}
    assert sum(want.values()) == 3 and len(want) == 19
