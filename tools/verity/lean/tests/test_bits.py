"""`tools/verity/lean/bits.py` (`verity lean bits`) and its mirrors (`bits_mirror.py`), on this tree's three security locks
and one Lean source, with no Lean build:

- **pins**: every mirror's pinned guarantees and definitions hash in the locks as the mirror records, so a restated
  statement or error term fails here until someone re-reads it, updates the mirror and then its pin;
- **instances** (`INSTANCES`): a theorem proved by `norm_num` that bounds a mirrored definition at numerals; the mirror,
  evaluated in exact rationals at the arguments read from the theorem's source, is at most its right side, and the line
  the command prints for that point carries the same arguments and the same figure.

What it does not hold: that an instance's expression is the guarantee's bound. The instance bounds the mirrored
definition at its numerals, and which guarantee's bound reads that definition, and how (`SpotOneWaitTwo` reads
`hashBound` twice), is the mirror's own reading; the facts' `bound` field will check that. A pin on a guarantee whose
statement is still in the proofs tree is its `type_hash`, the statement's name, so a restated body leaves it unmoved:
the instance check is what holds that mirror to the Lean.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("lean_bits", HERE.parent / "bits.py")
B = importlib.util.module_from_spec(spec)
sys.modules["lean_bits"] = B
spec.loader.exec_module(B)
M = B.M

#: (Lean source, theorem): a theorem proved by `norm_num` stating `F a₁ … aₙ ≤ c` at numerals, `F` a mirrored definition
INSTANCES = [
    ("security/Proofs/Flock/SamplingGeneric/TwoOneWaitNumbers.lean", "Claims.Flock.Numbers.hashBound_oneWait_513"),
    ("security/Proofs/Flock/SamplingGeneric/TwoOneWaitNumbers.lean", "Claims.Flock.Numbers.hashBound_oneWait_513_k40"),
]


@pytest.fixture(scope="module")
def locks() -> dict[str, dict]:
    return {p: json.loads((ROOT / p / "lean-audit.json").read_text()) for p in B.LOCKS}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return B.lines([ROOT / p for p in B.LOCKS], text=False)


@pytest.mark.parametrize("mirror", M.MIRRORS, ids=lambda m: m.lean)
def test_pins_hold(mirror, locks):
    fix = (f"{mirror.lean}'s mirror in bits_mirror.py: re-read the statement and the definitions it pins, update the "
           f"mirror to what they now say, then its pin")
    for g, (key, h) in mirror.guarantees.items():
        records = [lock["guarantees"][g] for lock in locks.values() if g in lock.get("guarantees", {})]
        assert records, f"{g} is listed in no security lock: {fix}"
        for record in records:
            assert key == ("unfolded_hash" if "unfolded_hash" in record else "type_hash"), \
                f"{g}'s pin is its {key} but its record has an unfolded_hash, which says more: pin that"
            assert record[key] == h, f"{g}'s {key} moved ({h[:12]} → {record[key][:12]}): {fix}"
    for d, h in mirror.definitions.items():
        found = [group["definitions"][d] for lock in locks.values() for group in lock.get("reads", {}).values()
                 if d in group.get("definitions", {})]
        assert found, f"no security lock's reads records {d}: {fix}"
        assert set(found) == {h}, f"{d}'s reads hash moved ({h[:12]} → {sorted(set(found))}): {fix}"


def test_a_moved_pin_prints_no_number(tmp_path, locks):
    mirror = next(m for m in M.MIRRORS if m.definitions)
    g = next(iter(mirror.guarantees))
    where = next(p for p, lock in locks.items() if g in lock.get("guarantees", {}))
    lock = json.loads(json.dumps(locks[where]))
    key = mirror.guarantees[g][0]
    lock["guarantees"][g][key] = "0" * 64
    (tmp_path / "lean-audit.json").write_text(json.dumps(lock))
    out = [r for r in B.lines([tmp_path], text=False) if r["guarantee"] == g]
    assert len(out) == 1 and out[0]["bits"] is None and out[0]["source"] is None
    assert "re-read the statement, update the mirror, then its pin" in out[0]["note"]
    assert B.pins_hold(mirror, {where: locks[where]}) == []


def test_order_and_exit(rows, capsys):
    rank = {None: 0, "mirrored": 1}
    assert [rank[r["source"]] for r in rows] == sorted(rank[r["source"]] for r in rows)
    mirrored = [r["bits"] for r in rows if r["source"] == "mirrored"]
    assert mirrored and mirrored == sorted(mirrored)
    assert B.main(["--locks-only"]) == 0
    assert B.main(["--locks-only", "--json"]) == 0
    capsys.readouterr()


def test_text_pass(tmp_path):
    """A statement that applies a probability definition through another definition bounds a probability, with the right
    side of its `≤` as its error term and the definition that holds it; one that doesn't, doesn't; a witness, from the
    lock, bounds nothing."""
    (tmp_path / "A.lean").write_text(
        "namespace Claims\n\ndef Pr (p : Nat → Prop) : Nat := 0\n\nend Claims\n\nnamespace X\n\nopen Claims\n\n"
        "def Bad (n : Nat) : Prop := n = 1\n\ndef Sound (ε : Nat) : Prop :=\n  Pr Bad ≤ ε + 1\n\n"
        "def Plain : Prop := 1 = 1\n\nend X\n")
    lock = {"guarantees": {
        "X.G.sound": {"owner": "@a", "signature": ["X.G.sound : X.Sound 3"]},
        "X.G.plain": {"owner": "@a", "signature": ["X.G.plain : X.Plain"]},
        "X.G.at": {"owner": "@a", "signature": ["X.G.at : X.Sound 0"]},
    }, "witnesses": {"X.G.sound": {"honest": ["X.G.at"]}}}
    (tmp_path / "lean-audit.json").write_text(json.dumps(lock))
    got = {r["guarantee"]: r for r in B.lines([tmp_path])}
    assert (got["X.G.sound"]["probability"], got["X.G.sound"]["error"], got["X.G.sound"]["error_from"]) == \
        (True, "ε + 1", "X.Sound")
    assert got["X.G.plain"]["probability"] is False and got["X.G.plain"]["error"] is None
    assert got["X.G.at"]["witness"] and got["X.G.at"]["probability"] is False
    assert "| X.G.sound | @a | ~yes | ~ε + 1 |" in B.table(list(got.values()))
    assert "| X.G.at | @a | no, a witness | - |" in B.table(list(got.values()))

    (tmp_path / "facts.json.gz").write_bytes(gzip.compress(json.dumps({"facts": {"pins": {
        "X.G.sound": {"kind": "theorem", "reads": [{"name": "X.Sound", "module": "A"}]},
        "X.G.plain": {"kind": "theorem", "reads": [{"name": "Claims.Pr", "module": "A"}]},
        "X.G.at": {"kind": "missing"},
    }}}).encode()))
    pins = B.stored_pins(tmp_path / "facts.json.gz")
    got = {r["guarantee"]: r for r in B.lines([tmp_path], pins=pins)}
    assert got["X.G.at"]["probability_from"] == "lock"
    missing = {r["guarantee"]: r for r in B.lines([tmp_path], pins={"X.G.sound": {"kind": "missing"}})}
    assert missing["X.G.sound"]["probability_from"] == "text", "a pin with no reads leaves the text's reading"
    assert (got["X.G.sound"]["probability"], got["X.G.sound"]["probability_from"]) == (False, "facts"), \
        "the facts' reads decide, over the text"
    assert (got["X.G.plain"]["probability"], got["X.G.plain"]["probability_from"]) == (True, "facts")
    assert "| X.G.plain | @a | yes |" in B.table(list(got.values()))


def test_run_reads_its_facts_and_locks(tmp_path, monkeypatch):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"guarantees": {"X.g": {"owner": "@run", "signature": ["X.g : True"]}}}))
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"pins": {"X.g": {"reads": [{"name": "Verity.Game.prob"}]}}}))
    held, missing = tmp_path / "held", tmp_path / "missing"

    def fetch(run, pkg):
        if pkg != held:
            raise SystemExit(f"{run} holds nothing for {pkg}")
        return facts, lock, tmp_path / "report.json"

    monkeypatch.setattr(B, "fetch_run", fetch)
    pins, lock_files = B.fetch_runs("r1", [held, missing])
    assert pins == {"X.g": {"reads": [{"name": "Verity.Game.prob"}]}} and lock_files == {held: lock}
    [line] = B.lines([held], text=False, pins=pins, lock_files=lock_files)
    assert (line["owner"], line["probability"], line["probability_from"]) == ("@run", True, "facts")


def test_periods(tmp_path, rows):
    (tmp_path / "deployments.json").write_text(json.dumps([
        {"id": "dc", "machines": 2 ** 10, "gpus_per_machine": 8, "window_s": 600, "role": "reference"},
        {"id": "small", "machines": 1, "gpus_per_machine": 8, "window_s": 600}]))
    year = 600 * 2 ** 20
    (tmp_path / "adversary.json").write_text(json.dumps([
        {"id": "world", "role": "reference", "monitoring_period_s": year, "total_flop": 2 ** 13 * 10 * year,
         "flop_per_s_per_h100": 10}]))
    c = B.periods(census=tmp_path)
    assert (c["deployment"], c["adversary"]) == ("dc", "world")
    assert c["audits"]["a year"] == {"datacenter": 2 ** 30, "world": 2 ** 30}
    assert c["audits"]["a month"]["datacenter"] == 2 ** 30 / 12
    assert B.periods("small", census=tmp_path)["audits"]["a year"]["datacenter"] == 2 ** 20
    with pytest.raises(SystemExit, match="its ids: dc, small"):
        B.periods("nowhere", census=tmp_path)

    out = B.lines([ROOT / p for p in B.LOCKS], text=False, census=c)
    line = next(r for r in out if r["source"] == "mirrored")
    assert line["periods"]["a year"] == {"datacenter": line["bits"] - 30, "world": line["bits"] - 30}
    assert all(r["periods"] is None for r in out if r["source"] is None)
    assert B.fmt_bits(-1.0) == "vacuous"


def lean_value(src: str) -> Fraction:
    """A Lean numeral expression (naturals, `+ - * / ^`, parentheses) as an exact rational."""
    toks = re.findall(r"\d+|[-+*/^()]", src)
    assert "".join(toks) == "".join(src.split()), f"not a numeral expression: {src!r}"
    pos = 0

    def atom() -> Fraction:
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "(":
            v = expr()
            assert toks[pos] == ")"
            pos += 1
            return v
        return Fraction(int(t))

    def power() -> Fraction:
        nonlocal pos
        base = atom()
        if pos < len(toks) and toks[pos] == "^":
            pos += 1
            e = power()
            assert e.denominator == 1
            return base ** int(e)
        return base

    def term() -> Fraction:
        nonlocal pos
        v = power()
        while pos < len(toks) and toks[pos] in "*/":
            op = toks[pos]
            pos += 1
            v = v * power() if op == "*" else v / power()
        return v

    def expr() -> Fraction:
        nonlocal pos
        v = term()
        while pos < len(toks) and toks[pos] in "+-":
            op = toks[pos]
            pos += 1
            v = v + term() if op == "+" else v - term()
        return v

    v = expr()
    assert pos == len(toks), f"not one expression: {src!r}"
    return v


def read_instance(path: str, theorem: str) -> tuple[str, list[Fraction], Fraction]:
    """The applied definition's full name, its arguments and the right side of ``theorem``, read from its source; the
    theorem must sit directly in its namespace and be proved by `norm_num`."""
    ns, short = theorem.rsplit(".", 1)
    text = (ROOT / path).read_text()
    assert f"\nnamespace {ns}\n" in text, f"{path} opens no namespace {ns}"
    m = re.search(rf"^theorem {re.escape(short)} :\s*(.*?)\s*:= by\s+norm_num\b", text, re.M | re.S)
    assert m, f"{path} has no `theorem {short} : … := by norm_num …`"
    lhs, rhs = m.group(1).split("≤")
    fn, rest = lhs.strip().split(None, 1)
    args, depth, cur = [], 0, ""
    for c in rest.strip() + " ":
        if c == " " and depth == 0:
            if cur:
                args.append(lean_value(cur))
            cur = ""
            continue
        depth += (c == "(") - (c == ")")
        cur += c
    return f"{ns}.{fn}", args, lean_value(rhs)


@pytest.mark.parametrize("path,theorem", INSTANCES, ids=[t for _, t in INSTANCES])
def test_instance(path, theorem, rows, capsys):
    fn, args, rhs = read_instance(path, theorem)
    mirror = next((m for m in M.MIRRORS if m.lean == fn), None)
    assert mirror is not None, f"{theorem} applies {fn}, which no mirror restates"
    assert len(mirror.params) == len(args), f"{fn}'s mirror lists {mirror.params} as its arguments, {theorem} gives {len(args)}"
    at = dict(zip(mirror.params, args))
    value = mirror.value(**at)
    assert isinstance(value, Fraction) and value <= rhs, \
        f"{fn}'s mirror at {theorem}'s arguments is {float(value):.4g}, above its right side {float(rhs):.4g}"

    printed = [r for r in rows if r["row"] == theorem]
    assert len(printed) == 1, f"the command prints {len(printed)} lines whose row is {theorem}"
    line = printed[0]
    assert {k: Fraction(v) for k, v in line["args"].items()} == at
    assert line["bits"] == B.bits_of(value)
    B.main(["--locks-only"])
    out = [s for s in capsys.readouterr().out.splitlines() if f"| {theorem} |" in s]
    assert len(out) == 1 and f"| {B.fmt_bits(B.bits_of(value))} | mirrored |" in out[0]
