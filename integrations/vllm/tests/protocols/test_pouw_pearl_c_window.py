"""Pearl+'s window record W (`pearl_c.window`): over the rounds the prover retains of the timed call, each under its round
(its pass's place among ``served.json``'s passes with calls, so the warm-up's are rounds 0 and 1 and a dropped pass keeps
its number), each call's A rows and y a 32-bit value at its manifest's roots under the domains the verdict recomputes; a
retained warm-up round is refused.

CPU only: the passes are manifests the test writes.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.core.protocols.pouw.audit import HM96, activation_binding, commit_rows, rows_domain
from verity.core.protocols.sampled_proofs.one_stage import registered as G, registration as REG
from verity_vllm.program.predict.window_class import PearlTables
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import served_y as SY
from verity_vllm.protocols.pouw.pearl_c import window as PWR

SCHEME = "pearl-c-sm120-v1-h3"
ENTRY = f"pouw:{SCHEME}"
SALT = bytes(range(64))


def manifest(d, base: int, calls: int, salt: bytes = SALT, y: bool = True) -> str:
    """A retained pass's manifest in ``d``, the fields W reads: ``calls`` calls numbered on from ``base``."""
    mm = [{"index": base + c, "m": 32, "k": 128, "n": 64 * (c + 1), "weight": c, "real_rows": 20 + c,
           "root_a": f"{base + c:02x}" * 64, **({"y": {"root": f"{base + c + 128:02x}" * 64}} if y else {})}
          for c in range(calls)]
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"entry": ENTRY, "scheme": SCHEME, "salt": salt.hex(), "matmuls": mm}))
    return str(d / "manifest.json")


def served(tmp_path: Path) -> list[dict]:
    """A served run's passes as ``served.json`` lists them: the warm-up's two (rounds 0 and 1, ``timed`` false), one
    without calls, and the timed call's two (rounds 2 and 3)."""
    return [{"units": 1, "timed": False, "manifest": manifest(tmp_path / "pass-00000", 0, 1)},
            {"units": 1, "timed": False, "manifest": manifest(tmp_path / "pass-00001", 1, 1)},
            {"units": 0, "timed": True},
            {"units": 2, "timed": True, "manifest": manifest(tmp_path / "pass-00003", 2, 2)},
            {"units": 1, "timed": True, "manifest": manifest(tmp_path / "pass-00004", 4, 1)}]


def test_w_registers_each_retained_timed_rounds_calls_under_the_rounds_served_number(tmp_path):
    """The timed call's rounds 2 and 3 retained: each call's A rows (m₁₆ rows of K words) and y (its real rows of N words)
    at the manifest's roots, 32-bit words, under A's activation domain and y's served domain, round by round under the
    round's own number; the record is v2 and chained after ``previous``, and the names are the ones the window circuit
    reads (`PearlTables`). Round 2 dropped, round 3 keeps its number."""
    passes = served(tmp_path)
    w = PWR.write(tmp_path / "out" / "w.json", passes, [2, 3])
    assert json.loads((tmp_path / "out" / "w.json").read_text()) == w
    assert (w["format"], w["protocol"], w["previous"]) == (REG.FORMAT_V2, REG.PROTOCOL, None)
    assert w["window"] == {"kind": "served", "entry": ENTRY, "scheme": SCHEME, "salt": SALT.hex()}
    assert [e["value"] for e in w["registered"]] == ["pearl-a/2/0", "pearl-y/2/0", "pearl-a/2/1", "pearl-y/2/1",
                                                      "pearl-a/3/0", "pearl-y/3/0"]
    mans = [json.loads(Path(passes[i]["manifest"]).read_text()) for i in (3, 4)]
    calls = [u for man in mans for u in man["matmuls"]]
    for (a, y), u in zip(zip(w["registered"][::2], w["registered"][1::2]), calls, strict=True):
        assert set(a) == set(y) == {*G.ENTRY_KEYS, G.WIDTH_KEY}
        assert a == {"value": a["value"], "root": u["root_a"], "leaves": u["m"], "words": u["k"], "word_bits": 32,
                     "domain": rows_domain(activation_binding(SALT, u["index"]), u["index"], u["m"], HM96).domain_id.hex()}
        assert y == {"value": y["value"], "root": u["y"]["root"], "leaves": u["real_rows"], "words": u["n"], "word_bits": 32,
                     "domain": SY.y_domain(SALT, u["index"], u["real_rows"]).domain_id.hex()}
    assert (PearlTables.a(1, 0), PearlTables.y(0, 1)) == (PWR.a_value(1, 0), PWR.y_value(0, 1)) == ("pearl-a/1/0", "pearl-y/0/1")
    assert PWR.record(mans, rounds=[2, 3]) == w and PWR.record(mans, "ab" * 64, [2, 3])["previous"] == "ab" * 64
    assert [e["value"] for e in PWR.write(tmp_path / "w3.json", passes, [3])["registered"]] == ["pearl-a/3/0", "pearl-y/3/0"]


def test_w_is_refused_for_rounds_it_cant_register(tmp_path, monkeypatch):
    """A retained warm-up round (the verdict ties every retained round to W, which lists only the timed call's), a round
    that is no pass with calls or retained no manifest, no round, rounds out of order, a call that commits no y, passes
    under two salts and rows that aren't hm96's: each refused, naming why, and nothing is written."""
    out, passes = tmp_path / "w.json", served(tmp_path)
    good = json.loads(Path(passes[3]["manifest"]).read_text())
    cases = [(lambda: PWR.write(out, passes, [1, 2]), "round 1 is the warm-up's (timed false) and retained"),
             (lambda: PWR.write(out, passes, [4]), "no round 4 among the run's 4 passes with calls"),
             (lambda: PWR.write(out, [{"units": 1}], [0]), "round 0 retained no manifest"),
             (lambda: PWR.write(out, passes, []), "it retains none"),
             (lambda: PWR.record([good, good], rounds=[3, 2]), "are not distinct rounds in order"),
             (lambda: PWR.record([good], rounds=[2, 3]), "are not distinct rounds in order"),
             (lambda: PWR.write(out, [{"units": 1, "manifest": manifest(tmp_path / "b", 0, 1, y=False)}], [0]),
              "round 0's call 0 (index 0) commits no y"),
             (lambda: PWR.write(out, [*passes[3:], {"units": 1, "manifest": manifest(tmp_path / "c", 9, 1, salt=bytes(64))}],
                                [0, 2]), "W is one epoch's")]
    for call, why in cases:
        with pytest.raises(ValueError, match=why.replace("(", r"\(").replace(")", r"\)")):
            call()
    monkeypatch.setattr(PPD, "scheme_of", lambda name: SimpleNamespace(commitment_hash="sha256"))
    with pytest.raises(ValueError, match=f"commits sha256 rows, and W registers {HM96} rows"):
        PWR.write(out, passes, [3])
    assert not out.exists()


def committed(key: bytes, calls: list[tuple[int, int, int, int, int]], s: int = 0) -> tuple[dict, dict[str, np.ndarray]]:
    """A pass's manifest whose calls (``(index, m16, real rows, K, N)``) commit random BF16 x and y as Pearl+ does (A: x
    widened to FP32 then zero rows up to m₁₆, `audit.commit_rows` salted at the index; y: `served_y.commit_y`), with the
    prover's key, and each value's rows, as round ``s``'s, as the circuit's proof blocks make them (u16 words, an FP32
    word's low first)."""
    rng, mm, rows = np.random.default_rng(1), [], {}
    for c, (i, m, real, k, n) in enumerate(calls):
        x, y = (rng.integers(0, 1 << 16, (real, w), dtype=np.uint16) for w in (k, n))
        a = np.zeros((m, k), "<u4")
        a[:real] = x.astype("<u4") << 16
        root_a = commit_rows(activation_binding(SALT, i), i, [r.tobytes() for r in a], HM96, key, i).commitment.root
        mm.append({"index": i, "m": m, "k": k, "n": n, "real_rows": real, "root_a": root_a.hex(),
                   "y": {"root": SY.commit_y(y.tobytes(), n, SALT, i, key)[0].hex()}})
        rows[PWR.a_value(s, c)] = a.view("<u2")
        rows[PWR.y_value(s, c)] = (y.astype("<u4") << 16).view("<u2")
    return {"entry": ENTRY, "scheme": SCHEME, "salt": SALT.hex(), "hm96_salt_key": key.hex(), "matmuls": mm}, rows


def test_the_audited_windows_values_are_ws_from_the_rows_its_proof_blocks_make():
    """`reuse` commits each value W registers from the rows the audited circuit's proof blocks make of it, as Pearl+
    committed the call's A rows and y, so each value's entry is W's and its tree opens each row at W's root, at the
    round W numbers it (round 2, the timed call's first). W holds no salt key. Rows the proof blocks make otherwise (one bit
    of one A row), another key, a W other than the passes' record, or the passes at another round is refused."""
    key = bytes(range(100, 132))
    man, rows = committed(key, [(7, 16, 5, 64, 128), (8, 32, 17, 128, 64)], s=2)
    w = PWR.record([man], rounds=[2])
    assert key.hex() not in json.dumps(w)
    got = PWR.reuse(w, [man], lambda name, n, empty: rows[name], rounds=[2])
    theirs = {e["value"]: e for e in w["registered"]}
    assert sorted(got) == sorted(theirs) == ["pearl-a/2/0", "pearl-a/2/1", "pearl-y/2/0", "pearl-y/2/1"]
    assert {v: x.entry() for v, x in got.items()} == theirs
    for v, x in got.items():
        root = bytes.fromhex(theirs[v]["root"])
        assert all(G.verify_read(x.domain, root, p, x.bcs[p], x.path(p), G.SCHEMA_SEG) for p in range(x.leaves))
    flipped = {**rows, "pearl-a/2/1": rows["pearl-a/2/1"].copy()}
    flipped["pearl-a/2/1"][3, 0] ^= 1
    for bad, mans, of, at, why in (
            (w, [man], flipped, [2], "pearl-a/2/1: the rows the audited circuit's proof blocks make of it are not"),
            (w, [{**man, "hm96_salt_key": bytes(32).hex()}], rows, [2], "pearl-a/2/0: the rows"),
            ({**w, "registered": w["registered"][:-1]}, [man], rows, [2], "W is not the window record of these passes"),
            (w, [man], rows, None, "W is not the window record of these passes")):
        with pytest.raises(ValueError, match=why):
            PWR.reuse(bad, mans, lambda name, n, empty, of=of: of[name], rounds=at)
