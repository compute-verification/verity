"""Pearl+'s window record W (`pearl_c.window`): over the rounds the prover retains of the timed call, each under its round
(its pass's place among ``served.json``'s passes with calls, so the warm-up's are rounds 0 and 1 and a dropped pass keeps
its number), each call's A rows and y a 32-bit value at its manifest's roots under the domains the verdict recomputes,
and each weight's B rows once, at PoUW's weight root, which is the hm96 root of the same rows under the same domain; a
retained warm-up round is refused.

CPU only: the passes are manifests the test writes.
"""
from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.protocols.pouw.verifier.audit import (
    HM96,
    HM96_WEIGHTS,
    activation_binding,
    commit_rows,
    hm96_salt,
    rows_domain,
    weight_binding,
)
from verity.protocols.spot.verifier.one_stage import registered as G, registration as REG
from verity.commitments.merkle import tree_levels
from verity_vllm.program.predict.window_class import PearlTables
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import served_y as SY
from verity_vllm.protocols.pouw.pearl_c import window as PWR

SCHEME = "pearl-c-sm120-v1-h3"
ENTRY = f"pouw:{SCHEME}"
SALT = bytes(range(64))


def manifest(d, base: int, calls: int, salt: bytes = SALT, y: bool = True) -> str:
    """A retained pass's manifest in ``d``, the fields W reads: ``calls`` calls numbered on from ``base``, call c on weight c
    (its root ``c0c0…`` + c, the install's)."""
    mm = [{"index": base + c, "m": 32, "k": 128, "n": 64 * (c + 1), "weight": c, "real_rows": 20 + c,
           "root_a": f"{base + c:02x}" * 64, **({"y": {"root": f"{base + c + 128:02x}" * 64}} if y else {})}
          for c in range(calls)]
    weights = {str(c): {"n": 64 * (c + 1), "k": 128, "root": f"{0xc0 + c:02x}" * 64} for c in range(calls)}
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"entry": ENTRY, "scheme": SCHEME, "salt": salt.hex(), "weights": weights,
                                                 "matmuls": mm}))
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
    round's own number, then each weight the rounds read once, its B rows (n rows of K words) at its root under
    `audit.weight_binding`'s domain; the record is v2 and chained after ``previous``, and the names are the ones the window
    circuit reads (`PearlTables`). Round 2 dropped, round 3 keeps its number and registers only the weight it reads."""
    passes = served(tmp_path)
    w = PWR.write(tmp_path / "out" / "w.json", passes, [2, 3])
    assert json.loads((tmp_path / "out" / "w.json").read_text()) == w
    assert (w["format"], w["protocol"], w["previous"]) == (REG.FORMAT_V2, REG.PROTOCOL, None)
    assert w["window"] == {"kind": "served", "entry": ENTRY, "scheme": SCHEME, "salt": SALT.hex(), "version": PWR.VERSION}
    assert PWR.version_of(w) == PWR.VERSION == 2
    assert [e["value"] for e in w["registered"]] == ["pearl-a/2/0", "pearl-y/2/0", "pearl-a/2/1", "pearl-y/2/1",
                                                      "pearl-a/3/0", "pearl-y/3/0", "pearl-b/0", "pearl-b/1"]
    mans = [json.loads(Path(passes[i]["manifest"]).read_text()) for i in (3, 4)]
    calls = [u for man in mans for u in man["matmuls"]]
    for (a, y), u in zip(zip(w["registered"][:6:2], w["registered"][1:6:2]), calls, strict=True):
        assert set(a) == set(y) == {*G.ENTRY_KEYS, G.WIDTH_KEY}
        assert a == {"value": a["value"], "root": u["root_a"], "leaves": u["m"], "words": u["k"], "word_bits": 32,
                     "domain": rows_domain(activation_binding(SALT, u["index"]), u["index"], u["m"], HM96).domain_id.hex()}
        assert y == {"value": y["value"], "root": u["y"]["root"], "leaves": u["real_rows"], "words": u["n"], "word_bits": 32,
                     "domain": SY.y_domain(SALT, u["index"], u["real_rows"]).domain_id.hex()}
    for b, wid in zip(w["registered"][6:], (0, 1), strict=True):
        n = 64 * (wid + 1)
        assert b == {"value": f"pearl-b/{wid}", "root": f"{0xc0 + wid:02x}" * 64, "leaves": n, "words": 128, "word_bits": 32,
                     "domain": rows_domain(weight_binding(wid), -2, n, HM96).domain_id.hex()}
    assert (PearlTables.a(1, 0), PearlTables.y(0, 1), PearlTables.b(2)) == (PWR.a_value(1, 0), PWR.y_value(0, 1), PWR.b_value(2)) \
        == ("pearl-a/1/0", "pearl-y/0/1", "pearl-b/2")
    assert PWR.record(mans, rounds=[2, 3]) == w and PWR.record(mans, "ab" * 64, [2, 3])["previous"] == "ab" * 64
    assert [e["value"] for e in PWR.write(tmp_path / "w3.json", passes, [3])["registered"]] == ["pearl-a/3/0", "pearl-y/3/0",
                                                                                               "pearl-b/0"]
    old = PWR.record(mans, rounds=[2, 3], version=1)
    assert old == {**w, "window": {"kind": "served", "entry": ENTRY, "scheme": SCHEME, "salt": SALT.hex()},
                   "registered": w["registered"][:6]} and PWR.version_of(old) == 1


def test_w_at_version_1_is_the_one_written_before_w_registered_the_weights():
    """A W states its version in its header: version 2, or none, version 1 (the run of record's W, its header the kind,
    entry, scheme and salt, its values the calls' alone). A header stating any other version, 1 included, is refused, as
    is a W built at a version that isn't one."""
    head = {"kind": "served", "entry": ENTRY, "scheme": SCHEME, "salt": SALT.hex()}
    assert (PWR.version_of({"window": head}), PWR.version_of({}), PWR.VERSIONS) == (1, 1, (1, 2))
    for v in (1, 3, "2", 2.0, True, None):
        with pytest.raises(ValueError, match=re.escape(PWR.UNKNOWN_VERSION)):
            PWR.version_of({"window": {**head, "version": v}})
    with pytest.raises(ValueError, match="W is version 1 or 2, not 3"):
        PWR.record([{"salt": SALT.hex()}], version=3)


def test_w_is_refused_for_rounds_it_cant_register(tmp_path, monkeypatch):
    """A retained warm-up round (the verdict ties every retained round to W, which lists only the timed call's), a round
    that is no pass with calls or retained no manifest, no round, rounds out of order, a call that commits no y, passes
    under two salts, a weight a pass commits no root of, a weight two rounds commit as other rows (another root, or another
    n), a weight two calls of one pass read at other shapes, and rows that aren't hm96's: each refused, naming why, and
    nothing is written."""
    out, passes = tmp_path / "w.json", served(tmp_path)
    good = json.loads(Path(passes[3]["manifest"]).read_text())
    later = json.loads(Path(passes[4]["manifest"]).read_text())
    rooted = {**later, "weights": {"0": {**later["weights"]["0"], "root": "c1" * 64}}}
    wider = {**later, "matmuls": [{**later["matmuls"][0], "n": 128}]}
    reshaped = {**good, "matmuls": [good["matmuls"][0], {**good["matmuls"][1], "weight": 0}]}
    cases = [(lambda: PWR.write(out, passes, [1, 2]), "round 1 is the warm-up's (timed false) and retained"),
             (lambda: PWR.write(out, passes, [4]), "no round 4 among the run's 4 passes with calls"),
             (lambda: PWR.write(out, [{"units": 1}], [0]), "round 0 retained no manifest"),
             (lambda: PWR.write(out, passes, []), "it retains none"),
             (lambda: PWR.record([good, good], rounds=[3, 2]), "are not distinct rounds in order"),
             (lambda: PWR.record([good], rounds=[2, 3]), "are not distinct rounds in order"),
             (lambda: PWR.write(out, [{"units": 1, "manifest": manifest(tmp_path / "b", 0, 1, y=False)}], [0]),
              "round 0's call 0 (index 0) commits no y"),
             (lambda: PWR.write(out, [*passes[3:], {"units": 1, "manifest": manifest(tmp_path / "c", 9, 1, salt=bytes(64))}],
                                [0, 2]), "W is one epoch's"),
             (lambda: PWR.record([{**good, "weights": {"0": good["weights"]["0"]}}], rounds=[2]),
              "round 2's pass commits no root of weight 1, whose B rows W registers"),
             (lambda: PWR.record([good, rooted], rounds=[2, 3]),
              "round 3's pass commits pearl-b/0 as other rows than an earlier round's"),
             (lambda: PWR.record([good, wider], rounds=[2, 3]),
              "round 3's pass commits pearl-b/0 as other rows than an earlier round's"),
             (lambda: PWR.record([reshaped], rounds=[2]),
              "round 2's call 1 (index 3) reads weight 0 as 128 rows of 128 words, and an earlier call as 64 of 128: W "
              "registers one shape of each weight's B rows")]
    for call, why in cases:
        with pytest.raises(ValueError, match=why.replace("(", r"\(").replace(")", r"\)")):
            call()
    monkeypatch.setattr(PPD, "scheme_of", lambda name: SimpleNamespace(commitment_hash="sha256"))
    with pytest.raises(ValueError, match=f"commits sha256 rows, and W registers {HM96} rows"):
        PWR.write(out, passes, [3])
    assert not out.exists()


def committed(key: bytes, calls: list[tuple[int, int, int, int, int]], s: int = 0
              ) -> tuple[dict, dict[str, np.ndarray], dict[int, np.ndarray]]:
    """A pass's manifest whose calls (``(index, m16, real rows, K, N)``, call c on weight c) commit random BF16 x and y as
    Pearl+ does (A: x widened to FP32 then zero rows up to m₁₆, `audit.commit_rows` salted at the index; y:
    `served_y.commit_y`), with the prover's key, each value's rows, as round ``s``'s, as the circuit's proof blocks make
    them (u16 words, an FP32 word's low first), and each weight's B rows (N rows of K FP32 words, a random BF16 weight
    widened), committed as PoUW commits them (`audit.commit_rows` at `weight_binding`, owner -2, salted at `HM96_WEIGHTS`
    plus the weight: `clear_audit.Prover`'s tree, the device's `weight_roots`)."""
    rng, mm, rows, b, weights = np.random.default_rng(1), [], {}, {}, {}
    for c, (i, m, real, k, n) in enumerate(calls):
        x, y = (rng.integers(0, 1 << 16, (real, w), dtype=np.uint16) for w in (k, n))
        a = np.zeros((m, k), "<u4")
        a[:real] = x.astype("<u4") << 16
        root_a = commit_rows(activation_binding(SALT, i), i, [r.tobytes() for r in a], HM96, key, i).commitment.root
        mm.append({"index": i, "m": m, "k": k, "n": n, "weight": c, "real_rows": real, "root_a": root_a.hex(),
                   "y": {"root": SY.commit_y(y.tobytes(), n, SALT, i, key)[0].hex()}})
        rows[PWR.a_value(s, c)] = a.view("<u2")
        rows[PWR.y_value(s, c)] = (y.astype("<u4") << 16).view("<u2")
        b[c] = rng.integers(0, 1 << 16, (n, k), dtype=np.uint16).astype("<u4") << 16
        root_b = commit_rows(weight_binding(c), -2, [r.tobytes() for r in b[c]], HM96, key, HM96_WEIGHTS + c).commitment.root
        weights[str(c)] = {"n": n, "k": k, "root": root_b.hex()}
    return ({"entry": ENTRY, "scheme": SCHEME, "salt": SALT.hex(), "hm96_salt_key": key.hex(), "weights": weights,
             "matmuls": mm}, rows, b)


def test_the_audited_windows_values_are_ws_from_the_rows_its_proof_blocks_make():
    """`reuse` commits each value W registers from the rows the audited circuit's proof blocks make of it, as Pearl+
    committed the call's A rows and y, so each value's entry is W's and its tree opens each row at W's root, at the
    round W numbers it (round 2, the timed call's first), W at version 2 or at version 1 (built at the version it states).
    W holds no salt key. Rows the proof blocks make otherwise (one bit of one A row), another key, a W other than the
    passes' record at its version (a version-2 W's values under a header that states none), or the passes at another
    round is refused."""
    key = bytes(range(100, 132))
    man, rows, _b = committed(key, [(7, 16, 5, 64, 128), (8, 32, 17, 128, 64)], s=2)
    w = PWR.record([man], rounds=[2])
    assert key.hex() not in json.dumps(w)
    got = PWR.reuse(w, [man], lambda name, n, empty: rows[name], rounds=[2])
    theirs = {e["value"]: e for e in w["registered"] if not e["value"].startswith("pearl-b/")}
    assert sorted(got) == sorted(theirs) == ["pearl-a/2/0", "pearl-a/2/1", "pearl-y/2/0", "pearl-y/2/1"]
    assert {v: x.entry() for v, x in got.items()} == theirs
    for v, x in got.items():
        root = bytes.fromhex(theirs[v]["root"])
        assert all(G.verify_read(x.domain, root, p, x.bcs[p], x.path(p), G.SCHEMA_SEG) for p in range(x.leaves))
    old = PWR.record([man], rounds=[2], version=1)
    assert {v: x.entry() for v, x in PWR.reuse(old, [man], lambda name, n, empty: rows[name], rounds=[2]).items()} == theirs
    unstated = {**w, "window": {k: x for k, x in w["window"].items() if k != "version"}}
    flipped = {**rows, "pearl-a/2/1": rows["pearl-a/2/1"].copy()}
    flipped["pearl-a/2/1"][3, 0] ^= 1
    for bad, mans, of, at, why in (
            (w, [man], flipped, [2], "pearl-a/2/1: the rows the audited circuit's proof blocks make of it are not"),
            (w, [{**man, "hm96_salt_key": bytes(32).hex()}], rows, [2], "pearl-a/2/0: the rows"),
            ({**w, "registered": w["registered"][:-1]}, [man], rows, [2], "W is not the window record of these passes"),
            (unstated, [man], rows, [2], "W is not the window record of these passes"),
            (w, [man], rows, None, "W is not the window record of these passes")):
        with pytest.raises(ValueError, match=why):
            PWR.reuse(bad, mans, lambda name, n, empty, of=of: of[name], rounds=at)


def test_ws_b_rows_are_at_pouws_weight_root_the_hm96_root_of_the_same_rows():
    """PoUW's tree of a weight's B rows is a ``frame-v3-sha512`` tree of hm96 ``row-seg/v1`` leaves over 32-bit words
    (`audit.Hm96Domain`), so its root is the one-stage registered commitment's (`registered.Registered.commit`, as `reuse`
    commits A and y) of the same rows and salts in a tree under the same domain, and W's ``pearl-b/<w>`` at PoUW's root is
    that value's entry: B needs no second tree, and PoUW's root is the entry's. A row other than the committed one moves
    the root."""
    key = bytes(range(100, 132))
    man, _rows, b = committed(key, [(7, 16, 5, 64, 128), (8, 32, 17, 128, 64)])
    theirs = {e["value"]: e for e in PWR.record([man])["registered"] if e["value"].startswith("pearl-b/")}
    assert sorted(theirs) == ["pearl-b/0", "pearl-b/1"]

    def registered(wid: int, rows: np.ndarray) -> PWR.Committed:
        n = len(rows)
        base = G.Registered.commit(SCHEME, PWR.b_value(wid), rows.tolist(), [hm96_salt(key, HM96_WEIGHTS + wid, j) for j in range(n)],
                                   word_bits=32, schema=G.SCHEMA_SEG)
        dom = rows_domain(weight_binding(wid), -2, n, "sha512")
        levels = tree_levels(dom, [G.leaf(dom, j, bc, G.SCHEMA_SEG) for j, bc in enumerate(base.bcs)])
        return PWR.Committed(**{**{f.name: getattr(base, f.name) for f in fields(base)}, "levels": levels}, call_domain=dom)

    for wid, rows in b.items():
        got = registered(wid, rows)
        assert got.domain.domain_id == rows_domain(weight_binding(wid), -2, len(rows), HM96).domain_id
        assert got.root.hex() == man["weights"][str(wid)]["root"] and got.entry() == theirs[PWR.b_value(wid)]
    flipped = b[1].copy()
    flipped[3, 0] ^= 1 << 16
    assert registered(1, flipped).root.hex() != man["weights"]["1"]["root"]
