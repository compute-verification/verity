"""Pearl+'s window record W (`pearl_c.window`): at release, after the device run's finish, the install writes W over every
retained pass with calls, round s the s-th such pass in the order the run ended them (a pass without calls is no round),
each call's A rows and y a 32-bit value at its manifest's roots under the domains the verdict recomputes; unset, the
release is the run's own and nothing is written.

CPU only: the device run is a stand-in whose ended passes name manifests the test writes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.core.protocols.pouw.audit import HM96, activation_binding, commit_rows, rows_domain
from verity.core.protocols.pouw.coins import Challenger
from verity.core.protocols.sampled_proofs.one_stage import registered as G, registration as REG
from verity_vllm import protocols as PO
from verity_vllm.program.predict.window_class import PearlTables
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import served_y as SY
from verity_vllm.protocols.pouw.pearl_c import window as PWR

SCHEME = "pearl-c-sm120-v1-h3"
ENTRY = f"pouw:{SCHEME}"
ROOTS = {0: b"\x05" * 64}
SALT = bytes(range(64))


def manifest(d, base: int, calls: int, salt: bytes = SALT, y: bool = True) -> str:
    """A retained pass's manifest in ``d``, the fields W reads: ``calls`` calls numbered on from ``base``."""
    mm = [{"index": base + c, "m": 32, "k": 128, "n": 64 * (c + 1), "weight": c, "real_rows": 20 + c,
           "root_a": f"{base + c:02x}" * 64, **({"y": {"root": f"{base + c + 128:02x}" * 64}} if y else {})}
          for c in range(calls)]
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"entry": ENTRY, "scheme": SCHEME, "salt": salt.hex(), "matmuls": mm}))
    return str(d / "manifest.json")


def test_w_registers_each_calls_a_rows_and_y_round_by_round_over_the_passes_with_calls(tmp_path):
    """Two passes with calls around one without: rounds 0 and 1, each call's A rows (m₁₆ rows of K words) and y (its real
    rows of N words) at the manifest's roots, 32-bit words, under A's activation domain and y's served domain; the record
    is v2 and chained after ``previous``, and the names are the ones the window circuit reads (`PearlTables`)."""
    m0, m1 = manifest(tmp_path / "pass-00000", 0, 2), manifest(tmp_path / "pass-00002", 2, 1)
    w = PWR.write(tmp_path / "out" / "w.json", [{"units": 2, "manifest": m0}, {"units": 0}, {"units": 1, "manifest": m1}])
    assert json.loads((tmp_path / "out" / "w.json").read_text()) == w
    assert (w["format"], w["protocol"], w["previous"]) == (REG.FORMAT_V2, REG.PROTOCOL, None)
    assert w["window"] == {"kind": "served", "entry": ENTRY, "scheme": SCHEME, "salt": SALT.hex()}
    assert [e["value"] for e in w["registered"]] == ["pearl-a/0/0", "pearl-y/0/0", "pearl-a/0/1", "pearl-y/0/1",
                                                      "pearl-a/1/0", "pearl-y/1/0"]
    mans = [json.loads(Path(m).read_text()) for m in (m0, m1)]
    calls = [u for man in mans for u in man["matmuls"]]
    for (a, y), u in zip(zip(w["registered"][::2], w["registered"][1::2]), calls, strict=True):
        assert set(a) == set(y) == {*G.ENTRY_KEYS, G.WIDTH_KEY}
        assert a == {"value": a["value"], "root": u["root_a"], "leaves": u["m"], "words": u["k"], "word_bits": 32,
                     "domain": rows_domain(activation_binding(SALT, u["index"]), u["index"], u["m"], HM96).domain_id.hex()}
        assert y == {"value": y["value"], "root": u["y"]["root"], "leaves": u["real_rows"], "words": u["n"], "word_bits": 32,
                     "domain": SY.y_domain(SALT, u["index"], u["real_rows"]).domain_id.hex()}
    assert (PearlTables.a(1, 0), PearlTables.y(0, 1)) == (PWR.a_value(1, 0), PWR.y_value(0, 1)) == ("pearl-a/1/0", "pearl-y/0/1")
    assert PWR.record(mans) == w and PWR.record(mans, previous="ab" * 64)["previous"] == "ab" * 64


def test_w_is_refused_for_passes_it_cant_register(tmp_path, monkeypatch):
    """A pass with calls that retained no manifest, a run with no pass with calls, a call that commits no y, passes under two
    salts and rows that aren't hm96's: each refused, naming why, and nothing is written."""
    out = tmp_path / "w.json"
    good = manifest(tmp_path / "a", 0, 1)
    cases = [([{"units": 1}], "pass 0 has calls and retained no manifest"),
             ([{"units": 0}], "the run has no pass with calls"),
             ([{"units": 1, "manifest": manifest(tmp_path / "b", 0, 1, y=False)}], "round 0's call 0 (index 0) commits no y"),
             ([{"units": 1, "manifest": good}, {"units": 1, "manifest": manifest(tmp_path / "c", 1, 1, salt=bytes(64))}],
              "W is one epoch's")]
    for passes, why in cases:
        with pytest.raises(ValueError, match=why.replace("(", r"\(").replace(")", r"\)")):
            PWR.write(out, passes)
    monkeypatch.setattr(PPD, "scheme_of", lambda name: SimpleNamespace(commitment_hash="sha256"))
    with pytest.raises(ValueError, match=f"commits sha256 rows, and W registers {HM96} rows"):
        PWR.write(out, [{"units": 1, "manifest": good}])
    assert not out.exists()


def committed(key: bytes, calls: list[tuple[int, int, int, int, int]]) -> tuple[dict, dict[str, np.ndarray]]:
    """A pass's manifest whose calls (``(index, m16, real rows, K, N)``) commit random BF16 x and y as Pearl+ does (A: x
    widened to FP32 then zero rows up to m₁₆, `audit.commit_rows` salted at the index; y: `served_y.commit_y`), with the
    prover's key, and each value's rows as the circuit's proof blocks make them (u16 words, an FP32 word's low first)."""
    rng, mm, rows = np.random.default_rng(1), [], {}
    for c, (i, m, real, k, n) in enumerate(calls):
        x, y = (rng.integers(0, 1 << 16, (real, w), dtype=np.uint16) for w in (k, n))
        a = np.zeros((m, k), "<u4")
        a[:real] = x.astype("<u4") << 16
        root_a = commit_rows(activation_binding(SALT, i), i, [r.tobytes() for r in a], HM96, key, i).commitment.root
        mm.append({"index": i, "m": m, "k": k, "n": n, "real_rows": real, "root_a": root_a.hex(),
                   "y": {"root": SY.commit_y(y.tobytes(), n, SALT, i, key)[0].hex()}})
        rows[PWR.a_value(0, c)] = a.view("<u2")
        rows[PWR.y_value(0, c)] = (y.astype("<u4") << 16).view("<u2")
    return {"entry": ENTRY, "scheme": SCHEME, "salt": SALT.hex(), "hm96_salt_key": key.hex(), "matmuls": mm}, rows


def test_the_audited_windows_values_are_ws_from_the_rows_its_proof_blocks_make():
    """`reuse` commits each value W registers from the rows the audited circuit's proof blocks make of it, as Pearl+
    committed the call's A rows and y, so each value's entry is W's and its tree opens each row at W's root. W holds no
    salt key. Rows the proof blocks make otherwise (one bit of one A row), another key, or a W other than the passes'
    record is refused, naming the value."""
    key = bytes(range(100, 132))
    man, rows = committed(key, [(7, 16, 5, 64, 128), (8, 32, 17, 128, 64)])
    w = PWR.record([man])
    assert key.hex() not in json.dumps(w)
    got = PWR.reuse(w, [man], lambda name, n, empty: rows[name])
    theirs = {e["value"]: e for e in w["registered"]}
    assert {v: x.entry() for v, x in got.items()} == theirs
    for v, x in got.items():
        root = bytes.fromhex(theirs[v]["root"])
        assert all(G.verify_read(x.domain, root, p, x.bcs[p], x.path(p), G.SCHEMA_SEG) for p in range(x.leaves))
    flipped = {**rows, "pearl-a/0/1": rows["pearl-a/0/1"].copy()}
    flipped["pearl-a/0/1"][3, 0] ^= 1
    for bad, mans, of, why in ((w, [man], flipped, "pearl-a/0/1: the rows the audited circuit's proof blocks make of it are not"),
                               (w, [{**man, "hm96_salt_key": bytes(32).hex()}], rows, "pearl-a/0/0: the rows"),
                               ({**w, "registered": w["registered"][:-1]}, [man], rows,
                                "W is not the window record of these passes")):
        with pytest.raises(ValueError, match=why):
            PWR.reuse(bad, mans, lambda name, n, empty, of=of: of[name])


def test_the_setting_is_off_by_default_and_is_pearl_cs_over_a_transcript_dir():
    """Unset, ``window_record`` is None; set, it is a path and needs ``transcript_dir`` (W reads the retained manifests),
    and an executor other than Pearl-C's is refused it."""
    assert pouw.Settings.of({}).window_record is None
    assert pouw.Settings.of({"window_record": "w.json", "transcript_dir": "t"}).window_record == "w.json"
    with pytest.raises(ValueError, match="'window_record' needs 'transcript_dir'"):
        pouw.Settings.of({"window_record": "w.json"})
    with pytest.raises(ValueError, match="'window_record' is a path string"):
        pouw.Settings.of({"window_record": 3, "transcript_dir": "t"})
    other = PO.Context(stage="serve", enabled=("pouw:ncp-v1",), settings={"pouw": {"window_record": "w", "transcript_dir": "t"}})
    assert any("'window_record' is Pearl-C's window record" in r for r in PO.refusals(other))


class Apply:
    def apply(self, layer, x, bias=None):
        return np.asarray(x, np.float32) @ np.asarray(layer.weight, np.float32).T


class Linear:
    def __init__(self, weight):
        self.weight = weight
        self.quant_method = Apply()


class Model:
    def __init__(self, *weights):
        self.layers = [Linear(w) for w in weights]

    def modules(self):
        return iter(self.layers)


class Run:
    """Pearl-C's device run with no device: its finish ends the passes `ENDED` names, as a served run's release does."""
    ENDED: list = []
    RUNS: list = []

    def __init__(self, s, entry, kernel, weights, roots, coins, opening, **kw):
        self.commits, self.details, self.cur = {"scheme": s.name}, {"passes": []}, SimpleNamespace(matmuls=[])
        Run.RUNS.append(self)

    def begin(self, *a, **kw):
        self.cur = SimpleNamespace(matmuls=[])

    def end_pass(self, *a, **kw):
        return {}

    def finish(self):
        self.details["passes"] += Run.ENDED


@pytest.fixture
def device_run(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(PPD.Kernel, "load", staticmethod(lambda *a: None))
    monkeypatch.setattr(PPD, "weight_roots", lambda kernel, weights, scheme, salt_key=None: ROOTS)
    monkeypatch.setattr(PPD, "PearlCDeviceRun", Run)
    monkeypatch.setattr(pouw, "_SALT_KEYS", {})
    Run.RUNS.clear()
    return Run


def test_the_install_writes_w_at_release_after_the_runs_finish_and_none_unset(tmp_path, device_run):
    """With ``window_record``, W is written once the run has finished, over the passes its finish ended, and the run's
    details name its path, digest and value count; without it the release is the run's own finish and nothing is written."""
    model = Model(np.random.default_rng(0).standard_normal((128, 1024), np.float32))
    device_run.ENDED = [{"units": 1, "manifest": manifest(tmp_path / "t" / "pass-00000", 0, 1)}, {"units": 0}]
    for path in (tmp_path / "w.json", None):
        settings = {"device": "cuda", "kernel": "k", "coins": Challenger(0, random.Random(0).randbytes),
                    "transcript_dir": str(tmp_path / "t"), **({"window_record": str(path)} if path else {})}
        ctx = PO.Context(stage="serve", enabled=(ENTRY,), settings={"pouw": settings})
        assert PO.refusals(ctx) == ()
        with PO.install(model, ctx):
            assert not (tmp_path / "w.json").exists()
        run = device_run.RUNS[-1]
        if path is None:
            assert "window_record" not in run.details and not (tmp_path / "w.json").exists()
            continue
        w = json.loads(path.read_text())
        assert [e["value"] for e in w["registered"]] == ["pearl-a/0/0", "pearl-y/0/0"]
        assert run.details["window_record"] == {"path": str(path), "digest": REG.digest(w), "values": 2}
        path.unlink()
