"""The salt-to-deadline window (benchmarks/pouw/exhaustion) off the card: its Merkle transcript, draw, keys and fraction; the
CPU audit on an honest window written as the cards write one (every drawn tile accepted, both controls rejected) and on
windows that break the clock, the roots or the draw; and window.py's parent and workers on the libcuda stand-in, every
launch inside its buffers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from test_pearl_c_sm120 import UUID, dry_env, kernel_names, room, ship, track  # noqa: F401

HERE = Path(__file__).resolve().parents[1] / "exhaustion"


def load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


A = load("exhaustion_audit", HERE / "audit.py")
W = A.W
H = W.F.H

from verity_pouw.audit import (ROW_SCHEMA, TILE_SCHEMA, Prover, activation_binding, rows_domain,  # noqa: E402
                               transcript_domain, weight_binding)
from verity_pouw.protocol import tagged  # noqa: E402

SMALL = (64, 64, 1024)                       # (m, n, k): one 64 x 64 tile, the scheme's least k


def test_merkle_root_path_and_stream_agree():
    for count in range(0, 19):
        leaves = [hashlib.sha256(bytes([i])).digest() for i in range(count)]
        root, s = W.merkle_root(leaves), W.Stack()
        for x in leaves:
            s.push(x)
        assert s.root() == root and s.count == count
        for pos in range(count):
            path = W.merkle_path(leaves, pos)
            assert W.root_from_path(leaves[pos], pos, count, path) == root
            assert W.root_from_path(bytes(32), pos, count, path) != root
            if path:
                assert W.root_from_path(leaves[pos], pos, count, path[:-1]) is None
                assert W.root_from_path(leaves[pos], pos, count, path + [root]) is None
            if count > 1:
                other = (pos + 1) % count
                assert W.root_from_path(leaves[pos], other, count, path) != root
        assert W.root_from_path(bytes(32), count, count, []) is None
    assert W.merkle_root([]) == tagged(W.NODE)


def test_draws_are_the_sources_and_uniform_over_credited_tiles():
    src, root, salt = b"\x01" * 32, b"\x02" * 32, b"\x03" * 32
    counts = [(4, 3), (1, 0), (2, 5)]
    d = W.draws(src, root, salt, counts, 4, 400)
    assert d == W.draws(src, root, salt, list(reversed(counts)), 4, 400) and len(d) == 400
    assert d != W.draws(b"\x04" * 32, root, salt, counts, 4, 400) and d != W.draws(src, root, b"\x05" * 32, counts, 4, 400)
    assert all({"card", "i", "tile"} == set(x) and x["card"] in (2, 4) and 0 <= x["tile"] < 4 for x in d)
    assert all(x["i"] < dict(counts)[x["card"]] for x in d)
    assert {(x["card"], x["i"]) for x in d} == {(2, i) for i in range(5)} | {(4, i) for i in range(3)}
    assert W.draws(src, root, salt, [(0, 0)], 4, 10) == []


def test_keys_are_the_fixtures_at_every_gemm():
    """A GEMM's keys are fixture.keys_of's for its one-unit epoch, and the shared ones the same at every matmul index."""
    sch, salt, root_b = W.scheme(), b"\x11" * 32, b"\x22" * 32
    seed_b = W.C._labelled(salt + root_b, "seed-B")
    const = W.constant_keys(sch, salt, root_b, W.SHAPE)
    for card, i in ((0, 0), (0, 1), (7, 12345)):
        mm = W.matmul(card, i)
        assert mm.index == card << 32 | i and mm.shape.m == mm.shape.k == mm.shape.n == 8192
        keys = W.F.keys_of(sch, W.gemm_epoch(salt, root_b, mm), mm, seed_b)
        assert W.gemm_keys(sch, salt, root_b, mm) == keys["dom_a"] + keys["unit"]
        assert len(keys["dom_a"]) == W.DOM and len(keys["unit"]) == W.UNIT
        assert const == {name: keys[name] for name in W.CONST_KEYS}
    assert W.gemm_keys(sch, salt, root_b, W.matmul(0, 0)) != W.gemm_keys(sch, salt, root_b, W.matmul(1, 0))


def test_salt_and_window_root_bind_their_inputs():
    src, root_b = b"\x07" * 32, b"\x08" * 32
    seed = W.window_seed(src, 10, ["GPU-a", "GPU-b"])
    assert seed != W.window_seed(src, 11, ["GPU-a", "GPU-b"]) and seed != W.window_seed(src, 10, ["GPU-b", "GPU-a"])
    salt = W.salt_of(seed, root_b, SMALL)
    assert salt == W.salt_of(seed, root_b, SMALL) != W.salt_of(seed, b"\x09" * 32, SMALL)
    cards = [{"card": 1, "uuid": "GPU-b", "count": 3, "root": "ab" * 32}, {"card": 0, "uuid": "GPU-a", "count": 2, "root": "cd" * 32}]
    r = W.window_root(salt, 10, 20, cards)
    assert r == W.window_root(salt, 10, 20, cards[::-1])
    assert r != W.window_root(salt, 10, 21, cards) and r != W.window_root(salt, 10, 20, [{**cards[0], "count": 4}, cards[1]])


def test_fraction_draws_and_gamma():
    assert A.draws_needed() == 59 and (1 - A.EPS_S) ** 59 <= A.DELTA_S < (1 - A.EPS_S) ** 58
    for t in (59, 60, 300, 600):
        e = A.eps_for(t)
        assert (1 - e) ** t <= A.DELTA_S < (1 - (e - Fraction(1, 10**6))) ** t and e <= A.EPS_S
    assert A.eps_for(300) == pytest.approx(0.00994, abs=1e-5)
    g = A.gamma(8192, 8192)
    assert g["gamma"] == "1648180439/446072750000" and g["status"] == "pinned"
    assert g["pin"] == "Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192"
    assert g["source"].endswith("lean-audit.json")
    f = A.fraction({0: 600, 3: 300}, {0: 1.5, 3: 2.0}, 1.8, Fraction(1, 100))
    assert f["capacity_gemms"] == 1200 + 900 and f["credited_gemms"] == 900
    assert f["fraction"] == pytest.approx(0.99 * 0.95 * 900 / 2100) and f["slowdown"] == pytest.approx(2100 / 900)
    assert "fraction" not in A.fraction({0: 1}, {0: 1.0}, 1.0, None)


def test_baseline_takes_the_fastest_cublaslt_median():
    items = [{"key": ["baseline", "m8-n8-k8", "fp8-e4m3", "lt_fp8-e4m3_0_0_algo35_tile20"], "method": "graph", "ms_per_call": x}
             for x in (1.5, 1.4, 1.6)]
    items += [{"key": ["baseline", "m8-n8-k8", "fp8-e4m3", "lt2_fp8-e4m3_algo1"], "method": "graph", "ms_per_call": 1.45},
              {"key": ["baseline", "m8-n8-k8", "fp8-e4m3", "verity_fp8_256x128_ew"], "method": "graph", "ms_per_call": 1.3},
              {"key": ["baseline", "m8-n8-k8", "fp8-e4m3", "lt_fp8-e4m3_1"], "method": "burst", "ms_per_call": 1.0},
              {"key": ["baseline", "m8-n8-k8", "fp8-e4m3", "lt_fp8-e4m3_1"], "method": "graph", "ms_per_call": 1.7},
              {"key": ["arm", "m8-n8-k8", "fp8-e4m3", "lt_fp8-e4m3_0"], "method": "graph", "ms_per_call": 0.1}]
    b = A.baseline({"exit": 0, "items": items}, "m8-n8-k8")
    assert b["cublaslt"] == ("lt2_fp8-e4m3_algo1", 1.45) and b["fastest"] == ("verity_fp8_256x128_ew", 1.3)
    assert b["medians_ms"]["lt_fp8-e4m3_0_0_algo35_tile20"] == 1.5 and b["medians_ms"]["lt_fp8-e4m3_1"] == 1.7


# ---- an honest window, as the cards write one ---------------------------------------------------------------------

T_S = 1_800_000_000 * 10**9
DURATION = 60.0
UUIDS = {0: "GPU-00000000-0000-0000-0000-000000000000", 1: "GPU-11111111-1111-1111-1111-111111111111"}


def dump(d: Path, name: str, data: bytes) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.bin").write_bytes(data)
    return {"path": f"{name}.bin", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def put(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def honest_window(out: Path, gemms: dict[int, int], shape=SMALL, hashing: str = "h1") -> None:
    """window.py's OUT for cards `gemms` ({card: credited GEMMs}), each GEMM proved by the scheme's reference prover and its
    trees laid out as the device's, the tampered control's leaves and tile tree a re-run's with one leaf byte flipped, and
    a cuBLASLt baseline of 1 ms a GEMM before and after."""
    m, n, k = shape
    sch, t_d = W.scheme(hashing), T_S + int(DURATION * 1e9)
    h = sch.commitment_hash
    b_rows = W.rows(W.SEED_B, 0, n, k)
    prover = Prover(sch, {W.WEIGHT: b_rows})
    root_b = prover.weight_roots[W.WEIGHT]
    tree_b = H.tree_bytes(rows_domain(weight_binding(W.WEIGHT), -2, n, h), b_rows, ROW_SCHEMA)
    assert tree_b[-32:] == root_b
    source = b"\x5a" * 32
    salt = W.salt_of(W.window_seed(source, T_S, [UUIDS[g] for g in sorted(gemms)]), root_b, shape)
    tiles = len(sch.tiles(W.mshape(shape)))
    put(out / "start.json", {"schema": W.SCHEMA, "scheme": sch.name, "hashing": hashing, "shape": {"m": m, "n": n, "k": k}, "source": source.hex(), "t_s_ns": T_S,
                             "t_s": W.utc(T_S), "t_d_ns": t_d, "t_d": W.utc(t_d), "duration_s": DURATION, "salt": salt.hex(),
                             "root_b": root_b.hex(), "cards": [{"card": g, "uuid": UUIDS[g]} for g in sorted(gemms)]})
    runs, closes = {}, []
    for g, count in sorted(gemms.items()):
        entries = []
        for i in range(count):
            mm = W.matmul(g, i, shape)
            a_rows = W.rows(W.a_seed(g, i & 1), 0, m, k)
            ep = W.gemm_epoch(salt, root_b, mm)
            tr = prover.prove(ep, {mm.index: a_rows})
            leaves = [prover.digest(t) for t in range(tiles)]
            tree_a = H.tree_bytes(rows_domain(activation_binding(salt, mm.index), mm.index, m, h), a_rows, ROW_SCHEMA)
            tree_t = H.tree_bytes(transcript_domain(ep, tiles, h), leaves, TILE_SCHEMA)
            assert (tree_a[-32:], tree_t[-32:]) == (tr.activation_roots[0], tr.commitment.root)
            runs[g, i] = (tree_a, tree_t, leaves, ep)
            entries.append(W.entry(g, i, tree_a[-32:], tree_t[-32:]))
        runs[g] = entries
        closes.append({"op": "close", "card": g, "uuid": UUIDS[g], "count": count, "root": W.merkle_root(entries).hex(),
                       "t_close_ns": t_d - 10**9, "t_recv_ns": t_d - 10**9 + 1})
    root = W.window_root(salt, T_S, t_d, closes)
    t_root = t_d - 10**8
    put(out / "window.json", {"schema": W.SCHEMA, "t_s_ns": T_S, "t_d_ns": t_d, "t_root_ns": t_root, "t_root": W.utc(t_root),
                              "on_time": True, "root": root.hex(), "cards": closes, "late": []})
    source2 = b"\xa5" * 32
    picked = W.draws(source2, root, salt, [(g, c) for g, c in gemms.items()], tiles, 60)
    opened = []
    for g in sorted(gemms):
        weights = {**dump(out / "weights" / f"card{g}", "tree_b", tree_b), "root": root_b.hex()}
        for i in sorted({d["i"] for d in picked if d["card"] == g}):
            tree_a, tree_t, leaves, _ = runs[g, i]
            d = out / "draws" / f"card{g}-gemm{i}"
            files = {"tree_a": dump(d, "tree_a", tree_a), "tree_t": dump(d, "tree_t", tree_t), "leaves": dump(d, "leaves", b"".join(leaves))}
            put(d / "manifest.json", {"card": g, "i": i, "index": W.index_of(g, i), "op_set": i & 1, "root_a": tree_a[-32:].hex(),
                                      "root_t": tree_t[-32:].hex(), "tiles": tiles, "flipped": False, "files": files,
                                      "path": [x.hex() for x in W.merkle_path(runs[g], i)]})
        opened.append({"op": "opened", "card": g, "weights": weights})
    tamper = picked[0]
    tree_a, _, leaves, ep = runs[tamper["card"], tamper["i"]]
    bad = [bytes([leaves[0][0] ^ 1]) + leaves[0][1:]] + leaves[1:]
    tree_t = H.tree_bytes(transcript_domain(ep, tiles, h), bad, TILE_SCHEMA)
    d = out / "controls" / "tampered"
    put(d / "manifest.json", {"card": tamper["card"], "i": tamper["i"], "index": W.index_of(tamper["card"], tamper["i"]),
                              "op_set": tamper["i"] & 1, "root_a": tree_a[-32:].hex(), "root_t": tree_t[-32:].hex(), "tiles": tiles,
                              "flipped": True, "files": {"tree_a": dump(d, "tree_a", tree_a), "tree_t": dump(d, "tree_t", tree_t),
                                                         "leaves": dump(d, "leaves", b"".join(bad))}})
    put(out / "draws.json", {"schema": W.SCHEMA, "source": source2.hex(), "t_source_ns": t_root + 1, "tiles_per_gemm": tiles,
                             "draws": picked, "tamper": tamper, "opened": opened, "worker_exits": [0] * len(gemms)})
    label = f"m{m}-n{n}-k{k}"
    for when in ("before", "after"):
        for g in gemms:
            put(out / f"bench-{when}" / f"card{g}" / "bench.json",
                {"exit": 0, "items": [{"key": ["baseline", label, "fp8-e4m3", "lt_fp8-e4m3_0_0_algo35_tile20"], "method": "graph",
                                       "ms_per_call": 1.0}]})


@pytest.fixture(scope="module", params=W.HASHINGS)
def honest(tmp_path_factory, request):
    out = tmp_path_factory.mktemp(f"exhaustion-window-{request.param}")
    honest_window(out, {0: 2, 1: 1}, hashing=request.param)
    return out


def failed(res: dict) -> set[str]:
    return {c["check"] for c in res["checks"] if not c["ok"]}


def by_name(res: dict) -> dict:
    return {c["check"]: c for c in res["checks"]}


@pytest.mark.slow
def test_audit_accepts_an_honest_window_and_rejects_both_controls(honest, monkeypatch):
    """Every drawn tile through Verifier.check_tile from the dumps (three GEMMs, each tile recomputed on the CPU), the
    tampered re-run's changed tile and the wrong salt's both rejected, and the fraction from the baselines and γ (8,192³'s:
    no twin is published at the test's shape, and without one the audit fails)."""
    monkeypatch.setattr(A, "gamma", lambda n, k, g=A.gamma(8192, 8192): g)
    argv = [str(honest), "--bench-before", str(honest / "bench-before"), "--bench-after", str(honest / "bench-after"), "--jobs", "4"]
    assert A.main(argv) == 0
    res = json.loads((honest / "audit.json").read_text())
    names = by_name(res)
    assert res["ok"] and {"salt", "deadline", "closes by t_d", "window root", "draw", "draws enough", "both controls", "gamma",
                          "baseline"} <= set(names)
    assert len(res["tiles"]) == 60 and all(t["ok"] for t in res["tiles"])
    assert {c["label"]: (c["ok"], c["reason"]) for c in res["controls"]} == {"tampered": (False, "checked values"),
                                                                              "wrong salt": (False, "activation opening")}
    f = res["fraction"]
    assert res["gemms"] == {"0": 2, "1": 1} and f["capacity_gemms"] == 2 * DURATION * 1000 and f["credited_gemms"] == 3
    gamma, eps = Fraction(res["gamma"]["gamma"]), A.eps_for(60)
    assert f["eps_s"] == str(eps) and f["fraction"] == pytest.approx(float((1 - gamma) * (1 - eps) * 3 / 120000))


@pytest.fixture
def quick(honest, tmp_path, monkeypatch):
    """A copy of the honest window, audited with every tile's recomputation taken as accepted (what each case breaks is
    upstream of the tile checks)."""
    out = tmp_path / "w"
    shutil.copytree(honest, out)
    monkeypatch.setattr(A, "check", lambda job: {**{x: job[x] for x in ("label", "card", "i", "tile")}, "ok": job["label"] not in
                                                ("tampered", "wrong salt"), "reason": "stub"})
    return out


def edit(path: Path, change) -> None:
    obj = json.loads(path.read_text())
    change(obj)
    path.write_text(json.dumps(obj))


def run(out: Path) -> dict:
    return A.audit(out, out / "bench-before", out / "bench-after", 1, "test")


@pytest.mark.slow
def test_audit_rejects_what_breaks_the_clock_the_roots_or_the_draw(quick, monkeypatch):
    assert failed(run(quick)) == {"gamma"}
    monkeypatch.setattr(A, "gamma", lambda n, k, g=A.gamma(8192, 8192): g)
    assert run(quick)["ok"]

    edit(quick / "window.json", lambda w: w["cards"][0].update(t_close_ns=w["t_d_ns"] + 1))
    assert failed(run(quick)) == {"closes by t_d"}
    edit(quick / "window.json", lambda w: w["cards"][0].update(t_close_ns=w["t_d_ns"] - 1))

    edit(quick / "window.json", lambda w: w.update(t_root_ns=w["t_d_ns"] + 1))
    assert "window root" in failed(run(quick))
    edit(quick / "window.json", lambda w: w.update(t_root_ns=w["t_d_ns"] - 1))

    edit(quick / "window.json", lambda w: w["cards"][1].update(count=w["cards"][1]["count"] + 1))
    assert {"window root", "draw"} <= failed(run(quick))
    edit(quick / "window.json", lambda w: w["cards"][1].update(count=w["cards"][1]["count"] - 1))

    edit(quick / "start.json", lambda s: s.update(t_s_ns=s["t_s_ns"] + 1))
    assert {"salt", "deadline"} <= failed(run(quick))
    edit(quick / "start.json", lambda s: s.update(t_s_ns=s["t_s_ns"] - 1))

    edit(quick / "draws.json", lambda d: d.update(t_source_ns=d["t_source_ns"] - 2))
    assert failed(run(quick)) == {"draw"}
    edit(quick / "draws.json", lambda d: d.update(t_source_ns=d["t_source_ns"] + 2))

    edit(quick / "draws.json", lambda d: d.update(draws=d["draws"][:58]))
    assert {"draw", "draws enough"} <= failed(run(quick))

    first = json.loads((quick / "draws.json").read_text())["draws"][0]
    man = quick / "draws" / f"card{first['card']}-gemm{first['i']}" / "manifest.json"
    edit(man, lambda m: m.update(root_t="00" * 32))
    assert any(x.startswith("draw ") and x.endswith(" entry") for x in failed(run(quick)))


# ---- window.py on the stand-in -------------------------------------------------------------------------------------

@pytest.mark.slow
def test_window_dry_run(ship, tmp_path):  # noqa: F811
    """The parent and one worker through the whole window on the stand-in, under an injected clock: identity, B's root, the
    warm-up, the salt, GEMMs in batches until the guard, the close by t_d, the draw and the re-runs, every launch and copy
    inside its buffers; then two workers (each its own card), and their dumps through the audit, which finds the window's
    clock, roots and draw in order and rejects every tile (the stand-in computes nothing)."""
    md = tmp_path / "server.md"
    md.write_text(f"| 3 | `{UUID}` |\n| 5 | `{UUID}` |\n")
    for schedule, hashing in (("serial", "h1"), ("overlap", "h1"), ("overlap", "h2")):
        log, out = tmp_path / f"launches-{schedule}-{hashing}.jsonl", tmp_path / f"one-{schedule}-{hashing}"
        env = dry_env(ship, log, CUDA_VISIBLE_DEVICES="3", EXH_CLOCK=f"fake:{T_S}:{10**8}", PEARLC_SHIP=str(ship))
        r = subprocess.run([sys.executable, str(HERE / "window.py"), "run", "--out", str(out), "--server-md", str(md),
                            "--duration", "2", "--guard", "1", "--draws", "8", "--shape", "128", "128", "1024",
                            "--schedule", schedule, "--hashing", hashing], env=env, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        assert r.returncode == 0, r.stderr
        start, win = json.loads((out / "start.json").read_text()), json.loads((out / "window.json").read_text())
        dr = json.loads((out / "draws.json").read_text())
        assert start["t_d_ns"] == start["t_s_ns"] + 2 * 10**9 and start["schedule"] == schedule
        assert start["scheme"] == f"pearl-c-sm120-v1-{hashing}"
        assert win["on_time"] and not win["late"] and len(win["cards"]) == 1
        (c,) = win["cards"]
        assert c["count"] > 0 and c["count"] % W.BATCH == 0 and c["late"] == 0 and c["t_close_ns"] <= start["t_d_ns"]
        assert dr["tiles_per_gemm"] == 4 and len(dr["draws"]) == 8 and dr["worker_exits"] == [0]
        assert dr["opened"][0]["mismatched"] == [] and dr["opened"][0]["tampered"]["i"] == dr["draws"][0]["i"]
        allocs, maps, seen = {}, set(), set()
        for ev in map(json.loads, log.read_text().splitlines()):
            if ev["op"] == "copy":                   # the roots' copies into the log ring: inside a buffer, not at its start
                assert room(allocs, ev["dst"]) >= ev["bytes"] and room(allocs, ev["src"]) >= ev["bytes"], ev
            else:
                track(ev, allocs, maps)
            if ev["op"] == "launch":
                seen.add(ev["name"])
        rows, leaves = ("hash_rows_w", "hash_tree_leaves") if hashing == "h1" else ("hash_rows_b3s", "hash_leaves_b3s")
        assert {"gemm_pearlc_b", "seed_line_a", "hash_leaf_p", rows, leaves} <= seen
        assert not seen & {"gemm_pearlc", "hash_msg_v", "gemm_ct", "gemm_plain"}

    out = tmp_path / "two"
    env = dry_env(ship, Path("/dev/null"), CUDA_VISIBLE_DEVICES="3,5", EXH_CLOCK=f"fake:{T_S}:{10**8}", PEARLC_SHIP=str(ship))
    r = subprocess.run([sys.executable, str(HERE / "window.py"), "run", "--out", str(out), "--server-md", str(md),
                        "--duration", "2", "--guard", "1", "--draws", "60", "--shape", "128", "128", "1024"],
                       env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    res = A.audit(out, None, None, 1, "test")
    names = by_name(res)
    for ok in ("salt", "deadline", "closes by t_d", "window root", "tiles per GEMM", "draw", "draws enough"):
        assert names[ok]["ok"], names[ok]
    assert all(c["ok"] for n, c in names.items() if n.endswith(" entry") or n.endswith(" B tree"))
    assert not res["ok"] and not any(t["ok"] for t in res["tiles"]) and not names["baseline"]["ok"]

    r = subprocess.run([sys.executable, str(HERE / "window.py"), "run", "--out", str(tmp_path / "bad"), "--server-md", str(md),
                        "--duration", "2", "--shape", "128", "128", "1024"],
                       env={**env, "CUDA_VISIBLE_DEVICES": "3,4"}, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert r.returncode == W.EXIT_USAGE
