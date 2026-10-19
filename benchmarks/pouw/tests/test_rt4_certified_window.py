"""RT4-INT's witnesses on the certified served window (#2847, `served_commit.Challenger(certifier=…)`): what its credit
reads, from the network certifier's reveal to `salt.Credits`.

A certified window's drawn unit is `Pc8TileSide` (`served_commit.template`), whose proof reads the drawn tile's two
committed side rows and nothing seeded (`pc8.SIDE_PORTS` has no public input region), and `core_check.opening` holds the
proof's rows to the window root only. So the salt `core_check.pass_epoch` checks in the manifest reaches nothing a drawn
unit proves: rows formed under any salt, or under none, are credited in every certified window and under every call.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
from verity.ml.crypto import lms
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.active import Proxy
from verity.core.protocols.certifier.certificate import LmsKey
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root
from verity.core.protocols.pouw.circuit import pc8
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.one_stage import partition as PT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402
import of_record as OR  # noqa: E402

K, M, N = 128, 3, 4
EGRESS, INGRESS = GridParams(buckets=6, allocation=3, frame_bytes=512), GridParams(buckets=6, allocation=2, frame_bytes=512)
#: a PoUW window is two certifier windows; the proxy serves four, so PoUW windows 0 and 1
POUW_WINDOW, WINDOWS = 2, 4
SCHEME = C.PearlC(k_min=K, forming="v1", device=DEVICES["sm120"], hashing="h3")
MATMUL = Matmul(5, Shape(16, K, N), 2)


def _rows(seed: int, n: int) -> list[bytes]:
    return [r.tobytes() for r in np.random.default_rng(seed).standard_normal((n, K)).astype("<f4")]


def _words(salt: bytes) -> dict:
    """The fp8-p rows a committer forms for one call under `salt`, as `served_commit.form` forms them."""
    tag = lambda s: hashlib.sha256(s).digest()  # noqa: E731
    ctx = Context(salt, MATMUL, tag(b"root_a"), tag(b"root_b"))
    noise = SCHEME.noise(ctx)
    a = _rows(1, M) + [bytes(4 * K)]
    return {"a": SC.form(SCHEME, ctx, noise, K, "a", a, list(range(M + 1))),
            "b": SC.form(SCHEME, ctx, noise, K, "b", _rows(2, N), list(range(N)))}


#: rows formed once, before any certifier salt exists, under a salt that is no window's
PRECOMPUTED = _words(hashlib.sha256(b"formed before any certifier window").digest())


def _calls(words) -> list[dict]:
    """Two calls (matmuls 5 and 9) over the same formed rows."""
    return [{"record": {"matmul": m, "shape": [16, K, N], "tile0": m, "weight": 2, "real_rows": M + 1}, "words": words}
            for m in (5, 9)]


def _pass(run: str) -> tuple[dict, dict[int, bytes], Workload]:
    roots = {2: hashlib.sha512(run.encode()).digest()}
    man = {"run": run, "matmuls": [{"index": 5, "m": 16, "k": K, "n": N, "weight": 2}],
           "weights": {"2": {"root": roots[2].hex()}},
           "commitment": {"root": hashlib.sha512(f"{run}/tiles".encode()).hexdigest(), "count": 1}}
    return man, roots, Workload((MATMUL,))


def _prover(seed: bytes):
    k, leaves = lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(seed).digest()), itertools.count()
    return LmsKey(k.params, lms.public_key(k)), lambda m: lms.sign_at(k, next(leaves), m).encode()


def _certifier(sid: bytes, key, sign, record: bytes) -> tuple[Proxy, dict[int, bytes]]:
    """The certifier with PoUW windows 0 and 1, the prover's record taken for both before their salts are drawn, both
    published; and the salts it drew, which the prover's server receives in-band."""
    drawn: dict[int, bytes] = {}

    def source(p: int, n: int) -> bytes:
        drawn[p] = hashlib.shake_256(b"certifier salt %d" % p).digest(n)
        return drawn[p]

    c = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SyncSet((0, 1), (1, 2)), [ClockSync(0, 1)] * WINDOWS, origin=0.0,
              clock=lambda: 0.0, salt=source, registered=PS.registered, prover_key=key, prover_context=sid,
              pouw_window=POUW_WINDOW)
    for p in range(WINDOWS // POUW_WINDOW):
        c.commit_weights(p, record, sign(active.weights_message(c.bucketer.certifier.link, p, record, sid)))
    for _ in range(WINDOWS * INGRESS.buckets):
        c.bucketer.tick()
    return c, drawn


def _live(path: Path) -> tuple[Ch.LiveChallenger, Ch.LiveCommitment]:
    p = lms.Params(64, 4, 2)
    k = lms.keygen(p, hashlib.sha512(str(path).encode()).digest())
    live = Wd.live_challenger(0, 1, random.Random(str(path)).randbytes)
    Ch.write_record(path, live.sign_record("auditor-host", lms.public_key(k), lambda m: lms.sign_at(k, 0, m).encode()))
    return live, Ch.trusted_commitment(path, Ch.ChallengerKey(4, 2, lms.public_key(k)),
                                       lambda key, m, s: lms.verify(p, key.public, m, lms.Signature.decode(p, s)))


def _window(p: int, reveal: bytes, sid: bytes, prover: bytes, words) -> dict:
    """A certified window of PoUW window `p`: its manifest salted `pouw_salt(reveal, p, sid, prover)`, its entries the
    rows `words`, committed as `served_commit.commit_window` commits them."""
    run = "r-rt4"
    man, roots, work = _pass(run)
    raw = json.dumps({**man, "schema": CC.MANIFEST_CERTIFIED,
                      "salt": Epoch.certified(work, roots, [POUW_WINDOW * p + i for i in range(POUW_WINDOW)], POUW_WINDOW,
                                              reveal, sid, prover).salt.hex()}, sort_keys=True).encode()
    params = {"context": PS.context(f"served-zk/{run}", p, hashlib.sha512(raw).hexdigest()),
              "deployment": f"served-zk/{run}", "k": K, "eps": "1/2", "delta": "1/10", "grid_bits": Wd.GRID_BITS,
              "range": [1, 1 << 40], "entries": 4, "w_rows": N}
    root_w, _ = SC.commit_weights(params, [np.arange(K, dtype="<u2").tobytes() for _ in range(N)])
    public, advice, priv = SC.commit_window(params, _calls(words), root_w, SC.root_h3_of(man))
    return {"params": params, "raw": raw, "rec": SC.register(params, public), "advice": advice, "priv": priv,
            "salt": bytes.fromhex(json.loads(raw)["salt"])}


def _pub(priv, advice, t: int) -> bytes:
    """The drawn unit's proof's public file as `verity_flock.circuit.write` lays it out: the header over the circuit the
    verifier pins, then the unit's A row's and B row's b ‖ c."""
    from verity.core.protocols.pouw.circuit import leaves as L
    e, i, j = Wd.tile_at(SC.entries_of(advice), t)
    schemas = {"a0": L.ROW_SCHEMA["fp8-pi", SC.SIDES["a"]], "b0": L.ROW_SCHEMA["fp8-pi", SC.SIDES["b"]],
               "o0": L.ROW_SCHEMA["tile", SC.SIDES["a"]]}
    head = {"circuit_sha512": CC.TILES[SC.template(K).id].circuit, "instances": 1, "frame_v3": {"schemas": schemas}}
    return json.dumps(head).encode() + b"\n" + priv[e]["a"]["bcs"][i] + priv[e]["b"]["bcs"][j] + bytes(128)


def _tile_words(w: dict, t: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    e, i, j = Wd.tile_at(SC.entries_of(w["advice"]), t)
    return tuple(int(x) for x in w["priv"][e]["a"]["words"][i]), tuple(int(x) for x in w["priv"][e]["b"]["words"][j])


def test_the_certified_windows_drawn_unit_reads_no_salt():
    """The served window's population is one template, `Pc8TileSide`, and its ports carry no public input region, so no
    seed, noise line or salt is an input of any drawn unit's proof; the row units that read the seeds' noise regions
    (`Pc8RowA`, `Pc8RowB`) are in no served window."""
    d = SC.template(K)
    assert d.id.startswith("Pc8TileSide")
    assert pc8.SIDE_PORTS["regions"] == {} and pc8.PORTS["row_a"]["regions"] and pc8.PORTS["row_b"]["regions"]
    prog, q, part = SC.population(K, 26)
    assert q == PT.template_instance_query(PT.template_id(d))


def test_rows_formed_before_any_salt_are_credited_in_two_certified_windows(tmp_path):
    """RT4-INT-1. The prover forms one call's rows once, under a salt that is neither window's, and commits them, twice
    each (matmuls 5 and 9), in PoUW windows 0 and 1, each salted by its own certifier reveal. Both windows pass the
    certifier's record, the reveal, `pass_epoch` of record, `Credits` (shared) and the live draw under the trusted
    commitment, and every drawn unit's opening passes `core_check.opening` of record. The two windows' tiles are the
    same words, and each window's 24 real tiles are 12 distinct products: computed once, credited four times."""
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), SC.stream_id({"deployment": "served-zk/r-rt4"}, "window")
    record = PS.pouw_record(weights_root(_pass("r-rt4")[1]), sid, prover)
    c, drawn = _certifier(sid, key, sign, record)
    credits = PS.Credits()
    windows = []
    for p in (0, 1):
        w = _window(p, drawn[p], sid, prover, PRECOMPUTED)
        assert _words(w["salt"]) != PRECOMPUTED, "the honest committer's rows under this window's salt differ"
        live, trusted = _live(tmp_path / f"w{p}.json")
        ch = SC.Challenger(w["params"], live, clock=lambda: 0.0, record=OR.Record(lean=False), certifier=c,
                           prover=prover, trusted=trusted, credits=credits)
        reg = ch.register(f"w{p}", w["rec"], manifest=w["raw"])
        assert ch.passes == {f"w{p}": "certifier"}
        d = ch.select(f"w{p}")
        assert d.units
        for u in d.units:
            assert SC.check(w["params"], w["rec"], {"units": list(d.units)}, u, _pub(w["priv"], w["advice"], u),
                            SC.opening(w["advice"], w["priv"], w["params"], u), record=OR.Record(lean=False)) is None
        windows.append({**w, "reg": reg, "units": d.units})
    assert set(credits._credited) == {(0, sid + prover), (1, sid + prover)}
    assert windows[0]["salt"] != windows[1]["salt"]
    real = 2 * M * N
    tiles = [[_tile_words(w, t) for t in range(real)] for w in windows]
    assert tiles[0] == tiles[1]
    assert len(set(tiles[0])) == M * N == real // 2
    ell = SC.level_of(windows[0]["rec"])
    assert Wd.Work.at(ell, Wd.GRID_BITS).credited > M * N


def test_a_certified_challenger_takes_any_live_commitment_as_trusted(tmp_path):
    """RT4-INT-5. `Challenger(certifier=…, trusted=…)` asks only that `trusted` be a live epoch's `LiveCommitment`: the
    commitment of a challenger object the caller holds, never read from a signed record, is taken as the trusted one,
    and the draw's check of record (`core_check.draw`, Lean `drawCheck`) reads that commitment's JSON and no signature."""
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), SC.stream_id({"deployment": "served-zk/r-rt4"}, "window")
    record = PS.pouw_record(weights_root(_pass("r-rt4")[1]), sid, prover)
    c, drawn = _certifier(sid, key, sign, record)
    w = _window(0, drawn[0], sid, prover, PRECOMPUTED)
    held = Wd.live_challenger(0, 1, random.Random("held by whoever runs the challenger").randbytes)
    ch = SC.Challenger(w["params"], held, clock=lambda: 0.0, record=OR.Record(lean=False), certifier=c, prover=prover,
                       trusted=held.commitment)
    ch.register("w", w["rec"], manifest=w["raw"])
    assert ch.select("w").units and ch.coins_of("w")["commitment"] == held.commitment.to_json()
