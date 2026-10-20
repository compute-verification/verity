"""RT4-INT's witnesses on the certified served window (#2847, `served_commit.Challenger(certifier=…)`) as SALT-WIRE left
it at a6d6159a4a: what a drawn unit's proof is held to, from the network certifier's reveal to `salt.Registrations`.

Every served-zk certified window is refused ``unsalted`` today (`salt.unit_reason` of `Pc8TileSide`), so nothing is
credited. `unit_reason` reads a template's ports alone: it admits `PouwH1Tile`, whose proof takes a salt as input, and
nothing in the certified path holds that input to the salt the verifier computes (`salt.pouw_salt`). `pouw_salt` reaches
the manifest check (`core_check._certified_salt`) only; the drawn unit's check of record (`core_check.opening`, Lean
`Opening`) takes no salt and pins no salted circuit, and `Compiled.inputs` has no caller outside tests. So past
`unit_reason`, rows formed before any certifier salt pass every certified check, and a salted unit wired into this path as
it stands is proved under whatever salt its prover chose.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from verity.ml.crypto import lms
from verity.ml.randomness import derive
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.active import RUN_ID_BYTES, Proxy, RemoteCertifier
from verity.core.protocols.certifier.certificate import LmsKey, LmsSigner
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import pc8
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.one_stage import partition as PT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402
import of_record as OR  # noqa: E402

H1 = Path(__file__).resolve().parents[1] / "h1_prove"
sys.path.insert(0, str(H1))
import h1unit as U  # noqa: E402

_spec = importlib.util.spec_from_file_location("rt4_h1_prove_run", H1 / "run.py")
RN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RN)

K, M, N = 128, 3, 4
EGRESS, INGRESS = GridParams(buckets=6, allocation=3, frame_bytes=512), GridParams(buckets=6, allocation=2, frame_bytes=512)
LINK = "requests"
#: a PoUW window is two certifier windows; the proxy serves four, so PoUW windows 0 and 1
POUW_WINDOW, WINDOWS = 2, 4
SCHEME = C.PearlC(k_min=K, forming="v1", device=DEVICES["sm120"], hashing="h3")
MATMUL = Matmul(5, Shape(16, K, N), 2)
RUN = "r-rt4"


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


def _sid(stream: str = "window") -> bytes:
    return SC.stream_id({"deployment": f"served-zk/{RUN}"}, stream, {"link": LINK, "origin": (0.0).hex()})


def _certifier(sid: bytes, key, sign, record: bytes) -> tuple[Proxy, dict[int, bytes]]:
    """The certifier on `LINK` from origin 0 with PoUW windows 0 and 1, the prover's record taken for both before their
    salts are drawn, both published; and the salts it drew, which the prover's server receives in-band."""
    drawn: dict[int, bytes] = {}

    def source(p: int, n: int) -> bytes:
        drawn[p] = hashlib.shake_256(b"certifier salt %d" % p).digest(n)
        return drawn[p]

    c = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SyncSet((0, 1), (1, 2)), [ClockSync(0, 1)] * WINDOWS, origin=0.0,
              clock=lambda: 0.0, ingress_link=LINK, salt=source, registered=PS.registered, prover_key=key,
              prover_context=sid, pouw_window=POUW_WINDOW)
    for p in range(WINDOWS // POUW_WINDOW):
        c.commit_weights(p, record, sign(active.weights_message(LINK, p, record, sid)))
    for _ in range(WINDOWS * INGRESS.buckets):
        c.bucketer.tick()
    return c, drawn


def _live(path: Path) -> tuple[Ch.LiveChallenger, Ch.ChallengerKey]:
    """A live epoch's challenger, its record signed and written to `path`, and the verifier's key for that record."""
    k = lms.keygen(lms.Params(64, 4, 2), hashlib.sha512(str(path).encode()).digest())
    live = Wd.live_challenger(0, 1, random.Random(str(path)).randbytes)
    Ch.write_record(path, live.sign_record("auditor-host", lms.public_key(k), lambda m: lms.sign_at(k, 0, m).encode()))
    return live, Ch.ChallengerKey(4, 2, lms.public_key(k))


def _challenger(params, c, prover, path: Path, ledger: PS.Registrations) -> SC.Challenger:
    live, ck = _live(path)
    return SC.Challenger(params, live, clock=lambda: 0.0, record=OR.Record(lean=False), certifier=c, prover=prover,
                         coin_record=path, coin_key=ck, registrations=ledger)


def _window(p: int, reveal: bytes, sid: bytes, prover: bytes, words) -> dict:
    """A certified window of PoUW window `p`: its manifest salted `pouw_salt(reveal, p, sid, prover)`, its entries the
    rows `words`, committed as `served_commit.commit_window` commits them."""
    man, roots, work = _pass(RUN)
    raw = json.dumps({**man, "schema": CC.MANIFEST_CERTIFIED,
                      "salt": Epoch.certified(work, roots, [POUW_WINDOW * p + i for i in range(POUW_WINDOW)], POUW_WINDOW,
                                              reveal, sid, prover).salt.hex()}, sort_keys=True).encode()
    params = {"context": PS.context(f"served-zk/{RUN}", p, hashlib.sha512(raw).hexdigest()),
              "deployment": f"served-zk/{RUN}", "k": K, "eps": "1/2", "delta": "1/10", "grid_bits": Wd.GRID_BITS,
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


def _proved(w: dict, units) -> None:
    for u in units:
        assert SC.check(w["params"], w["rec"], {"units": list(units)}, u, _pub(w["priv"], w["advice"], u),
                        SC.opening(w["advice"], w["priv"], w["params"], u), record=OR.Record(lean=False)) is None


def test_the_certified_windows_drawn_unit_reads_no_salt_and_is_refused_unsalted():
    """The served window's population is one template, `Pc8TileSide`, whose ports carry no public input region, so no
    seed, noise line or salt is an input of any drawn unit's proof; `salt.unit_reason` refuses it, so every certified
    window is refused ``unsalted`` and nothing is credited."""
    d = SC.template(K)
    assert d.id.startswith("Pc8TileSide")
    assert pc8.SIDE_PORTS["regions"] == {} and pc8.PORTS["row_a"]["regions"] and pc8.PORTS["row_b"]["regions"]
    prog, q, part = SC.population(K, 26)
    assert q == PT.template_instance_query(PT.template_id(d))
    assert PS.unit_reason(d) is not None and d.id in PS.unit_reason(d)


def test_past_unit_reason_rows_formed_before_any_salt_pass_every_certified_check(tmp_path, monkeypatch):
    """RT4-INT-10, the served path. `unit_reason` is the one rule between RT4-INT-1 and a credit: lift it (as a salted
    template would pass it) and the prover's rows, formed once under a salt that is neither window's, committed twice
    each in PoUW windows 0 and 1, pass the certifier's record and reveal, `pass_epoch` of record, the registration
    ledger, the live draw under the signed record, and `core_check.opening` of record for every drawn unit. The two
    windows' tiles are the same words: nothing after `unit_reason` reads a drawn proof's salt."""
    monkeypatch.setattr(PS, "unit_reason", lambda unit: None)
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), _sid()
    c, drawn = _certifier(sid, key, sign, PS.pouw_record(weights_root(_pass(RUN)[1]), sid, prover))
    ledger, windows = PS.Registrations(), []
    for p in (0, 1):
        w = _window(p, drawn[p], sid, prover, PRECOMPUTED)
        assert _words(w["salt"]) != PRECOMPUTED, "the honest committer's rows under this window's salt differ"
        ch = _challenger(w["params"], c, prover, tmp_path / f"w{p}.json", ledger)
        ch.register(f"w{p}", w["rec"], manifest=w["raw"])
        assert ch.passes == {f"w{p}": "certifier"} and (p, sid, prover) in ledger
        d = ch.select(f"w{p}")
        assert d.units
        _proved(w, d.units)
        windows.append(w)
    assert len(ledger) == 2 and windows[0]["salt"] != windows[1]["salt"]
    real = 2 * M * N
    tiles = [[_tile_words(w, t) for t in range(real)] for w in windows]
    assert tiles[0] == tiles[1] and len(set(tiles[0])) == M * N == real // 2


def test_nothing_holds_a_drawn_salted_proofs_salt_input_to_pouw_salt():
    """RT4-INT-10, the salted units. `unit_reason` admits the H1 tile unit on its ports. Its statement's only salt is
    the public row ``pub`` (salt ‖ position ‖ roots), whose content the verifier must fill itself: H1-PROVE's verifier
    does (`run.unit_bcs`, from the salt its caller passes, its own harness coin's `unit_salt`), and the certified path has
    no caller that passes `pouw_salt`: `core_check.opening` takes no salt and pins only `Pc8Tile*` circuits, `pouw_salt`
    enters `core_check` only at the manifest check, and nothing in `served_commit` or `core_check` calls
    `Compiled.inputs` or `h1unit.pub_words`. A prover that picks its own salt s* gets the exact z every salt gives and
    the receipt of s*, all computable before any certifier salt; its statement differs from the one under
    `pouw_salt(reveal, p, stream, prover)` in the public row alone."""
    shape = U.Shape(64)
    assert PS.unit_reason(shape.definition()) is None
    assert "salt" not in inspect.signature(CC.opening).parameters
    assert CC.TILES and all(t.startswith("Pc8Tile") for t in CC.TILES)
    core, served = inspect.getsource(CC), inspect.getsource(SC)
    assert core.count("PS.pouw_salt(") == 1 and "PS.pouw_salt(" in inspect.getsource(CC._certified_salt)
    assert not any(s in src for src in (core, served) for s in (".inputs(", "pub_words", "unit_salt"))

    key, _ = _prover(b"the prover")
    certified = PS.pouw_salt(hashlib.shake_256(b"certifier salt 0").digest(PS.SALT_BYTES), 0, _sid(),
                             PS.prover_id(key.public))
    own = hashlib.sha512(b"the prover's own salt, chosen before any certifier window").digest()
    rng = np.random.default_rng(10)
    a = rng.integers(-126, 126, (4, 64)).tolist()
    b = rng.integers(-110, 111, (8, 64)).tolist()
    roots, position = (hashlib.sha512(b"root_a").digest(), hashlib.sha512(b"root_b").digest()), 5
    z_own, receipt_own = H.reference(a, b, own, position, roots)
    z_cert, receipt_cert = H.reference(a, b, certified, position, roots)
    assert z_own == z_cert and receipt_own != receipt_cert

    com = {"window": "w-rt4", "calls": [{"name": "c", "tile0": 0, "tiles": 8, "n": 8, "root_a": roots[0].hex(),
                                         "root_b": roots[1].hex()}]}
    op = {"unit": position, "rows": {p: {"bc": hashlib.sha512(p.encode()).digest().hex() * 2}
                                     for p in [*shape.in_rows, "o0", "o1"]}}
    held = {s: RN.unit_bcs(shape, com, s, op) for s in (own, certified)}
    assert held[own]["pub"] != held[certified]["pub"]
    assert {p: v for p, v in held[own].items() if p != "pub"} == {p: v for p, v in held[certified].items() if p != "pub"}
    assert not np.array_equal(U.pub_words(own, position, roots), U.pub_words(certified, position, roots))


def _prover_record(seed: bytes, coins_seed: str, reg: str, stream: str, receipt: dict, n: int, k: int) -> dict:
    """A live draw whose challenger the prover runs on coins of its own (`coins_seed`), its record signed under a key the
    prover made from `seed`: the draw's check of record's input as `pouw_pass_cases.honest_live_draw` lays it out."""
    ch = Wd.live_challenger(0, 1, random.Random(coins_seed).randbytes)
    sk = lms.keygen(lms.Params(64, 4, 2), hashlib.sha512(seed).digest())
    message = Ch.record_message(lms.public_key(sk)[:16], "provers-node", ch.commitment)
    record = Ch.ChallengerRecord(ch.commitment, "provers-node", lms.sign_at(sk, 0, message).encode())
    o = ch.open(Wd.DRAW, 0, bytes.fromhex(reg))
    ctx, size = CC.draw_key_context(stream, reg, receipt), 2 * k * CC.width(n) + 64
    units = CC.stream_subset(k, n, derive(o.coin, CC.DRAW_DOMAIN, ctx).stream("draw", size))
    return {"coins": {"key": Ch.ChallengerKey(4, 2, lms.public_key(sk)).to_json(), "record": record.to_json()},
            "recorded": False, "opening": o.to_json(), "stream": stream, "registration": reg, "receipt": receipt, "k": k,
            "population": n, "units": units}


def test_the_draws_check_of_record_verifies_the_record_under_the_key_its_input_carries():
    """RT4-INT-5, narrowed. The certified challenger now reads its commitment itself from the signed record under the
    `coin_key` its caller gives, but the draw's check of record (`core_check.draw`, Lean `drawCheck`) verifies the record
    under ``coins["key"]``, which the same input carries, and no parameter pins it. A record a prover signed under its
    own key over coins it holds passes; holding the coins, the prover sees each candidate registration's draw before it
    registers and keeps one whose draw misses the unit it can't prove."""
    stream, n, k, bad = hashlib.sha512(b"stream").hexdigest(), 26, 3, 0
    for i in itertools.count():
        reg = hashlib.sha512(b"candidate %d" % i).hexdigest()
        receipt = {"format": "verity/registration-receipt/v1", "received": str(i), "registration": reg,
                   "verifier": "served-zk/challenger/window"}
        d = _prover_record(b"the prover's challenger key", "the prover's coins", reg, stream, receipt, n, k)
        if bad not in d["units"]:
            break
    assert CC.draw(*(d[x] for x in ("coins", "recorded", "opening", "stream", "registration", "receipt", "k",
                                    "population", "units"))) is None
    other = _prover_record(b"another key", "the prover's coins", reg, stream, receipt, n, k)
    assert other["coins"]["key"] != d["coins"]["key"] and other["units"] == d["units"]
    assert CC.draw(*(other[x] for x in ("coins", "recorded", "opening", "stream", "registration", "receipt", "k",
                                        "population", "units"))) is None
    assert "key" not in inspect.signature(CC.draw).parameters


def test_a_registered_window_is_admitted_before_its_draw_and_its_failure_closes_nothing(tmp_path, monkeypatch):
    """RT4-INT-6, renamed. `Registrations` admits a certified window when the challenger receives it, before its draw
    coin is open and before any proof (now named a registration ledger, not credit). Window 0 is received, drawn and
    never proved (a visible failure with no outcome), and window 1 of the same stream and prover is received and drawn
    after it: nothing in the served path holds a period (#2722's TODO), so no refusal closes its stream's credit."""
    monkeypatch.setattr(PS, "unit_reason", lambda unit: None)
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), _sid()
    c, drawn = _certifier(sid, key, sign, PS.pouw_record(weights_root(_pass(RUN)[1]), sid, prover))
    ledger, chs = PS.Registrations(), []
    for p in (0, 1):
        w = _window(p, drawn[p], sid, prover, PRECOMPUTED)
        ch = _challenger(w["params"], c, prover, tmp_path / f"w{p}.json", ledger)
        ch.register(f"w{p}", w["rec"], manifest=w["raw"])
        assert (p, sid, prover) in ledger and "draw" not in ch.openings
        assert ch.select(f"w{p}").units
        chs.append(ch)
    assert chs[0].sessions == {} and len(ledger) == 2


def test_the_network_certifiers_client_is_no_certifier_a_certified_challenger_takes(tmp_path):
    """RT4-INT-2's link 8, with NET-50 (#2879 @ 1e1ad3967d) merged on SALT-WIRE (#2847 @ a6d6159a4a). NET-50's
    `RemoteCertifier` is to replace the in-process `Proxy` (`Challenger(certifier=RemoteCertifier(...))`, its PR's
    text), but the certified challenger names its certifier's run by `certifier_run`, which reads the proxy's
    `bucketer.certifier.link` and `origin`; the client holds `link` and `run_id` and no origin, so the challenger can't
    be built on it. And the two name a run apart: NET-50 by the run id each proxy draws, the stream id by link and
    origin, so two proxies on one link from one origin are two runs to the certifier and one stream to PoUW."""
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), _sid()
    _, drawn = _certifier(sid, key, sign, PS.pouw_record(weights_root(_pass(RUN)[1]), sid, prover))
    w = _window(0, drawn[0], sid, prover, PRECOMPUTED)
    ck = lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"the network certifier").digest())
    challenger = LmsSigner(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"the challenger").digest()), tmp_path / "s")
    remote = RemoteCertifier("127.0.0.1", 9, LmsKey(ck.params, lms.public_key(ck)), challenger, LINK,
                             bytes([7]) * RUN_ID_BYTES)
    assert not hasattr(remote, "bucketer") and not hasattr(remote, "origin")
    with pytest.raises(AttributeError, match="bucketer"):
        _challenger(w["params"], remote, prover, tmp_path / "w.json", PS.Registrations())

    runs = [Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SyncSet((0, 1), (1, 2)), [ClockSync(0, 1)] * WINDOWS, origin=0.0,
                  clock=lambda: 0.0, ingress_link=LINK, run_nonce=lambda n, b=b: bytes([b]) * n) for b in (1, 2)]
    assert runs[0].run_id != runs[1].run_id and SC.certifier_run(runs[0]) == SC.certifier_run(runs[1])
