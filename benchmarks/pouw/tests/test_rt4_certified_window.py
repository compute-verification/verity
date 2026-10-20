"""RT4-INT's witnesses on the certified served window (#2847, `served_commit.Challenger(certifier=…)`) at c50849bb22,
which merges SALT-BIND (#2912, the opening of record under the certified salt) and NET-50 (#2879, `RemoteCertifier`),
and at 154aa408ac (the deployment's pinned challenger key, RT4-INT-5).

Every served-zk certified window is still refused: its population is one template, `Pc8TileSide`, which reads no salt.
RT4-INT-10's two witnesses (rows formed before any salt passing every certified check, and a salted unit proved under a
salt its prover chose) now refuse at the opening, and `benchmarks/pouw/tests/test_salt_bind.py` holds them as refusals;
the opening difftest's ``salted-position`` and ``salted-roots`` cases hold the public row's position and roots. Link 8's
witness refuses too, and is held here as the binding it now has; so is RT4-INT-5's at 154aa408ac (the draw
under the deployment's pinned challenger key). RT4-INT-6 stays open.
"""
from __future__ import annotations

import hashlib
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
from verity.core.protocols.pouw.circuit import pc8
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import service
from verity.core.protocols.sampled_proofs.one_stage import partition as PT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402
import of_record as OR  # noqa: E402

K, M, N = 128, 3, 4
EGRESS, INGRESS = GridParams(buckets=6, allocation=3, frame_bytes=512), GridParams(buckets=6, allocation=2, frame_bytes=512)
LINK = "requests"
#: a PoUW window is two certifier windows; the proxy serves four, so PoUW windows 0 and 1
POUW_WINDOW, WINDOWS = 2, 4
SCHEME = C.PearlC(k_min=K, forming="v1", device=DEVICES["sm120"], hashing="h3")
MATMUL = Matmul(5, Shape(16, K, N), 2)
RUN = "r-rt4"
#: the network certifier's LMS key and the challenger's, which the certifier registers, and the certifier run's id
CERTIFIER, CHALLENGER = (lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(s).digest())
                         for s in (b"rt4 certifier", b"rt4 challenger"))
RUN_ID = bytes(range(32))


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


def _run(run_id: bytes = RUN_ID) -> dict[str, str]:
    return {"key_id": CERTIFIER.I.hex(), "link": LINK, "run_id": run_id.hex()}


def _sid(stream: str = "window", run_id: bytes = RUN_ID) -> bytes:
    return SC.stream_id({"deployment": f"served-zk/{RUN}"}, stream, _run(run_id))


def _signers(tmp_path: Path) -> list[LmsSigner]:
    out = []
    for name, k in (("certifier", CERTIFIER), ("challenger", CHALLENGER)):
        (tmp_path / name).mkdir(parents=True)
        out.append(LmsSigner.create(k, tmp_path / name / "signer-state.json"))
    return out


def _certifier(sid: bytes, key, sign, record: bytes, tmp_path: Path, reveals) -> tuple[RemoteCertifier, dict[int, bytes]]:
    """The certifier's run `RUN_ID` on `LINK` with PoUW windows 0 and 1, the prover's record taken for both before their
    salts are drawn, both published, served on localhost (`reveals`): the challenger's client of it, and the salts it
    drew."""
    drawn: dict[int, bytes] = {}
    signers = _signers(tmp_path)

    def source(p: int, n: int) -> bytes:
        drawn[p] = hashlib.shake_256(b"certifier salt %d" % p).digest(n)
        return drawn[p]

    c = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SyncSet((0, 1), (1, 2)), [ClockSync(0, 1)] * WINDOWS, origin=0.0,
              clock=lambda: 0.0, ingress_link=LINK, salt=source, registered=PS.registered, prover_key=key,
              prover_context=sid, pouw_window=POUW_WINDOW, challenger_key=signers[1].public_key(), signer=signers[0],
              run_nonce=lambda n: RUN_ID)
    for p in range(WINDOWS // POUW_WINDOW):
        c.commit_weights(p, record, sign(active.weights_message(LINK, p, record, sid)))
    for _ in range(WINDOWS * INGRESS.buckets):
        c.bucketer.tick()
    return RemoteCertifier(*reveals(c), signers[0].public_key(), signers[1], LINK, RUN_ID), drawn


def _challenger(params, c, prover, path: Path, ledger: PS.Registrations) -> SC.Challenger:
    k = lms.keygen(lms.Params(64, 4, 2), hashlib.sha512(str(path).encode()).digest())
    live = Wd.live_challenger(0, 1, random.Random(str(path)).randbytes)
    Ch.write_record(path, live.sign_record("auditor-host", lms.public_key(k), lambda m: lms.sign_at(k, 0, m).encode()))
    deployment = SC.Deployment(c.certifier_key, c.run_id, Ch.ChallengerKey(4, 2, lms.public_key(k)))
    return SC.Challenger(params, live, clock=lambda: 0.0, record=OR.Record(lean=False), certifier=c, prover=prover,
                         coin_record=path, deployment=deployment, registrations=ledger)


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
    return {"params": params, "raw": raw, "rec": SC.register(params, public), "advice": advice, "priv": priv}


def test_the_certified_windows_drawn_unit_reads_no_salt_and_is_refused_unsalted():
    """The served window's population is one template, `Pc8TileSide`, whose ports carry no public input region, so no
    seed, noise line or salt is an input of any drawn unit's proof; `salt.unit_reason` refuses it, and the opening
    refuses its units in a certified window (`core_check._salt_reason`), so nothing certified is credited."""
    d = SC.template(K)
    assert d.id.startswith("Pc8TileSide")
    assert pc8.SIDE_PORTS["regions"] == {} and pc8.PORTS["row_a"]["regions"] and pc8.PORTS["row_b"]["regions"]
    prog, q, part = SC.population(K, 26)
    assert q == PT.template_instance_query(PT.template_id(d))
    assert PS.unit_reason(d) is not None and d.id in PS.unit_reason(d)
    assert CC.TILES[d.id].salted is None


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


def test_a_draw_record_the_prover_signs_over_its_own_coins_is_refused_under_the_pinned_key():
    """RT4-INT-5, closed at #2847 @ 154aa408ac. The draw's check of record (`core_check.draw`, Lean `drawCheck`) takes
    the challenger key the window's deployment pins (`served_commit.Deployment`) and verifies the record under it, not
    under ``coins["key"]``: a record the prover signed under its own key over coins it holds, ground so its draw misses
    the unit it can't prove, is refused ``coin-key``, and so is a live draw with no pin. The pin is an input beside the
    coins (`of_record.py_draw` reads ``x["key"]``), so a replay that took it from the record would pass the prover's own
    record: the pin is the deployment's, which its caller gives (NET-50's TODO)."""
    stream, n, k, bad = hashlib.sha512(b"stream").hexdigest(), 26, 3, 0
    for i in itertools.count():
        reg = hashlib.sha512(b"candidate %d" % i).hexdigest()
        receipt = {"format": "verity/registration-receipt/v1", "received": str(i), "registration": reg,
                   "verifier": "served-zk/challenger/window"}
        d = _prover_record(b"the prover's challenger key", "the prover's coins", reg, stream, receipt, n, k)
        if bad not in d["units"]:
            break
    args = [d[x] for x in ("coins", "recorded", "opening", "stream", "registration", "receipt", "k", "population",
                           "units")]
    auditor = lms.keygen(lms.Params(64, 4, 2), hashlib.sha512(b"the auditor's challenger key").digest())
    pinned = Ch.ChallengerKey(4, 2, lms.public_key(auditor)).to_json()
    assert pinned != d["coins"]["key"]
    assert CC.draw(*args, pinned) == (CC.COIN_KEY, CC.OTHER_KEY)
    code, why = CC.draw(*args)
    assert code == CC.COIN_KEY and why.startswith(CC.NO_PINNED_KEY)
    assert OR.py_draw({**d, "key": pinned}) == {"refused": [CC.COIN_KEY, CC.OTHER_KEY]}
    assert OR.py_draw({**d, "key": d["coins"]["key"]}) is None


def test_a_registered_window_is_admitted_before_its_draw_and_its_failure_closes_nothing(tmp_path, monkeypatch, reveals):
    """RT4-INT-6. `Registrations` admits a certified window when the challenger receives it, before its draw coin is open
    and before any proof. Window 0 is received, drawn and never proved (a visible failure with no outcome), and window 1
    of the same stream and prover is received and drawn after it: nothing in the served path holds a period (#2722's
    TODO), so no refusal closes its stream's credit."""
    monkeypatch.setattr(PS, "unit_reason", lambda unit: None)
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), _sid()
    c, drawn = _certifier(sid, key, sign, PS.pouw_record(weights_root(_pass(RUN)[1]), sid, prover), tmp_path, reveals)
    ledger, chs = PS.Registrations(), []
    for p in (0, 1):
        w = _window(p, drawn[p], sid, prover, PRECOMPUTED)
        ch = _challenger(w["params"], c, prover, tmp_path / f"w{p}.json", ledger)
        ch.register(f"w{p}", w["rec"], manifest=w["raw"])
        assert (p, sid, prover) in ledger and "draw" not in ch.openings
        assert ch.select(f"w{p}").units
        chs.append(ch)
    assert chs[0].sessions == {} and len(ledger) == 2


def test_a_certified_challenger_takes_the_certifiers_client_and_its_stream_names_the_run(tmp_path, monkeypatch, reveals):
    """RT4-INT-2's link 8, now held. The certified challenger is built on NET-50's `RemoteCertifier` and names its stream
    by the client's `certifier_run`, (key id, run id, link), the same as its proxy's. Two runs on one link are two
    streams: a window whose manifest is salted under another run's stream is refused at registration, since the salt
    the challenger computes for its own run is not the manifest's (window 1, whose reveal is still fresh). The run id and key are still the caller's to give
    (NET-50's TODO: the auditor's deployment record)."""
    monkeypatch.setattr(PS, "unit_reason", lambda unit: None)
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), _sid()
    c, drawn = _certifier(sid, key, sign, PS.pouw_record(weights_root(_pass(RUN)[1]), sid, prover), tmp_path, reveals)
    assert SC.certifier_run(c) == _run()
    proxies = []
    for b in (1, 2):
        cert, chal = _signers(tmp_path / f"p{b}")
        proxies.append(Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SyncSet((0, 1), (1, 2)), [ClockSync(0, 1)] * WINDOWS,
                             origin=0.0, clock=lambda: 0.0, ingress_link=LINK, challenger_key=chal.public_key(), signer=cert,
                             run_nonce=lambda n, b=b: bytes([b]) * n))
    runs = [SC.certifier_run(p) for p in proxies]
    assert runs[0] != runs[1] and {r["run_id"] for r in runs} == {(bytes([b]) * RUN_ID_BYTES).hex() for b in (1, 2)}
    params = {"deployment": f"served-zk/{RUN}"}
    assert SC.stream_id(params, "window", runs[0]) != SC.stream_id(params, "window", runs[1])

    w = _window(0, drawn[0], sid, prover, PRECOMPUTED)
    ch = _challenger(w["params"], c, prover, tmp_path / "w.json", PS.Registrations())
    ch.register("w", w["rec"], manifest=w["raw"])
    assert ch.passes == {"w": "certifier"}

    other = _sid(run_id=bytes([7]) * RUN_ID_BYTES)
    assert other != sid
    w2 = _window(1, drawn[1], other, prover, PRECOMPUTED)
    ch2 = _challenger(w2["params"], c, prover, tmp_path / "w2.json", PS.Registrations())
    with pytest.raises(service.Refused) as refused:
        ch2.register("w2", w2["rec"], manifest=w2["raw"])
    detail = refused.value.verdict.detail
    assert detail == {"coin": "the pass's salt is not the certifier's for its window, stream and prover"}, detail
    assert "w2" not in ch2.passes
