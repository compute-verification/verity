"""RT4-INT-10 on the certified served window (#2847, `served_commit.Challenger(certifier=…)`): a drawn unit's proof is
held to the window's certified salt at its opening of record (`core_check.opening`'s ``salt``, Lean `PouwCheck.opening`'s).

RT4-INT's witness lifted `salt.unit_reason` and passed rows formed before any certifier salt through every certified
check, the opening included. The opening now opens a certified window's unit only when its template is salted
(`core_check.Salted`) and only under the 64-byte salt the verifier holds (`core_check.certified_salt`, the salt
`pass_epoch` checked), and requires the salted unit's public row to be `core_check.public_bc` of that salt, the unit's
position and its entry's side roots. So the precomputed `Pc8TileSide` rows are refused at the opening, whether or not the
verifier passes the salt, and a salted unit's proof formed under a salt the prover chose is refused under the certified
one.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
from verity.ml.crypto import lms
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.active import Proxy
from verity.core.protocols.certifier.certificate import LmsKey, LmsSigner
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES
from verity.core.protocols.sampled_proofs import challenger as Ch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402
import of_record as OR  # noqa: E402

_CASES = Path(CC.__file__).resolve().parent / "lean/check/tests"
for _name in ("pouw_rounds_cases", "pouw_pass_cases", "pouw_check_cases", "pouw_opening_cases"):
    _spec = importlib.util.spec_from_file_location(_name, _CASES / f"{_name}.py")
    sys.modules[_name] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules[_name])
openings, canon = sys.modules["pouw_opening_cases"], sys.modules["pouw_check_cases"]._canon

K, M, N = 128, 3, 4
EGRESS, INGRESS = GridParams(buckets=6, allocation=3, frame_bytes=512), GridParams(buckets=6, allocation=2, frame_bytes=512)
LINK = "requests"
#: a PoUW window is two certifier windows; the proxy serves four, so PoUW windows 0 and 1
POUW_WINDOW, WINDOWS = 2, 4
SCHEME = C.PearlC(k_min=K, forming="v1", device=DEVICES["sm120"], hashing="h3")
MATMUL = Matmul(5, Shape(16, K, N), 2)
RUN = "r-salt-bind"
#: the runner's record of a request served at the verifier's fixed length under ignore_eos (ASK-39)
SERVED = {"ignore_eos": True, "max_tokens": [SC.MAX_TOKENS], "served": [SC.MAX_TOKENS]}
#: the network certifier's LMS key and the challenger's, which the certifier registers, and the certifier run's id
CERTIFIER, CHALLENGER = (lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(s).digest())
                         for s in (b"certifier", b"challenger"))
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
    return SC.stream_id({"deployment": f"served-zk/{RUN}"}, stream,
                        {"key_id": CERTIFIER.I.hex(), "link": LINK, "run_id": RUN_ID.hex()})


def _certifier(sid: bytes, key, sign, record: bytes, tmp_path: Path,
               reveals) -> tuple[active.RemoteCertifier, dict[int, bytes]]:
    """The certifier's run `RUN_ID` on `LINK` with PoUW windows 0 and 1, the prover's record taken for both before their
    salts are drawn, both published, served on localhost (`reveals`): the challenger's client of it, and the salts it
    drew."""
    drawn: dict[int, bytes] = {}
    signers = []
    for name, k in (("certifier", CERTIFIER), ("challenger", CHALLENGER)):
        (tmp_path / name).mkdir()
        signers.append(LmsSigner.create(k, tmp_path / name / "signer-state.json"))

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
    return active.RemoteCertifier(*reveals(c), signers[0].public_key(), signers[1], LINK, RUN_ID), drawn


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
              "range": [1, 1 << 40], "entries": 4, "w_rows": N,
              "length": {"max_tokens": [SC.MAX_TOKENS], "stop": Wd.STOP_FIXED}}
    root_w, _ = SC.commit_weights(params, [np.arange(K, dtype="<u2").tobytes() for _ in range(N)])
    public, advice, priv = SC.commit_window(params, _calls(words), root_w, SC.root_h3_of(man),
                                            length=SC.served_length(SERVED))
    return {"params": params, "raw": raw, "rec": SC.register(params, public), "advice": advice, "priv": priv}


def _pub(priv, advice, t: int) -> bytes:
    """The drawn unit's proof's public file as `verity_flock.circuit.write` lays it out: the header over the circuit the
    verifier pins, then the unit's A row's and B row's b ‖ c."""
    from verity.core.protocols.pouw.circuit import leaves as L
    e, i, j = Wd.tile_at(SC.entries_of(advice), t)
    schemas = {"a0": L.ROW_SCHEMA["fp8-pi", SC.SIDES["a"]], "b0": L.ROW_SCHEMA["fp8-pi", SC.SIDES["b"]],
               "o0": L.ROW_SCHEMA["tile", SC.SIDES["a"]]}
    head = {"circuit_sha512": CC.TILES[SC.template(K).id].circuit, "instances": 1, "frame_v3": {"schemas": schemas}}
    return json.dumps(head).encode() + b"\n" + priv[e]["a"]["bcs"][i] + priv[e]["b"]["bcs"][j] + bytes(128)


def test_rows_formed_before_any_salt_are_refused_at_a_certified_windows_opening(tmp_path, monkeypatch, reveals):
    """RT4-INT-10, the served path. RT4-INT's witness: lift `unit_reason` and the prover's rows, formed once under a salt
    that is neither window's and committed in PoUW windows 0 and 1, pass the certifier's record and reveal (through its
    client, `RemoteCertifier`), `pass_epoch` of record, the registration ledger and the live draw. They now stop at the
    opening of record: the window is certified and `Pc8TileSide` reads no salt, so every drawn unit is refused, through
    `Challenger.check` (under the certified salt its pass was checked against, `certified_salt`) and through
    `core_check.opening` under that salt and under none."""
    monkeypatch.setattr(PS, "unit_reason", lambda unit: None)
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), _sid()
    committed = PS.pouw_record(weights_root(_pass(RUN)[1]), sid, prover)
    c, drawn = _certifier(sid, key, sign, committed, tmp_path, reveals)
    ledger = PS.Registrations()
    refused = "the window is certified, and its template's unit reads no salt"
    for p in (0, 1):
        w = _window(p, drawn[p], sid, prover, PRECOMPUTED)
        ch = _challenger(w["params"], c, prover, tmp_path / f"w{p}.json", ledger)
        ch.register(f"w{p}", w["rec"], manifest=w["raw"])
        assert ch.passes == {f"w{p}": "certifier"} and (p, sid, prover) in ledger
        d = ch.select(f"w{p}")
        assert d.units
        salt = CC.certified_salt(w["raw"], w["params"]["context"], drawn[p], committed, sid, prover)
        assert salt == PS.pouw_salt(drawn[p], p, sid, prover) == ch.salts[f"w{p}"]
        params, rec = SC.core_params(w["params"]), CC.commitment(bytes.fromhex(OR.hexed(w["rec"])))
        for u in d.units:
            pub, op = _pub(w["priv"], w["advice"], u), SC.opening(w["advice"], w["priv"], w["params"], u)
            why = ch.check(f"w{p}", u, pub, op)
            assert why is not None and why.startswith(refused), why
            for held in (salt, None):
                assert CC.opening(CC.params(params), rec, list(d.units), u, pub, op, held).startswith(refused)


def test_a_salted_unit_formed_under_its_own_salt_is_refused_under_the_certified_one():
    """RT4-INT-10, the salted unit. A certified window's salt is `pouw_salt` of the certifier's reveal; the prover can
    pick its own salt s* before any reveal and form the H1 unit's public row under it, its rows the same. Under the
    certified salt the verifier holds, the opening of record (`of_record.py_opening`, the difftest's input) refuses the
    s* proof at its public row and opens the proof formed under the certified salt (the honest twin: every rule of the
    opening met), so the one statement credited is the certified salt's."""
    key, _ = _prover(b"the prover")
    sid, prover = _sid(), PS.prover_id(key.public)
    reveal = hashlib.shake_256(b"certifier salt 0").digest(PS.SALT_BYTES)
    w = _window(0, reveal, sid, prover, PRECOMPUTED)
    ctx = w["params"]["context"]
    certified = CC.certified_salt(w["raw"], ctx, reveal, PS.pouw_record(weights_root(_pass(RUN)[1]), sid, prover), sid,
                                  prover)
    own = hashlib.sha512(b"the prover's own salt, chosen before any certifier window").digest()
    assert certified != own
    for seed in range(3):
        held = {}
        for s in (certified, own):
            case = openings.honest(random.Random(f"salt-bind/{seed}"), salted=True, context=ctx, salt=s)
            held[s] = {"params": case["params"], "commitment": canon(case["record"]).encode().hex(), "units": case["units"],
                       "unit": case["unit"], "pub": case["pub"].hex(), "opening": case["opening"], "salt": certified.hex()}
        assert held[certified]["opening"] == held[own]["opening"] and held[certified]["pub"] != held[own]["pub"]
        assert OR.py_opening(held[certified]) is None
        why = OR.py_opening(held[own])
        assert why is not None and why.startswith("the proof's public row is not the window's certified salt"), why
