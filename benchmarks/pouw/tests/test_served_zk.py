"""served_zk's window through the proof service (`served_zk/served_commit.py` over P1, `sampled_proofs.service`): a
window of served calls' side rows committed as C-Flock commits a unit's rows, its tiles padded with filler tiles to a
grid level, registered with exactly PoUW's public items, received by a `Stream`, drawn once, each drawn tile's proof
opened against the window root, delivered once, and the final outcome publishing exactly the public items."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import itertools
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from verity.ml.commitments.merkle import tree_levels
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
from verity.core.protocols.pouw.circuit import leaves as L
from verity.core.protocols.pouw.coins import Challenger as Coins
from verity.core.protocols.pouw.coins import CoinOpening, coin_reason
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import registration as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402
import of_record as OR  # noqa: E402

#: the challenger's checks are of record in Lean (`of_record`), which `lake` builds
LEAN = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
PD = SC._pd()

K, M, N = 128, 3, 4


def _rows(seed: int, n: int) -> list[bytes]:
    x = np.random.default_rng(seed).standard_normal((n, K)).astype("<f4")
    return [r.tobytes() for r in x]


def _coins(seed: int = 7) -> Coins:
    """The epoch's coin challenger on a seeded entropy (the challenger's own generator stands in)."""
    return Coins(0, random.Random(seed).randbytes)


def _pass(run: str) -> tuple[dict, dict[int, bytes], Workload]:
    """A served pass of `run` before its salt: its manifest's calls, weights' roots and tiles' commitment, the roots and
    the workload."""
    roots = {2: hashlib.sha512(run.encode()).digest()}
    man = {"run": run, "matmuls": [{"index": 5, "m": 16, "k": K, "n": N, "weight": 2}], "weights": {"2": {"root": roots[2].hex()}},
           "commitment": {"root": hashlib.sha512(f"{run}/tiles".encode()).hexdigest(), "count": 1}}
    return man, roots, Workload((Matmul(5, Shape(16, K, N), 2),))


def _manifest(run: str, coins: Coins | None) -> dict:
    """A served pass's manifest as the challenger reads it: its calls, its weights' roots and its salt, the salt coin of
    `coins` opened after the weights' root, or (None) a beacon's, as a pass recorded before epoch coins carries."""
    man, roots, work = _pass(run)
    if coins is None:
        beacon = bytes(32)
        return {**man, "schema": PD.MANIFEST_SCHEMA_RECORDED, "beacon": beacon.hex(),
                "salt": Epoch.unverified(work, roots, beacon).salt.hex()}
    opening = coins.open("salt", weights_root(roots))
    return {**man, "schema": PD.MANIFEST_SCHEMA, "coins": coins.commitment.to_json(), "salt_opening": opening.to_json(),
            "salt": Epoch.start(work, roots, coins.commitment, opening).salt.hex()}


def _raw(man: dict) -> bytes:
    return json.dumps(man, sort_keys=True).encode()


def _params(run: str, raw: bytes) -> dict:
    return {"context": f"served-zk/{run}/{hashlib.sha512(raw).hexdigest()}", "deployment": f"served-zk/{run}",
            "k": K, "eps": "1/2", "delta": "1/10", "grid_bits": Wd.GRID_BITS, "range": [1, 1 << 40], "entries": 4, "w_rows": N,
            "length": {"max_tokens": [SC.MAX_TOKENS], "stop": Wd.STOP_FIXED},
            "registration_deadline": SC.REGISTRATION_DEADLINE}


#: the runner's record of a request served at the verifier's fixed length under ignore_eos
SERVED = {"ignore_eos": True, "max_tokens": [SC.MAX_TOKENS], "served": [SC.MAX_TOKENS]}


#: each window's pass by its parameters' context: (the epoch's coins, the manifest's bytes, the manifest)
PASSES: dict[str, tuple[Coins, bytes, dict]] = {}


@pytest.fixture(scope="module")
def served():
    scheme = C.PearlC(k_min=K, forming="v1", device=DEVICES["sm120"], hashing="h3")
    tag = lambda s: hashlib.sha256(s).digest()  # noqa: E731
    ctx = Context(tag(b"salt"), Matmul(5, Shape(16, K, N), 2), tag(b"root_a"), tag(b"root_b"))
    noise = scheme.noise(ctx)
    a = _rows(1, M) + [bytes(4 * K)]                 # a zero row, which forming refuses
    words = {"a": SC.form(scheme, ctx, noise, K, "a", a, list(range(M + 1))),
             "b": SC.form(scheme, ctx, noise, K, "b", _rows(2, N), list(range(N)))}
    return scheme, ctx, noise, words


def _calls(words) -> list[dict]:
    """Two calls of the same rows: 2·M·N = 24 real tiles, grid level 23, padded to 26 with a filler entry of 1 x 2 zero rows."""
    return [{"record": {"matmul": m, "shape": [16, K, N], "tile0": m, "weight": 2, "real_rows": M + 1}, "words": words}
            for m in (5, 9)]


def _weights() -> list[bytes]:
    return [np.arange(K, dtype="<u2").tobytes() for _ in range(N)]


def _window(words, run="r-request", coins=None, recorded=False, served=SERVED):
    """A window of two calls from a pass of `run`, salted by `coins`' salt coin (a seeded one by default) or, `recorded`,
    by a beacon, its length registered from the runner's record `served` (none: no length item)."""
    coins = None if recorded else coins or _coins()
    man = _manifest(run, coins)
    params = _params(run, _raw(man))
    PASSES[params["context"]] = (coins, _raw(man), man)
    root_w, _ = SC.commit_weights(params, _weights())
    length = SC.served_length(served) if served is not None else None
    public, advice, priv = SC.commit_window(params, _calls(words), root_w, SC.root_h3_of(man), length=length)
    return params, public, advice, priv


def _register(params, public, **extra):
    return SC.register(params, public, extra)


def _codes(e: pytest.ExceptionInfo) -> set[str]:
    return set(e.value.verdict.codes)


def _pub(bca: bytes, bcb: bytes) -> bytes:
    """A tile proof's public file as `verity_flock.circuit.write` lays it out: its header (over the circuit the verifier
    pins for the window's template), then instance 0's b || c in port order (the A row, the B row, the tile digest's
    row)."""
    schemas = {"a0": L.ROW_SCHEMA["fp8-pi", SC.SIDES["a"]], "b0": L.ROW_SCHEMA["fp8-pi", SC.SIDES["b"]],
               "o0": L.ROW_SCHEMA["tile", SC.SIDES["a"]]}
    head = {"circuit_sha512": CC.TILES[SC.template(K).id].circuit, "instances": 1, "frame_v3": {"schemas": schemas}}
    return json.dumps(head).encode() + b"\n" + bca + bcb + bytes(128)


def _unit_pub(advice, priv, t: int) -> bytes:
    e, i, j = Wd.tile_at(SC.entries_of(advice), t)
    return _pub(priv[e]["a"]["bcs"][i], priv[e]["b"]["bcs"][j])


def test_rows_are_the_schemes_operands(served):
    """Each formed row is the fp8-p row of the operands `PearlC.encode` forms for a tile holding it, and the refused row is
    none of the call's kept rows."""
    scheme, ctx, noise, words = served
    assert words["a"][M] is None and all(w is not None for w in words["a"][:M] + words["b"])
    from verity.core.protocols.pouw.protocol import Tile
    ops = scheme.encode(ctx, noise, _rows(1, M)[1:2], _rows(2, N)[3:4], Tile((1,), (3,)))
    assert words["a"][1] == L.words_fp8_p(ops.a.codes[0], ops.pa[0])
    assert words["b"][3] == L.words_fp8_p(ops.b.codes[0], ops.pb[0])


def test_the_window_is_padded_to_a_grid_level_with_filler(served):
    """The calls' kept tiles, then a filler entry of zero rows up to the padded count, then empty entries to the bound, each
    an entry row; the window statement holds on the rows, each side tree holds its kept rows only under the window's side
    domain (from the window root's one-stage served domain), the record keeps the rows' places in the call, and replaying
    the salts gives the same commitment."""
    *_, words = served
    params, public, advice, priv = _window(words)
    work = Wd.bucket(2 * M * N)
    assert (advice["real"], public["level"], work.credited, work.n) == (2 * M * N, work.level, 24, 26)
    entries = SC.entries_of(advice)
    assert [e.kind for e in entries] == ["call", "call", "filler", "none"]
    call, filler = entries[0], entries[2]
    assert (call.n_a, call.n_b, call.lo, call.hi) == (M, N, 0, M * N) and (filler.lo, filler.hi) == (24, 26)
    assert (filler.n_a, filler.n_b) == (1, 2) and filler.record == filler.root_h3 == bytes(64) and entries[3] == Wd.Entry("none")
    root_h3 = SC.root_h3_of(PASSES[params["context"]][2])
    assert entries[0].root_h3 == entries[1].root_h3 == root_h3 and advice["root_h3"] == root_h3.hex()
    assert advice["records"][0]["keep"] == {"a": list(range(M)), "b": list(range(N))} and advice["records"][2:] == [None, None]
    assert call.record == Wd.record_digest(advice["records"][0]) and len(priv[0]["a"]["bcs"]) == M
    assert not priv[2]["a"]["words"].any() and not priv[2]["b"]["words"].any()
    assert Wd.window_problem([e.row() for e in entries], work, params["entries"], root_h3) is None
    _, _, part = SC.population(K, work.n)
    (wroot,) = [r for r in public["roots"] if r["port"] == SC.PORT]
    assert wroot["domain"] == R.window_domain(part["program"], SC.PT.digest(part), SC.PORT, SC.ENTRY_SCHEMA, params["entries"],
                                              params["context"])
    for e in (0, 2):
        for side in "ab":
            dom, schema = SC.side_domain(params, public["level"], side)
            assert dom == Wd.side_domain(bytes.fromhex(wroot["domain"]), side) and dom.count == 1 << Wd.DEPTHS[side]
            leaves = SC.side_leaves(dom, schema, priv[e][side]["bcs"])
            root = tree_levels(dom, leaves + [dom.pad(r) for r in range(len(leaves), dom.count)])[-1][0]
            assert root == (entries[e].root_a if side == "a" else entries[e].root_b)
    assert priv[0]["b"]["bcs"][2] == L.commit("fp8-pi", SC.SIDES["b"], Wd.side_row(words["b"][2], 2), priv[0]["b"]["salts"][2])
    assert [Wd.side_index(r) for r in priv[0]["a"]["words"]] == advice["records"][0]["keep"]["a"]
    assert (call.tile0, call.cols, call.m, entries[1].tile0) == (5, 1, 16, 9) and filler.m == filler.tile0 == 0
    salts = {e: {s: priv[e][s]["salts"] for s in "ab"} for e in range(3)}
    again, advice2, _ = SC.commit_window(params, _calls(words), public["items"]["root_W"], root_h3,
                                         {**salts, "entries": priv["entries"]}, length=public["items"]["length"])
    assert again == public and advice2["entries"] == advice["entries"]
    for bad in ({}, {"commitment": {"root": "ab" * 32}}):
        with pytest.raises(SystemExit, match="not a SHA-512"):
            SC.root_h3_of(bad)


def test_register_makes_exactly_the_public_items_public(served):
    """The record's window holds exactly the registered public items (work, check, sampling, length, root_W) and nothing of the
    calls: no shape, no call index, no row count.  One more item, a root_W that is no registered root, or a commitment
    whose level the range refuses, is refused."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    assert rec["window"] == {"kind": "served", "context": params["context"], "public": public["items"]}
    assert set(rec["window"]["public"]) == set(Wd.REGISTERED) and rec["population"] == 26 and rec["scheme"] == S.SCHEME
    assert rec["law"] == {"law": "subset", "k": rec["window"]["public"]["sampling"]["law"]["k"]}
    assert [r["port"] for r in rec["roots"]] == [SC.PORT, "W"]
    assert S.check_public(SC.spec_of(params, public["level"]), rec).ok and SC.public_problem(params, rec).ok
    text = json.dumps(rec)
    for leak in ('"matmul"', '"shape"', '"real_rows"', '"rows"', '"keep"'):
        assert leak not in text
    with pytest.raises(S.Refused) as e:
        _register(params, public, shapes=[[16, K, N]])
    assert _codes(e) == {"public-items"}
    with pytest.raises(S.Refused) as e:
        _register(params, public, root_W="00" * 64)
    assert _codes(e) == {"public-value"}
    undeclared = SC.spec_of(params, public["level"])
    undeclared = S.Spec(undeclared.values, undeclared.selection, undeclared.check, undeclared.public[:2], undeclared.context)
    assert set(S.check_public(undeclared, rec).detail["undeclared"]) == {f"leaves:{SC.PORT}", f"domain:{SC.PORT}", "leaves:W",
                                                                          "domain:W"}
    with pytest.raises(ValueError, match="outside"):
        SC.work_at({**params, "range": [100, 1 << 40]}, public["level"])


#: runner records the verifier refuses, each with its `window.length_problem` code: served until EOS, served short of the
#: verifier's max_tokens, and served at a max_tokens of the runner's own
UNFIXED = {"eos-keyed-row": {**SERVED, "ignore_eos": False},
           "length-mismatch": {**SERVED, "served": [SC.MAX_TOKENS - 1]},
           "max-tokens": {**SERVED, "max_tokens": [2 * SC.MAX_TOKENS], "served": [2 * SC.MAX_TOKENS]}}


def test_rows_of_record_serve_a_public_fixed_length(served):
    """The window registers its rows' fixed length, the verifier's max_tokens per slot served under ignore_eos, so its
    useful work is public.  A fixed-length window registers; a row served until EOS, a served length other than
    max_tokens, or a max_tokens other than the verifier's is refused as ``window-form``; and a window with no length
    item (registered before the rule) is refused as not of record.  The committer's max_tokens is the served request's
    (`e2e.SERVED_TOKENS`), and the runner's record of it has the name the committer reads."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
    import e2e
    assert (SC.MAX_TOKENS, SC.SERVED_RECORD) == (e2e.SERVED_TOKENS, e2e.SERVED_RECORD)
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    assert rec["window"]["public"]["length"] == Wd.length_item([SC.MAX_TOKENS]) == SC.served_length(SERVED)
    for code, record in UNFIXED.items():
        p, pub, _, _ = _window(words, served=record)
        with pytest.raises(S.Refused) as e:
            _register(p, pub)
        assert _codes(e) == {"window-form"} and e.value.verdict.detail["window-form"].startswith(code), code
    p, pub, _, _ = _window(words, served=None)
    assert set(pub["items"]) == set(Wd.LEGACY_REGISTERED)
    with pytest.raises(S.Refused) as e:
        _register(p, pub)
    assert _codes(e) == {"window-form"} and e.value.verdict.detail["window-form"].startswith("no-fixed-length")


def _salt_after(params) -> bytes:
    """What the epoch's salt coin is opened after: the weights' root of the window's pass."""
    man = PASSES[params["context"]][2]
    return weights_root({int(w): bytes.fromhex(v["root"]) for w, v in man["weights"].items()})


def _challenger(params, units=None, coins=None, sampler=None, record=None) -> SC.Challenger:
    """The challenger holding the epoch's coins (the ones that salted the window's pass, by default), its salt coin opened
    after the pass's weights' root as the job opens it; `units` a stand-in for Lean's draw."""
    sampler = sampler or (SC.python_sampler if units is None else (lambda law, n, src: list(units)[:law["k"]]))
    ch = SC.Challenger(params, coins or _coins(), sampler, clock=lambda: 0.0, record=record)
    ch.open("salt", _salt_after(params))
    return ch


def _held(ch, name, rec, **kw):
    """`ch` receives window `rec` from its pass (the manifest its parameters name)."""
    return ch.register(name, rec, manifest=PASSES[ch.params["context"]][1], **kw)


@LEAN
def test_the_challenger_on_the_serving_controllers_state_holds_its_coins(served, tmp_path, monkeypatch, capsys):
    """e2e.py's coins under ``POUW_COIN_STATE`` and `served_commit.py challenger --state` on the same file publish one
    commitment and open the one salt coin; the state file is its owner's alone, and a missing one is refused, never
    replaced by fresh coins."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
    import e2e
    *_, words = served
    params, *_ = _window(words)
    state, after = tmp_path / "state" / "r1", bytes(range(64))
    monkeypatch.setenv("POUW_COIN_STATE", str(state))
    controller = e2e.epoch_coins(type("A", (), {"retain": None})(), 0)
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
    assert e2e.epoch_coins(type("A", (), {"retain": None})(), 0).commitment == controller.commitment
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"op": "salt", "after": after.hex(), "out": f"{tmp_path}/salt.json"}) + "\n"))
    (tmp_path / "params.json").write_text(json.dumps(params))
    capsys.readouterr()
    assert SC.main(["challenger", f"{tmp_path}/params.json", f"{tmp_path}/log", "--state", str(state)]) == 0
    assert capsys.readouterr().out.startswith("OPENED")
    assert json.loads((tmp_path / "log" / "coins.json").read_text()) == controller.commitment.to_json()
    assert json.loads((tmp_path / "salt.json").read_text()) == controller.open("salt", after).to_json()
    with pytest.raises(SystemExit, match="no challenger state"):
        SC.main(["challenger", f"{tmp_path}/params.json", f"{tmp_path}/log2", "--state", str(tmp_path / "none")])


def test_a_served_challenger_refuses_coins_no_claim_instance_names(tmp_path, monkeypatch):
    """CMP-52: the serving controller's coins (e2e.py's `Coins`) and `served_commit.py challenger`'s are named by a claim
    instance, the Lean verifier of record's (`pouw.verifier.CLAIM`) or its state file's, which an earlier start drew from
    it; a source no claim instance names, `os.urandom` among them, is refused before any coin is drawn or published."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
    import e2e
    for source in (os.urandom, random.Random(0).randbytes):
        with pytest.raises(Ch.CoinError, match="coins-without-claim"):
            e2e.Coins(0, source)
    state = tmp_path / "state"
    state.write_bytes(random.Random(1).randbytes(512))
    assert e2e.Coins(0, e2e.state_entropy(str(state))).challenger.claim == SC.PV.CLAIM == "uniform/io-getrandombytes"
    monkeypatch.setattr(SC.PV, "coins", random.Random(2).randbytes)
    (tmp_path / "params.json").write_text("{}")
    with pytest.raises(SystemExit, match="coins-without-claim"):
        SC.main(["challenger", f"{tmp_path}/params.json", f"{tmp_path}/log"])
    assert not (tmp_path / "log" / "coins.json").exists()


@LEAN
def test_the_challenger_holds_every_item_to_its_own(served):
    """The challenger's stream registers the honest record and its receipt names it.  An extra public key, a raised level
    (more credit), a K′ other than its own, another request's record (its own context), a field one-stage doesn't define,
    a length that isn't its fixed one or no length item is refused, and so is a second registration while the first has
    no outcome."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ch = _challenger(params)

    def refused(r) -> set[str]:
        with pytest.raises(S.Refused) as e:
            _held(ch, "x", r)
        return _codes(e)

    extra = json.loads(json.dumps(rec))
    extra["window"]["public"]["shapes"] = [[16, K, N]]
    assert refused(extra) == {"public-items"}
    raised = json.loads(json.dumps(rec))
    raised["window"]["public"]["work"]["level"] += 1
    assert refused(raised) == {"public-value"}
    k = json.loads(json.dumps(rec))
    k["window"]["public"]["sampling"]["law"]["k"] -= 1
    assert refused(k) == {"public-value"}
    oparams, opublic, _, _ = _window(words, run="r-other")
    assert refused(_register(oparams, opublic)) == {"window"}
    assert refused({**rec, "tiles": [2, 3]}) == {"keys"}
    for record in UNFIXED.values():
        unfixed = json.loads(json.dumps(rec))
        unfixed["window"]["public"]["length"] = SC.served_length(record)
        assert refused(unfixed) == {"window-form"}
    legacy = json.loads(json.dumps(rec))
    del legacy["window"]["public"]["length"]
    assert refused(legacy) == {"window-form"}
    assert "x" not in ch.regs
    reg = _held(ch, "honest", rec)
    assert reg.receipt["registration"] == R.digest(rec) == reg.digest and reg.order == 0
    assert refused(rec) == {"open"}


@LEAN
def test_select_draws_once_over_the_draw_coin_opened_after_the_registration(served):
    """The draw is the law's subset:K′ over the padded population, over the key of the epoch's draw coin, which the
    challenger opens after the registration (its digest) and checks against its commitment; Lean's draw on that key's
    stream is `stream_subset`'s.  A second select is refused, and so is a retry after a failed draw: the coin opens once.
    A replay from the recorded opening draws the same units; a tampered recorded coin is refused."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    k = SC.law_of(params, public["level"])["k"]
    ch = _challenger(params)
    reg = _held(ch, "honest", rec)
    assert ch.passes == {"honest": "coin"} and "draw" not in ch.openings
    d = ch.select("honest")
    assert len(d.units) == k and d.draw["population"] == 26 and d.units == sorted(d.units) and all(0 <= u < 26 for u in d.units)
    o = ch.openings["draw"]
    assert o.after == bytes.fromhex(reg.digest) and coin_reason(ch.coins.commitment, o, "draw", o.after) is None
    st = ch.regs["honest"][0]
    src = SC.draw_bytes(SC.draw_key(o.coin, st, reg), d.law, 26)
    assert len(src) == 2 * k * CC.width(26) + 64 and d.units == CC.stream_subset(k, 26, src)
    assert ch.coins_of("honest") == {"commitment": ch.coins.commitment.to_json(), "pass": "coin", "draw": o.to_json()}
    with pytest.raises(S.Refused) as e:
        ch.select("honest")
    assert _codes(e) == {"drawn"}

    salt = ch.openings["salt"]
    again = _challenger(params, coins=SC.Recorded(ch.coins.commitment, [salt, o]))
    _held(again, "honest", rec)
    assert again.select("honest").draw == d.draw
    forged = CoinOpening(o.epoch, o.label, o.after, bytes([o.coin[0] ^ 1]) + o.coin[1:], o.salt, o.path)
    tampered = _challenger(params, coins=SC.Recorded(ch.coins.commitment, [salt, forged]))
    _held(tampered, "honest", rec)
    with pytest.raises(S.Refused) as e:
        tampered.select("honest")
    assert _codes(e) == {"coin"} and "does not open against" in e.value.verdict.detail["coin"]

    params, public, _, _ = _window(words)
    rec = _register(params, public)
    bad = _challenger(params, sampler=lambda law, n, src: list(range(law["k"] + 1)))
    _held(bad, "honest", rec)
    with pytest.raises(S.Refused):
        bad.select("honest")
    with pytest.raises(S.Refused) as e:
        bad.select("honest")
    assert _codes(e) == {"coin"} and "already open" in e.value.verdict.detail["coin"]


@LEAN
def test_a_window_received_past_the_deadline_from_the_salt_coins_release_earns_no_credit(served):
    """Under ``registration_deadline`` (`Pouw.AcceptsBy`'s td, seconds on the challenger's clock), the salt coin's opening
    reads its release, and each stream records the epoch's salt at that release before its first registration.  A window
    received within the deadline registers with both readings.  One received later, or before the salt is open, is
    refused (`late-registration`) and ends the epoch: the stream takes nothing after it."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    coins, _, man = PASSES[params["context"]]
    salt = CoinOpening.from_json(man["salt_opening"])
    timed = {**params, "registration_deadline": 5.0}

    def challenger():
        t = [0.0]
        return SC.Challenger(timed, SC.Recorded(coins.commitment, [salt]), clock=lambda: t[0]), t

    ch, t = challenger()
    t[0] = 10.0
    ch.open("salt", salt.after)
    t[0] = 14.0
    reg = _held(ch, "honest", rec)
    assert (reg.salted_at, reg.received_at) == (10.0, 14.0) and ch.salt_released_at == 10.0
    assert ch.regs["honest"][0].spec.check.registration_deadline == 5.0

    late, t = challenger()
    late.open("salt", salt.after)
    t[0] = 5.5
    with pytest.raises(S.Refused) as e:
        _held(late, "late", rec)
    assert _codes(e) == {"late-registration"} and "late" not in late.regs
    with pytest.raises(S.Refused) as e:
        _held(late, "again", rec)
    assert _codes(e) == {"epoch-ended"}

    unsalted, _ = challenger()
    with pytest.raises(S.Refused) as e:
        _held(unsalted, "early", rec)
    assert _codes(e) == {"late-registration"}


def test_a_window_without_a_registration_deadline_is_refused(served, tmp_path):
    """Parameters without ``registration_deadline`` are refused, so the service accepts no window `EndToEndWindow`
    doesn't cover: the challenger won't hold them, `spec_of` won't state a check under them, and the committer won't
    register under them.  `params` writes the deadline.  Only a replay of a record whose parameters predate it runs
    without one (`predates_deadline`), and never with parameters that carry one."""
    *_, words = served
    params, public, _, _ = _window(words)
    bare = {k: v for k, v in params.items() if k != "registration_deadline"}
    for refuse in (lambda: SC.Challenger(bare, _coins()), lambda: SC.spec_of(bare, public["level"]),
                   lambda: _register(bare, public)):
        with pytest.raises(ValueError, match="no registration_deadline"):
            refuse()
    with pytest.raises(ValueError, match="carry a registration_deadline"):
        SC.Challenger(params, _coins(), predates_deadline=True)
    assert SC.Challenger(bare, _coins(), predates_deadline=True).predates_deadline
    assert SC.spec_of(bare, public["level"], predates_deadline=True).check.registration_deadline is None
    assert SC.spec_of(params, public["level"]).check.registration_deadline == SC.REGISTRATION_DEADLINE

    man = {"schema": PD.MANIFEST_SCHEMA, "run": "r", "matmuls": [{"index": 0, "m": 1, "k": SC.K, "n": SC.K, "weight": 2}]}
    (tmp_path / "pass").mkdir()
    (tmp_path / "pass" / "manifest.json").write_text(json.dumps(man))
    assert SC.main(["params", str(tmp_path / "pass"), str(tmp_path / "p.json")]) == 0
    assert json.loads((tmp_path / "p.json").read_text())["registration_deadline"] == SC.REGISTRATION_DEADLINE
    assert SC.main(["params", str(tmp_path / "pass"), str(tmp_path / "q.json"), "--registration-deadline", "60"]) == 0
    assert json.loads((tmp_path / "q.json").read_text())["registration_deadline"] == 60.0


def test_a_pass_registered_again_on_another_stream_is_refused(served):
    """Windows credited together hold each call index of their salt once (`credit.passes_distinct`): the same pass
    registered again on another stream would credit every call twice under the same seeds and tickets, so it is refused
    (``repeated-call``) and not held, while the first stays (CMP-84)."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ch = _challenger(params, record=OR.Record(lean=False))
    _held(ch, "honest", rec)
    with pytest.raises(S.Refused) as e:
        _held(ch, "again", rec, stream="again")
    assert _codes(e) == {"repeated-call"} and "both hold call index" in e.value.verdict.detail["repeated-call"]
    assert "again" not in ch.regs and list(ch.manifests) == ["honest"] and ch.passes == {"honest": "coin"}


def test_a_window_received_after_the_epochs_draw_coin_is_refused(served, tmp_path):
    """The record's order is each window's receipt before the epoch's draw coin's release: once the draw coin is open, a
    window is refused (`late-registration`), even one whose pass and record are honest and within the deadline, and its
    stream is left as it was.  A control replaying the honest draw opens no coin and still registers.  A record whose
    parameters predate the deadline replays as it ran (`replay`), byte for byte: its post-draw registration of the same
    pass on another stream is refused there too, as a repeated call (`credit.passes_distinct`)."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ch = _challenger(params, record=OR.Record(lean=False))
    _held(ch, "honest", rec)
    ch.select("honest")
    with pytest.raises(S.Refused) as e:
        _held(ch, "after", rec, stream="later")
    assert _codes(e) == {"late-registration"} and "after the epoch's draw coin" in e.value.verdict.detail["late-registration"]
    assert "after" not in ch.regs and ("later", SC.level_of(rec)) not in ch.streams and not ch.witness["after"]
    ch.register("ctl", rec, stream="ctl", replay="honest")
    assert ch.select("ctl").units == ch.draws["honest"].units

    bare = {k: v for k, v in params.items() if k != "registration_deadline"}
    old = SC.Challenger(bare, _coins(), record=OR.Record(lean=False), predates_deadline=True)
    log = tmp_path / "log"
    log.mkdir()
    (log / "coins.json").write_text(json.dumps(old.coins.commitment.to_json()))
    (tmp_path / "rec.json").write_text(json.dumps(rec))
    (tmp_path / "man.json").write_bytes(PASSES[params["context"]][1])
    reg = {"registration": f"{tmp_path}/rec.json", "manifest": f"{tmp_path}/man.json"}
    reqs = [{"op": "salt", "after": _salt_after(params).hex()}, {"op": "register", "name": "honest", **reg},
            {"op": "select", "name": "honest"}, {"op": "register", "name": "after", "stream": "later", **reg}]
    replies = [SC.answer(old, json.dumps(r)) for r in reqs]
    assert [x.split(" ", 1)[0] for x in replies] == ["OPENED", "REGISTERED", "DRAWN", "REFUSED"] and "repeated-call" in replies[3]
    (log / "requests.jsonl").write_text("".join(json.dumps({"request": json.dumps(r), "reply": x}) + "\n"
                                                for r, x in zip(reqs, replies)))
    (log / "openings.jsonl").write_text("".join(json.dumps(o.to_json()) + "\n" for o in old.openings.values()))
    assert SC.replay(bare, log, tmp_path / "replay", record=OR.Record(lean=False)) == []
    (tmp_path / "bare.json").write_text(json.dumps(bare))
    assert SC.main(["replay", str(tmp_path / "bare.json"), str(log), str(tmp_path / "replay2"), "--python-record"]) == 0


#: the network certifier's grids, as `certifier.active` runs them; a PoUW window is two certifier windows of CW seconds
EGRESS, INGRESS = GridParams(buckets=6, allocation=3, frame_bytes=512), GridParams(buckets=6, allocation=2, frame_bytes=512)
CW = INGRESS.buckets * INGRESS.bucket_ms / 1000
#: the certifier's run ids, fixed before each proxy starts so the prover can name its run in the stream it records
RUN_ID, OTHER_RUN_ID = bytes(range(32)), bytes(range(1, 33))


def _prover(seed: bytes):
    """A prover's LMS key as the certifier registers it, and a signer at one-time keys 0, 1, ... in turn."""
    k, leaves = lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(seed).digest()), itertools.count()
    return LmsKey(k.params, lms.public_key(k)), lambda m: lms.sign_at(k, next(leaves), m).encode()


def _keys(tmp_path: Path) -> tuple[LmsSigner, LmsSigner]:
    """The network certifier's LMS signer and the challenger's, whose key the certifier registers (N = 64, W = 1, 2^6
    leaves each), each with fresh leaf state under `tmp_path`."""
    out = []
    for name in ("certifier", "challenger"):
        (tmp_path / name).mkdir(exist_ok=True)
        k = lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(name.encode()).digest())
        out.append(LmsSigner.create(k, tmp_path / name / "signer-state.json"))
    return out[0], out[1]


def _certifier(sid: bytes, key, sign, record: bytes, keys, seed: bytes = b"", clock=lambda: 0.0, windows: int = 1,
               link: str = "requests", run: bytes = RUN_ID) -> tuple[Proxy, dict[int, bytes]]:
    """The certifier's run `run` on ingress link `link`, answering under `keys`' certifier signer the requests its
    challenger signs, PoUW's reader and the prover's `key` registered, the stream `sid` its context, on `clock`: the
    prover records `record` for PoUW windows 0 … `windows` − 1 before the first tick, signed, then they are published.
    With the salt it drew for each, which the prover's server receives in-band."""
    drawn: dict[int, bytes] = {}

    def source(p: int, n: int) -> bytes:
        drawn[p] = hashlib.shake_256(b"certifier salt %d %s" % (p, seed)).digest(n)
        return drawn[p]

    c = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SyncSet((0, 1), (1, 2)), [ClockSync(0, 1)] * 4, origin=0.0, clock=clock,
              ingress_link=link, salt=source, registered=PS.registered, prover_key=key, prover_context=sid, pouw_window=2,
              challenger_key=keys[1].public_key(), signer=keys[0], run_nonce=lambda n: run)
    for p in range(windows):
        c.commit_weights(p, record, sign(active.weights_message(link, p, record, sid)))
    for _ in range(2 * windows * INGRESS.buckets):
        c.bucketer.tick()
    return c, drawn


def _client(c: Proxy, at: tuple[str, int], keys) -> active.RemoteCertifier:
    """The challenger's client of certifier `c` served at `at` (the `reveals` fixture), for its key, run and link."""
    _, run_id, link = c.certifier_run
    return active.RemoteCertifier(*at, c.signer.public_key(), keys[1], link, run_id)


def _sid(run: str, keys, stream: str = "window", link: str = "requests", run_id: bytes = RUN_ID) -> bytes:
    """The stream id the prover records with the certifier's run `run_id` on `link` under `keys`' certifier key
    (`stream_id`)."""
    return SC.stream_id({"deployment": f"served-zk/{run}"}, stream,
                        {"key_id": keys[0].key_id.hex(), "link": link, "run_id": run_id.hex()})


def _certified_window(words, run: str, reveal: bytes, sid: bytes, prover: bytes,
                      p: int = 0) -> tuple[dict, bytes, dict, dict, dict]:
    """A window of two calls from a certified pass of `run` (`MANIFEST_CERTIFIED`) in PoUW window `p`, salted by
    `Epoch.certified` of the certifier's salt `reveal` under the stream `sid` and `prover`, its rows `words`: its
    parameters, which name window `p` and carry no registration_deadline, the manifest's bytes, the window's
    registration, and its advice and private rows (`commit_window`), which open its units."""
    man, roots, work = _pass(run)
    raw = _raw({**man, "schema": CC.MANIFEST_CERTIFIED,
                "salt": Epoch.certified(work, roots, [2 * p, 2 * p + 1], 2, reveal, sid, prover).salt.hex()})
    params = {k: v for k, v in _params(run, raw).items() if k != "registration_deadline"}
    params["context"] = PS.context(f"served-zk/{run}", p, hashlib.sha512(raw).hexdigest())
    root_w, _ = SC.commit_weights(params, _weights())
    public, advice, priv = SC.commit_window(params, _calls(words), root_w, SC.root_h3_of(man),
                                            length=SC.served_length(SERVED))
    return params, raw, _register(params, public), advice, priv


def _live(path: Path, entropy: str | None = None) -> tuple[Ch.LiveChallenger, Ch.ChallengerKey]:
    """A live epoch's challenger on its host, its record signed and written to `path`, and the verifier's key for that
    record (`challenger.ChallengerKey`); `entropy` names another challenger's coins, whose record isn't the one written."""
    k = lms.keygen(lms.Params(64, 4, 2), hashlib.sha512(str(path).encode()).digest())
    live = Wd.live_challenger(0, 1, random.Random(str(path)).randbytes)
    Ch.write_record(path, live.sign_record("vy-nebius-cpu-1", lms.public_key(k), lambda m: lms.sign_at(k, 0, m).encode()))
    return (live if entropy is None else Wd.live_challenger(0, 1, random.Random(entropy).randbytes)), \
        Ch.ChallengerKey(4, 2, lms.public_key(k))


def _deployment(client: active.RemoteCertifier, ck: Ch.ChallengerKey) -> SC.Deployment:
    """What the verifier pins of the deployment: `client`'s certifier key and run id, and its challenger's key `ck`."""
    return SC.Deployment(client.certifier_key, client.run_id, ck)


def _unasked(monkeypatch) -> list[str]:
    """What the challenger asks the certifier (`salt.committed`, `salt.revealed`), recorded and never answered."""
    asked: list[str] = []
    monkeypatch.setattr(PS, "committed", lambda *a: asked.append("committed"))
    monkeypatch.setattr(PS, "revealed", lambda *a: asked.append("revealed"))
    return asked


def test_rows_formed_before_any_salt_are_refused_in_every_certified_window(served, tmp_path, monkeypatch, reveals):
    """RT4-INT-1's witness, refused.  The fixture's rows are formed once, under SHA-256(b"salt"), which is no certifier
    window's salt, and committed twice (matmuls 5 and 9) in PoUW windows 0 and 1, each window's pass salted by its own
    certifier reveal under the recorded stream and prover.  A certified window credits only units that read its salt
    (`salt.unit_reason`), and the window's drawn unit, `Pc8TileSide`, reads two committed rows and no salt, so both are
    refused (``unsalted``) before the stream receives them and before the certifier is asked for its record or salt:
    nothing is registered, received or drawn."""
    *_, words = served
    run, keys = "r-certified", _keys(tmp_path)
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), _sid(run, keys)
    c, salts = _certifier(sid, key, sign, PS.pouw_record(weights_root(_pass(run)[1]), sid, prover), keys, windows=2)
    client = _client(c, reveals(c), keys)
    asked, ledger = _unasked(monkeypatch), PS.Registrations()
    unread = PS.unit_reason(SC.template(K))
    assert unread is not None and SC.template(K).id in unread
    for p in (0, 1):
        params, raw, rec, *_ = _certified_window(words, run, salts[p], sid, prover, p)
        live, ck = _live(tmp_path / f"w{p}.json")
        ch = SC.Challenger(params, live, clock=lambda: 0.0, record=OR.Record(lean=False), certifier=client,
                           prover=prover, coin_record=tmp_path / f"w{p}.json", deployment=_deployment(client, ck),
                           registrations=ledger)
        with pytest.raises(S.Refused) as e:
            ch.register(f"w{p}", rec, manifest=raw)
        assert _codes(e) == {"unsalted"} and e.value.verdict.detail["unsalted"] == unread
        assert (ch.regs, ch.streams, ch.passes, ch.openings) == ({}, {}, {}, {})
    assert salts[0] != salts[1] and asked == [] and len(ledger) == 0


def test_one_windows_rows_are_refused_under_two_streams(served, tmp_path, monkeypatch, reveals):
    """RT4-INT-9's witness, refused.  One prover registers streams "a" and "b", each behind its own certifier link, and
    commits the same rows, formed before any salt, in PoUW window 0 of both: their stream ids differ (each names its
    stream and its certifier run, whose link differs, `stream_id`), and each window is refused (``unsalted``) before it
    is received, so no product is registered once, let alone once per stream."""
    *_, words = served
    run, keys = "r-certified", _keys(tmp_path)
    key, sign = _prover(b"the prover")
    prover, asked, ledger = PS.prover_id(key.public), _unasked(monkeypatch), PS.Registrations()
    sids = {s: _sid(run, keys, s, f"requests-{s}") for s in "ab"}
    for s, sid in sids.items():
        c, salts = _certifier(sid, key, sign, PS.pouw_record(weights_root(_pass(run)[1]), sid, prover), keys,
                              s.encode(), link=f"requests-{s}")
        client = _client(c, reveals(c), keys)
        assert SC.stream_id({"deployment": f"served-zk/{run}"}, s, SC.certifier_run(client)) == sid
        params, raw, rec, *_ = _certified_window(words, run, salts[0], sid, prover)
        live, ck = _live(tmp_path / f"{s}.json")
        ch = SC.Challenger(params, live, clock=lambda: 0.0, record=OR.Record(lean=False), certifier=client,
                           prover=prover, coin_record=tmp_path / f"{s}.json", deployment=_deployment(client, ck),
                           registrations=ledger)
        with pytest.raises(S.Refused) as e:
            ch.register(s, rec, stream=s, manifest=raw)
        assert _codes(e) == {"unsalted"} and not ch.streams and not ch.regs
    assert sids["a"] != sids["b"] and asked == [] and len(ledger) == 0


def test_past_the_salt_rule_a_certified_window_is_recorded_revealed_drawn_under_a_signed_record_and_registered_once(
        served, tmp_path, monkeypatch, reveals):
    """What a certified window of salted units runs through after `salt.unit_reason`, which no served-zk window passes
    today (the two tests above): this test lifts that one rule (monkeypatched to accept) and holds the rest end to end
    against the network certifier, which the challenger asks through its client (`RemoteCertifier`, a signed request
    and a signed answer it verifies, over localhost to `serve_reveals`).  The prover records its pass's weights, the
    window's stream (`stream_id`, naming the certifier's run, (key id, run id, ingress link), which the proxy and its
    client give alike) and its prover id (`salt.prover_id`) with the certifier for PoUW window 0 before its salt is
    drawn, signed, and salts its pass from the salt it receives in-band.  `register` reads that record on the window's
    registration and only then has the certifier reveal window 0's salt, so a challenger whose stream or prover isn't
    the recorded one is refused (``coin``) and takes no reveal: the honest one after it passes (``certifier``), holds
    the salt its pass was checked against, and its draw is the live epoch's ``pouw/draw`` coin 0 opened after the
    registration, under the commitment the challenger read itself from its challenger's signed record, which the draw's
    check of record verifies again under the challenger key the deployment pins (`Deployment`), and refuses under any
    other pin (``coin-key``).  Its drawn units are opened under that salt (`Challenger.check`, SALT-BIND-1),
    where `check` alone passes none.  The certifier reveals window 0 once and within its reveal deadline on its own
    clock, so a second registration, or one past the deadline, is refused (``certifier``).  A second proxy of the same
    run (its key, run id and link) gives the same stream id, and the deployment's shared `Registrations` refuse its
    window 0 (``reregistered``); another run (another run id) is another stream, so a record naming the first run's
    stream is refused (``coin``) and one naming its own registers, though coins that aren't its signed record's
    challenger's are refused at the draw.  A certified challenger opens no salt coin, and is refused parameters with a
    registration deadline, a certifier that names no run (an in-process proxy, which has no signer), no deployment, a
    client for another certifier key or run id than the deployment pins, coins it can't read from a signed record under
    the deployment's challenger key, a prover id that isn't 64 bytes or a context that names no PoUW window; parameters
    that name one are refused without a certifier.  A deployment is a certifier's public key, a 32-byte run id and a
    challenger key, or it is refused (`DEPLOYMENT_FORM`)."""
    monkeypatch.setattr(PS, "unit_reason", lambda unit: None)
    *_, words = served
    run, keys = "r-certified", _keys(tmp_path)
    key, sign = _prover(b"the prover")
    prover, sid = PS.prover_id(key.public), _sid(run, keys)
    other_prover = PS.prover_id(_prover(b"another prover")[0].public)
    record = PS.pouw_record(weights_root(_pass(run)[1]), sid, prover)
    t = [0.0]
    c, salts = _certifier(sid, key, sign, record, keys, clock=lambda: t[0])
    client = _client(c, reveals(c), keys)
    assert SC.certifier_run(client) == SC.certifier_run(c) == \
        {"key_id": keys[0].key_id.hex(), "link": "requests", "run_id": RUN_ID.hex()}
    with pytest.raises(TypeError, match="ingress_link"):
        Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SyncSet((0, 1), (1, 2)), [ClockSync(0, 1)] * 4, origin=0.0)
    params, raw, rec, advice, priv = _certified_window(words, run, salts[0], sid, prover)
    ledger = PS.Registrations()

    def challenger(name: str, certifier=client, prover=prover, p=params, entropy=None) -> SC.Challenger:
        live, ck = _live(tmp_path / f"{name}.json", entropy)
        return SC.Challenger(p, live, clock=lambda: 0.0, record=OR.Record(lean=False), certifier=certifier,
                             prover=prover, coin_record=tmp_path / f"{name}.json",
                             deployment=_deployment(certifier, ck), registrations=ledger)

    def refused(ch: SC.Challenger, code: str, r=rec, m=raw, **kw) -> str:
        with pytest.raises(S.Refused) as e:
            ch.register("x", r, manifest=m, **kw)
        assert _codes(e) == {code} and "x" not in ch.regs and "x" not in ch.passes
        return e.value.verdict.detail[code]

    unrecorded = "the pass's weights, stream and prover are not those the certifier recorded before it drew the window's salt"
    assert refused(challenger("other-stream"), "coin", stream="another") == unrecorded
    assert refused(challenger("other-prover", prover=other_prover), "coin") == unrecorded
    ch = challenger("honest")
    with pytest.raises(S.Refused) as e:
        ch.open("salt", bytes(64))
    assert _codes(e) == {"coin"} and "opens no salt coin" in e.value.verdict.detail["coin"]
    reg = ch.register("honest", rec, manifest=raw)
    salt = PS.pouw_salt(salts[0], 0, sid, prover)
    assert ch.passes == {"honest": "certifier"} and ch.salts == {"honest": salt} and "draw" not in ch.openings
    assert (0, sid, prover) in ledger
    drawn, draw_of_record = [], ch.record.draw
    ch.record.draw = lambda x: drawn.append(x) or draw_of_record(x)
    d = ch.select("honest")
    assert [x["key"] for x in drawn] == [ch.deployment.challenger_key.to_json()] == [ch.signed["key"]]
    assert OR.py_draw({**drawn[0], "key": _live(tmp_path / "pinned.json")[1].to_json()}) == \
        {"refused": [CC.COIN_KEY, CC.OTHER_KEY]}
    o, st = ch.openings["draw"], ch.regs["honest"][0]
    assert (o.label, o.index, o.after) == (Wd.DRAW, 0, bytes.fromhex(reg.digest))
    assert Ch.live_reason(ch.trusted, o, Wd.DRAW, 0, o.after) is None and ch.trusted == ch.coins.commitment
    assert ch.trusted == Ch.trusted_commitment(tmp_path / "honest.json", Ch.ChallengerKey.from_json(ch.signed["key"]),
                                               CC.lms_verify)
    k = SC.law_of(params, SC.level_of(rec))["k"]
    assert d.units == CC.stream_subset(k, 26, SC.draw_bytes(SC.draw_key(o.coin, st, reg), d.law, 26))
    assert ch.coins_of("honest") == {"commitment": ch.trusted.to_json(), "pass": "certifier", "draw": o.to_json()}

    held, opening = [], CC.opening
    monkeypatch.setattr(CC, "opening", lambda *a: held.append(a[6]) or opening(*a))
    u = d.units[0]
    pub, op = _unit_pub(advice, priv, u), SC.opening(advice, priv, params, u)
    unread = "the window is certified, and its template's unit reads no salt"
    assert ch.check("honest", u, pub, op).startswith(unread)
    assert SC.check(params, rec, d.draw, u, pub, op, record=OR.Record(lean=False)).startswith(unread)
    assert held == [salt, None]
    monkeypatch.setattr(CC, "opening", opening)
    assert refused(challenger("again"), "certifier").startswith("the certifier doesn't reveal PoUW window 0's salt")

    late, late_salts = _certifier(sid, key, sign, record, keys, seed=b"late", clock=lambda: t[0])
    lparams, lraw, lrec, *_ = _certified_window(words, run, late_salts[0], sid, prover)
    t[0] = 4 * CW + 1e-3
    assert "the certifier doesn't give what PoUW recorded for window 0" in \
        refused(challenger("late", certifier=_client(late, reveals(late), keys), p=lparams), "certifier", lrec, lraw)
    t[0] = 0.0

    twin, twin_salts = _certifier(sid, key, sign, record, keys, seed=b"twin")
    tclient = _client(twin, reveals(twin), keys)
    assert twin_salts[0] != salts[0] and SC.certifier_run(tclient) == SC.certifier_run(client)
    tparams, traw, trec, *_ = _certified_window(words, run, twin_salts[0], sid, prover)
    assert f"PoUW window 0 is already registered ({reg.digest[:16]})" in \
        refused(challenger("twin", certifier=tclient, p=tparams), "reregistered", trec, traw)

    rsid = _sid(run, keys, run_id=OTHER_RUN_ID)
    stale, stale_salts = _certifier(sid, key, sign, record, keys, seed=b"stale", run=OTHER_RUN_ID)
    sparams, sraw, srec, *_ = _certified_window(words, run, stale_salts[0], sid, prover)
    assert refused(challenger("stale", certifier=_client(stale, reveals(stale), keys), p=sparams), "coin", srec,
                   sraw) == unrecorded
    again, again_salts = _certifier(rsid, key, sign, PS.pouw_record(weights_root(_pass(run)[1]), rsid, prover), keys,
                                    seed=b"restarted", run=OTHER_RUN_ID)
    aclient = _client(again, reveals(again), keys)
    assert rsid != sid and SC.stream_id({"deployment": f"served-zk/{run}"}, "window", SC.certifier_run(aclient)) == rsid
    rparams, rraw, rrec, *_ = _certified_window(words, run, again_salts[0], rsid, prover)
    rch = challenger("restarted", certifier=aclient, p=rparams, entropy="not the record's challenger")
    rch.register("restarted", rrec, manifest=rraw)
    assert rch.passes == {"restarted": "certifier"} and (0, sid, prover) in ledger and (0, rsid, prover) in ledger
    with pytest.raises(S.Refused) as e:
        rch.select("restarted")
    assert _codes(e) == {"coin"} and "restarted" not in rch.draws

    live, ck = _live(tmp_path / "init.json")
    _, foreign = _live(tmp_path / "foreign.json")
    dep = _deployment(client, ck)
    base = dict(certifier=client, prover=prover, coin_record=tmp_path / "init.json", deployment=dep)
    in_process = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SyncSet((0, 1), (1, 2)), [ClockSync(0, 1)] * 4, origin=0.0,
                       ingress_link="requests")
    pinned = "is for the certifier key and run id its deployment pins"
    for p, kw, why in (({**params, "registration_deadline": 60.0}, base, "carry no registration_deadline"),
                       (params, {**base, "certifier": in_process}, "names its run.*without a signer"),
                       (params, {**base, "deployment": None}, "takes its deployment"),
                       (params, {**base, "deployment": ck}, "takes its deployment"),
                       (params, {**base, "deployment": dataclasses.replace(dep, run_id=OTHER_RUN_ID)}, pinned),
                       (params, {**base, "deployment": dataclasses.replace(dep, certifier_key=keys[1].public_key())},
                        pinned),
                       (params, {**base, "coin_record": None}, "signed record"),
                       (params, {**base, "coin_record": tmp_path / "missing.json"}, "opens against its challenger's"),
                       (params, {**base, "deployment": dataclasses.replace(dep, challenger_key=foreign)},
                        "opens against its challenger's"),
                       (params, {**base, "prover": prover[:32]}, "64-byte id"),
                       ({**params, "context": _params(run, raw)["context"]}, base, "names its PoUW window"),
                       (params, {}, "so it is certified")):
        with pytest.raises(ValueError, match=why):
            SC.Challenger(p, live, **kw)
    for edit in ({"challenger_key": _coins().commitment}, {"challenger_key": ck.to_json()}, {"run_id": RUN_ID[:16]},
                 {"run_id": RUN_ID.hex()}, {"certifier_key": keys[0]}, {"certifier_key": keys[0].public_key().public}):
        with pytest.raises(ValueError, match=re.escape(SC.DEPLOYMENT_FORM)):
            dataclasses.replace(dep, **edit)


def test_a_deployments_challenger_key_is_infras_published_pin(tmp_path):
    """`Deployment.pinned` takes the challenger key from infra's published pin (`CHALLENGER_KEY_PIN`; the fixture is its
    payload's bytes): vy-challenger-1's LMS key at W = 1, H = 20, whose SHA-512 is `CHALLENGER_KEY_SHA512`, stated whole.
    A record of any other key is refused even when it names its own key's SHA-512: the key infra retired (RT4-INT-18), a
    tampered public key, or the pinned key under another named SHA-512; so is a record of another schema or form."""
    pin = Path(__file__).parent / "fixtures" / "challenger-key.json"
    assert hashlib.sha256(pin.read_bytes()).hexdigest() == \
        "8f7a5515c5f6c5e22db062ca378c6b6e907e25bad6fda7b301efe22b3829db32"
    assert re.fullmatch(r"art:[0-9a-f]{64}", SC.CHALLENGER_KEY_PIN) and re.fullmatch(r"[0-9a-f]{128}",
                                                                                     SC.CHALLENGER_KEY_SHA512)
    doc = json.loads(pin.read_bytes())
    ck = SC.pinned_challenger_key(pin)
    assert (ck.W, ck.H, ck.public.hex()) == (1, 20, doc["pubkey"])
    assert hashlib.sha512(ck.public).hexdigest() == doc["pubkey_sha512"] == SC.CHALLENGER_KEY_SHA512
    certifier = _keys(tmp_path)[0].public_key()
    dep = SC.Deployment.pinned(certifier, RUN_ID, pin)
    assert (dep.certifier_key, dep.run_id, dep.challenger_key) == (certifier, RUN_ID, ck)

    retired = doc["retired"][0]
    flipped = ("0" if doc["pubkey"][0] != "0" else "1") + doc["pubkey"][1:]
    other = {**doc, "pubkey": flipped, "pubkey_sha512": hashlib.sha512(bytes.fromhex(flipped)).hexdigest()}
    for edit, why in (({"pubkey": retired["pubkey"], "pubkey_sha512": retired["pubkey_sha512"]}, SC.NOT_PINNED),
                      (other, SC.NOT_PINNED), ({"pubkey": flipped}, SC.NOT_PINNED),
                      ({"pubkey_sha512": retired["pubkey_sha512"]}, SC.NOT_PINNED),
                      ({"schema": "challenger-key/v2"}, SC.CHALLENGER_KEY_FORM),
                      ({"pubkey": doc["pubkey"].upper()}, SC.CHALLENGER_KEY_FORM),
                      ({"pubkey": doc["pubkey"][:-1]}, SC.CHALLENGER_KEY_FORM),
                      ({"pubkey": None}, SC.CHALLENGER_KEY_FORM), ({"W": 3}, SC.CHALLENGER_KEY_FORM),
                      ({"H": True}, SC.CHALLENGER_KEY_FORM)):
        f = tmp_path / "record.json"
        f.write_text(json.dumps({**doc, **edit}))
        with pytest.raises(ValueError, match=re.escape(why)):
            SC.Deployment.pinned(certifier, RUN_ID, f)
    for raw in (b"[]", b"not json", None):
        f = tmp_path / "raw.json"
        f.unlink(missing_ok=True)
        if raw is not None:
            f.write_bytes(raw)
        with pytest.raises(ValueError, match=re.escape(SC.CHALLENGER_KEY_FORM)):
            SC.pinned_challenger_key(f)


@LEAN
def test_a_window_whose_pass_coin_is_foreign_early_mislabelled_or_tampered_is_refused(served):
    """The challenger re-starts the window's pass's epoch from its manifest under its own coin commitment: a salt coin of
    another challenger, opened after what it had not recorded, labelled the draw coin or tampered is refused, and so is a
    pass whose commitment is not the epoch's, whose salt is not its coin's, which carries no coin or names no pass, or
    which is not the pass the window's parameters name.  A pass recorded before epoch coins (a v1 manifest, a beacon's
    salt) is received only as a recorded replay, and a manifest of any other schema not at all.  Nothing refused opens a
    coin."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    coins, raw, man = PASSES[params["context"]]
    opening = CoinOpening.from_json(man["salt_opening"])

    def refused(m, recorded=False, code="coin"):
        ch = _challenger(params)
        with pytest.raises(S.Refused) as e:
            ch.register("x", rec, manifest=None if m is None else _raw(m), recorded=recorded)
        assert _codes(e) == {code} and set(ch.openings) == {"salt"} and "x" not in ch.regs
        return e.value.verdict.detail[code]

    for case, why in (("foreign", "coin does not open against the epoch's coin commitment"),
                      ("early", "coin was not opened after what the verifier recorded"),
                      ("mislabelled", "coin is the epoch's 'draw' coin, not its 'salt' coin"),
                      ("tampered", "coin does not open against the epoch's coin commitment")):
        assert refused(SC.coin_variant(man, coins.commitment, opening, case)) == f"the salt's {why}", case
    assert "not the epoch's" in refused({**man, "coins": _coins(8).commitment.to_json()})
    assert "salt is not its coin's" in refused(SC.coin_variant({**man, "salt": "00" * 32}, coins.commitment, opening, "resalted"))
    assert "no salt coin" in refused({k: v for k, v in man.items() if k not in ("coins", "salt_opening")})
    assert "names no pass" in refused(None)
    assert "not the one the window's parameters name" in refused({**man, "note": "another pass"}, code="manifest")
    assert _held(_challenger(params), "honest", rec).digest == R.digest(rec)

    bparams, bpublic, _, _ = _window(words, run="r-beacon", recorded=True)
    brec, braw = _register(bparams, bpublic), PASSES[bparams["context"]][1]
    with pytest.raises(S.Refused) as e:
        _challenger(bparams, coins=_coins()).register("x", brec, manifest=braw)
    assert _codes(e) == {"coin"} and f"is {PD.MANIFEST_SCHEMA_RECORDED}, recorded before epoch coins" in e.value.verdict.detail["coin"]
    _, _, bman = PASSES[bparams["context"]]
    for schema, code, why in (("verity-vllm/pouw-pearl-c-run/v0", "coin", f"schema 'verity-vllm/pouw-pearl-c-run/v0', not {PD.MANIFEST_SCHEMA}"),
                              (None, "manifest", "the pass's manifest names no schema")):
        ch = _challenger(bparams, coins=_coins())
        with pytest.raises(S.Refused) as e:
            ch.register("x", brec, manifest=_raw({**bman, "schema": schema}), recorded=True)
        assert why in e.value.verdict.detail[code]
    ch = _challenger(bparams, coins=_coins())
    ch.register("honest", brec, manifest=braw, recorded=True)
    assert ch.passes == {"honest": "recorded beacon"} and len(ch.select("honest").units) == SC.law_of(bparams, bpublic["level"])["k"]


def test_stream_subset_is_the_exact_partial_fisher_yates_on_one_stream():
    """`stream_subset` reads `width(n - i)`-byte words in order, skips a word at or above the largest multiple of the
    range, and runs out on a short stream, as Lean's ``Draw.subset`` does."""
    n, k = 26, 3
    w = CC.width(n)
    word = lambda x: x.to_bytes(w, "big")  # noqa: E731
    top = (1 << (8 * w)) - (1 << (8 * w)) % n
    src = word(top) + word(5) + word(0) + word(199)          # top is refused; then 5 % 26, 0 % 25, 199 % 24 = 7
    assert CC.stream_subset(k, n, src) == [1, 5, 9]
    assert CC.stream_subset(k, n, src[:-1]) is None
    key = SC.derive(b"\x01" * 32, SC.DRAW_DOMAIN, {"x": "y"})
    units = CC.stream_subset(8, 1000, SC.draw_bytes(key, {"law": "subset", "k": 8}, 1000))
    assert len(set(units)) == 8 and units == sorted(units) and all(0 <= u < 1000 for u in units)


@LEAN
def test_check_opens_the_drawn_units_rows_only(served, tmp_path):
    """A proof's public file opens a drawn unit, a call's tile or the filler's, only when its rows are that unit's rows of
    this window: another row, another window's rows, another window's registration, or an undrawn unit is rejected."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec = _register(params, public)
    SC.save(tmp_path / "p.npz", priv)
    priv = SC.load(tmp_path / "p.npz")
    draw = {"law": "subset", "population": 26, "k": 3, "units": [5, 13, 25]}
    for t in draw["units"]:
        assert SC.check(params, rec, draw, t, _unit_pub(advice, priv, t), SC.opening(advice, priv, params, t)) is None
    assert SC.opening(advice, priv, params, 25)["entry_value"]["kind"] == "filler"
    op = SC.opening(advice, priv, params, 5)
    assert "not entry 0's row" in SC.check(params, rec, draw, 5, _unit_pub(advice, priv, 6), op)
    assert "not drawn" in SC.check(params, rec, draw, 6, _unit_pub(advice, priv, 6), SC.opening(advice, priv, params, 6))
    assert SC.check(params, rec, draw, 6, _unit_pub(advice, priv, 6), SC.opening(advice, priv, params, 6), drawn=False) is None
    assert "not the unit's row" in SC.check(params, rec, draw, 6, _unit_pub(advice, priv, 6), op, drawn=False)
    oparams, opublic, oadvice, opriv = _window(words, run="r-other")
    orec = _register(oparams, opublic)
    assert "row" in SC.check(params, rec, draw, 5, _unit_pub(oadvice, opriv, 5), op)
    assert "window" in SC.check(params, orec, draw, 5, _unit_pub(advice, priv, 5), op)
    forged = {**op, "entry_value": {**op["entry_value"], "hi": op["entry_value"]["hi"] + 1}}
    assert "not the registered window's entry" in SC.check(params, rec, draw, 5, _unit_pub(advice, priv, 5), forged)
    for bad in ({**op["entry_value"], "kind": "tile"}, {**op["entry_value"], "root_a": "00"}, {"kind": "call"}, None):
        assert SC.check(params, rec, draw, 5, _unit_pub(advice, priv, 5), {**op, "entry_value": bad}) == "the opening names no entry"


def _session(t: int) -> str:
    """Unit `t`'s session record's SHA-512, as the firewall delivers it and Lean's verdict names it."""
    return hashlib.sha512(f"session {t}".encode()).hexdigest()


def _unit(t: int, lean: bool = True, opening: bool = True, session: int | None = None) -> dict:
    stage = {"definition": "Pc8TileSide", "circuit_sha512": "11" * 64, "class_sha512": "22" * 64, "ands": 7, "k_log": 3,
             "binding": "hash scratch"}
    return {"unit": t, "statement": SC.statement(stage, b"pub"),
            "verdicts": {"lean": {"of_record": True, "accepted": lean, "why": "", "digest": "33" * 64,
                                  "sessions": [_session(t if session is None else session)]},
                         "opening": {"of_record": False, "accepted": opening, "why": ""}}}


@LEAN
def test_the_outcome_publishes_exactly_the_public_items(served):
    """The outcome over every drawn unit accepts only when each unit's Lean verdict and opening check accept; PoUW's public
    items of it are exactly `window.PUBLIC`; it waits for the proof's delivery, is final, and units other than the draw's
    are refused, and so is a Lean verdict about sessions other than the delivered ones.  A control stream replays the
    honest draw: a reject there is final, and its failure budget stops it."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ch = _challenger(params)
    _held(ch, "honest", rec)
    d = ch.select("honest")
    units, delivered = [_unit(t) for t in d.units], [_session(t) for t in d.units]
    with pytest.raises(S.Pending):
        ch.outcome("honest", units)
    assert ch.deliver("honest", delivered) is None
    with pytest.raises(S.Refused) as e:
        ch.outcome("honest", [_unit(t, session=99 if t == d.units[1] else None) for t in d.units])
    assert _codes(e) == {"sessions"}
    with pytest.raises(S.Refused) as e:
        ch.outcome("honest", units[:-1])
    assert _codes(e) == {"units"}
    o, pub, body = ch.outcome("honest", units)
    assert o.accepted and o.profile is not None and body["audit"]["proved_units"] == d.units and body["sessions"] == delivered
    assert set(pub) == {p[0] for p in Wd.PUBLIC} and pub["verdict"] is True and pub["drawn"] == d.units
    assert pub["work"] == public["items"]["work"] and pub["root_W"] == public["items"]["root_W"]
    assert body["audit"]["verdicts"]["lean"]["sessions"] == delivered
    for again in (lambda: ch.outcome("honest", units), lambda: ch.deliver("honest", delivered)):
        with pytest.raises(S.Refused) as e:
            again()
        assert _codes(e) == {"final"}
    for i, bad in enumerate(([_unit(t, lean=t != d.units[1]) for t in d.units], [_unit(t, opening=False) for t in d.units])):
        name = f"ctl-{i}"
        ch.register(name, rec, stream=name, replay="honest")
        assert ch.select(name).units == d.units
        ch.deliver(name, delivered)
        o, pub, body = ch.outcome(name, bad)
        assert not o.accepted and o.verdict.codes == ("reject",) and pub["verdict"] is False and o.profile is None
        with pytest.raises(S.Refused) as e:
            ch.register(f"{name}-retry", rec, stream=name, replay="honest")
        assert _codes(e) == {"stopped"}
        with pytest.raises(S.Refused) as e:
            _held(ch, f"{name}-window", rec, stream=name)
        assert _codes(e) == {"late-registration"}
    assert body["audit"]["verdicts"]["opening"]["units"] == [False] * 4


def _through(ch, rec, name="honest"):
    """Window `rec` received, drawn, delivered and its outcome, every drawn unit accepted."""
    _held(ch, name, rec)
    d = ch.select(name)
    ch.deliver(name, [_session(t) for t in d.units])
    return ch.outcome(name, [_unit(t) for t in d.units])


@LEAN
def test_lean_is_of_record_and_python_its_witness(served, tmp_path):
    """Each of the window's checks runs Lean's port and Python's reference on one input: the witness holds both verdicts,
    the same, Lean's of record, and the published items are Lean's.  Undone (`Record(lean=False)`), the references alone
    give the same outcome and witness no Lean verdict."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec = _register(params, public)
    ch = _challenger(params)
    o, pub, _ = _through(ch, rec)
    w = ch.witness["honest"]
    assert [e["check"] for e in w] == ["registered", "pass", "registration", "public", "draw", "answer"]
    assert all(e["same"] and e["of_record"] == "lean" and e["lean"] == e["python"] for e in w)
    answer = w[-1]["lean"]
    assert o.accepted and answer["verdict"] == "accept" and pub == answer["public"]
    record = OR.Record(lean=False)
    py = _challenger(params, record=record)
    o2, pub2, _ = _through(py, rec)
    assert o2.accepted and pub2 == pub and all(e["lean"] is None and e["of_record"] == "python" for e in py.witness["honest"])
    t = ch.draws["honest"].units[0]
    lean = OR.Record()
    assert SC.check(params, rec, ch.draws["honest"].draw, t, _unit_pub(advice, priv, t), SC.opening(advice, priv, params, t),
                    record=lean) is None
    assert [(e["check"], e["same"], e["lean"]) for e in lean.witness] == [("opening", True, None),
                                                                           ("prelude", True, {"reason": None})]
    lean.close()


@LEAN
def test_r1_r6_of_a_window_are_leans(served):
    """One-stage's R1-R6 of a PoUW registration, honest and with each field changed, through the switch: Lean's codes are
    Python's, none for the honest record and some for each change."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ell, n = SC.level_of(rec), rec["population"]
    prog, q = SC.PT.population_program(SC.template(K), n), SC.PT.template_instance_query(SC.template(K))
    law, ports = SC.law_of(params, ell), SC.ports(params, ell)
    p = OR.registration_params(prog, q, law, S.SCHEME, ports, [], params["delta"])
    record = OR.Record()

    def codes(r) -> list[str]:
        return record.registration(p, r, lambda: R.check(r, prog, q, law, S.SCHEME, ports, None, (), [], None, None, None).codes)

    assert codes(rec) == []
    root = rec["roots"][0]
    changes = {"protocol": "verity/other/v1", "program": "00" * 64, "partition": "00" * 64, "query": "00" * 64,
               "population": n + 1, "law": {**law, "k": law["k"] + 1}, "scheme": "frame-v3/hm96", "leaf_layer": "x",
               "anchors": [{"x": 1}], "roots": [{**root, "domain": "00" * 64}, *rec["roots"][1:]]}
    got = {k: codes({**rec, k: v}) for k, v in changes.items()}
    got["leaves"] = codes({**rec, "roots": [{**root, "leaves": root["leaves"] + 1}, *rec["roots"][1:]]})
    got["one-root"] = codes({**rec, "roots": rec["roots"][:1]})
    assert all(got.values()), got
    assert all(e["same"] for e in record.witness) and len(record.witness) == len(got) + 1
    record.close()


def test_the_record_follows_the_switch(monkeypatch):
    """`Record()` takes its side of record from `pouw.verifier.RECORD_VERDICT`, and an explicit `lean` overrides it."""
    assert OR.Record().lean is (OR.V.RECORD_VERDICT == "lean")
    monkeypatch.setattr(OR.V, "RECORD_VERDICT", "python")
    assert OR.Record().lean is False and OR.Record(lean=True).lean is True
    monkeypatch.setattr(OR.V, "RECORD_VERDICT", "lean")
    assert OR.Record().lean is True and OR.Record(lean=False).lean is False


@LEAN
def test_the_public_items_rule_of_a_window_is_leans(served):
    """The service's public-items rule over a window's Spec is Lean's (`Service.checkPublic`): on generated registrations
    (keys added and dropped, roots renamed, repeated, dropped or given fields, the scheme, query, leaf layer, anchors and
    registered values changed, several at once) `service.check_public` gives exactly its verdict, codes and detail, none
    for the honest record and some for many of the changed ones."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ell, n = SC.level_of(rec), rec["population"]
    spec = SC.spec_of(params, ell)
    record = OR.Record(lean=True)

    def verdict(r: dict) -> dict:
        return record.public(OR.public_params(spec, r, S.SCHEME), lambda: S.check_public(spec, r).to_json())

    assert verdict(rec) == {"ok": True, "codes": [], "detail": {}}

    def mutate(rng: random.Random, r: dict) -> None:
        roots = r.get("roots")
        match rng.randrange(12):
            case 0:
                r[rng.choice(("tiles", "registered", "values", "extra"))] = rng.choice(([2, 3], [], {"x": 1}, None))
            case 1:
                r.pop(rng.choice(sorted(r)), None)
            case 2 if roots:
                roots[rng.randrange(len(roots))]["port"] = rng.choice(("W", SC.PORT, "X", "values", 7))
            case 3 if roots:
                roots.append(dict(rng.choice(roots)))
            case 4 if roots:
                roots.pop(rng.randrange(len(roots)))
            case 5 if roots:
                e = roots[rng.randrange(len(roots))]
                e[rng.choice(("leaves", "domain", "root", "extra", "value"))] = rng.choice((1, "00" * 64, None, "x"))
            case 6 if roots:
                roots[rng.randrange(len(roots))].pop(rng.choice(("leaves", "domain", "root", "port")), None)
            case 7:
                r["scheme"] = rng.choice(("frame-v3/hm96", S.SCHEME + "x", None))
            case 8:
                r["query"] = rng.choice(("00" * 64, None, r.get("query")))
            case 9:
                r["leaf_layer"] = rng.choice(("x", "00" * 64, None))
            case 10:
                r["anchors"] = rng.choice(([{"x": 1}], [], None))
            case _:
                r["population"] = rng.choice((n, n + 1, 0))

    refused = 0
    for seed in range(300):
        rng = random.Random(seed)
        r = json.loads(json.dumps(rec))
        for _ in range(rng.choice((1, 1, 2, 3))):
            mutate(rng, r)
        try:
            refused += not verdict(r).get("ok")
        except OR.Disagree as e:
            raise AssertionError(f"seed {seed}: Python {e.python} where the Lean, of record, gives {e.lean}") from None
    assert refused >= 100, refused
    assert all(e["same"] for e in record.witness)
    record.close()


@LEAN
def test_a_disagreement_rejects_the_session(served, monkeypatch):
    """A Python reference that disagrees with Lean, at each check, refuses the session as ``lean-python-disagree: <check>``
    with both verdicts in the witness; nothing is retried, and a refused draw is never redrawn."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec = _register(params, public)

    def disagrees(check: str, patch, step) -> S.Refused:
        ch = _challenger(params, coins=_coins())
        with monkeypatch.context() as m:
            patch(m)
            with pytest.raises(S.Refused) as e:
                step(ch)
        assert _codes(e) == {f"{OR.DISAGREE}: {check}"}
        bad = [x for x in ch.witness["honest"] if not x["same"]]
        assert [x["check"] for x in bad] == [check] and bad[0]["lean"] != bad[0]["python"]
        return ch

    disagrees("registered", lambda m: m.setattr(OR, "py_registered", lambda x: {"reason": "forced"}),
              lambda ch: _held(ch, "honest", rec))
    disagrees("pass", lambda m: m.setattr(OR, "py_pass", lambda x: {"how": "forced"}), lambda ch: _held(ch, "honest", rec))
    disagrees("registration", lambda m: m.setattr(SC.R, "check", lambda *a: SC.Verdict(False, ("R1-law",), {})),
              lambda ch: _held(ch, "honest", rec))
    disagrees("public", lambda m: m.setattr(SC.S, "check_public", lambda *a: SC.Verdict(False, ("keys",), {})),
              lambda ch: _held(ch, "honest", rec))
    ch = disagrees("draw", lambda m: m.setattr(OR, "py_draw", lambda x: {"refused": ["draw", "forced"]}),
                   lambda ch: (_held(ch, "honest", rec), ch.select("honest")))
    with pytest.raises(S.Refused) as e:
        ch.select("honest")
    assert _codes(e) == {"coin"} and "already open" in e.value.verdict.detail["coin"]
    case = CC.case
    ch = disagrees("answer", lambda m: m.setattr(OR.CC, "case", lambda j: {**case(j), "reason": "forced"}),
                   lambda ch: _through(ch, rec))
    with pytest.raises(S.Refused) as e:
        ch.outcome("honest", [_unit(t) for t in ch.draws["honest"].units])
    assert _codes(e) == {"final"}
    t = ch.draws["honest"].units[0]
    monkeypatch.setattr(OR, "py_opening", lambda x: "forced")
    record = OR.Record()
    assert SC.check(params, rec, ch.draws["honest"].draw, t, _unit_pub(advice, priv, t), SC.opening(advice, priv, params, t),
                    record=record) == f"{OR.DISAGREE}: opening"
    assert record.witness[0]["python"] == "forced" and record.witness[0]["lean"] is None
    record.close()
    monkeypatch.undo()
    op, served_domain = SC.opening(advice, priv, params, t), SC.window_domain
    monkeypatch.setattr(SC, "window_domain", lambda params, ell: served_domain({**params, "context": "other"}, ell))
    record = OR.Record()
    assert SC.check(params, rec, ch.draws["honest"].draw, t, _unit_pub(advice, priv, t), op, record=record) == \
        f"{OR.DISAGREE}: prelude"
    assert [(e["check"], e["same"]) for e in record.witness] == [("opening", True), ("prelude", False)]
    record.close()


def test_lean_and_rust_verdicts_are_read_from_their_outputs(tmp_path):
    """Lean's verdict is of record and names, for each session it printed a VERDICT of, that session record's SHA-512 (the
    bytes `deliver` hashes); it accepts only when every session's does.  Rust's is beside it, not of record."""
    a, b = tmp_path / "s-a", tmp_path / "s-b"
    for d in (a, b):
        d.mkdir()
        (d / "session.json").write_text(json.dumps({"session": d.name}))
    sha = {d: hashlib.sha512((d / "session.json").read_bytes()).hexdigest() for d in (a, b)}
    line = {"session": f"{a}/", "accepted": False, "why": "ring-switch claim 2 mismatch"}
    (tmp_path / "lean.out").write_text(f"setup: 3 ms\nVERDICT {json.dumps(line)}\n")
    v = SC.lean_verdict(tmp_path / "lean.out")
    assert v["of_record"] and v["accepted"] is False and "ring-switch" in v["why"] and len(v["digest"]) == 128
    assert v["sessions"] == [sha[a]] and v["digest"] == hashlib.sha512(SC.canonical_json(line)).hexdigest()
    two = [{"session": str(a), "accepted": True, "why": ""}, {"session": str(b), "accepted": True, "why": ""}]
    (tmp_path / "two.out").write_text("".join(f"VERDICT {json.dumps(x)}\n" for x in two))
    assert SC.lean_verdict(tmp_path / "two.out")["sessions"] == [sha[a], sha[b]] and SC.lean_verdict(tmp_path / "two.out")["accepted"]
    (tmp_path / "two.out").write_text("".join(f"VERDICT {json.dumps(x)}\n" for x in (two[0], {**two[1], "accepted": False, "why": "w"})))
    assert SC.lean_verdict(tmp_path / "two.out")["accepted"] is False and SC.lean_verdict(tmp_path / "two.out")["why"] == "w"
    (tmp_path / "gone.out").write_text(f"VERDICT {json.dumps({**line, 'session': str(tmp_path / 'gone')})}\n")
    (tmp_path / "bare.out").write_text('VERDICT {"accepted":true,"why":""}\n')
    assert SC.lean_verdict(tmp_path / "gone.out")["sessions"] == [None] == SC.lean_verdict(tmp_path / "bare.out")["sessions"]
    (tmp_path / "none.out").write_text("")
    assert SC.lean_verdict(tmp_path / "none.out")["accepted"] is None and SC.lean_verdict(tmp_path / "none.out")["sessions"] == []
    (tmp_path / "prove.out").write_text('LIVE\t{"verdict": {"accepted": true, "why": ""}}\n')
    assert SC.rust_verdict(tmp_path / "prove.out")["accepted"] is True and not SC.rust_verdict(tmp_path / "prove.out")["of_record"]


@LEAN
def test_the_cli_runs_the_window_through_the_service(served, tmp_path, monkeypatch, capsys):
    """The job's steps through `main`: register (and the refused extra item), then one challenger process's requests,
    as the job sends them a line at a time: its coin commitment, its salt coin opened after the pass's weights' root (as
    serving asks for it), the refused edited record and the passes whose salt coin is a control's, the honest one,
    Lean's draw (a stand-in, on the draw coin's key) and the refused second select, the opening checks, the delivery, the
    outcome and the refused second one.  `replay` re-runs them from the recorded coins byte for byte, and refuses a
    tampered record."""
    *_, words = served
    params, public, advice, priv = _window(words)
    coins, raw, man = PASSES[params["context"]]
    (tmp_path / "params.json").write_text(json.dumps(params))
    (tmp_path / "manifest.json").write_bytes(raw)
    opening = CoinOpening.from_json(man["salt_opening"])
    c = tmp_path / "commit"
    c.mkdir()
    (c / "commitment.json").write_text(json.dumps(public))
    p, t = str(tmp_path / "params.json"), str(tmp_path)
    assert SC.main(["register", str(c), p, f"{t}/registration.json"]) == 0
    assert SC.main(["register", str(c), p, f"{t}/x.json", "--extra", "shapes=[[16,128,4]]", "--expect-refused"]) == 0
    assert SC.main(["edit", f"{t}/registration.json", f"{t}/extra.json", 'window.public.calls=2']) == 0
    (tmp_path / "coins.json").write_text(json.dumps(coins.commitment.to_json()))
    (tmp_path / "opening.json").write_text(json.dumps(opening.to_json()))
    (tmp_path / "resalt.json").write_text(json.dumps({**man, "salt": "00" * 32}))
    for case in SC.CASES:
        src = "resalt.json" if case == "resalted" else "manifest.json"
        assert SC.main(["coin-variant", f"{t}/{src}", f"{t}/coins.json", f"{t}/opening.json", f"{t}/coin-{case}.json", case]) == 0
    lean = tmp_path / "lean"
    lean.write_text(f"#!{sys.executable}\nimport json, sys\nfrom verity.core.protocols.pouw.core_check import stream_subset\n"
                    "a = sys.argv\np, k = int(a[a.index('--population') + 1]), int(a[a.index('--k') + 1])\n"
                    "w = (p.bit_length() + 64 + 7) // 8\nsrc = bytes.fromhex(a[a.index('--stream') + 1])\n"
                    "sys.exit(1) if len(src) != 2 * k * w + 64 else None\n"
                    "print('DRAW ' + json.dumps({'law': 'subset', 'population': p, 'k': k, 'units': stream_subset(k, p, src)}))\n")
    lean.chmod(lean.stat().st_mode | stat.S_IEXEC)
    rec = json.loads((tmp_path / "registration.json").read_text())
    dry = _challenger(params, coins=_coins())        # the process's coins: its draw is the one the job opens
    _held(dry, "honest", rec)
    drawn = dry.select("honest").units
    rows = []
    for u in drawn:
        (tmp_path / f"pub{u}.bin").write_bytes(_unit_pub(advice, priv, u))
        (tmp_path / f"op{u}.json").write_text(json.dumps(SC.opening(advice, priv, params, u)))
        assert SC.check(params, rec, {"units": drawn}, u, _unit_pub(advice, priv, u), SC.opening(advice, priv, params, u)) is None
        (tmp_path / f"open{u}.json").write_text(json.dumps({"of_record": False, "accepted": True, "why": ""}))
        (tmp_path / f"s{u}").mkdir()
        (tmp_path / f"s{u}" / "session.json").write_text(json.dumps({"unit": u}))
        (tmp_path / f"lean{u}.out").write_text(f"VERDICT {json.dumps({'session': f'{t}/s{u}/', 'accepted': True, 'why': ''})}\n")
        (tmp_path / f"stage{u}.json").write_text(json.dumps({"definition": "d", "circuit_sha512": "1", "class_sha512": "2",
                                                             "ands": 1, "k_log": 1, "binding": "hash scratch"}))
        rows.append({"unit": u, "stage": f"{t}/stage{u}.json", "pub": f"{t}/pub{u}.bin", "lean": f"{t}/lean{u}.out",
                     "opening": f"{t}/open{u}.json", "session": f"{t}/s{u}/session.json"})
    (tmp_path / "units.json").write_text(json.dumps(rows))
    honest = {"registration": f"{t}/registration.json", "manifest": f"{t}/manifest.json"}
    reqs = [{"op": "coins"},
            {"op": "salt", "after": opening.after.hex(), "out": f"{t}/salt.json"},
            {"op": "register", "name": "extra", "registration": f"{t}/extra.json", "manifest": f"{t}/manifest.json"},
            *({"op": "register", "name": case, "registration": f"{t}/registration.json", "manifest": f"{t}/coin-{case}.json"}
              for case in SC.CASES),
            {"op": "register", "name": "honest", **honest, "out": f"{t}/receipt.json"},
            {"op": "select", "name": "honest", "out": f"{t}/draw.json"},
            {"op": "select", "name": "honest", "out": f"{t}/draw2.json"},
            {"op": "outcome", "name": "honest", "units": f"{t}/units.json"},
            {"op": "deliver", "name": "honest", "units": f"{t}/units.json"},
            {"op": "outcome", "name": "honest", "units": f"{t}/units.json", "out": f"{t}/outcome"},
            {"op": "outcome", "name": "honest", "units": f"{t}/units.json", "out": f"{t}/outcome2"},
            {"op": "quit"}, {"op": "select", "name": "honest"}]
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(r) + "\n" for r in reqs)))
    monkeypatch.setattr(SC, "Coins", lambda epoch, entropy: _coins())   # the process's coins are the ones that salted the pass
    capsys.readouterr()
    assert SC.main(["challenger", p, f"{t}/log", "--lean", str(lean)]) == 0
    replies = capsys.readouterr().out.splitlines()
    assert [x.split(" ", 1)[0] for x in replies] == ["COINS", "OPENED", "REFUSED", *["REFUSED"] * len(SC.CASES), "REGISTERED",
                                                     "DRAWN", "REFUSED", "PENDING", "DELIVERED", "ACCEPT", "REFUSED"]
    assert all('"coin"' in r for r in replies[3:3 + len(SC.CASES)]) and "salt is not its coin's" in replies[2 + len(SC.CASES)]
    log = tmp_path / "log"
    assert json.loads((log / "coins.json").read_text()) == coins.commitment.to_json()
    assert json.loads((tmp_path / "salt.json").read_text()) == man["salt_opening"]
    salt, draw = (json.loads(x) for x in (log / "openings.jsonl").read_text().splitlines())
    assert salt == man["salt_opening"] and draw["label"] == "draw" and draw["after"] == R.digest(rec)
    d = json.loads((tmp_path / "draw.json").read_text())
    assert d["units"] == drawn and json.loads((tmp_path / "receipt.json").read_text())["registration"] == R.digest(rec)
    assert not (tmp_path / "draw2.json").exists() and not (tmp_path / "outcome2").exists()
    o = json.loads((tmp_path / "outcome" / "outcome.json").read_text())
    assert o["accepted"] and set(o["public"]) == {x[0] for x in Wd.PUBLIC} and o["public"]["drawn"] == drawn
    assert o["coins"] == {"commitment": coins.commitment.to_json(), "pass": "coin", "draw": draw}
    assert Fraction(o["public"]["sampling"]["delta"]) == Fraction(1, 10)
    audit = json.loads((tmp_path / "outcome" / "audit.json").read_text())
    assert audit["sessions"] == [hashlib.sha512((tmp_path / f"s{u}" / "session.json").read_bytes()).hexdigest() for u in drawn]
    assert audit["audit"]["verdicts"]["lean"]["sessions"] == audit["sessions"]
    assert len((log / "requests.jsonl").read_text().splitlines()) == len(reqs) - 2
    witness = [json.loads(x) for x in (log / "witness.jsonl").read_text().splitlines()]
    assert [w["reply"] for w in witness] == [x.split(" ", 1)[0] for x in replies]
    noted = [e for w in witness for e in w["of_record"]]
    assert {e["check"] for e in noted} >= {"registered", "registration", "pass", "draw", "answer"}
    assert all(e["same"] and e["of_record"] == "lean" for e in noted)
    accepted = [e for w in witness if w["reply"] == "ACCEPT" for e in w["of_record"]]
    assert accepted and all(e in o["of_record"] for e in accepted)

    assert SC.main(["replay", p, str(log), f"{t}/replay", "--lean", str(lean)]) == 0
    assert capsys.readouterr().out.startswith("SAME ")
    assert (tmp_path / "replay").is_dir() and not SC.replay(params, log, tmp_path / "replay2", SC.python_sampler)  # the twin
    forged = {**draw, "coin": ("00" if draw["coin"][:2] != "00" else "01") + draw["coin"][2:]}
    (log / "openings.jsonl").write_text(json.dumps(salt) + "\n" + json.dumps(forged) + "\n")
    assert SC.main(["replay", p, str(log), f"{t}/replay3", "--lean", str(lean)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("DIFFERENT ") and "does not open against" in out


def _staged(d: Path, rec: dict, proof: int) -> Path:
    s = d / "sessions" / "l0000-1"
    s.mkdir(parents=True)
    (d / "stage.json").write_text(json.dumps({"circuit_sha512": "11", "class_sha512": "22", "ands": 7, "k_log": 3}))
    (d / "pub.bin").write_bytes(b"h\n" + bytes(384))
    (s / "session.json").write_text(json.dumps(rec))
    (s / "verifier.json").write_text(json.dumps({"host": str(d)}))
    (s / "t.rep0.bin").write_bytes(bytes(proof))
    return d


def test_shape_compares_what_the_prover_sends(tmp_path):
    """Two units' proofs have one shape when the circuit, the public file, each proof file's size and the session record's
    skeleton agree; the verifier's own verdict and timings, and its own files, are not the prover's."""
    rec = {"streams": [{"rounds": [{"k": 0, "msg_len": 96, "coins": ["ab" * 16]}]}], "verdict": {"verify_s": 1.5}, "g": 0.25}
    a = _staged(tmp_path / "a", rec, 100)
    b = _staged(tmp_path / "b", {**rec, "verdict": {"verify_s": 12.25, "why": "x"}, "g": 3.125}, 100)
    assert SC.shape(a, b)["same"]
    c = _staged(tmp_path / "c", {**rec, "streams": [{"rounds": [{"k": 0, "msg_len": 97, "coins": ["ab" * 16]}]}]}, 100)
    v = SC.shape(a, c)
    assert not v["same"] and any("msg_len" in x for x in v["diffs"])
    assert not SC.shape(a, _staged(tmp_path / "d", rec, 101))["same"]


@pytest.mark.slow
def test_a_staged_unit_opens_under_the_window_root(served, tmp_path):
    """stage_tile.py stages a call's tile and a filler tile of a committed window as `Pc8TileSide` units, each in its own
    process, on the committer's side rows and salts: each is over the circuit the verifier pins for the template
    (`core_check.TILES`, Lean's `PouwCheck.tiles`), and each public file's rows are the side trees' leaves, so the opening
    accepts.  Nothing cheaper catches a unit whose committed rows are not the side rows (the index field dropped or moved),
    or a change to the template or to C-Flock's lowering that moves the circuit off its pin, which only staging shows;
    about a minute at K = 128."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec = _register(params, public)
    (tmp_path / "advice.json").write_text(json.dumps(advice))
    (tmp_path / "params.json").write_text(json.dumps(params))
    SC.save(tmp_path / "prover.npz", priv)
    stage = Path(__file__).resolve().parents[1] / "served_zk" / "stage_tile.py"
    draw = {"law": "subset", "population": 26, "k": 2, "units": [7, 25]}
    runs = {t: subprocess.Popen([sys.executable, str(stage), str(tmp_path), str(tmp_path / "params.json"), str(t),
                                 str(tmp_path / f"u{t}")], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for t in draw["units"]}
    for t, p in runs.items():
        _, err = p.communicate()
        assert p.returncode == 0, f"unit {t}: {err[-2000:]}"
        out = tmp_path / f"u{t}"
        st = json.loads((out / "stage.json").read_text())
        assert st["definition"] == SC.template(K).id and st["row_schemas"][:2] == [
            [L.ROW_SCHEMA["fp8-pi", SC.SIDES[s]], 16 * L.ROW_WORDS["fp8-pi"](K)] for s in "ab"]
        assert st["circuit_sha512"] == CC.TILES[st["definition"]].circuit, (
            f"unit {t}: {st['definition']} builds circuit {st['circuit_sha512']}, not the one the verifier pins: a change "
            "to the template or to C-Flock's lowering lands its new digest in PouwCheck/Tile.lean and core_check.TILES")
        op = json.loads((out / "opening.json").read_text())
        assert SC.check(params, rec, draw, t, (out / "pub.bin").read_bytes(), op) is None, t
    assert json.loads((tmp_path / "u25" / "stage.json").read_text())["kind"] == "filler"


def _recorded(words, root: Path, units=(1, 13)) -> Path:
    """A committed window laid out as a served_zk run records it (`job.sh`): the request's advice, params and
    registration, the draw, and each drawn unit's and the filler unit's opening."""
    params, public, advice, priv = _window(words)
    rec = _register(params, public)
    out = root / "r-window" / "out"
    (out / "request").mkdir(parents=True)
    for name, x in (("advice", advice), ("params", params), ("registration", rec)):
        (out / "request" / f"{name}.json").write_text(json.dumps(x))
    filler = next(e.lo for e in SC.entries_of(advice) if e.kind == "filler")
    (out / "draw.json").write_text(json.dumps({"units": list(units)}))
    (out / "summary.json").write_text(json.dumps({"units": list(units), "filler_unit": filler}))
    for t, d in [*((t, f"u{t}") for t in units), (filler, "filler")]:
        (out / d).mkdir()
        (out / d / "opening.json").write_text(json.dumps(SC.opening(advice, priv, params, t)))
    return root / "r-window"


def test_replay_window_holds_a_recorded_window_and_refuses_its_breaks(served, tmp_path):
    """`replay_window` on a recorded window of entry rows: its statement, rows, records, root, draws and registered items
    are the recorded values and every negative control is refused; a tampered salt fails the window's root."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
    import replay_window as RW
    *_, words = served
    run = _recorded(words, tmp_path)
    rep = RW.replay(RW.Sides(lean=False), [run])
    t = rep["totals"]
    assert RW.passed(rep) and (t["entry_rows"], t["draws"]) == (1, 3), t
    w = rep["runs"]["r-window"]["request"]
    assert {c["control"] for c in w["controls"]} >= {f"window-{k}" for k in RW.REASONS} | {"root-salt", "tile-at-past"}
    adv = run / "out" / "request" / "advice.json"
    a = json.loads(adv.read_text())
    a["entry_salts"][1] = "00" * (len(a["entry_salts"][1]) // 2)
    adv.write_text(json.dumps(a))
    bad = RW.replay(RW.Sides(lean=False), [run])
    assert not RW.passed(bad) and [c["check"] for c in bad["runs"]["r-window"]["request"]["checks"] if not c["ok"]] == ["root-1"]


@LEAN
def test_replay_window_lean_is_the_python(served, tmp_path):
    """Every input `replay_window` makes of a recorded window, the controls' included, gives Lean's
    `pouw-window-difftest` the outcome Python gives."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
    import replay_window as RW
    *_, words = served
    sides = RW.Sides()
    try:
        rep = RW.replay(sides, [_recorded(words, tmp_path)])
    finally:
        sides.close()
    t = rep["totals"]
    assert RW.passed(rep) and t["compared"] >= 40 and t["lean_same"] == t["compared"], rep["apart"][:3]


#: a per-unit check of a drawn tile: open-u<t> or lean-u<t> for a unit index t
_UNIT_CHECK = re.compile(r"^(open|lean)-u[0-9]+$")


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("SZ_PARITY_E2E"),
                    reason="SZ_PARITY_E2E unset: the parity needs the flock/Lean builds and a served window on the node")
def test_fan_out_verdicts_match_the_serial_loop(tmp_path):
    """served_zk/job.sh's parallel fan-out over the drawn units (SZ_JOBS=4) gives the same verdicts as the serial loop
    (SZ_JOBS=1).  The two runs draw different unit indices -- each run's commitment uses fresh salts, so its registration
    root, and the draw coin opened after it, differ -- so the parity is draw-independent: the structural and control checks
    (coins, receive-*, select, open/lean-filler, open/lean-foreign-*, lean-st-*, outcome-*, deliver-*, shape-filler, replay,
    ...) must have byte-identical verdicts, and both runs must draw the same number of units (pinned ε, δ, calls fix K'),
    every one opening and Lean-verifying ACCEPT.  Opt in with SZ_PARITY_E2E on a node that has the served window
    ($WORK/passes/$SERVED_RUN) and the flock/Lean builds ($HZ_CACHE); it runs the whole pipeline twice over a tiny window
    (ε=1/8, δ=1/2, K'=6), a few minutes each.  The realistic-parameter audits run this fan-out at scale."""
    root = Path(__file__).resolve().parents[3]
    base = {**os.environ, "HZ_JOB": "benchmarks/pouw/served_zk/job.sh", "SZ_EPS": "1/8", "SZ_DELTA": "1/2",
            "SZ_CALL_LIST": os.environ.get("SZ_CALL_LIST", "77 225"), "SERVED_RUN": os.environ["SERVED_RUN"],
            "HZ_MEM_GB": os.environ.get("HZ_MEM_GB", "120")}
    unit, rest = {}, {}
    for jobs in ("1", "4"):
        rundir = tmp_path / f"jobs{jobs}"
        (rundir / "out").mkdir(parents=True)
        r = subprocess.run(["bash", "benchmarks/pouw/hidden_zk/job.sh"], cwd=root, text=True, capture_output=True,
                           env={**base, "SZ_JOBS": jobs, "RESEARCH_RUN_DIR": str(rundir)})
        summary = rundir / "out" / "summary.json"
        assert r.returncode == 0 and summary.is_file(), f"SZ_JOBS={jobs} failed:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"
        checks = {k: v["verdict"] for k, v in json.loads(summary.read_text())["checks"].items()}
        unit[jobs] = {k: v for k, v in checks.items() if _UNIT_CHECK.match(k)}
        rest[jobs] = {k: v for k, v in checks.items() if not _UNIT_CHECK.match(k)}
    assert rest["1"] == rest["4"], f"the fan-out's structural verdicts differ from the serial loop's: {rest}"
    assert len(unit["1"]) == len(unit["4"]) > 0, f"the fan-out drew a different number of units: {unit}"
    for jobs in ("1", "4"):
        assert all(v == "ACCEPT" for v in unit[jobs].values()), f"SZ_JOBS={jobs} has a non-ACCEPT drawn unit: {unit[jobs]}"
