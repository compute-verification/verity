"""served_h1's same-GEMM gate (`served_h1/same_gemm.py`) on small CPU windows: two steps of two K = N = 64 linears over a
64-slot batch with 20 served rows, each call's y feeding the next call's A rows, under a real live challenger
(`pouw.window.live_challenger`, `LiveRecord`, an injected clock).

A beside-path window, today's served_h1 (y from `torch._int_mm` in the call, receipts in a separate pass after the step's
round), is refused: the trace names the served y's GEMM, and the order refuses every call but a step's last.  The same
beside path under a round per call keeps the order, so only the trace refuses it.  A fused window (one launch per call
writes y, z and the H1 receipts, which are `receipts.gated_receipts`' bit for bit; a round per call, z committed after its
coin) is accepted.  A fused window that runs a second GEMM in the call, inside the fused launch, or outside every call, a
GEMM as elementwise ops inside the launch, a write after it, a returned y that isn't the launch's buffer, or receipts from
a separate pass is refused, each with its reason.  Seeded coins, a non-live commitment, a release log out of order and a
round tree that doesn't recompute refuse the window.  A drawn unit's z rows open against its z table, its cells are the
unit's reference z, and its served rows are dequant(z); a moved served row or z cell is refused."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.sampled_proofs import challenger as Ch

HERE = Path(__file__).resolve().parents[1] / "served_h1"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"served_h1_{name}", HERE / f"{name}.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RC, CM, SG, SV = _load("receipts"), _load("commit"), _load("same_gemm"), _load("serve")
WINDOW, T, K, M, STEPS, LINEARS = "test-window", 64, 64, 20, 2, 2
N = K
KEY = bytes(range(64))
ZERO = "00" * 64
KERNEL = "verity_h1_fused_s8"
WORDS = 2 * RC.layout(K).sets
UNITS = (T // 32) * (N // 32)


class _Model:
    def __init__(self, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.b = [torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8) for _ in range(LINEARS)]
        self.bt = [b.t().contiguous() for b in self.b]
        self.s_w = [torch.rand(N, generator=g) * 1e-3 + 1e-4 for _ in range(LINEARS)]
        self.x0 = torch.randn(M, K, generator=g)


def _quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s_a = x.abs().amax(1).clamp(min=1e-8) / 127
    return (x / s_a[:, None]).round().clamp(-127, 127).to(torch.int8), s_a


def _dequant(z: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    return (z.to(torch.float32) * s_a[:, None] * s_w[None, :]).to(torch.bfloat16)


def _bf16_words(f: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _unit_receipts(a: np.ndarray, b: np.ndarray, admit: np.ndarray, salt: bytes, base: int, ra: bytes, rb: bytes) -> np.ndarray:
    """`receipts.gated_reference` on host words, no dispatcher op: the fixture's stand-in for the fused kernel's receipts."""
    out = np.zeros((UNITS, WORDS), dtype=np.uint32)
    for u, (i, j) in enumerate(RC.units(T, N)):
        if admit[32 * i:32 * i + 32].any():
            ai = (a[32 * i:32 * i + 32].astype(np.int64) & 0xFF).tolist()
            bj = (b[32 * j:32 * j + 32].astype(np.int64) & 0xFF).tolist()
            out[u] = HT.reference(ai, bj, salt, base + u, (ra, rb))[1]
    return out


class _Window:
    """One served window: `path` 'beside' (today's) or 'fused', `form` 'step' or 'call' rounds, `secret` a cheat."""

    def __init__(self, path: str, form: str, secret: str | None = None, seed: int = 0) -> None:
        self.path, self.form, self.secret = path, form, secret
        self.model = _Model(seed)
        rounds = STEPS * LINEARS + 1 if form == "call" else STEPS
        rng = random.Random(seed)
        self.lch = Wd.live_challenger(7, rounds, entropy=rng.randbytes)
        self.log = Ch.LiveRecord(self.lch, Wd.ROUND, clock=itertools.count().__next__)
        self.calls: list[dict] = []
        self.data: list[dict] = []
        self.round_log: list[dict] = []
        self.prev = (ZERO, ZERO)
        self.admit = np.arange(T) < M
        self.rec = SG.Recorder("cpu")
        with self.rec:
            self._serve()
        self.ledger = self.rec.ledger()
        if form == "call":
            self._round(STEPS * LINEARS, [ZERO, ZERO, *self.prev])
        self.coins = {"mode": "live", "commitment": self.lch.commitment.to_json(), "rounds": self.round_log,
                      "release_log": list(self.log.log)}
        if form == "call":
            self.coins["round_form"] = SG.ROUND_FORM_CALL
        else:
            self.coins["rule"] = "round s = step s: ..."

    def _round(self, r: int, leaves: list[str]) -> bytes:
        tree = {"kind": "round", "fields": {"window": WINDOW, "round": r, "leaves": "a, admit, z(prev), receipts(prev)"},
                "leaves": leaves}
        root = SG.tree_root(tree)
        opening = self.log.record(root)
        self.round_log.append({"round": r, "root": root.hex(), "tree": tree, "opening": opening.to_json()})
        return HT.live_salt(self.lch.commitment, opening.coin, r, root)

    def _call_meta(self, s: int, w: int, xq: torch.Tensor, s_a: torch.Tensor) -> dict:
        slots = np.zeros((T, K), dtype=np.int8)
        slots[:M] = xq.numpy()
        scales = np.zeros(T, dtype=np.float32)
        scales[:M] = s_a.numpy()
        fields = {"window": WINDOW, "step": s, "linear": w, "module": f"l{w}", "slots": T, "k": K, "schema": CM.SCHEMA}
        root_a = CM.s8_table(CM.domain(fields, T, "a-table"), KEY, f"a/{s}/{w}", slots, scales).root
        c = {"step": s, "linear": w, "module": f"l{w}", "k": K, "n": N, "slots": T, "selected": M,
             "base": s * LINEARS * UNITS + w * UNITS, "root_a": root_a.hex(),
             "root_b": hashlib.sha512(f"b/{w}".encode()).hexdigest(),
             "root_admit": hashlib.sha512(f"admit/{s}/{w}".encode()).hexdigest()}
        self.calls.append(c)
        self.data.append({"slots": slots, "scales": scales})
        return c

    def _serve(self) -> None:
        x = self.model.x0
        for s in range(STEPS):
            for w in range(LINEARS):
                t = s * LINEARS + w
                if self.secret == "outside" and t == 1:
                    torch._int_mm(_quant(x)[0], self.model.bt[w])
                y = self._beside(t, s, w, x) if self.path == "beside" else self._fused(t, s, w, x)
                x = y.to(torch.float32)
            if self.form == "step":
                cs = self.calls[-LINEARS:]
                st = SV.step_roots(WINDOW, s, cs)
                opening = self.log.record(st["root"])
                salt = HT.live_salt(self.lch.commitment, opening.coin, s, st["root"])
                self.round_log.append({"round": s, "root": st["root"].hex(), "fields": st["fields"],
                                       "a_step": st["a_step"].hex(), "admit_step": st["admit_step"].hex(),
                                       "opening": opening.to_json()})
                for i in range(len(self.calls) - LINEARS, len(self.calls)):
                    self._beside_receipts(i, salt)

    def _beside(self, t: int, s: int, w: int, x: torch.Tensor) -> torch.Tensor:
        """Today's path: the served y from the call's own `_int_mm`, its receipts later, beside it."""
        with self.rec.call(t, key=t):
            xq, s_a = _quant(x)
            y = _dequant(torch._int_mm(xq, self.model.bt[w]), s_a, self.model.s_w[w])
            self.rec.served(t, y, M)
        self._call_meta(s, w, xq, s_a)
        if self.form == "call":
            c = self.calls[-1]
            self._beside_receipts(len(self.calls) - 1, self._round(t, [c["root_a"], c["root_admit"], *self.prev]))
            self.prev = (hashlib.sha512(f"z/{t}".encode()).hexdigest(), hashlib.sha512(f"r/{t}".encode()).hexdigest())
        return y

    def _beside_receipts(self, i: int, salt: bytes) -> None:
        c, d = self.calls[i], self.data[i]
        rc, _ = RC.gated_receipts(RC.Call(torch.from_numpy(d["slots"]), self.model.b[c["linear"]]), self.admit, salt,
                                  c["base"], bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"]))
        c["salt"] = salt.hex()
        d["receipts"] = rc

    def _fused(self, t: int, s: int, w: int, x: torch.Tensor) -> torch.Tensor:
        """The fused path: the call's A table and admit root in round t's root, its coin, then one launch that writes the
        call's y, z and receipts; the call returns a view of that y."""
        b, b_np, s_w = self.model.b[w], self.model.b[w].numpy(), self.model.s_w[w]
        with self.rec.call(t, key=t):
            xq, s_a = _quant(x)
            if self.secret == "beside":
                torch._int_mm(xq, self.model.bt[w])
            if self.secret == "host-y":
                zh = (xq.numpy().astype(np.int64) @ b_np.astype(np.int64).T).astype(np.float32)
                prod_h = (zh * s_a.numpy()[:, None]).astype(np.float32) * s_w.numpy()[None, :]
                fake = torch.from_numpy(_bf16_words(prod_h).view(np.int16)).view(torch.bfloat16)
            c = self._call_meta(s, w, xq, s_a)
            d = self.data[-1]
            salt = self._round(t, [c["root_a"], c["root_admit"], *self.prev])
            y = torch.empty(T, N, dtype=torch.bfloat16)
            z = torch.empty(T, N, dtype=torch.int32)
            r = torch.empty(UNITS, WORDS, dtype=torch.int32)
            with self.rec.fused(t, [KERNEL]) as f:
                if self.secret == "inside":
                    torch._int_mm(xq, self.model.bt[w])
                if self.secret == "elementwise":
                    (xq.to(torch.int32)[:, None, :] * b.to(torch.int32)[None, :, :]).sum(-1)
                zz = (d["slots"].astype(np.int64) @ b_np.astype(np.int64).T).astype(np.int32)
                prod = (zz.astype(np.float32) * d["scales"][:, None]).astype(np.float32) * s_w.numpy()[None, :]
                y.view(torch.int16).numpy()[:] = _bf16_words(prod).view(np.int16)
                z.numpy()[:] = zz
                rr = _unit_receipts(d["slots"], b_np, self.admit, salt, c["base"], bytes.fromhex(c["root_a"]),
                                    bytes.fromhex(c["root_b"]))
                r.numpy()[:] = rr.view(np.int32)
                f.outs(rows=M, y=y, z=z, receipts=r)
            if self.secret == "late":
                y.mul_(1)
            out = y[:M].clone() if self.secret == "clone" else fake if self.secret == "host-y" else y[:M]
            self.rec.served(t, out, M)
        receipts = r
        if self.secret == "receipts-beside":
            rc, _ = RC.gated_receipts(RC.Call(torch.from_numpy(d["slots"]), b), self.admit, salt, c["base"],
                                      bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"]))
            receipts = torch.from_numpy(rc.view(np.int32))
        self.rec.committed(t, "receipts", receipts)
        self.rec.committed(t, "z", z)
        z_dom = CM.domain({"window": WINDOW, "step": s, "linear": w, "module": f"l{w}", "slots": T, "n": N,
                           "schema": CM.SCHEMA}, T, "z-table")
        table = SG.z_table(z_dom, KEY, f"z/{s}/{w}", zz)
        r_dom = CM.domain({"window": WINDOW, "step": s, "linear": w, "units": UNITS}, UNITS)
        root_r = CM.levels(r_dom, [CM.leaf(r_dom, KEY, f"r/{s}/{w}", u, bytes(row))[0]
                                   for u, row in enumerate(CM.receipt_rows(rr))])[-1][0]
        c.update(salt=salt.hex(), root_z=table.root.hex())
        d.update(receipts=rr, z=zz, y=y[:M].clone(), z_table=table, z_domain=z_dom, salt=salt)
        self.prev = (table.root.hex(), root_r.hex())
        return out

    def verdict(self, coins: dict | None = None, ledger: dict | None | str = "own") -> dict:
        return SG.verdict(WINDOW, self.calls, [c["selected"] for c in self.calls], self.coins if coins is None else coins,
                          self.ledger if ledger == "own" else ledger, keys=list(range(len(self.calls))))


@pytest.fixture(scope="module")
def fused() -> _Window:
    return _Window("fused", "call")


def _reasons(v: dict) -> list[str]:
    return [x for r in v["refused"] for x in r["reasons"]]


def test_beside_window_is_refused():
    win = _Window("beside", "step")
    v = win.verdict()
    assert not v["ok"] and v["window"] == [] and v["gated_counts"] == [0] * len(win.calls)
    assert v["refused_calls"] == v["calls_with_rows"] == STEPS * LINEARS
    assert v["by_check"] == {"order": STEPS * (LINEARS - 1), "trace": STEPS * LINEARS}
    trace = [r["reasons"][-1] for r in v["refused"]]
    assert all("served y from aten::_int_mm call #" in x and x.endswith("receipts from a separate pass") for x in trace)
    assert trace[0].startswith("credit.same_gemm: linear 0 (l0) step 0: served y from aten::_int_mm call #0")
    order = [r for r in v["refused"] if len(r["reasons"]) == 2]
    assert [(r["step"], r["linear"]) for r in order] == [(s, 0) for s in range(STEPS)]
    assert "served y existed before its receipts' noise" in order[0]["reasons"][0]
    assert v["first"] == order[0]["reasons"][0]


def test_beside_path_that_keeps_the_order_is_refused_by_the_trace():
    v = _Window("beside", "call").verdict()
    assert not v["ok"] and v["by_check"] == {"order": 0, "trace": STEPS * LINEARS}
    assert all("receipts from a separate pass" in r["reasons"][0] for r in v["refused"])


def test_record_without_a_ledger_is_refused(fused):
    v = fused.verdict(ledger=None)
    assert not v["ok"] and v["by_check"]["trace"] == STEPS * LINEARS and v["by_check"]["order"] == 0
    assert "the record has no launch ledger entry" in v["first"]


def test_fused_window_is_accepted(fused):
    v = fused.verdict()
    assert v["ok"], v["first"]
    assert v["gated_counts"] == [M] * len(fused.calls) and v["refused"] == [] and v["window"] == []
    for c, d in zip(fused.calls, fused.data):
        rc, _ = RC.gated_receipts(RC.Call(torch.from_numpy(d["slots"]), fused.model.b[c["linear"]]), fused.admit,
                                  d["salt"], c["base"], bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"]))
        assert np.array_equal(rc, d["receipts"])
        sa = torch.from_numpy(d["scales"][:M])
        want = _dequant(torch._int_mm(torch.from_numpy(d["slots"][:M]), fused.model.bt[c["linear"]]), sa,
                        fused.model.s_w[c["linear"]])
        assert torch.equal(d["y"].view(torch.int16), want.view(torch.int16))
    assert fused.ledger["gemm_launches"] == 0 and fused.ledger["outside"] == []


@pytest.mark.parametrize("secret,why", [
    ("beside", "a second GEMM in the call: aten::_int_mm call #0 beside the fused launch"),
    ("inside", "aten::_int_mm call #0 inside the fused launch"),
    ("elementwise", "ran inside the fused launch: it is one raw launch"),
    ("late", "aten::mul_ ran after the fused launch"),
    ("clone", "aten::clone ran after the fused launch"),
    ("host-y", "is not the fused launch's y buffer"),
    ("receipts-beside", "the committed receipts are not the fused launch's receipts buffer"),
])
def test_fused_window_with_a_cheat_is_refused(secret, why):
    v = _Window("fused", "call", secret).verdict()
    assert not v["ok"] and v["gated_counts"] == [0] * STEPS * LINEARS
    assert v["by_check"]["order"] == 0 and v["window"] == []
    assert v["first"].startswith("credit.same_gemm: linear 0 (l0) step 0: ") and why in v["first"], v["first"]


def test_fused_window_with_a_gemm_outside_every_call_is_refused():
    v = _Window("fused", "call", "outside").verdict()
    assert not v["ok"] and v["refused"] == [] and v["gated_counts"] == [0] * STEPS * LINEARS
    assert v["window"] == [f"credit.same_gemm: window: aten::_int_mm call #0 with a linear's (K, N) = ({K}, {N}) ran "
                           "outside every call's range"]


def _seeded(c):
    return {"mode": "seeded", "ok": True}


def _scheme(c):
    c["commitment"] = {**c["commitment"], "scheme": "verity/seeded"}
    return c


def _release_first(c):
    log = c["release_log"]
    log[0], log[1] = log[1], log[0]
    return c


def _tree(c):
    c["rounds"][1]["tree"]["leaves"][0] = ZERO
    return c


@pytest.mark.parametrize("tamper,why", [
    (_seeded, "coins are seeded, not live"),
    (_scheme, "the round coins: the coin commitment's scheme is verity/seeded"),
    (_release_first, "the release log does not have round 0's root, then its coin"),
    (_tree, "round 1's tree does not recompute to its recorded root"),
])
def test_coins_that_are_not_live_refuse_the_window(fused, tamper, why):
    v = fused.verdict(coins=tamper(json.loads(json.dumps(fused.coins))))
    assert not v["ok"] and v["gated_counts"] == [0] * STEPS * LINEARS and v["refused_calls"] == STEPS * LINEARS
    assert len(v["window"]) == 1 and why in v["window"][0], v["window"]


def _tie(win: _Window, i: int, u: int, served: list[str] | None = None, z: np.ndarray | None = None,
         ref: int | None = None) -> str | None:
    c, d = win.calls[i], win.data[i]
    zz = d["z"] if z is None else z
    ri = u // (N // 32)
    ranks = range(32 * ri, 32 * ri + 32)
    rows = SG.z_rows(zz)
    call = RC.Call(torch.from_numpy(d["slots"]), win.model.b[c["linear"]])
    ref_z = RC.reference(call, d["salt"], c["base"], bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"]),
                         u if ref is None else ref)[0]
    served = win.ledger["calls"][i]["served"]["rows"] if served is None else served
    return SG.tie_refusal(c, u, d["z_table"].open(ranks), {r: rows[r][0] for r in ranks}, KEY,
                          f"z/{c['step']}/{c['linear']}", d["z_domain"].domain_id, T, N, ref_z, d["scales"],
                          win.model.s_w[c["linear"]].numpy(), served)


def test_tie_of_a_drawn_unit(fused):
    assert all(_tie(fused, i, u) is None for i in (0, 3) for u in (0, 1, 2))
    served = list(fused.ledger["calls"][2]["served"]["rows"])
    served[5] = hashlib.sha512(b"another y row").hexdigest()
    assert _tie(fused, 2, 1, served=served).endswith("unit 1's served y rows are not dequant of its committed z")
    moved = fused.data[2]["z"].copy()
    moved[5, 40] += 1
    assert _tie(fused, 2, 1, z=moved).endswith("unit 1's z rows don't open against its z table")
    assert _tie(fused, 2, 1, ref=0).endswith("unit 1's committed z is not its reference z")


def test_cli_refuses_the_beside_record(tmp_path):
    win = _Window("beside", "step")
    calls = [{**c, "tap": i} for i, c in enumerate(win.calls)]
    manifest, ledger = tmp_path / "manifest.json", tmp_path / "ledger.json"
    manifest.write_text(json.dumps({"window": WINDOW, "calls": calls, "coins": win.coins}))
    ledger.write_text(json.dumps(win.ledger))
    for args, why in (([f"--ledger={ledger}"], "receipts from a separate pass"), ([], "has no launch ledger entry")):
        out = tmp_path / "verdict.json"
        assert SG.main(["--manifest", str(manifest), "--out", str(out), *args]) == 1
        v = json.loads(out.read_text())
        assert v["refused_calls"] == STEPS * LINEARS and v["by_check"] == {"order": STEPS, "trace": STEPS * LINEARS}
        assert all(why in r["reasons"][-1] for r in v["refused"])
