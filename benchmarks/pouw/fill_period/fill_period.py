"""One filler period end to end on a development gate (CMP-172): the fill period ``<name>:fill`` opened on the auditor's
ledger, one filler window under its live coins, and `fill._period_outcomes`' verdict on it, with the verifier of record a
stand-in.

* **The circuit** is the filler window's Program (`program`): `h1window.program` over the torch loop's two linears
  (`LOOP_LINEARS`, K = n = 128, its checkpoint role `ROLE`) at one A block each, slot M = R, its units J10-R tiles
  (`h1tile.PouwH1Tile5`) of `TILE`, so 16 units a call and 32 in all.
* **The gate** is `a5_coins.Local` (`window_salt.local`): `benchmarks/live_challenger/auditor.py`'s verbs on this machine
  under a key made here. It registers the loop's committed checkpoint under the circuit as vy-admin does
  (``model-register``), opens the fill period with the draw K `DRAW_K` and ``--model ROLE`` (so its open signs the
  registry's weights' root and model), reserves the window's session and opens it on `fill.open_root` of that root and
  the session, with one salt coin and one draw coin (`fill.FILL_FAMILIES`), and closes the period once the session ended.
* **The window** (`fill_window`): the salt coin opened after the open root, the Program salt `h1tile.live_salt` of it;
  each unit's A rows the verifier's pad (`fill.pad_codes`) and B rows the checkpoint's, its z and receipt
  `h1tile.j10r_reference` under that salt, its position, its call and blocks and the roots of the pad and weights tables,
  its y the served epilogue (`h1zy.dequant`); each K's pad (`fill.pad_entry`), each call's z/y and receipt tables
  committed under fresh hiding salts, the weights the registration's entries; then the draw coin after
  `fill.draw_root` over the salt and every entry.
* **The verifier of record is a stand-in** (`STAND_IN`): for each drawn unit it opens the unit's z/y and receipt rows
  against their registered roots and recomputes both from the public pad, the checkpoint's B rows and the live salt;
  it accepts the unit only when they match. At the real run it is C-Flock's Lean ``flock-verify`` on each drawn
  statement (`fill.fill_period`).

The record is ``OUT/fill.json``: the fill (``{work, k, delta, eps}``) or the refusal, the drawn units and the stand-in's
verdicts, and two negatives the fill must refuse (``negatives``): a z/y table registered after the draw coin (its root
changed), and a drawn unit the verifier rejects. ``OUT/evidence.json`` is what the fill was checked on, the
`fill._period_outcomes` arguments ``{key, ledger, pins, main, name, fills}``.

    uv run python benchmarks/pouw/fill_period/fill_period.py --out OUT [--coins stub]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import fill as F
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.circuits.codec import canonical_json, encode_program

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "window_salt"))
import window_salt as WS  # noqa: E402

TREE = WS.TREE
LOOP = TREE / "integrations" / "torch_loop"
STUB = TREE / "benchmarks" / "live_challenger" / "tests" / "stub_serve.py"
ROLE = "TORCH_LOOP_MLP_H128_W8A8"
LOOP_LINEARS = (("model.layers.0.mlp.up_proj", 128, 128), ("model.layers.1.mlp.down_proj", 128, 128))
TILE = (8, 16)
PREFIX = "fill/"
DRAW_K = 4
LANE = "compute-accounting"
STAND_IN = ("stand-in verifier of record: each drawn unit's z/y and receipt rows opened at their registered roots and "
            "recomputed from the public pad, the checkpoint's B rows and the live salt (h1tile.j10r_reference)")
#: the pad rows' scale s_a, as f32 bits
PAD_SA = struct.unpack("<I", F.PAD_SCALE)[0]


def program() -> tuple[Any, bytes, dict]:
    """The filler window's Program, its descriptor (canonical JSON without annotations, the bytes the period's circuit is
    the SHA-512 of) and its layout (`credit.layout_of`)."""
    calls = [(name, 0) for name, _k, _n in LOOP_LINEARS]
    prog = HW.program(LOOP_LINEARS, calls, TILE[0], TILE, PREFIX, unit=lambda k: HW.tile_unit(k, TILE, H.PouwH1Tile5))
    desc = canonical_json({k: v for k, v in encode_program(prog).items() if k != "annotations"})
    layout, why = Cr.layout_of(desc)
    if why is not None:
        raise ValueError(f"the filler Program's layout is refused: {why}")
    return prog, desc, layout


def checkpoint(directory: Path) -> None:
    """The loop's committed checkpoint files copied into `directory` (0755, its files 0644, as the registry takes them)."""
    directory.mkdir(mode=0o755, parents=True)
    entry = json.loads((LOOP / "manifests" / "checkpoints-int8.json").read_text())["checkpoints"][0]
    for n in entry["files"]:
        shutil.copyfile(LOOP / "checkpoint" / n, directory / n)
        (directory / n).chmod(0o644)


def b_rows(registry: Any, directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """K = 128's B rows as the registry reads them: each linear's int8 codes in turn ``(256, 128)``, and their f32 s_w
    as u32 bits ``(256,)``."""
    ck = registry.Checkpoint(directory, sorted(p.name for p in directory.iterdir()))
    codes, scales = [], []
    for name, _k, _n in LOOP_LINEARS:
        codes.append(np.asarray(ck.tensor(f"{name}.weight"), np.int8))
        scales.append(np.asarray(ck.tensor(f"{name}.weight_scale"), "<f4").reshape(-1).view("<u4"))
    return np.concatenate(codes), np.concatenate(scales)


def pad_rows(k: int, rows: int) -> np.ndarray:
    """The verifier's pad at K = `k`: `rows` rows of its int8 codes."""
    return np.frombuffer(b"".join(F.pad_codes(k, i) for i in range(rows)), np.int8).reshape(rows, k)


def units_of(layout: dict) -> list[dict]:
    """Each unit of the layout's calls, in the Program's order: its position, its call, the call's first unit, its tile
    (I, J), its pad rows' first row and its B rows' first row."""
    (r, c), m = layout["tile"], layout["m"]
    ks = {name: (k, n) for name, k, n in layout["linears"]}
    b0, at = {}, {}
    for name, k, n in layout["linears"]:
        b0[name], at[k] = at.get(k, 0), at.get(k, 0) + n
    out, u, seen = [], 0, {}
    for call, (lin, _blk) in enumerate(layout["calls"]):
        k, n = ks[lin]
        j = seen[k] = seen.get(k, -1) + 1
        u0 = u
        for ti in range(m // r):
            for tj in range(n // c):
                out.append({"unit": u, "call": call, "u0": u0, "k": k, "tile": (ti, tj), "a": m * j + r * ti,
                            "b": b0[lin] + c * tj})
                u += 1
    return out


def compute(unit: dict, pad: np.ndarray, b: np.ndarray, sw: np.ndarray, salt: bytes, roots: tuple[bytes, bytes],
            layout: dict) -> tuple[list[int], list[int]]:
    """A unit's z/y record and receipt as u16 words (`h1zy.RECORD`: its cells' s32 z then their bf16 y; the receipt's
    u32 words low half first), under the window's Program salt `salt` and its K's pad and weights roots `roots`."""
    (r, c), m = layout["tile"], layout["m"]
    ta = pad[unit["a"]:unit["a"] + r].view(np.uint8).tolist()
    tb = b[unit["b"]:unit["b"] + c].view(np.uint8).tolist()
    blocks = CC.j10r_blocks(m, unit["k"], (r, c), *unit["tile"])
    z, receipt = H.j10r_reference(ta, tb, salt, unit["unit"], roots, unit["u0"], blocks)
    y = [Z.dequant(z[i * c + j], PAD_SA, int(sw[unit["b"] + j])) for i in range(r) for j in range(c)]
    zw = np.asarray([v & 0xFFFFFFFF for v in z], "<u4").view("<u2").tolist()
    return zw + [v & 0xFFFF for v in y], np.asarray(receipt, "<u4").view("<u2").tolist()


def bc_of(words: list[int], salt: bytes) -> bytes:
    return G._INST.commit_string(G._INST.default_key, G.row_digest(words, G.WORD_BITS, G.SCHEMA_V2), salt)


def tables(units: list[dict], words: dict[int, tuple[list[int], list[int]]], program_sha: str, window: str
           ) -> dict[str, G.Registered]:
    """Each call's z/y and receipt tables, a row a unit of the call, committed under fresh hiding salts."""
    out: dict[str, G.Registered] = {}
    for call in sorted({u["call"] for u in units}):
        mine = [u["unit"] for u in units if u["call"] == call]
        for name, at in (("zy", 0), ("receipt", 1)):
            v = f"{PREFIX}{name}@{call}"
            out[v] = G.Registered.commit(program_sha, v, [words[u][at] for u in mine], None, G.WORD_BITS, G.SCHEMA_V2,
                                         window=window)
    return out


def stand_in(drawn: list[int], units: list[dict], committed: dict[str, G.Registered], entries: list[dict],
             pad: np.ndarray, b: np.ndarray, sw: np.ndarray, salt: bytes, roots: tuple[bytes, bytes], layout: dict,
             program_sha: str, window: str) -> list[dict]:
    """Each drawn unit's outcome (`credit._proved`'s form): accepted when its z/y and receipt rows open at the roots the
    window registers and are the ones it computes from the public pad, the checkpoint's B rows and `salt`."""
    by = {e["value"]: e for e in entries}
    out = []
    for u in drawn:
        unit = units[u]
        t = u - unit["u0"]
        want = compute(unit, pad, b, sw, salt, roots, layout)
        ok = True
        for name, words in zip(("zy", "receipt"), want):
            reg, e = committed[f"{PREFIX}{name}@{unit['call']}"], by[f"{PREFIX}{name}@{unit['call']}"]
            dom = G.domain(program_sha, e["value"], e["words"], e["leaves"], G.WORD_BITS, window)
            ok &= dom.domain_id.hex() == e["domain"] and G.verify_read(
                dom, bytes.fromhex(e["root"]), t, reg.bcs[t], reg.path(t), G.SCHEMA_V2) and \
                reg.bcs[t] == bc_of(words, reg.salts[t])
        a, bv = f"{PREFIX}a/{unit['k']}@0", f"{PREFIX}b/{unit['k']}"
        zy = f"{PREFIX}zy@{unit['call']}"
        out.append({"unit": u, "verdict": "accept" if ok else "reject", "program": program_sha,
                    "registered": {"a0": {"value": a, "root": by[a]["root"]}, "b0": {"value": bv, "root": by[bv]["root"]},
                                   "zy": {"value": zy, "root": by[zy]["root"]}}})
    return out


def fill_window(g: Any, A: Any, layout: dict, desc: bytes, weights: list[dict], weights_root: str, pad: np.ndarray,
                b: np.ndarray, sw: np.ndarray, k: int, dest: Path) -> dict:
    """One filler window on gate `g` (module docstring): its evidence (`fill.FILL_KEYS`), with the stand-in's outcomes,
    and what it drew."""
    program_sha = hashlib.sha512(desc).hexdigest()
    W = bytes.fromhex(weights_root)
    address, secret, commitment = g.open(lambda s: F.open_root(W, s).hex(), [list(f) for f in F.FILL_FAMILIES],
                                         scheme=Wd.LIVE_SCHEME)
    r = F.open_root(W, g.session)
    with remote.RemoteChallenger.connect(address, bytes.fromhex(secret)) as client:
        c = client.commitment
        if c.root.hex() != commitment:
            raise Ch.CoinError("the challenger's hello answered another commitment than the gate's open")
        opening = client.open(Wd.SALT, 0, r)
        salt = H.live_salt(c, opening.coin, 0, r)
        pads = [F.pad_entry(program_sha, PREFIX, r.hex(), kk, rows) for kk, rows in F.pad_rows(layout).items()]
        roots = (bytes.fromhex(pads[0]["root"]), bytes.fromhex(weights[0]["root"]))
        units = units_of(layout)
        words = {u["unit"]: compute(u, pad, b, sw, salt, roots, layout) for u in units}
        committed = tables(units, words, program_sha, r.hex())
        entries = [dict(e) for e in weights] + pads + [t.entry() for t in committed.values()]
        receipts = F.entries_root(entries)
        root = F.draw_root(salt, receipts)
        draw = client.open(Wd.DRAW, 0, root)
    directory, _ledger, _pins, key = g.finish(dest)
    d = {"window": {"root": r.hex()}, "salt": salt.hex(), "opening": opening.to_json(), "entries": entries,
         "descriptor": desc.hex(), "session": remote.session_files(directory),
         "draw": {"receipts": receipts.hex(), "root": root.hex(), "coin": draw.to_json()}}
    drawn, why = Cr.drawn(d["draw"]["coin"]["coin"], r.hex(), root.hex(), {"receipts": receipts.hex()}, layout, k)
    if why is not None:
        raise Ch.CoinError(f"the window's draw is refused: {why}")
    d["proofs"] = stand_in(drawn, units, committed, entries, pad, b, sw, salt, roots, layout, program_sha, r.hex())
    return {"evidence": {x: d[x] for x in F.FILL_KEYS}, "drawn": drawn, "key": key, "session": g.session}


def stub(g: Any, log: Path) -> None:
    """Gate `g`'s sessions served under live_challenger's stub of the Lean verifier's coins, each request logged to
    `log`."""
    real = g.A.serve_argv
    g.A.serve_argv = lambda *a, **k: [sys.executable, str(STUB), *real(*a, **k)[2:]]
    os.environ["COINS_LOG"] = str(log)


def run(g: Any, main: list[str], out: Path, name: str = "fill-demo") -> dict:
    """The fill period of served period `name` on gate `g` (`window_salt.local`'s), its record written to
    ``out/fill.json`` and returned."""
    A, host = g.A, g.host
    now = datetime.now(timezone.utc)
    hooks = dict(by=LANE, now=now, own=lambda p: None, seal=lambda p: None,
                 register=lambda args: A.model_register(host, args, pinned=g.pin, now=now),
                 period=lambda args: A.period(host, args, signer=g.signer, pinned=g.pin, now=now, active=lambda s: False))

    def gate(*argv: str) -> dict:
        rc, got = A.gate(list(argv), host, **hooks)
        doc = json.loads(got)
        if rc:
            raise Ch.CoinError(f"the gate refused {argv[0]}: {doc.get('refused')}")
        return doc

    out.mkdir(parents=True, exist_ok=True)
    _prog, desc, layout = program()
    program_sha = hashlib.sha512(desc).hexdigest()
    (out / "fill-program.json").write_bytes(desc)
    ck = out / "checkpoint"
    checkpoint(ck)
    registered = gate("model-register", "--model", ROLE, "--circuit", program_sha, "--descriptor",
                      str(out / "fill-program.json"), "--checkpoint", str(ck))
    line = json.loads(host.models.read_text().splitlines()[-1])
    nm = F.period_name(name)
    opened = gate("period-open", "--name", nm, "--lane", LANE, "--cap", "1", "--circuit", program_sha, "--draw-k",
                  str(DRAW_K), "--model", ROLE)
    pad = pad_rows(128, F.pad_rows(layout)[128])
    b, sw = b_rows(A.registry(), ck)
    w = fill_window(g, A, layout, desc, line["entries"], line["weights_root"], pad, b, sw, DRAW_K, out / "challenger")
    closed = gate("period-close", "--name", nm)
    lines = remote.ledger_lines(host.ledger)
    pins = [x for x in host.pins.read_text().split("\n") if x]
    key = w["key"].to_json()

    def verdict(fills: list[dict]) -> dict:
        got, why = F._period_outcomes(key, lines, pins, main, name, fills, CC.lms_verify)
        return {"fill": got} if why is None else {"refused": list(why)}

    ev = w["evidence"]
    zy0 = f"{PREFIX}zy@0"
    after = hashlib.sha512(b"a z/y table registered after the draw coin").hexdigest()
    stale = {**ev, "entries": [{**x, "root": after} if x["value"] == zy0 else x for x in ev["entries"]]}
    failed = {**ev, "proofs": [{**p, "verdict": "reject"} if i == 0 else p for i, p in enumerate(ev["proofs"])]}
    doc = {"verifier_of_record": STAND_IN, "circuit": program_sha, "layout": layout, "period": nm,
           "registered": registered, "period_open": opened["body"], "period_close": closed["body"],
           "session": w["session"], "window": ev["window"]["root"], "salt": ev["salt"], "draw_k": DRAW_K,
           "drawn": w["drawn"], "verdicts": [p["verdict"] for p in ev["proofs"]], **verdict([ev]),
           "negatives": {"stale-entry": verdict([stale]), "unit-rejected": verdict([failed])},
           "evidence": {"key": key, "ledger": lines, "pins": pins, "main": main, "name": name, "fills": [ev]}}
    (out / "fill.json").write_text(json.dumps({k: v for k, v in doc.items() if k != "evidence"}, indent=1,
                                              sort_keys=True) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--name", default=f"fill-demo-{secrets.token_hex(3)}")
    p.add_argument("--coins", choices=("lean", "stub"), default="lean",
                   help="the challenger's coin source: the Lean verifier's (its sampled-proofs-verifier built), or "
                        "live_challenger's test stub of it (seeded bytes)")
    a = p.parse_args(argv)
    main_line = [WS.a5_coins().on_main([])]
    g = WS.local(a.out / "gate", main_line)
    if a.coins == "stub":
        stub(g, a.out / "coins.log")
    doc = run(g, main_line, a.out, a.name)
    (a.out / "evidence.json").write_text(json.dumps(doc["evidence"], sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("fill", "refused", "drawn", "verdicts", "negatives") if k in doc}, indent=1))
    return 0 if "fill" in doc and all("refused" in v for v in doc["negatives"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
