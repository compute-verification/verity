"""One filler period end to end on a development gate: the fill period ``<name>:fill`` opened on the
auditor's ledger, one filler window of Dither calls under its live coins, and `fill.period_fill`'s verdict on it, with
the verifier of record a stand-in.

* **The circuit** is the filler descriptor (`fill.descriptor`) of `CALLS` calls of `SHAPE` (128 × 128 cells, K = 128 in
  two segments of 64, so each block is `PouwDitherBlock_v1` at FIRST or not), B the first n rows of the H200 example's
  seeded served weights' layer-0 up projection at Phase 1's shape (`e2e_h200.served_weights`, `LINEAR` under `SEED`,
  its int8 codes, pinned by `dither.checkpoint_digest`).
* **The gate** is `a5_coins.Local` (`window_salt.local`): `benchmarks/live_challenger/auditor.py`'s verbs on this machine
  under a key made here. It opens the fill period on the circuit with the draw K `DRAW_K`, reserves the window's session
  and opens it on `fill.open_root` of the circuit and the session, with a salt coin, a call coin a call and a draw coin
  (`fill.families`), and closes the period once the session ended.
* **The window** (`fill_window`): the salt coin and every call coin opened after the open root, the window's salt
  `h1tile.live_salt` of the first; call c's seeds `dither.fill_seeds` of the salt and call coin c, its A the pad
  (`dither.pad_rows`), its T (`dither.values`) committed under the prover's private salts (`dither.commit`); then the draw
  coin after `fill.draw_root` over the salt and every call's value root.
* **The verifier of record is a stand-in** (`STAND_IN`, `stand_in`): for each drawn block it opens the block's rows
  against its call's value root and checks `PouwDitherBlock_v1` in the clear on them, the pad row, B's row and the coins
  (`dither.check_block`). At the real run it is C-Flock's verifier on each block's proof.

The record is ``OUT/fill.json``: the fill (``{work, k, delta, eps}``) or the refusal, the drawn blocks and the stand-in's
verdicts, and two negatives the fill must refuse (``negatives``): a value root changed after the draw coin, and a drawn
block the verifier rejects. ``OUT/evidence.json`` is what the fill was checked on, the `fill.period_fill` arguments
``{key, ledger, pins, main, name, fills}``.

    uv run python benchmarks/pouw/fill_period/fill_period.py --out OUT [--coins stub]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import fill as F
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import dither as D
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "window_salt"))
import window_salt as WS  # noqa: E402

TREE = WS.TREE
sys.path.insert(0, str(TREE / "examples" / "h200"))
from e2e_h200 import served_weights as SW  # noqa: E402

STUB = TREE / "benchmarks" / "live_challenger" / "tests" / "stub_serve.py"
#: B's source: the seeded served weights' layer-0 up projection at Phase 1's shape (512 × 128 int8 codes)
WEIGHTS, LINEAR, SEED = SW.PHASE1, "l0.wup", b"0"
SHAPE = D.Shape(128, 128, 128, lp=32, L=64, o=False)
CALLS = 2
DRAW_K = 4
LANE = "compute-accounting"
STAND_IN = ("stand-in verifier of record: each drawn block's rows opened at its call's value root and PouwDitherBlock_v1 "
            "checked in the clear on them, the pad row, B's row and the coins (dither.check_block)")


def b_rows(sh: D.Shape) -> list[list[int]]:
    """B, n × K: the first n output channels of `LINEAR`'s seeded int8 codes, each its first K inputs."""
    return SW.generate(WEIGHTS, SEED, LINEAR, range(sh.n))[:, :sh.k].tolist()


def call(sh: D.Shape, salt: bytes, coin: bytes, B: list[list[int]], ks: bytes, c: int
         ) -> tuple[dict[str, bytes], list[list[int]], D.Tree]:
    """Filler call c: its seeds, its pad A and its values' tree under the prover's salt key `ks`."""
    seeds = D.fill_seeds(salt, coin)
    A = D.pad_rows(seeds, sh.k, range(sh.m))
    T, _ = D.values(sh, seeds, A, B)
    return seeds, A, D.commit(sh, D.values_buffer(sh, T), D.salts(ks, c * sh.leaves))


def stand_in(sh: D.Shape, units: list[int], calls: list[tuple[dict, list, D.Tree]], B: list[list[int]], circuit: bytes,
             salt: bytes, coins: list[bytes]) -> list[dict]:
    """Each drawn block's outcome (`credit._proved`'s form, `fill.statement` its program): accepted when the rows it opens
    open at its call's value root and `PouwDitherBlock_v1` holds on them."""
    dom, out = sh.domain(), []
    for u in units:
        b = c, i, j, s = F.block_of(sh, u)
        seeds, A, tree = calls[c]
        rows, ok = {}, True
        for pos in sh.opened(i, j, s):
            row, sl, path = tree.open(pos)
            ok &= D.opens(dom, tree.root, sh.leaves, pos, row, sl, path)
            rows[pos] = row
        ins = D.block_inputs(sh, seeds, A[i], B[j], i, j, s, D.word_at(sh, rows, i, j, s - 1), D.word_at(sh, rows, i, j, s))
        ok = ok and D.check_block(ins) == 1
        out.append({"unit": u, "verdict": "accept" if ok else "reject",
                    "program": F.statement(circuit, salt, coins[c], b).hex(), "registered": F.registered(c, tree.root)})
    return out


def fill_window(g: Any, desc: bytes, B: list[list[int]], k: int, dest: Path) -> dict:
    """One filler window on gate `g` (module docstring): its evidence (`fill.FILL_KEYS`), with the stand-in's outcomes,
    and what it drew."""
    circuit = hashlib.sha512(desc).digest()
    address, secret, commitment = g.open(lambda s: F.open_root(circuit, s).hex(), [list(f) for f in F.families(CALLS)],
                                         scheme=Wd.LIVE_SCHEME)
    r = F.open_root(circuit, g.session)
    ks = secrets.token_bytes(64)
    with remote.RemoteChallenger.connect(address, bytes.fromhex(secret)) as client:
        c = client.commitment
        if c.root.hex() != commitment:
            raise Ch.CoinError("the challenger's hello answered another commitment than the gate's open")
        opening = client.open(Wd.SALT, 0, r)
        salt = HT.live_salt(c, opening.coin, 0, r)
        coins = [client.open(F.CALL, i, r) for i in range(CALLS)]
        calls = [call(SHAPE, salt, o.coin, B, ks, i) for i, o in enumerate(coins)]
        roots = [t.root for _s, _a, t in calls]
        receipts = F.receipts(roots)
        root = F.draw_root(salt, receipts)
        draw = client.open(Wd.DRAW, 0, root)
    directory, _ledger, _pins, key = g.finish(dest)
    d = {"window": {"root": r.hex()}, "salt": salt.hex(), "opening": opening.to_json(),
         "calls": [o.to_json() for o in coins], "roots": [x.hex() for x in roots], "descriptor": desc.hex(),
         "session": remote.session_files(directory),
         "draw": {"receipts": receipts.hex(), "root": root.hex(), "coin": draw.to_json()}}
    drawn, why = Cr._window_drawn(d["window"], d["draw"], F.stratum(SHAPE, CALLS), k)
    if why is not None:
        raise Ch.CoinError(f"the window's draw is refused: {why}")
    d["proofs"] = stand_in(SHAPE, drawn, calls, B, circuit, salt, [o.coin for o in coins])
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
    now = datetime.now(UTC)
    hooks = dict(by=LANE, now=now, own=lambda p: None, seal=lambda p: None,
                 period=lambda args: A.period(host, args, signer=g.signer, pinned=g.pin, now=now, active=lambda s: False))

    def gate(*argv: str) -> dict:
        rc, got = A.gate(list(argv), host, **hooks)
        doc = json.loads(got)
        if rc:
            raise Ch.CoinError(f"the gate refused {argv[0]}: {doc.get('refused')}")
        return doc

    out.mkdir(parents=True, exist_ok=True)
    B = b_rows(SHAPE)
    desc = F.descriptor(SHAPE, CALLS, D.checkpoint_digest(B))
    (out / "fill-program.json").write_bytes(desc)
    nm = F.period_name(name)
    opened = gate("period-open", "--name", nm, "--lane", LANE, "--cap", "1", "--circuit",
                  hashlib.sha512(desc).hexdigest(), "--draw-k", str(DRAW_K))
    w = fill_window(g, desc, B, DRAW_K, out / "challenger")
    closed = gate("period-close", "--name", nm)
    lines = remote.ledger_lines(host.ledger)
    pins = [x for x in host.pins.read_text().split("\n") if x]
    key = w["key"].to_json()

    def verdict(fills: list[dict]) -> dict:
        got, why = F.period_fill(key, lines, pins, main, name, fills, CC.lms_verify)
        return {"fill": got} if why is None else {"refused": list(why)}

    ev = w["evidence"]
    after = hashlib.sha512(b"a value root committed after the draw coin").hexdigest()
    stale = {**ev, "roots": [after, *ev["roots"][1:]]}
    failed = {**ev, "proofs": [{**p, "verdict": "reject"} if i == 0 else p for i, p in enumerate(ev["proofs"])]}
    doc = {"verifier_of_record": STAND_IN, "circuit": hashlib.sha512(desc).hexdigest(), "descriptor": json.loads(desc),
           "period": nm, "period_open": opened["body"], "period_close": closed["body"], "session": w["session"],
           "window": ev["window"]["root"], "salt": ev["salt"], "draw_k": DRAW_K, "drawn": w["drawn"],
           "blocks": [list(F.block_of(SHAPE, u)) for u in w["drawn"]], "verdicts": [p["verdict"] for p in ev["proofs"]],
           **verdict([ev]), "negatives": {"stale-root": verdict([stale]), "block-rejected": verdict([failed])},
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
    print(json.dumps({k: doc[k] for k in ("fill", "refused", "drawn", "blocks", "verdicts", "negatives") if k in doc},
                     indent=1))
    return 0 if "fill" in doc and all("refused" in v for v in doc["negatives"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
