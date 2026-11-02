"""M3's drawn H200 units checked on CPU from their bytes alone: `m3draw.py open`'s ``drawn.json`` and ``drawn.bin``,
held to the registration (`m3reg.py`) and the live coin (`m3coin.py`).

    python examples/h200/m3check.py --drawn DIR --registration REG.json --coin coin.json [--payload BIN] [--out FILE]

* **The sample.** The registration's digest is its record's. The coin's units are
  ``derive(coin, DOMAIN, {"registration": digest}).subset(k, population)`` again. They are the opened units, in order.
* **The openings.** Each unit's rows and salts are its committed strings (``hm96-sha512/row/v2``). The unit's tree reaches
  its slot root, and the slot root reaches the registered step root (`slots.verify_unit`, under the registered context
  and the layout this tree builds, whose digest must be the registered one).
* **The replay.** Each unit's template is recomputed bit for bit on its recorded inputs (`units.py`'s Definitions,
  evaluated here). A linear's or an embedding's weight inputs are the seeded model's own rows, regenerated from the
  public shape and seed. Nothing else about its inputs is checked: they are the re-run's tables, which reproduce every
  registered root, and they are opened nowhere.

Exit 0 when every check passes, else 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3coin as COIN
import m3reg as REG
from e2e_h200 import units as UN
from e2e_h200.commit import Committer, fixture_secrets
from e2e_h200.plan import Map
from e2e_h200.shape import SHAPES
from e2e_h200.weights import Weights
from e2e_h200 import served_weights as SW


def arr(pay: bytes, d: dict) -> np.ndarray:
    o, n = d["bytes"]
    return np.frombuffer(pay[o:o + n], dtype=np.dtype(d["dtype"])).reshape(d["shape"])


def replay(shape, mp: Map, weights: Weights, fam: str, template: str, idx: int, xs: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """The unit's Definition on its inputs, and the checks of its weight inputs against the seeded model."""
    name, u, extra = template.split("{")[0], mp.units()[fam], {}
    if name == "EmbeddingCoordinate_v1":
        tok, col, pos = xs
        extra["weights_equal"] = bool(np.array_equal(col, np.asarray(weights.array("tok"), np.uint16)[idx]))
        want = UN._ev(UN.tpl(name, V=len(col)), tok, col[None], pos)[0]
    elif name == "AllGather_v1":
        want = xs[0]
    elif name == "RmsSumSqChunks_v1":
        want = UN.sums_chunks(xs[0], mp.f.chunk)
    elif name == "RmsNormOutSums_v1":
        n = shape.width
        want = UN.norm_out_sums(*xs, n, mp.f.groups(n))
    elif name == "RowAbsMax_v1":
        want = np.array([UN.row_abs_max(xs[0], u.table[:-len("_amax")].endswith("down"))], np.uint16)
    elif name == "QuantChunkS8_v1":
        codes, scale = UN.quant_chunk(xs[0], xs[1][0], u.table.endswith("down_quant"))
        want = np.concatenate([codes, np.array([scale], np.uint32).view(np.uint8)])
    elif name == "ScaledMmS8CoordinateRows_v1":
        x, sx, wrow, sw = xs
        tensor = u.table
        extra["weights_equal"] = bool(np.array_equal(wrow, weights.row(tensor, idx).view(np.uint8)))
        extra["scale_equal"] = bool(np.array_equal(
            sw, np.asarray(weights.array(f"{tensor}.scale")[idx:idx + 1], np.float32).view(np.uint32)))
        want = UN._s8mm(weights.dims[tensor][1]).evaluate(x[None], sx, wrow[None], sw)[0]
    elif name == "QkSoftmaxRowPad_v1":
        q, keys, n = xs
        want = UN.qk_softmax_pad(q, keys[:int(n[0])])
    elif name == "PvRow_v1":
        want = UN.pv_row(*xs)
    elif name == "Bf16AddRow_v2":
        want = UN.bf16_add_row(*xs)
    elif name == "Argmax_v2":
        want = UN.argmax_row(xs[0][None])
    else:
        raise ValueError(f"{template}: no Definition here")
    return np.asarray(want).reshape(-1), extra


def lms_verify(key, message: bytes, signature: bytes) -> bool:
    from verity.ml.crypto import lms
    p = lms.Params(64, key.W, key.H)
    try:
        return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))
    except ValueError:
        return False


def session_audit(coin: dict, digest: str, d: Path, key_record: Path, main: Path) -> dict:
    """The coin's session as core's verifier takes it: the one session the auditor's ledger opens on the registration's
    digest under the pinned challenger key, run from a commit on main, whose signed log shows that digest as the only root
    it received and whose commitment opens the recorded coin after it."""
    from verity.core.protocols.sampled_proofs import challenger as Ch
    from verity.core.protocols.sampled_proofs import remote as R
    sid, window = coin["reserve"]["session"], {"root": digest}
    out: dict = {"session": sid}
    try:
        key = Ch.key_of_record(json.loads(key_record.read_text()))
        sdir = d / sid / "out" if (d / sid / "out").is_dir() else d / sid
        ledger = d / f"{sid}.ledger.jsonl"
        commitment, roots = R.trusted_session(sdir, [ledger], window, key, lms_verify)
        pins = [x for x in (d / f"{sid}.pins.jsonl").read_text().split("\n") if x]
        audited = Ch.pinned_session(Ch.ledger_head([R.ledger_lines(ledger)], key, lms_verify), window, key, pins,
                                    R.main_lines(main), lms_verify)
        after = bytes.fromhex(digest)
        out.update(trusted=True, pinned=audited.session == sid,
                   roots_ok=roots == {COIN.LABEL: (after,)},
                   commitment_ok=commitment.root.hex() == coin["commitment"],
                   coin_ok=Ch.live_reason(commitment, Ch.LiveOpening.from_json(coin["opening"]), COIN.LABEL, 0, after) is None)
    except (Ch.CoinError, OSError, ValueError, KeyError) as e:
        out.update(trusted=False, why=str(e))
    out["ok"] = all(out.get(k) for k in ("trusted", "pinned", "roots_ok", "commitment_ok", "coin_ok"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drawn", required=True)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--coin", required=True)
    ap.add_argument("--payload", default="", help="drawn.bin (default: beside drawn.json)")
    ap.add_argument("--out", default="")
    ap.add_argument("--key-record", default="", help="the challenger's challenger-key/v1 record: audit the coin's session")
    ap.add_argument("--main", default="", help="main's first-parent history, with --key-record")
    a = ap.parse_args(argv)
    d = Path(a.drawn)
    doc = json.loads((d / "drawn.json").read_text())
    pay = Path(a.payload or d / doc["payload"]["file"]).read_bytes()
    reg = json.loads(Path(a.registration).read_text())
    coin = json.loads(Path(a.coin).read_text())
    rec, digest = reg["record"], reg["digest"]
    out: dict = {"format": "verity/e2e-h200/m3-check/v1", "registration": digest}
    out["registration_digest_ok"] = REG.digest(REG.FORMAT, rec) == digest
    out["payload_ok"] = hashlib.sha512(pay).hexdigest() == doc["payload"]["sha512"]
    law, N = rec["law"], int(rec["population"])
    coin_bytes = bytes.fromhex(coin["opening"]["coin"])
    again = COIN.draw(coin_bytes, digest, law["k"], N)
    units = [u["index"] for u in doc["units"]]
    out["draw_ok"] = (coin.get("registration") == digest and again == coin["units"] == list(doc["draw"]["units"]) == units
                      and len(units) == law["k"] and N == doc["population"])
    step = int(doc["step"])
    root = next(r for r in rec["roots"] if r["port"] == f"step{step}")
    out["step_root_ok"] = root["root"] == doc["step_root"]

    shape = SHAPES[{"cmp92-h200-mqa-w16384-l52": "h200"}.get(rec["partition_record"]["shape"], rec["partition_record"]["shape"])]
    seed = {"test": SW.TEST_SEED, "demo": SW.DEMO_SEED}[rec["weights"]["seed"]]
    ctx, key = fixture_secrets(doc["run_id"])
    committer = Committer("slots", ctx, key, shape, "cpu", lambda s: b"\0" * 32)
    S, lay = committer.S, committer.layout
    out["context_ok"] = ctx.hex() == rec["context"] == doc["context"]
    out["layout_ok"] = S.layout_digest(lay).hex() == rec["partition_record"]["layout_digest"] == doc["layout_digest"]
    U = lay.unit_count
    mp, weights = Map(shape), Weights(shape, seed)
    rows = []
    for x in doc["units"]:
        b, u = divmod(int(x["index"]), U)
        op = x["opening"]
        o = S.UnitOpening.of_json(op)
        rs = {k: (pay[r["row"][0]:r["row"][0] + r["row"][1]], pay[r["salt"][0]:r["salt"][0] + r["salt"][1]])
              for k, r in enumerate(op["rows"])}
        opened = (o.slot == b == x["row"] and o.unit == u == x["commit_unit"]
                  and S.verify_unit(bytes.fromhex(root["root"]), ctx, lay, step, int(doc["slots"]), o, rs))
        xs = [arr(pay, i) for i in x["inputs"]]
        got = arr(pay, x["outputs"]).reshape(-1)
        want, extra = replay(shape, mp, weights, x["family"], x["template"], int(x["unit"]), xs)
        replayed = got.shape == want.shape and np.array_equal(got, want.astype(got.dtype))
        committed = b"".join(r for r, _ in rs.values())
        row = {"index": x["index"], "template": x["template"], "family": x["family"], "unit": x["unit"], "slot": b,
               "commit_unit": u, "opened": bool(opened), "replayed": bool(replayed),
               "outputs_are_committed_rows": committed == np.ascontiguousarray(got).tobytes(), **extra}
        row["ok"] = row["opened"] and row["replayed"] and row["outputs_are_committed_rows"] and all(extra.values())
        rows.append(row)
        print("CHECK " + json.dumps(row), flush=True)
    out["units"] = rows
    if a.key_record:
        out["session"] = session_audit(coin, digest, Path(a.coin).parent, Path(a.key_record), Path(a.main))
        print("SESSION " + json.dumps(out["session"]), flush=True)
    out["ok"] = all(out[k] for k in ("registration_digest_ok", "payload_ok", "draw_ok", "step_root_ok", "context_ok", "layout_ok")) \
        and all(r["ok"] for r in rows) and (not a.key_record or out["session"]["ok"])
    print("VERDICT " + json.dumps({k: v for k, v in out.items() if k != "units"}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if out["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
