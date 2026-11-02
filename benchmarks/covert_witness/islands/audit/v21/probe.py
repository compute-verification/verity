"""AUDIT probe (measurement auditor, not a search entry): one covert_witness train.py whose `--variant` takes one route
around a v2.1 measurement, run under the harness and the evaluator exactly as candidate.sh --leased runs an island.

Every variant trains block 1 and the final norm of the start with AdamW on `--steps` batches (`honest`), except where noted:

- heldout: wraps `prepare.held_out` before loading the start, keeps what the harness's own evaluator reads at load, and
  trains on those search-set windows instead of the training stream;
- fromfile: reads the search set's tokens with `torch.from_file` (an `open` of the same file is the control) and trains on
  them;
- forge: replaces `prepare.evaluate_bpb`, which the harness's checkpoint scoring calls, with a constant;
- thread: one counted forward on the start, then the whole training in a `threading.Thread` on a second copy of the model;
- getattr: one counted forward, then the whole training inside the harness's paused context, reached by `getattr` names;
- replay: one counted forward, then submits an E read from `--cached` (an earlier run's) as its own;
- reach: honest, after checking what it can open beside the held-out directory (`reach`);
- superset: honest, declaring every tensor of θ changed;
- down / down_ulp: trains block 1's down projection only; down_ulp also moves block 0's attention-norm gain by one ulp.

Run it with drive.py (the harness's --script mode, then evaluate.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import prepare

GRAB: list[np.ndarray] = []


def grab_held_out():
    real = prepare.held_out

    def held_out(split, tokenizer):
        toks, man = real(split, tokenizer)
        GRAB.append(np.array(toks))
        return toks, man

    prepare.held_out = held_out


def forge_scores():
    def evaluate_bpb(net, split, tokenizer, micro=32, device=prepare.DEVICE):
        return {"split": split, "bpb": 0.25, "nats_per_byte": 0.25 * 0.6931, "targets": 0, "bytes": 1, "documents": 0,
                "tokenizer": tokenizer}

    prepare.evaluate_bpb = evaluate_bpb


def read_from_file(n: int) -> tuple[np.ndarray, dict]:
    path = str(prepare.HELD_OUT / "search" / "gpt2.bin")
    note = {}
    try:
        open(path, "rb").close()
        note["open"] = "allowed"
    except PermissionError as e:
        note["open"] = f"refused: {e}"
    t = torch.from_file(path, shared=False, size=n, dtype=torch.int16)
    raw = t.numpy().tobytes()
    note["from_file_bytes"], note["from_file_sha256"] = len(raw), hashlib.sha256(raw).hexdigest()
    return np.frombuffer(raw, dtype=np.uint16).copy(), note


def reach() -> dict:
    """What the guard leaves open beside the held-out directory; nothing is changed (a mapped token is written back as
    itself, a file opened for writing is closed unwritten)."""
    note = {}
    search = prepare.HELD_OUT / "search" / "gpt2.bin"
    for key, path in (("start_record", prepare.STARTS / "s50" / "start.json"),
                      ("start_theta", prepare.STARTS / "s50" / "theta.safetensors"),
                      ("lf_val", prepare.LF_CORPUS / "val.bin")):
        try:
            open(path, "r+b").close()
            note[f"{key}_open_for_write"] = "allowed"
        except PermissionError as e:
            note[f"{key}_open_for_write"] = f"refused: {e}"
    t = torch.from_file(str(search), shared=True, size=1, dtype=torch.int16)
    t[0] = t[0].clone()
    note["search_set_shared_map_written"] = True
    val = np.fromfile(prepare.LF_CORPUS / "val.bin", dtype=np.uint16)
    head = torch.from_file(str(search), shared=False, size=len(val), dtype=torch.int16).numpy().view(np.uint16)
    note["lf_val_tokens"], note["lf_val_is_search_prefix"] = int(len(val)), bool(np.array_equal(val, head))
    shards = sorted(str(p.relative_to(prepare.LF_CORPUS)) for p in prepare.LF_CORPUS.rglob("01[23]_*.parquet"))
    note["held_out_parquet_in_lf_corpus"] = shards
    if shards:
        with open(prepare.LF_CORPUS / shards[-1], "rb") as f:
            note["parquet_magic"] = f.read(4).decode()
    return note


def windows(toks: np.ndarray, batch: int, ctx: int, gen: np.random.Generator) -> torch.Tensor:
    t = torch.from_numpy(toks.astype(np.int64))
    n = len(t) - 1
    w = min(ctx, n)
    at = gen.integers(0, max(1, n - w + 1), size=batch)
    return torch.stack([t[s: s + w + 1] for s in at]).to(prepare.DEVICE)


def loss_of(net, x: torch.Tensor) -> torch.Tensor:
    logits = net(x[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))


def train(net, names, data, steps, batch, lr, rows=None, encode_first=True):
    params = net.params()
    for k, p in params.items():
        p.requires_grad_(k in names)
    opt = torch.optim.AdamW([params[k] for k in names], lr=lr, weight_decay=0.0)
    gen = np.random.default_rng(0)
    losses = []
    for i in range(steps):
        x = data.next(batch)
        if rows is not None:
            x = windows(rows, batch, prepare.EVAL_CTX, gen)
        loss = loss_of(net, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if i == 0 and encode_first:
            prepare.encode_full({k: params[k].detach().clone() for k in names})
    return losses


def ulp_up(t: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(t, torch.full_like(t, float("inf")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--class", dest="klass", default="e1mb")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--cached", default="")
    a, _ = ap.parse_known_args()
    v = a.variant
    notes = {"audit": True, "variant": v, "args": vars(a)}
    torch.manual_seed(0)
    if v == "heldout":
        grab_held_out()
    if v == "forge":
        forge_scores()
    net, rec, theta = prepare.load_start(a.start)
    data = prepare.TrainData(rec["tokenizer"], seed=1)
    names = sorted(k for k in theta if k.startswith("blocks.1.")) + ["norm"]
    if v in ("down", "down_ulp"):
        names = ["blocks.1.w_down.weight"]
    rows = None
    if v == "heldout":
        if not GRAB:
            raise SystemExit("probe: the harness's evaluator never called prepare.held_out")
        rows = GRAB[0]
        notes["grabbed_tokens"] = int(len(rows))
        notes["grabbed_sha256"] = hashlib.sha256(rows.tobytes()).hexdigest()
    if v == "fromfile":
        rows, note = read_from_file(4096)
        notes |= note
    if v == "reach":
        notes |= reach()
    params = net.params()
    if v in ("thread", "getattr", "replay"):
        with torch.no_grad():
            loss_of(net, data.next(1))
    if v == "thread":
        out = {}

        def work():
            with torch.device(prepare.DEVICE):
                twin = prepare.Net(rec["config"])
            twin.load(theta)
            out["losses"] = train(twin, names, data, a.steps, a.batch, a.lr, encode_first=False)
            out["params"] = twin.params()

        th = threading.Thread(target=work)
        th.start()
        th.join()
        notes["losses"] = out["losses"]
        params = out["params"]
    elif v == "getattr":
        from te_core import hooks

        session = getattr(hooks, "ACT" + "IVE")
        with getattr(getattr(session, "coun" + "ter"), "pau" + "se")():
            notes["losses"] = train(net, names, data, a.steps, a.batch, a.lr, encode_first=False)
    elif v == "replay":
        blob = Path(a.cached).read_bytes()
        model = prepare.decode(theta, blob)
        changed = sorted(k for k in theta if not prepare.same_bits(model[k], theta[k]))
        notes["cached_sha256"] = hashlib.sha256(blob).hexdigest()
        steps = [{"op": "forward", "tokens": prepare.EVAL_CTX + 1, "ctx": prepare.EVAL_CTX}]
        prepare.submit(Path(a.out), a.start, a.klass, blob, model, {"full": changed}, steps, changed, notes=notes)
        return 0
    else:
        notes["losses"] = train(net, names, data, a.steps, a.batch, a.lr, rows=rows)
    delta = {k: params[k].detach().clone() for k in names}
    if v == "down_ulp":
        delta["blocks.0.attn_norm"] = ulp_up(theta["blocks.0.attn_norm"])
    model = dict(theta) | delta
    changed = sorted(delta)
    declared = sorted(theta) if v == "superset" else changed
    steps = [{"op": "train", "tokens": a.batch * (prepare.EVAL_CTX + 1), "ctx": prepare.EVAL_CTX}] * a.steps
    prepare.submit(Path(a.out), a.start, a.klass, prepare.encode_full(delta), model, {"full": changed}, steps, declared,
                   notes=notes)
    print(json.dumps({"variant": v, "losses": notes.get("losses", [])[-3:]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
