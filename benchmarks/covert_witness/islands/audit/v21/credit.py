"""AUDIT probe (measurement auditor, not a search entry): counterexample crediting (`prepare.declare_step`) when the
decoder the harness's check calls is the program's own `prepare.decode`.

Each variant trains the tied embedding, block 1 and the final norm with AdamW on `--steps` batches:

- dense: declares nothing;
- declare_full: each step declares its weights as encode_full of their current values (the real decoder rebuilds them);
- forge_decode: replaces `prepare.decode` with one that returns, for a 16-byte tag it declared, the weights the step
  read, and declares that tag each step.

Run it with drive.py (the harness's --script mode, then evaluate.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare

SNAP: dict[str, dict[str, torch.Tensor]] = {}


def forge_decoder():
    real = prepare.decode

    def decode(theta, blob, device=prepare.DEVICE):
        got = SNAP.get(hashlib.sha256(bytes(blob)).hexdigest())
        if got is None:
            return real(theta, blob, device=device)
        return {k: (got[k] if k in got else v).to(device) for k, v in theta.items()}

    prepare.decode = decode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--class", dest="klass", default="reference")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3)
    a, _ = ap.parse_known_args()
    v = a.variant
    torch.manual_seed(0)
    if v == "forge_decode":
        forge_decoder()
    net, rec, theta = prepare.load_start(a.start)
    data = prepare.TrainData(rec["tokenizer"], seed=1)
    names = ["embed", "norm"] + sorted(k for k in theta if k.startswith("blocks.1."))
    params = net.params()
    for k, p in params.items():
        p.requires_grad_(k in names)
    opt = torch.optim.AdamW([params[k] for k in names], lr=a.lr, weight_decay=0.0)
    losses = []
    for i in range(a.steps):
        x = data.next(a.batch)
        if v == "declare_full":
            prepare.declare_step(prepare.encode_full({k: params[k].detach().clone() for k in names}))
        elif v == "forge_decode":
            tag = b"audit-tag-" + i.to_bytes(6, "little")
            SNAP[hashlib.sha256(tag).hexdigest()] = {k: params[k].detach().clone() for k in names}
            prepare.declare_step(tag)
        logits = net(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    delta = {k: params[k].detach().clone() for k in names}
    model = dict(theta) | delta
    steps = [{"op": "train", "tokens": a.batch * (prepare.EVAL_CTX + 1), "ctx": prepare.EVAL_CTX}] * a.steps
    notes = {"audit": True, "variant": v, "args": vars(a), "losses": losses}
    prepare.submit(Path(a.out), a.start, a.klass, prepare.encode_full(delta), model, {"full": sorted(names)}, steps,
                   sorted(names), notes=notes)
    print(json.dumps({"variant": v, "losses": losses[-2:]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
