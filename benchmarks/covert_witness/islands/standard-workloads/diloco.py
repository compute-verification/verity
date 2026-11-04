"""Island `standard-workloads`: DiLoCo from the start (te_core.workloads' `diloco`), its M workers simulated in turn on
one GPU.

    python guard.py islands/standard-workloads/diloco.py --start s-t20 --out SUB [--workers 4] [--inner-steps 50] [...]

Every worker starts from the outer weights (θ at first) and runs H inner AdamW steps on its own share of the start's
stream (`--batch` windows a worker a step, in bf16 autocast; its AdamW state kept across syncs, as DiLoCo keeps it);
then an outer step: each worker's pseudo-gradient, the outer weights minus its own, passed at `--pass-dtype`, averaged
over the M workers, and applied by Nesterov SGD (`--outer-lr`, `--outer-momentum`), after which every worker restarts
from the new outer weights. Each step is explicit (hooks.step()): an inner step is all M workers' (B = M x batch
windows), and each sync is its own step, which reads no tokens. E is the outer weights after the last sync, every
tensor in full (`prepare.encode_full`).
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

import prepare
from te_core import hooks

DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inner-steps", type=int, default=50)
    ap.add_argument("--syncs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16, help="windows a worker a step")
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--pass-dtype", choices=tuple(DTYPES), default="f32")
    ap.add_argument("--data-seed", type=int, default=11)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = prepare.DEVICE
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    names = sorted(params)
    outer = {n: params[n].detach().float().clone() for n in names}
    mom = {n: torch.zeros_like(outer[n]) for n in names}
    workers = []
    for w in range(a.workers):
        with torch.device(dev):
            wnet = prepare.Net(rec["config"])
        own = wnet.params()
        with torch.no_grad():
            for n in names:
                own[n].data = outer[n].clone()
        opt = torch.optim.AdamW(list(own.values()), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                                fused=dev.startswith("cuda"))
        workers.append((wnet, own, opt))
    n_inner = a.inner_steps * a.syncs
    hooks.plan(n_inner + a.syncs)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    tokens = a.workers * a.batch * prepare.EVAL_CTX
    steps_log: list[dict] = []
    pass_dt = DTYPES[a.pass_dtype]
    print(json.dumps({"workers": a.workers, "inner_steps": a.inner_steps, "syncs": a.syncs, "tokens_per_step": tokens}),
          flush=True)
    t0 = time.perf_counter()
    for s in range(a.syncs):
        for h in range(a.inner_steps):
            i = s * a.inner_steps + h
            frac = min(1.0, (i + 1) / a.warmup) * (1 - i / n_inner)
            with hooks.step():
                for wnet, own, opt in workers:
                    for grp in opt.param_groups:
                        grp["lr"] = a.lr * frac
                    rows = data.next(a.batch)
                    for j in range(0, a.batch, a.micro):
                        x = rows[j: j + a.micro]
                        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                            logits = wnet(x[:, :-1])
                        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
                        (loss * (len(x) / a.batch)).backward()
                    torch.nn.utils.clip_grad_norm_(list(own.values()), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            steps_log.append({"op": "train", "tokens": tokens})
            if (i + 1) % 50 == 0:
                print(json.dumps({"step": i + 1, "loss": float(loss.detach()), "s": round(time.perf_counter() - t0, 1)}),
                      flush=True)
        with hooks.step(), torch.no_grad():
            for n in names:
                pseudo = torch.stack([(outer[n] - own[n].detach()).to(pass_dt) for _, own, _ in workers]).float().mean(0)
                mom[n].mul_(a.outer_momentum).add_(pseudo)
                outer[n].sub_(pseudo.add(mom[n], alpha=a.outer_momentum), alpha=a.outer_lr)
                for _, own, _ in workers:
                    own[n].copy_(outer[n])
        steps_log.append({"op": "extra", "flops": 0, "what": "outer sync (elementwise; the harness counts it)"})
        print(json.dumps({"sync": s + 1, "s": round(time.perf_counter() - t0, 1)}), flush=True)
    model = {n: outer[n].to(theta[n].dtype).cpu() for n in names}
    n_values = sum(t.numel() for t in outer.values())
    notes = {"workload": "diloco", "args": vars(a), "parameters": n_values,
             "pseudo_gradient_bits": n_values * torch.finfo(pass_dt).bits, "reduction_fan_in": a.workers,
             "outer": "Nesterov SGD on the mean pseudo-gradient", "inner": "AdamW, state kept across syncs"}
    prepare.submit(prepare.Path(a.out), a.start, prepare.encode_full(model), model, {"full": names}, steps_log,
                   changed=names, notes=notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
