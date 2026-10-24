"""B2: fine-tune MODEL (full, AdamW 5e-5, batch 8 x 256, the sweep's recipe) and keep fp32 snapshots; also the cheap proxies.

Conditions (all draw batches from one stream: generator seed 0, batch 8, so "same data" means the same token windows):
  main   lr 5e-5, batch 8, 1000 steps, snapshots SNAPS_MAIN
  b1     lr 5e-5, batch 1 (the first sequence of each main batch), 1000 steps: 1/8 of main's FLOPs per step
  sub10  lr 5e-4, batch 8, every 10th main batch, 100 steps: 1/10 of main's steps over the same data span
"""
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL, TAG, SCRATCH, OUT, SEQ, BATCH, tokens, snap_path

SNAPS_MAIN = (1, 2, 3, 5, 10, 20, 30, 50, 100, 1000)
CONDITIONS = {
    "main": dict(lr=5e-5, steps=1000, snaps=SNAPS_MAIN, batch=BATCH, stride=1),
    "b1": dict(lr=5e-5, steps=1000, snaps=(1, 10, 100, 1000), batch=1, stride=1),
    "b1lr": dict(lr=1.8e-5, steps=1000, snaps=(1, 10, 100, 1000), batch=1, stride=1),
    "sub10": dict(lr=5e-4, steps=100, snaps=(1, 10, 100), batch=BATCH, stride=10),
    "sub10lr1": dict(lr=5e-5, steps=100, snaps=(1, 10, 100), batch=BATCH, stride=10),
    "sub10lr3": dict(lr=1.5e-4, steps=100, snaps=(1, 10, 100), batch=BATCH, stride=10),
}
DEV = os.environ.get("B2_DEV", "cuda")
if os.environ.get("SMOKE"):
    for c in CONDITIONS.values():
        c["steps"] = min(c["steps"], 10 if c["stride"] == 1 else 1)
        c["snaps"] = tuple(s for s in c["snaps"] if s <= c["steps"])


def batches(n_main_steps, seed=0):
    """the main run's batch start indices, step by step"""
    toks = tokens()["train"]
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_main_steps):
        yield torch.randint(0, len(toks) - SEQ - 1, (BATCH,), generator=g)


def run(name, lr, steps, snaps, batch, stride):
    toks = tokens()["train"]
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to(DEV).train()
    if name == "main":
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, snap_path("main", 0))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    log, t0, step = [], time.time(), 0
    for i, idx in enumerate(batches(steps * stride)):
        if i % stride:
            continue
        step += 1
        idx = idx[:batch]
        x = torch.stack([toks[j:j + SEQ] for j in idx]).to(DEV)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log.append(loss.item())
        if step in snaps:
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, snap_path(name, step))
            print(f"{TAG} {name} step {step} loss(avg last 20) {sum(log[-20:]) / len(log[-20:]):.3f} "
                  f"t={time.time() - t0:.0f}s", flush=True)
    json.dump({"loss": log, "seconds": time.time() - t0}, open(os.path.join(OUT, f"loss_{TAG}_{name}.json"), "w"))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "tok":
        print({k: len(v) for k, v in tokens().items()})
    else:
        run(which, **CONDITIONS[which])
