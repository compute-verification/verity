"""Island `standard-workloads`: distillation from the registered start into a fresh student (te_core.workloads'
`distill`).

    python guard.py islands/standard-workloads/distill.py --start s-t20 --out SUB [--width 0.5] [--steps 600] [...]

The teacher is the start itself, θ unchanged: each step its forward (bf16 autocast, no gradient) reads only θ and the
step's windows. The student is an lf model of θ's depth, vocabulary and context at `--width` of its d_model, heads and
MLP width, trained from a fresh init (`--seed`) by AdamW on the teacher's next-token distributions at every position
(KL(teacher || student), `--temperature`). Each step is explicit (hooks.step()). E is the student in the model format
(`prepare.encode_model`), every tensor its own.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

import prepare
from te_core import hooks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--data-seed", type=int, default=13)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = prepare.DEVICE
    teacher, rec, theta = prepare.load_start(a.start)
    for p in teacher.parameters():
        p.requires_grad_(False)
    c = rec["config"]
    heads = max(1, round(c["n_head"] * a.width))
    d = heads * (c["d_model"] // c["n_head"])
    cfg = c | {"name": f"{c['name']}-student", "d_model": d, "n_head": heads,
               "d_ff": 16 * max(1, round(c["d_ff"] * a.width / 16))}
    torch.manual_seed(a.seed)
    with torch.device(dev):
        student = prepare.Net(cfg)
    params = student.params()
    for p in params.values():
        p.data = p.data.float()
    opt = torch.optim.AdamW(list(params.values()), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                            fused=dev.startswith("cuda"))
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    hooks.plan(a.steps)
    tokens = a.batch * prepare.EVAL_CTX
    teacher_fwd = prepare.train_flops(rec, {}, [{"op": "forward", "tokens": tokens}])["flops"]
    steps_log: list[dict] = []
    print(json.dumps({"student": cfg, "student_parameters": sum(p.numel() for p in params.values()),
                      "teacher_parameters": rec["parameters"], "steps": a.steps}), flush=True)
    t0 = time.perf_counter()
    for step in range(a.steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / a.steps)
        for grp in opt.param_groups:
            grp["lr"] = a.lr * frac
        with hooks.step():
            rows = data.next(a.batch)
            for j in range(0, a.batch, a.micro):
                x = rows[j: j + a.micro, :-1]
                with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                    with torch.no_grad():
                        target = F.log_softmax(teacher(x).float() / a.temperature, -1)
                    logits = student(x)
                logp = F.log_softmax(logits.float() / a.temperature, -1)
                loss = F.kl_div(logp.view(-1, logp.shape[-1]), target.view(-1, target.shape[-1]), log_target=True,
                                reduction="batchmean") * a.temperature ** 2
                (loss * (len(x) / a.batch)).backward()
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        steps_log += [{"op": "train", "tokens": tokens}, {"op": "extra", "flops": teacher_fwd, "what": "teacher forward"}]
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1, "kl": float(loss.detach()), "s": round(time.perf_counter() - t0, 1)}),
                  flush=True)
    model = {n: p.detach().float().cpu() for n, p in params.items()}
    notes = {"workload": "distill", "args": vars(a), "student": cfg,
             "student_parameters": sum(p.numel() for p in params.values()), "teacher": "the start, theta unchanged"}
    prepare.submit(prepare.Path(a.out), a.start, prepare.encode_model(cfg, model), model, {"full": sorted(model)},
                   steps_log, changed=sorted(model), notes=notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
