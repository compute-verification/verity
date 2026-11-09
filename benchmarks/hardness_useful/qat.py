"""Strassen-hostile QAT (HT-USEFUL): a short distillation pass that keeps every block linear's int8 codes spread to a
target Strassen overflow rate, then its quality and its exported codes.

The student is the model with every block linear a `quant.QLinear` calibrated to --target (a Strassen level-1 overflow
rate; `absmax` is stock per-channel int8, `bf16` evaluates the unquantized model and trains nothing). With --rotate
each linear quantizes W·R and reads x·R (`rotation.py`). The teacher is the same model in bf16, frozen. The loss is the
forward KL(teacher ‖ student) at temperature 1 over every position, on FineWeb-Edu packed into --seq windows in file
order (`prep.py`); every arm reads the same tokens in the same order.
AdamW (0.9, 0.95), no weight decay, linear warmup then cosine to 0, gradients clipped at 1. Every --recal steps each
matrix's c is solved again, so the overflow stays at the target as the weights move. Every parameter trains, or with
--lora R only a rank-R update B·A of each block linear's latent weight (`quant.add_lora`) and the RMSNorm gains, which
fits an 8B student in fp32 on one GPU; B·A is merged into the latent weight before the final c, codes and evals.
--teacher-device puts the teacher on another GPU.

Quality, before training (the post-training quantization at the same scales) and after:
- perplexity at --seq on WikiText-2's test split (raw, joined by blank lines, non-overlapping windows) and on the first
  --val-tokens of the FineWeb-Edu val shard;
- zero-shot accuracy with lm-eval (--tasks), the model evaluated as served: codes and scales frozen.

--serve is how activations reach the codes in training, the headline evals and zero-shot (`quant.Served`: bf16, int8
per token, design U's int8 plus live 2-bit noise, or its parts, chosen per matrix); perplexity is also measured under
every --eval-modes spec, on the same noise draws for every arm. --probe N measures, per matrix, how much error each
part of the service puts in its input and output (`quant.Probe`) over N WikiText-2 windows. With --device-map cpu the
model is evaluated on the CPU, whose noise draws are not the GPU's.

Outputs in --out: result.json (config, per-matrix c and overflow profile, evals, the loss curve, throughput) and, with
--price, the route pricing of the codes (`price.py`, run on the CPU while zero-shot runs on the GPU). With --codes FILE
the codes are written there (each matrix's int8 codes `NAME.q` and fp32 scales `NAME.s`, about 1 GB for a 1B model, so
outside the run dir) and result.json records the file's sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quant  # noqa: E402

TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande"
EVAL_MODES = "a16,a8,e2,e2:down_proj"
KD_CHUNK = 2048
EVAL_SEED, TRAIN_SEED = 1234, 1


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


class KD(torch.autograd.Function):
    """Mean over positions of KL(teacher ‖ student); the gradient is formed chunk by chunk in the forward and kept in
    the logits' dtype, so no fp32 vocabulary-sized activation is saved."""

    @staticmethod
    def forward(ctx, sl, tl):
        shape = sl.shape
        s2, t2 = sl.reshape(-1, shape[-1]), tl.reshape(-1, shape[-1])
        n = s2.shape[0]
        grad = torch.empty_like(s2)
        loss = torch.zeros((), device=sl.device, dtype=torch.float64)
        for i in range(0, n, KD_CHUNK):
            ls = F.log_softmax(s2[i:i + KD_CHUNK].float(), -1)
            lt = F.log_softmax(t2[i:i + KD_CHUNK].float(), -1)
            pt = lt.exp()
            loss += (pt * (lt - ls)).sum(dtype=torch.float64)
            grad[i:i + KD_CHUNK] = ((ls.exp() - pt) / n).to(grad.dtype)
        ctx.save_for_backward(grad)
        ctx.shape = shape
        return (loss / n).float()

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return (grad * g.to(grad.dtype)).reshape(ctx.shape), None


def wikitext_ids(tok) -> np.ndarray:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return np.asarray(tok("\n\n".join(ds["text"]))["input_ids"], dtype=np.int64)


@torch.no_grad()
def perplexity(model, ids: np.ndarray, seq: int, bs: int = 4, max_windows: int | None = None) -> dict:
    n = len(ids) // seq
    if max_windows:
        n = min(n, max_windows)
    was = model.training
    model.eval()
    nll, cnt, windows = 0.0, 0, []
    dev = model.get_input_embeddings().weight.device
    for i in range(0, n, bs):
        x = torch.from_numpy(ids[i * seq:min(i + bs, n) * seq].astype(np.int64).reshape(-1, seq)).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            logits = model(x).logits
        w = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                            x[:, 1:].reshape(-1).to(logits.device), reduction="none").view(len(x), -1).sum(1)
        windows += (w / (seq - 1)).tolist()
        nll += w.sum().item()
        cnt += x[:, 1:].numel()
        del logits
    model.train(was)
    return {"ppl": math.exp(nll / cnt), "nll": nll / cnt, "tokens": cnt, "window_nll": windows}


def zeroshot(model, tok, tasks: list[str], bs: int, limit: int | None) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs, max_length=2048)
    out = {}
    for t in tasks:
        t0 = time.time()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = lm_eval.simple_evaluate(model=lm, tasks=[t], num_fewshot=0, limit=limit, log_samples=False,
                                            bootstrap_iters=0)
            out[t] = {k: v for k, v in r["results"][t].items() if isinstance(v, (int, float))}
            out[t]["n"] = r.get("n-samples", {}).get(t, {}).get("effective")
        except Exception as e:  # noqa: BLE001 — a task that can't load is reported, not fatal
            out[t] = {"error": f"{type(e).__name__}: {e}"}
        out[t]["seconds"] = time.time() - t0
        log("zeroshot", t, json.dumps(out[t]))
    return out


def served(model, args, spec: str):
    if args.target != "bf16":
        quant.set_served(model, spec).reseed(EVAL_SEED)


def ppls(model, wt_ids, val_ids, args, spec: str) -> dict:
    served(model, args, spec)
    out = {}
    if "wikitext2" in args.ppl_sets:
        out["wikitext2"] = perplexity(model, wt_ids[args.wt_skip * args.seq:], args.seq, max_windows=args.wt_windows)
    if "fineweb_val" in args.ppl_sets:
        out["fineweb_val"] = perplexity(model, val_ids[args.val_skip * args.seq:], args.seq,
                                        max_windows=args.val_tokens // args.seq)
    return out


def evaluate(model, tok, wt_ids, val_ids, args, zs: bool) -> dict:
    """Perplexities served as --serve and under every --eval-modes spec, then zero-shot served as --serve."""
    t0 = time.time()
    ev = ppls(model, wt_ids, val_ids, args, args.serve)
    log("mode", args.serve, json.dumps(ev))
    if args.target != "bf16":
        ev["modes"] = {}
        for m in filter(None, args.eval_modes.split(",")):
            ev["modes"][m] = ppls(model, wt_ids, val_ids, args, m)
            log("mode", m, json.dumps(ev["modes"][m]))
    log("eval", json.dumps({m: {k: round(v["ppl"], 4) for k, v in e.items() if k != "modes"}
                            for m, e in [(args.serve, ev)] + list(ev.get("modes", {}).items())}),
        f"{time.time() - t0:.0f}s")
    served(model, args, args.serve)
    if zs and args.tasks:
        ev["zeroshot"] = zeroshot(model, tok, args.tasks.split(","), args.zs_batch, args.zs_limit)
    return ev


def even_memory(model: str, slack_gib: float = 8) -> dict[int, str] | None:
    """Per visible GPU an even share of a local sharded checkpoint plus slack, so a device map spreads it instead of
    filling GPU 0 first, and every GPU keeps room for `quant.wrap`'s float64 temporaries; None for anything else."""
    idx = Path(model) / "model.safetensors.index.json"
    n = torch.cuda.device_count()
    if not idx.exists() or n == 0:
        return None
    total = json.loads(idx.read_text())["metadata"]["total_size"]
    return {i: f"{math.ceil(total / n / 2**30 + slack_gib)}GiB" for i in range(n)}


def export(model, path: Path) -> dict:
    from safetensors.torch import save_file

    model.eval()
    tensors = {}
    for name, parent, attr in quant.block_linears(model):
        if not isinstance(getattr(parent, attr), quant.QLinear):
            continue
        q, s = getattr(parent, attr).frozen()
        tensors[name[: -len(".weight")] + ".q"] = q.to(torch.int8).cpu().contiguous()
        tensors[name[: -len(".weight")] + ".s"] = s.float().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"file": str(path), "sha256": digest, "bytes": path.stat().st_size, "matrices": len(tensors) // 2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--target", required=True, help="bf16, absmax or a Strassen level-1 overflow rate (0.02 = 2%%)")
    ap.add_argument("--rotate", action="store_true", help="quantize W·R and feed x·R (`rotation.py`)")
    ap.add_argument("--targets", type=Path, default=None,
                    help="a JSON map of tensor name to its own target (a rate, absmax, bf16: left unquantized, or deflR: "
                    "W·R less its top-R SVD part, served apart, at --target; with --device-map), the rest at --target; "
                    "out (tables.py --map's served exact) is absmax, served as --serve says")
    ap.add_argument("--calib", type=Path, default=None,
                    help="an earlier arm's result.json on the same weights and targets (with --device-map): each "
                    "matrix's c from its calibration_before instead of solved again (`quant.wrap`'s calib)")
    ap.add_argument("--device-map", default=None, help="auto: load in bf16 across the visible GPUs (cpu: on the CPU), "
                    "codes frozen (`quant.FrozenQLinear`); evaluation only (--tokens 0)")
    ap.add_argument("--serve", default="a16", help="how activations are served in training, the headline evals and "
                    "zero-shot: MODE[/wF][:X+...][;MODE=X+...]... (`quant.Served`, `quant.set_served`)")
    ap.add_argument("--eval-modes", default=EVAL_MODES, help="more serving specs to measure perplexity under")
    ap.add_argument("--tokens", type=int, default=100_000_000, help="training tokens; 0 = no training")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64, help="sequences a step")
    ap.add_argument("--micro", type=int, default=4, help="sequences a forward")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.03, help="share of steps")
    ap.add_argument("--recal", type=int, default=50, help="steps between solving every matrix's c again")
    ap.add_argument("--lora", type=int, default=0, metavar="R",
                    help="train a rank-R update of each block linear and the RMSNorm gains only (0: every parameter)")
    ap.add_argument("--teacher-device", default="cuda", help="the teacher's device (default the student's)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-tokens", type=int, default=1_048_576)
    ap.add_argument("--ppl-sets", default="wikitext2,fineweb_val", help="the perplexity sets to measure")
    ap.add_argument("--wt-windows", type=int, default=None, help="WikiText-2's first this many windows (default all)")
    ap.add_argument("--wt-skip", type=int, default=0, help="WikiText-2 windows to skip first (as --val-skip)")
    ap.add_argument("--probe", type=int, default=0, help="WikiText-2 windows to run `quant.Probe` over, served as "
                    "--serve, before training")
    ap.add_argument("--val-skip", type=int, default=0, help="FineWeb-Edu val windows to skip first (a model too slow "
                    "for one lease measures --val-tokens in parts and sums their nll)")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("HU_DATA", "data")))
    ap.add_argument("--tasks", default=TASKS, help="lm-eval tasks, comma-separated; empty for perplexity only")
    ap.add_argument("--zs-batch", type=int, default=64)
    ap.add_argument("--zs-limit", type=int, default=None)
    ap.add_argument("--no-zs-before", action="store_true", help="skip zero-shot before training")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps of the --tokens schedule (smoke tests)")
    ap.add_argument("--codes", type=Path, default=None, help="write the codes to this safetensors file")
    ap.add_argument("--price", action="store_true", help="price the exported codes with price.py on the CPU")
    ap.add_argument("--price-workers", type=int, default=6)
    ap.add_argument("--price-args", default="", help="more arguments for price.py")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    assert args.codes or not args.price, "--price prices the --codes file"
    assert args.target != "bf16" or args.serve == "a16", "bf16 has no codes to serve noised"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    meta = json.loads((args.data / "meta.json").read_text())
    if meta["model"] != args.model:
        data_tok = AutoTokenizer.from_pretrained(meta["model"])
        assert (data_tok.get_vocab(), data_tok.bos_token_id, data_tok.eos_token_id) == \
            (tok.get_vocab(), tok.bos_token_id, tok.eos_token_id), \
            f"{args.data} is tokenized by {meta['model']}, whose vocabulary isn't {args.model}'s: prep.py --model it"
    train_ids = np.memmap(args.data / "train.bin", dtype=np.uint32, mode="r")
    val_ids = np.memmap(args.data / "val.bin", dtype=np.uint32, mode="r")
    wt_ids = wikitext_ids(tok)
    res = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "data": meta,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__,
           "threads": torch.get_num_threads(), "wikitext2_tokens": int(len(wt_ids))}
    import transformers

    res["transformers"] = transformers.__version__
    if args.device_map:
        assert args.tokens == 0, "--device-map evaluates only"
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map=args.device_map, max_memory=even_memory(args.model))
        res["device_map"] = {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()}
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                     attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    target = None if args.target in ("absmax", "bf16") else float(args.target)
    targets = None
    if args.targets:
        targets = {k: None if v in ("absmax", "out") else v if str(v).startswith("defl") or v == "bf16" else float(v)
                   for k, v in json.loads(args.targets.read_text()).items()}
        res["targets"] = targets
    calib = None
    if args.calib:
        assert args.device_map, "--calib reuses a frozen wrap's c"
        calib = json.loads(args.calib.read_text())["calibration_before"]
        res["calib"] = {"file": str(args.calib), "sha256": hashlib.sha256(args.calib.read_bytes()).hexdigest()}
    total = 0 if args.target == "bf16" else args.tokens // (args.batch * args.seq)
    steps = total if args.max_steps is None else min(total, args.max_steps)
    res["steps"], res["schedule_steps"] = steps, total
    assert steps * args.batch * args.seq <= len(train_ids), \
        f"{steps} steps need {steps * args.batch * args.seq} tokens, {args.data} has {len(train_ids)}: prep more shards"
    if args.target != "bf16":
        res["calibration_before"] = quant.wrap(model, target, args.rotate, log, targets, frozen=bool(args.device_map),
                                               calib=calib)
        res["matrices_before"] = quant.stats(model)
    res["eval_before"] = evaluate(model, tok, wt_ids, val_ids, args, zs=steps == 0 or not args.no_zs_before)
    if args.probe and args.target != "bf16":
        served(model, args, args.serve)
        quant.PROBE = quant.Probe(model)
        t0 = time.time()
        perplexity(model, wt_ids, args.seq, max_windows=args.probe)
        res["probe"] = {"spec": args.serve, "windows": args.probe, "seconds": time.time() - t0,
                        "matrices": quant.PROBE.summary()}
        quant.PROBE = None
        log("probe", f"{res['probe']['seconds']:.0f}s")
    (args.out / "result.json").write_text(json.dumps(res, indent=1))

    if steps > 0:
        tdev = torch.device(args.teacher_device)
        teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                       attn_implementation="sdpa").to(tdev).eval()
        teacher.config.use_cache = False
        for p in teacher.parameters():
            p.requires_grad_(False)
        if args.lora:
            res["lora_params"] = quant.add_lora(model, args.lora)
            for n, p in model.named_parameters():
                if n.endswith("norm.weight"):
                    p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        res["trained_params"] = sum(p.numel() for p in params)
        log("training", res["trained_params"], "parameters", f"(lora {args.lora})" if args.lora else "(all)")
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8, fused=True)
        warm = max(1, int(args.warmup * total))
        accum = args.batch // args.micro
        curve = [{"step": 0, "fineweb_val_64": perplexity(model, val_ids, args.seq, max_windows=64)["ppl"]}]
        recals = []
        quant.set_served(model, args.serve).reseed(TRAIN_SEED, "cuda")
        model.train()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for step in range(steps):
            lr = args.lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = lr
            tot = 0.0
            for j in range(accum):
                off = (step * args.batch + j * args.micro) * args.seq
                x = torch.from_numpy(train_ids[off:off + args.micro * args.seq].astype(np.int64)
                                     .reshape(args.micro, args.seq)).cuda(non_blocking=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tl = teacher(x.to(tdev)).logits.to(x.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sl = model(x).logits
                loss = KD.apply(sl, tl) / accum
                loss.backward()
                tot += loss.item()
                del tl, sl, loss
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if target is not None and (step + 1) % args.recal == 0:
                info = quant.recalibrate(model, target)
                ovs = sorted(v["strassen1"] for v in info.values())
                recals.append({"step": step + 1, "min": ovs[0], "median": ovs[len(ovs) // 2], "max": ovs[-1]})
            rec = {"step": step + 1, "kd": tot, "lr": lr, "grad_norm": gn,
                   "tok_s": (step + 1) * args.batch * args.seq / (time.time() - t0)}
            if (step + 1) % args.eval_every == 0 or step + 1 == steps:
                rec["fineweb_val_64"] = perplexity(model, val_ids, args.seq, max_windows=64)["ppl"]
            curve.append(rec)
            if (step + 1) % 10 == 0 or "fineweb_val_64" in rec or step < 3:
                log("step", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()}))
        res["train"] = {"curve": curve, "recalibrations": recals, "seconds": time.time() - t0,
                        "tokens": steps * args.batch * args.seq,
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        del teacher, opt
        torch.cuda.empty_cache()
        if args.lora:
            res["train"]["lora_merged"] = quant.merge_lora(model)
        if target is not None:
            res["calibration_after"] = quant.recalibrate(model, target)
    model.eval()
    pricer = None
    if args.target != "bf16":
        res["matrices_after"] = quant.stats(model) if steps > 0 else res["matrices_before"]
        if args.codes:
            res["codes"] = export(model, args.codes)
        if args.price:
            cmd = [sys.executable, str(HERE / "price.py"), "--codes", str(args.codes),
                   "--out", str(args.out / "price"), "--workers", str(args.price_workers), "--full-check",
                   *args.price_args.split()]
            log("price:", " ".join(cmd))
            pricer = subprocess.Popen(cmd, stdout=open(args.out / "price.log", "w"), stderr=subprocess.STDOUT)
    res["eval_after"] = evaluate(model, tok, wt_ids, val_ids, args, zs=True) if steps > 0 else res["eval_before"]
    if pricer is not None:
        res["price_rc"] = pricer.wait()
        log("price rc", res["price_rc"])
    res["seconds"] = time.time() - t_start
    res["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    status = Path("/proc/self/status")
    if status.exists():  # the peak counts a CPU model's checkpoint pages in node RAM (RssShmem), shared and fetched once
        res["rss_kib"] = {k: int(v.split()[0]) for k, v in (ln.split(":", 1) for ln in status.read_text().splitlines())
                          if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem")}
    (args.out / "result.json").write_text(json.dumps(res, indent=1))
    log("done", json.dumps({k: res["eval_after"][k]["ppl"] for k in ("wikitext2", "fineweb_val") if k in res["eval_after"]}))
    return 0 if res.get("price_rc", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
