"""Fine-tune MODEL from its pretrained weights under one condition and save fp32 snapshots to $SCRATCH."""
import copy, glob, hashlib, json, os, sys, sysconfig, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRATCH = os.environ["SCRATCH"]
OUT = os.environ["OUT"]
MODEL = os.environ.get("MODEL", "EleutherAI/pythia-160m")
TAG = MODEL.split("/")[-1]
SEQ, BATCH = 256, 8


def corpus():
    lib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(lib, "*.py")) + glob.glob(os.path.join(lib, "*/*.py")))
    files = [f for f in files if "test" not in f and "site-packages" not in f]
    train, held = [], []
    for f in files:
        h = int(hashlib.sha256(os.path.relpath(f, lib).encode()).hexdigest(), 16) % 10
        (held if h == 0 else train).append(open(f, encoding="utf-8", errors="replace").read())
    return "\n\n".join(train), "\n\n".join(held)


def tokens():
    path = os.path.join(SCRATCH, f"tokens_{TAG}.pt")
    if os.path.exists(path):
        return torch.load(path)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tr, he = corpus()
    d = {"train": torch.tensor(tok(tr)["input_ids"]), "held": torch.tensor(tok(he)["input_ids"])}
    torch.save(d, path)
    return d


def run(name, lr, steps, snaps, lora=False, seed=0):
    toks = tokens()["train"]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear"))
    model.to("cuda").train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(toks) - SEQ - 1, (BATCH,), generator=g)
        x = torch.stack([toks[i:i + SEQ] for i in idx]).to("cuda")
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log.append(loss.item())
        if step in snaps:
            if lora:
                m2 = copy.deepcopy(model).to("cpu").merge_and_unload()
                sd = {k: v.detach().clone() for k, v in m2.state_dict().items()}
                nl = sum(p.numel() for p in params)
                del m2
            else:
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                nl = None
            torch.save({"sd": sd, "lora_params": nl}, os.path.join(SCRATCH, f"snap_{TAG}_{name}_s{step}.pt"))
            print(f"{TAG} {name} step {step} loss(avg last 20) {sum(log[-20:]) / len(log[-20:]):.3f} "
                  f"t={time.time() - t0:.0f}s", flush=True)
    json.dump(log, open(os.path.join(OUT, f"loss_{TAG}_{name}.json"), "w"))


CONDITIONS = {
    "ft5e-5": dict(lr=5e-5, steps=1000, snaps={1, 10, 100, 1000}),
    "ft1e-5": dict(lr=1e-5, steps=100, snaps={1, 100}),
    "ft3e-4": dict(lr=3e-4, steps=1000, snaps={100, 1000}),
    "lora": dict(lr=2e-4, steps=1000, snaps={1, 100, 1000}, lora=True),
}

def snaps_override(name, snaps):
    """SNAPS="ft5e-5:100,1000;lora:1000" keeps only those snapshot steps of a condition (training still runs all its steps)."""
    for part in filter(None, os.environ.get("SNAPS", "").split(";")):
        cond, steps = part.split(":")
        if cond == name:
            return {int(s) for s in steps.split(",")}
    return snaps


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "tok":
        print({k: len(v) for k, v in tokens().items()})
    else:
        c = dict(CONDITIONS[which])
        c["snaps"] = snaps_override(which, c["snaps"])
        run(which, **c)
