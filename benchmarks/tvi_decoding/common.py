"""Shared by train.py and predict.py: the model, the corpus (the sweep's: stdlib .py files, hashed split), snapshot paths."""
import glob, hashlib, os, sysconfig
import torch

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
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tr, he = corpus()
    d = {"train": torch.tensor(tok(tr)["input_ids"]), "held": torch.tensor(tok(he)["input_ids"])}
    torch.save(d, path)
    return d


def snap_path(name, step):
    return os.path.join(SCRATCH, f"snap_{TAG}_{name}_s{step}.pt")
