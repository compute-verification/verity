"""The honest cost of randomized int8 arithmetic: a model's accuracy under W8A8 with every int8 dot exact, and each
randomization the stacking defence S3 considers applied with fresh coins, against deterministic round-to-nearest-even.

The int8 model (plain PyTorch, one GPU). Every decoder linear is W8A8: weights per output channel and activations per
token row, symmetric on ±127, the dot an exact int32 `torch._int_mm`. Attention's two dots are int8 too: Q and K per
(token, head) row in s8, the probabilities in u8 (×255) and V per head in s8, each dot exact (|sum| < 2^24 in fp32).
The softmax, norms and the LM head stay as the model has them. An arm changes only how the int8 codes are rounded, what is
added to the int32 accumulators, or the basis the linears' dots run in:

* `bf16`: the model as published (no int8), through the same eager attention.
* `rtn`: round-to-nearest-even everywhere (the deterministic reference every other arm is paired with).
* `sr`: stochastic rounding of every activation code (Q, K, P, V and every linear input), a fresh uniform coin each.
* `dither`: one uniform offset a row of each requantization (a verifier offset), `⌊y + u⌋`.
* `noise1`: RTN, then ±1 LSB uniform on every activation code (bounded noise that survives integer inputs).
* `accG`: RTN, then a uniform integer in `[−2^(G−1), 2^(G−1))` added to every int32 accumulator of every dot.
* `sr+accG`: both.
* `had`: a fresh random-sign block-Hadamard rotation (block 128, or the largest power of two dividing a narrower
  width) of each linear's input dimension, weights rotated and
  requantized after the coins; `perm`: a fresh signed permutation (exact, so bit-identical to `rtn`).

Metrics: perplexity on the ppl bench's pinned windows (WikiText-2 and C4 at 2048, `benchmarks/ppl/windows.py`), MMLU
5-shot on a fixed subset (the first `--mmlu-per` test questions of each subject, HF `cais/mmlu`), and GSM8K 8-shot
chain-of-thought (the first `--gsm8k-n` test problems, greedy, the standard 8 exemplars). Every result is a line of
`results.jsonl` in `--out`, written as it lands. `--smoke` runs every arm on a few inputs of a small model and checks the
harness's exactness first (the int8 dot against float64, the eager attention against the model's own)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ppl"))

import windows as W  # noqa: E402  (the ppl bench's pinned windows)


@dataclass(frozen=True)
class Arm:
    name: str
    quant: bool = True
    act: str = "rtn"  # rtn | sr | dither | noise
    noise_lsb: int = 0
    acc_bits: int = 0
    rot: str = "none"  # none | hadamard | signperm

    @property
    def random(self) -> bool:
        return self.act != "rtn" or self.acc_bits > 0 or self.rot != "none"


ARMS = {a.name: a for a in [
    Arm("bf16", quant=False), Arm("rtn"), Arm("sr", act="sr"), Arm("dither", act="dither"),
    Arm("noise1", act="noise", noise_lsb=1), Arm("acc4", acc_bits=4), Arm("acc6", acc_bits=6), Arm("acc8", acc_bits=8),
    Arm("acc10", acc_bits=10), Arm("acc12", acc_bits=12), Arm("sr+acc6", act="sr", acc_bits=6),
    Arm("sr+acc8", act="sr", acc_bits=8), Arm("had", rot="hadamard"), Arm("perm", rot="signperm"),
]}


class State:
    arm: Arm = ARMS["bf16"]
    gen: torch.Generator | None = None
    gen_tag: object = None  # a rotated arm's weights are re-drawn when this changes (a new arm or seed)
    stats: dict | None = None  # when a dict, each dot's log2 rms accumulator is recorded under its kind
    attn_calls: int = 0


S = State()


def coins_uniform(shape, device) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=S.gen)


def quantize(y: torch.Tensor, lo: int, hi: int, row_dims: int = 1) -> torch.Tensor:
    """Round `y` (already in code units) to integer codes in `[lo, hi]` under the arm's rounding."""
    a = S.arm
    if a.act == "rtn" or a.act == "noise":
        q = torch.round(y)
        if a.act == "noise":
            q = q + torch.randint(-a.noise_lsb, a.noise_lsb + 1, y.shape, device=y.device, generator=S.gen)
    elif a.act == "sr":
        q = torch.floor(y + coins_uniform(y.shape, y.device))
    elif a.act == "dither":
        q = torch.floor(y + coins_uniform((*y.shape[:-row_dims], *([1] * row_dims)), y.device))
    else:
        raise ValueError(a.act)
    return q.clamp_(lo, hi)


def acc_noise(acc: torch.Tensor) -> torch.Tensor:
    g = S.arm.acc_bits
    if g <= 0:
        return acc
    h = 1 << (g - 1)
    return acc + torch.randint(-h, h, acc.shape, device=acc.device, generator=S.gen, dtype=acc.dtype)


def record(kind: str, acc: torch.Tensor) -> None:
    if S.stats is not None:
        rms = acc.double().pow(2).mean().sqrt().item()
        S.stats.setdefault(kind, []).append(math.log2(max(rms, 1e-30)))


def hadamard(n: int, device) -> torch.Tensor:
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


ROT_BLOCK = 128


class QLinear(nn.Module):
    """A decoder linear, run per the current arm: as published (`bf16`), or W8A8 with an exact int32 dot."""

    def __init__(self, lin: nn.Linear, name: str):
        super().__init__()
        self.lin, self.name = lin, name
        self.k = lin.in_features
        self._q: dict = {}

    def weights(self) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """The int8 weights (n, k), their per-channel scales (n, 1), and the rotation they were taken in."""
        a = S.arm
        if a.rot == "none":
            if "rtn" not in self._q:
                self._q["rtn"] = (*quant_weight(self.lin.weight.float()), None)
            return self._q["rtn"]
        if self._q.get("rot_for") is not S.gen_tag:
            dev = self.lin.weight.device
            sign = torch.randint(0, 2, (self.k,), device=dev, generator=S.gen).float() * 2 - 1
            rot = {"sign": sign}
            if a.rot == "signperm":
                rot["perm"] = torch.randperm(self.k, device=dev, generator=S.gen)
            w = rotate(self.lin.weight.float(), rot)
            self._q["rot"] = (*quant_weight(w), rot)
            self._q["rot_for"] = S.gen_tag
        return self._q["rot"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = S.arm
        if not a.quant:
            return self.lin(x)
        wq, ws, rot = self.weights()
        shape = x.shape
        x2 = x.reshape(-1, self.k).float()
        if rot is not None:
            x2 = rotate(x2, rot)
        sx = x2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        xq = quantize(x2 / sx, -128, 127).to(torch.int8)
        acc = int_mm(xq, wq)
        record("linear", acc)
        acc = acc_noise(acc)
        out = acc.float() * sx * ws.t()
        if self.lin.bias is not None:
            out = out + self.lin.bias.float()
        return out.to(x.dtype).reshape(*shape[:-1], -1)


def quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ws = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
    return torch.round(w / ws).clamp_(-127, 127).to(torch.int8).contiguous(), ws


def rotate(x: torch.Tensor, rot: dict) -> torch.Tensor:
    """`x Q` along the last dimension: a signed permutation, or signs then block Hadamard (Q orthogonal, so
    `(x Q)(W Q)ᵀ = x Wᵀ` in exact arithmetic)."""
    if "perm" in rot:
        return x[..., rot["perm"]] * rot["sign"]
    k = x.shape[-1]
    blk = min(ROT_BLOCK, k & -k)
    xs = (x * rot["sign"]).reshape(*x.shape[:-1], k // blk, blk)
    return (xs @ hadamard(blk, x.device)).reshape(x.shape)


def int_mm(xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """`xq @ wqᵀ` exactly in int32 (cuBLASLt's int8 GEMM), padding rows to its shape rules."""
    m = xq.shape[0]
    mp = max(32, -(-m // 8) * 8)
    if mp != m:
        xq = torch.cat([xq, xq.new_zeros(mp - m, xq.shape[1])])
    return torch._int_mm(xq, wq.t())[:m]


def repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention (fp32 softmax) with both dots int8 and exact under `quant`, in the model's own signature."""
    a = S.arm
    S.attn_calls += 1
    groups = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    k = repeat_kv(key, groups).float()
    v = repeat_kv(value, groups).float()
    q = query.float()
    if a.quant:
        qs = q.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
        ks = k.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
        qc = quantize(q / qs, -128, 127)
        kc = quantize(k / ks, -128, 127)
        acc = torch.matmul(qc, kc.transpose(-1, -2))
        record("qk", acc)
        acc = acc_noise(acc)
        scores = acc * qs * ks.transpose(-1, -2) * scaling
    else:
        scores = torch.matmul(q, k.transpose(-1, -2)) * scaling
    T, Sk = scores.shape[-2], scores.shape[-1]
    if attention_mask is not None:
        m = attention_mask[..., :Sk]
        if m.dtype == torch.bool:
            scores = scores.masked_fill(~m, float("-inf"))
        else:
            scores = scores + m.float()
    elif T > 1:
        causal = torch.ones(T, Sk, dtype=torch.bool, device=scores.device).tril(Sk - T)
        scores = scores.masked_fill(~causal, float("-inf"))
    p = torch.softmax(scores, dim=-1, dtype=torch.float32)
    if a.quant:
        pc = quantize(p * 255, 0, 255)
        vs = v.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12) / 127
        vc = quantize(v / vs, -128, 127, row_dims=2)
        acc = torch.matmul(pc, vc)
        record("pv", acc)
        acc = acc_noise(acc)
        out = acc * (vs / 255)
    else:
        out = torch.matmul(p, v)
    return out.to(query.dtype).transpose(1, 2).contiguous(), None


def install(model) -> int:
    """Wrap every decoder linear and route attention through `attention`; returns the linears wrapped."""
    import importlib
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and ".layers." in f"{name}.{cname}":
                setattr(mod, cname, QLinear(child, f"{name}.{cname}"))
                n += 1
    mt = model.config.model_type
    mm = importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")
    mm.eager_attention_forward = attention
    model.config._attn_implementation = "eager"
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    return n


def set_arm(arm: Arm, seed: int) -> None:
    S.arm = arm
    S.gen = torch.Generator(device="cuda")
    tag = int.from_bytes(hashlib.sha256(f"{arm.name}/{seed}".encode()).digest()[:8], "little") >> 1
    S.gen.manual_seed(tag)
    S.gen_tag = object()


# ---- evaluations ------------------------------------------------------------------------------------------------------


@torch.no_grad()
def perplexity(model, wins: torch.Tensor) -> dict:
    nll, n, per = 0.0, 0, []
    for w in wins:
        ids = w[None].cuda()
        logits = model(ids).logits.float()[0, :-1]
        loss = F.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
        nll += loss
        n += ids.shape[1] - 1
        per.append(loss / (ids.shape[1] - 1))
    return {"ppl": math.exp(nll / n), "tokens": n, "window_nll": per}


MMLU_API = "https://datasets-server.huggingface.co/rows?dataset=cais/mmlu&config=all&split={split}&offset={off}&length=100"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def fetch(url: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise AssertionError


def mmlu_rows(split: str, data: Path) -> list[dict]:
    path = data / f"mmlu-{split}.json"
    if not path.exists():
        rows, off = [], 0
        while True:
            page = json.loads(fetch(MMLU_API.format(split=split, off=off)))
            got = [r["row"] for r in page["rows"]]
            rows += got
            off += len(got)
            if not got or off >= page.get("num_rows_total", off):
                break
        path.write_text(json.dumps(rows))
    return json.loads(path.read_text())


def gsm8k_rows(data: Path) -> list[dict]:
    path = data / "gsm8k-test.jsonl"
    if not path.exists():
        path.write_bytes(fetch(GSM8K_URL))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mmlu_prompts(test: list[dict], dev: list[dict], per: int) -> list[tuple[str, int]]:
    shots: dict[str, list[dict]] = {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    taken: dict[str, int] = {}
    out = []
    for r in test:
        s = r["subject"]
        if taken.get(s, 0) >= per:
            continue
        taken[s] = taken.get(s, 0) + 1
        head = f"The following are multiple choice questions (with answers) about {s.replace('_', ' ')}.\n\n"
        body = "".join(fmt_mmlu(x) + f" {'ABCD'[x['answer']]}\n\n" for x in shots[s][:5])
        out.append((head + body + fmt_mmlu(r), r["answer"]))
    return out


def fmt_mmlu(r: dict) -> str:
    return r["question"].strip() + "\n" + "".join(f"{'ABCD'[i]}. {c}\n" for i, c in enumerate(r["choices"])) + "Answer:"


@torch.no_grad()
def mmlu(model, tok, prompts: list[tuple[str, int]], batch: int) -> dict:
    letters = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i][0]))
    right = [0] * len(prompts)
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        enc = tok([prompts[i][0] for i in idx], return_tensors="pt", padding=True).to("cuda")
        logits = model(**enc).logits[:, -1, letters].float()
        for j, i in enumerate(idx):
            right[i] = int(logits[j].argmax().item() == prompts[i][1])
    return {"acc": sum(right) / len(right), "n": len(right), "right": right}


GSM8K_SHOTS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there "
     "will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been "
     "21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they "
     "had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason "
     "give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The "
     "answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The "
     "answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to "
     "thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers "
     "were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls "
     "did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he "
     "had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars "
     "left. 23 - 15 is 8. The answer is 8."),
]
GSM8K_HEAD = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in GSM8K_SHOTS)
NUM = re.compile(r"-?[\d,]*\.?\d+")


def gsm8k_answer(text: str) -> str | None:
    text = text.split("\n\nQ:")[0].split("\nQ:")[0]
    m = re.search(r"The answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    hit = m.group(1) if m else (NUM.findall(text) or [None])[-1]
    return None if hit is None else hit.replace(",", "").rstrip(".")


def same_number(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


@torch.no_grad()
def gsm8k(model, tok, rows: list[dict], batch: int, max_new: int) -> dict:
    right, outs = [], []
    for b0 in range(0, len(rows), batch):
        rs = rows[b0:b0 + batch]
        enc = tok([GSM8K_HEAD + f"Q: {r['question']}\nA:" for r in rs], return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                             stop_strings=["\n\nQ:", "\nQ:"], tokenizer=tok)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(rs, texts):
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            got = gsm8k_answer(t)
            right.append(int(same_number(got, gold)))
            outs.append(got)
    return {"acc": sum(right) / len(right), "n": len(right), "right": right, "answers": outs}


# ---- the run ----------------------------------------------------------------------------------------------------------


def load(snap: Path):
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(str(snap))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.bfloat16,
                                                              attn_implementation="eager").cuda().eval()
    return tok, model


def checks(model, tok) -> dict:
    """The harness's exactness: the int8 dot against float64, and the eager attention against the model's own at bf16."""
    torch.manual_seed(0)
    a = torch.randint(-128, 128, (40, 4096), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 128, (1024, 4096), dtype=torch.int8, device="cuda")
    exact = torch.equal(int_mm(a, b).double(), a.double() @ b.double().t())
    ids = tok("The quick brown fox jumps over the lazy dog. " * 20, return_tensors="pt")["input_ids"][:, :128].cuda()
    set_arm(ARMS["bf16"], 0)
    S.attn_calls = 0
    with torch.no_grad():
        ours = model(ids).logits.float()
    calls = S.attn_calls
    import importlib
    mt = model.config.model_type
    mm = importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")
    mm.eager_attention_forward = ORIG_EAGER[mt]
    with torch.no_grad():
        ref = model(ids).logits.float()
    mm.eager_attention_forward = attention
    dev = (ours - ref).abs().max().item()
    agree = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
    return {"int_mm_exact": exact, "attention_calls": calls, "eager_vs_model_max_abs_logit": dev,
            "eager_vs_model_top1_agree": agree, "eager_ok": calls > 0 and agree >= 0.9}


ORIG_EAGER: dict = {}


def emit(out: Path, rec: dict) -> None:
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    short = {k: v for k, v in rec.items() if k not in ("window_nll", "right", "answers")}
    print(json.dumps(short), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--family", default="", help="the ppl lock's tokenizer family (llama3), to check the windows")
    ap.add_argument("--data", type=Path, required=True, help="the pinned ppl sets; MMLU and GSM8K are fetched here")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=3, help="seeds a random arm's perplexity runs")
    ap.add_argument("--evals", default="wikitext2,c4,mmlu,gsm8k")
    ap.add_argument("--task-arms", default="", help="arms MMLU runs (default every arm)")
    ap.add_argument("--gen-arms", default="", help="arms GSM8K runs (default MMLU's)")
    ap.add_argument("--mmlu-per", type=int, default=20)
    ap.add_argument("--gsm8k-n", type=int, default=500)
    ap.add_argument("--gsm8k-max-new", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="a few inputs each, and the exactness checks")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok, model = load(a.snapshot)
    import importlib
    mt = model.config.model_type
    ORIG_EAGER[mt] = importlib.import_module(f"transformers.models.{mt}.modeling_{mt}").eager_attention_forward
    n_lin = install(model)
    meta = {"kind": "meta", "snapshot": str(a.snapshot), "model_type": mt, "linears": n_lin,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    import transformers
    meta["transformers"] = transformers.__version__
    meta.update(checks(model, tok))
    emit(a.out, meta)
    if not (meta["int_mm_exact"] and meta["eager_ok"]):
        print("exactness checks failed", file=sys.stderr)
        return 1
    arms = [ARMS[x] for x in a.arms.split(",")]
    task_arms = [ARMS[x] for x in a.task_arms.split(",")] if a.task_arms else arms
    gen_arms = [ARMS[x] for x in a.gen_arms.split(",")] if a.gen_arms else task_arms
    evals = a.evals.split(",")
    encode, bos = W.encoder(tok)
    for name in ("wikitext2", "c4"):
        if name not in evals:
            continue
        wins, rec = W.build(name, 2048, encode, bos, a.data)
        if a.family:
            W.check(a.family, name, 2048, wins)
        wins = torch.from_numpy(wins)
        if a.smoke:
            wins = wins[:1, :256]
        emit(a.out, {"kind": "windows", "set": name, **rec})
        for arm in arms:
            for seed in range(a.seeds if arm.random else 1):
                set_arm(arm, seed)
                S.stats = {} if (arm.name == "rtn" and name == "wikitext2") else None
                t0 = time.time()
                r = perplexity(model, wins)
                rec = {"kind": "ppl", "set": name, "ctx": int(wins.shape[1]), "arm": arm.name, "seed": seed, **r,
                       "seconds": round(time.time() - t0, 1)}
                if S.stats is not None:
                    rec["log2_rms_acc"] = {k: sorted(v)[len(v) // 2] for k, v in S.stats.items()}
                    rec["log2_rms_acc_p10_p90"] = {k: (sorted(v)[len(v) // 10], sorted(v)[(9 * len(v)) // 10])
                                                   for k, v in S.stats.items()}
                S.stats = None
                emit(a.out, rec)
    if "mmlu" in evals:
        test, dev = mmlu_rows("test", a.data), mmlu_rows("dev", a.data)
        prompts = mmlu_prompts(test, dev, 1 if a.smoke else a.mmlu_per)[: 8 if a.smoke else None]
        emit(a.out, {"kind": "data", "set": "mmlu", "test_sha256": sha256_file(a.data / "mmlu-test.json"),
                     "dev_sha256": sha256_file(a.data / "mmlu-dev.json"), "questions": len(prompts)})
        for arm in task_arms:
            set_arm(arm, 0)
            t0 = time.time()
            r = mmlu(model, tok, prompts, 8)
            emit(a.out, {"kind": "mmlu", "arm": arm.name, "seed": 0, **r, "seconds": round(time.time() - t0, 1)})
    if "gsm8k" in evals:
        rows = gsm8k_rows(a.data)[: 4 if a.smoke else a.gsm8k_n]
        emit(a.out, {"kind": "data", "set": "gsm8k", "sha256": sha256_file(a.data / "gsm8k-test.jsonl"),
                     "problems": len(rows)})
        for arm in gen_arms:
            set_arm(arm, 0)
            t0 = time.time()
            r = gsm8k(model, tok, rows, 4 if a.smoke else 64, 64 if a.smoke else a.gsm8k_max_new)
            emit(a.out, {"kind": "gsm8k", "arm": arm.name, "seed": 0, **r, "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
