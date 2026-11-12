"""Channel swaps that bring a Llama checkpoint's massive-activation rows into Pearl+'s forming domain, by reparametrizing
its weights so the function is unchanged.

Pearl+ credits a row of a linear's input only inside TT_OUT's domain (`pearl_c`'s docstring, "The domain"): the noise floor
ρ·α ≥ 1, that is s ≤ 444·ρ, and liveness (`pearl_c.live_row`), with ρ the RMS of the row's entries at positions ≡ 0 (mod
8). A Llama's attention-sink rows (BOS, and a few tokens like it) carry a handful of massive channels; where those sit off
the stride, ρ misses them and the row fails. Swapping each such channel with a quiet channel on the stride, in every
weight that reads or writes it, computes the same function and puts the channel inside ρ.

Three kinds of channel space, each swapped consistently:

- **hidden** (one for the model): the residual stream. A swap c ↔ c′ permutes the embedding's columns (the tied LM head's
  too, or `lm_head`'s), every RMSNorm weight, the input columns of q, k, v, gate and up, and the output rows of o and
  down.
- **intermediate L**: down's input in layer L. A swap permutes gate's and up's output rows and down's input columns.
- **attention L**: o's input in layer L, channel h·d_h + d of q head h. A swap of dims d ↔ d′ within kv head g permutes
  v's output rows g·d_h + d ↔ g·d_h + d′ and o's input columns h·d_h + d ↔ h·d_h + d′ for each q head h of g (attention's
  output is equivariant in V's dims; RoPE touches q and k only). Since d_h ≡ 0 (mod 8), d′ ≡ 0 (mod 8) puts every such
  channel on the stride.

The swaps come from a public calibration forward (`CALIBRATION`, public-domain text, BOS first, run in BF16 on the CPU),
never from served values, by one pinned rule (`search`): while some calibration row of a space fails the per-row rules,
take the off-stride unit with the largest |x|/ρ over the failing rows and swap it with the on-stride unit of that space
that is quietest over all its rows; stop when every row passes (refuse past `MAX_SWAPS` a space). The rows are screened in
float64, and a row within a band of a threshold is decided exactly (`pearl_c.s5_stats`, `pearl_c.live_row`).

    python channel_swaps.py make --out DIR [--checkpoint SNAPSHOT | --rev REV] [--report F]
        the swaps (DIR/channel_swaps.json), the permuted snapshot (DIR, a copy of the checkpoint's files with its
        model.safetensors permuted) and the logit check (`check`)
    python channel_swaps.py check --original SNAPSHOT --permuted SNAPSHOT [--report F]
        the two snapshots' logits on the calibration and a held-out public text, in float64 and in BF16
    python channel_swaps.py screen-case CASE --swaps F [--out F]
        the per-row rules on a served Pearl+ case's rows (``served_passes.py case``'s parts.json and blobs) with the
        swaps applied: counts only

The float64 check can't be bit-exact: a swap reorders the sums over the swapped dimension (RMSNorm's mean of squares, each
linear's accumulation over k), and floating-point addition isn't associative. In float64 the logits differ only by that
reordering's rounding, so `check` holds them to `F64_TOL` of the largest logit and prints the BF16 forward's
difference and its greedy tokens beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.prover.schemes import pearl_kw as P

REPO = "unsloth/Llama-3.2-1B"
REV = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
FILES = ("config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json")
SCHEMA = "verity-pouw/pearl-plus-channel-swaps/v1"
MAX_SWAPS = 32
F64_TOL = 1e-9
STRIDE = C.S5_STRIDE
KINDS = ("qkv", "o", "gate_up", "down")  # a served pass's weights, four a layer in this order (weight index = 4·L + kind)

#: Public-domain text: the Gettysburg Address (Bliss copy), in its three paragraphs, and the Preamble of the US
#: Constitution, each a prompt of its own after BOS.
CALIBRATION = (
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and "
    "dedicated to the proposition that all men are created equal.",
    "Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, "
    "can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, "
    "as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting "
    "and proper that we should do this.",
    "But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave "
    "men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The "
    "world will little note, nor long remember what we say here, but it can never forget what they did here. It is for "
    "us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so "
    "nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these "
    "honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that "
    "we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a "
    "new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from "
    "the earth.",
    "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic "
    "Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to "
    "ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.",
)
#: The held-out text of the logit check (public domain: the opening of A Tale of Two Cities), which no swap reads.
HELD_OUT = ("It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, "
            "it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season "
            "of Darkness, it was the spring of hope, it was the winter of despair.")


# ---- the checkpoint ---------------------------------------------------------------------------------------------------

def fetch(rev: str, out: Path) -> Path:
    """REPO's files at `rev` from the Hugging Face hub into `out` (the snapshot), each kept as downloaded."""
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (out / name).exists():
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{rev}/{name}"
        tmp = out / (name + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 24)
        tmp.rename(out / name)
    return out


def read_safetensors(path: Path) -> tuple[bytearray, dict, int]:
    """(the file's bytes, its header, the data's offset)."""
    data = bytearray(path.read_bytes())
    (n,) = struct.unpack_from("<Q", data, 0)
    return data, json.loads(data[8:8 + n]), 8 + n


def tensors(data: bytearray, header: dict, base: int) -> dict:
    """Each tensor of a BF16 safetensors file as a torch view on `data` (writes go to the file's bytes)."""
    import torch
    out = {}
    for name, h in header.items():
        if name == "__metadata__":
            continue
        if h["dtype"] != "BF16":
            raise ValueError(f"{name}: {h['dtype']}, not BF16")
        lo, hi = h["data_offsets"]
        out[name] = torch.frombuffer(memoryview(data)[base + lo:base + hi], dtype=torch.bfloat16).view(h["shape"])
    return out


# ---- the tokenizer (byte-level BPE, ASCII text) -------------------------------------------------------------------------

#: Llama 3's pre-tokenizer pattern with \p{L} and \p{N} as their ASCII classes, exact on ASCII text
_PRE = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]{1,3}| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|"
                  r"\s+(?!\S)|\s+")


def _byte_chars() -> dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    def __init__(self, path: Path):
        doc = json.loads(path.read_text())
        model = doc["model"]
        self.vocab: dict[str, int] = model["vocab"]
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model["merges"]]
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.ignore_merges = bool(model.get("ignore_merges"))
        self.bos = next(t["id"] for t in doc["added_tokens"] if t["content"] == "<|begin_of_text|>")
        self.chars = _byte_chars()
        self.inv = {v: k for k, v in self.vocab.items()}

    def _bpe(self, word: str) -> list[str]:
        if self.ignore_merges and word in self.vocab:
            return [word]
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((a, b), math.inf), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            rank, i = min(pairs)
            if rank == math.inf:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return parts

    def encode(self, text: str) -> list[int]:
        """BOS, then the text's tokens; refuses text that isn't ASCII or doesn't decode back to itself."""
        if not text.isascii():
            raise ValueError("the tokenizer here is exact on ASCII text only")
        ids = [self.bos]
        for piece in _PRE.findall(text):
            ids += [self.vocab[p] for p in self._bpe("".join(self.chars[b] for b in piece.encode()))]
        back = {c: b for b, c in self.chars.items()}
        if bytes(back[c] for i in ids[1:] for c in self.inv[i]).decode() != text:
            raise AssertionError("the tokens don't decode back to the text")
        return ids


# ---- the model ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    inter: int
    eps: float
    theta: float
    scaling: dict | None
    tied: bool

    @classmethod
    def of(cls, config: dict) -> Shape:
        heads = config["num_attention_heads"]
        return cls(config["num_hidden_layers"], config["hidden_size"], heads, config["num_key_value_heads"],
                   config.get("head_dim") or config["hidden_size"] // heads, config["intermediate_size"],
                   config["rms_norm_eps"], config["rope_theta"], config.get("rope_scaling"),
                   bool(config.get("tie_word_embeddings")))


def inv_freq(shape: Shape):
    """RoPE's inverse frequencies, with Llama 3's scaling when the config names it."""
    import torch
    d = shape.head_dim
    f = 1.0 / (shape.theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    sc = shape.scaling
    if not sc or sc.get("rope_type", sc.get("type")) != "llama3":
        return f
    factor, lo, hi, old = sc["factor"], sc["low_freq_factor"], sc["high_freq_factor"], sc["original_max_position_embeddings"]
    wavelen = 2 * math.pi / f
    scaled = torch.where(wavelen > old / lo, f / factor, f)
    smooth = (old / wavelen - lo) / (hi - lo)
    medium = (wavelen >= old / hi) & (wavelen <= old / lo)
    return torch.where(medium, (1 - smooth) * scaled / factor + smooth * scaled, scaled)


def forward(w: dict, shape: Shape, ids: list[int], dtype, capture: dict | None = None):
    """The logits of a Llama on one prompt (rows = positions), its weights upcast to `dtype` as they are read; with
    `capture`, each linear's input rows (BF16 values as float32) by (kind, layer): ``qkv``, ``o``, ``gate_up``, ``down``,
    and ``lm_head`` (layer -1)."""
    import torch
    W = lambda name: w[name].to(dtype)  # noqa: E731
    T, hd = len(ids), shape.head_dim

    def keep(kind: str, layer: int, x) -> None:
        if capture is not None:
            capture.setdefault((kind, layer), []).append(x.to(torch.bfloat16).float().numpy())

    def norm(x, name):
        v = x.to(torch.float32 if dtype == torch.bfloat16 else dtype)
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + shape.eps)
        return W(name) * v.to(dtype)

    ang = torch.arange(T, dtype=torch.float64)[:, None] * inv_freq(shape)[None, :]
    cos, sin = (torch.cat([f(ang), f(ang)], -1).to(dtype) for f in (torch.cos, torch.sin))

    def rope(x):
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    mask = torch.full((T, T), float("-inf"), dtype=dtype).triu(1)
    x = W("model.embed_tokens.weight")[torch.tensor(ids)]
    group = shape.heads // shape.kv_heads
    for L in range(shape.layers):
        p = f"model.layers.{L}."
        h = norm(x, p + "input_layernorm.weight")
        keep("qkv", L, h)
        q = (h @ W(p + "self_attn.q_proj.weight").T).view(T, shape.heads, hd).transpose(0, 1)
        k = (h @ W(p + "self_attn.k_proj.weight").T).view(T, shape.kv_heads, hd).transpose(0, 1)
        v = (h @ W(p + "self_attn.v_proj.weight").T).view(T, shape.kv_heads, hd).transpose(0, 1)
        q, k = rope(q), rope(k)
        k, v = k.repeat_interleave(group, 0), v.repeat_interleave(group, 0)
        scores = (q @ k.transpose(1, 2)) / math.sqrt(hd) + mask
        att = torch.softmax(scores.to(torch.float32) if dtype == torch.bfloat16 else scores, -1).to(dtype)
        o = (att @ v).transpose(0, 1).reshape(T, shape.heads * hd)
        keep("o", L, o)
        x = x + o @ W(p + "self_attn.o_proj.weight").T
        h = norm(x, p + "post_attention_layernorm.weight")
        keep("gate_up", L, h)
        a = torch.nn.functional.silu(h @ W(p + "mlp.gate_proj.weight").T) * (h @ W(p + "mlp.up_proj.weight").T)
        keep("down", L, a)
        x = x + a @ W(p + "mlp.down_proj.weight").T
    h = norm(x, "model.norm.weight")
    keep("lm_head", -1, h)
    return h @ W("lm_head.weight" if not shape.tied and "lm_head.weight" in w else "model.embed_tokens.weight").T


# ---- the per-row rules ----------------------------------------------------------------------------------------------------

def exact_passes(row: np.ndarray) -> bool:
    """`clear_work.row_passes` on one row of BF16 values (as float32): the noise floor and `pearl_c.live_row`."""
    words = [int(u) for u in row.astype(np.float32).view(np.uint32)]
    _, rho, alpha, _ = C.s5_stats(words)
    return P.f32_of_bits(rho) * P.f32_of_bits(alpha) >= P.SIGMA_MIN and C.live_row(words)


def failing(x: np.ndarray) -> np.ndarray:
    """Which rows of `x` (rows × k, BF16 values) fail the per-row rules: float64, the rows within 1e-6 of a threshold
    decided by `exact_passes`."""
    k = x.shape[1]
    d = x.astype(np.float64)
    rho2 = (d[:, ::STRIDE] ** 2).sum(1) / len(range(0, k, STRIDE))
    rho = np.sqrt(rho2)
    ra = rho * 448 / np.maximum(np.abs(d).max(1) + 4 * rho, 2.0 ** -32)
    sq, thr = d * d, 64 * rho2[:, None]
    dead = (sq >= thr).sum(1)
    band = (np.abs(sq - thr) <= 1e-6 * thr).sum(1)
    fail = (ra < 1) | (64 * dead > k)
    near = (np.abs(ra - 1) < 1e-6) | ((band > 0) & (64 * (dead - band) <= k) & (64 * (dead + band) > k))
    for r in np.nonzero(near)[0]:
        fail[r] = not exact_passes(x[r])
    return fail


# ---- the swaps ------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Space:
    """A channel space: `units` of `width` channels each (unit u = (g, d) is channels base(h) + d for each h of group g)."""
    kind: str          # hidden | intermediate | attention
    layer: int         # -1 for hidden
    k: int
    groups: int        # attention: kv heads; else 1
    per: int           # attention: q heads a kv head; else 1
    dims: int          # attention: head_dim; else k

    def channels(self, g: int, d: int) -> list[int]:
        if self.kind != "attention":
            return [d]
        return [(g * self.per + j) * self.dims + d for j in range(self.per)]


def perm_of(space: Space, swaps: list[tuple[int, int, int]]) -> np.ndarray:
    """The column permutation of `space`'s rows under `swaps` ((g, d, d′) each): column i of the permuted row is column
    p[i] of the original."""
    p = np.arange(space.k)
    for g, a, b in swaps:
        for ca, cb in zip(space.channels(g, a), space.channels(g, b)):
            p[ca], p[cb] = p[cb], p[ca]
    return p


def search(space: Space, rows: np.ndarray) -> tuple[list[tuple[int, int, int]], int]:
    """The pinned rule (module docstring) on `space`'s calibration rows: (its swaps, the rows failing before them)."""
    swaps: list[tuple[int, int, int]] = []
    moved: set[tuple[int, int]] = set()
    before = int(failing(rows).sum())
    for _ in range(MAX_SWAPS + 1):
        x = rows[:, perm_of(space, swaps)]
        bad = failing(x)
        if not bad.any():
            return swaps, before
        if len(swaps) == MAX_SWAPS:
            break
        d = np.abs(x.astype(np.float64))
        rho = np.maximum(np.sqrt((d[:, ::STRIDE] ** 2).mean(1)), 2.0 ** -126)
        loud = (d[bad] / rho[bad, None]).max(0)
        quiet = d.max(0)
        best = max(((max(loud[c] for c in space.channels(g, u)), g, u) for g in range(space.groups)
                    for u in range(space.dims) if u % STRIDE and (g, u) not in moved), default=None)
        if best is None:
            break
        _, g, u = best
        _, t = min((max(quiet[c] for c in space.channels(g, t)), t) for t in range(0, space.dims, STRIDE)
                   if (g, t) not in moved)
        swaps.append((g, u, t))
        moved |= {(g, u), (g, t)}
    raise RuntimeError(f"{space.kind} {space.layer}: {int(failing(rows[:, perm_of(space, swaps)]).sum())} calibration rows "
                       f"still fail after {len(swaps)} swaps (MAX_SWAPS {MAX_SWAPS})")


def spaces(shape: Shape) -> list[Space]:
    out = [Space("hidden", -1, shape.hidden, 1, 1, shape.hidden)]
    for L in range(shape.layers):
        out.append(Space("intermediate", L, shape.inter, 1, 1, shape.inter))
        out.append(Space("attention", L, shape.heads * shape.head_dim, shape.kv_heads, shape.heads // shape.kv_heads,
                         shape.head_dim))
    return out


def rows_of(space: Space, cap: dict) -> np.ndarray:
    """The calibration rows a space's swaps are chosen on: the hidden space's are q/k/v's and gate/up's inputs in every
    layer (the LM head's input is screened, not searched on); an intermediate space's, down's; an attention space's, o's."""
    if space.kind == "hidden":
        keys = [(kind, L) for L in range(max(l for _, l in cap) + 1) for kind in ("qkv", "gate_up")]
    else:
        keys = [({"intermediate": "down", "attention": "o"}[space.kind], space.layer)]
    return np.concatenate([a for key in keys for a in cap[key]])


def calibrate(w: dict, shape: Shape, tok: Tokenizer) -> dict:
    """The swaps of every space from the calibration forward, and the screen of every linear's rows before and after."""
    import torch
    cap: dict = {}
    with torch.no_grad():
        for text in CALIBRATION:
            forward(w, shape, tok.encode(text), torch.bfloat16, cap)
    found = {}
    for sp in spaces(shape):
        swaps, before = search(sp, rows_of(sp, cap))
        if swaps or before:
            found[f"{sp.kind}:{sp.layer}"] = {"kind": sp.kind, "layer": sp.layer, "failing_before": before,
                                              "swaps": [list(s) for s in swaps]}
    doc = {"schema": SCHEMA, "repo": REPO, "calibration_sha256": hashlib.sha256("\n".join(CALIBRATION).encode()).hexdigest(),
           "tokens": sum(len(tok.encode(t)) for t in CALIBRATION), "max_swaps": MAX_SWAPS,
           "shape": shape.__dict__, "spaces": found}
    doc["swap_count"] = sum(len(s["swaps"]) for s in found.values())
    doc["screen"] = screen_calibration(cap, shape, doc)
    return doc


def column_perm(doc: dict, shape: Shape, kind: str, layer: int) -> np.ndarray:
    """The permutation of the input columns of linear `kind` in `layer` (``lm_head``: layer -1) under `doc`'s swaps."""
    sp = {s.kind + ":" + str(s.layer): s for s in spaces(shape)}
    name = "hidden:-1" if kind in ("qkv", "gate_up", "lm_head") else \
        f"{'intermediate' if kind == 'down' else 'attention'}:{layer}"
    return perm_of(sp[name], [tuple(s) for s in doc["spaces"].get(name, {"swaps": []})["swaps"]])


def screen_calibration(cap: dict, shape: Shape, doc: dict) -> dict:
    """Each linear's calibration rows failing before and after the swaps; ``total_after`` counts the linears Pearl+
    credits (`KINDS`), and the LM head's input is listed beside them."""
    out = {}
    for (kind, L), parts in sorted(cap.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        x = np.concatenate(parts)
        before, after = int(failing(x).sum()), int(failing(x[:, column_perm(doc, shape, kind, L)]).sum())
        if before or after:
            out[f"{kind}:{L}"] = {"rows": len(x), "failing_before": before, "failing_after": after}
    out["total_after"] = sum(v["failing_after"] for key, v in out.items() if not key.startswith("lm_head"))
    return out


def apply(w: dict, shape: Shape, doc: dict) -> None:
    """`doc`'s swaps on the weights in place (BF16 views on the file's bytes): every weight that reads or writes each
    swapped channel (module docstring)."""
    def cols(name, a, b):
        t = w[name]
        t[:, [a, b]] = t[:, [b, a]].clone()

    def rows(name, a, b):
        t = w[name]
        t[[a, b]] = t[[b, a]].clone()

    hd, per = shape.head_dim, shape.heads // shape.kv_heads
    for s in doc["spaces"].values():
        p = f"model.layers.{s['layer']}."
        for g, a, b in s["swaps"]:
            if s["kind"] == "hidden":
                cols("model.embed_tokens.weight", a, b)
                if "lm_head.weight" in w:
                    cols("lm_head.weight", a, b)
                for name in [f"model.layers.{l}.{n}" for l in range(shape.layers)
                             for n in ("input_layernorm.weight", "post_attention_layernorm.weight")] + ["model.norm.weight"]:
                    t = w[name]
                    t[[a, b]] = t[[b, a]].clone()
                for l in range(shape.layers):
                    q = f"model.layers.{l}."
                    for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"):
                        cols(q + n + ".weight", a, b)
                    for n in ("self_attn.o_proj", "mlp.down_proj"):
                        rows(q + n + ".weight", a, b)
            elif s["kind"] == "intermediate":
                rows(p + "mlp.gate_proj.weight", a, b)
                rows(p + "mlp.up_proj.weight", a, b)
                cols(p + "mlp.down_proj.weight", a, b)
            else:
                rows(p + "self_attn.v_proj.weight", g * hd + a, g * hd + b)
                for h in range(g * per, (g + 1) * per):
                    cols(p + "self_attn.o_proj.weight", h * hd + a, h * hd + b)


# ---- the logit check ------------------------------------------------------------------------------------------------------

def load(snapshot: Path):
    data, header, base = read_safetensors(snapshot / "model.safetensors")
    return data, header, base, tensors(data, header, base), Shape.of(json.loads((snapshot / "config.json").read_text()))


def logits_check(orig: dict, perm: dict, shape: Shape, tok: Tokenizer) -> dict:
    """Both models' logits on each calibration prompt and the held-out text, float64 and BF16."""
    import torch
    out = {"f64_tol": F64_TOL, "texts": []}
    with torch.no_grad():
        for name, text in [*((f"calibration {i}", t) for i, t in enumerate(CALIBRATION)), ("held-out", HELD_OUT)]:
            ids = tok.encode(text)
            a64, b64 = forward(orig, shape, ids, torch.float64), forward(perm, shape, ids, torch.float64)
            a16, b16 = forward(orig, shape, ids, torch.bfloat16), forward(perm, shape, ids, torch.bfloat16)
            top = float(a64.abs().max())
            out["texts"].append({
                "text": name, "tokens": len(ids), "max_abs_logit_f64": top,
                "f64_max_abs_diff": float((a64 - b64).abs().max()), "f64_rel": float((a64 - b64).abs().max()) / top,
                "f64_bit_exact": bool(torch.equal(a64, b64)),
                "bf16_max_abs_diff": float((a16.float() - b16.float()).abs().max()), "bf16_bit_exact": bool(torch.equal(a16, b16)),
                "bf16_argmax_agree": float((a16.argmax(-1) == b16.argmax(-1)).float().mean()),
                "f64_argmax_agree": float((a64.argmax(-1) == b64.argmax(-1)).float().mean())})
    out["f64_rel_max"] = max(t["f64_rel"] for t in out["texts"])
    out["ok"] = out["f64_rel_max"] <= F64_TOL and all(t["f64_argmax_agree"] == 1.0 for t in out["texts"])
    return out


# ---- a served case ----------------------------------------------------------------------------------------------------------

def screen_case(case: Path, doc: dict) -> dict:
    """The per-row rules on every real A row of each call a served case holds, and on its weights' B rows, before (the
    manifest's declaration) and after `doc`'s swaps (a weight's columns permuted as its kind's inputs are)."""
    shape = Shape(**doc["shape"])
    parts = json.loads((case / "parts.json").read_text())
    files = parts["evidence"]["files"]

    def rows(pi: int, spec: dict, m: int, k: int) -> np.ndarray | None:
        h = files[pi].get(spec["path"])
        if h is None:
            return None
        raw = np.frombuffer((case / "blobs" / h).read_bytes(), "<u2")
        if raw.size != m * k:
            raise ValueError(f"{spec['path']}: {raw.size} values, not {m}×{k}")
        return (raw.reshape(m, k).astype(np.uint32) << 16).view(np.float32)

    out = {"calls": 0, "real_rows": 0, "excluded_before": 0, "failing_after": 0, "newly_failing": 0, "weights_b": 0,
           "b_failing_before": 0, "b_failing_after": 0, "left": []}
    seen: set[int] = set()
    for pi, man in enumerate(parts["manifests"]):
        for rec in man["matmuls"]:
            w, m, k = rec["weight"], rec["m"], rec["k"]
            p = column_perm(doc, shape, KINDS[w % 4], w // 4)
            a = rows(pi, rec["rows"], m, k)
            if a is not None:
                real = a[:rec["real_rows"]]
                after = failing(real[:, p])
                ex = np.zeros(len(real), bool)
                ex[rec["excluded_a"]] = True
                out["calls"] += 1
                out["real_rows"] += len(real)
                out["excluded_before"] += int(ex.sum())
                out["failing_after"] += int(after.sum())
                out["newly_failing"] += int((after & ~ex).sum())
                out["left"] += [{"pass": pi, "index": rec["index"], "weight": w, "row": int(r)} for r in np.nonzero(after)[0]]
            if w not in seen:
                wt = man["weights"][str(w)]
                b = rows(pi, wt["rows"], wt["n"], wt["k"])
                if b is not None:
                    seen.add(w)
                    out["weights_b"] += 1
                    out["b_failing_before"] += int(sum(failing(b[i:i + 4096]).sum() for i in range(0, len(b), 4096)))
                    out["b_failing_after"] += int(sum(failing(b[i:i + 4096][:, p]).sum() for i in range(0, len(b), 4096)))
    out["ok"] = out["failing_after"] == 0 and out["b_failing_after"] == 0
    return out


# ---- the command ------------------------------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: str | None, doc: dict) -> None:
    text = json.dumps(doc, indent=1) + "\n"
    if path:
        Path(path).write_text(text)
    print(text, end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--out", required=True, help="the permuted snapshot's directory")
    mk.add_argument("--checkpoint", help="the original snapshot (default: fetched at --rev into OUT.original)")
    mk.add_argument("--rev", default=REV)
    mk.add_argument("--report")
    ck = sub.add_parser("check")
    ck.add_argument("--original", required=True)
    ck.add_argument("--permuted", required=True)
    ck.add_argument("--report")
    sc = sub.add_parser("screen-case")
    sc.add_argument("case")
    sc.add_argument("--swaps", required=True)
    sc.add_argument("--out")
    a = ap.parse_args(argv)

    if a.cmd == "screen-case":
        got = screen_case(Path(a.case), json.loads(Path(a.swaps).read_text()))
        _write(a.out, got)
        return 0 if got["ok"] else 1
    if a.cmd == "check":
        tok = Tokenizer(Path(a.original) / "tokenizer.json")
        orig, perm = load(Path(a.original)), load(Path(a.permuted))
        got = logits_check(orig[3], perm[3], orig[4], tok)
        _write(a.report, got)
        return 0 if got["ok"] else 1

    out = Path(a.out)
    src = Path(a.checkpoint) if a.checkpoint else fetch(a.rev, Path(str(out) + ".original"))
    tok = Tokenizer(src / "tokenizer.json")
    data, header, base, w, shape = load(src)
    doc = calibrate(w, shape, tok)
    doc["source"] = {"repo": REPO, "rev": a.rev, "files": {n: _sha256(src / n) for n in FILES if (src / n).exists()}}
    if doc["screen"]["total_after"]:
        _write(a.report, doc)
        return 1
    orig = {n: t.clone() for n, t in w.items()}
    apply(w, shape, doc)
    out.mkdir(parents=True, exist_ok=True)
    for n in FILES:
        if n != "model.safetensors" and (src / n).exists():
            shutil.copyfile(src / n, out / n)
    (out / "model.safetensors").write_bytes(data)
    doc["permuted"] = {"model.safetensors": _sha256(out / "model.safetensors")}
    (out / "channel_swaps.json").write_text(json.dumps({k: v for k, v in doc.items() if k != "logits"}, indent=1) + "\n")
    doc["logits"] = logits_check(orig, w, shape, tok)
    _write(a.report, doc)
    return 0 if doc["logits"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
