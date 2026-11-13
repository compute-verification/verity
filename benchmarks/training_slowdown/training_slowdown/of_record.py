"""A config's counted training step beside its run of record's: `python -m training_slowdown.of_record [ENTRY ...]
--records DIR` says, for each entry of `attacks/` that `of_record.toml` holds to a run, equal or what differs.

`of_record.toml` names an entry's run of record: the run, the store artifact that holds its record, the record's
path there and its sha256. `--records DIR` is where the records are, each at `DIR/RUN/PATH` (what `research data
fetch ART --to DIR/RUN --path PATH` leaves); a record that is missing is refused with that line, and so is a file of
another sha256.

What is held is the step and the kind of the encoding, not the weights, E's bytes or the loss:

- **The step.** The record's side is its modal step structure, the one most of its steps have
  (`Counter.record()["structures"]`): `updated`, one `[MACs a token, updated MACs a token upstream, MACs, tokens,
  bits]` for each use of an updated weight, and `log`, one `[phase, kind, op, operand shapes, count, MACs]` for each
  product. The config's side is the same count of its own step: its route built on the start's shape and stepped
  twice under the harness's counter (`run.train`), the second step read, since by then every trained weight has left
  θ as in every later step. Both sides are stated per row a forward (`per_row`): the record's rows a forward are
  divided out of its token shapes and their MACs, and the config is stepped at one row a forward, its batch over the
  record's rows a forward, so a step has the config's own forwards. `updated` must be equal as its `(MACs a token,
  upstream)` pairs, and `log` by phase, kind, op, shapes, count and MACs. The start and the steps must be the
  record's too.
- **Attention's kernel.** The counter writes a fused attention kernel as the ops the math kernel computes attention
  by (`flops.Counter._attention`), so a config's step holds those. A record that holds the kernel as one row is
  rewritten the same way before it is compared (`as_math.rewrite`, on its modal structure). The row does not say
  which gradients a backward call asked for: that is read from the record's trace where it holds one, and is
  otherwise the entry's `partial` in `of_record.toml` (`none`, every call asks for q's, k's and v's, or
  SUBSET:CALLS), a stated value and never a default: an entry whose record needs one and has none is refused, with
  `as_math`'s words for what to state, and so is a record `as_math` cannot rewrite. The output says whose word the
  ask is. A stated ask the record cannot confirm is still held: a wrong one writes products the config's step does
  not have, or leaves out ones it has, so the comparison shows it as a difference. A `partial` whose record has no
  fused backward row to read it for is a difference too.
- **Why a count stands for the run.** θ here is the model's own initialisation and the rows are random tokens, on a
  CPU. The counter reads operand shapes and which weights have left θ, never a value, so it counts the step a job of
  the config takes on the start's θ and data whenever that step is one fixed graph. The capture decides that: the
  step is traced (`run.trace_step`), and an entry whose trace the capture refuses, or whose step reads a value back
  that decides what it computes, is refused here. No route says of itself that it branches on values.
- **Expected differences.** A route's CPU count differs from a program's CUDA record in ways that are the device's
  (`width`, `norm`) and in ways that are the program's own arrangement of a computation the route
  takes too: the loss as the rows' means over the rows kept where a route means over the tokens (`loss`), and the
  average of the weights as one op over every tensor of the model where a route averages each tensor it trains
  (`average`). Each is named by a rule that fires only on its own exact case (`width`, `RULES`); the output names
  each one that fired with the MACs on both sides. Anything else is printed as a difference, and the command exits
  1: a route that steps AdamW otherwise than the record's program did, in torch's default step where the program
  asked for the fused kernel, differs, and its entry sets the route's dial for the kernel.
- **A line of the record's own program.** A product the record's program computes and a route rightly doesn't (a
  logging line's arithmetic) is no rule's: the entry of `of_record.toml` names it (`record_only`: the line, the
  count and MACs the record's step has of it over the config's, and why), and it is printed as named and taken out
  of the differences. A named line that the record's step does not have over the config's as named is a difference.
- **The encoding.** Each codec of the config's plan must be of the record's kind (`E_KINDS`: the record's `format`
  beside the codecs of that kind); E's size after the counted steps is printed beside the record's last
  checkpoint's, with how a codec packs the same encoding another way than the record's program did (`E_PACKING`).

`--out DIR` keeps the comparisons (`DIR/of_record.json`) and each config's counted record (`DIR/ENTRY/run.json`).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
import tempfile
import tomllib
from fractions import Fraction
from pathlib import Path

import torch

from . import as_math, config, library, run, starts
from .model import LM

HELD = Path(__file__).with_name("of_record.toml")
SCHEMA = "of-record/v1"
#: a record's E `format` beside the codec kinds (`codecs.Codec.kind`) that are the same kind of encoding: the trained
#: tensors whole, and their deltas in codes with a scale a group, the codes a rounding of the trained delta
#: (`scaledquant`) or the ternary ones the run trained (`sparseternary`). A format may have several kinds
E_KINDS = {"full": ("identity",), "quant": ("scaledquant", "sparseternary")}
#: a codec that packs the same encoding another way than the records' program did, so that its size is expected to
#: differ from a record's by the packing alone: named in the entry's E line
E_PACKING = {"sparseternary": "one xz stream a matrix at one setting, where the record's program kept the smallest "
                              "of several settings' streams of all its matrices"}
#: how many of a config's steps are counted; the last is the one compared
STEPS = 2
FAIR = ("Each config's step was counted at the model's own initialisation on random tokens, one row a forward, on a "
        "CPU: the count reads operand shapes and which weights have left the start, never a value, and the capture "
        "traced the step as one fixed graph that reads no value back, so it is the step a job of the config takes on "
        "the start's weights and data. An expected difference is named by a rule that fires on its exact case "
        "alone: the device's, and the record's program's own arrangement of a computation the route takes too. "
        "Not held: the weights, E's bytes and the loss, which only a run has.")
#: the argument by which the records' programs drop rows from a step's loss: that share of a forward's rows, the ones
#: of the highest loss, get no gradient
DROPS_ROWS = "--trunc"
#: the argument by which the records' programs keep an average of the weights: its horizon, a share of the steps so far
AVERAGES = "--ema-frac"
#: a number as a program's arguments write one
NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def held(path: Path = HELD) -> dict:
    """Each held entry's run of record: `{entry: {"run", "art", "path", "sha256"}}`; where its record holds fused
    attention backward rows and no trace, `"partial"`, the gradients those calls asked for (`as_math.parse`); and
    where its record's program computes lines the route rightly doesn't, `"record_only"`, those lines (`named`)."""
    return tomllib.loads(Path(path).read_text())["held"]


def read(name: str, h: dict, root: Path) -> dict:
    """`name`'s record under `root`, refused unless it is the file `of_record.toml` names."""
    path = Path(root) / h["run"] / h["path"]
    fetch = f"`research data fetch {h['art']} --to {Path(root) / h['run']} --path {h['path']}`"
    if not path.is_file():
        raise SystemExit(f"of_record: {name}'s record is not at {path} ({fetch})")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != h["sha256"]:
        raise SystemExit(f"of_record: {path} is not the record {HELD.name} names for {name}: its sha256 is "
                         f"{hashlib.sha256(raw).hexdigest()} ({fetch})")
    return json.loads(raw)


def modal(rec: dict) -> tuple[dict, int]:
    """The step structure most of the record's steps have, and how many have it."""
    sid, n = collections.Counter(s["struct"] for s in rec["steps"]).most_common(1)[0]
    return rec["structures"][sid], n


def rows_a_forward(struct: dict, ctx: int) -> int:
    """The rows each forward of the step reads: every use of an updated weight states its forward's tokens."""
    tokens = {u[3] for u in struct["updated"]}
    if len(tokens) != 1 or next(iter(tokens)) % ctx:
        raise ValueError(f"the step's uses of an updated weight read {sorted(tokens) or 'no'} tokens: not one whole "
                         f"number of rows of {ctx}, so its rows a forward can't be divided out")
    return tokens.pop() // ctx


def as_counted(struct: dict, rec: dict, said: str | None) -> tuple[dict, str | None, list[str]]:
    """A record's step structure with each fused attention kernel's row as the ops the counter writes the kernel as
    (`as_math.rewrite`); what the output says of that, None where the structure holds no such row; and the
    differences. The gradients a backward row's calls asked for are the record's trace's where it holds one and
    otherwise `said`, the entry's `partial`: a ValueError in `as_math`'s words where neither says, where the two
    disagree and where a row can't be rewritten. A `partial` that no backward row was read for is a difference."""
    trace = (rec.get("trace") or {}).get("rows") or []
    log, moved, asks = as_math.rewrite(struct["log"], trace, as_math.parse(None if said is None else str(said)))
    unread = [] if said is None or asks else [
        f"{HELD.name} states `partial = {json.dumps(said)}` and the record's step has no fused attention backward "
        "row to read it for"]
    if not (moved["by"] or moved["macs"]):
        return struct, None, unread
    stated = f"{HELD.name}'s word (`partial`) and not the record's, which holds no trace to read it from"
    asked = "; ".join(f"`{a['asked']}` at {at}, " + (stated if a["from"] == "--partial" else "read from its trace")
                      for at, a in sorted(asks.items()))
    return struct | {"log": log}, (
        "the record holds a fused kernel as one row: compared as the ops the math kernel computes attention by, as "
        "the counter writes the config's (`as_math`). The gradients its backward calls asked for (`none`: every "
        f"call q's, k's and v's; SUBSET:CALLS: that many calls a step SUBSET's alone): {asked or 'it has no backward'}"
    ), unread


def _whole(x: Fraction):
    return int(x) if x.denominator == 1 else float(x)


def per_row(struct: dict, shape: dict) -> dict:
    """A step structure of a model of `shape` per row a forward: `rows`, its rows a forward; `updated`, each use's
    `(MACs a token, upstream, bits)`; `log`, `{(phase, kind, op, shapes): [count, MACs]}` with the rows divided out of
    every token shape and out of the MACs of each entry whose shapes that changed. A dimension of rows x context
    becomes the context, and a leading dimension of rows before a context becomes 1. A product of attention's is
    batched over rows x heads: a `bmm` whose every operand leads with rows x heads over a context and nothing but
    contexts and a head's width, and that leading dimension becomes the heads. An optimizer's entries have no such
    shape and stay whole. A `_foreach_` op's operands are sorted: their order is the order a program lists its
    parameters in, which is no part of the step."""
    ctx, heads = shape["ctx"], shape["n_head"]
    r = rows_a_forward(struct, ctx)
    a_head = {ctx, shape["d_model"] // heads}

    def of_tokens(s: list) -> list:
        s = [ctx if d == r * ctx else d for d in s]
        return [1, *s[1:]] if len(s) > 1 and s[0] == r and ctx in s[1:] else s

    def over_heads(s: list) -> bool:
        return len(s) == 3 and s[0] == r * heads and ctx in s[1:] and set(s[1:]) <= a_head

    log: dict[tuple, list] = {}
    for phase, kind, op, shapes, count, macs in struct["log"]:
        was = json.loads(shapes)
        now = ([[heads, *s[1:]] for s in was] if op == "bmm" and all(map(over_heads, was)) else
               [of_tokens(s) for s in was])
        e = log.setdefault((phase, kind, op, json.dumps(sorted(now) if op.startswith("_foreach_") else now)), [0, 0])
        e[0] += count
        e[1] += _whole(Fraction(macs, r)) if now != was else macs
    return {"rows": r, "updated": sorted((u[0], u[1], u[4]) for u in struct["updated"]), "log": log}


# -- the expected differences: each rule takes what it covers out of the two sides' unequal entries, `cfg` and `rec`,
# given the start's `shape` and the record's program's arguments `args`


def width(mine: list, theirs: list) -> tuple[dict | None, list[str]]:
    """The uses of an updated weight, the config's beside the record's: the expected difference, if it fired, and the
    differences. The `(MACs a token, upstream)` pairs must be equal. Expected: uses that read their weight at 32 bits
    where the record's read it at 16, since a route casts to bf16 on CUDA alone."""
    a, b = (collections.Counter(u[:2] for u in side) for side in (mine, theirs))
    if a != b:
        return None, [f"updated, the {who}'s step only: " + ", ".join(
            f"({p:,} MACs a token, {u:,} upstream) x{n}" for (p, u), n in sorted(only.items()))
            for who, only in (("config", a - b), ("record", b - a)) if only]
    mine, theirs = collections.Counter(mine), collections.Counter(theirs)
    wide, narrow = mine - theirs, theirs - mine
    if not wide:
        return None, []
    if {u[2] for u in wide} == {32} and {u[2] for u in narrow} == {16}:
        return {"rule": "width", "what": f"{sum(wide.values())} of the uses read their weight at 32 bits where the "
                                         f"record's read it at 16: a route casts to bf16 on CUDA alone"}, []
    return None, ["updated, the weights' bits: " + "; ".join(
        f"the {who}'s " + ", ".join(f"({p:,}, {u:,}) at {bits} x{n}" for (p, u, bits), n in sorted(only.items()))
        for who, only in (("config", wide), ("record", narrow)))]


def norm(cfg: dict, rec: dict, shape: dict, args: list) -> dict | None:
    """The norm: CUDA runs it as one fused kernel, which the counter has no price for and counts only where its gain
    is an updated weight, one MAC an element; a CPU runs it as elementwise ops, each counted. Fires on the record's
    `fwd/updated _fused_rms_norm` over a row's residual stream and a gain beside the config's `fwd/updated mul` over
    the same shapes with its count and MACs. It then covers the norm's other products on a CPU, none of which the
    record may have: the gain's multiply where the gain is θ's (`fwd/theta`), the normalisation (`fwd/act`, the
    stream by a scalar a token), which must be one for each gain's multiply, and the backward's multiplies and
    quotients over those three shapes alone. A step that trains no gain has neither of the first two, a record with
    nothing of the kernel beside a config with no updated gain's multiply: the rule then fires on the config's
    multiplies by θ's gains, and covers the same products."""
    x, s, g = [1, shape["ctx"], shape["d_model"]], [1, shape["ctx"], 1], [shape["d_model"]]
    gain = json.dumps([x, g])
    fused, mul = ("fwd", "updated", "_fused_rms_norm", gain), ("fwd", "updated", "mul", gain)
    frozen, scale = ("fwd", "theta", "mul", gain), ("fwd", "act", "mul", json.dumps([x, s]))
    trained = fused in rec
    if cfg.get(mul) != rec.get(fused) or not trained and frozen not in cfg:
        return None
    updated = rec[fused] if trained else [0, 0]
    inside = [k for k in cfg if k in (frozen, scale) or k[:2] == ("bwd", "grad") and k[2] in ("mul", "div")
              and k[3] != "[]" and all(o in (x, s, g) for o in json.loads(k[3]))]
    if cfg.get(scale, [0])[0] != updated[0] + cfg.get(frozen, [0])[0] or any(k in rec for k in inside):
        return None
    gains = (f"{updated[0]} multiplies by an updated gain, the same on both sides" if trained else
             "no gain is updated, so the record has nothing of the kernel")
    found = {"rule": "norm", "record_macs": updated[1], "config_macs": updated[1] + sum(cfg[k][1] for k in inside),
             "what": f"the norm as the CPU's elementwise ops for CUDA's fused kernel: {gains}, and on the config's "
                     f"alone {cfg.get(frozen, [0])[0]} by θ's gains, {cfg[scale][0]} normalisations and "
                     f"{sum(cfg[k][0] for k in inside if k[0] == 'bwd')} products of the backward's, none of which "
                     "CUDA's kernel shows the counter"}
    for k in (*((mul,) if trained else ()), *inside):
        del cfg[k]
    if trained:
        del rec[fused]
    return found


def stated(args: list, flag: str) -> str | None:
    """What a program's arguments state for `flag`: the argument after it or after its `=`, None where they have no
    such flag."""
    for i, a in enumerate(args):
        if a == flag:
            return args[i + 1] if i + 1 < len(args) else ""
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def number(args: list, flag: str) -> float | None:
    """The number a program's arguments state for `flag`: 0 where they have no such flag, and None where what they
    state is no number, which no rule takes for its case."""
    said = stated(args, flag)
    if said is None:
        return 0.0
    return float(said) if NUMBER.fullmatch(said) else None


def loss(cfg: dict, rec: dict, shape: dict, args: list) -> dict | None:
    """The loss's arrangement: a record's program may mean each row's loss over its tokens, sum the rows and divide
    by the rows the step keeps, where a route means over a forward's tokens and scales by the forward's share of the
    batch. Where the step drops no row the two are one value and one gradient, so this is the program's choice and
    no other loss. Fires on the record's backward quotient over a row's tokens with its scalar quotient, forward and
    backward, beside the config's scalar product, forward and backward: all five of one count, and neither side with
    a line of the other's. What is checked for a dropped row: the record's program's arguments (`script_args`),
    where the record keeps them, state no `DROPS_ROWS` but 0; with one, or with one that is no number, the rule does
    not fire, and of a record that keeps none the output says that nothing was checked."""
    scalar, a_row = "[[]]", json.dumps([[1, shape["ctx"]]])
    theirs = (("bwd", "grad", "div", a_row), ("fwd", "theta", "div", scalar), ("bwd", "grad", "div", scalar))
    mine = (("fwd", "theta", "mul", scalar), ("bwd", "grad", "mul", scalar))
    n = rec.get(theirs[0], [0])[0]
    if (not n or [rec.get(k) for k in theirs] != [[n, n * shape["ctx"]], [n, n], [n, n]]
            or [cfg.get(k) for k in mine] != [[n, n]] * 2 or any(k in cfg for k in theirs)
            or any(k in rec for k in mine)):
        return None
    if number(args, DROPS_ROWS) != 0:
        return None
    checked = (f"the record's program's arguments state no {DROPS_ROWS}" if args else
               "the record keeps no arguments of its program, so that was not checked")
    found = {"rule": "loss", "record_macs": sum(rec[k][1] for k in theirs), "config_macs": sum(cfg[k][1] for k in mine),
             "what": "the loss as the route's mean over a forward's tokens times the forward's share of the batch for "
                     "the record's program's mean of each row, the rows' sum over the rows kept, "
                     f"{_plural(n, 'time')} a step: one value and one gradient where the run drops no row ({checked})"}
    for k in theirs:
        del rec[k]
    for k in mine:
        del cfg[k]
    return found


def average(cfg: dict, rec: dict, shape: dict, args: list) -> dict | None:
    """The average of the weights: a record's program may keep one of every tensor of the model, those it never
    trains too, in one `_foreach_lerp_` a step, where route `ema` keeps one of each tensor it trains, a `lerp_` a
    tensor. Fires where the record's program's arguments state an average (`AVERAGES`, a number other than 0), on the
    record's one `_foreach_lerp_`, once a step, over tensors each beside its average and priced at one MAC an
    element, beside the config's `lerp_` of a tensor with one of its shape, priced alike, for some of the record's
    tensors and no other.
    Neither side may have another `lerp_` or `_foreach_lerp_`, and the config's unequal entries no `addcmul_` or
    `addcdiv_`: an unfused AdamW step has those and a `lerp_` of its own, which one of these could then be. The
    record's tensors the config has none of are those the route does not train: the output counts them."""
    name = ("other", "param")
    theirs = [k for k in rec if k[:3] == (*name, "_foreach_lerp_")]
    mine = {k: json.loads(k[3]) for k in cfg if k[:3] == (*name, "lerp_")}
    if (not number(args, AVERAGES) or len(theirs) != 1 or not mine or any(k[:3] == (*name, "lerp_") for k in rec)
            or any(k[:2] == name and k[2] in ("_foreach_lerp_", "addcmul_", "addcdiv_") for k in cfg)):
        return None
    left = collections.Counter(tuple(t) for t in json.loads(theirs[0][3]))
    if any(n % 2 for n in left.values()) or rec[theirs[0]] != [1, sum(math.prod(t) * n // 2 for t, n in left.items())]:
        return None
    for k, operands in mine.items():
        n, t = cfg[k][0], tuple(operands[0])
        if len(operands) != 2 or operands[0] != operands[1] or cfg[k] != [n, n * math.prod(t)] or 2 * n > left[t]:
            return None
        left[t] -= 2 * n
    trained, more = sum(cfg[k][0] for k in mine), sum(left.values()) // 2
    untrained = (f"those {trained} and {more} more that the route does not train "
                 f"({sum(math.prod(t) * n // 2 for t, n in left.items()):,} MACs a step): the program averages every "
                 "tensor of the model, those it never trains too" if more else f"the same {trained}")
    found = {"rule": "average", "record_macs": rec[theirs[0]][1], "config_macs": sum(cfg[k][1] for k in mine),
             "what": f"the route's average of each tensor it trains, {trained} of them, an op a tensor, for the "
                     f"record's program's in one op: {untrained}"}
    del rec[theirs[0]]
    for k in mine:
        del cfg[k]
    return found


RULES = (norm, loss, average)


def _plural(n: int, one: str, many: str = "") -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"


def _shapes(shapes: str) -> str:
    """Operand shapes as printed: a run of one shape once, with its length."""
    out: list[list] = []
    for s in json.loads(shapes):
        if out and out[-1][0] == s:
            out[-1][1] += 1
        else:
            out.append([s, 1])
    return "[" + ", ".join(json.dumps(s, separators=(",", ":")) + (f" x{n}" if n > 1 else "") for s, n in out) + "]"


def _entry(key: tuple, value: list) -> str:
    return f"{key[0]}/{key[1]} {key[2]} {_shapes(key[3])} x{value[0]}, {value[1]:,} MACs"


def named(lines: list[dict], cfg: dict, rec: dict) -> tuple[list[dict], list[str]]:
    """The lines of the record's own program that its entry of `of_record.toml` names (`record_only`): products the
    route rightly doesn't compute, each by `phase`, `kind`, `op` and `shapes`, with the `count` and `macs` the
    record's step has of it over the config's (all of the line where the config has none of it) and `why`. No rule
    in code knows them. A line is taken out of the two sides' unequal entries `cfg` and `rec` where the record's
    excess is the named one; a named line whose excess is another, or none, is a difference, so a name outlives
    neither its line nor a route that computes it."""
    found, differences = [], []
    for line in lines:
        key = (line["phase"], line["kind"], line["op"], json.dumps(line["shapes"]))
        value = [line["count"], line["macs"]]
        mine, theirs = cfg.get(key, [0, 0]), rec.get(key, [0, 0])
        if key in rec and [theirs[0] - mine[0], theirs[1] - mine[1]] == value:
            del rec[key]
            cfg.pop(key, None)
            found.append({"line": _entry(key, value), "why": line["why"]})
        else:
            differences.append(f"{HELD.name} names a line the record's step does not have over the config's as "
                               f"named: {_entry(key, value)}")
    return found, differences


def compare(mine: dict, theirs: dict, shape: dict, lines=(), args=()) -> dict:
    """The config's step (`mine`) beside the record's (`theirs`), each `per_row`'s: how many `uses` of an updated
    weight and `products` of the log are equal, the `expected` differences (`width`, `RULES`: each with what it
    covers and the MACs on each side), the `named` lines of the record's own program (`lines`: the entry's
    `record_only`) and the `differences`, everything else. `args` are the record's program's arguments, where the
    record keeps them."""
    wide, differences = width(mine["updated"], theirs["updated"])
    cfg, rec = dict(mine["log"]), dict(theirs["log"])
    same = [k for k in cfg if rec.get(k) == cfg[k]]
    for k in same:
        del cfg[k], rec[k]
    expected = [wide] if wide else []
    expected += [found for rule in RULES if (found := rule(cfg, rec, shape, list(args)))]
    only, unnamed = named(lines, cfg, rec)
    differences += unnamed
    for k in sorted(set(cfg) | set(rec)):
        if k in cfg and k in rec:
            differences.append(f"log, unequal: {_entry(k, cfg[k])} in the config's step, x{rec[k][0]}, "
                               f"{rec[k][1]:,} MACs in the record's")
        else:
            differences.append(f"log, the {'config' if k in cfg else 'record'}'s step only: "
                               f"{_entry(k, (cfg if k in cfg else rec)[k])}")
    return {"uses": len(mine["updated"]) if not differences or not differences[0].startswith("updated") else 0,
            "products": len(same), "expected": expected, "named": only, "differences": differences}


# -- the config's counted step


class Tokens:
    """What stands in for the start's data stream: random tokens at the start's context, a batch cut to one row for
    each of the record's `rows` a forward."""

    def __init__(self, shape: dict, rows: int):
        self.vocab, self.ctx, self.rows = shape["vocab"], shape["ctx"], rows
        self.generator = torch.Generator().manual_seed(0)

    def next(self, n: int) -> torch.Tensor:
        return torch.randint(0, self.vocab, (n // self.rows, self.ctx + 1), generator=self.generator)


def counted(cfg: config.Config, rows: int, out: Path) -> dict:
    """The record of `cfg`'s first `STEPS` steps as a job counts them (`run.train` under `run.Harness`, with the
    job's trace), θ the model's own initialisation and the stream `Tokens` at the record's `rows` a forward.
    Written to `out/run.json`."""
    route, shape = config.route(cfg.route), cfg.start.shape
    torch.manual_seed(0)
    theta = {k: v.detach().clone() for k, v in LM.of(shape).state_dict().items()}
    te = run.Harness(theta, out, start={"shape": dict(shape)}, digests=False)
    with te.counter:
        model, optimizers = route.build(shape, cfg.dials)
        run.train(te, cfg, route, Tokens(shape, rows), model, optimizers, steps=STEPS)
    return te.finish({"trace": run.trace_step(cfg, route, model, optimizers, "cpu")})


def refusal(rec: dict) -> str | None:
    """Why a counted step doesn't stand for a run's steps, if it doesn't: the capture refused the step, the step
    reads a value back, or the count itself is invalid."""
    trace = rec["trace"]
    if "refused" in trace:
        r = trace["refused"]
        return f"the capture refused the step ({r['reason']}, {r['op']}: {r['message']})"
    if read_back := [*trace["data_dependent"], *trace["carried_scalars"]]:
        return f"the step reads a value back that decides what it computes ({read_back[0]})"
    if rec["invalid"]:
        return f"the count is invalid ({'; '.join(rec['invalid'])})"
    return None


def row(cfg: config.Config, h: dict, rec: dict, out: Path) -> dict:
    """An entry beside its run of record `rec`: `facts` (what was read, as printed), the `expected` differences, the
    `named` lines of the record's own program, the `differences`, and `refused`, why nothing else was compared: the
    record's fused attention rows can't be written as the counter writes them (`as_counted`), or the entry's count
    stands for no run (`refusal`)."""
    route, shape = config.route(cfg.route), cfg.start.shape
    r = {"entry": cfg.name, "run": h["run"], "record": h["path"], "expected": [], "named": [], "differences": [],
         "refused": None}
    struct, n = modal(rec)
    try:
        struct, attention, unread = as_counted(struct, rec, h.get("partial"))
    except ValueError as e:
        return r | {"refused": f"its record's fused attention rows can't be compared as the math kernel's ops: {e} "
                               f"(what `--partial` would say is the entry's `partial` in {HELD.name})"}
    r["differences"] += unread
    theirs = per_row(struct, shape)
    batch, rows = cfg.budget["batch"], theirs["rows"]
    if batch % rows:
        r["differences"].append(f"the batch: the config's {batch} rows are no whole number of the record's {rows} "
                                "rows a forward")
        return r
    mine = counted(cfg, rows, out)
    if why := refusal(mine):
        return r | {"refused": why, "expected": [], "differences": []}
    start, cp, last = rec.get("start_record") or rec["start"], mine["checkpoints"][-1], rec["checkpoints"][-1]
    if start["digest"] != cfg.start.digest:
        r["differences"].append(f"the start: the config's is {cfg.start.name} ({cfg.start.digest[:12]}), the "
                                f"record's {start['name']} ({start['digest'][:12]})")
    if len(rec["steps"]) != cfg.budget["steps"]:
        r["differences"].append(f"the steps: the config's {cfg.budget['steps']}, the record's {len(rec['steps'])}")
    found = compare(per_row(mine["structures"][mine["steps"][STEPS - 1]["struct"]], shape), theirs, shape,
                    h.get("record_only", []), rec.get("script_args") or [])
    kinds = sorted({c.kind for _, c in route.plan(cfg.dials).rules})
    found["differences"] += e_kind(kinds, last["format"])
    r["expected"] += found["expected"]
    r["named"] = found["named"]
    r["differences"] += found["differences"]
    r["facts"] = [
        ("start", f"{cfg.start.name} ({cfg.start.digest[:12]})"),
        ("steps", f"{cfg.budget['steps']}"),
        ("step", f"the config's step {STEPS}, {_plural(batch // rows, 'forward')} of one row, counted on a CPU, "
                 f"beside the record's modal structure, which {n} of its {len(rec['steps'])} steps have, "
                 f"{_plural(rows, 'row')} a forward divided out"),
        ("updated", f"{found['uses']} of the record's {len(theirs['updated'])} uses of an updated weight, by MACs a "
                    "token and updated MACs upstream"),
        *([("attention", attention)] if attention else []),
        ("log", f"{found['products']} of the record's {len(theirs['log'])} products, by phase, kind, op, operand "
                "shapes, count and MACs"),
        ("E", e_sizes(kinds, cp["e_bytes"], last, len(rec["steps"])))]
    return r


def e_sizes(kinds: list[str], e_bytes: int, last: dict, steps: int) -> str:
    """An entry's E line: the plan's `kinds` and E's size after the counted steps beside the record's format and its
    last checkpoint's size after its `steps`, which are not compared, and each packing that differs (`E_PACKING`)."""
    return (f"{', '.join(kinds)}, {e_bytes:,} B after {STEPS} steps; the record's {last['format']}, "
            f"{last['e_bytes']:,} B after {steps}"
            + "".join(f"; expected to differ in size by the packing too: {k} is {E_PACKING[k]}"
                      for k in kinds if k in E_PACKING))


def e_kind(kinds: list[str], fmt: str) -> list[str]:
    """The difference, if any, between a plan's codec `kinds` and a record's E `fmt`: every codec of the plan must be
    one of the format's kind of encoding (`E_KINDS`)."""
    same = E_KINDS.get(fmt, ())
    if kinds and set(kinds) <= set(same):
        return []
    return [f"E's kind: the config's plan is {', '.join(kinds) or 'empty'}, the record's format {fmt} "
            f"({' or '.join(same) or 'none known here'})"]


def verdict(r: dict) -> str:
    if r["refused"]:
        return f"refused: {r['refused']}"
    if r["differences"]:
        return f"differs, {_plural(len(r['differences']), 'difference')}"
    also = [_plural(len(r["expected"]), "expected difference")] if r["expected"] else []
    also += [f"{_plural(len(r['named']), 'named line')} of the record's program alone"] if r["named"] else []
    return ", ".join(["equal", *also])


def text(rows: list[dict]) -> str:
    """The rows as the command prints them: each entry with its run of record and its verdict, what was read, each
    expected difference by its rule with the MACs on both sides, each line its entry of the data file names, and each
    difference; then the count of each verdict and what makes the comparison fair."""
    out = []
    for r in rows:
        out.append(f"{r['entry']}, held to {r['run']} ({r['record']}): {verdict(r)}")
        lines = list(r.get("facts", []))
        for e in r["expected"]:
            macs = f"; MACs a step at one row a forward: the record's {e['record_macs']:,}, the config's " \
                   f"{e['config_macs']:,}" if "record_macs" in e else ""
            lines.append(("expected", f"{e['rule']}: {e['what']}{macs}"))
        lines += [("named", f"in the record's step and not the config's: {e['line']}; {HELD.name} names it: "
                            f"{e['why']}") for e in r["named"]]
        lines += [("DIFFERS", d) for d in r["differences"]]
        out += [f"  {k:<9} {v}" for k, v in lines] + [""]
    n = collections.Counter(verdict(r).split(",")[0].split(":")[0] for r in rows)
    out.append(f"{_plural(len(rows), 'entry', 'entries')}: {n['equal']} equal, {n['differs']} differ, "
               f"{n['refused']} refused.")
    return "\n".join([*out, FAIR])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("entry", nargs="*", help="an entry: a config's file, or its name in attacks/ (default: every "
                                             "entry the data file holds to a run)")
    ap.add_argument("--records", type=Path, required=True, help="where the runs' records are, each at RUN/PATH")
    ap.add_argument("--held", type=Path, default=HELD, help="the data file: each entry's run of record")
    ap.add_argument("--starts", type=Path, default=starts.REGISTRY, help="the starts registry")
    ap.add_argument("--out", type=Path, default=None, help="where of_record.json and each entry's counted records go")
    a = ap.parse_args(argv)
    runs, reg = held(a.held), starts.registry(a.starts)
    paths = library.entries(a.entry or sorted(runs))
    if unheld := [p.stem for p in paths if p.stem not in runs]:
        raise SystemExit(f"of_record: {a.held.name} holds {', '.join(unheld)} to no run (it holds {', '.join(runs)})")
    with tempfile.TemporaryDirectory() as scratch:
        rows = []
        for p in paths:
            cfg = config.load(p, reg)
            rows.append(row(cfg, runs[cfg.name], read(cfg.name, runs[cfg.name], a.records),
                            (a.out or Path(scratch)) / cfg.name))
    print(text(rows))
    if a.out:
        kept = {"schema": SCHEMA, "rows": rows, "fair": FAIR}
        (a.out / "of_record.json").write_text(json.dumps(kept, indent=1) + "\n")
    return int(any(r["refused"] or r["differences"] for r in rows))


if __name__ == "__main__":
    sys.exit(main())
