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
- **Why a count stands for the run.** θ here is the model's own initialisation and the rows are random tokens, on a
  CPU. The counter reads operand shapes and which weights have left θ, never a value, so it counts the step a job of
  the config takes on the start's θ and data whenever that step is one fixed graph. The capture decides that: the
  step is traced (`run.trace_step`), and an entry whose trace the capture refuses, or whose step reads a value back
  that decides what it computes, is refused here. No route says of itself that it branches on values.
- **Expected differences.** A CPU count differs from a CUDA record in ways that are the device's, each named by a
  rule that fires only on its own exact case (`width`, `RULES`); the output names each one that fired with the MACs
  on both sides. Anything else is printed as a difference, and the command exits 1.
- **The position tables.** One expected difference is the harness's (`HARNESS`): the records' harness checked the
  model's buffers that θ doesn't hold, the position tables, like θ's weights, so a product with one is θ's, where
  `run.Harness` takes such a buffer for a weight the route made. `tables` counts the step again as the records'
  harness did, and that count is the one compared; the output says how many products and uses that moves and that
  the step's MACs are the same on both counts.
- **A line of the record's own program.** A product the record's program computes and a route rightly doesn't (a
  logging line's arithmetic) is no rule's: the entry of `of_record.toml` names it (`record_only`: the line as the
  command prints it, and why), and it is printed as named and taken out of the differences. A named line that is
  not the record's step's alone is a difference.
- **The encoding.** Each codec of the config's plan must be of the record's kind (`E_KINDS`: the record's `format`
  beside the codecs of that kind); E's size after the counted steps is printed beside the record's last
  checkpoint's, with how a codec packs the same encoding another way than the record's program did (`E_PACKING`).

`--out DIR` keeps the comparisons (`DIR/of_record.json`) and each config's counted records: the harness's own count
(`DIR/ENTRY/run.json`) and the count compared (`DIR/ENTRY/tables/run.json`).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import tempfile
import tomllib
from fractions import Fraction
from pathlib import Path

import torch

from . import config, library, run, starts
from .model import LM

HELD = Path(__file__).with_name("of_record.toml")
SCHEMA = "of-record/v1"
#: a record's E `format` beside the codec kinds (`codecs.Codec.kind`) that are the same kind of encoding: the trained
#: tensors whole, and their deltas in codes with a scale a group, the codes a rounding of the trained delta
#: (`scaledquant`) or the ternary ones the run trained (`sparseternary`)
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
        "the start's weights and data. An expected difference is the device's or the harness's, named by a rule "
        "that fires on its exact case alone. Not held: the weights, E's bytes and the loss, which only a run has.")
#: CUDA's attention kernels beside the CPU's, which the counter prices alike (`flops.FLOP_FORMULAS`)
ATTENTION = {"_scaled_dot_product_flash_attention": "_scaled_dot_product_flash_attention_for_cpu",
             "_scaled_dot_product_flash_attention_backward": "_scaled_dot_product_flash_attention_for_cpu_backward"}
#: the operands CUDA's attention backward takes after the CPU's: its dropout's seed and offset
CUDA_ONLY_OPERANDS = [[2], []]
#: the products of torch's unfused AdamW step on one parameter with no weight decay, each with its tensor operands
UNFUSED_ADAMW = (("lerp_", 2), ("mul_", 1), ("addcmul_", 3), ("div", 1), ("addcdiv_", 3))


def held(path: Path = HELD) -> dict:
    """Each held entry's run of record: `{entry: {"run", "art", "path", "sha256"}}`, and where its record's program
    computes lines the route rightly doesn't, `"record_only"`, those lines (`named`)."""
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


def _whole(x: Fraction):
    return int(x) if x.denominator == 1 else float(x)


def per_row(struct: dict, ctx: int) -> dict:
    """A step structure per row a forward: `rows`, its rows a forward; `updated`, each use's `(MACs a token, upstream,
    bits)`; `log`, `{(phase, kind, op, shapes): [count, MACs]}` with the rows divided out of every token shape (a
    dimension of rows x context becomes the context, a leading dimension of rows before a context becomes 1) and out
    of the MACs of each entry whose shapes that changed. An optimizer's entries have no such shape and stay whole. A
    `_foreach_` op's operands are sorted: their order is the order a program lists its parameters in, which is no
    part of the step."""
    r = rows_a_forward(struct, ctx)

    def shape(s: list) -> list:
        s = [ctx if d == r * ctx else d for d in s]
        return [1, *s[1:]] if len(s) > 1 and s[0] == r and ctx in s[1:] else s

    log: dict[tuple, list] = {}
    for phase, kind, op, shapes, count, macs in struct["log"]:
        was = json.loads(shapes)
        now = [shape(s) for s in was]
        e = log.setdefault((phase, kind, op, json.dumps(sorted(now) if op.startswith("_foreach_") else now)), [0, 0])
        e[0] += count
        e[1] += _whole(Fraction(macs, r)) if now != was else macs
    return {"rows": r, "updated": sorted((u[0], u[1], u[4]) for u in struct["updated"]), "log": log}


# -- the expected differences: each rule takes what it covers out of the two sides' unequal entries


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


def attention(cfg: dict, rec: dict, shape: dict) -> dict | None:
    """Attention's kernel: a CPU runs `ATTENTION`'s second names where CUDA runs its first. Fires on an entry of the
    record's under CUDA's name beside one of the config's under the CPU's with its phase, kind, count, MACs and
    operand shapes (`CUDA_ONLY_OPERANDS` aside)."""
    pairs = []
    for key in rec:
        phase, kind, op, shapes = key
        s = json.loads(shapes)
        for n in (len(s), len(s) - len(CUDA_ONLY_OPERANDS)) if op in ATTENTION else ():
            mine = (phase, kind, ATTENTION[op], json.dumps(s[:n]))
            if n >= 0 and s[n:] in ([], CUDA_ONLY_OPERANDS) and cfg.get(mine) == rec[key]:
                pairs.append((key, mine))
                break
    if not pairs:
        return None
    macs = sum(rec[key][1] for key, _ in pairs)
    for key, mine in pairs:
        del rec[key], cfg[mine]
    return {"rule": "attention", "record_macs": macs, "config_macs": macs,
            "what": f"the CPU's attention kernels for CUDA's, {_plural(len(pairs), 'entry', 'entries')} of the same "
                    "phase, kind, count, shapes and MACs"}


def norm(cfg: dict, rec: dict, shape: dict) -> dict | None:
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


def adamw(cfg: dict, rec: dict, shape: dict) -> dict | None:
    """AdamW's kernel: the record's program steps CUDA's fused kernel, one op over every parameter, which the counter
    prices at eight multiplications a parameter element; a route steps torch's unfused AdamW, whose five in-place
    products a parameter the counter counts as they run (`UNFUSED_ADAMW`). Fires on the record's one `_fused_adamw_`
    over its parameters, their gradients and their two moments beside the config's five products once for each of
    the same parameter shapes."""
    key = next((k for k in rec if k[:3] == ("other", "param", "_fused_adamw_")), None)
    if key is None:
        return None
    tensors = [s for s in json.loads(key[3]) if s]
    params = tensors[:len(tensors) // 4]
    if tensors != params * 4 or rec[key] != [1, 8 * sum(math.prod(p) for p in params)]:
        return None
    want: dict[tuple, list] = {}
    for p in params:
        for op, operands in UNFUSED_ADAMW:
            e = want.setdefault(("other", "param", op, json.dumps([p] * operands)), [0, 0])
            e[0] += 1
            e[1] += math.prod(p)
    if any(cfg.get(k) != v for k, v in want.items()):
        return None
    found = {"rule": "adamw", "record_macs": rec[key][1], "config_macs": sum(v[1] for v in want.values()),
             "what": f"torch's unfused AdamW for CUDA's fused kernel over the same {_plural(len(params), 'parameter')}"
                     f": {len(UNFUSED_ADAMW)} products a parameter element counted as they run, where the kernel is "
                     "priced at 8"}
    del rec[key]
    for k in want:
        del cfg[k]
    return found


RULES = (attention, norm, adamw)


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


def named(lines: list[dict], rec: dict) -> tuple[list[dict], list[str]]:
    """The lines of the record's own program that its entry of `of_record.toml` names (`record_only`), out of the
    record's unequal entries `rec`: products the route rightly doesn't compute, each `phase`, `kind`, `op`, `shapes`,
    `count` and `macs` as the command prints the line, and `why`. No rule in code knows them. A named line that is
    not there as named is a difference, so a name outlives neither its line nor a route that computes it."""
    found, differences = [], []
    for line in lines:
        key = (line["phase"], line["kind"], line["op"], json.dumps(line["shapes"]))
        value = [line["count"], line["macs"]]
        if rec.get(key) == value:
            del rec[key]
            found.append({"line": _entry(key, value), "why": line["why"]})
        else:
            differences.append(f"{HELD.name} names a line that is not the record's step's alone: {_entry(key, value)}")
    return found, differences


def compare(mine: dict, theirs: dict, shape: dict, lines=()) -> dict:
    """The config's step (`mine`) beside the record's (`theirs`), each `per_row`'s: how many `uses` of an updated
    weight and `products` of the log are equal, the `expected` differences (`width`, `RULES`: each with what it
    covers and the MACs on each side), the `named` lines of the record's own program (`lines`: the entry's
    `record_only`) and the `differences`, everything else."""
    wide, differences = width(mine["updated"], theirs["updated"])
    cfg, rec = dict(mine["log"]), dict(theirs["log"])
    same = [k for k in cfg if rec.get(k) == cfg[k]]
    for k in same:
        del cfg[k], rec[k]
    expected = [wide] if wide else []
    expected += [found for rule in RULES if (found := rule(cfg, rec, shape))]
    only, unnamed = named(lines, rec)
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


class AsRecorded(run.Harness):
    """The harness as the records' was (`tables`): a buffer of the model that θ doesn't hold is checked like θ's
    weights, so a product with one is θ's."""

    def register(self, model: torch.nn.Module) -> torch.nn.Module:
        tables = {k: b for k, b in model.named_buffers() if k not in self.theta}
        self.counter.theta_cpu = self.theta | {k: b.detach().clone() for k, b in tables.items()}
        self.counter.register(model.state_dict(keep_vars=True) | tables)
        self.model = model
        self._measure(model)
        return model


def counted(cfg: config.Config, rows: int, out: Path, harness=run.Harness, trace: dict | None = None) -> dict:
    """The record of `cfg`'s first `STEPS` steps as a job counts them (`run.train` under `harness`, with the job's
    trace, taken here unless `trace` is given), θ the model's own initialisation and the stream `Tokens` at the
    record's `rows` a forward. Written to `out/run.json`."""
    route, shape = config.route(cfg.route), cfg.start.shape
    torch.manual_seed(0)
    theta = {k: v.detach().clone() for k, v in LM.of(shape).state_dict().items()}
    te = harness(theta, out, start={"shape": dict(shape)}, digests=False)
    with te.counter:
        model, optimizers = route.build(shape, cfg.dials)
        run.train(te, cfg, route, Tokens(shape, rows), model, optimizers, steps=STEPS)
    return te.finish({"trace": trace or run.trace_step(cfg, route, model, optimizers, "cpu")})


def _log(rec: dict, weights_as_one: bool = False) -> dict:
    """The log of a counted record's compared step as it was counted, no rows divided out: `{(phase, kind, op,
    shapes): [count, MACs]}`, with the forward's products of an updated weight and of θ's as one kind if asked."""
    out: dict[tuple, list] = {}
    for phase, kind, op, shapes, count, macs in rec["structures"][rec["steps"][STEPS - 1]["struct"]]["log"]:
        if weights_as_one and (phase, kind) in (("fwd", "updated"), ("fwd", "theta")):
            kind = "weight"
        e = out.setdefault((phase, kind, op, shapes), [0, 0])
        e[0] += count
        e[1] += macs
    return out


def tables(cfg: config.Config, rows: int, out: Path, own: dict) -> tuple[dict, dict | None, list[str]]:
    """The position tables. `run.Harness` names θ's weights alone, so its counter takes a buffer of the model that θ
    doesn't hold for a weight the route made, and a product with one for an updated weight's; the records' harness
    checked such a buffer like θ's weights, so in a record that product is θ's. Counts the step again as the records'
    harness did (`AsRecorded`, into `out/tables`): the count the comparison then reads, the expected difference if it
    fired, and the differences. Fires when the two counts differ. Its exact case: they differ in the forward's
    products alone, some an updated weight's in the harness's own count (`own`) that are θ's in the other, the same
    op, shapes, count and MACs over the two kinds together, so the step's MACs are equal. The two counts are set side
    by side as counted, both at the same rows a forward: a table is no token's, so the harness's own count of its
    uses has no one number of rows to divide out."""
    again = counted(cfg, rows, out / "tables", AsRecorded, own["trace"])
    a, b = _log(own), _log(again)
    uses = [c["structures"][c["steps"][STEPS - 1]["struct"]]["updated"] for c in (own, again)]
    if a == b and uses[0] == uses[1]:
        return again, None, []
    macs = [c["steps"][STEPS - 1]["macs"] for c in (own, again)]
    if _log(own, True) != _log(again, True) or macs[0] != macs[1]:
        return again, None, [
            "the position tables: the harness's own count and the count with them checked like θ's weights differ "
            f"in more than which of the forward's products are an updated weight's (the step's MACs: {macs[0]:,} "
            f"and {macs[1]:,})"]
    moved = [(v[0] - b.get(k, [0, 0])[0], v[1] - b.get(k, [0, 0])[1])
             for k, v in a.items() if k[:2] == ("fwd", "updated") and b.get(k) != v]
    return again, {"rule": "tables", "what": (
        "the model's position tables checked like θ's weights, as the records' harness checked them, where the "
        f"harness takes them for weights the route made: {sum(n for n, _ in moved):,} of the forward's products "
        f"({sum(m for _, m in moved):,} MACs a step at one row a forward) are an updated weight's in the harness's own "
        f"count and θ's here, and its own count has {len(uses[0])} uses of an updated weight for these "
        f"{len(uses[1])}; the step's MACs are {macs[0]:,} on both counts")}, []


#: what `run.Harness` counts otherwise than the records' harness did. Each entry counts the config's step again as
#: theirs did, and the comparison reads that count; an entry goes when `run.Harness` counts its case as the records do
HARNESS = (tables,)


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
    `named` lines of the record's own program, the `differences`, and `refused`, why the entry's count stands for no
    run, in which case nothing else was compared."""
    route, shape = config.route(cfg.route), cfg.start.shape
    r = {"entry": cfg.name, "run": h["run"], "record": h["path"], "expected": [], "named": [], "differences": [],
         "refused": None}
    struct, n = modal(rec)
    theirs = per_row(struct, shape["ctx"])
    batch, rows = cfg.budget["batch"], theirs["rows"]
    if batch % rows:
        r["differences"].append(f"the batch: the config's {batch} rows are no whole number of the record's {rows} "
                                "rows a forward")
        return r
    mine = counted(cfg, rows, out)
    for rule in HARNESS:
        if refusal(mine):
            break
        mine, found, differences = rule(cfg, rows, out, mine)
        r["expected"] += [found] if found else []
        r["differences"] += differences
    if why := refusal(mine):
        return r | {"refused": why, "expected": [], "differences": []}
    start, cp, last = rec.get("start_record") or rec["start"], mine["checkpoints"][-1], rec["checkpoints"][-1]
    if start["digest"] != cfg.start.digest:
        r["differences"].append(f"the start: the config's is {cfg.start.name} ({cfg.start.digest[:12]}), the "
                                f"record's {start['name']} ({start['digest'][:12]})")
    if len(rec["steps"]) != cfg.budget["steps"]:
        r["differences"].append(f"the steps: the config's {cfg.budget['steps']}, the record's {len(rec['steps'])}")
    found = compare(per_row(mine["structures"][mine["steps"][STEPS - 1]["struct"]], shape["ctx"]), theirs, shape,
                    h.get("record_only", []))
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
        lines += [("named", f"the record's step only: {e['line']}; {HELD.name} names it: {e['why']}")
                  for e in r["named"]]
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
