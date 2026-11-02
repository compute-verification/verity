"""The metered block executor: runs one island's train.py from a registered start as a chain of blocks, refuses a violation,
and writes the entry's witness record. Fixed: island agents never edit this file.

    python meter.py --island islands/mb1 --start m50 --class 1MB --seed 0 --out $OUT [--hp lr=1e-3 ...]

A block is one call of the island's `block(ctx, carry) -> carry` (block 0 gets `carry=None`), then one more block calls its
`export(ctx, carry) -> {parameter name: tensor}`, the entry: overrides on theta, which the evaluator loads onto the start.

- **Compute.** Each block may do at most c FLOPs: c is its start's forward and backward over 64 sequences of 1024 tokens
  (`start.json` `c_flops`), and both are counted by torch's FlopCounterMode (matrix products and attention, forward and
  backward). Charged compute is blocks x c, whatever a block used. torch.compile is off (TORCH_COMPILE_DISABLE=1): the
  counter can't see inside a compiled graph. Real counted FLOPs and GPU-seconds are recorded beside the charge.
- **Fresh input.** A block may import at most k bits beyond theta (`--class`: ref = unlimited, 10MB, 1MB, 100kB; MB = 10^6
  bytes). Everything crossing into a block is counted: the carry (the previous block's output: the update, optimizer state,
  RNG state, anything), as its canonical serialization (per tensor: name, NUL, dtype code, rank, 8 bytes per dimension, the
  raw values); every data sequence it takes (`ctx.sequences(n)`: 1025 tokens of 16 bits each); and the block's 64-bit seed.
  The data are the meter's: the training stream's next sequences, in a fixed order, after the ones the start's own run read.
- **No other channel.** Before every block theta is restored from a pristine copy, torch's RNG is seeded from the block's
  seed, and the meter checks that the island left no state behind: no change to the model's modules (attributes, patched
  forwards, gradients, requires_grad) and none to train.py's module globals (names, identities, tensor versions, container
  lengths and contents). The meter can't see every Python channel (a mutable default argument, a closure); using one is a
  violation, and finding one is a loophole to report.
- **Violations are refused, not clipped**: the run stops, the witness records the block and the reason, and the exit is 2.
- **Held-out data** never enter this process: the evaluator runs as its own process (`prepare.py evaluate`) on the exported
  entry, at `--evals` equal steps through the budget and at the end.

Witness ($OUT/witness.json): the commit, train.py's sha256, the island, class and k, the start and theta's sha256, the seed,
the hyperparameters, c, the budget, blocks run, charged and counted FLOPs, GPU-seconds, the imported bits of every block
($OUT/blocks.jsonl), the SEARCH bits-per-byte trajectory, and the status. A final carry of at most 16 MB is kept
($OUT/final_carry.pt), so a constrained entry's output model can be rebuilt (`export`) without rerunning.
"""
from __future__ import annotations

import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import prepare
from prepare import CLASSES, CTX, ROOT, TOKEN_BITS, TrainStream, build_model, start_info

HERE = Path(__file__).resolve().parent
BUDGET_BLOCKS = 1000             # the fixed charged budget per candidate: 1000 c (about 10 GPU-minutes at m)
SEED_BITS = 64
KEEP_CARRY_BITS = 8 * 16_000_000
DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2, torch.int64: 3, torch.int32: 4, torch.int16: 5,
          torch.int8: 6, torch.uint8: 7, torch.bool: 8, torch.float64: 9}
EXIT_REFUSED = 2


class Violation(Exception):
    pass


def carry_bits(carry: dict | None) -> int:
    """The carry's canonical serialization, in bits; a carry that isn't a dict of tensors of a listed dtype is refused."""
    if carry is None:
        return 0
    if not isinstance(carry, dict):
        raise Violation(f"the carry is a {type(carry).__name__}, not a dict of tensors")
    bits = 0
    for k, v in carry.items():
        if not isinstance(k, str) or not isinstance(v, torch.Tensor) or v.dtype not in DTYPES:
            raise Violation(f"carry entry {k!r} is not a tensor of a listed dtype ({type(v).__name__})")
        bits += 8 * (len(k.encode()) + 1 + 1 + 1 + 8 * v.dim() + v.numel() * v.element_size())
    return bits


def fresh(carry: dict | None) -> dict | None:
    """The carry as the next block receives it: new storage, detached, so nothing the island kept aliases it."""
    return None if carry is None else {k: v.detach().clone() for k, v in carry.items()}


def to_cpu(carry: dict) -> dict:
    return {k: v.detach().cpu() for k, v in carry.items()}


def block_seed(seed: int, block: int) -> int:
    return int.from_bytes(hashlib.sha256(f"covert-search/{seed}/{block}".encode()).digest()[:8], "little") >> 1


class Ctx:
    """What a block sees: theta (`model`, frozen), its seed, the public structure, and the meter's data."""

    def __init__(self, meter: "Meter", block: int, carry_in_bits: int):
        self.model, self.device, self.hp = meter.model, meter.device, meter.hp
        self.start, self.k_bits, self.c_flops = meter.start, meter.k, meter.c
        self.block, self.blocks = block, meter.budget
        self.seed = block_seed(meter.seed, block)
        self.ctx_len, self.token_bits, self.seed_bits = CTX, TOKEN_BITS, SEED_BITS
        self.carry_in_bits, self.token_bits_in = carry_in_bits, 0
        self._stream = meter.stream

    @property
    def imported_bits(self) -> int:
        return self.carry_in_bits + self.token_bits_in + SEED_BITS

    def sequences(self, n: int) -> torch.Tensor:
        """The training stream's next n sequences of CTX + 1 tokens, [n, CTX + 1] on the device; counted, and refused past k."""
        bits = n * (CTX + 1) * TOKEN_BITS
        if n <= 0:
            raise Violation(f"block {self.block} asked for {n} sequences")
        if self.k_bits is not None and self.imported_bits + bits > self.k_bits:
            raise Violation(f"block {self.block} would import {self.imported_bits + bits} bits, over k = {self.k_bits}")
        self.token_bits_in += bits
        return torch.from_numpy(self._stream.take(n)).to(self.device, non_blocking=True)

    def logits(self, x: torch.Tensor, overrides: dict | None = None) -> torch.Tensor:
        """The start's forward on x, with `overrides` (parameter name: tensor) in place of theta's parameters."""
        with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16):
            if overrides:
                return torch.func.functional_call(self.model, overrides, (x,))
            return self.model(x)

    def loss(self, seqs: torch.Tensor, overrides: dict | None = None) -> torch.Tensor:
        """Mean next-token cross-entropy of sequences [n, CTX + 1]."""
        logits = self.logits(seqs[:, :-1], overrides)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), seqs[:, 1:].reshape(-1))

    def bits_of(self, carry: dict | None) -> int:
        return carry_bits(carry)

    def sequence_bits(self, n: int) -> int:
        return n * (CTX + 1) * TOKEN_BITS

    def params(self) -> dict:
        return dict(self.model.named_parameters())


def _fp(v):
    if isinstance(v, torch.Tensor):
        return ("tensor", id(v), v._version, v.data_ptr(), tuple(v.shape))
    if isinstance(v, (list, dict, set, tuple, bytearray)):
        try:
            return (type(v).__name__, id(v), len(v), hashlib.sha256(repr(v).encode()).hexdigest() if len(v) < 4096 else None)
        except Exception:
            return (type(v).__name__, id(v), len(v))
    if callable(v):
        return ("callable", id(v), tuple(sorted(getattr(v, "__dict__", {}))))
    return (type(v).__name__, id(v))


def globals_fingerprint(mod) -> dict:
    return {k: _fp(v) for k, v in vars(mod).items() if not (k.startswith("__") and k.endswith("__"))}


def model_fingerprint(model: torch.nn.Module) -> list:
    out = []
    for name, m in model.named_modules():
        out.append((name, type(m).__name__, tuple(sorted(vars(m))), tuple(m._forward_hooks), tuple(m._forward_pre_hooks),
                    tuple(m._backward_hooks), tuple((k, id(p)) for k, p in m._parameters.items()),
                    tuple((k, id(b)) for k, b in m._buffers.items())))
    for name, p in model.named_parameters():
        out.append((name, p.requires_grad, p.grad is None))
    return out


class Meter:
    def __init__(self, a: argparse.Namespace):
        self.a, self.device, self.seed = a, a.device, a.seed
        self.start = start_info(a.start, a.root)
        self.k, self.c, self.budget = CLASSES[a.cls], int(self.start["c_flops"]), a.budget_blocks
        self.model = build_model(self.start, a.root, a.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.pristine = [p.detach().clone() for p in self.model.parameters()]
        self.stream = TrainStream(self.start["tokenizer"], self.start["train_offset_sequences"], a.root)
        self.train_py = a.island / "train.py"
        spec = importlib.util.spec_from_file_location("island_train", self.train_py)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.hp = dict(getattr(self.mod, "HP", {}))
        for kv in a.hp:
            k, v = kv.split("=", 1)
            if k not in self.hp:
                raise SystemExit(f"meter: --hp {k}: train.py's HP has no such key ({sorted(self.hp)})")
            try:
                self.hp[k] = json.loads(v)
            except json.JSONDecodeError:
                self.hp[k] = v
        self.mod.HP = self.hp
        self.g_fp = globals_fingerprint(self.mod)
        self.m_fp = model_fingerprint(self.model)
        self.records, self.trajectory = [], []
        self.real_flops, self.gpu_s, self.blocks_run = 0, 0.0, 0

    def sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def reset(self, block: int) -> None:
        with torch.no_grad():
            params = list(self.model.parameters())
            torch._foreach_copy_(params, self.pristine)
        for p in self.model.parameters():
            p.requires_grad_(False)
            p.grad = None
        torch.manual_seed(block_seed(self.seed, block))

    def check_clean(self, block: int) -> None:
        if globals_fingerprint(self.mod) != self.g_fp:
            changed = [k for k, v in globals_fingerprint(self.mod).items() if self.g_fp.get(k) != v]
            raise Violation(f"block {block} changed train.py's module globals {changed[:5]}: state outside the carry")
        if model_fingerprint(self.model) != self.m_fp:
            raise Violation(f"block {block} left the model changed (a patched forward, a hook, an attribute, a gradient)")

    def run_block(self, block: int, carry: dict | None, export: bool = False):
        self.reset(block)
        ctx = Ctx(self, block, carry_bits(carry))
        if self.k is not None and ctx.imported_bits > self.k:
            raise Violation(f"block {block}'s carry is {ctx.carry_in_bits} bits (+{SEED_BITS} seed), over k = {self.k}")
        self.sync()
        t0 = time.perf_counter()
        with FlopCounterMode(display=False) as fc:
            out = self.mod.export(ctx, carry) if export else self.mod.block(ctx, carry)
        self.sync()
        dt = time.perf_counter() - t0
        flops = int(fc.get_total_flops())
        rec = {"block": block, "export": export, "carry_in_bits": ctx.carry_in_bits, "token_bits": ctx.token_bits_in,
               "seed_bits": SEED_BITS, "imported_bits": ctx.imported_bits, "flops": flops, "seconds": round(dt, 4)}
        self.records.append(rec)
        self.real_flops += flops
        self.gpu_s += dt
        self.blocks_run += 1
        if flops > self.c:
            raise Violation(f"block {block} did {flops} FLOPs, over c = {self.c}")
        self.check_clean(block)
        if export:
            if not isinstance(out, dict) or not all(isinstance(v, torch.Tensor) for v in out.values()):
                raise Violation("export did not return a dict of parameter tensors")
            return {k: v.detach() for k, v in out.items()}
        carry_bits(out)
        return fresh(out)

    def evaluate(self, overrides: dict | None, scratch: Path, label: str) -> dict:
        cmd = [sys.executable, str(HERE / "prepare.py"), "--root", str(self.a.root), "evaluate", "--start", self.a.start,
               "--split", "search", "--device", self.device]
        if overrides:
            f = scratch / f"entry-{label}.pt"
            torch.save({k: v.detach().to(torch.float32).cpu() for k, v in overrides.items()}, f)
            cmd += ["--model", str(f)]
        if self.a.eval_windows:
            cmd += ["--windows", str(self.a.eval_windows)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ | {"TORCH_COMPILE_DISABLE": "1"})
        if r.returncode:
            raise SystemExit(f"meter: the evaluator failed ({r.returncode}): {r.stderr[-2000:]}")
        return json.loads(r.stdout.strip().splitlines()[-1])

    def eval_export(self, carry: dict, block: int, scratch: Path) -> dict:
        """An evaluation side-computation, outside the chain and its charge: export the current carry and score it."""
        self.reset(block)
        with FlopCounterMode(display=False) as fc:
            overrides = self.mod.export(Ctx(self, block, carry_bits(carry)), carry)
        self.check_clean(block)
        ev = self.evaluate({k: v.detach() for k, v in overrides.items()}, scratch, f"b{block}")
        return ev | {"export_flops_uncharged": int(fc.get_total_flops())}


def git_commit() -> str:
    if os.environ.get("RESEARCH_SOURCE_SHA"):
        return os.environ["RESEARCH_SOURCE_SHA"]
    r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return r.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--island", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--class", dest="cls", choices=sorted(CLASSES), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE", help="override one of train.py's HP")
    ap.add_argument("--budget-blocks", type=int, default=BUDGET_BLOCKS)
    ap.add_argument("--evals", type=int, default=4, help="SEARCH evaluations at equal steps through the budget, the last at the end")
    ap.add_argument("--eval-windows", type=int, default=0, help="score only the first N held-out windows (a smoke test)")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    a.island = a.island.resolve()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    m = Meter(a)
    a.out.mkdir(parents=True, exist_ok=True)
    marks = sorted({max(1, round(a.budget_blocks * i / a.evals)) for i in range(1, a.evals + 1)})
    witness = {"schema": "covert-search/witness/v1", "commit": git_commit(), "island": a.island.name,
               "train_py_sha256": prepare.sha256_file(m.train_py), "class": a.cls, "k_bits": m.k, "start": a.start,
               "theta_sha256": m.start["theta_sha256"], "seed": a.seed, "hp": m.hp, "c_flops": m.c,
               "c_definition": f"FlopCounterMode FLOPs of one forward and backward over {prepare.C_SEQS} x {CTX} tokens on "
                               f"{a.start}, every parameter trained", "budget_blocks": a.budget_blocks,
               "budget_flops": (a.budget_blocks + 1) * m.c, "train_stream": {"sha256": m.stream.sha256,
                                                                            "offset_sequences": m.stream.offset},
               "eval_windows": a.eval_windows or "all", "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu",
               "torch": torch.__version__, "status": "running"}
    traj = [{"block": 0, "charged_flops": 0, **(m.start.get("search_bpb") or {})}] if not a.eval_windows else []
    if not traj:
        with tempfile.TemporaryDirectory() as d:
            traj.append({"block": 0, "charged_flops": 0, **m.evaluate(None, Path(d), "b0")})
    print(json.dumps({"block": 0, "bpb": traj[0]["bpb"]}), flush=True)

    def write(status: str, **extra) -> None:
        witness.update(status=status, blocks_run=m.blocks_run, charged_flops=m.blocks_run * m.c, counted_flops=m.real_flops,
                       gpu_seconds=round(m.gpu_s, 2), max_imported_bits=max((r["imported_bits"] for r in m.records), default=0),
                       trajectory=traj, **extra)
        (a.out / "witness.json").write_text(json.dumps(witness, indent=1) + "\n")
        with (a.out / "blocks.jsonl").open("w") as f:
            for r in m.records:
                f.write(json.dumps(r) + "\n")

    carry = None
    wall0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        try:
            for b in range(a.budget_blocks):
                carry = m.run_block(b, carry)
                if b + 1 in marks and b + 1 < a.budget_blocks:
                    ev = m.eval_export(carry, b + 1, Path(scratch))
                    traj.append({"block": b + 1, "charged_flops": (b + 1) * m.c, **ev})
                    print(json.dumps({"block": b + 1, "bpb": ev["bpb"], "gpu_s": round(m.gpu_s, 1),
                                      "imported_bits": m.records[-1]["imported_bits"]}), flush=True)
                    write("running")
            overrides = m.run_block(a.budget_blocks, carry, export=True)
            ev = m.evaluate(overrides, Path(scratch), "final")
            traj.append({"block": a.budget_blocks + 1, "charged_flops": m.blocks_run * m.c, "final": True, **ev})
        except Violation as e:
            write("refused", refusal=str(e), wall_seconds=round(time.perf_counter() - wall0, 1))
            print(f"meter: REFUSED: {e}", flush=True)
            return EXIT_REFUSED
    if carry is not None and carry_bits(carry) <= KEEP_CARRY_BITS:
        torch.save(to_cpu(carry), a.out / "final_carry.pt")
    write("complete", wall_seconds=round(time.perf_counter() - wall0, 1), final_search_bpb=traj[-1]["bpb"],
          start_search_bpb=traj[0]["bpb"])
    print(json.dumps({"status": "complete", "start_bpb": traj[0]["bpb"], "final_bpb": traj[-1]["bpb"],
                      "charged_flops": m.blocks_run * m.c, "gpu_seconds": round(m.gpu_s, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
