"""Pearl+ for `verity bench overhead`: the PoUW harness's arm (`benchmarks/pouw/harness/arm.py`, v0.6) over the served
Pearl-C call on an H100 or H200 (`verity_vllm.protocols.pouw.pearl_c` under `SERVED`, on this directory's kernels, the
serial schedule), and its ε. The door registers both in `verity_bench.overhead.SCHEMES`; the three names it fixes are each
one definition here: `SCHEME` (Pearl+'s id in the door), `verifier` (the reference verifier's declaration) and `epsilon`.

- **The cubin** is the file `$VERITY_BENCH_CUBIN` names (the art the door sent), listed in the build record's ``binaries``
  so the SASS gate passes it before anything is timed. Without it the arm builds one into the run's directory
  (``build.sh`` with ``CUBIN_ONLY=1``), for development only (the record's ``cubin.source``). The cubin of record is
  `PINNED`'s art (`pinned`), the one `verity bench overhead run --scheme pearl-plus --cubin ART` sends: the record's
  ``cubin.art`` names it when the loaded bytes are its, and is None otherwise.
- **The call** (`prepare`): one `PearlCDeviceRun` a weight (the operand set's B, BF16, read in place; its root and weight
  side at construction), whose served call (`gemm`, at call index `INDEX` in a pass of its own) makes the call's shape,
  pipeline and keys. `launch` is that call's device work on the harness's stream, into the buffers a served call at a
  pass's first position writes: x widened into A's FP32 words, the call's steps, and with hashing A's root into the pass and
  the records' screen. The keys are the served call's, uploaded by it, so `launch` makes no host-device copy; their SHA-512
  and the pass's bookkeeping, a served call's host work too, are not in it (the record's ``host``). `weight_side` is the
  served weight side's device work: B widened to FP32, B's row tree, then the job (B's screen, on the host, is not in it).
- **The gate**: y's m rows and, with hashing, A's root, the tile leaves and the screen's flags equal, byte for byte, the
  served call's on the same operands at the same index (``all`` each).
- **The transcript** (`dump`): the harness poisons what it reads (`transcript_buffers`: A's words and tree, A's root, the
  leaves and the flags) and launches the call, which is then recorded and its pass ended as a retained served pass ends
  (`end`: the tile tree; A's and B's rows and trees, the leaves and the manifest `verify_run.py` reads), with the epoch's
  coin commitment beside it (``coins.json``). The no-write control is the same over the poisoned buffers.
- **The stock scaling** (`FAMILY`) is rowwise: the chain multiplies A's and B's FP8 codes, each row under its own scale
  (``stats_s5``), and y is its words under A's row scales and B's (``ia_a``, ``ia_b``).
- **ε** (`epsilon`) is the window of record's, at its sample (`PINNED`'s ``window``), the same at every linear; refused
  while no window is stated.

The arm imports `arm`, torch and the served path only where the harness runs it, so the door's node side can import it
for `epsilon` alone.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arm import GateResult, OperandSet, Shape, Work

__all__ = ["PearlPlus", "epsilon"]

HERE = Path(__file__).resolve().parent
#: Pearl+'s id in the door (`verity_bench.overhead.SCHEMES`), and the arm's name in its transcripts' paths
SCHEME = "pearl-plus"
#: the served scheme the arm runs: `verity_vllm.protocols.pouw.pearl_c.SCHEMES`' H100 record, this directory's kernels
SERVED = "pearl-c-h100-v1-h3"
#: the stock scaling the door times beside it, its useful output's (`baselines.FAMILIES`)
FAMILY = "fp8-e4m3-rowwise"
#: the build target this directory's cubin is for (`identity.CARDS`)
ARCH = "sm_90a"
#: where the door's cubin arrives (`verity_bench.overhead_node.CUBIN_ENV`)
CUBIN_ENV = "VERITY_BENCH_CUBIN"
#: the cubin of record (``cubin``: its art id, sha256 and size) and, once one is stated, the window of record `epsilon`
#: reads (``window``)
PINNED = HERE / "overhead_arm.json"
#: the reference verifier, from the run's working directory (the source tree's root)
VERIFY_RUN = "benchmarks/pouw/pearl_c_vllm/verify_run.py"
#: the call index of every call the arm makes: the served reference's, whose keys the launch reads
INDEX = 0
#: the challenger's epoch for the arm's coins
EPOCH = 1
#: the price unit of `work`'s W_ref and credit: one dense FP8 MAC on the device (`pearl_c_device.Prices`)
UNIT = "fp8-mac"
#: the δ `epsilon` states ε at unless given one; γ doesn't depend on it.
DELTA = Fraction(1, 100)


def verifier() -> dict:
    """The build record's ``verifier`` (`arm.check_verifier`): `verify_run.py` on each shape's retained pass, which names
    the manifest's sha256 on its ACCEPT or REJECT line, under the coins `dump` writes beside it; the no-write control's at
    the same path under ``negative/``, which it must reject."""
    return {"name": "verify_run.py: the reference replay of a retained Pearl-C pass", "per": "shape",
            "transcript": "transcripts/{arm}/{shape}/manifest.json", "control": "negative/{arm}/{shape}/manifest.json",
            "commands": [["{python}", VERIFY_RUN, "{transcript}", "--coins", "{dir}/coins.json", "--commit", "{commit}"]]}


def epsilon(m: int, n: int, k: int, card: str, delta: Fraction = DELTA) -> dict:
    """Pearl+'s soundness error beside γ at one linear (m, n, k) on `card`, as the door records it: ``{"epsilon", "delta",
    "basis"}``. PoUW's ε is a window's sample's, not a linear's, so it is the window of record's at every linear:
    `PINNED`'s ``window`` states the window (``basis``), the card it ran on, its sample's K and its credited and whole work
    (C, W), and ε is `window.eps_at(K, δ, C, W, 1)` at `delta`, as `credit.established` gives a window's. Refused
    while no window is stated, and on another card than the window's."""
    w = json.loads(PINNED.read_text()).get("window")
    if w is None:
        raise LookupError(f"{SCHEME} has no ε at {m} x {n} x {k} on {card}: PoUW's ε is a window's sample's "
                          f"(window.eps_at), and {PINNED.name} states no window of record")
    if w["card"] != card:
        raise ValueError(f"the window of record ({w['basis']}) ran on the {w['card']}, not the {card}")
    from verity.protocols.pouw.verifier import window as Wd

    eps = Wd.eps_at(w["k"], Fraction(delta), w["credited"], w["work"], 1)
    return {"epsilon": float(eps), "delta": float(delta),
            "basis": f"{w['basis']}: window.eps_at(K={w['k']}, δ={delta}, C={w['credited']}, W={w['work']}, 1) = {eps}"}


def pinned() -> dict:
    """The cubin of record, `PINNED`'s ``{"art", "sha256", "bytes"}``."""
    return json.loads(PINNED.read_text())["cubin"]


def cubin_of(sent: str | None, out_dir: Path, build: Callable[[Path], Path] | None = None) -> tuple[Path, str]:
    """The cubin the arm loads and how it came: the file `sent` names (``$VERITY_BENCH_CUBIN``'s, ``sent``), else one
    `build` (default `build_dev`) makes in `out_dir` (``development``)."""
    if sent:
        path = Path(sent)
        if not path.is_file():
            raise FileNotFoundError(f"${CUBIN_ENV} names {sent}, which isn't a file")
        return path.absolute(), "sent"
    return (build or build_dev)(out_dir / "pearl-plus-cubin"), "development"


def build_dev(out: Path) -> Path:
    """build.sh's cubin and SASS gates alone (``CUBIN_ONLY=1``) into `out`."""
    log = out.with_name(out.name + ".log")
    out.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        rc = subprocess.run(["bash", str(HERE / "build.sh"), str(out)], env={**os.environ, "CUBIN_ONLY": "1"}, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"build.sh exited {rc} (see {log}): send the cubin instead (`verity bench overhead run --cubin ART`)")
    return out / "pearl_c_sm90.cubin"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def m16(m: int) -> int:
    """The served call's rows: m rounded up to the domain's row granularity, the filler rows zero."""
    return -(-m // 16) * 16


def settings() -> Any:
    from verity_config.settings import of

    return of(__file__)


def kernels() -> Any:
    """This directory's run.py, for its constants off the device (on it, `Kernel.load` loads its own copy, whose
    `Pipeline` the arm's buffers are sized by)."""
    name = "pearl_c_sm90_overhead_run"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "run.py")
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


def tiles_of(m: int, n: int) -> int:
    R = kernels()
    return R.cdiv(m16(m), R.TILE) * (n // R.TILE)


class PearlPlus:
    name = SCHEME
    stream = "fp8"
    precision = FAMILY
    inputs = "bf16"
    flag_sets = {"hash": frozenset({"hash"}), "nohash": frozenset()}

    def __init__(self) -> None:
        self.cubin: Path | None = None
        self._kernel: Any = None
        self._held: dict[tuple, _Held] = {}
        self._streams: dict[int, Any] = {}

    def build(self, ctx: dict) -> dict:
        if ctx["arch"] != ARCH:
            raise RuntimeError(f"{SCHEME} runs on {ARCH} (an H100 or H200), not {ctx['arch']}: time it with --card h200")
        cubin, source = cubin_of(settings().VERITY_BENCH_CUBIN, Path(ctx["out_dir"]))
        self.cubin, sha, pin = cubin, sha256(cubin), pinned()
        return {"scheme": SCHEME, "served": SERVED, "binaries": [str(cubin)],
                "cubin": {"path": str(cubin), "sha256": sha, "source": source,
                          "art": pin["art"] if sha == pin["sha256"] else None},
                "schedule": "serial", "graphs": False, "index": INDEX,
                "host": "launch enqueues the call's device work; the served call's keys (native SHA-512 and a pinned "
                        "upload), the pass's bookkeeping and B's screen run once, in prepare",
                "verifier": verifier()}

    def domain(self, shape: Shape) -> str | None:
        from verity_vllm.protocols.pouw import pearl_c as PPD

        bad = PPD.refusals(SERVED, {0: (shape.n, shape.k)})
        if bad:
            return "; ".join(b.split(": ", 1)[-1] for b in bad)
        R = kernels()
        if 4 * shape.k > R.RS_MAX_SEGS * 1024:
            return f"A's row digests take rows of at most {R.RS_MAX_SEGS} 1 KiB segments (k <= {R.RS_MAX_SEGS * 256})"
        if m16(shape.m) > PPD.FLAG_ROWS:
            return f"a pass screens at most {PPD.FLAG_ROWS} rows"
        return None

    def work(self, shape: Shape) -> Work:
        """The call's W_ref and credit at its m₁₆ rows (the scheme's `wref` and `credit_of`), the chain's FP8 MACs, A's
        rows (FP32 words) and the tile leaves (C̃ and U, FP32 words, over the m₁₆ rows a partial row tile has) the call
        hashes, and B's rows its weight side does."""
        from arm import Work
        from verity.protocols.pouw.verifier.protocol import Shape as Unit
        from verity_vllm.protocols.pouw import pearl_c as PPD

        s, m = PPD.scheme_of(SERVED), m16(shape.m)
        unit = Unit(m, shape.k, shape.n)
        return Work(tensor_macs={FAMILY: m * shape.k * shape.n}, w_ref=str(s.wref(unit)), credited=str(s.credit_of(unit)),
                    unit=UNIT, hashed_bytes={"a_rows": 4 * m * shape.k, "tile_leaves": 8 * m * shape.n},
                    weight_hashed_bytes={"b_rows": 4 * shape.n * shape.k})

    def prepare(self, dev: Any, shape: Shape, flags: frozenset[str], operands: OperandSet) -> _Call:
        key = (operands.b.ptr, shape.m, shape.n, shape.k)
        if key not in self._held:
            self._held[key] = _Held(self, key, operands, shape)
        return _Call(self._held[key], shape, "hash" in flags, operands)

    def gate(self, dev: Any, call: _Call, shape: Shape, flags: frozenset[str]) -> GateResult:
        from arm import GateResult

        dev.sync()
        return GateResult(*call.check())

    def dump(self, dev: Any, call: _Call, shape: Shape, out: Path) -> None:
        call.retain(Path(out))

    # ---- the device ---------------------------------------------------------------------------------------------------

    def kernel(self) -> Any:
        """The served path's kernel on the built cubin, loaded once."""
        if self._kernel is None:
            from verity_vllm.protocols.pouw import pearl_c as PPD

            if self.cubin is None:
                raise RuntimeError("prepare before build: the harness builds an arm first")
            k = PPD.Kernel.load(str(HERE), str(self.cubin))
            if not k.dev.has("widen_a"):
                raise RuntimeError(f"{self.cubin} has no widen_a: the launch widens x on the device, as the served call does")
            self._kernel = k
        return self._kernel

    def on(self, stream: int) -> Any:
        """torch's stream object for the harness's `stream`."""
        import torch

        if stream not in self._streams:
            self._streams[stream] = torch.cuda.ExternalStream(stream)
        return self._streams[stream]


class _Cuda:
    """A harness buffer as torch takes one in place (the CUDA array interface): `shape` 16-bit words."""

    def __init__(self, ptr: int, shape: tuple[int, int]):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": "<i2", "data": (ptr, False), "version": 3,
                                         "strides": None}


def _bf16(buf, rows: int, cols: int) -> Any:
    import torch

    if buf.nbytes < 2 * rows * cols:
        raise ValueError(f"a {buf.nbytes}-byte buffer holds no {rows} x {cols} BF16")
    return torch.as_tensor(_Cuda(buf.ptr, (rows, cols)), device="cuda").view(torch.bfloat16)


def _buffer(t: Any):
    from arm import Buffer

    return Buffer(t.data_ptr(), t.numel() * t.element_size())


def _on(kernel: Any, stream: int, steps: list) -> None:
    kernel.dev.stream = stream
    try:
        for f in steps:
            f()
    finally:
        kernel.dev.stream = None


class _Held:
    """One weight's `PearlCDeviceRun` at one shape, shared by the calls prepared on it, and its weight side's device work
    (`PearlCDeviceRun._weight_side`'s pipeline over the run's weight buffers)."""

    def __init__(self, arm: PearlPlus, key: tuple, operands, shape):
        import torch
        from verity.protocols.pouw.verifier import verifier as PV
        from verity.protocols.pouw.verifier.audit import weights_root
        from verity.protocols.pouw.verifier.coins import Challenger
        from verity_vllm.protocols.pouw import pearl_c as PPD

        self.arm, self.key, self.users = arm, key, 0
        self.w = _bf16(operands.b, shape.n, shape.k)
        kernel = arm.kernel()
        salt_key = PPD.new_salt_key(SERVED)
        roots = PPD.weight_roots(kernel, {0: self.w}, SERVED, salt_key=salt_key)
        ch = Challenger(EPOCH, PV.coins)
        self.run = PPD.PearlCDeviceRun(PPD.scheme_of(SERVED), f"pouw:{SERVED}", kernel, {0: self.w}, roots, ch.commitment,
                                       ch.open("salt", weights_root(roots)), capacity=tiles_of(shape.m, shape.n),
                                       flag_rows=m16(shape.m), salt_key=salt_key)
        R, wt = kernel.run, self.run.weights[0]
        size = R.Pipeline.sizes(R.BM, wt.n, wt.k, R.BM)
        self.xb = torch.empty((wt.n, wt.k), dtype=torch.float32, device=self.run.device)
        self.seg = PPD._zeros(size["segdig_b"], self.run.device)
        self.wpipe = PPD._pipeline(kernel, R.BM, wt.n, wt.k,
                                   {**{x: t.data_ptr() for x, t in wt.b.items()}, "line_a": 0, "segdig_b": self.seg.data_ptr()},
                                   self.xb.data_ptr(), {"line_b": wt.keys.data_ptr(), "dom_b": wt.keys.data_ptr() + 64,
                                                        **self.run.const_keys}, variant=self.run.variant, weight=0)
        self.wsteps = PPD._steps(self.wpipe, PPD.EPOCH_STEPS + PPD.JOB_STEPS)

    def weight_side(self, stream: int) -> None:
        import torch

        with torch.cuda.stream(self.arm.on(stream)):
            self.xb.copy_(self.w)
        _on(self.run.kernel, stream, self.wsteps)

    def release(self) -> None:
        self.users -= 1
        if self.users:
            return
        import torch

        for pipe, _, _ in self.run.pipes.values():
            pipe.free()
        self.wpipe.free()
        self.arm._held.pop(self.key, None)
        self.run = self.w = self.xb = self.seg = None
        torch.cuda.empty_cache()


class _Call:
    """One operand set's call under one flag set (the arm interface's `Call`)."""

    def __init__(self, held: _Held, shape, hashing: bool, operands):
        import torch

        held.users += 1
        self.held, self.hashing = held, hashing
        self.m, self.m16, self.n, self.k = shape.m, m16(shape.m), shape.n, shape.k
        self.x = _bf16(operands.a, shape.m, shape.k)
        run = held.run
        self.reference()
        self.sh = run.shapes[(self.m16, self.n, self.k)]
        self.pipe, steps, side = run._pipe(0, self.sh)
        if side:
            raise RuntimeError(f"{SCHEME}: the serial schedule put steps on a side lane")
        self.steps = list(steps)
        self.tiles = tiles_of(self.m, self.n)
        self.y = torch.empty((self.sh.mp, self.n), dtype=torch.bfloat16, device=run.device)
        nb = run.nb
        self.views = {"y": self.y[:self.m]}
        if hashing:
            self.views |= {"root_a": run.roots_buf[:nb], "leaves": run.leaves[:nb * self.tiles],
                           "flags": run.flag_buf[:self.m]}
        self.outputs = {name: _buffer(t) for name, t in self.views.items()}
        if hashing:
            from arm import Buffer

            self.transcript_buffers = {name: self.outputs[name] for name in ("root_a", "leaves", "flags")} | {
                "xa": Buffer(self.sh.xa.data_ptr(), 4 * self.m16 * self.k),
                "tree_a": Buffer(self.sh.b["tree_a"].data_ptr(), self.pipe.size["tree_a"])}

    def launch(self, stream: int) -> None:
        run = self.held.run
        K, nb = run.kernel, run.nb
        K.run.widen_a(K.dev, self.x.data_ptr(), self.sh.xa.data_ptr(), self.m, self.m16, self.k, stream=stream)
        self.pipe.b["y"], self.pipe.b["leaves"] = self.y.data_ptr(), run.leaves.data_ptr()
        keys = self.sh.keys.data_ptr()
        self.pipe.keys.update(dom_a=keys, unit=keys + nb)
        _on(K, stream, self.steps)
        if self.hashing:
            import torch
            from verity_vllm.protocols.pouw import pearl_c as PPD

            size = self.pipe.size["tree_a"]
            K.dev.copy_async(run.roots_buf.data_ptr(), self.sh.b["tree_a"].data_ptr() + size - nb, nb, stream)
            with torch.cuda.stream(self.held.arm.on(stream)):
                PPD.screen_into(self.views["flags"], self.x, self.sh.b["alpha_a"], self.sh.b["aux_a"], run.fused_screen)

    def weight_side(self, stream: int) -> None:
        self.held.weight_side(stream)

    def reference(self) -> Any:
        """The served call on the call's operands at `INDEX`, in a pass of its own: y's m rows (with hashing, A's root,
        the leaves and the flags are where the launch's views read them)."""
        import torch

        run = self.held.run
        run.hashing = self.hashing
        run.begin(at=INDEX)
        sh = run.shapes.get((self.m16, self.n, self.k))
        if sh is not None:
            sh.dirty = self.m16
        y = run.gemm(0, self.x, self.held.w, None)
        torch.cuda.synchronize()
        return y

    def check(self) -> tuple[bool, dict[str, str], dict]:
        """The gate: the launch's outputs against the served call's, byte for byte."""
        import torch

        mine = {name: t if name == "y" else t.clone() for name, t in self.views.items()}
        ref = {"y": self.reference()} | {name: t for name, t in self.views.items() if name != "y"}
        same = {name: bool(torch.equal(mine[name], ref[name])) for name in mine}
        return all(same.values()), {name: "all" for name in mine}, {"equal_to_served_call": same, "index": INDEX}

    def retain(self, out: Path) -> None:
        """The pass the launch wrote, ended as a retained served pass ends, into `out`, and the epoch's coin commitment
        beside it."""
        import torch
        from verity.protocols.pouw.verifier.protocol import Matmul
        from verity.protocols.pouw.verifier.protocol import Shape as Unit
        from verity_vllm.protocols.pouw import pearl_c as PPD

        if not self.hashing:
            raise RuntimeError(f"{SCHEME}: only a hashing call has a transcript")
        run = self.held.run
        torch.cuda.synchronize()
        run.hashing = True
        run.begin(at=INDEX)
        cur = run.cur
        cur.retain, cur.run = out, settings().RESEARCH_RUN_ID
        mm = Matmul(PPD.call_index(run), Unit(self.m16, self.k, self.n), 0)
        cur.matmuls.append(mm)
        cur.flag_rows, cur.tiles = self.m, self.tiles
        run._retained(cur, mm, self.m, self.views["flags"], *run._kept(self.sh, self.pipe))
        run.end()
        (out / "coins.json").write_text(json.dumps(run.coins.to_json()) + "\n")

    def free(self) -> None:
        if self.held is None:
            return
        self.views, self.outputs, self.y = {}, {}, None
        self.held.release()
        self.held = None
