"""The linears a served model runs: `ve.cu` through ctypes, one object per linear layer, holding either the plain FP8 fragments
(`PlainLinear`, the native arm's operand in the fused kernel's own order) or only the codec words (`EncodedLinear`).

vLLM calls a linear through the op ``ve_enc::linear(x8, xs, idx)`` (opaque to torch.compile), on activations its own FP8
kernel's quantization produced (dynamic, per token where CUTLASS FP8 is supported).  At M <= 32 rows (a decode batch) it runs
one of `wc.cu`'s three GEMM forms (the `Config` the arm picked for this shape and M) and reduces the chunks' sums in order to
BF16 (`ve_reduce`), so a plain and an encoded linear over the same weight return the same bits.  At more rows (prefill) it
decodes the layer into a scratch FP8 matrix for vLLM's CUTLASS GEMM: prefill is compute-bound and isn't what's timed.

Kernels are pluggable: a `Lib` is any build of `ve.cu` (packing, k range), and a later fused kernel (a tag stage, a wider
codec) is another `Lib` whose `ve_gemm` takes the same arguments.  `WideLinear` runs ra-codec-wide's fused GEMM instead
(`benchmarks/word_codec/wide.cu`, a `WideLib` a field: Sloth over p = 2^127 − 1, 2^255 − 19, 2^521 − 1, 2^256 − 189 or
2^512 − 569, and with a tag seed set the bandwidth audit's ro-blake3/v1 tag of every block of encoded words it reads, or
with its sketch set ro-sketch-blake3/v0 over the encoded words its CTAs hold, `wide_sketch`); its chunk sums have the
same layout, so `ve_reduce` turns them into the same BF16.  `TensorLinear` runs ra-codec-tensor's
(`benchmarks/codec_tensor/wt.cu`, a `TensorLib` a field: decoder warps decoding in shared memory ahead of one MMA warp, and
tag warps hashing the store's 4,088-byte blocks, or sketch warps sketching the stored bytes on the tensor cores before the
decoders overwrite them, ro-sketch-blake3/v0), with the same chunk sums.
"""
from __future__ import annotations

import ctypes
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import layout
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORD_CODEC = HERE.parent / "word_codec"
CODEC_TENSOR = HERE.parent / "codec_tensor"
MAX_ROWS = 32
FORMS = ("simple", "pipe", "ws", "wide", "tensor")
SMEM_PER_BLOCK = 101376  # sm_120's opt-in dynamic shared memory a CTA


def _load(name: str, file: str, where: Path = WORD_CODEC):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, where / file)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _codec():
    return _load("wc_codec", "codec.py")


def _wide():
    return _load("wc_wide", "wide.py")


def _wide_sketch():
    _wide()
    sys.modules.setdefault("wide", sys.modules["wc_wide"])  # wide_sketch.py imports its sibling as `wide`
    return _load("wc_wide_sketch", "wide_sketch.py")


def _ct():
    return _load("ct_wide", "wide.py", CODEC_TENSOR)


def _sketching():
    if str(CODEC_TENSOR) not in sys.path:
        sys.path.append(str(CODEC_TENSOR))  # sketching.py imports its sibling `tagging`
    return _load("ct_sketching", "sketching.py", CODEC_TENSOR)


def stream() -> int:
    return torch.cuda.current_stream().cuda_stream


class Lib:
    """One build of `ve.cu`: its packing (`bits`) and largest k (`kmax`)."""

    def __init__(self, path: str | Path) -> None:
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.path = str(path)
        so = self.so = ctypes.CDLL(self.path)
        so.wc_set_key.argtypes = [v, i, q]
        so.wc_encode.argtypes = [v, v, q, q, i, i, v]
        so.wc_decode.argtypes = [v, v, q, q, i, v, i, v]
        so.ve_prepare.argtypes = [i, i]
        so.ve_occupancy.argtypes = [i, i, i, i, i]
        so.ve_gemm.argtypes = [i, i, v, v, v, i, i, i, q, i, i, i, i, v]
        so.ve_reduce.argtypes = [v, i, i, i, v, i, v, v, v]
        so.ve_constants.argtypes = [v]
        so.wc_spin.argtypes = [ctypes.c_longlong, v]
        cs = (ctypes.c_int * 8)()
        so.ve_constants(cs)
        self.constants = list(cs)
        self.bits, self.kmax = self.constants[6], self.constants[7]
        lw = layout.lane_words(self.bits)
        if self.constants[:5] != [layout.LANES, layout.TASK_TILES, lw, lw * layout.LANES, layout.TASK_COLS]:
            raise RuntimeError(f"{path}: layout constants {self.constants} are not layout.py's")
        self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.key = None

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{Path(self.path).name}: {fn} error {rc}")

    def set_key(self, seed: bytes, k: int):
        key = _codec().key(seed, k)
        c = (ctypes.c_uint64 * k)(*key.c)
        self.call("wc_set_key", c, k, key.tau)
        self.key = key
        return key


class WideLib:
    """One build of `wide.cu` (one field, `WW_BITS`): its key, its tag seed and the tag buffer its GEMMs write."""

    def __init__(self, path: str | Path) -> None:
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.path = str(path)
        so = self.so = ctypes.CDLL(self.path)
        so.ww_constants.argtypes = [v]
        so.ww_set_key.argtypes = [v, i, v, v, v, v]
        so.ww_decode.argtypes = [v, v, q, q, i, v, i, v]
        so.ww_encode.argtypes = [v, v, q, q, i, i, v]
        so.ww_gemm.argtypes = [i, v, v, v, i, i, i, q, i, v, v, i, v]
        so.ww_gemm_occupancy.argtypes = [i, i, i, i]
        so.ww_gemm_sketch.argtypes = [i, v, v, v, i, i, i, q, i, v, v, q, v, v, i, v]
        so.ww_workspace_bytes.argtypes = [q]
        so.ww_workspace_bytes.restype = ctypes.c_size_t
        so.ww_workspace_tags.argtypes = [v, q]
        so.ww_workspace_tags.restype = v
        so.ww_sketch_commit.argtypes = [v, q, v, v, v]
        so.ww_sketch_leaves.argtypes = [v, q, v, q, q, v, v]
        so.ww_sketch_fold.argtypes = [v, q, q, q, v]
        so.ww_sketch_tree.argtypes = [v, q, i, v, v]
        cs = (ctypes.c_int * 32)()
        so.ww_constants(cs)
        self.constants = list(cs[:21])
        self.bits, self.limbs, self.payload, self.lane_words, self.kmax = (cs[0], cs[1], cs[2], cs[3], cs[7])
        self.kw, self.sub = cs[15], cs[18]
        self.field = _wide().field(self.bits)
        f = self.field
        if ((self.limbs, self.payload, self.lane_words, cs[4], cs[5]) !=
                (f.limbs, f.payload, layout.wide_lane_words(f.payload), layout.LANES, layout.TASK_TILES)):
            raise RuntimeError(f"{path}: constants {self.constants} are not wide.py's and layout.py's")
        self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.key = None
        self.tag_seed: bytes | None = None
        self._seed = None
        self.tags: torch.Tensor | None = None
        self.sketch: dict | None = None
        self.signs: torch.Tensor | None = None
        self.step: dict | None = None

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{Path(self.path).name}: {fn} error {rc}")

    def set_sketch(self, seed: bytes) -> None:
        """Tag under ro-sketch-blake3/v0 with this window seed (`wide_sketch`: k_s = 16, a vector a row of a CTA's tile of
        T row blocks at one chunk): the GEMMs sketch the encoded words they hold and hash the sketches."""
        ws, f = _wide_sketch(), self.field
        if self.kw != 1:
            raise ValueError(f"{Path(self.path).name}: the sketch is a KW = 1 build's")
        self.set_tag(seed)
        self.sign_matrix = ws.signs(seed, f)
        self.signs = torch.from_numpy(ws.sign_fragments(self.sign_matrix, ws.steps(f))).cuda()
        self.sketch = {"name": ws.NAME, "k": ws.K, "t": ws.T, "n": ws.vector_bytes(f)}

    def vectors(self, n: int, kd: int) -> int:
        return 16 * _wide_sketch().tiles(n, kd)

    def commit_every_step(self, per_linear: list[int]) -> None:
        """Each decode step's commit as the step goes (``per_linear``: each linear's vectors, in step order): a linear's
        tags go to one slot of a two-slot ring, which L2 holds, and the next linear hashes their leaves into the step's
        tree on a second stream beside its own GEMM; when every linear's tags are whole 16·2^sub, a third stream folds
        those leaves `sub` levels, waiting only for them, so the ring slot is free once the leaves have read it
        (ra-codec-wide's run E); the step's last linear hashes its own and the tree."""
        if self.sketch is None:
            raise ValueError("the in-step commit is the sketch tag's")
        vectors, n = sum(per_linear), -(-sum(per_linear) // 16)
        nodes = 0
        while True:
            nodes += n
            if n == 1:
                break
            n = (n + 1) // 2
        nodes_t = torch.empty(8 * nodes, dtype=torch.int64, device="cuda")
        # `ww_sketch_*` find the tree's nodes after a tag area of the step's tags that they never read: only the nodes
        # are allocated, and the workspace pointer is placed that area before them.
        tag_area = -(-32 * vectors // 256) * 256
        self.step = {"ring": [torch.empty(8 * max(per_linear), dtype=torch.int32, device="cuda") for _ in range(2)],
                     "nodes": nodes_t, "ws": nodes_t.data_ptr() - tag_area, "vectors": vectors,
                     "fold": all(v % (16 << self.sub) == 0 for v in per_linear), "at": 0, "slot": 0, "pending": None,
                     "side": torch.cuda.Stream(priority=-1), "folds": torch.cuda.Stream(priority=-1), "unjoined": None,
                     "roots": {}}

    def leaves(self, tags: int, at: int, nvec: int, s: int) -> None:
        """The step's leaves of its tags [at, at + nvec), read from ``tags``."""
        st = self.step
        self.call("ww_sketch_leaves", st["ws"], st["vectors"], tags, at, nvec, self._seed, s)

    def fold(self, tags: int, at: int, nvec: int, s: int) -> None:
        """Those leaves folded `sub` levels, when the step folds (``tags`` unread: the leaves are in the nodes)."""
        st = self.step
        if st["fold"]:
            self.call("ww_sketch_fold", st["ws"], st["vectors"], at, nvec, s)

    def tree(self, s: int) -> None:
        st = self.step
        self.call("ww_sketch_tree", st["ws"], st["vectors"], self.sub if st["fold"] else 0, None, s)

    def step_root(self) -> bytes:
        """The root the step's tree left on the device (its last node; SHA-512's state words, big-endian)."""
        return self.step["nodes"][-8:].cpu().numpy().astype(">i8").tobytes()

    def set_key(self, seed: bytes, k: int):
        wide, f = _wide(), self.field
        key = wide.key(f, seed, k)

        def arr(vals):
            return (ctypes.c_uint32 * len(vals))(*vals)

        c, tau = arr([x for cj in key.c for x in wide.limbs(cj, f)]), arr(wide.limbs(key.tau, f))
        if f.three_mod_four:
            self.call("ww_set_key", c, k, tau, None, None, None)
        else:
            i_, c1 = wide.SQRT_M1, wide.C1
            self.call("ww_set_key", c, k, tau, arr(wide.limbs(i_, f)), arr(wide.limbs(c1, f)),
                      arr(wide.limbs(c1 * i_ % f.p, f)))
        self.key = key
        return key

    def set_tag(self, seed: bytes | None) -> None:
        """Tag every block the GEMMs read under ro-blake3/v1 with this window seed (None: no tag)."""
        if seed is not None and len(seed) != 32:
            raise ValueError("a tag seed is 32 bytes")
        self.tag_seed, self._seed = seed, None if seed is None else ctypes.create_string_buffer(seed, 32)

    def tag_buffer(self, blocks: int) -> torch.Tensor:
        """The tags of one call's blocks (32 bytes a block), shared by every linear as the step's partial sums are."""
        if self.tags is None or self.tags.numel() < 8 * blocks:
            self.tags = torch.empty(8 * blocks, dtype=torch.int32, device="cuda")
        return self.tags


class TensorLib:
    """One build of ra-codec-tensor's `wt.cu` (one field, `WT_FIELD`): its key, its tag key and the tag buffer its fused
    GEMMs write, and each launch's tweak steps."""

    def __init__(self, path: str | Path) -> None:
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.path = str(path)
        so = self.so = ctypes.CDLL(self.path)
        so.wt_constants.argtypes = [v]
        so.wt_set_key.argtypes = [v, i, v, v]
        so.wt_encode.argtypes = [v, v, q, q, i, i, v]
        so.wt_decode.argtypes = [v, v, q, q, i, v, i, v]
        so.wt_fused_sketch.argtypes = [i, v, v, v, i, i, i, q, i, i, i, i, i, v, v, q, i, v, v, i, v, v]
        so.wt_sketch_vectors.argtypes = [i, i, i, i]
        so.wt_sketch_vectors.restype = q
        so.wt_set_tag.argtypes = [v, v]
        so.wt_tag_leaves.argtypes = [v, q, v, v]
        so.wt_tag_tree.argtypes = [v, q, v, v]
        cs = (ctypes.c_int * 15)()
        so.wt_constants(cs)
        self.constants = list(cs)
        (self.bits, self.limbs, self.word_bytes, self.payload_bytes, self.task_words, self.kmax, self.sb,
         self.fused_warps, self.fused_span, self.tag_block) = (cs[0], cs[1], cs[2], cs[3], cs[4], cs[10], cs[11], cs[12],
                                                                cs[13], cs[14])
        ct = _ct()
        self.field = next(f for f in ct.FIELDS.values() if f.n == self.bits)
        f = self.field
        if ((self.word_bytes, self.payload_bytes, self.task_words, cs[6], self.sb) !=
                (f.word_bytes, f.payload_bytes, ct.TASK_WORDS[f.name], f.c, ct.TASK_WORDS[f.name] * f.word_bytes)):
            raise RuntimeError(f"{path}: constants {self.constants} are not codec_tensor/wide.py's {f}")
        self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.key = None
        self.tag_seed: bytes | None = None
        self.sketch: dict | None = None
        self.tags: torch.Tensor | None = None
        self.step: dict | None = None
        self._steps: dict[tuple, tuple[int, torch.Tensor]] = {}
        self._vectors: dict[tuple, int] = {}

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{Path(self.path).name}: {fn} error {rc}")

    def _limbs(self, x: int) -> bytes:
        return x.to_bytes(4 * self.limbs, "little")

    def set_key(self, seed: bytes, k: int):
        ct = _ct()
        key = ct.key(self.field.name, seed, k)
        c = ctypes.create_string_buffer(b"".join(self._limbs(x) for x in key.c))
        tau, p = ctypes.create_string_buffer(self._limbs(key.tau)), ctypes.create_string_buffer(self._limbs(self.field.p))
        self.call("wt_set_key", c, k, tau, p)
        self.key, self._steps = key, {}
        return key

    def set_tag(self, seed: bytes | None) -> None:
        """Tag every block of E under ro-blake3/v1 with this window seed (None: no tag)."""
        if seed is not None:
            from verity.core.protocols.bandwidth import tags
            if len(seed) != 32:
                raise ValueError("a tag seed is 32 bytes")
            key = ctypes.create_string_buffer(tags.Blake3Tag().key(seed))
            pre = ctypes.create_string_buffer(tags.prefix(tags.LEAF_DOMAIN, "ro-blake3/v1", seed))
            self.call("wt_set_tag", key, pre)
        self.tag_seed, self.sketch = seed, None

    def set_sketch(self, seed: bytes, k: int, t: int = 8) -> None:
        """Tag the store under ro-sketch-blake3/v0 instead (`codec_tensor/sketching.py`, k = 1 only): the sketch warps'
        k_s = k signed sums of each vector of t consecutive tasks of a CTA, then each vector's BLAKE3 tag under this
        window seed."""
        skt = _sketching()
        if len(seed) != 32:
            raise ValueError("a tag seed is 32 bytes")
        bits = skt.sign_bits(seed, k, skt.vector_bytes(self.sb, t))
        words = skt.sign_fragments(bits, skt.steps(self.sb), t)
        key, pre = ctypes.create_string_buffer(skt.key(seed)), ctypes.create_string_buffer(skt.leaf_prefix(seed))
        self.call("wt_set_tag", key, pre)
        self.tag_seed, self._vectors = seed, {}
        self.sketch = {"name": skt.NAME, "k": k, "t": t, "bits": bits,
                       "signs": torch.from_numpy(words.view("<i4").copy()).cuda()}

    def tag_vectors(self, n: int, kd: int) -> int:
        """The most tags one launch over an n × kd linear writes: its 4,088-byte blocks, or sketched, its vectors at the
        CTA groups that maximise them (`wt_sketch_vectors`; the launch's groups depend on the batch and config)."""
        if self.sketch is None:
            return -(-layout.tasks(n, kd) * self.sb // self.tag_block)
        nbs, t, skt = n // 16, self.sketch["t"], _sketching()
        return 16 * (kd // layout.TASK_COLS) * max(g * skt.tiles_per_cta(nbs, g, t) for g in range(1, nbs + 1))

    def tag_buffer(self, blocks: int) -> torch.Tensor:
        if self.tags is None or self.tags.numel() < 8 * blocks:
            self.tags = torch.empty(8 * blocks, dtype=torch.int32, device="cuda")
        return self.tags

    def vectors(self, n: int, kd: int, groups: int) -> int:
        """The sketch vectors (tags) one sketched launch over an n × kd linear at ``groups`` CTA groups writes."""
        at = (n, kd, groups)
        if at not in self._vectors:
            self._vectors[at] = self.so.wt_sketch_vectors(n, kd, groups, self.sketch["t"])
        return self._vectors[at]

    def commit_every_step(self, vectors: int) -> None:
        """Commit each decode step (sketched only): the step's sketched launches write their tags one after another into
        one buffer of at most ``vectors`` tags, vector v of a launch at index j = (the tags before it in the step) + v, and
        the step's last linear hashes the leaves over them (16 tags a leaf) and the RFC 6962 tree over the leaves
        (`wt_tag_leaves`, `wt_tag_tree`), on the stream, so a CUDA graph of the step holds the commit."""
        if self.sketch is None:
            raise ValueError("the in-step commit is the sketch tag's")
        leaves = -(-vectors // 16)
        self.step = {"tags": torch.empty(8 * vectors, dtype=torch.int32, device="cuda"),
                     "nodes": torch.empty(8 * (2 * leaves + 64), dtype=torch.int64, device="cuda"),
                     "vectors": vectors, "at": 0, "roots": {}}

    def commit(self, nvec: int, m: int) -> None:
        st = self.step
        if not 0 < nvec <= st["vectors"]:
            raise RuntimeError(f"a step's {nvec} tags overflow its commit buffer of {st['vectors']}")
        root = ctypes.c_ulonglong(0)
        self.call("wt_tag_leaves", st["tags"].data_ptr(), nvec, st["nodes"].data_ptr(), stream())
        self.call("wt_tag_tree", st["nodes"].data_ptr(), -(-nvec // 16), ctypes.byref(root), stream())
        st["roots"][m] = (nvec, root.value)

    def configs(self, tagged: bool) -> list[tuple[int, int, int, int]]:
        """(decoders, stages, taggers, k_s) as ra-codec-tensor's `fused_cfgs("auto")` picks them: untagged (taggers
        -1), the most decoder warps the field's registers allow at 4 and 6 stages fewer than fit at M = 32; tagged, 0
        or 2 of them tagging from the start instead (every other warp tags once its own work is done); sketched, the
        most decoder warps and 2 fewer (ra-codec-tensor's p1536 picks at T = 32: 8 x 8 for qkv and o, 10 x 9 for
        gate_up and down), all of them sketching the words they decode, at 0 to 4 stages fewer than fit with the
        sketch's buffers and fewer than 2 t (the kernel puts k_s = 16's signs in shared memory when they fit beside
        the stages).  Not 4 or 6 tag warps: with that many gate_up's sums at M = 32 differ from launch to launch (p511 and
        p767, r20261012-092414-7dcc and r20261012-092945-296e)."""
        def smax(ks: int) -> int:
            return (SMEM_PER_BLOCK - 4 * 8 * 528 - 32 - 192 * ks) // (self.sb + 40)

        dmax = self.fused_warps - 2

        def need(d: int) -> int:
            return -(-32 * d // self.task_words)

        if self.sketch is not None:
            ks, t = self.sketch["k"], self.sketch["t"]
            return list(dict.fromkeys((d, min(2 * t - 1, max(need(d), smax(ks) - f)), -1, ks)
                                      for d in (dmax, dmax - 2) if d >= self.fused_span for f in range(5)))
        if not tagged:
            return list(dict.fromkeys([(dmax, max(need(dmax), smax(0) - 4), -1, 0),
                                       (dmax, max(need(dmax), smax(0) - 6), -1, 0)]))
        return [(dmax - t, max(need(dmax - t), smax(0) - 4), t, 0) for t in (0, 2) if dmax - t >= self.fused_span]

    def fused(self, k: int, w, x, y, m: int, n: int, kd: int, base: int, partials: int, cfg: Config, groups: int,
              tsteps, tags, tag_base: int, launched=None) -> None:
        sk = self.sketch if cfg.sketch_k else None
        self.call("wt_fused_sketch", k, w, x, y, m, n, kd, base, partials, cfg.consumers, cfg.stages, groups,
                  max(cfg.taggers, 0), tsteps, tags, tag_base, cfg.sketch_k,
                  sk["signs"].data_ptr() if sk else None, None, sk["t"] if sk else 1, launched, stream())

    def steps(self, m: int, n: int, kd: int, cfg: Config) -> tuple[int, torch.Tensor]:
        """The launch's CTA groups (as many as fill the device) and its tweak steps on the device (`bench.tsteps`): a
        lane's next unit is 32·D words on, its task A or A + 1 on, consecutive tasks groups·nchunks apart."""
        nb8 = 1 if m <= 8 else 2 if m <= 16 else 4
        at = (nb8, n, kd, cfg)
        if at not in self._steps:
            launched = ctypes.c_int(0)
            self.fused(self.key.rounds, None, None, None, m, n, kd, 0, 0, cfg, 0, None, None, 0, ctypes.byref(launched))
            nch = kd // layout.TASK_COLS
            groups = launched.value // nch
            adv, rem = divmod(32 * cfg.consumers, self.task_words)
            gq = groups * nch
            d = (adv * gq * self.task_words + rem, (adv + 1) * gq * self.task_words + rem - self.task_words)
            st = b"".join(self._limbs(self.key.tau * x % self.field.p) for x in d)
            self._steps[at] = (groups, torch.frombuffer(bytearray(st), dtype=torch.uint8).cuda())
        return self._steps[at]


@dataclass(frozen=True)
class Config:
    """One GEMM form and its launch: `groups` CTA groups a chunk for the persistent forms (0: as many as fit), the
    warp-specialized form's consumer warps and ring stages (the tensor GEMM's decoder warps and stages), the wide GEMM's
    L2 prefetch distance in CTAs, and the tensor GEMM's warps tagging from the start (-1: no tag) or its decoders'
    sketch k_s."""

    form: int
    groups: int = 0
    consumers: int = 0
    stages: int = 0
    ahead: int = 0
    taggers: int = 0
    sketch_k: int = 0

    @property
    def name(self) -> str:
        if self.form == 3:
            return "wide" + (f"-ahead{self.ahead}" if self.ahead else "")
        if self.form == 4:
            return f"tensor-d{self.consumers}s{self.stages}" + (
                f"k{self.sketch_k}" if self.sketch_k else f"t{self.taggers}" if self.taggers >= 0 else "")
        return FORMS[self.form] + (f"-{self.consumers}x{self.stages}" if self.form == 2 else "")


#: the warp-specialized form's (consumers, stages), equal: `wc.cu`'s consumers take tasks i = c, c + C, ... from stage i mod S
#: and wait on its parity alone, which is unambiguous only when each stage has one consumer
WS = ((2, 2), (3, 3), (4, 4), (6, 6), (8, 8))


def candidates(lib: Lib, k: int, m: int, nchunks: int) -> list[Config]:
    out = [Config(0)]
    per = lib.so.ve_occupancy(1, k, m, 0, 0)
    if per > 0:
        out.append(Config(1, max(1, lib.sms * per // nchunks)))
    for c, s in WS:
        lib.call("ve_prepare", k, s)
        per = lib.so.ve_occupancy(2, k, m, c, s)
        if per > 0:
            out.append(Config(2, max(1, lib.sms * per // nchunks), c, s))
    return out


class Linear:
    """One layer's operand on the device.  `k` is the codec's rounds (0: plain fragments)."""

    k = 0

    def __init__(self, lib: Lib, n: int, kd: int, weight_scale: torch.Tensor) -> None:
        layout.check_shape(n, kd)
        self.lib, self.n, self.kd, self.ws = lib, n, kd, weight_scale.reshape(1).float()
        self.nchunks = kd // layout.TASK_COLS
        self.configs: dict[int, Config] = {}
        self.base = 0
        self.store: torch.Tensor

    @property
    def nbytes(self) -> int:
        return self.store.numel() * self.store.element_size()

    def default(self) -> Config:
        return Config(0)

    def candidates(self, m: int) -> list[Config]:
        return candidates(self.lib, self.k, m, self.nchunks)

    def config(self, m: int) -> Config:
        for mm in sorted(self.configs):
            if mm >= m:
                return self.configs[mm]
        return self.default()

    def gemm(self, x8: torch.Tensor, cfg: Config | None = None, partials: bool = True,
             part: torch.Tensor | None = None) -> torch.Tensor:
        m = x8.shape[0]
        cfg = cfg or self.config(m)
        shape = (self.nchunks if partials else 1, m, self.n)
        if part is None:
            part = (torch.empty if partials else torch.zeros)(shape, dtype=torch.float32, device=x8.device)
        elif tuple(part.shape) != shape or part.dtype != torch.float32 or not part.is_contiguous() or not partials:
            raise ValueError(f"partials buffer {tuple(part.shape)} {part.dtype} for {shape}")
        self._launch(cfg, x8, part, partials)
        return part

    def _launch(self, cfg: Config, x8: torch.Tensor, part: torch.Tensor, partials: bool) -> None:
        self.lib.call("ve_gemm", cfg.form, self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), x8.shape[0], self.n,
                      self.kd, self.base, int(partials), cfg.groups, cfg.consumers, cfg.stages, stream())

    def reduce(self, part: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """BF16 out of the chunks' sums; ``xs`` is one activation scale or one a row (FP32, contiguous)."""
        m = part.shape[1]
        if xs.numel() not in (1, m) or xs.dtype != torch.float32 or not xs.is_contiguous():
            raise ValueError(f"activation scale {tuple(xs.shape)} {xs.dtype} for {m} rows")
        out = torch.empty((m, self.n), dtype=torch.bfloat16, device=part.device)
        self.lib.call("ve_reduce", part.data_ptr(), part.shape[0], m, self.n, xs.data_ptr(), int(xs.numel() > 1),
                      self.ws.data_ptr(), out.data_ptr(), stream())
        return out

    def dense(self) -> torch.Tensor:
        """W (n × k, FP8) rebuilt from the operand (prefill's scratch copy)."""
        raise NotImplementedError

    def forward(self, x8: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """BF16 x W^T from FP8 rows ``x8`` and their activation scale ``xs`` (one, or one a row)."""
        if x8.shape[0] > MAX_ROWS:
            from vllm import _custom_ops as ops
            return ops.cutlass_scaled_mm(x8, self.dense().t(), out_dtype=torch.bfloat16, scale_a=xs, scale_b=self.ws, bias=None)
        return self.reduce(self.gemm(x8.contiguous()), xs.reshape(-1).float().contiguous())


class PlainLinear(Linear):
    def __init__(self, lib: Lib, w: torch.Tensor, weight_scale: torch.Tensor) -> None:
        super().__init__(lib, *w.shape, weight_scale)
        self.store = layout.plain_fragments(w)

    def dense(self) -> torch.Tensor:
        return layout.unplain(self.store, self.n, self.kd).view(torch.float8_e4m3fn)


class EncodedLinear(Linear):
    """Only the codec words: `store` holds W's payloads encoded under the lib's key at word indices base, base + 1, ..."""

    def __init__(self, lib: Lib, w: torch.Tensor, weight_scale: torch.Tensor, k: int, base: int, grid: int = 0) -> None:
        super().__init__(lib, *w.shape, weight_scale)
        if lib.key is None or lib.key.rounds != k or not 1 <= k <= lib.kmax:
            raise ValueError(f"the lib's key is not set for k = {k}")
        self.k, self.base, self.bits = k, base, lib.bits
        payloads = layout.payloads(w, self.bits)
        self.store = torch.empty_like(payloads)
        lib.call("wc_encode", payloads.data_ptr(), self.store.data_ptr(), payloads.numel(), base, k,
                 grid or 4 * lib.sms * 8, stream())
        self.payloads = payloads

    def decoded(self) -> torch.Tensor:
        out = torch.empty_like(self.store)
        xr = torch.zeros(1, dtype=torch.int64, device=self.store.device)
        self.lib.call("wc_decode", self.store.data_ptr(), out.data_ptr(), self.store.numel(), self.base, self.k,
                      xr.data_ptr(), 4 * self.lib.sms * 8, stream())
        return out

    def dense(self) -> torch.Tensor:
        return layout.unpayload(self.decoded(), self.n, self.kd, self.bits).view(torch.float8_e4m3fn)

    def drop_payloads(self) -> None:
        self.payloads = None


class WideLinear(Linear):
    """Only the wide field's words: `store` (int32) holds W's payloads (`layout.wide_payloads`) encoded k rounds under the
    wide lib's key at word indices base, base + 1, ... (k = 0: the payloads themselves, the wide kernel's plain operand).
    With the lib's tag seed set, each GEMM also writes the ro-blake3/v1 tag of block j = base / L + 32·task + lane (that
    lane's L encoded words) to the lib's tag buffer, or with its sketch set, the ro-sketch-blake3/v0 tag of its vector v
    at index tag_base + v.  `lib` (a `ve.cu` build) only reduces the chunk sums."""

    def __init__(self, lib: Lib, wlib: WideLib, w: torch.Tensor, weight_scale: torch.Tensor, k: int, base: int,
                 grid: int = 0, tag_base: int | None = None) -> None:
        super().__init__(lib, *w.shape, weight_scale)
        if k and (wlib.key is None or wlib.key.rounds != k or not 1 <= k <= wlib.kmax):
            raise ValueError(f"the wide lib's key is not set for k = {k}")
        if base % (layout.LANES * wlib.lane_words):
            raise ValueError("a wide store starts at a multiple of 32·L words")
        self.wlib, self.k, self.base, self.bits = wlib, k, base, wlib.payload
        self.words = layout.wide_store_words(self.n, self.kd, wlib.payload)
        self.blocks = wlib.vectors(self.n, self.kd) if wlib.sketch else self.words // wlib.lane_words
        self.tag_base = base // wlib.lane_words if tag_base is None else tag_base
        self.first = self.last = self.seal = False
        self.at: int | None = None
        self.fork = self.join = self.folded = None
        payloads = layout.wide_payloads(w, wlib.payload, wlib.limbs)
        if k:
            self.store = torch.empty_like(payloads)
            wlib.call("ww_encode", payloads.data_ptr(), self.store.data_ptr(), self.words, base, k, grid or 4 * wlib.sms,
                      stream())
        else:
            self.store = payloads
        self.payloads = payloads

    @property
    def tag_block_base(self) -> int:
        return self.tag_base

    def default(self) -> Config:
        return Config(3)

    def candidates(self, m: int) -> list[Config]:
        w = self.wlib
        occ = w.so.ww_gemm_occupancy(self.k, m, 3 if w.sketch else int(w.tag_seed is not None), 1)
        return [Config(3)] + ([Config(3, ahead=occ * w.sms)] if occ > 0 else [])

    def _launch(self, cfg: Config, x8: torch.Tensor, part: torch.Tensor, partials: bool) -> None:
        if cfg.form != 3:
            raise ValueError(f"the wide GEMM has one form, not {cfg.name}")
        w = self.wlib
        if w.sketch and self.at is not None:
            self.sketched(x8, part, partials, cfg, w.step["ring"][w.step["slot"]].data_ptr(), self.at)
        elif w.sketch:
            self.sketched(x8, part, partials, cfg, w.tag_buffer(self.blocks).data_ptr(), self.tag_base)
        else:
            tags = w.tag_buffer(self.blocks).data_ptr() if w.tag_seed is not None else None
            w.call("ww_gemm", self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), x8.shape[0], self.n,
                   self.kd, self.base, int(partials), w._seed, tags, cfg.ahead, stream())

    def sketched(self, x8: torch.Tensor, part: torch.Tensor, partials: bool, cfg: Config, tags: int, jbase: int) -> None:
        """The GEMM with its vectors' tags, vector v's (index jbase + v) at ``tags`` + 32 v."""
        w = self.wlib
        w.call("ww_gemm_sketch", self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), x8.shape[0], self.n,
               self.kd, self.base, int(partials), w.signs.data_ptr(), None, jbase, w._seed, tags, cfg.ahead, stream())

    def forward(self, x8: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """As `Linear.forward`; under the lib's in-step commit, the step's first linear starts its tags at index 0 and
        each next one's after the previous one's, in the other ring slot, while the previous one's leaves are hashed on
        the second stream (forked and joined within this call) and, folding, folded on the third, which waits only for
        them and joins at the next `seal`ed linear (the last before a CUDA-graph piece ends, at attention), so every
        graph vLLM captures, piecewise or whole, holds both ends of each fork; the last one hashes its own leaves and
        the tree."""
        w, m = self.wlib, x8.shape[0]
        st = w.step
        if st is None or m > MAX_ROWS:
            return super().forward(x8, xs)
        if self.first:
            st["at"], st["slot"], st["pending"], st["unjoined"] = 0, 1, None, None
        at, end, pend = st["at"], st["at"] + self.blocks, st["pending"]
        if end > st["vectors"]:
            raise RuntimeError(f"a step's tags overflow its commit of {st['vectors']}")
        cur = torch.cuda.current_stream()
        if pend is not None:
            if self.fork is None:
                self.fork, self.join, self.folded = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
            self.fork.record(cur)
            st["side"].wait_event(self.fork)
            w.leaves(*pend, st["side"].cuda_stream)
            self.join.record(st["side"])
            if st["fold"]:
                st["folds"].wait_event(self.join)
                w.fold(*pend, st["folds"].cuda_stream)
                self.folded.record(st["folds"])
                st["unjoined"] = self.folded
        st["slot"], self.at = 1 - st["slot"], at
        try:
            y = super().forward(x8, xs)
        finally:
            self.at = None
        if pend is not None:
            cur.wait_event(self.join)
        st["at"], st["pending"] = end, (st["ring"][st["slot"]].data_ptr(), at, self.blocks)
        if self.last:
            w.leaves(*st["pending"], cur.cuda_stream)
            w.fold(*st["pending"], cur.cuda_stream)
        if (self.seal or self.last) and st["unjoined"] is not None:
            cur.wait_event(st["unjoined"])
            st["unjoined"] = None
        if self.last:
            w.tree(cur.cuda_stream)
            st["pending"], st["roots"][m] = None, end
        return y

    def decoded(self) -> torch.Tensor:
        if not self.k:
            return self.store
        out = torch.empty_like(self.store)
        xr = torch.zeros(1, dtype=torch.int64, device=self.store.device)
        self.wlib.call("ww_decode", self.store.data_ptr(), out.data_ptr(), self.words, self.base, self.k, xr.data_ptr(),
                       4 * self.wlib.sms, stream())
        return out

    def dense(self) -> torch.Tensor:
        return layout.wide_unpayload(self.decoded(), self.n, self.kd, self.wlib.payload, self.wlib.limbs).view(
            torch.float8_e4m3fn)

    def drop_payloads(self) -> None:
        self.payloads = None


class TensorLinear(Linear):
    """Only `wt.cu`'s words (ra-codec-tensor's fused decode GEMM): `store` (S bytes a word, held as int16 so no uint8
    tensor of a linear's size is in the process: `bytes` is its uint8 view) holds W's payloads (`layout.tensor_limbs`)
    encoded k >= 1 rounds under the tensor lib's key at word indices base, base + 1, ...  With the lib's tag seed set,
    its tag warps write the ro-blake3/v1 tag of each 4,088-byte block jl of the store at index tag_base + jl to the lib's
    tag buffer.  `lib` (a `ve.cu` build) only reduces the chunk sums."""

    def __init__(self, lib: Lib, tlib: TensorLib, w: torch.Tensor, weight_scale: torch.Tensor, k: int, base: int,
                 tag_base: int, grid: int = 0) -> None:
        super().__init__(lib, *w.shape, weight_scale)
        if tlib.key is None or tlib.key.rounds != k or not 1 <= k <= tlib.kmax:
            raise ValueError(f"the tensor lib's key is not set for k = {k}")
        self.tlib, self.k, self.base, self.tag_base, self.bits = tlib, k, base, tag_base, 8 * tlib.payload_bytes
        payloads = layout.tensor_limbs(w, tlib.task_words, tlib.payload_bytes, tlib.limbs)
        self.words = payloads.numel() // tlib.limbs
        if tlib.word_bytes % 2:
            raise ValueError(f"{tlib.field.name}: a word of {tlib.word_bytes} bytes is not whole int16s")
        self.store = torch.empty(self.words * tlib.word_bytes // 2, dtype=torch.int16, device=w.device)
        tlib.call("wt_encode", payloads.data_ptr(), self.store.data_ptr(), self.words, base, k, grid or 4 * tlib.sms,
                  stream())
        self.blocks = tlib.tag_vectors(self.n, self.kd)
        self.groups = 0
        self.first = self.last = False
        self.at: int | None = None
        self.payloads = payloads

    @property
    def bytes(self) -> torch.Tensor:
        return self.store.view(torch.uint8)

    def default(self) -> Config:
        return self.candidates(1)[0]

    def candidates(self, m: int) -> list[Config]:
        return [Config(4, consumers=d, stages=s, taggers=t, sketch_k=ks)
                for d, s, t, ks in self.tlib.configs(self.tlib.tag_seed is not None)]

    def _launch(self, cfg: Config, x8: torch.Tensor, part: torch.Tensor, partials: bool) -> None:
        if cfg.form != 4:
            raise ValueError(f"the tensor GEMM has one form, not {cfg.name}")
        t = self.tlib
        groups, steps = t.steps(x8.shape[0], self.n, self.kd, cfg)
        if self.at is not None:
            tags, tag_base = t.step["tags"].data_ptr() + 32 * self.at, self.at
        else:
            tags = t.tag_buffer(self.blocks).data_ptr() if cfg.taggers >= 0 or cfg.sketch_k else None
            tag_base = self.tag_base
        t.fused(self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), x8.shape[0], self.n, self.kd, self.base,
                int(partials), cfg, groups, steps.data_ptr(), tags, tag_base)
        self.groups = groups

    def forward(self, x8: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """As `Linear.forward`; under the lib's in-step commit, the step's first linear starts its tags at index 0, each
        sketched launch writes its tags after the previous one's, and the last one commits them."""
        t, m = self.tlib, x8.shape[0]
        st = t.step
        if st is None or m > MAX_ROWS:
            return super().forward(x8, xs)
        at = 0 if self.first else st["at"]
        end = at + t.vectors(self.n, self.kd, t.steps(m, self.n, self.kd, self.config(m))[0])
        if end > st["vectors"]:
            raise RuntimeError(f"a step's tags overflow its commit buffer of {st['vectors']}")
        self.at = at
        try:
            y = super().forward(x8, xs)
        finally:
            self.at = None
        st["at"] = end
        if self.last:
            t.commit(end, m)
        return y

    def sketch_vectors(self, vs: list[int], groups: int) -> np.ndarray:
        """Vectors ``vs`` of a sketched launch at ``groups`` CTA groups, as `sketching.vectors` lays them out (vector
        16 (b·TPC + j) + r of CTA b = g0·chunks + kc is row r of tasks T j .. T j + T − 1 of the CTA, row blocks
        g0 + i·groups at chunk kc, zero past the last), read from the device store a task at a time."""
        skt, t = _sketching(), self.tlib.sketch["t"]
        nbs, sb = self.n // 16, self.tlib.sb
        tpc, w = skt.tiles_per_cta(nbs, groups, t), 32 * skt.steps(sb)
        store = self.bytes.view(nbs, self.nchunks, sb)
        out = np.zeros((len(vs), t * w), np.int8)
        for o, v in enumerate(vs):
            q, r = divmod(v, 16)
            b, j = divmod(q, tpc)
            g0, kc = divmod(b, self.nchunks)
            for ii in range(t):
                g = g0 + (t * j + ii) * groups
                if g < nbs:
                    out[o, ii * w:(ii + 1) * w] = skt.task_rows(store[g, kc].cpu().numpy(), 1, 1, sb)[0, 0, r]
        return out

    def decoded(self) -> torch.Tensor:
        t = self.tlib
        out = torch.empty(self.words * t.limbs, dtype=torch.int32, device=self.store.device)
        xr = torch.zeros(1, dtype=torch.int64, device=self.store.device)
        t.call("wt_decode", self.store.data_ptr(), out.data_ptr(), self.words, self.base, self.k, xr.data_ptr(),
               4 * t.sms, stream())
        return out

    def dense(self) -> torch.Tensor:
        t = self.tlib
        return layout.tensor_unlimbs(self.decoded(), self.n, self.kd, t.task_words, t.payload_bytes, t.limbs).view(
            torch.float8_e4m3fn)

    def drop_payloads(self) -> None:
        self.payloads = None


#: the linears ``ve_enc::linear`` dispatches to, by index
LINEARS: list[Linear] = []


@torch.library.custom_op("ve_enc::linear", mutates_args=())
def linear_op(x8: torch.Tensor, xs: torch.Tensor, idx: int) -> torch.Tensor:
    return LINEARS[idx].forward(x8, xs)


@linear_op.register_fake
def _(x8: torch.Tensor, xs: torch.Tensor, idx: int) -> torch.Tensor:
    return x8.new_empty((x8.shape[0], LINEARS[idx].n), dtype=torch.bfloat16)


__all__ = ["FORMS", "LINEARS", "MAX_ROWS", "Config", "EncodedLinear", "Lib", "Linear", "PlainLinear", "TensorLib",
           "TensorLinear", "WideLib", "WideLinear", "candidates", "linear_op", "stream"]
