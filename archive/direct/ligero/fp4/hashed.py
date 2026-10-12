"""``fp4-nvf4+poseidon2`` (alias ``+hash``): the NVFP4 relation with its operands hashed in-circuit (lane fp4-decode).

The NVFP4 unit (``relation.py``) takes 141 verifier-side pins per unit -- the signed E2M1 numerators ``a[i].n`` /
``b[i].n`` (in halves), per scale block ``g[g].sm`` (the two UE4M3 mantissas' product), ``g[g].X`` (the block's weight
exponent), ``g[g].part`` (the block participates: a nonzero product and two nonzero scales) and the anchor ``anc.G``
(the max over participating blocks of ``X - FRAC_BITS``, ``GRID_FLOOR`` when none participates).  :class:`FP4Format`
is the relation's ``hash_format``: ``hashchain.compose`` converts every pin to a hint row re-derived from the committed
operand bits (``pin_names`` / ``pin_order`` / ``decode`` / ``finish``), and the Poseidon2 leaf packs the format's lane
elements (``leaf.poseidon2.LaneFormat``: ``Poseidon2Leaf.with_lanes``):

* **E2M1 code** ``s e1 e0 m`` (4 bit rows): ``p = e0 e1``, ``q = m (1 + e1 + 2 p)``, ``mag = 2 e0 + 4 e1 + 2 p + q``
  (= ``NUMERATOR[code & 7]``, exhaustive in the test), ``n = (1 - 2 s) mag`` -- three product rows per code.
* **UE4M3 scale byte** (8 bit rows): ``b7 = 0`` (the padding bit; PTX ISA: a set bit is invalid), ``e = b3..b6``,
  ``m = b0..b2``, ``nz_e = [e != 0]`` (an inverse pair), ``mant = m + 8 nz_e``, ``exp = e - 9 - nz_e`` (subnormal
  scales: ``m 2^-9``), finite: ``127 - low7 != 0`` (the 0x7F NaN), ``nz_m = [low7 != 0]``.
* **Block** g: ``sm = mant_a mant_b``, ``X = -2 + exp_a + exp_b``, ``any = [sum_i mag_a[i] mag_b[i] != 0]`` (the 16
  magnitude products are >= 0, so the sum is 0 iff every one is), ``part = any nz_m,a nz_m,b``,
  ``cand = GRID_FLOOR + part (X - FRAC_BITS - GRID_FLOOR)``.
* **Anchor**: ``G - cand_g in [0, 256)`` for every block (8 bit rows each) and ``prod_g (G - cand_g) = 0`` (two
  product rows and one quadratic; p prime, every factor < 2^8): G is the maximum.
* **Lanes**: the 64 codes + 4 scale bytes of one operand per step are 72 nibbles = 288 bits, packed 6 nibbles per
  24-bit lane (< 2^24 < p: injective) into rate lanes 0..11, lanes 12..15 zero: one permutation per operand per unit,
  as for BF16 / E4M3.  IV = (role, word_bits 4, 24 x 72 nibbles): its own domain.

The relation object :data:`FP4_HASHED` is what ``relchain.HashedRelationRunner`` drives (``run.py --relation fp4-nvf4
--auth included-hash``, or ``--relation fp4-nvf4+poseidon2``): the same protocol, statement (v5, relation ``fp4-nvf4``:
the Rust verifier's ``hashed_sys_id`` pin on ``FP4_NVF4``), gate and bench as ``fp8-ada+hash``.
"""
from __future__ import annotations

import hashlib

import numpy as np

from verity.ml.circuits.targets.target import NVFP4_SM120
from verity_numerical.bench import contract

from .. import poseidon2 as h2
from ..compile import Expr, LigeroCtx, P, Quadratic, System
from .relation import FRAC_BITS, GRID_FLOOR, GROUP, GROUPS, K, NUMERATOR, NVFP4, SCALES, STEPS, WORDS, compile_fp4_unit
from .witness import MODEL, hints_fp4, public_vectors_fp4

__all__ = ["FP4_HASHED", "FP4Format", "FP4HashedRelation", "HASHED_NAMES", "NIBBLES", "PIN_NAMES", "fp4_hashed_relation"]

#: The 141 verifier-side pins of the unit in the order the device hint generator emits them (``hints_device.PIN_NAMES``;
#: repeated here so this module imports without torch): the order compose converts them to hint rows in.
PIN_NAMES = tuple([f"a[{i}].n" for i in range(K)] + [f"b[{i}].n" for i in range(K)]
                  + [f"g[{g}].sm" for g in range(GROUPS)] + [f"g[{g}].X" for g in range(GROUPS)]
                  + [f"g[{g}].part" for g in range(GROUPS)] + ["anc.G"])
INSTANCE_SEED = 20260922            # fp4/chain.py INSTANCE_SEED (the synthetic VU families)
NIBBLES = K + 2 * SCALES            # 72 lane elements per operand per unit
LANE_BITS = 24
PER_LANE = LANE_BITS // 4           # 6 nibbles per lane
N_LANES = NIBBLES // PER_LANE       # 12 of the 16 rate lanes
BIT_LAYOUT = [4] * K + [8] * SCALES
#: The composition's names on the command line: ``fp4-nvf4+poseidon2`` and its alias ``fp4-nvf4+hash``.
HASHED_NAMES = ("fp4-nvf4+poseidon2", "fp4-nvf4+hash")


def _reduce(e):
    if isinstance(e, Expr):
        return Expr(e.ctx, {k: v % P for k, v in e.lin.items()}, e.c % P)
    return e % P


def _nz(ctx: LigeroCtx, sys: System, tag: str, t: Expr) -> Expr:
    """``nz = [t != 0]`` as the inverse pair ``t * inv = nz``, ``(1 - nz) t = 0``, ``nz`` boolean (2 rows)."""
    one = Expr(ctx, {}, 1)
    inv = ctx._row("inv", f"{tag}.inv", spec=(t,))
    sys.program.append(("inv", next(iter(inv.lin)), t))
    nz = ctx.product(t, inv)
    ctx.boolean(f"{tag}.nz", nz)
    sys.quadratic.append(Quadratic(f"hash.{tag}.nz0", one - nz, t, Expr(ctx)))
    return nz


def sponge_iv(role: int) -> list[int]:
    """The sponge IV of operand ``role``: ``IV(role, 4, 24 x 72)`` (4-bit elements, 72 per step)."""
    return h2.IV(role, 4, STEPS * NIBBLES)


class FP4Format:
    """The NVFP4 operand format (module doc): ``hashchain.compose``'s ``hash_format`` and the Poseidon2 leaf's
    ``LaneFormat`` (68 words per column -> 72 nibbles on 24-bit lanes)."""

    k = WORDS
    elem_bits = 4
    n_elems = NIBBLES
    lane_bits = LANE_BITS
    bit_layout = BIT_LAYOUT

    def pin_names(self) -> set:
        return set(PIN_NAMES)

    def pin_order(self, want: set) -> list:
        return [n for n in PIN_NAMES if n in want]

    def decode(self, ctx: LigeroCtx, sys: System, pins: dict, nm: str) -> tuple[list, list, int, dict]:
        """Operand ``nm``: (72 nibble elements, 288 bit row indices word-major, rows added, aux) -- the codes' numerator
        pins re-derived; the magnitudes, scale mantissas / exponents / nonzero flags handed to :meth:`finish`."""
        one = Expr(ctx, {}, 1)
        E = lambda idx: Expr(ctx, {idx: 1}, 0)  # noqa: E731
        m0 = sys.m
        elems, bit_rows, mags = [], [], []
        for i in range(K):
            tag = f"{nm}[{i}]"
            bits = [ctx.hint(f"{tag}.b{j}") for j in range(4)]
            bit_rows += [next(iter(b.lin)) for b in bits]
            for j, b in enumerate(bits):
                ctx.boolean(f"{tag}.bool{j}", b)
            m, e0, e1, s = bits
            elems.append(_reduce(m + e0 * 2 + e1 * 4 + s * 8))
            p = ctx.product(e0, e1)
            q = ctx.product(m, one + e1 + p * 2)
            mag = e0 * 2 + e1 * 4 + p * 2 + q
            mags.append(mag)
            n = ctx.product(one - s * 2, mag)
            ctx.eq(f"{tag}.n", E(pins[f"{tag}.n"]), n)
        mants, exps, nzm = [], [], []
        for g in range(SCALES):
            tag = f"{nm}[{K + g}]"
            bits = [ctx.hint(f"{tag}.b{j}") for j in range(8)]
            bit_rows += [next(iter(b.lin)) for b in bits]
            for j, b in enumerate(bits):
                ctx.boolean(f"{tag}.bool{j}", b)
            elems.append(_reduce(sum((bits[j] * 2**j for j in range(4)), Expr(ctx))))         # the low nibble
            elems.append(_reduce(sum((bits[4 + j] * 2**j for j in range(4)), Expr(ctx))))     # the high nibble
            ctx.eq(f"{tag}.pad", bits[7], 0)
            m = bits[0] + bits[1] * 2 + bits[2] * 4
            e = bits[3] + bits[4] * 2 + bits[5] * 4 + bits[6] * 8
            low7 = m + e * 8
            nz_e = _nz(ctx, sys, f"{tag}.e", e)
            ctx.inverse_nonzero(f"{tag}.finite", 127 - low7)
            nz_m = _nz(ctx, sys, f"{tag}.m", low7)
            mants.append(m + nz_e * 8)
            exps.append(e - 9 - nz_e)
            nzm.append(nz_m)
        return elems, bit_rows, sys.m - m0, {"mag": mags, "mant": mants, "exp": exps, "nzm": nzm}

    def finish(self, ctx: LigeroCtx, sys: System, pins: dict, A: dict, B: dict) -> int:
        """The block pins (``g[g].sm / .X / .part``) and the anchor ``anc.G`` from both operands' decodes."""
        E = lambda idx: Expr(ctx, {idx: 1}, 0)  # noqa: E731
        m0 = sys.m
        G = E(pins["anc.G"])
        diffs = []
        for g in range(GROUPS):
            sm = ctx.product(A["mant"][g], B["mant"][g])
            ctx.eq(f"g[{g}].sm", E(pins[f"g[{g}].sm"]), sm)
            X = A["exp"][g] + B["exp"][g] - 2
            ctx.eq(f"g[{g}].X", E(pins[f"g[{g}].X"]), X)
            S = Expr(ctx)
            for i in range(GROUP * g, GROUP * (g + 1)):
                S = S + ctx.product(A["mag"][i], B["mag"][i])
            any_ = _nz(ctx, sys, f"g[{g}].any", S)
            part = ctx.product(ctx.product(any_, A["nzm"][g]), B["nzm"][g])
            ctx.eq(f"g[{g}].part", E(pins[f"g[{g}].part"]), part)
            cand = ctx.product(part, X - FRAC_BITS - GRID_FLOOR) + GRID_FLOOR
            d = G - cand
            ctx.rng(f"anc.ge{g}", d, 0, 256)
            diffs.append(d)
        d01 = ctx.product(diffs[0], diffs[1])
        d23 = ctx.product(diffs[2], diffs[3])
        sys.quadratic.append(Quadratic("hash.anc.max", d01, d23, Expr(ctx)))    # G equals one of the candidates
        return sys.m - m0

    # --- witness side ---------------------------------------------------------------------------------------------------
    @staticmethod
    def nibbles_np(words) -> np.ndarray:
        """(N, 68) words -> (N, 72) nibbles: the 64 codes, then each scale byte's low and high nibble."""
        w = np.asarray(words, dtype=np.int64)
        if w.shape[-1] != WORDS or (w < 0).any() or (w[..., :K] >= 16).any() or (w[..., K:] >= 256).any():
            raise ValueError(f"NVFP4 operand rows are (N, {WORDS}): {K} E2M1 codes (< 16) + {SCALES} UE4M3 bytes (< 256)")
        sc = w[..., K:]
        nib = np.stack([sc & 15, sc >> 4], -1).reshape(*w.shape[:-1], 2 * SCALES)
        return np.concatenate([w[..., :K], nib], -1)

    def elements_np(self, words) -> np.ndarray:
        return self.nibbles_np(words)

    def elements_torch(self, words):
        """(l, 68) int64 torch -> (l, 72) nibbles on the same device."""
        import torch
        w = words.to(torch.int64)
        sc = w[:, K:]
        nib = torch.stack([sc & 15, sc >> 4], -1).reshape(w.shape[0], 2 * SCALES)
        return torch.cat([w[:, :K], nib], 1)

    def lanes_np(self, words) -> np.ndarray:
        """(N, 68) words -> (N, 16) rate lanes: 6 nibbles per 24-bit lane in lanes 0..11, 0 in 12..15 (what the leaf absorbs)."""
        from ..leaf.poseidon2 import _pack_elements_np
        return _pack_elements_np(self.nibbles_np(words), self)

    def bit_rows_torch(self, w):
        """(l, 68) int64 torch -> (288, l): the hint rows ``<nm>[i].b<j>`` word-major (4 per code, 8 per scale byte)."""
        import torch
        l = w.shape[0]
        s4 = torch.arange(4, dtype=torch.int64, device=w.device)
        s8 = torch.arange(8, dtype=torch.int64, device=w.device)
        codes = ((w[:, :K, None] >> s4) & 1).permute(1, 2, 0).reshape(4 * K, l)
        scales = ((w[:, K:, None] >> s8) & 1).permute(1, 2, 0).reshape(8 * SCALES, l)
        return torch.cat([codes, scales], 0)


class FP4HashedRelation:
    """The relation object ``relchain.HashedRelationRunner`` / ``hashchain.compose`` drive for ``fp4-nvf4+poseidon2``: the
    attributes of ``relations.Relation`` the hashed path reads, on the NVFP4 unit (``fp4/relation.py``), its hint
    generator (``fp4/witness.py``) and its synthetic instance families (``fp4/chain.py``)."""

    name = "fp4-nvf4"
    relation_name = "GemmAccumulator<1536> over sm120.mma.m16n8k64.e2m1.nvf4"      # fp4/chain.py RELATION_NAME
    model_name = MODEL.name
    witness_generation = ("fp4/hints_device.py device hint generator (one graph replay per sub-batch; witness.hints_fp4 numpy "
                          "on cpu) + hashchain.hash_hints (operand bit rows, Poseidon2 sponges of fp4/hashed.py FP4Format)")
    claim_note = "public claim per VU = the final FP32 accumulator word (three chain-end components: sign, exponent, fraction)"
    model = MODEL
    params = NVFP4
    k = WORDS                       # operand words per unit (the chain step)
    steps = STEPS
    y_bits = 32
    y_dtype = "<u4"
    word_dtype = "<u1"
    tag = b"|rel=fp4-nvf4|v1"       # fp4/chain.py FP4_HOOKS.tag
    fold = 1
    folded_from = None
    frozen_tier = None
    target = NVFP4_SM120
    rust_verifiable = True
    backend_name = ("b-ligero chain, fp4-nvf4 relation (archive/direct/ligero/fp4/relation.py) with the in-circuit operand "
                    "decode + Poseidon2 sponges of fp4/hashed.py (hashchain.compose)")
    # the frozen NVFP4 set IS fp4/chain.py instances_fp4 at seed 20260922 (contract.NVFP4_INSTANCES_DIGEST ==
    # chain.instances_digest(4096), which instances_digest_fn returns): the bare cell's dataset / tier, so the committed
    # column names the same set at 4096 VUs
    instances_dataset = contract.NVFP4_INSTANCES_DATASET
    instances_tier = contract.NVFP4_INSTANCES_TIER
    instances_recipe = ("fp4-nvf4 synthetic (fp4/chain.py instances_fp4: VU i draws its 24 steps' operands from edge family "
                        "VU_FAMILIES[i % 14] of seed 20260922, accumulators recorded by BLACKWELL_SM120_NVF4.step)")
    instance_seed = INSTANCE_SEED
    vu_words = STEPS * WORDS        # words per committed row (the tree bindings' K): 24 x 68, not K_VU = 24 x 64 codes
    reference_hints = None
    hash_format = FP4Format()
    #: words off the finite domain (``witness._check_domain``): the UE4M3 NaN byte and a byte with the padding bit set; the
    #: gate plants them at a scale-byte position (``off_domain_index``), E2M1 codes have no invalid encoding
    off_domain = (0x7F, 0xFF, 0x80)
    off_domain_index = K
    unchecked = False               # True: the prover's decode does NOT refuse an off-domain word (the negatives prove it)
    _bare_sys = None                # memo: the bare unit the device hint generator is keyed on (``hints``)

    @staticmethod
    def y_public(y: int) -> int:
        return int(y)

    def compile(self, p, chain: bool = True) -> System:
        return compile_fp4_unit(p, chain=chain)

    @staticmethod
    def _np(x, dtype=np.int64):
        import torch
        return x.cpu().numpy().astype(dtype) if isinstance(x, torch.Tensor) else np.asarray(x, dtype=dtype)

    def _domain_checked(self, fn, *args):
        """``fn(*args)`` with ``witness._check_domain`` disabled while ``unchecked`` (the negatives force the prover on)."""
        if not self.unchecked:
            return fn(*args)
        from . import witness as w
        chk, w._check_domain = w._check_domain, lambda *a: None
        try:
            return fn(*args)
        finally:
            w._check_domain = chk

    def _device_gen(self, p, l: int, device, slot=None):
        """The bare unit's device hint generator (``fp4/hints_device.py``, the bare cell's: one graph replay per sub-batch
        yields the 141 decode pins AND the 86 hint rows).  Built on the BARE unit (its pins in compile order); the composed
        system's base hints are that unit's hint rows in the same order (``hashchain.compose`` keeps them as the prefix).
        ``slot``: the pipeline slot index (sub-batches in flight on different streams must not share the static buffers)."""
        from .hints_device import hints_device_for
        if self._bare_sys is None:
            self._bare_sys = compile_fp4_unit(p, chain=True)
        return hints_device_for(self._bare_sys, l, device, slot=slot)

    def hints(self, p, sys: System, c_in, a, b, y_out, device, slot=None):
        """The base unit's 86 hint rows: the device generator on CUDA, ``witness.hints_fp4`` (numpy) elsewhere and for the
        forced negatives (``unchecked``: the device path has no domain check to switch off, the numpy path is the one the
        negatives reason about)."""
        import torch
        if str(device).startswith("cuda") and not self.unchecked:
            gen = self._device_gen(p, a.shape[0], device, slot)
            if [n for _, n in sys.hints] != [n for _, n in self._bare_sys.hints]:
                raise ValueError("fp4-nvf4+poseidon2: the composed system's base hints are not the bare unit's hint rows")
            _pub, H, _y = gen(c_in, a, b, check=True)
            return H
        H = self._domain_checked(hints_fp4, sys, self._np(c_in), self._np(a), self._np(b))
        return torch.as_tensor(H, dtype=torch.int64, device=device)

    def public_vectors(self, p, c, a, b, y, device="cpu", slot=None):
        """The 141 decode pins of the bare unit -- hint rows of the composed system (``hashchain.hash_hints`` asks for them
        per sub-batch): the device generator's ``pub_rows`` on CUDA (they do not depend on the accumulator: c = 0), the
        numpy ``public_vectors_fp4`` elsewhere and for the forced negatives."""
        import torch
        if str(device).startswith("cuda") and not self.unchecked:
            from .hints_device import PIN_NAMES
            a = torch.as_tensor(a, dtype=torch.int64, device=device)
            b = torch.as_tensor(b, dtype=torch.int64, device=device)
            gen = self._device_gen(p, a.shape[0], device, slot)
            pub, _H, _y = gen(torch.zeros(a.shape[0], dtype=torch.int64, device=device), a, b, check=True)
            return {name: pub[i] for i, name in enumerate(PIN_NAMES)}
        vals = self._domain_checked(public_vectors_fp4, self._np(a), self._np(b))
        return {k: torch.as_tensor(v, dtype=torch.int64, device=device) for k, v in vals.items()}

    # --- the gate's FP4 negative family (relchain.gate_hashed calls ``extra_negatives``) --------------------------------
    #: (name, word index in the a row of step 0, forged byte): malformed UE4M3 / E2M1 encodings and a wrong (valid) scale
    MALFORMED = (("UE4M3 NaN scale byte 0x7F", K, 0x7F),
                 ("UE4M3 scale byte with the padding bit set (0x80)", K, 0x80),
                 ("scale byte 0xFF (padding bit + NaN mantissa)", K + 1, 0xFF),
                 ("E2M1 code word 16 (a fifth bit)", 0, 16),
                 ("E2M1 code word 0xFF", 1, 0xFF))

    def extra_negatives(self, R, base: list) -> list[tuple[str, tuple[bool, str]]]:
        """Malformed encodings the prover is FORCED to compute on (``unchecked``: the hint generator's domain check off, the
        model's accumulators recomputed where the model accepts the bytes, the committed ones kept otherwise) -- the
        verifier alone must reject each; plus a wrong scale: a valid UE4M3 byte other than the committed one, the chain
        recomputed consistently on it (a forged computation on operands other than the committed row)."""
        a0, b0, accs0, _ = base[0]
        a0 = list(int(x) for x in a0)
        s_alt = 0x44 if a0[K + 2] != 0x44 else 0x45
        cases = list(self.MALFORMED) + [(f"wrong scale: committed byte 0x{a0[K + 2]:02x} proved as 0x{s_alt:02x} (valid UE4M3)", K + 2, s_alt)]
        negs = []
        self.unchecked = True
        try:
            for name, idx, val in cases:
                a = list(a0)
                a[idx] = val
                accs, c = [], 0
                try:
                    for s in range(STEPS):
                        c = MODEL.step(c, a[WORDS * s:WORDS * (s + 1)], b0[WORDS * s:WORDS * (s + 1)])
                        accs.append(int(c))
                except Exception:
                    accs = list(accs0)
                try:
                    pf, pb, ly = R.prove_vus([(a, b0, accs, accs[-1]), base[1]], check=False, vu_ids=[0, 1])
                    negs.append((name, R.verify_vus(pf, pb, ly)))
                except Exception as e:                      # the forced hint generator still cannot build the row: a rejection too
                    negs.append((name, (False, f"prover: {type(e).__name__}: {e}")))
        finally:
            self.unchecked = False
        return negs

    # --- instances: the synthetic edge families of fp4/chain.py, flattened to the runner's (a, b, accs, y) rows ---------
    @staticmethod
    def instances_fn(n: int, **kw) -> list[tuple]:
        from .chain import instances_fp4
        return [(A.reshape(-1).tolist(), B.reshape(-1).tolist(), list(accs), int(y)) for A, B, accs, y in instances_fp4(n)]

    @staticmethod
    def instances_digest_fn(n: int, **kw) -> str:
        from .chain import instances_digest
        return instances_digest(n)


FP4_HASHED = FP4HashedRelation()


def fp4_hashed_relation() -> FP4HashedRelation:
    return FP4_HASHED


def chain_sponge_np(lanes: np.ndarray, iv: list[int], n_vus: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """numpy twin of ``hashchain.sponge_states``: ``lanes`` (l, 16) rate elements per column (VU v at columns
    [v steps, (v + 1) steps), pad columns 0) -> (cap_in (8, l): the linked rows ``cap - IV`` entering every column,
    digests (n_vus, 8)).  The torch-free reference the laptop test evaluates the composed system with."""
    l = lanes.shape[0]
    cap_in = np.zeros((h2.CAPACITY, l), dtype=np.int64)
    st = np.zeros((h2.WIDTH, n_vus), dtype=np.int64)
    for x in range(h2.CAPACITY):
        st[h2.RATE + x] = iv[h2.RATE + x]
    cols0 = np.arange(n_vus) * steps
    for s in range(steps):
        cols = cols0 + s
        cap_in[:, cols] = (st[h2.RATE:] - np.asarray(iv[h2.RATE:], dtype=np.int64)[:, None]) % P
        st = np.concatenate([lanes[cols].T, st[h2.RATE:]], 0)
        st = h2.permute_np(st)
    return cap_in, st[:h2.OUT].T.copy()


def hashed_hints_np(hr, a: np.ndarray, b: np.ndarray, n_vus: int, unchecked: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """The full hint block (len(sys.hints), l) of the composed system for operand columns ``a``, ``b`` (l, 68) and
    accumulators ``c_in`` supplied by the caller through ``base``: returns (extra hints in ``hr.extra_hint_names`` order,
    digests (n_vus, 16), the public vectors ``hash.is_end`` / ``hash.d[x]``).  numpy; the pod path is ``hashchain.hash_hints``."""
    from .. import hashchain
    from . import witness as w
    l = a.shape[0]
    if unchecked:
        chk, w._check_domain = w._check_domain, lambda *args: None
        try:
            vals = public_vectors_fp4(a, b)
        finally:
            w._check_domain = chk
    else:
        vals = public_vectors_fp4(a, b)
    fmt = hr.fmt
    digests = []
    for _, nm, role in hashchain.ROLES:
        x = a if nm == "a" else b
        r = 0
        for i in range(WORDS):
            for j in range(fmt.bit_layout[i]):
                vals[f"{nm}[{i}].b{j}"] = (x[:, i] >> j) & 1
                r += 1
        cap, dg = chain_sponge_np(fmt.lanes_np(x), sponge_iv(role), n_vus, hr.steps)
        for xx in range(h2.CAPACITY):
            vals[f"{nm}.cap[{xx}]"] = cap[xx]
        digests.append(dg)
    digests = np.concatenate(digests, 1)
    extra = np.stack([vals[n] for n in hr.extra_hint_names])
    table = hashchain.public_table(digests, n_vus, hr.steps, l)
    pub = {hashchain.HASH_PIN_END: table[:, 0]}
    for x in range(hashchain.N_DIGEST):
        pub[hashchain.hash_pin(x)] = table[:, 1 + x]
    return extra, digests, pub


def numerator_table_check() -> bool:
    """The algebraic magnitude ``2 e0 + 4 e1 + 2 e0 e1 + m (1 + e1 + 2 e0 e1)`` equals ``NUMERATOR[code & 7]`` on every code."""
    for code in range(8):
        m, e0, e1 = code & 1, (code >> 1) & 1, (code >> 2) & 1
        p = e0 * e1
        if 2 * e0 + 4 * e1 + 2 * p + m * (1 + e1 + 2 * p) != NUMERATOR[code]:
            return False
    return True


def system_id_np(sys: System) -> bytes:
    """``protocol.system_id`` transcribed torch-free (the laptop pins the Rust ``hashed_sys_id`` without importing the
    prover): rows (kind, name, interval), linear and quadratic tables, pins, hints, chain rows and component ends."""
    h = hashlib.blake2b(digest_size=32)
    h.update(f"m={sys.m};chain={sys.chain is not None}".encode())
    for r in sys.rows:
        h.update(f"{r.idx}:{r.kind}:{r.name}:{r.lo}:{r.hi};".encode())
    for c in sys.linear:
        h.update(f"L:{c.name}:{sorted(c.lin.items())}:{c.c}:{c.pub};".encode())
    for q in sys.quadratic:
        h.update(f"Q:{q.name}:{q.a.key()}:{q.b.key()}:{q.c.key()};".encode())
    h.update(repr(sys.pins).encode())
    h.update(repr(sys.hints).encode())
    if sys.chain is not None:
        h.update(repr(sys.chain["c"]).encode())
        if sys.chain.get("y16") is not None:
            h.update(repr([e.key() for e in sys.chain["y"]] + [sys.chain["y16"].key()]).encode())
        else:
            h.update(repr([e.key() for e in sys.chain["y"]]).encode())
            h.update(repr([("end", Y.key(), int(lo), int(bits)) for Y, lo, bits in sys.chain["y_end"]]).encode())
    return h.digest()
