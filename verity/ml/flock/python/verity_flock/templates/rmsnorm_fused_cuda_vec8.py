"""Flock lowering of ``rmsnorm-fused-cuda-vec8`` (RMSNormFusedCudaVector8_v1, vLLM's served width-8
``fused_add_rms_norm_kernel``; GLM-5.3's hidden 6144): the IR graph cut at its reduction boundary
(``verity_flock.ir_lower.reduction_cut``) into identical units of whole CUDA warps of the kernel's cub BlockReduce. A warp is
its threads' residual adds, square partials and shfl-down tree (its aggregate is a cut word) and the epilogue of its elements,
which reads the row scalar ``s`` (a cut word). The tail, the sequential sum of the warp aggregates, ``/ N``, ``+ eps`` and
``RsqrtApprox``, reads no input word; ``verity/flock-circuit`` proves it as stage circuits and a MUFU lookup wired to the units'
cut ports (``verity_flock.tail``).

What differs from ``rmsnorm-fused-cuda`` (RMSNormFusedCuda_v2, the generic ``<bf16, 0>`` kernel): a thread holds 8-element
vectors t, t + BLOCK, ... rather than elements t, t + T, ...; a vector's partial is four adjacent square-pair sums added left
to right, and a thread adds its vector partials with F32Add, not an FMA chain; BLOCK is a static (the launch takes N or 256
threads by the token count), and threads beyond N / 8 hold nothing. :func:`warps` says when every warp is the same unit:
``min(BLOCK, N / 8)`` threads in whole warps, each holding the same number of vectors. A unit is the warp count's odd part of
consecutive warps, so an instance holds a power-of-two number of units (GLM-5.3: 24 warps of one vector at BLOCK 1024, three
a unit; 8 warps of three vectors at BLOCK 256; 8 units of 768 elements either way). At N = 6144 the tail's ``/ N`` divides by a
constant that is not a power of two, which the tail lowers through the IEEE divider (``verity_flock.tail``) where 2048 and 4096
took an exact reciprocal. The unit netlist is pinned per
(N, EPS, BLOCK) by :data:`PINS`, its leaf maps and cut structure (the tail with N and eps, the cut ports) included.

A unit of 768 elements is about 2^21.8 rows, more than frame-v3's 2^22-bit block holds beside its compressions, so these
templates are proved under ``verity/flock-circuit`` only (:data:`SCHEMES`)."""
TEMPLATE = "rmsnorm-fused-cuda-vec8"
FAMILY = "RMSNormFusedCudaVector8_v1"
UNIT = "rmsnorm-fused-cuda-vec8/bf16/warps"
FUSED = True
VECTOR = 8
PINS: dict[tuple[int, float, int], str] = {
    (6144, 1e-05, 1024): "08346ff85c92817c6f29b4cd358cef09930a9aeaae0112eb5da81316bc4c0eee",
    (6144, 1e-05, 256): "fb72ffc60d5acaa81904d0a2574b8939fad634e76183d70cad3b9c8b31b049af",
}
#: ``verity_flock.circuit.SCHEME``, the commitment scheme of ``verity/flock-circuit``
SCHEMES = ("frame-v3-sha512/hm96-sha512",)


def warps(n: int, block: int) -> tuple[int, int] | None:
    """(warps, vectors per thread) of the launch when its warps are identical units, else None."""
    vectors = n // VECTOR
    threads = min(block, vectors)
    if n % VECTOR or threads < 1 or vectors % threads or threads % 32:
        return None
    return threads // 32, vectors // threads


def cut(n: int, block: int, fused: bool = FUSED) -> tuple[int, int] | None:
    """(the reduction cut's width, warps a unit): a warp's input leaves (x and res at each element for the fused kernel, x for
    the plain one) and the odd part of the warp count."""
    w = warps(n, block)
    if w is None:
        return None
    count, per_thread = w
    return (2 if fused else 1) * 32 * VECTOR * per_thread, count // (count & -count)


def definition(subcircuit):
    return subcircuit.definition()


def _statics(subcircuit) -> tuple[int, float, int]:
    p = subcircuit.params
    return int(p["N"]), float(p["EPS"]), int(p["BLOCK"])


def lowering(subcircuit, fused: bool = FUSED, unit: str = UNIT):
    from verity_flock import ir_lower

    n, eps, block = _statics(subcircuit)
    width, group = cut(n, block, fused)
    name = f"{unit}/n{n}/b{block}/eps{eps:g}"
    return ir_lower.lowering(definition(subcircuit), name, ir_lower.reduction_cut(width), group=group)


def circuit_lowering(subcircuit):
    return lowering(subcircuit)


def supports(subcircuit, scheme, family: str = FAMILY, unit: str = UNIT):
    n, _eps, block = _statics(subcircuit)
    if warps(n, block) is None:
        return (f"C-Flock lowers {family} when its warps are identical units (min(BLOCK, N / 8) threads in whole warps, each "
                f"holding N / 8 / threads vectors), not N = {n}, BLOCK = {block}")
    if scheme in SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under verity/flock-circuit ({', '.join(SCHEMES)}: unit {unit}, the row tail in the "
            f"statement), not under {scheme}")


def relation(subcircuit, scheme):
    return "verity/flock-circuit"
