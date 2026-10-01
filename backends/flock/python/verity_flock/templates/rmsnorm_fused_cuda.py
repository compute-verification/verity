"""C-Flock lowering of ``rmsnorm-fused-cuda`` (RMSNormFusedCuda_v2): the IR graph cut at its reduction boundary
(``verity_flock.ir_lower.reduction_cut``) into identical units, one per CUDA warp of the kernel's cub BlockReduce: the warp's
residual adds, squares and shfl-down tree (its aggregate is a cut word) and the epilogue of its elements, which reads the row
scalar ``s`` (a cut word). The tail, the sequential sum of the warp aggregates, ``/ N``, ``+ eps`` and ``RsqrtApprox``,
reads no input word.

A warp's input leaves (:func:`width`): 32 threads, each reading x and res at every position it owns (threads hold elements
t, t + T, ... with T = min(N, 1024) threads); when T does not divide N the warps differ and there is no single unit.

eps is the program's, so it is a value (:func:`values`): the instance's ``values`` row holds it, every warp reads it and
returns it to the tail, and the unit netlist is pinned per N by :data:`PINS`, its leaf maps and cut structure included. The
frame-v3 statement's verifier evaluates the tail itself on public cut words, so its lowering (:func:`frame_lowering`) keeps
eps a tail constant, pinned per (N, EPS) by :data:`FRAME_PINS`."""
import struct

TEMPLATE = "rmsnorm-fused-cuda"
UNIT = "rmsnorm-fused-cuda/bf16/warp"
PINS = {2048: "9cd666d23a8e3279d57f09fd9e7c41b958ef00f7f5c5e4046237a291234ac7e9",
        4096: "edfd1271030935da4904402908f6dba94ec2a09ffabfb9b7a837256aed702b8b"}
FRAME_PINS = {(2048, 1e-05): "b18b3a5995ac32acab7377c653554b648dc5b9452ca4526bbfacc9d1eeaedced",
              (4096, 1e-05): "2d738630c45d35554222541059ee0db4841973444b82d664a6bdb4a5168ca97b"}


def width(n: int) -> int | None:
    threads = min(n, 1024)
    return None if n % threads or threads % 32 else 2 * 32 * (n // threads)


def definition(subcircuit):
    return subcircuit.definition()


def values(subcircuit) -> list[tuple[int, int]]:
    """The program's constants the units take as values, ``(width, value)``: eps, as the f32 the Definition holds."""
    return [(32, struct.unpack("<I", struct.pack("<f", float(subcircuit.params["EPS"])))[0])]


def lowering(subcircuit):
    from verity_flock import ir_lower

    n = int(subcircuit.params["N"])
    return ir_lower.lowering(definition(subcircuit), f"{UNIT}/n{n}", ir_lower.reduction_cut(width(n)), values=values(subcircuit))


def circuit_lowering(subcircuit):
    """The lowering ``verity/flock-circuit`` proves: :func:`lowering`, eps a value."""
    return lowering(subcircuit)


FRAME_SCHEMES = ("frame-v3/blake3-keyed",)


def frame_lowering(subcircuit):
    from verity_flock import ir_lower

    n, eps = int(subcircuit.params["N"]), float(subcircuit.params["EPS"])
    return ir_lower.lowering(definition(subcircuit), f"{UNIT}/n{n}/eps{eps:g}", ir_lower.reduction_cut(width(n)))


def supports(subcircuit, scheme):
    n = int(subcircuit.params["N"])
    if width(n) is None:
        return f"C-Flock lowers RMSNormFusedCuda_v2 when its warps are identical units (min(N, 1024) threads dividing N), not N = {n}"
    if scheme in FRAME_SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under {', '.join(FRAME_SCHEMES)} (verity/flock-ir-frame/v3: unit {UNIT}, the row tail "
            f"evaluated by the verifier); binding its rows under {scheme} is not wired yet")


def relation(subcircuit, scheme):
    return "irf:rmsnorm-fused-cuda"


def stage(input_set, lo, hi, out_dir):
    """The frame-v3 instance file for ids [lo, hi) (``verity_flock.ir_frame.stage``): each pod stages its own."""
    from verity_flock import ir_frame

    return ir_frame.stage(frame_lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
