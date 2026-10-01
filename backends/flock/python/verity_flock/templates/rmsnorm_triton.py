"""C-Flock lowering of ``rmsnorm-triton`` (RMSNormTriton_v1): the IR graph cut at its reduction boundary
(``verity_flock.ir_lower.reduction_cut``) into identical units, one per warp of the Triton kernel's 128-thread CTA: its
threads' 8-column square chains, the xor-butterfly to lane 0 (the warp sum is a cut word) and the epilogue of its 256
columns, which reads the row scalar ``inv`` (a cut word). The tail, the CTA combine ``(w0 + w2) + (w1 + w3)``, the chunks
summed from +0, ``fma(rcp(N), sum, eps)``, ``MufuSqrtFtz`` and ``div.full``, reads no input word.

A warp's input leaves (:func:`width`): 32 threads x 8 columns of x, when every lane of every 1024-column chunk holds
columns (N a multiple of 1024), or 32 threads x N/128 columns when N is a power of two under 1024 (one N-column block over
the 128 threads); otherwise the warps differ.

eps is the program's, so it is a value (:func:`values`): the instance's ``values`` row holds it, every warp reads it and
returns it to the tail, and the unit netlist is pinned per N by :data:`PINS`, its leaf maps and cut structure included. The
frame-v3 statement's verifier evaluates the tail itself on public cut words, so its lowering (:func:`frame_lowering`) keeps
eps a tail constant, pinned per (N, EPS) by :data:`FRAME_PINS`."""
import struct

TEMPLATE = "rmsnorm-triton"
UNIT = "rmsnorm-triton/bf16/warp"
PINS = {2048: "17618e2cd612924291f52b3e9ecb6ea4f39d55c193ba8e452388bf8112d55cb0",
        4096: "8d3eb58195e3203ea6355cb5264a3412bfcd4081d44434f6624dcb8ea7f8a79b",
        128: "c92c2d4ba3aa6eb4efaaa4fa3808f093860c7385b369c5a24631a0760f4638a4"}
FRAME_PINS = {(2048, 1e-05): "7400301dd799baf5b46ebf70a25f6e2e7ef13b5f4a2ae821f7acdd3d162f9308",
              (4096, 1e-05): "e7d3976fe7e959fe55a85818a9ad3796bd5ebd1ae1d5ba9660f491bd92034d2a",
              (128, 1e-06): "ba1bd0a72fe51f0e1ce8d798aa4e18df55f9e643b1c503873787b14bdc8352b8"}


def width(n: int) -> int | None:
    if n % 1024 == 0:
        return 256
    return n // 4 if 128 <= n < 1024 and n & (n - 1) == 0 else None


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
        return (f"C-Flock lowers RMSNormTriton_v1 when its warps are identical units (N a multiple of 1024, or a power of two from 128 "
                f"to 512), not N = {n}")
    if scheme in FRAME_SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under {', '.join(FRAME_SCHEMES)} (verity/flock-ir-frame/v3: unit {UNIT}, the row tail "
            f"evaluated by the verifier); binding its rows under {scheme} is not wired yet")


def relation(subcircuit, scheme):
    return "irf:rmsnorm-triton"


def stage(input_set, lo, hi, out_dir):
    """The frame-v3 instance file for ids [lo, hi) (``verity_flock.ir_frame.stage``): each pod stages its own."""
    from verity_flock import ir_frame

    return ir_frame.stage(frame_lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
