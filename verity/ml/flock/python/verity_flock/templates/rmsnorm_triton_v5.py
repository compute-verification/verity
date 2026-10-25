"""C-Flock lowering of ``rmsnorm-triton-v5`` (RMSNormTriton_v5, ``registry.dead_free``: the registered compiler's
``RMSNormTriton_v1`` without its dead adds): ``rmsnorm-triton``'s lowering, byte for byte. v5 keeps lane 0's 31 adds of each warp's
xor butterfly where v1 states all 160, but a lowering reads only the live gates (``ir_lower.live_gates``), so v1's lowering never
held the other 129: the two graphs give the same warp units (a warp's 8-column square chains, lane 0's adds, whose sum is a cut
word, and the epilogue of its 256 columns, which reads the row scalar ``inv``), the same cut words and the same tail (the CTA
combine ``(w0 + w2) + (w1 + w3)``, the chunks summed from +0, ``fma(rcp(N), sum, eps)``, ``MufuSqrtFtz`` and ``div.full``). The
unit netlist is ``rmsnorm-triton``'s, under its name and its :data:`PINS`; what binds the statement to v5 is the Program it
names (``verity_flock.circuit.program_digests``).

v5 states the kernel's 1024-column chunks at every N, so its warps are identical units only when every lane of every chunk holds
columns (:func:`width`: N a multiple of 1024; Llama's N = 2048 and 4096 are 8 and 16 units). These templates are proved under
``verity/flock-circuit`` only (:data:`SCHEMES`), its row tail in the statement (``verity_flock.tail``)."""
from verity_flock.templates import rmsnorm_triton as _v1

TEMPLATE = "rmsnorm-triton-v5"
FAMILY = "RMSNormTriton_v5"
UNIT = _v1.UNIT
PINS = {k: v for k, v in _v1.PINS.items() if k[0] % 1024 == 0}
#: ``verity_flock.circuit.SCHEME``, the commitment scheme of ``verity/flock-circuit``
SCHEMES = ("frame-v3-sha512/hm96-sha512",)


def width(n: int) -> int | None:
    return 256 if n > 0 and n % 1024 == 0 else None


def definition(subcircuit):
    return subcircuit.definition()


def lowering(subcircuit):
    from verity_flock import ir_lower

    n, eps = int(subcircuit.params["N"]), float(subcircuit.params["EPS"])
    return ir_lower.lowering(definition(subcircuit), f"{UNIT}/n{n}/eps{eps:g}", ir_lower.reduction_cut(width(n)))


def circuit_lowering(subcircuit):
    return lowering(subcircuit)


def supports(subcircuit, scheme):
    n = int(subcircuit.params["N"])
    if width(n) is None:
        return f"C-Flock lowers {FAMILY} when its warps are identical units (N a multiple of 1024), not N = {n}"
    if scheme in SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under verity/flock-circuit ({', '.join(SCHEMES)}: unit {UNIT}, the row tail in the "
            f"statement), not under {scheme}")


def relation(subcircuit, scheme):
    return "verity/flock-circuit"
