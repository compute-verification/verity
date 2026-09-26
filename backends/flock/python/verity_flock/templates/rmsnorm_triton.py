"""C-Flock lowering of ``rmsnorm-triton`` (RMSNormTriton_v1): the IR graph cut at its reduction boundary
(``verity_flock.ir_lower.reduction_cut``) into identical units, one per warp of the Triton kernel's 128-thread CTA: its
threads' 8-column square chains, the xor-butterfly to lane 0 (the warp sum is a cut word) and the epilogue of its 256
columns, which reads the row scalar ``inv`` (a cut word). The tail, the CTA combine ``(w0 + w2) + (w1 + w3)``, the chunks
summed from +0, ``fma(rcp(N), sum, eps)``, ``MufuSqrtFtz`` and ``div.full``, reads no input word: the verifier evaluates it
with the IR evaluator and supplies the cut words as public unit words.

A warp's input leaves (:func:`width`): 32 threads x 8 columns of x, when every lane of every 1024-column chunk holds
columns (N a multiple of 1024); otherwise the warps differ. The unit netlist is pinned per (N, EPS) by :data:`PINS`, its cut
structure (the tail with N and eps, the cut ports) included."""
TEMPLATE = "rmsnorm-triton"
UNIT = "rmsnorm-triton/bf16/warp"
PINS = {(2048, 1e-05): "950d95a130531afb00951a53d64d45a4549933669029b155b237e21fc82014e6"}


def width(n: int) -> int | None:
    return 256 if n % 1024 == 0 else None


def definition(subcircuit):
    return subcircuit.definition()


def lowering(subcircuit):
    from verity_flock import ir_lower

    n, eps = int(subcircuit.params["N"]), float(subcircuit.params["EPS"])
    return ir_lower.lowering(definition(subcircuit), f"{UNIT}/n{n}/eps{eps:g}", ir_lower.reduction_cut(width(n)))


def supports(subcircuit, scheme):
    n = int(subcircuit.params["N"])
    if width(n) is None:
        return f"C-Flock lowers RMSNormTriton_v1 when its warps are identical units (N a multiple of 1024), not N = {n}"
    return (f"C-Flock lowers {subcircuit.id} exactly (verity_flock.templates.rmsnorm_triton: unit {UNIT}, the row tail "
            f"evaluated by the verifier) and proves it with public unit words (verity/flock-ir-block/v1); binding its rows under "
            f"{scheme} is not wired yet")


def stage(input_set, lo, hi, out_dir):
    from verity_flock import ir_lower

    return ir_lower.stage(lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
