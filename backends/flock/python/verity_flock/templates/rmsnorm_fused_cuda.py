"""C-Flock lowering of ``rmsnorm-fused-cuda`` (RMSNormFusedCuda_v2): the IR graph cut at its reduction boundary
(``verity_flock.ir_lower.reduction_cut``) into identical units, one per CUDA warp of the kernel's cub BlockReduce: the warp's
residual adds, squares and shfl-down tree (its aggregate is a cut word) and the epilogue of its elements, which reads the row
scalar ``s`` (a cut word). The tail, the sequential sum of the warp aggregates, ``/ N``, ``+ eps`` and ``RsqrtApprox``,
reads no input word: the verifier evaluates it with the IR evaluator and supplies the cut words as public unit words.

A warp's input leaves (:func:`width`): 32 threads, each reading x and res at every position it owns (threads hold elements
t, t + T, ... with T = min(N, 1024) threads); when T does not divide N the warps differ and there is no single unit. The
unit netlist is pinned per (N, EPS) by :data:`PINS`, its cut structure (the tail with N and eps, the cut ports) included."""
TEMPLATE = "rmsnorm-fused-cuda"
UNIT = "rmsnorm-fused-cuda/bf16/warp"
PINS = {(2048, 1e-05): "47d753967af84bf44a05fbd6d8fdae2d8ff9c4fba6117043b9fd6f4c1a7f4111"}


def width(n: int) -> int | None:
    threads = min(n, 1024)
    return None if n % threads or threads % 32 else 2 * 32 * (n // threads)


def definition(subcircuit):
    return subcircuit.definition()


def lowering(subcircuit):
    from verity_flock import ir_lower

    n, eps = int(subcircuit.params["N"]), float(subcircuit.params["EPS"])
    return ir_lower.lowering(definition(subcircuit), f"{UNIT}/n{n}/eps{eps:g}", ir_lower.reduction_cut(width(n)))


def supports(subcircuit, scheme):
    n = int(subcircuit.params["N"])
    if width(n) is None:
        return f"C-Flock lowers RMSNormFusedCuda_v2 when its warps are identical units (min(N, 1024) threads dividing N), not N = {n}"
    return (f"C-Flock lowers {subcircuit.id} exactly (verity_flock.templates.rmsnorm_fused_cuda: unit {UNIT}, the row tail "
            f"evaluated by the verifier) and proves it with public unit words (verity/flock-ir-block/v1); binding its rows under "
            f"{scheme} is not wired yet")


def stage(input_set, lo, hi, out_dir):
    from verity_flock import ir_lower

    return ir_lower.stage(lowering(input_set.subcircuit), input_set, lo, hi, out_dir)
