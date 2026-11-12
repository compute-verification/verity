"""circuit-check's bindings of `verity.definitions.registry.tp_allreduce`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: GLM-5.3's hidden size, whose TP8 all-reduces `_glm53_served_roots` binds
GLM53_N = 6144
#: the rows (SIZE = rows * GLM53_N) `_glm53_served_roots` and `_glm53_multimem_alone_roots` bind GLM-5.3's multimem
#: all-reduce at: one row; the fewest a launch past MNNVL's 42 tokens served; the input budget's edge (85 rows read whole, 86
#: lazily); and the most served (256)
GLM53_MULTIMEM_ROWS = (1, 41, 85, 86, 256)
#: the multimem rows `_glm53_multimem_alone_roots` binds, each its own binding's one target, so its task runs apart from the
#: other served all-reduces': the most served, whose correspondence reads all 8 ranks' 1,572,864 elements lazily on each vector
GLM53_MULTIMEM_ALONE = (256,)


def _glm53_served(mnnvl: bool, multimem_rows: tuple[int, ...]) -> list[Any]:
    from verity.circuits.defs import bind
    from verity.definitions.registry import tp_allreduce as TA

    out: dict[str, Any] = {}
    for c in TA.CAPTURED:
        if mnnvl and (c.N, c.world, c.device, c.route) == (GLM53_N, 8, None, "fi_ar_comm"):
            calls = [(M, rows) for M in range(c.lo, c.hi + 1) for rows in range(1, M + 1)]
        elif ((c.N, c.world, c.route) == (GLM53_N, 8, "symm_mem_comm")
              and (c.device, c.node) == (TA.UNSTATED_NODE[0], TA.UNSTATED_NODE[1:])):
            calls = [(max(rows, c.lo), rows) for rows in multimem_rows]
        else:
            continue
        for M, rows in calls:
            d, statics = TA.route_binding(c, 8, M, GLM53_N, rows)
            fn = bind(d, **statics)
            out[fn.id] = fn
    return [out[k] for k in sorted(out)]


def _glm53_served_roots() -> list[Any]:
    """GLM-5.3 TP8's all-reduces of its hidden size on H200 at served statics, as `tp_allreduce.route_binding` binds a
    request's rows among a captured launch's M tokens (`CAPTURED`): every MNNVL call (at most 42 rows, one-shot in a launch
    of at most 10 tokens, `mnnvl_shot`), and multimem at `GLM53_MULTIMEM_ROWS` on the node a registration stating no tree
    is served on (`tp_allreduce.UNSTATED_NODE`), but for `GLM53_MULTIMEM_ALONE`'s rows (`_glm53_multimem_alone_roots`).
    All but one row of multimem are over `targets.MAX_GATES`, so each holds a ``digests`` pin, which keeps it in ``--all``;
    `test_tp_allreduce_served` holds each to one."""
    return _glm53_served(True, tuple(r for r in GLM53_MULTIMEM_ROWS if r not in GLM53_MULTIMEM_ALONE))


def _glm53_multimem_alone_roots() -> list[Any]:
    """GLM-5.3 TP8's multimem all-reduce at `GLM53_MULTIMEM_ALONE`'s served rows, as `_glm53_served_roots` binds the others."""
    return _glm53_served(False, GLM53_MULTIMEM_ALONE)


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The word Definitions at WORLD 8: the MNNVL all-reduce of 8 elements two-shot, its fused norm at N = 512 over a 2-CTA
    cluster (one warp per CTA) one-shot with the + EPS unfused, and its two parts (its all-reduce and residual add of 8 elements
    two-shot, its norm at that N, cluster and shot), vLLM's custom all-reduce of 64 elements 2-stage (every
    rank owns one pack), torch's multimem all-reduce of 128 elements (16 per owner) at a chain for owners 0-3 and a balanced
    tree for 4-7, and one 16-byte vector of it at GLM-5.3's tree; and NCCL's tree all-reduce of 8 elements at WORLD 4 down the
    TP4 capture's chain."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import tp_allreduce as TA

    trees = (TA.chain(8),) * 4 + ((((0, 1), (2, 3)), ((4, 5), (6, 7))),) * 4
    return [
        ("AllReduceMnnvl_v1", lambda: bind(TA.AllReduceMnnvl, WORLD=8, N=8, SHOT=2)),
        ("AllReduceRmsNormMnnvl_v1",
         lambda: bind(TA.AllReduceRmsNormMnnvl, WORLD=8, N=512, SHOT=1, CLUSTER=2, EPS=1e-5, FMA=False)),
        ("AllReduceResidualMnnvl_v1", lambda: bind(TA.AllReduceResidualMnnvl, WORLD=8, N=8, SHOT=2)),
        ("RmsNormMnnvl_v1", lambda: bind(TA.RmsNormMnnvl, N=512, SHOT=1, CLUSTER=2, EPS=1e-5, FMA=False)),
        ("AllReduceCustom_v1", lambda: bind(TA.AllReduceCustom, WORLD=8, SIZE=64, STAGE=2)),
        ("AllReduceMultimem_v1", lambda: bind(TA.AllReduceMultimem, WORLD=8, SIZE=128, TREES=trees, FTZ=False)),
        ("AllReduceMultimemVector_v1", lambda: bind(TA.AllReduceMultimemVector, WORLD=8, TREE=TA.GLM53_H200_TREE, FTZ=False)),
        ("AllReduceNcclTree_v1", lambda: bind(TA.AllReduceNcclTree, WORLD=4, SIZE=8, CHAIN=(0, 1, 2, 3))),
    ]
