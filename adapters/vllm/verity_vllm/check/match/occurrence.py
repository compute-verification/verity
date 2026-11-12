"""Occurrence resolution (the code says 'attribution'): which request a fold instance belongs to by the fold's row-count rule
(`attribute`), which instances belong to no request because they read only shared weights (`is_weight_only`, owner `SHARED`), and
each request's instances with the shared ones of its engine steps merged in (`shared_legs`)."""

from __future__ import annotations

SHARED = "*weight-only*"   # owner sentinel of a weight-only instance; never a request id


def is_weight_only(r: dict, weights_nodes: frozenset, shared_nodes: set, compact: bool) -> bool:
    """True when every node operand of fold instance `r` is a shared weight or an earlier weight-only instance (node id
    in `shared_nodes`) and there is at least one: the instance carries no request's data (gemma2's RMSNorm scale `1 + w`).  The rule
    `decompose` and GM-01 (global_match.check, G2) apply after the meta channel and before the row-count rule -- one definition."""
    if compact:
        from verity_vllm.program.compact import idx_values as _idx_values
    reads = False
    for a in r["args"]:
        for run in a:   # record (space, idx, lo, hi) | compact (space, (idx0, lo, hi), (dI, dL, dH), count): space / idx only
            if run[0] != "n":
                continue
            for idx in (_idx_values(run) if compact else (run[1],)):
                if idx in weights_nodes or idx in shared_nodes:
                    reads = True
                else:
                    return False
    return reads


def shared_legs(per_req: dict, shared_idx: list, steps: dict) -> dict:
    """Each request's instance list with the weight-only instances of every engine step the request has rows in
    (StepStart / logits rows) merged in, in node order: the request's own derive computes that node inside its forward at that step.
    A request with no rows in the step does not see it.  Request order is preserved."""
    present: dict[int, set[str]] = {s: {r for r, _ in st.get("rows", [])} | {r for r, _ in st.get("logits_rows", [])} for s, st in steps.items()}
    return {rid: sorted(idxs + [i for i, s in shared_idx if rid in present.get(s, ())]) for rid, idxs in per_req.items()}


def attribute(meta: dict, st: dict) -> tuple[str | None, str]:
    """(request_id | None, how) for one fold instance meta {step,row,rows} given its engine step's StepStart."""
    row, rows = meta.get("row"), meta.get("rows")
    if row is None or rows is None:
        return None, "no-row"
    n_tok, n_log = len(st["rows"]), len(st["logits_rows"])
    if rows == n_tok and rows == n_log:
        if row >= n_tok:
            return None, "row-out-of-range"
        a, b = st["rows"][row][0], st["logits_rows"][row][0]
        return (a, "token=logits") if a == b else (None, f"ambiguous:{a}!={b}")
    if rows == n_tok:
        return (st["rows"][row][0], "token") if row < n_tok else (None, "row-out-of-range")
    if rows == n_log:
        return (st["logits_rows"][row][0], "logits") if row < n_log else (None, "row-out-of-range")
    # fused-MoE expert instances (MoeExpertGemm / SiluMul / MoeExpertGemmW) run over the TOPK*tokens pair rows, pair-major
    # p = t*TOPK + k (the Program's addressing, independent of the sorted block schedule): row p belongs to token p // TOPK.  The MoE
    # X-09 rule; without it every MoE-block instance is "attributed to no request" (OLMoE b1: 110 208 of 165 950).
    if n_tok and rows > n_tok and rows % n_tok == 0:
        k = rows // n_tok
        return (st["rows"][row // k][0], f"moe-pair(x{k})") if row < rows else (None, "row-out-of-range")
    return None, f"rows={rows}!=tokens({n_tok})|logits({n_log})"
