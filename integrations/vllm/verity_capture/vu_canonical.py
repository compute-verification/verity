"""The canonical VU query for the supported vLLM Programs (design doc `docs/vu-query-design-2026-09-17.md` §6), expressed
with the frozen query API (`verity.ir.query_ast`, `docs/query-interfaces-freeze-2026-09-17.md`).

    canonical_vu = override(default(32), {
        "//Gemm_v1{*}":          members("//Gemm_v1{*}", "@batch:GemmCoordinate_v1"),        # one VU per output word, Out = {y[n]} 16 b
        "//AttentionHead_v3{*}": each("//AttentionHead_v3{*}", concat(
                                     each("@call:AttnBlock_v3", concat(
                                         zip(".", ["@batch:AmpereBF16TcDot16_v1"]),        # PV chain per d over the k-step batches, 32 b
                                         instances("@call:DotBf16_v1"),                    # s[c] = q . k[c] as one atomic call, 32 b
                                         rest())),                                         # max chain, ex2 / fma, lane sums, cvt
                                     rest())),                                             # epilogue adds, Fa2InvSum, mul / cvt members
        "//RMSNormTriton_v1{*}": each(..., concat(zip(".", ["@batch:F32Mul_v1", "@batch:F2fpBf16_v1"]), rest())),  # normalize path per output, 16 b
        "//RMSNormFusedCuda_v2{*}": the same shape,
        "//SiluMul_v1{*}":       members(..., "@batch:SiluMulBf16_v1"),                                            # 16 b
        "//MoeExpertGemm_v1{*}": members(..., "@batch:MoeExpertCoordinate_v1"),                                    # gather row + dot inside, 16 b
        "//MoeExpertGemmW_v1{*}": members(..., "@batch:MoeExpertCoordinateW_v1"),
        "//MoeRouterTopK_v1{*}": each(..., concat(instances("@call:NvExpf_v1"), each("@scan:RouterArgmaxStep_v1", default(32)), rest())),
    })

Every family not named falls to `default(32)` (a node is one set per member iff its declared distinct return width is
<= 32 bits, else it is descended) -- a valid partition with w_out <= 32 for every registered Definition, the 64-bit
`Const64` gates of `GumbelArgmaxBf16_v1` excepted (design doc §7: they are a Definition change, not a query).  The v1/v2
factored variants of the GEMM and attention chains (`Gemm_v2{K,N,DOT}`, `AttentionHead_v1/v2`) get the same shapes with
their own callee ids, so one query text serves `Serve_v1`, `Serve_v2` and `Serve_v4` programs of a profile.

Profiles select the tensor-core step primitive the PV zip names: `ampere` (sm_80/sm_89 eager, `AmpereBF16TcDot16_v1`)
and `hopper` (sm_90, `HopperBF16WgmmaDot16_v1`); the query text is otherwise identical.  `canonical_query_document`
is the `verity-ir/query/v1` artifact (no `kind` / `w_max`: the use configuration is `verity_vllm.ir.query_artifact`'s).

Why attention is not a population keyed by T: the head body is refined into block calls (`AttnBlock_v3{D,NVIS,FIRST,BN}`
-- the statics do not include T) and epilogue singletons; every VU kind is a callee id (`DotBf16_v1{K=D}`, the k-step
batch members, the scalar gates), so a T=72 head produces more VUs of the SAME kinds as a T=40 head, never new kinds.
`tests/ir/test_vu_canonical.py` asserts exactly that.

This module imports `verity.ir.query_ast` lazily: `canonical_query_text` works without it; the AST constructors need it.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

SCHEMA = "verity-ir/query/v1"
AUTHOR = "verity_capture.vu_canonical"
VERSION = "2026-09-17.1"
#: [COORD R17-10 (Daniel, 17:41Z)] this module's query is ONE query among others -- the FINE relation-level partition (Gemm coordinates,
#: AttnBlock chains, norm interiors) -- labelled `Q_fine_v1`; nothing in the infrastructure is "canonical".  The sweep's query OF RECORD is
#: `Q_module_v1` (`verity_capture.commit.required_manifest.Q_MODULE_ID`: one unit per Program row / module boundary), which uses this
#: query's family selection (`query_digest`) and not its unit.  The module keeps its historical name for its importers (16 files); the
#: label is what records and renders carry.
QUERY_ID = "Q_fine_v1"

#: the tensor-core k16 step primitive per profile family (the attention PV zip names it)
PROFILES: dict[str, dict[str, Any]] = {
    "ampere": {"dot": "AmpereBF16TcDot16_v1", "inv": "Fa2InvSum_v1",
               "doc": "vLLM eager bf16 on sm_80/sm_89 (FA2 split-KV, Triton batch-invariant GEMM): Serve_v1 / Serve_v2{GEMM_AMPERE} / Serve_v4"},
    "hopper": {"dot": "HopperBF16WgmmaDot16_v1", "inv": "Fa3InvSum_v1",
               "doc": "vLLM eager bf16 on sm_90 (FA3, wgmma): Serve_v2{hopper.GEMM_HOPPER, ATTENTION_FA3}"},
}
#: aliases -> profile key (the semantic-profile ids the capture pipeline uses start with `vllm-<rev>-sm<cc>-...`)
_ALIASES = {"default": "ampere", "vllm": "ampere", "sm80": "ampere", "sm86": "ampere", "sm89": "ampere", "fa2": "ampere",
            "sm90": "hopper", "fa3": "hopper"}

#: (family path, GemmCoordinate callee) for each registered GEMM version
GEMM_VERSIONS: tuple[tuple[str, str], ...] = (("//Gemm_v1{*}", "@batch:GemmCoordinate_v1"), ("//Gemm_v2{*}", "@batch:GemmCoordinate_v2"))
#: (head path, block callee, dot callee) for each registered attention chain version
ATTENTION_VERSIONS: tuple[tuple[str, str, str], ...] = (("//AttentionHead_v1{*}", "@call:AttnBlock_v1", "@call:DotBf16_v1"),
                                                        ("//AttentionHead_v2{*}", "@call:AttnBlock_v2", "@call:DotBf16_v2"),
                                                        ("//AttentionHead_v3{*}", "@call:AttnBlock_v3", "@call:DotBf16_v1"))
#: norm bodies whose normalize path is `F32Mul` batches feeding an `F2fpBf16` batch per output word
NORM_VERSIONS: tuple[str, ...] = ("//RMSNormTriton_v1{*}", "//RMSNormFusedCuda_v1{*}", "//RMSNormFusedCuda_v2{*}")
NORM_ZIP: tuple[str, ...] = ("@batch:F32Mul_v1", "@batch:F2fpBf16_v1")
#: single-batch bodies: one VU per member
MEMBER_FAMILIES: tuple[tuple[str, str], ...] = (("//SiluMul_v1{*}", "@batch:SiluMulBf16_v1"),
                                                ("//MoeExpertGemm_v1{*}", "@batch:MoeExpertCoordinate_v1"),
                                                ("//MoeExpertGemmW_v1{*}", "@batch:MoeExpertCoordinateW_v1"))
ROUTER_PATH = "//MoeRouterTopK_v1{*}"
ROUTER_EXPF = "@call:NvExpf_v1"
ROUTER_SCAN = "@scan:RouterArgmaxStep_v1"
W_MAX = 32


def resolve_profile(profile: str) -> str:
    """The profile key of `profile` (a key, an alias, or a semantic-profile id containing `sm89` / `sm90` / `fa3`...)."""
    p = (profile or "default").strip().lower()
    if p in PROFILES:
        return p
    if p in _ALIASES:
        return _ALIASES[p]
    tokens = set(p.replace("-", " ").replace("_", " ").split())
    # the architecture-specific tokens decide before the generic ones (`vllm-...-sm90-...` is hopper, not "vllm")
    if tokens & {"sm90", "fa3", "hopper", "h100", "h200"}:
        return "hopper"
    if tokens & {"sm80", "sm86", "sm89", "fa2", "ampere", "ada", "a100", "4090", "l40s", "vllm"}:
        return "ampere"
    raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)} and aliases {sorted(_ALIASES)}")


# ---------------------------------------------------------------------------------------------------------
# the query, as text (no module dependency) and as the frozen AST
# ---------------------------------------------------------------------------------------------------------


def _attention_text(head: str, block: str, dot: str, dot_prim: str) -> str:
    return (f'each("{head}", concat(each("{block}", concat(zip(".", ["@batch:{dot_prim}"]), instances("{dot}"), rest())), rest()))')


def _norm_text(path: str) -> str:
    sels = ", ".join(f'"{s}"' for s in NORM_ZIP)
    return f'each("{path}", concat(zip(".", [{sels}]), rest()))'


def _router_text() -> str:
    return f'each("{ROUTER_PATH}", concat(instances("{ROUTER_EXPF}"), each("{ROUTER_SCAN}", default({W_MAX})), rest()))'


def canonical_cases_text(profile: str = "default") -> list[tuple[str, str]]:
    """The override table as (path, family text) pairs, in canonical order."""
    key = resolve_profile(profile)
    dot_prim = PROFILES[key]["dot"]
    cases: list[tuple[str, str]] = []
    for path, coord in GEMM_VERSIONS:
        cases.append((path, f'members("{path}", "{coord}")'))
    for head, block, dot in ATTENTION_VERSIONS:
        cases.append((head, _attention_text(head, block, dot, dot_prim)))
    for path in NORM_VERSIONS:
        cases.append((path, _norm_text(path)))
    for path, sel in MEMBER_FAMILIES:
        cases.append((path, f'members("{path}", "{sel}")'))
    cases.append((ROUTER_PATH, _router_text()))
    return cases


def canonical_query_text(profile: str = "default") -> str:
    """The canonical VU query in the constructor syntax `verity.ir.query_ast.parse` accepts."""
    body = ", ".join(f'"{p}": {f}' for p, f in canonical_cases_text(profile))
    return f"override(default({W_MAX}), {{{body}}})"


def _gs():
    from verity.ir import query_ast as gateset  # the Query API owner's module (lazy: the text form needs nothing)
    return gateset


def _zip(GS, nodesels: list[str]):
    """`zip(path, [nodesel...])` of the freeze doc, relative to the enclosing body (`.`); the module exports it as `zip_`
    (the builtin's name) -- accept either spelling."""
    z = getattr(GS, "zip", None) or getattr(GS, "zip_")
    return z(".", list(nodesels))


def _attention_family(GS, head: str, block: str, dot: str, dot_prim: str):
    inner = GS.concat(_zip(GS, [f"@batch:{dot_prim}"]), GS.instances(dot), GS.rest())
    return GS.each(head, GS.concat(GS.each(block, inner), GS.rest()))


def _norm_family(GS, path: str):
    return GS.each(path, GS.concat(_zip(GS, list(NORM_ZIP)), GS.rest()))


def _router_family(GS):
    return GS.each(ROUTER_PATH, GS.concat(GS.instances(ROUTER_EXPF), GS.each(ROUTER_SCAN, GS.default(W_MAX)), GS.rest()))


def canonical_cases(profile: str = "default") -> dict[str, Any]:
    """The override table `{path: Family query}` built with the frozen constructors."""
    GS = _gs()
    key = resolve_profile(profile)
    dot_prim = PROFILES[key]["dot"]
    cases: dict[str, Any] = {}
    for path, coord in GEMM_VERSIONS:
        cases[path] = GS.members(path, coord)
    for head, block, dot in ATTENTION_VERSIONS:
        cases[head] = _attention_family(GS, head, block, dot, dot_prim)
    for path in NORM_VERSIONS:
        cases[path] = _norm_family(GS, path)
    for path, sel in MEMBER_FAMILIES:
        cases[path] = GS.members(path, sel)
    cases[ROUTER_PATH] = _router_family(GS)
    return cases


def canonical_vu_query(profile: str = "default"):
    """`override(default(32), canonical_cases(profile))` -- the Query AST (a `verity.ir.query_ast` Query)."""
    GS = _gs()
    return GS.override(GS.default(W_MAX), canonical_cases(profile))


def gemm_query(version: int = 1):
    GS = _gs()
    path, coord = GEMM_VERSIONS[version - 1]
    return GS.members(path, coord)


def attention_query(profile: str = "default", version: int = 3):
    GS = _gs()
    head, block, dot = ATTENTION_VERSIONS[version - 1]
    return _attention_family(GS, head, block, dot, PROFILES[resolve_profile(profile)]["dot"])


def canonical_query_id(profile: str = "default") -> str:
    return _gs().query_id(canonical_vu_query(profile))


def canonical_query_document(profile: str = "default", *, created_utc: str | None = None, extra_provenance: dict | None = None) -> dict:
    """The `verity-ir/query/v1` document of the canonical query: schema, `query_id` (sha256 of the canonical JSON AST,
    program-independent), the AST, provenance.  Carries no `kind` / `w_max` (freeze doc: the use configuration and the
    binding to a Program are `verity_vllm.ir.query_artifact`'s document)."""
    GS = _gs()
    key = resolve_profile(profile)
    q = canonical_vu_query(key)
    prov = {"author": AUTHOR, "version": VERSION, "profile": key, "profile_doc": PROFILES[key]["doc"],
            "requested_profile": profile, "text": canonical_query_text(key),
            "created_utc": created_utc or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if extra_provenance:
        prov.update(extra_provenance)
    return {"schema": SCHEMA, "query_id": GS.query_id(q), "query": GS.to_json(q), "provenance": prov}


def query_from_document(doc: dict):
    """The Query AST of a `verity-ir/query/v1` document; refuses a document whose `query_id` does not match its AST."""
    GS = _gs()
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document: schema={doc.get('schema')!r}")
    q = GS.from_json(doc["query"])
    qid = GS.query_id(q)
    if doc.get("query_id") not in (None, qid):
        raise ValueError(f"query_id {doc.get('query_id')} does not match the AST ({qid})")
    return q


__all__ = ["SCHEMA", "AUTHOR", "VERSION", "PROFILES", "W_MAX", "GEMM_VERSIONS", "ATTENTION_VERSIONS", "NORM_VERSIONS", "NORM_ZIP",
           "MEMBER_FAMILIES", "ROUTER_PATH", "resolve_profile", "canonical_cases_text", "canonical_query_text", "canonical_cases",
           "canonical_vu_query", "gemm_query", "attention_query", "canonical_query_id", "canonical_query_document", "query_from_document"]
