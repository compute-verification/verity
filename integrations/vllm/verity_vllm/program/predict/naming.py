"""What a request Program's Build says about it beyond its digest, written down from φ: the audit's names without a trace.

A Build dir (`pipeline.build`) holds, beside the Program, what no Program digest covers and the audit reads
(`check.replay.index.ProgramIndex`, `correspondence.reader_for_query`): each Call's name (`<module>/<fx node>[/<member>][<row>]`,
the instances' rows and the descriptor's `node_hints`), the parameters' names and types (the weights record's fields under the
request wrapper's `model.`), the runtime correspondence (`correspondence.runtime`) and the cached prefix's layout
(`prefix_cache.json`). `named_program` builds a request Program (`build.build_program`) with them (`name_request`), as vLLM's
trace of that Program names them, and `write_build` writes the Build dir, so a verifier's own Build of a Program needs neither
the trace nor vLLM.

The names are vLLM's dense Llama-family decoder's under the request wrapper on one rank. Each forward pass of M rows at
positions p0.. (the prefill's LP − C rows at C, then each decode's one row) is these fx nodes in call order, each Call a row r of
its node's output unless said otherwise: the embedding; layer 0's input norm (`triton_launch`); per layer L, its input norm
(`fused_add_rms_norm`, L > 0), the q/k/v linear, the rotary embedding (one node: every row's query, then every row's key, named
`<node>/query[r]` and `<node>/key[r]`), the attention (named `<node>[t=p0+r]`), the o linear, the post-attention norm, the
gate/up linear, `silu_and_mul` and the down linear; the final norm; then the sampled row's logits (`triton_launch`, row 0 of
[1, V]) and its select (`argmax`, or `gumbel_topp_token_select` with its constants `top_p`, `pos`, `temperature` and, at a
constant split count, `splits`, named `<node>/<member>`, before its row 0), under the root module. A linear is one
`triton_launch` GEMM, or under the exact-int8 linears `row_quant` then `scaled_mm`. An fx node is named by its op, counted over
the request (`embedding`, `embedding_1`, ...). Every node of the emitted body is held to the kind its place names
(`build._Specs`), so a body these names don't fit is refused, never misnamed; the dead units' Calls are named, then dropped with
them (`prune.kept_records`). Anything else (the fused MoE, q/k norms, biases, Gemma-2's chains and caps, an activation other than
SiLU, a sliding window, a step wrapper, tensor parallelism, the DSA and DSV4 decoders) is `Unexpressible`: its names are its
trace's.

The correspondence states what the audit's reader reads (`reader_for_query.recorded_names`): each occurrence's module, fx node and
op, each Call's member, row or naming hint, and its return slots, each in an fx output or in an argument the node writes in place
(the occurrence's `inputs` lists those, with their views). It leaves unrecorded what only a trace sees and only the prover's Match
reads: the Calls' argument slots (`args` empty), the other fx arguments, strides, lifetimes, module classes and lowering rules.
"""
from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from verity.ml.circuits.types import Record

from .build import BuildResult, _Decoder, build_program, decoder
from .common import Unexpressible
from .phi import Phi

#: the request wrapper's attribute holding the model (`vllm_meta.GreedyRequest.model`): the root of every module path and weight name
WRAPPER = "model."
#: the op of each fx node a dense request's Calls are lowered from, by the node's base name
OPS = {"embedding": "aten.embedding.default", "triton_launch": "verity_cba.triton_launch.default",
       "row_quant": "verity_int8.row_quant.default", "scaled_mm": "verity_int8.scaled_mm.default",
       "rotary_embedding": "_C.rotary_embedding.default", "unified_attention_with_output": "vllm.unified_attention_with_output.default",
       "fused_add_rms_norm": "_C.fused_add_rms_norm.default", "silu_and_mul": "_C.silu_and_mul.default",
       "argmax": "aten.argmax.default", "gumbel_topp_token_select": "vllm_iface.gumbel_topp_token_select.default"}
#: what `write_build`'s `result.json` names as the Build's maker
BUILT_BY = "verity_vllm.program.predict.naming"
BF16 = "bfloat16"


@dataclass(frozen=True)
class Named:
    """A request Program's names: each Call's by node index (`names`), the parameters' (`params`: [name, type] each, as an
    instances head lists them), the runtime correspondence document, the cached prefix's layout (None without one) and the
    request's shape (`shape`: wrapper, LP, T, C and sampling, as a Build's `result.json` states them)."""

    names: dict[int, str]
    params: list[list[str]]
    correspondence: dict
    prefix: dict | None
    shape: dict


def _view(shape: tuple[int, ...], dtype: str) -> dict:
    return {"shape": list(shape), "dtype": dtype}


class _Walk:
    """The trace's fx nodes over a request's emitted body, in Call order: each node's occurrence, each Call's record and name."""

    def __init__(self, nodes: list) -> None:
        self.nodes, self.k = nodes, 0
        self.seen: Counter = Counter()
        self.occurrences: list[dict] = []
        self.calls: list[dict] = []
        self.names: dict[int, str] = {}

    def site(self, op: str, module: str, views: list[dict], written: tuple = ()) -> tuple[int, str]:
        """A new fx node of `op` under `module` -> its occurrence and node name. `views` are its outputs, or with `written` the fx
        arguments it writes in place ((argument, source) each, the occurrence's `inputs` in that order)."""
        n = self.seen[op]
        self.seen[op] += 1
        node = op if n == 0 else f"{op}_{n}"
        occ: dict[str, Any] = {"node": node, "module": module, "op": OPS[op], "collective": None, "outputs": [None] if written else views}
        if written:
            occ["inputs"] = [{"node": "", "arg": a, "source": s, "view": v} for (a, s), v in zip(written, views, strict=True)]
        self.occurrences.append(occ)
        return len(self.occurrences) - 1, node

    def call(self, ok: Callable[[Any], bool], what: str, occ: int, name: str, returns: list[dict], *, row: int | None = None,
             hint: str | None = None, member: str | None = None) -> None:
        """The next node of the body as Call `name`, refused unless `ok` holds of its Definition."""
        k = self.k
        if k >= len(self.nodes) or not ok(self.nodes[k].fn):
            got = self.nodes[k].fn.id if k < len(self.nodes) else "past the body's end"
            raise Unexpressible(f"naming: node {k} is {got}, not the {what} the trace names {name}")
        self.calls.append({"occ": occ, "member": member, "row": row, "hint": hint, "returns": returns, "args": []})
        self.names[k] = name
        self.k += 1

    @staticmethod
    def slots(views: list[dict], r: int, in_place: bool, ports: range | None = None) -> list[dict]:
        """Row r of each view as a return slot: port p in view p, an fx output or the p-th written argument."""
        out = []
        for p in ports or range(len(views)):
            w = 1
            for s in views[p]["shape"][1:]:
                w *= s
            out.append({"port": len(out), "output": None if in_place else p, **({"written_arg": p} if in_place else {}),
                        "leaves": [r * w, (r + 1) * w]})
        return out

    def rows(self, fn: Any, what: str, op: str, module: str, M: int, views: list[dict], written: tuple = ()) -> None:
        """A node of `op` under `module` over M rows and its Calls of `fn`, one per row."""
        occ, node = self.site(op, module, views, written)
        for r in range(M):
            self.call(lambda f: f is fn, what, occ, f"{module}/{node}[{r}]", self.slots(views, r, bool(written)), row=r)


def _family(fn: Any) -> str:
    return str(getattr(fn, "base_id", fn.id)).split("{")[0].split("[")[0].rsplit("_v", 1)[0]


def refuse(net: Any) -> None:
    """`Unexpressible` unless `name_request` names `net`'s request: the dense Llama-family decoder on one rank."""
    if type(net) is not _Decoder or net.world != 1:
        raise Unexpressible(f"naming: the {net.model_class} decoder at TP{getattr(net, 'world', '?')} (the dense decoder on one "
                            f"rank only)")
    C = net.C
    facts = [f for f, on in (("the fused MoE", C.E), ("q/k norms", C.qk_norm), ("a q/k/v bias", C.qkv_bias), ("the unfused norm chain", C.norm_chain),
                             ("the extra MLP norms", C.extra_norms), ("the embedding scale", C.embed_scale), ("soft-caps", C.attn_cap or C.final_cap),
                             (f"the activation {C.act!r}", C.act != "silu"), ("a sliding window", C.window is not None)) if on]
    if facts:
        raise Unexpressible(f"naming: {', '.join(facts)} (vLLM's dense decoder with SiLU and fused norms only)")


def _forward(w: _Walk, S: Any, C: Any, M: int, p0: int) -> None:
    """One forward pass of M rows at positions p0.. (module docstring)."""
    H, bf = C.H, BF16
    lin_in_place = ((2, "other"),)

    def linear(fn: Any, module: str, K: int, N: int) -> None:
        if not C.int8:
            w.rows(fn, "GEMM", "triton_launch", module, M, [_view((M, N), bf)], lin_in_place)
            return
        quant, mm = fn
        w.rows(quant, "row quantizer", "row_quant", module, M, [_view((M, K), "int8"), _view((M,), "float32")])
        w.rows(mm, "scaled int8 GEMM", "scaled_mm", module, M, [_view((M, N), bf)])

    def fused(module: str) -> None:
        w.rows(S.fused, "fused add-norm", "fused_add_rms_norm", module, M, [_view((M, H), bf)] * 2, ((0, "call"), (1, "call")))

    w.rows(S.embedding, "embedding", "embedding", f"{WRAPPER}model.embed_tokens", M, [_view((M, H), bf)])
    w.rows(S.norm0, "input norm", "triton_launch", f"{WRAPPER}model.layers.0.input_layernorm", M, [_view((M, H), bf)], lin_in_place)
    attn = {id(f) for f in S._attn.values()}
    qw, kw = C.NH * C.D, C.KVH * C.D
    for L in range(C.NL):
        p = f"{WRAPPER}model.layers.{L}."
        if L:
            fused(p + "input_layernorm")
        linear(S.qkv, p + "self_attn.qkv_proj", H, C.QKV)
        module = p + "self_attn.rotary_emb"
        views = [_view((M, qw), bf), _view((M, kw), bf)]
        occ, node = w.site("rotary_embedding", module, views, ((1, "call"), (2, "call")))
        for half, fn, port in (("query", S.rope_q, 0), ("key", S.rope_k, 1)):
            for r in range(M):
                hint = f"{module}/{node}/{half}[{r}]"
                w.call(lambda f, fn=fn: f is fn, f"RoPE {half}", occ, hint, w.slots(views, r, True, range(port, port + 1)), hint=hint)
        module = p + "self_attn.attn"
        views = [_view((M, C.NH, C.D), bf)]
        occ, node = w.site("unified_attention_with_output", module, views, ((3, "call"),))
        for r in range(M):
            hint = f"{module}/{node}[t={p0 + r}]"
            w.call(lambda f: id(f) in attn, "attention", occ, hint, w.slots(views, r, True), hint=hint)
        linear(S.o, p + "self_attn.o_proj", qw, H)
        fused(p + "post_attention_layernorm")
        linear(S.gate_up, p + "mlp.gate_up_proj", H, 2 * C.I)
        w.rows(S.silu, "SiluMul", "silu_and_mul", p + "mlp.act_fn", M, [_view((M, C.I), bf)], ((0, "other"),))
        linear(S.down, p + "mlp.down_proj", C.I, H)
    fused(f"{WRAPPER}model.norm")


def _sample(w: _Walk, S: Any, C: Any, sampling: dict) -> None:
    """The sampled row's logits and its select (module docstring)."""
    from .common import kind
    w.rows(S.logits, "LM head GEMM", "triton_launch", f"{WRAPPER}logits_processor", 1, [_view((1, C.V), BF16)], ((2, "other"),))
    if sampling.get("kind", "greedy") == "greedy":
        occ, node = w.site("argmax", "", [_view((1,), "int64")])
        select = _family(kind("token_select", V=C.V))
        w.call(lambda f: _family(f) == select, "token select", occ, f"{node}[0]", w.slots([_view((1,), "int64")], 0, False), row=0)
        return
    derived = sampling["splits"] == "derived"
    select = _family(kind("token_select_stochastic", V=C.V, CONSTRUCTION=S.tp.effective_sampler_construction()) if derived
                     else kind("token_select_stochastic_split", V=C.V, S=int(sampling["splits"])))
    occ, node = w.site("gumbel_topp_token_select", "", [_view((1,), "int32")])
    for member, W in (("top_p", 32), ("pos", 64), ("temperature", 32), *(() if derived else (("splits", 32),))):
        const = _family(kind("const", W=W, BITS=0))
        w.call(lambda f, c=const: _family(f) == c, f"{W}-bit constant", occ, f"{node}/{member}",
               [{"port": 0, "output": None, "leaves": None}], member=member)
    w.call(lambda f: _family(f) == select, "token select", occ, f"{node}[0]", w.slots([_view((1,), "int32")], 0, False), row=0)


def name_request(raw: Any, net: Any, params: tuple, *, LP: int, T: int, C: int, sampling: dict) -> tuple[Any, Named]:
    """The request Program's body `raw` (`build_program`'s, before `prune.dead_free`) of `net` (a `build._Decoder` that `refuse`
    passes) over `params` (its user inputs, then the weights record), prefill LP rows after a cached prefix of C, then T decodes,
    sampled under `sampling` -> (its body in the registered compiler's form, its names), or `Unexpressible` (module docstring)."""
    from verity_vllm.correspondence.runtime import SCHEMA
    from verity_vllm.program.prune import kept_records
    S, Cm = net.specs, net.C
    w = _Walk(list(raw.nodes))
    for M, p0 in [(LP - C, C)] + [(1, LP + s) for s in range(T)]:
        _forward(w, S, Cm, M, p0)
        _sample(w, S, Cm, sampling)
    if w.k != len(w.nodes):
        raise Unexpressible(f"naming: the body has {len(w.nodes)} nodes and the trace names {w.k}, node {w.k} "
                            f"({w.nodes[w.k].fn.id}) first unnamed")
    body, calls, names = kept_records(raw, w.calls, w.names, registered=True)
    *user, (wname, wrec) = params
    weights = Record(*((WRAPPER + f, t) for f, t in wrec.fields))
    doc = {"schema": SCHEMA, "body": "program_fn", "inputs": [n for n, _ in user], "weights": [n for n, _ in weights.fields],
           "rank": None, "occurrences": w.occurrences, "calls": calls}
    names_of = [n for n, _ in params]
    prefix = {"input": names_of.index("prefix"), "C": C, "row_leaves": Cm.KVH * Cm.D,
              "layers": [f"model.layers.{L}.self_attn.attn" for L in range(Cm.NL)]} if C else None
    shape = {"wrapper": "request", "LP": LP, "T": T, "C": C, "sampling": dict(sampling)}
    return body, Named(names, [[n, str(t)] for n, t in user] + [[wname, str(weights)]], doc, prefix, shape)


def named_program(phi: Phi, LP: int, T: int, C: int = 0) -> BuildResult:
    """φ's request Program at (LP, T) after a cached prefix of C rows, in the registered compiler's form (`build.build_program`,
    ``dead_free``), with its names (`BuildResult.named`), or `Unexpressible` where `refuse` refuses its decoder, before any build."""
    refuse(decoder(phi, 1, int(LP) + int(T)))
    sampling = dict(phi.sampling or {"kind": "greedy"})
    return build_program(phi, "request", LP=LP, T=T, C=C, dead_free=True, finish=lambda raw, net, params: name_request(
        raw, net, params, LP=int(LP), T=int(T), C=int(C), sampling=sampling))


def write_build(out: str, program: Any, named: Named) -> str:
    """`program`'s Build dir at `out`, as a trace's reads to the audit (`prove_path.build_of`, `ProgramIndex`): its descriptor
    under `named`'s `node_hints` and correspondence, its instances under `named`'s Call and parameter names, its `result.json`
    (the Program's function, digest and correspondence digest, the request's shape) and its `prefix_cache.json` with a cached
    prefix -> the Program's digest."""
    from verity.ml.circuits.codec import encode_program, program_digest
    from verity_vllm.correspondence.runtime import NAMESPACE, correspondence_digest
    from verity_vllm.pipeline.build_instances import write_instances
    desc = encode_program(program)
    digest = program_digest(desc)
    desc["annotations"] = {"node_hints": {str(k): v for k, v in sorted(named.names.items())}, NAMESPACE: named.correspondence}
    cd = correspondence_digest(desc)
    os.makedirs(out, exist_ok=True)
    with gzip.open(os.path.join(out, "descriptor.json.gz"), "wb") as f:
        f.write(json.dumps(desc, sort_keys=True).encode())
    write_instances(out, program, digest, cd, {}, names=named.names, params=named.params)
    if named.prefix is not None:
        with open(os.path.join(out, "prefix_cache.json"), "w") as f:
            json.dump(named.prefix, f, indent=1)
    res = {"ok": True, **named.shape, "tp": {"world": 1, "rank": 0}, "built_by": BUILT_BY,
           "program": {"function": program.fn.id, "digest": digest, "n_root_nodes": len(program.fn.body.nodes), "correspondence_digest": cd},
           **({"prefix": named.prefix} if named.prefix is not None else {})}
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    return digest


__all__ = ["Named", "named_program", "write_build"]
