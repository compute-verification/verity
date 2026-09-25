"""``description.json`` (contract section 6): the node list the checker runs from.

Layout::

    {
      "schema": "verity-vllm/description/v1",
      "profile_id": "0123456789ab",
      "model": "HuggingFaceTB/SmolLM2-135M",
      "symbols": {"H": 576, "V": 49152},
      "nodes": [
        {"op": "layers.0.mlp.act_mul", "layer": "0", "op_type": "silu_mul",
         "dtype": "bfloat16", "shape": ["I"], "ports": 1,
         "phases": ["prefill", "decode"], "positions": "all",
         "args": [{"name": "gate_up", "source": "node",
                   "op": "layers.0.mlp.gate_up_proj.gemm", "port": 0}],
         "relation": "silu_mul.bf16.v1", "origin": "captured",
         "decompositions": [{"id": "row",
                             "interface": {"input_bytes": 6144, "output_bits": 24576}}],
         "runtime_impl": "vllm::silu_and_mul"}
      ]
    }

- ``op`` is the ``op=`` field of the node's outputs and is unique; ``layer`` is its ``l=`` field.
- ``shape`` is the per-position row shape of port 0 (ints or symbol names); extra ports are
  listed under ``outputs`` as ``{"port": 1, "dtype": ..., "shape": [...]}``.
- ``positions`` is ``"all"`` (every processed position) or ``"sampled"`` (only positions whose
  sampler produced a token: ``lm_head``, ``sampler``).
- ``args`` name the node's inputs by identity rule.  Sources:
  ``node`` (``op``, ``port``, optional ``t``: ``same`` | ``prev`` | ``all_le`` | ``all_lt``,
  optional ``shard``/``expert``), ``weight`` (``param``, optional ``shard``), ``token``
  (``which``: ``input`` | ``sampled``, from the boundary tree), ``boundary`` (``field``), and
  ``const`` (``value``; part of the description's digest, not a committed leaf).
- ``relation`` is a registry id (:mod:`verity_vllm.program.backends.relations`); ``origin`` is ``captured`` or
  ``derived``; ``decompositions`` are the candidates of Table 2.

:class:`Description` validates the document, resolves symbolic shapes, enumerates the node
instances a request should have produced, and resolves an instance's argument identities.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.program import dtypes
from verity_vllm.commit import hashing, identity
from verity_vllm.observe.capture_v1 import Boundary

DESCRIPTION_SCHEMA = "verity-vllm/description/v1"

ARG_SOURCES = ("node", "weight", "token", "boundary", "const")
NODE_T_RULES = ("same", "prev", "all_le", "all_lt")
# ``expert`` on an arg: an int names one slot; "same" = the consuming instance's own slot (a
# per-expert weight ``<param>.e<slot>`` or the same slot of a slotted producer); "all" = the list of
# every slot of a slotted producer (the combination reads all of them; which are non-zero is the
# checked router result's business, never a stored mask).
EXPERT_RULES = ("same", "all")
POSITION_RULES = ("all", "sampled")
OP_TYPES = (
    "embedding",
    "gemm",
    "rmsnorm",
    "layernorm",
    "rope",
    "attention",
    "silu_mul",
    "residual_add",
    "lm_head",
    "argmax",
    "sampler",
    "router",
    "expert",
    "combine",
    "allreduce",
    "quantize",
    "dequantize",
    "other",
)


class DescriptionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Arg:
    name: str
    source: str
    op: str | None = None
    port: int = 0
    t: str = "same"
    shard: int | None = None
    expert: int | None = None
    param: str | None = None
    which: str | None = None
    field: str | None = None
    value: Any = None
    # adapter-specific keys the schema does not interpret (e.g. W1's ``cols`` column slice,
    # ``unbound``); preserved verbatim so checkers can read them and digests are stable
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"name": self.name, "source": self.source, **self.extra}
        if self.source == "node":
            doc.update({"op": self.op, "port": self.port})
            if self.t != "same":
                doc["t"] = self.t
            if self.shard is not None:
                doc["shard"] = self.shard
            if self.expert is not None:
                doc["expert"] = self.expert
        elif self.source == "weight":
            doc["param"] = self.param
            if self.shard is not None:
                doc["shard"] = self.shard
            if self.expert is not None:
                doc["expert"] = self.expert
        elif self.source == "token":
            doc["which"] = self.which
        elif self.source == "boundary":
            doc["field"] = self.field
        else:
            doc["value"] = self.value
        return doc

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Arg:
        if not isinstance(doc, Mapping) or "name" not in doc or "source" not in doc:
            raise DescriptionError(f"arg needs name and source: {doc!r}")
        source = doc["source"]
        if source not in ARG_SOURCES:
            raise DescriptionError(f"arg {doc['name']!r}: source must be one of {ARG_SOURCES}")
        known = {
            "name", "source", "op", "port", "t", "shard", "expert", "param", "which", "field",
            "value",
        }  # fmt: skip
        arg = cls(
            name=str(doc["name"]),
            source=source,
            op=doc.get("op"),
            port=int(doc.get("port", 0)),
            t=str(doc.get("t", "same")),
            shard=doc.get("shard"),
            expert=doc.get("expert"),
            param=doc.get("param"),
            which=doc.get("which"),
            field=doc.get("field"),
            value=doc.get("value"),
            extra={k: v for k, v in doc.items() if k not in known},
        )
        if arg.expert is not None and not (
            isinstance(arg.expert, int) or arg.expert in EXPERT_RULES
        ):
            raise DescriptionError(
                f"arg {arg.name!r}: expert must be a slot index or one of {EXPERT_RULES}"
            )
        if source == "node":
            if not arg.op:
                raise DescriptionError(f"arg {arg.name!r}: node source needs op")
            if arg.t not in NODE_T_RULES:
                raise DescriptionError(f"arg {arg.name!r}: t must be one of {NODE_T_RULES}")
        elif source == "weight":
            if not arg.param:
                raise DescriptionError(f"arg {arg.name!r}: weight source needs param")
        elif source == "token":
            if arg.which not in ("input", "sampled"):
                raise DescriptionError(f"arg {arg.name!r}: which must be input|sampled")
        elif source == "boundary":
            if arg.field not in identity.BOUNDARY_FIELDS_WITHOUT_T:
                raise DescriptionError(
                    f"arg {arg.name!r}: field must be one of {identity.BOUNDARY_FIELDS_WITHOUT_T}"
                )
        elif "value" not in doc:
            raise DescriptionError(f"arg {arg.name!r}: const source needs value")
        return arg


@dataclass(frozen=True, slots=True)
class Decomposition:
    id: str
    interface: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "interface": dict(self.interface)}


@dataclass(frozen=True, slots=True)
class Node:
    op: str
    layer: str
    op_type: str
    dtype: str
    shape: tuple[int | str, ...]
    relation: str
    origin: str = "captured"
    ports: int = 1
    outputs: tuple[dict[str, Any], ...] = ()
    phases: tuple[str, ...] = ("prefill", "decode")
    positions: str = "all"
    args: tuple[Arg, ...] = ()
    decompositions: tuple[Decomposition, ...] = (Decomposition("row"),)
    runtime_impl: str | None = None
    shards: int | None = None  # TP ranks producing s=<rank> outputs; None -> no s field
    experts: int | None = None  # expert slots producing e=<slot> outputs; None -> no e field
    declared_domain: int | None = None
    """Declared mathematical domain width in bits of this node's (integer) outputs — GUIDANCE
    §31.2/§32.5 (lane OUT16).  For B0 token outputs: 16 (b = ceil(log2 V), V = 49,152) inside the
    32-bit storage container.  Additive and optional: absent, nothing changes anywhere (old
    descriptions are byte-identical and no domain check runs).  Present, the verifier's
    output-linkage check unconditionally requires every authenticated output value of this node
    (and the boundary record leaf linked to it) to satisfy ``value >> declared_domain == 0`` —
    the high container bits are zero — on every audit, independent of both sampling draws
    (:func:`verity_vllm.program.backends.relations.check_output_linkage` / ``check_output_domains``)."""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "op": self.op,
            "layer": self.layer,
            "op_type": self.op_type,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "ports": self.ports,
            "phases": list(self.phases),
            "positions": self.positions,
            "args": [a.to_json() for a in self.args],
            "relation": self.relation,
            "origin": self.origin,
            "decompositions": [d.to_json() for d in self.decompositions],
        }
        if self.outputs:
            doc["outputs"] = [dict(o) for o in self.outputs]
        if self.runtime_impl is not None:
            doc["runtime_impl"] = self.runtime_impl
        if self.shards is not None:
            doc["shards"] = self.shards
        if self.experts is not None:
            doc["experts"] = self.experts
        if self.declared_domain is not None:
            doc["declared_domain"] = self.declared_domain
        doc.update(self.extra)
        return doc

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Node:
        known = {
            "op", "layer", "op_type", "dtype", "shape", "ports", "phases", "positions", "args",
            "relation", "origin", "decompositions", "outputs", "runtime_impl", "shards", "experts",
            "declared_domain",
        }  # fmt: skip
        for key in ("op", "layer", "op_type", "dtype", "shape", "relation"):
            if key not in doc:
                raise DescriptionError(f"node {doc.get('op')!r} lacks {key}")
        node = cls(
            op=str(doc["op"]),
            layer=str(doc["layer"]),
            op_type=str(doc["op_type"]),
            dtype=dtypes.normalize_dtype(doc["dtype"]),
            shape=tuple(_shape_dim(d) for d in doc["shape"]),
            relation=str(doc["relation"]),
            origin=str(doc.get("origin", "captured")),
            ports=int(doc.get("ports", 1)),
            outputs=tuple(dict(o) for o in doc.get("outputs", [])),
            phases=tuple(doc.get("phases", ["prefill", "decode"])),
            positions=str(doc.get("positions", "all")),
            args=tuple(Arg.from_json(a) for a in doc.get("args", [])),
            decompositions=tuple(
                Decomposition(str(d["id"]), dict(d.get("interface", {})))
                for d in doc.get("decompositions", [{"id": "row"}])
            ),
            runtime_impl=doc.get("runtime_impl"),
            shards=doc.get("shards"),
            experts=doc.get("experts"),
            declared_domain=doc.get("declared_domain"),
            extra={k: v for k, v in doc.items() if k not in known},
        )
        node.validate()
        return node

    def validate(self) -> None:
        if not identity.OP_RE.match(self.op):
            raise DescriptionError(f"node op {self.op!r} must match {identity.OP_RE.pattern}")
        try:
            identity._validate_layer(self.layer)
        except identity.IdentityError as error:
            raise DescriptionError(f"node {self.op}: {error}") from error
        if self.op_type not in OP_TYPES:
            raise DescriptionError(f"node {self.op}: op_type {self.op_type!r} not in {OP_TYPES}")
        if self.origin not in ("captured", "derived"):
            raise DescriptionError(f"node {self.op}: origin {self.origin!r}")
        if self.ports < 1:
            raise DescriptionError(f"node {self.op}: ports must be >= 1")
        if not self.phases or any(p not in identity.PHASES for p in self.phases):
            raise DescriptionError(f"node {self.op}: phases {self.phases!r}")
        if self.positions not in POSITION_RULES:
            raise DescriptionError(f"node {self.op}: positions must be one of {POSITION_RULES}")
        names = [a.name for a in self.args]
        if len(set(names)) != len(names):
            raise DescriptionError(f"node {self.op}: duplicate arg names")
        if not self.decompositions:
            raise DescriptionError(f"node {self.op}: at least one candidate decomposition")
        for out in self.outputs:
            if "port" not in out or int(out["port"]) < 1 or int(out["port"]) >= self.ports:
                raise DescriptionError(f"node {self.op}: outputs[].port must be in 1..ports-1")
            dtypes.normalize_dtype(out.get("dtype", self.dtype))
        if self.declared_domain is not None:
            if isinstance(self.declared_domain, bool) or not isinstance(self.declared_domain, int):
                raise DescriptionError(f"node {self.op}: declared_domain must be an int (bits)")
            size = dtypes.itemsize(self.dtype)
            if size is None or self.dtype not in (
                "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
            ):  # fmt: skip
                raise DescriptionError(
                    f"node {self.op}: declared_domain needs an integer container dtype, got {self.dtype}"
                )
            if not 1 <= self.declared_domain <= 8 * size:
                raise DescriptionError(
                    f"node {self.op}: declared_domain {self.declared_domain} outside 1..{8 * size} "
                    f"({self.dtype} container)"
                )

    def port_dtype(self, port: int) -> str:
        if port == 0:
            return self.dtype
        for out in self.outputs:
            if int(out["port"]) == port:
                return dtypes.normalize_dtype(out.get("dtype", self.dtype))
        return self.dtype

    def port_shape(self, port: int) -> tuple[int | str, ...]:
        if port == 0:
            return self.shape
        for out in self.outputs:
            if int(out["port"]) == port:
                return tuple(_shape_dim(d) for d in out.get("shape", self.shape))
        return self.shape

    def arg(self, name: str) -> Arg:
        for a in self.args:
            if a.name == name:
                return a
        raise KeyError(f"node {self.op} has no arg {name!r}")


def _shape_dim(d: Any) -> int | str:
    if isinstance(d, bool):
        raise DescriptionError(f"bad shape dim {d!r}")
    if isinstance(d, int):
        if d < 0:
            raise DescriptionError(f"negative shape dim {d}")
        return d
    if isinstance(d, str) and d.isidentifier():
        return d
    raise DescriptionError(f"shape dims are ints or symbol names, got {d!r}")


@dataclass(frozen=True, slots=True)
class NodeInstance:
    """A description node bound to one request and position: the default VU."""

    node: Node
    request_id: str
    t: int
    phase: str
    port: int = 0
    shard: int | None = None
    expert: int | None = None

    def output_id(self, profile_id: str) -> str:
        return str(
            identity.value_id(
                profile_id,
                self.request_id,
                self.t,
                self.phase,
                self.node.layer,
                self.node.op,
                self.port,
                shard=self.shard,
                expert=self.expert,
            )
        )

    @property
    def dtype(self) -> str:
        return self.node.port_dtype(self.port)


class Description:
    def __init__(self, doc: Mapping[str, Any]) -> None:
        if not isinstance(doc, Mapping):
            raise DescriptionError("description must be a JSON object")
        if doc.get("schema") != DESCRIPTION_SCHEMA:
            raise DescriptionError(f"description.schema must be {DESCRIPTION_SCHEMA!r}")
        if "nodes" not in doc or not isinstance(doc["nodes"], list):
            raise DescriptionError("description.nodes must be a list")
        self.doc = dict(doc)
        self.profile_id = doc.get("profile_id")
        self.model = doc.get("model")
        self.symbols: dict[str, int] = {}
        for name, value in dict(doc.get("symbols", {})).items():
            if not (isinstance(name, str) and name.isidentifier()):
                raise DescriptionError(f"symbol name {name!r}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DescriptionError(f"symbol {name} must be a non-negative int")
            self.symbols[name] = value
        self.nodes: list[Node] = [Node.from_json(n) for n in doc["nodes"]]
        self._by_op: dict[str, Node] = {}
        for node in self.nodes:
            if node.op in self._by_op:
                raise DescriptionError(f"duplicate node op {node.op!r}")
            self._by_op[node.op] = node
        self.validate()

    # --- validation ---

    def validate(self, known_relations: Iterable[str] | None = None) -> None:
        known = set(known_relations) if known_relations is not None else None
        for node in self.nodes:
            for port in range(node.ports):
                for dim in node.port_shape(port):
                    if isinstance(dim, str) and dim not in self.symbols:
                        raise DescriptionError(f"node {node.op}: unknown symbol {dim!r}")
            for arg in node.args:
                if arg.source == "node":
                    target = self._by_op.get(arg.op or "")
                    if target is None:
                        raise DescriptionError(
                            f"node {node.op}: arg {arg.name} refers to {arg.op!r}"
                        )
                    if arg.port >= target.ports:
                        raise DescriptionError(f"node {node.op}: arg {arg.name} port {arg.port}")
            if known is not None and node.relation not in known:
                raise DescriptionError(f"node {node.op}: unknown relation {node.relation!r}")

    def to_json(self) -> dict[str, Any]:
        doc = dict(self.doc)
        doc["nodes"] = [n.to_json() for n in self.nodes]
        return doc

    def digest(self) -> bytes:
        return hashing.json_digest(self.to_json())

    # --- lookups ---

    def node(self, op: str) -> Node:
        try:
            return self._by_op[op]
        except KeyError as error:
            raise KeyError(f"description has no node {op!r}") from error

    def __contains__(self, op: str) -> bool:
        return op in self._by_op

    def nodes_of_type(self, op_type: str) -> list[Node]:
        return [n for n in self.nodes if n.op_type == op_type]

    def op_families(self) -> list[str]:
        return sorted({n.op_type for n in self.nodes})

    def resolve_shape(self, shape: Sequence[int | str]) -> tuple[int, ...]:
        out = []
        for dim in shape:
            if isinstance(dim, str):
                if dim not in self.symbols:
                    raise DescriptionError(f"unknown symbol {dim!r}")
                out.append(self.symbols[dim])
            else:
                out.append(int(dim))
        return tuple(out)

    def leaf_spec(self, node: Node, port: int = 0) -> tuple[str, tuple[int, ...]]:
        """(dtype, concrete shape) a leaf of this node's port must have (N04's schema check)."""
        return node.port_dtype(port), self.resolve_shape(node.port_shape(port))

    # --- instances ---

    def positions_of(self, node: Node, boundary: Boundary) -> list[tuple[int, str]]:
        """(t, phase) pairs at which ``node`` has an output for this request."""
        out = []
        sampled = set(boundary.sampled_positions())
        seen: set[tuple[int, str]] = set()
        for step in boundary.steps:
            for t in step.positions:
                # capture-format v1.1: the id's phase is a function of the position (prefill iff
                # t < prompt_len), not of the step that happened to compute it, so a recomputed
                # position after a preemption keeps its identity.  On non-preempted trajectories
                # this equals the step's phase.
                phase = boundary.phase_at(t)
                if phase not in node.phases:
                    continue
                # a logits-row node (lm_head, sampler.select) exists only where the runtime
                # computed a logits row: the last scheduled position of the step, and only when
                # that position is a sampled one (a partial prefill chunk's row is discarded)
                if node.positions == "sampled" and (t not in sampled or t != step.positions[-1]):
                    continue
                if (t, phase) in seen:  # a recompute step repeats identities; list each once
                    continue
                seen.add((t, phase))
                out.append((t, phase))
        return out

    def instances(self, boundary: Boundary, ops: Iterable[str] | None = None) -> list[NodeInstance]:
        nodes = [self.node(op) for op in ops] if ops is not None else self.nodes
        out: list[NodeInstance] = []
        for node in nodes:
            shards: list[int | None] = list(range(node.shards)) if node.shards else [None]
            experts: list[int | None] = list(range(node.experts)) if node.experts else [None]
            for t, phase in self.positions_of(node, boundary):
                for port in range(node.ports):
                    for s in shards:
                        for e in experts:
                            out.append(
                                NodeInstance(node, boundary.request_id, t, phase, port, s, e)
                            )
        return out

    def expected_ids(self, boundary: Boundary, profile_id: str) -> dict[str, NodeInstance]:
        return {inst.output_id(profile_id): inst for inst in self.instances(boundary)}

    def instance_of(self, canonical_id: str, boundary: Boundary) -> NodeInstance:
        cid = identity.parse(canonical_id)
        if cid.kind != "value":
            raise DescriptionError(f"{canonical_id} is not a value id")
        node = self.node(cid.op or "")
        if node.layer != cid.layer:
            raise DescriptionError(f"{canonical_id}: layer {cid.layer} != node layer {node.layer}")
        if cid.port is None or cid.port >= node.ports:
            raise DescriptionError(f"{canonical_id}: port out of range")
        if cid.shard is not None and not (node.shards and cid.shard < node.shards):
            raise DescriptionError(f"{canonical_id}: node does not produce shard {cid.shard}")
        if cid.expert is not None and not (node.experts and cid.expert < node.experts):
            raise DescriptionError(
                f"{canonical_id}: node does not produce expert slot {cid.expert}"
            )
        if cid.shard is None and node.shards:
            raise DescriptionError(f"{canonical_id}: node outputs are sharded (s= is required)")
        if cid.expert is None and node.experts:
            raise DescriptionError(
                f"{canonical_id}: node outputs are per expert slot (e= required)"
            )
        assert cid.t is not None and cid.phase is not None
        if (cid.t, cid.phase) not in self.positions_of(node, boundary):
            raise DescriptionError(f"{canonical_id}: node has no output at t={cid.t}/{cid.phase}")
        return NodeInstance(
            node, boundary.request_id, cid.t, cid.phase, cid.port, cid.shard, cid.expert
        )

    # --- argument identities ---

    def resolve_arg(
        self, arg: Arg, inst: NodeInstance, boundary: Boundary, profile_id: str
    ) -> str | list[str] | dict[str, Any]:
        """The canonical id(s) an argument denotes for ``inst`` (a ``const`` returns its value).

        ``token/input`` is the boundary leaf holding the token fed at position ``t``: the
        prompt token at ``t`` for ``t < prompt_len``, else the output token sampled at ``t-1``.
        ``token/sampled`` is the output token sampled at ``t``.
        """
        rid, t = inst.request_id, inst.t
        if arg.source == "const":
            return {"const": arg.value}
        if arg.source == "weight":
            param = arg.param or ""
            if arg.expert == "same":
                if inst.expert is None:
                    raise DescriptionError(f"{inst.node.op}: arg {arg.name} needs an expert slot")
                param = f"{param}.e{inst.expert}"
            elif isinstance(arg.expert, int):
                param = f"{param}.e{arg.expert}"
            elif arg.expert is not None:
                raise DescriptionError(f"{inst.node.op}: arg {arg.name}: expert {arg.expert!r}")
            # a sharded node instance reads its own rank's weight leaf (`w=<param>/s=<rank>`; the
            # per-rank capture records every parameter per rank, replicated ones included) unless
            # the arg names a shard explicitly
            shard = arg.shard if arg.shard is not None else inst.shard
            return str(identity.weight_id(param, shard))
        if arg.source == "boundary":
            return str(identity.boundary_id(profile_id, rid, arg.field or ""))
        if arg.source == "token":
            if arg.which == "sampled":
                return str(identity.boundary_id(profile_id, rid, "output_token", t))
            if t < boundary.prompt_len:
                return str(identity.boundary_id(profile_id, rid, "prompt_token", t))
            return str(identity.boundary_id(profile_id, rid, "output_token", t - 1))
        target = self.node(arg.op or "")
        shard = arg.shard if arg.shard is not None else (inst.shard if target.shards else None)
        if arg.expert == "all":
            if not target.experts:
                raise DescriptionError(f"{inst.node.op}: arg {arg.name}: {target.op} has no slots")
            experts: list[int | None] = list(range(target.experts))
        elif arg.expert == "same" or arg.expert is None:
            experts = [inst.expert if target.experts else None]
            if target.experts and inst.expert is None:
                raise DescriptionError(f"{inst.node.op}: arg {arg.name} needs an expert slot")
        else:
            experts = [int(arg.expert)]

        def at(pos: int, expert: int | None) -> str:
            # capture-format v1.1 (R3): a position served from the prefix cache is bound to the
            # SOURCE request's value at that position (same tokens on 0..pos, checked by the
            # verifier from both boundary trees); it is never a fresh input of this request.
            cached = boundary.cached_source(pos)
            if cached is not None:
                return str(
                    identity.value_id(
                        profile_id, cached.source_request_id, pos,
                        "prefill" if pos < cached.source_prompt_len else "decode",
                        target.layer, target.op, arg.port, shard=shard, expert=expert,
                    )
                )  # fmt: skip
            step = boundary.step_of(pos)
            if step is None:
                raise DescriptionError(f"{inst.node.op}: arg {arg.name} needs position {pos}")
            return str(
                identity.value_id(
                    profile_id, rid, pos, boundary.phase_at(pos), target.layer, target.op, arg.port,
                    shard=shard, expert=expert,
                )
            )  # fmt: skip

        if arg.expert == "all":
            if arg.t != "same":
                raise DescriptionError(f"{inst.node.op}: arg {arg.name}: expert=all needs t=same")
            return [at(t, e) for e in experts]
        expert = experts[0]
        if arg.t == "same":
            return at(t, expert)
        if arg.t == "prev":
            return at(t - 1, expert)
        upper = t + 1 if arg.t == "all_le" else t
        return [at(pos, expert) for pos in range(upper)]

    def resolve_args(
        self, inst: NodeInstance, boundary: Boundary, profile_id: str
    ) -> dict[str, str | list[str] | dict[str, Any]]:
        return {a.name: self.resolve_arg(a, inst, boundary, profile_id) for a in inst.node.args}


def load(doc: Mapping[str, Any]) -> Description:
    return Description(doc)
