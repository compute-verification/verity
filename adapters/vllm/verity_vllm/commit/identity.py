"""Canonical identities (contract section 5): construction, parsing, validation, tree order.

Three id kinds share one syntax -- ``/``-joined ``key=value`` fields in a fixed order, absent
optional fields omitted (never blank), every value non-empty and free of ``/`` and ``=``::

    value     p=<profile_id>/r=<request_id>/t=<pos>/ph=<prefill|decode>/l=<layer>/op=<op>/port=<n>
              [/s=<tp rank>][/e=<expert slot>][/c=<coordinate>][/x=<speculative step>]
    weight    w=<param>[/s=<tp rank>][/c=<row>]
    boundary  p=<profile_id>/r=<request_id>/b=<field>[/t=<pos>]

``c=<row>`` on a weight id (capture format v2) names one row block of a 2-D
parameter in its own ``[N, K]`` layout: the leaf holding rows ``row .. row + rows_per_leaf - 1``.  For a ``torch.nn.Linear`` weight that is the
GEMM's weight *column* ``W[:, j]`` in the ``x @ W.T`` sense.  Without ``c`` the id names the whole
tensor (format v1 leaves, and the range opening of a v2 tensor).

- ``profile_id``: 12 lowercase hex chars (:func:`verity_vllm.commit.hashing.profile_id`).
- ``request_id``: ``[A-Za-z0-9_.:-]+``.
- ``t``, ``port``, ``s``, ``e``: canonical decimal non-negative ints (no leading zeros).
- ``l``: a decimal layer index or one of ``embed``, ``norm``, ``lm_head``, ``sampler``.
- ``op``: ``[A-Za-z0-9_.]+`` (``layers.3.self_attn.qkv_proj.gemm``, ``sampler.select``).
- ``c``: a coordinate, decimal ints joined by ``,`` (``c=17`` or ``c=3,7``).
- ``x`` (capture-format speculative extension): the
  request-local step in which a *speculative* value was computed.  Without ``x`` a value id names the
  committed value at position ``t`` (unique per request, position, op, port — the format's identity
  rule); with ``x=<step>`` it names a value computed at ``t`` during step ``<step>``'s speculation that
  the step did not commit — a rejected verification slot's target value, or the draft model's
  proposal-row value — kept as a first-class leaf of that step's replay block and never confused with
  the committed value at the same position (which a later step produces under the unqualified id).
- ``param``: ``[A-Za-z0-9_.]+`` (the safetensors / ``named_parameters`` name).
- ``field`` (boundary tree leaves; a POC choice, the contract leaves them open): one of
  ``prompt_token`` (with ``t``), ``output_token`` (with ``t``), ``output_presence`` (with ``t``),
  ``sampling_params``, ``seed``, ``finish_reason`` (without ``t``).

**Tree order** is the code-point order of the id string (equal to UTF-8 byte order), i.e.
Python's ``sorted(ids)``; :func:`sort_key` returns that key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PROFILE_ID_RE = re.compile(r"^[0-9a-f]{12}$")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
INT_RE = re.compile(r"^(0|[1-9][0-9]*)$")
OP_RE = re.compile(r"^[A-Za-z0-9_.]+$")
PARAM_RE = re.compile(r"^[A-Za-z0-9_.]+$")
COORD_RE = re.compile(r"^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$")

PHASES = ("prefill", "decode")
NAMED_LAYERS = ("embed", "norm", "lm_head", "sampler")
# ``output_logprob``: the chosen token's logprob as the API reported it, one
# ``float32`` leaf per present output record; a boundary written before the field has no such leaves.
BOUNDARY_FIELDS_WITH_T = ("prompt_token", "output_token", "output_presence", "output_logprob")
BOUNDARY_FIELDS_WITHOUT_T = ("sampling_params", "seed", "finish_reason")
BOUNDARY_FIELDS = BOUNDARY_FIELDS_WITH_T + BOUNDARY_FIELDS_WITHOUT_T

VALUE_ORDER = ("p", "r", "t", "ph", "l", "op", "port", "s", "e", "c", "x")
VALUE_REQUIRED = ("p", "r", "t", "ph", "l", "op", "port")
WEIGHT_ORDER = ("w", "s", "c")
BOUNDARY_ORDER = ("p", "r", "b", "t")

Kind = Literal["value", "weight", "boundary"]


class IdentityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CanonicalId:
    """A parsed identity.  ``str(cid)`` is the canonical string."""

    kind: Kind
    profile_id: str | None = None
    request_id: str | None = None
    t: int | None = None
    phase: str | None = None
    layer: str | None = None
    op: str | None = None
    port: int | None = None
    shard: int | None = None
    expert: int | None = None
    coordinate: str | None = None
    param: str | None = None
    field: str | None = None
    speculative: int | None = None  # ``x=<step>``: speculative (uncommitted) value of that step; value ids only

    def __str__(self) -> str:
        return format_id(self)


# --- construction --------------------------------------------------------------------------------


def value_id(
    profile_id: str,
    request_id: str,
    t: int,
    phase: str,
    layer: str | int,
    op: str,
    port: int = 0,
    *,
    shard: int | None = None,
    expert: int | None = None,
    coordinate: str | tuple[int, ...] | None = None,
    speculative: int | None = None,
) -> CanonicalId:
    if isinstance(coordinate, tuple | list):
        coordinate = ",".join(str(int(c)) for c in coordinate)
    cid = CanonicalId(
        kind="value",
        profile_id=profile_id,
        request_id=request_id,
        t=t,
        phase=phase,
        layer=str(layer),
        op=op,
        port=port,
        shard=shard,
        expert=expert,
        coordinate=coordinate,
        speculative=speculative,
    )
    validate(cid)
    return cid


def weight_id(
    param: str, shard: int | None = None, coordinate: int | str | None = None
) -> CanonicalId:
    if isinstance(coordinate, int) and not isinstance(coordinate, bool):
        coordinate = str(coordinate)
    cid = CanonicalId(kind="weight", param=param, shard=shard, coordinate=coordinate)
    validate(cid)
    return cid


def boundary_id(profile_id: str, request_id: str, field: str, t: int | None = None) -> CanonicalId:
    cid = CanonicalId(
        kind="boundary", profile_id=profile_id, request_id=request_id, field=field, t=t
    )
    validate(cid)
    return cid


# --- validation ----------------------------------------------------------------------------------


def _need_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IdentityError(f"{name} must be a non-negative int, got {value!r}")


def _need_none(name: str, value: object) -> None:
    if value is not None:
        raise IdentityError(f"{name} must be absent for this id kind")


def validate(cid: CanonicalId) -> None:
    if cid.kind == "value":
        _validate_profile(cid.profile_id)
        _validate_request(cid.request_id)
        _need_int("t", cid.t)
        if cid.phase not in PHASES:
            raise IdentityError(f"ph must be one of {PHASES}, got {cid.phase!r}")
        _validate_layer(cid.layer)
        if not isinstance(cid.op, str) or not OP_RE.match(cid.op):
            raise IdentityError(f"op must match {OP_RE.pattern}, got {cid.op!r}")
        _need_int("port", cid.port)
        if cid.shard is not None:
            _need_int("s", cid.shard)
        if cid.expert is not None:
            _need_int("e", cid.expert)
        if cid.coordinate is not None and (
            not isinstance(cid.coordinate, str) or not COORD_RE.match(cid.coordinate)
        ):
            raise IdentityError(f"c must match {COORD_RE.pattern}, got {cid.coordinate!r}")
        if cid.speculative is not None:
            _need_int("x", cid.speculative)
        _need_none("param", cid.param)
        _need_none("field", cid.field)
    elif cid.kind == "weight":
        if not isinstance(cid.param, str) or not PARAM_RE.match(cid.param):
            raise IdentityError(f"param must match {PARAM_RE.pattern}, got {cid.param!r}")
        if cid.shard is not None:
            _need_int("s", cid.shard)
        for name in ("profile_id", "request_id", "t", "phase", "layer", "op", "port", "expert", "speculative"):
            _need_none(name, getattr(cid, name))
        if cid.coordinate is not None and (
            not isinstance(cid.coordinate, str) or not INT_RE.match(cid.coordinate)
        ):
            raise IdentityError(f"weight c must be one canonical row index, got {cid.coordinate!r}")
        _need_none("field", cid.field)
    elif cid.kind == "boundary":
        _validate_profile(cid.profile_id)
        _validate_request(cid.request_id)
        if cid.field not in BOUNDARY_FIELDS:
            raise IdentityError(f"b must be one of {BOUNDARY_FIELDS}, got {cid.field!r}")
        if cid.field in BOUNDARY_FIELDS_WITH_T:
            _need_int("t", cid.t)
        else:
            _need_none("t", cid.t)
        for name in ("phase", "layer", "op", "port", "shard", "expert", "coordinate", "param", "speculative"):
            _need_none(name, getattr(cid, name))
    else:
        raise IdentityError(f"unknown id kind {cid.kind!r}")


def _validate_profile(profile_id: object) -> None:
    if not isinstance(profile_id, str) or not PROFILE_ID_RE.match(profile_id):
        raise IdentityError(f"p must be 12 lowercase hex chars, got {profile_id!r}")


def _validate_request(request_id: object) -> None:
    if not isinstance(request_id, str) or not REQUEST_ID_RE.match(request_id):
        raise IdentityError(f"r must match {REQUEST_ID_RE.pattern}, got {request_id!r}")


def _validate_layer(layer: object) -> None:
    if not isinstance(layer, str) or not (INT_RE.match(layer) or layer in NAMED_LAYERS):
        raise IdentityError(f"l must be a layer index or one of {NAMED_LAYERS}, got {layer!r}")


# --- formatting and parsing ----------------------------------------------------------------------


def format_id(cid: CanonicalId) -> str:
    validate(cid)
    if cid.kind == "value":
        fields = [
            ("p", cid.profile_id),
            ("r", cid.request_id),
            ("t", str(cid.t)),
            ("ph", cid.phase),
            ("l", cid.layer),
            ("op", cid.op),
            ("port", str(cid.port)),
        ]
        if cid.shard is not None:
            fields.append(("s", str(cid.shard)))
        if cid.expert is not None:
            fields.append(("e", str(cid.expert)))
        if cid.coordinate is not None:
            fields.append(("c", cid.coordinate))
        if cid.speculative is not None:
            fields.append(("x", str(cid.speculative)))
    elif cid.kind == "weight":
        fields = [("w", cid.param)]
        if cid.shard is not None:
            fields.append(("s", str(cid.shard)))
        if cid.coordinate is not None:
            fields.append(("c", cid.coordinate))
    else:
        fields = [("p", cid.profile_id), ("r", cid.request_id), ("b", cid.field)]
        if cid.t is not None:
            fields.append(("t", str(cid.t)))
    return "/".join(f"{k}={v}" for k, v in fields)


def _split_fields(text: str) -> list[tuple[str, str]]:
    if not isinstance(text, str) or text == "":
        raise IdentityError("id must be a non-empty string")
    pairs: list[tuple[str, str]] = []
    for piece in text.split("/"):
        if piece.count("=") != 1:
            raise IdentityError(f"field {piece!r} must be exactly key=value")
        key, value = piece.split("=")
        if key == "" or value == "":
            raise IdentityError(f"field {piece!r} has an empty key or value")
        pairs.append((key, value))
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise IdentityError(f"duplicate field in {text!r}")
    return pairs


def _check_order(keys: list[str], order: tuple[str, ...], text: str) -> None:
    positions = []
    for key in keys:
        if key not in order:
            raise IdentityError(f"unexpected field {key!r} in {text!r}")
        positions.append(order.index(key))
    if positions != sorted(positions):
        raise IdentityError(f"fields out of order in {text!r}; order is {'/'.join(order)}")


def _parse_int(name: str, value: str) -> int:
    if not INT_RE.match(value):
        raise IdentityError(f"{name} must be a canonical decimal int, got {value!r}")
    return int(value)


def parse(text: str) -> CanonicalId:
    """Parse any id kind; raise :class:`IdentityError` unless ``text`` is exactly canonical."""
    pairs = _split_fields(text)
    keys = [k for k, _ in pairs]
    fields = dict(pairs)
    if keys[0] == "w":
        _check_order(keys, WEIGHT_ORDER, text)
        cid = CanonicalId(
            kind="weight",
            param=fields["w"],
            shard=_parse_int("s", fields["s"]) if "s" in fields else None,
            coordinate=fields.get("c"),
        )
    elif "b" in fields:
        _check_order(keys, BOUNDARY_ORDER, text)
        for key in ("p", "r", "b"):
            if key not in fields:
                raise IdentityError(f"boundary id {text!r} lacks {key}")
        cid = CanonicalId(
            kind="boundary",
            profile_id=fields["p"],
            request_id=fields["r"],
            field=fields["b"],
            t=_parse_int("t", fields["t"]) if "t" in fields else None,
        )
    else:
        _check_order(keys, VALUE_ORDER, text)
        for key in VALUE_REQUIRED:
            if key not in fields:
                raise IdentityError(f"value id {text!r} lacks required field {key}")
        cid = CanonicalId(
            kind="value",
            profile_id=fields["p"],
            request_id=fields["r"],
            t=_parse_int("t", fields["t"]),
            phase=fields["ph"],
            layer=fields["l"],
            op=fields["op"],
            port=_parse_int("port", fields["port"]),
            shard=_parse_int("s", fields["s"]) if "s" in fields else None,
            expert=_parse_int("e", fields["e"]) if "e" in fields else None,
            coordinate=fields.get("c"),
            speculative=_parse_int("x", fields["x"]) if "x" in fields else None,
        )
    validate(cid)
    if format_id(cid) != text:
        raise IdentityError(f"{text!r} is not in canonical form (expected {format_id(cid)!r})")
    return cid


def kind_of(text: str) -> Kind:
    return parse(text).kind


# --- ordering ------------------------------------------------------------------------------------


def sort_key(canonical_id: str) -> str:
    """The tree order: code-point order of the id string (``sorted(ids)``)."""
    return canonical_id


def tree_sorted(ids: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return sorted(ids, key=sort_key)
