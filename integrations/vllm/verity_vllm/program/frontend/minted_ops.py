"""The custom ops the frontend mints, `verity::<kind>_<hash>(Tensor[] args, str name) -> Tensor[]`: one per fully qualified
call, batch or scan, whose fake implementation supplies shapes only and whose real one refuses to run; and the trace depth
that tells an authored body's shape-only pre-pass from its export."""

from __future__ import annotations

import contextvars
import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils import _pytree as pytree

from verity.ml.circuits.codec import canonical_json, descriptor_id
from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.types import Type
from verity_vllm.program.frontend.correspond import compare_bodies
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.value_types import CARRIER, flat_types, shape_of, tensors_of, unflatten

_LIB = torch.library.Library("verity", "FRAGMENT")


@dataclass
class _OpInfo:
    kind: str  # call | batch | scan
    fn: PrimitiveDefinition | SpecializedDefinition
    arg_types: tuple[Type, ...]  # collection type of each argument as passed (batch: Array<n, P> for mapped)
    axes: tuple
    n: int
    n_xs: int
    out_type: Type


_OPINFO: dict[str, _OpInfo] = {}
_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("verity_trace_depth", default=0)


def _tracing() -> bool:
    return _DEPTH.get() > 0


def _no_semantics(*_a, **_k):
    raise NotImplementedError("Verity ops have no torch execution semantics; export them to a Program")


def _mint(kind: str, fn, arg_types: Sequence[Type], axes: tuple, n: int, n_xs: int, out_type: Type) -> str:
    key = canonical_json([kind, descriptor_id(fn), [t.to_json() for t in arg_types], list(axes), n, n_xs])
    opname = f"{kind}_{hashlib.sha256(key).hexdigest()[:16]}"
    info = _OPINFO.get(opname)
    if info is not None and info.fn is not fn:
        # another definition object already claims this fully qualified specialisation: it must be the same body
        # (§8.1: never first-wins)
        fid = descriptor_id(fn)
        if isinstance(fn, PrimitiveDefinition) or isinstance(info.fn, PrimitiveDefinition):
            if not (isinstance(fn, PrimitiveDefinition) and isinstance(info.fn, PrimitiveDefinition)
                    and fn.params == info.fn.params and fn.ret == info.fn.ret):
                raise FrontendError("E_IDENTITY", f"two different definitions claim {fid}", fid)
        else:
            v = compare_bodies(info.fn.body, fn.body, fid=fid)
            if v.verdict not in ("identical", "same-references"):
                raise FrontendError("E_IDENTITY", f"two different bodies claim {fid} ({v.verdict}: {v.detail}); give one of them a new version", fid)
    if info is None:
        _LIB.define(f"{opname}(Tensor[] args, str name) -> Tensor[]")
        outs = [shape_of(t) for t in flat_types(out_type)]

        def fake(args, name, _outs=outs):
            dev = args[0].device if args else torch.device("meta")
            return [torch.empty(s, dtype=CARRIER[w], device=dev) for s, w in _outs]

        torch.library.register_fake(f"verity::{opname}", fake)
        _LIB.impl(opname, _no_semantics, "CompositeExplicitAutograd")
        _OPINFO[opname] = _OpInfo(kind, fn, tuple(arg_types), tuple(axes), n, n_xs, out_type)
    return opname


def _emit(kind: str, fn, args: Sequence[Any], arg_types: Sequence[Type], out_type: Type, axes=(), n=1, n_xs=0, name: str = ""):
    if not _tracing():  # pre-pass: shapes only
        return tensors_of(out_type, "meta")
    opname = _mint(kind, fn, arg_types, tuple(axes), n, n_xs, out_type)
    flat = [t for a in args for t in pytree.tree_leaves(a)]
    outs = getattr(torch.ops.verity, opname)(flat, name)
    return unflatten(out_type, list(outs))
