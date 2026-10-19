"""served_h1's same-GEMM gate: a served window credits a call's admitted rows only if the call's served y, the tensor its
linear returns to vLLM and the model's next ops read, is the receipt kernel's own output.  One launch computes the call's y,
z and receipts, and no other GEMM runs for that linear.  A call that fails credits nothing (`credit.gated`), and its reason
reads ``credit.same_gemm: linear L step s: ...``.

The gate reads three kinds of evidence, each catching what the others miss:

  order  the challenger's log (`order_refusals`), auditor-trusted only when the challenger runs off the prover's node
         (`trusted-nebius-host`); serve.py runs it in its own process, so that log can refuse a window, not vouch for one.
         Call (s, w)'s receipts take their noise from a round coin.
         The next linear's A rows are computed from (s, w)'s served y, and they are committed in a round root.  If that
         root was recorded before the coin was released, the served y existed before its own noise, and no noised GEMM
         produced it.  It needs a round per call (`ROUND_FORM_CALL`): with a round per step, it refuses every call but the
         step's last.  Seeded coins refuse the window: their salt is fixed before the pass, so no coin follows any y.
         It misses a beside path that keeps the order: a round per call, with y from a second GEMM.
  trace  the hook's launch ledger (`Recorder`, in process, `trace_refusals`).  The profiler records every op and kernel
         in each call's range.  The call must have exactly one GEMM, its declared fused launch, a raw launch in whose range
         only its declared kernels, views and allocations run, each one of the window's pinned fused kernels (none pinned
         refuses every fused launch) and, on a CUDA trace, each seen there.  That launch wrote the returned y (same
         storage, version and rows), and the committed receipts and z are its buffers.  Only views may run after it.  The
         window is refused (`window_refusals`) by a host-only trace, by device work the trace places at no host call (a
         kernel torch attached to no op goes to the launch call with its CUPTI correlation id, `LAUNCH_CALLS`), and by a
         GEMM of a linear's (K, N) outside every call.  It misses a kernel named like a pinned one (CUPTI gives names, not
         code), what the fused kernel itself computes (a GEMM inside it, or a y copied from an input it was handed: the
         kernel's SASS gate), a GEMM written as elementwise ops and a reduction, which reaches y only through such a
         kernel, and a GEMM in another process.
         The prover writes the ledger, so it stops a silent beside path, not a forged ledger.
  tie    drawn units (`tie_refusal`).  The unit's 32 z rows open against the call's committed z table, its cells are the
         unit's reference z, and each served y row is dequant(z) bit for bit (`dequant`, `engine.int8_exact.dequantize`'s
         order).  It ties the committed z to the served value, but an s8 GEMM is exact, so it can't tell which GEMM made y.

    python same_gemm.py --manifest OUT/manifest.json [--ledger OUT/launch-ledger.json] [--out VERDICT.json]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import commit as CM  # noqa: E402

TAG = "credit.same_gemm"
LEDGER = "verity/pouw-served-h1/launch-ledger/v1"
VERDICT = "verity/pouw-served-h1/same-gemm/v1"
CALL_RANGE = "verity/served-h1/call/"
FUSED_RANGE = "verity/served-h1/fused/"
#: the recorder's own reads of the buffers it digests (a device-to-host copy on CUDA); a GEMM in one still counts
READ_RANGE = "verity/served-h1/ledger-read"
ROUND_FORM_STEP = "step"
ROUND_FORM_CALL = "call"

#: the dispatcher's GEMM-class ops; one counts only when no op under it is GEMM-class (aten::matmul runs aten::mm)
GEMM_OPS = re.compile(r"^aten::(_int_mm|mm|matmul|bmm|addmm|addmv|mv|dot|vdot|baddbmm|addbmm|_scaled_mm\w*|_scaled_grouped_mm|"
                      r"_grouped_mm|linear|einsum|tensordot|_addmm_activation|conv\w*|_convolution\w*|chain_matmul|"
                      r"linalg_multi_dot|_weight_int\d+pack_mm\w*)$")
#: device kernels that are GEMMs (cuBLAS, cuBLASLt, CUTLASS and their tensor-core tilings), wherever they were launched from
GEMM_KERNELS = re.compile(r"(?i)(gemm|gemv|imma|hmma|wgmma|cutlass|cublas|xmma|nvjet|splitk|tensorop)")
#: the runtime and driver calls that put work on a device (cudaLaunchKernel, cuLaunchKernelEx, cudaGraphLaunch,
#: cudaMemcpyAsync, cuMemsetD32, ..): device work torch attached to no op is placed at the one whose CUPTI correlation id
#: it carries
LAUNCH_CALLS = re.compile(r"^cu(da)?(\w*Launch|Memcpy|Memset)\w*$")
#: the device activities the ledger places (kineto's names; None where the trace names none)
DEVICE_WORK = (None, "kernel", "gpu_memcpy", "gpu_memset")
#: ops that make views or fresh allocations, so write no existing buffer: the only ops that may run in a call after its
#: fused launch.  A write is a childless dispatcher op (``ns::name``; `aten::to` with no `_to_copy` under it returned its
#: input) not in this set, or an op with one under it; the runtime's own events (``cudaLaunchKernel``, autograd's
#: ``detach``) show as their kernels.
VIEW_OPS = frozenset(f"aten::{n}" for n in (
    "view", "reshape", "_reshape_alias", "_unsafe_view", "as_strided", "slice", "select", "narrow", "t", "transpose",
    "permute", "alias", "detach", "unsqueeze", "squeeze", "expand", "expand_as", "view_as", "flatten", "unflatten",
    "split", "chunk", "unbind", "lift_fresh", "contiguous", "_local_scalar_dense", "item", "empty", "empty_strided",
    "empty_like", "new_empty", "to", "resolve_conj", "resolve_neg"))


def _hex(b: bytes) -> str:
    return bytes(b).hex()


# -- the hook's launch ledger ------------------------------------------------------------------------------------------

def tensor_facts(t: Any, rows: int | None = None) -> dict[str, Any]:
    """What the ledger keeps of a tensor: its storage and data pointers, bytes, version counter and, with `rows`, the SHA-512
    of each of its first `rows` rows' bytes (bf16 y rows as their u16 words), read inside `READ_RANGE`."""
    import torch
    out = {"storage": int(t.untyped_storage().data_ptr()), "ptr": int(t.data_ptr()), "nbytes": int(t.numel() * t.element_size()),
           "version": int(t._version), "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
    if rows is not None:
        with torch.profiler.record_function(READ_RANGE):
            out["rows"] = row_digests(t.detach()[:rows].reshape(rows, -1).cpu())
    return out


def row_digests(t: Any) -> list[str]:
    import torch
    t = t.contiguous()
    raw = t.view(torch.int16 if t.element_size() == 2 else torch.int32 if t.element_size() == 4 else torch.uint8).numpy()
    return [hashlib.sha512(raw[i].tobytes()).hexdigest() for i in range(raw.shape[0])]


class _Fused:
    def __init__(self) -> None:
        self.facts: dict[str, Any] = {}

    def outs(self, rows: int | None = None, **tensors: Any) -> None:
        """The launch's output buffers (y, z, receipts), as it leaves them; y's first `rows` rows' digests too."""
        self.facts = {k: tensor_facts(v, rows if k == "y" else None) for k, v in tensors.items()}


class Recorder:
    """The hook's launch ledger over a served pass: a torch profiler (CPU ops, and CUDA kernels on a device) with one range
    per linear call (`call`) and one per fused launch inside it (`fused`); the hook adds the tensor it returned to vLLM
    (`served`) and the buffers the window committed (`committed`).  `ledger` turns it into the gate's input."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.meta: dict[int, dict[str, Any]] = {}
        self.prof = None

    def __enter__(self) -> Recorder:
        import torch
        acts = [torch.profiler.ProfilerActivity.CPU]
        if self.device == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(activities=acts, record_shapes=True)
        self.prof.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self.prof.__exit__(*exc)

    def _m(self, i: int) -> dict[str, Any]:
        return self.meta.setdefault(int(i), {"fused": None, "served": None, "committed": {}, "key": None})

    @contextlib.contextmanager
    def call(self, i: int, key: Any = None) -> Iterator[int]:
        """Call `i`'s range: everything its linear runs, from its input to the tensor it returns."""
        import torch
        self._m(i)["key"] = key
        with torch.profiler.record_function(f"{CALL_RANGE}{int(i)}"):
            yield i

    @contextlib.contextmanager
    def fused(self, i: int, kernels: Sequence[str] = ()) -> Iterator[_Fused]:
        """Call `i`'s fused launch, the one that makes its y, z and receipts: a raw kernel launch (its declared `kernels`)
        that writes buffers it names with `outs` before the range closes."""
        import torch
        f = _Fused()
        with torch.profiler.record_function(f"{FUSED_RANGE}{int(i)}"):
            yield f
        self._m(i)["fused"] = {"kernels": list(kernels), "outs": f.facts}

    def served(self, i: int, y: Any, rows: int) -> None:
        """The tensor call `i` returned to vLLM, and its first `rows` (the served rows') digests."""
        self._m(i)["served"] = tensor_facts(y, rows)

    def committed(self, i: int, kind: str, t: Any) -> None:
        """The buffer the window committed as call `i`'s `kind` (receipts, z); a numpy array is not a device buffer."""
        import torch
        self._m(i)["committed"][kind] = tensor_facts(t) if isinstance(t, torch.Tensor) else {"host": type(t).__name__}

    def ledger(self) -> dict[str, Any]:
        return ledger_of(self.prof.events(), self.meta, self.device)


def _kn(name: str, shapes: Sequence[Sequence[int]]) -> list[int] | None:
    """A GEMM op's (K, N) from its input shapes, for the forms with one: x [.., K] by w [K, N] (or linear's w [N, K])."""
    s = [list(x) for x in shapes if isinstance(x, (list, tuple)) and len(x) >= 1]
    op = name.removeprefix("aten::")
    if op == "linear" and len(s) >= 2 and len(s[1]) == 2:
        return [s[0][-1], s[1][0]]
    if op in ("addmm", "baddbmm", "addbmm", "_addmm_activation") and len(s) >= 3 and len(s[1]) >= 2 and len(s[2]) >= 2:
        return [s[1][-1], s[2][-1]]
    if op in ("_int_mm", "mm", "matmul", "bmm", "_scaled_mm") and len(s) >= 2 and len(s[0]) >= 2 and len(s[1]) >= 2:
        return [s[0][-1], s[1][-1]]
    return None


def _descendants(e: Any) -> Iterator[Any]:
    for c in e.cpu_children:
        yield c
        yield from _descendants(c)


def _on_host(e: Any) -> bool:
    """A host event (an op, a range, a runtime call): CUPTI also reports each range as a device-side annotation of the same
    name, with no children, which is not the range."""
    return str(getattr(e, "device_type", "DeviceType.CPU")).endswith("CPU")


def ledger_of(events: Sequence[Any], meta: Mapping[int, Mapping[str, Any]], device: str) -> dict[str, Any]:
    """The ledger from profiler events (torch's FunctionEvents, their `cpu_children` trees and attached `kernels`).

    A GEMM launch is a GEMM-class op none of whose descendants is one, or a GEMM-class kernel no such op launched.  They are
    numbered #0, #1, .. in the pass's order.  Each call's events are those in its range, in order, each flagged ``fused``
    (inside its fused launch's range) or ``after_fused`` (after that range closed); of the recorder's reads (`READ_RANGE`)
    only GEMM launches."""
    evs = sorted(filter(_on_host, events), key=lambda e: (e.time_range.start, -e.time_range.end))
    extra, unattributed = _orphans(events, evs)

    def gemm_op(e) -> bool:
        return bool(GEMM_OPS.match(e.name)) and not any(GEMM_OPS.match(d.name) for d in _descendants(e))

    under_gemm: set[int] = set()
    for e in evs:
        if gemm_op(e):
            under_gemm.update(id(d) for d in _descendants(e))
            under_gemm.add(id(e))
    def kernels_of(e) -> list[str]:
        return [k.name for k in (getattr(e, "kernels", ()) or ()) if not k.name.startswith("verity/")] + extra.get(id(e), [])

    launches: list[tuple[Any, str, str | None]] = []
    for e in evs:
        if gemm_op(e):
            launches.append((e, "op", None))
        if id(e) not in under_gemm:
            for k in kernels_of(e):
                if GEMM_KERNELS.search(k):
                    launches.append((e, "kernel", k))
    seq = {(id(e), kind, k): n for n, (e, kind, k) in enumerate(launches)}

    def writes(e) -> bool:
        if e.cpu_children:
            return any(writes(c) for c in e.cpu_children)
        return "::" in e.name and e.name not in VIEW_OPS

    def entry(e, fr, members: set[int], kind: str = "op", kname: str | None = None) -> dict[str, Any]:
        inside = fr is not None and (e is fr or id(e) in members)
        after = fr is not None and not inside and e.time_range.start >= fr.time_range.end
        g = seq.get((id(e), kind, kname))
        d = {"name": kname or e.name, "kind": kind, "gemm": g is not None, "seq": g, "fused": inside, "after_fused": after}
        if kind == "kernel":
            d["kernel_of"] = e.name
        if kind == "op" and g is not None:
            d["kn"] = _kn(e.name, e.input_shapes or [])
            d["kernels"] = sorted({k for x in [e, *_descendants(e)] for k in kernels_of(x)})
        if kind == "op" and "::" in e.name and writes(e):
            d["writes"] = True
        return d

    reads = {id(x) for e in evs if e.name == READ_RANGE for x in (e, *_descendants(e))}
    ranges = {int(e.name[len(CALL_RANGE):]): e for e in evs if e.name.startswith(CALL_RANGE)}
    fused_of = {int(e.name[len(FUSED_RANGE):]): e for e in evs if e.name.startswith(FUSED_RANGE)}
    inside_any: set[int] = set()
    calls = []
    for i, r in sorted(ranges.items()):
        fr = fused_of.get(i)
        members = {id(d) for d in _descendants(fr)} if fr is not None else set()
        body = [r, *_descendants(r)]
        inside_any.update(id(d) for d in body)
        rows = []
        for e in body:
            read = id(e) in reads
            if e is not r and e is not fr and (not read or (id(e), "op", None) in seq):
                rows.append(entry(e, fr, members))
            rows += [entry(e, fr, members, "kernel", k) for k in kernels_of(e) if not read or (id(e), "kernel", k) in seq]
        m = meta.get(i, {})
        calls.append({"call": i, "key": m.get("key"), "events": rows, "fused": m.get("fused"), "served": m.get("served"),
                      "committed": m.get("committed", {})})
    outside = []
    for e, kind, k in launches:
        if id(e) in inside_any or e.name.startswith(CALL_RANGE):
            continue
        d = {"seq": seq[(id(e), kind, k)], "name": k or e.name, "kind": kind}
        if kind == "op":
            d["kn"] = _kn(e.name, e.input_shapes or [])
        outside.append(d)
    return {"schema": LEDGER, "device": device, "gemm_launches": len(launches), "calls": calls, "outside": outside,
            "unattributed": [{"name": k, "count": n, "gemm": bool(GEMM_KERNELS.search(k))}
                             for k, n in sorted(Counter(unattributed).items())],
            "trace": "torch.profiler: dispatcher ops" + (" and CUDA kernels (CUPTI)" if device == "cuda" else ""),
            "gemm_rule": {"ops": GEMM_OPS.pattern, "kernels": GEMM_KERNELS.pattern}}


def _orphans(events: Sequence[Any], host: Sequence[Any]) -> tuple[dict[int, list[str]], list[str]]:
    """Device kernels torch attached to no op.  Its `kernels` link a kernel only to the synchronous host op whose id is the
    kernel's ``linked_correlation_id``, and a raw driver-API launch (`receipts.Cubin`) in a bare range carries none.  Each
    such kernel, copy or set (`DEVICE_WORK`) goes to the host call (`LAUNCH_CALLS`) with its CUPTI correlation id, or else
    its flow id: ({id(host call): [names]}, names of the device work no host call put there)."""
    ops = {e.id for e in host if not getattr(e, "linked_correlation_id", 0) and not getattr(e, "is_async", False)
           and getattr(e, "activity_type", None) != "overhead"}
    calls: dict[tuple[str, int], Any] = {}
    for e in host:
        if LAUNCH_CALLS.match(e.name):
            calls.setdefault(("id", e.id), e)
            if getattr(e, "flow_id", 0):
                calls.setdefault(("flow", e.flow_id), e)
    extra: dict[int, list[str]] = {}
    lost: list[str] = []
    for d in events:
        if (_on_host(d) or getattr(d, "is_user_annotation", False) or getattr(d, "activity_type", None) not in DEVICE_WORK
                or d.name.startswith("verity/") or (getattr(d, "linked_correlation_id", 0) or None) in ops):
            continue
        h = calls.get(("id", d.id)) or (calls.get(("flow", d.flow_id)) if getattr(d, "flow_id", 0) else None)
        if h is None:
            lost.append(d.name)
        else:
            extra.setdefault(id(h), []).append(d.name)
    return extra, lost


# -- the checks ---------------------------------------------------------------------------------------------------------

def _who(c: Mapping[str, Any]) -> str:
    m = f" ({c['module']})" if c.get("module") else ""
    return f"linear {c['linear']}{m} step {c['step']}"


def _why(c: Mapping[str, Any], text: str) -> str:
    return f"{TAG}: {_who(c)}: {text}"


def _same_buffer(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    return (a is not None and b is not None and "storage" in a and "storage" in b
            and (a["storage"], a["ptr"], a["nbytes"], a["version"]) == (b["storage"], b["ptr"], b["nbytes"], b["version"]))


def call_refusal(c: Mapping[str, Any], e: Mapping[str, Any] | None, pinned: Sequence[str] = (),
                 device: str = "cuda") -> str | None:
    """Why call `c`'s ledger entry `e` doesn't show its served y as its receipt kernel's own output; None when it does.
    `pinned`: the window's fused kernels, the only ones a fused launch may declare.  On a `device` trace ("cuda") each
    declared kernel must be seen in the fused range; a host trace sees no kernel."""
    if e is None:
        return _why(c, f"the record has no launch ledger entry ({LEDGER}): what produced the served y is not recorded")
    evs = e["events"]
    gemms = [x for x in evs if x["gemm"] and not x["fused"]]
    f = e.get("fused")
    if f is None:
        if not gemms:
            return _why(c, "no GEMM launch and no fused launch in the call's range: the served y's producer is not in the "
                           "trace")
        g = gemms[0]
        kern = f", kernels {', '.join(g['kernels'])}" if g.get("kernels") else ""
        more = f" (and {len(gemms) - 1} more GEMM launches)" if len(gemms) > 1 else ""
        return _why(c, f"served y from {g['name']} call #{g['seq']}{kern}{more}, receipts from a separate pass")
    unpinned = [k for k in f["kernels"] if k not in pinned]
    if unpinned or not f["kernels"]:
        return _why(c, f"the fused launch declares kernel {unpinned[0] if unpinned else 'none'}, not one of the window's "
                       f"pinned fused kernels ({', '.join(pinned) or 'none pinned'})")
    inner = [x for x in evs if x["fused"] and x["gemm"] and not (x["kind"] == "kernel" and x["name"] in f["kernels"])]
    if inner:
        return _why(c, f"{inner[0]['name']} call #{inner[0]['seq']} inside the fused launch: the fused kernel is one raw "
                       "launch, not a dispatcher or library GEMM")
    seen = sorted({x["name"] for x in evs if x["fused"] and x["kind"] == "kernel"})
    alien = [k for k in seen if k not in f["kernels"]]
    if alien:
        return _why(c, f"kernel {alien[0]} inside the fused launch is not one it declares ({', '.join(f['kernels']) or 'none'})")
    unseen = [k for k in f["kernels"] if k not in seen] if device == "cuda" else []
    if unseen:
        return _why(c, f"the fused launch's kernel {unseen[0]} never ran in its range: the device trace has no such kernel "
                       "there")
    busy = [x for x in evs if x["fused"] and x["kind"] == "op" and x.get("writes")]
    if busy:
        return _why(c, f"{busy[0]['name']} ran inside the fused launch: it is one raw launch, in which only its declared "
                       "kernels, views and allocations run")
    if gemms:
        g = gemms[0]
        return _why(c, f"a second GEMM in the call: {g['name']} call #{g['seq']} beside the fused launch")
    late = [x for x in evs if x["after_fused"] and (x["kind"] == "kernel" or x.get("writes"))]
    if late:
        return _why(c, f"{late[0]['name']} ran after the fused launch, before y was returned: only views may")
    y, s = f["outs"].get("y"), e.get("served")
    if s is None or y is None:
        return _why(c, "the hook recorded no served y" if s is None else "the fused launch names no y buffer")
    inside = s["storage"] == y["storage"] and y["ptr"] <= s["ptr"] and s["ptr"] + s["nbytes"] <= y["ptr"] + y["nbytes"]
    if not inside:
        return _why(c, f"served y (ptr {s['ptr']:#x}, {s['nbytes']} bytes) is not the fused launch's y buffer "
                       f"(ptr {y['ptr']:#x}, {y['nbytes']} bytes)")
    if s["version"] != y["version"]:
        return _why(c, f"served y was written after the fused launch (version {y['version']} -> {s['version']})")
    if s.get("rows") != y.get("rows"):
        return _why(c, "served y's rows are not the rows the fused launch wrote")
    for kind in ("receipts", "z"):
        if kind in f["outs"] and not _same_buffer(e.get("committed", {}).get(kind), f["outs"][kind]):
            return _why(c, f"the committed {kind} are not the fused launch's {kind} buffer")
    if "receipts" not in f["outs"]:
        return _why(c, "the fused launch names no receipts buffer: receipts from a separate pass")
    return None


def trace_refusals(calls: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any] | None, keys: Sequence[Any],
                   pinned: Sequence[str] = ()) -> dict[int, str]:
    """Each call's refusal from the ledger (`call_refusal`); call i's entry is the one whose key is ``keys[i]``."""
    by_key = {json.dumps(e.get("key")): e for e in (ledger or {}).get("calls", [])}
    device = (ledger or {}).get("device", "cuda")
    out = {}
    for i, c in enumerate(calls):
        why = call_refusal(c, by_key.get(json.dumps(keys[i])) if ledger else None, pinned, device)
        if why:
            out[i] = why
    return out


def window_refusals(ledger: Mapping[str, Any] | None, shapes: Sequence[tuple[int, int]], host_ok: bool = False) -> list[str]:
    """The ledger's own refusals: a host-only trace (unless `host_ok`, a window served on the host), device work the trace
    can't place in or out of a call, and GEMM launches outside every call whose (K, N) is a credited linear's."""
    if not ledger:
        return []
    out = []
    if ledger.get("device") != "cuda" and not host_ok:
        out.append(f"{TAG}: window: the launch ledger is a {ledger.get('device')} trace, which sees no device kernel, so a "
                   "raw-launched GEMM is invisible to it; credit needs a CUDA trace")
    out += [f"{TAG}: window: {x['name']} ({x['count']} launches{', a GEMM' if x['gemm'] else ''}) ran with no host call "
            "the trace places in or out of a call" for x in ledger.get("unattributed", [])]
    kn = {tuple(x) for x in shapes}
    return out + [f"{TAG}: window: {x['name']} call #{x['seq']} with a linear's (K, N) = {tuple(x['kn'])} ran outside every "
                  "call's range" for x in ledger.get("outside", []) if x.get("kn") and tuple(x["kn"]) in kn]


# -- order: the challenger's log ----------------------------------------------------------------------------------------

def tree_root(node: Any) -> bytes:
    """A round's tree, a leaf (hex) or {"kind", "fields", "leaves"}: `commit.step_root` over its leaves' roots under
    ``commit.domain(fields, len(leaves), kind)``."""
    if isinstance(node, str):
        return bytes.fromhex(node)
    leaves = [tree_root(x) for x in node["leaves"]]
    return CM.step_root(CM.domain(node["fields"], len(leaves), node["kind"]), leaves)


def tree_leaves(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
        return
    for x in node["leaves"]:
        yield from tree_leaves(x)


def step_trees(window: str, calls: Sequence[Mapping[str, Any]], rounds: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Each round's tree in the round-per-step form (`serve.step_roots`): [the step's A root over its calls' A-table roots in
    linear order ('a-step'), its admit root]."""
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for c in calls:
        by_step.setdefault(int(c["step"]), []).append(c)
    out = []
    for r in rounds:
        cs = sorted(by_step.get(int(r["round"]), []), key=lambda c: int(c["linear"]))
        a = {"kind": "a-step", "fields": {"window": window, "step": int(r["round"]), "linears": len(cs)},
             "leaves": [c["root_a"] for c in cs]}
        out.append({"kind": "round", "fields": r["fields"], "leaves": [a, r["admit_step"]]})
    return out


def successors(calls: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """The public dataflow: call (s, w)'s served y feeds call (s, w + 1)'s A rows, and the step's last linear (through the
    sampled token) feeds (s + 1, 0)'s."""
    at = {(int(c["step"]), int(c["linear"])): i for i, c in enumerate(calls)}
    width = 1 + max(int(c["linear"]) for c in calls)
    out = {}
    for (s, w), i in at.items():
        j = at.get((s, w + 1) if w + 1 < width else (s + 1, 0))
        if j is not None:
            out[i] = j
    return out


def order_refusals(window: str, calls: Sequence[Mapping[str, Any]], coins: Mapping[str, Any] | None) -> tuple[dict[int, str], list[str]]:
    """(per call, window) refusals from the live round log: each round's tree recomputes to its recorded root; the coins are
    the live epoch's, each opened after its round's root, and the challenger's release log has root r, then coin r, then
    root r + 1; then call i's receipts' salt is round rs's key, its A rows are in round ra <= rs, and its successor's A
    rows, and its own z table when it has one, are in a round after rs."""
    from verity.core.protocols.pouw import window as Wd
    from verity.core.protocols.sampled_proofs import challenger as Ch
    mode = (coins or {}).get("mode")
    if mode != "live":
        return {}, [f"{TAG}: window: coins are {mode or 'missing'}, not live: a seeded salt is fixed before the served "
                    "pass, so no coin follows the served y and the receipts certify nothing about it; credit needs live "
                    "rounds"]
    if not coins.get("rounds"):
        return {}, [f"{TAG}: window: live coins with no round log: the challenger's log can't order the receipts' noise "
                    "against the served y"]
    rounds = list(coins["rounds"])
    form = coins.get("round_form") or (ROUND_FORM_STEP if str(coins.get("rule", "")).startswith("round s = step s")
                                       else None)
    if form == ROUND_FORM_STEP:
        trees = step_trees(window, calls, rounds)
    elif all("tree" in r for r in rounds):
        trees = [r["tree"] for r in rounds]
    else:
        return {}, [f"{TAG}: window: the round log names no round form and carries no round trees"]
    roots = [bytes.fromhex(r["root"]) for r in rounds]
    bad = [r["round"] for r, t, root in zip(rounds, trees, roots) if tree_root(t) != root]
    if bad:
        return {}, [f"{TAG}: window: round {bad[0]}'s tree does not recompute to its recorded root"]
    commitment = Ch.LiveCommitment.from_json(coins["commitment"])
    openings = [Ch.LiveOpening.from_json(r["opening"]) for r in rounds]
    why = Wd.rounds_reason(commitment, roots, openings)
    if why:
        return {}, [f"{TAG}: window: the round coins: {why}"]
    log = coins.get("release_log")
    if not log:
        return {}, [f"{TAG}: window: the record has no challenger release log"]
    pos = {(x["event"], int(x["index"])): n for n, x in enumerate(log) if x.get("label") == Wd.ROUND}
    for r in range(len(rounds)):
        a, b, nxt = pos.get(("root", r)), pos.get(("release", r)), pos.get(("root", r + 1))
        if a is None or b is None or not a < b or (r + 1 < len(rounds) and (nxt is None or not b < nxt)):
            return {}, [f"{TAG}: window: the release log does not have round {r}'s root, then its coin, then round {r + 1}'s root"]
    key_round = {Wd.round_key(commitment, o.coin, int(r["round"]), root).stream(0, 64).hex(): int(r["round"])
                 for r, o, root in zip(rounds, openings, roots)}
    leaf_round: dict[str, int] = {}
    for r, t in zip(rounds, trees):
        for x in tree_leaves(t):
            leaf_round.setdefault(x, int(r["round"]))
    succ = successors(calls)
    out = {}
    for i, c in enumerate(calls):
        rs, ra = key_round.get(str(c.get("salt", ""))), leaf_round.get(c["root_a"])
        if rs is None:
            out[i] = _why(c, "its receipts' salt is no round's key")
        elif ra is None:
            out[i] = _why(c, "its A table's root is in no round's root")
        elif rs < ra:
            out[i] = _why(c, f"its receipts' noise is round {rs}'s coin, opened before its A rows were committed (round {ra})")
        elif c.get("root_z") and leaf_round.get(c["root_z"], -1) <= rs:
            rz = leaf_round.get(c["root_z"])
            out[i] = _why(c, f"its z table is committed in round {rz}'s root, not after round {rs}'s coin, which seeds its "
                             "noise" if rz is not None else "its z table's root is in no round's root")
        elif i in succ:
            n = calls[succ[i]]
            rn = leaf_round.get(n["root_a"])
            if rn is None or rn <= rs:
                out[i] = _why(c, f"served y existed before its receipts' noise: {_who(n)}'s A rows, computed from it, are "
                                 f"in round {rn}'s root, recorded before round {rs}'s coin, which seeds this call's receipts, "
                                 "so no noised GEMM made that y")
    return out, []


# -- tie: drawn units against the committed z table ---------------------------------------------------------------------

def dequant(z: np.ndarray, s_a: np.ndarray, s_w: np.ndarray) -> np.ndarray:
    """RN_bf16(RN32(RN32(RN32(z) * s_a[i]) * s_w[n])) as u16 words (`engine.int8_exact.dequantize`)."""
    import torch
    y = (torch.from_numpy(np.asarray(z, dtype=np.int32)).to(torch.float32)
         * torch.from_numpy(np.asarray(s_a, dtype=np.float32).reshape(-1))[:, None]
         * torch.from_numpy(np.asarray(s_w, dtype=np.float32).reshape(-1))[None, :]).to(torch.bfloat16)
    return y.view(torch.int16).numpy()


def z_rows(z: np.ndarray) -> list[tuple[bytes, int]]:
    """A call's z table rows (AROW's row-table format, `row_table.commit`): row i its N s32 cells little-endian."""
    z = np.ascontiguousarray(z, dtype="<i4")
    return [(z[i].tobytes(), 32 * z.shape[1]) for i in range(z.shape[0])]


def z_table(dom, key: bytes, label: str, z: np.ndarray):
    """Call (s, w)'s z table: `z_rows` under ``commit.salt(key, label, i)``, domain kind 'z-table'."""
    return CM.RT.commit(dom, z_rows(z), [CM.salt(key, label, i) for i in range(z.shape[0])])


def tie_refusal(c: Mapping[str, Any], u: int, opening, z_rows_: Mapping[int, bytes], key: bytes, label: str, dom_id: bytes,
                slots: int, n: int, ref_z: Sequence[int], s_a: np.ndarray, s_w: np.ndarray, served_rows: Sequence[str]) -> str | None:
    """Drawn unit u = (row block ri, column block cj) of call c: its 32 z rows open against c's z table root, its 32 x 32
    cells are the unit's reference z (u32, row-major), and each of its served rows (< m) digests to dequant of its z row."""
    ri, cj = divmod(u, n // 32)
    ranks = range(32 * ri, 32 * ri + 32)
    rows = {r: (z_rows_[r], 32 * n, CM.salt(key, label, r)) for r in ranks}
    if not CM.RT.verify_rows(bytes.fromhex(c["root_z"]), dom_id, slots, opening, rows):
        return _why(c, f"unit {u}'s z rows don't open against its z table")
    z = np.stack([np.frombuffer(z_rows_[r], dtype="<i4") for r in ranks])
    if [int(v) & 0xFFFFFFFF for v in z[:, 32 * cj:32 * cj + 32].reshape(-1)] != list(ref_z):
        return _why(c, f"unit {u}'s committed z is not its reference z")
    m = len(served_rows)
    live = [r for r in ranks if r < m]
    if live:
        y = dequant(z[[r - 32 * ri for r in live]], np.asarray(s_a)[live], s_w)
        if [hashlib.sha512(y[k].tobytes()).hexdigest() for k in range(len(live))] != [served_rows[r] for r in live]:
            return _why(c, f"unit {u}'s served y rows are not dequant of its committed z")
    if any(z[r - 32 * ri].any() for r in ranks if r >= m):
        return _why(c, f"unit {u}'s z is not zero on rows past the served ones")
    return None


# -- the verdict --------------------------------------------------------------------------------------------------------

def verdict(window: str, calls: Sequence[Mapping[str, Any]], counts: Sequence[int], coins: Mapping[str, Any] | None,
            ledger: Mapping[str, Any] | None, keys: Sequence[Any] | None = None, tie: Sequence[str] = (),
            pinned: Sequence[str] = (), host_ok: bool = False) -> dict[str, Any]:
    """The gate over a window: per call, the order and trace refusals; the window's own (order, the ledger's
    `window_refusals`, and `tie`, the drawn units' tie refusals).  `pinned` names the window's fused kernels; `host_ok`
    takes a host-only ledger (a window served on the host).  A refused call credits nothing; a window refusal refuses
    every call.  Only calls with admitted rows are reported."""
    order, w_order = order_refusals(window, calls, coins)
    trace = trace_refusals(calls, ledger, list(keys) if keys is not None else [None] * len(calls), pinned)
    shapes = sorted({(int(c["k"]), int(c["n"])) for c in calls})
    win = [*w_order, *window_refusals(ledger, shapes, host_ok), *tie]
    credited = [i for i, k in enumerate(counts) if k]
    refused = {i: [x for x in (order.get(i), trace.get(i)) if x] for i in credited}
    refused = {i: r for i, r in refused.items() if r or win}
    gated = [0 if (win or i in refused) else int(k) for i, k in enumerate(counts)]
    first = (win or [r[0] for i, r in sorted(refused.items()) if r] or [None])[0]
    return {"schema": VERDICT, "ok": not win and not refused, "window": win, "first": first,
            "calls_with_rows": len(credited), "refused_calls": len(refused) if not win else len(credited),
            "by_check": {"order": sum(1 for i in credited if i in order), "trace": sum(1 for i in credited if i in trace)},
            "refused": [{"call": i, "step": calls[i]["step"], "linear": calls[i]["linear"], "reasons": r}
                        for i, r in sorted(refused.items()) if r],
            "gated_counts": gated, "ledger": None if ledger is None else ledger.get("schema")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="a served_h1 window's manifest.json")
    ap.add_argument("--ledger", help="its launch ledger (launch-ledger.json), when it has one")
    ap.add_argument("--out", help="write the verdict here")
    a = ap.parse_args(argv)
    m = json.loads(Path(a.manifest).read_text())
    ledger = json.loads(Path(a.ledger).read_text()) if a.ledger else None
    keys = [c.get("tap") for c in m["calls"]]
    v = verdict(m["window"], m["calls"], [int(c.get("selected", 0)) for c in m["calls"]], m.get("coins"), ledger, keys,
                pinned=m.get("fused_kernels", ()))
    v.pop("gated_counts")
    v["record"] = {"manifest": a.manifest, "window": m["window"], "calls": len(m["calls"]), "framing": m.get("framing"),
                   "fused_kernels": m.get("fused_kernels"),
                   "coins": (m.get("coins") or {}).get("mode"), "rounds": len((m.get("coins") or {}).get("rounds", [])),
                   "ledger": a.ledger}
    text = json.dumps(v, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(json.dumps({k: v[k] for k in ("ok", "first", "calls_with_rows", "refused_calls", "by_check", "window")}, indent=1))
    return 0 if v["ok"] else 1


__all__ = ["CALL_RANGE", "DEVICE_WORK", "FUSED_RANGE", "GEMM_KERNELS", "GEMM_OPS", "LAUNCH_CALLS", "LEDGER", "READ_RANGE",
           "ROUND_FORM_CALL", "ROUND_FORM_STEP", "Recorder",
           "TAG", "VERDICT", "call_refusal", "dequant", "ledger_of", "order_refusals", "row_digests", "step_trees",
           "successors", "tensor_facts", "tie_refusal", "trace_refusals", "tree_leaves", "tree_root", "verdict",
           "window_refusals", "z_rows", "z_table"]

if __name__ == "__main__":
    sys.exit(main())
