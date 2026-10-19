"""The request components GP-01 composes (`global_program`): a derived dir's descriptor, its wrapper parameters and its provenance, and
the component decoded from them -- one at a time, or each as its derive finishes while the others still run (`follow_components`)."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import REGISTRY
from verity_vllm.check.match.batch_decomp import derived_shape
from verity_vllm.program import workload as W


def load_desc(d: str) -> dict:
    with gzip.open(os.path.join(d, "descriptor.json.gz"), "rb") as f:
        return json.loads(f.read())


def _params_before_rows(path: str) -> list | None:
    """`params` of an instance sequence that derive_step wrote with `params` before `rows`, read without the rows (a long context's
    rows are the file); None when the head does not read that way."""
    dec, head = json.JSONDecoder(), ""
    with gzip.open(path, "rt") as f:
        while '"rows":' not in head:
            chunk = f.read(1 << 16)
            if not chunk:
                return None
            head += chunk
    i = head.find('"params": ')
    if i < 0 or i > head.find('"rows":'):
        return None
    ps, _end = dec.raw_decode(head, i + len('"params": '))
    return ps if isinstance(ps, list) else None


def wrapper_params(d: str) -> list[tuple[str, str]]:
    """The request wrapper's own parameters [(name, type repr)] (derive_step instances.json.gz `params`): needed to group the
    Input batches (one per weight field) back onto `weights` so it can be shared."""
    p = os.path.join(d, "instances.json.gz")
    if not os.path.exists(p):
        raise ValueError(f"{d}: no instances.json.gz (the wrapper parameters are needed to identify the shared weights)")
    ps = _params_before_rows(p)
    if ps is None:
        with gzip.open(p, "rt") as f:
            ps = json.load(f).get("params")
    if not ps:
        raise ValueError(f"{d}: instances.json.gz has no params")
    return [(n, t) for n, t in ps]


def recorded_component_digest(d: str) -> str | None:
    """The program digest derive_step RECORDED for the component dir (result.json `program.digest`, else artifact.json `program_digest`)."""
    p = os.path.join(d, "result.json")
    if os.path.exists(p):
        r = json.load(open(p))
        dg = (r.get("program") or {}).get("digest")
        if dg:
            return str(dg)
    p = os.path.join(d, "artifact.json")
    if os.path.exists(p):
        dg = json.load(open(p)).get("program_digest")
        if dg:
            return str(dg)
    return None


def recorded_registration(d: str) -> dict | None:
    """What a component registers (result.json `program.supplied`, a registered_values derive's): its `values` fields with their bits
    and its tables' names; None for a component that holds its values as constants. Parameter positions are left out: they shift with
    the request-local parameters before them."""
    try:
        sup = (json.load(open(os.path.join(d, "result.json"))).get("program") or {}).get("supplied") or {}
    except (OSError, ValueError):
        return None
    out = {}
    if sup.get("registration"):
        out["values"] = sup["registration"].get("fields")
    if sup.get("tables"):
        out["tables"] = [t["name"] for t in sup["tables"]]
    return out or None


def registration_refusal(dirs: list[str]) -> str | None:
    """Why the components cannot share one registration (`Workload_v1`'s `shared.values` and tables), or None: each registers the same
    fields with the same bits and the same tables, or none does."""
    regs = {json.dumps(recorded_registration(d), sort_keys=True) for d in dirs}
    return None if len(regs) <= 1 else (f"the components register different values (result.json program.supplied, {len(regs)} distinct): "
                                        "not one program's registration")


def component_provenance(d: str, desc: dict) -> dict:
    """[rev-pm M-0123 E] the component descriptor GP-01 composes must be the Program the derive recorded: its recomputed digest must equal
    result.json/artifact.json's program_digest.  `refusal` names the mismatch (fail closed); a component dir with no recorded digest is
    refused too (a descriptor without provenance is not a Build artifact)."""
    from verity.ml.circuits.codec import program_digest
    recomputed = program_digest(desc)
    recorded = recorded_component_digest(d)
    refusal = None
    if recorded is None:
        refusal = f"{d}: no recorded program digest (result.json program.digest / artifact.json program_digest): the component has no provenance"
    elif recorded != recomputed:
        refusal = (f"{d}: descriptor digest {recomputed[:16]} != the digest the derive recorded {recorded[:16]}: the component descriptor is not "
                   f"the Program derive_step built")
    return {"dir": d, "descriptor_digest": recomputed, "recorded_digest": recorded,
            "descriptor_sha256": hashlib.sha256(canonical_json(desc)).hexdigest(), "refusal": refusal}


def load_component(d: str, canon: dict) -> dict:
    """A derived dir as GP-01 composes it: {"prov", "comp" | "error", "secs"}; the component is decoded (into the shared-definition set
    `canon`) only when its provenance holds, and what decoding raised is kept for `component_of` to raise."""
    t1 = time.time()
    desc = load_desc(d)
    pre: dict = {"prov": component_provenance(d, desc)}
    if not pre["prov"]["refusal"]:
        try:
            pre["comp"] = W.request_component(desc, REGISTRY, wrapper_params(d), canon=canon, digest=pre["prov"]["descriptor_digest"])
        except Exception as e:  # noqa: BLE001 -- raised by component_of, where GP-01 has always raised it
            pre["error"] = e
    pre["secs"] = round(time.time() - t1, 2)
    return pre


def component_of(pre: dict) -> W.RequestComponent:
    if "error" in pre:
        raise pre["error"]
    return pre["comp"]


def _ready(d: str) -> bool | None:
    """A derived dir's outcome once the derive has written it: True (ok, its shape readable), False (it failed), None (not yet)."""
    try:
        ok = json.load(open(os.path.join(d, "result.json"))).get("ok")
        if ok is not True:
            return False
        return derived_shape(d) is not None or None
    except (OSError, ValueError):
        return None


def follow_components(derived_dirs: list[str], need: set[tuple[int, int]], canon: dict, *, timeout: float | None = None,
                      done: str | None = None, stop: str | None = None) -> dict[str, dict] | None:
    """Load, check and decode each needed component as its derive finishes, while the others still run: {dir: {"prov", "comp" | "error",
    "secs"}}, or None when a derive failed or `stop` appeared (the Build fails; nothing is composed).  Once `done` exists the derives
    are over and a dir still unreadable is left to `build`'s own pass, which reports it as it always did.  Components are decoded in the
    order they finish into one shared-definition set (`canon`): a definition is shared by content, so the order changes no output.
    `timeout` (seconds) bounds the wait for dirs no marker ends; None waits on the markers alone."""
    deadline, pending, out = None if timeout is None else time.time() + timeout, list(derived_dirs), {}
    while pending:
        progressed = False
        for d in list(pending):
            st = _ready(d)
            if st is None:
                continue
            pending.remove(d)
            progressed = True
            if st is False:
                return None
            s = derived_shape(d)
            if s not in need:
                continue
            out[d] = load_component(d, canon)
        if not pending or (done and os.path.exists(done)):
            break
        if stop and os.path.exists(stop):
            return None
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"follow: {len(pending)} derived dir(s) not written in {timeout:.0f} s: {pending[:3]}")
        if not progressed:
            time.sleep(1)
    return out
