"""A `registered_values` fold's supplied inputs: one run parameter `run` / `run_<i>` per request (its sampling values) after the
peers, and the registered `values` (the program's) right before the weights.  Layout: prompts..., seeds..., peers..., runs...,
values, weights.  A field is allocated at its first `Supplied` operand, at the next leaf offset; both Records are finalised when
the fold finishes."""

from __future__ import annotations

from verity.ml.circuits.refs import Affine
from verity.ml.circuits.types import Record, Type, Value

from verity_vllm.observe.fold.resolver import Supplied


class SuppliedInputs:
    def __init__(self) -> None:
        #: the parameter index of each owner ("values", or a request index for its `run_<i>`) and its fields so far
        #: ({name, width, bits, offset})
        self.param: dict[str | int, int] = {}
        self.fields: dict[str | int, list[dict]] = {}

    def __bool__(self) -> bool:
        return bool(self.param)

    def declare(self, params: list, n_requests: int) -> None:
        """Append the run parameters and the registered `values` (empty Records until `finalise`) to `params`."""
        for i in range(n_requests):
            self.param[i], self.fields[i] = len(params), []
            params.append(("run" if n_requests == 1 else f"run_{i}", Record()))
        self.param["values"], self.fields["values"] = len(params), []
        params.append(("values", Record()))

    def leaf(self, op: Supplied, ptype: Type, what: str, req_index: dict[str, int], err: type[Exception] = ValueError) -> Affine:
        """The leaf of the field `op` names in its owner's parameter; a contract violation raises `err`."""
        if ptype != Value(int(op.width)):
            raise err(f"{what}: a {op.width}-bit supplied value for a parameter of type {ptype!r}")
        if op.binding == "registration":
            owner = "values"
        elif op.binding == "run":
            if op.request not in req_index:
                raise err(f"{what}: run value {op.name!r} names request {op.request!r}, which is not in the run header")
            owner = req_index[op.request]
        else:
            raise err(f"{what}: a value is bound per run or at registration, not {op.binding!r}")
        if owner not in self.param:
            raise err(f"{what}: a Supplied operand under a profile without registered_values")
        fields = self.fields[owner]
        f = next((f for f in fields if f["name"] == op.name), None)
        if f is None:
            f = {"name": op.name, "width": int(op.width), "bits": int(op.bits), "offset": len(fields)}
            fields.append(f)
        elif (f["width"], f["bits"]) != (int(op.width), int(op.bits)):
            raise err(f"{what}: value {op.name!r} is {op.width}-bit 0x{int(op.bits):x} here and {f['width']}-bit "
                      f"0x{f['bits']:x} at its first use")
        return Affine("p", self.param[owner], f["offset"], 1)

    def finalise(self, params: tuple) -> tuple:
        """`params` with every supplied Record's fields."""
        params = list(params)
        for owner, pidx in self.param.items():
            params[pidx] = (params[pidx][0], Record(*[(f["name"], Value(f["width"])) for f in self.fields[owner]]))
        return tuple(params)

    def layout(self, params: tuple, requests: list[dict], program=None) -> dict:
        """The supplied inputs as the record states them: the registered `values` and each request's `run_<i>`, every field with its
        leaf offset and bits (hex) -- what registration commits and what a run binds; {} without registered values.  With the
        `program`, `input_nodes` names every parameter's Input nodes: an empty `run_<i>` lowers to none, so a node is not its index."""
        def fields(owner):
            return [{"name": f["name"], "width": f["width"], "bits": f"0x{f['bits']:0{f['width'] // 4}x}", "offset": f["offset"]}
                    for f in self.fields[owner]]
        if not self.param:
            return {}
        out = {"registration": {"param": self.param["values"], "name": "values", "fields": fields("values")},
               "run": [{"param": self.param[i], "name": params[self.param[i]][0], "request": requests[i]["request_id"], "fields": fields(i)}
                       for i in sorted(k for k in self.param if k != "values")]}
        if program is not None:
            out["input_nodes"] = input_nodes(program)
        return out


def program_registered_values(program_dirs) -> bool:
    """Whether the Build's Program dirs register the program's values (`result.json` `program.supplied.registration`, a
    `registered_values` derive's), so the fold the Match compares against them folds under `Profile.registered_values` too.  Dirs
    that disagree, or one without a readable result.json, are refused by name (SystemExit)."""
    import json
    import os
    got = set()
    for d in program_dirs:
        try:
            prog = json.load(open(os.path.join(d, "result.json"))).get("program") or {}
        except (OSError, ValueError, AttributeError) as e:
            raise SystemExit(f"{d}: no readable result.json ({type(e).__name__}): whether its Program registers its values is unknown") from None
        got.add(bool((prog.get("supplied") or {}).get("registration")))
    if len(got) > 1:
        raise SystemExit(f"{list(program_dirs)}: some Programs register their values and some hold them as constants")
    return got == {True}


def input_nodes(program) -> dict[str, list[int]]:
    """Each root parameter's Input nodes (root body node indices, ascending), by parameter name, from the lowered root itself."""
    from verity.ml.circuits.refs import runs
    root = program.root
    out: dict[str, list[int]] = {}
    for (name, _t), coll in zip(root.program_fn.signature()[0], root.input_colls):
        nodes: set[int] = set()
        for _space, _i, lo, hi in runs(coll.refs):
            if hi > lo:
                nodes.update(range(root.body.node_at(lo), root.body.node_at(hi - 1) + 1))
        out[name] = sorted(nodes)
    return out
