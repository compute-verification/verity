"""A profile's witness inputs (`Profile.witnesses`): one parameter `<name>` / `<name>_<i>` per request and name after the peers, an
Array of 32-bit words whose leaves the `WitnessParam` reads take in issue order.  Each is sized when the fold finishes, and an empty one is
refused, since it would shift the input layout."""

from __future__ import annotations

from verity.ml.circuits.refs import Affine
from verity.ml.circuits.types import Array, Record, Type, Value

from verity_vllm.observe.fold.resolver import WitnessParam


class WitnessInputs:
    def __init__(self) -> None:
        #: (request index, name) -> the witness parameter's index, and the reads that took its leaves in issue order
        self.param: dict[tuple[int, str], int] = {}
        self.fields: dict[tuple[int, str], list[dict]] = {}

    def declare(self, params: list, names: tuple[str, ...], n_requests: int) -> None:
        """Append every name's parameter per request (empty Records until `finalise`) to `params`."""
        for name in names:
            for i in range(n_requests):
                self.param[(i, name)], self.fields[(i, name)] = len(params), []
                params.append((name if n_requests == 1 else f"{name}_{i}", Record()))

    def leaves(self, op: WitnessParam, i: int, ptype: Type, what: str, req_index: dict[str, int], read: dict,
               err: type[Exception] = ValueError) -> Affine:
        """Row i's next leaves of its request's `op.name` parameter; `read` (step, event, spec, hint) is kept with their range."""
        key = (req_index.get(op.keys[i], -1), op.name)
        if key not in self.param:
            raise err(f"{what}: request {op.keys[i]!r} has no {op.name!r} witness parameter (Profile.witnesses)")
        if ptype.homogeneous_width() != 32:
            raise err(f"{what}: a witness parameter holds 32-bit words, the parameter wants {ptype!r}")
        fields = self.fields[key]
        base = fields[-1]["offset"] + fields[-1]["leaves"] if fields else 0
        fields.append({"offset": base, "leaves": ptype.leaves, **read})
        return Affine("p", self.param[key], base, ptype.leaves)

    def finalise(self, params: tuple, requests: list[dict]) -> tuple[tuple, list[tuple[str, dict]]]:
        """`params` with every witness parameter sized to its reads, and a refusal (message, detail) for each that none took."""
        params, empty = list(params), []
        for (ri, name), pidx in self.param.items():
            n = sum(f["leaves"] for f in self.fields[(ri, name)])
            if not n:
                req = requests[ri]["request_id"]
                empty.append((f"request {req!r}: no read took its {params[pidx][0]!r} witness parameter, which would be empty (input layout "
                              f"shift); refused", {"request": req, "param": params[pidx][0]}))
            params[pidx] = (params[pidx][0], Array(n, Value(32)) if n else Record())
        return tuple(params), empty

    def layout(self, params: tuple, requests: list[dict]) -> list[dict]:
        """The witness parameters as the record states them: parameter index, request, witness name and every read's leaf range in
        issue order, so a checker binds each to the Program's input of that name."""
        return [{"param": pidx, "name": params[pidx][0], "witness": name, "request": requests[ri]["request_id"], "request_index": ri,
                 "leaves": sum(f["leaves"] for f in self.fields[(ri, name)]), "fields": list(self.fields[(ri, name)])}
                for (ri, name), pidx in self.param.items()]
