"""A profile's witness inputs (`Profile.witnesses`): one parameter `<name>` / `<name>_<i>` per request and name after the peers, an
Array of 32-bit words whose leaves the `Witness` reads take in issue order, and under a batch one `launch_<name>` per name of
`Profile.launch_witnesses` after them, whose leaves the launches several requests share take (`Witness.members`).  Each is sized when
the fold finishes, and an empty one is refused, since it would shift the input layout."""

from __future__ import annotations

from verity.ml.circuits.refs import Affine
from verity.ml.circuits.types import Array, Record, Type, Value

from verity_vllm.observe.fold.resolver import Witness

#: the request index under which a run's `launch_<name>` parameter is kept
LAUNCH = -1


class WitnessInputs:
    def __init__(self) -> None:
        #: (request index | LAUNCH, name) -> the witness parameter's index, and the reads that took its leaves in issue order
        self.param: dict[tuple[int, str], int] = {}
        self.fields: dict[tuple[int, str], list[dict]] = {}

    def declare(self, params: list, names: tuple[str, ...], n_requests: int, launch_names: tuple[str, ...] = ()) -> None:
        """Append every name's parameter per request, then under a batch each launch name's launch parameter (empty Records until
        `finalise`), to `params`."""
        keys = [(i, name) for name in names for i in range(n_requests)] + ([(LAUNCH, name) for name in launch_names] if n_requests > 1 else [])
        for i, name in keys:
            self.param[(i, name)], self.fields[(i, name)] = len(params), []
            params.append((f"launch_{name}" if i == LAUNCH else name if n_requests == 1 else f"{name}_{i}", Record()))

    def _take(self, key: tuple[int, str], n: int, field: dict) -> int:
        fields = self.fields[key]
        base = fields[-1]["offset"] + fields[-1]["leaves"] if fields else 0
        fields.append({"offset": base, "leaves": n, **field})
        return base

    def leaves(self, op: Witness, i: int, ptype: Type, what: str, req_index: dict[str, int], read: dict,
               err: type[Exception] = ValueError) -> Affine:
        """Row i's next leaves of its request's `op.name` parameter, or of the launch's with `op.members`, whose members each reserve
        their linked words; `read` (step, event, spec, hint) is kept with every range."""
        key = (LAUNCH if op.members else req_index.get(op.keys[i], -2), op.name)
        if key not in self.param:
            owner = f"launch {op.keys[i]!r} (several requests)" if op.members else f"request {op.keys[i]!r}"
            raise err(f"{what}: {owner} has no {op.name!r} witness parameter (Profile.witnesses)")
        if ptype.homogeneous_width() != 32:
            raise err(f"{what}: a witness parameter holds 32-bit words, the parameter wants {ptype!r}")
        launch = {"launch": op.keys[i]} if op.members else {}
        base = self._take(key, ptype.leaves, {**launch, **read})
        for req, slot, pairs in op.members:
            if (req_index.get(req, -2), op.name) not in self.param:
                raise err(f"{what}: launch member {req!r} has no {op.name!r} witness parameter (Profile.witnesses)")
            self._take((req_index[req], op.name), pairs + 1, {"form": "linked", **launch, "slot": slot, "pairs": pairs, **read})
        return Affine("p", self.param[key], base, ptype.leaves)

    def finalise(self, params: tuple, requests: list[dict]) -> tuple[tuple, list[tuple[str, dict]]]:
        """`params` with every witness parameter sized to its reads, and a refusal (message, detail) for each that none took."""
        params, empty = list(params), []
        for (ri, name), pidx in self.param.items():
            n = sum(f["leaves"] for f in self.fields[(ri, name)])
            if not n:
                owner = "the run" if ri == LAUNCH else f"request {requests[ri]['request_id']!r}"
                empty.append((f"{owner}: no read took its {params[pidx][0]!r} witness parameter, which would be empty (input layout "
                              f"shift); refused", {"request": None if ri == LAUNCH else requests[ri]["request_id"], "param": params[pidx][0]}))
            params[pidx] = (params[pidx][0], Array(n, Value(32)) if n else Record())
        return tuple(params), empty

    def layout(self, params: tuple, requests: list[dict]) -> list[dict]:
        """The witness parameters as the record states them: parameter index, request (None for a launch parameter), witness name and
        every read's leaf range in issue order, so a checker binds each to the Program's input of that name."""
        return [{"param": pidx, "name": params[pidx][0], "witness": name, "request": None if ri == LAUNCH else requests[ri]["request_id"],
                 "request_index": None if ri == LAUNCH else ri, "leaves": sum(f["leaves"] for f in self.fields[(ri, name)]),
                 "fields": list(self.fields[(ri, name)])}
                for (ri, name), pidx in self.param.items()]
