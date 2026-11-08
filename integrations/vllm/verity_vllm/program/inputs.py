"""What fills each input of a served Program, declared, and the refusals that keep every value it reads one whose bits are
accounted for.

A served Program's outputs are a function of what it reads, so they carry no more than the advice's bits and the wrong proof
blocks' only where every value it reads is one of these kinds (`Kind.kind`):

- ``block``: another proof block's output, which the link checks bind (a carried request's history, a peer rank's partial);
- ``certified``: the system's ingress (a request's prompt, its seed);
- ``registered``: fixed before any coin (the weights);
- ``advice``: decided by serving and charged at its width by a term of the window charge (`CHARGED`);
- ``witness``: decided by serving and determined by the rest: the Program returns a check bit, which the verifier requires to
  be 1, from a relation that holds for one value alone of what reaches the outputs (`RELATIONS`, each with its evidence).

A value of none of these is free: the prover picks it, nothing charges it, and it can move output bits. Three refusals keep
each new one out, each naming the fix:

- `parameter`: a Program parameter `PARAMS` doesn't declare (`predict.build.build_program`, `predict.node.node_parameters`);
- `run_field`: a field of the `run` parameter `RUN_FIELDS` doesn't declare (`frontend.registration.RunInputs.value`);
- `check`: a check bit no relation of `RELATIONS` computes (`frontend.derive.Ctx.check`, `predict.common.with_checks`).

What is free today is `ALLOWED`, each entry with its reason, its bound in bits and its fix; the list only shrinks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

KINDS = ("block", "certified", "registered", "advice", "witness")

#: the window charge's terms that count a Program input's bits: none yet (`ChargeCount_v2` counts advice v1's decisions, its
#: hits past the pool bound and its order entries), so no Program input is advice until a term counts it
CHARGED: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Kind:
    """A declared input: its kind, why it is that kind, and for ``advice`` the charge's term that counts its bits, for
    ``witness`` the relation (a Definition's base id, a key of `RELATIONS`) whose check holds it to one value."""

    kind: str
    why: str
    charge: str = ""
    relation: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"an input's kind is one of {KINDS}, not {self.kind!r}")
        if self.kind == "advice" and self.charge not in CHARGED:
            raise ValueError(f"advice is charged by a term of the window charge ({sorted(CHARGED)}), not {self.charge!r}")
        if self.kind == "witness" and self.relation not in RELATIONS:
            raise ValueError(f"a witness names its relation in RELATIONS ({sorted(RELATIONS)}), not {self.relation!r}")


#: each relation whose check bit a served Program returns: its Definition's base id -> the evidence that it holds for one value
#: alone of what reaches the outputs, a test (``path::function``) or a Lean theorem (``path::name``)
RELATIONS: dict[str, str] = {
    "DsaTopKLongRow_v1":
        "integrations/vllm/tests/program/test_dsa_topk.py::test_the_long_row_accepts_exactly_the_top_set_in_ascending_order",
}

#: a Program parameter's name (a node Program's ``rank<r>.`` prefix and a leaf's ``[i]`` aside) -> its kind; ``run`` is
#: declared field by field (`RUN_FIELDS`)
PARAMS: dict[str, Kind] = {
    "prompt": Kind("certified", "the request's prompt ids: ingress v0's parse unit outputs them, else its certified body "
                                "(`predict.steps.request_of`)"),
    "input_ids": Kind("certified", "a step wrapper's token ids (the Build's step trace): the prompt's and earlier samples'"),
    "seed": Kind("certified", "the request's seed, its certified body's or the public default (`predict.steps.policy_of`)"),
    "prefix": Kind("block", "a request's first C positions' rows: the producers' outputs (a prefix-cache hit) or the last "
                            "window's edge ports (a carried request)"),
    "prefix_ik": Kind("block", "the indexer's key rows of the history, as ``prefix``"),
    "prefix_iks": Kind("block", "the indexer's key scales of the history, as ``prefix``"),
    "last_token": Kind("block", "a carried request's last kept output, the last window's ``edge.last``"),
    "peers_*": Kind("block", "another rank's partial of a collective, its rank Program's output (`pipeline.tp.links`)"),
    "weights": Kind("registered", "the model's weights, registered before any coin"),
}

#: a field of the `run` parameter (`frontend.registration.RunInputs`), as a pattern over its name -> its kind
RUN_FIELDS: dict[str, Kind] = {
    r"[\w.]+/topk_switches\[t=\d+\]": Kind(
        "witness", "a DSA row past TOPK keys: the Waksman settings that route its top-k order. Only the top-k set in ascending "
                   "index order passes `DsaTopKLongRow_v1`, and every reader reads the routed rows, a function of that order, "
                   "so the settings' redundant bits reach no output", relation="DsaTopKLongRow_v1"),
}

#: what is free today: (``parameter`` or ``run field``, a pattern over the name) -> its reason, its bound and its fix
ALLOWED: dict[tuple[str, str], str] = {
    ("parameter", "schedule"):
        "DeepSeek-V4's `MoeAlignRanks` key of each (token, slot) pair, the position serving's atomicAdd gave it, which no "
        "check holds. The key puts the pair in one of its expert's moe blocks, and the block sets the Marlin split-K cut bits "
        "(`MarlinMoeSchedule`), so it moves output bits: at most sum_e log2(c_e! / prod_b s_b!) bits a MoE layer a step, c_e "
        "the expert's pairs and s_b its blocks' sizes. Fix: serve the align in pair order and key `MoeAlignRanks` by the pair "
        "index, or charge the key as advice",
    ("parameter", "splits"):
        "a derived top-p split count, SplitsFor_v1 of the batch's live rows, per sampling event, at most log2 6 bits an event; "
        "`predict.batch.check_compiled` refuses it (R3) in every compiled session, and serving pins `registered_splits`. Fix: "
        "delete the derived form",
    ("run field", r"hole\d+\.\d+:.+"):
        "a survey's stand-in for a node every rule refuses (`frontend.rules.training_survey.Hole`): `check_identity` refuses "
        "every Program with one, so no registered Program reads it",
}


class Free(ValueError):
    """A value a Program would read that no declaration covers: one serving decides and nothing computes, certifies, registers
    or charges."""


def _fix(what: str) -> str:
    return (f"{what} is undeclared in `verity_vllm.program.inputs`: if a proof block computes it, the ingress certifies it or "
            f"the registration fixes it, declare its kind; if serving decides it, serve it canonically so a proof block computes "
            f"it (or a relation holds it to one value: a witness, its relation in RELATIONS with the test or theorem that shows "
            f"it), or charge it at its width in a term of the window charge (CHARGED) and declare it as advice")


def _allowed(where: str, name: str) -> str | None:
    return next((why for (w, pat), why in ALLOWED.items() if w == where and re.fullmatch(pat, name)), None)


def parameter(name: str) -> Kind | str:
    """Program parameter `name`'s declaration, or its `ALLOWED` reason; `Free` otherwise."""
    base = re.sub(r"\[\d+\]$", "", re.sub(r"^rank\d+\.", "", name))
    base = "peers_*" if re.fullmatch(r"peers_\d+", base) else base
    if base in PARAMS:
        return PARAMS[base]
    if (why := _allowed("parameter", base)) is not None:
        return why
    raise Free(_fix(f"Program parameter {name!r}"))


def check_params(params: Any) -> None:
    """Refuse a Program whose parameters (``(name, type)`` pairs) aren't each declared (`parameter`); ``run``'s fields are
    each refused where they're read (`run_field`)."""
    for name, _ in params:
        if name != "run":
            parameter(name)


def run_field(name: str) -> Kind | str:
    """Field `name` of the `run` parameter's declaration, or its `ALLOWED` reason; `Free` otherwise."""
    for pat, kind in RUN_FIELDS.items():
        if re.fullmatch(pat, name):
            return kind
    if (why := _allowed("run field", name)) is not None:
        return why
    raise Free(_fix(f"run field {name!r}"))


def check(B: Any, ok: Any, what: str) -> None:
    """Refuse check bit `ok` (a `Coll` of builder `B`'s body) unless it is an output of a call to a relation of `RELATIONS`."""
    ref = ok.refs[0] if ok.refs.count else None
    if ref is None or ref[0] != "n" or not 0 <= ref[1] < len(B.nodes):
        raise Free(f"check {what!r} is no call's output in the Program's body: a check bit is a relation's verdict, its "
                   f"relation in `verity_vllm.program.inputs.RELATIONS`")
    fn = B.nodes[ref[1]].fn
    base = getattr(fn, "base_id", None) or fn.id
    if base not in RELATIONS:
        raise Free(f"check {what!r} is the verdict of {base}, which RELATIONS doesn't list: a relation whose check bit a "
                   f"Program returns needs a test or theorem that it holds for one value alone of what reaches the outputs, "
                   f"else its witness is free (`verity_vllm.program.inputs`)")
