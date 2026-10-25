"""The fold's input layout in the comparator: which Input nodes of the oracle Prog each request owns, and which are the shared weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verity_vllm.check.match import instance_sequence as SEQ


def weights_input_nodes(B: "SEQ.Prog", n_request_params: int, n_seed_params: int = 0, n_peer_params: int = 0,
                        n_witness_params: int = 0) -> set[int]:
    """The fold's shared WEIGHTS parameter as a SET of Input nodes: every Input node of the oracle
    Prog that no request owns.  The fold's root parameters are `prompt_<r>` (one per request, header order), then `seed_<i>` (one per
    seeded request), then ONE `weights` Record (`fold._header`).  `verity.ml.circuits.program._lower_input` lowers a width-homogeneous
    Record to ONE `Input` batch (the bf16 dense models: `n_input_nodes - 1` was that node) but a MIXED-width Record to one batch PER LEAF
    FIELD (Qwen3-4B-FP8: e4m3 weight + f32 `weight_scale_inv` per linear + bf16 norms -> 435 nodes at B=1), so the last Input node is not
    "the weights" and the node count says nothing about the number of requests.  The request-owned Inputs are exactly the first
    `n_request_params + n_seed_params` nodes (the codec emits one node per prompt/seed parameter, in parameter order, before the Record's
    leaves); everything after them is the weights, shared by every request subcircuit exactly once (one global address per leaf).
    Refuses by name a Prog whose Input nodes cannot hold the header's parameters or whose request-owned Inputs are not the leading ones.
    A tensor-parallel rank fold adds ONE `peers_<i>` Record per request (homogeneous bf16 -> one Input node each) after the seeds
    and before the weights: `n_peer_params` of them, request-owned.
    A profile's witness parameters (`schedule_<i>`, one Array of 32-bit words per request and name: `compare.Prog.witness_index`) follow
    the peers, one Input node each: `n_witness_params` of them, request-owned.
    A `registered_values` fold's supplied Input nodes (`run_<i>`, `values`: `compare.Prog.supplied_fields`) sit between the witnesses
    and the weights and are neither: they are left out of both counts, and the weights follow them."""
    n_owned = n_request_params + n_seed_params + n_peer_params + n_witness_params
    supplied = set(getattr(B, "supplied_fields", {}))
    if supplied and min(supplied) < n_owned:
        raise ValueError(f"supplied input nodes {sorted(supplied)[:4]} precede the {n_owned} request-owned ones")
    inputs = sorted(set(B.inputs) - supplied)
    if len(inputs) <= n_owned:
        raise ValueError(f"program has {len(inputs)} input nodes but the header has {n_request_params} requests + {n_seed_params} seeds "
                         f"+ {n_peer_params} peers + {n_witness_params} witnesses (>= 1 weights node expected)")
    owned = inputs[:n_owned]
    if owned != list(range(n_owned)):
        raise ValueError(f"the request-owned input nodes are not the leading Input nodes 0..{n_owned - 1}: {owned[:8]}")
    w = set(inputs[n_owned:])
    # the codec's layout contract (compare.load_oracle `input_base`): the weights leaves are contiguous after the owned parameters
    if min(w) != n_owned + len(supplied):
        raise ValueError(f"first weights input node is {min(w)}, expected {n_owned + len(supplied)} (request params {n_request_params}, seeds {n_seed_params}, "
                         f"peers {n_peer_params}, witnesses {n_witness_params})")
    # cross-check against the fold's OWN statement of where the weights parameter is: load_oracle keys `input_base` by every Input node at or
    # after the weights parameter index the instances actually read (`operand_refs` space "p", fold contract: weights = the LAST parameter).
    # A header claiming more (or fewer) requests than the Program has prompt parameters puts the boundary elsewhere -> refused by name.
    base = getattr(B, "input_base", None)
    if base:
        if set(base) != w:
            raise ValueError(f"header and Program disagree on the input layout: the header's {n_request_params} requests + {n_seed_params} seeds "
                             f"+ {n_peer_params} peers + {n_witness_params} witnesses put "
                             f"the weights at input nodes {min(w)}..{max(w)}, the fold's instances read the weights parameter as input nodes "
                             f"{min(base)}..{max(base)}")
    return w


def request_inputs(B: "SEQ.Prog", reqs: list[dict]) -> tuple[dict[str, int], dict[str, int], list[tuple[str, int]], set[int]]:
    """(seed node of request, peers node of request, (request, witness node) pairs, weights input nodes): the fold's request-owned
    parameters by request id, from `compare.Prog.seed_index`, `peer_index` and `witness_index` (node -> header request index), and the
    weights after them.  Refused by name when the Program has a parameter for a request the header does not have, or a TP rank fold's
    peers parameters are not one per request."""
    def owners(index: dict[int, int], what: str) -> list[tuple[str, int]]:
        out = []
        for k, i in sorted(index.items()):
            if i >= len(reqs):
                raise ValueError(f"program has a {what} input node for request index {i} but the header has {len(reqs)} requests")
            out.append((reqs[i]["request_id"], k))
        return out
    seed_node_of = dict(owners(getattr(B, "seed_index", {}), "seed"))
    peer_node_of = dict(owners(getattr(B, "peer_index", {}), "peers"))
    if peer_node_of and len(peer_node_of) != len(reqs):
        raise ValueError(f"program has {len(peer_node_of)} peers input nodes for {len(reqs)} requests (a TP rank fold declares one per request)")
    witnesses = owners(getattr(B, "witness_index", {}), "witness")
    return seed_node_of, peer_node_of, witnesses, weights_input_nodes(B, len(reqs), len(seed_node_of), len(peer_node_of), len(witnesses))
