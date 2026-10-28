"""A `registered_values` Program's supplied inputs in the comparator: the registered `values` and a request's `run_<i>`.

`RootView.supplied_fields[node]` = sorted [(leaf offset, name, width)] of the one Input node a supplied Record lowers to, and
`RootView.supplied_owner[node]` = the request index of a run parameter, None for the registered values.  A read canonicalises as
("SUPPLIED", name, lo, hi) with lo/hi relative to the field on both sides: the fold's layout comes from program.json (`supplied`,
`supplied.input_nodes`), the derived Program's from its derive-report (`provenance.supplied`)."""

from __future__ import annotations


def field_of(P, idx: int, lo: int, hi: int):
    """(offset, name) of the supplied field holding leaves [lo, hi) of Input node `idx`; None when the range is not inside one."""
    for off, name, _w in P.supplied_fields.get(idx, ()):
        if off <= lo and hi <= off + 1:
            return off, name
    return None


def src(P, idx: int, lo: int, hi: int) -> tuple:
    f = field_of(P, idx, lo, hi)
    return ("SUPPLIED", f[1], lo - f[0], hi - f[0]) if f else ("SUPPLIED?", idx, lo, hi)


def srcP(P, p: tuple, unfold: bool) -> list[tuple]:
    """`src` over a run progression: uniform iff the whole span lies in one field, else run by run."""
    _space, (idx0, lo0, hi0), (_dI, dL, dH), c = p
    f = field_of(P, idx0, min(lo0, lo0 + (c - 1) * dL), max(hi0, hi0 + (c - 1) * dH))
    if f is None:
        return P._per_run(p, unfold)
    return [("SUPPLIED", f[1], (lo0 - f[0], hi0 - f[0]), (dL, dH), c)]


def values_nodes(P) -> frozenset[int]:
    """The registered `values` Input node(s): shared by every request, read like a weight."""
    return frozenset(k for k, o in getattr(P, "supplied_owner", {}).items() if o is None)


def run_owners(P, reqs: list[dict]) -> list[tuple[str, int]]:
    """(request id, Input node) of every non-empty `run_<i>`: request i's own, like its prompt."""
    return [(reqs[i]["request_id"], k) for k, i in getattr(P, "supplied_owner", {}).items() if i is not None]


def set_supplied(P, entries: list[tuple[int, int | None, dict]], d: str) -> None:
    """`P.supplied_fields` / `supplied_owner` from (Input node, owner, layout entry {name, fields: [{name, width, offset}]}); a
    supplied node is never a seed, whatever its width."""
    for node, owner, ent in entries:
        if node not in P.inputs:
            raise ValueError(f"{d}: supplied parameter {ent.get('name')!r} (node {node}) is not an Input node of the Program")
        flds = sorted((int(f["offset"]), str(f["name"]), int(f["width"])) for f in ent.get("fields") or [])
        if len({w for _o, _n, w in flds}) > 1:
            raise ValueError(f"{d}: supplied parameter {ent.get('name')!r} mixes widths, so it is not one Input node")
        if [o for o, _n, _w in flds] != list(range(len(flds))) or len(flds) != int(P.inputs[node][1]):
            raise ValueError(f"{d}: supplied parameter {ent.get('name')!r}: fields at offsets {[o for o, _n, _w in flds][:8]} do not "
                             f"tile its Input node of {P.inputs[node][1]} leaves")
        P.supplied_fields[node] = flds
        P.supplied_owner[node] = owner
    if P.supplied_fields and P.seed_nodes & set(P.supplied_fields):
        seeds = sorted(P.seed_nodes - set(P.supplied_fields))
        P.seed_nodes, P.seed_index = set(seeds), {k: i for i, k in enumerate(seeds)}


def derived_registration(prov: dict, n_user: int, d: str) -> dict | None:
    """The derived Program's registered `values` entry: one homogeneous Record, never empty, right after the user inputs."""
    reg = (prov.get("supplied") or {}).get("registration")
    if reg and int(reg["param"]) != n_user:
        raise ValueError(f"{d}: derive-report puts `values` at parameter {reg['param']}, after {n_user} user inputs it is {n_user}")
    return reg


def load_oracle_supplied(P, pj: dict, d: str) -> set[int] | None:
    """The fold's supplied parameters at the Input nodes program.json's `supplied.input_nodes` names (an empty `run_<i>` lowers to
    none, so a parameter's node is not its index); returns the weights' Input nodes when it names them."""
    sup = pj.get("supplied") or {}
    in_nodes = sup.get("input_nodes")
    if sup and not in_nodes:
        raise ValueError(f"{d}: program.json states supplied parameters but not the parameters' Input nodes (`input_nodes`)")
    ents = []
    for ent, owner in [(sup.get("registration"), None)] + [(r, i) for i, r in enumerate(sup.get("run") or [])]:
        if not ent:
            continue
        nodes = in_nodes.get(ent["name"]) or []
        if not ent.get("fields"):
            if nodes:
                raise ValueError(f"{d}: supplied parameter {ent['name']!r} has no fields but Input nodes {nodes[:4]}")
            continue
        if len(nodes) != 1:
            raise ValueError(f"{d}: supplied parameter {ent['name']!r} lowers to {len(nodes)} Input nodes, one homogeneous Record is one")
        ents.append((nodes[0], owner, ent))
    set_supplied(P, ents, d)
    return set(in_nodes["weights"]) if in_nodes and "weights" in in_nodes else None
