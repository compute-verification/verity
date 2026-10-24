"""REDTEAM phase 14 target 3 (#3790): can a served window make its Programs read ids other than the port's, or let the served
rows' positions disagree with the statics, unseen?  The window binds its one committed ids copy as the manifest's single ids
identity (`serve_seal.port_identities`), and refuses any other tensor bound to that identity (`serve_seal.copies_beside_port`).

These are pure-function probes of the guard, no torch: `copies_beside_port` reads the binding map's identities, which are what
`binding.build_binding_map` writes from `Binder.bind` -- AFTER `op_path_aliases` is applied (`query.manifest.bind.Binder.bind`).
So a second ids copy a prover commits under a decoy name whose op_path ALIASES to the port's identity resolves, in the map, to the
port's identity and is caught like any raw second copy.  The one place the guard does not reach -- the eager row's positions, which
are the Program's RoPE statics and not a committed port identity -- is asserted as the boundary the served-rows tie must hold."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity_vllm.commit.served_record import PORT_NAMES
from verity_vllm.pipeline import serve_seal as ss


def _bmap_from_bind(binder, step, names):
    """The binding map entries `binding.build_binding_map` writes for `names` at `step`: each entry's identity is the
    (op_path, output_member) `Binder.bind` resolves the name to (aliases applied), and its name is the committed name."""
    entries = []
    for name, (op, member, _k, _how) in zip(names, binder.bind(step, names)):
        entries.append({"kind": "leaf", "name": name, "identity": {"op_path": op, "output_member": member, "step": step}})
    return {"entries": entries}


def _eager(aliases=None):
    """An eager row's manifest: its one ids identity the Embedding's input (`model.embed_tokens/in`, family input_token_ids), a
    layer-1 Gemm, and the `op_path_aliases` the opened store resolves the window's names through."""
    return {"identities": [{"op_path": "model.embed_tokens", "output_member": "in", "family": "input_token_ids", "engine_step": 0},
                           {"op_path": "model.layers.1.mlp.gate_up_proj", "output_member": "out", "family": "instance_outputs", "engine_step": 0}],
            "op_path_aliases": dict(aliases or {})}


def _port_binder(manifest):
    """The binder `serve_seal.seal_window` builds for a served window with a step-input port (`served_binder`): the manifest's
    identities placed at the step, the port's ids/positions rebound as the identities the Programs read them under."""
    from verity_vllm.commit.served_record import StepInputPort

    return ss.served_binder(manifest, [0], [StepInputPort(SimpleNamespace(chunk=4096), 512)])


def test_phase14_port_identities_binds_one_ids_and_refuses_two():
    """The eager row's port ids are the Embedding's input, and it reads no committed positions (RoPE's are static); a manifest
    naming two input-ids identities is refused -- the port's ids can be one of them only."""
    assert ss.port_identities(_eager()) == {PORT_NAMES[0]: ("model.embed_tokens", "in")}
    two = {"identities": [{"op_path": op, "output_member": "in", "family": "input_token_ids"} for op in ("model.embed_tokens", "model.mm_proj")]}
    with pytest.raises(ValueError, match="2 input-ids identities"):
        ss.port_identities(two)


def test_phase14_a_second_ids_copy_under_a_raw_name_is_caught():
    """A tensor committed beside the port under the port's own identity (name != the port's) is a second copy of the ids the
    Programs could read in place of the port's: `copies_beside_port` names it."""
    manifest = _eager()
    binder = _port_binder(manifest)
    bmap = _bmap_from_bind(binder, 0, [PORT_NAMES[0], "model.embed_tokens/in"])
    assert ss.copies_beside_port(bmap, ss.port_identities(manifest)) == ["step 0: model.embed_tokens/in"]


def test_phase14_a_second_ids_copy_aliased_to_the_port_identity_is_caught():
    """The alias path the PR's own suite does not probe: a prover commits a second ids copy under a DECOY name (`model.decoy/in`)
    and aliases its op_path to the port's (`model.embed_tokens`).  `Binder.bind` applies `op_path_aliases`, so the map entry's
    identity is the port's and `copies_beside_port` catches it -- the raw name never has to equal the port's for the guard to see
    the collision.  Without the alias the decoy resolves to its own identity, which the Programs (reading the port's) never read."""
    manifest = _eager({"model.decoy": ["model.embed_tokens", 0]})
    binder = _port_binder(manifest)
    bmap = _bmap_from_bind(binder, 0, [PORT_NAMES[0], "model.decoy/in"])
    assert [e["identity"]["op_path"] for e in bmap["entries"]] == ["model.embed_tokens", "model.embed_tokens"]
    assert ss.copies_beside_port(bmap, ss.port_identities(manifest)) == ["step 0: model.decoy/in"]

    no_alias = _eager()
    bmap0 = _bmap_from_bind(_port_binder(no_alias), 0, [PORT_NAMES[0], "model.decoy/in"])
    assert [e["identity"]["op_path"] for e in bmap0["entries"]] == ["model.embed_tokens", "model.decoy"]
    assert ss.copies_beside_port(bmap0, ss.port_identities(no_alias)) == []


def test_phase14_copies_beside_port_spares_the_ports_own_entries_and_other_members():
    """The port's own ids/positions entries, the embedding's OUTPUT (a different member of the port's op), and an unrelated tensor
    are not second copies -- the guard names only a tensor bound to an identity the port's tensors are, under another name."""
    compiled = {"identities": [{"op_path": "c00.input", "output_member": m, "family": "served_kernel_values", "engine_step": 0}
                               for m in ("ids", "positions")]}
    aliased = ss.port_identities(compiled)
    assert aliased == {PORT_NAMES[0]: ("c00.input", "ids"), PORT_NAMES[1]: ("c00.input", "positions")}
    bmap = {"entries": [
        {"kind": "leaf", "name": PORT_NAMES[0], "identity": {"op_path": "c00.input", "output_member": "ids", "step": 0}},
        {"kind": "leaf", "name": PORT_NAMES[1], "identity": {"op_path": "c00.input", "output_member": "positions", "step": 0}},
        {"kind": "leaf", "name": "c00.input/out", "identity": {"op_path": "c00.input", "output_member": "out", "step": 0}},
        {"kind": "leaf", "name": "model.embed_tokens/out", "identity": {"op_path": "model.embed_tokens", "output_member": "out", "step": 0}},
        {"kind": "leaf", "name": "evil/ids", "identity": {"op_path": "c00.input", "output_member": "ids", "step": 1}},
    ]}
    assert ss.copies_beside_port(bmap, aliased) == ["step 1: evil/ids"]


def test_phase14_eager_positions_are_not_a_port_identity_the_guard_reaches():
    """The boundary: on the eager row the port's positions bind as no Program identity (`port_identities` returns no positions
    entry), so `copies_beside_port` does not guard a second positions copy -- the served rows' positions agreeing with the RoPE
    statics is the served-rows tie's job, not this seal's.  A regression here would be a positions entry appearing in the eager
    port's identities."""
    manifest = _eager()
    aliased = ss.port_identities(manifest)
    assert PORT_NAMES[1] not in aliased
    bmap = {"entries": [
        {"kind": "leaf", "name": PORT_NAMES[1], "identity": {"op_path": "step.input", "output_member": "positions", "step": 0}},
        {"kind": "leaf", "name": "model.layers.0.self_attn.rotary_emb/second_positions", "identity": {"op_path": "step.input", "output_member": "positions", "step": 0}},
    ]}
    assert ss.copies_beside_port(bmap, aliased) == []
