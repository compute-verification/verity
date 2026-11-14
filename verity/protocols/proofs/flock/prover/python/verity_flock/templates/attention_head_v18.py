"""Flock lowering of ``attention-head-v18`` (``registry.dead_free.AttentionHead_v18{T, D, BN, DOT, INV, MASKED_FROM}`` on sm120's
FA2 steps at a decode-only launch): every tensor-core k-step of the head is one unit (``verity_flock.ir_lower.tc_units``;
``fp.tc_dot16`` with ``HopperBF16WgmmaDot16``'s groups), ``ceil(D/16)`` QK steps a key (``acc + q . k``) and D per k-step of 16
visible keys PV steps (``acc + p . v``). The key and value words come from their committed rows, a value row past T in the first
visited block's last k-step is the zero word v18 pads with (it pads nothing else), the query is a public port, and the softmax
(running max, the rescale by the guarded running max at ``CHECK`` blocks and by the running max as it is at the others, MUFU.EX2,
lane sums, ``Fa2InvSum``, the output cast) is the tail, which ``verity/flock-circuit`` proves as stage circuits.

What differs from ``attention-head`` (AttentionHead_v3): the tensor-core step (sm120's ``HopperBF16WgmmaDot16_v1``, one
16-product group, where v3 has the Ampere step), MASKED_FROM (T's, ``ceil(T/BN) - 1``: only the first visited block runs
``Check_inf``), and the first block's padding. The units' shape is v3's, so the same ``tc_units`` reads them off the graph.

T is per instance: a statement covers the instances of one T (:func:`lowering` takes it; :func:`lowering_for_set` reads it off a
set whose instances share it). The one unit netlist is the same for every T; the leaf maps and the tail are T's. Proved under
``verity/flock-circuit`` only (:data:`SCHEMES`)."""
TEMPLATE = "attention-head-v18"
FAMILY = "AttentionHead_v18"
UNIT = "attention-head-v18/sm120-fa2-bf16/tc-step"
#: ``verity_flock.circuit.SCHEME``, the commitment scheme of ``verity/flock-circuit``
SCHEMES = ("frame-v3-sha512/hm96-sha512",)


def definition(subcircuit, T: int):
    return subcircuit.definition(int(T))


def lowering(subcircuit, T: int):
    from verity_flock import ir_lower

    D, BN = int(subcircuit.params["D"]), int(subcircuit.params["BN"])
    return ir_lower.lowering(definition(subcircuit, T), f"{UNIT}/d{D}-bn{BN}/t{int(T)}", unit_fn=ir_lower.tc_units)


def key_count(input_set) -> int:
    """The one key count of a set's instances (``len(k) / D``)."""
    D = int(input_set.subcircuit.params["D"])
    counts = {len(k) // D for k in input_set.port("k")}
    if len(counts) != 1:
        raise ValueError(f"{input_set.name}: instances of key counts {sorted(counts)}; a statement covers one T")
    return counts.pop()


def lowering_for_set(input_set):
    return lowering(input_set.subcircuit, key_count(input_set))


def supports(subcircuit, scheme):
    if int(subcircuit.params["D"]) % 16:
        return f"C-Flock lowers {FAMILY} with D a multiple of 16, not D = {subcircuit.params['D']}"
    if scheme in SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under verity/flock-circuit ({', '.join(SCHEMES)}: tensor-core steps as units, the "
            f"softmax in the statement), not under {scheme}")


def relation(subcircuit, scheme):
    return "verity/flock-circuit"
