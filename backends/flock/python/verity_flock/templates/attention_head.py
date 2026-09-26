"""C-Flock lowering of ``attention-head`` (FA2 AttentionHead_v3{T, D, BN}): every tensor-core k-step of the head is one unit
(``verity_flock.ir_lower.tc_units``; ``fp.tc_dot16`` = ``AmpereBF16TcDot16`` on every encoding), 4T QK steps (``acc + q . k``)
and D per 16 visible keys PV steps (``acc + p . v``): the key and value words come from their committed rows, a key past T is
the zero word the IR pads with, the query is a public port, and the softmax (running max, MUFU.EX2, lane sums, rescaling,
MUFU.RCP of the sum, the output cast) is the native tail the verifier evaluates on the public words.

T is per instance: a statement covers the instances of one T (:func:`lowering` takes it; :func:`lowering_for_set` reads it
off a set whose instances share it). The one unit netlist is the same for every T; the leaf maps and the tail (the CUT line)
are T's, so the pin is per T."""
TEMPLATE = "attention-head"
UNIT = "attention-head/fa2-bf16/tc-step"
FRAME_SCHEMES = ("frame-v3/blake3-keyed",)


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


def frame_lowering(subcircuit, T: int):
    return lowering(subcircuit, T)


def supports(subcircuit, scheme):
    if int(subcircuit.params["D"]) % 16:
        return f"C-Flock lowers AttentionHead_v3 with D a multiple of 16, not D = {subcircuit.params['D']}"
    if scheme in FRAME_SCHEMES:
        return None
    return (f"C-Flock proves {subcircuit.id} under {', '.join(FRAME_SCHEMES)} (verity/flock-ir-frame/v3: tensor-core steps as units, "
            f"the softmax native); binding its rows under {scheme} is not wired yet")


def relation(subcircuit, scheme):
    return "irf:attention-head"


def stage(input_set, lo, hi, out_dir):
    """The frame-v3 instance file for ids [lo, hi) of a one-T set (``verity_flock.ir_frame.stage``): each pod stages its own."""
    from verity_flock import ir_frame

    return ir_frame.stage(lowering_for_set(input_set), input_set, lo, hi, out_dir)
