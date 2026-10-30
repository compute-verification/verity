"""N2's head check (`head_check`) on the service's own link-windows: its AND counts, each part against its reference on
honest heads and on heads wrong in one input, and the whole Definition against its reference."""

from dataclasses import replace

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.identity import tagged_sha512
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier.certificate import RECORD_TAG, CertifiedRecord, certificate_digest, parameters_digest
from verity.core.protocols.certifier.grid import Reason, Status

from test_network_certifier_service import PARAMS, Run, honest_occupancy


def _checks(lw, opening):
    return lw.head_checks(opening, bytes.fromhex(lw.root(opening)))


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    """Head checks of the service's link-windows under the honest profile, by name: (instance, the part that refuses it
    or None). The honest heads of both ends in each window, and heads each wrong in one input; an edited head is
    committed under its own root and its certificate's signed digest is its own, so only the edit can refuse it."""
    run = Run(tmp_path_factory.mktemp("heads"), occupancy=honest_occupancy())
    lw, honest = run.lw, [run.honest(w) for w in range(3)]
    out = {f"{end} {w}": (i, None) for w, o in enumerate(honest) for end, i in zip(("egress", "ingress"), _checks(lw, o))}
    e0, i0 = _checks(lw, honest[0])

    def edited(d, **fields):
        recs = list(honest[0].records)
        r = recs[d]
        head = replace(r.head, **fields)
        recs[d] = CertifiedRecord(head, r.salt, replace(r.certificate, digest=certificate_digest(r.salt, head.digest())))
        return _checks(lw, replace(honest[0], records=tuple(recs)))[d]

    swapped = replace(honest[0], records=honest[0].records[::-1])
    return out | {
        "another window": (replace(e0, window=1), "fields"),
        "another link": (replace(e0, link=b"responsez"), "fields"),
        "the ingress head at the egress leaf": (replace(_checks(lw, swapped)[0], profile=i0.profile), "fields"),
        "another parameters digest": (replace(e0, grid=HC.grid_bytes(PARAMS, parameters_digest(PARAMS))), "fields"),
        "an incomplete status": (edited(0, status=Status(Reason.OVERFLOW)), "fields"),
        "another profile row": (replace(e0, profile=tuple((n + 1) % (PARAMS.allocation + 1) for n in e0.profile)),
                                "counts"),
        "counts other than the profile's": (edited(1, counts=(0,) * PARAMS.buckets), "counts"),
        "another root": (replace(e0, record=_checks(lw, honest[1])[0].record), "opening"),
        "another salt": (replace(e0, salt=i0.salt), "opening"),
        "another path": (replace(e0, path=e0.path[::-1]), "opening"),
        "another signed digest": (replace(e0, signed=i0.signed), "signed"),
        "another certifier salt": (replace(e0, salt_c=i0.salt_c), "signed"),
        "a certificate that signs the unsalted record digest": (replace(e0, signed=tagged_sha512(RECORD_TAG, e0.head)),
                                                                "signed"),
        "an unset deadline bit": (edited(0, deadline=False), None),
        "another frames root": (edited(1, root=bytes(64)), None),
    }


@pytest.fixture(scope="module")
def bare(tmp_path_factory):
    """The honest heads of a link with no profile: COUNTS = 0."""
    run = Run(tmp_path_factory.mktemp("bare"), occupancy=(None, None))
    return [i for w in range(3) for i in _checks(run.lw, run.honest(w))]


def _parts(inst):
    """The fields' and the counts' circuits on the instance, each evaluated alone."""
    sh = HC.Shape(len(inst.link), inst.T)
    a = {n: np.asarray([bits], dtype=np.uint8) for (n, _), bits in zip(inst.unit.signature()[0], HC.arguments(inst))}
    head = a["head"]

    def out(fn, *args):
        return bool(np.asarray(evaluate_bits(fn, *args)).reshape(-1)[0])

    return {"fields": out(bind(HC.NetworkHeadFields, L=sh.L), head[:, :8 * sh.CNT], a["link"], a["direction"],
                          a["window"], a["grid"]),
            "counts": out(bind(HC.NetworkHeadCounts, T=sh.T), head[:, 8 * sh.CNT:8 * sh.ROOT], a["profile"])
            if inst.counts else True}


def test_the_count_check_is_32t_minus_1_ands_and_the_fields_736_plus_8l_minus_1():
    for T in (1, 4, 600):
        assert HC.ands(bind(HC.NetworkHeadCounts, T=T)) == 32 * T - 1
    for L in (1, 9, 40):
        assert HC.ands(bind(HC.NetworkHeadFields, L=L)) == 736 + 8 * L - 1


def test_each_part_is_its_reference_and_each_wrong_input_is_refused_where_it_belongs(cases, bare):
    for name, (inst, refused) in cases.items():
        want = HC.reference_parts(inst)
        assert _parts(inst) == {k: want[k] for k in ("fields", "counts")}, name
        assert [k for k, ok in want.items() if not ok] == ([] if refused is None else [refused]), name
    assert all(HC.reference(i) and not i.counts for i in bare)


def test_a_head_of_another_length_is_no_instance_of_the_unit(cases):
    inst, _ = cases["egress 0"]
    with pytest.raises(ValueError, match="the head is 181 bytes"):
        HC.reference(replace(inst, head=inst.head[:-1]))
    with pytest.raises(ValueError, match="argument head has"):
        HC.arguments(replace(inst, head=inst.head[:-1]))


@pytest.mark.slow
def test_the_head_check_is_its_reference(cases, bare):
    """The whole Definition on every case, with and without a profile, against its reference, and its ANDs by part at the
    deployed T = 600. What the parts' tests can't reach is the opening's wiring: the head's bytes into its row digest,
    its salt into hm96's leaf, and the leaf's position (direction, 1) in N2's tree; and the signed digest's: the head's
    bytes into its record digest, and that digest and the certifier's salt into the digest the certificate signs.
    Building and compiling the commitment gadgets' Boolean bodies takes about 75 s, once."""
    insts = [i for i, _ in cases.values()]
    assert HC.evaluate(*insts) == [int(HC.reference(i)) for i in insts]
    assert HC.evaluate(*bare) == [1] * len(bare)
    size = HC.size(HC.unit(9, 600, True))
    assert (size["fields"], size["counts"], size["other"]) == (807, 19_199, 3)
    assert size["opening"] == HC.ands(bind(HC.NetworkHidingRead, N_BITS=8 * HC.Shape(9, 600).HL, D=2)) > 32 * 600
    assert size["signed"] == (HC.ands(bind(HC.NetworkRecordDigest, HL=HC.Shape(9, 600).HL))
                              + HC.ands(bind(HC.NetworkSignedDigest))) == 1_333_292
