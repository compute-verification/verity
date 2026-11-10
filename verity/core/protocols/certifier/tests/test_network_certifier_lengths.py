"""The network certifier with lengths rows (`WardenParams.lengths`, `warden`'s "The lengths rows"), driven tick by tick
on `test_network_certifier_warden`'s harness. Each test holds a schedule with lengths rows to one rule of that section
and the same schedule without them to what it did before they were a parameter."""

from __future__ import annotations

import dataclasses

import pytest

from verity.core.protocols.certifier.warden import (INTEGER_FIELDS, OPTIONAL_FIELDS, WardenParams)

#: a schedule whose cells hold an id request's head, and the same with lengths rows
PLAIN = WardenParams(cell_bytes=512, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                     ingress_buckets=3, egress_cells=9, every=2)
LEN = dataclasses.replace(PLAIN, lengths=1)
B, P, R, T, D, E = LEN.cell_bytes, LEN.payload, LEN.slots, LEN.window, LEN.delay, LEN.every
#: the default schedule's digest, PLAIN's, and PLAIN's under a response cap of 4, as each was before the lengths rows'
#: form was a parameter
DEFAULT_DIGEST = ("9a727d16f6c1f351d316465c101269c2a4666cbcdf1dbaaef01579fd7e72315a"
                  "ae2be1451412c6328d9f6688f534fb8bbe5416bbaeb5cfc0cae48597c5fd0d50")
PLAIN_DIGEST = ("b064f2c3e90f5059b7393bcbca02b7b94d514041946423a253fa1cca6de11ee4"
                "1898e2e0368e1a03d9e5f8a910fc91ac2f8154924ccd7f669f8b0f4d99964808")
CAPPED_DIGEST = ("6dc8692fb13c37e45ce4d4229a8377931b434ff5c72fd243aeb2d0596dc0226e"
                 "b987b7a37d81a0baeed20ac5eb52ee614b991797a45e85c3fba8916f34361c22")


# --- the mode ----------------------------------------------------------------------------------------------------


def test_the_lengths_rows_form_is_a_parameter_no_other_schedule_hashes():
    """A schedule without lengths rows hashes what it hashed before they were a parameter (the integer fields, the
    egress form, the kinds and the reasons), so its digest, and every head and certificate that binds it, is unchanged;
    the form enters the digest of a schedule that has it, and is no integer field of the schedule the period check
    reads."""
    assert WardenParams().lengths is None and PLAIN.lengths is None
    assert WardenParams().digest.hex() == DEFAULT_DIGEST and PLAIN.digest.hex() == PLAIN_DIGEST
    assert LEN.digest != PLAIN.digest and dataclasses.replace(LEN, lengths=None) == PLAIN
    assert OPTIONAL_FIELDS == ("max_tokens_cap", "lengths") and not set(OPTIONAL_FIELDS) & set(INTEGER_FIELDS)
    assert LEN.reasons == PLAIN.reasons
    capped = dataclasses.replace(PLAIN, max_tokens_cap=4)
    assert capped.digest.hex() == CAPPED_DIGEST
    assert len({PLAIN.digest, LEN.digest, capped.digest, dataclasses.replace(capped, lengths=1).digest}) == 4


def test_a_schedule_with_lengths_rows_is_one_whose_releases_fit_them():
    """The form is 1 or unset; with it the egress form is id, ingress ends by the delay, a connection ends within a
    window of its admission, its egress cells count in a u16 and a cell holds an id request's head and first prompt
    field. K, a row's slots, is the most releases a slot stride can hold."""
    for bad in (0, 2, True, "1", 1.0):
        with pytest.raises(ValueError, match="lengths is 1"):
            dataclasses.replace(PLAIN, lengths=bad)
    for change, why in (({"egress": "bytes"}, "egress form is id"), ({"ingress_buckets": D + 1}, "at most the delay"),
                        ({"window": 8, "every": 1}, "ends within a window"), ({"delay": 13}, "ends within a window"),
                        ({"egress_cells": (1 << 16) + 1, "window": 1 << 16}, "count in a u16"),
                        ({"cell_bytes": 256}, "no room for an id request|doesn't fit its first cell")):
        dataclasses.replace(PLAIN, **change)
        with pytest.raises(ValueError, match=why):
            dataclasses.replace(LEN, **change)
    assert dataclasses.replace(LEN, delay=12, window=16, egress_cells=9).lengths == 1
    assert LEN.stride_releases == 2 == LEN.lanes
    # every = 1: a bucket releases at most a request a lane; long strides: a lane's releases are D + 2 − ingress_buckets
    # apart, and a connection holds its lane D + 1 buckets
    for change, K in (({"every": 1}, 2), ({"every": 4}, 2), ({"every": 8}, 4), ({"every": 16}, 8),
                      ({"every": 16, "ingress_buckets": 1}, 6), ({"every": 8, "lanes": 5, "ingress_buckets": 5}, 15)):
        assert dataclasses.replace(LEN, **change).stride_releases == K, change
