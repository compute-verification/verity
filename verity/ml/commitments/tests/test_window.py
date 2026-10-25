"""The window-domain rule (`verity.ml.commitments.window`): a pinned domain id, and every field it binds changes it."""
from __future__ import annotations

from verity.ml.commitments.window import WINDOW_DOMAIN, window_domain

FIELDS = {"program": "aa" * 64, "partition": "bb" * 64, "port": "m0/x", "schema": "hm96-sha512/row/v1", "leaves": 40,
          "run": "cc" * 64}
PINNED = ("34842d71d8cc5b3d8e6a2b9178bbf727e17878fb82f3924d235c526d03fb88fd"
          "4c5ac5fbf94e871b8841cc24381a8d90c1f6263bc17869987bd3c7194a9ce5ce")


def test_the_window_domain_is_pinned():
    assert WINDOW_DOMAIN == "verity/one-stage/served-domain/v1"
    assert window_domain(**FIELDS) == PINNED


def test_every_field_is_bound():
    other = {"program": "ab" * 64, "partition": "ba" * 64, "port": "m0/y", "schema": "hm96-sha512/row-seg/v1", "leaves": 41,
             "run": "cd" * 64}
    for k, v in other.items():
        assert window_domain(**{**FIELDS, k: v}) != PINNED, k
