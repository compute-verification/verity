"""Pluggable operand-commitment leaves for ``authentication = included-hash`` (campaign brief §2, lane leaf-iface).

``base.py`` is the contract (:class:`LeafScheme`), ``registry.py`` the name -> scheme table (``get("poseidon2")``,
``"hash"`` an alias), one module per scheme: ``poseidon2.py`` (the first plugin, moved from ``hashchain.py``; byte-identical
systems), ``dummy.py`` (test-only), ``blake3.py`` / ``ajtai.py`` (lanes blake3-leaf / ajtai-leaf).  ``conformance_test.py``
is what every scheme must pass.
"""
from __future__ import annotations

from .base import LeafScheme, WitnessHook
from .registry import (DEFAULT, by_schema, canonical, get, names, register, register_family, relation_name, split_relation_name,
                       suffix_of)

__all__ = ["DEFAULT", "LeafScheme", "WitnessHook", "by_schema", "canonical", "get", "names", "register", "register_family",
           "relation_name", "split_relation_name", "suffix_of"]
