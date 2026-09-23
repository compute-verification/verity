"""Operand-commitment leaf schemes behind one interface (the leaf campaign, 2026-09-23).

A *leaf* is the row digest the committer puts under the SHA-256 tree (``hashauth.py``) and the in-circuit gadget the
hashed relation (``hashchain.compose``) evaluates on its private operand bits so the two agree.  :mod:`.base` is the
contract, :mod:`.registry` resolves a name (``poseidon2`` / ``hash``, ``ajtai-n256``, ...) to a scheme.
"""
