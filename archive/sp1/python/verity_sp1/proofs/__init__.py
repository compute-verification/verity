"""verity_sp1.proofs: backend-independent single statements over Verity programs, and how a backend is judged on them.

Formerly ``verity.verification`` (item 25).  Two layers, both promoted from veritor (``interface/``,
``protocol/proofs/``) at eb865ce2:

- ``typed_obligation`` is the ``typed-obligation/v0`` interface object: family, types, references, roots,
  subcircuits, obligations -- the smallest backend-independent statement of "these exact relations over these
  committed values must hold".  ``validate`` is its checker, ``families`` resolves the pinned gate sets,
  ``binding`` derives the 65 public bytes a proof binds (``h_O || h_L || verdict``).
- ``lowering`` maps the interface object onto the flat ``statement`` (obligations over positions in commitments,
  a ``programs`` table) that ``wire`` encodes and every backend consumes; ``transparent`` is the reference
  checker for that statement (the proof *is* the openings), ``schema``/``packed`` its witness formats.
  ``lowering.lower_subcircuit`` is the IR-level adapter: one IR subcircuit -> its ``typed-obligation/v0/lowering``
  section.

Verification plans and sampling are a protocol, ``verity/core/protocols/sampled_proofs`` (decision 41).  ``conformance`` is the Python
column of the cross-language vector suite in ``fixtures/typed-obligation-v0``.

Import from the owning submodule.  This package depends on ``verity.ml.circuits``, ``verity.ml.circuits.ops`` (for the pinned
tensor-core family) and ``verity.ml.commitments``; never on a backend.
"""
