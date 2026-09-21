"""Backend-independent verification plans and obligations over Verity programs.

Two layers, both promoted from veritor (``interface/``, ``protocol/proofs/``) at eb865ce2:

- ``typed_obligation`` is the ``typed-obligation/v0`` interface object: family, types, references, roots,
  subcircuits, obligations -- the smallest backend-independent statement of "these exact relations over these
  committed values must hold".  ``validate`` is its checker, ``families`` resolves the pinned gate sets,
  ``binding`` derives the 65 public bytes a proof binds (``h_O || h_L || verdict``).
- ``lowering`` maps the interface object onto the flat ``statement`` (obligations over positions in commitments,
  a ``programs`` table) that ``wire`` encodes and every backend consumes; ``transparent`` is the reference
  checker for that statement (the proof *is* the openings), ``schema``/``packed`` its witness formats.

``plan`` is the IR-level selection layer (verification plans, sampled gates, the lowering adapter);
``conformance`` is the Python column of the cross-language vector suite in ``fixtures/typed-obligation-v0``.

Import from the owning submodule.  This package depends on ``verity.ir``, ``verity.ml`` (for the pinned
tensor-core family) and ``verity.commitments``; never on a backend.
"""
