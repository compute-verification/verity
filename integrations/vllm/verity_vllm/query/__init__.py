"""The module-body query over a derived Program and the bridge to v1's identity encoding.

    program_view   a `verity_vllm.query.units.CallGraph` view of a derived Program: from its descriptor (verity.ml.circuits codec) or
                   from the Build's instance sequence (`instances.json.gz`: the same flat body, one row per Call, names inline)
    correspondence the per-Call runtime-correspondence annotations (owning module path, fx node, member hint, row) read off
                   the descriptor's `node_hints` / the instance names, and the Program-path -> implementation-path map, in
                   `verity_vllm.correspondence.reader_for_query`
    module_body    `Q_module_body_v1(P, correspondence)`: units = module bodies; `module_body_boundary` = its crossing Values minus
                   the Program's literals (Calls that read no Value: the descriptor's, never acquired)
    required       required values under the query of record (Q_word v1: every Call boundary; Q_module_body_v1 for A/B) spelt as v1
                   IDENTITY_KEY rows; the partition of record (`partition_of`)
    manifest       the required-value manifest's format (`format`), its binding to what the runtime commits (`bind`) and
                   the check that a Commit used the manifest of record (`verify`)
    (the CLI)      `verity-vllm manifest build|build-global ... --out manifest.json`

Nothing in this package names a model or decides a member by a module-name or Definition-family table
(`tests/program/test_lint.py` holds that): what a Call produces comes from its Definition's output signature, where it lives from the
recorded correspondence, what crosses from the Program's own dataflow.  The committed tensors are bound to those identities by
structure (`verity_vllm.query.manifest.bind`), never by respelling an identity as a tap names it.

The gate's per-row records are `tests/query/gate/row<N>_*.json`.
"""
