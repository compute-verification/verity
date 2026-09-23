"""vLLM integration v2, Phase 2: the module-body query over a derived Program and the bridge to v1's identity encoding.

    program_view   a `verity.verification.query.Program` view of a derived Program: from its descriptor (verity.ir codec) or
                   from the Build's instance sequence (`instances.json.gz`: the same flat body, one row per Call, names inline)
    correspondence the per-Call runtime-correspondence annotations (owning module path, fx node, member hint, row) read off
                   the descriptor's `node_hints` / the instance names, and the Program-path -> implementation-path map
    module_body    `Q_module_body_v1(P, correspondence)`: units = module bodies
    v1_bridge      required values under Q_module_body_v1 spelt as v1 IDENTITY_KEY rows (+ the named residuals)
    compare        identity-set comparison against a v1 manifest, every difference classified
    cli            `python -m verity_vllm.query.cli build|build-global ... --out manifest.json`

Nothing in this package names a model or decides a member by a module-name or Definition-family table
(`tests/query/test_lint.py` holds that): what a Call produces comes from its Definition's output signature, where it lives from the
recorded correspondence, what crosses from the Program's own dataflow.  What v1 named that the structure cannot is a *named residual*
on the output, never a rule.
"""
