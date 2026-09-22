"""research.store: the evidence store (README.md in this directory is the contract).

Provenance half (lane store-prov): tool, tools_registry, derivation, attempt, cli_prov (+ research.env for conditions()).
Storage half (lane store-core): canon, ids, model, local, index, remote, remote_s3, kinds, cli.
Modules lazy-import across the halves inside functions so each half imports on its own.
"""
