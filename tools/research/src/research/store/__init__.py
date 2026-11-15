"""research.store: the evidence store (README.md in this directory is the contract).

Four objects (Tool, Derivation, Attempt, Artifact), two relations (Replica, Label).  This package holds the storage half:
canonical JSON and ids (`canon`, `ids`), the artifact/label envelopes (`model`), the local store and its layout (`local`), the
disposable sqlite catalog (`index`), remotes (`remote`, `remote_s3`), known kinds (`kinds`) and the `research data` CLI (`cli`).
Import from the owning module; nothing is re-exported here.
"""
