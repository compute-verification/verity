# PoUB's devices (`verity/protocols/poub/device`)

What runs PoUB on the auditor's side of a deployment. It holds one directory.

## Interface

Nothing is imported from this directory itself: its `__init__.py` is empty.

- `live_verifier/` is the verifier's auditor host: `host.py`, the service and gate that run PoUB's verifier and the
  memory challenger's parties on a host the developer doesn't control. `live_verifier/README.md` states its interface.
