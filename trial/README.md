# Layout trial

A toy of the proposed layout. Run an epoch:

~~~bash
uv sync && uv run python -m toy_serving [--integrity replay|sp1] [--mode honest|lazy|rollout] [--no-space] [--k K] [--verbose]
lake build    # the Lean beside the Python
~~~

## Layers

Imports point down only, in Python and in Lean:

~~~text
verity/protocol        the composition: accounting, compliance, configuration, epoch
verity/subprotocols    each standalone: sampled_proofs, pous, pouw, network_warden, nci; never one another
verity/primitives      commitments, randomness, the by-name registry
verity/circuits        what a circuit is and computes
~~~

`integrations/` sit on top and play the developer (`verity.protocol.epoch.Developer`).

Each subprotocol states in its own terms what it takes and what it establishes (its profile). The protocol
composes them, and owns everything between them: phase order, the shared weights commitment, each one's challenger,
cross-subprotocol refusals, and the δ budget.

## Conventions

- **Lean sits beside the Python it states.** `verity/circuits/Circuit.lean` sits beside `circuit.py`. There is one
  Lake package at the root (`lakefile.toml`), and a module's name is its path. The wheel excludes `*.lean` and
  `.lake`.
- **`__init__.py` holds only a docstring.** Code goes in named modules, so the files can be dropped (making the
  folders namespace packages) once `packages/verity` is gone and nothing else named `verity` is on the path.
- **Heavy constructions have their own `pyproject.toml`.** It sits beside the source, declares the dependencies and
  registers the construction under the `verity.constructions` entry points (`sampled_proofs/sp1/`). Light ones are
  listed in their subprotocol (`SHIPPED`).
