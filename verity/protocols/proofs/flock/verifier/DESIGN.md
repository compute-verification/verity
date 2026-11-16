# Flock's verifier: design

The packages it requires and the layers of its libraries. `README.md` says what the verifier is and how to build and
run it.

## Requirements

It fetches nothing. It requires, by path, the three base packages every Lean package of the tree loads:
`verity/circuits/base` (`Base`: bytes, SHA-512, canonical JSON and the JSON parser), `circuits` (`Circuits`: the
program format, its decoder, the partition and annotations) and `commitments` (`Commitments`: frame-v3 and the sampling
of proof blocks). Their declarations keep the namespace `Flock`. A module that moved there keeps its old name here as a
re-export, one `public import` and nothing else (`Flock/Bytes.lean` re-exports `Base.Bytes`), until every importer names
the base module. A job that builds this package outside the checkout copies it with those three
(`verity/protocols/proofs/flock/prover/pod/lake_closure.py`).

## Layers

The libraries are layers, each loading only the ones before it: the verifier (`Flock`'s modules outside the next two),
the ZK server (`FlockZkServer`, which imports `Flock.ZkServer` and the modules under it) and the recursion (`FlockRec`,
which imports the `Flock.Rec*` modules, with `VStar`). `tests/verity/protocols/proofs/flock/prover/test_lean_layers.py` gives each module its layer and fails on an import
of a layer above; the imports that still point up are its `UP`, each with what removes it.
