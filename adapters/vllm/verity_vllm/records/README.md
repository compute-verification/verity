# verity_vllm.records

The records the adapter's stages hand each other. One stage writes a JSON record and a later stage reads it: the Build
writes a summary and each run of the Commit writes a record, and the Commit's verdict reads both. This package gives
each kind of record one typed class, so that a reader gets the fields its writer wrote, or a refusal that names the
file and the missing field.

It loads nothing else of the adapter, so every stage may import it. The package's docstring lists the record kinds,
and [`DESIGN.md`](DESIGN.md) says how each kind is read and which names other code imports from each module.
