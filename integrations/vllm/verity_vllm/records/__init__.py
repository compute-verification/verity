"""Stage records, typed: one frozen dataclass per record kind one stage writes and another reads.

A record's `load` reads the JSON as its writer writes it and refuses a missing required field, naming the file and the field
(`fields.MissingField`).  A field a writer may leave out is `X | None`, with the reason beside it; nothing is defaulted.  The
package imports nothing else from `verity_vllm`, so every stage may import it.

* `required_manifest` -- the required-value manifest (`verity-sweep/required-manifest/v1`), as the replay's coverage reads it.
"""
