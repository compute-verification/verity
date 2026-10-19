"""A derive dir's instance sequence (`instances.json.gz`, cb-a/instance-sequence/v2): the Program's flat body as (spec, argument
items) rows, the comparator's input, streamed to a gzip file."""
from __future__ import annotations

import gzip
import json
import os


def _instance_rows(program):
    """The flat body as (spec, argument items) rows — the comparator's unit (compare.py), one at a time.  An argument is stored in the
    progressions form (instances v2, `compact.file_items_from_refs`, the fold writer's): O(rows) for a scattered Strided where `runs()`
    is O(leaves) — the GLM-5.3 MLA-absorb transposed-weight read is ~1e6 leaves per operand.  Every reader expands back to the
    per-leaf run lists, bit-identically (`compact.expanded_file_args` at `program_view.iter_instance_rows` and the raw doc loaders)."""
    from verity.ml.circuits.codec import descriptor_id
    from verity_vllm.program.compact import file_items_from_refs
    for k, n in enumerate(program.fn.body.nodes):
        yield {"i": k, "spec": descriptor_id(n.fn), "form": n.form, "name": n.name, "args": [file_items_from_refs(a.refs) for a in n.args]}


def _instance_sequence(program) -> list[dict]:
    return list(_instance_rows(program))


def _write_instances(f, head: dict, rows, chunk: int = 4096) -> None:
    """`json.dump({**head, "rows": list(rows)}, f)`, the same text, written a chunk of rows at a time with `json.dumps` (its C encoder;
    `json.dump` to a file always runs the Python one) and without the rows list."""
    f.write(json.dumps(head)[:-1] + (', "rows": [' if head else '"rows": ['))
    batch, first = [], True
    for r in rows:
        batch.append(r)
        if len(batch) == chunk:
            f.write(("" if first else ", ") + json.dumps(batch)[1:-1])
            batch, first = [], False
    if batch:
        f.write(("" if first else ", ") + json.dumps(batch)[1:-1])
    f.write("]}")


#: `instances.json.gz`'s gzip level.  Its readers parse its text, and no digest or published output reads its bytes
#: (`research_outputs.NEVER`).  A long context's text is mostly the attention Calls' key and value runs (B1 4096/512: 8.6 GB), which
#: level 6 compresses in under a third of level 9's CPU time, 2% larger, and which inflates as fast at either level.
INSTANCES_GZ_LEVEL = 6


def _write_instances_file(path: str, head: dict, rows) -> None:
    with gzip.open(path, "wt", compresslevel=INSTANCES_GZ_LEVEL) as f:
        _write_instances(f, head, rows)


def write_instances(out: str, program, digest: str, corr_digest: str | None, supplied: dict, *, names: dict[int, str | None] | None = None,
                    params: list[list[str]] | None = None) -> None:
    """A derive dir's `instances.json.gz`: the Program's instance sequence (`_instance_rows`) under its digests.  `names` (by Call)
    and `params` stand in for the Program's own Call names and parameters, which no digest covers (`predict.aot.emit`: as its
    trace named them)."""
    rows = _instance_rows(program)
    if names is not None:
        rows = ({**r, "name": names.get(r["i"])} for r in rows)
    if params is None and hasattr(program.fn.body, "params"):
        params = [[n, str(t)] for n, t in program.fn.body.params]
    _write_instances_file(os.path.join(out, "instances.json.gz"),
                          {"schema": "cb-a/instance-sequence/v2", "digest": digest, "correspondence_digest": corr_digest, "params": params, **supplied},
                          rows)
