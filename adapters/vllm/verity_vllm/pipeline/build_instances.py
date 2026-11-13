"""A derive dir's instance sequence (`instances.json.gz`, cb-a/instance-sequence/v2): the Program's flat body as (spec, argument
items) rows, the comparator's input, streamed to a gzip file.  The rows and the writers are `program.instances_file`'s: the text is
`json.dumps({**head, "rows": list(_instance_rows(program))})`, and the Build writes a word Program's from its logs
(`write_program_instances`), not row dict by row dict.

A Boolean Program that names its word Program (`program.frontend.boolean.derived`: σ-checked node by node, so every Call, node index and
word leaf is the word Program's) is written through that link: row k is word node k -- the word Definition its Boolean Definition
computes (`verity.circuits.boolean.trace.word_of`, checked against the word node), its operand runs over word leaves, the word
parameters -- with ``bits``, the Boolean Definition the Call is of.  The header says so (``ir: "boolean"``) and keeps the Boolean
Program's digest.  Every reader of the rows (the manifest and its word check, the replay, the serving rows) reads the word
Program's units, boundaries and committed values, whichever IR the Program is in."""
from __future__ import annotations

import os

from verity_vllm.program.instances_file import (
    INSTANCES_GZ_LEVEL,  # noqa: F401 (read through here)
    instance_rows,
    write_program_instances,
)
from verity_vllm.program.instances_file import write_instances as _write_instances  # noqa: F401 (read through here)
from verity_vllm.program.instances_file import (
    write_instances_file as _write_instances_file,
)

__all__ = ["write_instances"]


def _instance_rows(program):
    """`instance_rows(program)`, and a Boolean Program's rows through its word link (`_word_rows`)."""
    word = getattr(program, "word", None)
    if word is not None:
        return _word_rows(program, word)
    return instance_rows(program)


def _word_rows(program, word):
    """A Boolean Program's rows through its word link: word node k's spec, form, name and runs, with ``bits`` = Boolean node k's
    Definition.  A Boolean Definition whose word is not the Definition word node k Calls is refused by name."""
    from verity.circuits.boolean.trace import word_of
    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import CompositeDefinition, bind

    from verity_vllm.program.compact import file_items_from_refs
    links: dict[str, str] = {}
    for k, (b, w) in enumerate(zip(program.fn.body.nodes, word.fn.body.nodes, strict=True)):
        bits, spec = descriptor_id(b.fn), descriptor_id(w.fn)
        link = links.get(bits)
        if link is None:
            if b.fn is w.fn or bits == spec:
                link = spec
            else:
                wf = word_of(b.fn)
                link = descriptor_id(bind(wf) if isinstance(wf, CompositeDefinition) else wf)
            links[bits] = link
        if link != spec:
            raise ValueError(f"row {k}: {bits} names the word Definition {link}, and the word Program's node {k} Calls {spec}")
        yield {"i": k, "spec": spec, "bits": bits, "form": w.form, "name": w.name, "args": [file_items_from_refs(a.refs) for a in w.args]}


def _instance_sequence(program) -> list[dict]:
    return list(_instance_rows(program))


def check_rows(body, n: int) -> list[list[int]]:
    """The row and output leaf `[i, lo, lo + 1]` of each of `body`'s last `n` returned leaves, in order: where the Program's checks
    output (`program.predict.common.with_checks`, the returned tuple's last member) is computed."""
    from verity.circuits.refs import runs
    leaves: list[tuple[str, int, int]] = []
    for space, i, lo, hi in reversed(runs(body.ret)):
        take = min(hi - lo, n - len(leaves))
        leaves += [(space, i, k) for k in range(hi - 1, hi - 1 - take, -1)]
        if len(leaves) == n:
            break
    if len(leaves) != n or any(s != "n" for s, _i, _k in leaves):
        raise ValueError(f"the Program's last {n} returned leaves are not {n} checks computed by its rows: {leaves[::-1][:4]}")
    return [[i, k, k + 1] for _s, i, k in reversed(leaves)]


def write_instances(out: str, program, digest: str, corr_digest: str | None, supplied: dict, *, names: dict[int, str | None] | None = None,
                    params: list[list[str]] | None = None, checks: dict | None = None) -> None:
    """A derive dir's `instances.json.gz`: the Program's instance sequence (`_instance_rows`) under its digests.  `names` (by Call)
    and `params` stand in for the Program's own Call names and parameters, which no digest covers (`predict.aot.emit`: as its
    trace named them).  `checks` (`BuildResult.checks`) is written with the row and leaf of each check (`check_rows`) under ``at``,
    never ``rows``, which ends the header for a reader of it alone (`program_view._read_instance_header`); the replay evaluates
    each and requires it to be `accept` (`check.replay.checks`).  A Boolean Program's rows are row dicts (`_word_rows`),
    which `RowTexts` doesn't write."""
    word = getattr(program, "word", None)
    body = (word or program).fn.body
    if params is None and hasattr(body, "params"):
        params = [[n, str(t)] for n, t in body.params]
    head = {"schema": "cb-a/instance-sequence/v2", "digest": digest, "correspondence_digest": corr_digest, "params": params,
            **({"ir": "boolean"} if word is not None else {}), **supplied,
            **({"checks": {**checks, "at": check_rows(body, len(checks["of"]))}} if checks else {})}
    path = os.path.join(out, "instances.json.gz")
    if word is None:
        write_program_instances(path, head, program, names=names)
        return
    rows = _word_rows(program, word)
    if names is not None:
        rows = ({**r, "name": names.get(r["i"])} for r in rows)
    _write_instances_file(path, head, rows)
