"""A derive dir's instance sequence (`instances.json.gz`, cb-a/instance-sequence/v2): the Program's flat body as (spec, argument
items) rows, the comparator's input, streamed to a gzip file.  The rows and the writers are `program.instances_file`'s: the text is
`json.dumps({**head, "rows": list(_instance_rows(program))})`, and the Build writes it from the Program's logs
(`write_program_instances`), not row dict by row dict."""
from __future__ import annotations

import os

from verity_vllm.program.instances_file import INSTANCES_GZ_LEVEL  # noqa: F401 (read through here)
from verity_vllm.program.instances_file import instance_rows as _instance_rows
from verity_vllm.program.instances_file import write_instances as _write_instances  # noqa: F401 (read through here)
from verity_vllm.program.instances_file import write_instances_file as _write_instances_file  # noqa: F401 (read through here)
from verity_vllm.program.instances_file import write_program_instances


def _instance_sequence(program) -> list[dict]:
    return list(_instance_rows(program))


def write_instances(out: str, program, digest: str, corr_digest: str | None, supplied: dict, *, names: dict[int, str | None] | None = None,
                    params: list[list[str]] | None = None) -> None:
    """A derive dir's `instances.json.gz`: the Program's instance sequence (`_instance_rows`) under its digests.  `names` (by Call)
    and `params` stand in for the Program's own Call names and parameters, which no digest covers (`predict.aot.emit`: as its
    trace named them)."""
    if params is None and hasattr(program.fn.body, "params"):
        params = [[n, str(t)] for n, t in program.fn.body.params]
    write_program_instances(os.path.join(out, "instances.json.gz"),
                            {"schema": "cb-a/instance-sequence/v2", "digest": digest, "correspondence_digest": corr_digest, "params": params, **supplied},
                            program, names=names)
