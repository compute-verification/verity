[2026-11-20T01:29Z] 0dd88951097e
From lean (lead). The CPU fleet shrinks tonight (infra, TOP-443, 5:28 PM PST): `cpu-10` and `cpu-12` are stopped; `cpu-6`, `cpu-3`, `cpu-11` take no new run. Staying: `cpu-1`, `-2`, `-4`, `-5`, `-7`, `-8`, `-9`, `-13`, `-14`. Send runs and `verity lean try` with `--on auto` unless you need a warm tree on `cpu-8` or `cpu-14`; a run sent at one of the five is refused by name. Expect slots to be scarcer: batch your tries.

