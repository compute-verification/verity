[2026-11-12T06:59Z] f09f844f8002
ci 10:5x PM PST: infra filed your work as Linear CI-526 (the full list of errored and faulted test files, read from the executor's db at 10:42 PM PST, and how to read them again on cpu-7: graph2.db read-only, 'select id, state, error from tasks where submission >= 's20261112T05' and state in ('faulted','errored')'). Name CI-526 in your PR bodies (not in the tree). Done = a submission run to its end has no faulted or errored task of the graph's own.

