[2026-11-11T03:51Z] 542ba912e22c
From circuits' coordinator, one line from infra's announcement, no change of order: after you merge or check out the moved main on your VM, run once: uv sync --all-packages --extra torch-cpu, before any local test or import. Jobs on a node already sync for themselves.

