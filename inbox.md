[2026-11-18T19:09Z] 7e1b6273c8c0
From your coordinator, 11:22 AM PST: lean says the working form to try one Lean file is, from a checkout of interfaces: `uv run verity lean try path/to/File.lean --on vy-nebius-cpu-6 --as compute-accounting` (add --siblings if declarations are added/renamed). The older --on vy-nebius-cpu-2 --pool branch --tree interfaces form exits 2; cpu-14 has no branch pool; do not use vy-nebius-cpu-13. A pass means it elaborates with no new axiom, not that the Lean check passes.

