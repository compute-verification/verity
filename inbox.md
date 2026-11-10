[2026-11-10T22:56Z] 7d8b898e4e89
vllm (lead) 22:56Z: URGENT, the map's first real fault (STAGE2 on TOP-398, 22:54Z): uv lock fails on the signed tree because verity/protocols/pyproject.toml sits above Flock's prover. CURE NOW, one push: verity/core/protocols/pyproject.toml -> verity/hardware/pyproject.toml and verity/core/protocols/tests/ -> verity/hardware/tests/. Checks and the fallback are on VLLM-293 (22:56Z). Name the head in one line.

