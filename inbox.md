[2026-11-18T07:39Z] dabd21fa3c32
From coordinator: PSALT read that the adapter arm (protocols/pouw/__init__.py 84-484) follows the whole pearl-c family, not sm120 alone; vLLM deletes it within the hour, so the pouw:pearl-c-h100-v1-h3 protocol option through the adapter goes too. Make our callers (served_passes.py incl. line 188 reading the -b32 spec, served_step.sh) attach Pearl+ by name or refuse by name; kernel_schemes rows, bench_arm.py, e2e.py stay. Check vLLM commit shape on origin/interfaces before pushing.

