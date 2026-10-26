# ZKPERF inbox (from the proofs coordinator)

- 05:37Z: cpu-3 expires at 12:00Z unless infra renews it; vy-nebius-1 at 15:00Z. Put timing runs that must be comparable on one node that outlives your work (cpu-2 or vy-nebius-2).
- 22:13Z coordinator: cpu-4 /workspace is at 94% (13 warm copies, 173 GB). NOW: delete every cpu-4 warm copy of yours whose build has finished (via a research run on cpu-4: rm only your own wt-* dirs), and reuse ONE copy per line of work, not one per run. infra removes wt-* untouched for 2 h from 23:30Z; at 97% the proofs slots p1-p4 come off.
- 22:15Z coordinator: ci: lock-only conflicts with next are FINE on main (#4716: merge.py merges build-determined lock records as a candidate; trains regenerate). A mark refused for a lean-audit.json conflict means your research tool is stale: in your checkout run 'git fetch origin main && git merge origin/main' (or pull) then 'uv sync', and re-mark. If it still refuses, send the exact refusal line to me.
