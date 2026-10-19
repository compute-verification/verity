"""Pod fleet: RunPod lifecycle, the lease every pod carries (lease.py, sh/lease.sh), the one budgets guard (guard.py, budgets.py),
and the pod-side idle guard under sh/.

Promoted from veritor zk/campaign @ df7fbea3d (2026-09-21).
"""

# Name suffixes of this project's pods.  runpod.py `create` (standalone, spells its own) still appends the first; parsers accept all.
POD_SUFFIXES = ("-veritor-campaign", "-verity-campaign")


def strip_pod_suffix(name: str) -> str:
    for s in POD_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name
