"""Pod fleet: RunPod lifecycle, budget cap, memory guard, and the pod-side shell guards under sh/.

Promoted verbatim from veritor zk/campaign @ df7fbea3d (deadline.sh, rearm.sh from the r17 deadman pod, 2026-09-21).
budget_cap.py and mem_guard.py import runpod.py as a sibling via sys.path so the same files run standalone on a pod.
"""

# Name suffixes of this project's pods.  runpod.py `create` (standalone, spells its own) still appends the first; parsers accept all.
POD_SUFFIXES = ("-veritor-campaign", "-verity")


def strip_pod_suffix(name: str) -> str:
    for s in POD_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name
