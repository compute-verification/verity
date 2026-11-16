"""Pod fleet: RunPod lifecycle, leases, the budgets guard and the pod-side idle guard."""

POD_SUFFIXES = ("-veritor-campaign", "-verity-campaign")


def strip_pod_suffix(name: str) -> str:
    for s in POD_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name
