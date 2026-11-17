"""The point's lease and the CUDA toolkit on its node: the timed step's argv under the kit's lease of one card,
the refusal of a node that can't run `gpu-lease`, the lease's card as the host stamp names it, and the toolkit's
version as the node's nvcc says it."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from verity_bench import kit, steady_spec
from verity_bench.steady_cards import Asker, Refused

#: the exit code `_ask` gives for a command the node couldn't run at all, as a shell gives it
NOT_RUN = 127
#: the lease tool in the tree, which a node has on its path as `gpu-lease`
LEASE_SOURCE = "tools/research/src/research/pods/sh/gpu_lease.sh"


def lease_argv(spec: steady_spec.SteadySpec, inside: Sequence[str], on: Sequence[int] | None = None,
               cores: str | None = None) -> list[str]:
    """The timed step under the kit's lease (`kit.lease`) of one card for the spec's bound, with what a point's lease
    says beside it, before the command: the cards it may be (`on`, gpu-lease's `--on`: `cards_near`), the run's CPUs
    (`cores`, its slot's, gpu-lease's `--cores`: the lease declares them, so gpu-lease refuses or holds a lease whose
    CPUs and the live leases' pass the machine's CPU quota), preemptible, as a guest's GPU job must be (a preempted
    point has no passing run, so the next `run` launches it again), and the host memory the spec admits. Whether the
    machine throttled them while the row ran is the row's check `no_throttling`."""

    minutes = spec.admission.max_minutes
    if minutes is None:
        raise Refused(f"spec {spec.id!r} has no admission max_minutes, which bounds the point's lease")
    near = ["--on", ",".join(map(str, on))] if on else []
    pinned = ["--cores", cores] if cores else []
    memory = ["--mem-gb", str(spec.admission.mem_gb)] if spec.admission.mem_gb is not None else []
    *lease, command = kit.lease(1, minutes)
    return [*lease, *near, *pinned, "--preemptible", *memory, command, *inside]


def lease_tool(ask: Asker) -> None:
    """Refuses a node on which `gpu-lease` can't be run, before a statement is staged or fetched for a lease that could
    not be taken: the kit's lease calls it by its bare name."""

    tool = kit.lease(1, 1)[0]
    rc, text = ask([tool, "status"])
    if rc == NOT_RUN:
        raise Refused(f"`{tool}` can't be run on this node ({text.strip()[-200:]}), and the timed step runs under its "
                      f"lease of one card: put {LEASE_SOURCE} of this tree on the node's path as `{tool}`. A copy on "
                      "the node's own disk goes when the node is reset")


# ---- inside the lease ---------------------------------------------------------------------------------------------------------


def leased_card(stamp: Mapping[str, Any]) -> Mapping[str, Any]:
    """The kit's host stamp, once it is of the lease's one card: its name, its UUID and the driver's version."""

    gpus = stamp.get("gpus") or []
    if stamp.get("why") or len(gpus) != 1 or not stamp.get("driver"):
        raise Refused("the host stamp doesn't say what the lease's card is ("
                      + (stamp.get("why") or f"it names {len(gpus)} GPUs and the driver {stamp.get('driver')!r}")
                      + "): inside a gpu-lease of one card it is that card's, which gpu-lease names in GPU_LEASE_UUID")
    return stamp


def toolkit_on_node(ask: Asker) -> str:
    """The CUDA toolkit's version: what the nvcc on the path the spec's setup leaves says it is."""

    rc, text = ask(["nvcc", "--version"])
    found = re.search(r"release [\d.]+, V(\d+(?:\.\d+)+)", text) if rc == 0 else None
    if found is None:
        raise Refused("the nvcc on the path the spec's setup leaves doesn't say its version (exit "
                      f"{rc}: {text.strip()[-200:]!r}): the setup puts the toolkit the prover was built with first")
    return found.group(1)


def toolkit(spec: steady_spec.SteadySpec, ask: Asker) -> str:
    """The toolkit's version (`toolkit_on_node`), read before any statement is staged or fetched and any card is asked
    for, since it needs neither: a node at another toolkit is refused for nothing. The timed step reads it again inside
    the lease, beside the driver's, which only the lease's card gives."""

    read = toolkit_on_node(ask)
    if read != spec.cuda_toolkit:
        raise Refused(f"the node's CUDA toolkit is {read} (its nvcc), and spec {spec.id!r} says {spec.cuda_toolkit}: a "
                      "row at another toolkit is another point. Set the spec's `cuda_toolkit` to the node's, which times "
                      "its rows again there")
    return read
