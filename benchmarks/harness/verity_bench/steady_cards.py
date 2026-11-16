"""The cards of a steady-state point's node (`steady_node`): which cards are on the NUMA node of the run's CPUs, which
its lease is asked of (`cards_near`, `booked`); the NUMA node of the lease's card (`card_numa_node`), which its row
keeps; and what else is on the node at a row's start and end (`snapshot`)."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: where a node lists each NUMA node's CPUs (`nodeN/cpulist`) and each PCI device's NUMA node (`ADDRESS/numa_node`), and
#: the query that gives each card's index, as gpu-lease counts them, with its PCI address
NODES, DEVICES = Path("/sys/devices/system/node"), Path("/sys/bus/pci/devices")
CARDS = ("nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader")

#: a command's exit code and what it printed
Asker = Callable[[Sequence[str]], tuple[int, str]]


class Refused(ValueError):
    """The point can't be timed as its spec says; nothing was timed."""


def cards_near(cpus: Sequence[int], ask: Asker) -> dict[str, Any]:
    """The cards on the NUMA node of the run's CPUs (`cards`, by index), which the point's lease is asked of: a slot of
    the node gives a run its CPUs and no card, and a card across the machine from its provers' CPUs is another
    measurement. `cards` is None, with `why`, where the node doesn't say: the run's CPUs aren't on one NUMA node it
    lists (of a timed row, the check `one_numa_node` says so), or no card says it is on that node (a machine with one
    node, or a VM, says none). The lease then takes any card."""

    mine, held = set(cpus), []
    for listed in sorted(NODES.glob("node[0-9]*/cpulist")):
        spans = [span.split("-") for span in listed.read_text().strip().split(",") if span]
        if any(mine & set(range(int(span[0]), int(span[-1]) + 1)) for span in spans):
            held.append(int(listed.parent.name[4:]))
    if len(held) != 1:
        return {"numa_nodes": sorted(held), "cards": None, "why": f"the node lists {len(held)} NUMA nodes for the run's CPUs"}
    where = card_nodes(ask)
    if isinstance(where, str):
        return {"numa_nodes": held, "cards": None, "why": where}
    if not (cards := sorted(index for index, node in where.items() if node == held[0])):
        return {"numa_nodes": held, "cards": None, "why": f"none of the node's {len(where)} cards says it is on NUMA node "
                                                          f"{held[0]}"}
    return {"numa_nodes": held, "cards": cards}


def card_nodes(ask: Asker) -> dict[int, int] | str:
    """The NUMA node the kernel says each card is on (-1 where it says none, as a VM does), by the card's index as
    gpu-lease counts them; or why the cards can't be listed."""

    rc, text = ask(list(CARDS))
    if rc:
        return f"`{' '.join(CARDS)}` exited {rc}"
    where: dict[int, int] = {}
    for line in text.splitlines():
        index, _, address = (part.strip() for part in line.partition(","))
        # nvidia-smi writes the address with eight digits of domain and in capitals, the kernel with four and without
        said = DEVICES / address[-12:].lower() / "numa_node"
        node = said.read_text().strip() if index.isdigit() and said.is_file() else ""
        if re.fullmatch(r"-?\d+", node):
            where[int(index)] = int(node)
    return where


def card_numa_node(device: str, ask: Asker) -> int | None:
    """The NUMA node of the lease's card (`device`, its index), which the row's check `one_numa_node` holds to its CPUs'
    node; None where the node doesn't say."""

    where = card_nodes(ask)
    node = None if isinstance(where, str) or not device.isdigit() else where.get(int(device))
    return node if node is not None and node >= 0 else None


def booked(card: int, near: Mapping[str, Any]) -> int:
    """The card the point was launched on (`--card`, a booked card's index), which its lease is then asked of alone, once
    it is one of the cards on the NUMA node of the run's CPUs (`cards_near`): refused, before any statement is staged or
    fetched, where it isn't or the node doesn't say."""

    if near["cards"] is None:
        raise Refused(f"the point was launched on card {card}, and the node doesn't say which cards are on the NUMA node "
                      f"of the run's CPUs ({near['why']}), so card {card} can't be held to them")
    if card not in near["cards"]:
        raise Refused(f"the point was launched on card {card}, which isn't on NUMA node {near['numa_nodes'][0]} of the "
                      f"run's CPUs (its cards are {', '.join(map(str, near['cards']))}): launch it with the cores of "
                      f"card {card}'s slot")
    return card


def snapshot(path: Path, run_id: str, card: str, ask: Asker, now: Callable[[], float]) -> None:
    """What else is on the node now, as `steady.card_file` says a run keeps it: each card another run's lease holds and
    each compute process on a card, or a line saying which of the two the node gave no answer on (a `gpu-lease status`
    without a `free` or `held` line of the row's card, `card`, is no answer: it doesn't say that no other lease holds
    it)."""

    lines = [f"epoch {int(now())}"]
    rc, text = ask(["gpu-lease", "status"])
    said = [line.split()[1:2] for line in text.splitlines() if line.split()[:1] == [card]] if rc == 0 else []
    if rc != 0:
        lines.append(f"gpu-lease status gave no answer (exit {rc})")
    elif not any(state in (["free"], ["held"]) for state in said):
        why = f"card {card} reads `{' '.join(said[0]) or 'nothing'}`" if said else f"no line of card {card}"
        lines.append(f"gpu-lease status gave no answer ({why} in its output)")
    for line in text.splitlines() if rc == 0 else ():
        if re.match(r"\d+ held\b", line) and f" run={run_id} " not in f"{line} ":
            lines.append("lease " + line.split(" cmd=")[0][:200])
    cards_rc, cards = ask(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
    apps_rc, apps = ask(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"])
    if cards_rc != 0 or apps_rc != 0 or not cards.strip():
        lines.append("nvidia-smi gave no answer")
    else:
        index = {uuid.strip(): number.strip() for number, _, uuid in (line.partition(",") for line in cards.splitlines())}
        for line in apps.splitlines():
            fields = [field.strip().replace(" ", "") for field in line.split(",")]
            if len(fields) == 3 and fields[0]:
                lines.append(f"app GPU {index.get(fields[0], '?')} pid {fields[1]} {fields[2]}")
    path.write_text("\n".join(lines) + "\n")
