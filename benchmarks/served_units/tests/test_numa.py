"""numa.py: each card's CPUs from its own NUMA node, and the check that reads a run's groups back against them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load


@pytest.fixture
def numa(monkeypatch):
    return load(monkeypatch, "numa.py")


#: node A's layout: cards 0 and 1 on node 0, 2 and 3 on node 1, 4 and 5 on node 2, 6 and 7 on node 3; 24 CPUs a node
CSV = "".join(f"{i}, GPU-{i}, 00000000:{0x1b + 0x10 * i:02X}:00.0\n" for i in range(8))


def sysfs(root: Path) -> Path:
    for i in range(8):
        d = root / f"bus/pci/devices/0000:{0x1b + 0x10 * i:02x}:00.0"
        d.mkdir(parents=True)
        (d / "numa_node").write_text(f"{i // 2}\n")
    for n in range(4):
        (root / f"devices/system/node/node{n}").mkdir(parents=True)
        (root / f"devices/system/node/node{n}/cpulist").write_text(f"{24 * n}-{24 * n + 23}\n")
    return root


def test_cards_2_3_4_on_cores_24_95_get_their_own_nodes_cpus(numa, tmp_path: Path):
    pin = numa.pin(["2", "3", "4"], CSV, tmp_path / "t.json", sysfs(tmp_path / "sys"), list(range(24, 96)))
    assert pin == "2=" + ",".join(map(str, range(24, 36))) + ";3=" + ",".join(map(str, range(36, 48))) \
        + ";4=" + ",".join(map(str, range(48, 60)))
    t = json.loads((tmp_path / "t.json").read_text())
    assert [c["numa_node"] for c in t["cards"]] == [1, 1, 2] and t["nodes"]["1"] == list(range(24, 48))


def test_a_card_whose_node_holds_none_of_the_jobs_cpus_is_refused(numa, tmp_path: Path):
    with pytest.raises(SystemExit, match="node 0 \\(cards 1\\): 0"):
        numa.pin(["1", "2"], CSV, tmp_path / "t.json", sysfs(tmp_path / "sys"), list(range(24, 96)))


def test_the_check_finds_a_prover_off_its_cards_node(numa, tmp_path: Path):
    """Wave 1's own pinning: card 3's prover on 40-43 is on node 1; card 2's on 66,67,82,83 is not."""
    numa.pin(["2", "3"], CSV, tmp_path / "t.json", sysfs(tmp_path / "sys"), list(range(24, 96)))
    for g, (card, prover, verifier) in enumerate([(3, "40,41,42,43", "44,45"), (2, "66,67,82,83", "84,85")]):
        (tmp_path / f"out/gpu{g}/out").mkdir(parents=True)
        (tmp_path / f"out/gpu{g}/gpu-lease-usage.jsonl").write_text(json.dumps({"gpus": [{"index": card}]}) + "\n")
        (tmp_path / f"out/gpu{g}/out/host.txt").write_text(f"device=gpu cpus=x prover={prover} verifier={verifier}\n")
    assert numa.check(tmp_path / "t.json", tmp_path / "out") == 1
    rows = (tmp_path / "out/pinning.tsv").read_text().splitlines()
    assert [r.split("\t")[5] for r in rows[1:]] == ["1", "0"]
