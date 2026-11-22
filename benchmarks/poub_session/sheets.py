"""The session's node figures, read from the node's data sheets: its node file and the GPU and host cards it names
(`verity.device`'s `load_node` and `load_card`), through `timing.Figures.of_sheets`, with the figures no sheet states
(`UNSHEETED`).  `python sheets.py NODE` writes them as the verifier's `--figures` reads them, and `python sheets.py --sha512
NODE` their digest, as its `--figures-sha512` pins it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.device import load_card, load_node, node_kinds, number  # noqa: E402
from verity.protocols.accounting.memory.poub.verifier import timing  # noqa: E402

CARD_FILES = Path(__file__).resolve().parents[2] / "calibration/hardware/cards"
NODE_FILES = Path(__file__).resolve().parents[2] / "calibration/hardware/nodes"

UNSHEETED = {
    "rtx-pro-6000-server-x8": {
        # block-sha512's 33 compressions a block over 2,097,152 blocks in 9.87 ms on one RTX PRO 6000
        "gpu_sha512_per_s": 7.0e9,
        # Assumption 5 per width, one GPU's share of the node: s·(1 + h) over the reference (codec-mix's 5.13e10 at
        # 512 bits, `kv.cu`'s 1.499e10 at 1,024), s the best form on the GPU less its MOV and SEL and h the host's 128
        # cores at their IFMA bound over 8 reference GPUs
        "gpu_squarings_per_s": {"p1024": 1.67 * 1.499e10, "p512": 1.567 * 5.13e10},
        # `net.ipv4.tcp_rmem`'s max on the measured node, the most its boundary link lands at once beyond its rate
        "link_burst_bytes": 6_291_456,
    },
}
# No figure here is measured on an H200 or H100 node, nor its card's on-chip storage counted: their sessions read the
# RTX PRO 6000 node's (its card's `store.chip` for the GPU's), as the measured node's.
UNSHEETED |= {kind: UNSHEETED["rtx-pro-6000-server-x8"]
              | {"gpu_chip_bytes": int(number(load_card("rtx-pro-6000-server", directory=CARD_FILES)["store"]["chip"]))}
              for kind in ("h200-sxm-141gb-x1", "h100-sxm-80gb-x1")}
"""Per node kind, the figures a session's ledger reads that no data sheet states."""


def figures(node: str, *, count: int | None = None) -> timing.Figures:
    """The figures of a node of kind `node`, from its sheets and `UNSHEETED`, at `count` of its GPUs (by default its
    node file's); refused by name for a kind that has no node file or no `UNSHEETED` entry. A host card that states no
    measured disk read is taken at its ceiling (`disk.read_max`)."""
    if node not in node_kinds(directory=NODE_FILES):
        raise ValueError(f"no node file {node}.toml under {NODE_FILES}: name one of "
                         f"{', '.join(node_kinds(directory=NODE_FILES))}")
    if node not in UNSHEETED:
        raise ValueError(f"the node kind {node} has no SHA-512 rate, squaring ceiling or link burst measured "
                         f"(sheets.UNSHEETED): measure them on it, or hand the verifier a --figures file")
    n = load_node(node, directory=NODE_FILES, cards_directory=CARD_FILES)
    if count is not None:
        n = {**n, "device": {**n["device"], "count": {**n["device"]["count"], "value": count}}}
    host = load_card(n["host"]["sku"], directory=CARD_FILES)
    if "read" not in host.get("disk", {}):
        host = {**host, "disk": {**host["disk"], "read": host["disk"]["read_max"]}}
    return timing.Figures.of_sheets(n, load_card(n["device"]["sku"], directory=CARD_FILES), host, **UNSHEETED[node])


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    digest = args[:1] == ["--sha512"]
    if len(args) != 1 + digest:
        print("usage: sheets.py [--sha512] NODE: the figures of a node of kind NODE (a node file's name under "
              "calibration/hardware/nodes), as the session verifier's --figures reads them, or (--sha512) their "
              "digest, which its --figures-sha512 pins", file=sys.stderr)
        return 2
    try:
        fig = figures(args[-1])
        print(fig.sha512() if digest else json.dumps(fig.to_json(), sort_keys=True))
    except ValueError as e:
        print(f"sheets.py: {e}", file=sys.stderr)
        return 3
    return 0


__all__ = ["UNSHEETED", "figures", "main"]

if __name__ == "__main__":
    sys.exit(main())
