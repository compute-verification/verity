"""The session's node figures, read from the node's data sheets: its node file and the GPU and host cards it names
(`verity.device`'s `load_node` and `load_card`), through `timing.Figures.of_sheets`, with the figures no sheet states
(`UNSHEETED`) and each GPU's squaring ceiling, a function of its card's clock readings (`squaring_ceilings`).  `python sheets.py NODE` writes them as the verifier's `--figures` reads them, and `python sheets.py --sha512
NODE` their digest, as its `--figures-sha512` pins it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "calibration/rate_squaring"))

import gamma  # noqa: E402
from verity.device import load_card, load_node, node_kinds, number  # noqa: E402
from verity.protocols.accounting.memory.poub.verifier import timing  # noqa: E402

CARD_FILES = Path(__file__).resolve().parents[2] / "calibration/hardware/cards"
NODE_FILES = Path(__file__).resolve().parents[2] / "calibration/hardware/nodes"

UNSHEETED = {
    "rtx-pro-6000-server-x8": {
        # block-sha512's 33 compressions a block over 2,097,152 blocks in 9.87 ms on one RTX PRO 6000
        "gpu_sha512_per_s": 7.0e9,
        # `net.ipv4.tcp_rmem`'s max on the measured node, the most its boundary link lands at once beyond its rate
        "link_burst_bytes": 6_291_456,
    },
}
# No figure here is measured on an H200 or H100 node, nor its card's on-chip storage counted: their sessions read the
# RTX PRO 6000 node's SHA-512 rate and link burst (its card's `store.chip` for the GPU's), as the measured node's. No
# squaring ceiling is borrowed: theirs is refused (`squaring_ceilings`).
UNSHEETED |= {kind: UNSHEETED["rtx-pro-6000-server-x8"]
              | {"gpu_chip_bytes": int(number(load_card("rtx-pro-6000-server", directory=CARD_FILES)["store"]["chip"]))}
              for kind in ("h200-sxm-141gb-x1", "h100-sxm-80gb-x1")}
"""Per node kind, the figures a session's ledger reads that no data sheet states, but its squaring ceiling."""

P512_FORM = 1.567
"""One RTX PRO 6000's squaring rate at 512 bits over codec-mix's reference (`gamma.REF` squarings a second at
`gamma.REF_MHZ`), at the reference's clock, as it was typed: s·(1 + h), s the best form on the GPU less its MOV and SEL
and h the host's 128 cores at their IFMA bound over 8 reference GPUs.  Its source (the form, the run behind it and its
split between s and h) is not in the tree, and it is none of `gamma.GPU`'s forms.  `squaring_ceilings` takes it whole
as the GPU's part, so the host share beside it counts the host again: the higher ceiling."""

FORMS = {"rtx-pro-6000-server-x8": {"p512": P512_FORM}}
"""Per node kind, per codec width, one GPU's squaring rate over `gamma.REF` at the same clock, where a ceiling is
established.  Only the node `gamma.py` reads has one: its host is the one `HOST_SQUARINGS_PER_S` counts."""

WIDTHS = ("p512", "p1024")
"""The codec widths a node's figures state a squaring ceiling at, or refuse one."""

CLOCK_READINGS = ("sm", "sm_boost")
"""The readings of a GPU card's SM clock its squaring ceiling is scaled to, each where the card states it: `clock.sm`
and the data sheet's `clock.sm_boost`."""

HOST_SQUARINGS_PER_S = max(gamma.HOST.values())
"""The host's squarings a second, the larger of `gamma.HOST`'s readings: its cores at their IFMA-only port bound,
above their measured rate a core.  No GPU clock scales it."""

REFUSED = {"p1024": "the clock of the run behind its reference rate at this width (`kv.cu`'s) is stated nowhere, so the "
                    "ceiling can't be scaled to the card's clock readings: state that run's clock, or measure the best "
                    "form at this width at a stated clock, and give it a form in benchmarks/poub_session/sheets.py's "
                    "FORMS"}
"""Per codec width, why a node kind in `FORMS` has no ceiling at it."""


def squaring_ceilings(node: str, gpu: dict, gpus: int) -> dict[str, float | str]:
    """One GPU's squaring ceiling per codec width on a node of kind `node` with `gpus` GPUs of card `gpu`, or the
    refusal that names why none is established.  In squarings a second per GPU at a width, the node's ceiling over its
    GPUs, the largest over the card's clock readings f (MHz, `CLOCK_READINGS`):

        C_w = max_f (s_w · REF · f / REF_MHZ) + H / gpus

    s_w the form over the reference at its clock (`FORMS`, dimensionless), REF the reference (`gamma.REF`, squarings
    a second per GPU) at REF_MHZ (`gamma.REF_MHZ`, MHz), and H the host's (`HOST_SQUARINGS_PER_S`, squarings a second),
    which the GPUs' clock doesn't scale.  The node's ceiling is gpus·C_w."""
    readings = [float(number(gpu["clock"][r])) for r in CLOCK_READINGS if r in gpu["clock"]]
    out: dict[str, float | str] = {}
    for width in WIDTHS:
        if width in FORMS.get(node, {}):
            gpu_part = max(FORMS[node][width] * gamma.REF * f / gamma.REF_MHZ for f in readings)
            out[width] = gpu_part + HOST_SQUARINGS_PER_S / gpus
        else:
            why = REFUSED[width] if node in FORMS else (
                "no squaring rate is measured on it, and another card's ceiling is not its bound: measure the best "
                "form on it at a stated clock (calibration/rate_squaring) and give it a form in "
                "benchmarks/poub_session/sheets.py's FORMS")
            out[width] = f"no squaring ceiling is established for {gpu['sku']} at {width[1:]} bits: {why}"
    return out


def figures(node: str, *, count: int | None = None) -> timing.Figures:
    """The figures of a node of kind `node`, from its sheets and `UNSHEETED`, at `count` of its GPUs (by default its
    node file's); refused by name for a kind that has no node file or no `UNSHEETED` entry. A host card that states no
    measured disk read is taken at its ceiling (`disk.read_max`)."""
    if node not in node_kinds(directory=NODE_FILES):
        raise ValueError(f"no node file {node}.toml under {NODE_FILES}: name one of "
                         f"{', '.join(node_kinds(directory=NODE_FILES))}")
    if node not in UNSHEETED:
        raise ValueError(f"the node kind {node} has no SHA-512 rate or link burst measured "
                         f"(sheets.UNSHEETED): measure them on it, or hand the verifier a --figures file")
    n = load_node(node, directory=NODE_FILES, cards_directory=CARD_FILES)
    if count is not None:
        n = {**n, "device": {**n["device"], "count": {**n["device"]["count"], "value": count}}}
    host = load_card(n["host"]["sku"], directory=CARD_FILES)
    if "read" not in host.get("disk", {}):
        host = {**host, "disk": {**host["disk"], "read": host["disk"]["read_max"]}}
    gpu = load_card(n["device"]["sku"], directory=CARD_FILES)
    return timing.Figures.of_sheets(n, gpu, host, **UNSHEETED[node], gpu_squarings_per_s=squaring_ceilings(
        node, gpu, int(number(n["device"]["count"]))))


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


__all__ = ["FORMS", "UNSHEETED", "figures", "main", "squaring_ceilings"]

if __name__ == "__main__":
    sys.exit(main())
