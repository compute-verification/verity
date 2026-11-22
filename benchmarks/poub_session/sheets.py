"""The session's node figures, read from the node's data sheets: its node file and the GPU and host cards it names
(`verity.device`'s `load_node` and `load_card`), through `timing.Figures.of_sheets`, with the figures no sheet states
(`UNSHEETED`) and each GPU's share of the card's maximum squaring rate, a function of its card's clock readings
(`squaring_ceilings`).  `python sheets.py NODE` writes them as the verifier's `--figures` reads them, and `python sheets.py --sha512
NODE` their digest, as its `--figures-sha512` pins it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "calibration/rate_squaring"))

import gamma
from verity.device import load_card, load_node, node_kinds, number
from verity.protocols.accounting.memory.poub.verifier import timing

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
BORROWED = ("h200-sxm-141gb-x1", "h100-sxm-80gb-x1")
"""The node kinds with a node file and no SHA-512 rate, link burst or squaring rate measured on them."""


def _sku(kind: str) -> str:
    return load_node(kind, directory=NODE_FILES, cards_directory=CARD_FILES)["device"]["sku"]


def _chip(sku: str) -> int | str:
    store = load_card(sku, directory=CARD_FILES).get("store", {})
    if "chip" in store:
        return int(number(store["chip"]))
    return (f"no on-chip storage is established for {sku}: its card file states no store.chip, and another card's is "
            "not its bound, since S_chip bounds the bytes the node's GPUs hold on chip from above: state store.chip in "
            f"calibration/hardware/cards/{sku}.toml with its source")


# Each figure below bounds what the audited node can do from above (its compressions a second, the bytes its links land
# at once, the bytes its GPUs hold on chip), so another node's figure is not its bound: each is refused by name, the
# on-chip storage only where the node's own card states no `store.chip`.
UNSHEETED |= {kind: {
    "gpu_sha512_per_s": f"no SHA-512 rate is established for {_sku(kind)}: none is measured on it, and another card's "
                        "rate is not its bound, since the rate bounds the prover's compressions a second from above: "
                        "measure block-sha512 on it and state it in benchmarks/poub_session/sheets.py's UNSHEETED",
    "link_burst_bytes": f"no link burst is established for the node kind {kind}: none is read on it, and another node's "
                        "is not its bound, since the burst bounds the bytes its links land at once from above: read "
                        "`net.ipv4.tcp_rmem`'s max on it and state it in benchmarks/poub_session/sheets.py's UNSHEETED",
    "gpu_chip_bytes": _chip(_sku(kind))}
    for kind in BORROWED}
"""Per node kind, the figures a session's ledger reads that no data sheet states, but the card's maximum squaring
rate, each a number or the refusal that says none is established."""

FORMS = {"rtx-pro-6000-server-x8": {"p512": max(gamma.GPU.values())}}
"""Per node kind, per codec width, one GPU's squaring rate over `gamma.REF` (squarings a second) at `gamma.REF_MHZ`
(MHz), dimensionless, where the card's maximum squaring rate is established.  At 512 bits it is the largest of
`gamma.GPU`'s variants, measured and modelled alike, every one a 512-bit rate (mod 2^512 - 569) taken at `gamma.MHZ`
and scaled to `gamma.REF_MHZ`: a maximum squaring rate is an upper bound, so the fastest variant stated is taken.  The
premise that stays open: no variant faster than the fastest `gamma.py` states exists.  Only the node `gamma.py` reads
has one: its host is the one `HOST_SQUARINGS_PER_S` counts."""

WIDTHS = ("p512", "p1024")
"""The codec widths a node's figures state the card's maximum squaring rate at, or refuse one."""

CLOCK_READINGS = ("sm", "sm_boost")
"""The readings of a GPU card's SM clock the card's maximum squaring rate is scaled to, each where the card states it:
`clock.sm` and the data sheet's `clock.sm_boost`."""

HOST_SQUARINGS_PER_S = max(gamma.HOST.values())
"""The host's squarings a second, the larger of `gamma.HOST`'s readings: its cores at their IFMA-only port bound,
above their measured rate a core.  No GPU clock scales it."""

REFUSED = {"p1024": "the clock of the run behind its reference rate at this width (`kv.cu`'s) is stated nowhere, so the "
                    "rate can't be scaled to the card's clock readings: state that run's clock, or measure the fastest "
                    "variant at this width at a stated clock, and give it in benchmarks/poub_session/sheets.py's FORMS"}
"""Per codec width, why a node kind in `FORMS` has no maximum squaring rate at it."""


def squaring_ceilings(node: str, gpu: dict, gpus: int) -> dict[str, float | str]:
    """One GPU's share of the card's maximum squaring rate per codec width on a node of kind `node` with `gpus` GPUs of
    card `gpu`, or the refusal that names why none is established.  In squarings a second per GPU at a width, the
    largest over the card's clock readings f (MHz, `CLOCK_READINGS`):

        C_w = max_f (s_w · REF · f / REF_MHZ) + H / gpus

    s_w the fastest variant over the reference at its clock (`FORMS`, dimensionless), REF the reference (`gamma.REF`,
    squarings a second per GPU) at REF_MHZ (`gamma.REF_MHZ`, MHz), and H the host's (`HOST_SQUARINGS_PER_S`,
    squarings a second), which the GPUs' clock doesn't scale.  s_w is the GPU's alone, so the host is added once, in
    H / gpus.  The node's rate is gpus·C_w."""
    readings = [float(number(gpu["clock"][r])) for r in CLOCK_READINGS if r in gpu["clock"]]
    out: dict[str, float | str] = {}
    for width in WIDTHS:
        if width in FORMS.get(node, {}):
            gpu_part = max(FORMS[node][width] * gamma.REF * f / gamma.REF_MHZ for f in readings)
            out[width] = gpu_part + HOST_SQUARINGS_PER_S / gpus
        else:
            why = REFUSED[width] if node in FORMS else (
                "no squaring rate is measured on it, and another card's is not its bound: measure the fastest "
                "variant on it at a stated clock (calibration/rate_squaring) and give it in "
                "benchmarks/poub_session/sheets.py's FORMS")
            out[width] = f"no maximum squaring rate is established for {gpu['sku']} at {width[1:]} bits: {why}"
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


__all__ = ["BORROWED", "FORMS", "UNSHEETED", "figures", "main", "squaring_ceilings"]

if __name__ == "__main__":
    sys.exit(main())
