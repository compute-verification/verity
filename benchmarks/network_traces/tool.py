"""Store declarations for the network traces' harness (research.store README §6.1).

    research run --on MACHINE ... --tool benchmarks.network_traces.tool:COVERT_SYNC -- \
      bash benchmarks/network_traces/covert.sh last cover-3-2 envelope

Key params: the sync policies. The traces and the calibration are inputs. Imports nothing but ``research.store.tool``.
"""

from __future__ import annotations

from research.store.tool import Tool


def parse_covert(argv: list[str]) -> dict:
    return {"policies": " ".join(a for a in argv[2:] if not a.startswith("-"))}


COVERT_SYNC = Tool(name="network_covert_sync", version="1", command=["bash", "benchmarks/network_traces/covert.sh"],
                   closure=["benchmarks/network_traces/**", "verity/protocols/accounting/communication/**"],
                   key_params=["policies"], produces=["run-files/v1"], parse=parse_covert)
