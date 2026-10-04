"""Running the protocol: one deployment, under one configuration, over many epochs.

    refuse      every reason the configuration cannot run, once, before anything runs
    set up      the commitment scheme and the auditor's challenge source, shared by every epoch
    repeat      `epoch.run_epoch`, each epoch's record chained to the previous one's digest

Every epoch yields its report as it ends: each layer's verdict comes every epoch, and silence is never an accept. An
integration runs this with itself as the developer; `epochs` is unbounded by default, so a deployment runs until it stops.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

from verity.primitives.commitments import SCHEMES
from verity.primitives.randomness import SOURCES
from verity.primitives.registry import NotInstalled
from verity.protocol.accounting.account import Inventory
from verity.protocol.configuration import Configuration, refusals
from verity.protocol.epoch import Developer, EpochRecord, EpochReport, make_audits, run_epoch


class Refused(ValueError):
    def __init__(self, reasons: tuple[str, ...]):
        self.reasons = reasons
        super().__init__("configuration refused: " + "; ".join(reasons))


def run(
    config: Configuration, developer: Developer, inventory: Inventory, epochs: Iterable[int] | None = None
) -> Iterator[tuple[EpochReport, EpochRecord]]:
    why = list(refusals(config))
    if not why:
        try:
            make_audits(config, SCHEMES[config.commitments]())
        except NotInstalled as e:
            why.append(str(e))
    if why:
        raise Refused(tuple(why))

    scheme, source = SCHEMES[config.commitments](), SOURCES[config.challenges]()
    previous = None
    for epoch in itertools.count(1) if epochs is None else epochs:
        report, record = run_epoch(config, scheme, source, developer, inventory, epoch, previous)
        previous = record.digest()
        yield report, record
