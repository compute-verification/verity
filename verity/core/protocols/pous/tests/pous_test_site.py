"""A wipe test's registered site: a remote composite's registration log (`verity.core.site.Registration`) registering the
wipe's server as a host with one memory leaf for each tier the wipe declares, so that `streamed_wipe.WipeParams.check_site`
holds the test's tiers to it. Kept apart from `pous_test_signer`, which other suites load by path."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.site import Loc, Registration, Site


def site_log(host: str, tiers: Sequence[tuple[str, int]]) -> Registration:
    """A log registering `host` with one memory leaf a tier, each `(name, bytes)` under its wipe name
    (`streamed_wipe.TIER_NAMES`), directly under the host."""
    spec = {name: t for t, name in sw.TIER_NAMES.items()}
    leaves = [Loc(f"{host}/{name}", "memory", host, tier=spec[name], bytes=b) for name, b in tiers]
    log = Registration("remote")
    return log.append(log.entry(locs=[Loc("site", "site"), Loc(host, "host", "site"), *leaves]))


def site_of(host: str, tiers: Sequence[tuple[str, int]]) -> Site:
    """The site `site_log(host, tiers)` registers: the wipe's tiers for `host` are `tiers`."""
    return Site.of_log(site_log(host, tiers))


def write_site(directory: Path, host: str, tiers: Sequence[tuple[str, int]]) -> str:
    """`site_log(host, tiers)` written to `directory/site.json`; its name, as `period.json`'s `site` names it."""
    (directory / "site.json").write_text(json.dumps(site_log(host, tiers).to_json()))
    return "site.json"
