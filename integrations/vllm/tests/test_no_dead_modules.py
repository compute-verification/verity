"""Dead-code lint: no new dead modules in `integrations/vllm` (shrink-only keep-list).

`tests/dead_code_census.py` builds the static import graph of `verity_vllm` / `verity_vllm` / `verity.ml.definitions.numerics` /
`verity_vllm` from the declared entry points in `census_roots.txt` (shell runners -- every `verity_vllm/ops/*.sh` and
every `*_negatives.sh` / `*_negative_*.sh` by rule --, Python entry points, the regression harness).  A production module no root
reaches is dead unless `dead_code_keep.json` names it, with the reason it stays.  The test fails on

* a non-live module the keep-list does not name: new dead code -- delete it, reach it from a declared entry point, or add a reviewed
  keep entry;
* a keep entry that names no non-live module (the module was deleted, or became live): remove the entry, so the list only shrinks;
* a roots entry that matches nothing, and a negatives runner that is not a root.

Shell scripts are entry points: a runner is live because `census_roots.txt` declares it, not because something imports it.

Dynamic evidence: the modules seen imported (`sys.modules` at exit of every process) by the ENGINE=v2 regression harness over the 13
fixture workloads, the offline suites and `python -m <entry point> --help` of every root module on the tree of the deletions is the
artifact named by `dynamic_record` in `dead_code_keep.json`; `python tests/dead_code_census.py --imported <its imported.json>` merges it.
"""
from __future__ import annotations

import fnmatch
import functools

import pytest

from tests import dead_code_census as census


@functools.lru_cache(maxsize=1)
def _state():
    g = census.Graph()
    by_mod, by_shell, unmatched = census.resolve_roots(g)
    return g, by_shell, unmatched, census.non_live_modules(g), census.load_keep()


@pytest.mark.slow
def test_every_roots_entry_matches():
    _, _, unmatched, _, _ = _state()
    assert not unmatched, f"census_roots.txt entries that match no tracked shell, directory or module: {unmatched}"


@pytest.mark.slow
def test_negatives_runners_are_roots():
    g, by_shell, _, non_live, _ = _state()
    negatives = [s for s in census.shell_files() if fnmatch.fnmatchcase(s.rsplit("/", 1)[-1], "*_negative*.sh")]
    assert negatives, "no negatives runner found"
    assert not [s for s in negatives if s not in by_shell], "negatives runners must be entry points (census_roots.txt)"
    ran = {m for s in negatives for m in by_shell[s]}
    assert ran and not ran & set(non_live), f"modules a negatives runner runs must be live: {sorted(ran & set(non_live))}"


@pytest.mark.slow
def test_no_new_dead_modules():
    _, _, _, non_live, keep = _state()
    new = {m: rel for m, rel in sorted(non_live.items()) if m not in keep}
    assert not new, ("modules no declared entry point reaches (delete them, reach them from census_roots.txt, or keep them by name in "
                     f"dead_code_keep.json with the reason): {new}")


@pytest.mark.slow
def test_keep_list_only_shrinks():
    _, _, _, non_live, keep = _state()
    stale = sorted(m for m in keep if m not in non_live)
    assert not stale, f"dead_code_keep.json entries that name no non-live module (deleted or now live): remove them: {stale}"
