"""Phase 17 red-team (#3862 N2-ZK), Target 5a: is the advised driver's clock source labelled a stand-in everywhere it
reports? The driver carries no clock certificate (#3881 adds those); every window's floor is computed under a clock the
driver declares is a stand-in, and the summary's ``clocks`` is meant to say so.

* The simulated link's clock (`advised_traffic.StandInClocks`) reports ``source == "stand-in"``, and both links' tie
  (`advised_traffic.StandInTie`) reports ``source == "stand-in"``: a reviewer scanning a ``summary.json`` for the word
  flags them.
* The served link's clock (`CertifierClocks`, read from a run's ``proxy.json.gz``) is a stand-in too -- its docstring
  says so and `CertifierClocks.of` refuses a run without the log -- but its summary label
  (`CertifierClocks.label`) names only the proxy log and carries no ``stand-in`` marker, unlike the tie and the
  simulated clock. FINDING (low): the one summary field a reviewer would grep for "stand-in" doesn't flag the served
  run's clock. One-line fix: mark `CertifierClocks.label`'s source as a stand-in. Owner: N2-ZK (#3862).

Nothing here lowers or proves."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_n2_advised_zk", HERE / "n2_advised_zk.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


n2 = _load()
AT = n2.AT


def test_the_simulated_links_clock_and_both_ties_are_marked_stand_in():
    """The simulated link's clock and the tie on either link report ``source == "stand-in"``, so a reviewer grepping a
    summary for the word flags them."""
    sim = n2.simulated(n2.SEED, AT.SHAPE)
    assert sim.clocks["source"] == "stand-in"
    assert "StandInClocks" in sim.clocks["by"]
    assert sim.tie["source"] == "stand-in"
    served = n2.standin_tie(sim.ingress, n2.OWN)
    assert served["source"] == "stand-in"
    assert "stand-in" in json.dumps(sim.clocks).lower() and "stand-in" in json.dumps(sim.tie).lower()


def test_the_served_runs_clock_names_the_proxy_log_but_carries_no_stand_in_marker(tmp_path):
    """`CertifierClocks` is a stand-in: its docstring says so and `CertifierClocks.of` refuses a run without the log.
    But `CertifierClocks.label` names only the proxy log, with no ``stand-in`` marker -- the asymmetry a reviewer
    scanning for the word would miss. The log's ``as_declared`` flag still toggles on whether it matches the plan."""
    assert "stands in" in " ".join(n2.CertifierClocks.__doc__.split())
    with gzip.open(tmp_path / "proxy.json.gz", "wt") as fh:
        json.dump({"syncs": [[s.delta, str(s.rho)] for s in AT.SYNCS]}, fh)
    clocks = n2.CertifierClocks.of(tmp_path, "n2")

    label = clocks.label(AT.SYNCS)
    assert "proxy.json.gz" in label["source"]
    assert label["as_declared"] and not clocks.label(AT.SYNCS[::-1])["as_declared"]
    # The finding: the served clock's summary label carries no "stand-in" marker, unlike the tie and simulated clock.
    assert "stand-in" not in json.dumps(label).lower()

    with pytest.raises(SystemExit, match="no clock source"):
        n2.CertifierClocks.of(tmp_path / "none", "n2")


def test_the_served_clock_reports_only_through_its_label_so_the_marker_would_live_there():
    """`CertifierClocks` reports its syncs to the driver only through `syncs` (to the view) and `label` (to the
    summary); the summary's clock field is exactly ``label``'s output, so the marker, if added, belongs in `label`."""
    clocks = n2.CertifierClocks("n2", tuple(AT.SYNCS))
    assert list(clocks.syncs("n2")) == list(AT.SYNCS) and clocks.syncs("other") == ()
    public = [n for n in vars(type(clocks)) if not n.startswith("_") and callable(getattr(type(clocks), n, None))]
    assert set(public) <= {"of", "syncs", "label"}
