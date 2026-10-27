"""The space leg: one PoUS period on the node, answered by PoUS's run check (``verity/pous/p2/v4/run``), its hook
`verity.core.protocols.pous.period`'s ``leg`` (`verity_all_protocols.table`'s interface: ``leg(period_dir, node, out) ->
answer``). With ``pous-run-protocol`` and ``pous-setup`` among the period's tools (``run.sh`` builds them) the Lean answers
and must agree with the Python; it writes ``out/pous.case.json``, onsite's space case."""

from verity.core.protocols.pous.period import leg

__all__ = ["leg"]
