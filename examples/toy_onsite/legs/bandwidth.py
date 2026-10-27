"""The bandwidth leg: one remote bandwidth session a period on the node, answered by ``verity/remote-bandwidth/v0``, its
hook `verity.core.protocols.remote_bandwidth.period`'s ``leg`` (`verity_all_protocols.table`'s interface:
``leg(period_dir, node, out) -> answer``). With ``remote-audit-protocol`` among the period's tools (``run.sh`` builds it)
the Lean answers and must agree with the Python; it writes ``out/bandwidth.case.json``, onsite's bandwidth case."""

from verity.core.protocols.remote_bandwidth.period import leg

__all__ = ["leg"]
