"""The wipe leg: the streamed wipe of the node's store a period, answered by ``verity/wipe/v0``, its hook
`verity.core.protocols.pous.wipe_period`'s ``leg`` (`verity_all_protocols.table`'s interface:
``leg(period_dir, node, out) -> answer``). With ``wipe-protocol`` among the period's tools (``run.sh`` builds it) the Lean
answers and must agree with the Python. Not one of onsite's components until ``verity/wipe/v0`` joins them."""

from verity.core.protocols.pous.wipe_period import leg

__all__ = ["leg"]
