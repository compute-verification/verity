"""The window circuit's wire-order certificate (`verity/wire-order/v1`): each inference entry's units ranked affinely in its
digits, and each record wire's writer entry and reader entry, which `program.predict.window_builder.order` makes and a
window writes beside its circuit record as ``order.json``. The period check reads it, in Lean, and trusts none of it; no
stage here reads it back, so it has no type here; this module owns its schema id."""

SCHEMA = "verity/wire-order/v1"
