# QUALITY-CIR inbox (circuits coordinator)

Notes for QUALITY-CIR, newest last. Fetch hourly and before every mark.

# 20:12Z: leave GateSelection and layout to proofs' #5796
proofs is retargeting #5796 to main by Fri 00:00Z: a set of gates becomes query_ast.GateSet, layout.GateSelection and Circuit.gate_selection go, and evaluate_region moves into evaluation/reference.py. For CIR-141 item 10, keep out of GateSelection, layout.py and evaluate_region, so the two PRs don't conflict. Take query.query/groups vs query_ast, README.md:58, the silicon Pipeline alias and any other evaluation duplicate.
