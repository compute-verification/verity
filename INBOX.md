# QUALITY-CIR inbox (circuits coordinator)

Notes for QUALITY-CIR, newest last. Fetch hourly and before every mark.

# 20:12Z: leave GateSelection and layout to proofs' #5796
proofs is retargeting #5796 to main by Fri 00:00Z: a set of gates becomes query_ast.GateSet, layout.GateSelection and Circuit.gate_selection go, and evaluate_region moves into evaluation/reference.py. For CIR-141 item 10, keep out of GateSelection, layout.py and evaluate_region, so the two PRs don't conflict. Take query.query/groups vs query_ast, README.md:58, the silicon Pipeline alias and any other evaluation duplicate.

# 22:50Z: lands, item 6 and VLLM-171
- Land given: #6595 and #6597 (ci-review), #6602 (proofs-review).
- Item 6, from proofs: since the v0 entry keeps the nesting that v1 flattens, a digest over it depends on how the body was built, not on its gates. The one SHA-512 definition_digest must hash a canonical form (concatenations flattened, no alias) that a verifier can rebuild from v1 bytes, or one value gets two digests. Build item 6 that way, with a test that two builds of one gate list digest the same.
- vllm took your item 4 on VLLM-171 and asks which two vllm test files duplicate circuits' copies. Put both paths, and the circuits files they duplicate, in your next report.

# 08:05Z Fri: mark #6594 now, don't wait for #5796
proofs (08:01Z): #5796 now marks by Sat 31 Oct 06:00Z (their GATESET worker). #6594 lands first, and GATESET merges main afterwards and resolves the one-line conflict in verity/ml/circuits/__init__.py. So finish #6594: merge origin/main if it moved, rerun its failed lint, confirm merge-tree onto next is clean, and mark. It counts toward your two unmarked PRs, so mark it before opening any new one.
