[2026-11-15T00:36Z] 823a68678f4d
ci to LOCKPASS, 00:3xZ: one more check for your report (network-accounting, TOP-424 4:34 PM PST): the inference-only lock (verity/protocols/policy/inference_only/verifier/lean/lean-audit.json) pins two record-difftest data digests that are stale on the branch (the generator writes 708 cases, same bytes on two runs). Say whether your --update pass rewrites them, and whether `test_the_data_is_what_the_audit_pins` passes after it.

