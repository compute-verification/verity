[2026-11-11T07:22Z] 5031c97d573b
bounds, TRN-67, Wed 11 Nov 07:22Z. #8593 IS MARKED.
Result line (07:21Z, checkout of origin/main 8201dacff26e, plain): "research queue: #8593 is ready at 1a3cc0b7a385 (pr:8593@1a3cc0b7a385c6c3ac507b62367551ac055119c2)". Preflight: "merged onto main at 8201dacff26e: checks skipped: Lean: not built: 213 modules are or import what it changes, over the 40 the preflight builds; the lock run after its landing builds them; collect:adapters/vllm not checked: still running when the 120 s budget ran out; (24 more suites' collection not checked, the budget spent); the checks that ran passed (127 s; collected 52 suite(s))". Nothing refused. Records gate warned for security; no --update run.
What covers the skipped parts (read, at 1a3cc0b7a385): Lean check r20261111-044612-4d76 exit 0, AUDIT: PASS, 3 packages (706 + 3,806 + 94 modules), 0 failed, regenerated {}, every module replayed, none reused; unread ratchet exit 0; suites r20261111-044646-6ba0 exit 0: 52 suites, 52 passed (43 ran, 9 kept passes), 0 failed.
The queue's line comes with the four, after #8595's mark, which is running now.

