[2026-11-10T18:31Z] f54fe4aadf65
TRN-76 FORM, 10:35 AM PST. Both refusals built and checked; #8676 at 13718f2ae76f (draft, unmarked, body's first lines say it refuses today's served record; merges clean onto main e7b03e4a6c).
RUNS at that head: r20261110-181602-844f Lean check AUDIT PASS, 1627 declarations, record difftest 633 cases 0 differences. r20261110-181201-06ab suites network 79 passed, repository 279 passed (verifier-cost in it, ALLOWED unchanged).
YOUR YES HOLDS. Counted by the reference's rule at that head, one wire at a time, on the small served circuit: 126 wires, 124 admitted, exactly two refused (TokenFeed_v1.o0 and SelectCollect_v1.o0 into EmbeddingRead_v1.tok). Bits under a word are admitted: 12 served wires today (SelectCollect_v1.head from ScanSlots_v2.o3 among them), and the difftest holds both directions at the served key's shape in Lean and reference (word under bits refused, bits under a word accepted). With tok taken as one 32-bit leaf: 126 admitted, 0 refused; ScanSlots_v2.o0 and o4 into tok are then bits under a word. own-rows refuses nothing served.
WHAT THAT COUNT IS NOT: the Lean form stops at the first refused wire, so the per-wire count is the reference's. The Lean on the whole served record is queued on a node (r20261110-182739-fe4d, waiting for a slot); I report its verdict on TRN-76.
ONE DIFFERENCE FROM Aligned: the rule is a little stronger. A port whose leaves have two widths is refused whichever way they lie. No served port has two widths.
NO TEST GOES RED: nothing on main, on vllm's remote branch or on lean's property branch names the form; on #8676 only its difftest and its own test read it, and neither reads a served record.
(a)-(e) are on TRN-76. (d): the verdict's 1 is held by SPOT's registered reads (interface.committed_reads, opened by Flock's checkRegistered); owner proofs. (e): done-when and dates unchanged.
Rerun the count: branch cursor/training-served-record-probe-3b1c, integrations/vllm/served_wires.py (95 s).

