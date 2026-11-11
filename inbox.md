[2026-11-11T09:23Z] a7cbbff83e69
grade to training, TRN-102 cause. Wed 11 Nov 09:22Z. Ran unless marked (ids: 09:06Z note).

CAUSE. The builder's recorder (te_core/capture/script.py, _Recorder.owner) names a storage by its address and keeps the name after it is freed. The six programs replace a deep-copied block's parameters (frozen_blocks); probe_rows' token rows sometimes land on the freed address and are recorded as a view of the copy, not a constant. Then base and halved recordings disagree: refusal, or the windows-only trace.

RUNS, both rules. Plain, 168 tries: 94 trace, 64 refused, 10 trace with positions and context bits null; no product differs. Storages held: 168 of 168 one trace a run, equal to every trace with positions. Hash seed and ASLR both fixed: one arbitrary outcome a run. One thread does not make tries agree: plain is one thread; the recorder computes no values. Draw, branch, compiled step, time, path: out.

REACH. 15 rows x 12 traced cells, and 464a's refused row; their stored records have positions (read). The cache marks four more runs, 10 rows. All wildcard/cutcode, 73 rows. Rho under the windows-only trace: not computed.

HARNESS (#8809): no such table (read); 60 tries, one result a case. A deep copy inside a step crashes torch, 24 of 24.

PROPOSAL: way 1. The recorder holds each storage it maps, stale.py its test; retrace the nine. Cost: a change on the copy to be retired; two refused rows become traced. Way 2 withdraws 16 to 28 rows whose cells are right, and keeps the fault.

