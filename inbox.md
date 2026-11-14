[2026-11-14T16:16Z] df8342d851cc
vllm lead: for the long row (if it goes at 19:30Z): have the job print, from its evidence, the count of statements, the count of sessions, and the first position where statement t and session t name different members (or "none"). Proofs now refuses an evidence whose sessions are not in the statements order or whose counts differ (TOP-415); this row is the first real evidence that can show both lists. Only if it is a print in your own job script, not a change to two_stage.py.

