"""verity_capture.tp — tensor-parallel (multi-rank) capture tooling, lane tp2 (window 2026-09-17).

Torch-free.  Reads the per-rank raw logs the observer already writes (`log.rank<r>.jsonl.gz`, one process per rank) and the fold's
per-rank outputs; produces the cross-rank *collective linkage record* — for every collective launch, which launch on each rank produced
its operand and which launches consume its result, joined across ranks by NCCL's call-order contract `(communicator, step, k)`.

  collective_record.py   the record schema: what one collective launch record must contain (per rank), what the cross-rank link
                         record contains, and what the current observer record lacks
  collective_link.py     `extract` (per-rank shard from a raw log) · `link` (join the shards, optionally with the fold's instances)

Nothing here changes observer / fold / patterns (frozen this window); the record is the input a future `Collective` pattern needs.
"""
