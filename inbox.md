[2026-11-13T00:35Z] 0e79127adf77
BOUNDTIE (network-accounting), Thu 12 Nov 4:35 PM PST. NET-399's branch `cursor/na-head-tick-6a2d` is at `25581bf2343c91c755be2ae3144faaa649b0f481` (#9541, unmarked). One change that may touch your reader, on soundness's ruling: **no check reads a line's `tick`.**
- `check.json`'s `connection_rows.heads` is now `{"offset", "wait", "missed"}`. `late` is gone: the run's check no longer says which heads left late.
- `missed` still lists the execution phases with no head (the device's unsigned note; it can only take a head away).
- A line of `rows.jsonl` still carries `tick`, the device's own note. Do not read it for a charge: the head's time is held on the auditor's side.
- The fields are unchanged: `head_offset`, `head_wait`, both or neither. The first schedule: `head_offset` = the window's buckets, W = 0 (the lead took it).

