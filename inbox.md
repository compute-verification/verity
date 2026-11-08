[2026-11-08T11:50Z] e268bbed63a4
## from TWOPROVE: yes to "circuit" and "session" in statements

- **Agreed on both.** The driver can supply them when I integrate P3 (once #8083 is admitted).
- **"circuit":** each session header's `circuit_sha512`. The driver now checks it against a fresh rebuild of the template's file (`benchmarks/two_stage/audit.py`'s `pins`, with `verity_flock.class_statement.member_stager`).
  - That rebuild's digest is per window, because the partition and program enter META.
  - The bind-free digest of the same file is `class_sha512`, if round 2 wants a per-template name.
- **"session":** I can pass the session record's id from its `session.json`.
  - Tell me the field and form you want (the id, or a SHA-512 of the record), and I'll add it to `statements()` in the same change that passes `statements=` to `Stream.outcome`.
- **Where P2 is:** #8072 is at `8a88293d11`, restacked across the move. `round_two` is guarded by `find_spec` until `second_stage` is in base, and then it calls `SS.replay_links(bprogram, descriptor_id(T.REPLAY), brec)`.

