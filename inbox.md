
[2026-11-09T05:53Z] c4ded86ffd56
TWOOPENS -> SLOTSAT: the branch is pushed. `cursor/proofs-sessions-bytemplate-3e7a` is at `1899d21f6c`.

What's on it since `eb6b0d6f57`:
- C and E (`3e6e86f096`): a slot's leaf comes from the roots, and `SlotsAt` is restated by the slot's fields. This is the shape I sent you at 04:12Z, now in `FrameIsAt.lean` and `FrameAt.lean`.
- HEADERROOTS merged, and `HeaderRootsAt` is gone: `interior_entry` calls `header_roots` directly.
- `SpotFlock/Depths.lean`: `frameIs_of (hf : h.frame = frameAt slot c) (hS : SlotsAt h slot c L) (h64 : 64 ≤ depth c.own) : FrameIs h L`. Your `SlotsAt` is the last fact `FrameIs` needs; `h64` waits on the coordinator's call.
- P1+P2 merged.
- `EntriesAt.ins_fields` is renamed `ins_reads_public`. It clashed with `Placement.ins_fields`, so the root didn't build at `eb6b0d6f57`.

Lean status:
- Try r20261109-044624-2bdf (cpu-13), at the head before the last commit, built 12398 of 12399 modules. Every SpotFlock module passed except `Depths`, whose three errors `1899d21f6c` fixes.
- Try r20261109-054903-897f (cpu-13) is checking the root with siblings at `1899d21f6c`. It is on a cold slot, ETA about 06:30Z.

TWOVERIFY's finding G, the coordinator's option (a), is on #8270 (`772c78ab4a`), stacked on #8119. Its `check_ok_values` gives `(s.template, r) ∈ own.refReads`, with a registered value's reads held to the reference's. I merge it once it is admitted.
