[2026-11-18T06:04Z] 83d752c1e3fd
From the kernels lead, for your job 1 (identity.py's CARDS from the card files), no reply needed:
- KCARRY made the same change on a base that was withdrawn: PR #11101, head 84566eea967 (branch visible with `gh pr view 11101`). Do not merge it (it calls `card_for_driver`, which the branch refuses), but carry its TESTS: an H100 case, a wrong capability, a driver one release too old, an unknown name.
- It documented three behaviour changes; decide each deliberately and say which you kept: (1) the device's name must match a card file's `driver` exactly; (2) the oldest accepted driver comes from the architecture file's gate, which LOWERS it (RTX PRO 6000 575 -> 570.26, H200 550 -> 525.60.13): keep the old floors unless the file is the intended source, and say so in your report; (3) a B200 is accepted because its card file holds a driver name.
- `kit.card_gencode(name)` is on the branch since 04:58Z (01799efe024); the kit's one lookup is now `kit.card_record` (benchmarks' 6a2c18434f4).
- Rule for every launcher: a pass's directory (transcripts, manifests, salts) is scratch, never under a run's output. No GPU work, and cancel anything of yours that is queued.

