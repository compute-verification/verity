[2026-11-18T13:01Z] 2f19cee6af12
From the kernels lead, two facts for your gate-step job, no reply needed:
- KSALT's mend is on `interfaces` at `e576be54965` (commits `c566d6f509d`, `24caac3fc96`): `smoke_h100.sh` names no architecture; it reads its target from a new verb, `pearl_c_sm90/build.sh --arch`. Build on that tip.
- `pearl_c_sm90/build.sh` changed by two lines for that verb. When you run the script `--dry` at your new commit on the node, the `pearl_c.build` step rebuilds Pearl-C off the card: say whether the cubin is the same as the one run 2 executed (the debit cubin digest your earlier dry runs printed, `5e609a74809be810`, or whatever run 2 printed: compare like with like and paste both). If it differs, tell me first.

