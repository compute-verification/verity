
## 22:30Z census is in
art:35e0fc4c6e108487ac4125267fcc208294991d15bc2df7e4f6ecb86c7eb82edb (fetch it with `research data fetch`). The plan, owners by step, and rulings 1-6 are on PRF-268.
Your steps: step 2 is the ONE registration digest in the code twin (REG-CORE's file: Flock/Registration.lean or core_protocol). Collapse SampledProofs.Record's and Flock.RecVStar's copies and the two in Security/Properties onto it, keeping the bytes (ruling 1). Then step 6 for the Lean readers. Post the digest's location on PRF-268 as soon as it compiles; REG-SP-PY, REG-POUW, REG-NET and REG-ADV read it.

## 22:58Z: don't write a second digest
ONE4 already put the one window-record digest in `Flock/WindowRecord.lean` (flock_verifier) on `cursor/soundness-reg-digest-7827` at 6325a253a8; PoUW, NCI and the network check read it. Not core_protocol, because RecVStar is in flock_verifier and core_protocol requires it, so that would be a cycle; this amends ruling 5. Merge that branch (not next) and point your readers at it: `SampledProofs.Record.registrationDigest`, `Flock.RecVStar`, and the two in Security/Properties. Rename the file if you like, but keep one definition. Fix `Record.registrationDigest`'s non-ASCII disagreement with it through ruling 6 (ASCII-only records). Post on PRF-268.

## 02:45Z: new deadline, new nodes (all soundness workers)
- Daniel, 7:30 PM PDT: everything in P0 to P2 is done by Wed 15:00Z. The last mark for any P1 or P2 PR is **09:00Z**, and the registration cluster must be at zero on staging by 10:00Z. P3 is not part of it: no new P3 work, only landing PRs already open.
- infra: vy-nebius-cpu-9 has been open since 02:15Z and cpu-8 opens by 03:00Z, each with a soundness pool and 6-8 warm Lean build slots. cpu-7 is 14 deep. Cancel by id any run of yours that hasn't started on cpu-7, cpu-4 or cpu-6, and resend it on cpu-9 (or cpu-8 once it is up), with `--cwd clone`.
