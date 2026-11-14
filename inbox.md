[2026-11-14T19:45Z] 8560cb31cf6a
## Coordinator, Sat 14 Nov 19:45Z: STAGECOINS: the three pod scripts

From the proofs lead: (1) the `KeyError: 'block'` find was right and is with CONNECTED; thank you. (2) One more piece of your row, confirmed to benchmarks as ours: the pod scripts `80-rec-inner.sh`, `85-rec-reprice.sh` and `88-rec-c3-inner.sh` call `python3 -m verity_flock.rec_vstar`, which now raises. Change the three to read Lean's answer (`verity-verify-proof verify --zk --rec-part J` for V*, `rec-compose` for the composition), with the same exit-code meaning the scripts' callers expect; say in your note what each script's caller reads. Unrun is fine (recursion is parked), but `bash -n` each.

