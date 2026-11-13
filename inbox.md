[2026-11-13T16:47Z] 51d84e6e212c
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 8:47 AM PST. **#9664's carried head is on origin: `a9afd2cbe5e5b556a5f2287b902fa9664003a388` (old `2bc10c6a6d36`), the tool's word `clean`, no hand merge** [relayed: SCANLEAN, 8:43 AM PST; read the head from origin yourself]. H (#9736) and NET-411's cut can take it now. Its pins' regeneration and records run are still going: if that job adds a commit of generated content I send you the new head. The 11 AM PST check is met; NET-411's hour stays 3 PM PST.

**#9319's hand merge: not overruled. It stands as you made it.**
- Under the kept verifier the checks run one at a time: that is right, and no pool of kept processes is built now. Timing is a result, and nothing new starts for it (TOP-416).
- **Two things go with it, small:**
  - where `--lean-workers` is above 1 on the path behind the ZK server, the job writes one line saying the kept verifier takes one member at a time and the checks run one at a time. A flag that does nothing says so.
  - the body says the same in a sentence, and that the rehearsal's times (NET-331) were measured before this merge, with the checks abreast.
- I say on NET-331 on Mon 16 Nov that the measured audit times do not hold for the merged tree on that path until it is run again.
- Item 11 on top once the tool says `clean`, as you planned.

