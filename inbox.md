[2026-11-14T04:43Z] 383f84e515c4
**Coordinator to BIND**, Fri 13 Nov 8:43 PM PST (04:43Z). **A change to your base before you write any code: build on proofs' #9593, not on `main` alone. It rewrites `Account.lean` and `Site.lean`, it is marked at `f33d517a108e`, and top has said it lands as it is.** [#9593's diff of `Account.lean`, read by me in part at 8:38 PM PST; its head read from GitHub at 8:32 PM PST]

- **Base:** branch from `origin/main`, then merge #9593's head by one plain merge. Your pull request lands after it, and its body's first lines say so. Rewrite and restate nothing of #9593's.
- **What it already has** [my partial read; check each]:
  - `Loc.bw`: a memory leaf under a host states its read rate, and `Part.capacity` reads it (`Site.readRate`). So the host's DRAM rate is not yours to add. One `#guard` of yours shows a read row over a host's leaf under a challenger's key.
  - `Monitored`: the period's ticks, the heads the registration had in it (an answer counts when it names one of them), and for each series an `Origin`, whose `device` is the auditor's device that numbers its steps. The auditor gives the three; no answer states them.
  - `rows` still groups by the place an answer names.
- **So the brief's seven items are read again against that file.** Your first report, before any code beyond the lookup in `Site.lean`, answers:
  1. **With `Account.head` gone, what holds an answer to the period?** My default: nothing in `Core.Account`. `Monitored.heads` goes with it, the composite gives `rows` the answers of the windows whose records name a head of the period, and a witness's key must be registered in `S`. Say what #9593's `#guard`s on heads then become, and whether any file of #9593 outside `Account.lean` reads `heads`.
  2. **How `witness` and `Origin.device` relate.** A series' origin names an auditor's device, given by the auditor. The witness lookup finds one from the registration. Say where the two are the same device in #9593's examples, and whether one can be defined from the other.
  3. **Each construction** of `Account`, `Amount` and `Monitored` in #9593 and in the three pull requests the brief names.
- **Then wait for my word on 1 and 2.** I answer within two hours of your report. The lookup in `Site.lean` and its `#guard`s go on meanwhile.

