[2026-11-13T00:51Z] 0111c0235b60
From your coordinator (memory-accounting), Thu 12 Nov 4:51 PM PST (00:51Z) (the same text is in your inbox; no reply wanted). **ASKPAGE: two more things for the page, from top (PRF-611, 4:47 PM PST). One is a number, the other corrects a sentence of mine. Your final is still due Fri 13 Nov 7 AM PST.**

**1. The number: how much of the store the device has seen after a week, a month and a year of sessions**
- top's words: "Give the page the sessions in a monitoring period, at the demo's rate and at the rate the design intends, and the share of the store the device has seen by the period's end. If that share is near all of it, (b) exposes nothing that (a) does not, only sooner, and the two parts get one answer. If it stays small, they are different questions."
- **What I read** at #9375's head `34c5cbb3705c`:
  - An audit has one salt and one session (`challenger_party.py` l.23-27).
  - A store is set up once and audited every period (`verifier/period.py` l.4-5).
  - `demos/onsite_served/config.toml` sets `challenges = 320` and no period's length. `verity run` takes `--periods N`.
  - I found no period's length in clock time by grep in the demo's README or PoUS's `PROTOCOL.md` [searched, not read whole].
- **My arithmetic, to check, not to copy** [derived]: at 983,040 labels a session is 1 part in 3,072. If places are uniform and independent between sessions, the share of distinct labels seen after n sessions is 1 − (1 − 1/3,072)^n:

  | sessions | a week | 30 days | 365 days |
  |---|---|---|---|
  | one a day | 0.23% | 0.97% | 11.2% |
  | one an hour | 5.3% | 20.9% | 94.2% |
  | one a minute | 96.2% | all | all |

  Half the store at 2,129 sessions, 90% at 7,072.
- **Read and say:**
  - how a session's places are drawn (uniform? repeats inside a session? independent of earlier sessions?);
  - how many periods Tue 24 Nov's demo runs, over how long;
  - whether the design names a rate anywhere (the Glossary, PoUS's `PROTOCOL.md`, `security/Definitions/Pous/`). If it names none, the page says so and gives the table by rate.
- **The share falls as the store grows.** Give one more row for a store of production size: `period.py`'s docstring names Demo B's `C*`, 1,060,008,348,186 bits. By my arithmetic that is about 4% in a year at one session an hour. The page says which store each row is.
- **On the page:** three or four lines and the small table, in Background beside "Today the device sees". If the share stays small at the demo's rate and at any rate the design names, A stays the recommendation. If it does not, say so at the top of your note and I decide.

**2. A correction of mine: one statement already names the two premises**
- I wrote "No PoUS statement names either yet". That is wrong. `MemoryChallenger.Attest.Properties.AuditorSeesSkeleton` (`security/Properties/Attest.lean` l.35-38, proved in `security/Proofs/Properties/Pous/MemoryChallenger/Attest.lean`, in the lock) says: under `Faithful`, `Confined` and `WitnessPathPrivate`, what the auditor sees of the party is what its code sends in the clear on the skeleton of its inputs, so it learns nothing of the records from the party.
- **Read what it covers:** which code its `Step` and `honestCode` are (the signing of a period's root and the liveness records, as I read the header), and whether the timed session's answers (the labels the device receives) are among its inputs. I expect not.
- **top recommends yes to (a)** on one condition: "PoUS's statements name `Confined` and `witness-path-private`. The page names who states them and by when." Who is us.
- **So write in your note, not on the page:** what is already stated, what is missing for the session's labels, the candidate statements with your recommendation, and how large the Lean is. I set the date from that. lean's cut holds Lean from Fri 7 PM PST to Sat noon PST.
- The page then says, under (a): what `AuditorSeesSkeleton` already gives, the one statement that is missing, and "memory-accounting, by DATE" with DATE left for me.
- top's sentence stays quoted; add after it that the 320 answers are labels that decode alone to 2,054 bytes of the weights each.

