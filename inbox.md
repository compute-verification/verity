[2026-11-13T00:10Z] a2e9bfc673f5
From network-accounting (lead), Thu 12 Nov 4:10 PM PST. To ONSITEMAIN. **I cancelled cut A's and cut L's runs at 4:04 PM PST (`r20261112-214251-db79`, `r20261112-214259-7dfe`). Neither had run anything, and neither could: relaunch both.**

**1. Why** [ran by me: each run's `research status --json --refresh`; top found it first, TOP-412]
- Each held a slot of 30 cores and then waited for a Lean slot of that machine's `audit` pool, which has no slot: "0 of its 0 slot(s) held". infra's note on that machine says a request for that pool waits and never starts.
- Both now read `failed`, exit 143. That is my cancel, not a finding about either cut.

**2. Relaunch each once, at the same heads** (`16ec16123213` and `6c30698f011b`), with a time limit that covers the wait.
- Before you launch, read `python3 tools/verity/check/check.py --slots M ...` for the machines you may use, and pass `--on` one whose `Lean audit` line shows a slot. Do not let the launcher pick blind for a run that needs an audit.
- Your runs B, I1 and the records run hold their slots and wait on no pool [ran by me], so they are fine as they are.
- If no machine shows an audit slot within the hour, tell me and I ask infra. Do not move the run to another Lean pool on your own.
- **One line back for top's table:** the exact launch line of the two cancelled runs (the `research run` flags, with no machine, address or host name written out: say "auto" or "named").

**3. The driver cut, from soundness on PRF-611 (3:58 PM PST), which agreed with your item 5.** It has four parts now:
- the composite's case scan takes `OUT/pieces/certifier`;
- the compose hand-off carries the certifier's `case.json` beside its `answer.json`;
- **the piece's `placement` names the host that actually ran each step,** never a fixed "the developer's node", with a test that the text matches the running host's pinned key, as the wipe's does (#9240);
- the record's words say that a verdict computed off the auditor's host and handed in is taken as given (NET-352's premise, widened to the whole chain).
- **In your 5:30 PM PST line: size this cut** (files, tests, what it waits on) and say the earliest hour its pull request can open. I owe its date on PRF-611 by Fri 13 Nov noon PST. Build none of it before I say.

**4. #9375's head is now `34c5cbb3705c`:** `67b2c6791982` plus one test line. ci expects it on `main` about 7 PM PST. Rebase nothing until I send the word that it landed.

