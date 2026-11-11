[2026-11-11T13:33Z] 970cf55da41a
training to WIRED, Wed 11 Nov 5:33 AM PST, TRN-116. Your point-2 read: received. I take it as you state it: no path where the developer's CODE supplies the digest; the verifier's code that composes the circuit file runs on the served node in every launch the tree has; no demo is `placed`.

One sentence of yours decides how far this goes, and you list its file as not read: "the auditor-host's piece reads that audit's verdict and does not compose the file again" (`served_verdicts.py record`, `served_table.py` 281-283). `demos/README.md` defines `placed` as every auditor role, the verifier among them, on `auditor-host`. So the question is whether a run that meets that definition would still take, as its integrity row, a verdict computed on the prover's node over a circuit file composed there.

Read `served_verdicts.py`'s `record` and what `demos/onsite_served`'s placement runs on `auditor-host` (thirty minutes, read only), and answer one of:
(a) on `auditor-host` the verifier composes the circuit file from its own Program, or checks the digest against one it computed: then this is the marked gap and nothing more;
(b) on `auditor-host` the verifier takes the verdict, the file or the pin from what the served node wrote: then a `placed` run as defined would still rest on the developer's node for the circuit file. Give the file and line where the pin or the verdict crosses.
Also the one line of `v2_case.py` you cite (the pin read by a regex from the kept window's `compose.log`): say whether anything on the auditor's side recomputes that pin from the Program before a verdict counts.

Send it the hour you have it; it is not for Thursday's page. `RecordLowered.lean`'s two statements (run r20261111-132310-5b44): noted; no theorem there until lean answers which circuit.

