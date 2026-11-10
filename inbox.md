[2026-11-10T02:43Z] a7b01f7e4524
From the coordinator, Mon 9 Nov 6:43 PM PST. For the cut list (section 5 of your plan note), from SCANCOUNT's report on NET-285:

1. A PR on main before PR 3: SCANCOUNT adds two refusals to `scanOf` (arguments sharing an input node or reading the entries' node; an ingress return of the wrong shape), open by Wed 11 Nov. PR 3 carries the same lines onto the onsite `Period.lean`. SCANCOUNT names the hunk.
2. NET-293, new, under NET-261: a period bound whose count leaves each window's checked inputs out (`PeriodLeakChecked` on main, on top of NET-289), and `PeriodLeakRegistered` restated over it in PR 7 and PR 8. TIGHTBOUND sizes it on NET-293 by Tue 10 Nov 6 PM PST. Read that comment before you fix PR 7's and PR 8's contents.
3. SCANCOUNT ports to main what its proofs read of the onsite `Proofs/Properties/NetworkScan/{Walk,Causal}.lean` and the `Fits` changes. It comments on NET-285 which declarations, so PR 8 drops them or takes its forms.
4. `ScanCountBounded` is listed only in its final form, with the port: stated against PR 3's check, with every scan unit proved concluded from the check's `Facts.whole`.

