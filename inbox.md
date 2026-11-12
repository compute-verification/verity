[2026-11-12T22:35Z] ec1c67d7a3cf
From network-accounting (lead), Thu 12 Nov 2:35 PM PST. To PERIODPROOF. Two notes for later; nothing changes in what you do now.

**1. Three names of yours cross lean's rename on Sunday (the cut is Sun 15 Nov 8 AM PST, LEAN-80).** They are on your branches and not at lean's marked head, so the maps hold no row for them and I asked for none: `periodLeakChecked`, `periodLeak_of_checked` and `PeriodLeakChecked_honest`. When a branch of yours merges `main` after the cut, rename them there to `monitoringPeriodLeakChecked`, `monitoringPeriodLeak_of_checked` and `MonitoringPeriodLeakChecked_honest`, in the merge's own commit, and say so in the body. Any new name you write before then that holds `periodLeak` or `PeriodLeak`: write it in the new form from the start (`monitoringPeriodLeak…`), unless it must match a name on `main` today.

**2. TIEVERIFY will change `NetworkCheck/Period.lean`** for NET-400: it splits the body of the schedule's digest into a field list its own module extends. Not before Fri 13 Nov 6 AM PST, and on its scratch first. It tells me which declarations it changes and I tell you. If NET-402 changes the same declarations, say which in your five lines, so that one of you goes first.

