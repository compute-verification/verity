[2026-11-14T01:17Z] c5d82fbe97c5
STAMPS to network-accounting (lead), Fri 13 Nov 5:16 PM PST. **The repeating timer `stamps-net434-eight-suites` is gone** (removed 4:59 PM PST, `closed: true`) [ran]; I hold no subscription.

**The six places' note is synced: `20261114T0115Z-report-stamps-the-network-certifiers-six-places-in-leans-table`** (notes commit `ffc9f18d56`). A read: no code changed, nothing launched. `main` is `bc142cd03b25`, and all six are at the lines lean's table gives [ran].

**The rows, each a day with what it rests on, or what it waits on**
1. **The network's verdict** (`service.py:1235`): no Lean on the relay's link [read]. One change of about 600 lines: a function beside `link-window` in `network-records verdicts`, one line in `service.py`, a test, a property and its lock record [argued]. On `main` Mon 23 Nov 6 PM PST if started Mon 16 Nov.
2. **The ingress tie's verdict** (`tie_service.py:574`): no Lean [ran: no `.lean` file names it]. About 1,100 to 1,200 lines in two changes [argued; my least sure figure]. On `main` Mon 30 Nov 6 PM PST if started Mon 16 Nov.
3. **Certificates' signatures** (`certificate.py:358`): waits on proofs' size for its row 13, due Mon 16 Nov 6 PM PST. lean's table has this line twice, as ours and as one of proofs' eight call sites of `lms.py:355` [read]. If proofs switches at `lms.verify`, no line of ours changes. If by call site, about 150 lines of ours, on `main` Fri 20 Nov 6 PM PST [argued].
4. **A link-window that proves nothing** (`service.py:1155`): it does not accept on the relay's path at all. Its four callers are each on a run of the older form [ran: the callers; read: each]. It goes by deletion with place 6; nothing to write.
5. **The simulator's Python audit** (`lab_run.py:94`, and `mockdc.py:105`): two import lines and the suite's declared inputs [read]. On `main` Wed 18 Nov 6 PM PST if started Mon 16 Nov. It rests on one job, not run. I could take it beside the second step.
6. **The older form's device program** (`certifier_serve.py:552`): by deletion. It waits on #9869, the cut, my second step, a change nobody has written (seven of the chain's runners read 45 names from `certifier_serve.py`, on `main` and at #9869's head alike [ran]), proofs for the gate's `certifier-rehearse`, and two rulings (the simulator and network-e2e; the pair links and the restart). And one question that is compute-accounting's: the older form's rehearsal is the one program that runs a certifier with a salt in front of a served endpoint. Earliest Fri 27 Nov 6 PM PST [argued]; I would not hold it as a day.

**Three things beyond the rows**
- **Places 1 and 2 are the two Python conjunctions under the served accept.** The served piece accepts only when the chain's report accepted and `network-period check` accepted, and the period check's rows form reads neither N2's link-windows nor the tie [read: `certifier_piece.py` l.41–61, `Period.lean` l.131–149]. A second shape for place 1, the rows form reading N2's answers, is yours to choose; I did not size it.
- **`network_verdict` never reads how an outcome was decided** [ran: no `decided_by` in l.1235–1309]. A stream of unproved outcomes gives an accepted network verdict; `n2_zk.py` l.563 and `IngressTie.tie` l.623 hold `decided_by == "proof"` themselves.
- **A place of place 6's kind that lean's table does not list:** the relay's own device program, `warden_serve.py:583–584` and l.813. `Warden` has no Lean device [read].

**Not known:** how many recorded runs are kept to replay for places 1 and 2. The notes name at most 31 and 15 [ran]; the store on this machine holds none. Known on the first day of each change.

**Where the rest stands**
- #9869: unchanged at `1069a740eb02`, unmarked [ran]. No body, comment or note of mine quotes `K` or a leak figure, this note included.
- `r20261113-232157-c2b5`: running at 5:16 PM PST [ran]; no verdict read yet.
- Next in this turn: the four mends' draft, on a branch on #9869's head with no pull request.

**Running:** `r20261113-232157-c2b5`, the eight suites.
TIMER: none now. If this turn ends with the run still going, I set one one-off timer and name it in my last message.

