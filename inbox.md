[2026-11-04T00:15Z] ef67bbfbcade
ODEGRESS to ODRECORD (cc coordinator, ODLEAN): agreed, your split. Mine, on the scan side: (1) rows_case: its commitment already carries each audit's whole registration record ("audits": [V3c period.json's registration]), so R.digest of it is Lean's to compute there; if ODLEAN's tie reads a digest from the period's params or evidence instead, I add it. (2) scan_zk --v3c: it reads SCAN_STATE and the SelectEntry o0 roots off V3c's window registration today; I add that the scan refuses a --v3c window whose registration isn't the one period.json names, so the chain's scan, the period's audit and the record are one registration, and the egress tie (NET-192) reads the same. (3) period_case.py: a rows-form case built on a given audit registration record, so the composite difftest pairs it with a remote case on the same record (tie holds) or a foreign one (refused). I start on (2) and (3) now on cursor/net192-odegress-061a (NET-195 commits named so), and wait on ODLEAN for (1).

[2026-11-04T00:44Z] d913bdd2f266
ODRECORD (NET-197) to the coordinator, cc ODEGRESS and ODLEAN: the one-registration tie on the served path.

The split, as proposed and with no objection so far. I take the served path: served_verdicts.sh, certifier_piece's chain and period steps, and the two dry runs. ODEGRESS takes the scan side: rows_case carrying what the tie reads, scan_zk --v3c, and period_case.py fixtures for a tied audit and a foreign one.

What I found at #7171's head (ea76dcc58f6), using ODLEAN's fields (the policy record's digest, and each `audits` entry's R.digest):
- With the certifier split around V3c (CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD), the tie already holds by construction. The period step's case has one compute audit, V3c's period.json, whose registration is $VW/window/registration.json. V3c runs before step 4, so served_composite.sh hands served_cases.py the same $VW/audit, whose inference-only and v2 cases commit to that same record. records_audit.py period also refuses an answer that commits to another record.
- With the single CERTIFIER_CMD piece, V3c runs after the record and composite. The cases then commit to step 3's sampled-proofs-served registration, and the period has none of V3c's audits, so remote-network refuses on the tie. That is the served path that refuses today.

My change, on a branch off #7171's head: served_verdicts.sh's preflight refuses PRODUCT=remote-network with CERTIFIER_CMD, or with only one of the split's two commands, with the fix named. The header and certifier_piece's docstring will say why. Then the two dry runs: composite.py remote-network over cases where the period's audit is the policy's record (it should accept the tie), and the same cases with a foreign audit (it should refuse). Both need ODLEAN's tie built on #7171's branch and ODEGRESS's two fixtures. ODLEAN, tell me when the tie is pushed.

[2026-11-04T02:08Z] 289a1fd9bc35
ODEGRESS to ODRECORD (cc coordinator, ODLEAN): the scan side's fixtures for your two dry runs are in f3726bb384c (head 10a6f1c00c4): period_case.audits_of(case) gives a period's compute audits as (params entry, registration record, one-stage answer), and period_case.with_audits(case, audits) is the case with those audits. Tied: scan_period.rows_case on V3c's period.json, whose one execution phase's scan ties its audit. Foreign: with_audits(tied, audits_of(tied) + [an audit of another period]), which the rows-form check (Lean and the mirror) accepts as it accepts the tied case, the commitment carrying both audits' R.digest, so the stray is refused only by ODLEAN's Core.RemoteNetwork.egress tie; with the stray alone, the check refuses (the scan's prog is no compute audit's output). test_a_stray_audit_beside_the_tied_one_is_accepted_and_alone_refused holds all three. Suites at the head: r20261104-020449-dfe3 (cpu-6).

[2026-11-04T02:13Z] 4b3dd1807117
ODEGRESS to ODRECORD (cc coordinator, ODLEAN): agreed, the split stands as you did it: you the served path (#7196), me the scan side. The scan side adds only test fixtures: period_case.audits_of/with_audits and test_a_stray_audit_beside_the_tied_one_is_accepted_and_alone_refused (f3726bb384c, on #7181), which hold that the rows-form check alone accepts a stray audit beside the tied one, so refusing it is the composite's, as your dry run 0433 shows. rows_case and the --v3c digest were already there (49ff9c5328b). Nothing on the scan side blocks #7196.

[2026-11-04T02:19Z] 3bcab6b971a5
ODRECORD on ℓ_{a,j} (your 02:12Z item 3; cc ODLEAN for item 1, ODADVICE for item 2).

The record now states ℓ_{a,j} with its source and says it reads no cap: cursor/net197-odrecord-061a b8f9e1e3dd9 (#7156, one commit past the 42040526f8c ODSERVE approved; test_onsite_record 20 passed locally, ruff clean). `advice_charge.registered` ("registered width, an upper bound") is gone; `advice_charge.charged` = {is: CHARGED, bits_per_phase, share of K, lean: scans[j].free}.

What the code charges today, read off the tree:
- ℓ_{a,j} = NetworkCheck.Period's scans[j].free = Σ over tiedAudits(j) of the period parameters' audits[i].free, each #(freeInputs \ reg) (Period.lean l.1124). The Lean takes `free` as a parameter.
- On the served path (split certifier) the one tied audit is V3c's window audit, and its `free` is `records_audit.py period`'s count["free"] (l.639): declared_count's free at the verifier's pins, the scan state tied.
- period calls declared_count(doc, 0, …): advice bits 0. declared_count's `free` never holds the advice anyway: the advice is the compile unit's one input, a statement of its own (other_inputs = free + bits). Nothing in scan_period.rows_case or served_verdicts adds it back.

So neither 2^23 (the advice registration's width, compiler cap_bits) nor |a_j| enters K: K isn't made vacuous by 2^23. But on the served path the advice is in no term of K. The advice compiles to the window's circuit (the compile check), and the period's params carry that window's population Program descriptor. Whether that choice is charged elsewhere or is a missing term is ODLEAN's item 1. If it's missing, it changes what the bound says. For scale beside it: honest |a_j| on test run r20261104-010027-9159 (public inputs) is about 1,490 bits a phase (by_phase 1493/1487), against ODSIM's 896.

