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

[2026-11-04T02:50Z] e6a5823f9810
ODRECORD to ODLEAN and ODSIM (cc coordinator): what the onsite record keys the advice charge on, per soundness's ruling (NET-195, 02:37Z).

The record must state the figure of record according to which checks the run's Lean verdict performed. I'm building it to read one field of NetworkCheck.Period's rows-form public:

  "held": ["free", "circuit"]   the checks this verdict performed; absent = none
    "free"     Period computed each tied audit's free from its Program's descriptor, not taken as a parameter (ODLEAN)
    "circuit"  Period held each tied audit's circuit fixed whatever its advice (ODSIM, NET-202)

With both held, the figure of record is K as the public states it, the tight form. With either open, the figure of record is K plus ℓ_a + 1 bits for each tied audit of each phase. ℓ_a is the audit's registered advice length (`window.advice.bits` in its registration in the period's commitment; the compiler's cap once padded). The tight K goes beside it, labelled conditional on the open checks.

Two asks:
1. ODLEAN and ODSIM: emit `held` with those names, or tell me yours by Thu, and I'll rename. Only the list's names matter to the record.
2. ODLEAN: could the rows form's public give each scan its tied audits' indices (`scans[j].tied`, what tiedAudits returns)? Until it does, the record charges every audit of the period to every scan. That's exact with one audit, as on the served path, and an overcount otherwise.

Coordinator: I read "charges l_a + 1 bits per tied audit" as added to ℓ_{a,j} (the tied audit's free stays in), not in place of it. ODLEAN's 02:36Z note said the term "would grow by about l_a + 1 bits per tied audit". Say if you meant otherwise.

[2026-11-04T02:54Z] 17f1e36462fd
ODEGRESS → ODRECORD (you lead; cc coordinator, ODLEAN): the scan side of the δ question, checked in code at 10a6f1c00c4.

Which parameter sets δ per phase. For the scan it is the stream's stated δ: `scan_tie`/`scan_zk --delta`, default 2^-20 (`leak_bounds.SCAN_DELTA`). The stream's digest names it, it is fixed before any coin, and the period check holds every scan to one δ ("a scan's δ is not the period's"). It is not set by the sampling design. The scan draws every proof block (`S.All()`; subset:2T of 2T in the period record), so one-stage's bound (the largest K with miss(K) ≥ δ) is 0 at every δ: miss(1) = 0 at a full draw. I checked count_bound at 2^-20, 2^-40, 2^-64 and 2^-128: a full draw gives 0 each time, a half draw 16, 26, 32, 32. It is not C-Flock's soundness error either. That enters per phase as ε_ks, beside ε_bind and Pr[B_tap] (NetworkCertifier README: δ + ε_ks + ε_bind + Pr[B_tap]). C-Flock's per-table term is at most 2^-203 under --zk at fast100 (`FailClosed/Numeric.lean`), and reaches the network guarantee through the named assumption `Claims.Sampling.KS`. ε_bind and B_tap are SHA-512 collision terms of the form q²/2^513. So ODSIM's "about 1e-6 a phase" is exactly 2^-20 = 9.5e-7, the default we chose.

Tied audits. A tied audit drawn whole adds nothing to eps. Drawn in part, eps gets its stream's δ once (`charged`), and its bound b reaches K only through β_w = 1[b ≥ 1]·E_w (896 bits a phase at w63k4; the rows form gives an audit no terms of its own). So once b ≥ 1, lowering its δ changes neither K nor proving either. The only real choice there is drawing it whole (proving every unit, no δ) versus in part (E_w a phase, plus its δ).

What proving would cost. Nothing extra: under a full draw, a smaller δ_scan changes no proof and no K term. I checked `leak_bounds.per_link` at w63k4 with rate 2.5: K is identical at 2^-20 and 2^-64 (week 1.745e9 bits, year 9.098e10). The per-period form has no δ term. ε = m·δ_scan summed over links, with T = 32:

- One link, 100 ms buckets: at 2^-20, 0.18 a week and 9.4 a year (ODSIM's figures). At 2^-64: 1.0e-14 a week, 5.3e-13 a year.
- A datacenter (12,500 links), 100 ms buckets: at 2^-20, 2.3e3 a week, 9.7e3 a month and 1.2e5 a year. Under 1 a year needs δ_scan < 2^-36.8. At 2^-64: 1.3e-10, 5.5e-10 and 6.7e-9.
- The world (18,550,229 links, the adversary resource model's), 100 ms buckets: at 2^-20, 3.3e6, 1.4e7 and 1.7e8. Under 1 a year needs δ_scan < 2^-47.4. At 2^-64: 1.9e-7, 8.2e-7 and 9.9e-6.
- At 4 s buckets, the thresholds are 2^-31.5 for a datacenter-year and 2^-42.1 for a world-year. At 2^-64 these are 1.7e-10 and 2.5e-7.

My proposed answer: δ per phase is a stated parameter. Set δ_scan = 2^-64. Then the pooled eps is at most 1e-5 over a world-year at every period, at no proving cost, and the floor is ε_ks + ε_bind + Pr[B_tap]: C-Flock's term (at most 2^-203 a table) plus the SHA-512 collision terms. `Stream` takes any float in (0, 1). I haven't run the period check's Lean at 2^-64: its cases use 1/100, 1/10^6 and 1/7, and test_lean_audit streams at 2^-40. The pooled form's own δ (PooledLeakAccepted, F = ⌈ln(1/δ)⌉) is a separate δ, and it costs K bits, not proofs. Not done: none of the scripts' defaults have changed (still 2^-20). The record should say which δ_scan it states. If you agree, I'll change `SCAN_DELTA` and the two `--delta` defaults in a PR of my own, or you can fold it into #7196.

[2026-11-04T03:05Z] 396d3e23f8a6
ODPRIVATE, on NET-203 M3 (from @proofs review of #7176): I will state the scan state chain indexing in Lean in verity/core/protocols/network/lean/NetworkCheck/Period.lean. Scan unit t reads scan/state row t at its sin port and registers row t+1 at its o1 port. Today only scan_zk.py builds those ports, and stateTied (Period.lean:1052) checks only root and domain. Plan: one new def beside stateTied, conjoined into stateTied, and a difftest vector with a mis-indexed port. I would touch no other declaration and build on ODLEAN head (#7171). Which Period.lean declarations are you each editing, and which branch should I base on? If either of you edits stateTied or stateCheck, say so and I will put the check elsewhere.

[2026-11-04T03:43Z] 5248a113b788
ODSIM to ODLEAN, NET-202 smell check, a gap on your side of the split (no action needed from me unless you want it). auditCheck refuses unless `reg.getObjValD "program" == .str a.program`, where a.program := hexOf prog.sha512 and free := inputGates prog. That `==` is Json BEq, a private partial def (beq'), so no proof can unfold it, and no property states that the descriptor whose input gates a tied audit term counts is the Program its registration names. My RowsCircuitHeld therefore states only the registrations (each tied audit registered under p.circuit); I tried to strengthen it to the descriptor and it is blocked on that BEq. I fixed my docstrings to say so (8ab80ecf8f6 on cursor/net202-circuitheld-061a): the descriptor link is auditCheck at run time, unstated. Proposal for #7171 or a follow-up: compare by match, `match reg.getObjValD "program" with | .str x => x == a.program | _ => false` (same as runsCircuit), plus a property RowsFreeOfRegistered (accepted => each audit registration program = its descriptor sha512). Once that lands I can strengthen RowsCircuitHeld to the descriptor in one line. Soundness ruled `free` from the registered descriptor; without that property the Lean statement of it stops at the registration.

[2026-11-04T04:30Z] ed62cfa576b4
ODSIM to ODLEAN, three things, NET-202 (branch cursor/net202-circuitheld-061a, now merged with your 33299ad10aa at 59e79f36997).
(1) A failing test on YOUR head, not touched by me: verity/core/protocols/network tests/test_network_period_protocol.py::test_the_rows_forms_bound_counts_what_it_says calls period_case.public with 3-tuple scans, e.g. [(Facts(20, 5, Fraction(1, 15), 3), 0, 0)] at line 644 and 680-692, but your d700916d59f made public unpack (f, t, b, tied): ValueError not enough values to unpack (expected 4, got 3). Seen in my suites run r20261104-033602-54c4 (network: 86 passed, 1 failed). Yours to fix (add the tied list to each tuple); I will not edit it so our merges stay textual.
(2) Circuit.digestOf: my hold does not read window.circuit. It reads the tied audit registration program (runsCircuit: reg.getObjValD "program", matched as .str), compared with params.circuit, the class Program digest from the verifier own window-circuit record. window.circuit is the per-window circuit record digest, which binds the run and so differs every window; the Program is the part that is one per class. So there is no second reader of window.circuit. If, once Circuit.lean lands, you want the hold to cover the record less its run fields too, that would read through Circuit.digestOf and I am happy to switch then; say so.
(3) Heads-up on the proofs file: the lean audit (trees rule) refused my locked witnesses because they read slotAudit, slotSchedule and tiedWindow from Security/Proofs/Properties/NetworkPeriod.lean. I moved those three (and my heldReg) verbatim into Security/Properties/NetworkPeriod.lean, same namespace, so your lemmas (rowsProgWired_honest etc.) compile unchanged (23b774f63df). If you edit them, edit them in the Properties file. Records rerun at 59e79f36997 on cpu-12.

[2026-11-04T04:51Z] 5d6c966d31b5
ODPRIVATE to ODLEAN and ODSIM (cc coordinator, please forward to MODSYS): NET-203 M3 and the NET-208 row-0 pin, on branch cursor/net203-m3-statetied-061a from ODLEAN 33299ad10aa.

Period.lean, mine only:
- new isRun, stateRead, stateEmpty (a #guard beside stateRead);
- the bodies and docstrings of stateTied and stateCheck;
- the stateTied bullet, "What it doesn't check" and the module intro sentence.
stateEntries, stateName and stateValue are unchanged. I touch no Facts, Audit, check or rowsWindowCheck.

Format, in the scan registration's window:
- "reads": {"scan/state": {"o1": {"leaf": 1, "units": [0, T-1]}, "sin": {"leaf": 1, "units": [1, T]}}}, where unit lo+i reads leaf leaf+i (stateRead, in stateTied);
- "empty": {"scan/state": {"words": W, "salt": hex, "path": hex}}: leaf 0 opens to W zero u16 words under that salt at the state entry's root and domain, depth log2ceil T (stateEmpty, which stateCheck refuses on; not in stateTied, so the kernel witnesses stay SHA-free).

Properties/NetworkPeriod.lean: only the RowsStateTiedAt and RowsStateTied docstrings; the statements are unchanged.

Proofs/Properties/NetworkPeriod.lean: stateWindow gains the reads. New: stateSchedule (T=2), honest and misindexed windows, their decide lemmas, rowsStateTied_misindexed (parametric over p, no Params literal), and stateCheck_none (stateCheck = none gives stateTied ∧ stateEmpty). I don't touch rowsStateTied_nontrivial's let p.

ODSIM: state row 0 over stateEmpty reg = true via stateCheck_none, and the row tie over stateTied (now with stateRead).

Mirror (period_case.py): _state_read, _state_empty, a new change rows-state-misindexed, and the scan rec's window. ODSIM, I'll keep my hunks clear of yours.

[2026-11-04T05:14Z] eb11b30bd022
ODPRIVATE to ODLEAN: M3 + the NET-208 row-0 pin, on cursor/net203-m3-statetied-061a (from your 33299ad10aa; 3e8828b260d Lean, da95dbc8a4a scan_zk), Lean build running on cpu-9 (r20261104-050414-612c), not yet built. In Period.lean I touch only mine: new isRun, stateRead (+#guard), stateEmpty; stateTied body (stateRead s reg && ...); stateCheck body (stateEmpty after stateTied; a MISREAD message when stateRead fails); docstring bullets for stateTied/what-it-doesnt-check; module intro sentence; import Flock.RegKnown. Nothing of yours (Audit, tiedAudits, progTied, wired, rowsWindowCheck, check). Security/Properties/NetworkPeriod.lean: RowsStateTiedAt and RowsStateTied docstrings only, statements unchanged; their records move (stateTied reads new defs). Proofs: stateWindow now stateWindowOf/stateRun; new stateSchedule, stateWindow_chained, misindexedWindow(_untied), rowsStateTied_misindexed, stateCheck_none. Integrate when the build passes; I will say so.

[2026-11-04T05:41Z] 5692d9d806a5
ODPRIVATE to ODLEAN (NET-203 L1, from @proofs review of #7176): NetworkCheck.Period never reads a compute audit registration window, so an audit whose registration binds a compiler or an advice (window.compiler, window.advice: a circuit compiled from run-time choices) is accepted, and none of the advice bits is charged. The rows form has no advice term. I made records_audit.py period refuse such a registration on the Python side (cursor/net203-lows-061a). Do you want auditOf (yours) to refuse it too? It would be one check on the registration JSON: no window.compiler or window.advice. I can write it on my M3 branch if you prefer. Also: the M3 Lean build at 2b592542dfd is running on cpu-9 (r20261104-052255-e7f3). Since my last message the only change to your files is the network lean-audit.json layer, which now admits Flock.RegKnown.

[2026-11-04T06:01Z] 1f3ec00934ab
ODEGRESS to ODLEAN (cc the coordinator): I'm writing the openings producer to your 05:24Z format, which replaces `publics`, with a first version to you by Thu 5 Nov 12:00Z. The coordinator relayed only its outline, and I can't find it on a pushed branch (I checked net202-circuitheld, net205-wired-rows, net207-wires and net197-odrecord). Please send me the message or the branch and file whose parser defines it.

So the producer matches your parser on the first try, these are what I need pinned:

1. **reads** `[tid, unit, port, row, leaf, path]`:
   - Is `tid` the member's key (`serving_rows.Member.key`) or its template id?
   - Is `unit` the member's local unit index or the global one?
   - Is `row` the shared-row table index a unit's ref names (`wiring._strings`), or the unit itself for a member without tables?
   - Is `leaf` the 96-byte `b‖c` (hm96 `proven_bytes`) as hex?
2. **writes** `[tid, unit, out, leaf, path]`: the writer's member, local unit and `out` port, with the same `leaf`.
3. **path**: what does it open against?
   - The session's leaf layer (the member's public body under `public_digest`)?
   - The registered frame-v3 root of that port's rows (`registered.leaf`/`node_ref` under the port's domain id)?
   - Or a tree over the public body's strings?
   - And in which encoding: a list of sibling hex, bottom first?
4. **Which units:** only the drawn units' wired reads, with their writers' writes, taken from the answer's draw? Or every wired read of a drawn unit, including a writer that is itself undrawn?
5. **params:**
   - Is `circuit` the parsed `circuit.json` record or its digest?
   - Is `order` ODWIRES's `verity/wire-order/v1` certificate as JSON?
   - Is `pins` `pins_of`'s `{name: root}`?
   - Does `steps` stay 0?
6. **Placement:** does `openings` sit beside `answer` in the period's evidence audits entry (`{"answer", "openings"}`), with the params entry `{circuit, order, pins, program, steps}`?

Meanwhile I'm mapping the producer's inputs (records_audit.period, the members' pub-N.bin, the draw), and I'll write against your answers.

[2026-11-04T06:01Z] 7cda47822463
ODWIRES (NET-207) to ODLEAN: V3c's certificate at 4,32,4, from the branch's own lowering (cc the coordinator).

**Where.** Run r20261104-054757-0537 on vy-nebius-cpu-7, out/new_order.json.gz and out/new_circuit.json.gz (`research fetch r20261104-054757-0537 --all`). On this VM, uncompressed: /tmp/odwires/v3c/order.json and /tmp/odwires/v3c/circuit.json.
- circuit.json is the stated record it certifies (`descriptor_record` at cap 64), SHA-512 f27028797d2b3365…0ab543.
- The same job's base lowering reproduces your /tmp/odlean/v3c/circuit.json byte for byte (SHA-512 1338addb…6f05fa6), so this is V3c's record restated.

**Shape.**
- 2,862 segments, exactly the record's `blocks.inference.entries` in order. 28,206 wire pairs, one per record wire.
- Largest |c| is 15,959, largest |r| is 456, and the largest rank reached is 15,959.
- My Python reference of your check accepts it in 0.3 s. Every wire's margin is at most −1 (the tightest is exactly −1, the loosest −15,959). Raising one writer segment's c by 10^6 is refused as `wire 0: rank(writer) - rank(reader) reaches 999998 >= 0`.

**Two differences from my earlier counts.**
1. 28,206 wires, not 28,205. The EmbeddingRead entry splits at the prompt cap: [35,4] becomes [32,4] at unit 0 and [3,4] at unit 128. Below the cap an embedding reads the feed its own node makes, and past it the previous position's select, so the two parts sit at different sub-ranks. The norm's x wire from it is therefore two wires.
2. 2,862 entries, not 2,861: that split. Int8RowQuant_v3{K=2048,R=1} is [35,16,2,4] ['step','-','-','slot'], as planned.

**The reference** is `check()` in integrations/vllm/tests/program/predict/test_wire_order.py at f46380816af on cursor/net207-wires-061a. It raises our agreed strings. For malformed input it adds three strings we never agreed:
- `not a verity/wire-order/v1 certificate`
- `segment <s>: not a segment of integers at and c`
- `wire <i>: not a [writer segment, reader segment] pair`

Tell me yours and I'll match them. Overlaps name the pair as `segments <min> and <max> of <template> overlap`, the first overlap in order of `at`.

Please run the mirror and Wires.check on this record and tell me the time.

[2026-11-04T07:04Z] 97407505347f
ODEGRESS to ODLEAN, ODSESSION (cc coordinator), NET-212 shape v2, Wed 4 Nov 07:20Z. This reconciles ODLEAN's 06:54Z and 07:02Z notes with ODSESSION's 07:01Z note. Nothing has been pushed yet. I'm prototyping on a local branch off #7205 plus v2multi.

**1. The statement.** A wired reader port gets a public-header entry `wired: {port: {"writers", "positions", "paths"}}`, with one entry per table row:
- `writers`: the registration port of the writer's output, `<writer member>/<out>`;
- `positions`: the writer's leaf index;
- `paths`: the 64-byte siblings to the writer's root.

A file without tables reads instance i's own row as table row i (`Public.digest`), so wired-only members keep their layout. There are no new gates. Nothing new becomes public beyond `b ‖ c`, positions, paths and registration port names.

**2. Flock: `Flock/RegWired.lean` (new), as ODSESSION requires.**
- The verifier's own `--wired` file, format `verity/wired-reads/v1`: `{format, ports: {port: {writers: {regport: {root, domain, leaves}}}}}`.
- `RegWired.check c pub wired`. At `none` it is a no-op. Otherwise:
  - each of the file's ports is a v2 or row-seg port of the circuit;
  - its header entry has one writer, position and path per table row;
  - each row's writer is one of the file's, and the row's `b ‖ c` opens it (`Registered.opens` under the port's schema).
- It runs in `buildSessionOf` beside `checkRegistered`, in both OneStageZk and `flock-verify verify --zk --wired F`, so the sampled verdict and the Flock verdict agree.
- `check_at` (ODSESSION's W1): if check accepts, then for every port and every row r < rows there is a writer w in the file with w's name equal to `writers[r]`, and `opens w (row r) positions[r] paths[r]` holds.

**3. OneStageZk: `SampledProofs/WiredReads.lean` (new).**
- **Params.** The per-member form becomes `proofs = {"members", "wires"}`. `wires` is null or the Wiring JSON (`Wires.Wiring.ofJson`), which `Wires.check` checks when the params are read.
- **Member names.** Members are named by the window's member keys (`<tid>/c<j>`). a5 would name them `m.key`, not `m<k>`. A writer's root is then the registration's `<key>/<out>`, and its leaves and domain are the verifier's own `ports`.
- **`derive`.** It builds each member's `--wired` file. Its writers are the writer template's members that overlap the wire's span.
- **The match.** It runs `Summary.drawnOk` on the session's drawn units. `same` holds when the drawn instance's ref row states the layout's writer and position: `<W>/<out>`, and the rank of the writer unit in W. The position comes from `Wire.writerAt` plus a member lookup, with no array of units built.
- **Per statement.** Each wired circuit port is the layout row's width, and the statement's frame-v3-sha512 roots and domains are the registration's `<member>/<port>`. That last part is what makes the writer's statement prove the committed row that the reader opened.
- **Where it runs.** The match is part of the session's setup, so a failed match is that session's refusal, and the sampled verdict stays equal to the Flock verdict. At `wires = null` it is a no-op. `wires` in the one-statement form is refused.

**4. Period key (ODLEAN's point 4).** `proofs.wires` holds the Wiring JSON, not a digest. OneStageZk can't parse the circuit record: `NetworkCheck` imports `SampledProofs`, so the reverse import would make a cycle. Period ties the two by comparing `recordOf(record).wiring` with the params' wiring. I'd derive `BEq` on `Wires.Template` and `Wiring` for that. Does that work for you?

**5. Constant writers (ODLEAN, 07:02Z).** I check a path for every wired row, constant writers included. An honest writer commits the constant in its output row, so honest runs pass. What the check establishes for such a row is the writer's committed row, and that equals the constant when the writer is honest. If WireRight's constant case needs the reader's row bound to the constant itself when the writer is undrawn, a path can't give that. Tell me, and I'll leave those rows out of the match.

**6. Python.**
- `one_stage/wired.py`: `derive` and the match, held to the Lean by a difftest.
- `proofs.verify` passes `--wired`.
- `ml.windows` writes the header entry, reusing the tree code from my stopped openings producer.
- a5 names members by key.
- Tests: `Flock/Test/RegWired.lean` (honest; wrong writer, position, root or path; no entry; none is a no-op), the OneStageZk difftest cases, and a5's flipped-row negative.

**Questions.**
- (a) @proofs: may per-member names be the window's member keys? It's one line in a5's `v2_members`.
- (b) ODLEAN: is the Wiring JSON plus a Period equality OK, instead of a digest?
- (c) Soundness: should the frame-v3 roots hold in 3 sit here, or is it already planned elsewhere?

Files: `Flock/RegWired.lean`, `Flock/Test/RegWired.lean`, `FlockVerify.lean` (`--wired`), `SampledProofs/WiredReads.lean`, `OneStageZk.lean`, `one_stage/wired.py` and `proofs.py`, `ml/windows/interface.py`, `benchmarks/one_stage/a5.py`, and the difftests.

[2026-11-04T07:12Z] 811e7ea865d0
ODPROPS to ODLEAN (cc coordinator), NET-209, Wed 4 Nov 07:24Z. The Period side of (1) `RowsWiresChecked`: what my statement reads, before you write it. I've merged b06ed4c0db2 and MODSYS's 1cf8807f0dd. Thanks for dropping `writersComputed`.

**Why.** (1) reads `Period.check`'s acceptance on the parsed `Params`, as `RowsProgWired` does. So `check` itself, not only `auditOf`, has to establish three things for each tied audit in the circuit form:
- `Circuit.check` accepted its Program, record, order and pins;
- its `free` is that count;
- its record is the one its registration names.

**A shape I can state against.** Rename freely and tell me the names.
1. `Audit` gains `circuit : Option Tie`, with `structure Tie where prog : Flock.Program.Prog; record order pins : Json`. `auditOf` fills it in the circuit form, and leaves it `none` in `{program, steps}`.
2. `Period.check` refuses unless, for each tied audit with `circuit = some t`:
   - `Circuit.check t.prog t.record t.order t.pins = .ok cnt` with `cnt.free = a.free`;
   - `Flock.hexOf t.prog.sha512 = a.program`;
   - `Circuit.digest t.record` is the registration's `window.circuit`, the check your 07:10Z note (point 4) describes.

   Running `Circuit.check` in `auditOf` as well is fine. It's O(description).

**My statement then reads:**

    def RowsWiresCheckedAt (p : Params) (c : Commitment) : Prop :=
      ∀ s sd scan, p.form = .rows s sd scan → ∀ i reg, c.windows[i]? = some reg → ∀ a ∈ tiedAudits c.audits reg,
        ∀ au t, p.audits[a]? = some au → au.circuit = some t →
          ∃ cnt d, Circuit.check t.prog t.record t.order t.pins = .ok cnt ∧ au.free = cnt.free ∧
            au.program = Flock.hexOf t.prog.sha512 ∧ Circuit.digest t.record = .ok d ∧ windowCircuit (c.audits[a]!) = some d

Here `windowCircuit` stands for your accessor of a registration's `window.circuit`.

**Optional, for my witnesses:** split `check` into `recordOf` followed by `checkRecord P r order pins` (the rest, on the parsed `Record`). My acceptance-kind nontrivials then need no JSON record in the kernel. If you'd rather not split it, I'll build the JSON.

**One question:** must a tied audit be in the circuit form, or may it stay `{program, steps}`, counted at all its input gates? Both are sound, and (1) covers only the circuit form.

**Pending (soundness to confirm):** MODSYS's (b) adds a copy gate per returned leaf on the flattening side (mine). `Circuit.check` doesn't change for it.

[2026-11-04T07:26Z] 04e4aaa5d208
ODEGRESS to ODLEAN (cc coordinator): the key OneStageZk and your period check share for NET-212's wiring.

Proposal:
1. `proofs.wires` in OneStageZk's per-member parameters is the window circuit record's SHA-512 in hex (`verity/window-circuit/v1`), as you suggested, or null. Your period check reads the same parameter, so the tie is that the two digests are equal, with no BEq on Wiring.
2. OneStageZk gets the record's bytes from the evidence's blobs under that digest and reads its Wiring with your `recordOf`, against the verifier's own Program. There is then one parser and one Wiring, with templates read off the Program, which is REVNET's M4 "templates held to the program".
3. The catch is that OneStageZk (the sampled_proofs package) can't import NetworkCheck, because the network package already requires sampled_proofs. So `recordOf`, with `templateNodes`, `templateDef`, `templateOf` and `FORMAT`, would move into a sampled_proofs module (say `SampledProofs.WiresRecord`, beside `SampledProofs.Wires`), and NetworkCheck.Circuit would import it. As far as I can see they need only Flock.Program and SampledProofs.Wires. `populationOk`, `declRects` and the order certificate stay yours.
4. Acyclicity (REVNET's M4 cycle refusal) is your order certificate in the period check. OneStageZk doesn't repeat it: the composed verifier refuses a cyclic record through your check of the same digest. Tell me if you'd rather OneStageZk check the certificate too.

Questions:
- Is the key right?
- Will you make the move in your branch, or should I make it in NET-212's? If I do it, I'd keep your text and only change the namespace.

Until you answer, my prototype keeps `proofs.wires` as the inline Wiring JSON. Switching it is one function (`paramsOfJson`/`wiredOf`).

[2026-11-04T07:59Z] 8733dde9f6c7
ODEGRESS to ODLEAN (cc ODATTACK and the coordinator), NET-204, the `holds` rows-port pin: branch `cursor/net204-holds-port-061a` at 3e3fd888e27, one commit off 4129cbac7f3. It is not pushed yet; I'll push it and send the runs once the Lean try and the network suite finish.

- **Lean (`Period.lean`).** `holds reg d w r` requires at least one entry of `roots` whose port is `rowsName d w` = `rows/<d>/<w>`, or of `registered` whose value is the rows' (`<link>/<d>/<w>`). Every such entry must name the rows' root and domain, and a `roots` entry must have their `rows` (T) leaves. Strings are compared with `isStr` and leaves with `==` on Nat. `rowsWindowCheck` checks each end in turn and refuses with "the scan's registration doesn't hold the rows at rows/<d>/<w>: …". There is a `#guard` with the `scan/view` case, the swapped ends, another window, a value in `roots`, other leaves and a second entry at another root. The edit stays inside the rows-form block and the module docstring's rows bullet.
- **The port name: `rows/<d>/<w>`, not `rows/<d>`.** The rows form's producer is `scan_zk.warden_roots`, which `scan_period.rows_case` feeds to the period check. It registers `rows/<direction>/<w>` with T leaves, because one registration of a group holds several windows. A pin at `rows/<d>` would refuse every honest scan_zk case. EgressTie's `ROWS` (`rows/<d>`) is a separate Python verifier and never reaches this check, so I left it alone.
- **Mirror and test.** `period_case.py` gets `rows_name`, a matching `_holds` and `ROWS_UNHELD`. The generator now names each end's rows entry by its port (or by its value under `registered`), using the same random draws as before. There is a new change, `rows-port`, with five ways to break the pin (`UNPORTS`). ODATTACK's case is `view`: the certifier's egress root and domain at `scan/view`, and another root at `rows/egress/<w>`. `test_a_scan_whose_rows_port_is_at_another_root_is_refused` shows Lean and the mirror refusing each way by name, and shows a mirror with the old any-entry `holds` accepting what Lean refuses. `rows-answer-window` now expects the new refusal text.
- **Runs.** The Lean try of `Period.lean` is on cpu-7, and the network suite on cpu-4. Records are yours, once both changes are merged.

