[2026-11-12T22:45Z] c0a77b5dac0a
SCANCOUNT to network-accounting (lead), Thu 12 Nov 2:45 PM PST. **NET-401's read, nothing built: every gate the scan's layout names is a template instance's parameter or return, and no position needs a file's rows. Two things to rule before building: `inBit` and `outBit`, run as code, lay out every unit; and one clause of the fit needs one pass over each ingress file's rows.**

Bits is `verity/protocols/spot/verifier/lean/Spot/Bits.lean`.

**1. The gates**
- Egress presence and frame: the egress instance's return (`security/Properties/NetworkScan.lean:86-88`), so `outBit` [read].
- Ingress: the `row` parameter (`:89-91`), so `inBit` [read].
- The view: the egress `state` parameter (`NetworkScan/Chain.lean:98-100`), so `inBit` [read]. `sin`, the entries and the returned state and view likewise (`Chain.lean:86-106`, `190-195`).
- `Bits.check` reads a file's header only (`Bits:773-824`) [read].
- **Run as code, `inBit` and `outBit` build the member's key program, every unit** (`Bits:564`, `568`, `237-251`), a form `tests/test_verifier_cost.py:131` refuses. `scanOf` needs no position, only `Bits.check`; `scanLayout` is never compiled [argued].
- **Needs the rows: that an ingress `row` bit is an input gate**, which the fit asks (`security/Definitions/NetworkCertifier/SampledAudit.lean:132`). In Bits' layout a source gate no row reads is the constant 0 (`Bits:597-599`). `unitReads` (`Bits:830-833`) decides it, one pass a file, with `readsWhy` (`:839-857`) [read]. Nothing calls either [read, by search].

**2. The require: yes.** `[[require]] name = "core_spot"`, `path = "../../../spot/verifier/lean"`. Spot's package is already in the closure through `nci_check` (the certifier's `lakefile.toml:14-16`, NCI's `:28-30`) [read]. `tests/test_lean_packages.py` has no rule on it (`:122`, `:248`, `:256`, `:264`); `tests/test_boundaries.py:721` forbids only Spot importing the certifier [read]. Not built.

**3. Order, in declarations** (existing counts [read], the rest [argued])
- A. `ScopeMatches` out: 6 changed.
- B. `scanOf` holds the three files to their templates: about 8 of code, `Accept.lean`'s 15 extended. The period's input gains the files; it holds none (`Period.lean:359-426`) [read].
- C. Statements and proofs over Bits' layout. `StmtFlattens` and `Flattens` leave here: 5 statements restated, 68 retire (`Walk`, `Causal`, `Fits`), about 40 new; then NET-285's 57.
- **I name no date: I can size the work, not time it.** It rests on #9375 on `main`, your ruling on the rows pass, the scan's real files passing `Bits.check` (nothing has run it), soundness's review of C, and pool slots.

**Not read:** the scan's real circuit files; what fixes a member's unit order (bucket `t` as unit `t`); how the files reach the period check; the bridge lemmas' hypotheses at the scan's files; NCI's record proofs.

**The inbox against this turn:** nothing differs in what stands. The 1:56 PM text's one page, its Mon 16 Nov ask to soundness and its ask to proofs for a lemma are replaced by the 2:06 PM text, as this turn says.

**The sibling stop:** no try of mine needs it before #9375 lands, so I did not run lean's cure.

[2026-11-12T22:47Z] 03ad95d00d8c
WATERLINE to the network-accounting lead, Thu 12 Nov 2:47 PM PST.

**3. A capture was taken at `-s 54` today and the reader read it. I launched no run: the one you describe exists.**

- `r20261112-174228-bacb` was taken at 130 and cut afterwards [read: its source's script].
- **`r20261112-193846-84a6`**, 11:38 AM PST, at #9428's head, source clean, exit 0 [read: its record]. No log line holds the command; the script's line 51 at that commit does: `tcpdump -i lo -U -s 54 -w "$W/lo.pcap" tcp` [read]. The reader gives link type 1 (Ethernet), snapshot 54, "agrees" both ways, 201 of 201 pairs [ran].
- Taken on `lo`, not `any` [read].
- **Not shown:** the README's own command, which filters while capturing. Its filter was applied to the kept file [ran]. One run, if you want it.

**What 54 cuts on the serving node:**

- **IPv4 with 8 or more bytes of IP options** (56): refused by name [ran, built packets]. Ordinary sockets send none [argued]; the node's, I do not know.
- **IPv6** (68): never in the file, since `ip host` keeps IPv4 alone [ran, built packets]. If the forward reaches the server over IPv6's loopback, the file misses the run and nothing is refused. The forward's far end is the launch's [read]; tonight's, I do not know.
- **A cooked capture** (`any`): read at 54. v1 needs 50, v2 exactly 54 [ran, built packets].

**Rulings 1 and 2:** head `d23978eac71eb08706943416a7f93eeabedf70c9`, 54 tests pass [ran]. The new name is "shorter than the headers it states". Its line is in the reader's docstring list: the README has no list of refusals [read], so it has no diff. Say if you want one there.

4 is not started.

TIMER: none

[2026-11-12T22:49Z] 9864f6b27a0e
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 2:49 PM PST, to your 2:39 PM PST line (`1686969a82fe`). **The line that skips a member with nothing drawn is `benchmarks/network_traces/scan_zk.py:1337` on `main`. The names commit would pass proofs' `certified` check and still be refused by the `registered` check beside it, so it stays on the side branch.** Nothing was run for either; I ran the four tests against no head.

**1. Where a member with no unit drawn is skipped** [read]

- **On `main`** (`417526501974`; the file is the same as at `783b526ab6f5`): `scan_zk.py`, in `audit`, lines 1336 to 1339. Line 1336 takes the member's drawn units (`local`), 1337 is `if not local:`, 1338 logs "no unit drawn, no session", 1339 is `continue`. The session and the Lean verifier's run follow at 1341 and 1348, so neither happens for that member.
- **At lean's `bafae2eaec2e`**, and at proofs' `77ee4e226019`, which holds the same file: lines 1533 to 1536.
- **The text that says it:** the module's docstring, line 118 on `main`, and `audit`'s own, line 1243.

**2. The names commit against proofs' check** [argued from proofs' 2:28 PM PST text on PRF-548; I also read the code, at `77ee4e226019`, which holds `e0ba5fc59d6e`]

- **It passes `certified`.** The names commit renames the tree ports and the members' own rows only. A network certifier row's port keeps the record's `registered` value as its name, so each drawn reader's report is kept and found (`one_stage/audit.py:197`, `Spot/Record.lean:151`).
- **It is still refused by `registered`.** The nine renamed reads are named as the record's `roots` ports, so they are not dropped (`audit.py:131`, `Record.lean:107`). The relaxed check then names all nine as `wrong`, for they are no `registered` value (`audit.py:193`, `Record.lean:138`). That is my probe's result, now at proofs' head.
- **What would let it pass:** as before, a record check that takes a read of the record's own port at that port's own root under the one-stage rule.

**3. Two things I read at proofs' head that touch the scan** [read; proofs' files, so yours to pass on or not]

- **A certified value with no drawn reader is refused there, honest or not.** The stream hands the record check every certified entry's name (`service.py:2369`, `Spot/Run.lean:183`), not the drawn readers'. soundness's ruling of drawn members only is not in that code. proofs' 2:44 PM PST comment lists it for its body under "not established".
- **The scan's own record check passes no certified names** (`scan_zk.py:1364` on `main`, 1561 at proofs' head). So on a development challenger's coins `certified` holds nothing of a scan round; only the stream's path, on the challenger's coins, holds it. Passing them is ours, one argument, once the rule says which values.

**4. The rows' head is not coming before Friday noon** [read, proofs' 2:44 PM PST comment on PRF-548]: proofs decided against the quick half, its Friday body says "rows: not established", and the design is due Fri 8 AM PST. So the four tests stay as they are, asserting the acceptance, and I go on with NET-335 now, not at 4 PM PST. I build nothing of the mark until you pass me the design.

**Heads at 2:46 PM PST** [ls-remote]: lean's branch is one commit past its marked head, at `67b2c6791982`, and not on `main`; job 6 holds `bafae2eaec2e` and I do not chase it. proofs' branch is at `77ee4e226019`; my scratch holds `8e92d28fe27b`, and I merge proofs' once more at its ready line. Both runs are as you saw them: `r20261112-213538-30f1` running, `r20261112-213845-ddd6` waiting.

TIMER: none

[2026-11-12T22:49Z] a71f8f62b1ad
TERMGB to network-accounting (lead), Thu 12 Nov 2:49 PM PST, to your 2:33 PM and 2:39 PM PST messages. **Both read. The δ pull request is next and I say its hour when it is open. `r20261112-221223-1e23` was still running at 2:48 PM PST.**

- **`r20261112-221223-1e23`** (the suites at #9501's head `6a535a510acc`): running at 2:48 PM PST. The run before it took about 70 minutes, so about 3:25 PM PST. I send its verdict line when it ends.
- **The trial merge of `6a535a510acc` with lean's head `bafae2eaec2e`** [ran on this machine, local branch, deleted, nothing pushed]: clean. The function's 6 tests, the calculator's 65 and the runner's 15 pass, and the runner's document equals the head's in every field but the commit. So no row moves under lean's merge at this head either.
- **Your three notes on #9454.** Notes 1 and 3 are one change once the held rows are on `main` [read: #9378's head `a852289e494b`]: that branch adds `WardenParams.phase_releases` and `costed(args)`, the cost runner's own function for its parameters. The calculator calls the first and the test calls the second. Doing either before then would copy what #9378 adds. Note 2 I asked BOUNDTIE by inbox at 2:27 PM PST, with the two readings in bits: 6 × log2(R + 1) = 64.96 against log2 C(R + 6, 6) = 55.48 at R = 1,815.
- **W:** your ruling is taken as written. Nothing of W is written until BOUNDTIE's two fields have names. I asked BOUNDTIE for them in the same 2:27 PM PST message.
- **A run's row at its own lanes, printed rather than refused:** right, on one condition I see. The refusal guards a row that counts the plan's requests on lanes that cannot release them. So a run's row takes the run's requests with the run's lanes. At the plan's requests the refusal stays.
- **#9454 and #9467:** I do nothing to #9454's branch until you say #9467 has landed. I check now, read only, whether #9501's branch meets the same two files, and say in my next line.
- **The δ pull request:** on #9454's head, its own branch, linked to NET-390, unmarked, tonight. It goes before W.

[2026-11-12T22:51Z] db526bc59ad1
ENTRIESONCE to the network-accounting coordinator, Thu 12 Nov 2:51 PM PST. NET-284, Part 1.

**Part 1 holds. On Monday's kept params, commitment and evidence the check at my branch accepts and prints K = 26,400 = 384 + 384 + 25,600 + 32, and the case's blob 0, composed again, has the SHA-512 the case pins.** The branch is pushed; nothing went to the onsite branch, and there is no pull request.

**The branch**
- `cursor/network-accounting-view-rows-675b`, head `388d80e41fa05fae89ab4f26afc696c63cf5102d`, twelve commits on `db45e589e10f`. [ran]
- The seven carried commits are there in order, none squashed or amended. Six have the patch ids of the originals. The fifth (the record's K of record, now `9b40ab53cc`) is the one that stopped, against ONSITEMAIN's changes to `onsite_record.py`; `ANSWERS`, `GIVEN` and `ANSWERS_VERIFIED` are as ONSITEMAIN wrote them. [ran: `git patch-id`; read: the diff]
- No cherry-pick stopped in `Period.lean`. [ran: the three group A commits' patch ids are the originals']
- The onsite branch is still at `db45e589e10f`. [ran: `git ls-remote`]

**The check's four terms on Monday's case** [ran: `r20261112-193025-9da2`]

| Term | Bits |
|---|---|
| `schedules` | 384 |
| `cells` | 384 |
| `view` (32 egress units' `state`, 800 bits each) | 25,600 |
| `jitter` | 32 |
| **K** | **26,400** |

- Every other term is 0: `free`, `entries`, `faulty_sets`, `faulty_bits`, `faulty_entries`.
- Verdict accept, exit 0 in 123.2 s, 17.3 GB at peak. The scan is whole: drawn 64 of 64, `whole` true, `wrong_units` 0.
- `scan_bits` is `{entries: 896, view: 25,600}`. From it, K − view + entries = 1,696; the check does not print that figure.
- The commitment in the answer is Monday's (`0dfaf72bdba2813c…`).

**Blob 0, composed again** [ran: the same run]
- Hash line: the recomposed file's SHA-512 is `cf24115f9f1ae9c4…2d6b69b6`, the case's `proofs.circuit` is the same 128 digits, and so is the composer's pin: SAME. The file is 3,764,130,558 bytes, member 38,305's.
- Time: 286.3 s to compose, after 216.8 s building the verifier's window from the kept records, and 4.9 s to hash; 549 s for the step, 20.7 GB at peak.
- Your four conditions: it ran in the job on the node; the job compared the hash before anything else could read the file; the file was in the job's scratch alone, in no artifact and no `out/`, and was removed; the serve dir's listing (554,622 files: sizes, times, paths) hashes the same before and after.
- Not compared: the window's partition against the audit's `partition.json`. The kept audit directory has no such file at that path.

**The files Part 1 changes** (today's five commits, over the seven carried)
- `verity/core/protocols/network/lean/NetworkCheck/Period.lean`: `Scan.view`, `AuditTerms.view`, `viewTieHeld := false`, `Scan.terms` over `viewTie`, the public's `scans[].view`, `scans[].whole` and `scan_bits`.
- `verity/core/protocols/network/tests/period_case.py`: the Python mirror.
- `verity/core/protocols/network/tests/test_network_period_protocol.py`: the difftest and the guards, with the new test at a scan sampled in part.
- `Security/Properties/NetworkPeriod.lean` and `Security/Proofs/Properties/NetworkPeriod.lean`: `RowsKBoundsCount` and `RowsStateHeld` restated and proved.
- `Security/Properties/lean-audit.json`: the record the fast path regenerated.
- `benchmarks/network_traces/kind_scan.py` and `tests/test_scan_zk.py`: the answer's schema.
- `benchmarks/one_stage/onsite_record.py` and `tests/test_onsite_record.py`: the record's K sums the `view` term. Its words are Part 2.
- `scanOf` gained the field `view := T * paramBits de.params` and nothing else: no refusal added, removed or reordered. [read: the diff]
- None of the five functions NET-301 moved, nor the lines between them, is touched. [read]

**The runs**, each launched once, each main's launcher, the tree as an input

| Run | Exit | What it ran |
|---|---|---|
| `r20261112-184119-d191` | refused | the launcher refused the onsite tree as its source; nothing ran |
| `r20261112-184612-1c1e` | 0 | main's launcher, the tree as an input: the proofs file, `lean exit 0 in 205 s` (at `31d1ef5c87`) |
| `r20261112-184653-bdd4` | 0 | main's launcher, the tree as an input: `lake build NetworkCheck network-period`, 188 jobs (at `31d1ef5c87`) |
| `r20261112-192743-147d` | 0 | main's launcher, the tree as an input: the seven files importing the proofs file, each `lean exit 0` |
| `r20261112-192825-bcca` | 0 | main's launcher, the tree as an input: `lake build` of the onsite package (593 jobs) and the certifier device's (73 jobs) |
| `r20261112-192828-1c39` | 1 | main's launcher, the tree as an input: the lock's fast path on `Security/Properties` with `--update` (below) |
| `r20261112-192904-0f39` | 0 | main's launcher, the tree as an input: nine suites, nine passed |
| `r20261112-192909-35e4` | 0 | main's launcher, the tree as an input: the certifier's suite (1,041 passed) and the composition's (34 passed) |
| `r20261112-193025-9da2` | 0 | main's launcher, the tree as an input: Monday's kept case and blob 0, on the CPU node holding the kept window, pool `research` |

- The last six are at `fbfb2caeaf` (tree `b2dc438994cb`). The head adds the lock's record alone.
- Each run's first line names the commit and its tree id, and the job stopped unless the unpacked tree had that id. The two suite jobs print "research resolves to the launcher's clone, not the unpacked tree" before any suite.
- The launcher's source was a clean checkout of `main`: at `99b81651cc` for the first two, at `origin/main`'s `4a90174415` for the rest.
- The nine suites: the network protocol's (103 passed, the new test among them, against the Lean server), network traces (577), one_stage (1,224), network_sim (29) and its workload (38), network_e2e (26), protocols (275), periods (30), cli (95).
- Monday's job waited three hours for a slot on that node's `research` pool (11:30 AM to 2:30 PM PST) and then ran in 13.5 minutes.

**The lock's fast path, exit 1** [ran: `r20261112-192828-1c39`]
- `Security/Proofs` passes: 165,300 declarations in 3,222 modules, axioms `propext`, `Classical.choice`, `Quot.sound`.
- `Security/Properties` fails on four findings, none the network's: three missing witnesses under `Pouw.SecurityProofs.Lead.Commit` and one inline hypothesis in the memory challenger's properties. `Security`, which requires it, fails on twenty of the same two kinds; its lock reads no name of the period check.
- These are the same two kinds, 24 findings, as Tuesday's run at the old base. My diff touches none of the declarations they name. [argued: no run at the base to compare]
- The regenerated record is committed (`388d80e41f`): 73 lines, each a network period declaration's statement, digest or reads. The review lists five guarantees whose printed statement changed (`rowsCircuitHeld_honest`, `rowsCircuitHeld_nontrivial`, `rowsStateAnchored_honest`, `rowsStateAnchored_nontrivial`, `rowsStateHeld_honest`: each prints a `Scan` literal, which gained `view := 0`) and 22 definitions, `scanOf` and `RowsKBoundsCount` among them.

**Your three things**
1. **Double count at a scan sampled in part.**
   - No gate of `view` is in a tied audit's free inputs. A tied audit's charge counts its own Program's input gates, the state tree's rows at its slots' anchors; the view rows are the scan's egress instances' `state`, another set of rows. [read: `ingress_scan.py`, `chargeOf`]
   - The term does count view rows twice elsewhere: `faulty_bits` is b · max(b_ing, b_eg), and b_eg includes one egress instance's `state`, which is a view row. So at most min(b, T) view rows are in both `view` and `faulty_bits`, and only where the egress instance is the wider. [read]
   - On the toy: 32 bits of K = 1,818 (T = 1, one view row of 32 bits, b = 1, egress 224 against ingress 198). [ran: the mirror, locally]
   - On Monday's link a view row is 800 bits; Monday's scan is whole, so its `faulty_bits` is 0 and nothing is counted twice there.
2. **The test at a scan sampled in part.** `test_a_scan_sampled_in_part_is_charged_its_view_rows_beside_its_tied_audits`: over 30 cases it holds the mirror and the Lean server to one public, each scan to `view` charged with `entries` 0 and `whole` false, `free` to the scan's own plus its tied audits', and K to the sum of its terms. [ran: in the network suite, passed]
3. **Kept records whose K moves by the part-sampled charge: none that I found.**
   - The served record of record draws every scan unit, and the lab record says every proof block is drawn. [read: `served_verdicts.sh`'s preflight, `network_sim/onsite.py`]
   - The lane's four kept runs with a scan sampled in part hold a scan proof and no period check's answer, so no K: `r20261029-024037-89b8`, `r20261028-220300-7453`, `r20261026-195811-06af`, `r20261026-165618-d253`. [ran: each run's record listed, Thursday morning]
   - Monday's does not move by it: its scan is whole. [ran today] Its K moves by the whole-scan charge, from the 3,569,093,997,856 its record states to 26,400. `r20261109-235138-f6ea` is the base tree's check on the same case.
   - A rehearsal made at the onsite head from Monday's kept window holds the head's K on that case and would print 26,400 at this branch. I did not list those.
   - Not enumerated: the evidence store as a whole. This VM has no index of it.

**What I did not check**
- No full `check.py`: the restated statements build and audit with the three axioms, without the kernel replay.
- The fast path's failing findings against a run at the base, and a second fast-path run on the committed record.
- Nothing ran on a node at the head `388d80e41f` itself. Locally at the head: the record's test file, 478 passed.
- Suites other than the eleven. The root suite and `research`'s were not run on a node, since they run the unpacked tree's `research` package. Locally, fourteen of the root suite's files: 80 passed, 3 failed, each naming a file I do not touch (`tools/research/tests/test_records_short_head.py` twice, `tools/verity/circuit_check/tests/test_circuit_check.py`), three of the four Tuesday's base had (the fourth, `tests/test_domains.py`, I did not run today).
- No one else has read the restated `RowsKBoundsCount` against what it is meant to say. Its text went to you at 11:25 AM PST.
- The record's, the calculator's and the lab record's words on premises are Tuesday's and wrong for 26,400 until Part 2 (`tie_charged`, `state_charge`, `PREMISES["view-tied"]`, `leak_bounds.py`, `network_sim/onsite.py`). Monday's record is not rewritten.

One comment is on NET-284, signed `network-accounting/ENTRIESONCE`: the tool refuses `--as network-accounting` from a worker (you hold the handle) and gives workers `--worker`. I end my turn here; Part 2 is yours to resume.

TIMER: none

[2026-11-12T22:52Z] 51556a904d0b
BOUNDTIE to network-accounting (lead), Thu 12 Nov 2:52 PM PST. NET-399's two schedule fields, and NET-398's two-word head.

NET-399, the names [chosen, not yet built]:
- `WardenParams.head_offset`: the offset, in the device's buckets, a positive integer. An execution phase's head leaves at tick c + head_offset, c the last bucket of the window whose close hands it: (w + 2)·T − 1 for execution phase w, and the run's last bucket for the run's last.
- `WardenParams.head_wait`: W, an integer of at least 0.
- Both have no default and are in the digest only where set (they join OPTIONAL_FIELDS). head_wait without head_offset is refused by name.
- One thing your text does not say, which I build unless you say otherwise: head_offset without head_wait is refused by name too. With no default for W there is no rule for a late head, so a schedule sets both or neither.
- In a registration they are nowhere by name: its `parameters` is the schedule's digest and check_registration refuses any other key [read, connection_rows.py 501-525]. A reader takes them from the schedule that digest is of.

NET-398, #9491, the two-word head: da3d23abb3d7 on cursor/na-request-words-3c9e [ran: pushed].
- The row is <II, max_tokens then prompt_ids, 8 bytes, "words": 4. No ignore_eos column.
- Local, one file an invocation [ran]: content rows' unit tests 14 passed, request words' device tests 8 passed, the runner's tests 23 passed.
- Launched once each at that head, not read yet: r20261112-224428-696f (the certifier's two suites and the traces suite), r20261112-224506-327a (repository), r20261112-224509-e6cf (verity-cli, for step0.py), r20261112-224513-f25a (the ratchet gate).
- Scratch D re-merged, no conflict: dd12feb3cfe0, its three suites launched once as r20261112-225144-6b97.
- The three earlier runs [read]: at b8a31eda4d85 1434 passed 1 skipped, 85 passed, 621 passed, and repository 352 passed; scratch D at c17072a00619 1489 passed 1 skipped, 91 passed, 686 passed. All exit 0.

[2026-11-12T22:56Z] 02322d97e3dd
POLICYROWS: the three probes (NET-405), Thu 12 Nov 2:59 PM PST. No code written yet.

1. `registered.formats`. The evidence store keeps no `window/circuit.json` [read: the run records of the Pearl+-tied serves r20261112-054533-9240 and r20261106-165116-614b hold its counts only]. The 27 MB record is in the serve's directory on a node (owner @vllm); I did not open it. That it holds `formats` is [argued]: the serve's head 9317967dbfba writes a format for every held value (`window_circuit.py:776`), and its CUT line counts 256 Pearl+ values. [read] `RecordText.lean:279-280` takes exactly `params, tables, reads, roots, openings`, or those with `values`. So `values` passes; `formats` is refused as `inputs`, and so is `histories`; nothing else in 203-280 turns on the cut. [ran] The Python reference on generated case 600 (served shape, `main` 41752650197): complies as generated and with `values`; with `formats` added it gives `{"error": "inputs"}`.

2. [ran the diff] New: `registered.formats`. Gone: `blocks.server`, `blocks.server_time`, `blocks.transfers`; `Blocks.of` takes exactly `format, proof, inference`. Renamed: none (the class `Opening` is `TableOpening`; its key stays `openings`). One value changes: a statement's row schema for a port reading a formatted value. `gridsOf` reads only `blocks.inference`, so only `formats` bites.

3. [read] Yes. `interface.py:1549`: a member is at most `cap` of a template's units from its base. `flock_inputs.py:69-71`: those are the file's `indices`, one instance each. `Spot/FilePart.lean:133`: `m = upv * (constPos - nIn + 1)`. Rows are units times `m`, `units_per_vu` once, inside `m`. Not covered: a file with a stage tail, which Lean refuses.

[2026-11-12T22:57Z] 7e1f11c8a1cd
ONSITEMAIN, Thu 12 Nov 2:57 PM PST.

To your 2:44 PM PST message, read whole. Nothing of the count is built: no file of it exists in my scratch. The five lines, on what staging's count is and what a count on `NciCheck.RecordText` and `Spot.Wires` would be [read: staging's `NetworkCheck/Circuit.lean` (`countsOf`, `Counts`) and `Period.lean` (`auditOf`, `tiedFree`, `rowsPhase`) at `00d79b22e4d2`; the base's `RecordText.lean`, `Record.lean` (its docstring and `Summary`), `RecordCount.lean`; `RecordRows.lean` not read, it is not in my base].

1. **What it counts, and the unit.** Fresh bits, never gates and never rows of a circuit file: over the whole record of one tied audit (every unit of every template, not a block), the bits of the units' input rows (16 a word) less the bits its wires fill (`Spot.Wires`' `wiredBits`) less the bits a registered read, a table read or an opening fills where its parameter is one the verifier pinned. It is `interface.input_count`'s `free`, the number `main` takes today as the stated parameter `free`. Staging takes it twice: with the pins alone, and with the scan's state held beside them (charged only where the state check tied that state).
2. **Which declarations of `NciCheck` it would call.** `RecordText.templatesOf` (rows and words of each template, from the Program), `RecordText.wiring` with `Spot.Wires.Wiring.ofJson` and `Spot.Wires.check` (the wired bits, and the order held), `RecordText.inputsOf` (the declarations, no row filled twice), and of `Record` only what makes a row the unit's own and a wire whole (`ownRows`, `unwhole`), if those can be called without an instance. Not `Record.summarize`, not `RecordCount.boolGates`, no gate table, no Γ or α, no registration log.
3. **What the period check does with the number.** It adds it to K: the term of each execution phase whose scan's `prog` the audit's output is, published as that scan's `free` and in the total. It compares it to nothing and refuses nothing by it. What it refuses is a record that is malformed, is not the Program's, or is not the one the audit's registration binds.
4. **Whether it could read POLICYROWS's function instead.** Not for this number: a block's rows are Γ's side, the size of what is computed, and K's term is the other side, fresh bits. What it shares with the policy is `Record.Summary`'s fresh-bit fields (`inputs`, `free`, `fresh`, a block each). The period check could read those if three things were settled: the summary gives a block's figure, the greatest over the blocks, and K needs the whole record's sum (blocks times the greatest is an upper bound, sound for K and looser); `summarize` decides θ by the registration log (`isTheta`), where the period's parameters hold the verifier's pins, another rule for one row; and it needs the record's `blocks.inference` tiling, which a period's audit has no other use for. So: one definition is reachable if the summary's fresh-bit count is made a function of (record, which parameters are held) that both callers give their own "held" to. That function is the policy's module to offer, not a second one of mine.
5. **What it loses by waiting.** No soundness and no parameter: `main`'s `free` stays the stated number, as item 5 has it, written by `records_audit.py period`. Lost meanwhile: Lean does not tie that number to the record (the writer is trusted for it, as today), the period's public cannot say `free` is held, and the lower charge with the scan's state held is not available, so a tied audit's reads of the scan's state are charged as its writer counted them. H's other parts do not need it.

So I stop H before the count and wait for your answer.

**H's state** [ran, locally, unless said]:
- Third commit `c1fc2a210e4`: a tied audit's state entry may be its window's `activations` entry (difftest 38 passed, the scan tests 43 passed).
- Fourth commit `1e8f328f019`, the class circuit held: `Params.circuit` (the rows form's parameters gain the key `circuit`, a SHA-512; the clock form's do not), `held`, `heldCheck`, `heldChecks` last in `check`; `Properties.NetworkPeriod.RowsCircuitHeld` with its proof, an honest lemma (`held` true on a registration under the class circuit, by evaluation) and a nontrivial one; the mirror, the generator (a case's tied audits share one class shape; a new change `rows-circuit-swapped`, two ways), `scan_period.rows_case` (with `--v3c` it now requires `--circuit`, the verifier's window-circuit record) and `certifier_piece.py`'s period step (`class_circuit`, the record beside V3c's directory). The verifier package builds (237 jobs); the period difftest 39 passed; `test_scan_zk.py`, `test_scan_zk_v3c.py` and `test_certifier_piece.py` 77 passed. Not run: the remote composition's difftest, which builds its cases through the same generator. It goes in H's check run.
- Tries [ran, through `research run`, each once]: at the first commit, `r20261112-221442-56a6`, exit 0: `security/Proofs/Properties/NetworkPeriod.lean` (63 theorems, no axiom beyond the three), `security/Proofs/Properties/NetworkScan.lean` (11), `CoreOnsite.lean`. At the second commit, `r20261112-223125-77a9`, exit 0: the same three files, `NetworkPeriod.lean` at 70 theorems. So `decide +kernel` does evaluate `stateEmpty … = false`, which I was least sure of. At the fourth commit (it holds the third): `r20261112-225637-3632`, launched once at 2:57 PM PST, running.
- Two things in it for your word. The public gains no `held` key in my text (staging publishes `"held": ["circuit", "free"]`; without the count `free` is not held, and a key of one name seemed a format change to no reader's use): say if you want `"held": ["circuit"]`. And `class_circuit` reads `window-circuit-w<K>-<class>.json`, as staging does; under the tie form the base's script names that record with a `-tie` suffix, which neither staging nor this reads, so the period step there reports the record missing, or refuses the phase by name. It never accepts a wrong one. I leave it as staging has it and say so in the body unless you say otherwise.
- Every positional `Params` literal in the two Security `NetworkPeriod` files gains `, none` (eight of them), which is the only edit to `main`'s lemmas there besides one more `List.mem_append_left _` in five membership chains. No statement of `main`'s changes.
- The lock: H adds two guarantees (`rowsStateEmpty`, `rowsCircuitHeld`) and changes what `rowsStateTied` unfolds to (its statement is `main`'s, `stateTied` under it now asks the reads too). No record is written yet: `check.py --update` goes in H's one check run, at its last head.

**Your two facts for `081ffe2a710`**: both go into H's body in those words: the two keys by name (`window.reads`, `window.empty`), that the record check's registered layout is changing under PRF-548 so whoever merges second reads the other, and that the honest side of `RowsStateEmpty` is shown by the difftest and not by a Lean lemma. If the witnesses rule asks an honest lemma of it I send you what the check prints first.

**Runs of the cuts, and one thing that needs your word.** A's and L's checks are running (they waited for a slot until after 2:33 PM PST). **B's and I1's never started**: `r20261112-214254-8906` and `r20261112-214257-d77f` were cancelled by the run's own time limit (class `STAGE_TIMEOUT`, about 61 minutes of wall time, no line of output) while they waited for a slot. I had given A and L `--timeout 2h` and B and I1 none, so they had the default hour, and the wait counts in it. My mistake in the launcher, not a failure of either cut. Each run is launched once, so I do not relaunch them myself: **say "relaunch B and I1" and I launch each once more with `--timeout 3h`, nothing else changed**. Both are Python cuts (two commits and one), no Lean.

Least sure of: line 2's "if those can be called without an instance" (I read `Record`'s docstring and declaration list, not the bodies of `ownRows` and `unwhole`); and line 4's reading of `Summary.free`, from its docstring alone.

