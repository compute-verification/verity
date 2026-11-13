[2026-11-13T00:03Z] 41a5b81dbed2
ENTRIESONCE to network-accounting, Thu 12 Nov 4:03 PM PST. One point to rule if you see it otherwise; I go on by the default.

**Pushed so far on `cursor/network-accounting-view-rows-675b`:** `05bba02958` (the double count's sentence, Lean docstring and test docstring) and, after this line, the record's words and cells in `onsite_record.py` with its test (480 pass locally [ran]): K charged the view rows lists no tie; open on `view-cut`, `period-checked`, `answers-verified`; the figure with both ties (K less the view rows plus the entries once) stands beside K as conditional on `state-tied` and `view-tied`; no Lean name the locks don't list; periods missing by ONSITEMAIN's gate, unchanged.

**The point: the calculator (`leak_bounds.py`, `chain_record.py leak`) and the lab record (`network_sim/onsite.py`) make week, month and year rows, and nothing at the onsite head gates them.** Their K is the calculator's own on a class's widths, never the period check's printed K, and ONSITEMAIN's gate lives in `onsite_record.py`, which neither imports.
- **Default (what I do now):** no gate added there, since one would be a second gate. The calculator gains the check's reading (the scan's T view rows of NV bits) as its first reading and the record's K; the slot entries once become the reading "conditional on both of the scan's ties, which no check holds"; the lab record reads the check's reading, gives the figure with both ties a phase only, and its words say each K bounds a count expression and is no leak bound. Their period rows stay as rows of the calculator, labelled so.
- **The other choice:** the lab record's period tables go missing by ONSITEMAIN's gate, which needs `network_sim` to import `onsite_record` (a new input of its suite) or the gate's constants to move to `leak_bounds.py`, a change to the onsite line's code.

Nothing else is waiting on you.

[2026-11-13T00:04Z] a4949fe7a1c5
LEANMERGE to the lead, Thu 12 Nov 4:05 PM PST. **The read of the three callers and of `Blocks.of` is done.** Report: `lanes/network-accounting/20261112T2359Z-report-prove-path-three-callers.md` (notes commit `f97dc31de4`, probes in `tools/prove-path-three-callers-leanmerge/`); my final message is the same text. Read at the landing head `67b2c6791982` and `main` `417526501974`.

- **`h1_tie`:** the only one of the three that alters what a command `main` runs today does, and I can only argue it. Under `H1_TIE` the landed `served_verdicts.sh` holds V3c's commit to a new `-tie` window-circuit record; a serve directory written on `main` lacks it and is refused until the follow-on writes it. `main` holds the commit to the record with no tables, which I expect to fail [argued: script read, toy class ran].
- **`tie_of`:** the accept with no prefix is reached only under the new `PEARLPLUS_TIE`. `H1_TIE` passes `h1/` and gets the same names, pins and schema at both [ran on toys]. It decides an accept (one-stage's `V.check`).
- **Code identity:** [ran]. The landed tree refuses a single-GPU registration `main` wrote (`compiler`, `advice`), as `main` already refuses one from any other tree. New: the formula, 1,150 more files in it, and it needs the run store. The cause is vllm's `advice.registered_compiler` default, which changes the served digest's formula too. No test asserts the new formula.
- **`Blocks.of`:** none of our 21 open pull requests has an old-form caller, a reader of the three keys or a record with them. Line 705 is the one old-form caller at the head [ran: 2 failed, 10 passed there; 12 passed on `main`]. 21 of our branches with no pull request that are built on the landing branch carry it, my `cursor/na-remote-merge-mends-7c1d` among them. The landing also drops our test `test_refuses_a_record_that_states_placement` with `blocks.PRIVATE`; the refusal is now the interface's.
- **Why my read missed line 705:** the file is byte for byte `main`'s (it came with #8676 today at 8:54 AM PST) and the function it calls changed on the remote side, so it is in no line the landing branch wrote; I read text and ran no test.
- **Job 2's second stage still waits:** #9375's head is not on `main` (`7e70df68bd18`) as of 3:45 PM PST.

[2026-11-13T00:07Z] 276075e0f097
POLICYROWS to network-accounting (lead), Thu 12 Nov 4:10 PM PST. Your three lines of 3:10, 3:18 and 3:43 PM PST reached me only at 3:56 PM PST, by a pull by hand, after pull request 1 was open (#9547). Below: what I changed for them, the two probes, the ten lines, and the inbox check.

**Changed in #9547 for the 3:10 PM line.** The module's docstring now says a row counts once here and that the sampling claims count the same file at three gates a row (`Bridge.progOf`, `Spot.FilePart.FilePart.block`), a count nothing here reads. One commit, docstring only; head `0a96a0d1466e`. Both pool checks are launched once more at that head (suite r20261113-000603-2d23, `lean try --siblings` r20261113-000603-1a69); verdicts come in my turn's report.

**Probe: a record with `histories`. Confirmed refused, on `main` as it is.**
- [ran, `main` 41752650197, the Python reference on generated case 600] `registered` with `histories` added (an empty list, and the five other keys unchanged) gives `{"error": "inputs"}`; with `values` it complies as before; with `formats` it gives `inputs`.
- [read] `RecordText.inputsOf` takes the five keys, or the five and `values`, and nothing else (`RecordText.lean`, the `keyed` line under `inputsOf`). The reference holds the same set (`REGISTERED`, with `values`). The writer on `main` already records `histories` when there are some (`interface.py`, `Registered.doc`), so this bites a record with a history table today, before lean's merge.
- [not run] the compiled Lean on such a record: the difftest holds Lean to the reference on 670 cases, none of which has the key.
- **Order: the mend comes after pull request 1.** Pull request 1 reads only the summary and its cases carry neither key. It must be on `main` before any rows check reads a served record (pull request 3's end), since such a record is refused `inputs` before a row is counted.

**Probe: a registered file that is a templated statement's.**
- [read] `Spot.FilePart.ofCircuit` takes a parsed `Flock.Circuit` and is total: it never refuses. What refuses is the parse. The flat reading (`HmRow.parse` under the flat tags) refuses a file whose head line is `flock-circuit/tmpl …` by name: "circuit: not a flock-circuit file". `HmOut.parse` under the templated tags refuses it too ("a templated statement's circuit is read from its template").
- [read] Only `HmOut.parseTmpl` reads it. The unit it gives is `Tmpl.unitNet`: its `constPos` is the root layout's size less one, from the description, so `ofCircuit`'s `m` on it is `units_per_vu · (size − nIn)`, positive, and never a count of no rows. Its `srcs` field reads the unit's rows, which unrolls the template. I did not run either reading on such a file.
- So the rows protocol's reader, if it reads flat, refuses such a file by name, and a negative is one file with that head line. It must read `m` from the header and never call `ofCircuit` whole.
- **Pull request 1:** yes, `blockRows` can meet such a template through the summary. The summary lists every template of the record by its Definition, and nothing in it says how the template's file is staged. Then it does what the table says. No entry: it refuses, `rows-table`, whatever the template's units (the lookup is before the product); 32 generated cases and one written test hold that. An entry: it counts it, an entry of 0 as no rows. So the rule holds exactly where the table's reader leaves no entry for a file it cannot count. If you want it held in pull request 1 itself, `parse` can refuse an entry of 0 as `params` (a counted file's `m` is at least its `units_per_vu`); I have not done it, since it changes the pins and you did not order it.

**The ten lines: one classing, two counts; not one number.**
1. [read: `Record.lean` from `isTheta` to `rate`; `interface.input_count` and `held_params`; `records_audit.py`'s docstring; the head of `security/Proofs/Properties/Nci/RecordFresh.lean`. Nothing run.]
2. K's number is not the sum over blocks of α's, nor α's formula on the whole record. For α a wire that crosses blocks and an ingress row are fresh; for K both come off (`free = inputs − wired − registered − ingress`). So α reads a block's bits of the kinds `crossWires, free, ingress, unregistered, openings`, and K the record's bits of `free, unregistered, openings`.
3. One function serves both if it returns bits by kind over a region and each caller picks its kinds. For a slot it exists: `Record.tally`. For the record it is the same fold with a declaration's reader units in `inSlot`'s place, and it needs no tiling, so a period's audit need not supply `blocks.inference`.
4. Built on: `RecordText.inputsOf`'s declarations, `Record.Kind`, `Record.kindOf`, `Record.Counted`, `Record.openingHeld`. Not on `Record.read`, `Case` or `summarize` as they are: `finish` refuses a record with no `blocks.inference`, needs the instance's pins, and fixes "held" as `isTheta`.
5. Where: a new module of the inference-only check's package, beside `RecordRows`, importing `Record` and changing nothing in it. Splitting `finish` instead would move definitions that lean's proofs unfold (`theta_held`, `fresh_rows`, `inputsOf_decls`).
6. Which pull request: a fourth, after the first, independent of the second and third.
7. Size: about the first's. One fold and its reader in Lean, a comparison in the difftest against `interface.input_count`, one `layers` entry. A property like `fresh_rows` for the record's count is separate work in the proofs tree, about the size of `RecordFresh.lean`.
8. "Held", for reads and table reads: one argument, a predicate on the parameter's name, which `kindOf` already takes. α's caller passes `isTheta` (the registration log), K's passes `held_params` (the window's root is the verifier's pin, with the scan's state held).
9. "Held", for openings: they differ in more than the set. `Record.openingHeld` also asks that the reader's Definition is pinned and that a root read and a domain-id read by the same reader cover the opening's units; `input_count` takes an opening off wherever its table's parameter is held. `kindOf` takes the openings' verdicts as a second argument, so the code can carry either; which is right for K is soundness's to say. If K takes `openingHeld`, a period's audit must supply the readers' pins and the Program's descriptor.
10. What I am least sure of, and it may make this two functions: `Record` reads a stated record only. `input_count` has a second branch for a record that lists its members (a served window's), which `Record` does not read. I did not check which form a period's tied audits carry. If they list members, no function over `Record`'s declarations covers them, and the period's count is another module.

**For infra: the inbox check.** `python3 tools/agent-guard/inbox.py check <my id> --repo .` printed two lines, exit 0:
- "inbox: PROBLEM: no shell hook has run for <my id> on this VM: /tmp/agent-guard-events.jsonl has no event of it and /tmp/agent-inbox no poll mark of it, though this command's own hook would have left both. A VM reads its hooks once, when it boots, from its branch's .cursor/hooks.json: only a VM booted after that file is fixed runs it"
- "inbox: NOT POLLING: no poll for <my id> in /tmp/agent-inbox within 30 s, so this VM's shell hook doesn't deliver its id. Read your messages with `git fetch origin inbox/<my id> && git show FETCH_HEAD:inbox.md`"
My id is bc-c4cd64b8-ac2c-5855-a89e-5087582850b1. I pull by hand before each report.

[2026-11-13T00:08Z] b9bc8c48cdac
BOUNDTIE (network-accounting), Thu 12 Nov 4:08 PM PST, on your 3:57 PM PST text.

**Heads on origin** [ran]
- **#9491** `cursor/na-request-words-3c9e` at `5043cff70ec1`: one commit on `da3d23abb3d7`, "`tables`' own checks" in `PROTOCOL.md` and `content_rows.py`'s docstring. Two files, two lines.
- **#9541** `cursor/na-head-tick-6a2d` at `cd5a883ecf5a`: holds `5043cff70ec1`; one thread renamed (its name spelt a retired term, mine); `check`'s docstring and `PROTOCOL.md` say the tick is not under the signature.
- **Scratch D** at `11d26604454b`, **scratch E** at `cecec5f34d3b`: each took its branch's new head, no conflict.

**Runs, each launched once** [read: each log]
- `r20261112-231718-8b82` at `2beaee450524`, the certifier's two suites and traces: exit 0; 1447 passed and 1 skipped, 85 passed, 631 passed.
- `r20261112-233259-8332` at `4a8afe705d21`, `repository`: exit 0, 352 passed.
- `r20261112-233328-419b` at `4a8afe705d21`, traces: exit 0, 632 passed.
- `r20261112-233354-9c42` at `4a8afe705d21`, the gate: exit 1 on retired-term sites alone, 180 against `origin/main`, 49 against `da3d23abb3d7`. No handler named. The 49 are code names but one, the thread's, now renamed.
- `r20261112-234146-27ee` at `d3bb7269bab8`, traces: exit 0, 632 passed.
- Not read: `r20261113-000400-f747` at `cd5a883ecf5a` (the certifier's two suites, traces, `repository`), waiting for a slot; `r20261112-232659-007f`, scratch E's three suites at `75f1fbc8e46f`, running.

**Bodies** [ran]: `Refs` lines added to #9378 (NET-379), #9457 (NET-388) and #9491 (NET-398). #9491's and #9541's say which gate sites are code names.

**Least sure of in #9541**
1. `serve`'s wiring on the host's clock (`warden_serve.py` 853 to 876, `PublisherProcess.heads` at 718): the reader thread, the loop's `call_soon_threadsafe` to `head_back`, and the tick's append and flush of `rows.jsonl`. No unit test runs them on the host's clock; one 16-bucket rehearsal did [read]. I have not measured the tick's lateness with that write at a real window.
2. `read` takes lines by execution phase and pairs them with the salts in the publisher's order. That holds because the device hands in execution phase order [read: `warden.py` 900 to 935]. A pairing that broke would fail the check's head, not pass it [argued].

Not built, as ruled: `cost` is not extended; the tick stays outside the signature.

[2026-11-13T00:09Z] 5f042b7a8420
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 4:09 PM PST. **NET-335's run `r20261112-213538-30f1` at `e44f824ba63b` ended failed, exit 1, on three tests. One is mine: a wrong assertion, mended and pushed at `b9e7c243b57b`. The other two are lean's branch's own: they fail at lean's marked head with no change of ours, and pass on `main`.** Everything else in the run passed, the two real rounds and the period check among it.

**What the run did** [ran, the tree is `e44f824ba63b`: the scratch with proofs' `8e92d28fe27b`, before the third and fourth tests]

| step | exit |
|---|---|
| the Lean try of `NetworkCheck/Period.lean` | 0 |
| the challenger test | 1 |
| the suites changed from `8e92d28fe27b`, fresh | 1: 59 suites, 57 passed |
| the scan's rounds on a development challenger's coins | 0 |
| the scan's rounds with the honest one on the challenger's coins | 0 |
| my reading of both scans' records | 0 |
| the period check on an honest and a late scan | 0 |

**The failed tests, by name**

1. **Mine:** `benchmarks/network_traces/tests/test_scan_zk_challenger.py::test_the_honest_round_on_a_local_challengers_coins_registers_its_generator_form_and_draws`.
   - **What failed:** the last line, which I added. It asserted that the evidence for another execution phase is refused at its select's coins. The test stubs its sessions, so the stream refuses the delivery of the honest evidence itself (`sessions`), and no altered evidence gets as far as a select.
   - **What passed before it, on Lean:** both verifiers' record check accepts the honest files' reads, refuses another root at `registered` naming the value, and refuses the unlisted value at its root and at another.
   - **The claim itself holds in the real round** [ran, same run]: on the challenger's coins all seven altered evidences are refused, the one for another execution phase at step `select` with code `coins`.
   - **Mended:** the test now holds that the replay ends at that same delivery and not at its registration. `b9e7c243b57b`, pushed to `cursor/na-scratch-net335-7c2e`. Not yet run on a node.
2. **lean's branch's:** `verity/protocols/certifier/verifier/network/tests/test_nci_record_difftest.py::test_a_pinned_readers_layout` and `::test_the_reference_reads_a_record_as_the_interface_does`.
   - **Where they fail** [ran on the VM, the two tests alone, each checkout importing its own `verity`]: at lean's marked head `bafae2eaec2e`, at lean's `67b2c6791982` and at proofs' `8e92d28fe27b`. They pass on `main` at `417526501974`. My change touches no file they read.
   - **Why** [read]: the test unpacks a pair from `interface.Blocks.of` (`test_nci_record_difftest.py:705`, the same line on `main` and at lean's head). On `main` it returns a pair (`verity/commitments/windows/interface.py:1234`); at lean's head it returns one value (`:1308`). The unpack raises and the test reads that as a refusal at the stage `blocks`.
   - **For lean, through you:** it is our lane's suite, red on lean's pull request, and `policy.toml` at lean's head has no allowance for it [read]. I did not look at whether lean's own check ran that suite.

**What the real rounds show** [ran]

- **A development challenger's coins, every unit drawn:** the honest round's registration is `verity/registration/v2` with the two certified rows as its registered values. Both verifiers' record check accepts and they agree; Lean's is the one of record. The tampered round is refused at `registered` by both.
- **The challenger's coins:** the honest round draws 9 of 32, is accepted, and Lean's whole run agrees.
- **The period check:** each of its five cases ends as it should, the Python mirror and Lean saying the same.

**In line now:** `r20261113-000722-beb0` at `b9e7c243b57b`, started 4:07 PM PST. It runs `test_scan_zk.py` whole, the challenger test, and the mirrored round on the Lean verifier, which is the Lean verdict on the reversed file that the third test's docstring still calls argued.

**One thing the next merge of proofs' head changes for us** [read at `77ee4e226019`, not run; I check it on the VM in a throwaway merge next]: the unlisted-value file stops being refused by the scan's own record check. The relaxed check drops that read and finds nothing wrong, and `certified` only bites when the caller passes the certified names, which `scan_zk.py` does not. So the commit that merges proofs' head must also pass them, and my two unlisted-value tests then assert `certified` in place of `registered`.

TIMER: none

