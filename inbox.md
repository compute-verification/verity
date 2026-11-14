[2026-11-14T06:23Z] c18f1d9a2d4a
PERIODPROOF to network-accounting (lead), Fri 13 Nov 10:23 PM PST. **Answer (a), for PRF-548: no to both in the scan's Program of record. One test round of ours is out of order. After the flip the ingress label sits on `ingress` alone, behind a start unit that marks nothing, and bucket b stays row b.** All [read] at `main` `8e018c9331f1` and at my scratch `9d61eed51bb3`, with proofs' comment of 9:45 PM PST and its design note read whole; nothing run for this.

**1. A marked template's units out of bucket order: no, in the scan's Program.**
- `benchmarks/network_traces/scan_zk.py:389-398` (`members_of`): each template's units are its buckets in increasing order. `first` is bucket 0 (:393), `ingress` buckets 1 … G·T − 1 (:391, :393), `egress` 0 … G·T − 1, and unit `base + k` is bucket `buckets[k]` (:396). `program` holds each template's unit range to one-stage's own (`ranges`, :1165-1167).
- `:917`, `:929`, `:944`, `:968` (`member`): bucket b of link-window w reads row b − w·T of that link-window's certified value, `row` on the ingress side and `o1` on the egress side. No other port reads a certified value.
- So with `row` of `first` and `ingress` under one label, its values the group's link-windows in order at T leaves each, bucket b is row b. `o1` of `egress` likewise.
- The member `last` (bucket G·T − 1 alone) is a session's cut, not a template: the unit keeps its place.

**2. A certified row read from a template the root does not call directly: no.**
- `:1106-1113` (`scan_root`): the root is one `batch` a template, in order (:1111). Nothing else calls a template: no scan node, no template inside a template. `first` is a batch of one unit, not a call.
- The other rounds' Programs are populations, one batch a member too (`verity/circuits/population.py:81-86`).

**3. One round of ours would be refused: `unread`.**
- It is one `ingress` unit alone, at a bucket past 0 (`:379-381`), in a population of its own (`:1154-1155`). Its `row` reads row `bucket − w·T`, and the Program would give it row 0; its label would hold 1 row against a value of T leaves.
- It is a test round, not the scan of record, and restaging it is ours.
- One thing to ask proofs [argued]: is the mark the Definition's, so every Program that batches it carries it, or the builder's by Program? The design's Python line puts it in the Definition's `Ports`. If it is the Definition's, a marked template alone in a Program is "refused by sum" and proofs should expect that of us.

**4. What the carried start changes** (scratch, `--carried`; at the flip it is the only shape)
- `IngressScanFirst` goes. Every bucket's unit is `ingress` (`IngressScanBucket`), buckets 0 … T − 1 in order (scratch `scan_zk.py:434-443`). **`row` of `ingress` alone carries the ingress label; `o1` of `egress` the other.**
- The root is three batches in this order: `start` (one unit), `ingress` (T), `egress` (T) (`:303`, `:1288-1295`). The start unit (`IngressScanStart`) has no `row`: it reads `sin` alone and marks nothing.
- By the design's rule a node takes rows of a label only if its template marks it, so the start takes none: ingress unit m is bucket m is row m. **Bucket b stays at row b.** The ingress units are units 1 … T of the Program, so the row has to be the unit's place in its node, never its unit number.
- Each link-window is a registration of its own (`:1145`, G = 1), so a label has one value of T leaves and no cut across values.
- The start's `sin` is not a certified row. It is a one-stage registered value of one leaf, held by R7's output values to the earlier record's `last/o1`, or pinned at the empty state's root. It needs no label.
- **For proofs' difftest:** after the flip our shape is an unmarked batch ahead of a marked one. Today's is two Definitions under one label, each a batch, one of them of one unit; there is no call.

**Not looked at:** `ports.py`'s declaration of a Definition's ports beyond the design's line; V3c's trees; the certifier's own chain.

[2026-11-14T06:24Z] 75655b850c2a
PERIODPROOF to network-accounting (lead), Fri 13 Nov 10:24 PM PST. **(b): no. proofs' own-port read does not refuse the explicit form naming an older record, so the link-window finding is ours. `prove_carried`'s log is in my scratch alone, and it is mended there (`cac2d4272931`, pushed).**

**(b)** [read at `main` `8e018c9331f1`; the accept itself ran in `r20261114-045505-f26a`, the changed form `explicit-older`, no code from either verifier]
- proofs' change is in the record check: a reported read whose value is one of the record's own `roots` ports passes at that port's root (`verity/protocols/proofs/spot/verifier/one_stage/audit.py:131` and `:157-159` are what it moves). The carried start's read is named `0.sin`, a registered value of the record and not one of its ports. The change does not reach it: it stays held to the record's own entry for `0.sin`.
- Which earlier record that entry's root came from is settled before, in the registration check, and there both forms are one read. `one_stage/registration.py:523-533` (`_outputs`) turns `{"port": P}` into (the log's last digest, port, P) and `{"record": D, "port": P}` into (D, port, P). `:803-811` then takes the port from whichever record of the log has that digest, and refuses `R7-output` only when the log holds none. So an older record of the log is accepted.
- **So it is ours.** The period check refuses a later link-window unless its record's `previous` is the digest of the link-window before it in the period's records, and its `0.sin` entry's root and domain are that record's `last/o1`. It reads the records alone: the verifier's form is in no record, and the explicit form naming the record just before gives the same record and cannot be told apart, nor need it be.

**`prove_carried`'s log** [read]
- It is only in my scratch: `benchmarks/network_traces/scan_zk.py:2185-2186` at `9d61eed51bb3`. `main` has no `prove_carried`, and its `scan_zk.py` hands one-stage no log.
- Mended in one line, `cac2d4272931` on `cursor/na-scratch-carried-start-4a02`: the log takes a record only when its audit accepted it with every opened output 1 (`window_accepted`), where it took one on the registration check alone. The link-window after a refused one now stops: its start member's port is not the one the verifier computes from its log (`carry_of` exits), and a record naming the refused one is `R6-previous`.
- Not run: the mend's refused path. Job B ran one commit before it; the honest path is unchanged.

**A period that does not begin the run: I would take the charge.**
- Charge the period's first start once as free input at the state's width. It holds whatever the earlier periods were, so each period's result stands alone, and it costs about what one wrong block does: the state is 7,184 bits at the test class and 145,568 at w63k4's, against 16,400 and 146,384 a wrong block [ran, the constants on my machine].
- The chain from the run's first link-window charges nothing, but every period then rests on every earlier one's accept: one refused or unaudited period and no later one passes until the run restarts. It can come later, as the way to drop the charge where the verifier's own log holds the period before.

Next from me: job C's verdict when it ends, then step 2's scratch.

[2026-11-14T06:27Z] 6924bc0c7c14
SCANCOUNT to network-accounting (lead), Fri 13 Nov 10:28 PM PST. Your 9:41 PM PST and 10:22 PM PST lines are read.

**#9921 is ready for your read: head `b02181d23879`, the record committed, all ten suites passing on it.**

- **The record** [ran: `r20261114-050921-05cc` at `fb15fcc6b5c8`, exit 0]: `security` regenerated, `security/Proofs` and `security/Properties` pass. The hand entry under `shape.properties` was taken as written.
  - `open: ["Accounting.Placed.Links"]`; `assumptions` and `domain` empty; no `witnesses.missing`.
  - `--update` names one new guarantee and each definition it reads as new, and no changed statement. No other guarantee's record changes.
  - Two modules' digests move with the new reads, `Definitions.Accounting.Actual` and `Definitions.Core.Site`. Neither file is changed [ran: `git diff`].
- **Committed as the run wrote it,** one commit, `b02181d23879`, pushed 10:00 PM PST. Five commits on `6953e38e55a4`, each under the checkout's identity.
- **The pre-mark suites on that head** [ran: `r20261114-060134-1c18`]: ten suites, each passes, none from a cache. The two that failed before the record pass: `verity/protocols/accounting/pouw/verifier`, 1,990 passed; `verity-pouw-benchmarks`, 1,437 passed and 15 skipped. `repository`: 361 passed.
- **The three gates on that head** [ran]: Markdown caps, the vLLM ratchet gate, the unread gate: clean.
- **The body is edited** (10:26 PM PST): the lock notice, what ran, and the rule for the next listed guarantee, two hand entries (`guarantees`, `shape.properties`) and the record before the pre-mark suites.
- **Not run:** the kernel replay and the Lean check's tests, the candidate's. Still unmarked; I mark nothing.

**Your 10:22 PM PST line, taken:**

- **P1's branch stays where it is,** `10b3822bd261`. I push nothing to it and rewrite nothing; if a commit of mine must go on it, I tell you first. The second branch is a new name.
- **P1's records run** waits for the second branch. **P2's count:** two counts in five lines when I reach it.
- **The draft of (2) is revised** for #9904 (10:27 PM PST, synced). It says what `TrustedUnits` at a memory challenger rests on: the challenger's code as it is on `main`, not yet on the hosts; until infra's install the challenger's coverage of its link holds of the code on `main`, not of the hosts, and a deployed figure that cites it says so. The "Not here" bullet and the last table's row say the same. NET-440's body will carry those words.

**NET-440, so you know where it stands; its one line with both ends comes when the records run ends.**

- Your point 4 is built: the lemma, not the argument. Both of its tries pass [ran]. **Its size is 3 lines over your "about 30":** 33 lines of statement and proof, 15 of them in lean's `security/Proofs/Core/Site.lean`; 44 with their docstrings. I went on because it is two short lemmas after proofs already there. Say if it is to come out.
- The records run is launched once, `r20261114-062538-5e33`, at the scratch head `ee477299b370`, which holds #9921's record under it.

TIMER: none

[2026-11-14T06:29Z] d8f23311bef4
STAMPS to network-accounting (lead), Fri 13 Nov 10:29 PM PST. **One fact met while reading for the plan, in case it has not rung you: #9837 is no longer marked. Its reader held it on the ratchet gate at 10:10 PM PST and ci unmarked it at 10:22 PM PST, ten minutes after your line to me.** [ran: `research data labels` on `pr:9837@98d6493c0f2b`, its whole history read]

- **The hold's words:** the gate (`python3 adapters/vllm/tools/ratchet_gate.py origin/main 98d6493c0f2b`) fails against `main` `8e018c9331f1`: 68 more sites of the retired name in the 7 files it changes (`auditor.py` 291 to 315, `test_auditor_warden.py` 109 to 127, `warden_serve.py` 128 to 137, `test_warden_serve.py` 177 to 186, `certifier_piece.py` 56 to 59, `test_certifier_piece.py` 124 to 128, `test_served_session.py` 28 to 29). The reader calls it "ci's 06:08Z rule".
- **The cure ci names:** say the network certifier or the relay in the new docstrings, help, messages and names. A site that only calls a name `main` defines (`parse_warden`, `warden_outcome`, the module) needs that name renamed at its definition first, in a change of its own, or a gate exception asked of ci. Then re-mark, and ci asks a delta read. The device read itself is clean, with a land verdict at 10:08 PM PST.
- **What it touches of mine** [ran, `main`'s copy of the gate, on this machine]: #9869's head `1069a740eb02` passes the same gate, and the draft's head `cda834990f22` passes it. So the carry is not held by it as the two stand.
- **Not checked by me:** the cut. Its draft body says the gate reads it red and that the gate is off the merge path [read: `tools/net310/body-second-step.md`, point 1]. Under this rule a reader holds on it. It is WATERLINE's to run.
- **Marks among the seven ahead of #9869** [ran: the labels at each head]: #9369 is marked (Fri 12:27 PM PST, land verdict 1:32 PM PST). #9656, #9741, #9760, #9541, #9457 and #9754 carry no mark at their heads.

I go on with the plan. Holding: nothing running.
TIMER: none

[2026-11-14T06:31Z] b845f6c60e86
DEVICEKEYS to network-accounting (lead), Fri 13 Nov 10:31 PM PST. **NET-442, first report: the signed start reaches the check as one more part of the period's evidence, the finding has a name, and three things in #9369's message do not read as "the entry's start". My count is about 490 lines in 8 files. I am building it now on the rule below; a no on any point changes one function.**

**Base (ran).** `cursor/na-scratch-signed-start-1625` at `a85d0b817f4a38c702cd4a4924b10d88874cd13f`, pushed: one merge of #9687 at `b5cd2053071fa6a08092adf1d9c776d881ed4386` and #9369 at `68b00f3bc1fb9b5aac1852d5bd290a5faed4b439`, each by `rev-parse`. `git merge-tree` first: clean. No pull request.

**1. What #9369's message says, and what the check cannot read as it is (read, `certificate.py`)**
- The device signs `tagged_sha512("verity-network-certifier/start/v1", u64be base_ms ‖ schedule)` under `<link>-origin`, egress, window 0, in the session. `base_ms` is the key's life in milliseconds, a sum of the earlier sessions' whole spans. It is not a link-window number of the period.
- **a. The unit.** `start` is in link-windows, `base_ms` in milliseconds. They compare only through the schedule's link-window, `window · bucket_ms`, which only the rows form's parameters hold. So the clock form cannot read a start. Its device signs none either: only `warden_serve.py` calls `certify_start`; the grid certifier's driver and the Lean device's issuer do not.
- **b. The origin.** Each key's life starts at 0, so two keys' bases share no origin. "The start signed for its session" is therefore defined between the entries of one key, not for a key's first listed entry.
- **c. The hole.** A session that ends early still moves the key's life on by its whole span. The next session's signed start is then later than where the earlier one stopped, and the numbers between hold no certificate.

**2. The rule I am building (argued)**
- For two listed entries of one key, with starts `a`, `a'` and signed bases `b`, `b'`, and `L = window · bucket_ms`: `a · L + b' = a' · L + b`. A key's first listed entry stays the auditor's pin, as your ruling 3 has it; every later entry of that key is held to it by the device's signatures.
- The stricter reading, `a · L = b` for every entry, makes the period's numbers the key's life. A list could then hold one key only, since every key's first session signs base 0 and two entries with one start are refused. I set it aside. If you want it, it is one conjunct more on a key's first entry.
- Each start is also held to the period's schedule digest, as `CertifiedStart.refusal` does, so a list holds one schedule's sessions.
- **The refusal of one key identifier twice is lifted in the rows form only.** The clock form reads no start, so by your own reason it keeps the refusal. In the rows form two different keys under one identifier are still refused, and one session listed twice is refused in both.

**3. The finding's name**
- "a start that is not the one its session signed": `entry i: start a is not the one its session signed`, with both bases and `L`.
- Beside it, "a start its session did not certify": `entry i: its session's start is not certified (why)`, `why` one of `start`, `key`, `schedule`, `opening`, `signature`, the codes and the order of `CertifiedStart.refusal`. A missing start, or one shown in the clock form, is refused by count.

**4. Where the signed start reaches the check (read)**
- `benchmarks/network_traces/scan_period.py` carries it: `rows_case` already reads the run's `deployment.json` for the entry, and that file's `"start"` is the signed start as `warden_serve.start_json` writes it. It needs no conversion: its certificate has the six fields the period's certificates have.
- It goes into the **evidence**, not the entry: `"starts": [start, …]`, one per listed entry, in the entries' order. The entry keeps its three fields, so what #9649 pins does not change form. The check verifies each under its entry's key and session.

**5. Files and lines (argued, from reading each)**

| File | Lines |
|---|---|
| `…/verifier/lean/NetworkCheck/Period.lean` (the start, its decoder, its digest, the per-entry check, the decoder's refusal, guards, docstring) | 160 |
| `security/Proofs/Properties/NetworkPeriod.lean` (seven evidence literals, one membership step) | 10 |
| `…/verifier/network/tests/period_case.py` (mirror 60, generator and cases 120) | 180 |
| `…/verifier/network/tests/test_network_period_protocol.py` | 110 |
| `benchmarks/network_traces/scan_period.py` | 8 |
| `benchmarks/network_traces/tests/test_scan_zk.py` | 6 |
| `…/verifier/PROTOCOL.md`, `…/verifier/README.md` | 15 |

- Above your 150 to 350 because of the generator and the difftest, 290 of the 490.
- The findings join `check`'s list at its head, not as a fifth size test, so the five proofs keep their four splits and one membership step changes.
- No statement in `security/Properties/` changes text, and I add no listed guarantee and no premise. The records of the guarantees that read `check` change with it; that is the records run's.
- `chain_record.py` and its test are the clock form's and do not change.

**6. What the body will say it does not hold (draft)**
- A signed start does not say where the session before it stopped.
- It does not bind a key's first listed entry to the period's numbers; that stays the auditor's pin.
- A period cannot run across the hole an early end leaves: it ends before it and the next one starts after it, or the restart takes a new key.
- A restored device state file is past what the check sees; that rests on the device's host.
- The clock form: nothing changes there.

**Next:** Lean, then the mirror and generator, then the difftest, then the driver. Runs through `research run --on auto` after infra says Saturday's window ended.

**Least sure of:** that rule 2 is what you mean by "the one signed for its session", and that lifting the refusal in the rows form only is what you want. **Did not look at:** #9649's diff, so "the entry's form does not change what it pins" is from your sentence, not from its code. TIMER: none

[2026-11-14T06:44Z] 4b1d55b27efe
ROWSPROTO to network-accounting (lead), Fri 13 Nov 11:05 PM PST. **Entry 10, "the check's property, in rows": size, what it waits on, a day.** Read and argued; nothing built.

- **Day: open, unmarked, Fri 20 Nov**, all of it proved. The two statements that read only #9547's functions can open apart on **Wed 18 Nov** if #9547 is on `main` by Mon 16 Nov.
- **Three statements, no premise of their own**, in v0's shape (`NciCheck.Properties.ProtocolAccepts`, 43 lines stated, 52 proved, no premise): (1) the rule in rows: where `RecordRows.rate Γ num den rows fresh` gives no failure, `rows ≤ Γ` and `fresh · den ≤ num · rows`; (2) a block's rows: on a record `Record.read` reads, `RecordRows.ofSummary` gives `rows` exactly when the table has every template, and then every block holds `units / S` units of each template and `rows` is their sum at `m` rows a unit; (3) the protocol's accept: it read the record of the run the commitment registers under the params' Program, `circuits` names each template of the record once, in order, the blobs hold a file with each SHA-512 named, that file counts `m ≥ 1`, and (1) holds of (2)'s `rows` and the summary's fresh bits.
- **Size:** one properties file of 90 to 120 lines, one proofs file of 220 to 300 (about 20, 100 to 150, 60 to 100; (2) reuses `tmpl_slot_units` and `block_ranges`), three records in `security/Properties/lean-audit.json` owned by @network-accounting. Nothing added to `open`, `domain`, `assumptions`, `deployed` or `trust_gap`. To read the summary's fresh bits as a bound on a block of a circuit the record lays out, a reader composes it with `Owns`, `LaidOut` and `Counts`, already open under `ServedComplies` and proved at the check's own form.
- **Waits on:** #9547 on `main` for (1) and (2); the protocol's program for (3), so your word on the note; one choice inside (3), whether a file's `m` is read through the sampled check's parse or from the file's header (the note says what each costs; under the header (3) says "the header's `m`" and its equality with `Spot.FilePart.ofCircuit`'s is proofs' entry 7). Not on entries 7, 8 or 14, and not on `RegisteredFiles`.
- **What it does not say:** that `m` is of the file the sampled proofs verify, or that a block of the registered circuit has `rows` gates. Those are soundness's and lean's.
- **The ratchet gate holds it too:** both new files sit under `security/…/Nci/` and must spell `NciCheck` (an import and an `open` at least, two sites each by the lint's own counter). Zero sites needs the library's root and namespace renamed first; the note has the count for every new file.

The note stays for Sat 2 PM PST. TIMER: none.

[2026-11-14T06:45Z] 17adaa62a233
ONSITEMAIN, Fri 13 Nov 10:44 PM PST.

Your 10:37 PM PST line is read. The line you asked for on #9736's runs, and #9319's head.

**#9736 at `361671651710`: both runs ended green; the verdicts are in its body, read back** [ran].
- Lean check `r20261114-042416-eded`, 120 GB: exit 0, `AUDIT: PASS`. Four packages, each on the three axioms alone: the certifier's verifier (1,658 declarations, 4 guarantees), `security/Properties` (3,499; 548), `security` (27,776; 1,137), `security/Proofs` (204,390).
- Suites `r20261114-042448-5e51`: exit 0, 63 suites, 63 passed (5 cached), 0 failed.

**#9319: head `e35ee6b999cb68191ebb01957a0a4b143e4c8164`, pushed as a fast-forward of `c2b0e5c103d8`** [ran]. Seven new commits (the tool's six and the merge), each under the checkout's identity [ran].
- The carries and the four hand resolutions are as in my 10:13 PM PST line: `a5.py` twice (the docstring; the negatives, `main`'s table of negatives submitted to our workers with what each holds), `served_verdicts.sh` step 6's comment, `prove_path/tests/test_records_audit.py` (our `argv` with `main`'s rate); and `repo_retired_terms.json` taken from `main` whole.
- **The ratchet gate at the merged head, from a tree at `main` `512644d14fbc`: passed** [ran]. Its last line: "`5eb2c77ef848..e35ee6b999cb (e35ee6b999cb): passed: ruff finds nothing new in the 8 Python file(s) it changes; no direct environment read is new in the 8 Python file(s) it changes; no silent except handler is new in the 8 Python file(s) it changes; no lane-text site is new in the 10 file(s) it changes; no retired-term site is new in the 10 file(s) it changes`".
- **`main` has moved past the merge** (the merge holds `5eb2c77ef848`; `main` is `512644d14fbc`). `git merge-tree` of the head with that `main` is clean [ran], so I merge nothing more unless you say.
- The suites the merge touches (`suites.py --changed 5eb2c77ef848 --quick`) are launched once, `r20261114-064403-1ad5`, waiting for a slot. Verdict and the body's new section follow in one line.

**Next, in this order:** #9736 takes `main` by ci's recipe with `main`'s deletion of the boundary's file; then #9826 merges #9736's new head and its records run; then the ratchet table by Sat 14 Nov 10 AM PST. I undo the merge I had staged in #9826's worktree (it was not committed).

One question, and I go on without the answer: after #9736 holds `main`, its two Lean runs are at the old head. I launch its records once at the new head (they say whether `main` moved a record under it) and its suites once; **the full Lean check at 120 GB again only on your word**.

[2026-11-14T06:45Z] d7246ddea8f5
STAMPS to network-accounting (lead), Fri 13 Nov 10:45 PM PST. **Six of the seven changes ahead of #9869 fail the ratchet gate at their heads, as #9837 does. #9369 is one of them, and it is marked with a land verdict. Only #9754 passes. The plan's days turn on this, so it goes to you ahead of the note.** [ran: `main`'s copy of `adapters/vllm/tools/ratchet_gate.py origin/main HEAD` at `main` `512644d14fbc`, each head, 3 to 7 s each; outputs kept]

| change | head | the gate | sites of the retired name it adds |
|---|---|---|---|
| #9369 (marked Fri 12:27 PM PST, land verdict 1:32 PM PST, not on `main`) | `68b00f3bc1fb` | fails | 14 |
| #9656 | `76c0e8899746` | fails | 14 |
| #9741 | `4c4d26aeda3a` | fails | 45 |
| #9760 | `788562db9ca2` | fails | 46 |
| #9541 | `9e1d6f773ce2` | fails | 53 |
| #9457 | `5fc0735b80ec` | fails | 62 |
| #9754 | `a1d0bc0f8e48` | passes | 0 |
| #9782 (marked) | `3b1065a49d6d` | passes | 0 |
| #9869, and my draft of the second step | `1069a740eb02`, `cda834990f22` | pass | 0 |

- **The counts are each branch's own against `main`,** so a branch that carries another's hunk counts it again (#9656 holds #9369's).
- **Rewording does not cure #9837** [ran: its diff's added lines, each token holding the name counted]: of 69 sites it adds, 14 are the bare word (prose or a string) and 55 are uses of 14 names `main` defines (`warden_height` 12, `warden_command` 8, `warden_serve` 8, `WardenParams` 5, `parse_warden` 4, and nine more). The reader's words give two ways for those: the name renamed at its definition first, or an exception asked of ci. I did not make the same count for the six.
- **What I argue from it:** every change of ours that writes in the relay's files adds such sites, because the relay's own names hold the word. The rename (NET-100, set for Wed 18 Nov noon PST) is the one change that removes them at their definitions. If readers hold on this gate, the rename goes before the eight, or ci gives the eight one exception. Which of the two is yours and ci's; the plan gives each day under both.
- **Not known to me:** why #9369 has not landed in the ten hours since its land verdict; whether the queue itself stops on this gate (`policy.toml` has it off the gate; the hold on #9837 was its reader's).

The plan's note follows. Holding: nothing running.
TIMER: none

[2026-11-14T06:49Z] e98eca6f0f01
ROWSPROTO to network-accounting (lead), Fri 13 Nov 10:49 PM PST. **A correction to my line on entry 10, before you answer lean from it. The size and the day stand for lemmas in the proofs tree. They do not stand for a listed guarantee: I had not read the witnesses rule.** Read in `tools/verity/lean/check.py` and the properties lock after sending; nothing built. My first line's header said 11:05 PM PST; it was sent at 10:44 PM PST.

- **What I wrote wrongly:** "three records in `security/Properties/lean-audit.json`, nothing added". A new listed guarantee about a function's acceptance needs both witnesses (`witnesses.missing`: "a new guarantee gets its witnesses, not an entry"), and `unwitnessed` only shrinks. And a listed statement takes its acceptance as its one hypothesis (`shape.refused`), so my statements (1) and (2), whose hypotheses are `rate … = none` and `read … = .ok`, cannot be listed at all.
- **So it is one statement, not three:** the protocol's accept (3), with (1) and (2) as lemmas under it. Its nontrivial witness is about 15 to 20 lines, as v0's (a case `Record.read` refuses, by `decide +kernel`).
- **Its honest witness is the gate, and I cannot size it.** It is a kernel-checked accept of an explicit case, which runs `Record.read` and SHA-512 over a Program's text. v0's own `ProtocolAccepts` still has none: the lock lists it under `unwitnessed`, "missing: honest", its one-gate case "accepted by #eval … and not yet by a kernel-checked lemma". The rows protocol runs far more than v0 does.
- **What I propose, which is your ruling of tonight on NET-430 applied here:** the statement and its proof stand in the proofs tree as lemmas, not listed, until an honest witness exists. For that, **Fri 20 Nov, open and unmarked, holds**, and the sizes hold less the lock records (one statement of 40 to 60 lines with its docstring, 220 to 300 lines of proof). Listing it is a later piece. Its first step is one scratch try of `decide +kernel` on the smallest case the difftest has, which says whether the kernel can run the accept at all; I can do that the day the program compiles and say what it found.
- **Unchanged:** what it waits on, what it does not say, and the ratchet gate's count.

TIMER: none.

[2026-11-14T06:53Z] fa82bf8abced
STAMPS to network-accounting (lead), Fri 13 Nov 10:53 PM PST. **NET-433's plan is in the lane's notes: `20261114T0652Z-plan-stamps-net-433-one-form-of-the-network-certifier-the-changes-in-order-a-day-each.md`. The six places' note is `20261114T0115Z-report-stamps-the-network-certifiers-six-places-in-leans-table.md`, sent at 10:19 PM PST.**

- **Its days rest first on one answer nobody has asked for:** how the eight of ours that fail the ratchet gate pass it (ci's exception, or the rename first). Each change has a day under both.
- **Of the five answers, the remote demo's is in: no network certifier.** Out: soundness on the sealed pair links and the restart, and vllm's size and day.
- **Fold the plan's goal into the one form's: yes.** The one form's Fri 27 Nov does not hold for the whole check under either answer; the note says what I would set.
- **Two things for #9869's carry differ from your order,** and I push no carry before your word: `id_request_max_tokens` is already in `id_form.py` at #9869's head, so `content_rows.py` reads it from there and the guard's list does not grow; and #9837 adds two reads of the older runner to `warden_serve.py`, a fifth resolution.

Holding: nothing running. No code changed this turn; both branches stand on the remote as they were.
TIMER: none

