[2026-11-13T02:08Z] fe403d61d7ef
OPTIN (TIEVERIFY) to the network-accounting coordinator, Thu 12 Nov 6:07 PM PST. Your 5:42 PM and 6:04 PM PST are read whole, and the three comments on NET-399. **The three suites pass whole at `233d7c8e184`, with the two reds gone. And every stream of the certifier's is one template, under a subset design.**

1. **The suites** [ran] r20261113-012352-29bd at `233d7c8e184` (on origin, `cursor/na-scratch-net386-5924`), each verdict line as the run printed it:
   - `passed  verity/protocols/certifier/verifier 1491 passed, 1 skipped in 1021 s`
   - `passed  verity/protocols/certifier/verifier/network 104 passed in 824 s`
   - `passed  verity-network-traces        689 passed in 428 s`
   - `network` was 102 passed and 2 failed before lean's one line; it is 104 passed with it merged.
2. **One template: yes, always, for each of the certifier's three streams in the tree; and none meets proofs' refusal.** [read at `233d7c8e184`]
   - Each one's Program is one proof block's Definition repeated (`population_program` of one unit): the ingress tie over a phase's T bucket rows (`tie_service.py`), the response tie over a class's rows (`response_tie_service.py`), the egress record's at a population of 1 (`service.py`). So each has one stratum [ran, my machine, for the response tie alone: on a toy at 1, 3 and 6 rows, with and without the stop, one stratum each].
   - Each one's design is a subset, not a stratified one: the response tie samples at `rate:NUM/DEN`, which the stream turns into `subset:K` [ran, same]; the ingress tie's selection is `All` and its record `subset:T`; the egress record's is `All` and `subset:1`. So none's registration has a `law.strata`, and a refusal of a design that is no subset meets nothing in them.
   - I found no stream named a stop tie in the tree: if "both ties" on Tue 17 Nov are other than the ingress tie and the response tie, say which and I read it.
   - None of the three goes through `served_verdicts.sh`'s `V3C_LAW` preflight, the one proofs says refuses `subset:` under `H1_TIE` [read].
   - **Not read:** the compute audit's own window (V3c's class window under `H1_TIE`, where a stratum is a table's readers). That registration is serving's. No kept tied run is in my reach: the windows' registrations I hold are all untied and `subset`. If your question is about that window, it is vllm's to read on a run.
3. **Two stamps of mine were written before I read the clock, again.** My last message's header reads 5:27 PM PST and its notes sync 5:26 PM PST; it was sent at 5:25 PM PST and the sync ran in the minute before. The header above is the clock's.
4. **Read and held, nothing built:** a stream a phase; the close on the stream's own clock at the registered end; the salt from the challenger's coins; d = (offset + W) ticks in seconds, inclusive. The salt fact you list as owed for Fri 10 AM PST is the one sent at 4:45 PM PST and taken at 5:04 PM PST; say if you want more than that. What a stream a phase costs a period's registrations: by Mon 16 Nov noon PST, as you set.

TIMER: none tonight unless you send work; next is the Mon 16 Nov noon PST line.

[2026-11-13T02:10Z] 717e5cd1e24b
**TERMGB to the lead, 6:10 PM PT, Thu 12 Nov: #9454's branch is at `f0616c75f9ba`, the merge of `main` with both sides' marks, and `verity-bench`'s suite passed at it: `r20261113-015915-cdc9`, exit 0.**

**1. The head.** `cursor/na-certifier-term-simulated-b4e6` is at `f0616c75f9ba510241b99c6eac1875f889e2bf97` [ran: pushed as a fast-forward from `4916fc023070`; no force]. Its parents are `4916fc023070` and `origin/main` at `636d15d041b5`. It met #9421 only: #9467 was still open at 6:08 PM PT.

**2. The run.** `r20261113-015915-cdc9`: done, exit 0, tree `f0616c75f9ba`, on a CPU node of the pool. `verity-bench`: 566 passed, 0 failed, 1 suite of 1 [ran: the masked reader].

**3. What the merge changed by hand: three conflicts and one pin.** Everything else is git's own merge.

- **`MARKS` is the union:** `("measured", "modeled", "simulated", "assumed", "missing")` in `benchmarks/harness/verity_bench/exfiltration.py`. `main`'s comment above it and its new constants below it are kept as they are.
- **The line that counts the inputs by mark names both.** With no missing input it reads "N modeled, 1 simulated and K assumed.", as this branch pinned it. With some it reads "N modeled, 1 simulated, K assumed and J missing."
- **The README's Inputs paragraph** (`benchmarks/exfiltration/README.md`) describes `simulated` and `missing` both, each in its side's words.
- **One pin of `main`'s side moved,** and it is the one line I changed outside a conflict: in `test_a_field_the_record_lacks_names_its_gap_and_no_share_is_claimed`, "modeled, {assumed} assumed and 4 missing." became "modeled, 1 simulated, {assumed} assumed and 4 missing." It failed at the merge without it [ran: 20 passed and 1 failed before, 21 passed after, on this machine]. It is benchmarks' test, so please read that line.

**4. One thing I left as the merge made it.** Three sentences say every input is "labelled measured, modeled, simulated or assumed": the README's opening, the module's docstring and the command's help. They name four marks and `MARKS` holds five. `main`'s own text there named three of its four, so `missing` was never in those sentences. I added nothing.

**5. The same commit is also on my scratch** `cursor/na-certifier-term-simulated-merge-wip-b4e6`, pushed before the suite ran. I delete no branch, so it stays.

**6. W's two suites runs are still running** at 6:08 PM PT, and the first will end red on the harness's seven, as expected.

- **Why they are long:** the change from `0e8d16c3b8be` touches 57 suites at W's head, `verity-flock` and Flock's Lean build among them, because it holds `verity/protocols/certifier/verifier/class_charge.py` from the branch's base [ran: the run's own first lines].
- **`r20261113-005354-0e28` so far** (at `9096bb240a2b`): 46 suites passed and one failed [ran: the masked reader on the running run]. The one is `verity-bench`: 556 passed, 7 failed, the same seven tests by name as in my table on NET-403. So its exit will not be 0, and the same holds for `r20261113-012901-0c7e` at `d7d44f849cec`.
- I send each run's verdict line when it ends.

[2026-11-13T02:15Z] 43368400e502
BOUNDTIE to network-accounting (lead), Thu 12 Nov 6:15 PM PST. NET-399, #9541, your 5:11 PM text: done.

Head: cursor/na-head-tick-6a2d at f1b9feaa5726 on origin [ran: git rev-parse]. One commit on 25581bf2343c, three files, texts only, no code and no test [ran: git diff --stat]. Unmarked.

The two phrases, at every site you named [ran: rg]:
- "No check and no verifier reads it. Nothing holds the head's time yet: that is the auditor's side, which is not built." PROTOCOL.md's "Who publishes" and "Not held here"; connection_rows_run.py's module docstring and check's docstring; warden_serve.py's module docstring.
- "it stands where a head or the word would, so it can only take one away. What an execution phase with no head is charged is its reader's: each class's whole schedule, which no reader here charges yet. The summary's heads names those execution phases for it." PROTOCOL.md's "A head not back"; connection_rows_run.py's "The check"; check's docstring.
warden.py's docstring and the flags' help are as they were.

Runs, each launched once:
- f1b9feaa5726: r20261113-013236-a59e, exit 0. The certifier's verifier suite 1447 passed, 1 skipped; verity-network-traces 635 passed; repository 352 passed [read: the run's stdout].
- 25581bf2343c: r20261113-003225-7034, exit 0 (1447 passed, 1 skipped; network 85; traces 635; repository 352) [read]. r20261113-003309-99da, the served rehearsal at offset 5, W = 0: accepted, 6 of 6 registered, heads {"offset": 5, "wait": 0, "missed": []} [read].

The body has the heading "Not held" with your three lines, and ends Refs NET-399 [ran: gh pr view].

The verifier's side: the two functions under verity/protocols/certifier/ that read a rows line are connection_rows.check_registration and connection_rows.check_no_head. Each holds its document to its exact keys, so a tick inside one is refused by name [ran: the test].

Not built, and smaller: keeping the tick out of rows.jsonl, in the device's own log alone. Your word first.

Scratch E: cursor/na-scratch-e-399-lean-4d7a at d1bd0282a3b0, holding this head. No suite run there at this head.

I rebased nothing and merged no main. I wait for your word on #9375.

[2026-11-13T02:15Z] fd1be9e971ad
From PERIODPROOF, 6:15 PM PST. NET-402: the scratch's head, the runs, the reading at the period's own δ, the gate's count.

**Head on origin:** `cursor/network-accounting-period-bound-scratch-4a02` at `e01f34ec3a3db48d77445be7dfd826a23b570732` [run: `git ls-remote`]. It holds lean's new head `34c5cbb3705c` by one merge (`cd8e597981`), nothing rebased. Its diff against `34c5cbb3705c` is six files: the check, its mirror, the tests, the two security Lean files and the properties lock's listing; no lakefile [run: `git diff --stat`].

**Runs**

- Tries on the proofs file: `r20261113-000705-77cc` (form (b): 56 theorems), then `r20261113-005515-e87d`, `r20261113-011038-e63a` and, at the final tree, `r20261113-013901-1e0a` (57 theorems): each Lean exit 0 and "none of its … theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; the last rebuilt `NetworkCheck.Period` and `Properties.NetworkPeriod` with their guards. On the check's own file, in its own package: `r20261113-003152-3570`, Lean exit 0 [run].
- Suites `r20261112-230333-160e` (at `ca31f85639de`, rc 1): "59 suites: 58 passed (11 cached), 1 failed". The failed tests, in the log's words: `FAILED tests/test_nci_record_difftest.py::test_a_pinned_readers_layout` and `FAILED tests/test_nci_record_difftest.py::test_the_reference_reads_a_record_as_the_interface_does`. No third. Not run again for them [run].
- Suites at the merged head `cd8e597981`, `r20261113-010115-949e` (`--changed 34c5cbb3705c --quick`): rc 0, "60 suites: 60 passed (5 cached), 0 failed" [run].
- The network suite whole, slow tests included: at `cd8e597981` `r20261113-012246-5345` rc 0, 675 passed; at the head `e01f34ec3a3d` `r20261113-014345-3c87` rc 0, 675 passed [run].
- The changed suites at the head: `r20261113-020437-a214`, launched 6:04 PM PST, running. One line from me when it ends.
- Records: `r20261113-001533-9f61` (at `6f275e53a39f`) failed rc 1 in the build of `Proofs.Properties.Verity.Sampled.Instance`, the old base's own, and on nothing of mine. `r20261113-010111-9359` (at `cd8e597981`): rc 0, `regenerated`, the audit passes after the update. Before the update the properties lock's check named: my six entries with no record (the two guarantees and their four witnesses); `rowsCapPublished_nontrivial`'s signature, the base's (`entries := 0`); and `reads` of eight modules, two of them the ones I edit (`NetworkCheck.Period`, `Properties.NetworkPeriod`) and six the base's. The security lock moved in eight `reads`, `NetworkCheck.Period` and seven the base's. No bookkeeping line names an entry of mine [run; read with a three-way script: base, head, kept].
- **I commit no record.** The run is one commit behind the head, and the base's entries are not mine to write. The records run goes on `main` after the landing, as for B and C.

**`RowsTiedCharged`'s lock notice** [run: the same records run]. Its own record does not move: `type_hash` and `unfolded_hash` read `RowsTiedCharged`, not what it unfolds to. The notice is "definition Properties.NetworkPeriod.RowsTiedChargedAt (Properties.NetworkPeriod), read by Properties.NetworkPeriod.rowsTiedCharged: changed", with `tiedInPart` and `AuditFacts` the same way. It stays under `unwitnessed`, same entry.

**The reading at the period's own δ (ONSITEMAIN's part), as built** [read: `NetworkCheck/Period.lean` at the head]

- `Facts.design` (line 453), a fifth field after the four, default `subset 0`. Staging calls it `draw`; I named it `design` so no new name holds a retired term.
- `Facts.readAt f δ` (1223): the same sample at δ, its count `max f.wrong (sampleBound f.units f.design δ)`. Staging replaces the count; mine never lowers one the check holds.
- `tiedDelta := ⟨1, 2 ^ 64⟩` (1396); `chargedOf` (1401) returns `f.readAt tiedDelta`; the rows public's JSON gains `"charged"`.
- In the security Lean [read: `security/Properties/NetworkPeriod.lean`]: `SampleHeld` (152) gains `f.design = d`; `tiedInPart` (230) maps `.readAt tiedDelta`; `RowsTiedChargedAt` (243), sampled in part: `f.readAt tiedDelta ∈ pub.charged ∧ SampleCharged reg (f.readAt tiedDelta) ∧ (1 ≤ (f.readAt tiedDelta).wrong → …)`. Proved: `sampleCharged_readAt`, `charged_eq`, `rowsTiedCharged` [run: the tries].
- **What it moves** [run: a scratch script outside the tree, the mirror standing in for Lean, 48 seeds]: against form (b), 9,888 cases, no verdict and no reason moves; 384 accepted periods have another public, in `eps` only, which is lower in each: `tiedDelta` stands where the answer's δ stood. The answer's δ stays in `audits`.
- The whole change against the base, 9,312 cases: accepted then refused 0; a refusal's reason moved 0; 464 accepted with another public; K raised in 229, lowered in none; `faulty_entries` from 0 to its scan's entries in 85 terms.

**Positional literals** [read: the diff]. I changed `Facts`' only: the guards of `terms`' and `rowsTerms`' `eps`, the w63k4 guard, one pattern of the last guard, and in the proofs file `rowsTiedCharged_honest`'s two and the negatives' stated publics. None of `Params`, the evidence or `cappedPeriod`. No field reordered. My hunks in `Period.lean` are the module docstring, `Facts`, and from `sampleOf` down; `Schedule.digest` and `Difftest.lean` are untouched, and the lakefile is unchanged: `Spot.Record` resolves without a direct `require`.

**The ratchet gate** [run: `adapters/vllm/tools/ratchet_gate.py 34c5cbb3705c HEAD`]

- At `7b241884bc` it failed with "adds 197 retired-term site(s) to the files it changes". The head's last commit rewrites my own locals, docstrings and two refusal strings. Now: "adds 39".
- The 39 by term: `law` 26 (the registration's wire key, in `sampleOf`'s read and in the registrations the guards, witnesses and tests build), `cappedPeriod` 9 (the six witnesses' period), `drawn` 6 and `Draw` 1 (`Facts.drawn`, `profileDraw`, the public's key), `Cap` 4 (`rowsCapPublished_honest`, `maxTokensCap`), `draw` -7. By file: proofs 85 to 97, properties 86 to 94, `Period.lean` 255 to 264, the mirror 176 to 177, the tests 64 to 73.
- All 39 are names `main` gives or wire keys. **I added no entry.** The gate's message, whole: "Say the term the row gives (GLOSSARY.md's older names). A use the table keeps (a recorded key, a wire format) stays only where it already stands. If the file moved, move its entry in the same change: `python3 tests/lint/test_repo_retired_terms.py --move OLD NEW`; a new row of the table records the list in its change (`--record`)." It names no fix for a wire key or a name of `main`'s read in a new place. Your word on them.
- **The two refusals' text changed with it**, in Lean and the mirror alike: "…: registration: no sample: it names no subset or stratified sample of a population" and "…: profile: not the registration's sample: its level of N units and its sampling design are not the registration's M units and theirs". Both are refusals this change adds; the names the tests match (`registration: no sample`, `not the registration's sample`) are as they were.

**An honest lemma at an accepted period** [argued; not tried]. It needs `check` to accept a period with one audit, so the kernel must evaluate two SHA-512 digests inside it, the Program descriptor's and the canonical registration's, at a written descriptor one-stage's answer names. I did not measure what the kernel takes on that.

**The body file** `tools/net402/pr-body.md` in the notes: all sections filled but the records line, which waits for the run on `main`. It names the three negatives and why the lock lists one, the two pinned tests, the counts above (464 periods with another public now, where form (b) alone had 352; K raised in 229, lowered in none), the hand-built `Facts`, the three refusals kept, and the gate's count.

**Least sure of**

1. `readAt`'s `max` against staging's plain replacement. Mine is the stricter; say if staging's is what H needs.
2. `SampleCharged reg (f.readAt tiedDelta)` inside `RowsTiedChargedAt`: it is more than you asked for there.
3. `eps` lower in 384 periods. It follows from the ruling as I read it, but it is a number a reader may take for the audit's error.
4. The 39 sites.

**Where I did not look:** the whole `verity check`; `circuit-check`; staging's branch since its commit `00d79b22e4d2`; TIEVERIFY's scratch beyond your lines on it.

