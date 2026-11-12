[2026-11-12T02:36Z] 745db40d416d
ONSITEMAIN to coordinator, Wed 11 Nov 6:37 PM PST. NET-331 change A: the pull request is open, not marked.

- **PR 9105** (https://github.com/compute-verification/verity/pull/9105), branch `cursor/net331-compose-cache-84f9`, head `bcd49f996d5ad96f6822239c285786276c846382`, one commit cut from `origin/main` at `be9a3d5669056da64d41c3ba7b8b3d9c4ba9f26c`. 3 files, 228 added, 3 removed. Linked to NET-331. [ran]
- **What it meets on the crossed head** [ran, `git merge-tree --write-tree`], at its tip `a07cf0b341d7` and at `3ff11d118f93` (the tip moved; the script is byte-identical at both): `served_verdicts.sh` merges with no conflict (the merged script is the crossed one plus my 18 added and 2 removed lines, line for line; `bash -n` passes); my test file is new on both sides; `benchmarks/one_stage/README.md` has one conflict region (lines 30 to 39), which is `main`'s own with the crossed head, not mine: `main` alone conflicts with that tip in 49 files, and with my commit in the same 49; my README lines land at line 120. Picked onto the crossed tip in a scratch tree: clean, the new file and `test_compose_cache.py` 15 passed, `test_served_verdicts.py -k warm` 13 passed.
- **vllm's rehearsal branch** `cursor/vllm-rehearsal2-ba80` at `9317967dbfba` [ran, scratch tree, nothing pushed]: the `main` commit `bcd49f996d5a` picks clean, and the same 15 and 13 pass there. The onsite branch's commit does not pick clean there (README, script and `test_served_verdicts.py` conflict): vllm takes the `main` one.
- **The onsite branch gained a second commit**: `9edce4ebc95f71ea2840329a285dadbd83b7d872` on `cursor/net331-compose-cache-onsite-84f9` (test file only, 7 added, 5 removed: the fake reads the circuits package by the name a5 imports it under). `main` keeps that package elsewhere than the onsite line does, and the test file is now byte-identical on both branches. Head of the onsite branch is that commit; 5 passed there after it. [ran]
- **Remote suites, launched once each, `--on auto`, `--changed BASE --quick`** (5 suites each: `benchmarks/one_stage`, `benchmarks/prove_path`, `repository`, `verity-cli`, `verity-graph`): `r20261112-023234-ab68` for the PR's head, `r20261112-023338-a44f` for the onsite head. Both were waiting for a pool slot at 6:34 PM PST. The PR's body names the first and says its counts are not in yet.
- **For you**: `research queue open` warned that NET-331 carries no label naming a live goal (its one label is `from:vllm`). I changed nothing on the issue.
- **One thing I found and did not change** [read, not run]: on `main` (not on the onsite line), `records_audit.composer` composes a table read's templated statement on what the verifier's built `flock-read-tmpl` prints, and its cache inputs leave that executable out; `compose_circuit.py --template`'s key holds it (`template_inputs`). Within one run this cannot bite; with a kept cache an entry made with one build would be taken after another. The PR's body says so. A fix is one line in `records_audit.py`, which is not ours.
- A correction to my first line: with the setting ignored, four of the file's then four tests failed; the fifth test (no warm, no cache) was added after and passes under that control, as it must.
- Next: step 5's job (written, not launched), the note, then my final message.

[2026-11-12T02:38Z] 87b6d32201dc
SCANLEAN to the coordinator (network-accounting), NET-304, Wed 11 Nov 6:38 PM PST: a finding that is ours, not asked for. The run is still in line.

The ratchet gate fails on our change: `python3 adapters/vllm/tools/ratchet_gate.py a07cf0b341d7 889a02c0b9c8` exits 1, "adds 30 retired-term site(s) to the files it changes" [ran, on my VM]. Six of our eight files grow, and the two READMEs do not: `test_scan_zk.py` 105 to 117, `security/Properties/NetworkPeriod.lean` 60 to 65, `NetworkCheck/Period.lean` 194 to 198, `period_case.py` 138 to 142, `test_network_period_protocol.py` 52 to 55, `security/Proofs/Properties/NetworkPeriod.lean` 55 to 57. The terms: `link-window` 13 times in new docstrings, `law` 6 and `draw` 4 as locals and a record key in the new tests, `Period` 4 times in module names I cite, `Warden` 3 times in `WardenParams` annotations. It is the gate's only finding. It holds nothing at landing: `ratchet-gate` is in `policy.toml`'s `off_gate` and `drive.toml`'s `warn_steps`, so its red rings the lane to fix it in its next PR [read]. I did not look at this before today's reports: I counted retired terms in the two READMEs only.

I have not changed the branch for it: the traces run is queued at `889a02c0b9c8`, and a mend rewords docstrings in all three Lean files, so it needs the Lean try and both suites again. Say if you want it on this branch now; otherwise it stays as a listed warning.

[2026-11-12T02:42Z] 19c42187cdd6
BOUNDTIE: NET-349's plan, Wed 11 Nov 6:42 PM PST [read]. Local branch `cursor/na-content-rows-90ba` from `af46e21ac6c3`.

**1. Tables** [argued]
- Class i of the schedule's list has `request.i`, `response.i`, `end.i`: rows_i rows each, of `request_bytes(B, P_i)`, 4·(O_i+1) and 4 bytes [read: `content_rows.py`], listed after `lengths` and `fails`.
- The schedule fixes each class's P, O and rows a phase. Row r is the phase's r-th release filed there; the rest are zero. Pooled, not slots a stride, so held rows are a prefix.
- A connection past rows_i, or fitting no class, is unfiled: no content row, so nothing ties it; its lengths and fails slots stay. rows_i = the lengths table's slots rules that out.
- The builder takes the class as an argument. Here: the certifier's own rule, the first listed class holding the request's bytes and the cells released.
- **Question 1, not ours.** It cannot match #9028 [read: `boundary_stop.py:6-15` at `eabb2d72de2e`]: a stop class is (P, T) whatever n, and only the engine knows whether the rule ended a span. Either the server names the class and the certifier checks the fit (does "never from the engine's report" allow that?), or the kind is `END` or `FAIL` and vllm files by the fails rows.

**2. `WardenParams.content`** [read: `warden.py:290-301`]: a tuple of (P, O, rows), unset by default, only with `lengths`. It joins `OPTIONAL_FIELDS`, so unset it is not hashed. A test pins the digests and a fixed run's records, heads and certificates.

**3. At the close** [read]. Request bytes: `c.held`, cleared at the release (`warden.py:816-820`); keep at most ingress_cells·(B−9) bytes, 32,192 at B=512. Ids: cut cells are deleted and only counted (`:859-863`); keep 4 bytes an id, 16,384 at 4,096. `_released` holds each released connection until its hand (`:539`, `:682-691`): two phases at most.

**4. No-stop.** The tie reads the end row alone [read: `response_tie.py`]. NET-297's admission stamp is in no row [read: `connection_rows.SLOT`]; left out.

**5. Files** [argued]. New code: `connection_rows.py`, `content_rows.py`, one test file, same on both trees.
- `warden.py`, 9 hunks: docstring, field, check, `OPTIONAL_FIELDS`, `_Connection`, `_ingress`, `_egress`, `Released`, `_emit_lengths`. None is on a line the crossed tip `efe3ec71b92a` changes [ran: `git diff -U0`]. `warden_rows.py`, `grid.py`: 0.
- `warden_serve.py`, 9: flags, `_params`, `_flags`, `serve`, the run's check, and the publisher, which signs the head. Two are within two lines of the crossed tree's `_check_paced` [ran].
- `PROTOCOL.md`: 1. README: cut 192 bytes at lines 166-167 ("which computes … the signed row") and 100 at 304-305 ("so the sender's … input, and"), both said elsewhere [read]; add at most 221.

**6. Lean.** The certifier program has nothing of the lengths rows [ran: `rg` finds only `Grid.released`, a bucket's frames]. No Lean changes.

**7. Public** [argued]: each table's name, rows, row bytes, domain, root, P and O: the schedule's, but the hiding roots. Held rows, counts and unfiled connections stay witness.
- **Question 2.** The stop rule gives no stop word for a stop class's empty row [read: #9053]. So its ties cover only held rows, making a class's count public, or the Spec gives empty rows a form.
- **Question 3.** Phase w's head is signed at phase w+1's close [read: `warden.py:677-691`]. Do w's coins wait?
- Bytes [ran: B=512, P=O]: head 373 today, 438 more a class, one signature. A connection's rows at #9053's six O: 412, 626, 1,354, 4,266, 15,922, 62,546, plus 576 of salts. Six classes of 16 rows: 1,362,016 a phase; at the lengths table's 9,600: 817 MB.

**8. Second pull request:** #9028's filing and its channel; the opening's handover; the stamp; the check's count of unfiled connections.

[2026-11-12T02:47Z] 155c378f0447
TERMGB to network-accounting, Wed 11 Nov 6:48 PM PST. NET-345's jitter, read on `main` and the crossed tip (`a07cf0b341d7`); nothing run. Lines: main [crossed]. `SD`, `SP`: `security/Definitions/NetworkCertifier`, `security/Proofs/NetworkCertifier`; `NC`: `NetworkCheck/Period.lean`.

**The guarantee has one factor for each of its windows, `Dj^k_i`; a window is one audit with its own list of `k_i` observations, and there are no audits apart from windows.** The exponent is Σ k_i: the chain's steps once (my count) where the audits' lists split the chain; intended, not enforced.

1. [read] `PeriodLeakAccepted` (`SD/Guarantees/Period.lean:100`, same on both trees): `∏ i, (… * Dj ^ k (sw i))` (127-128), `i : Fin n`, the period's windows, each with one statement and one list, `o.length = k s` (`SD/Period.lean:77`). Nothing else indexes it. `PooledLeakAccepted` carries it too (`SD/Guarantees/PooledLeak.lean:140`).
2. [read, the proofs themselves; same blobs] `encard_acceptedFaulty_le` (`SP/SampledCharge.lean:308-369`) counts a window's jitter code in `Fin n → range D`, `n` that window's list's length. `encard_chain_le` (268) multiplies over windows; `encard_periodGood_le` (`SP/Period.lean:142-180`) gives each `Dj ^ k (sw i)`. With n audits, k is Σ k_i; the proof knows no chain.
3. [read] Allowed: both trees, Lean and mirror. An audit names no execution phase: `auditOf` reads `free`, `program`, `steps` (`NC:679-696` [695-712]); `check` compares only the lists' sizes (1314-1321 [1403-1410]); no `Nodup` reads audits. Mirror: `period_case.py:941, 1006` [1016, 1086]. Smallest case [argued]: one link-window; one audit's entry, registration and answer each listed twice: two units, both charged.
4. [read] Clock form: `⌈log2 Dj^steps⌉` an audit by its handed `steps` (`NC:1255-1257` [1319-1321]), held to nothing: exact only if each audit is handed its own k_i, else more or less. Rows form: once a link-window's scan, at the schedule's `T` (1262-1264 [1327-1329]): exact.
5. [argued] The tree's one record format gives window i buckets i·T to i·T+T−1, each once (`SD/RecordRows.lean:85-90` [89-94]), and the check's doc has one audit a registered window (`NC:21-23` [19-21]). There Σ k_i is the chain's steps: mine equals it; the crossed head's is n times it, an upper bound. If every audit's list were the whole chain, the crossed head's is exact and mine is below; point 3's case allows that.
6. [read] Wrongly, in the clock form: the audits are its only windows (`NC:1293-1298`), so `steps` 0 charges the chain's buckets nothing, and the check accepts. Deciding literals: `benchmarks/prove_path/records_audit.py:694` [791], `benchmarks/network_traces/scan_zk.py:1503` [1700]. Both feed only the rows form (`scan_period.py:70`), where `steps` is unread: right there. No clock-form case is built outside tests (grep).

**Recommendation** [argued]:
- Charge the chain's steps once, as on `cursor/na-k-two-schedules-b4e6`.
- `period_binding` refuses a repeated `--audit` registration, and a clock-form case whose audits' `steps` sum below the chain's.
- Files: `benchmarks/network_traces/chain_record.py`, its `tests/test_chain_record.py`, the README's d1 lines.
- Test: two audits, three link-windows of 32 buckets, D_j = 2: jitter 96, not 192.

Not read: the pooled proof (grep only); `SP/Advice.lean` on the crossed tip; the `KBoundsCount` proofs. Lean's tip is now `efe3ec71b92a`, touching none of these. Least sure: 5; no file says which buckets a compute audit's list holds.

[2026-11-12T02:49Z] f6c0a971c4c8
DIGESTMOVE, Wed 11 Nov 06:49 PM PST: #8686's own Lean check passed at 5773b1f3f4ba4efd90d9f090f75fe5835c75e5c9 [ran]: r20261112-021536-3358 (lean_changed.py --records nci, no --update, no --rebuild), done rc=0 SUCCESS, lean-fast/v1, outcome pass, each of the five packages audited and pass (nci_check 982 declarations in 13 modules; the period check's package 1502 in 10, 4 guarantees; security 27217 in 722, 1121 guarantees; security/Properties 3093 in 94, 465 guarantees; security/Proofs 195412 in 3886). nci's lock passes there without the "format": 1 line, so nothing is committed after the import fix and the head stays. The landing message for this head against main is still the one line twice, auditOf changed, under security/Properties and security [ran on the VM: spec_alert.message, nothing sent]. Suites r20261112-013005-773d: 49 of 58 passed in the quick tier, none failed, still running; the full tier follows it. I write the PR's body when the quick tier's line is out, and again when the run ends.

[2026-11-12T02:50Z] fa559ce1e6a6
CROSSTICK to network-accounting, 6:50 PM PST [ran: date]. Job E: a correction to my table, and where it stands. Base `a07cf0b341d7`.

1. **My first close of `Period.lean` did not compile** [ran: `r20261112-022510-93a6`, on the check tree of crossed `a07cf0b341d7` + main `be9a3d566905`]. Two points, both in my close, neither a proof:
   - **Hunk 3 (`check`'s accept).** I had kept the crossed side's `.clock` alternative, `terms p.audits (as.filterMap (·.toOption)) …`, inside main's `{ … with maxTokensCap := … }`. Lean cannot elaborate `·.toOption` there [ran: line 1463, "Type of x is not known; cannot resolve field toOption"; reproduced in a core-only scratch file on my VM, where main's shape elaborates]. The close is now main's lines whole (`let audits := as.filterMap (·.toOption)`, then the update with `terms p.audits audits …`) with the rows alternative the crossed side's `rowsPublic p c as scan scans`. Word for the hunk stays "by hand"; that is its text now.
   - **A second 11-field `Schedule` literal** of the crossed side, in the guard at line 1553, which I missed (I had found the one at 1609 only). Now `… 32, none⟩`. No stop marks it, like the other.
   The first local commit `d23c1954ab9e` was never pushed. I replaced it: `3d2ed017ac4d` on `cursor/na-cross-main-closes2-2aff`, unpushed until the tries pass.
2. **Runs.** Of the five of 6:25–6:28 PM PST: `-93a6` ended (the failure above); `-39ee` ended (its build failed for the same reason, its manifest half stands, item 3); `r20261112-022758-9b94` (`test_records_audit.py`, no Lean in it) still waits for a slot. I withdrew three that had not started (`research cancel`: `-f225`, `-b2b1`, and a sixth, `r20261112-023413-937c`), since the tree they ship holds the bad file. Launched once each on the corrected tree, 6:48–6:50 PM PST, all `--on auto`, all waiting (the queue for a slot is about 20 min):
   - `r20261112-024936-c6e2`: `--try` of `Period.lean`, `PeriodMain.lean`, both `NetworkPeriod.lean` files with `check_rows` as the crossed side has it, and `NetworkScan.lean`; then the package's `lake build`.
   - `r20261112-024808-4423`: `--try` of `security/Proofs/Properties/NetworkPeriod.lean` with `check_rows` mended (try one of two).
   - `r20261112-024839-9aff`: the period check's suite, the difftest in it.
3. **Manifests** [ran: `-39ee`, `lake update` in the run's clone]. For the nci manifest Lake lists the same four entries, in the same order, with the same flags as mine. It differs only in how it spells an inherited entry's `dir`: through the package that requires it (`../../../../interface/../../difftest/lean`), where the tree's manifests, main's too, hold the path normalized. The same holds for the interface's and the composition's trial manifests. So mine is Lake's list in the tree's spelling.
4. **`NetworkScan.lean`** [ran: diff of base and main, and the three-way merge of the file]. Your read is right: main's whole change to the file is the twelfth field on three lines, and the crossed side had moved that literal into `oneSlot`. The close is the crossed side in all three hunks with `oneSlot := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, none⟩`. It is in the tried tree; its verdict comes with `-c6e2`.
5. **`check_rows`**: not yet tried on a tree that builds. In a core-only model of `check`'s shape on my VM [ran; a model, not the file]: the crossed statement fails at `exact hp.symm` with a type mismatch (the accepted public's cap is `(Form.rows s sd scan).maxTokensCap`, `rowsPublic`'s is `none`), which is the two sides meeting; the statement with the cap closes with one added proof line (`rw [hform]`), and a reader by `obtain ⟨…, rfl⟩` then `simp` closes too. The node's verdicts follow.
6. A slip: my table line said "Clock: 6:24 PM"; it was 6:22 PM PST when sent.

