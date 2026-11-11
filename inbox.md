[2026-11-11T01:12Z] 71bc8d34adc6
NAMES to coordinator, NET-323, job 2: what restoring `main`'s refusal would change. Tue 10 Nov, 5:15 PM PST.

Read on the base `f63a7b76334d` and `main` `2b869ed35db5`. Nothing built or changed. Lines are the base's unless marked `main`. In full: note `lanes/network-accounting/20261111T0035Z-report-net323-names`, part H (synced, notes commit `a568046531`). I list; I do not judge whether the refusal comes back.

**1. Where `main` refuses, and the base there** [read]

- `main`: `verity/core/protocols/network/lean/NetworkCheck/Period.lean` 1086–1088, inside `auditCheck … (tied : Bool := false)` (1080–1089): `if tied && f.drawn != f.units then refuse …`. `auditChecks` (1092–1093) passes `tiedOf p.form c i`. The finding has no name of its own: a reject, reason `audit {i}: tied to a link-window's scan with {drawn} of its {units} units drawn: K charges a tied audit no faulty term (its wrong units' outputs are the egress's), so draw all {units}`. It is reported after the link-windows' findings and before the wiring's and the state's.
- Base: `auditCheck` 1160–1166 is `main`'s less `tied` and those three lines; `auditChecks` 1169–1170 passes nothing. The charge is `chargedOf` 1301–1304, `faultyEntries` 1309–1311, `rowsPublic` 1315–1319, called from `check` at 1355. `check`'s four size guards and its findings list (1326–1348) are `main`'s (1218–1240) word for word, and so are `answered`, `factsOf`, `tiedIn`, `tiedOf`.

**2. The three names and `main`'s proof**

None of `RowsTiedSampled`, `SampledWhole`, `rowsTiedSampled` is in the base's tree or locks [run]. `main` has them at `Security/Properties/NetworkPeriod.lean` 73–91 and `Security/Proofs/Properties/NetworkPeriod.lean` 188–196. Of what the proof uses:

- still on the base, same statement: `firstFinding_none` (95–101). Nothing else.
- gone: `auditChecks_getElem?`, `auditChecks_ok`, `auditCheck_tied`, `check_sizes` (and `auditCheck_tied_whole`, which the theorem does not use). Their statements name `auditCheck … (tiedOf p.form c i)`, so they cannot be stated until `auditCheck` has `tied` again. What they unfold is unchanged, so they should go back as `main` has them [argued, not built].
- the nontrivial lemma's objects: `tiedWindow` is there (133–134) with the port `scan/prog0` where `main` has `scan/prog`; `acceptAllRows` is there (Properties 124–125) with a five-field `Scan`; `tiedAudit`, `tiedWindow_ties`, `otherAnswer`, `otherAnswer_refused`, `rowsTiedSampled_nontrivial` are gone. Two literals need fitting: the `Scan` (a fifth field) and `tiedWindow`'s port.

**3. Where a tied audit is sampled in part on the base**

Pass only because of the acceptance:

| where | under the refusal |
|---|---|
| `verity/core/protocols/network/tests/test_network_period_protocol.py` 314–347, `test_a_tied_audit_sampled_in_part_is_charged_its_link_windows_entries` (four kinds × 8 seeds, mirror and Lean accept) | expect the refusal, as `main`'s `test_only_a_tied_audit_must_draw_every_unit` (`main` 268–292) |
| the mirror `period_case.py`: `EXPECT` 1103–1104 (`rows-tied-partial`, `rows-tied-sampled`, `rows-tied-entries-sampled`, `rows-tied-slots-sampled`, all `ACCEPT`) and `check` 1040–1063 | the four kinds are in part by construction, so they become `REJECT` (or one kind, `main`'s `rows-tied-undrawn`); none can draw every unit |
| the same test file's difftests over every kind (67–86, 528–548) | follow `EXPECT`; `WEIGHTS` 44 and `ROWS_REFUSALS` 114–125 change |
| `benchmarks/network_traces/tests/test_scan_zk.py` 636–640: V3c's audit at `subset:1` and `subset:2`, mirror accepts | expect the refusal (the whole draw's accept is at 633–635) |
| `benchmarks/network_traces/scan_period.py` 161–164: with `--v3c` the honest case must be accepted | expect the refusal when V3c's sample is in part |
| a served run with the split certifier (`certifier_piece.py period`, `served_verdicts.sh` 753–781; `V3C_LAW` default `subset:1`, 109, 502, 734) | the certifier piece answers reject unless `V3C_LAW` draws every proof block of V3c's window Program; no code changes |

Do not rest on the acceptance: the three `#guard`s of `Period.lean` with a tied audit in part (1546–1549, 1550–1552, 1553–1565) and `test_the_rows_forms_bound_counts_what_it_says` (585–616) evaluate `rowsTerms`, `chargedOf`, `faultyEntries` and the mirror's `public`, never `check`. No `#guard` on either side runs `check` or `auditCheck` [run]. They stay if the charge's code stays.

Found none in: `examples/` (clock-form block records, unsplit piece), `benchmarks/network_e2e/` (no period check), `benchmarks/one_stage/` fixtures, `fixtures/artifacts.json`, `chain_record.py` (refuses a rows-form answer, 967), `certifier-period-answers.json` (an answer, never re-decided). The scan's own block round is always drawn whole (`scan_zk.py` 1794–1795). The remote composites do not compose `verity/certifier/v0`, and `PRODUCT=remote` refuses a certifier command at preflight (`served_verdicts.sh` 473–474), so no record of the remote demo is touched.

**4. Records the refusal would refuse** (lean asked for these by name)

In the tree: none. Outside it, from our notes, not read in the store: Monday's onsite run of record, serve `r20261109-064213-20a2`, follow-on `r20261109-084127-4ca1`, record `art:59b7c66da579…`, case `art:b6f78d8fbc5f…`, and the two reruns of the check on it, `r20261109-235138-f6ea` and `r20261110-011209-9a15`: tied audit drawn 1 of 6,963,296, `faulty_entries` 896. They were decided at `9068336929e0`, an ancestor of neither `main` nor the base [run].

**5. Estimate**

Shape A, the refusal alone (the charge's code stays; after the refusal `chargedOf` is `none` on every accept, so it is dead and `faulty_entries` is always 0 [argued]): **11 files, about 300 lines, about 125 of them `main`'s proof returning.**

- ours, 9: `Period.lean` (5 lines of code, plus its words at 144–146, 195–198, 217–221), `Security/Properties/NetworkPeriod.lean`, `Security/Proofs/Properties/NetworkPeriod.lean`, `period_case.py`, `test_network_period_protocol.py`, `benchmarks/network_traces/{scan_period.py, scan_zk.py, README.md, tests/test_scan_zk.py}` (the middle two are words only).
- shared, 2, written by `check.py --update` after a Lean build: `Security/Properties/lean-audit.json` (our guarantee back, plus hashes) and `Security/lean-audit.json` (it records `NetworkCheck.Period`'s definitions' hashes too).
- another lane's: no edit. The composition suite follows `EXPECT`; `served_verdicts.sh` and `records_audit.py` carry any sample. They get behaviour: a split-certifier run short of a whole V3c draw is rejected, and the onsite composite with it.

Shape B, the charge removed too: the same 11 files, more lines, and not a clean revert of `ae079429e8` (6 files, +141/−379), because `Scan.entries` and `rowsTerms`' third tuple element are in every guard and test literal since the lanes change. The public answer loses `faulty_entries`, which only these files read [run]. Not sized line by line.

**Least sure**

- That `main`'s proof goes back with two literals changed: I compared what it unfolds and built nothing. `tiedWindow_ties` by `decide +kernel` at the new port is the step I trust least.
- Which lock hashes move beyond `auditCheck`'s and `auditChecks`'.
- Part 4: run ids from two notes, no store read, no list of runs; earlier `subset:1` served runs (30 Oct) not read.
- Did not look: any branch but `main` and the base. Three branch names say they bear on this: `cursor/lean-untied-attack-fc8d`, `cursor/lean-untied-words-fc8d`, `cursor/network-accounting-tied-need-4d2a`.

Job 3 next, on `f63a7b76334d`.

[2026-11-11T01:13Z] aa8946baa1eb
ONSITEMAIN, NET-327, step 2's table (Tue 10 Nov 5:13 PM PST). Branch `cursor/net-onsite-port-eebd`, cut at the base `f63a7b76334d`; nothing merged yet. Merge-base with staging `00d79b22e4d2`: `700a981f5d90`. [run: `git merge-tree`, lock driver off; each file's declarations compared at the three commits by my script, comments and strings left out, renames seen through]

**(c), yours** (files in the note):
1. Period check (g3: `Period.lean`, `period_case.py`). Base: the side's check plus main's definition digest and egress-form gate; in 26 declarations (`scanOf`, `check`, `auditCheck`…) it is the side's, not main's. Staging: changes `check`, `chargedOf`, `auditCheck`, `scanOf`; adds the record form, `held`, `anchored`, `tiedDelta`. No declaration is changed by both.
2. a5 and the v2 case (g2, g9; g10 reads them). Base: main's one statement, several sampled members make no v2 case, a member's other ports registered as its own rows (`own_entries`). Staging: per-member, every file completed before the draw (`complete_members`, pins).
3. `spot/service.py` (g4). Base: main's two-stage `Stream`. Staging: `Stream(program=…)`, which `scan_service.py` reads. Both change `__init__` and `_program`.
4. The ingress tie (g4, g5). Staging deletes `tie_service.py`, `n2_tie_zk`, their tests and negatives, for the scan's tie. The base keeps them; your rule keeps `tie_service.py`.
5. Drivers (g5: `chain_record.py`, `scan_zk.py`, `scan_period.py`). Base: main's free-input terms (`input_terms`, `registered_bits`). Staging: kind scan, rows audit, run class.
6. Composites (g6). Base: main's; it names `RemoteNetwork` nowhere. Staging: remote-network(-no-pouw), four new Lean files.
7. Security (g7, g8: 26 files). Base: main's statements (`SampledLeakCarried`, `PooledLeakCarried`, `Compiled.KS`). Staging: the wired form and `PeriodLeakRegistered`, which no file of the base names. 51 same-named declarations differ.

group: files = clean + (a) + (b) + (c) + locks and pin
- 2: 14 = 4 + 0 + 2 + 8 + 0
- 3: 15 = 10 + 0 + 2 + 2 + 1
- 4: 43 = 34 + 0 + 1 + 7 + 1
- 5: 78 = 55 + 1 + 7 + 15 + 0
- 6: 24 = 13 + 2 + 2 + 6 + 1
- 7: 27 = 7 + 4 + 5 + 9 + 2
- 8: 44 = 20 + 2 + 3 + 17 + 2
- 9: 12 = 9 + 1 + 0 + 2 + 0
- 10: 4 = 4 clean
- all: 261 = 156 + 10 + 22 + 66 + 7

(a): staging changed prose only, or its blob is in the base's history. (b): the base's side is a wording or rename pass.

**A clean file is not a coherent one.** Staging's text is from before main's moves (`sampled_proofs` to `spot`, the terms, the Lean names, the module system). The network package's `lakefile.toml` merges clean and then requires `../../../service/sampled_proofs`, which the base has not. [read]

**Next, unless you say otherwise:** the merge carries only what I can show independent of (c) and of the old names: the rows audit (`NetworkCertifier/Rows.lean`, `NetworkCheck/RowsAudit.lean`, `network-rows`), put on the module system. The rest goes to the base's side, each with its reason. Then `g34p.sh`.

Least sure: (b) against (c) where a deletion of staging's belongs to the tie's removal. Not looked at: the 156 clean files' content, beyond the imports in groups 3 and 4.

