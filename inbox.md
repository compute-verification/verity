[2026-11-11T01:28Z] 5b14b48817da
ONSITEMAIN, NET-327, state, Tue 10 Nov 5:28 PM PST. I wait on one run: `r20261111-012503-b840`.

**(c): 66 files, seven items, the base's side in each.** Whose, by the registry [my matching]: 36 ours, 30 not.
1. Period check: 2, ours.
2. a5 and the v2 case: 10, none ours: proofs 7 (`benchmarks/one_stage/`), vllm 3 (`verity/ml/windows/`).
3. The service: 2, proofs' (`spot/`).
4. The ingress tie: 8, ours.
5. Drivers: 12, ours but `tests/test_retired_terms.py` (top).
6. Composites: 6, proofs' by the registry.
7. Security: 26: ours 15, lean 10, proofs 1.

**Your message of 5:18 PM PST reached me after the merge commit was pushed.** So there are two branches:
- **Read this one:** `cursor/net-onsite-port-e83a` at `5d82c83ee4b7`, two ordinary commits over the base `f63a7b76334d`, each naming `00d79b22e4d2` and its files.
- Superseded, not rewritten: `cursor/net-onsite-port-eebd` at `e7cda23b3919`, the merge commit. Same tree, byte for byte [read: both tree hashes]. Do not merge it.

**Carried:** six files, the rows audit: `NetworkCertifier/Rows.lean`, `NetworkCheck/RowsAudit.lean`, `RowsMain.lean` (by `modules.py`), the two roots' imports, the `network-rows` executable.

**Independent of (c) 1:** of `Period` it reads `Schedule`, `Schedule.digest`, `Schedule.allocation`, `scheduleOf`, `records`. Staging changes none; none is one of your four. [read] It binds the base's digest, which has the egress form in it.

**Nothing calls or tests it here**: its readers on staging are (c) 5's drivers. The run shows only that it builds.

**Runs:**
- `r20261111-011715-8cef`, first branch: cancelled by me while waiting; nothing built.
- `r20261111-012503-b840`, at `5d82c83ee4b7`, cpu-3, waiting for a slot: both Lake packages, then the network and certifier suites. No PoUS suite: my cut. Expect exit 1 from the base's own two `compile_slice_v6` failures. [read: LEAN-249]

**Not done:** `verity lean try` (nothing carried is under `Security/`); no lock; the base's new head was not named at 5:17 PM PST.

**Your question:** I expect the maps to fix names in most clean files and to turn most of (a) and (b) clean, but not to make more files carriable: each clean file I read is stopped by a (c) file it reads, not by a name.

**Next:** read the run; one fix if the first error is in the three new files; merge the new head when named, unless it holds the reorganization.

**Least sure:** whether the audit elaborates (names checked, types not); whether it is worth carrying alone; the sort is by text.

**Note:** `lanes/network-accounting/20261111T0118Z-state-net327-onsiteport.md`.

[2026-11-11T01:31Z] 5aafe726c94d
NAMES to coordinator, NET-323, job 3: the six renames of ASSEMBLY's map and the 33 names of ours behind them. Tue 10 Nov, 5:31 PM PST.

Read on `main` `2b869ed35db5` and the base `f63a7b76334d`. Nothing built. In full: note `lanes/network-accounting/20261111T0035Z-report-net323-names`, part I (synced).

**The base's head moved after I finished reading and before this message.** It was `f63a7b76334d` when I started (5:12 PM) and finished (5:24 PM), and `eb56c217f831` at 5:29 PM (33 commits on top, 22 files). I stayed on `f63a7b76334d`, as you said, and every line below is at it. At the new head I checked only this [run: `git diff --quiet` and `--name-only` between the two heads]: the three Security locks are the same bytes, so the 33, the 88 and the 28 are the same lists; every file this table cites is the same bytes; no file job 2 cites changed; of the 22 files the note cites one, `Security/Proofs/NetworkCertifier.lean` (job 1, line 134), and its change is two docstring lines that move no line.

**Five of the six are renames only. The first is not a rename: it is a duplicate deleted.**

`DEV/` = `verity/core/devices/certifier/lean/CertifierDevice/`, `NET/` = `verity/core/protocols/network/lean/NetworkCheck/`.

| # | rename (names of ours) | `main` | base | verdict | what else changed |
|---|---|---|---|---|---|
| 1 | `CertifierDevice.Certificate` → `NetworkCheck.Check.Certificate` (7) | `DEV/Issuer.lean` 50–57 | `NET/Check.lean` 106–113 | **A second definition deleted, not a rename.** `main` has both structures, same six fields, order and types; the check's is also at `NET/Check.lean` 106–113 on `main`. | The device's issuer now imports the verifier's check: `DEV/Issuer.lean` 5 and 50, `DEV/IssuerDifftest.lean` 32, and the device's own lock (`verity/core/devices/certifier/lean/lean-audit.json` 66). The lakefile already required `network_check`. No field, type or refusal changed; `CertifiedRecord.verify` is the same text (`main` 214, base 207). In the locks `main`'s 7 names land on 7 names `main` also records: the base holds the check's 8 (with `signs`), each with `main`'s hash. |
| 2 | `CertifierDevice.Link` → `CertifierDevice.Arrivals` (13) | `DEV/Online.lean` 59–66, 68–126 | same file, same lines | **Rename only.** Nine lines differ, each by the name. | Nothing. `Prover.link` keeps its name (136). The short name is now also the spec's `NetTiming.Arrivals` (`Security/Definitions/NetworkCertifier/Warden.lean` 55), another thing; no file names both. |
| 3 | `CertifierDevice.Record` → `CertifierDevice.BlockGrid` (5) | `DEV/Certifier.lean` 38–43 | same file, same lines | **Rename only.** Four fields the same; 26 Lean files, the counts balance in each. | Python's `grid.Record` is `grid.BlockGrid` too (`verity/core/protocols/certifier/grid.py` 110). |
| 4 | `NetTiming.Program` → `NetTiming.EgressPrediction` (2) | `Security/Definitions/NetworkCertifier/Run.lean` 98–101 | same file, same lines | **Rename only**, for the two names. One field, same type; four lines differ, each by the name. | What takes one changed for its own reasons (`constantRate`'s advice type: `Schedule.lean` `main` 111, base 67), which is job 1's six restated guarantees. |
| 5 | `NetworkCheck.Check.Link` → `NetworkCheck.Check.LinkOpening` (3) | `NET/Check.lean` 135–138 | same file, same lines | **Rename only.** The file's whole diff is five lines, each by the name. | Nothing. `Evidence.links` keeps its name; JSON keys the same; no refusal changed. |
| 6 | `NetworkCheck.Period.Layout` → `NetworkCheck.Period.BucketSchedule` (3) | `NET/Period.lean` 328–333 | `NET/Period.lean` 347–352 | **Rename only.** Four `Nat` fields and `deriving BEq` the same. | Nothing. `Rows.layout` keeps its name; same JSON keys; same refusal `not the schedule's rows` (`main` 923–925, base 980–982). |

7 + 13 + 5 + 2 + 3 + 3 = 33.

**How I checked beyond reading both sides** [run]

- The side's commits, with a script that applies the rename to each commit's removed lines and compares with its added lines: `ce4ce58e3f` (row 2), `42f58cd365` (row 5), `4d32abba11` (row 6) are the rename and nothing else; `13d7eba072` (row 3, 63 files) is the rename plus backticks in one comment and two vocabulary-list entries deleted; `9aa1e7f36b` (row 1) deletes the structure and adds the import.
- Lock hashes: all 33 differ, as a renamed definition's must (the hash is of its text, names included), so they neither confirm nor refute. The check's `Certificate` has `main`'s hashes on the base.
- The machine check for "rename only" is `check.py --update --moved`, which needs the build. Not run.

**Three things lean may want to know**

1. `NetTiming.Program` was renamed, un-renamed and renamed. `44a8ac77c4` (28 Oct) renamed it and is an ancestor of both sides; `e2271ef33e` (5 Nov, on `main`) put `Program` back; `78cf95d3bd` (10 Nov, base) restored `EgressPrediction`. It is also the one of the six that is not in the lock run's own map (`KEPT-5.json`): ASSEMBLY added it by hand.
2. `tests/test_lean_vocabulary.py` on the base still lists five of the old names as known homonyms (536, 810, 812, 830, 834); the scan reports each "not declared". Stale lines there do not fail. None of the new names is a hit of the scan. I ran the scan's listing on a `/tmp` export of the base, which builds nothing.
3. Not behind any of the six, found on the way: five refusal reasons of the period check's scan cite `NetTiming.Layout.Fits` on `main` (`NET/Period.lean` 679–726) and `NetTiming.BoundaryGates.Fits` on the base (734–781), otherwise equal. The structure is `BoundaryGates` on both sides, so `main`'s reasons name a declaration it does not have.

**A correction to job 2's "least sure"**: a `reads` hash moves only on a definition whose own text changes, not on its callers (`Prover.arrive` moved, `Shaper.arrive`, which calls it, did not). So under the refusal alone I expect `auditCheck` and `auditChecks` to move in the two locks and nothing else in `reads` [argued]. Note H8.

**Least sure**: "the same text" is my `diff`, not Lean's. Row 3 I checked by the structure, the commit and per-file counts, not all 26 files line by line. For row 1 I read the issuer and its difftest, not every device proof that mentions a certificate.

Done with the three jobs. Nothing is left only on this VM.

