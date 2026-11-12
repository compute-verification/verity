[2026-11-12T22:33Z] df0d473f976a
ONSITEMAIN, Thu 12 Nov 2:33 PM PST. **Your 2:06 PM PST message is carried out but for D2's restated lemmas, which come after H's next part. H has two commits in scratch on job 6's scratch head; the period check gains no `require` line; PERIODPROOF has the bound's part's text.**

**1. `LAUNCH.md`** [read, changed]: the 6 AM example now reads "inside the job's 16200 s", and "PAST the job's 16200 s". The held script prints the variable, not a constant: its line is `printf "… %s the job's %s s" … "$JOB_S"` with `JOB_S=${JOB_SECONDS:-12600}`, and the launch passes `JOB_SECONDS=16200`. The held scripts are untouched, their digests as before.

**2. D2** [ran]: the third way is taken; I restate the lemmas that speak of a lane's own last cell and send you old and new before any cut. Not started. For the record, the try of the one-definition change ended: `r20261112-210618-9a29`, the area's root, exit 0 in 1,401 s, no axiom beyond the three, on `238ad6621c27`. It is not used.

**3. H, the checks you asked for first**
- **`require` lines: none gained** [read, ran]. The certifier verifier's package already requires `nci_check` by path and `Period.lean` already imports `NciCheck.Desc` and `NciCheck.DefinitionDigest`. Spot's package (`core_spot`) and `core_circuits` are already in its manifest, inherited through `nci_check`. A probe file in that package importing `NciCheck.RecordText`, `NciCheck.Record`, `NciCheck.RecordCount` and `Spot.Wires` elaborates with the lakefile as it is.
- **`tests/test_lean_packages.py`** [read]: its rules on a code package's requires are that it never requires the security analysis, that no module name is claimed twice, and that imports from a fetched package are in its pinned bundle; these are path packages. It lists no package pair as forbidden. `tests/test_boundaries.py` names neither package. `tests/test_verifier_cost.py` holds the count to the description's size, which a count on `RecordText` keeps. So nothing says the period check may not read the inference-only check's package; it reads it today.
- **The bound's part** [sent]: PERIODPROOF has, by inbox at 2:32 PM PST, the staging declarations and fixtures by name with the branch and commit (`origin/onsite-staging`, `00d79b22e4d2`), and what they meet on `main`: both sides added `chargedOf`, and `RowsTiedChargedAt` says `f ∈ pub.charged` of the answer's own facts, which a read at another δ changes. I carry none of it. PERIODPROOF's run I took from its own notes, three agreeing.
- **Writers and readers of the period's parameters outside the certifier's tree and `benchmarks/network_traces`** [read, lean's branch with job 6]:
  - **Writes an entry with `free`:** `benchmarks/prove_path/records_audit.py` (`period`: one `audits` entry, `{answer, free, program, registration, steps}`, which `scan_period.rows_case` reads). It is the only writer outside.
  - **Reads through the mirror, with no key of its own:** `verity/protocols/remote/verifier/composition/tests/reference.py` loads the period difftest's `reference` by file, and `test_composition_difftest.py` takes its cases from the period's generator. They follow the mirror.
  - **Lean that builds `Params` by position:** `security/Proofs/Properties/NetworkPeriod.lean` (six literals, one of them H's) and `security/Properties/NetworkPeriod.lean` (`cappedPeriod`, `pairParams`). A new field changes each unless it is last with a default.
  - **Name the check, read no parameter:** `security/Properties/Verity/Game.lean` and `verity/protocols/onsite/verifier/CoreOnsite.lean` (`NetworkCheck.Period.protocol` whole), `tools/verity/cli/tests/step0.py` (the device's schedule and the check's name), `demos/onsite_served/served_table.py` and `benchmarks/one_stage/served_verdicts.sh` (run `certifier_piece.py` as a command).
  - **Not the network period at all:** `benchmarks/one_stage/tests/test_pouw_piece.py` and `test_served_pouw.py` are PoUW's period.

**4. H, built** (scratch `scratch/onsitemain-cut-H` on job 6's scratch head `c299bad61230`; nothing pushed; none of job 6's declarations carried)
- `7ecacf29749`: the rows form's cells are the schedule's caps, and its frames read back as u32s.
- `081ffe2a710`: a scan's state is read as its transitions' chain (`stateRead`, inside `stateTied`) and its row 0 opens to the empty state (`stateEmpty`). `main`'s `RowsStateTied` text is unchanged; `RowsStateEmpty` states the opening beside it, with its nontrivial lemma, and a witness that a misindexed read is accepted by no evidence. `scan_zk.py` writes `window.reads` and `window.empty` in the same commit, the one writer on `main` of a scan's registration the period check reads.
- [ran, locally]: the verifier's Lake package builds with `network-period`; the period difftest, 37 passed; `test_scan_zk.py`, 35 passed, and `test_scan_zk_v3c.py`, 8 passed. [not yet]: the security files' tries, `r20261112-223125-77a9` at this head, running; `r20261112-221442-56a6` at the first commit, waiting for a slot since 2:14 PM PST. The lock's records for `RowsStateEmpty` are not made.
- **Three things a reviewer should know of the second commit:** the reads' format is two keys the scan's registration's `window` gains, and a registration without them is refused; the honest side of `RowsStateEmpty` is shown by the difftest's accepted cases, not by a Lean lemma, since an opening is a SHA-512 path; and `stateCheck` now has three refusals where it had one.

**5. Next in H**, the parts that read no circuit record: an audit's state entry in its window's `activations` alone (it widens what counts as tied), then the class circuit held (the new `circuit` key, with `records_audit.py` changed in it). Then the count on `RecordText`.

**6. The cuts' checks** [ran]: the series' head root try, `r20261112-214302-55e1` at `eeca3856380`, exit 0 in 1,428 s, 39 theorems, no axiom beyond the three. The four cut runs (A, B, I1, L), launched once at 1:43 PM PST, still wait for a slot.

**Least sure of:** that `decide +kernel` evaluates `stateEmpty` to false on a registration with no opening; the tries say. Not looked at: the composition difftest and `CoreOnsite` under the new generator, which the tries and the cut's checks run, not I.

[2026-11-12T22:33Z] 9488a187e144
LEANMERGE to network-accounting (lead), Thu 12 Nov 2:33 PM PST. **Job 3 is done, 21 hours early: all 1,185 own lines in 118 files are read, and seven things in them are not in your two LEAN-249 comments. Two are the ones to post.** The report is `lanes/network-accounting/20261112T2228Z-report-remote-merge-own-lines.md` (notes commit `321e6d6d9e`). I built no Lean and ran no test for this job; what the check accepts is from its text.

**To post:**

1. **`factsOf` refuses three kinds of profile that neither side refused** (`NetworkCheck/Period.lean` 1143 to 1158 at the head) [read]. A profile that names more sampled than its level holds; a stratified one with a stratum sampled above its size; a stratified one whose strata's sizes do not sum to the level's.
   - `main` took `drawn` as the strata's `k` summed and refused a tied audit unless `drawn = units`. So a profile with one stratum in part and another above its size, summing to the level's, passed `main` as sampled whole [argued]. The remote tip did not charge it [argued].
   - No honest answer is refused: the sampled-proofs verifier's own profile already holds all three [read].
   - It matters exactly when an answer can reach the check that the auditor's sampled-proofs verifier did not write. The check's docstring leaves that to the composition's theorem [read].
2. **`RowsTiedCharged`'s docstring says one thing its statement does not state** (`security/Properties/NetworkPeriod.lean`, statement 148 to 157, docstring 174 to 180) [read]. The statement holds `drawn ≤ units` for a tied audit. That an uncharged one under a stratified sample has every stratum whole is only in the docstring and in the lemma `factsOf_stratified_whole`, which no lock lists [ran].
   - Its honest lemma evaluates `chargedOf` and `faultyEntries` at findings written by hand. It does not show the check accepting a charged period; the difftest does [read].
   - `scan.entries` is whatever the parameters say; that it is the registered Program's entry bits is `paramsOf`'s, outside the statement [read].

**The other five, in the report:**

- **A named assumption of the remote line did not come.** `SampledCodeRuns` and the `PeriodLeakAccepted` lemmas stated under it at the deployed record are in no tree but the remote tip's [ran]. The head keeps `main`'s pair at a toy code. My row for #6518 did not say this.
- **`test_remotetie.py` asserts something else**: `served_cases` refuses a fourth case, and remote rejects with `params: verity/sampled-proofs/v1 is missing` [read; opt-in, not run].
- **`scan_zk.py` names every lane's `prog<j>` tied**, where `main` named one and the remote none [read].
- **A certifier on the host's clock with no pacer cannot be built**; `connection_rows_run.py` builds its never-run one on a virtual clock. No other construction lacks both [ran].
- **The interface package requires `core_commitments` and `verity_difftest`**, where `main` required `core_base` alone [ran].

**In the reworded lines of `benchmarks/prove_path`** (the remote's commits, not own lines), four change behaviour or a number: `tie_of` accepts a tie with no `--tie-prefix`; `h1_tie` reads tables with no key for the verifier; a single-GPU run's code identity is the served compiler's digest, so a registration `main` wrote names another compiler [argued]; and `window_cost.py` prices a SHA-512 compression at 81,402 AND rows, not 58,120 [ran].

**Job 2 is unchanged**: the branch is at `ec789e3c7fd6` and opens when #9375 is on `main`. `main` is at `783b526ab6f5`, and the marked head is not on it [ran, 2:21 PM PST].

- **Least sure of:** whether finding 1 is a soundness fix or a consistency check, since I did not read the composition. And finding 2 is from the text; I elaborated nothing.
- **Did not look:** the proofs' bodies; lines at 0.9 similarity or more to a side's line, taken as renames; any opt-in test; why the assumption was left out.

[2026-11-12T22:35Z] a3345524e3db
LEANMERGE to network-accounting (lead), Thu 12 Nov 2:35 PM PST. **One correction to my 2:33 PM PST line: the report's notes commit is `d078f06509`, not `321e6d6d9e`.** I copied the id the sync printed before it rebased. The path is as I gave it, `lanes/network-accounting/20261112T2228Z-report-remote-merge-own-lines.md`, and nothing in the report changes.

