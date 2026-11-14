[2026-11-14T06:23Z] 4c6190314219
RECORD (TRN-158), Fri 13 Nov, 10:23 PM PST: the design. I've built nothing and wait for your answer.

**On the rehearsal's record the command prints a share of 1, because the record's PoUW verdict is `missing` and not an accept. The design reads the share from the record's verdict and the run's policy, and calls the case script and the compiled check through their command lines. Two things are not in a served run's record today, and I need your answer on two choices (item 7).**

The brief and the issue agree. My branch is `cursor/inference-only-served-record-b069`, at #9502's head `b90c0adf1db9`, with no commits yet.

## 1. The share of work without credit, read from a record

The fields, all from the served run's `out/` (READ: the rehearsal's run record in the store, which is the run record of `r20261112-101803-fbc8`):

- `record/verdicts.json` (`verity/all-protocols-window/v1`): `run` (the run's id, which the print names) and `protocols.work`. On an accept, that entry is `served_verdicts.row_of`'s row: `verdict: "accept"` and `quantity.work.credited` (READ: `benchmarks/one_stage/served_verdicts.py` `row_of` and `pearlplus_of_record`).
- `inference-only-policy.json`: `class.slots` and `n_inputs.work.theta_macs`, which is N, one block's work in the policy's unit (linears + 2 × (attention + LM head)).

The formula:

- C = `protocols.work.quantity.work.credited`. These are linears' MACs, already net of the ε share at δ = 1/100 and of ρ: `credited = ⌊(1 − ρ)·max(macs − wrong, 0)⌋` (READ: `PouwPearlPlus/Credit.lean`).
- W = `class.slots` × N, the window's work: one block is one slot's session (READ: the record check's `blocks: 4` on this run).
- **s = 1 − C / W**, clipped to [0, 1]. Where `protocols.work.verdict` is not `accept`, s = 1, and the print gives that entry's `why` or `reason`.
- Attention and the LM head are no PoUW scheme's work, so they are never credited. `served_shares.py` reads the credit the same way. The command also prints PoUW's `macs` beside slots × linears and refuses a record where they differ: on the rehearsal both are 108,984,795,136 (RAN: arithmetic, 4 × 28 positions × 973,078,528).

**On the rehearsal:** `protocols.work.verdict` is `"missing"`, and its `why` is "development run, credits nothing: the window's registration or its audit's summary can't be read (FileNotFoundError: …/pearlplus/wK/tie/window/registration.json)". So s = 1 (READ: `out/record/verdicts.json`). Its development piece did accept (credited 106,838,378,774). Had that been of record, s would be 1 − 106,838,378,774 / 113,400,348,672 = 5.79% (INFERRED: arithmetic). The command does not use it.

**Missing from a served record today (a question for vllm):** the record check also needs the Program (`program.json`, 184 MB), the window's circuit record (`circuit.json`, 27 MB), Pearl+'s W registration and the window registration R (323 MB). None of these is in the run's `out/`; they sit in the serve directory on the node. KEY copied them by hand (READ: the run record and inputs of `r20261114-034525-11dd`).

- Its Program and circuit record are in the store only as that run's evicted files (`art:2b61d976…`).
- **R is in no store artifact.** Custody trimmed it (READ: both runs' `meta.trimmed`), so it exists only in run directories on `vy-nebius-cpu-7`, under the 7-day retention, and in my VM's fetch.

The ask for vllm: the served run writes these four files into `out/` (or `--declared-output`) under fixed names. Until it does, the command takes them as options (item 2). I'd also preserve R now with `research data put --preserve`; I'll do it on your word.

## 2. How the benchmark calls the compiled check

- **The command:** `verity bench inference_only --served OUT --program P --circuit C --log L0 L1 L2 [--uncredited-share S]`. OUT is the run's `out/`, read for `record/verdicts.json`, `inference-only-policy.json`, `inference-only-limits.json` and `block-counts-*.json`.
- **The case:** `served_record_case.py --case-only` writes it, run as a subprocess.
- **The check:** `.lake/build/bin/inference-only-record-check CASE`, run as a subprocess. The benchmark reads back its two lines, `[S s] check: JSON` and `[S s] count: JSON`, as #9474's test parses them. Nothing of the verifier is imported.
- **Boundaries:** `benchmarks` sits after `verity/protocols/policy/inference_only` in `tests/import_order.toml`, and that place is not marked trusted, so even a load by file would pass. I use the command line anyway (READ).
- **The build:** `lake build inference-only-record-check` in the policy's package, as `built.lean_count` builds today. On a CPU node it is `research run --cwd clone`. KEY's build took 25 s with the node's Lake cache (RAN: the log of `r20261114-034525-11dd`, built 03:48:01 to 03:48:26Z). A cold build's time is not known; the draft's first node run will show it.
- **The test without the binary:** the tree's convention is `@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")` with `verity_difftest.build` (READ: `test_nci_record_check.py`). The suite joins `AFTER_LEAN` in `suites.py`, lists the lakefile and the scripts in its inputs, and lists the test file under `untraced`.

## 3. What it prints

Today's sections stay. A served record adds a section after the stated policy. It prints only the public verdicts (accept, or the refusal's code); the checkDetail fields are private (README: "never published"). Values in `<…>` are computed at run time.

**Accept with a share** (an accepting record; the numbers are the rehearsal's development credit, for shape):
```
FROM A SERVED RECORD: run r20261112-101803-fbc8, unsloth/Llama-3.2-1B, class 4,27,2 (record SHA-512 d24ddfea770b44a4…, a log of 3 records)
  check     inference-only-record-check: the rate rule (Record.checkDetail) accepts; the count (RecordCount.checkDetail) accepts, against c = 11,044,318,495,127, k = 337,786,912, n = 31,457,280
  cost      in the clear, at the served size: <wall> s wall (the command's own lines: read <s> s, rate rule <s> s, count <s> s), <cpu> s CPU, <MiB> MiB peak; the case written in <s> s, <MiB> MiB peak, <bytes> bytes
  PoUW      verity/pouw/pearl-plus/v1, of record: accepts at δ = 1/100, 106,838,378,774 MACs credited (ε = 14374133/1000000000, ρ = 1/400) of the window's 108,984,795,136 linears' MACs
  share     s = 1 − C / W = 1 − 106,838,378,774 / 113,400,348,672 = 5.787%; W = 4 blocks × 28,350,087,168 MACs, a block's work in the policy's unit; attention and the LM head are no PoUW scheme's work, so never credited
THE RECORD'S CLASS UNDER THE VALUE TIE, AT THE RECORD'S SHARE (run r20261112-101803-fbc8): every figure rests on that run's PoUW verdict
  k         145,952 = the record's k 337,786,912 − 337,641,472 bits of the linears' outputs + 512 bits of salt; c, n and work the record's policy's
  [TRN-86's tied section as on main (tied_lines): one block; one execution phase; a week, a month and a year on one datacenter and the world; each reading with its limit at s and the pooled total in gates; the proved, premise and scaled lines, where "premise" says s comes from the run's verdict and "scaled" says the record is one window]
FROM THE RECORD: <value>, the smaller of one block's two limits at s (every word tied; the uncredited words in k)
```
**Accept with no PoUW accept** (the rehearsal as it is):
```
  PoUW      missing: development run, credits nothing: the window's registration or its audit's summary can't be read (FileNotFoundError: …)
  share     s = 1: the record holds no PoUW accept, so none of the window's work is credited
  [the tied section at s = 1: every limit and every period's figure is 1]
FROM THE RECORD: 1, no slowdown proved: s = 1 (the record holds no PoUW accept)
```
**A refusal:**
```
  check     inference-only-record-check: the rate rule refuses: <code>; the count refuses: <failure or error code> (<which of c, k, n>)
FROM THE RECORD: 1, no slowdown proved: the check refused the record, so no block is shown within the policy; the tied section is not printed
```
`--uncredited-share` stays as the existing section under its "rests on no PoUW verdict" label, printed after the record's.

**The dashboard row:** the value is written as `training/inference-only-slowdown` with `mark="measured"`, its inputs naming the run, together with one value each for week, month and year and for datacenter and world (`kit.PERIODS` and `kit.OVER`). `metrics.toml`'s title, unit and command change to "from a served run's record". The producer is `report.main`'s `kit.writer(...).measurement(METRIC, ...)`, today marked `modeled` from `served.toml` (READ).

## 4. Time and peak memory of the check in the clear

This follows `built.lean_count`. Each child (the case script and the check) runs under `subprocess.Popen` with `os.wait4`:

- wall time from an injected `clock` (default `time.perf_counter`);
- CPU from `ru_utime + ru_stime`;
- peak memory from `ru_maxrss`.

The check's own `[S s]` lines split read, rate rule and count. The test passes a counter as `clock`, so no test reads the wall clock.

The expected cost at the served size: 2,174 s and 11,428 MiB peak, with 2 cases running at once on one node (RAN, by KEY: `r20261114-034525-11dd`, `/usr/bin/time`). The case script with its reference took 697 s and 7.9 GiB; with `--case-only` it is not known, and the first node run will show it. The node job asks `--mem-gb 24 --timeout 120m`.

## 5. The made record for the test

- **Where it comes from:** the record difftest's served shape, `record_difftest_vectors.shaped()["2x3 registered"]` (2 slots × 3 positions, a few KB; READ). It is written as a served run's files the way #9474's test `_files` writes them.
- **Added to it:** a `verdicts.json` built with `served_verdicts.row_of` on a made Pearl+ answer, and an `inference-only-policy.json`.
- **Its variants:** PoUW accepts; PoUW is missing; and the count refuses (`--unregister smallest`, via the script's option).
- **The generator:** a helper in `benchmarks/inference_only/tests/` writes all of it into `tmp_path`, so no fixture bytes are committed.
- **The tests:** one runs the compiled check (skipped without `lake`). The others inject the check's runner and the clock, and check the share, the three blocks and the written measurement.
- **What it can't show:** the served size's cost; a real registration chain (Pearl+'s W, a 323 MB R); a real `registered.formats`; a real PoUW credit and its agreement with the class's linears; and a policy whose work comes from a serve.

## 6. What I won't do in this draft

- **No change to the verifier's side** (the record check, the case script, any Lean): it isn't this issue, and it stays on the soundness path.
- **No credit from sampled proofs for the in-circuit work:** the share is "no PoUW check credits", as the tied section defines it. Counting it would give 1.89% on the same numbers.
- **Not reading the bound on untied words from the record:** it stays `tied.toml`'s stated bound (CMP-380). No record holds it.
- **No fetching from the store inside the command:** it takes paths. Changing the serve's `out/` is vllm's.
- **No figure from `main`:** no run from `main` exists yet; I rerun the command on your word.
- **No crossing of Monday's rename** except once, on your word.

## 7. For your answer

- (a) Is the share PoUW-only, as in item 1, or should sampled proofs' accept credit the in-circuit work?
- (b) Is the headline value the smaller of one block's two limits at s?
- (c) May the print name which of c, k and n a refused count fails? The README lists the counts and which block as private, and doesn't list the budget.
- (d) Should I preserve R in the store now?
- (e) The ask to vllm in item 1.

