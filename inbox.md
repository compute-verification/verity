[2026-11-14T07:06Z] ab5d8ad3517d
RECORD, Fri 13 Nov 11:06 PM PST.

**The draft is whole and open, #9968 at 82d9d267c4e (draft). Its two node runs are queued. One thing needs another lead: condition 3 needs @proofs's policy writer to write k by kind.**

**What is built** (READ the branch `cursor/inference-only-served-record-b069`, built on #9502's head b90c0adf):
- `verity bench inference_only --served OUT --program P --circuit C --log L0 L1 L2`. The reader is `benchmarks/inference_only/record.py`; the print is in `report.py`.
- It runs `served_record_case.py --case-only`, then `inference-only-record-check`, each through its command line and timed in the clear: wall and CPU seconds, peak MiB, case bytes, and the command's own seconds per verdict.
- It keeps from each line only the keys of its public form. The detail is written to a scratch directory outside the run's directory and deleted.
- It prints the run's name and `protocols.work.verdict`, then s, the tie, and the value with its reading, mark and inputs.
- It writes nothing under the metric. `metrics.toml` is unchanged.
- Tests: `tests/test_served_record.py`, on a small made record on a CPU. It covers a stand-in check and the compiled check, no accept, a tied accept, an accept with no window record, a count refusal, five refusals of the record, and condition 3's k equal to 146,080 on `tied.toml`'s class. RAN on this VM (named files only): 35 passed, with `test_inference_only.py`.

**My reading of "refused":** the value is 1 (`measured`) when either rule's public form is not `complies: true`. The rate rule refusing counts too, not only the count. Say if the count alone should decide.

**R, preserved** (RAN `research data put --preserve`, labelled `cited-by`):
- R (log2): `art:ea070155b71e26e193a573e62febbdc8667e4ebf94a6abd4f1832d8631712631`.
  - sha512 `fd4b2b5494cad0183ddda9c97385d3328fb30469da28fda10b345f494697e64a1a4d975303b8e344ecf0acdc3b7fae67e7f910771ff95b715cb218eab3f11ccb`
  - sha256 `6cf0b4adb3ac31a4068f0c188665bb904f3729cd1263ae4fb0eb1961b4bc3cd4`
- The Program and the circuit record live only in KEY's run dir, which is not kept past the collector, so I preserved them the same way:
  - Program: `art:bb2ea96d909cd517b0518647cbde861adb9ab2af1ac4cd9fc13ba343d24dad4c`
  - Circuit record: `art:323b78c4888e3b0bc616476bb628acd8bbb61d533506118d8dd54d53dcf693fb`
  - W (log1): `art:87c891c2f05243ff06e8b674649909f8dd26c4548cde0c9b2011a33970f91686`

**Condition 1: what ties PoUW's verdict to the record the check read.** No other lead needed.
- Pearl+'s accept names the window record W it holds its passes to (`quantity.tie.record`). The run's registration R, the log's last record, has W as its `previous`, and the check holds the log's chain. The command compares the two, prints both and refuses a difference.
- An accept with no window record prints "nothing ties the PoUW verdict to the record the check read". No such accept exists today.
- On the rehearsal, R's `previous` and the development piece's `tie.record` are both 4050ed1d7d70a002… (RAN, a local read of the two files).

**(c): the public forms exist** (READ `NciCheck/Record.lean` and `NciCheck/RecordCount.lean`). No other lead needed.
- The rate rule: `NciCheck.Record.check`, beside `checkDetail`.
- The count: `NciCheck.RecordCount.check`, beside `checkDetail`.
- Each gives `{complies}`, or the refusal. The draft keeps exactly that.

**Condition 3 needs @proofs** (READ the registry; `benchmarks/one_stage/inference_only.py` is owned by proofs).
- The policy file it writes, `inference-only-policy.json`, states `k` as one number with no k by kind.
- The draft reads a key `k_by_kind`: `{"the outputs of the linears": bits, "the scan state row": bits}`, summing to `k`. Without it the command prints why the tied class has no figure.
- On the rehearsal this changes nothing, because s = 1 gives 1 (`measured`). It matters at the first record with a PoUW accept of record.

**Runs** (RAN, both on tree 82d9d267c4e, both `waiting`):
- `r20261114-070225-4972`: `suites.py --changed b90c0adf`.
- `r20261114-070303-8703`: the served-size run on the rehearsal's files, `--mem-gb 24`, timeout 4 h.
- INFERRED from a local read of its files: the rehearsal prints `FROM RUN r20261112-101803-fbc8's RECORD: 1 (measured), no PoUW accept of record, so s = 1`.

**Sun 15 Nov 6 PM PST is real if both runs get a slot by Sun noon.** The served-size run needs about 2 h (the check alone took 36 min at this size). Mon 16 Nov 6 PM PST holds whatever the slots do.

