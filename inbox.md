[2026-11-12T08:25Z] dc6bdb6ce776
grade to training, Thu 12 Nov 08:25Z. #9029 at its new head: the carrier, #8910 and #9187's head are merged, the gate passes, the runs at the head are read, the body is rewritten. Not marked, still a draft.

**Head.** `cursor/training-trace-job-dtype-0cd4` at `cfb7a3ae5c3862685d27e3ba2b40c975bff2bf91`, pushed with a plain `git push`. No rebase, no force push, main itself never merged. First parents since `435bfa4688c7`:

- `0078a5d0f23` merge of #9003's head `034f76f61368`
- `1a4e836f646`, `34a57c08736`, `ad11d4e7d86`: three mends, one cause each
- `01e8bd81627`: 4.23.0's own, the grader's test (your 05:23Z word)
- `64c3156dd31` merge of #8910's head `4e221ac275bc` (your 06:36Z word)
- `d47f246d956`: 4.23.0's own, `qat`'s cast on every device
- `cfb7a3ae5c3` merge of #9187's head `866e44a846e9` (your 07:43Z word)

**`git show --remerge-diff 0078a5d0f23`: eight files, no other (ran).** Each keeps both sides. None is plain in the sense of one side taken; what each side did and what I kept:

1. `training_slowdown/__init__.py`. Mine: `VERSION = "4.23.0"`. Carrier: 4.22.0 and main's names after it (`STOPPED`, `TOLD`, `SCHEMA`, `BUDGET_SLACK`, `NOTHING`, `stop_record`, `told`). Kept: 4.23.0, then main's names.
2. `capture/trace.py`. Mine: `_math_attention`, and `_trace` records under `_marked_backward(), _math_attention()`. Carrier: `_DeepCopy`, `_no_deep_copy`, and `_trace` records under `_marked_backward(), _no_deep_copy(), sdpa_kernel([SDPBackend.MATH])`. Kept: both functions and `with _marked_backward(), _no_deep_copy(), _math_attention():` (`_math_attention` holds the math kernel, so the carrier's `sdpa_kernel` line is inside it and not written twice).
3. `library.py`. Mine: an import of `capture.products`, for `priced_in`. Carrier: `traced` imported in place of `run`. Kept: both imports.
4. `README.md`, two places. Mine: the library's table (the job's program on a CPU, `priced in`) and the routes' cast on every device. Carrier: `traced.traced_step`, and the route contract's `LATER` and `EITHER`. Kept: each sentence with both.
5. `tests/test_entries.py`. Mine: the docstring's paragraph on the float32 forward. Carrier: the docstring's sentence on an average's reference. Kept: both.
6. `tests/test_grade.py`. Mine: `trace()`'s docstring names the stored records' capture. Carrier: `traced.trace_step` in the same docstring. Kept: my text with the carrier's name.
7. `tests/test_routes.py`. Mine: the test that a route's forward is cast on a CPU. Carrier: a schedule test at the same place. Kept: both tests.
8. `tests/test_run.py`. Mine: `import contextlib`; the width test traces the route as stepped, cast or in float32. Carrier: `import copy`; `trace_step` imported from `traced`. Kept: both imports, and the carrier's `trace_step` called on my `stepped`.

No resolution changes what a record measures. `git show --remerge-diff 64c3156dd31` is empty (ran): #8910 merged with no conflict, as you read it. `git show --remerge-diff cfb7a3ae5c3` is empty too (ran).

**Your 07:43Z step.** #9187 is not on main (OPEN at `866e44a846e9`; main at `bcaf58a45d53`), so I merged #9187's head, not main. It is one commit above the carrier's head, so the merge brings that commit alone: the tree differs from `d47f246d956` in `work.he_of_rows`' docstring only (3 lines in, 2 out). Then, on the VM (RAN):

- `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD` at `cfb7a3ae5c3`: "ratchet gate bcaf58a45d53..cfb7a3ae5c38: 1 change(s), 0 failed". Its last line: "034f76f61368..cfb7a3ae5c38 (cfb7a3ae5c38): passed: ruff finds nothing new in the 27 Python file(s) it changes; no direct environment read is new in the 27 Python file(s) it changes; no silent except handler is new in the 27 Python file(s) it changes; no lane-text site is new in the 32 file(s) it changes; no retired-term site is new in the 32 file(s) it changes".
- It names nothing, so I reworded nothing and added no list entry. My added lines over the carrier's head hold `cap` or `caps` as a whole word nowhere (ran: grep of the diff, 0).
- That gate is my tree's copy, which is main's at `be9a3d566905`. So I also ran main's own copy as the queue would: a scratch merge of `cfb7a3ae5c3` into main `bcaf58a45d53` (`4a5f891c220`, unpushed, merges clean), then the gate from that tree: "ratchet gate bcaf58a45d53..4a5f891c220d: 1 change(s), 0 failed", the same five clauses. Both judge #8910's files with mine, since #8910 is not on main yet.

**What merged clean and did not work together, one commit a cause.**

- `1a4e836f646`: the carrier's new test of the counter's split called `products.priced(r)`; 4.23.0's `priced` takes the trace's capture.
- `34a57c08736`: the carrier's three-rates test holds torch's AdamW in float32 against a route 4.23.0 casts ("AssertionError: embed", r20261112-045843-dd60). The route's forward runs in float32 there.
- `ad11d4e7d86`: the `width` rule goes, as you allowed after I told you (all 29 equal with no `width` line while the rule was still in: r20261112-050442-ed69, r20261112-050516-d2f6, r20261112-050543-3c6a at `1a4e836f646`). `width` becomes `uses`, which returns differences only.
- `d47f246d956`: `routes/qat.py`'s autocast loses `enabled=`; a new test holds that `qat`'s forward is under bf16 autocast on a CPU; #8910's hand-written `qat` step test runs the route's forward in float32.

**One thing in `d47f246d956` no failure forced.** I gave #8910's hand-written `qat` step test the float32 switch before running it under the cast. A control at a scratch commit without the switch passes it (r20261112-070234-e939 at `a28d7a35df7`, unpushed: "2 passed, 17 deselected", last line "pytest exit 0"). I kept the switch because the test holds the step's rule against a float32 reference, as `tests/test_entries.py` does. Say the word and I drop it in a commit of its own.

**Runs, each read whole by me (RAN and READ).** At the head `cfb7a3ae5c3`:

| run | what | last line |
| --- | --- | --- |
| r20261112-075945-6fcf | the three commands, with `VERITY_GRADER_SLOWDOWN_CODE` set: `suites.py --changed be9a3d566905 --fresh` ("4 suites: 4 passed (0 cached), 0 failed": training_slowdown 561 passed, grader 31 passed and none skipped, repository 311, verity-graph 234; "exit 0"), `pins --stale` ("exit 0"), ruff ("All checks passed!") | "ruff exit 0" |
| r20261112-080017-27cc | the grader's suite with the harness: 31 passed | "grader suite with the harness exit 0" |
| r20261112-080040-0e2e | port check, the three `qat` entries: "3 entries: 3 equal, 0 differ, 0 refused.", one expected difference each (`norm`), no `width` line | "port check exit 0" |
| r20261112-080107-96ad | port check, 22 entries: "22 entries: 22 equal, 0 differ, 0 refused." | "port check exit 0" |

At `d47f246d956`, the head before #9187's docstring:

| run | what | last line |
| --- | --- | --- |
| r20261112-064139-5dca | the three commands with the variable: 561, 31, 311, 234 passed; every exit 0 | "ruff exit 0" |
| r20261112-064211-97f0 | the grader's suite with the harness: 31 passed | "grader suite with the harness exit 0" |
| r20261112-064244-4fcf | port check, the three `qat` entries: 3 equal, `norm` each, no `width` line | "port check exit 0" |
| r20261112-064310-a862 | port check, 22 entries: 22 equal | "port check exit 0" |
| r20261112-064335-9570 | port check, the six l16 entries: "6 entries: 6 equal, 0 differ, 0 refused." | "port check exit 0" |
| r20261112-071854-480e | port check, the xl entry: "1 entry: 1 equal, 0 differ, 0 refused.", `norm`, no `width` line. It waited 25 minutes for its slot. | "port check exit 0" |
| r20261112-072729-5f13 | the stored point graded from its records, and its trace over its own circuit, by `d47f246d956` | "f99e5a32…  own.txt" after "grade rc 0", "own rc 0" |
| r20261112-072720-e6dd (at #8910's head `4e221ac275bc`) | the same by #8910's head, to compare | "d92d4873…  own.txt" after "grade rc 0", "own rc 0" |

The six l16 entries and the xl entry I did not run again at `cfb7a3ae5c3`: the two heads differ in a docstring, and the two port checks I did run again (0e2e, 96ad) are line for line the earlier ones but the head's, node's and thread-count lines. Say the word and I run them at the head.

The `qat` entries against #8910's own run r20261112-060031-77ea: only the head and node lines, the three `width` lines and E's size after two steps differ (top4 2,584,507 → 2,584,527 B, top3 1,165,698 → 1,165,622 B, top2 1,067,711 → 1,067,499 B). The port check does not compare E's size. They did not differ for any other reason, so I added no rule.

The stored point: the printed grade is byte for byte the same by both heads (sha256 `b12822dc1e3f…`, S 169,479 = p 1.311 × ρ 129,298); `slowdown.json` differs in the version alone, two fields; over its own circuit the trace gives 92,859.86 by #8910's head and "unmeasured: 34 of the traced step's matrix products are priced in another dtype…" by `d47f246d956`. I ran these two unasked, to hold the body's section on stored records at the new base. CPU, 8 GB, read-only.

Runs at earlier heads (`1a4e836f646`, `01e8bd81627`, the control at `435bfa4688c7`, the carrier) are in the body's section 9 with what each showed. Every launch and read, stamped: `/cursor/stores/self/merge9003/launches.txt`.

**Which O is the job's: what I put in the body (section 4), marked.**

- The GPU kernel's width: INFERRED, bf16 in and out. No GPU run, and I read nothing of what the fused kernel holds inside.
- The capture before 4.23.0 traced attention float32 (RAN, r20261112-053627-c5bd, the carrier's code on 29 traces). 4.23.0's traces it bf16 (RAN, r20261112-063335-2666).
- The grader's circuit wires a bf16 tiny step's attention at 32 bits (RAN: the test that asserts it passes, r20261112-080017-27cc; READ: `grader/tests/test_grader.py`).
- What the calculator was handed did NOT change in width. Before 4.23.0 `priced(row)` restated a float32 attention row at bf16 by its default `gemm_dtype="bfloat16"`: 1,480.125 gates a MAC, 16-bit outputs. The other reading, `rows(g, gemm_dtype=None)`, 32-bit, went to the trace's own |C| and to the grader's test, which is why the test agreed until now. 4.23.0 has the one reading, at 16.
- Through O no slowdown moves: M holds no attention product (READ, `work.he_of_rows`). RAN on 29 entries (2666): bound as traced and with `bmm` rows at float32; N, O, b_y, β_w, β_x, F, M and what binds are equal at all 29.
- Through |C| it moves, and that is the price fix itself: with attention at float32 the slowdown is 0.432 to 0.591 of as-traced, exactly the |C| ratio. The carrier's code on its own CPU traces gives 0.119 to 0.121 of this branch's, all |C| (float32 circuit under bf16-priced work).
- No entry of record's slowdown moves: the 32 held runs of record are GPU records with no `trace` (RAN, r20261112-064615-38ab and r20261112-064629-fa97), so this capture prices nothing of theirs.

I changed no line of the grader's circuit or of the capture for it.

**For onecalc's #9039** (head `99271e981455`, read now; a trial merge into `cfb7a3ae5c3` in a scratch worktree, aborted, nothing pushed). Plainly:

- `capture/trace.py`: no name or signature of 4.23.0's changed. `_math_attention` is as it was. Added by the merges: `DEEP_COPY`, `_DeepCopy`, `_no_deep_copy`, the refusal kind `deep_copy`; and from #8910 `_fake_copy` now returns four things (`fake_model, fake_opts, held, originals`), with `_taken` and `_refuse_originals` beside it. #9039 calls none of those.
- `library.py`: it imports `traced` in place of `run` and calls `traced.traced_step`, same arguments. `run.traced_step` no longer exists; `run.trace_step` still resolves. #9039's own lines call only `run.trace_step`, so nothing of its breaks on the name.
- The op log: THIS ONE BITES. #9039 writes a seven-field row, the dtype last. Three readers that come in with the carrier take six and merge with no conflict: `run.py`'s sum of two segments' structures (`for *sig, count, macs in mine["log"] + its["log"]`: with seven fields it adds MACs into count and the dtype into MACs), `of_record.per_row` (`for phase, kind, op, shapes, count, macs in struct["log"]`: raises), and `tests/test_of_record.py`'s made-up logs. The first is silent.
- The trial merge conflicts in seven files, one hunk each: `README.md`, `tests/test_flops.py`, `tests/test_grade.py`, `__init__.py` (4.24.0 against 4.23.0 and main's names), `capture/trace.py` (`CaptureRefused`'s docstring: `deep_copy` beside `unpriced_dtype`, text only), `flops.py` (the carrier's `structure_id(struct)` against #9039's row order and inline sha256), `run.py` (the module docstring).

**The body** is rewritten with `research queue edit 9029 --body-file`: it says it stands on #8910 and through it on #9003, and holds #9187's head; it has the width section for your ruling (section 4), the three merges and the gate (section 6), the mends, and every run (section 9). Read back from GitHub: draft, head `cfb7a3ae5c3`, no label. One correction in it: the old body said of the grader's suite "28 passed, 3 skipped (the GPU ones)". They are the three tests that import the harness, skipped because the variable was not set.

**Not done, and not mine to do.** Not marked; still a draft. No GPU run. No stored record changed, nothing published, nothing written to Linear or Slack. The six l16 entries' and the xl entry's port checks are of `d47f246d956`, not of the head (above). main itself is not merged; when #9187 is on main and you want main in, that is one more merge and one more round of the runs. No run of mine is out or queued.

