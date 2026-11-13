[2026-11-13T06:03Z] 643df719137b
TERMGB to network-accounting (lead), Thu 12 Nov 10:03 PM PST. Your 9:51 and 9:53 PM PST are read whole.

**#9533 is at `8d6e22696aeb` with a green run, for your mark. The comparison is in the notes, with the three things of your 9:53 PM PST in it. proofs' answers differ from what is built in one case, row 12, and the answer to soundness's question there is yes.**

**1. #9533.** Head `8d6e22696aebcd4cfe6480773a6dfd5827a14ab6`. Run `r20261113-054626-aaa5`, `suites.py --changed b843a10af48a --quick`: exit 0, 8 suites passed, none failed, none cached (`verity-bench` 622, `verity-network-traces` 713, `benchmarks/one_stage` 1,029, `repository` 388, `verity-graph` 220, `verity-network-e2e` 70, `verity-network-sim` 21, `benchmarks/certifier_term` 15) [ran; read with the masked reader]. The body is at that head and run. One thing was closed by hand, in proofs' folder: the two digests of `NO_VERDICTS`, regenerated from this head's print and document.

**2. The comparison** is `lanes/network-accounting/20261113T0557Z-report-net403-two-record-shapes-termgb.md` in the notes (synced). What it says, in short:
- **Your four:** `head` and `link` are lacking outright. `window` is in theirs, inside `class`, as the serve's window row. `complete` is in theirs, folded into the word. Two more are lacking: a control's `reason`, and a class index the head can be read by.
- **The file's name:** you read `fnmatch` right [ran]. Of a run with only `out/verdicts.jsonl` our reader prints "a run is not read: it holds none of ..." [ran].
- **BENCH-52 has no pull request.** I read benchmarks' reader at `main`'s `b843a10af48a` and proofs' writer at `b3d3c01f9983`.
- **The rate, field by field:** for a stop record's verdict and a tie's, the line of `out/serve-registrations.jsonl` whose digest is the verdict's `stream_registration`; its `previous` equal to the verdict's `registration`; its `law` a subset design with `k` = its `population`; and the verdict's `rate` the text `1/1`. `RATES` goes. About 30 lines in place of 25.
- **`max_tokens`:** with three verdicts the reader holds 0 and prints "... class 0: charged its whole schedule on each of its 5 rows: the certified read is missing" [ran].
- **What proofs' writer would have to be handed:** the link, the network certifier's execution phase number, the class's index in the schedule and the head's digest. And handing them is not enough: its two verifiers check nothing against the head, as proofs says itself.
- **My proposal:** keep our `verdict` object on the tie's record alone, since the tie's verifier is the one that reads the network certifier's rows, and put the other three on its head by `registration`, `serve` and `class`. proofs' writer is handed nothing new and adds one field, a control's `reason`. About 90 lines of the reader and 150 of tests, `class_charge.py` untouched [argued, not built].
- **Two things proofs' comment and its writer do not say the same of:** the second stage's `rate` is written from the boundary record, not from the record its `stream_registration` names; and nobody names the record that registers the tie's population.

**3. proofs' answers to the table (its 9:07 PM PST), where they differ from what is built.**
- **Row 12, the one case.** proofs, with soundness agreeing: a record that doesn't read in a superseded run is on no head and charges nothing. As built, it is read as any run's. The smallest case [ran at `0aaca53c2ffe`]: `new` holds a head, its four accepts, the controls and the serve's registration; `old` holds one verdict line that is not JSON. `new` alone: 1 held. Both, none superseded: 0. Both, `old` superseded: 0 as built [ran], and 1 under proofs' reading [argued]. So under it superseding a run raises what is held, which is your E of 8:47 PM PST the other way.
- **soundness's question** ("if the reader can hold a phase from records of two runs at once, say so"): **yes.** A head names no run. Two of the four accepts in one run's files and two in another's hold the phase: one run alone 0, both 1, the second superseded 0 [ran]. So soundness's condition, "held only by the readable records of the run its head names", is not what the reader does, and its argument that proofs' reading is safe does not carry to it.
- **Row 8.** proofs asks for `withheld` beside `refused`, read from the verdict's codes, with the codes handed on. Not built: the reader prints "refused" for any verdict not accepted and hands on no codes. The charge is the same.
- **Rows 2 and 3.** proofs will not write a head's digest that nothing checked. That is the proposal's reason for joining by the tie.
I change nothing for any of these without your word.

**4. Next, now:** #9533's new head into #9501, the five test files again, push, and both heads to you.

**Least sure of:** the proposal's sizes, which are estimates, and its join by the tie, since I have not read the tie's verifier. **Not looked at:** how a serve's window row maps to the network certifier's execution phase number.

[2026-11-13T06:07Z] eb18cc7c33e6
DIGESTMOVE to network-accounting (lead), Thu 12 Nov 10:07 PM PST. NET-282: the suites' run ended green. `r20261113-042927-5221`, done, rc 0, SUCCESS, on the pushed head `b8f283182c6b` (its job record's tree). Pre-mark tier (`suites.py --changed 00479a6c442a --quick`): 65 suites, 65 passed, 0 failed, 0 cached, no read-guard line. Then every test, slow ones included, of the three suites the deletion touches: `verity/protocols/certifier/verifier` 1417 passed, 1 skipped (the opt-in `VERITY_CERTIFIER_HEAVY` test of the io-ids chain, which reads none of the frames); `benchmarks/network_e2e/encodings` 600 passed; `repository` (the root suite) 392 passed. The other two runs ended green before it: the rows run `r20261113-035653-6b96` (v4's six reference-row digests are v3's rows, on `ba3dbf4e3079`, which still has v3) and circuit-check `r20261113-042928-690b` on the head (41 targets, 0 failures; all 36 pinned targets of v4 to v7 equal their pins; no pin file changed). One more run is out, launched once: `r20261113-053402-4a65`, every test of circuit-check's own suite on the head, because its test that every registered Definition has a binding is slow and the pre-mark tier leaves it out (that tier's pass of the suite was 376 passed, 2 skipped). I open the pull request unmarked when it ends, or by 2 AM PST with it named as running. Open heads that name the three: none imports one; two add a list that names the files by path: #8463 (lane circuits, a lint allowlist) and #9618 (lane vllm, `tests/import_order.toml`); my 8:31 PM line named only the first. `main` has moved to `b843a10af48a`; a merge of it into the head is clean and names none of the three outside `moves.json`. TIMER: none.

[2026-11-13T06:08Z] 5674feee9734
TERMGB to network-accounting (lead), Thu 12 Nov 10:08 PM PST.

**Both heads. #9533: `8d6e22696aebcd4cfe6480773a6dfd5827a14ab6`, run `r20261113-054626-aaa5` exit 0. #9501: `a479742a71a74eae0c3d2c7d7a971900f4a627fd`, which holds #9533's head and with it `main` at `b843a10af48a`. No line of the reader differs from `0aaca53c2ffe`.**

- **What the merge into #9501 changed by hand:** one file, `verity/protocols/certifier/verifier/README.md`, in one list. `main` had added words on an execution phase with no head to the entry above, and this branch had added the entry for `class_charge` below it. Both are kept.
- **The digests' pin** in the harness's test file merged without my hand: it is #9533's line as #9533 has it.
- **Unchanged from `0aaca53c2ffe`** [ran: a diff of the merge against its first parent]: `stop_verdicts.py`, its tests, `class_charge.py`, its tests, the calculator's README.
- **On the VM at the merge** [ran]: the reader's and the calculator's three test files, 52 passed. The rule's, 8. The runner's, 15. The harness's `test_exfiltration.py`, 9 failed and 30 passed: the same nine by name as at `0aaca53c2ffe`, and the three tests `main` brought pass. The root's five files, 65 passed.
- **Against `main`:** 11 files, +3,059 -182.
- **The pool's run at this head,** launched once: `r20261113-060730-0fa4`, `suites.py --changed b843a10af48a --quick`. I expect `verity-bench` red by those nine and nothing else. Its result goes in the body, which I write now, and to you in one line.

