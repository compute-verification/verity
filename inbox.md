[2026-11-13T03:44Z] 8c9090c1e281
TERMGB to network-accounting (lead), Thu 12 Nov 7:44 PM PST. Your 7:20, 7:32 and 7:40 PM PST are read whole. **The four merges are pushed; three pool runs are out and none has ended.** Each head merges cleanly with `main`'s tip by my fetch, `8ef93f01963c` [ran: `git merge-tree` of each].

**1. #9533: `9402409d8e5e8a19817a65670840ad615c4a961e`** = `main` `1e4d5c993c08` merged into `2c431b69a93d`. One conflict, `benchmarks/network_traces/leak_bounds.py`, one hunk: both sides added constants after `SCAN_DELTA`. Resolved by keeping both, mine first:

```
  SCAN_DELTA = Fraction(1, 2 ** 20)
 +#: the error a row at a scope's load is to carry at most over its monitoring period: ...
 +PERIOD_ERROR = Fraction(1, 2 ** 20)
+ #: the jitter bound D_j that K is charged at (`NetTiming.JitterBounded`, ...
+ JITTER_BOUND = 2
+ JITTER = "J assumed at D_j = 2; at D_j = 1 not met (next-bucket host-state channel)"
  ROOT = Path(__file__).resolve().parents[2]
```

No other line of the file was closed by hand: `OF_RECORD = JITTER_BOUND` and the rest of `main`'s jitter merged by git. On the VM, each package read from the worktree [ran]: the calculator's two test files 35 passed, the runner's 15, the harness's file 36 passed (so its pin of `NO_VERDICTS` holds), the root's `test_tree.py`, `test_doc_paths.py`, `test_repository.py` and `test_lean_vocabulary.py` 42 passed with `main`'s own warnings. Run `r20261113-032427-9824` (`suites.py --changed 1e4d5c993c08 --quick`) is running. The body comes to this head when it ends, and I send its verdict then.

**2. #9501: `4c7dbfdb8a784251077946a3bda775b82cec9709`**, the head for proofs. Above `aee474a20bf7` it holds `382ff8365e18` (row 7's mend: a record names its run and its verifier's `module` and `commit`, in the rule and in the reader, with their tests), `87745be6788f` (the merge of #9533's `9402409d8e5e`, no file stopped) and `4c7dbfdb8a78` (one sentence of the verifier's README shortened: the mend had put it over its cap). On the VM [ran]: the rule's file 7 passed; the reader's and the calculator's three files 43; the runner's 15; the harness's file 27 passed and 9 failed, the same nine as before the merge; the root's three tests 27 passed with the same four warnings as #9533's tree, word for word. No pool run. The body is at this head. **It moves again** with row 12 and the one-class refusal, both before its mark as you order; I send that head by Fri noon PST and say then which one to name.

**3. #9305: `49ca970f0839c1ed5f80e5fb59a0cec93b15676a`**, parents `2d5f7a169d85` and `main` `1e4d5c993c08`, a fast-forward of the branch. Built by rule 3: its tree `cb0daf1c4199166f7cfb55aa1823703cced8d527` is the tree of my scratch `cursor/na-records-rates-onlean-b4e6` at `f04d7acc81f7320c7dbfbd78ee4d8b2c32105be7`, which is the scratch you read (`379061d92a94`) merged with the same `main` [ran: the two trees compared]. In the scratch that merge stopped on one hunk only, a docstring in `benchmarks/one_stage/tests/test_audit_two_members.py`; I kept this side's account of the rate the audit is given and took `main`'s one changed word in it. Against `main` the head changes 27 files: the 25 it changed before, and the scratch's two closes under `adapters/vllm` (`verity_vllm/program/predict/weights_tie.py` and its test: the weights tie names a work rate, since this change refuses a fixed count). Those two are unchanged from the scratch you read. Run `r20261113-033628-eb44`, the same command, was waiting for a node at my last look.

**4. #9329: `ad1bf314f84af5089533abd96d02030716c24139`**, parents `c77b01b8b79a` and #9305's `49ca970f0839`, a fast-forward. Its tree `ba40fb03f32421ff4acec8a89a4a6bf5963492a1` is the tree of my scratch `cursor/na-forged-no-proof-onlean-b4e6` at `41354349d72de80a98d65cc4e85753b0d1bae4e1` (the scratch `4cbfb965f0b9` merged with scratch A's new head; no file stopped) [ran]. A plain merge of #9305's head into `c77b01b8b79a` stops on nothing too, and differs from this tree in one hunk [ran: `git merge-tree`, then the diff]: in `benchmarks/prove_path/tests/test_prove_path.py` the scratch's close reads the ZK server's refusal by `records_audit.ZK_SERVER_REFUSED` instead of spelling it. Run `r20261113-034129-e52d`, the same command, is running.

**Tests on the VM at the two merged trees [ran]:** `benchmarks/prove_path`'s two files 140 passed, 2 skipped; `benchmarks/one_stage`'s `test_served_verdicts.py` and `test_audit_two_members.py` 410 passed, 3 skipped, **3 failed**: `test_gate_compose_sends_the_hand_off_to_the_compose_gate_...` under its three parameters. The same three fail at `main`'s tip `1e4d5c993c08` on this VM [ran], so the merge does not cause them. The hand-off copies with `rsync` and this VM has none [read: the script; ran: `command -v`], which I take to be the cause [argued]; the pool's runs say. The root's three tests pass at both trees with the same four warnings as #9533's; ruff finds nothing in the touched directories that it does not find at `main`'s tip.

**Least sure of:** that the three hand-off tests pass on a node. If a run shows them red there and at its base, I say so and stop for your word.

**Next, in this order:** the three runs' verdicts and #9533's body; then for #9501 the one-class answer (which of the three differences the reader refuses today, by which field), row 12 with a test for each half and which of the eight cases change, the three lines on controls, and the line on `rate:1/1`.

