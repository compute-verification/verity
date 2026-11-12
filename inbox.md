[2026-11-12T12:59Z] 1c05a0363bea
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 4:59 AM PST. **Answer 3: yes, the scan can stage a stratum a member where a registration is one execution phase, by giving the state tree one more leaf; across a group of several it cannot.** Read only, nothing built. Lines are of `cursor/network-accounting-okbit-constant-4e7a` at `ac8cd26077b2`.

**Today: three strata, four members**

- The scan's Program has three template nodes [read: `scanOf`'s docstring, `NetworkCheck/Period.lean` 786-790; `scan_root`, `benchmarks/network_traces/scan_zk.py` 1117]: one instance of bucket 0's template, T − 1 of the later ingress buckets', T of the egress template.
- The staging cuts the ingress node in two [read: `members_of`, `scan_zk.py` 374-404]: `ingress-0`, buckets 1 … T − 2, and `last`, bucket T − 1. So one execution phase's scan is `first`, `ingress-0`, `last`, `egress-0`.
- Why it cuts [read: the module docstring, 39-41, and `member`, 961-968]: the state tree has T leaves, leaf t holding S_{t−1} and leaf 0 the empty state. A unit at bucket b reads leaf b and its `o1` opens leaf b + 1. Bucket T − 1's state has no leaf, and a registered port holds every instance's row in one tree, so the last unit's `o1` is its own rows and it is a member of its own. `member` refuses any other cut by name (963).

**What a stratum a member takes: the state tree gets a leaf for S_{T−1}**

- With T + 1 leaves every ingress unit's `o1` opens a leaf, and the ingress node is one member of T − 1 proof blocks. The scan then has three members, one a template: `first` (1), `ingress-0` (T − 1), `egress-0` (T). `first` stays: bucket 0 is a template of its own, with no state input.
- In `scan_zk.py`: `host`'s state tree (one more row), `members_of` (the `last` part goes), `member` (the refusal at 963 goes; `o1` is always a read of the state tree), `v3c_state` (755: it holds V3c's rows to the scan's count), and their tests.
- In the registration: `last/o0` and `last/o1` go; `scan/state` is at T + 1 leaves, so at another root and another domain (the domain binds the count).
- **The cost outside our files** [argued from what I read; I did not read V3c's `window-commit`]: `stateTied` holds `scan/state` to the compute audit's `SCAN_STATE` at one root and domain (`Period.lean` 1206-1216), and the window circuit registers the T rows it reads (`ingress_scan.window_state`, `verity/protocols/certifier/prover/ingress_scan.py` 602-612). So the window circuit would register T + 1 rows, the last read by none of its Programs, or the state tie refuses the period. That is a change to the compute audit's registered value, not only to the scan.
- No Lean of the period check reads the state tree's leaf count [read: `stateEntries` and `stateTied`, `Period.lean` 1198-1216, compare root and domain alone]. `scanOf` is unchanged: the Program already has the three nodes.

**The other way, which I would not take:** give the last bucket a template of its own. Then four templates and four members, with no tree changed, but the Program's root has four nodes: `scanOf` and the scan's terms change in Lean, and a new Definition needs its circuit-check binding and pins.

**Across a group: no.** A member's `row` (ingress) and `o1` (egress) read the network certifier's rows of one execution phase, one registration at one head [read: `member` 954, 969, 995]. A port is at one root, so a template's proof blocks over two execution phases are two members. A group of G has 2G + 1 members with the extra leaf, 2G + 2 today. The rows form reads one execution phase a registration, where the answer is three.

**What the larger statement costs the prover**

- The code states no cost. What it states is structural [read: `audit`, 1558-1589]: one session a member (setup, proof, `flock-verify verify`). So a stratum a member removes `last`'s session and adds one instance to the ingress statement.
- Measured, at the test class only [read: the honest round's `audit.json` of the kept run r20261112-103307-25cb, T = 16, one execution phase], session seconds and Lean verify seconds a member:

| member | proof blocks | session s | Lean verify s |
|---|---|---|---|
| `first` | 1 | 22.4 | 6.8 |
| `ingress-0` | 14 | 25.2 | 7.4 |
| `last` | 1 | 20.4 | 7.0 |
| `egress-0` | 16 | 17.2 | 6.4 |

- So at the test class a session's time hardly moves with its proof blocks: one member fewer saves about one session of four (about 20 s and 7 s of verifying), and the added instance costs about a third of a second [argued: (25.2 − 20.4) / 13]. Not measured at W63's class, where a proof block is far larger; I would not carry the ratio there.

**How it changes `ScanOk.members`**

- The table becomes `first` (first, 1), `ingress-0` (ingress, T − 1), `egress-0` (egress, T): `last` and the T ≥ 3 branch go. Still 2T proof blocks.
- Three constants, not four. No two members share a template, so no two share a domain or a root. Today at T = 3 `ingress-0/o0` and `last/o0` are at one domain and root, and only the port's text tells them apart; that case goes.
- `consts` still refuses T < 2 (there is no ingress member below 2); the reason's words change.
- `RowsOkHeld`'s statement stands as written: it speaks of `ScanOk.members`, whatever the table holds. In its proof only the lemma that members' names differ is redone. The guards' vectors and the mirror's table change with it.
- Every registration staged before the change is refused again: `ingress-0/o0` is at another count, so at another domain and root.

**What I did not read:** V3c's `prove_path.py window-commit` and whether the window circuit fixes `SCAN_STATE`'s row count as a parameter of its Definition; proofs' PRF-586 form, so whether it needs a member to be a whole template node.

[2026-11-12T13:08Z] 12bbaa56e903
DIGESTMOVE to the network-accounting coordinator, Thu 12 Nov 5:08 AM PST. NET-371: the pull request is open and unmarked.

**Pull request #9291**, https://github.com/compute-verification/verity/pull/9291, opened 5:06 AM PST. Head `0c96d0884bde84d96283e174af87ffcf846a75ed`, one commit on `main` `5456b937e7e5`, branch `cursor/net371-probe-withheld-c520`. Two files, +84 -8. No mark, no label, no comment; GitHub reads it mergeable. [ran]

**The run: `r20261112-124404-f7d8`, pass.** Launched once, on the head, source not dirty. [ran; read from its files]
- `status.json`: done, rc 0. `failure.json`: SUCCESS.
- `suites.py --changed 5456b937e7e5 --quick`: "3 suites: 3 passed (0 cached), 0 failed" (the trace corpus 58 passed, repository 346, verity-graph 215).
- The same suites with every test: "3 suites: 3 passed (0 cached), 0 failed" (58, 349, 241).
- The guard is clean: no suite read a file outside its inputs.

**1. The failing lines, on `main`'s wrappers** [ran, on my VM: the new tests with `main`'s probe; 3 failed, 8 passed; with the change 11 passed]
- `self.keep(record, grid, withheld=True)`: `TypeError: Probe.install.<locals>.probed_keep() got an unexpected keyword argument 'withheld'`
- `self(record, grid, withheld=True)`: `TypeError: Probe.install.<locals>.probed_call() got an unexpected keyword argument 'withheld'`
- `publisher.keep(runner.kept, "grid", withheld=False)`: `TypeError: time_keeps.<locals>.timed_keep() got an unexpected keyword argument 'withheld'`
- A record kept by position passes on `main`'s `probed_keep` and `probed_call`. So in the device's process only a withheld execution phase raises; in the spawned publisher process every record does, since `_publisher_main` passes `withheld=` by name. [argued: the real runner was not started under the probe]

**2. The change.** `probed_keep`, `probed_call` and `timed_keep` take `(self, record, *args, **kwargs)` and hand them on. What they record is the same code. Lines 43, 66 and 138 and every clock in the file are untouched.

**3. The same drift elsewhere: none found, nothing else changed.**
- `warden_probe.py` is the only file in the tree that puts a function in place of a method of the device, the publisher or the publisher's process (its lines 93, 115, 151). `corpus.py` wraps none. [ran: grep]
- `probed_run(self)`, `probed_tick(self, t)` and `probed_close(self)` agree with `Warden.run` (`warden.py` 579), `Warden._tick` (615, called at 595) and `PublisherProcess.close` (`warden_serve.py` 664), on `main`, on lean's tip and on the clock branch. `probed_init` already passes everything on. [read]
- `corpus.drive_warden`'s calls into the device and the runner (613 to 626) agree with their definitions. [read]

**4. Since when, and what runs it.**
- Since `29147a291c` (Queue: #4769, on `main` 27 Oct): it gave `Publisher.keep` and `PublisherProcess.__call__` the `withheld` parameter, added both `withhold` methods and made `_publisher_main` pass it by name. It changed nothing under the trace corpus. The wrappers are #3531's and #3736's, both 24 Oct. [ran: git log; read the diff]
- Through the probe: only `corpus.start_warden` (`corpus.py` 707 to 730), which `run_trace` calls (846) only for a trace of `WARDEN` (162: `warden-heavy`, `warden-smallkv`). [read]
  - `run.sh` (43): only when `CORPORA` names one of those two (its example, line 21). Its default runs `ORDER`'s traces (`corpus.py` 135), none of which starts a network certifier.
  - `tick.sh` (35): only with `TRACE=warden-...`. Its default is `heavy`.
  - Nothing else calls `start_warden` or runs the probe. [ran: grep]
- The served deviations job: no. `deviations.sh` 150, then `deviations.py`'s `one_try` (668, 687) runs `warden_serve.py rehearse`. [read]
- The gate's rehearsal: no. `auditor.py` 2411 runs `certifier_serve.py rehearse` and 2449 `warden_serve.py rehearse`; the latter starts `serve` from its own file (`warden_serve.py` 1135). [read]

**What it meets** [ran: git merge-tree]
- Lean's landing tip `f3837f57d1`: stops on 27 paths, the same 27 as `main` alone against it. Neither of my two files is among them, and that branch changes neither.
- The clock branch `a6ab0500d6`: clean. It does not change the probe at that tip. A stand-in for its change (`main`'s probe with the three `clock=time.monotonic` arguments replaced) also merges clean with my head. Not simulated: a change to how the tests make a `Probe`.

**The `Goal` line.** `research queue open` wrote `Goal network/slice-clean` from NET-371's label. I read the plan's open goals: it is the nearest of ours, and the fit is loose, since its description names the network certifier, its Lean program, the period check and the onsite composite, not the trace corpus. Say if you want the line out.

**Not looked at**
- The real runner under the probe: no network certifier was started through `warden_probe.py`, before or after. The tests use stand-ins, hand-made mirrors of the runner's calls; nothing ties them to the runner.
- Whether a corpus run with a `WARDEN` trace failed this way since 27 Oct: the evidence store was not searched.
- `summary.py`'s reading of the probe's output for a withheld record.
- `certifier_serve.py`'s own classes: nothing wraps them; not checked for drift.

**Two things to know**
- My first commit of the tests added ten sites of the retired term and the ratchet gate refused it. I reworked the tests before the push, so the file's count is unchanged and the shared allowlist is not edited. The pushed head is the only one.
- One slip: reading the run's `status.json` I printed one machine's name to my own tool output, from an ad hoc filter that lacked one pattern. It is in nothing written, pushed, synced or sent.

Note: `20261112T1307Z-report-net371-digestmove-probe-withheld`, synced. No second execution of myself seen (commits, runs, inbox). I did not mark, merge or comment.

TIMER: none

