[2026-11-12T02:17Z] a293924b9290
DIGESTMOVE, Wed 11 Nov 06:17 PM PST, where #8686 stands at 5773b1f3f4ba (unchanged since my sha line; no records commit is needed). [ran] lean try DefinitionDigest.lean r20261112-013005-21ad: SUCCESS, 'lean exit 0 in 21 s' (its siblings step imported 0 and left 11 unchecked for want of their builds; the package's own audit below builds all 13 modules). [ran] lean try NetworkCheck/Period.lean r20261112-013005-aff8: SUCCESS, 'lean exit 0 in 19 s'. [ran] records, lean_changed.py --records nci --update --rebuild, r20261112-013005-fb14: SUCCESS, lean-fast/v1, outcome regenerated: the period check's package, security, security/Properties and security/Proofs each pass on the merged locks as committed; nci_check is 'regenerated' by one line, "format": 1 at the top, the tool's stamp you ruled out of the PR, so I commit nothing unless you say so. The PR's own Lean check (the same fast path, no --update, as e1f9 was) is r20261112-021536-3358, launched 6:15 PM PST, waiting for a slot: I expect it before 7:15 PM PST, well inside 8:30 PM. The suites (r20261112-013005-773d: --changed be9a3d566905 --quick, then the same without --quick) are running, no failure so far; the full tier may run past 8:30 PM PST, I say at 7:30 PM PST where it is. No lakefile changed.

[2026-11-12T02:19Z] 8bdf67d34fde
TERMGB to coordinator (network-accounting), Wed 11 Nov, 6:19 PM PST. Part C is pushed.

**Sha:** `850a314f7372230ddbb99015296e8b38336d441f` on `cursor/na-cross-leak-bounds-term-b4e6`, a new branch of mine cut from `102668539b17`, pushed at 6:17:45 PM PST. One commit, one file: `benchmarks/network_traces/leak_bounds.py`, blob `294d80ee05ef`, 995 lines, +199 −76 against the close's. No pull request. Nothing pushed anywhere else: the close's branch, lean's, yours and NET-345's read the same heads after the push as before. [ran]

**When lean may take it whole:** when `main`'s `leak_bounds.py` is #8777's blob `80f836f36996`. At 6:17:55 PM PST #8777 was still open and `main` was still `be9a3d566905` (the file `97ea444b8661`) [ran]. If another change reaches that file first, the file must be made again.

**1. The file and its base.** Your finding holds: `git merge-file -p` of the close's file (`b14db4d89c0a`), `main`'s (`97ea444b8661`) and #8777's (`80f836f36996`) is clean, 994 lines, `JITTER_BOUND = 2` at line 112 and `OF_RECORD = 2` at line 464, and the other order gives the same bytes [ran]. The base is right because `main`'s file is one blob at both ends: the close's commit message says it was made against `main` `75ef75838f99` [read], the file there is `97ea444b8661`, it is the same blob at `be9a3d566905`, which #8777 sits on, and no commit of `main` between them changes it [ran]. So each side is a change to the same file: the close adds 20 lines and removes 4 (the jitter), #8777 adds 198 and removes 76 (its table), and the merged file is #8777's plus exactly the close's 20 and 4 [ran].

**2. One definition.** One line changed by hand and one added: line 464 is now `OF_RECORD = JITTER_BOUND`, under a comment line saying they are one bound, defined there. `JITTER_BOUND = 2` stays at line 112. No reader needs it the other way round: each reader of either name reads it by attribute at run time (`scan_period.py`, `chain_record.py`, `tests/test_scan_zk.py`, `tests/test_leak_bounds.py`, `tests/test_chain_record.py`, `tests/test_certifier_term.py`) [read]. Both import as 2 and are the same object [ran]. Ruff passes on the file [ran].

**3. What lean's merge meets on the nine paths.** lean's tip is no longer the one you named: it is `a07cf0b341d7`. `3ff11d118f93` is its ancestor, and none of the ten commits since touches the nine paths, `chain_record.py`, `scan_period.py` or `tests/test_scan_zk.py` [ran]. `git merge-tree` of that tip with #8777's head stops on 49 paths, the same 49 as with `main` alone and as for the tip you named: #8777 adds no conflict and removes none [ran]. The nine:

| Path | lean's merge meets |
|---|---|
| `benchmarks/network_traces/leak_bounds.py` | conflict, 2 hunks: take this commit's file whole |
| `benchmarks/network_traces/tests/test_leak_bounds.py` | conflict, 2 hunks; both sides unchanged since the close: the close's file stands |
| `benchmarks/network_traces/tests/test_chain_record.py` | conflict, 1 hunk; both sides unchanged: the close's file stands |
| `verity/protocols/certifier/device/README.md` | conflict, 1 hunk (both added it); both sides unchanged: the close's file stands |
| `benchmarks/README.md` | clean, #8777's blob |
| `benchmarks/certifier_term/README.md` | clean, #8777's blob |
| `benchmarks/harness/tests/test_exfiltration.py` | clean, #8777's blob |
| `benchmarks/network_traces/tests/test_certifier_term.py` | clean, #8777's blob |
| `benchmarks/network_traces/pyproject.toml` | clean, a true merge: #8777's lines plus the crossed head's three |

So I wrote no other file. One thing in `pyproject.toml`: the suite's inputs name `benchmarks/census/deployments.json` twice. The second entry is on the crossed tip already, not made by the merge [ran]. It parses; I did not check whether anything refuses a repeated input.

**4. Tests, local, on the assembled tree** (how: a worktree at `a07cf0b341d7`; `git merge --no-commit --no-ff` of #8777's head, left uncommitted; the close's three other files written whole from `102668539b17`; this commit's file put in; the other 45 conflicted files as git left them, but two the tests import, closed for the run only, below; tests run with the main checkout's interpreter and a path naming only the worktree's packages; each run reported 0 modules read from the main checkout) [ran]:

| Test | With this commit's file | With the close's own file |
|---|---|---|
| `tests/test_leak_bounds.py` | 9 passed | 9 passed |
| `tests/test_certifier_term.py` | 20 passed | 8 passed, 12 failed |
| `benchmarks/harness/tests/test_exfiltration.py` | 11 passed | 10 passed, 1 failed |
| `tests/test_chain_record.py` | 18 passed, 5 failed | 18 passed, the same 5 failed |
| `tests/test_scan_zk.py`, the one test that pins `JITTER_BOUND` | 1 passed | not run |

- The five failures are one cause, and it is lean's merge to close, not this file's: `lake build network-records` stops on `verity/protocols/interface/lakefile.toml` line 2, a conflict marker in one of the 49 files [ran]. None of the five reads `leak_bounds`, and the same five fail with the close's file [ran]. They have no verdict of their own here; this VM holds no Lean build.
- The right-hand column is why the file is needed: the close's file as it is fails 13 tests of #8777 on the merged tree (first causes: the schema reads `certifier-term/v1`, and `inputs()` takes no `requests`) [ran].
- Figures on the same tree [ran]: K by links, 168 figures, equal to the close's on 168 and to #8777's on 0, as it should be (#8777's file charges K by links at one schedule). The certifier term's document at the plan's and the census's requests, with and without four lanes: equal to #8777's whole on 4 of 4, and its print equal character for character. #8777's file gives the same four documents on #8777's own tree as on the assembled tree, so the crossed head moves no input of the table [ran].

**What lean must know beyond the file.**
- The tip moved; the nine paths did not.
- Two of lean's own 49 stop our tests until closed. `verity/protocols/certifier/verifier/network/tests/period_case.py`, in `public`: the code hunk needs both sides, `main`'s two `max_tokens_cap` lines and the crossed head's K and eps lines; I ran it three ways (ours, theirs, both) with the same counts. `verity/protocols/spot/verifier/one_stage/proofs.py`: both sides added a function at one place; I kept both. Both are closes I made for the run only, on the VM, in no commit [ran].
- On my branch alone the file sits on the crossed head with no `main` in it, as the close's files do. There `tests/test_leak_bounds.py` is 8 passed, 1 failed with this file and with the close's alike (the crossed head's `chain_record.py` reads `leak_bounds.world`, which `main`'s side removed) [ran]. It is to be taken whole into the merge, not judged there.

**Not checked.** The five Lean-building tests to a verdict, and anything in Lean. The whole check, whole suites, and the root's tests on the assembled tree. `benchmarks/certifier_term/run.py` from its command line (I ran `document` and `render` through the module). `verity/protocols/certifier/device/README.md` beyond that both its sides are the blobs the close was made on. No remote run: the assembled tree is not a commit, and each test took under a minute here. I am least sure of the two files I closed for the run: they are lean's, and a different close of `period_case.py` could change what `tests/test_chain_record.py` and the `tests/test_scan_zk.py` test see.

[2026-11-12T02:21Z] 7a7a817db4cd
SCANLEAN to the coordinator (network-accounting), NET-304, Wed 11 Nov 6:22 PM PST: the merged head is pushed.

`cursor/na-tied-need-crossed-37ad` is at `889a02c0b9c8`, a merge commit of `72caf456be00` and the crossed tip `a07cf0b341d7`, pushed with no force [ran]. The merge stopped on no file and touched none of our eight: `git diff 72caf456be00 889a02c0b9c8` on them is empty, and the head against the crossed tip is the same eight files, 503 lines added and 36 removed, as before [ran]. It changed none of the three Lean files, no file they import and no file that imports them, so I do not rerun the Lean try [ran: the tree's import graph, over-approximated where a module name is ambiguous]. The suite `verity-network-traces` is launched at the merged head, whole and fresh, `--on auto`; its id and verdict follow.

