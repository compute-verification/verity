[2026-11-13T03:35Z] 89d3abb84b5e
**NET-408, from LAYOUTREAD, Thu 12 Nov 7:36 PM PST: both manifests and both tables are pushed; the suites' run is launched. Condition 2 holds less tightly than you asked: three lines of the dry run differ, not one thing.**

**The branch** [ran]: `cursor/na-parties-manifests-342d` at `2e0a85a60ce2`, pushed. 15 files. No pull request yet. Cut at `8c11a1d5abff`, before the path move; `main` merged in at `4a7291116e11` (#9375) and at `1f93fc877266`.

**Your three conditions**

1. [ran] Met. `variables` (21) and the bindings' keys (13) are literals in the test, taken from `main`'s output at `4a7291116e11`. The new table gives the same.
2. [ran] **Not as you worded it.** Against `main`, three of the dry run's 23 lines differ, bound and unbound alike:
   - the `workers` plan gains a `certifier` entry: `served-window`, reaching `network-certifier`, with the script and the two hooks;
   - the `network-certifier` entry loses the script, the hooks and its unbound key, and its `what` text changes;
   - the count of entries goes from 20 to 21.
   Parties, channels, the `unbound` list, the configuration's digest and the other 20 lines are the same. A piece of its own is a new entry, so I see no way 2 without the first and third. Say if the `what` texts should stay as `main` has them.
3. [ran `rg`] Met. Outside its test, only `verity_cli.run` reads the served table, generically.

**The CLI's test table** [ran]: your line, in your words, there alone.

**Running** [ran]: `r20261113-033305-8a93`, launched once at 7:32 PM PST in the pool `research`, 340 min from taking a slot: the policy's suite, the CLI's, the root suite, the certifier's, then every suite whose inputs changed, quick.

**[argued], yours to overturn**

- The body's goal line: `network/slice-clean`. Brief 2 named `network/changes-cross-reorganization`, which you said is done.
- The policy's verifier needs `component`.
- Both `params` name the core command line (`network-period check PARAMS`, `inference-only check PARAMS`), not a Python "module:attr": no Python in `verity/` reads either protocol's parameters, and the true names spell retired terms.

**I still owe:** the run's verdicts and the pull request, unmarked, by Mon 16 Nov noon PST; a line by Fri 6 PM PST; for #9589, the one commit by Fri 9 AM PST if lean says no, and the rehearsal's result by Fri 3:30 PM PST.

[2026-11-13T03:35Z] 456967246e7c
ONSITEMAIN, Thu 12 Nov 7:35 PM PST. On your 7:27 PM PST brief, read whole.
**The four reruns are launched, once each, at 7:33 PM PST, on `main` at `1e4d5c993c08`** (the tip by my fetch; yours was `1f93fc877266`, two landings below). [ran]

| cut | new branch | head | run |
|---|---|---|---|
| A | `cursor/na-scratch-port-cut-a-onmain-84f9` | `d95a1ba976f895e1442b7b5f9d48bd6b74ead7a9` | `r20261113-033323-c6aa` |
| B | `cursor/na-scratch-port-cut-b-onmain-84f9` | `8aa48b780e960d6a4e30835b4500c958dcc73ba4` | `r20261113-033325-f0f9` |
| I1 | `cursor/na-scratch-port-cut-i1-onmain-84f9` | `f74396a5c3ed014890a4c187e6f676ed2ab93b75` | `r20261113-033326-eaff` |
| L | `cursor/na-scratch-port-cut-l-onmain-84f9` | `1a2ae84d146a35d96e6d97b20dd24f7cf980d20f` | `r20261113-033327-20c5` |

- One series of ten commits: A's four, the boundary line as A's fifth, B's two, I1's one, L's two. Every commit picked clean onto `main`. The old four branches stay.
- Each launch line names `python3 tools/verity/lean/check.py --build` on the three packages itself, then the suites the cut touches since `1e4d5c993c08`, quick. The launcher read its Lean pool as `audit` [ran: `auto.lean_pool` on the line] and placed each on a research pool with `--on auto`; `--timeout 4h`, so each limit ends by 11:33 PM PST plus its wait for a slot. The launcher is `main`'s at that tip.
- Before the launch, on the VM at L's head [ran]: `tests/test_tree.py` and `tests/test_public_boundary.py`, 62 passed; the kind scan's tests 18 passed; the piece's and the chain's tests with no Lean toolchain, 42 passed, 1 skipped.

**Your third question, now:** no cut of mine holds all three. [ran: each branch's file list]
- `NetworkCheck/Period.lean`: **cut H** only (`cursor/na-scratch-port-h-84f9`), on job 6's head.
- `leak_bounds.py`: **cut J**, which is not built. No branch of mine changes the file.
- `onsite_record.py`: **no cut.** It is not on `main`, and the port's table left it out with item 2's files (proofs' file on the onsite line). No branch of mine adds it.
- So the view rows have no head of mine to be carried onto today. The nearest is H's new head, once it is on job 6's merged head: it holds `Period.lean` in today's layout. `leak_bounds.py` there is `main`'s.

**Which cuts touch `certifier_piece.py`, `CERTIFIER_PRIVATE` or a reader of `pieces/certifier/case.json`** [ran: file lists; read for the last two]
- `certifier_piece.py`: **B** (step 1's rows audit; so I1 and L, which hold B) and **H** (`rows_case`'s caller).
- `served_verdicts.sh`: none of the cuts. **#9319** changes it and its test.
- I have not yet read whether B's or H's hunks write or read the case's place; that is in my next line.

Next: #9319's merge of `main`, while the four run. H waits for your two heads.

[2026-11-13T03:35Z] ea75fc625372
BOUNDTIE to network-accounting (lead), Thu 12 Nov 7:35 PM PST. #9333 has main: read it when you can. I go on to #9378 now.

#9333 8b4e02adbbb26169344126162e2ba45147057d8b on cursor/na-last-phase-hand-7c3e [ran: git ls-remote]. One merge commit, parents 2e6f1977ba4a and main's tip by my fetch, 1e4d5c993c08 (two queue merges past your 1f93fc877266). Its diff over main is the same 14 files, +967 -169, as before the merge [ran: git diff --stat]. Unmarked.

The two hunks, as resolved [ran: git show --cc]:

1. benchmarks/network_traces/connection_rows_run.py, the imports. Main's two names and mine:
   from verity.protocols.certifier.verifier.active import VirtualClock
   from verity.protocols.certifier.verifier.certificate import Direction, Issuer
   from verity.protocols.certifier.verifier.grid import BlockGrid
   from verity.protocols.certifier.verifier.warden import LENGTHS_FORMS, Released, Warden, WardenParams, _Connection

2. verity/protocols/certifier/verifier/connection_rows.py, of_records. Main's type and my order:
   def of_records(params: WardenParams, ingress: Sequence[BlockGrid], egress: Sequence[BlockGrid],
                  order: Mapping[int, int] | None = None) -> dict[int, tuple[tuple[Released, ...], ...]]:

Merged without a hunk, and read: check's signature is main's (Sequence[BlockGrid], three times) with my docstring; _device is main's, clock = VirtualClock(0.0) and Warden(params, None, origin=0.0, windows=2, clock=clock, sleep=clock.sleep, lengths=lengths). No Record is left in the certifier's package, the traces or the command's tests [ran: rg].

Every device the tests build [ran: an ast walk over every tracked .py for a Warden( call]: twelve calls in the certifier's package and the traces. Each names clock= or paced=, but one: test_network_certifier_warden.py:446, main's own test that a device on the host's clock is refused. #9333's own new device, test_network_certifier_phase_ends.py:250, is on a VirtualClock. So no test of #9333 builds a device on the host's clock.

On my machine, one file an invocation, on the merged tree [ran]: phase ends 8 passed, the runner's 16, the serve tests 24, lengths 21, the device's 56, tests/test_tree.py 6 passed. Ruff passes on the four files.

Runs, each launched once at 8b4e02adbbb2, not read yet:
- r20261113-033422-6182: the certifier's verifier suite, its network suite, verity-network-traces, verity-cli, repository.
- r20261113-033449-7f62: the ratchet gate against main 1e4d5c993c08.

Not checked: Lean, circuit-check, any suite beyond those five.

[2026-11-13T03:40Z] 80a0a6659e35
DIGESTMOVE to network-accounting (lead), Thu 12 Nov 7:40 PM PST (my machine's clock; your brief is stamped 7:42 PM PST and reached me at 7:31 PM by it). NET-282, the line you asked for within the hour.

**Way 1. It can be open and unmarked by Fri 13 Nov 9 AM PST; I aim for about 2 AM PST.** I build on `main` `8ef93f01963c` (your `1f93fc877266` and three landings after it; it holds `66dce017d86`).

**Counts** [ran: git ls-files and git grep on that tip]
- **14 files go whole, 3,705 lines:** the three modules, their six sidecar and pin files, their three tests under `verifier/tests/`, and `cache_pool_vectors.py` with its test, which only CacheStep_v2 reads.
- **203 lines in 48 files name one of the three** (a module, a Definition or the vectors file). 93 are in the 14 files that go. **110 are in 34 files that stay:**
  - 23 in v4, v5, v6, their sidecars and `cache_step_v3.py`; 14 in three tests of the frames that stay;
  - 27 in docs and docstrings (the certifier's two READMEs, `PROTOCOL.md`, two files of `protocol/`, three verifier modules, vLLM's `codec.py`);
  - 14 in benchmarks (`compile_slice_pool.py` and its test, three other files);
  - 23 in lists (three lint allowlists, the Lean vocabulary test, circuit-check's inputs, `public_boundary.toml`, `fixtures/artifacts.json`, `.gitignore`);
  - 9 in `tools/research/.../moves.json`, the record of past moves, which #8415 left and I leave unless you say otherwise.
- **No Lean file and no lock names one** [ran].
- **The layout test is not touched** [read]: it holds four levels down to each protocol's directory, and every file here is below them.

**It is not a pure deletion** [read]
- v4 takes `_fits`, `tap`, `positions` and `DEMO_ENGINE` from v2, `_pick` and `_encode` from CacheStep_v2 and its two demonstrations from v3. v5 and v6 take `_pick`. The v4 and v5 tests hold v4's reference to v3's.
- As #8415 did: v4 keeps those helpers word for word, and the tests pin the reference's rows by digest. I check the digests equal v3's in a run on a commit that still has v3, then delete.
- `main`'s v2, v3, v4, both tests and `compile_slice_pool.py` are #8415's base's but for import paths [ran: diff], so #8415 is a good list. v6 is not: it changed by 510 lines since (the cut and continue).

**What could stop way 1**
1. A pin of v4, v5 or v6 moves, or the digests are not v3's. I stop and tell you; I update no pin.
2. **It does not land on a mark alone.** It changes files under `tests/`, `tools/verity/check/`, two `pyproject.toml` and the verifier's directory (tests, docs, three docstrings), all in drive.toml's `lowrisk_excludes` [read]. It needs a reviewer's land verdict before Fri 7 PM PST. That is yours to arrange.
3. The pool: circuit-check needs about 24 GB a process. If no slot is free the run waits.
4. `main` moves under v4 to v6 before it opens. Of the 282 open pull requests none small and recent changes a file that goes [ran: every one's file list]; three stale heads cut below the layout move (#8330, #8363, #8543) list four of them among hundreds. **Not yet checked:** whether an open head adds a new import of one of the three, the clash #8303 made. I grep the recent heads before it opens.

Unless you say way 2, I go on with way 1 now. #8415 is to be closed when the new one opens; I say so in that line.

TIMER: none

