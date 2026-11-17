[2026-11-17T02:21Z] 1e249ecf40a8
RECORD to the training lead, 02:21Z.

- #10665: ready again at f1ec87b9f5a3497ea86f5b05738c004424163146 (main merged) [ran]. r20261117-014757-cb74 is done, 4 of 4 passed: repository 377, training_slowdown 763, grader 36, verity-graph 248. The preflight onto main at 9f9b66b31d71 refused nothing and reported no conflict this time. The body says what the merge took (top's README whole, the grade's two statements in ρ). The goal warning stands until you send the id. Its earlier land verdict, if any, was at the old head, so it needs a reader again.
- #10564: its lock is done at e1e3e1d42f22 [ran].
  - r20261117-014318-06b2: security AUDIT PASS. security/Properties was regenerated, and its new record drops only the cover theorem's open premise Slowdown.Cover.MulBlocks.Upstream, now committed.
  - Across both locks, the diff against main is +2/−238.
  - These are the same removals NAMES2 listed at 01:50Z (the Desc.Cover root, Upstream in both locks, the Witnesses reads). I made them at f1430cae before their message reached me, so #10687's merge should be clean. NAMES2, please confirm against e1e3e1d42f22.
  - Not marked.
- #10570: main merged at 078ec4c2c33a. Lock update r20261117-012717-6d1c: done, AUDIT PASS for security and security/Properties, OUTCOME pass (lock current, nothing to commit) [ran]. Not marked.
- #10607: lock update r20261117-014447-2fac at 4725f7d8a412 (main merged), running.
- #10619: suites r20261117-014630-9aec at 03b32d558eb0, running.
- #10556's suite run r20261116-215836-155e is cancelled at the slot's 2 h limit with verity-flock and verity-vllm unfinished, as its body already says.
- #10611: still open as of 02:17Z.

[2026-11-17T02:22Z] 232a76e8ec16
From NAMES2, 6:22 PM PST. #10687 (tiers): the Lean builds and the records update passes. The lock commits are pushed onto its branch, head b6512c57167, a fast-forward from PAGE's efbdf5ab5f4. No mark.
- Runs, by id and state:
  - r20261117-011615-5a40 failed: the first run, at bbc079433d2, with the body's 18/4/2/12/9/13/8 entries removed by hand. security PASS and regenerated; its requirer security/Properties failed to build on #10564's leftover root.
  - r20261117-015124-4f06 done: at b4d1f4d8b46. security PASS with nothing more to regenerate. Properties PASS after it regenerated one entry, which I committed as b6512c57167. [ran]
- Lock diff against efbdf5ab5f4 [ran]: 2 files, +4/-705.
  - Exactly the body's list: guarantees 18, witnesses 4, unwitnessed 2, shape.frozen 12, shape.inline 9, open 13, reads 8 module records, and the Witnesses and Witness definitions. Every entry of either lock that names a deleted declaration or module is in that list; nothing outside it does.
  - Also regenerated: reads drops Verity.Game.prPMFReal (Definitions.Core.Prob), Slowdown.Witness.mb0 and σ0, and recomputes those three digests. Only the deleted guarantees read them.
  - Also four #10564 leftovers, which #10687 needs in order to build:
    - security/Properties roots: Properties.InferenceOnly.Desc.Cover, whose file #10564 deletes;
    - open Slowdown.Cover.MulBlocks.Upstream, in both locks;
    - the Witnesses reads' CoverTraining_{honest,nontrivial}.
    RECORD has the list (5:50 PM). If #10564 makes the same removals, the merge into #10687 is clean.
- The merge with my two, once they land [ran: trial merge of my #10613 merged with main into efbdf5ab5f4]:
  - Two files conflict, three hunks, each the union of both deletions.
    - security/Proofs/Slowdown.lean's docstring becomes "...the work and product counts of threads, fit and layered references, covering at one and two costs, the training formalism (`Formalism`), and their witnesses."
    - compose's remote README: "`Verity.Guarantees.Counting` and the floor."; mine drops "The local bounds" bullets and #10687 drops "Results per monitoring period", so both go.
  - Definitions.lean, the Witnesses imports, security/README.md, the retired-terms list and both locks merge on their own. merge.py merged the locks; both keep their layout, and every security import resolves.
- #10703 (threads) after mine: the same two content conflicts, plus security/Proofs/Slowdown/Witnesses.lean (a modify/delete), which resolves as the deletion. Then security/Definitions/Slowdown/Capability/Toy.lean, the one Capability file #10611 keeps, is imported by nothing, so the Lean check fails it as orphan.file. #10703's merge of main must delete Toy.lean and its lock record reads."Definitions.Slowdown.Capability.Toy". Nothing else names it [ran].
#10703's records are running at 5058ca7762c, which merges #10687's b4d1f4d8b46: security r20261117-020910-2560 and security/Properties r20261117-021001-6997. #10713: r20261117-020728-2db2, running.

[2026-11-17T02:26Z] d394a48d714d
From PAGE (training). TRN-177, item 1: the benchmarks/nci deletion.

Draft #10731, `cursor/training-delete-nci-4207`, head `40a00eed117`, base #10703's branch (`3438c2ec93a`). Not marked; it is compute-accounting's to read. It is not linked in Linear (no credentials); `research queue link 10731 --issue TRN-177` links it.

- The base needs #10597 too. With #10597 still in, main's `families.py` and three slowdown tests load nci's `algorithms.json`, `bracket.py` and `tiers.py` [ran]. So the branch is #10703, then a local merge of #10597 (commit `d4dec775a3b`), then the deletion (commit `40a00eed117`, 55 files, +34/−11,624). Compute-accounting reads the last commit. In the merge, three files main changed after #10597's base and #10597 deletes are taken as deleted: `envelopes.py`, `operating.py` and `test_envelopes.py`. #10597 meets the same conflict when it merges main.
- What goes:
  - `benchmarks/nci` and `tests/benchmarks/nci` (4,091 Python lines with the tests);
  - `theorem.py` (`multiple_bound`) and `test_theorem.py`;
  - `hardware.toml`'s `location.rtx-pro-6000-server` and `fleet.tiers-reference`, which restated `tiers.py`'s server and which nothing reads [ran];
  - two calibrations nothing else reads.
- Rows that go:
  - `benchmarks/README.md`'s index row;
  - compute-accounting's `owns` entry for `benchmarks/nci/` in `registry.json`;
  - the `artifacts.json` fixture row and its `.gitignore` line;
  - 26 allowlist rows;
  - `test_lean_vocabulary.py`'s entry for `theorem.py::Partition`.
- Metrics removed: none. There is no `metrics.toml` in nci, and none names it.
- What moves: `benchmarks/nci/calibrations.json` becomes `benchmarks/slowdown/data/calibrations.json`, with the three entries `params.py` reads. The registry's `policy.kimi-k2-1m-raw.{k,gamma}` and the range of `policy.v4-pro-session.k` come from it, so it can't simply go.
- Readers in other lanes' files. None stops this; tell me if you read any as a stop:
  - `benchmarks/spot_figures/figure1.py` (proofs') reads `policy.v4-pro-session.k` through the registry. It is unchanged, and the move keeps it working: its nine figure test files gave 112 passed, 1 skipped [ran].
  - Edited, path lines only: the harness suite's `inputs` (`tests/benchmarks/harness/pyproject.toml`, benchmarks') listed five nci files.
  - Edited, prose only:
    - `benchmarks/README.md` (benchmarks') loses its index row;
    - `.gitignore` (top's) loses one line;
    - the inference-only verifier README (network-accounting's) loses a sentence and one "older name" that named `benchmarks/nci`.
  - Not edited: `GLOSSARY.md:510` (top's) names `benchmarks/nci`. It goes to top in the sweep.
- Ran:
  - slowdown suite: 96 passed.
  - `inference_only`: 86 passed, as at the merge commit.
  - `circuit_properties`: 3 failed and 1 error, the same ones at the merge commit (environment).
  - Root and lint, 13 files: 185 passed, 2 failed.
    - `test_every_open_conjecture_names_a_test_that_exists` fails until #10703's lock update removes the three conjectures that name nci tests [inferred].
    - `test_the_scan_sees_verity` fails at the merge commit too (environment).
- Not run: `training_slowdown` (its conftest imports `torch`, absent here) and the check's runner. No Lean is touched.

The sweep comes next.

[2026-11-17T02:31Z] 2cd75ac2dc64
From NAMES2, 6:31 PM PST. #10713 (untied) builds at its head 3e528fedc6e, and its lock needs no change [ran: r20261117-020728-2db2 done].
- security/Proofs built: AUDIT PASS.
- security PASS, with regenerated {}.
- Its requirer security/Properties PASS, with regenerated {}.
- No record/ directory was written.
So HECharged's proof and its witnesses build without Untied.lean or CutWitness's untied toys, and the body's "Lock records: none" holds. I committed and pushed nothing, and there is no mark. #10703's two runs are still going.

