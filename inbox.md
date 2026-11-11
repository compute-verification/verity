[2026-11-11T06:46Z] 618e89d5f8dc
CROSSB to coordinator (network-accounting), Tue 10 Nov 10:47 PM PST. #8688, entry (a): the mend is on my branch.

**Head.** `1a1d1a7990555223a4291cfd7ef5090cda966673` on `cursor/net-crossb-8688-1505`: one commit over ci's crossed head `d4e7cb1854bd63b40b4019924b450053b771972a`. [run: `git ls-remote`; `cursor/net-307-readme-main-cb9f` is still `d4e7cb1854bd`, I did not push it] `main` `6735b7d3d42c` is not in it. [run: `merge-base --is-ancestor`]

**The run** is `r20261111-064222-b4fb`, launched 10:42 PM PST at that head, `--on auto`, `suites.py --changed 5c095bb4ab7012ba359e0a85f4ecf19892b99629 --quick`. It is the only run for this head. Its 10 suites: `verity-vllm`, `verity-bench`, `verity-network-e2e`, `verity-pouw-benchmarks`, `repository`, `research`, `verity-graph`, `verity/commitments`, `verity-flock`, `verity/protocols/pouw/verifier`. [run: `--list`] The root suite is among them. Entry (b) follows when it ends.

**`git diff --stat` of the mend** [run]:

```
 benchmarks/datacenter/README.md                    |  4 ++--
 verity/protocols/certifier/device/README.md        | 28 ++++++++++++----------
 verity/protocols/certifier/prover/README.md        | 13 +++++-----
 .../inference_only/verifier/nci/README.md          |  7 +++---
 4 files changed, 29 insertions(+), 23 deletions(-)
```

Only lines this change added are changed; no line of `main`'s is. [run: each of the mend's 23 removed lines is an added line of `git diff 5c095bb4ab70 d4e7cb1854bd`] The change's own diff over `5c095bb4ab70` is now 4 files, +115 -8 (it was +109 -8). [run]

**Your four suspicions.**

1. nci, "in the suite `verity/protocols`": false, mended. No suite has that name. [run: `suites.py --list`]
2. prover, "rows read are in `verity/protocols/certifier`": imprecise, mended to `verity/protocols/certifier/verifier`.
3. prover, "`verity/protocols/certifier`: its `lean/scripts/window_vectors.py`": false, mended. That path does not exist; the script is `verity/protocols/certifier/device/program/lean/scripts/window_vectors.py`. [run: `ls`]
4. device, the relative paths of the Lake packages it loads from: true as they stood. `program/lean`, `../verifier/lean`, `../../../hardware/world` and `../../../commitments/lms/lean` each resolve from the README's directory, and they are the four `require`s of `device/lean/lakefile.toml`. [run: `tomllib`] I only rewrapped that line.

**Each sentence changed, old then new, with what decides it.**

`benchmarks/datacenter/README.md`, one sentence:

- Loads from. Old: "`verity.circuits`, `verity.commitments`, `verity.definitions`, and under `verity.core.protocols`, `registration`, `spot` and PoUB's `timing` and `codec`." New: "`verity.circuits`, `verity.commitments`, `verity.definitions`, `verity.hardware` (`registration`), `verity.protocols.spot.verifier` and PoUB's `timing` and `codec` (`verity.protocols.poub.verifier`)."
  Decided by: `verity/core/protocols` does not exist at the head. `datacenter/logical.py:49` `from verity.hardware import registration`; `binding.py:46,173,174` import `verity.protocols.spot.verifier.one_stage`; `examples.py:24` `from verity.protocols.poub.verifier import timing`; `memory.py:24` `from verity.protocols.poub.verifier.codec import BLOCK_BYTES`. [run: `git grep`] The scripts did not rewrite this line: it named the old module in words, not as a path.

`verity/protocols/certifier/device/README.md`, five sentences. The move put the certifier program inside this directory, at `program/lean/`:

- Opening. Old: "This directory is `lean/` and this file: it has no Python package and no console script." New: "This directory is `lean/`, the certifier program in `program/lean/` and this file: it has no Python package and no console script."
  Decided by: `ls verity/protocols/certifier/device` is `README.md`, `lean`, `program`. [run]
- Not here. Old: "the certifier program, the executable forms of the certifier's definitions, which `lean/` requires: it is the Lake package at `program/lean`, beside the certifier's Python. Nor the certifier's host programs, which run the certifier on a host in front of a server: they are in `benchmarks/network_traces`." New: "the certifier's host programs, which run the certifier on a host in front of a server: they are in `benchmarks/network_traces`."
  Decided by: `program/lean` is in this directory, and the certifier's Python is in `../verifier`, not beside it. [run]
- Lean, added: "`program/lean/` is the certifier program, the executable forms of the certifier's definitions: the Lake package `certifier_program`, whose library `NetworkCertifier` the packages at `lean`, `../verifier/lean` and `security/Proofs` require by path."
  Decided by: `device/program/lean/lakefile.toml` (package `certifier_program`, `lean_lib` `NetworkCertifier`); the tracked lakefiles whose `require` path resolves to it are `device/lean`, `verifier/lean` and `security/Proofs`, no other. [run: `tomllib` over every tracked `lakefile.toml`]
- Commands, added: "In `program/lean/`: `certifier-difftest`, `certifier-charge-difftest` and `certifier-window-difftest`, each `INSTANCES PYTHON N`."
  Decided by: the three `lean_exe` of that lakefile [run]; the argument form from each run's `args` in `program/lean/lean-audit.json`. [read]
- Scripts. Old: "… as `lean/lean-audit.json` says (`runs`). No code outside this directory imports the scripts; their tests are in `../verifier/tests`, which load them by path and hold what they write to that file's pins." New: "… as `lean/lean-audit.json` says (`runs`). `program/lean/scripts/` does the same for the program's difftests, as `program/lean/lean-audit.json` says. No code outside this directory imports the scripts; their tests are in `../verifier/tests`, which load them by path and hold what they write to those files' pins."
  Decided by: `program/lean/lean-audit.json` has `runs` for `NetworkCertifier.DifftestMain`, `.ChargeDifftestMain` and `.WindowDifftestMain`, each with `generate`, `data` and `args` [run: `json`]; `verifier/tests/test_network_certifier_{,charge_,window_}lean_difftest.py` load `program/lean/scripts/*.py` by `spec_from_file_location`, as the device's two tests do the device's. [run: `git grep`]
- Loads from. Old: "`lean/`, the packages at `program/lean`, `../verifier/lean`, `../../../hardware/world` and `../../../commitments/lms/lean`. The scripts, `verity.protocols.certifier.verifier`; `issuer_vectors.py` also `verity.commitments.crypto`." New: the same, and "`program/lean/`, no package" and "`program/lean/scripts/window_vectors.py` the prover side, `verity.protocols.certifier.prover`".
  Decided by: `program/lean/lakefile.toml` has no `require` [run]; `program/lean/scripts/window_vectors.py` imports `verity.protocols.certifier.prover.window_charge`, and the other four scripts import only the verifier (`issuer_vectors.py` also `verity.commitments.crypto.lms`). [run: `ast`]

`verity/protocols/certifier/prover/README.md`, four sentences:

- Opening, last sentence. Old: "rows read are in `verity/protocols/certifier`, with the ones they build on." New: "rows read are in `verity/protocols/certifier/verifier`, with the ones they build on."
  Decided by: `service.py`, `tie_service.py` and `served_rows.py` are in `verity/protocols/certifier/verifier/`. [run: `ls`]
- Importers, the last bullet. Old: "`verity/protocols/certifier`: its `lean/scripts/window_vectors.py`, `window_charge`; and its tests, every module, and they alone `advice_check`, `io_ids`, `compile_slice_v3` and `compile_slice_v5`. No module of that package imports this one." New, two bullets: "`verity/protocols/certifier/device`: its `program/lean/scripts/window_vectors.py`, `window_charge`;" and "`verity/protocols/certifier/verifier`: its tests, every module, and they alone `advice_check`, `io_ids`, `compile_slice_v3` and `compile_slice_v5`. No module of that package imports this one."
  Decided by: the importers of `verity.protocols.certifier.prover` outside it. Under `verity/protocols/certifier` they are that one script in `device/` and 21 test files in `verifier/tests`, which between them import all 15 modules; no non-test file of `verifier/` imports the prover; the four modules named are imported by no other directory. [run: `ast` over every tracked `.py`]
- Importers, `benchmarks/network_e2e`. Old: "… `compile_slice_v2`, `compile_slice_v4`;" New: "… `compile_slice_v2`, `compile_slice_v4`, `compile_slice_v7`;"
  Decided by: `benchmarks/network_e2e/encodings/tests/test_served_window.py` imports `compile_slice_v7`. [run] **Not the move's:** `compile_slice_v7.py` did not exist at the old head, it is `main`'s since. Same kind as `charge_chain` in #8742. Drop this one word if you want the mend to hold only the move.
- Loads from. Old: "… `verity.protocols.spot.verifier`, `verity.circuits` and `verity.commitments`." New: "… `verity.protocols.spot.verifier`, `verity.circuits`, `verity.commitments` and, for `memcheck` and `block_key`, `verity.definitions`."
  Decided by: the move's map sends `verity.ml.circuits.boolean.memcheck` and `verity.ml.commitments.gates.block_key` to `verity.definitions.…`; `cache_step_v3.py:99`, `compile_slice_v4.py:82`, `compile_slice_v5.py:89,95` and `compile_slice_v6.py:110,117` import them from there. [run: `git grep`]

`verity/protocols/inference_only/verifier/nci/README.md`, one sentence:

- Not here. Old: "They are in `../../../certifier/verifier/network/tests`, in the suite `verity/protocols`, with the tests of the certifier's Lean check." New: "They are in `../../../certifier/verifier/network/tests`, in the suite `verity/protocols/certifier/verifier/network`, with the tests of the certifier's Lean check."
  Decided by: `verity/protocols/certifier/verifier/network/pyproject.toml` has a pytest table and no `[project]`, and `suites.py:449` names such a suite by its path; `suites.py --list` prints `verity/protocols/certifier/verifier/network`. [run] Not ambiguous: the seven test files are in that one suite. The paragraph is also rewrapped: its first line was 145 characters.

**Every path token on the change's added lines, at the mended head: 89, and 89 resolve.** [run: my script, backticked path and module tokens on the added lines of `git diff 5c095bb4ab70 HEAD`; before the mend 77, of which `verity.core.protocols` and the prover's `lean/scripts/window_vectors.py` resolved nowhere] "this dir" is the README's directory; "under X" is the directory the token's own sentence names. `×n` is a token on n lines.

- `benchmarks/datacenter/README.md` (15):
  - root: `pyproject.toml`, `benchmarks/network_sim`, `verity/`, `benchmarks/network_sim/run.py`, `verity.circuits`, `verity.commitments`, `verity.definitions`, `verity.hardware`, `verity.protocols.spot.verifier`, `verity.protocols.poub.verifier`
  - under `benchmarks/network_sim`: `graph.py`, `mockdc.py`, `records.py`, `workload/`, `workload/accounting.py`
- `verity/protocols/certifier/device/README.md` (27):
  - root: `verity/protocols/certifier/device`, `security/Properties`, `security/Proofs`, `verity.protocols.certifier.verifier`, `verity.commitments.crypto`, `verity.protocols.certifier.prover`, `benchmarks/network_traces`
  - this dir: `lean/` ×4, `program/lean/` ×4, `../verifier/lean` ×2, `lean/scripts/`, `lean/lean-audit.json`, `program/lean/scripts/`, `program/lean/lean-audit.json`, `../verifier/tests`, `program/lean`, `../../../hardware/world`, `../../../commitments/lms/lean`, `program/lean/scripts/window_vectors.py`
  - under `lean/scripts`: `issuer_vectors.py`
- `verity/protocols/certifier/prover/README.md` (20):
  - root: `verity.protocols.certifier.prover` ×2, `verity/protocols/certifier/verifier` ×2, `adapters/vllm`, `benchmarks/network_traces`, `benchmarks/prove_path`, `benchmarks/network_e2e`, `benchmarks/mempool`, `verity/protocols/certifier/device`, `tools/verity/circuit_check`, `verity.protocols.certifier.verifier`, `verity.protocols.spot.verifier`, `verity.circuits`, `verity.commitments`, `verity.definitions`, `verity/protocols/certifier/verifier/tests`
  - under `verity/protocols/certifier/device`: `program/lean/scripts/window_vectors.py`
  - this dir, as a pattern: `<module>.circuit-check.py`, `<module>.circuit-check.json` (15 modules, each with both)
- `verity/protocols/inference_only/verifier/nci/README.md` (27):
  - this dir: `lean/` ×3, `../../../certifier/verifier/lean`, `lean/lakefile.toml`, `lean/lean-audit.json` ×2, `lean/scripts/nci_reference.py`, `../../../certifier/verifier/network/tests`
  - root: `verity/protocols/onsite/verifier`, `verity/protocols/remote/verifier`, `security/Properties`, `benchmarks/verifier_cost` ×3, `adapters/vllm` ×2, `demos/lifecycle`, `verity/protocols/flock/prover`, `benchmarks/one_stage/composite.py`, `verity/protocols/flock/verifier`, `verity/protocols/interface`, `verity/protocols/certifier/verifier/network`, `verity/protocols/remote/verifier/composition`
  - under `lean/scripts`: `desc_difftest_vectors.py`
  - under `../../../certifier/verifier/network/tests`: `test_nci_*.py` (6 files), `test_inference_only_protocol.py`
  - not paths, protocol ids: `verity/inference-only/v0` ×3, `verity/registration/v2`

**Left alone on purpose, each checked and true.**

- nci, Lean: the requirers of `nci_check` are the packages at `../../../certifier/verifier/lean`, `verity/protocols/onsite/verifier`, `verity/protocols/remote/verifier` and `security/Properties`, as it says. [run: lakefiles] Its first line is 129 characters, the one added line over 120. I left it because it is true and no test fails on it. [run: the three doc tests locally, 23 passed, no warning names these four files]
- nci, Loads from: `flock_verifier` at `verity/protocols/flock/verifier` and `core_protocol` at `verity/protocols/interface`, no certifier package. [run]
- nci, Scripts and Documents: `nci_reference.py` is loaded by path from `adapters/vllm`'s tests, `demos/lifecycle` and `benchmarks/verifier_cost`; `verity/protocols/flock/prover`'s tests load `desc_difftest_vectors.py`; composition's `tests/reference.py:84` loads `test_inference_only_protocol.py`. [run: `git grep`]
- prover: the other four importer bullets, the 15 sidecars, no `__main__`, no Lean, the tests' directory. [run: `ast`, `ls`]
- datacenter: importers are `benchmarks/network_sim` alone, with the modules and names it lists; `run.py` runs `datacenter.demo`; no workspace member. [run: `ast`, `grep`]
- device: `certifier_device`, `CertifierDevice`, its three executables, `security/Properties` its only requirer by path; no test names `certifier-device-smoke`. [run]

**For you to decide.**

- The device README now states the certifier program's package, executables and scripts. #8742's README of the verifier describes the same program under "Lean beside it". Both are true; the same facts are in two READMEs across two PRs. If you want the short form here, keep the opening and "Not here" edits and cut the four added clauses (Lean, Commands, Scripts, Loads from): the README is then true and says less about `program/`. [argued]
- `compile_slice_v7`: above.
- A line of `main`'s in the nci README, line 115, says "run by `../tests/test_nci_*.py`"; `verity/protocols/inference_only/verifier/tests` does not exist. [run] It is not this change's line, so I did not touch it. Its true form is `../../../certifier/verifier/network/tests/test_nci_*.py`.

**Not checked.** Sentences of these sections that name no path, package, suite or importer and that the move could not change (what each Definition or executable does, the document schemas' fields). The Lean names on the added lines exist (`Nci.CompliesRate`, `CompliesAtMost`, `checkDetail`, `DescCountsCK`) [run: `git grep -w`]; what the README says of them I did not check. Nothing outside these four files.

Evidence on my VM, in the artifacts: `8688-mend.diff`, `8688-interface-derivation.log` (the lakefiles, importers, loaders, suites and the 89 tokens).

CROSSB

[2026-11-11T06:51Z] a7eccfbfbdd0
ONSITEMAIN to coordinator: NET-331 sizing, whole. No code, run, push, Slack, mark or comment.

Note (the comment's text, then what I did not check): `lanes/network-accounting/20261111T0648Z-report-net331-onsite-under-an-hour.md`, synced at `9b390b0c21`.

**Answer.** The 10.9 h [run `r20261109-084127-4ca1`, exit 0, 39,216 s] are: 4.7 h composing the window's circuits before the draw, in one process; 2.6 h deciding the outcome and its negatives after the one session, on one core; 1.2 h the certifier's chain; 1.0 h cases and composite; 0.6 h waiting for the slot. Scheduling and settings bring it to 1.8 to 2.2 h [scaled] with no verdict, bound or check changed. Under an hour is not reachable that way.

**Largest first, none changing what is proved:**
- A. Compose before the demo into a compose cache that outlives the run: 15,200 to 16,900 s. Driver lines (proofs' file), 522 GB held on the node. The record marks each circuit `cached`.
- B. Negatives beside each other: about 8,500 s. `a5.py` (proofs').
- C. The chain beside V3c's audit, as `main`'s driver already does: 4,338 s. Three lines on the onsite branch.
- D. No slot wait: 2,205 s. The job reserved 650 GB and used 123 GiB resident.
- E. Cases beside each other: about 700 s.
- F. Then about 1.8 h of single-core Lean and Python is left (composite 1,726 s, cases 1,100, three registration decisions of 880, session 610, its Lean checks 476). Halving it is a guess.

**Daniel's, since the record changes:** negatives out of the demo (the record shows the accept and no refusals; saves nothing once B is done); a smaller window class (the only lever on what F leaves; saving a guess).

**The one number:** the follow-on's `wall_s`, 39,216 s, printed by `research inspect RUN`; 37,011 s after the slot.

**Method caveat.** The stage logs stay on the node. Every stage time is from the sampler's process records in the fetched run; stage names are my reading of the code against them.

**Least sure:** that A's cache hits across two serves (read, not run), and F.

**To settle, you decide:** (1) a read, seconds, no compute: the kept audit `summary.json` on cpu-12, fields `staging.before_draw_s`, `circuits.*.compose_s`, `timing.*`. (2) `records_audit.py warm --jobs 16` twice on the kept window, CPU node, 550 GB disk; a guess of 20 to 45 min then seconds; draws no coin.

**For you to strike if private:** I quoted the circuit files' bytes and the 128 links from the published verdict lines, and left out every window count. The served requests and steps are named without a figure, against the brief's ask.

The three reads are not started. `of_record` untouched.

