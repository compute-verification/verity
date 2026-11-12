---
id: top/20261112T1707Z-report-layout-rest-sized
campaign: top-coordination
lane: top
kind: report
status: done
repo: compute-verification/verity
origin: LAYOUTSIZE, worker of top (bc-af075d78-d5b9-590b-acbb-eb47d6f88f39); TOP-398; brief is top's message of Thu 12 Nov 8:57 AM PST
---

# The rest of the signed layout, sized (TOP-398), Thu 12 Nov 9:55 AM PST (17:55Z)

**Two moves can be staged today as certified pure moves: `fixtures/bench-instances/` to `benchmarks/numerical/bench-instances/` (84 files) and `verity/configuration/config/` back to `tools/config/` (9 files); three more certify in a trial but each needs a small hand-written pull request first, for the tables the mover does not rewrite (the rest of `verity/configuration/` onto the four files and `runtime/`, the Site types onto `verity/site.py`, and `verity/protocols/interface/` if it moves, which rides a Lean batch). The rest is code: the hardware loaders (267 lines) must find their data elsewhere before the data sheets move, nine protocol files import the runtime, ten files of Flock's prover and five of circuit-check load the served kernels' circuits (their move certifies but leaves two root tests red and conflicts with the vLLM refactor, #9364, in 434 files), and `adapters/torch/` does not exist (69 modules, 16,611 lines, reach torch and never vLLM). Decisions: Daniel's on the signed rules, which the wire, `errors.py` and `settings.py` break as written (files that must import nothing import them); proofs' on the served circuits, `interface/` and `world/`; the vLLM lead's on the torch package; top's on where `verity_config` and `fixtures/artifacts.json` go.**

`main` read: `da1bdbda0648db77753955a1985ea7d2fa46b7aa` (15:15Z). Its `AGENTS.md` is the one in my prompt. Nothing in the
repository was changed, no branch pushed, no pull request opened, nothing written to Linear or Slack.

How each figure is marked:

- **[run]**: a command I ran on that commit on my VM: my import scan, `git`, the mover (`tools/move/stage.py`) in a
  scratch clone outside the repository, or the scan functions of a root test called on a trial's tree.
- **[read]**: read by me in a file, an issue or a pull request.
- **[inferred]**: my conclusion from those. Not run.

Lean's branch is `cursor/lean-remote-merge-landing-fc8d` at `f3ffe4e2bb` (#9363, 1,588 files) [run]. A plain
`git merge-tree` of `main` with it already conflicts in one file, `tools/verity/check/public_boundary.toml` [run]; the
counts below include that file. An open branch crosses a landed move with `restack.py`, which reruns the map on the
branch, so a merge-tree count is what a plain merge would hit, not what the branch's owner has to resolve by hand
[read, `tools/move/README.md`].

## The trials

Each row is one draft map run by `stage.py` in a scratch clone at `main` (`--no-lock`, 58 to 139 seconds a map) [run].
"Boundary checks" are the 58 checks of `tests/test_boundaries.py` called as functions on the trial's tree (58 pass on
`main`). "New sites" is `tests/test_integration_boundaries.py`'s scan (0 on `main`). "Public" is
`tools/verity/check/public_boundary.py` on the tree (exit 0 on `main`, 8,372 public files).

| Trial | What moves | Files moved | Files rewritten | Certificate | Boundary checks | New sites | Public boundary | Conflicted with lean's branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | `verity/configuration/config/` to `tools/config/` | 9 | 89 | pure move, second run changes nothing | 58 of 58 | 0 | exit 0 | 5 files |
| B | `deploy/party.py`, `manifest.py`, `config.py` to `verity/party.py`, `protocol.py`, `config.py`; rest of `deploy/` to `verity/runtime/`; `wire/` to `verity/wire/` (a trial name) | 19 | 78 | pure move, after four `[split]` rows | 55 of 58 | 0 | 14 new edges | 6 files |
| C | all of `verity/definitions/` in one map | 751 | 1,741 | refused, 5 paths | not run | not run | not run | not run |
| C2 | the part a protocol names (`ops`, `silicon`, `kernels`, `private_circuits`, `evaluate`, `requirement`) to `verity/circuits/` | 234 | 804 | refused, 2 paths | 52 of 58 | 0 | 5 new edges | 13 files |
| C3 | the served part (`registry`, `numerics`, nine files, `tests`) to `adapters/vllm/verity_vllm/definitions/` | 493 | 1,149 | pure move, second run changes nothing | 57 of 58 | 24 | exit 0 | 10 files |
| D | the eight data sheets to `hardware/cards/`, `hardware/nodes/` | 8 | 14 | pure move, and it deleted both loaders | 58 of 58 | 0 | exit 0 | 1 file |
| E | `fixtures/bench-instances/` and `fixtures/artifacts.json` (to `tools/verity/artifacts.json`, a trial name) | 85 | 92 | refused, 3 paths | not run | not run | not run | not run |
| E2 | `fixtures/bench-instances/` to `benchmarks/numerical/bench-instances/` | 84 | 17 | pure move, second run changes nothing | 58 of 58 | 0 | exit 0 | 1 file |
| F | `verity/hardware/site/__init__.py` to `verity/site.py` | 1 | 22 | pure move, second run changes nothing | 56 of 58 | 0 | 2 new edges | 1 file |
| G | `verity/protocols/interface/` to `verity/interface/` (a trial name) | 21 | 64 | pure move, second run changes nothing | 58 of 58 | 0 | 19 new edges | 11 files |

Three things the trials show that hold for every move of this kind:

- **The mover does not rewrite a place written from `verity/`.** `PLACE`, `NO_UNIT` and the tree lists of
  `tests/test_boundaries.py` name places without the `verity/` prefix (`"configuration/deploy"`), and trials B, C2 and F
  left them as they were, so the unit checks fail on the moved tree [run]. Proofs listed `protocols/compose` in
  `CORE_TREES` ahead of its move [read]; each move of Python inside `verity/` needs the same hand-written pull request
  first.
- **The mover does not rewrite a brace list of `public_boundary.toml`.** The row
  `"verity/{claims,commitments,configuration,definitions,difftest,hardware,protocols}"` and the row
  `"verity/hardware/{...,cards,nodes,...}"` stayed, so the moved files left the public set in trials B, D, F and G [run]. That is a hand edit in ci's file; five open pull requests are
  about the boundary or the export (#9345, #8701, #8559, #8556, #9222) [read].
- **A certificate is not a green tree.** Trials B, C3, F and G certify and still fail a root check [run]. Trial D
  certifies, passes all three checks and breaks every import of the cards (below).

## Found on the way

**A certified pure move can delete code.** In trial D the map moved only the eight TOML files. The mover then deleted
`verity/hardware/cards/__init__.py` (113 lines) and `verity/hardware/nodes/__init__.py` (154 lines) as "`__init__.py`
left alone in a package the move empties", and `move_check.py` certified the commit as a pure move (8 renames, 15
path-only edits) [run]. On that tree `from verity.hardware import cards` is an ImportError; on `main` it lists six
cards [run]. Neither side looks at what the file holds: `layout.py`'s `_leftover` (line 591) takes any `__init__.py`
left in a directory that holds nothing else, and `move_check.py` accepts the deletion of one whose package the moves
emptied (line 1864) [read]. The fix is ci's (owner of `tools/move/` [run, the owners registry]): refuse to delete a non-empty one. Until then a
map that empties a package whose `__init__.py` holds code must give that file a row.

**LAYOUTMOVES' staged tip fails two boundary checks.** On `cursor/top-layout-moves-9b85` at `37c80625da` (17:12Z),
`test_every_source_belongs_to_a_unit_or_is_a_grouping` fails (`NO_UNIT` still names
`composite/device/live_challenger/...`) and `test_spot_never_imports_the_network_certifier_in_python` fails
(`verity/protocols/spot/device/live_challenger/coin_link.py:40` imports `verity.protocols.certifier.verifier.coin_link`)
[run, the same 58 checks called as functions]. Its checkpoint note already says the live challenger imports the
certifier; this is the check that refuses it.

## 1. `verity/configuration/` to four files and `verity/runtime/`

| What | Files | Lines | Goes to (TOP-398 section 7) | How |
| --- | --- | --- | --- | --- |
| `deploy/party.py` | 1 | 109 | `verity/party.py` | [run] count, [read] docstring |
| `deploy/manifest.py` (roles, channels, messages, placement) | 1 | 272 | `verity/protocol.py` | [run], [read] |
| `deploy/config.py` (`verity/config/v0`: check, digest, parse) | 1 | 407 | `verity/config.py` | [run], [read] |
| `deploy/` runtime, channels, sources, stream, transcript, process | 6 | 1,621 | `verity/runtime/` | [run], [read] |
| `deploy/` tests, README, pyproject, `__init__.py` | 6 | 762 | with `runtime/` | [run] |
| `wire/` (frames, handshake, canonical JSON, message schemas) | 4 | 706 | no place in the signed tree | [run], [read] |
| `config/` (`verity_config`: environment settings, a strict TOML loader, GPU names) | 9 | 672 | no place in the signed tree; it was `tools/config/` before #8769 | [run], [read] |
| `verity/configuration/__init__.py` | 1 | 0 | goes with the directory | [run] |
| Total | 29 | 4,549 | | [run] |

Importers outside the directory, by my import scan [run]:

| Part | Non-test files | Test files | By top-level directory (non-test / test) |
| --- | --- | --- | --- |
| `verity_config` | 80 | 10 | benchmarks 78 / 8, verity 1 / 1, adapters 1 / 1 |
| `deploy` | 15 | 14 | verity 9 / 9, demos 4 / 0, tools 1 / 3, benchmarks 1 / 2 |
| `wire` | 11 | 8 | verity 7 / 5, demos 3 / 0, tools 1 / 3 |

**As a move.** Trial A certifies and passes every check I ran: `verity_config` imports from `tools/config/`, the
workspace member is rewritten, the public set is unchanged [run]. Trial B certifies too, with four `[split]` rows for
references to `verity/configuration/deploy` in prose and in `drive.toml` [run]. After it three boundary checks fail
(`core.deploy` and `core.wire` have no source; `config.py` is outside every unit) and 14 edges cross the public
boundary [run].

**Code and decisions first.**

- *The wire has no place, and the rules cannot hold with it.* `party.py` imports `KINDS`, `Message` and the handshake's
  `Key` from it, `manifest.py` imports `Schema` and `MessageTypes`, `config.py` imports `Refused`, and the Site imports
  `canonical` [read]. The signed rules say `party.py` and `site.py` import nothing. Either the rules gain a line (a
  wire module at the bottom that imports nothing) or the wire is split into the four files. Daniel's rule, so his
  decision; the first is one line and no code.
- *`verity_config` has no place.* 78 of its 80 importers are benchmarks [run]. `tools/config/` is where it was. Top's
  decision.
- *Nine protocol files import the runtime* (10 imports; section 7) [run]. Four import `runtime.ProtocolDef`, an
  eight-line dataclass (`runtime.py:72`) that belongs in `protocol.py` [read]. Six files of PoUS and PoUB import
  `stream` to run a party over a byte stream, which the rule gives to "what starts a run" [read]. That is
  memory-accounting's code to change.

**Lean.** No Lake package is inside, and neither trial touches a manifest, a lakefile or a lock [run].

**Conflicts.** Lean's branch changes 3 files under the directory, and 14 (trial A) and 37 (trial B) of the files a
trial touches [run]. Proofs' two maps touch 3 to 4 of trial A's files and 9 to 11 of trial B's, all lists and
allowlists both moves rewrite (`pyproject.toml`, `drive.toml`, `moves.json`, the retired-terms list) [run]. The public
boundary has one row for `verity/configuration/config`, which trial A rewrites [run].

**Owners.** memory-accounting 15 files (`deploy/`), ci 9 (`config/`), proofs 5 (`wire/`, by the `verity/` row) [run].

**Smallest sequence.**

1. A map moves `verity/configuration/config/` to `tools/config/` (trial A; today, once top says that is its place).
2. A hand-written pull request lists the new places in `tests/test_boundaries.py` and `public_boundary.toml` beside
   the old ones.
3. A map moves the rest (trial B), with the wire where Daniel's ruling puts it.
4. A code pull request moves `ProtocolDef` into `protocol.py` and takes `stream` out of the six PoUS and PoUB files,
   and adds the rule's check.

## 2. `verity/hardware/` to a top-level `hardware/` and `verity/site.py`

| What | Files | Lines | Goes to | How |
| --- | --- | --- | --- | --- |
| Six card and two node data sheets (TOML) | 8 | 498 | `hardware/` | [run] |
| Loaders `cards/__init__.py`, `nodes/__init__.py` | 2 | 267 | stay code; they read the TOML beside them (`Path(__file__).parent`) | [run], [read] |
| `site/__init__.py` (the Site types and their bytes) | 1 | 818 | `verity/site.py` | [run], [read] |
| `registration.py`, `record.py` | 2 | 408 | `verity/site.py` if "one file" is meant literally | [run], [inferred] |
| `lean.py` (writes the Lean card tables under `security/Definitions/Core/Accel`) | 1 | 214 | not placed | [run], [read] |
| `world/` (Lake package `core_world`, `World.lean`) | 6 | 98 (70 of Lean) | not placed | [run] |
| Tests, README, pyprojects, `__init__.py` | 10 | 1,800 | with what they test | [run] |
| Total | 30 | 4,103 | | [run] |

Importers outside the directory: 15 non-test files and 11 test files (benchmarks 13 / 5, verity 1 / 5, adapters 1 / 1);
the cards' loader is imported by 16 files, the Site by 6, registration by 6, the record by 2 [run].

**As a move.** Not yet. Trial D is the hazard above: the data sheets move, the loaders are deleted, the certificate
passes. Trial F (the Site alone to `verity/site.py`) certifies; two boundary checks then fail (`core.site` has no
source) and two edges cross the public boundary [run].

**Code and decisions first.**

- The loaders must find the data outside their package. The smallest change is two lines (`CARDS`, `NODES`) and two
  messages [read]. The signed picture says the data is "passed in by a run", which is a larger change: the 12
  non-test files that import the loaders (benchmarks 11, the vLLM adapter 1) would pass a directory [run, inferred].
  compute-research owns the loaders; which reading is meant is top's to ask.
- "One file" for the types means folding registration and the record into the Site by hand (1,226 lines together), or
  leaving them as files of their own. Proofs' decision.
- `world/` is neither data nor a Site type. It is required by the certifier device's package and named in the
  manifests of `security/Proofs` and `security/Properties` [run]. Proofs' decision; its move rides a Lean batch.
- `lean.py` and the two Lean tables are section 9's open point (CMP-392) [read].

**Lean.** Only `world/`: its own lock, one lakefile that requires it and three manifests that name it [run]. Trial F
rewrites a module name in a comment of one lakefile (`verity/protocols/spot/verifier/lean/lakefile.toml`), so even the
Site's move is a Lean input [run].

**Conflicts.** Lean's branch changes one file under the directory and none under `cards/`, `nodes/` or `site/` [run].
Proofs' maps touch `verity/hardware/registration.py` (a path in it) [run]. The public boundary has four rows here: the
brace list that makes `cards` and `nodes` public, two private tests, and `verity/hardware/world` as needed by the
certifier [read]. Open pull requests #9336, #9308 and #9188 touch the directory [read].

**Owners.** compute-research 19 files (cards, nodes, `lean.py`, tests), proofs 11 (Site, `world/`, registration,
record) [run].

**Smallest sequence.**

1. A code pull request makes the loaders read a directory outside their package and gives each loader a file name the
   mover will not delete.
2. A map moves the eight data sheets to `hardware/`, with the boundary's brace list edited by hand in the same batch.
3. A hand-written pull request lists `site.py` in the unit table, then a map moves the Site (trial F).
4. A Lean batch moves `world/` to where proofs decides.

## 3. `verity/definitions/` split between `verity/circuits/` and the vLLM adapter

| Part | Files | Python lines | How |
| --- | --- | --- | --- |
| Stays in `verity/`: `ops` 154, `evaluate` 26, `silicon` 25, `circuits` 18, `kernels` 11, `requirement` 11, `commitments` 7, `private_circuits` 7 | 259 | 30,485 | [run] |
| Served: `registry` 364, `numerics` 67, `tests` 53, twelve top-level files | 496 | 73,718 | [run] |
| Total | 755 | 104,203 | [run] |

TOP-398 says 324 files for the served part; `registry/` has 275 Python files and 364 files in all [run].

Importers from outside `verity/definitions/` [run]:

| Part | Non-test files | Test files | By top-level directory (non-test / test) |
| --- | --- | --- | --- |
| Served | 243 | 448 | adapters/vllm 182 / 391, verity/protocols 10 / 22, tools/verity 5 / 10, benchmarks the rest |
| Stays | 154 | 250 | verity 65 / 103, benchmarks 61 / 40, adapters 27 / 93, tools 1 / 13 |

No non-test file of the part that stays imports the served part [run]. The ten non-test files under `verity/` that
import the served part are all in Flock's prover (`verity_flock/`: `boolean_export`, `class_statement`, `ir_lower`,
`ir_sampling`, `partition_units`, `subcircuit_templates`, `tail`, `tail_pieces`, `templates/silu_mul_v2`, `topp_word`);
the five under `tools/` are circuit-check's `cache`, `checks`, `sidecars`, `targets` and `units` [run].

**As a move.**

- *One map for all of it is refused* on 5 paths (trial C): merging `definitions/circuits/` into `circuits/` gives a
  duplicated prefix in `research/queue.py`, a collision of two `__init__.py` in `moves.json` and a glob in
  `public_boundary.toml`; a README's links and one test's anchor are the other two [run].
- *The part that stays* (trial C2, without the two directories that merge) is refused on 2 paths: three links in
  `verity/circuits/README.md` and `CIRCUITS = Path(instructions.__file__).resolve().parents[1]` in
  `ops/tests/test_evidence_ids.py` [run]. On its tree 6 boundary checks fail: four units have no source, `circuits`
  is held numpy-free and `ops` uses numpy, and an exception list names a test of `requirement/` by its old path [run].
- *The served part* (trial C3) certifies. Then 24 import sites in 13 files under `verity/` are not in `KNOWN`
  (two non-test, both in Flock's prover; the rest tests of flock, circuits, the certifier and spot), and
  `test_circuit_check_imports_no_workspace_distribution_but_verity` fails on four files [run]. Both lists only shrink
  [read].

**Code and decisions first.**

- The served part: either Flock's prover (10 files) and circuit-check (5 files) stop loading the served circuits,
  which TOP-398 section 8 lists as work to build, or a ruling lets `KNOWN` and `CIRCUIT_CHECK_IMPORTS` grow by the
  sites the move exposes. 37 of `KNOWN`'s 44 entries already name `verity.definitions.*`, left from before #8769, so most of
  the debt is recorded, not new [run]. Proofs' decision with circuits.
- The part that stays: a hand-written pull request for the two refused hunks, the unit table, the numpy list and the
  boundary rows; then `definitions/circuits/` and `definitions/commitments/` by hand or a second map.

**Lean.** Two Lake packages are inside (`verity_evaluate`, `core_requirement`). Trial C2 moves 30 Lean inputs and edits
four in place: `security/Properties/lakefile.toml`, the manifests of `security/Proofs` and `security/Properties`, and a
`why` text in `security/lean-audit.json` [run]. Trial C3 edits one lakefile (`verity/protocols/flock/sigma`) [run].
Both ride a Lean batch.

**Conflicts.**

| Against | Part that stays (C2) | Served part (C3) | How |
| --- | --- | --- | --- |
| Lean's branch: files it changes that the move touches | 81 | 57 | [run] |
| Lean's branch: files it changes or adds under the moved paths | 6 | 12 (7 added) | [run] |
| Lean's branch: conflicted files in a plain merge | 13 | 10 | [run] |
| The vLLM refactor (#9364, 1,463 files, all under `adapters/vllm`): conflicted files in a plain merge | not run | 434 (3 on `main` alone) | [run] |
| LAYOUTMOVES' branch: files in common | 35 | 11 | [run] |
| Proofs' two maps: files in common | 11 to 13 | 4 | [run] |

The public boundary names `verity/definitions` in three rows (`requirement_units*`, `{evaluate,requirement}` and the
brace list) [read].

**Owners.** proofs 590 files (mostly by the `verity/` row), circuits 154 (`ops`), memory-accounting 11 [run].

**Smallest sequence.**

1. A hand-written pull request fixes the two refused hunks and lists the new places.
2. A map in a Lean batch moves the six directories that stay, after LAYOUTMOVES' difftest move, which rewrites the
   same lakefiles.
3. A hand-written pull request merges `definitions/circuits/` and `definitions/commitments/` into their targets.
4. After #9364 and lean's merge land, and after proofs' decision, a map moves the served part (trial C3).

## 4. `adapters/torch/` beside `adapters/vllm/`

`adapters/torch/` does not exist. `adapters/vllm/` has 2,073 files; `verity_vllm` has 743 modules and 197,189 lines
[run].

| Count, following every import statement | Modules | Lines | How |
| --- | --- | --- | --- |
| Name `vllm` directly | 51 | | [run] |
| Name `torch` directly | 186 | | [run] |
| Never reach `vllm` | 363 | 64,969 | [run] |
| of those, reach `torch` | 69 | 16,611 | [run] |
| of those, reach neither | 294 | 48,358 | [run] |
| Never reach `vllm` when a parent package's `__init__.py` counts | 181 | 21,294 | [run] |

TOP-398 has 355 of 750 and 67 [read]; the gap is two days of the refactor. The 363 are spread over every subpackage
(`program` 80 of 179, `pipeline` 46 of 149, `check` 44 of 78, `protocols` 33 of 63, `observe` 31 of 60, `engine` 31
of 55, `commit` 26 of 39, `acquire` 20 of 49, `records` 19 of 19, `query` 13 of 16, `properties` 9 of 18) [run].

**As a move.** No. There is no package to move into, and a module's parent `__init__.py` pulls vLLM in for 182 of the
363 [run]. Once a package exists and its boundary is drawn, the modules that move can go by a map [inferred].

**Code and decisions first.** The vLLM lead decides which modules are torch's (the 69, or the 363) and where the 294
that load neither belong; then `adapters/torch/pyproject.toml`, the package, the workspace member, the owners row and
the boundary rows are written by hand. `tests/test_boundaries.py` already names `verity_torch` among the
distributions `verity` may not import [read].

**Lean.** None [run].

**Conflicts.** The refactor's snapshot (#9364) changes 1,463 files under `adapters/vllm`; lean's branch changes 284
there [run]. `adapters/vllm` is in 54 rows of the public boundary [run].

**Owners.** vllm 1,189 files, circuits 854, compute-accounting 28 [run].

**Smallest sequence.**

1. The vLLM lead states the boundary, after the refactor's snapshot lands.
2. A hand-written pull request adds the empty `adapters/torch/` distribution and its rows.
3. A code pull request makes the package `__init__.py` files stop importing vLLM for the modules that move.
4. A map moves them, and `verity_vllm` depends on the new distribution.

## 5. `fixtures/` gone

228 tracked files, about 3.9 MB [run].

| What | Files | Bytes | Reader | How |
| --- | --- | --- | --- | --- |
| `bench-instances/` | 84 | 1,012,333 | `verity_numerical.bench.instances` (`DEFAULT_ROOT`), `benchmarks/tc_probe/`, three scripts of `verity/protocols/flock/pod/` | [run] |
| `typed-obligation-v0/` | 141 | 2,692,438 | none found by name for the tracked vectors; the one code reader reads a fetched file | [run] |
| `proof-format-v3/` | 2 | 144,035 | none: one docstring and one test's exception list name it | [run] |
| `artifacts.json` (147 `art:` ids, 6 of them for files under `fixtures/`) | 1 | | named in about 60 files: `tools/verity/check` 10, Flock 9, `tools/research` 15, the vLLM adapter's tests 8, root tests 5 | [run] |

**As a move.** Trial E2 certifies and passes every check: the reader's root exists with 84 files, and 15 other files are
rewritten (the readers, `census/input_sets.json`, the pod scripts, `artifacts.json`, the owners registry, the Glossary)
[run]. Trial E, with
`artifacts.json` too, is refused on 3 paths, all one expression: `Path("fixtures") / "artifacts.json"` in the store's
replica map, in a test and in a comment, which the mover rewrote to a path of three parts [run]. A destination of
two parts, or a one-line change to that expression first, removes the refusal [inferred].

**Code and decisions first.** Where `artifacts.json` goes is top's (its owner). The public boundary lists it under
`structure`, so the place must be one the export keeps [read]. The two directories without a reader are deletions,
not moves; proofs confirms nobody reads them and deletes them with their `art:` rows.

**Lean.** None [run].

**Conflicts.** Lean's branch changes 3 of the files trial E2 touches and nothing under `fixtures/bench-instances/`;
proofs' maps share `moves.json` and the owners registry; no boundary file is edited [run].

**Owners.** proofs 143 files, compute-accounting 84 (`bench-instances/`), top 1 (`artifacts.json`) [run].

**Smallest sequence.**

1. A map moves `fixtures/bench-instances/` beside its reader (trial E2; today).
2. Proofs deletes `proof-format-v3/` and `typed-obligation-v0/`, or names their readers.
3. A map moves `artifacts.json` to the place top names.

## 6. The five pieces the Map does not list

| Piece | What it is | Size | Non-test / test importers | Where the signed rules put it | How |
| --- | --- | --- | --- | --- | --- |
| `verity/claims/` | the security claims' ids a render cites; standard library only | 3 files, 599 lines of code | 2 / 11 (`pous/verifier/protocol.py`, `spot/device/challenger.py`) | no place named; nearest is `protocol.py` ("what a protocol declares") | [run], [read], [inferred] |
| `verity/errors.py` | three exception classes | 26 lines | 20 / 22 (commitments 8, adapters 4, hardware 3, protocols 3, definitions 2) | nowhere as written: `commitments/` may import only `circuits/`, and it imports this in 8 files | [run], [inferred] |
| `verity/settings.py` | the environment-settings machinery and `verity`'s own registry | 447 lines | 1 / 11 (`verity_config/settings.py`) | no place named; with `verity_config`, since no unit of `verity` imports it | [run], [inferred] |
| `verity/protocols/reporting.py` | the reporting grid: three periods and two δs | 15 lines | 14 / 4 (benchmarks 10, protocols 3, demos 1) | not a protocol, so not in the flat list; `protocol.py` | [run], [inferred] |
| `verity/protocols/interface/` | Lake package `core_protocol`: the Lean interface each protocol's check implements, with the Site and the registration check in Lean | 21 files | required by 12 lakefiles, named in 16 manifests | not a protocol; below Flock, since it requires only the base package | [run], [read], [inferred] |

Three more are in the same position and were not in the brief: the wire and `verity_config` (section 1) and
`verity/hardware/world/` (section 2).

**As moves.** `interface/` certifies as a move to a directory of its own (trial G), touching 17 Lake packages'
lakefiles or manifests and no lock record; 19 edges then cross the public boundary [run]. Lean's branch adds three
files under it and changes six, and a plain merge conflicts in 11 files [run]. `claims/` and `reporting.py` to flat
files are the shape of trial F [inferred]. Folding `reporting.py` into `protocol.py` is a 15-line hand edit that waits
for section 1.

**Decisions.** All five are proofs' files [run]. The smallest answers: `interface/` stays where it is and the rule
says it sits under Flock (proofs; no work); `errors.py` and `settings.py` get one line in the rules, "import nothing,
and anything may import them" (Daniel's rules); `claims/` stays a package with the same line, or becomes a flat file
(proofs); `reporting.py` goes into `protocol.py` after section 1.

## 7. The rules that are not enforced

An edge is one (file, imported module) pair, a `from P import sub` counted once. The scan reads every tracked `.py`
(5,820 files, none unparsed), follows import statements and strings handed to an import call, and reads Lake `require`
paths for the Lean side [run].

    cp -r ~/.research/notes/lanes/top/tools/layoutsize /tmp/layoutsize   # the scripts read /tmp/layoutsize
    cd <a checkout of main> && python3 /tmp/layoutsize/scan.py && python3 /tmp/layoutsize/rules.py -v

| Rule | Edges in non-test files | In test files | Where | How |
| --- | --- | --- | --- | --- |
| The runtime imports no protocol | 0 | 0 | | [run] |
| No protocol imports the runtime | 10 in 9 files | 12 in 8 files | PoUS 6 files, PoUB 2, spot 1 | [run] |
| No adapter imports the runtime | 0 | 0 | | [run] |
| `circuits/` imports nothing | 1 in 1 file | 70 in 25 files | `population.circuit-check.py` imports `verity.definitions` | [run] |
| the same, with the part of `definitions/` that stays counted in | 8 in 5 files | 45 in 14 files | `census.py` imports PoUW's prover; two gate files import `commitments`; `silicon/errors.py` imports `verity.errors` | [run] |
| `commitments/` imports only `circuits/` | 9 in 9 files | 24 in 15 files | 8 import `verity.errors`; `windows/interface.py` imports `verity.definitions.ops` | [run] |
| No code imports `security/` | 0 Lake packages outside it require one of its packages | | | [run] |
| No code imports the hardware data | 12 in 12 files import the loaders | 4 in 4 files | benchmarks 11, the vLLM adapter 1; 18 non-test files name `hardware/cards` or `nodes` by path | [run] |
| Flock imports no other protocol | 27 in 14 files | 61 in 35 files | Flock's prover (`rec_*`, `inner_registered`, `pod/rec_inner`, `pod/rec_reprice`) imports spot's `one_stage` and `device`; Lean 0 | [run] |
| Spot and recursion import no audit and not compose | 0 | 4 in 2 files | Lean 0 | [run] |
| No audit imports compose | 0 | 0 | Lean 0 | [run] |
| No protocol imports a configuration (remote, onsite) | 0 | 0 | Lean 0 | [run] |

Two more counts the rules do not state: an audit imports another audit in 11 edges of 6 non-test files (PoUB imports
PoUS) and in two Lake requirements (`network_check` requires `nci_check`, `remote_audit` requires `pous_check`); and
the four flat files as they stand break their own rules only through the wire (`party.py` 2 edges, the Site 1,
`manifest.py` 1, `config.py` 1) [run].

**Smallest sequence.** One pull request adds each rule as a check with today's offenders as a list that only shrinks,
in full paths so a move rewrites them (#8656, "Import order: one listing", is the open pull request this fits)
[inferred]. Emptying the lists is the code of sections 1 and 2, the errors line of section 6, and Flock's 27 imports
of spot, which is proofs' and the largest.

## Ranked by how soon each could land

1. **`fixtures/` (5).** One move can be staged today; the rest is one decision of top's and two deletions.
2. **`verity/configuration/` (1).** One move can be staged today once top places `verity_config`; the second needs a
   hand-written pull request and Daniel's line on the wire; the runtime rule is code after that.
3. **The import rules (7).** A check with shrinking lists can land this week; empty lists wait for the code.
4. **The unlisted pieces (6).** No code beyond 15 lines; it waits on two rule lines and on section 1.
5. **`verity/hardware/` (2).** Loader code first, then two small moves; `world/` rides a Lean batch.
6. **`verity/definitions/` (3).** The part that stays needs a hand-written pull request and a Lean batch; the served
   part waits for a decision, for #9364 and for lean's merge.
7. **`adapters/torch/` (4).** New code in the files the refactor is rewriting; last.

Stageable today as a pure move: `fixtures/bench-instances/` to `benchmarks/numerical/bench-instances/` (trial E2) and
`verity/configuration/config/` to `tools/config/` (trial A). Not to be staged as tried: the data sheets (trial D).

## What is beside this note

`lanes/top/tools/layoutsize/`: the scan (`scan.py`, `roots.py`, `rules.py`, `q.py`), the Lake reader (`lake.py`), the
owners count (`owners.py`), the adapter's reach count (`vllm_split.py`), the trial runner (`trial.sh`), the overlap
count (`overlap.py`), the two root tests' scans called as functions (`sites.py`, `bound.py`), the ten draft maps
(`maps/`) and each trial's log and the rule scan's full output (`out/`). The maps are drafts for a scratch clone: their
`[split]` rows were acknowledged without being read, and none was committed to the repository.
