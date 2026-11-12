# tools/verity/lean: the Lean audit

One audit for every Lake package (`python tools/verity/lean/check.py --all --build`), which `check` runs on the packages
whose inputs changed (`check.py --build PKG...`), and Lean's job in `check`. This file is an outline: what the audit is,
its commands, what it checks, the lock's sections, and one line per module. The detail is in the docstring each line
names, and new detail goes there: here it gets one line at most.

No package code runs where the audit decides anything. The facts come from the `.olean` files read as data
(`Facts.lean`), and the kernel replay (`Replay.lean`) loads them the same way. The only package code that runs is the
build's and, after the facts and the replay, the tests' under `runs` (each deciding only its own pass), in a sandbox
where the machine allows one; a test's data generator is the repository's code and runs outside it.

A weakened definition leaves every proof valid, so the axiom checks and the replay pass it: only the records see it, and
only a person can judge it. So the lock, `lean-audit.json`, records what each guarantee says and every definition its
statement reads, the audit fails on any difference, and a changed record is shown when it lands.

## Commands

~~~text
bash tools/verity/lean/setup.sh PKG
  elan, PKG's lean-toolchain, its pinned dependencies, Mathlib's cache when Mathlib is a dependency, and main's cached
  build of PKG (`cache.py`)
python tools/verity/lean/cache.py get [PKG...]
  main's cached build of each package, every file sha256-checked
python tools/verity/lean/check.py [--build] PKG...
  audit these packages (--build: setup.sh and lake build first)
python tools/verity/lean/check.py --update PKG...
  write the guarantees' records from the build; commit the diff (a copy of each record goes beside its reports, which a
  recorded run publishes; refused in a tree `research run --on` shipped: use --cwd clone)
python tools/verity/lean/check.py --update --owner @lane PKG...
  the same, with @lane as the owner of each new guarantee
python tools/verity/lean/check.py --update --moved MOVES.json PKG
  after a move, on the record as it was before it
python tools/verity/lean/check.py --all [--root DIR]
  the controls, then every package under DIR with a lean-audit.json
python tools/verity/lean/check.py --name-instances PKG...
  write each anonymous instance's name, the one Lean gave it, into its declaration (no record changes); it fails until
  you build and audit again
python tools/verity/lean/check.py --lift LIFTS.json PKG
  write each guarantee {"X.t": {"as": "X.Guarantees.G", "into": MODULE}} as `def G : Prop` in MODULE and `theorem
  X.SecurityProofs.G : X.Guarantees.G := @X.t` beside its proof (or in the module `"beside": M` names); then build and
  run `--update --moved LIFTS.json.moved.json PKG`
python tools/verity/lean/upstream.py PKG [--lean]
  what the pinned dependencies prove about PKG's watched statements
python tools/verity/lean/unread.py --out DIR --build
  the Lean nothing reads; lean-audit runs its ratchet on the audits' facts (its docstring: the seeds, the caps, the
  Properties gap)
python tools/verity/lean/merge.py --install
  once per clone: git merges lean-audit.json with merge.py, by origin/main's copy of it (fetch main first)
verity lean try FILE [--siblings] [--on NODE]
  the edit loop: `lake env lean FILE` in FILE's package against its built imports, the axiom probe's line last; neither
  runs this audit nor writes a record (`verity lean try --help`, `lean_changed.py`). `--siblings` checks each sibling
  as it is built: one built against an older build of an import is rebuilt when a refusal names it, and not otherwise
research run --on auto --mem-gb 32 --max-min 120 --project verity --source . --cwd clone \
  --declared-output 'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records PKG --update
  the fast path (the lean-proofs skill): PKG's records and its requirers'
research run --on auto --mem-gb 32 --max-min 180 --project verity --source . --cwd clone -- \
  python3 tools/verity/lean/check.py --build --update PKG
  a full record, left at lean-audit/audit-TAG/lean-audit.json
~~~

`check.py`'s docstring and `--help` have the other flags (`--closure`, `--replay-cache`, `--artifacts`, `--regenerate`,
`--bookkeeping warn`, `--facts`, the facts stages); `lean_audit.py`'s and `lean_changed.py`'s have the jobs' own.
`check.py` exits 0 when every package passes, 1 when one fails, and 2 on a usage error or a refusal.

`--fresh` also runs `leanchecker --fresh` on each root: the root's whole import closure, Mathlib and core included,
through the kernel into an empty environment. It needs one module per root and a lot of memory, and checks the closure
again on every run, so it runs nightly on `main` instead (`[[run]]` in `tools/research/README.md`, `tool.py`). `--closure`
with a kept cache, which `check` runs, checks the same closure once per change of its `.olean` files.

## What it checks

Every declaration of the package's audited modules (not a list of theorems), from the environment Lean builds. A failure
carries a code (`Failure`; `failure_codes` in `report.json`). Each line names where its rule is: a function or constant
of `check.py` unless it says otherwise, or the lock's section below.

- **Axioms**: only the allowed ones (default `propext`, `Classical.choice`, `Quot.sound`), so no `sorry` and no
  `native_decide`, and no `axiom` declaration: an assumption is a named `Prop`, taken as a hypothesis (`judge`).
- **`unreported`, `dropped`**: no `sorry` the elaborator made for an error it did not report, and every top-level
  declaration of a source lies in a range its `.olean` holds, in an audited module or a `proved_in` package's.
  `unreported` also fails `lean_changed.py --records`, where the axioms check doesn't run.
- **Kernel bypasses and compile-time code**, in the sources and the lakefile (`scans`, `UNCHECKED`, `COMPILE_TIME`,
  `listings`) and in what the build takes from fetched packages (`dependency_scans`); the fetched `.olean` files are
  the recorded ones (`dependency_digests`).
- **The lock's sections** hold, each as its line below says: `exempt`, `escapes`, `layers`, `trees`, the spec, the
  premises, `trust`, `verifier`, `claims`, `witnesses`, `shape`, `four_names`, `upstream`.
- **The guarantees say what their records say**, and each names its owner (`judge`).
- **Every instance has a name in its source** (`anonymous_failures`). Lean names an anonymous one from its type, and
  adds the project's root (`_proofs`) when nothing in the type is the project's own, so a move into another root or a
  split renames it, and every record whose definitions use it changes. `--name-instances` writes each current name.
- **The kernel accepts every declaration again** (`replay_step`, `Replay.lean`), and the axioms the replay finds are
  the ones the `.olean` files recorded; a replay killed by a signal is a kill, not a verdict (`killed`). `--closure`
  checks the whole import closure the same way, reusing each module that passed under its key, whichever package
  checked it (`closure_plan`, `replay_keys`, `build_roots_over`); the audit prints `closure: N modules checked, M
  reused` under each package's line.
- **`--build` recompiles a sample** of the modules the build didn't compile and compares the outputs
  (`sampled_recompile`, `olean_graph.py`).
- **Each test under `runs` passes** on the data whose SHA-256 the record holds (`tests`, `run_paths`).
- **Nothing is tampered with**: nothing changes this directory, the toolchain's `lean`, `lake` and `leanchecker`, or
  the `.olean` files of a package audited earlier while the audit runs: the Lean files run from a copy taken before the
  first build, and everything is sealed and checked after each package (`tampered`, `Seals`).

Where it runs: `--build` builds in the sandbox where the machine allows one (`sandbox.sh`, `sandboxed`), after taking
one of the node's Lean audit slots (`lean_slot`), and is refused below a package's memory floor (`memory_refusal`,
`MEM_FLOOR_GIB`, `SPEC_FLOOR_GIB`), with threads as its memory holds (`build_threads`, `replay_threads`). Packages are
audited side by side, each build after those of the packages it requires by path (`setup`, `writing`, `Seals`). `--all`
first runs the controls, which test the auditor on fixtures that must pass clean and fail once per violation
(`controls`; `--no-controls` skips them).

## The lock: `lean-audit.json`, beside the lakefile

The package's policy and its records, written in `check.py`'s `ORDER` at schema `FORMAT`; `check.py` prints a template
when a package has none (`template`).

~~~json
{
 "roots": ["Pkg"],
 "exempt": {"Pkg.Scratch": "why nothing needs to build it", "PkgBulk": "a test, not a proof"},
 "runs": {"PkgBulk": {"why": "what it tests", "args": ["{bulk.txt}", "1000"], "data": {"bulk.txt": "…"},
                      "generate": ["python3", "scripts/bulk.py", "{repo}", "--out", "{out}"]}},
 "axioms": ["Classical.choice", "Quot.sound", "propext"],
 "assumptions": {"Pkg.Hyp": {"owner": "@lane"}},
 "meaning": ["Pkg"],
 "escapes": {"Pkg.loop": "why compiled code no proof sees is acceptable here"},
 "compile_time": {"Pkg.Tactic": {"why": "why this module must run code while it compiles", "digest": "…"}},
 "dependency_unchecked": {"Dep.Loader": "why the package imports a dependency module that names a kernel bypass"},
 "layers": {"Pkg.Model": {"imports": ["Mathlib", "Pkg.Model"], "except": [], "why": "definitions import no proofs"}},
 "reads_exempt": {"Pkg.Exe": "why a guarantee may read these definitions though they are not spec"},
 "trees": {"definitions": ["Pkg.Model"], "properties": ["Pkg.Properties"], "proofs": ["Pkg.Proofs"],
           "except": {"Pkg.Proofs.Old": "why a statement may read it until its definitions move"}},
 "guarantees": {"Pkg.main_theorem": {"owner": "@lane"}}
}
~~~

Each section, with the function of `check.py` whose docstring has its rule. A list that "only shrinks" is one `--update`
keeps or drops from and never adds to.

- `roots`: the modules the audit imports; every module under them is audited. `exempt`: the module prefixes nothing
  imports, each with the reason; every other `.lean` file of the package is imported by a root, and an entry that
  matches no file fails (`judge`).
- `proved_in` (optional): the packages, as paths relative to this one, that prove its guarantees (below).
- `runs`: the package's tests, compiled evaluation that no proof reads (a bulk conformance check over more vectors
  than the kernel checks), each an exempt module that passes when it exits 0, on data used only with the SHA-256
  under `data` (`tests`, `run_paths`; `--no-runs` skips them). Generated data stays out of Git and out of the key of
  `check`'s audit step, so a test of the package's own checks that the generator still writes the recorded data
  (PoUW's `verity/protocols/pouw/verifier/tests/test_lean_package.py`).
- `axioms` (optional): the allowed axioms.
- `assumptions_module`, `assumptions`, `domain`, `open`: the premises the guarantees take, by kind
  (`assumption_records`, `unfolding`, `read_props`). A named assumption is declared in a module `assumptions_module`
  names or in one under it; it names one module or a list of them (`security/Properties` lists
  `Properties.Assumptions` beside `Definitions.Assumptions`, for an assumption stated over the code), each a module of
  the build or of a package it requires. A domain hypothesis is part of what its guarantee says, with a one-line
  `why`, or `cites` the listed guarantee whose statement it is; every other premise is open, with the Linear `issue`
  that discharges it and its `due` date, written by hand. A `Prop` a statement reads without taking it is open only by
  an entry that gives a `why`, for each guarantee that reads it (`on_read`). Each has an `owner` and a `digest`, so a
  change to a premise or to what it reads fails under its own name. `open_seed`, `open_default_issue` and
  `default_owner` are what `--update` and the records run fill an entry from (`bookkeeping_ticket`).
- `trust`: `base` ({premise: the trust assumption it states}, the same in every lock) and `extra` ({deployed guarantee:
  {premise: the ruling's rule}}, what it alone may rest on). `deployed`: the guarantees that state the deployed system.
  `trust_gap`: {deployed guarantee: its premises outside its trust}, which only a person adds to and `--update` drops
  from; the report prints each gap (`trust_failures`, `trust_gaps`).
- `meaning` (default `roots`): the modules whose definitions a guarantee's meaning is followed through. What a
  statement is about (an executable, Mathlib) stays outside (`record_scope`).
- `escapes`: every declaration a proof cannot unfold (`opaque`, which a `partial def` is, `unsafe`,
  `@[implemented_by]`, `@[extern]`), each with a reason; a listed one that is none fails (`judge`, `Facts.lean`).
- `verifier`: nothing its `entries` reach is `@[implemented_by]` or `@[extern]` outside `allow`, and they reach the
  `hash` namespace only through `stop` (`verifier_failures`, `verifier_allow`; Flock's lock lists names only).
- `claims`: every predicate the spec declares under `claims.modules` has its `fails` and `meets` witnesses under
  `claims.witnesses`, a reason under `claims.not`, or its guarantee under `claims.generic` (`claim_failures`).
- `witnesses`, `unwitnessed`, `witnesses_assumed`: a guarantee of an acceptance or an exclusion has `honest` and
  `nontrivial` witnesses, or an `unwitnessed` entry, which only shrinks (`witness_failures`, `witness_scope`,
  `assumed_failures`).
- `compile_time`: the modules that must run code while they compile, one by one, each with its `why` and the `digest`
  of its file as it was reviewed (`listings`, `scans`). None may declare a guarantee or a definition a guarantee reads,
  so a proof-only tactic belongs in a module of its own. The package has a `lakefile.toml`, not a `lakefile.lean`.
- `dependency_unchecked`: the modules of fetched packages whose source may name a kernel bypass, each with the reason
  (`dependency_scans`). An entry that no longer matches a hit fails, so the list stays current.
- `layers`: the modules under a prefix import only the listed prefixes (`judge`). A rule with `"sealed": true` is a
  layer of its own: its modules answer only to the rules at or under it, and another rule reaches it only by listing
  a prefix at or under it, never through a shorter one that holds it (an area's `Guarantees` under the area's
  vocabulary). **The spec**, what a guarantee is stated over, is the `layers` prefixes: every definition a guarantee's
  statement reads lies in the spec of its package or of one it requires by path, or in a pinned dependency
  (`spec_reads`). `reads_exempt`: the module prefixes a guarantee may read outside the spec, each with the reason.
- `trees`: the module prefixes of the definitions, properties and proofs trees; no statement reads a definition of the
  proofs tree, and no module of the other two imports one, `except` aside (`tree_failures`, `tree_roles`). The report
  lists, without failing, the spec definitions no guarantee reads (`unread_spec`).
- `upstream`: what each watched statement is about, and what has been acknowledged (`upstream.py`).
- `four_names`, `four_names_except`: each guarantee is `X.SecurityProofs.G : X.Guarantees.G` (`four_name_failures`).
- `guarantees`: the guarantees (GLOSSARY.md) and their witnesses, never another lemma, each with its `owner`, the
  registry handle of the lane that answers for it. `--update` fills each record, and the `reads` section, from the
  build (`judge`, `read_groups`, `PREMISE_KINDS`); without it the audit fails on any difference. A record holds the
  statement, not the proof: list a `sorry` stub, and the guarantee passes once a proof with allowed axioms replaces it.
- An unowned lock (`unowned_lock`): one that lists a guarantee or a premise and holds no lane field (`LANE_FIELDS`,
  `LANE_SECTIONS`), the lock of a tree with no lanes; no flag makes a lock unowned. No entry owes an owner (a premise
  the lock doesn't list still fails); its `AUDIT` line ends with `unowned`, and its report holds `"unowned": true`.
  One lane field, or `--owner` for that run, makes it owned and held to every rule. Every lock of this tree is owned
  (`tests/test_lean_packages.py`).
- `shape`: the shape every guarantee has (the lean-proofs skill, "One shape"): `properties`, `premises`, `frozen`,
  `statements`, `events`, `conjectures` (`shape`, `part_refusals`, `reordered`, `shares`), and `inline`, the inline
  hypotheses still allowed, which only shrinks (`inline_failures`).
- `dependencies`: written by `--update`, a digest of the `.olean` files the package imports, per fetched package and
  for the toolchain (`dependency_digests`).

The hashes are SHA-256, over a canonical form of the kernel term (`Facts.lean`'s `canonText`). Each
`set_option maxHeartbeats N` is in the package's report (`heartbeats`); `tests/lint/test_lean_bar.py` warns of a new one
with no reason and refuses one with no limit.

### A spec and its proofs in two packages: `proved_in`

A spec package (`Pkg.Protocol`, `Pkg.Assumptions`, `Pkg.Guarantees`: definitions and `def G : Prop` statements) keeps the
lock, the trusted object. Its proofs may live in another Lake package that `require`s the spec by path and proves
`theorem SecurityProofs.G : Guarantees.G`; that package is untrusted, and has a policy of its own with no guarantees.
The spec's lock names it (`"proved_in": ["../../security_proofs/pkg"]`), and the spec's audit reads, checks and records
each guarantee exactly as one proved in the spec itself (`proved`, `assign`, whose failures say what a named package
must be). The proofs' modules sit under a root of their own (`PkgProofs.…`), since Lean finds a module in the first
package on its search path that holds its root; the declarations keep their names (`namespace Pkg.SecurityProofs`).

## The records

One line per mechanism; a new one gets its line here and its detail in the docstring.

- **`--update`** writes the records and prints the review: each changed guarantee before and after, and each changed
  definition a guarantee reads (`review`, `statement_changes`; `review.txt` among the reports). Say so in the PR.
- **A record difference regenerates.** An audit whose only failures are record differences (`RECORD_DIFFERENCE`) writes
  the lock `--update` would among its reports (`regenerated`, `changes`). A candidate's records run runs with
  `--regenerate`, and the verified queue commits those records on top of the candidate, whose check passes
  `--no-regenerate`, so a record difference that reaches it fails there (`research.vqueue_records`).
- **An allowance is no pass.** A lane's fast path takes a finding ruled open as known, by its code and entry
  (`lean_changed.py --records --allow CODE:ENTRY`; `failure_codes` and `failure_entries` in `report.json`): the run
  exits 3 and keeps no pass, and the record it regenerates stays outside `record/` (`lean_changed.py`'s `ALLOWABLE`,
  `conclude` and `withheld`). `check.py` takes none.
- **Bookkeeping warns in a records run.** A candidate's records run (`--bookkeeping warn`) repairs a lock that fails
  only on bookkeeping, and the caps beside it, and warns each owner (`BOOKKEEPING`, `repaired`, `bookkeeping_repairs`;
  `lean_audit.py`'s `held_caps` and `raised_unread`; `research.merge.REPAIRED`).
- **The landing DM** lists each new, removed or changed guarantee and definition when the change lands on `main`;
  nobody approves it first (`research.spec_alert`). A statement whose type hash, named assumptions and premises are
  the same is one line, "printed differently", and stays out of it (`REPRINTED`).
- **A lift is not a change** when the type, with `G` unfolded once, hashes to the record's (`lifted_into`,
  `Facts.lean`'s `liftOf`); `--lift` writes lifts into the four-name form (`lift`).
- **A move is not a change**: `--update --moved MOVES.json` compares the build with the records under the old names
  (`review`, `facts_of`, `move_errors`, `Facts.lean`).
- **Merging**: `merge.py` merges `lean-audit.json` entry by entry and leaves git's conflict only at its three guards;
  a merged record is a candidate, which the Lean check rebuilds, so a merge that is clean but wrong (a definition that
  changed on one side without a guarantee there reading it, while the other side's new guarantee reads it) fails there
  (`merge.py`; the fast path's `merged` and `kept`, `lean_changed.py`).

## The modules

Each line says what the module is; its docstring (a Lean file's `/-! -/`, a script's header comment) has the detail.
Other tools run the commands here, and load `headers.py`, `module_keys.py`, `modules.py`, `lean_audit.py`,
`axiom_probe.py` and `siblings.py`, by path (`tools/verity/check/`, the task graph, the queue, the `verity` command).
They import only `tool.py`, as `tools.verity.lean.tool` (the run store's registry). `tests/test_gate_paths.py` fails
on a gate tool's path constant or a registered tool that no longer resolves.

The auditor:

- `check.py`: the audit of the named packages: the facts, the checks, the records and their review, the seals, the
  replay and its cache, the report.
- `Facts.lean`: the facts `check.py` judges, from the `.olean` files as data; the canonical texts the hashes are of.
- `Replay.lean`: the kernel replay of a package's declarations, and of its import closure.
- `Graph.lean`: the import graph of some modules, from their `.olean` headers.
- `Unread.lean`: what each declaration mentions, for `unread.py` and the audit.
- `sandbox.sh`: the build sandbox. `setup.sh`: Lean and a package's dependencies on a Linux machine.
- `settings.py`: the environment variables the Lean tools read. `moved.py`: where a path of an earlier layout is now
  (`setup.sh` runs it). `public_lock.py`: a lock as the public export ships it.
- `headers.py`: the one reading of a Lean file's import header. `modules.py`: the module-system codemod.
- `module_keys.py`: a Lake module's key, under which the node's artifact cache, which every sandboxed build fills,
  restores it.
- `olean_graph.py`: whether two builds of one module differ only in where its source was.
- `upstream.py`, `Upstream.lean`: the watch on what a pinned dependency proves about what a package assumes.
- `merge.py`: git's merge drivers for `lean-audit.json`, import lines, lakefile lists and the caps.
- `cache.py`: the build cache: main's compiled modules, published on merge and restored sha256-checked; the audit
  never reads it. `run_daily.py`: its daily publish.

Lean's job in `check` and the tools around the audit:

- `lean_audit.py`: the lean-audit step: which packages miss (the key of a pass; `--explain PKG` says what a miss
  changed), their scratch tree, the warm dependencies and builds, the replay cache, the unread ratchet.
- `lean_changed.py`: no step of `check`: a lane's fast path to its records (`--records`) and the edit loop on a
  node (`--try`).
- `lean-deps.json`: the pinned dependency bundles (`lean_audit.py --pin`).
- `warm.py`: the Lean audit of a ready mark's merge, which builds a Lean change before a candidate holds it.
- `siblings.py`, `axiom_probe.py`: what `verity lean try` runs beside lean: the sibling check and the axiom probe.
  The sibling check leaves out, and counts as unchecked, a module whose build, or an import's, is not of its source
  as it is: by the hash Lake's trace states for a module it built, and for one it took from its cache or an archive
  by Lake's own word, a `lake build --no-build -v` of the modules that could be siblings, which compiles nothing, run
  before any product is read and again after a rebuild. A word it can't take (an exit other than 0 or 3, a line it
  doesn't read, a write the try's sandbox refused) leaves every build Lake took unchecked; no time decides. On a node,
  a build Lake took and would compile (an import rebuilt since) counts where the node's artifact index records the
  SHA-512 of its source as it is under its trace's hash, read and never written; a checkout has Lake's word alone.
  It leaves out uncounted, FILE's own module aside, what no root loads beside the rest: the other executables' roots,
  and every module under an `exempt` prefix of the package's lock (the name or below it, as the audit matches it), so
  two scripts that each declare `main` don't clash. It never fails on the lock: with none, or one that can't be read
  (which it says on stderr), it leaves out the executables' roots alone.
  A sibling is checked as it is built. One built against an older build of an import is rebuilt when a refusal names
  it, and not otherwise: the stub's first refusal has Lake rebuild the siblings it names, once, with a line that says
  so, and what the stub of the plan made once more says is the verdict. That is the step's limit, and the full check,
  which builds every module, has none.
- `unread.py`, `unread-caps.json`, `unread_gate.py`: the Lean nothing reads, its ratchet's caps, and the queue's
  landing gate on its seeds (drive.toml's `landing_gates`). A PR that adds proof modules no guarantee reads yet sets
  its area's entry in the JSON file to the check's count in its own diff, with no room, and lists in its body each
  module, the guarantee that will read it and the date that guarantee is listed by (`ratchet`, whose failure prints
  the entry).
- `trusted_size.py`: the size of a package's spec, per area.
- `tool.py`: the store's declarations of the scheduled runs.
- `tests/`: the suite `verity-lean-audit`, the controls' fixtures among it.

## Using it from another repository

The audit needs the standard library, `lake` and the toolchain's `lean` and `leanc` (`Facts.lean` runs compiled); nothing
here imports Verity, so another repository can copy the auditor (`check.py`, `Facts.lean`, `Graph.lean`,
`Replay.lean`, `upstream.py`, `Upstream.lean`, `merge.py`, `headers.py`, `modules.py`, `olean_graph.py`,
`module_keys.py` with the task graph's `contract.py` it loads, `sandbox.sh`, `setup.sh`, `cache.py`, `tests/`) for its
own packages. The two jobs use `check`'s helpers (`tools/verity/check/common.py`, `lean_slot.py`), so they stay with
this repository.

Copy this directory at a commit of Verity's `main` and record the commit beside it. Write a `lean-audit.json` for each
package (`check.py` prints a template when one is missing), run `check.py --update` once on the trusted statements,
review and commit the records, and run `check.py --all --build` wherever the repository runs its checks.

These records suit statements that are merged and reviewed. An external, untrusted submission still needs its theorem checked by
`isDefEq` against a trusted `Prop` taken from the trusted commit, never from the submission's tree, and the submission
built in a sandbox, as POUS's grader does.
