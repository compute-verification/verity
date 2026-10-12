# tools/verity/lean: the Lean audit

One audit for every Lake package (`python tools/verity/lean/check.py --all --build`), which `check` runs on the packages whose inputs
changed (`audit.py --build PKG...`). Lean's jobs in `check` are here too: `lean_audit.py` (the lean-audit step: which
packages miss, their scratch tree, warm dependencies), `lean_changed.py` (the lean-changed step, and a lane's fast path to its
records) and `lean-deps.json` (the pinned dependency bundles). The audit needs the standard library, `lake` and the
toolchain's `lean`; nothing here imports Verity, so another repository can copy the auditor (`audit.py`, `Facts.lean`,
`Replay.lean`, `upstream.py`, `Upstream.lean`, `merge.py`, `sandbox.sh`, `setup.sh`, `cache.py`, `tests/`) and point it at its
own packages. The two jobs use `check`'s helpers (`tools/verity/check/common.py`, `lean_slot.py`), so they stay with this repository.
`check` reuses a package's pass while the content of everything its audit reads is unchanged, not its path: each file under its
path relative to the package, a package required by path as its own content, so a `git mv` of a package is no miss, and its
kept `.lake/build` follows it (`lean_audit.py`'s docstring has the key, `--explain PKG` what a miss changed).

No package code runs where the audit decides anything. The facts come from the `.olean` files read as data (`Facts.lean`
imports with initializers disabled and no environment extension loaded), and the kernel replay (`Replay.lean`) loads them
the same way. The only package code that runs is the build's and, after the facts and the replay, the tests' under `runs`
(each deciding only its own pass), in a sandbox where the machine allows one; a test's data generator is the repository's
code and runs outside it.

~~~text
bash tools/verity/lean/setup.sh PKG                      elan, PKG's lean-toolchain, its pinned dependencies, Mathlib's cache when
                                                  Mathlib is a dependency, and main's cached build of PKG (below)
python tools/verity/lean/cache.py get [PKG...]           main's cached build of each package, every file sha256-checked
python tools/verity/lean/check.py [--build] PKG...       audit these packages (--build: setup.sh and lake build first)
python tools/verity/lean/check.py --update PKG...        write the guarantees' records from the build; commit the diff (a copy of
                                                  each record goes beside its reports, which a recorded run publishes)
python tools/verity/lean/check.py --update --owner @lane PKG...   the same, with @lane as the owner of each new guarantee
python tools/verity/lean/check.py --all [--root DIR]     the controls, then every package under DIR with a lean-audit.json
python tools/verity/lean/check.py --name-instances PKG...  write each anonymous instance's name, the one Lean gave it, into its
                                                  declaration (no record changes); it fails until you build and audit again
python tools/verity/lean/check.py --lift LIFTS.json PKG  write each guarantee {"X.t": {"as": "X.Guarantees.G", "into": MODULE}}
                                                  as `def G : Prop` in MODULE and `theorem X.SecurityProofs.G :
                                                  X.Guarantees.G := @X.t` beside its proof (or in the module
                                                  `"beside": M` names); then build and run
                                                  `--update --moved LIFTS.json.moved.json PKG` (below)
python tools/verity/lean/upstream.py PKG [--lean]        what the pinned dependencies prove about PKG's watched statements
python tools/verity/lean/merge.py --install              once per clone: git merges lean-audit.json with merge.py (below)
~~~

## What it checks

Every declaration of the package's audited modules (not a list of theorems), from the environment Lean builds:

- only the allowed axioms (default `propext`, `Classical.choice`, `Quot.sound`): no `sorry`, no `native_decide`;
- no `axiom` declaration: an assumption is a named `Prop`, taken as a hypothesis;
- every declaration a proof cannot unfold (`opaque`, which a `partial def` is, `unsafe`, `@[implemented_by]`,
  `@[extern]`) is listed with a reason;
- every `.lean` file of the package is imported by a root, or exempt with a reason;
- no source file or lakefile names `debug.skipKernelTC`, an option that turns proofs into `sorry`, a `bootstrap.*`
  option, `addDeclWithoutChecking` or `addDeclCore` (comments and docstrings may);
- nor does the source of a module the package imports from a fetched package (`.lake/packages/*`: Mathlib, ArkLib and
  what they require), or the lakefile of a package it imports from, unless `dependency_unchecked` lists the module with
  the reason. The imported modules are the ones the `dependencies` record covers, and their sources are there once the
  package is built, so the scan builds nothing: soundness's 3,900 imported dependency modules take about a second. The
  toolchain, which defines these options, is trusted, and a path dependency is a package of the repository, audited on
  its own. Among the pinned dependencies only doc-gen4's loader names one, and no audited package
  imports it;
- no source file runs code while it compiles (`initialize`, `#eval`, `run_cmd`, `run_elab`, an `elab`, `macro` or `syntax`
  definition, an elaborator attribute), unless its module is listed under `compile_time` with a reason and the digest of
  its file as it was reviewed. No listed module may declare a guarantee or a definition a guarantee reads.
  The package has a `lakefile.toml`, not a `lakefile.lean` (which `lake` runs whenever it loads the package);
- the `.olean` files of the modules the package imports, as each fetched package (Mathlib's downloaded cache, ArkLib's
  build) and the toolchain provide them, are the ones recorded under `dependencies`. A dependency's other modules may be
  built or not. Lean's `.olean` files are reproducible, so the record holds on any machine. It changes when the package's
  imports change (run `--update`) or at a bump; checking it takes seconds;
- each `layers` rule holds: the modules under a prefix import only the listed prefixes;
- every instance has a name in its source. Lean names an anonymous one from its type, and adds the project's root
  (`_proofs`) when nothing in the type is the project's own, so a move into another root or a split renames it, and every
  record whose definitions use it changes. `--name-instances` writes each current name into its declaration;
- the guarantees say what their records say, and each names its owner (below);
- every definition a guarantee's statement reads is in the spec (below), or a pinned dependency's;
- no dependency the `upstream` section scans proves, at the revision `lake-manifest.json` pins, a theorem about a watched
  statement that nobody has acknowledged (below);
- the kernel accepts every declaration again (`Replay.lean`, on one import of the package's dependencies, each
  declaration in a task of its own on Lean's threads, `LEAN_NUM_THREADS` (by default the CPUs the audit may use: its
  affinity, bounded by the cgroup quota, where Lean's own default is the whole machine), checked against the import and the
  declarations it comes after): a declaration that skipped the kernel shows no axiom, and only the replay catches it.
  The replay also walks the environment it checked for the axioms the package uses, and the audit fails if they differ
  from the sets the `.olean` files recorded. No `#print axioms` output is parsed anywhere. A replay killed by a signal
  (out of memory) is reported as a kill, not as a verdict;
- each test under `runs` passes on the data whose SHA-256 the record holds (below);
- nothing changes this directory, the toolchain's `lean`, `lake` and `leanchecker`, or the `.olean` files of a package
  audited earlier while the audit runs: the Lean files run from a copy taken before the first build, and everything is
  sealed and checked after each package.

`--build` builds in `sandbox.sh` where `unshare` and unprivileged user namespaces are available: new user, mount, network
and PID namespaces, everything read-only but the build's `.lake` directories, no network, and nothing outliving the build.
The audit says when it had to build without one; the `compile-time` check then stands in for the sandbox.

`--fresh` also runs `leanchecker --fresh` on each root: the root's whole import closure, Mathlib and core included,
through the kernel into an empty environment. It needs one module per root and a lot of memory, so it is not part of
every run: the steward runs `audit.py --all --build --fresh` nightly on `main`, and `upstream.py --fetch` against
ArkLib's head weekly, each as a recorded run on a pod (`tool.py`; `[[run]]` in `tools/research/README.md`).

`--all` first runs the controls (`--no-controls` skips them; `check`'s audit does, and the `tools/verity/lean` suite runs them, only
when `tools/verity/lean/` changes): `tests/fixtures/control`, built with each audited package's toolchain (`--build`
installs it first, so a machine without elan works), must pass clean and fail once for each violation in
`tests/fixtures/violations.json`. Among them:

- a false theorem added through `debug.skipKernelTC`, spelled so that no scan finds it, which must pass every check but
  the replay and fail the replay;
- an initializer that tries to forge the facts, which must never run where the audit reads the package;
- compile-time IO that rewrites a checker input and restores it, which the compile-time check refuses and the sandbox
  blocks;
- a docstring that names `debug.skipKernelTC`, which must pass;
- a fetched dependency (a package at `.lake/packages/dep`) whose module sets `debug.skipKernelTC`, which must fail, and
  one whose docstring and comments name it, which must pass.

## The policy: `lean-audit.json`, beside the lakefile

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
 "guarantees": {"Pkg.main_theorem": {"owner": "@lane"}}
}
~~~

- `roots`: the modules the audit imports; every module under them is audited. `axioms` is optional.
- `proved_in` (optional): the packages, as paths relative to this one, that prove its guarantees (below).
- `runs`: the package's tests, compiled evaluation that no proof reads (a bulk conformance check over more vectors than
  the kernel checks). Each is an exempt module, run beside the facts and the replay by `lake env lean --run` in the build sandbox, and
  passes when it exits 0. An argument `{NAME}` is the path of data file NAME, which is what `generate` writes (run first,
  outside the sandbox, with `{repo}` the checkout, `--repo`, and `{out}` the directory it writes) or else the package's
  own file, either one possibly gzipped (`NAME.gz`). A file is used only with the SHA-256 under `data`: another fails the
  test unrun, and `--update` records it. Generated data stays out of Git and out of the key of `check`'s audit step, so
  a test of the package's own (PoUW's `verity/core/protocols/pouw/tests/test_lean_package.py`) checks that the generator still writes the recorded
  data. `--no-runs` skips them.
- `assumptions`: the named assumptions, by name, each with its `owner`: the `Prop` definitions a guarantee may take as a
  closed hypothesis, or reads in its statement. `--update` writes each one's `digest`, the one its `reads` record holds, so
  a changed assumption fails under its own name; an entry no guarantee reads fails. A list of modules is the old form:
  it fails, and `--update` converts it into the definitions the guarantees read there, each owned by their owner.
- `meaning` (default `roots`): the modules whose definitions a guarantee's meaning is followed through. What a
  statement is about (an executable, Mathlib) stays outside.
- `compile_time`: the modules that must run code while they compile, one by one. Write the entry with its `why`; `--update`
  adds the `digest` of the module's file. Any later edit of the file fails the audit until someone has reviewed its
  compile-time code and `--update` records the new digest, and `--update` prints the module in its review.
  No listed module may declare a guarantee or a definition a guarantee reads: a term elaborator, a notation
  or a macro there could change what a statement means. A proof-only tactic belongs in a module of its own.
- `dependency_unchecked`: the modules of fetched packages (for a lakefile, `<package>/lakefile.lean`) whose source may
  name a kernel bypass, each with the reason. An entry that no longer matches a hit fails, so the list stays current.
- `dependencies` is written by `--update`: per fetched package and for the toolchain, a digest of the `.olean` files of the
  modules the package imports.
- `guarantees`: the guarantees (README "Glossary"), and any lemma still listed beside them.
  `--update` fills each record, and a `reads` section, from the build: the signature, the named assumptions among the
  hypotheses, a hash of the type, and for each module the definitions the guarantees read, with a hash each, the
  guarantees that read them, and a `digest` of those hashes (which the record therefore determines). Without `--update`
  the audit fails on any difference, so a statement, or a definition it depends on, cannot change without a visible diff
  of this file. A record holds the statement, not the proof: list a `sorry` stub, and the guarantee passes once a proof
  with allowed axioms replaces it. A record written before guarantees were called that holds them (and each module's
  readers) under `pins`: the audit reads it so, `--update` rewrites it under today's names, and a record holding both
  is refused.
- `owner`, in each guarantee's entry: the handle of the lane that answers for it (`@proofs`, `@compute-accounting`), whom
  a broken or restated guarantee goes to. The audit fails an entry without one. Write it by hand, or pass
  `--update --owner @lane`, which gives it to the new entries only: `--update` keeps an entry's owner, and writes no
  entry without one. A changed owner is one line in the review and no change of the statement.
- `shape`: `{"premises": [module prefixes], "properties": {G: {"function": F, "outcome": O}}, "frozen": {name: why}}`,
  the shape every guarantee has (the lean-proofs skill, "One shape"). `Facts.lean` unfolds a lifted definition once,
  inlines its `let`s and splits a conjunction into its conjuncts; each takes its inputs, one hypothesis that applies `F`
  (a constant, several, or a field path such as `P.protocol.check`) and, unless `O` is `accept`, a constant named `O`,
  and premises (hypotheses whose head is under `premises`), up to the first input bound after that acceptance; what
  follows is its conclusion. With `O` `none` (a function that always returns) it takes no acceptance and must mention
  `F`. A conjunction whose conjuncts are about different functions (soundness of a check and simulation of a server)
  gives `{"conjuncts": [{"function": F1, "outcome": O1}, ...]}` in place of `function` and `outcome`, one per
  conjunct in order, so each is checked against its own. Several acceptances of `F` count as one, of the tuple of their
  inputs. A hypothesis is read by its body, under its leading binders: the acceptance's body applies `F` through
  definitions (mentioning a structure whose fields name `F`, such as an outer session, is not applying it) and the
  acceptance holds under no hypothesis of its own (`A → accepts` is not one), and a premise's head is its body's (`∀ j,
  P j` is headed by `P`). A `Prop` field of a structure under `meaning` that an input ranges over, or that an input
  returns (a family `Vs : Fin S → VStmt`, through reducible abbreviations), is a hypothesis, its
  own `Nonempty` among them. A guarantee under `frozen` (with why it doesn't fit yet) is
  exempt, and fails once it fits, so the list only shrinks. Every `def G : Prop` of a spec module that holds statements (`"statements"`: module prefixes; by default a module
  with a `Guarantees` component) must be the statement of a
  listed guarantee, or be under `frozen` until one is. A reason starts with its category (`restatable`, `frozen`,
  `lemma`, `completeness`, `falsified`, or `unstated` for a statement no guarantee states), so a record counts its
  guarantees by category. `"conjectures": {P: {"status": "open" | "falsified", "test": T, "why": W}}` names the
  conjectures a property may rest on, each a named assumption: one is a premise when it heads a hypothesis, and a
  property with a hypothesis reaching one lists it under its entry's `rests_on` (it is conditional). An open one names
  its falsification test `T`, a pytest node id `PATH::NAME` from the repository's root that a suite runs
  (`tests/test_lean_packages.py` checks the test exists), so a counterexample fails `check`; a falsified one refuses every
  property resting on it, which stays under `frozen` only as `falsified`. Every package is checked: without a `shape`
  section each guarantee is frozen or fails. An entry added to `frozen`, and a conjecture added or with a new status,
  is a claims change: the landing DM shows it with its reason (`research/spec_alert.py`), and nothing waits on it.
- **The spec**, the trusted text a guarantee is stated over, is the `layers` prefixes: the modules whose imports the
  policy rules. Every definition a guarantee's statement reads (theorems are not read: a statement means what it means
  whatever proves the propositions inside it), followed through `meaning` and where the walk leaves it, must lie in the
  spec of its package or of a package it requires by path (by that package's own `layers`), or in a pinned dependency
  (the toolchain, Mathlib, ArkLib). A definition in a spec module must also be under `meaning`, or the record would hold
  nothing of it. `reads_exempt` lists, each with the reason, module prefixes a guarantee may read outside the spec (a
  package whose statements are about its executable, not about a model of it), as an object {prefix: the reason}; an
  entry that exempts no read outside the spec fails, and no entry covers a spec module outside `meaning`.
  The report also lists, without failing, the spec definitions that no guarantee reads, its own or a dependent
  package's (`unread_spec` in `report.json`).
- **The hashes are SHA-256.** They are taken over a canonical form of the kernel term (`Facts.lean`'s `canonText`),
  which keeps what Lean's `Expr.hash` keeps (constant names, universe levels, bound variable indices, projections,
  literals) and drops what it drops (binder names and binder info, `let` names, metadata). A renamed hypothesis is
  therefore the same statement, and a different term is a different one. Records written with `Expr.hash`, which has
  32 bits, are still checked as they are, and the run notes them. `--update` rewrites them, and its review says
  "rehashed" where the 32-bit hash still matches, which Daniel's DM (below) leaves out.

**A changed guarantee reaches Daniel.** A weakened definition leaves every proof valid, so the axiom checks and the
replay pass it: only the records see it, and only a person can judge it. (A redefinition of `Collision` once made a
Merkle binding guarantee vacuous for unsalted schemes; the records flagged it and a review caught it.) `--update` records
the change and prints it: each changed guarantee's signature before and after, and each changed definition a guarantee
reads, with the guarantees that read it and its source now (also in `review.txt` among the reports). Say so in the PR.
An audit without `--update` whose only failures are record differences (a guarantee's signature, named assumptions or
type hash, a `reads` group, an assumption's digest: what a train's merge of two PRs, each recorded on its own side, can
leave) writes the `lean-audit.json` and `review.txt` that `--update` would among its reports, and its `report.json`'s
`regenerated` names that file; a missing record or name, an owner or any other failure writes none.
When the change lands on `main` (`research merge --push`, `research queue land`), Daniel gets a DM with each new, removed
or changed guarantee and each changed definition (`research.spec_alert`); nobody approves it first. A guarantee whose
statement is the same but prints differently (the same type hash and named assumptions, after a toolchain bump or a change
of notation) gets one line, "printed differently", and is left out of the DM. Definitions are compared by name, so one
read from another module with the same content is counted on one line.

**A lift is not a change.** Lifting a guarantee into the spec, `theorem X (xs) : e` becoming `theorem X (xs) : G as` with
`def G (ys) : Prop := e'` in a `Guarantees` module, changes its record's signature and type hash. `check.py --update`
certifies the lift as the same statement when the type, with `G` unfolded once, hashes to the record's type hash and the
named assumptions are the record's (`lifted_into`). `Facts.lean` (`liftOf`) unfolds by delta alone: the binders `xs` as they
are, and in place of `G as` the value of `G` at the type's universe levels with its leading `fun`s instantiated by `as` (as
`Meta.unfoldDefinition?` unfolds an applied constant), the proofs Lean abstracts out of a definition (its `G._proof_<n>`
theorems) put back in place, as a theorem's type keeps them. Nothing else is reduced, and the text is hashed as every
type is. A certified lift gets one line, "lifted into G: the same statement, its definition unfolded once", and `G`, new and
read by no other guarantee, needs no reading of its own; the review then holds no statement change, so the merge needs no
statement reviewer for it. Any other lift prints in full: a `G` whose value differs from `e` in anything the hash sees
(binder names and metadata aside), a `G` outside a `Guarantees` module, and a lift that moves a named assumption from the
theorem's binders into `G`, which the record's `assumptions` would lose. Daniel's DM reads only the records, so it still
lists a lifted guarantee as changed.

`four_names` lists the namespace roots (`["Nci", "NetTiming"]`) whose guarantees are all in the four-name form: each is
`X.SecurityProofs.G : X.Guarantees.G` (universe parameters aside), with `G` a definition of a `Guarantees` module, and the
audit fails on any other guarantee under a listed root. A package's owner adds its root once every guarantee there is
lifted, or no longer a guarantee.

`audit.py --lift LIFTS.json PKG` writes lifts into the four-name form: `def G : Prop` with the theorem's doc comment,
binders and type as its source writes them, after the `variable`s in scope that Lean includes in the theorem (those the
statement mentions, and the instances over them), in the theorem's namespace with its file's `open`s (less the namespaces
only proof modules declare) and the `open … in` right above it, so that its names resolve as they did, and
`theorem X.SecurityProofs.G : X.Guarantees.G := @X.t` beside the proof, whose module then imports the spec module;
`"beside": M` puts the theorem in `M` instead (a package that collects its guarantees' proofs in one module), which then
imports the proof's module too. The spec module imports the modules of the definitions the statement reads, and a
statement that reads a definition of a `proved_in` package (in `reads` or `outside`) is refused: move the definition
first. What it cannot follow (an `open` of a namespace a definition module opens only through a proof's declaration, a
`variable` an `omit` or `include` changes) fails the build or prints as a change, never as the same statement. It writes
the moves (`LIFTS.json.moved.json`, each `X.t` to `X.SecurityProofs.G`) for `--update --moved`,
which certifies each lift or prints it; `X.t` stays as a lemma. A guarantee that takes a named assumption as a hypothesis
moves it into `G`, which prints as a change.

**A move is not a change.** When declarations move (guarantees renamed, the definitions they read renamed or moved to other
namespaces), write the moves as a JSON object of old names to new ones, each a declaration or a namespace (the longest prefix
applies), and run `check.py --update --moved MOVES.json PKG` on the record as it was before the move. The record's guarantees
are renamed, and `Facts.lean` writes each guarantee's statement and each definition it reads once more with the build's names
mapped back to the old ones, a lifted statement also with its definition unfolded. The review compares those texts with the
records of the old names: a guarantee that says what its record says gets one line, "the same statement under its old
names" (moved from its old name, lifted as above, or both), and the definitions the same under their old names are counted
on one line. Daniel's DM, which reads only the records, leaves out a guarantee renamed with its type hash unchanged and a
definition moved with its content. Everything else prints as without the map. A map whose new names nest differently
from its old ones is refused, and so is one under which two constants of those texts (or two guarantees) would have one old
name, since the comparison could not tell them apart; an old theorem kept under its old name as the lemma behind a new
guarantee is no conflict.

## A spec and its proofs in two packages: `proved_in`

A spec package (`Pkg.Protocol`, `Pkg.Assumptions`, `Pkg.Guarantees`: definitions and `def G : Prop` statements) keeps the
lock, the trusted object. Its proofs may live in another Lake package (`security_proofs/<area>/`) that `require`s the spec
by path and proves `theorem SecurityProofs.G : Guarantees.G`; that package is untrusted. The spec's policy names it:

~~~json
{
 "roots": ["Pkg.Protocol", "Pkg.Assumptions", "Pkg.Guarantees"],
 "proved_in": ["../../security_proofs/pkg"],
 "guarantees": {"Pkg.SecurityProofs.g": {}}
}
~~~

and the proofs package has a policy of its own with no guarantees: its `roots` (all of its modules a proof uses), and its
`layers`, `exempt`, `escapes` and the rest as any package's. The spec's audit then reads each guarantee from the `.olean`
files of the package that declares it, with the spec's `meaning`, `axioms` and `assumptions`, and checks, records and
`--update`s it in the spec's lock exactly as a guarantee proved in the spec itself. So moving a proof out of the spec's
package changes no guarantee or `reads` record, and sends no DM (`tests/test_audit.py` builds the move). The spec's own
checks (layers, scans, escapes, its replay) run as before. It fails when:

- a guarantee is declared by no package among the spec and those named (`missing`), or in a module none of them audits, or
  by two of them;
- a named package does not require the spec by path, lists guarantees of its own, or lists under `compile_time` a module
  where a guarantee it proves is declared or read;
- its roots import a module of its own outside them, which its replay would skip;
- its modules share a root namespace with a package it requires. Lean finds a module in the first package on its search
  path that holds a directory for the module's root, so `lake env lean` (and the audit's reading) cannot load `Pkg.Spec`
  from one package and `Pkg.SecurityProofs` from another, although `lake build` can. Put the proofs' modules under a root
  of their own (`PkgProofs.…`); the declarations keep their names (`namespace Pkg.SecurityProofs`), which is all a record
  holds. For the same reason, and because a library whose root is `Pkg` claims every `Pkg.*` module, an umbrella module
  `Pkg` cannot move into the proofs package.

The proofs package's own audit replays its declarations through the kernel, so `audit.py SPEC` audits each package
`proved_in` names in the same run, without `--update` unless it is named too, and the spec's reading waits for its build.
`check`'s key for a spec covers the files of the packages it names, so the two miss and are audited together.

## Merging records: `merge.py`

Two branches that each add guarantees both change `lean-audit.json`, usually in the same `reads` module, so git's text merge
conflicts. `.gitattributes` gives the file to `merge.py` instead, once `python tools/verity/lean/merge.py --install` has registered
it in the clone's git config. It merges entry by entry: each side's change to a guarantee, a module under `reads`, a dependency
or a policy entry applies. Within a module both sides changed, the definitions merge the same way, the readers as sets, and
the digest is recomputed. Guarantees arrive byte for byte from their branches, and a hand-written section one side extends
(a module added to `meaning`) is kept. A side written before guarantees were called that (`pins`) is read under today's
names. A guarantee's `owner` merges apart from the rest of its entry, so one side's new owner and the other side's
restatement both apply.

It refuses when both sides changed one entry differently: a guarantee restated on both, a definition whose content both changed,
a dependency both bumped. It also refuses when one side stops reading a definition in a module the other side changed,
since the record doesn't say which guarantees read it. A refusal is a conflict in git, with the entries named. Resolve it the
old way: take either side and run `check.py --update` on the merged tree.

A merged record is a candidate. `check`'s Lean audit rebuilds every record of the merged tree and fails on any difference,
so a merge that is clean but wrong (a definition that changed on one side without a guarantee there reading it, while the
other side's new guarantee reads it) fails there. `merge.py --normalize FILE` recomputes a record's digests without a build.

## Noticing when a dependency proves what we assume: `upstream.py`

A named assumption, or a lemma we prove by hand, may be proved upstream one day if its statement can be written in a
dependency's constants, and the proof may arrive as a different theorem, with a different constant, under a name nobody
watches. One stated over the package's own definitions has no entry, since no upstream theorem can mention them, and
neither has one about a single number (PoUS's `PrimeP16448`: `Nat.Prime (2 ^ 16448 - 21065)`), since its entry could
mention only `Nat.Prime` and would match every theorem about primes. So the
watch list names what a statement is about, and the status comes from the dependency, not from a document. It is the `upstream` section of the package's policy:

~~~json
"upstream": {
 "scan": {"Arklib": {"baseline": "scripts/axiom_baseline.json", "sources": ["ArkLib"]}},
 "imports": ["ArkLib", "VCVio.CryptoFoundations.HardnessAssumptions.CollisionResistance"],
 "watch": {
  "A1-johnson-mca": {"about": "Assumptions.BCHKS25Thm46 (A1)",
                     "mentions": ["CoreDefinitions.mcaError", "CoreDefinitions.AffineLineGenerator", "ReedSolomon.code"],
                     "ack": ["ReedSolomon.mcaError_affineLine_johnson_le"]}
 }
}
~~~

A hit is a theorem whose statement mentions every constant under `mentions`. There are two passes.

- **No build.** The pass reads each `scan` package's sources with `git show`, in its checkout under `.lake/packages/`, at
  the pinned revision. It matches the constants' last components as words in each theorem's statement (comments aside),
  and takes proved or not from the package's own transitive baseline. ArkLib keeps `scripts/axiom_baseline.json` current
  with its `validate.sh --axioms` gate. The audit runs this pass for every package with an `upstream` section: a bump
  that brings a proof about a watched statement fails the audit until the proof is acknowledged. It takes a few seconds.
- **Exact.** `upstream.py PKG --lean` builds `imports` and runs `Upstream.lean`. That matches the constants in the
  elaborated types of everything imported, and walks each hit's proof for its axioms. It covers packages without a
  baseline (VCVio, CompPoly), and catches what text matching misses. Run it at a bump. The first build of all of ArkLib
  takes a while; after that the pass takes about half a minute.

A proved hit not under its entry's `ack` is new, and the exit code is 1. Someone checks whether it discharges the
assumption or replaces the hand proof, then records it with `--ack`: the list says what has been looked at, not what was
used. To see what upstream has proved since the pinned revision, run `upstream.py PKG --fetch --rev origin/main`; a weekly
`research run` of it tells us when a bump would gain something. A document's "status upstream" line cites this output
instead of asserting a status.

## The build cache: `cache.py`

Integration publishes the build of each package at every merged commit, and a development machine restores the newest one
that is an ancestor of its `HEAD`. Lake then unpacks the modules whose input hashes match and compiles the rest. Lake's
input hashes don't depend on the checkout's path: a build published from `/build/verity` restores in `/workspace`. The
module docstring has the layout and the rules. In short:
- The client checks every file's sha256 against the commit's record before Lake sees it, and hands the files to Lake with
  `lake cache unstage`. Lake never reads the bucket itself, because its hashes aren't cryptographic.
- A package's record also carries the build of each dependency that Mathlib's cache doesn't provide (ArkLib and the
  packages it brings, for `Security/Proofs`), unstaged into that dependency's scope. A fresh machine then needs
  `setup.sh` (the toolchain, Mathlib's cache, the dependencies' sources) and `cache.py get`, and compiles none of ArkLib.
- A miss or a mismatch means building from source. So do a missing read credential (`LEAN_CACHE_R2_*`), the kill switch
  (disabled, quarantined or unreachable), and `LEAN_CACHE=off`.
- The audit never reads the cache: `lake_env()` sets `LAKE_ARTIFACT_CACHE=false` and `LEAN_CACHE=off`, so the validator
  builds from source.
- Only integration publishes, and only a clean checkout of a commit on `origin/main`. It builds with `cache.py build` inside
  its sandbox, then runs `cache.py publish` as root outside it, which hashes the staged files itself and signs with the
  bucket-scoped key (`/etc/verity/lean-cache.env`). The build's user wrote the stage, so the publisher follows no link
  below `--out`, takes only regular files with one link, and uploads a private copy of the bytes it hashed: a link
  planted to the key uploads nothing.
- Every write is conditional. A blob that already exists (a 412 or a 409) must have the same MD5, and blobs go up in one
  part so that their ETag is their MD5. Records go last.
- On node 1, `run_daily.py` is the daily publish (`vy-lean-cache-daily.timer`, deployed with the `cache.py` beside it).
  It builds a fresh clone of main at `/build/verity` per package in infra's sandbox (`vy-lean-sandbox`), with the
  dependencies fetched outside as research and bound read-only. It runs `publish` and `prune` only after
  `vy-lean-sandbox clean` passes. A probe that got through, a refused sandbox or an unclean tree publishes nothing.

## Using it from another repository

Copy this directory at a commit of Verity's `main` and record the commit beside it. Write a `lean-audit.json` for each
package (`audit.py` prints a template when one is missing), run `check.py --update` once on the trusted statements,
review and commit the records, and run `audit.py --all --build` wherever the repository runs its checks.

These records suit statements that are merged and reviewed. An external, untrusted submission still needs its theorem checked by
`isDefEq` against a trusted `Prop` taken from the trusted commit, never from the submission's tree, and the submission
built in a sandbox, as POUS's grader does.
