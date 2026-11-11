# tools/verity/lean: the Lean audit

One audit for every Lake package (`python tools/verity/lean/check.py --all --build`), which `check` runs on the packages whose inputs
changed (`check.py --build PKG...`). Lean's jobs in `check` are here too: `lean_audit.py` (the lean-audit step: which
packages miss, their scratch tree, warm dependencies), `lean_changed.py` (the lean-changed step, a lane's fast path to its
records, and its edit loop, `--try`) and `lean-deps.json` (the pinned dependency bundles). The audit needs the standard library, `lake` and the
toolchain's `lean` and `leanc` (`Facts.lean` runs compiled); nothing here imports Verity, so another repository can copy the auditor (`check.py`, `Facts.lean`,
`Replay.lean`, `upstream.py`, `Upstream.lean`, `merge.py`, `headers.py`, `modules.py`, `sandbox.sh`, `setup.sh`, `cache.py`, `tests/`) for its
own packages. The two jobs use `check`'s helpers (`tools/verity/check/common.py`, `lean_slot.py`), so they stay with this repository.
`check` reuses a package's pass while the content of everything its audit reads is unchanged, not its path: each file under its
path relative to the package, a package required by path as its own content, so a `git mv` of a package is no miss, and its
kept `.lake/build` follows it (`lean_audit.py`'s docstring has the key, `--explain PKG` what a miss changed).

No package code runs where the audit decides anything. The facts come from the `.olean` files read as data (`Facts.lean`
imports with initializers disabled and no environment extension loaded), and the kernel replay (`Replay.lean`) loads them
the same way, a `module`'s from the private part holding its proofs. The only package code that runs is the build's and,
after the facts and the replay, the tests' under `runs` (each deciding only its own pass), in a sandbox where the machine
allows one; a test's data generator is the repository's code and runs outside it.

A run keeps the facts only gzipped (`facts.json.gz`), and only those read back from the store (`check.py --facts`): a
check run keeps each audit's and its provers' (`status_export.py --facts`), a records run only PKG's own
(`trusted_size.py --run`). `Facts.lean` writes its document in a temporary directory outside the run's files.

~~~text
bash tools/verity/lean/setup.sh PKG                      elan, PKG's lean-toolchain, its pinned dependencies, Mathlib's cache when
                                                  Mathlib is a dependency, and main's cached build of PKG (below)
python tools/verity/lean/cache.py get [PKG...]           main's cached build of each package, every file sha256-checked
python tools/verity/lean/check.py [--build] PKG...       audit these packages (--build: setup.sh and lake build first)
python tools/verity/lean/check.py --update PKG...        write the guarantees' records from the build; commit the diff (a copy of
                                                  each record goes beside its reports, which a recorded run publishes;
                                                  refused in a tree `research run --on` shipped: use --cwd clone)
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
python tools/verity/lean/unread.py --out DIR --build     the Lean nothing reads; lean-audit runs its ratchet on the
                                                  audits' facts (its docstring: the seeds, the caps, the Properties gap)
python tools/verity/lean/merge.py --install              once per clone: git merges lean-audit.json with merge.py (below)
~~~

The edit loop is `verity lean try FILE`: `lake env lean FILE` in the nearest Lake package above FILE, against its built
imports, with its `[leanOptions]` as a build has them, exiting with lean's status, the axiom probe's line last. `--siblings` then imports FILE's build beside each built
module of its package not importing it, failing on a name two declare (`siblings.py`). With `--on NODE`, through `research
run`, it runs `lean_changed.py --try FILE` on this checkout's working tree: in a node's `build` Lean slot, on the slot tree
`--records` keeps warm, `lake build +MODULE:deps` and then `lake env lean FILE`. Neither runs this audit or writes a record
(`lean_changed.py`'s docstring).

## What it checks

Every declaration of the package's audited modules (not a list of theorems), from the environment Lean builds:

- only the allowed axioms (default `propext`, `Classical.choice`, `Quot.sound`): no `sorry`, no `native_decide`;
- no `sorry` the elaborator made for an error it did not report (`sorryAx _ true`: `unreported`), in an audited module
  or a `proved_in` package's, which also fails `lean_changed.py --records`, where the axioms check doesn't run;
- every top-level declaration in an audited (or proving) module's source lies in a range its `.olean` holds (`dropped`:
  `#exit`, or a `variable` over a local notation, can make Lean drop the declarations after it, reporting nothing);
- no `axiom` declaration: an assumption is a named `Prop`, taken as a hypothesis;
- every declaration a proof cannot unfold (`opaque`, which a `partial def` is, `unsafe`, `@[implemented_by]`,
  `@[extern]`) is listed under `escapes` with a reason, and every listed one still is one;
- every `.lean` file of the package is imported by a root, or exempt with a reason;
- no source file or lakefile names `debug.skipKernelTC`, an option that turns proofs into `sorry`, a `bootstrap.*`
  option, `addDeclWithoutChecking` or `addDeclCore` (comments and docstrings may);
- nor does the source of a module the package imports from a fetched package (`.lake/packages/*`: Mathlib, ArkLib and
  what they require), or the lakefile of a package it imports from, unless `dependency_unchecked` lists the module with
  the reason. The imported modules are the ones the `dependencies` record covers, and their sources are there once the
  package is built, so the scan builds nothing. The
  toolchain, which defines these options, is trusted, and a path dependency is a package of the repository, audited on
  its own;
- no source file runs code while it compiles (`initialize`, `#eval`, `run_cmd`, `run_elab`, an `elab`, `macro` or `syntax`
  definition, an elaborator attribute), unless its module is listed under `compile_time` with a reason and the digest of
  its file as it was reviewed. No listed module may declare a guarantee or a definition a guarantee reads.
  The package has a `lakefile.toml`, not a `lakefile.lean` (which `lake` runs whenever it loads the package);
- the `.olean` files of the modules the package imports, as each fetched package (Mathlib's downloaded cache, ArkLib's
  build) and the toolchain provide them, are the ones recorded under `dependencies`. A dependency's other modules may be
  built or not. Lean's `.olean` files are reproducible, so the record holds on any machine. It changes when the package's
  imports change (run `--update`) or at a bump; checking it takes seconds;
- each `layers` rule holds: the modules under a prefix import only the listed prefixes. A rule with `"sealed": true` is
  a layer of its own: its modules answer only to the rules at or under it, and another rule reaches it only by listing
  a prefix at or under it, never through a shorter one that holds it (an area's `Guarantees` under the area's
  vocabulary);
- the trees hold (`trees`, below): no guarantee's statement reads a definition of the proofs tree, and no module of the
  definitions or properties tree imports one;
- every `Prop` a guarantee takes is named, another listed guarantee's statement or a plain range (`inline_failures`;
  `shape.inline`, which only shrinks, excepts some, and its size is the `AUDIT` line's inline debt);
- each deployed guarantee's gap is its `trust_gap` entry (`trust`, below);
- every instance has a name in its source. Lean names an anonymous one from its type, and adds the project's root
  (`_proofs`) when nothing in the type is the project's own, so a move into another root or a split renames it, and every
  record whose definitions use it changes. `--name-instances` writes each current name into its declaration;
- the guarantees say what their records say, and each names its owner (below);
- with a `verifier` section, nothing the verifier's `entries` reach is `@[implemented_by]` or `@[extern]` outside `allow`,
  and they reach the `hash` namespace only through `stop` (`verifier_failures`). `allow` is a record: `--update` writes
  each such constant on the path into it by name, so the review and the landing DM show one added, and a name that admits
  nothing on the path fails; a `module:M` entry, which admits every constant of the modules under `M`, is kept as
  written (Flock's lock lists names only). The walk follows definitions' values,
  `implemented_by` targets and `@[csimp]` replacements, the code compiled in place of a constant; a csimp rewrite itself
  is a theorem the replay checks, so it is listed (`csimp` in the facts) and not refused;
- with a `claims` section, every predicate the spec declares under `claims.modules` (a named assumption aside) has its
  witnesses under `claims.witnesses`, or a reason under `claims.not`: a guarantee `fails`, stating `¬ C …` past its
  binders, and guarantees `meets`, one per deployed record, that state the claim's checkable antecedents there and not
  `C` (`claim_failures`, not `shape`). A `meets` witness says something about the deployment: past its binders,
  split at `∧`, each conjunct is headed by a constant other than `C`, applies a definition under `meaning` (the
  deployed record or a field of it; not `True` or Init or Mathlib alone), is no reflexivity nor one of its hypotheses,
  and no hypothesis is `False` (`vacuous_meets`). A closed `def C : Prop` proved of every instance goes under
  `claims.generic` with its guarantee, which reads only the claims and Core and skips `shape` and the inline rule;
- a guarantee of an acceptance or an exclusion, unless it states `a = a` (`reflexive`), has `honest` and `nontrivial`
  witnesses of the form `witness_failures` states, or an `unwitnessed` entry, which only shrinks; `nontrivial` is one
  theorem or a list, each past the first in the exclusion form and dropping a premise no earlier one drops;
- every definition a guarantee's statement reads is in the spec (below), or a pinned dependency's;
- no dependency the `upstream` section scans proves, at the revision `lake-manifest.json` pins, a theorem about a watched
  statement that nobody has acknowledged (below);
- the kernel accepts every declaration again (`Replay.lean`, on one import of the package's dependencies, each
  declaration against the import and the declarations it comes after, on `LEAN_NUM_THREADS` threads, by default
  `cores`): a declaration that skipped the kernel shows no axiom, and only the replay catches it.
  The replay also walks the environment it checked for the axioms the package uses, and the audit fails if they differ
  from the sets the `.olean` files recorded. A replay killed by a signal
  (out of memory) is reported as a kill, not as a verdict. With `--closure` it checks the whole import closure the
  same way (below);
- with `--build`, a sample of the modules the build didn't compile (restored from the node's artifact cache, or kept
  from an earlier build), taken after the build with randomness nothing the build runs sees, compiles again in the
  sandbox with no cache, in an overlay that never writes the build, to the outputs the build holds: byte for byte, and
  `olean.server` and `olean.private`, which keep the source's absolute path, as object graphs (`sampled_recompile`,
  `olean_graph.py`; failure `sampled`). The report's `sampled` names the modules. With no sandbox none is sampled
  (`unsampled`), so nothing is restored or kept: every build is cold (`staged`), and a module restored anyway fails. A
  sample can miss a wrong module, so the uncached nightly check stays the backstop;
- each test under `runs` passes on the data whose SHA-256 the record holds (below), and its `generate` and `args` name no
  path outside the package except as `{repo}/...`, which check's scratch tree would not hold;
- nothing changes this directory, the toolchain's `lean`, `lake` and `leanchecker`, or the `.olean` files of a package
  audited earlier while the audit runs: the Lean files run from a copy taken before the first build, and everything is
  sealed and checked after each package.

`--build` builds in `sandbox.sh` where `unshare` and unprivileged user namespaces are available: new user, mount, network
and PID namespaces, everything read-only but the build's `.lake` directories and a private TMPDIR, no network, and nothing
outliving the build.
The audit says when it had to build without one; the `compile-time` check then stands in for the sandbox. `--build` first
takes one of the node's Lean audit slots (`lean_slot`; pool `LEAN_SLOT_POOL`, else `audit`) unless it holds one, and exits
2 when refused. Packages are audited side by side after one `setup.sh`, each build after those of the packages it
requires by path, never beside one that writes the same `.lake` (`writing`). `check.py` exits 0 when every package
passes, 1 when one fails, and 2 on a usage error or a refusal.

A package that pins ArkLib (`Security/Proofs`, and `Security`, which audits it beside itself) is refused, exit 2, on a
machine with less than `MEM_FLOOR_GIB` (24 GiB, memory and swap within the cgroup), by `check.py` and the fast path
alike; the refusal prints the fast path's `research run --on auto …` line (`BELOW_FLOOR_RUN`). `VERITY_LEAN_MEM_FLOOR_GIB` sets
another floor (0: none). Unless `LEAN_NUM_THREADS` is set, the build and the replay of a package that pins Mathlib or
ArkLib run only as many threads as its cgroup's memory holds (`build_threads`, `replay_threads`), one such replay at a
time. An audit of a spec beside the ArkLib package that proves its guarantees (`Security`)
is refused below `SPEC_FLOOR_GIB` (160 GiB; `check.py` says why), and the
refusal names `--kind lean-audit`, a 280 GiB shape.

**The replay cache and the import closure.** Without `--closure` the replay trusts the package's dependencies (the
toolchain's library, Mathlib, ArkLib) as their `.olean` files state them: a dependency module holding a theorem that
skipped the kernel passes it. `--closure` closes that gap module by module. Every module of the closure gets a key
(`replay_keys`): the toolchain, `Replay.lean`, the module's `.olean` files, and the name and key of each module it imports,
so it covers every `.olean` file under it. With `--replay-cache DIR`, a module whose key `DIR` holds passed the replay
before, with its imports in the environment, and a module checked is stored there once the replay passes. The replay of
a package then sends to the kernel each own module and each module of the closure that has not passed (`closure_plan`),
loads unchecked the modules that passed and import one of those, and imports the rest. By induction over the import DAG
every declaration of the closure has been through the kernel, with its imports present, under a key that covers them.
The cache is shared between packages, so
`Security` reuses the Mathlib modules `Security/Proofs` checked. So that a module's verdict is a function of its key,
`Replay.lean` pins what the kernel loads: no constant from outside the module's imports, an equation lemma two modules
realize checked once each, a reused public part's axiom copy taken as Lean's import does (`subsumesInfo`), and the
quotient's own constants. A change to `Replay.lean` changes every key, so the next run is cold.
A key vouches for what a module's files hold, not for who wrote the entry. So the cache is used only on a node whose
builds run in the sandbox, and only where no build can write it: `check.py` refuses a `--replay-cache` or
`--facts-cache` inside a build's `writable` roots (`build_roots_over`), and on a node without the sandbox the replay
neither reuses nor stores an entry (the report's `replay_cache` says so).
The audit prints `closure: N modules checked, M reused` under each package's line. `check`'s lean-audit step runs
`--closure` against the node's kept cache (`lean_audit.FLAGS`, `lean-replay/`), so a dependency module that skipped the
kernel fails the merge check, and the `closure_dependency` control fails without `--fresh`. `lean_audit.py --fresh` (the
uncached nightly check) gives its packages one cache in its scratch tree.

`--fresh` also runs `leanchecker --fresh` on each root: the root's whole import closure, Mathlib and core included,
through the kernel into an empty environment. It needs one module per root and a lot of memory, and checks the closure
again on every run, so it runs nightly on `main` instead (`[[run]]` in `tools/research/README.md`). `--closure` with a
kept cache, which `check` runs, checks the same closure once per change of its `.olean` files.

`--all` first runs the controls (`--no-controls` skips them; `check`'s audit does, and the `tools/verity/lean` suite runs them, only
when `tools/verity/lean/` changes): `tests/fixtures/control` and `tests/fixtures/modules`, built with each audited package's
toolchain (`--build` installs it first, so a machine without elan works), must pass clean and fail once for each violation
in `tests/fixtures/violations.json` or `module-violations.json`. Among them:

- a false theorem added through `debug.skipKernelTC`, spelled so that no scan finds it, which must pass every check but
  the replay and fail the replay;
- an initializer that tries to forge the facts, which must never run where the audit reads the package;
- compile-time IO that rewrites a checker input and restores it, which the compile-time check refuses and the sandbox
  blocks;
- a docstring that names `debug.skipKernelTC`, which must pass;
- a fetched dependency (a package at `.lake/packages/dep`) whose module sets `debug.skipKernelTC`, which must fail, and
  one whose docstring and comments name it, which must pass;
- a dependency that adds a false theorem through `debug.skipKernelTC`, spelled so that no scan finds it, which must pass
  every check, the replay of the package's own modules included, and fail the replay of the import closure
  (`--closure`).

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
 "trees": {"definitions": ["Pkg.Model"], "properties": ["Pkg.Properties"], "proofs": ["Pkg.Proofs"],
           "except": {"Pkg.Proofs.Old": "why a statement may read it until its definitions move"}},
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
  a test of the package's own (PoUW's `verity/protocols/pouw/verifier/tests/test_lean_package.py`) checks that the generator still writes the recorded
  data. `--no-runs` skips them.
- `assumptions_module`: the module (or a submodule) with named assumptions, or a list of such modules (`Security/Properties`
  lists `Properties.Assumptions` beside `Definitions.Assumptions`, for an assumption stated over the code); each must be
  a module of the build or of a package it requires. `assumptions`: the named assumptions (with
  `owner`), the premises declared there. `domain`: a premise that is part of what its guarantee says (a scope, the layer
  below's guarantee), with `owner` and a one-line `why`; `--update` adds one that is another listed guarantee's
  statement, citing it (`cites`), and an honest witness must conclude each. `open`: every other premise, with `owner`,
  the Linear `issue` that discharges it and its `due` date, by hand: an unlisted one fails, naming the entry to write,
  and only `open_seed` ({module prefix or owner: {issue, due[, owner]}}, consumed) lets `--update` add one. A premise is a closed hypothesis, or one a statement takes whose head the repository builds
  (`Facts.lean`'s `premises`); a `Prop` a statement only reads is in neither, held by its `reads` record. A new named
  assumption fails until `--update` adds it with its guarantees' owner. An alias of listed premises is no new one, open or closed: a definition whose value asserts, through `∧`, the bodies of
  `∀` and `→` and reducible unfolding, only listed premises or such aliases (`Facts.lean`'s `unfolds`; what it takes may
  mention only listed ones). Its parts are held to the lock in its place, and a closed one's are the guarantee's. One
  that asserts anything else (an unlisted `Prop`, `False`, a bound predicate, a constant applied to a proposition or a
  predicate, a recursive definition) is a new one.
  In a conclusion, and under a definition outside a `Guarantees` module, a premise is a constant of an assumptions
  module or of a module under `shape.premises` that a hypothesis mentions, or a hypothesis's head that the repository
  builds. `--update` writes each one's `digest`: its `reads`
  record's, else that of it and every definition it reads in `meaning` and this repository's modules
  (`assumption_reads`). So a change to a premise or to what it reads fails under its own name,
  and the landing DM shows it; one no guarantee reads fails.
- `trust`: `base` ({premise: the trust assumption it states}, the same in every lock) and `extra` ({deployed guarantee:
  {premise: the ruling's rule}}, what it alone may rest on). `deployed`: the guarantees that state the deployed system.
  `trust_gap`: {deployed guarantee: its premises outside its trust} (`trust_gaps`), which only a person adds to and
  `--update` drops from (`trust_failures`); the report prints each gap.
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
- `guarantees`: the guarantees (GLOSSARY.md) and their witnesses (`witnesses`, `claims.witnesses`), never another lemma.
  `--update` fills each record, and a `reads` section, from the build: the signature, the named assumptions among the
  hypotheses, the `premises` (its named assumptions, open premises, and the `Prop`s it only `reads`), a hash of the type, and for each module the
  definitions the guarantees read, with a hash each (not who
  reads them), and a `digest` of those hashes. Without `--update` the audit fails on any difference, so a statement, or
  a definition it depends on, cannot change without a visible diff of this file. A record holds the statement, not the
  proof: list a `sorry` stub, and the guarantee passes once a proof with allowed axioms replaces it. A record that
  holds them under their old name, `pins`, is read so and rewritten by `--update`; one holding both is refused.
- `owner`, in each guarantee's entry: the registry handle of the lane that answers for it, whom a broken or restated
  guarantee goes to; an entry without one fails. Write it by hand, or pass `--update --owner @lane`, which owns only the
  new entries (`--update` keeps an entry's owner). A changed owner is one line in the review, not a changed statement.
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
  own `Nonempty` among them. A statement that fits written in another order fits as written (outside a probability
  statement), and so does one with a hypothesis equating one field of two inputs, the field bound once: `check.py`'s
  `reordered` and `shares` say when. A guarantee under `frozen` (with why it doesn't fit yet) is
  exempt, and fails once it fits, so the list only shrinks. Every `def G : Prop` of a spec module that holds statements (`"statements"`: module prefixes; by default a module
  with a `Guarantees` name part) must be the statement of a
  listed guarantee, or be under `frozen` until one is. A reason starts with its category (`restatable`, `frozen`,
  `falsified`, or `unstated` for a statement no guarantee states), so a record counts its guarantees by category. A
  reason starting `lemma` or `completeness` is refused with its fix: unlist it (a spec statement only it states moves to
  `Proofs/<Area>/Lemmas/`), or list it under `witnesses` as its guarantee's honest witness. `"conjectures": {P: {"status": "open" | "falsified", "test": T, "why": W}}` names the
  conjectures a property may rest on, each a named assumption: one is a premise when it heads a hypothesis, and a
  property with a hypothesis reaching one lists it under its entry's `rests_on` (it is conditional). An open one names
  its falsification test `T`, a pytest node id `PATH::NAME` from the repository's root that a suite runs
  (`tests/test_lean_packages.py` checks the test exists), so a counterexample fails `check`; a falsified one refuses every
  property resting on it, which stays under `frozen` only as `falsified`. Every package is checked: without a `shape`
  section each guarantee is frozen or fails. An entry added to `frozen`, and a conjecture added or with a new status,
  is a claims change: the landing DM shows it with its reason, and nothing waits on it.
- **The spec**, what a guarantee is stated over, is the `layers` prefixes: the modules whose imports the
  policy rules. Every definition a guarantee's statement reads (theorems are not read: a statement means the same
  whatever proves the propositions in it), followed through `meaning` and where the walk leaves it, must lie in the spec
  of its package or of a package it requires by path (by that package's own `layers`), or in a pinned dependency (the
  toolchain, Mathlib, ArkLib). A definition in a spec module must also be under `meaning`, or the record would hold
  nothing of it. `reads_exempt` ({prefix: the reason}) lists the module prefixes a guarantee may read outside the spec
  (a package whose statements are about its executable, not about a model of it); an entry that exempts no read outside
  the spec fails, and no entry covers a spec module outside `meaning`, nor a module of the proofs tree (the trees check
  judges those).
- **The trees**: `trees` names the module prefixes of the definitions tree (each area's vocabulary, named assumptions
  and guarantee statements), the properties tree (statements about the code) and the proofs tree (every proof, lemma and
  proof-only definition); a module's tree is that of its longest prefix (`Proofs.Properties` before `Properties`), and a
  package's trees reach those that require it by path (`Security`'s reach `Security/Properties` and `Security/Proofs`).
  The check refuses a guarantee or property whose statement reads a definition in the proofs tree (move it to the
  definitions tree, or unlist it as a lemma), and a module of the definitions or properties tree that imports one.
  `except` ({prefix: the reason}, the package's own) lists exceptions by the read or importing module; an entry that
  excepts nothing fails. Excepted modules are still recorded. The report
  also lists, without failing, the spec definitions no guarantee of the package or a dependent one reads (`unread_spec`
  in `report.json`).
- **Heartbeat overrides are listed, never refused.** Each `set_option maxHeartbeats N` (and
  `synthInstance.maxHeartbeats`) in a package's sources is in its report (`heartbeats`, printed under its `AUDIT` line):
  its file and line, its value, its reason (a `--` comment ending its line, else those just above), and its module's
  build time in the audit's build.
- **The hashes are SHA-256.** They are taken over a canonical form of the kernel term (`Facts.lean`'s `canonText`),
  which keeps what Lean's `Expr.hash` keeps (constant names, universe levels, bound variable indices, projections,
  literals) and drops what it drops (binder names and binder info, `let` names, metadata). Each text holds the matchers and proofs Lean shares across a module in place, not by name, so
  neither a file split nor a module-system port changes a hash.

**A changed guarantee is shown when it lands.** A weakened definition leaves every proof valid, so the axiom checks and
the replay pass it: only the records see it, and only a person can judge it. `--update` records the change and prints
it: each changed guarantee's signature before and after, and each changed definition a guarantee
reads, with the guarantees that read it and its source now (also in `review.txt` among the reports). Say so in the PR.
An audit without `--update` whose only failures are record differences (a guarantee's signature, named assumptions,
premises or type hash, a `reads` group, a premise's digest) writes the `lean-audit.json` and `review.txt` that `--update`
would among its reports; its `report.json`'s `regenerated` names that file, and `changes` lists the review's
statement changes (`entry`, `kind`: the lock's section, `owners`, `line`). A missing record or name, an owner or any
other failure writes none. Code decides which
failures those are, not text: each failure's code (`Failure`) is in `report.json`'s `failure_codes`
(`settled_codes`: what `--regenerate` settled), and the record differences are `RECORD_DIFFERENCE`'s. The Lean check of a commit of `next` runs with
`--regenerate`, and drive commits the records it regenerated when it lands that commit (drive.toml's `regenerate_locks`);
drive's after-landing run passes `--no-regenerate`, so a record difference that reaches `checked` fails there.

**Bookkeeping warns after a landing.** Drive's lock run (the lean-audit step of the merge queue's check with
`--bookkeeping-warn`, reaching `check.py` as `--bookkeeping warn`) passes a package that fails only on bookkeeping
(`BOOKKEEPING`) on the lock `repaired` writes:
- an unlisted premise, an `assumptions` entry declared outside the assumptions module, or an uncovered `domain` entry is
  listed under `open`, owned by its guarantees' owner (the first, of several), on the lock's `open_default_issue` (else
  none, which its warning asks the next PR for), due seven days from the run;
- a guarantee the witnesses rule now holds to witnesses it lacks is listed under `unwitnessed` at its current hashes;
- a stale entry (a `trust_gap` premise too) is dropped, and a missing owner is set to the guarantees' owner, else the lock's `default_owner`.

The run then judges the policy's checks again on that lock, so whatever else fails still fails: the build, the replay,
an axiom beyond the standard three, `sorry`, `native_decide`, a kernel bypass and a malformed lock. `lean_audit.py` also
repairs the caps, in copies that audit.json's `repaired` names (`raised_unread`, `held_caps`): an unread area over its cap,
and a lock count over its cap in `tests/test_lean_packages.py` (UNWITNESSED, INLINE), are raised to the count, and such a
cap more than its `ROOM` above its count is lowered to the count and its room. Elsewhere a loose cap only warns (`_held`), so lowering a count never fails a check. Each repair is one
warning naming the entry and its owner, in audit.json's `bookkeeping`; a package's are its report's `repairs` (`entry`,
`code`, `owners`, `line`). Drive commits the repaired files with the locks
(`research.merge.REPAIRED`, held byte for byte to the run's copies), lists the warnings in the lock commit, and tells the
Lean lane and each owner they name; the next PR fixes them. No other check takes the flag.

When the change lands on `main`, the landing DM (`research.spec_alert`) lists each new, removed or changed guarantee and
each changed definition; nobody approves it first. A statement that only prints differently (the same type hash and named
assumptions: a toolchain bump, notation) gets one line, "printed differently", and stays out of the DM; a definition read
from another module with the same content is counted on one line.

**A lift is not a change.** Lifting a guarantee into the spec, `theorem X (xs) : e` becoming `theorem X (xs) : G as` with
`def G (ys) : Prop := e'` in a `Guarantees` module, changes its record's signature and type hash. `check.py --update`
certifies the lift as the same statement when the type, with `G` unfolded once, hashes to the record's type hash and the
named assumptions are the record's (`lifted_into`); `Facts.lean`'s `liftOf` says how it unfolds `G`, by delta alone, and
the text is hashed as every type is. A certified lift gets one line, "lifted into G: the same statement, its definition
unfolded once", and `G`, new and read by no other guarantee, needs no reading of its own; the review then holds no
statement change. Any other lift prints in full: a `G` whose value differs from `e` in anything the hash sees
(binder names and metadata aside), a `G` outside a `Guarantees` module, and a lift that moves a named assumption from the
theorem's binders into `G`, which the record's `assumptions` would lose. The landing DM reads only the records, so it
still lists a lifted guarantee as changed.

`four_names` lists the root of each of the lock's guarantees (`["Nci", "NetTiming", …]`), and each is `X.SecurityProofs.G
: X.Guarantees.G` (universe parameters aside), `G` a definition of a `Guarantees` module, but a witness, a frozen guarantee,
a generic claim's (`claims.generic`: stated as its claim) and one `four_names_except` holds ({G: why not yet}). An entry
there in the form, frozen, a witness, a generic claim's or gone fails.

`check.py --lift LIFTS.json PKG` writes lifts into the four-name form: `def G : Prop` in the spec module, and
`theorem X.SecurityProofs.G : X.Guarantees.G := @X.t` beside the proof or in the module `"beside": M` names (`check.py`'s
`lift` says what each copies and imports, and what it refuses). What it cannot follow (an `open` of a namespace a
definition module opens only through a proof's declaration, a `variable` an `omit` or `include` changes) fails the build
or prints as a change, never as the same statement. It writes
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
on one line. The landing DM leaves out a guarantee renamed with its type hash unchanged and
a definition moved with its content. Everything else prints as without the map. A map the comparison could not read back
is refused (`check.py`'s `move_errors`); an old theorem kept under its old name as the lemma behind a new guarantee is no
conflict.

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

The proofs package's own audit replays its declarations through the kernel, so `check.py SPEC` audits each package
`proved_in` names in the same run, without `--update` unless it is named too, and the spec's reading waits for its build.
`check`'s key for a spec covers the files of the packages it names, so the two miss and are audited together.

## Merging records: `merge.py`

Two branches that each add guarantees both change `lean-audit.json`, usually in the same `reads` module, so git's text merge
conflicts. `.gitattributes` gives the file to `merge.py` instead, registered by `merge.py --install`. It merges entry by entry: each side's change to a guarantee, a module under `reads`, a dependency or a policy entry applies.
Within a module both sides changed, the definitions merge the same way, and the digest is recomputed; an older lock's
readers and `pins` are read as today's. A guarantee's `owner` merges apart from the rest of its entry, so one side's new
owner and the other side's restatement both apply.

Apart from the guards below, the merge never conflicts. What the build determines (a guarantee's record, `reads`,
assumption digests) merges as the union or theirs' value, and the lock run regenerates it. What a person
writes (an owner, an `issue` or `due`, a `why`, a reason, the roots, an exemption, a dependency) merges as the union of
both sides' entries: one both sides changed differently is theirs', and one a side removed while the other changed it is
the changed one. An entry the sides moved to different sections of an exclusive group (`witnesses`, `not`, `generic`;
`assumptions`, `domain`, `open`) is where theirs put it, for `check.py --update` to classify. Each entry taken from one
side is one warning line. A side that isn't a record (bad JSON) yields to the other, with a warning; two bad sides leave
git's conflict.

Git runs a driver from the worktree you merge into, so the registered command (`research.clone.RECORD_DRIVER`) runs
`origin/main`'s copy of this directory when that ref exists, else the worktree's: fetch `main` first. Three guards leave
git's conflict, where it beats a silent loss: a lock `format` above `FORMAT` (bumped in both files with the lock's
schema), the worktree's copy merging a side with newer `merge.py` changes (`stale`), and a merge that drops most of a
side's section (`losses`).

A merged record is a candidate. `check`'s Lean audit rebuilds every record of the merged tree and fails on any difference,
so a merge that is clean but wrong (a definition that changed on one side without a guarantee there reading it, while the
other side's new guarantee reads it) fails there. `merge.py --normalize FILE` recomputes a record's digests without a build.
The fast path (`lean_changed.py --records`) takes such a candidate as it is: after a merge of `main` that takes every Lean
input of a package it changes whole from main's side and its lock from `merge.py`, on top of a commit whose records passed, it runs
nothing for that package (`merged`), and neither does it on inputs whose records already passed on that node or in a result
the node's store holds (`kept`, by `inputs_key`; `lean_changed.py --inputs-key PKG --head SHA` prints the key from git alone).

## Noticing when a dependency proves what we assume: `upstream.py`

A named assumption, or a lemma we prove by hand, may be proved upstream one day if its statement can be written in a
dependency's constants, and the proof may arrive as a different theorem, with a different constant, under a name nobody
watches. So the `upstream` section of the package's policy names what each watched statement is about (`mentions`), a
hit is a theorem whose statement mentions all of them, and the status comes from the dependency, not from a document;
`upstream.py`'s docstring has the section's format and its two passes. A statement over the package's own definitions
has no entry, since no upstream theorem can mention them, and neither has one about a single number, whose entry could
mention only `Nat.Prime` and would match every theorem about primes.

The audit runs the no-build pass for every package with an `upstream` section, so a bump that brings a proof about a
watched statement fails until someone has checked whether it discharges the assumption or replaces the hand proof and
recorded it with `--ack`: the list says what has been looked at, not what was used. Run the exact pass (`--lean`) at a
bump; its first run builds all of ArkLib. `upstream.py PKG --fetch --rev origin/main` shows what upstream has proved
since the pinned revision, and a weekly `research run` of it tells us when a bump would gain something. A document's
"status upstream" line cites this output instead of asserting a status.

## The build cache: `cache.py`

Integration publishes the build of each package at every merged commit, and a development machine restores the newest one
that is an ancestor of its `HEAD` (`cache.py get`, after `setup.sh`): every file is checked against the commit's sha256
record before Lake unpacks the modules whose input hashes match, and the rest compile, so a fresh machine compiles none of
ArkLib. A miss, a mismatch, a missing read credential (the `LEAN_CACHE_*` variables, else
`~/.config/verity/lean-cache-read.env`), the kill switch or `LEAN_CACHE=off` means building from source. The audit never
reads it: `lake_env()` sets `LEAN_CACHE=off`, and check's audit and a records run restore only from their node's own Lake
cache, which only a passing run there fills, by each module's compile key (`lean_audit.LakeArtifacts`, `module_keys.py`). `cache.py`'s
docstring has the bucket's layout and how integration publishes (from a clean
checkout of `origin/main`), and `run_daily.py`'s the daily publish.

## Using it from another repository

Copy this directory at a commit of Verity's `main` and record the commit beside it. Write a `lean-audit.json` for each
package (`check.py` prints a template when one is missing), run `check.py --update` once on the trusted statements,
review and commit the records, and run `check.py --all --build` wherever the repository runs its checks.

These records suit statements that are merged and reviewed. An external, untrusted submission still needs its theorem checked by
`isDefEq` against a trusted `Prop` taken from the trusted commit, never from the submission's tree, and the submission
built in a sandbox, as POUS's grader does.
