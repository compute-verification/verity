# tools/lean: the Lean audit

One audit for every Lake package, run by `check` (`python tools/lean/audit.py --all --build`). It needs the standard
library, `lake` and the toolchain's `lean`; nothing here imports Verity, so another repository can copy the directory
(`audit.py`, `Facts.lean`, `Replay.lean`, `sandbox.sh`, `setup.sh`, `tests/`) and point it at its own packages.

No package code runs where the audit decides anything. The facts come from the `.olean` files read as data (`Facts.lean`
imports with initializers disabled and no environment extension loaded), and the kernel replay (`Replay.lean`) loads them
the same way. The only package code that runs is the build's, in a sandbox where the machine allows one.

~~~text
bash tools/lean/setup.sh PKG                      elan, PKG's lean-toolchain, Mathlib's cache when Mathlib is a dependency
python tools/lean/audit.py [--build] PKG...       audit these packages (--build: setup.sh and lake build first)
python tools/lean/audit.py --update PKG...        write the pins' records from the build; commit the diff
python tools/lean/audit.py --all [--root DIR]     the controls, then every package under DIR with a lean-audit.json
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
- no source file runs code while it compiles (`initialize`, `#eval`, `run_cmd`, `run_elab`, an `elab`, `macro` or `syntax`
  definition, an elaborator attribute), unless its module is listed under `compile_time` with a reason, and the package
  has a `lakefile.toml`, not a `lakefile.lean` (which `lake` runs whenever it loads the package);
- the `.olean` files of the modules the package imports, as each fetched package (Mathlib's downloaded cache, ArkLib's
  build) and the toolchain provide them, are the ones recorded under `dependencies`. A dependency's other modules may be
  built or not. Lean's `.olean` files are reproducible, so the record holds on any machine. It changes when the package's
  imports change (run `--update`) or at a bump; checking it takes seconds;
- each `layers` rule holds: the modules under a prefix import only the listed prefixes;
- the pinned theorems say what their records say (below);
- the kernel accepts every declaration again (`Replay.lean`, on one import of the package's dependencies): a declaration
  that skipped the kernel shows no axiom, and only the replay catches it. The replay also walks the environment it
  checked for the axioms the package uses, and the audit fails if they differ from the sets the `.olean` files recorded.
  No `#print axioms` output is parsed anywhere. A replay killed by a signal (out of memory) is reported as a kill, not
  as a verdict;
- nothing changes this directory, the toolchain's `lean`, `lake` and `leanchecker`, or the `.olean` files of a package
  audited earlier while the audit runs: the Lean files run from a copy taken before the first build, and everything is
  sealed and checked after each package.

`--build` builds in `sandbox.sh` where `unshare` and unprivileged user namespaces are available: new user, mount, network
and PID namespaces, everything read-only but the build's `.lake` directories, no network, and nothing outliving the build.
The audit says when it had to build without one; the `compile-time` check then stands in for the sandbox.

`--fresh` also runs `leanchecker --fresh` on each root: the root's whole import closure, Mathlib and core included,
through the kernel into an empty environment. It needs one module per root and a lot of memory, so it is not part of
every run.

`--all` first runs the controls: `tests/fixtures/control`, built with each audited package's toolchain (`--build`
installs it first, so a machine without elan works), must pass clean and fail once for each violation in
`tests/fixtures/violations.json`. Among them:

- a false theorem added through `debug.skipKernelTC`, spelled so that no scan finds it, which must pass every check but
  the replay and fail the replay;
- an initializer that tries to forge the facts, which must never run where the audit reads the package;
- compile-time IO that rewrites a checker input and restores it, which the compile-time check refuses and the sandbox
  blocks;
- a docstring that names `debug.skipKernelTC`, which must pass.

## The policy: `lean-audit.json`, beside the lakefile

~~~json
{
 "roots": ["Pkg"],
 "exempt": {"Pkg.Scratch": "why nothing needs to build it"},
 "axioms": ["Classical.choice", "Quot.sound", "propext"],
 "assumptions": ["Pkg.Assumptions"],
 "meaning": ["Pkg"],
 "escapes": {"Pkg.loop": "why compiled code no proof sees is acceptable here"},
 "compile_time": {"Pkg.Tactic": "why this module must run code while it compiles"},
 "layers": {"Pkg.Model": {"imports": ["Mathlib", "Pkg.Model"], "except": [], "why": "definitions import no proofs"}},
 "pins": {"Pkg.main_theorem": {}}
}
~~~

- `roots`: the modules the audit imports; every module under them is audited. `axioms` is optional.
- `assumptions`: the modules whose `Prop` constants may appear as closed hypotheses of a pinned theorem.
- `meaning` (default `roots`): the modules whose definitions a pinned statement's meaning is followed through. What a
  statement is about (an executable, Mathlib) stays outside.
- `dependencies` is written by `--update`: per fetched package and for the toolchain, a digest of the `.olean` files of the
  modules the package imports.
- `pins`: the theorems a document or a PR cites as proved. `--update` fills each record, and a `reads` section, from the
  build: the signature, the named assumptions among the hypotheses, a hash of the type, and for each module the
  definitions the pinned statements read, with a hash each. Without `--update` the audit fails on any difference, so a
  statement, or a definition it depends on, cannot change without a visible diff of this file. A record holds the
  statement, not the proof: pin a `sorry` stub, and the pin passes once a proof with allowed axioms replaces it.

**A changed pin needs a named statement reviewer.** A weakened definition leaves every proof valid, so the axiom checks
and the replay pass it: only the pins see it, and only a person can judge it. (A redefinition of `Collision` once made
a Merkle binding theorem vacuous for unsalted schemes; the pins flagged it and review caught it.) `--update` records the
change and prints what the reviewer reads: each changed pin's signature before and after, and each changed definition a
pin reads, with the pins that read it and its source now (also in `review.txt` among the reports). The merge handoff
names the reviewer and the pins they read. A pin whose statement is the same but prints differently (the same type hash
and named assumptions, after a toolchain bump or a change of notation) gets one line, "printed differently", and needs no
reviewer.

## Using it from another repository

Copy this directory at a commit of Verity's `main` and record the commit beside it. Write a `lean-audit.json` for each
package (`audit.py` prints a template when one is missing), run `audit.py --update` once on the trusted statements,
review and commit the records, and run `audit.py --all --build` wherever the repository runs its checks.

Pins suit statements that are merged and reviewed. An external, untrusted submission still needs its theorem checked by
`isDefEq` against a trusted `Prop` taken from the trusted commit, never from the submission's tree, and the submission
built in a sandbox, as POUS's grader does.
