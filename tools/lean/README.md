# tools/lean: the Lean audit

One audit for every Lake package, run by `check` (`python tools/lean/audit.py --all --build`). It needs the standard
library, `lake` and the toolchain's `lean`; nothing here imports Verity, so another repository can copy the directory
(`audit.py`, `Facts.lean`, `Replay.lean`, `upstream.py`, `Upstream.lean`, `merge.py`, `sandbox.sh`, `setup.sh`, `tests/`) and
point it at its own packages.

No package code runs where the audit decides anything. The facts come from the `.olean` files read as data (`Facts.lean`
imports with initializers disabled and no environment extension loaded), and the kernel replay (`Replay.lean`) loads them
the same way. The only package code that runs is the build's and, after the facts and the replay, the tests' under `runs`
(each deciding only its own pass), in a sandbox where the machine allows one; a test's data generator is the repository's
code and runs outside it.

~~~text
bash tools/lean/setup.sh PKG                      elan, PKG's lean-toolchain, Mathlib's cache when Mathlib is a dependency
python tools/lean/audit.py [--build] PKG...       audit these packages (--build: setup.sh and lake build first)
python tools/lean/audit.py --update PKG...        write the pins' records from the build; commit the diff
python tools/lean/audit.py --all [--root DIR]     the controls, then every package under DIR with a lean-audit.json
python tools/lean/upstream.py PKG [--lean]        what the pinned dependencies prove about PKG's watched statements
python tools/lean/merge.py --install              once per clone: git merges lean-audit.json with merge.py (below)
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
  its file as it was reviewed. No listed module may declare a pin or a definition a pin reads.
  The package has a `lakefile.toml`, not a `lakefile.lean` (which `lake` runs whenever it loads the package);
- the `.olean` files of the modules the package imports, as each fetched package (Mathlib's downloaded cache, ArkLib's
  build) and the toolchain provide them, are the ones recorded under `dependencies`. A dependency's other modules may be
  built or not. Lean's `.olean` files are reproducible, so the record holds on any machine. It changes when the package's
  imports change (run `--update`) or at a bump; checking it takes seconds;
- each `layers` rule holds: the modules under a prefix import only the listed prefixes;
- the pins say what their records say (below);
- no dependency the `upstream` section scans proves, at the revision `lake-manifest.json` pins, a theorem about a watched
  statement that nobody has acknowledged (below);
- the kernel accepts every declaration again (`Replay.lean`, on one import of the package's dependencies, each
  declaration in a task of its own on Lean's threads, `LEAN_NUM_THREADS`, checked against the import and the
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

`--all` first runs the controls (`--no-controls` skips them; `check`'s audit does, and the `tools/lean` suite runs them, only
when `tools/lean/` changes): `tests/fixtures/control`, built with each audited package's toolchain (`--build`
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
 "assumptions": ["Pkg.Assumptions"],
 "meaning": ["Pkg"],
 "escapes": {"Pkg.loop": "why compiled code no proof sees is acceptable here"},
 "compile_time": {"Pkg.Tactic": {"why": "why this module must run code while it compiles", "digest": "…"}},
 "dependency_unchecked": {"Dep.Loader": "why the package imports a dependency module that names a kernel bypass"},
 "layers": {"Pkg.Model": {"imports": ["Mathlib", "Pkg.Model"], "except": [], "why": "definitions import no proofs"}},
 "pins": {"Pkg.main_theorem": {}}
}
~~~

- `roots`: the modules the audit imports; every module under them is audited. `axioms` is optional.
- `runs`: the package's tests, compiled evaluation that no proof reads (a bulk conformance check over more vectors than
  the kernel pins). Each is an exempt module, run after the replay by `lake env lean --run` in the build sandbox, and
  passes when it exits 0. An argument `{NAME}` is the path of data file NAME, which is what `generate` writes (run first,
  outside the sandbox, with `{repo}` the checkout, `--repo`, and `{out}` the directory it writes) or else the package's
  own file, either one possibly gzipped (`NAME.gz`). A file is used only with the SHA-256 under `data`: another fails the
  test unrun, and `--update` records it. Generated data stays out of Git and out of the key of `check`'s audit step, so
  a test of the package's own (PoUW's `tests/test_lean_package.py`) checks that the generator still writes the recorded
  data. `--no-runs` skips them.
- `assumptions`: the modules whose `Prop` constants may appear as closed hypotheses of a pin: its named assumptions.
- `meaning` (default `roots`): the modules whose definitions a pin's meaning is followed through. What a
  statement is about (an executable, Mathlib) stays outside.
- `compile_time`: the modules that must run code while they compile, one by one. Write the entry with its `why`; `--update`
  adds the `digest` of the module's file. Any later edit of the file fails the audit until someone has reviewed its
  compile-time code and `--update` records the new digest, and `--update` lists the module among what a reviewer reads.
  No listed module may declare a pin or a definition a pin reads: a term elaborator, a notation
  or a macro there could change what a statement means. A proof-only tactic belongs in a module of its own.
- `dependency_unchecked`: the modules of fetched packages (for a lakefile, `<package>/lakefile.lean`) whose source may
  name a kernel bypass, each with the reason. An entry that no longer matches a hit fails, so the list stays current.
- `dependencies` is written by `--update`: per fetched package and for the toolchain, a digest of the `.olean` files of the
  modules the package imports.
- `pins`: the guarantees, the theorems a document or a PR cites as proved (README "Glossary"), and any lemma pinned
  beside them. `--update` fills each record, and a `reads` section, from the build: the signature, the named assumptions
  among the hypotheses, a hash of the type, and for each module the definitions the pins read, with a hash each, the pins that read them, and a `digest` of those hashes (which
  the record therefore determines). Without `--update` the audit fails on any difference, so a statement, or a definition
  it depends on, cannot change without a visible diff of this file. A record holds the statement, not the proof: pin a
  `sorry` stub, and the pin passes once a proof with allowed axioms replaces it.
- **The hashes are SHA-256.** They are taken over a canonical form of the kernel term (`Facts.lean`'s `canonText`),
  which keeps what Lean's `Expr.hash` keeps (constant names, universe levels, bound variable indices, projections,
  literals) and drops what it drops (binder names and binder info, `let` names, metadata). A renamed hypothesis is
  therefore the same statement, and a different term is a different one. Records written with `Expr.hash`, which has
  32 bits, are still checked as they are, and the run notes them. `--update` rewrites them, and its review says
  "rehashed" where the 32-bit hash still matches, so no reviewer is needed for that.

**A changed pin needs a named statement reviewer.** A weakened definition leaves every proof valid, so the axiom checks
and the replay pass it: only the pins see it, and only a person can judge it. (A redefinition of `Collision` once made
a Merkle binding guarantee vacuous for unsalted schemes; the pins flagged it and review caught it.) `--update` records the
change and prints what the reviewer reads: each changed pin's signature before and after, and each changed definition a
pin reads, with the pins that read it and its source now (also in `review.txt` among the reports). The PR description
names the reviewer and the pins they read. A pin whose statement is the same but prints differently (the same type hash
and named assumptions, after a toolchain bump or a change of notation) gets one line, "printed differently", and needs no
reviewer. Definitions are compared by name, so one read from another module with the same content is counted on one line.

**A move is not a change.** When declarations move (pins renamed, the definitions they read renamed or moved to other
namespaces, a statement restated as a `def G : Prop` that a theorem `t : G` proves), write the moves as a JSON object of
old names to new ones, each a declaration or a namespace (the longest prefix applies), and run
`audit.py --update --moved MOVES.json PKG` on the record as it was before the move. The record's pins are renamed, and
`Facts.lean` writes each pin's statement and each definition it reads once more with the build's names mapped back to
the old ones, a statement whose type is a `Prop` definition under `meaning` also with that definition unfolded. The review
compares those texts with the records of the old names: a pin that says what its record says gets one line, "the same
statement under its old names" (moved from its old name, or restated as `G`, which then needs no reading of its own), and the
definitions the same under their old names are counted on one line; neither needs a reviewer. Everything else prints as
without the map. A map whose new names nest differently from its old ones is refused, and so is one under which two
constants of those texts (or two pins) would have one old name, since the comparison could not tell them apart; an old
theorem kept under its old name as the lemma behind a new pin is no conflict.

## Merging records: `merge.py`

Two branches that each pin guarantees both change `lean-audit.json`, usually in the same `reads` module, so git's text merge
conflicts. `.gitattributes` gives the file to `merge.py` instead, once `python tools/lean/merge.py --install` has registered
it in the clone's git config. It merges entry by entry: each side's change to a pin, a module under `reads`, a dependency
or a policy entry applies. Within a module both sides changed, the definitions merge the same way, the readers as sets, and
the digest is recomputed. Pins arrive byte for byte from their branches, and a hand-written section one side extends (a
module added to `meaning`) is kept.

It refuses when both sides changed one entry differently: a pin restated on both, a definition whose content both changed,
a dependency both bumped. It also refuses when one side stops reading a definition in a module the other side changed,
since the record doesn't say which pins read it. A refusal is a conflict in git, with the entries named. Resolve it the
old way: take either side and run `audit.py --update` on the merged tree.

A merged record is a candidate. `check`'s Lean audit rebuilds every record of the merged tree and fails on any difference,
so a merge that is clean but wrong (a definition that changed on one side without a pin there reading it, while the other
side's new pin reads it) fails there. `merge.py --normalize FILE` recomputes a record's digests without a build.

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
used. To see what upstream has proved since the pin, run `upstream.py PKG --fetch --rev origin/main`; a weekly
`research run` of it tells us when a bump would gain something. A document's "status upstream" line cites this output
instead of asserting a status.

## Using it from another repository

Copy this directory at a commit of Verity's `main` and record the commit beside it. Write a `lean-audit.json` for each
package (`audit.py` prints a template when one is missing), run `audit.py --update` once on the trusted statements,
review and commit the records, and run `audit.py --all --build` wherever the repository runs its checks.

Pins suit statements that are merged and reviewed. An external, untrusted submission still needs its theorem checked by
`isDefEq` against a trusted `Prop` taken from the trusted commit, never from the submission's tree, and the submission
built in a sandbox, as POUS's grader does.
