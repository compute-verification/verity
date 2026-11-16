# tools/pous_grader: design

## Conventions

Declarations kept their names
(`python3 tools/move/lean.py --resolve MODULE` maps a module's old name). Conventions follow the other Lean packages:
the same toolchain and Mathlib pin, `autoImplicit` off, no `axiom` (an assumption is a named `Prop` hypothesis), and a
check file.

## In this package

~~~text
lean-audit.json                     this package's audit policy for tools/verity/lean/check.py
Grader/{Main,Registry}.lean, grade.sh, TRUSTED.sha256   the grader (allowed axioms and pinned targets compiled in),
                                    its driver, the trusted-file hashes
submissions/reference.lean          the grader's positive control
~~~

~~~bash
lake build                          # the grader (the default target)
(cd ../../security/Proofs && lake exe cache get && lake build Proofs.Pous)   # PoUS's analysis; Mathlib prebuilt
python3 ../verity/lean/check.py --build --fresh . ../../security   # Verity's Lean audit
bash grade.sh submissions/reference.lean Pous.Guarantees.KFloor   # one JSON line, exit 0 iff PASS
bash check.sh                       # the audit and the grader's controls
~~~

## The 320-read certificate

`ObligationWide320` is stated at uniform indices. The 320-read certificate (`p2.wide320_certificate`, and
`PousCheck.RunProtocol.wide320Certificate` in Lean) adds `2^-127` for the memory attester's SHA-512 indices, assuming
`ε_prf ≤ 2^-127` under `prf/sha-512`: `Hash.Assumptions.Sha512Prf` takes the advantage as a parameter, which no model
fixes.

## The audit

**The audit.** `tools/verity/lean/check.py` audits every declaration of PoUS's modules: allowed axioms, named assumptions,
escapes, orphan files, import layers (Definitions imports no assumption, guarantee or proof; Assumptions only
Definitions; Guarantees no proof), a kernel replay, and the guarantees' records. `security/lean-audit.json` lists
as guarantees the sanity theorems, every `Pous.SecurityProofs.X` (among them the stacked one-way bound, the band
certificate, and the dense scheme at the 64 KB point, with the implementation's tagged keying and its per-segment tags),
the raw-storage refutations the `m1p` guarantees cite, and P2 in the cost model (`PousP2Ring`: `meets_of_meetsOn`, the
non-vacuity theorems `complete`, `tight` and `tight_p16448`, and the refutation `not_meetsOn_b19_k88`), and the
continuous audit's counting step (`PousContinuous.pr_misses_le`, `pr_misses_gt_le`). When a changed record lands on
`main`, a DM shows it (Verity's `AGENTS.md`).

## Submissions

**Submissions** are one file that imports only the trusted layers (`Definitions.Pous`, `Definitions.Pous.Assumptions`,
`Definitions.Pous.Guarantees`, `Proofs.Pous.Statements`), the proved `Proofs.Pous.Sanity` and `Proofs.Pous.Accounting.Numbers`, and `Mathlib.*`, and declares `theorem solution : Pous.Guarantees.<Name>`. `grade.sh` builds the trusted layer and the
grader first and records their hashes; compiles the submission with `lean` in a scratch directory inside a user, mount,
PID and network namespace where every other mount is read-only (so compile-time IO can change nothing, and its processes
die with the build); checks the hashes again; then runs the grader. The grader loads the submission's `.olean` as data,
replays every constant it adds through the kernel, requires `solution`'s type to be the pinned constant itself (a
syntactic comparison, so a near-miss statement cannot keep the kernel unfolding), and requires its axioms to be in
`allowedAxioms` (Lean's three, compiled in). The compile and the grader run under a time limit and a memory limit
(`POUS_GRADE_TIMEOUT`, `POUS_GRADE_MEMORY_MB`). `grade.sh` needs unprivileged user namespaces (`unshare --user`). A
graded proof moves into `security/Proofs/Pous/` as a module, with its theorem added to
`security/Proofs/Pous.lean` and listed as a guarantee in `security/lean-audit.json` (`check.py --update`,
then review the records' diff).
