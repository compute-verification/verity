# POUS Lean package

Definitions, pinned statements and proofs for the timed POUS audit game with perfect isolation (`../PROTOCOL.md`).
Conventions follow Flock's Lean packages (`backends/flock/verifier/lean/`): the same toolchain and Mathlib pin,
`autoImplicit` off, no `axiom` (an assumption is a named `Prop` hypothesis), and a check file.

~~~text
Pous/Game/{Prob,Oracle}.lean        trusted: bits, uniform probability, ideal primitive, parallel oracle programs
Pous/Model/{Scheme,Audit,Inc}.lean  trusted: encoding schemes, the audit game, INC and TimedINC
Pous/Model/{M1,HashChain,Pebbling,Labelling,Invertible,ColumnGame,Dense,SampledAudit}.lean
                                    trusted: model M1, B3's chain, Fisch's stacked DRGs and white/green game,
                                    Pietrzak's one-way labels, P3's invertible labels, the two-directional column
                                    game, the dense scheme S1, the state after a history
Pous/Accounting/{Params,LemmaA}.lean  trusted: ρ, δ, expansion, `Meets`; Lemma A's bound and A7′'s rows
Pous/Assumptions.lean               trusted: named hypotheses (Lemma A, Conjecture B1′)
Pous/Pinned.lean                    trusted: the 33 statements grade.sh grades against
Pous/Sanity.lean, Pous/Accounting/Numbers.lean   proved: the definitions are not vacuous; certified numbers
PousProofs/                         proved: Theorem 1, Track A, pebbling, B5, sampled audits, the dense scheme, P3
PousProofs/Pinned.lean              PousProofs.X : Pous.Pinned.X for each proved pinned statement
lean-audit.json                     the policy and pin records for tools/lean/audit.py
CheckAxioms.lean                    one #print axioms line per headline theorem
PousTargets.lean                    untrusted, not in the default build: the open statements as sorry targets
Grader/{Main,Registry}.lean, grade.sh, TRUSTED.sha256   the grader (allowed axioms and pinned targets compiled in),
                                    its driver, the trusted-file hashes
submissions/reference.lean          the grader's positive control
~~~

~~~bash
lake exe cache get                  # Mathlib's prebuilt oleans; never build Mathlib from source
lake build                          # Pous and PousProofs (default targets)
python3 ../../../tools/lean/audit.py --all --root . --fresh   # Verity's Lean audit
lake env lean CheckAxioms.lean      # the headline theorems' axioms
bash grade.sh submissions/reference.lean Pous.Pinned.KFloor   # one JSON line, exit 0 iff PASS
bash check.sh                       # the audit, the open targets, and the grader's controls
~~~

**Trusted without proof:** the statements in `Pous/Game`, `Pous/Model`, `Pous/Accounting/Params.lean`,
`Pous/Accounting/LemmaA.lean`, `Pous/Assumptions.lean` and `Pous/Pinned.lean`; Lean's kernel; the Mathlib pin; the
grader and its sandbox. Everything else is checked by machine. Docstring paths such as `lean/submissions/…` and
`docs/…` name the POUS Project store, where the provers' drafts and reviews live.

**The audit.** `tools/lean/audit.py` audits every declaration under `Pous` and `PousProofs`: allowed axioms, named
assumptions, escapes, orphan files, import layers (`Pous` imports no proofs), a kernel replay, and the pinned theorems'
records. `lean-audit.json` pins the sanity theorems, every `PousProofs.X`, the stacked one-way bound, the band
certificate, and the dense scheme at the deployment point (`dense_meets_64`, `seg_meets_14`, and `segTag_meets_14` for the
implementation's tagged keying, with `segTag_auditSecure`) and its per-segment tags (`domainSep_equiv`,
`tag_injective_14`). A changed record needs a named statement reviewer (Verity's `AGENTS.md`).

**Submissions** are one file that imports only `Pous` and `Mathlib.*` and declares
`theorem solution : Pous.Pinned.<Name>`. `grade.sh` builds the trusted layer and the grader first and records their
hashes; compiles the submission with `lean` in a scratch directory inside a user, mount, PID and network namespace
where every other mount is read-only (so compile-time IO can change nothing, and its processes die with the build);
checks the hashes again; then runs the grader. The grader loads the submission's `.olean` as data, replays every
constant it adds through the kernel, checks `solution`'s type against the pinned constant by kernel definitional
equality, and requires its axioms to be in `allowedAxioms` (Lean's three, compiled in). `grade.sh` needs unprivileged
user namespaces (`unshare --user`). A graded proof moves into `PousProofs/` as a module, with its theorem added to
`PousProofs/Pinned.lean` and pinned in `lean-audit.json` (`audit.py --update`, then review the records' diff).

**Changing the trusted layer** means editing the files listed in `TRUSTED.sha256` and regenerating it after review:

~~~bash
sha256sum lean-toolchain lakefile.toml lake-manifest.json grade.sh Grader/Registry.lean Grader/Main.lean Pous.lean \
  $(find Pous -name '*.lean' | sort) > TRUSTED.sha256
~~~
