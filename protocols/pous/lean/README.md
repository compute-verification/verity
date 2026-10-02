# POUS Lean package

Definitions, pinned statements and proofs for the timed POUS audit game with perfect isolation (`../PROTOCOL.md`).
Conventions follow Flock's Lean packages (`backends/flock/verifier/lean/`): the same toolchain and Mathlib pin,
`autoImplicit` off, no `axiom` (an assumption is a named `Prop` hypothesis), and a check file.

~~~text
Pous/Game/{Prob,Oracle}.lean        trusted: bits, uniform probability, ideal primitive, parallel oracle programs
Pous/Model/{Scheme,Audit,Inc}.lean  trusted: encoding schemes, the audit game, INC and TimedINC
Pous/Model/{M1,M1p,HashChain,Pebbling,Labelling,Invertible,ColumnGame,Dense,SampledAudit}.lean
                                    trusted: model M1, M1 on `[0, p)` (`m1p`, the abstract model at P2's domain),
                                    B3's chain, Fisch's stacked DRGs and white/green game,
                                    Pietrzak's one-way labels, P3's invertible labels, the two-directional column
                                    game, the dense scheme S1 (one segment, segmented, and tagged as the code
                                    keys it), the state after a history
Pous/Model/{P3Chain,BandChain,PubEncoder}.lean
                                    trusted: the overwrite chain on an ideal permutation, P3 with it as `H`, the
                                    band and the dense chain scheme over it, the public-encoder setup check
Pous/Accounting/{Params,LemmaA,PubMeets}.lean  trusted: ρ, δ, expansion, `Meets`; Lemma A's bound and A7′'s rows;
                                    `PubMeets`, the requirement with a public encoder
Pous/Assumptions.lean               trusted: named hypotheses (Lemma A, Conjecture B1′)
Pous/Pinned.lean                    trusted: 48 statements, the 47 grade.sh grades against and
                                    `InvExPostFactoTwoLayer`, which `N10Refuted` negates
Pous/P2/Pinned.lean                 trusted: six statements about the abstract model `m1p` at P2's parameters,
                                    which grade.sh also grades against; none is about the concrete P2
Pous/P2/Cost.lean                   trusted: P2's cost model, the generic ring over `Z/p` with random odd labels
                                    and the junk label (`ringRO`), the bound `RingBounded D Q L` beside `Bounded`,
                                    `MeetsOn`, `MeetsWithSlack`, and P2 in it (`p2ring`)
Pous/P2/Assumptions.lean            trusted: P2's named open step `PrimeP16448`, the primality of its modulus
Pous/Sanity.lean, Pous/Accounting/Numbers.lean   proved: the definitions are not vacuous; certified numbers
PousProofs/                         proved: Theorem 1, Track A, pebbling, B5, sampled audits, the dense scheme, P3
PousProofs/P2/                      proved: the `m1p` pins (Count, Rows, Domain, Margin); P2 in the cost model
                                    (RingBasic: `MeetsOn` at `L = Q` gives `Meets`, honest storage is accepted;
                                    RingTight: recomputing the roots breaks `MeetsOn`; RingTrunc: dropping low
                                    bits breaks `k = 88` from `L = 2^30`)
PousProofs/Chain/Flat.lean          proved, generated: P3 with a concrete `H`, the band, the dense chain scheme and
                                    the public encoder over the overwrite chain (the graded lanes, flattened)
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
`Pous/Accounting/LemmaA.lean`, `Pous/Accounting/PubMeets.lean`, `Pous/Assumptions.lean`, `Pous/Pinned.lean` and `Pous/P2`; Lean's kernel; the Mathlib pin; the
grader and its sandbox. Everything else is checked by machine. Docstring paths such as `lean/submissions/…` and
`docs/…` name the POUS Project store, where the provers' drafts and reviews live.

**The audit.** `tools/lean/audit.py` audits every declaration under `Pous` and `PousProofs`: allowed axioms, named
assumptions, escapes, orphan files, import layers (`Pous` imports no proofs), a kernel replay, and the pinned theorems'
records. `lean-audit.json` pins the sanity theorems, every `PousProofs.X`, the stacked one-way bound, the band
certificate, and the dense scheme at the deployment point (`dense_meets_64`, `seg_meets_14`, and `segTag_meets_14` for the
implementation's tagged keying, with `segTag_auditSecure`) and its per-segment tags (`domainSep_equiv`,
`tag_injective_14`), the raw-storage refutations the `m1p` pins cite, and P2 in the cost model (`PousP2Ring`:
`meets_of_meetsOn`, the non-vacuity pins `complete`, `tight` and `tight_p16448`, and the refutation
`not_meetsOn_b19_k88`). A changed record needs a named statement reviewer (Verity's `AGENTS.md`).

**Submissions** are one file that imports only `Pous` and `Mathlib.*` and declares
`theorem solution : Pous.Pinned.<Name>`. `grade.sh` builds the trusted layer and the grader first and records their
hashes; compiles the submission with `lean` in a scratch directory inside a user, mount, PID and network namespace
where every other mount is read-only (so compile-time IO can change nothing, and its processes die with the build);
checks the hashes again; then runs the grader. The grader loads the submission's `.olean` as data, replays every
constant it adds through the kernel, requires `solution`'s type to be the pinned constant itself (a syntactic
comparison, so a near-miss statement cannot keep the kernel unfolding), and requires its axioms to be in `allowedAxioms`
(Lean's three, compiled in). The compile and the grader run under a time limit and a memory limit
(`POUS_GRADE_TIMEOUT`, `POUS_GRADE_MEMORY_MB`). `grade.sh` needs unprivileged user namespaces (`unshare --user`). A graded proof moves into `PousProofs/` as a module, with its theorem added to
`PousProofs/Pinned.lean` and pinned in `lean-audit.json` (`audit.py --update`, then review the records' diff).

**Changing the trusted layer** means editing the files listed in `TRUSTED.sha256` and regenerating it after review:

~~~bash
sha256sum lean-toolchain lakefile.toml lake-manifest.json grade.sh Grader/Registry.lean Grader/Main.lean Pous.lean \
  $(find Pous -name '*.lean' | sort) > TRUSTED.sha256
~~~
