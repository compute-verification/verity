# POUS Lean: the grader, and where its security analysis is

This package (`pous_grader`) is code: the grader, its driver `grade.sh`, the trusted-file hashes and the submissions. It
imports only Lean. PoUS's definitions, guarantees and proofs for the timed POUS audit game, whose responder gets no
outside help (a deployment accounts for help: `../PROTOCOL.md`, "The problem"), are part of the security analysis,
`Security/`, listed below by their paths there. Declarations kept their names
(`python3 tools/move/lean.py --resolve MODULE` maps a module's old name). Conventions follow the other Lean packages:
the same toolchain and Mathlib pin, `autoImplicit` off, no `axiom` (an assumption is a named `Prop` hypothesis), and a
check file.

~~~text
Definitions/Pous.lean               trusted, the definitions: imports every module under Definitions/Pous/
Definitions/Pous/Game/{Prob,Oracle}.lean
                                    bits, uniform probability, ideal primitive, parallel oracle programs
Definitions/Pous/Model/{Scheme,Audit,Inc}.lean
                                    encoding schemes, the audit game, INC and TimedINC
Definitions/Pous/Model/{M1,M1p,HashChain,Pebbling,Labelling,Invertible,ColumnGame,Dense,SampledAudit}.lean
                                    model M1, M1 on `[0, p)` (`m1p`, the abstract model at P2's domain),
                                    B3's chain, Fisch's stacked DRGs and white/green game,
                                    Pietrzak's one-way labels, P3's invertible labels, the two-directional column
                                    game, the dense scheme S1 (one segment, segmented, and tagged as the code
                                    keys it), the state after a history
Definitions/Pous/Model/{P3Chain,BandChain,PubEncoder}.lean
                                    the overwrite chain on an ideal permutation, P3 with it as `H`, the
                                    band and the dense chain scheme over it, the public-encoder setup check
Definitions/Pous/Model/{BandGraph,SegLift}.lean
                                    the band graph the column game's band certificates are stated over; a
                                    segment's parents at their global positions, which the tagged transfer reads
Definitions/Pous/Model/Graded.lean  the graded audit: up to `t` late rounds, each within a hard `(D_late, Q_late)`
Definitions/Pous/Model/Continuous.lean
                                    the continuous audit: an explicit memory challenger (sequential reveal, a hard
                                    deadline, every answer past `Δ` counted) and a responder on its clock; the
                                    arrivals the memory challenger draws (queued, never dropped) and the memory at each time
Definitions/Pous/Model/{Digest,SecureErasure}.lean
                                    the keyed digest of a region (from PR #1010, without its bandwidth
                                    assumption); the secure-erasure audit: a keyed fill of all volatile memory,
                                    timed spot reads, and the foreign content the prover keeps (4 Oct model)
Definitions/Pous/Accounting/{Params,LemmaA,PubMeets,GradedMeets,Rebuild,EraseMeets}.lean
                                    ρ, δ, expansion, `Meets`; Lemma A's bound and A7′'s rows;
                                    `PubMeets`, the requirement with a public encoder; `GradedMeets`, with grace `t`;
                                    a band segment's re-encode cost `R(d)`, a graded row's round span and
                                    the memory challenger's cap on an audit's duration; `EraseMeets`,
                                    the secure-erasure requirement
Definitions/Pous/P2/Cost.lean       P2's cost model, the generic ring over `Z/p` with random odd labels
                                    and the junk label (`ringRO`), the bound `RingBounded D Q L` beside `Bounded`,
                                    `MeetsOn`, `MeetsWithSlack`, and P2 in it (`p2ring`)
Definitions/Pous/P2/Mutable.lean    the mutable store (PoUS as an obligation over the weights): one key pair per
                                    tag, a window's snapshot (`snap`) and the other tags' keys (`rest`)
Definitions/Pous/Assumptions.lean         trusted, the assumptions: named hypotheses (Lemma A, Conjecture B1′)
Definitions/Pous/Assumptions/P2.lean      P2's named open step `PrimeP16448`, the primality of its modulus
Definitions/Pous/Guarantees.lean          trusted, the frozen statements: 31, each with its category, the 28
                                    grade.sh grades against and three proved results cited outside the package
Definitions/Pous/Guarantees/P2.lean       seven frozen statements about the abstract model `m1p` at P2's parameters,
                                    six of which grade.sh grades against; none is about the concrete P2
Definitions/Pous/Guarantees/Obligation.lean
                                    the mutable store's audit window: tag freshness (`SnapshotFresh`), a block's
                                    offset absorbing its content (`P2ContentAbsorbs`), the window's bound from
                                    P2's static one at every block count (`ObligationOfStatic`), from
                                    `P2RingRoundSecure` (`ObligationOfRoundSecure`) and from its widening to `2^27`
                                    blocks, `P2RingRoundSecureWide` (`ObligationOfRoundSecureWide`), and the demo's
                                    memory term at `1038·2^16` blocks (`ObligationDemo`, `ObligationDemoWide`),
                                    and a 320-read window at `2^-20` from `2^19` to `2^27` blocks
                                    (`ObligationWide320`), proved in `Proofs/Pous/Obligation.lean`
Definitions/Pous/Guarantees/SecureErasure.lean
                                    two frozen statements, which grade.sh grades against: secure erasure on the
                                    band at every `N` and at 96 GiB (`SecureErasureMeetsFamily`,
                                    `SecureErasureMeets96GiB`); no named assumption, no bandwidth-separation premise;
                                    in the ideal-permutation model only, not a bound on a band-fill wipe at the
                                    reference `Π₂` (art:7617c8017e80)
Proofs/Pous/Statements.lean         trusted, the lemmas' statements (math, not claims): 42 lemmas, the
                                    completeness theorem `ContinuousComplete` and `InvExPostFactoTwoLayer`, which
                                    `N10Refuted` negates, among them `DigestAU` and `SecureErasureMeets`; it
                                    imports no proof, and grade.sh grades `KFloor`, `Theorem1ii` and the open
                                    `A2LemmaAM1`, `A5RabinGcd` and `A7Params` against it
Proofs/Pous.lean                    proved, the proofs: imports every proof module, and proves
                                    Pous.SecurityProofs.X : Pous.Guarantees.X for each proved guarantee
Proofs/Pous/{Sanity,Accounting/Numbers}.lean
                                    proved: the definitions are not vacuous; certified numbers
Proofs/Pous/                        proved: Theorem 1, Track A, pebbling, B5, sampled audits, the dense scheme, P3
Proofs/Pous/P2/                     proved: the `m1p` guarantees (Count, Rows, Domain, Margin); P2 in the cost model
                                    (RingBasic: `MeetsOn` at `L = Q` gives `Meets`, honest storage is accepted;
                                    RingTight: recomputing the roots breaks `MeetsOn`; RingTrunc: dropping low
                                    bits breaks `k = 88` from `L = 2^30`)
Proofs/Pous/Chain/Flat.lean         proved, generated: P3 with a concrete `H`, the band, the dense chain scheme and
                                    the public encoder over the overwrite chain (the graded lanes, flattened, then
                                    pruned to what their importers use and the docs cite)
Proofs/Pous/Continuous/             proved: the continuous audit's counting step (`MissCount.lean`), the three
                                    rules its memory challenger enforces (`Verifier.lean`), `PerAnswer` and
                                    `ContinuousAudit` (`PerAnswer.lean`), the 64 KiB band at `D = 5`
                                    (`Band.lean`, no record), completeness, `ContinuousComplete`
                                    (`Complete.lean`), the idle-slot time average `ContinuousTimeAverage`
                                    (`TimeAverage.lean`) and its band family at `D = 0, 1, 2, 5`
                                    (`BandTimeAverage.lean`, `BandFamilyContinuousTimeAverage`)
Proofs/Pous/Graded/                 proved: the tail rule (`Tail.lean`), Theorem 1(ii) for the graded audit
                                    (`Theorem.lean`) and the band with grace `t` at `D = 5` (`Band.lean`,
                                    `BandGradedMeetsFamily`)
Proofs/Pous/Rebuild/                proved: under the memory challenger's duration cap, no audit re-encodes a whole band
                                    segment from `W` alone (`Clock.lean`, the cumulative chain clock;
                                    `Window.lean`, the counts; `Exclusion.lean`, `BackgroundRebuildExcluded`)
Proofs/Pous/SecureErasure/          proved: `DigestAU` (Digest), secure erasure from the audit by counting
                                    (Reduction), and the band's audit at 2152 reads with `EraseMeets` at
                                    `f = 1/19`, `ε = 2^-128` (Band)
lean-audit.json                     the policy and guarantee records for tools/verity/lean/check.py
~~~

In this package:

~~~text
lean-audit.json                     this package's audit policy for tools/verity/lean/check.py
Grader/{Main,Registry}.lean, grade.sh, TRUSTED.sha256   the grader (allowed axioms and pinned targets compiled in),
                                    its driver, the trusted-file hashes
submissions/reference.lean          the grader's positive control
~~~

~~~bash
lake build                          # the grader (the default target)
(cd ../../../../../Security/Proofs && lake exe cache get && lake build Proofs.Pous)   # PoUS's analysis; Mathlib prebuilt
python3 ../../../../../tools/verity/lean/check.py --build --fresh . ../../../../../Security   # Verity's Lean audit
bash grade.sh submissions/reference.lean Pous.Guarantees.KFloor   # one JSON line, exit 0 iff PASS
bash check.sh                       # the audit and the grader's controls
~~~

**Trusted without proof:** the statements in the three layers `Security/lean-audit.json`'s `meaning` lists,
`Definitions/Pous`, `Definitions/Pous/Assumptions` and `Definitions/Pous/Guarantees` (each directory and its own `.lean` file),
and the lemmas' statements in `Proofs/Pous/Statements.lean` that grade.sh grades against;
Lean's kernel; the Mathlib pin; the grader and its sandbox. Everything else is checked by machine. Docstring paths such
as `lean/submissions/…` and `docs/…` name the POUS Project store, where the provers' drafts and reviews live.

**The audit.** `tools/verity/lean/check.py` audits every declaration of PoUS's modules: allowed axioms, named assumptions,
escapes, orphan files, import layers (Definitions imports no assumption, guarantee or proof; Assumptions only
Definitions; Guarantees no proof), a kernel replay, and the guarantees' records. `Security/lean-audit.json` lists
as guarantees the sanity theorems, every `Pous.SecurityProofs.X` (among them the stacked one-way bound, the band
certificate, and the dense scheme at the 64 KB point, with the implementation's tagged keying and its per-segment tags),
the raw-storage refutations the `m1p` guarantees cite, and P2 in the cost model (`PousP2Ring`: `meets_of_meetsOn`, the
non-vacuity theorems `complete`, `tight` and `tight_p16448`, and the refutation `not_meetsOn_b19_k88`), and the
continuous audit's counting step (`PousContinuous.pr_misses_le`, `pr_misses_gt_le`). When a changed record lands on
`main`, a DM shows it (Verity's `AGENTS.md`).

**Submissions** are one file that imports only the trusted layers (`Definitions.Pous`, `Definitions.Pous.Assumptions`,
`Definitions.Pous.Guarantees`, `Proofs.Pous.Statements`), the proved `Proofs.Pous.Sanity` and `Proofs.Pous.Accounting.Numbers`, and `Mathlib.*`, and declares `theorem solution : Pous.Guarantees.<Name>`. `grade.sh` builds the trusted layer and the
grader first and records their hashes; compiles the submission with `lean` in a scratch directory inside a user, mount,
PID and network namespace where every other mount is read-only (so compile-time IO can change nothing, and its processes
die with the build); checks the hashes again; then runs the grader. The grader loads the submission's `.olean` as data,
replays every constant it adds through the kernel, requires `solution`'s type to be the pinned constant itself (a
syntactic comparison, so a near-miss statement cannot keep the kernel unfolding), and requires its axioms to be in
`allowedAxioms` (Lean's three, compiled in). The compile and the grader run under a time limit and a memory limit
(`POUS_GRADE_TIMEOUT`, `POUS_GRADE_MEMORY_MB`). `grade.sh` needs unprivileged user namespaces (`unshare --user`). A
graded proof moves into `Security/Proofs/Pous/` as a module, with its theorem added to
`Security/Proofs/Pous.lean` and listed as a guarantee in `Security/lean-audit.json` (`check.py --update`,
then review the records' diff).

**Changing the trusted layer** means editing the files listed in `TRUSTED.sha256`, regenerating it from this directory,
and running the audit (`bash check.sh`):

~~~bash
sha256sum lean-toolchain lakefile.toml lake-manifest.json grade.sh Grader/Registry.lean Grader/Main.lean \
  ../../../../../Security/lean-toolchain ../../../../../Security/lakefile.toml ../../../../../Security/lake-manifest.json \
  ../../../../../Security/Proofs/lakefile.toml ../../../../../Security/Proofs/lake-manifest.json \
  ../../../../../Security/Definitions/Pous.lean \
  ../../../../../Security/Proofs/Pous/Statements.lean ../../../../../Security/Proofs/Pous/Sanity.lean \
  ../../../../../Security/Proofs/Pous/Accounting/Numbers.lean \
  $(find ../../../../../Security/Definitions/Pous ../../../../../Security/Definitions/Pous -name '*.lean' | LC_ALL=C sort -u) > TRUSTED.sha256
~~~
