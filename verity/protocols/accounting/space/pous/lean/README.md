# POUS Lean package

Definitions, guarantees and proofs for the timed POUS audit game with perfect isolation (`../PROTOCOL.md`).
Conventions follow Flock's Lean packages (`backends/flock/verifier/lean/`): the same toolchain and Mathlib pin,
`autoImplicit` off, no `axiom` (an assumption is a named `Prop` hypothesis), and a check file.

~~~text
Pous.lean                           the package's root: imports the four layers below
Pous/Protocol.lean                  trusted, the Protocol layer: imports every module under Pous/Protocol/
Pous/Protocol/Game/{Prob,Oracle}.lean
                                    bits, uniform probability, ideal primitive, parallel oracle programs
Pous/Protocol/Model/{Scheme,Audit,Inc}.lean
                                    encoding schemes, the audit game, INC and TimedINC
Pous/Protocol/Model/{M1,M1p,HashChain,Pebbling,Labelling,Invertible,ColumnGame,Dense,SampledAudit}.lean
                                    model M1, M1 on `[0, p)` (`m1p`, the abstract model at P2's domain),
                                    B3's chain, Fisch's stacked DRGs and white/green game,
                                    Pietrzak's one-way labels, P3's invertible labels, the two-directional column
                                    game, the dense scheme S1 (one segment, segmented, and tagged as the code
                                    keys it), the state after a history
Pous/Protocol/Model/{P3Chain,BandChain,PubEncoder}.lean
                                    the overwrite chain on an ideal permutation, P3 with it as `H`, the
                                    band and the dense chain scheme over it, the public-encoder setup check
Pous/Protocol/Model/{BandGraph,SegLift}.lean
                                    the band graph the column game's band certificates are stated over; a
                                    segment's parents at their global positions, which the tagged transfer reads
Pous/Protocol/Model/Graded.lean     the graded audit: up to `t` late rounds, each within a hard `(D_late, Q_late)`
Pous/Protocol/Model/Continuous.lean the continuous audit: an explicit timed verifier (sequential reveal, a hard
                                    deadline, every answer past `Δ` counted) and a responder on its clock; the
                                    arrivals the verifier draws (queued, never dropped) and the memory at each time
Pous/Protocol/Model/{Digest,SecureErasure}.lean
                                    the keyed digest of a region (from PR #1010, without its bandwidth
                                    assumption); the secure-erasure audit: a keyed fill of all volatile memory,
                                    timed spot reads, and the foreign content the prover keeps (4 Oct model)
Pous/Protocol/Accounting/{Params,LemmaA,PubMeets,GradedMeets,Rebuild,EraseMeets}.lean
                                    ρ, δ, expansion, `Meets`; Lemma A's bound and A7′'s rows;
                                    `PubMeets`, the requirement with a public encoder; `GradedMeets`, with grace `t`;
                                    a band segment's re-encode cost `R(d)`, a graded row's round span and
                                    the verifier's cap on an audit's duration; `EraseMeets`,
                                    the secure-erasure requirement
Pous/Protocol/P2/Cost.lean          P2's cost model, the generic ring over `Z/p` with random odd labels
                                    and the junk label (`ringRO`), the bound `RingBounded D Q L` beside `Bounded`,
                                    `MeetsOn`, `MeetsWithSlack`, and P2 in it (`p2ring`)
Pous/Assumptions.lean               trusted, the Assumptions layer: named hypotheses (Lemma A, Conjecture B1′)
Pous/Assumptions/P2.lean            P2's named open step `PrimeP16448`, the primality of its modulus
Pous/Guarantees.lean                trusted, the Guarantees layer: 73 statements, the 62 grade.sh grades against,
                                    `InvExPostFactoTwoLayer`, which `N10Refuted` negates, and ten proved results
                                    cited outside the package
Pous/Guarantees/P2.lean             six statements about the abstract model `m1p` at P2's parameters,
                                    which grade.sh also grades against; none is about the concrete P2
Pous/Guarantees/SecureErasure.lean  four statements, which grade.sh also grades against: the digest is almost
                                    universal (`DigestAU`), secure erasure from any sequential audit bound
                                    (`SecureErasureMeets`), and on the band at every `N` and at 96 GiB
                                    (`SecureErasureMeetsFamily`, `SecureErasureMeets96GiB`); no named assumption,
                                    no bandwidth-separation premise
Pous/SecurityProofs.lean            proved, the SecurityProofs layer: imports every proof module, and proves
                                    Pous.SecurityProofs.X : Pous.Guarantees.X for each proved guarantee
Pous/SecurityProofs/{Sanity,Accounting/Numbers}.lean
                                    proved: the definitions are not vacuous; certified numbers
Pous/SecurityProofs/                proved: Theorem 1, Track A, pebbling, B5, sampled audits, the dense scheme, P3
Pous/SecurityProofs/P2/             proved: the `m1p` guarantees (Count, Rows, Domain, Margin); P2 in the cost model
                                    (RingBasic: `MeetsOn` at `L = Q` gives `Meets`, honest storage is accepted;
                                    RingTight: recomputing the roots breaks `MeetsOn`; RingTrunc: dropping low
                                    bits breaks `k = 88` from `L = 2^30`)
Pous/SecurityProofs/Chain/Flat.lean proved, generated: P3 with a concrete `H`, the band, the dense chain scheme and
                                    the public encoder over the overwrite chain (the graded lanes, flattened)
Pous/SecurityProofs/Continuous/     proved: the continuous audit's counting step (`MissCount.lean`), the three
                                    rules its verifier enforces (`Verifier.lean`), `PerAnswer` and
                                    `ContinuousAudit` (`PerAnswer.lean`), the 64 KiB band at `D = 5`
                                    (`Band.lean`, no record), completeness, `ContinuousComplete`
                                    (`Complete.lean`), the idle-slot time average `ContinuousTimeAverage`
                                    (`TimeAverage.lean`) and its band family at `D = 0, 1, 2, 5`
                                    (`BandTimeAverage.lean`, `BandFamilyContinuousTimeAverage`)
Pous/SecurityProofs/Graded/         proved: the tail rule (`Tail.lean`), Theorem 1(ii) for the graded audit
                                    (`Theorem.lean`) and the band with grace `t` at `D = 5` (`Band.lean`,
                                    `BandGradedMeetsFamily`)
Pous/SecurityProofs/Rebuild/        proved: under the verifier's duration cap, no audit re-encodes a whole band
                                    segment from `W` alone (`Clock.lean`, the cumulative chain clock;
                                    `Window.lean`, the counts; `Exclusion.lean`, `BackgroundRebuildExcluded`)
Pous/SecurityProofs/SecureErasure/  proved: `DigestAU` (Digest), secure erasure from the audit by counting
                                    (Reduction), and the band's audit at 2152 reads with `EraseMeets` at
                                    `f = 1/19`, `ε = 2^-128` (Band)
lean-audit.json                     the policy and guarantee records for tools/lean/audit.py
CheckAxioms.lean                    one #print axioms line per headline theorem
PousTargets.lean                    untrusted, not in the default build: the open statements as sorry targets
Grader/{Main,Registry}.lean, grade.sh, TRUSTED.sha256   the grader (allowed axioms and pinned targets compiled in),
                                    its driver, the trusted-file hashes
submissions/reference.lean          the grader's positive control
~~~

~~~bash
lake exe cache get                  # Mathlib's prebuilt oleans; never build Mathlib from source
lake build                          # Pous, the four layers (the default target)
python3 ../../../../../../tools/lean/audit.py --all --root . --fresh   # Verity's Lean audit
lake env lean CheckAxioms.lean      # the headline theorems' axioms
bash grade.sh submissions/reference.lean Pous.Guarantees.KFloor   # one JSON line, exit 0 iff PASS
bash check.sh                       # the audit, the open targets, and the grader's controls
~~~

**Trusted without proof:** the statements in the three layers `lean-audit.json`'s `meaning` lists, `Pous/Protocol`,
`Pous/Assumptions` and `Pous/Guarantees` (each directory and its own `.lean` file); Lean's kernel; the Mathlib pin; the
grader and its sandbox. Everything else is checked by machine. Docstring paths such as `lean/submissions/…` and
`docs/…` name the POUS Project store, where the provers' drafts and reviews live.

**The audit.** `tools/lean/audit.py` audits every declaration under `Pous`: allowed axioms, named assumptions, escapes,
orphan files, import layers (Protocol imports no assumption, guarantee or proof; Assumptions only Protocol; Guarantees
no proof), a kernel replay, and the guarantees' records. `lean-audit.json` lists as guarantees the sanity theorems,
every `Pous.SecurityProofs.X` (among them the stacked one-way bound, the band certificate, and the dense scheme at the
64 KB point, with the implementation's tagged keying and its per-segment tags), the raw-storage refutations the `m1p`
guarantees cite, and P2 in the cost model (`PousP2Ring`: `meets_of_meetsOn`, the non-vacuity theorems `complete`,
`tight` and `tight_p16448`, and the refutation `not_meetsOn_b19_k88`), and the continuous audit's counting step
(`PousContinuous.pr_misses_le`, `pr_misses_gt_le`). When a changed record lands on `main`, Daniel gets a DM showing it
(Verity's `AGENTS.md`).

**Submissions** are one file that imports only the trusted layers (`Pous.Protocol`, `Pous.Assumptions`,
`Pous.Guarantees`), the proved `Pous.SecurityProofs.Sanity` and `Pous.SecurityProofs.Accounting.Numbers`, and
`Mathlib.*`, and declares `theorem solution : Pous.Guarantees.<Name>`. `grade.sh` builds the trusted layer and the
grader first and records their hashes; compiles the submission with `lean` in a scratch directory inside a user, mount,
PID and network namespace where every other mount is read-only (so compile-time IO can change nothing, and its processes
die with the build); checks the hashes again; then runs the grader. The grader loads the submission's `.olean` as data,
replays every constant it adds through the kernel, requires `solution`'s type to be the pinned constant itself (a
syntactic comparison, so a near-miss statement cannot keep the kernel unfolding), and requires its axioms to be in
`allowedAxioms` (Lean's three, compiled in). The compile and the grader run under a time limit and a memory limit
(`POUS_GRADE_TIMEOUT`, `POUS_GRADE_MEMORY_MB`). `grade.sh` needs unprivileged user namespaces (`unshare --user`). A
graded proof moves into `Pous/SecurityProofs/` as a module, with its theorem added to `Pous/SecurityProofs.lean` and
listed as a guarantee in `lean-audit.json` (`audit.py --update`, then review the records' diff).

**Changing the trusted layer** means editing the files listed in `TRUSTED.sha256` and regenerating it after review:

~~~bash
sha256sum lean-toolchain lakefile.toml lake-manifest.json grade.sh Grader/Registry.lean Grader/Main.lean \
  Pous/Protocol.lean Pous/Assumptions.lean Pous/Guarantees.lean Pous/SecurityProofs/Sanity.lean \
  Pous/SecurityProofs/Accounting/Numbers.lean \
  $(find Pous/Protocol Pous/Assumptions Pous/Guarantees -name '*.lean' | LC_ALL=C sort) > TRUSTED.sha256
~~~
