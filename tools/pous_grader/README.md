# tools/pous_grader: PoUS's grader

The grader checks a proof that someone submits for one of PoUS's statements. A submission is one Lean file that
declares `theorem solution : Pous.Guarantees.<Name>` and imports only PoUS's trusted layer, two proved modules and
Mathlib. `grade.sh` compiles it in a sandbox, replays it through Lean's kernel, and passes it only if `solution`'s type is
the named guarantee itself and its axioms are Lean's three. A graded proof then moves into `security/Proofs/Pous/`.
This package is code: the grader, its driver `grade.sh`, the trusted-file hashes and the submissions. It imports only
Lean.

## Try it

~~~bash
cd tools/pous_grader
lake build
bash grade.sh submissions/reference.lean Pous.Guarantees.KFloor
~~~

`lake build` builds the grader. `grade.sh` builds PoUS's trusted layer first, then grades the reference submission,
the grader's positive control: it prints one JSON line and exits 0 on PASS. It needs the Lean toolchain, Mathlib's
cache (`lake exe cache get` in `security/Proofs`) and unprivileged user namespaces (`unshare --user`), and its header
says each step and limit.

## Where PoUS's analysis is

PoUS's definitions, guarantees and proofs for the timed POUS audit game, whose responder gets no
outside help (a deployment accounts for help: `../../verity/protocols/resource_accounting/memory/pous/verifier/PROTOCOL.md`, "The problem"), are part of the security analysis,
`security/`, listed below by their paths there.

~~~text
Definitions/Pous.lean               trusted, the definitions: imports every module under Definitions/Pous/
Definitions/Pous/Game/{Prob,Oracle}.lean
                                    bits, uniform probability, ideal primitive, parallel oracle programs
Definitions/Pous/Model/{Scheme,Audit,Inc}.lean
                                    encoding schemes, the audit game and INC
Definitions/Assumptions/Pous.lean   TimedINC, INC against a time-bounded expander: a named assumption
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
                                    arrivals the memory challenger samples (queued, never dropped) and the memory at each time
Definitions/Pous/Accounting/{Params,LemmaA,PubMeets,GradedMeets,Rebuild}.lean
                                    ρ, δ, expansion, `Meets`; Lemma A's bound and A7′'s rows;
                                    `PubMeets`, the requirement with a public encoder; `GradedMeets`, with grace `t`;
                                    a band segment's re-encode cost `R(d)`, a graded row's round span and
                                    the memory challenger's limit on an audit's duration
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
                                    and a 320-read window at `2^-20` from `2^19` to `2^27` blocks (`ObligationWide320`),
                                    proved in `Proofs/Pous/Obligation.lean`
Proofs/Pous/Statements.lean         trusted, the lemmas' statements (math, not claims): 40 lemmas, the
                                    completeness theorem `ContinuousComplete` and `InvExPostFactoTwoLayer`, which
                                    `N10Refuted` negates; it imports no proof, and grade.sh grades `KFloor`, `Theorem1ii` and the open
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
Proofs/Pous/Rebuild/                proved: under the memory challenger's duration limit, no audit re-encodes a whole band
                                    segment from `W` alone (`Clock.lean`, the cumulative chain clock;
                                    `Window.lean`, the counts; `Exclusion.lean`, `BackgroundRebuildExcluded`)
lean-audit.json                     the policy and guarantee records for tools/verity/lean/check.py
~~~

**Trusted without proof:** the statements in the three layers `security/lean-audit.json`'s `meaning` lists,
`Definitions/Pous`, `Definitions/Pous/Assumptions` and `Definitions/Pous/Guarantees` (each directory and its own `.lean` file),
Core's `Verity.Game.prCoin` (`Definitions/Core/Game.lean`), which `Pous.pr` abbreviates, and the lemmas' statements in
`Proofs/Pous/Statements.lean` that grade.sh grades against;
Lean's kernel; the Mathlib pin; the grader and its sandbox. Everything else is checked by machine. Docstring paths such
as `lean/submissions/…` and `docs/…` name the POUS Project store, where the provers' drafts and reviews live.

## Changing the trusted layer

**Changing the trusted layer** means editing the files listed in `TRUSTED.sha256` (among them Core's `prCoin`, in
`Definitions/Core/Game.lean`, the named assumption `TimedINC`, in `Definitions/Assumptions/Pous.lean`, and the proved
files `Sanity.lean` imports) or adding a `.lean` file under
`Definitions/Pous`, regenerating it from this directory (`tools/check/pins.py --regen` runs this command, and
`--stale` fails while it would hash a file the list lacks), and running the audit (`bash check.sh`). A file the command
starts to hash outside the suite's `[tool.verity.tests] inputs` goes into them too (`tests/verity/protocols/resource_accounting/memory/pous/verifier/test_pous_trusted.py` names it):

~~~bash
sha256sum lean-toolchain lakefile.toml lake-manifest.json grade.sh Grader/Registry.lean Grader/Main.lean \
  ../../security/lean-toolchain ../../security/lakefile.toml ../../security/lake-manifest.json \
  ../../security/Proofs/lakefile.toml ../../security/Proofs/lake-manifest.json \
  ../../security/Definitions/Pous.lean ../../security/Definitions/Core/Game.lean \
  ../../security/Definitions/Assumptions/Pous.lean \
  ../../security/Proofs/Pous/Statements.lean ../../security/Proofs/Pous/Sanity.lean \
  ../../security/Proofs/Pous/Prob.lean ../../security/Proofs/Core/Game/Basic.lean \
  ../../security/Proofs/Pous/Accounting/Numbers.lean \
  $(find ../../security/Definitions/Pous ../../security/Definitions/Pous -name '*.lean' | LC_ALL=C sort -u) > TRUSTED.sha256
~~~
