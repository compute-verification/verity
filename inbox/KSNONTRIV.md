# KSNONTRIV (bc-c7d6462d-41a8-5405-83b0-88ef8803b116) inbox

- 21:27Z coordinator: launched for LEAN-66's nontrivial lemma for RecursiveKS (due Tue 08:00Z). Asked lean on LEAN-66 for the witnesses entry shape.
- 21:36Z coordinator: top's reading (21:35Z, lean rules on LEAN-66): a named premise discharged by a deployed run counts for the honest lemma ONLY if the lemma names that run (its run id or art: id) so the check can be rerun. So HonestRunAccepted, IrShaped and HonestWitSatisfies must each cite the PASSING witness run id / art: id in their docstrings (and the lock witnesses entry). Today no passing witness exists yet: #3744's fixture is being restaged (PRTRIAGE, new art via INF-270) and the deployed trace difftest is being profiled (CAPLOWER).
- 22:10Z KSNONTRIV (statement, task 1). Branch `cursor/proofs-ksnontriv-27b2` (from kslock 8811d713c4).
  **Accept-all means**: the compiled recursive verifier `(D.record IV).V`, whose acceptance `PrBad` reads, replaced by the
  one that keeps its rounds `k` and answers `.acc` at its first round with no SHA-512 call. Everything else (the parser,
  `xI`, the inner view, `IVStar`, `Bd`, `DO`, `Cp`, `DenseAt`) stays deployed:
  `def RecAt.acceptAll (D : RecAt Coin) : RecAt Coin := { D with record := fun IV => { D.record IV with V := ⟨(D.record IV).V.k, fun _ _ _ => .ret .acc⟩ } }`
  (`runs` is unchanged, by `rfl`). The right-hand side is re-read at accept-all too (`PrB = 0`: no verifier calls).
  **Full statement** (`¬` of RecursiveKS's conclusion at accept-all):
  `theorem recursiveKS_nontrivial : ¬ ∀ b aux nOf hd recRest, KSLin2On sha512 (recBoundAt relation {deployedB b with tr := recTr aux nOf hd recRest} (layR elemR)).acceptAll (bdDeployed relation (nextR rdR) (deployPlainTrimR relation rdR).cols wER joinR rowInDeployed finalR wR) (deployZkRec relation) CpDep DenseAt`.
  **Finding: this cannot be proved in Lean, for accept-all or for any other verifier.** It needs some `x` with
  `DenseAt x` and right-hand side `< 1`.
  - Where `RecDeploy.readX x = none`: `parse x = none`, `nS x = 0`, `xI = []`, so the inner Flock has no tables
    (`N₀ [] = 0`, `k = 0`). Then `κ ≥ e⁰ = 1 ≥ PrBad`, and the bound holds at every verifier.
  - `readX x = some _` needs `jsonOf "rec-inner" … = .ok _`, which runs `Lean.Json.Parser.anyCore`, a `partial def`,
    which the kernel treats as opaque. So no `x` is provably parsed. In the model where that parser always errors,
    every `x` is `DenseAt` with `κ ≥ 1`, and accept-all meets RecursiveKS. This is the 26-Oct vacuity class: what makes
    the statement non-trivial sits behind an opaque parser.
  **What I'll prove instead**:
  (A) `recursiveKS_acceptAll_of_readX_none : (∀ x, RecDeploy.readX x = none) → ∀ b aux nOf hd recRest, KSLin2On … (…).acceptAll … DenseAt`.
      This is the obstruction, machine-checked.
  (B) The narrowest nontrivial form: `¬ RecursiveKS-at-accept-all` from a named premise about one fixture statement
      `x₀`. The premise: the deployed parser takes `x₀` to a dense statement whose inner claim has no witness, with
      LamDeployed's `PlainIn`/`PartIn` facts there. Per top's 21:35Z reading, it is discharged by a deployed run that
      the lemma names by `art:` id. ExpectedCR (ecr/sha-512, in the lock) is read only at accept-all's four finders,
      whose declared cost is 0 under the silent prover.
      I'm checking that (B) closes, and that a `decOK` decoder exists for `Bd.deploy` and `DO`; if none does, the ∀ is
      itself vacuous. I'll report by 23:30Z. If (B) can't close, the RecursiveKS entry goes on LEAN-66's allowlist with
      (A) as its reason. The fix is then a total JSON reader on the `rec-inner`/partition path (the code side), so that a
      fixture parses in the kernel.
