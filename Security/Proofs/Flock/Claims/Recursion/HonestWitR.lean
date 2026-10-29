import Proofs.Flock.Claims.Recursion.DeployHonest

/-!
# Recursion completeness on the honest run, at its own statement

`vstar_complete_run_setup` takes V\*'s builder's facts `StagedSetsUp` and `WitSatisfies` at every statement and every
honest first part. On the honest compiled run at one statement `x` they are needed only there, and the first is a
consequence of the run accepting:

* the run's staging sets up (`stagedSetsUp_run`): an accepted run stages every part (`stagedPart_run`), so each part's
  statement staged at the run's inner and staging rounds passes the recursive setup;
* the honest witness satisfies each part's tables there: the named premise `HonestWitSatisfies`, V\*'s gate-level
  completeness at the run, witnessed outside Lean by the deployed V\* difftest.

`vstar_complete_run_setup_at` is `vstar_complete_run_setup` at the run, from those two, with no inner-acceptance
hypothesis: `InnerAccepts` was read only by the two builder facts.
-/

namespace RecDeploy

open Claims (Msg Out Prover run)
open Flock
open FlockDeploy (Coin)

/-- **An accepted run's staging sets up** (`StagedSetsUp` at the run): at an accepted run of the compiled verifier, each
part's V\* statement staged at the run's inner and staging rounds passes the recursive setup (`stagedPart_run`). -/
theorem stagedSetsUp_run (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (r : Fin ((V IV st).k x) → Coin) (hacc : (run H P (V IV st) x ω r).out = .acc) :
    ∀ s < nS x, (setupO (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)).isSome := by
  intro s hs
  obtain ⟨_, _, _, _, -, hsu, -⟩ := stagedPart_run H IV st x P ω r hacc hs
  simp [hsu]

/-- **The honest witness satisfies each part's tables on the honest run** (V\*'s gate-level completeness at the run;
witnessed outside Lean). On the reference prover's run of the compiled verifier at `x`, on tape `ω` and coins `r`, at
each part `s < nS x` whose statement staged at the run's inner and staging rounds sets up to tables `sts`, the outer
witness the reference prover hands part `s`'s session (`RefParts.wit` at the run's inner rounds) satisfies every table
of `sts` (`TablesSat`: each fits the first table's blocks, and its share of the witness's hidden values satisfies its
model statement). Not proved in Lean: it is the gate-level completeness of V\*'s part circuits, which compute the inner
verifier's checks. It is witnessed outside Lean by a satisfaction check of the honest witness against each part's
circuit in the deployed V\* difftest. Witness run: none passing yet; its run id or `art:` id goes here,
and in the lock's witnesses entry, when it passes. -/
def HonestWitSatisfies (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (x : Msg) (w : Claims.Flock.Wit)
    (ω : Tape b) (r : Fin ((V IV b.st).k x) → Coin) : Prop :=
  ∀ s < nS x, ∀ i spec sts,
    setupO (staged b.st x ((run H (refP IV b w) (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) =
        some (i, spec, sts) →
      TablesSat sts (b.ref.wit w ω.1 (((run H (refP IV b w) (V IV b.st) x ω r).msgs.take (IV.k (xIOf x))).map histOf) s)

/-- **Recursion completeness on the honest run, at its own statement**: `vstar_complete_run` with `StagedSetsUp` from
the run accepting (`stagedSetsUp_run`) and `WitSatisfies` at the run (`HonestWitSatisfies`). -/
theorem vstar_complete_run_at (H : Msg → Msg) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
    (ivScope : Msg → Bool) (b : Brings) (linkOk : RecIns → ByteArray → ByteArray → Array Stmt → Bool)
    (hN : NumsFixed b.st) (hL : LinksHonest linkOk H I.V b) {x : Msg} {w : Claims.Flock.Wit}
    (hR : Claims.Recursion.R (deployRec I.V ivScope b) I x w) (ω : Tape b) (r : Fin ((V I.V b.st).k x) → Coin)
    (hacc : (run H (refP I.V b w) (V I.V b.st) x ω r).out = .acc) (hW : HonestWitSatisfies H I.V b x w ω r)
    {s : ℕ} (hs : s < nS x) :
    ∃ i spec sts, setupO ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω
        r).getD s []) = some (i, spec, sts) ∧
      Claims.Flock.R (deployZkRec Readable.relation).parse
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω r).getD s [])
        (b.ref.wit w ω.1 (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))).map histOf) s) ∧
      linkOk i (b.st.transcript x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)))
        (b.st.record x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))) sts = true := by
  obtain ⟨s', hp, -, -⟩ := hR
  obtain ⟨xr, hx⟩ := readX_some_of_parse hp
  have hv := honestPre_run H I.V b w hx ω r
  have hA := run_take_asRounds H I.V b hx (refP I.V b w) ω r
  have hsu := stagedSetsUp_run H I.V b.st x (refP I.V b w) ω r hacc s hs
  obtain ⟨⟨i, spec, sts⟩, he⟩ := Option.isSome_iff_exists.1 hsu
  have hst : (vstar b.st x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] =
      staged b.st x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
    rw [vstar_getD _ _ _ hs]
    exact vstarAt_staged hN hsu
  have hrv : Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω r =
      vstar b.st x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) := rfl
  have hl := hL x w ω _ _ hv s hs i spec sts (by rw [hA]; exact he)
  rw [← transcript_honest b x hv.length, hA] at hl
  refine ⟨i, spec, sts, ?_, ?_, hl⟩
  · rw [hrv, hst]
    exact he
  · rw [hrv, hst]
    exact R_of_setupO he (hW s hs i spec sts he)

/-- **Recursion completeness on the honest run, at its own statement, with the deployed link checks**:
`vstar_complete_run_at` at `linkOkSetup`, whose `LinksHonest` is `linksHonest_setup`. -/
theorem vstar_complete_run_setup_at (H : Msg → Msg) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
    (ivScope : Msg → Bool) (b : Brings) (hN : NumsFixed b.st) {x : Msg} {w : Claims.Flock.Wit}
    (hR : Claims.Recursion.R (deployRec I.V ivScope b) I x w) (ω : Tape b) (r : Fin ((V I.V b.st).k x) → Coin)
    (hacc : (run H (refP I.V b w) (V I.V b.st) x ω r).out = .acc) (hW : HonestWitSatisfies H I.V b x w ω r)
    {s : ℕ} (hs : s < nS x) :
    ∃ i spec sts, setupO ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω
        r).getD s []) = some (i, spec, sts) ∧
      Claims.Flock.R (deployZkRec Readable.relation).parse
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω r).getD s [])
        (b.ref.wit w ω.1 (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))).map histOf) s) ∧
      linkOkSetup i (b.st.transcript x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)))
        (b.st.record x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))) sts = true :=
  vstar_complete_run_at H I ivScope b linkOkSetup hN (linksHonest_setup H I.V b) hR ω r hacc hW hs

end RecDeploy
