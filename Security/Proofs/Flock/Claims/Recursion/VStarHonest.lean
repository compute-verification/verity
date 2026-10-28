import Proofs.Flock.Claims.Recursion.VStarHh
import Proofs.Flock.Claims.Recursion.VStarPorts
import Proofs.Flock.Claims.Recursion.VStarPin
import Proofs.Flock.Claims.Recursion.VStarWit

/-!
# What each V\* part's circuit accepts on the honest run, at the deployed builder (`honest_partAccepts_deployed`)

`honest_partAccepts_layR` at `layR elemR` and the deployed brings at `x` (`bringsAt`): the reference prover's pieces are
`refPartsFin` at the deployed assigner `A0` and the fitted records `InnerRec.fit x`, so the inner prover sends hm96's
commit strings at the rounds before the last and `finMsg` at the last (`PFin`), and part `s`'s outer witness is
`witFin`. `bringsAt` changes only `ref`, so a brings that also registers (`withReg`, which changes only `setup`) is
covered by taking `b` to be it.

Of `honest_partAccepts_layR`'s premises about the witness holding the run's messages:

* `hr` is `rowsRead_layR`: `layR` shares no rows;
* `hp` at the last round (`hp_last_run`) is `hpLast` at `finMsg`, where the run's last inner message is the inner
  prover's (`hA`, the run lemma) and the run reaches it (`run_length_ge`);
* `hp` at the rounds before the last (`hpE`) and `hh` are this theorem's hypotheses.
-/

namespace RecDeploy.VStarHonest

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms)
open RecDeploy.VStarBuild

/-- **The deployed brings at `x`**: the reference prover's pieces at the deployed assigner and the fitted records. -/
def bringsAt (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b : Brings) (x : Msg) : Brings :=
  { b with ref := refPartsFin (A0 mask) (Ir.fit x) x }

/-- **hp at the last round**: where the run's last inner message is the inner prover's at the history before it, and
the history reaches it, every piece part `s` holds of it at `witFin` is that message's bytes at its offset. -/
theorem hp_last_run {A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit} {I : InnerRec} {x : Msg}
    {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {ms : List Msg} {s : ℕ} (hs : s < nS x)
    (hlen : RestRead.kI x - 1 ≤ hist.length) (hk : 0 < RestRead.kI x)
    (hA : ms.getD (RestRead.kI x - 1) [] =
      (((PFin A I x w).next (xIOf x) (hist.take (RestRead.kI x - 1)) ρ).eval FlockDeploy.sha512).1) :
    ∀ o m, (o, m) ∈ (DeployedR.layR OpenRows.elemR).piecesOf x s (witFin A I x w ρ hist s) (RestRead.kI x - 1) →
      ((ms.getD (RestRead.kI x - 1) []).drop o).take m.length = m := by
  have hl : (hist.take (RestRead.kI x - 1)).length + 1 = RestRead.kI x := by
    rw [List.length_take]
    omega
  apply hpLast_fin hs
  rw [hA, PFin_last A I x w (xIOf x) (hist.take (RestRead.kI x - 1)) ρ hl]
  rfl

theorem readX_of_lt_nS {x : Msg} {s : ℕ} (hs : s < nS x) : ∃ xr, readX x = some xr := by
  cases h : readX x with
  | none => simp [nS, h] at hs
  | some xr => exact ⟨xr, rfl⟩

/-- **Part `s`'s circuit accepts on the honest run, at the deployed builder**: `honest_partAccepts_layR` at
`layR elemR` and `bringsAt mask Ir b0 x`, with `hr` and the last round's `hp` proved. It takes, beside
`honest_partAccepts_layR`'s premises about the builder, the statement and the run (`hN` … `hacc`, `hW`, `hTm`, `hnc`):

* `hA`, the run lemma: each inner round's message is the inner prover's at the run's earlier inner rounds;
* `hpE`, `hp` at the rounds before the last;
* `hh`, the held rounds. -/
theorem honest_partAccepts_deployed (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    let b := bringsAt mask Ir b0 x
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR OpenRows.elemR
    let bL : Brings := { bR with rd := L.readers }
    let I := Claims.Recursion.Inner.ofFlock DeployedR.bdR.deploy decI
    let D := recBound _root_.Flock.Readable.relation bR L DeployedR.bdR.IVStar
    NumsFixed bL.st →
    ∀ {w : Claims.Flock.Wit} {st : Claims.Recursion.CompiledStatement}, Claims.Recursion.R D I x w →
      D.parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (ω : Tape bL) (r : Fin (D.V.k x) → Coin),
      (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).out = .acc →
      HonestWitSatisfies FlockDeploy.sha512 I.V bL x w ω r →
    ∀ s < nS x,
      let hist := ((Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.take (I.V.k (xIOf x))).map histOf
      let wO := bL.ref.wit w ω.1 hist s
      let ms := (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.map Prod.fst
      (∀ xr, readX x = some xr → ∀ p ∈ DeployedR.moreCalls xr s,
        FlockDeploy.sha512 (p.1 ++ readForms wO p.2.1) = readForms wO p.2.2) →
      (Claims.findColl (L.calls x s wO ++ Claims.Recursion.VStarB.vcallsAt FlockDeploy.sha512 D
        ((Claims.Recursion.runVstar FlockDeploy.sha512 D I (refP I.V bL w) x ω r).getD s []))).isSome = false →
      (∀ i < I.V.k (xIOf x), ms.getD i [] =
        (((bL.ref.PI w).next (xIOf x) (hist.take i) ω.1).eval FlockDeploy.sha512).1) →
      (∀ i < RestRead.kI x - 1, ∀ o m, (o, m) ∈ L.piecesOf x s wO i → ((ms.getD i []).drop o).take m.length = m) →
      (∀ i < L.rounds x, ∀ m, L.held x s wO i = some m → ms.getD i [] = m) →
      Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
        Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s x (Claims.coinSeq r) ms := by
  intro b bR L bL I D hN w st hR hx hf ω r hacc hW s hs hist wO ms hTm hnc hA hpE hh
  refine honest_partAccepts_layR OpenRows.elemR b aux nOf hd recRest decI hN hR hx hf ω r hacc hW s hs
    hTm hnc hh ?_ (rowsRead_layR _ _ _ _ _)
  intro i hi
  have hiK : i < RestRead.kI x := hi
  by_cases hlt : i < RestRead.kI x - 1
  · exact hpE i hlt
  · have he : i = RestRead.kI x - 1 := by omega
    subst he
    obtain ⟨xr, hxr⟩ := readX_of_lt_nS hs
    have hrun : I.V.k (xIOf x) + kS x ≤
        (Claims.run FlockDeploy.sha512 (refP I.V bL w) D.V x ω r).msgs.length :=
      run_length_ge FlockDeploy.sha512 I.V bL.st hxr (refP I.V bL w) ω r
    have hk : I.V.k (xIOf x) = RestRead.kI x := rfl
    have hlen : RestRead.kI x - 1 ≤ hist.length := by
      simp only [hist, List.length_map, List.length_take]
      omega
    exact hp_last_run (A := A0 mask) (I := Ir.fit x) hs hlen (by omega) (hA _ (by omega))

end RecDeploy.VStarHonest
