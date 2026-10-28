import Proofs.Flock.Claims.Recursion.PartAccepts
import Proofs.Flock.Claims.Recursion.VStarBDeploy

/-!
# What each V\* part's circuit accepts, on an accepted run (`PartAccepts`, with the vcalls escape)

`RecursiveKS` at the deployed layout reads `DecompB` at `PartAccepts`, which is non-vacuous only if `PartAccepts`
holds where an honest prover runs. On an accepted run, the staging (`StagedB`, at `deployRec`
`stagedB_deployRec_part`) gives each part's V\* statement showing the run's coins and its commit strings at every
inner round before the last. So `PartAccepts` at messages `ms` that are the run's at those rounds is what the part's
outer witness `w` brings at the V\* statement the run stages (`partAccepts_of_run`, `partAccepts_run_deployRec`):

* `Flock.check` passes;
* its recorded calls are true evaluations;
* none of them collides with another or with the compiled verifier's own evaluations there (`vcallsAt`, RECVCALLS'
  escape), a decidable equation about the run;
* it agrees with `ms` at the rounds it holds whole and at its pieces, and on the shared rows it reads (`RowsRead`).

The commit strings (`LayoutB.CmtAgrees`) and the coins (`shows`) come from the staging, not from the witness.
-/

namespace Claims.Recursion.VStarB

open Claims

section Honest

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Part `s`'s circuit accepts on an accepted run**: at a compiled statement of the shape and an accepted run, where
the staging shows the run (`StagedB`), `PartAccepts` holds at messages `ms` that are the run's at each inner round
before the last, wherever the part's outer witness `w` passes `Flock.check` at the V\* statement the run stages, its
recorded calls are true and collide neither with each other nor with the compiled verifier's evaluations there, and it
agrees with `ms` where it holds rounds, pieces and shared rows. -/
theorem partAccepts_of_run {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : LayoutB} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    (hSt : StagedB H D I L shape shows cmt) {x : Msg} {st : CompiledStatement} (hx : D.parse x = some st) (hf : shape st.form)
    {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin (D.V.k x) → Coin} (hacc : (run H P D.V x ω r).out = .acc)
    {s : ℕ} (hs : s < D.nS x) {w : Flock.Wit} {ms : List Msg}
    (hms : ∀ i, i + 1 < L.rounds x → ms.getD i [] = ((run H P D.V x ω r).msgs.map Prod.fst).getD i [])
    (hck : Flock.check DO.parse ((runVstar H D I P x ω r).getD s []) w = true)
    (hT : ∀ e ∈ L.calls x s w, H e.1 = e.2)
    (hnc : (findColl (L.calls x s w ++ vcallsAt H D ((runVstar H D I P x ω r).getD s []))).isSome = false)
    (hh : ∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m)
    (hp : ∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m)
    (hr : L.RowsRead x s w ms) :
    PartAccepts H D DO L shape shows cmt s x (coinSeq r) ms := by
  obtain ⟨hsh, hcm⟩ := hSt x st hx hf Ω P ω r hacc s hs
  exact ⟨⟨st, hx, hf⟩, _, w, hck, hT, hnc, hsh, hh, hp, LayoutB.cmtAgrees_of_eq fun i hi _ => (hms i hi).trans (hcm i hi).symm, hr⟩

end Honest

end Claims.Recursion.VStarB

namespace RecDeploy

open Claims (Msg)
open FlockDeploy (Coin)

/-- **Part `s`'s circuit accepts on an accepted run at `deployRec`**, with V\*'s staging (`showsPart`, `cmtTr`):
`partAccepts_of_run` at `stagedB_deployRec_part`, wherever the reader `untr` reads the coin server's transcript back
(`ReadsBackB`, which `readsBack_recTr` gives at `recRead`) and the layout's holdings pin its rounds (`HeldB`). -/
theorem partAccepts_run_deployRec {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (untr : TrReader) (DO : Claims.Flock.Deploy Coin Col T)
    (L : Claims.Recursion.VStarB.LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hrb : ReadsBackB H I.V b shape untr) (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope b) I L shape)
    {x : Msg} {st : Claims.Recursion.CompiledStatement} (hx : (deployRec I.V ivScope b).parse x = some st) (hf : shape st.form)
    {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {r : Fin ((deployRec I.V ivScope b).V.k x) → Coin}
    (hacc : (Claims.run H P (deployRec I.V ivScope b).V x ω r).out = .acc) {s : ℕ}
    (hs : s < (deployRec I.V ivScope b).nS x) {w : Claims.Flock.Wit} {ms : List Msg}
    (hms : ∀ i, i + 1 < L.rounds x →
      ms.getD i [] = ((Claims.run H P (deployRec I.V ivScope b).V x ω r).msgs.map Prod.fst).getD i [])
    (hck : Claims.Flock.check DO.parse
      ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s []) w = true)
    (hT : ∀ e ∈ L.calls x s w, H e.1 = e.2)
    (hnc : (Claims.findColl (L.calls x s w ++ Claims.Recursion.VStarB.vcallsAt H (deployRec I.V ivScope b)
      ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s []))).isSome = false)
    (hh : ∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m)
    (hp : ∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m)
    (hr : L.RowsRead x s w ms) :
    Claims.Recursion.VStarB.PartAccepts H (deployRec I.V ivScope b) DO L shape (showsPart untr b) (cmtTr untr) s x
      (Claims.coinSeq r) ms :=
  Claims.Recursion.VStarB.partAccepts_of_run (stagedB_deployRec_part H I ivScope b untr L shape hrb hHeld) hx hf hacc
    hs hms hck hT hnc hh hp hr

end RecDeploy
