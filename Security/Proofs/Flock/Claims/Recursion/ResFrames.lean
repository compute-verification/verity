import Proofs.Flock.Claims.Recursion.ElemRead
import Proofs.Flock.Claims.Recursion.RestReadsAt
import Proofs.Flock.Claims.Recursion.DecodeTiesVals
import Proofs.Flock.Claims.Recursion.DecodeTiesW
import Proofs.Flock.Claims.Recursion.FinalPiece
import Proofs.Flock.VBridge.RecOpen

/-!
# The frame values `framesRead` reads off the opened rounds

P2 of `ResLowered` (`ResLower.FramePorts`) asks a `rec-residuals` copy's frame ports to be `Rows.elemReadC`'s, which
reads rep `k`'s frame values off the opened messages (`Rows.framesRead`): a proof and `T` that fit the rep's opened
rounds at no coin, its values at table 0's rounds. This file reads those values back as the opened rows' elements.

* `framesRead_fits`: at the messages `Bound.opened` gives at the deployed join, wherever some proof fits rep `k`'s
  opened rounds at some coins (`OpenedFit`), `framesRead` is the values of a proof that fits them there.
-/

namespace RecDeploy.ResFrames

open Flock
open Claims (Msg)
open FlockDeploy (Coin setupOf repCtx repLens cutLens rowsOf nodeOf FitsRep joinR rowsOf_joinR')
open Flock.RecAlgebra (rounds)
open FlockDecodeTies (keyVal)
open RecDeploy.RestRead (rowsR nodesR mapM_map_some)

/-- **`framesRead` at the opened messages is a fitting proof's values**: at a statement that sets up as `v`, with round
0's opened rows `o₀` and the later rounds' `ot`, wherever a proof fits rep `k`'s block of `ot` at link coin `c₁` and
coins `cs`, `framesRead` at `Bound.opened`'s messages is the values, at each element of table 0's rounds, of a proof that
fits there (the one `framesRead` chooses, at no coin). -/
theorem framesRead_fits {x : Msg} {v : FlockDeploy.Ins × SessionSpec × Array Stmt} (hv : setupOf false x = some v)
    {o₀ : List (Msg × Salt)} {ot : List (List (Msg × Salt))} (last : Msg) {k : ℕ} (hk : k / 2 < v.2.2.size)
    {c₁ : Coin} {cs : List Coin} {pT : Flock.Proof × F128}
    (hfit : FitsRep (repCtx v.2.1 v.2.2 c₁ (nodesR o₀) cs k)
      ((cutLens (repLens v.2.1 v.2.2) (ot.map rowsR)).getD k []) pT.1 pT.2) :
    ∃ pT' : Flock.Proof × F128, FitsRep (repCtx v.2.1 v.2.2 c₁ (nodesR o₀) cs k)
        ((cutLens (repLens v.2.1 v.2.2) (ot.map rowsR)).getD k []) pT'.1 pT'.2 ∧
      Rows.framesRead x (((o₀ :: ot).map fun o => joinR (o.map Prod.fst)) ++ [last]) k =
        (rounds (Rows.shapeNoCoin v.2.1 v.2.2)).flatten.toList.map (keyVal pT'.1 pT'.2) := by
  have hex : ∃ pT : Flock.Proof × F128, FitsRep (repCtx v.2.1 v.2.2 default (nodesR o₀) [] k)
      ((cutLens (repLens v.2.1 v.2.2) (ot.map rowsR)).getD k []) pT.1 pT.2 :=
    ⟨pT, Rows.fitsRep_of (c := repCtx v.2.1 v.2.2 c₁ (nodesR o₀) cs k)
      (c' := repCtx v.2.1 v.2.2 default (nodesR o₀) [] k) rfl rfl rfl
      (Rows.repCtx_n hk c₁ default _ _ cs []) rfl rfl hfit⟩
  refine ⟨Classical.choose hex,
    Rows.fitsRep_of (c := repCtx v.2.1 v.2.2 default (nodesR o₀) [] k)
      (c' := repCtx v.2.1 v.2.2 c₁ (nodesR o₀) cs k) rfl rfl rfl
      (Rows.repCtx_n hk default c₁ _ _ [] cs) rfl rfl (Classical.choose_spec hex), ?_⟩
  have h0 : (rowsOf ((((o₀ :: ot).map fun o => joinR (o.map Prod.fst)) ++ [last]).headD [])).map
      (List.map nodeOf) = some (nodesR o₀) := by
    simp only [List.map_cons, List.cons_append, List.headD_cons, rowsOf_joinR', Option.map_some]
    rfl
  have h1 : ((((o₀ :: ot).map fun o => joinR (o.map Prod.fst)) ++ [last]).tail.dropLast).mapM rowsOf =
      some (ot.map rowsR) := by
    simp only [List.map_cons, List.cons_append, List.tail_cons, List.dropLast_concat]
    exact mapM_map_some (fun o => rowsOf_joinR' _) ot
  unfold Rows.framesRead
  rw [hv, h0, h1]
  dsimp only
  rw [dif_pos hex]

/-- **A fitting proof's values at the rounds are its chunks' values, round by round**: where a proof and `T` fit a
rep's rows at a context whose table folds at its own `k_log`, its values at each element of the rounds of its shape
(`shapeOf`, at the context's extra claims) are the values its rounds' chunks absorb, in order. -/
theorem frames_chunks {c : FlockDeploy.RepCtx} {rows : List (List ByteArray)} {p : Flock.Proof} {T : F128}
    (h : FitsRep c rows p T) (hkl : c.st.fold.kLog = c.st.kLog) :
    (rounds (FlockSoundness.Refine.shapeOf c.st c.sch (c.n - 2))).flatten.toList.map (keyVal p T) =
      (FlockDeploy.chunksOf c p T).flatMap FlockDeploy.vals := by
  obtain ⟨e, hne, hps⟩ := h.sizes
  obtain rfl : c.n - 2 = e := by omega
  unfold FlockDeploy.chunksOf
  rw [FlockDecodeTies.chunks_vals]
  exact (FlockDecodeTies.vals_repFrames_rounds c.dom p T rfl hkl (by simp [FlockSoundness.Refine.shapeOf]; omega)
    rfl rfl hps.r1ab hps.r1c hps.zPartial
    (fun k hk => FlockDecodeTies.rsEach_get hps (by simp [FlockSoundness.Refine.shapeOf]; omega))
    (FlockDecodeTies.yr_size hps)).symm

/-- **A value row's elements are its values**: bytes that start with the values' (`colBytes`) hold value `i` at
element `i`, below any count `n` past it. -/
theorem elemsOf_colBytes {vs : List F128} (pad : List UInt8) {n i : ℕ} (hi : i < vs.length) (hn : i < n) :
    elemsOf (FlockSoundness.Refine.colBytes vs ++ pad) n i = vs[i] := by
  unfold elemsOf
  rw [if_pos hn]
  refine FlockVBridge.AccChain.eq_of_bitOf fun t ht => ?_
  rw [FlockVBridge.bitOf_ofBits _ ht, FlockVBridge.bitL_append_left _
    (by rw [FlockSoundness.Refine.Frame.colBytes_length]; omega), FlockVBridge.bitL_colBytes, FlockVBridge.wbit,
    show (128 * i + t) / 128 = i by omega, show (128 * i + t) % 128 = t by omega,
    List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]

/-- **A round's value row holds the values its chunk absorbs**: where a round's rows hold values `vs` (`RowsAt`), the
row after its tops holds value `i` at element `i`. -/
theorem rowsAt_elem {s : RecVStar.SchedRound} {rs : List ByteArray} {vs : List F128} (h : FlockDeploy.RowsAt s rs vs)
    {n i : ℕ} (hi : i < vs.length) (hn : i < n) :
    elemsOf (rs.getD s.2.1.toList.sum default).data.toList n i = vs[i] := by
  obtain ⟨-, ⟨pad, -, hd⟩, hv⟩ := h
  cases hb : s.2.2.2 with
  | false =>
    have := hv hb
    subst this
    simp at hi
  | true =>
    rw [hb, if_pos rfl] at hd
    have e : rs[s.2.1.toList.sum]? = some ⟨(FlockSoundness.Refine.colBytes vs ++ pad).toArray⟩ := by
      have := congrArg (·[0]?) hd
      simpa [List.getElem?_drop] using this
    rw [List.getD_eq_getElem?_getD, e, Option.getD_some]
    show elemsOf (FlockSoundness.Refine.colBytes vs ++ pad).toArray.toList n i = vs[i]
    rw [List.toList_toArray]
    exact elemsOf_colBytes pad hi hn

end RecDeploy.ResFrames
