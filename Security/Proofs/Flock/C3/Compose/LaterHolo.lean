import Proofs.Flock.C3.Compose.HoloStageCoins
import Proofs.Flock.C3.RegLevels.LgHeld

/-!
# `LaterCoins` at the verifier's map, from its hold of the inner session

`laterCoins_holo`: `laterCoins_of_held` at the verifier's map (`holoMap`) at one run. The encodings are the map's
(`coinsEncoded_holoMap`, `coinWordsEncoded_holoMap`, `pointsEncoded_holoMap`), the sparse phases' stages are
`stageHeldW_sparse` from their starts (`sparseDrawnAt_holo`, `stageStarts`), and R's opening's and R's Ligerito's are
`stagesHeld_of_run`'s, within the history's rounds where the stages take `holoCTs`' count (`coins_total_holo`).
`stageCoins_holo_held`: `stageCoins_holo` at `holoMap` with `hlater` from `laterCoins_holo` at every accepted outer
proof and `hlt` from `srcLevelTie_of_maps`, both at R's run in V*'s sessions (`MapsAt`) at the `C3` the verifier
builds of its hold (`C3Held`), and the reps' `henc` at the map's one-round draws (`coinsEncoded_holoMap`). The
record facts the stages read are named below (`RecordPoints` … `RepStagesCounted`).
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (nCoins StageHeld StageHeldW SrcRepTie repsOfVals UnitVals ProofOfVals SetupOfVals FramesOfVals
  RepRoundsHeld RepsDrawn CoinsEncoded)
open Flock.Guarantees (holoCTs holoKinds heldBefore holoRounds c3Of)

theorem sparse_c3Of {ish : RecVStar.Inner} {cs : RecVStar.InnerCommits} {c3 : RecCoins.C3} {sp : RecSparse.Shape}
    (hsp : ish.sparse = some sp) (hc3 : c3Of ish (some cs) = some c3) : c3.sparse = sp := by
  unfold Flock.Guarantees.c3Of at hc3
  rw [hsp] at hc3
  cases hc3
  rfl

section held

variable {Cm Coin : Type}

/-- **Hypothesis: the link points are the session record's**: the coin server's record at the history decodes to a
session whose points read as the run's link points (`laterCoins_of_held`'s `hdec`, `hu`). -/
def RecordPoints (wr : List (Cm × Coin) → Lean.Json × String) {e : ExecHolo} (t : List (Cm × Coin))
    (u : C3Run e) : Prop :=
  ∃ (spec : Flock.SessionSpec) (sess : Flock.Session), Flock.Record.decode spec (wr t).2 = .ok sess ∧
    holoPts e.cls.mPts sess.points = u.pts

/-- **Hypothesis (`stageHeldW_sparse`'s `hh`): the verifier holds the inner transcript** at the history. -/
def HeldOk (wr : List (Cm × Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String RecVStar.InnerCommits) (t : List (Cm × Coin)) : Prop :=
  ∃ cs, held (wr t) = .ok cs

/-- **Hypothesis: what it holds counts the sparse streams' rounds** (`SparseRoundsCounted`). -/
def SparseCountedAt (wr : List (Cm × Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String RecVStar.InnerCommits) (sp : RecSparse.Shape)
    (t : List (Cm × Coin)) : Prop :=
  ∀ cs, held (wr t) = .ok cs → FlockC3.Sparse.SparseRoundsCounted cs sp

/-- **Hypothesis: the sparse stages are `sparseCoins`' at the `C3` of what it holds**: each sparse phase's steps are
the phase's at the verifier's sparse shape, on some words and the coins `sparseCoins` accepts there. -/
def SparseStagesAt (wr : List (Cm × Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String RecVStar.InnerCommits) (ish : RecVStar.Inner) (unit : String)
    (sp : RecSparse.Shape) {e : ExecHolo} (t : List (Cm × Coin)) (u : C3Run e) : Prop :=
  ∀ cs c3, held (wr t) = .ok cs → c3Of ish (some cs) = some c3 → ∀ x : Fin 2, ∃ w sc,
    RecCoins.sparseCoins c3 unit x.val = .ok sc ∧
      u.sp x = FlockC3.Sparse.stepsOf (sparseOf sp) w (FlockC3.Sparse.sparseCs sc)

/-- **Hypothesis: what it holds has R's stream**, a round of coins for each of R's rounds `rs`
(`stackedStreams_of_covers`' `hx`, `hxs`). -/
def StackedHeldAt (wr : List (Cm × Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String RecVStar.InnerCommits) (rs : Array RecVStar.SchedRound)
    (t : List (Cm × Coin)) : Prop :=
  ∀ cs, held (wr t) = .ok cs →
    ∃ x : RecVStar.InnerStream, cs.stream? "stacked/rep0" = some x ∧ x.coins.size = rs.size

/-- **Hypothesis: each rep's stage takes its stream's history rounds** (`stageStarts`' `hts`). -/
def RepStagesCounted (rep : Array RecVStar.SchedRound) {e : ExecHolo} (u : C3Run e) : Prop :=
  ∀ i : Fin 2, nCoins (u.ts i) = heldBefore (.rep i) rep rep.size

end held

/-- **`LaterCoins` at the verifier's map at one run** (`laterCoins_of_held` at `holoMap`): the link points the
record's (`hdec`, `hu`), the sparse stages `stageHeldW_sparse` at `sparseCoins`' coins (`hss`) from
`sparseDrawnAt_holo`'s starts (`stageStarts`), and R's stages `stagesHeld_of_run` at R's run (`hrun`, `hz`,
`hleft`), where the stages take `holoCTs`' count (`htot`). -/
theorem laterCoins_holo {st : Flock.Stmt} {hp : HoloParams} {Cm : Type}
    {t : List (Cm × ((k : Fin (1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k))}
    {u : C3Run (holoAt st hp)}
    {wr : List (Cm × ((k : Fin (1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)) →
      Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape}
    {ish : RecVStar.Inner} {tbl : String} {bc : Cm → String → Nat → List UInt8} {spec : Flock.SessionSpec}
    {sess : Flock.Session} {cs : RecVStar.InnerCommits} {c3 : RecCoins.C3} {unit : String}
    {rep rs : Array RecVStar.SchedRound} {sp : RecSparse.Shape} {sh : RecVStar.StackedShape}
    {x : RecVStar.InnerStream} {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt} {Reg' : Type}
    {M : FlockC3.Reps.StackedMaps stmt Vs} {zs : (s : Fin S) → (Vs s).Outer Reg'} {Lb : (s : Fin S) → (Vs s).Bits}
    {osh : Flock.Shape} {w : Array F128} {kap : Array (Array (Array F128))} {e0 sb e cap₀} {res : Array F128}
    {k₁ : ℕ}
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse (Flock.Guarantees.holoMap st hp Cm ish tbl bc))
    (hdec : Flock.Record.decode spec (wr t).2 = .ok sess) (hu : holoPts (holoAt st hp).cls.mPts sess.points = u.pts)
    (hlen : 1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length + 1 ≤ t.length)
    (hh : held (wr t) = .ok cs) (hc3 : c3Of ish (some cs) = some c3)
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rep) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some sh) (hcl : sh.claims = RecVStar.STACKED_CLAIMS)
    (hrs : RecVStar.stackedRounds sh = some rs) (hc : 0 < ish.claims)
    (hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims)
    (hshape : (holoAt st hp).cls.shape = sparseOf sp) (htblS : tbl ≠ s!"{RecVStar.SPARSE}/")
    (htbl : tbl ≠ s!"{RecVStar.STACKED}/") (hts : RepStagesCounted rep u)
    (hcnt : FlockC3.Sparse.SparseRoundsCounted cs sp)
    (hss : ∀ x : Fin 2, ∃ w sc, RecCoins.sparseCoins c3 unit x.val = .ok sc ∧
      u.sp x = FlockC3.Sparse.stepsOf (sparseOf sp) w (FlockC3.Sparse.sparseCs sc))
    (hx : cs.stream? "stacked/rep0" = some x) (hxs : x.coins.size = rs.size)
    (hrun : RecOpening.run osh (FlockC3.RegLevels.msgsO osh w (FlockC3.RegLevels.opsAt
        (FlockC3.RegLevels.readMaps M osh.sched) osh Lb kap c3 unit w)) (FlockC3.Opening.claimsO osh w c3 unit) 0
      (RecAlgebra.Coins.ofRounds (FlockC3.Opening.rep0 c3 unit)) = .ok (res, k₁))
    (hz : ∀ y ∈ res, y = 0) (hleft : FlockC3.RegLevels.MapsLeft M zs Lb osh unit c3 w kap e0 sb e cap₀ u)
    (htot : 1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op + nCoins u.lg =
      1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length) :
    LaterCoins (CT := holoCTs st hp) (t.map Prod.snd) u := by
  have hcm := FlockC3.RegLevels.commits_c3Of hc3
  have hss' : ∀ x : Fin 2, ∃ w cw, u.sp x = FlockC3.Sparse.stepsOf (sparseOf sp) w cw := fun x => by
    obtain ⟨w, sc, -, h⟩ := hss x
    exact ⟨w, _, h⟩
  have hst := FlockC3.Sparse.stageStarts (u := u) hts hss'
  have hW : ∀ x : Fin 2, StageHeldW t wr held (holoSelWords st hp ish tbl)
      (1 + 2 * heldBefore (.rep 0) rep rep.size + x.val * FlockVBridge.Sparse.nc (sparseOf sp)) (u.sp x) := by
    intro x
    obtain ⟨w, sc, hsc, he⟩ := hss x
    rw [he]
    exact FlockC3.Sparse.stageHeldW_sparse w (by rw [hcm]; exact hh) (by rw [hcm]; exact hcnt) (sparse_c3Of hsp hc3)
      hsc (FlockC3.Sparse.sparseDrawnAt_holo x hsp hrep ⟨_, FlockC3.RegLevels.holoStreams_four hrep hsp hsh hrs⟩ hc
        hA hshape htblS)
  obtain ⟨h4, h5⟩ := FlockC3.RegLevels.stagesHeld_of_run (t := t) hh hc3 hrep hsp hsh hcl hrs hx hxs htbl hts hss'
    hrun hz hleft htot.le
  refine laterCoins_of_held hcus coinsEncoded_holoMap (fun _ _ _ h => selC_of_holoSelOne h) coinWordsEncoded_holoMap
    (fun _ _ _ h => selC_of_holoSelWords h) (pointsEncoded_holoMap st hp Cm ish tbl bc) hdec hu (by omega) hlen ?_ ?_
    h4 h5
  · rw [hst.2.1]
    exact hW 0
  · rw [hst.2.2.1]
    exact hW 1

section src

variable {nT : ℕ} {T : Fin nT → Type} {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n}
  {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T (holoCTs st hp) st hp

/-- **Hypothesis: the run's `C3` is the one the verifier builds of its hold** (`c3Of`), at every history. -/
def C3Held (wr : List (Cm × (ℐ).Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String RecVStar.InnerCommits) (ish : RecVStar.Inner)
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3) : Prop :=
  ∀ R ω t Lb cs, held (wr t) = .ok cs → c3Of ish (some cs) = some (c3 R ω t Lb)

/-- **Hypothesis: R's opening is in V*'s sessions at the map `M`** (`srcLevelTie_of_maps`' `h`): at every accepted
outer proof, R's opening at the openings `opsAt` is in V*'s sessions, the claim tie holds, the source's opening is on
the ring switches V*'s layer holds (`OpenTieL`), and `MapsLeft`. -/
def MapsAt (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (M : FlockC3.Reps.StackedMaps stmt Vs) (sh : Flock.Shape)
    (unit : String) (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (kap : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array (Array (Array F128)))
    (e0 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ByteArray)
    (sb : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray)
    (e : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ByteArray)
    (cap₀ : Cap Dig) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    let Lb := layers zs dg N k Rw R ω t R₂ τ
    FlockC3.Opening.SessionAt (zs R ω t R₂) Lb sh unit (c3 R ω t Lb) (w R ω t Lb)
        (FlockC3.RegLevels.opsAt (FlockC3.RegLevels.readMaps M sh.sched) sh Lb (kap R ω t Lb) (c3 R ω t Lb) unit
          (w R ω t Lb)) ∧
      (∀ a b, FlockC3.Opening.coinClaim (c3 R ω t Lb) unit sh 0 = .ok a →
        FlockC3.Opening.coinClaim (c3 R ω t Lb) unit sh 1 = .ok b →
        ∀ i, ClaimRel (holoAt st hp).cls.shape.mReg (FlockC3.Opening.claimO sh (w R ω t Lb) ![a, b] i).toClaim
          ((tiedSrc stp R ω t Lb).claim i).toClaim) ∧
      FlockC3.Opening.OpenTieL (zs R ω t R₂) Lb sh (w R ω t Lb) (tiedSrc stp R ω t Lb) ∧
      FlockC3.RegLevels.MapsLeft M (zs R ω t R₂) Lb sh unit (c3 R ω t Lb) (w R ω t Lb) (kap R ω t Lb) (e0 R ω t Lb)
        (sb R ω t Lb) (e R ω t Lb) cap₀ (tiedSrc stp R ω t Lb)

/-- **`stageCoins_holo` at the verifier's map**, `hlater` discharged: at every accepted outer proof `laterCoins_holo`,
at R's run from `MapsAt`'s session (`run_of_sessionAt`) at the `C3` of the verifier's hold (`C3Held`), the stages'
total from their ties (`coins_total_holo`), the history's length from `hleaf` (`(ℐ).r` is the table's coin kinds'
`1 + length` and one more, by `schedInner`), and `hlt` from `MapsAt` (`srcLevelTie_of_maps`). The reps' rounds are
encoded and drawn at the map's one-round draws (`coinsEncoded_holoMap`; `hdr` at `holoSelOne`, as
`repsDrawn_holoMapAt` gives it). -/
theorem stageCoins_holo_held {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape}
    {ish : RecVStar.Inner} {tbl unit : String} {bc : Cm → String → Nat → List UInt8} {s : Fin 2 → String}
    {ix : Fin 2 → ℕ → ℕ} {rr rs : Array RecVStar.SchedRound} {sp : RecSparse.Shape} {shR : RecVStar.StackedShape}
    {M : FlockC3.Reps.StackedMaps stmt Vs} {osh : Flock.Shape}
    {c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {kap : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array (Array (Array F128))}
    {e0 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ByteArray}
    {sb : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray}
    {e : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ByteArray}
    {cap₀ : Cap Dig} (hleaf : HLeafC3 zs dg N k Rw)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hsv : SetupOfVals zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse (Flock.Guarantees.holoMap st hp Cm ish tbl bc))
    (hheldR : RepRoundsHeld zs dg N k Rw sh rep wr held s ix)
    (hdr : RepsDrawn zs dg N k Rw sh rep vals (holoSelOne ish tbl) s ix)
    (hrt : SrcRepTie zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hst : FlockC3.Sparse.SrcSparseTie zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hot : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => ∃ sh w ops,
      FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u)
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hosh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some osh)
    (hmaps : MapsAt zs dg N k Rw M osh unit (repsOfVals stp sh rep vals) c3 w kap e0 sb e cap₀)
    (hc3 : C3Held wr held ish c3) (hrep : holoRounds ish.m ish.kLog ish.claims = some rr)
    (hsp : ish.sparse = some sp) (hsh : ish.stacked = some shR) (hcl : shR.claims = RecVStar.STACKED_CLAIMS)
    (hrs : RecVStar.stackedRounds shR = some rs) (hc : 0 < ish.claims)
    (hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims)
    (hshape : (holoAt st hp).cls.shape = sparseOf sp) (htblS : tbl ≠ s!"{RecVStar.SPARSE}/")
    (htbl : tbl ≠ s!"{RecVStar.STACKED}/")
    (hpts : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u => RecordPoints wr t u)
    (hheld : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => HeldOk wr held t)
    (hcnt : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => SparseCountedAt wr held sp t)
    (hss : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      SparseStagesAt wr held ish unit sp t u)
    (hx : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => StackedHeldAt wr held rs t)
    (hts : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => RepStagesCounted rr u) :
    AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      StageCoins (CT := holoCTs st hp) (t.map Prod.snd) u := by
  have hlt := FlockC3.RegLevels.srcLevelTie_of_maps M hell hosh unit (repsOfVals stp sh rep vals) c3 w
    kap e0 sb e hmaps
  refine stageCoins_holo hleaf halg hacc hpr hsv hf hcus hheldR coinsEncoded_holoMap
    (fun _ _ _ h => selC_of_holoSelOne h) hdr hrt hst hot hlt ?_
  intro R ω t R₂ τ o hA'
  obtain ⟨cs, hh⟩ := hheld R ω t R₂ τ o hA'
  have hc3' := hc3 R ω t (layers zs dg N k Rw R ω t R₂ τ) cs hh
  obtain ⟨spec, sess, hdec, hu⟩ := hpts R ω t R₂ τ o hA'
  obtain ⟨xs, hxe, hxs⟩ := hx R ω t R₂ τ o hA' cs hh
  obtain ⟨hs, -, -, hleft⟩ := hmaps R ω t R₂ τ o hA'
  obtain ⟨_, _, _, _, -, -, hrun, hz⟩ := FlockC3.Opening.run_of_sessionAt hA' hosh hs
  obtain ⟨_, _, _, ho, hor⟩ := hot R ω t R₂ τ o hA'
  exact laterCoins_holo hcus hdec hu (hleaf R ω t R₂ τ o hA') hh hc3' hrep hsp hsh hcl hrs hc hA hshape htblS htbl
    (hts R ω t R₂ τ o hA') (hcnt R ω t R₂ τ o hA' cs hh) (hss R ω t R₂ τ o hA' cs _ hh hc3') hxe hxs hrun hz hleft
    (coins_total_holo (hrt R ω t R₂ τ o hA') (hst R ω t R₂ τ o hA') ho hor (hlt R ω t R₂ τ o hA'))

end src

end

end FlockC3.Compose
