import Proofs.Flock.C3.Reps.StepsTie
import Proofs.Flock.C3.Reps.Frames
import Proofs.Flock.C3.Reps.Vals
import Proofs.Flock.C3.Compose.LastRead
import Proofs.Flock.VBridge.Family.Encode

/-!
# `RepUnitOfAlg` at holo values

The four parts `repUnitOfAlg_of_parts` takes are proved at values `vals` meeting `HoloFramed` (`FramesOfVals`,
`framesOfVals_holo'`), `ProofFrom` (`ProofOfVals`) and `SetupFrom` (`SetupOfVals`), at the step source that reads its
reps off the values (`StepsOfVals`). This file makes values that meet all three from the rep's and the facts no
construction gives (`HoloSrc.OK`), and states the unit at the source C's `SrcOK` holds at.

* `HoloSrc.vals`: G's proof file (`proofOf`), the rep's extras, the stream of the rep's coins at `holoFrameAt`'s bytes,
  the claims, ring-switch coins and batching coins at the verifier's rows (`claimsAt`, `rppAt`, `gammaAt`), and the
  Ligerito's end from there (`ligEnd`). Unlike `ValsSrc.vals`, every part that `HoloFramed` ties to the rep is made
  from the rep.
* `HoloSrc.framed`: `HoloFramed` at those values, from the setup's facts, the rep's extras, the coin counts
  (`HoloCoinsShaped`), the proof file's decode and the openings (`OpenedAll`); the record, its domain, the proof's
  messages and the end hold by construction. `enc_holo` is the decode (G's `encodes_of` at the rep's extras).
* `proofFrom_of_framed`: `ProofFrom` from `HoloFramed` and the comb port (`CombFolds`).
* `stepsOfVals_norm_repsOfVals`: `StepsOfVals` at `tiedSrc (normSrc (repsOfVals stp sh rep vals))`, where
  `srcOK_repsOfVals` gives `SrcOK`: `normOs` keeps the reps' steps, link points, cap and openings.
* `repUnitOfAlg_holoVals`: `RepUnitOfAlg` there, from `HoloSrc.OK` at each rep, the class checks and the cap.
-/

namespace FlockC3.Reps

noncomputable section

section unit

local notation "𝔽" => FlockLevel3.GF128

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockC3.HoloValues
open FlockVBridge.Compiled (RepData AlgebraHolds AccHolds)
open FlockVBridge.Family (CombFolds OpenedAll SetupRealizes ShapeIs CapShaped LigShaped proofOf proofOf_zc
  streamRounds streamRounds_coins encodeProof encodeProof_size_le decodeProof_encodeProof fileBound fast100_file
  shapeOf_plain schedPart_plain proofOf_lengths)
open FlockSoundness.Discharge.Exec (chunks ligFrames)

local instance lawfulBEqByteArray : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-- The ring switches' coins at the verifier's rows, none where they refuse. -/
def rppAt (sh : Flock.Shape) (d : RepData) : ℕ → Array F128 :=
  match verifierRows sh d.rounds d.extras d.comb with
  | .ok (_, info, _) => fun k => (info.rs.sws[k]!).rpp
  | .error _ => fun _ => #[]

/-- The batching coins at the verifier's rows, none where they refuse. -/
def gammaAt (sh : Flock.Shape) (d : RepData) : Array F128 :=
  match verifierRows sh d.rounds d.extras d.comb with
  | .ok (_, info, _) => info.rs.gamma
  | .error _ => #[]

theorem rppAt_eq {sh : Flock.Shape} {d : RepData} {w info kappa}
    (h : verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa)) :
    rppAt sh d = fun k => (info.rs.sws[k]!).rpp := by
  unfold rppAt
  rw [h]

theorem gammaAt_eq {sh : Flock.Shape} {d : RepData} {w info kappa}
    (h : verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa)) : gammaAt sh d = info.rs.gamma := by
  unfold gammaAt
  rw [h]

/-- **The Ligerito's end** on `v`'s record past ring switching (`UnitFrames.lg`'s), `v.t` where it refuses. -/
def ligEnd (v : UnitVals) : Transcript :=
  match Flock.ligerito v.st.merkle v.sch (rsT v.claims v.p.ringSwitches v.rpp v.gamma) v.p.cap v.p.ligerito
      (Flock.basis (rsSws v.claims v.rpp v.gamma))
      ⟨v.t.rounds, v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 2 + v.claims.size + 1, ByteArray.empty⟩ with
  | .ok r => r.2
  | .error _ => v.t

theorem ligEnd_eq {v : UnitVals} {t' : Transcript}
    (h : Flock.ligerito v.st.merkle v.sch (rsT v.claims v.p.ringSwitches v.rpp v.gamma) v.p.cap v.p.ligerito
      (Flock.basis (rsSws v.claims v.rpp v.gamma))
      ⟨v.t.rounds, v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 2 + v.claims.size + 1, ByteArray.empty⟩ =
        .ok ((), t')) : ligEnd v = t' := by
  unfold ligEnd
  rw [h]

/-- What a holo rep unit's values are made from besides the rep's: the setup and schedule, the proof file's PCS
parameters, cap and Ligerito openings, and the stream's domain. -/
structure HoloSrc where
  st : Flock.Setup
  sch : Flock.Schedule
  params : Flock.PcsParams
  cap : Array ByteArray
  lig : Flock.LigeritoProof
  dom : ByteArray

/-- `HoloSrc.vals` before the Ligerito's end. -/
def HoloSrc.vals₀ (b : HoloSrc) (sh : Flock.Shape) (d : RepData) : UnitVals :=
  ⟨b.st, "flock-leaf/sha512-unsalted", b.sch, proofOf sh b.params b.cap b.lig d, d.extras,
    Transcript.init (streamRounds d.rounds (holoFrameAt b.st b.sch b.dom d.extras.size
      (proofOf sh b.params b.cap b.lig d) (valOf sh d.w d.comb d.cons .T) (hOfW sh d.w))) b.dom,
    claimsAt sh d, rppAt sh d, gammaAt sh d, Transcript.init #[] b.dom⟩

/-- **A holo rep unit's values** at the rep's values `d`: the plain leaf, G's proof file, the rep's extras, the stream
of the rep's coins at `holoFrameAt`'s bytes, the verifier rows' claims and ring-switching coins, and the Ligerito's
end. -/
def HoloSrc.vals (b : HoloSrc) (sh : Flock.Shape) (d : RepData) : UnitVals :=
  { b.vals₀ sh d with t₄ := ligEnd (b.vals₀ sh d) }

/-- **The facts a holo rep unit's values need at one rep** that their construction does not give: the setup's
(`SetupRealizes`, `ShapeIs`), the rep's extras at well-formed link points at `pts` (`RepExtrasAt`), the comb port
(`CombFolds`), the record's coin counts (`HoloCoinsShaped`, `TableSrc.Coins` at holo), the shapes the proof file's
decode needs (`CapShaped`, `LigShaped`, fewer than `2^48` extras: `encodes_of`'s) and the openings (`OpenedAll`). -/
structure HoloSrc.OK (b : HoloSrc) (S : FlockSoundness.Model.Statement) (ptLocal mPts : ℕ)
    (pts : Fin 2 → Fin mPts → 𝔽) (sh : Flock.Shape) (d : RepData) : Prop where
  realizes : SetupRealizes b.st S b.sch ptLocal mPts
  shape : ShapeIs sh b.st b.sch
  extras : RepExtrasAt b.st S ptLocal mPts pts d
  fold : CombFolds b.st sh d
  coins : HoloCoinsShaped b.st.m b.st.fold.kLog d.extras.size
    (chunks (ligFrames b.sch (valOf sh d.w d.comb d.cons .T) b.cap (proofOf sh b.params b.cap b.lig d).ligerito)).1.length
    d.rounds
  capShaped : CapShaped b.st b.sch b.cap
  ligShaped : LigShaped b.st b.sch b.lig
  fits : d.extras.size < 2 ^ 48
  opened : OpenedAll b.st b.sch sh (proofOf sh b.params b.cap b.lig d) d

/-- **G's proof file decodes at the holo rep's shape** (`encodes_of` at the rep's `|extras|` extra claims, which the
verifier's rows give V*'s shape: `verifierRows_claims`). -/
theorem enc_holo {st : Flock.Setup} {S : FlockSoundness.Model.Statement} {sch : Flock.Schedule} {ptLocal mPts : ℕ}
    {sh : Flock.Shape} {params : Flock.PcsParams} {cap : Array ByteArray} {lig : Flock.LigeritoProof} {d : RepData}
    {x} (hset : SetupRealizes st S sch ptLocal mPts) (hsh : ShapeIs sh st sch)
    (hrows : verifierRows sh d.rounds d.extras d.comb = .ok x) (hcap : CapShaped st sch cap)
    (hlig : LigShaped st sch lig) (hn : d.extras.size < 2 ^ 48) :
    ∃ bs, decodeProof (shapeOf st sch d.extras.size) bs = .ok (proofOf sh params cap lig d) := by
  obtain ⟨-, -, -, hms, hf, -, hkm, -, -⟩ := hset
  obtain ⟨hlv, hpart, hm35⟩ := fast100_file _ _ hf
  obtain ⟨hH, hsl⟩ := shapeOf_plain (sch := sch) (n := d.extras.size) hms
  have hpart' := (schedPart_plain (s := sch) hsl).trans hpart
  have hlen := proofOf_lengths (params := params) (d := d) hsh (verifierRows_claims hrows) hcap hlig
  have hle := encodeProof_size_le _ _ hlen
  have hfit : (encodeProof (shapeOf st sch d.extras.size) (proofOf sh params cap lig d)).size < 2 ^ 64 := by
    refine lt_of_le_of_lt hle ?_
    unfold fileBound
    rw [hH]
    have e : (shapeOf st sch d.extras.size).sched = sch := rfl
    have em : (shapeOf st sch d.extras.size).m = st.m := rfl
    have ek : (shapeOf st sch d.extras.size).kLog = st.kLog := rfl
    have ec : (shapeOf st sch d.extras.size).claims = 2 + d.extras.size := rfl
    rw [e, em, ek, ec]
    omega
  exact ⟨encodeProof _ _, decodeProof_encodeProof _ _
    ⟨by rw [hH]; omega, show 1 ≤ sch.levels.size - 1 by omega, hfit, hlen⟩⟩

/-- **`HoloFramed` at a holo rep unit's values**, from `HoloSrc.OK` and the verifier's rows accepting
(`AlgebraHolds`): the record, its domain, the proof's messages, the claims and coins and the end are the
construction's. -/
theorem HoloSrc.framed {b : HoloSrc} {S : FlockSoundness.Model.Statement} {ptLocal mPts : ℕ}
    {pts : Fin 2 → Fin mPts → 𝔽} {sh : Flock.Shape} {d : RepData} (h : b.OK S ptLocal mPts pts sh d)
    (halg : AlgebraHolds sh d) : HoloFramed sh d (b.vals sh d) b.dom := by
  obtain ⟨v, info, kappa, S', rs, hrows, hS, hw, hc, hres, hzero⟩ := halg
  have henc := enc_holo (params := b.params) h.realizes h.shape hrows h.capShaped h.ligShaped h.fits
  obtain ⟨-, -, -, -, hfast, hm14, hkm, hkL, -⟩ := h.realizes
  obtain ⟨_, ⟨e, he, -, hes⟩, he'⟩ := h.extras
  obtain rfl : e = d.extras := Except.ok.inj (he.symm.trans he')
  exact
    { record := rfl
      k0 := rfl
      pending := rfl
      coins := h.coins
      m14 := hm14
      m := h.shape.1.symm
      kL := hkL.trans h.shape.2.1.symm
      kS := h.shape.2.1.symm
      km := hkm
      sched := h.shape.2.2
      fast := hfast
      extras := rfl
      extShape := hes
      enc := henc
      zp := fun _ _ _ hr => proofOf_zc hr
      lp := ⟨rfl, rfl⟩
      rsw := rfl
      messages := rfl
      ood := rfl
      yr := rfl
      initialCap := beq_self_eq_true _
      opened := h.opened
      vals := fun _ _ _ hr => ⟨claimsAt_eq hr, rppAt_eq hr, gammaAt_eq hr⟩
      t₄ := fun _ hl => ligEnd_eq (v := b.vals₀ sh d) hl }

/-- **`ProofFrom` from `HoloFramed`** and the comb port: the rest of `ProofFrom` is `HoloFramed`'s. -/
theorem proofFrom_of_framed {sh : Flock.Shape} {d : RepData} {v : UnitVals} {dom : ByteArray}
    (H : HoloFramed sh d v dom) (hfold : CombFolds v.st sh d) : ProofFrom sh d v where
  m := H.m
  kLog := H.kS
  kLog_le := H.km
  fast := H.fast
  extras := H.extras
  extras_wf := H.extShape
  coins := by rw [H.record, streamRounds_coins]
  zerocheck := H.zp
  lincheck := H.lp
  ringSwitches := H.rsw
  foldPts := hfold
  claims := fun w info kappa hr => by rw [(H.vals w info kappa hr).1, H.extras]

end unit

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
  {sh : Flock.Shape}
  {rep : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {vals : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`SetupFrom` at a holo rep unit's values at holo**, at a statement whose class checks pass
(`ValsSrc.setupFrom_holo`'s, at `HoloSrc.vals`). -/
theorem HoloSrc.setupFrom_holo {b : HoloSrc} (hcls : ClassOk (Recursion.shapeOf st))
    {pts : Fin 2 → Fin (holoAt st hp).cls.mPts → GF128} {d : RepData}
    (h : b.OK ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal (holoAt st hp).cls.mPts
      pts sh d) :
    SetupFrom ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (holoAt st hp).cls.ptLocal
      (holoAt st hp).cls.mPts pts sh (b.vals sh d) := by
  refine ⟨h.realizes, h.shape, sched_holoAt hcls ?_, rfl, extras_of_at h.extras⟩
  obtain ⟨hmS, -, -, -, hfast, -⟩ := h.realizes
  rw [← show b.st.m = (holoAt st hp).cls.m from hmS]
  exact hfast

theorem repWeaveAt_norm_repsOfVals : RepWeaveAt zs dg N k Rw (normSrc (repsOfVals stp sh rep vals)) sh rep vals :=
  fun _ _ _ _ _ _ _ _ _ _ => rfl

/-- **The openings at the normalized `repsOfVals`' reps are the values' proof's rows**: `normOs` keeps strata 0–2. -/
theorem hos_norm_repsOfVals : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
    ∀ ℓ j, (normSrc (repsOfVals stp sh rep vals) R ω t Lb).os.1 i ℓ j =
      openRow (vals R ω t Lb i).p.ligerito (vals R ω t Lb i).sch ℓ j :=
  fun R ω t R₂ τ o hA i halg hacc ℓ j =>
    (congrFun (congrFun (normOs_os_of_lt (u := repsOfVals stp sh rep vals R ω t (layers zs dg N k Rw R ω t R₂ τ))
      (Nat.lt_trans i.isLt (by decide))) ℓ) j).trans
      (hos_repsOfVals (stp := stp) R ω t R₂ τ o hA i halg hacc ℓ j)

/-- **`StepsOfVals` at the reps' normalized tied source** (where `srcOK_repsOfVals` gives `SrcOK`), from the reps'
other parts and the cap: `stepsOfVals_repsOfVals` through `normOs`, which keeps the reps' steps, link points, cap and
openings. -/
theorem stepsOfVals_norm_repsOfVals (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hs : SetupOfVals zs dg N k Rw (tiedSrc (normSrc (repsOfVals stp sh rep vals))) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcap : AtRep zs dg N k Rw sh rep fun R ω t Lb i => capDig (vals R ω t Lb i).p.cap = (stp R ω t Lb).cap₀) :
    StepsOfVals zs dg N k Rw (tiedSrc (normSrc (repsOfVals stp sh rep vals))) sh rep vals :=
  stepsOfVals_of_woven (repStepsWoven_of_tie repWeaveAt_norm_repsOfVals hpr hs hf) hcap hos_norm_repsOfVals

/-- **`RepUnitOfAlg` at the reps' normalized tied source, at holo values** `(b R ω t Lb i).vals sh (rep R ω t Lb i)`:
the four parts from `HoloSrc.OK` at each rep at the run's link points, the class checks and the values' cap. -/
set_option maxHeartbeats 1000000 in
theorem repUnitOfAlg_holoVals (hcls : ClassOk (Recursion.shapeOf st))
    {b : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → HoloSrc}
    (hok : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      (b R ω t Lb i).OK ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (stp R ω t Lb).pts sh (rep R ω t Lb i))
    (hcap : AtRep zs dg N k Rw sh rep fun R ω t Lb i => capDig (b R ω t Lb i).cap = (stp R ω t Lb).cap₀) :
    RepUnitOfAlg zs dg N k Rw
      (tiedSrc (normSrc (repsOfVals stp sh rep fun R ω t Lb i => (b R ω t Lb i).vals sh (rep R ω t Lb i)))) sh rep := by
  have hf : FramesOfVals zs dg N k Rw sh rep fun R ω t Lb i => (b R ω t Lb i).vals sh (rep R ω t Lb i) :=
    framesOfVals_holo' fun R ω t R₂ τ o hA i halg hacc =>
      ⟨_, HoloSrc.framed (hok R ω t R₂ τ o hA i halg hacc) halg⟩
  have hpr : ProofOfVals zs dg N k Rw sh rep fun R ω t Lb i => (b R ω t Lb i).vals sh (rep R ω t Lb i) :=
    proofOfVals_of_from fun R ω t R₂ τ o hA i halg hacc =>
      proofFrom_of_framed (HoloSrc.framed (hok R ω t R₂ τ o hA i halg hacc) halg)
        (hok R ω t R₂ τ o hA i halg hacc).fold
  have hs : SetupOfVals zs dg N k Rw
      (tiedSrc (normSrc (repsOfVals stp sh rep fun R ω t Lb i => (b R ω t Lb i).vals sh (rep R ω t Lb i)))) sh rep
      fun R ω t Lb i => (b R ω t Lb i).vals sh (rep R ω t Lb i) :=
    setupOfVals_of_from fun R ω t R₂ τ o hA i halg hacc =>
      HoloSrc.setupFrom_holo hcls (hok R ω t R₂ τ o hA i halg hacc)
  exact repUnitOfAlg_of_parts hpr hs hf (stepsOfVals_norm_repsOfVals hpr hs hf hcap)

end

end FlockC3.Reps
