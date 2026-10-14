import Proofs.Flock.C3.Compose.HoloStageCoins

/-!
# `LaterCoins` from the inner session's records

`coinsAt_held`: a stage's `CoinsAt` from custody's coins half, at any stage whose coin steps are the coins the verifier
holds of one stream at rounds `ks` that the history's rounds from `j` on draw in order (`coinsAt_rep` is a rep's, its
rounds the record's). `StageHeld` names that of a stage, and `coinsAt_stageHeld` gives its `CoinsAt`.
`linkPoints_of_custody`: the link points are the history's first coin, from custody's third conjunct (the session
record's points are the map's at the first coin) where the map's points are the first coin's coordinate
(`PointsEncoded`) and the run's points are the record's.
`laterCoins_of_held`: `LaterCoins` at a run whose link points are the record's and whose later stages (both sparse
phases, R's opening, R's Ligerito) are each `StageHeld` from where the one before ends.
`stageHeld_repsOfVals`: the reps' stages are `StageHeld` at `repsOfVals`, rep 0's from round 1 and rep 1's after it,
from their rounds' custody (`RepRoundsHeld`) and draws (`RepsDrawn`).
-/

namespace FlockC3.Reps

noncomputable section

open FlockSoundness.Game (Msg Step)
open FlockVBridge.Family (coinSteps)
open FlockC3.Compose (CoinsAt)
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Assumptions (InnerRecordMap heldCoins InnerRecordCustody)

variable {r : ℕ} {CT : Fin r → Type}

/-- **A stage's coins are the history's from round `j` on**, at any stage: its coin steps are the coins `g q` the
verifier holds of stream `s` at round `ix q`, `q` over `ks` (`hts`, `hd`), and the history's round `j + p` draws the
`p`-th of them (`hdr`), which custody holds. The draws are read at any selector `sel` within the map's (`hsel`), where
the map's coins are the history's (`henc`): candidate 3's map's are, where a stream round takes one history round. -/
theorem coinsAt_held {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))} {ts : List Step} {ks : List ℕ}
    {g : ℕ → Array Flock.F128} {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
    {m : InnerRecordMap Cm ((k : Fin r) → CT k)} {sel : ℕ → String → ℕ → Prop} {s : String} {ix : ℕ → ℕ} {j : ℕ}
    (hts : ts.filterMap Sum.getRight? = ks.map (coinArr FlockSoundness.Refine.fOf ∘ g))
    (hcus : InnerRecordCustody wr held sparse m) (henc : CoinsEncoded sel m.coin)
    (hsel : ∀ i s q, sel i s q → m.selC i s q)
    (hd : ∀ q c, heldCoins (held (wr t)) s (ix q) = some c → g q = c)
    (hdr : j + ks.length ≤ r ∧ ∀ p q, ks[p]? = some q → sel (j + p) s (ix q))
    (hlen : r + 1 ≤ t.length) :
    nCoins ts = ks.length ∧ CoinsAt (t.map Prod.snd) j (nCoins ts) ts := by
  obtain ⟨hr, hdr'⟩ := hdr
  refine coinsAt_of_idx hts hr (by rw [List.length_map]; omega) ?_
  intro p q i hi hc hp hip
  have hs : sel i s (ix q) := by rw [hip]; exact hdr' p q hp
  have hc' : i < t.length := by rw [List.length_map] at hc; exact hc
  have hround := hd q _ (((hcus t).1 i t[i] (List.getElem?_eq_getElem hc')).2 s (ix q) (hsel i s (ix q) hs))
  show coinArr FlockSoundness.Refine.fOf (g q) = _
  rw [hround, List.map_take, henc i hi s (ix q) (t.map Prod.snd) hc hs]

/-- **Hypothesis: a stage's coins are one stream's, drawn in order from the history's round `j` on**: its coin steps
are the coins `g q` the verifier holds of a stream `s` at round `ix q`, `q` over `ks`, and the history's round `j + p`
draws round `ix q` of `s`, `q` the `p`-th of `ks`, below `r`. -/
def StageHeld {Cm : Type} (t : List (Cm × ((k : Fin r) → CT k)))
    (wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String)
    (held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits) (selC : ℕ → String → ℕ → Prop)
    (j : ℕ) (ts : List Step) : Prop :=
  ∃ (ks : List ℕ) (g : ℕ → Array Flock.F128) (s : String) (ix : ℕ → ℕ),
    ts.filterMap Sum.getRight? = ks.map (coinArr FlockSoundness.Refine.fOf ∘ g) ∧
      (∀ q c, heldCoins (held (wr t)) s (ix q) = some c → g q = c) ∧
      j + ks.length ≤ r ∧ ∀ p q, ks[p]? = some q → selC (j + p) s (ix q)

theorem coinsAt_stageHeld {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))} {ts : List Step}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
    {m : InnerRecordMap Cm ((k : Fin r) → CT k)} {sel : ℕ → String → ℕ → Prop} {j : ℕ}
    (hcus : InnerRecordCustody wr held sparse m) (henc : CoinsEncoded sel m.coin)
    (hsel : ∀ i s q, sel i s q → m.selC i s q) (hlen : r + 1 ≤ t.length)
    (h : StageHeld t wr held sel j ts) : CoinsAt (t.map Prod.snd) j (nCoins ts) ts := by
  obtain ⟨_, _, _, _, hts, hd, hr, hdr⟩ := h
  exact (coinsAt_held hts hcus henc hsel hd ⟨hr, hdr⟩ hlen).2

/-- **Hypothesis: the map's link points are the first coin's coordinate** (`InnerRecordMap.points`, the coin types'
shape): as a run's coin step, `pts` of the points the map gives at a coin is that coin's coordinate 0. -/
def PointsEncoded {P : Type} (pts : Array (Array Flock.F128) → P)
    (points : ((k : Fin r) → CT k) → Array (Array Flock.F128)) : Prop :=
  ∀ (h : 0 < r) (c : (k : Fin r) → CT k), (⟨P, pts (points c)⟩ : Msg) = ⟨CT ⟨0, h⟩, c ⟨0, h⟩⟩

/-- **The link points are the history's first coin**, from custody's third conjunct: the session record decodes to
`sess`, whose points are the map's at the first coin, and `u` is `pts` of them. -/
theorem linkPoints_of_custody {Cm P : Type} {t : List (Cm × ((k : Fin r) → CT k))}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
    {m : InnerRecordMap Cm ((k : Fin r) → CT k)} {pts : Array (Array Flock.F128) → P} {spec : Flock.SessionSpec}
    {sess : Flock.Session} {u : P}
    (hcus : InnerRecordCustody wr held sparse m) (henc : PointsEncoded pts m.points)
    (hdec : Flock.Record.decode spec (wr t).2 = .ok sess) (hu : pts sess.points = u) (hr : 0 < r)
    (ht : 0 < t.length) : (coinSteps CT 0 (t.map Prod.snd)).head? = some ⟨P, u⟩ := by
  cases t with
  | nil => exact absurd ht (Nat.lt_irrefl 0)
  | cons e t' =>
    have hp := (hcus (e :: t')).2 e (by simp) spec sess hdec
    simp only [List.map_cons, coinSteps, hr, ↓reduceDIte, List.singleton_append, List.head?_cons]
    rw [← hu, hp, henc hr e.2]

end

end FlockC3.Reps

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (nCoins CoinsEncoded StageHeld PointsEncoded)

/-- **`LaterCoins` from the inner session's records**: the link points are the session record's (`hdec`, `hu`, at
`PointsEncoded`), and each later stage is one stream's, drawn in order from the round after the stage before it
(`StageHeld`): both sparse phases after the reps, R's opening after them, R's Ligerito after the opening. -/
theorem laterCoins_of_held {r : ℕ} {CT : Fin r → Type} {e : ExecHolo} {Cm : Type}
    {t : List (Cm × ((k : Fin r) → CT k))} {u : C3Run e}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
    {m : FlockSoundness.Assumptions.InnerRecordMap Cm ((k : Fin r) → CT k)} {sel : ℕ → String → ℕ → Prop}
    {pts : Array (Array Flock.F128) → Fin 2 → Fin e.cls.mPts → GF128} {spec : Flock.SessionSpec}
    {sess : Flock.Session}
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m) (henc : CoinsEncoded sel m.coin)
    (hsel : ∀ i s q, sel i s q → m.selC i s q)
    (hpe : PointsEncoded pts m.points) (hdec : Flock.Record.decode spec (wr t).2 = .ok sess)
    (hu : pts sess.points = u.pts) (hr : 0 < r) (hlen : r + 1 ≤ t.length)
    (h2 : StageHeld t wr held sel (1 + nCoins (u.ts 0) + nCoins (u.ts 1)) (u.sp 0))
    (h3 : StageHeld t wr held sel (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0)) (u.sp 1))
    (h4 : StageHeld t wr held sel (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1))
      u.op)
    (h5 : StageHeld t wr held sel
      (1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) + nCoins u.op) u.lg) :
    LaterCoins (t.map Prod.snd) u :=
  ⟨FlockC3.Reps.linkPoints_of_custody hcus hpe hdec hu hr (by omega),
    FlockC3.Reps.coinsAt_stageHeld hcus henc hsel hlen h2, FlockC3.Reps.coinsAt_stageHeld hcus henc hsel hlen h3,
    FlockC3.Reps.coinsAt_stageHeld hcus henc hsel hlen h4, FlockC3.Reps.coinsAt_stageHeld hcus henc hsel hlen h5⟩

end

end FlockC3.Compose

namespace FlockC3.Reps

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)

theorem nCoins_of {ts : List Step} {ks : List ℕ} {f : ℕ → Msg} (h : ts.filterMap Sum.getRight? = ks.map f) :
    nCoins ts = ks.length := by
  show (ts.filterMap Sum.getRight?).length = _
  rw [h, List.length_map]

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
  {sh : Flock.Shape}
  {rep : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {vals : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String}
  {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits}
  {selC : ℕ → String → ℕ → Prop} {s : Fin 2 → String} {ix : Fin 2 → ℕ → ℕ}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The reps' stages are `StageHeld`**, rep 0's from the history's round 1 on and rep 1's after it: rep `i`'s coin
steps are its consumed rounds' coins (`repIdx`, from its steps' weave), each the coins the verifier holds of stream
`s i` at its round (`RepRoundsHeld`, through `ProofOfVals`' coins), drawn in order from `repStart` (`RepsDrawn`). A
sparse phase's and R's stage take the same four parts: the coin steps' rounds from the stage's weave, the rounds held of
its stream, and the draws from where the stage before it ends. -/
theorem stageHeld_repsOfVals
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hs : SetupOfVals zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hheld : RepRoundsHeld zs dg N k Rw sh rep wr held s ix)
    (hdr : RepsDrawn zs dg N k Rw sh rep vals selC s ix) :
    AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      StageHeld (CT := CT) t wr held selC 1 (u.ts 0) ∧
        StageHeld (CT := CT) t wr held selC (1 + nCoins (u.ts 0)) (u.ts 1) := by
  intro R ω t R₂ τ o hA
  have hw := repStepsWoven_of_tie repWeaveAt_repsOfVals hpr hs hf
  have one : ∀ i : Fin 2,
      nCoins ((tiedSrc (repsOfVals stp sh rep vals) R ω t (layers zs dg N k Rw R ω t R₂ τ)).ts i) =
          (repIdx ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
            (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) i)).length ∧
        StageHeld (CT := CT) t wr held selC
          (repStart (repIdx ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
            (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) 0)).length i)
          ((tiedSrc (repsOfVals stp sh rep vals) R ω t (layers zs dg N k Rw R ω t R₂ τ)).ts i) := by
    intro i
    have ha := halg R ω t R₂ τ o hA i
    have hc := hacc R ω t R₂ τ o hA i
    have hts : ((tiedSrc (repsOfVals stp sh rep vals) R ω t (layers zs dg N k Rw R ω t R₂ τ)).ts i).filterMap
          Sum.getRight? =
        (repIdx ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
          (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) i)).map (coinArr fOf ∘ fun q =>
            ((vals R ω t (layers zs dg N k Rw R ω t R₂ τ) i).t.rounds[q]!).coins) :=
      (Woven.steps (hw R ω t R₂ τ o hA i ha hc)).2.2.trans
        (unitCoins_eq ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
          (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) i))
    refine ⟨nCoins_of hts, _, _, s i, ix i, hts, fun q c hq => ?_, hdr R ω t R₂ τ o hA i ha hc⟩
    have hr := hheld R ω t R₂ τ o hA i ha hc q c hq
    rw [(hpr R ω t R₂ τ o hA i ha hc).coins] at hr
    exact getElem!_of_map hr
  obtain ⟨h0, s0⟩ := one 0
  refine ⟨s0, ?_⟩
  rw [h0]
  exact (one 1).2

end

end FlockC3.Reps
