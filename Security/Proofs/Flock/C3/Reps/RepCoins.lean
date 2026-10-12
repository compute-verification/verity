import Proofs.Flock.C3.Reps.RepsRead
import Proofs.Flock.Soundness.Assumptions.Inner

/-!
# The reps' coins at `repsOfVals`, from the inner session's records

`repCoins_repsOfVals`: `StageCoins`' two rep conjuncts at the tied source `tiedSrc (repsOfVals stp sh rep vals)`. Rep
`i`'s coin steps are its consumed rounds' coins (`repIdx`, `UnitWoven`'s), so its `p`-th is round `repIdx p`'s
(`unitCoins_eq`); the history's round `repStart c₀ i + p` draws it from the rep's stream (`RepsDrawn`), where custody
(`InnerRecordCustody`'s coins half) holds that round's coin, the rep's rounds are the record's (`RepRoundsHeld`), and
the map's coins are the round's coordinate (`CoinsEncoded`). `capRead_vals`: `CapRead` at values whose cap is one
constant.
-/

namespace FlockC3.Reps

noncomputable section

open FlockSoundness.Game (Msg Step)
open FlockVBridge.Family (coinSteps)
open FlockC3.Compose (CoinsAt)
open FlockSoundness.Discharge.PlaysCoin (coinArr)

section list

variable {r : ℕ} {CT : Fin r → Type}

theorem coinSteps_map {cs : List ((k : Fin r) → CT k)} {f : ℕ → Msg} : ∀ (ks : List ℕ) (j : ℕ),
    j + ks.length ≤ r → j + ks.length ≤ cs.length →
    (∀ p q i (hi : i < r) (hc : i < cs.length), ks[p]? = some q → i = j + p → f q = ⟨CT ⟨i, hi⟩, cs[i] ⟨i, hi⟩⟩) →
    coinSteps CT j ((cs.drop j).take ks.length) = ks.map f
  | [], j, _, _, _ => by rw [List.length_nil, List.take_zero]; rfl
  | q :: ks, j, hr, hc, h => by
      rw [List.length_cons] at hr hc ⊢
      have hjr : j < r := by omega
      rw [List.drop_eq_getElem_cons (by omega), List.take_succ_cons, List.map_cons]
      simp only [coinSteps, hjr, ↓reduceDIte, List.singleton_append]
      rw [h 0 q j hjr (by omega) rfl rfl, coinSteps_map ks (j + 1) (by omega) (by omega)
        fun p q' i hi hc' hq hip => h (p + 1) q' i hi hc' (by rw [List.getElem?_cons_succ]; exact hq) (by omega)]

theorem getElem!_of_map {α β : Type} [Inhabited α] {f : α → β} {a : Array α} {k : ℕ} {c : β}
    (h : (a.map f)[k]? = some c) : f a[k]! = c := by
  rw [Array.getElem?_map] at h
  cases hk : a[k]? with
  | none => rw [hk] at h; cases h
  | some x =>
      rw [hk] at h
      cases h
      rw [getElem!_def, hk]

/-- **A tape's coins from round `j` on**: its coin steps are `f` at `ks`, and the history's round `j + p` is `f` at
`ks`' `p`-th. -/
theorem coinsAt_of_idx {cs : List ((k : Fin r) → CT k)} {ts : List Step} {ks : List ℕ} {f : ℕ → Msg} {j : ℕ}
    (hts : ts.filterMap Sum.getRight? = ks.map f) (hr : j + ks.length ≤ r) (hc : j + ks.length ≤ cs.length)
    (h : ∀ p q i (hi : i < r) (hc : i < cs.length), ks[p]? = some q → i = j + p → f q = ⟨CT ⟨i, hi⟩, cs[i] ⟨i, hi⟩⟩) :
    nCoins ts = ks.length ∧ CoinsAt cs j (nCoins ts) ts := by
  have hn : nCoins ts = ks.length := by
    show (ts.filterMap Sum.getRight?).length = _
    rw [hts, List.length_map]
  refine ⟨hn, ?_⟩
  unfold CoinsAt
  rw [hn, hts]
  exact (coinSteps_map ks j hr hc h).symm

/-- Where rep `i`'s coins start in the history: after the points' coin, rep 1's after rep 0's `c₀`. -/
def repStart (c₀ : ℕ) (i : Fin 2) : ℕ := if (i : ℕ) = 0 then 1 else 1 + c₀

/-- **Hypothesis: the map's coins are the rounds' coordinates** (the coin types' shape): where round `i` draws round
`q` of stream `s`, the coins the map gives there, as a run's coin step, are coordinate `i` of the round's coin. -/
def CoinsEncoded {Cm : Type} (m : FlockSoundness.Assumptions.InnerRecordMap Cm ((k : Fin r) → CT k)) : Prop :=
  ∀ i (hi : i < r) s q (c : (k : Fin r) → CT k), m.selC i s q →
    coinArr FlockSoundness.Refine.fOf (m.coin c s q) = ⟨CT ⟨i, hi⟩, c ⟨i, hi⟩⟩

end list

section unit

open Flock FlockSoundness FlockSoundness.Refine
open FlockVBridge.Compiled (RepData)

/-- Rep `v`'s rounds whose coins its steps take (`UnitWoven`'s): all from `v.t.k` to `v.t₄.k` but `z_partial`'s. -/
def repIdx (S : FlockSoundness.Model.Statement) (v : UnitVals) : List ℕ :=
  List.range' v.t.k (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) - v.t.k) ++
    List.range' (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) + 1)
      (v.t₄.k - (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) + 1))

theorem unitCoins_eq (S : FlockSoundness.Model.Statement) (v : UnitVals) :
    (consumed v.t.rounds v.t.k (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6)) ++
      consumed v.t.rounds (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) + 1) v.t₄.k).map (coinArr fOf) =
    (repIdx S v).map (coinArr fOf ∘ fun q => (v.t.rounds[q]!).coins) := by
  unfold consumed repIdx
  rw [← List.map_append, List.map_map]

variable {r : ℕ} {CT : Fin r → Type}

/-- **One rep's coins are the history's from round `j` on**: its coin steps are its consumed rounds' coins (`hts`),
the rounds are the record's (`hd`, `hv`), and the history's round `j + p` draws the `p`-th of them (`hdr`), which
custody holds. -/
theorem coinsAt_rep {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))} {ts : List Step}
    {S : FlockSoundness.Model.Statement} {v : UnitVals} {d : RepData}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
    {m : FlockSoundness.Assumptions.InnerRecordMap Cm ((k : Fin r) → CT k)} {s : String} {ix : ℕ → ℕ} {j : ℕ}
    (hts : ts.filterMap Sum.getRight? = (repIdx S v).map (coinArr fOf ∘ fun q => (v.t.rounds[q]!).coins))
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m) (henc : CoinsEncoded m)
    (hd : ∀ q c, FlockSoundness.Assumptions.heldCoins (held (wr t)) s (ix q) = some c → d.rounds[q]? = some c)
    (hv : d.rounds = v.t.rounds.map (·.coins))
    (hdr : j + (repIdx S v).length ≤ r ∧ ∀ p q, (repIdx S v)[p]? = some q → m.selC (j + p) s (ix q))
    (hlen : r + 1 ≤ t.length) :
    nCoins ts = (repIdx S v).length ∧ CoinsAt (t.map Prod.snd) j (nCoins ts) ts := by
  obtain ⟨hr, hsel⟩ := hdr
  refine coinsAt_of_idx hts hr (by rw [List.length_map]; omega) ?_
  intro p q i hi hc hp hip
  have hs : m.selC i s (ix q) := by rw [hip]; exact hsel p q hp
  have hc' : i < t.length := by rw [List.length_map] at hc; exact hc
  have hround := hd q _ (((hcus t).1 i t[i] (List.getElem?_eq_getElem hc')).2 s (ix q) hs)
  rw [hv] at hround
  show coinArr fOf (v.t.rounds[q]!).coins = _
  rw [getElem!_of_map hround, henc i hi s (ix q) t[i].2 hs, List.getElem_map]

end unit

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf maskOf)
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
  {sh : Flock.Shape} {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {vals : ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
  {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String}
  {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {sparse : Option Flock.RecSparse.Shape}
  {m : FlockSoundness.Assumptions.InnerRecordMap Cm (holoInnerAt T CT st hp).Coin} {s : Fin 2 → String}
  {ix : Fin 2 → ℕ → ℕ}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Hypothesis: the reps' rounds are the inner session record's** (premise 2): at every accepted outer proof whose
rep `i` meets E's preconditions, its round `q`'s coins are those the verifier holds at round `ix i q` of stream `s i`. -/
def RepRoundsHeld (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (wr : List (Cm × (ℐ).Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits) (s : Fin 2 → String)
    (ix : Fin 2 → ℕ → ℕ) : Prop :=
  AtRep zs dg N k Rw sh rep fun _ _ t Lb i => ∀ q c,
    FlockSoundness.Assumptions.heldCoins (held (wr t)) (s i) (ix i q) = some c → (rep Lb i).rounds[q]? = some c

/-- **Hypothesis: the map draws the reps' consumed rounds** (`InnerRecordMap.selC`): at every accepted outer proof
whose rep `i` meets E's preconditions, the history's round `repStart c₀ i + p` (`c₀` rep 0's consumed rounds) draws
round `ix i q` of stream `s i`, `q` the `p`-th of rep `i`'s, and those rounds are below `r`. -/
def RepsDrawn (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals)
    (m : FlockSoundness.Assumptions.InnerRecordMap Cm (ℐ).Coin) (s : Fin 2 → String) (ix : Fin 2 → ℕ → ℕ) : Prop :=
  AtRep zs dg N k Rw sh rep fun _ _ _ Lb i =>
    repStart (repIdx ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (vals Lb 0)).length i +
        (repIdx ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (vals Lb i)).length ≤ r ∧
      ∀ p q, (repIdx ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (vals Lb i))[p]? = some q →
        m.selC (repStart (repIdx ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (vals Lb 0)).length i + p)
          (s i) (ix i q)

/-- **The reps' `StageCoins` conjuncts at `repsOfVals`**, from custody's coins half: rep 0's coins from the history's
round 1 on, rep 1's after them. -/
theorem repCoins_repsOfVals (hleaf : HLeafC3 zs dg N k Rw)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hs : SetupOfVals zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m)
    (hheld : RepRoundsHeld zs dg N k Rw sh rep wr held s ix) (henc : CoinsEncoded (CT := CT) m)
    (hdr : RepsDrawn zs dg N k Rw sh rep vals m s ix) :
    AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      CoinsAt (CT := CT) (t.map Prod.snd) 1 (nCoins (u.ts 0)) (u.ts 0) ∧
        CoinsAt (CT := CT) (t.map Prod.snd) (1 + nCoins (u.ts 0)) (nCoins (u.ts 1)) (u.ts 1) := by
  intro R ω t R₂ τ o hA
  have hw := repStepsWoven_of_tie repWeaveAt_repsOfVals hpr hs hf
  have one := fun i : Fin 2 =>
    have ha := halg R ω t R₂ τ o hA i
    have hc := hacc R ω t R₂ τ o hA i
    coinsAt_rep (CT := CT)
      (ts := (tiedSrc (repsOfVals stp sh rep vals) R ω t (layers zs dg N k Rw R ω t R₂ τ)).ts i)
      ((Woven.steps (hw R ω t R₂ τ o hA i ha hc)).2.2.trans
        (unitCoins_eq ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (vals (layers zs dg N k Rw R ω t R₂ τ) i)))
      hcus henc (hheld R ω t R₂ τ o hA i ha hc) (hpr R ω t R₂ τ o hA i ha hc).coins
      (hdr R ω t R₂ τ o hA i ha hc) (hleaf R ω t R₂ τ o hA)
  obtain ⟨h0, c0⟩ := one 0
  refine ⟨c0, ?_⟩
  rw [h0]
  exact (one 1).2

/-- **`CapRead` at values whose cap is one constant** (`proofOf`'s cap is `ValsSrc.cap`): at every accepted outer
proof `stp`'s cap is rep 0's (`hcap`, `stepsOfVals_repsOfVals`'), so it is `cap`'s digest at both. -/
theorem capRead_vals {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc} {cap : Array ByteArray}
    (hb : ∀ Lb i, (b Lb i).cap = cap)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hcap : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      capDig ((b Lb i).vals sh (rep Lb i)).p.cap = (stp R ω t Lb).cap₀) :
    CapRead zs dg N k Rw stp own sel := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' _
  have e := hcap R ω t R₂ τ o hA 0 (halg R ω t R₂ τ o hA 0) (hacc R ω t R₂ τ o hA 0)
  have e' := hcap R' ω' t' R₂' τ' o' hA' 0 (halg R' ω' t' R₂' τ' o' hA' 0) (hacc R' ω' t' R₂' τ' o' hA' 0)
  show (stp R ω t _).cap₀ = (stp R' ω' t' _).cap₀
  rw [← e, ← e']
  show capDig (b _ 0).cap = capDig (b _ 0).cap
  simp only [hb]

end

end FlockC3.Reps
