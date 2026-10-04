import FlockSoundness.Discharge.Recursion.Transfer
import FlockSoundness.Discharge.Recursion.Flock

/-!
# Recursion: V*'s stages, with one set of openings at every history

`zk_session_recursive_upto` takes one `Openings` for every history and stage, while the value binding of V*'s session at
a history (`rsZAtS … .bindingZ`) is built from that history's inputs. What the openings read is fixed by V*'s circuit
`c`: the commit strings (`PosK`: unit `u`'s row of port `q`, or a single gate), their values (`ValK`: bit strings up to
the widest row) and each one's leaf scheme (`schemeK`: `hm96-sha512/row/v1` of `rowEnc` at a row, `SHA-512(bitCode v)`
at a single gate). `obK` is those openings, and the value binding's openings at each history are `obK` by definition
(`ZkOuter.openingsOf_vb`); what a history's inputs change is only the registered digests (`ValueBindingZ.cs`, from the
public file).

* `candWeightK`: `candWeightZC` read off V*'s law alone (`ZkOuter.candWeight_vb`);
* `ZkOuter.candTransfer`: `zk_session_candValsR` with `candTermZC_le` at one history, the trial for each commit string a
  finder of `obK`'s openings (`ZkOuter.trial`), and the slack `ZkOuter.stageSlack`;
* `VCommits`, `VReads`: what V*'s inputs and circuit must do at each stage: the public file at history `t` registers
  `t`'s stage-`i` commitment at the stage's commit strings `Sk i`, and at an accepted outcome a round-`i` message of
  V*'s layer that the inner verifier would read differently from `msg i v` shows on a wire of `Sk i` a drawn unit reads;
* `zk_session_recursive_stages`: `zk_session_recursive_upto` with its transfer `htr` from `ZkOuter.candTransfer`, at
  `obK`, on `VCommits` and `VReads`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

section Fixed

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- **V*'s commit strings**: unit `u`'s row of port `q`, or a single gate (`Readers.ofRows`' positions at
`keyReaders`). -/
abbrev PosK : Type := (Fin n × Fin c.ports.size) ⊕ Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N

/-- **Their values**: bit strings of at most the widest row's `16 · words` bits, with their width. -/
abbrev ValK : Type := Layout.BitVal (Finset.univ.sup (Layout.portRowW c) + 1)

/-- **Each commit string's leaf scheme**: `hm96-sha512/row/v1` of `rowEnc` at a row, `SHA-512(bitCode v)` at a single
gate. -/
noncomputable def schemeK (p : PosK c hU n) : Binding.ExplicitLeaf (ValK c) :=
  if p.isLeft = true then Binding.ExplicitLeaf.hm96 Layout.rowEnc Layout.rowEnc_inj
  else Binding.ExplicitLeaf.plain Layout.bitCode Layout.bitCode_inj

/-- **V*'s openings**: a commit string, a value and a salt, opening the commit string's leaf of the value under the
salt; two openings of one commit string's leaf to different values are a SHA-512 collision (`ExplicitLeaf.pair`). -/
noncomputable def obK : Openings H512 (ValK c) where
  Op := PosK c hU n × ValK c × List UInt8
  Dg := PosK c hU n × Dig
  com o := (o.1, (schemeK c hU n o.1).leaf o.2.1 o.2.2)
  val o := o.2.1
  collide o o' := (schemeK c hU n o.1).pair o.2.1 o.2.2 o'.2.1 o'.2.2
  collide_spec o o' hc hv := by
    obtain ⟨p, v, y⟩ := o
    obtain ⟨p', v', y'⟩ := o'
    simp only [Prod.mk.injEq] at hc
    obtain ⟨rfl, hl⟩ := hc
    exact (schemeK c hU n p).pair_spec v y v' y' hv hl

/-- The commit string holding wire `g` (`Readers.ofRows`' `pos`): its row when `keyRowOf` has one, else its own. -/
noncomputable def posK (g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N) : PosK c hU n :=
  match Layout.keyRowOf c hU n (Layout.portRowW c) g with
  | some (u, ⟨q, _⟩) => .inl (u, q)
  | none => .inr g

/-- A draw of `Lv` reads commit string `pos`: a drawn unit reads a wire it holds. -/
def ReadsDrawK (Lv : Law n) (pos : PosK c hU n) (ω : Lv.Ω) : Prop :=
  ∃ u ∈ Lv.draw ω, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u, posK c hU n g = pos

open Classical in
/-- **Commit string `pos`'s weight** at the stage's commit strings `Sk`: on `Sk`, the fraction of `Lv`'s draws reading
it over `1 − ρ`. -/
noncomputable def candWeightK (Lv : Law n) (Sk : Finset (PosK c hU n)) (ρ : ℝ) (pos : PosK c hU n) : ℝ≥0∞ :=
  if pos ∈ Sk then
    ENNReal.ofReal ((Fintype.card {ω // ReadsDrawK c hU n Lv pos ω} : ℝ) / Fintype.card Lv.Ω / (1 - ρ))
  else 0

end Fixed

/-- **An event of the live audit's outcomes at the law's draws**: implications that hold at every outcome
`(R, L.draw ω, t)` bound probabilities. -/
theorem prob_auditLive_mono {n : ℕ} {L : Law n} {Reg : Type} (live : Finset (Fin n) → Reg → Game Exec.Transcript)
    (τ : Strategy (auditLive L Reg live)) {p q : Reg × Finset (Fin n) × Exec.Transcript → Prop}
    (h : ∀ R ω t, p (R, L.draw ω, t) → q (R, L.draw ω, t)) :
    prob p (auditLive L Reg live) τ ≤ prob q (auditLive L Reg live) τ := by
  show avg (fun ω => prob p ((live (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω)) ≤
    avg fun ω => prob q ((live (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω)
  refine avg_mono fun ω => ?_
  rw [prob_map, prob_map]
  exact prob_mono (fun t => h τ.1 ω t) _ _

namespace ZkOuter

variable {mPts nv : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv Reg' c hU hscope Lv) (dg : ByteArray → ByteArray) (N : ℕ)

/-- The value binding of V*'s session at the history (`rsZAtS`'s). -/
noncomputable abbrev vb :=
  (rsZAtS (Reg := Reg') zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope).bindingZ
    (rsZAtS_computes zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope)

/-- **The value binding's openings are `obK` at every history.** -/
theorem openingsOf_vb :
    openingsOf execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀)
      (tabZAt zo.dj zo.y₀) zo.vb = obK c hU nv := rfl

/-- **The trial for commit string `pos`** of V*'s session at the strategy `τ`, as a finder of `obK`'s openings. -/
noncomputable def trial (k Rw : ℕ) (τ : Strategy (zo.game dg N)) (pos : PosK c hU nv) :
    Finder (Finset (obK c hU nv).Op) :=
  trialFinderZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀)
    (tabZAt zo.dj zo.y₀) zo.vb (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw pos

/-- **A stage's slack at the strategy `τ`**: `ksAvgStrictZ + linkBoundZC + candSlackZC` of V*'s session. -/
noncomputable def stageSlack (k Rw M : ℕ) (ρ t' qF qS : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  ksAvgStrictZ (planZAt zo.dj zo.y₀) Lv k qF qS (reg (zo.sim dg N τ)) +
    linkBoundZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀)
      (tabZAt zo.dj zo.y₀) zo.vb k Rw M ρ t' (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) +
    candSlackZC (L := Lv) execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀)
      (tabZAt zo.dj zo.y₀) zo.vb k Rw ρ

/-- The value binding's commit string of each wire is `posK`'s. -/
theorem vb_pos (g : Fin ((keyProg c hU nv).circuit (Layout.outsOf c hU nv)).N) : zo.vb.pos g = posK c hU nv g := by
  dsimp only [vb, HmRowsZ.bindingZ, LayoutZ.bindingZ, layZAt, LayoutZ.ofShadow, Layout.keyedLayout, Layout.keyLayout,
    Layout.Readers.layout, Layout.ofBits, Layout.keyReaders]
  unfold posK
  rcases h : Layout.keyRowOf c hU nv (Layout.portRowW c) g with _ | ⟨u, q, k⟩
  · exact Layout.ofRows_pos_none _ h
  · exact Layout.ofRows_pos_some _ h

open Classical in
/-- **`candWeightZC` of V*'s session reads only V*'s law**: it is `candWeightK`. -/
theorem candWeight_vb (Sk : Finset (PosK c hU nv)) (ρ : ℝ) (pos : PosK c hU nv) :
    candWeightZC (L := Lv) execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀)
      (tabZAt zo.dj zo.y₀) zo.vb Sk ρ pos = candWeightK c hU nv Lv Sk ρ pos := by
  unfold candWeightZC candWeightK
  split_ifs with h1 h2 h2
  · exact congrArg (fun x : ℕ => ENNReal.ofReal ((x : ℝ) / Fintype.card Lv.Ω / (1 - ρ)))
      (Fintype.card_congr (Equiv.subtypeEquivRight fun ω => by
        unfold ReadsDrawZ ReadsZ ReadsDrawK
        simp only [vb_pos]
        exact Iff.rfl))
  · exact absurd h1 h2
  · exact absurd h2 h1
  · rfl

/-- **A stage's transfer at one history**, in `htr`'s shape with `obK`: `Pr[accept ∧ a drawn unit reads a wire of Sk
whose layer bit is not wv's] ≤ stageSlack + Σ_pos candWeightK pos · Pr[the trial for pos opens pos's registered digest
to a value other than wv pos]`; `zk_session_candValsR` with `candTermZC_le`. -/
theorem candTransfer {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (zo : ZkOuter mPts nv Reg' c hU hscope (Law.execOS (n := nv) src (.stratified Kd (Array.ofFn St)) pop))
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg') zo.dj zo.y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (τ : Strategy (zo.game dg N)) {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512
      (planZAt zo.dj zo.y₀ ((Law.execOS (n := nv) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (zo.sim dg N τ))) j (zcStrat execArith hs512 (planZAt zo.dj zo.y₀) (cont (zo.sim dg N τ) ω)) qF qS)
    (Sk : Finset (PosK c hU nv)) (wv : PosK c hU nv → ValK c) :
    prob (fun o => LiveAcceptsZKR zo.I zo.own o ∧
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u,
          CandBadZ execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀)
            (tabZAt zo.dj zo.y₀) zo.vb Sk wv (zo.layer dg N k Rw τ) g) (zo.game dg N) τ ≤
      zo.stageSlack dg N k Rw M ρ t' qF qS τ +
        ∑ pos, candWeightK c hU nv (Law.execOS (n := nv) src (.stratified Kd (Array.ofFn St)) pop) Sk ρ pos *
          prob (OffAt (obK c hU nv) (zo.vb.cs (reg (zo.sim dg N τ)) pos) (wv pos)) (zo.trial dg N k Rw τ pos).g
            (zo.trial dg N k Rw τ pos).s := by
  refine ((zk_session_candValsR zo.I zo.ht zo.own zo.dj zo.y₀ zo.wr dg N zo.hc hU zo.hpub hscope St Kd pop Lb src
    zo.hDraw zo.hTags zo.hRec hk hRw hM hρ hr ht' Sk wv τ hCR).trans (add_le_add le_rfl
      (candTermZC_le execArith hs512 _ _ _ zo.vb Sk wv _ _ k Rw hρ))).trans_eq ?_
  rw [← add_assoc]
  refine congrArg _ (Finset.sum_congr rfl fun pos _ => ?_)
  rw [candWeight_vb zo Sk ρ pos]
  rfl

end ZkOuter

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {mPts nv : ℕ} {Reg' : Type} {c : Flock.Circuit}
  {hU : Built c.unit} {hscope : Layout.scopeOk c = true} {Lv : Law nv}

/-- **V*'s public file registers the stages' commitments**: at every history `t`, where `t`'s stage-`i` commitment is
`e.1`, the digest V*'s value binding registers at each commit string of `Sk i` (`cs`, the public file's leaf there) is
`cmt e.1`'s. -/
def VCommits (zs : Reg → L.Ω → List (Cm × I.Coin) → ZkOuter mPts nv Reg' c hU hscope Lv)
    (Sk : ℕ → Finset (PosK c hU nv)) (cmt : Cm → PosK c hU nv → (obK c hU nv).Dg) : Prop :=
  ∀ i R ω t (e : Cm × I.Coin), t[i]? = some e → ∀ pos ∈ Sk i, ∀ R' : Reg', (zs R ω t).vb.cs R' pos = cmt e.1 pos

/-- **What V*'s circuit reads of each stage**: at every history and strategy, at every outcome flock-verify `--zk
--registered` accepts, when the inner verifier would read the round-`i` message of V*'s layer differently from
`msg i v` (`same i`), a unit of the outcome's draw reads a wire of a commit string of `Sk i` where the layer's bit is
not `v`'s (`CandBadZ`). -/
def VReads (zs : Reg → L.Ω → List (Cm × I.Coin) → ZkOuter mPts nv Reg' c hU hscope Lv) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (wmsg : (Fin (Layout.keyN c hU nv) → Bool) → List I.Msg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop) (Sk : ℕ → Finset (PosK c hU nv))
    (msg : ℕ → (PosK c hU nv → ValK c) → I.Msg) : Prop :=
  ∀ i < I.r, ∀ R ω t (τ : Strategy ((zs R ω t).game dg N)) (v : PosK c hU nv → ValK c) (R' : Reg') (ω' : Lv.Ω)
    (tr : Exec.Transcript), LiveAcceptsZKR (zs R ω t).I (zs R ω t).own (R', Lv.draw ω', tr) →
    ¬ Option.Rel (same i (t.map Prod.snd)) (wmsg ((zs R ω t).layer dg N k Rw τ))[i]? (some (msg i v)) →
    ∃ u ∈ Lv.draw ω', ∃ g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u,
      CandBadZ execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt (zs R ω t).dj (zs R ω t).y₀)
        (tabZAt (zs R ω t).dj (zs R ω t).y₀) (zs R ω t).vb (Sk i) v ((zs R ω t).layer dg N k Rw τ) g

/-- **The recursive audit with V*'s `--zk --registered` session as the outer proof is sound, with its stages' transfer
from V*'s session**: `zk_session_recursive_upto` at V*'s openings `obK`, stage `i`'s trial for each commit string the
session's (`ZkOuter.trial`), its slack `ZkOuter.stageSlack` and its weights `candWeightK` on `Sk i`, the transfer
`htr` from `ZkOuter.candTransfer` on `VCommits` and `VReads`. -/
theorem zk_session_recursive_stages
    -- V*'s law and A3
    {m : ℕ} (σs : Fin nv → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- V*'s session at each history
    (zs : Reg → L.Ω → List (Cm × I.Coin) →
      ZkOuter mPts nv Reg' c hU hscope (Law.execOS (n := nv) src (.stratified Kd (Array.ofFn St)) pop))
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t S R' j, rateZ (planZAt (Reg := Reg') (zs R ω t).dj (zs R ω t).y₀ S R') j k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t')
    -- the recursive prover, and cr/sha-512 for its outer simulation's finders at the leaves it reaches
    (σ : Strategy (recGame L Reg I Cm fun R ω t => (zs R ω t).game dg N))
    {qF qS qW : ℝ} (hCRo : AllAt σ fun ω t τ => (zs σ.1 ω t).CR dg N k Rw qF qS qW τ)
    -- the hidden circuit's statement, the messages V*'s layer holds, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : (Fin (Layout.keyN c hU nv) → Bool) → List I.Msg)
    (hbr : VBridge zs dg N k Rw x wmsg)
    -- what the inner verifier reads of each round's message
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    -- the stages: their commit strings, commitments and messages, V*'s registration and reading of them, and
    -- cr/sha-512 for the fork finders
    (Sk : ℕ → Finset (PosK c hU nv)) (cmt : Cm → PosK c hU nv → (obK c hU nv).Dg)
    (msg : ℕ → (PosK c hU nv → ValK c) → I.Msg) (hcmt : VCommits zs Sk cmt) (hrd : VReads zs dg N k Rw wmsg same Sk msg)
    (q : ℕ → ℝ)
    (hCR : ∀ i < I.r, ∀ ω pos,
      (forkAt (obK c hU nv) σ cmt (fun R ω t τ => (zs R ω t).trial dg N k Rw τ) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    prob (fun o => LiveAcceptsZKR (zs o.1 o.2.1 o.2.2.1).I (zs o.1 o.2.1 o.2.2.1).own o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => (zs R ω t).game dg N) σ ≤
      expAt σ (fun ω t τ => (zs σ.1 ω t).bound dg N (Law.stratified σs ks hks) k Rw M ρ t' qF qS qW τ) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (fun ω t τ => (zs σ.1 ω t).stageSlack dg N k Rw M ρ t' qF qS τ) +
            (∑ pos, candWeightK c hU nv (Law.execOS (n := nv) src (.stratified Kd (Array.ofFn St)) pop) (Sk i) ρ pos) *
              ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin :=
  zk_session_recursive_upto σs ks hks St hS Kd pop Lb src hA3 zs dg N hk hRw hM hρ hr ht' σ hCRo x wmsg hbr same hsame
    (obK c hU nv) cmt msg (fun _ R ω t τ => (zs R ω t).trial dg N k Rw τ)
    (fun _ R ω t τ => (zs R ω t).stageSlack dg N k Rw M ρ t' qF qS τ)
    (fun i => candWeightK c hU nv (Law.execOS (n := nv) src (.stratified Kd (Array.ofFn St)) pop) (Sk i) ρ) q
    (fun i hi ω => (hCRo ω).mono fun t s _ hs v e he => by
      refine le_trans (prob_auditLive_mono _ _ ?_)
        (((zs σ.1 ω t).candTransfer St Kd pop Lb src dg N hk hRw hM hρ (hr σ.1 ω t) ht' _ hs.1 (Sk i) v).trans
          (add_le_add le_rfl (Finset.sum_le_sum fun pos _ => ?_)))
      · intro R' ω' tr h
        exact ⟨h.1, hrd i hi σ.1 ω t _ v R' ω' tr h.1 h.2⟩
      · by_cases hp : pos ∈ Sk i
        · rw [hcmt i σ.1 ω t e he pos hp]
        · simp [candWeightK, hp])
    hCR hin

end FlockSoundness.Discharge.Recursion
