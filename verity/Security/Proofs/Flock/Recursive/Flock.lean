import Proofs.Flock.Recursive.Sound
import Proofs.Flock.Soundness.Discharge.ZkReg.Sound
import Proofs.Flock.Recursive.ZkReg.Vals

/-!
# Recursion: the outer proof a `--zk --registered` session of a fixed public verifier V*

`zk_session_recursive` is `recursive_sound` with the outer proof one `flock-verify verify --zk --registered F` session of
V*'s circuit `c` per history `(R, ω, t)` (the registration, the draw of the hidden circuit's units, the rounds'
commitments and coins), its inputs `ZkOuter` (public file, registered reads, the coin server's record), the verdict
`LiveAcceptsZKR`, and the decoded witness V*'s value layer (`XplurZC`, `ZkOuter.layer`). Outer soundness is
`zk_session_soundR` at one wrong unit and `zk_session_regValsR` at the history's registered-reads record, on their
hypotheses at each history and, for those that read the prover's strategy (`cr/sha-512` for the knowledge and
registered-value finders, `ZkOuter.CR`), at each leaf the prover reaches. What V*'s circuit must compute is one named
Prop, `VBridge`: at an accepted outcome, a value layer with no wrong unit whose drawn units read the registrant's values
at the registered reads (the hidden circuit `P`, the draw, the coins) holds an inner transcript the inner verifier accepts
on `P`'s statement at the rounds' coins. The stage hypotheses (`htr`, `hCR`) and `InnerSound` are `recursive_sound`'s;
`zk_session_recursive_upto` takes `recursive_sound_upto`'s instead.

The bound: `E[miss 1 + 2·(ksAvgStrictZ + linkBoundZC) + regTermZC] + Σ_{k < r} (E[εtr k] + (Σ_pos κ k pos)·√(q_k²/2^513))
+ ε_in`, the expectations over the draw and the rounds' coins at the prover's leaves (`expAt`). `ksAvgStrictZ +
linkBoundZC` is paid once by each of the two outer theorems.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

set_option genSizeOfSpec false in
set_option genInjectivity false in
/-- **V*'s `--zk --registered` session at one history**: what `zk_session_soundR` and `zk_session_regValsR` take besides
the circuit `c` (fixed: V* is public), the law `Lv` of V*'s units, the count curve and the prover's strategy; `rd` is the
registrant's record of the registered reads (the hidden circuit's rows from its registration, the draw and the coins
from the verifier's). -/
structure ZkOuter (mPts nv : ℕ) (Reg' : Type) (c : Flock.Circuit) (hU : Built c.unit) (hscope : Layout.scopeOk c = true)
    (Lv : Law nv) where
  I : Inputs
  ht : I.tags.typed = false
  own : Array Flock.Registered.Port
  dj : Finset (Fin nv) → Lean.Json
  S₀ : Finset (Fin nv)
  y₀ : DrawSetupZK I mPts S₀ (dj S₀)
  wr : Finset (Fin nv) → Reg' → LiveOut mPts → Exec.Transcript
  hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c
  pub : Public
  hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub
  hDraw : ∀ ω, DrawOkZK I dj mPts (Lv.draw ω)
  hTags : TagsOk I
  hRec : RecordsLiveZK I dj wr
  rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg') ht dj y₀ hc hU (pub := pub))
    (rowsSh ht dj y₀ hc hU hpub hscope)

namespace ZkOuter

variable {mPts nv : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv Reg' c hU hscope Lv) (dg : ByteArray → ByteArray) (N : ℕ)

/-- V*'s live session at the history. -/
abbrev game : Game (Reg' × Finset (Fin nv) × Exec.Transcript) := auditLive Lv Reg' (liveOf dg mPts N zo.wr)

/-- The prover's strategy in V*'s session, simulated in the compiled batched session (`simZ`). -/
noncomputable def sim (τ : Strategy (zo.game dg N)) :=
  simZ execArith hs512 _ zo.I zo.wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ zo.I zo.wr _ _ _ _
    (runsZC_live zo.I zo.ht zo.dj zo.y₀ zo.wr _ tablesAtZK_live zo.hTags zo.hRec zo.hDraw)) τ

/-- **V*'s value layer at the strategy `τ`**: the decoded witness (`XplurZC`), each wire's bit in its commit string's
plurality value. -/
noncomputable def layer (k Rw : ℕ) (τ : Strategy (zo.game dg N)) : Fin (Layout.keyN c hU nv) → Bool :=
  XplurZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀) (tabZAt zo.dj zo.y₀)
    ((rsZAtS zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope).bindingZ (rsZAtS_computes zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope))
    (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw

/-- **`cr/sha-512` at the strategy `τ`**: for the knowledge finders of its simulation (`zk_session_soundR`'s `hCR`) at
budgets `qF, qS`, and for its registered-value finders at `rd` (`zk_session_regValsR`'s `hW`) at budget `qW`. -/
def CR (k Rw : ℕ) (qF qS qW : ℝ) (τ : Strategy (zo.game dg N)) : Prop :=
  (∀ ω j, TableCRZ execArith hs512 (planZAt zo.dj zo.y₀ (Lv.draw ω) (reg (zo.sim dg N τ))) j
      (zcStrat execArith hs512 (planZAt zo.dj zo.y₀) (cont (zo.sim dg N τ) ω)) qF qS) ∧
    RegCRZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀) (tabZAt zo.dj zo.y₀)
      ((rsZAtS zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope).bindingZ (rsZAtS_computes zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope))
      (regOpenZ zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope zo.rd) (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw qW

/-- **The outer bound at the strategy `τ`**: `zk_session_soundR`'s at one wrong unit (`Lm` the model's law) plus
`zk_session_regValsR`'s. -/
noncomputable def bound (Lm : Law nv) (k Rw M : ℕ) (ρ t' qF qS qW : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  (Lm.miss 1 + ksAvgStrictZ (planZAt zo.dj zo.y₀) Lv k qF qS (reg (zo.sim dg N τ)) +
      linkBoundZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀)
        (tabZAt zo.dj zo.y₀)
        ((rsZAtS zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope).bindingZ (rsZAtS_computes zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope))
        k Rw M ρ t' (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ))) +
    (ksAvgStrictZ (planZAt zo.dj zo.y₀) Lv k qF qS (reg (zo.sim dg N τ)) +
      linkBoundZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀)
        (tabZAt zo.dj zo.y₀)
        ((rsZAtS zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope).bindingZ (rsZAtS_computes zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope))
        k Rw M ρ t' (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) +
      regTermZC execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt (Reg := Reg') zo.dj zo.y₀)
        (tabZAt zo.dj zo.y₀)
        ((rsZAtS zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope).bindingZ (rsZAtS_computes zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope))
        Lv k Rw ρ qW)

/-- **A layer agrees with the registrant at the outcome's drawn units**: every wire a drawn unit reads at a registered
read whose session leaf under the outcome's registration is the registrant's carries the registrant's bit. -/
def RegAgree (bits : Fin (Layout.keyN c hU nv) → Bool) (o : Reg' × Finset (Fin nv) × Exec.Transcript) : Prop :=
  ∀ u ∈ o.2.1, ∀ g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u,
    zo.rd.isRead ((laySh zo.ht zo.dj zo.y₀ zo.hc hU (pub := zo.pub)).pos g) →
    (laySh zo.ht zo.dj zo.y₀ zo.hc hU (pub := zo.pub)).regLeaf o.1 ((laySh zo.ht zo.dj zo.y₀ zo.hc hU (pub := zo.pub)).pos g) =
      zo.rd.leaf ((laySh zo.ht zo.dj zo.y₀ zo.hc hU (pub := zo.pub)).pos g) →
    bits g = (laySh zo.ht zo.dj zo.y₀ zo.hc hU (pub := zo.pub)).bit (zo.rd.wv ((laySh zo.ht zo.dj zo.y₀ zo.hc hU (pub := zo.pub)).pos g)) g

/-- **V*'s session at one history, at one strategy**: an accepted outcome at which V*'s value layer satisfies `Q`, when
every accepted outcome whose layer has no wrong unit and agrees with the registrant at its drawn units satisfies `Q`,
is at most `bound` likely to fail `Q`; `zk_session_soundR` at one wrong unit and `zk_session_regValsR` at `rd`. -/
theorem accepts_le {m : ℕ} (σs : Fin nv → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (zo : ZkOuter mPts nv Reg' c hU hscope (Law.execOS (n := nv) src (.stratified Kd (Array.ofFn St)) pop))
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg') zo.dj zo.y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    {qF qS qW : ℝ} (τ : Strategy (zo.game dg N)) (hCR : zo.CR dg N k Rw qF qS qW τ) (Q : Prop)
    (hQ : ∀ o, LiveAcceptsZKR zo.I zo.own o →
      ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).wrong (zo.layer dg N k Rw τ) = ∅ →
      zo.RegAgree (zo.layer dg N k Rw τ) o → Q) :
    prob (fun o => LiveAcceptsZKR zo.I zo.own o ∧ ¬ Q) (zo.game dg N) τ ≤
      zo.bound dg N (Law.stratified σs ks hks) k Rw M ρ t' qF qS qW τ := by
  have h1 := (zk_session_soundR zo.I zo.ht zo.own zo.dj zo.y₀ zo.wr dg N zo.hc hU zo.hpub hscope σs ks hks St hS Kd pop Lb src
    hA3 zo.hDraw zo.hTags zo.hRec hk hRw hM hρ hr ht' 1 τ hCR.1).1
  have h2 := zk_session_regValsR zo.I zo.ht zo.own zo.dj zo.y₀ zo.wr dg N zo.hc hU zo.hpub hscope St Kd pop Lb src zo.hDraw
    zo.hTags zo.hRec hk hRw hM hρ hr ht' zo.rd τ hCR.1 hCR.2
  refine (prob_mono (fun o h => ?_) _ _).trans ((prob_or_le _ _ _ _).trans (add_le_add h1 h2))
  obtain ⟨hacc, hnq⟩ := h
  by_cases hw : 1 ≤ (((keyProg c hU nv).partition (Layout.outsOf c hU nv)).wrong (zo.layer dg N k Rw τ)).card
  · exact Or.inl ⟨hacc, hw⟩
  · refine Or.inr ⟨hacc, ?_⟩
    by_contra hno
    refine hnq (hQ o hacc (Finset.card_eq_zero.1 (by omega)) fun u hu g hg hread hleaf => ?_)
    by_contra hne
    exact hno ⟨u, hu, g, hg, hread, hleaf, hne⟩

end ZkOuter

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {mPts nv : ℕ} {Reg' : Type} {c : Flock.Circuit}
  {hU : Built c.unit} {hscope : Layout.scopeOk c = true} {Lv : Law nv}

/-- **What V*'s circuit computes** (`k, Rw` the count curve's): at every history `(R, ω, t)` and strategy `τ` of V*'s
session there, every outcome flock-verify `--zk --registered` accepts at which V*'s value layer has no wrong unit and
agrees with the registrant at the outcome's drawn units holds, in the messages `wmsg` reads from the layer, an inner
transcript the inner verifier accepts on the hidden circuit's statement `x R ω` at the rounds' coins. -/
def VBridge (zs : Reg → L.Ω → List (Cm × I.Coin) → ZkOuter mPts nv Reg' c hU hscope Lv) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : (Fin (Layout.keyN c hU nv) → Bool) → List I.Msg) : Prop :=
  ∀ R ω t (τ : Strategy ((zs R ω t).game dg N)) o, LiveAcceptsZKR (zs R ω t).I (zs R ω t).own o →
    ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).wrong ((zs R ω t).layer dg N k Rw τ) = ∅ →
    (zs R ω t).RegAgree ((zs R ω t).layer dg N k Rw τ) o →
    I.accepts (x R ω) ((wmsg ((zs R ω t).layer dg N k Rw τ)).zip (t.map Prod.snd))

/-- **The recursive audit with V*'s `--zk --registered` session as the outer proof is sound, up to what the inner verifier
reads**: `recursive_sound_upto` at V*'s session, as `zk_session_recursive` is `recursive_sound` there. Stage `i` opens to
`msg i v`, and its transfer bounds the message V*'s layer holds for round `i` not being `same i` as that. -/
theorem zk_session_recursive_upto
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
    -- the stages: commitments' openings, transfer, and cr/sha-512 for the fork finders
    {V Pos : Type} [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos] (ob : Openings H512 V)
    (cmt : Cm → Pos → ob.Dg) (msg : ℕ → (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy ((zs R ω t).game dg N) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy ((zs R ω t).game dg N) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    (htr : ∀ i < I.r, AllAt σ fun ω t τ => ∀ (v : Pos → V) (e : Cm × I.Coin), t[i]? = some e →
      prob (fun b => LiveAcceptsZKR (zs σ.1 ω t).I (zs σ.1 ω t).own b ∧ ¬ Option.Rel (same i (t.map Prod.snd))
          (wmsg ((zs σ.1 ω t).layer dg N k Rw τ))[i]? (some (msg i v))) ((zs σ.1 ω t).game dg N) τ ≤
        εtr i σ.1 ω t τ +
          ∑ pos, κ i pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T i σ.1 ω t τ pos).g (T i σ.1 ω t τ pos).s)
    (hCR : ∀ i < I.r, ∀ ω pos, (forkAt ob σ cmt (T i) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    prob (fun o => LiveAcceptsZKR (zs o.1 o.2.1 o.2.2.1).I (zs o.1 o.2.1 o.2.2.1).own o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => (zs R ω t).game dg N) σ ≤
      expAt σ (fun ω t τ => (zs σ.1 ω t).bound dg N (Law.stratified σs ks hks) k Rw M ρ t' qF qS qW τ) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (εtr i σ.1) + (∑ pos, κ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  have h := recursive_sound_upto ob σ (fun R ω t o => LiveAcceptsZKR (zs R ω t).I (zs R ω t).own o) x
    (W := (Fin (Layout.keyN c hU nv) → Bool) × I.Stmt) (fun R ω t τ _ => ((zs R ω t).layer dg N k Rw τ, x R ω))
    Prod.snd (fun w => wmsg w.1) same hsame cmt msg T εtr κ q
    (εo := expAt σ fun ω t τ => (zs σ.1 ω t).bound dg N (Law.stratified σs ks hks) k Rw M ρ t' qF qS qW τ) (εP := 0)
    ?_ ?_ htr hCR hin
  · rwa [add_zero] at h
  · exact avg_mono fun ω => leafAvg_mono_all ((hCRo ω).mono fun t s _ hs =>
      ZkOuter.accepts_le σs ks hks St hS Kd pop Lb src hA3 (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht' _ hs _
        (hbr σ.1 ω t _))
  · refine le_of_eq ((congrArg avg (funext fun ω => ?_)).trans (avg_const 0))
    refine (leafAvg_congr_len (G := fun _ _ => 0) (fun t s _ => ?_) _).trans (leafAvg_const 0 _)
    exact le_antisymm ((prob_mono (fun b hb => hb.2 rfl) _ _).trans (prob_false _ _).le) zero_le

/-- **The recursive audit with V*'s `--zk --registered` session as the outer proof is sound.** The prover registers `R`,
the verifier draws `ω` by `L`, `I.r` rounds of a commitment and a live coin follow, and V*'s session at the history
(`zs R ω t`) ends the audit, which accepts when flock-verify `--zk --registered` accepts V*'s session. On
`zk_session_soundR`'s and `zk_session_regValsR`'s hypotheses at every history (and `cr/sha-512` for the outer
simulation's finders at the leaves the prover reaches, `hCRo`), V*'s bridge, the stages' transfer and `cr/sha-512` for
their fork finders, and `InnerSound`, the audit accepts while the hidden circuit's statement is false with probability at
most the expected outer bound, plus each stage's expected transfer slack and rewinding loss, plus `ε_in`. -/
theorem zk_session_recursive
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
    -- the stages: commitments' openings, transfer, and cr/sha-512 for the fork finders
    {V Pos : Type} [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos] (ob : Openings H512 V)
    (cmt : Cm → Pos → ob.Dg) (msg : (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy ((zs R ω t).game dg N) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy ((zs R ω t).game dg N) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    (htr : ∀ i < I.r, AllAt σ fun ω t τ => ∀ (v : Pos → V) (e : Cm × I.Coin), t[i]? = some e →
      prob (fun b => LiveAcceptsZKR (zs σ.1 ω t).I (zs σ.1 ω t).own b ∧
          (wmsg ((zs σ.1 ω t).layer dg N k Rw τ))[i]? ≠ some (msg v)) ((zs σ.1 ω t).game dg N) τ ≤
        εtr i σ.1 ω t τ +
          ∑ pos, κ i pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T i σ.1 ω t τ pos).g (T i σ.1 ω t τ pos).s)
    (hCR : ∀ i < I.r, ∀ ω pos, (forkAt ob σ cmt (T i) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    prob (fun o => LiveAcceptsZKR (zs o.1 o.2.1 o.2.2.1).I (zs o.1 o.2.1 o.2.2.1).own o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => (zs R ω t).game dg N) σ ≤
      expAt σ (fun ω t τ => (zs σ.1 ω t).bound dg N (Law.stratified σs ks hks) k Rw M ρ t' qF qS qW τ) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (εtr i σ.1) + (∑ pos, κ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin :=
  zk_session_recursive_upto σs ks hks St hS Kd pop Lb src hA3 zs dg N hk hRw hM hρ hr ht' σ hCRo x wmsg hbr
    (fun _ _ a b => a = b) I.accepts_of_rel_eq ob cmt (fun _ => msg) T εtr κ q
    (fun i hi => by simpa only [rel_eq_iff, ne_eq] using htr i hi) hCR hin

end FlockSoundness.Discharge.Recursion
