import Proofs.Flock.Recursive.Sound
import Proofs.Flock.Recursive.Multi
import Proofs.Flock.Soundness.Discharge.ZkReg.Sound
import Proofs.Flock.Recursive.ZkReg.Vals
import Proofs.Flock.Recursive.ZkReg.Off

/-!
# Recursion: the outer proof, `--zk --registered` sessions of fixed public verifiers V*

The outer proof at a history `(R, ω, t)` (the registration, the draw of the hidden circuit's units, the rounds'
commitments and coins) is `S` sessions in a row, session `j` one `flock-verify verify --zk --registered F` session of
the public circuit `(Vs j).c` under its law `(Vs j).law` (`VStmt`), with inputs `zs R ω t j` (`ZkOuter`: public file,
registered reads, the coin server's record). The audit accepts when every session accepts (`Accepts`), and the decoded
witness is each session's value layer (`XplurZC`, `ZkOuter.layer`).

* `ZkOuter.Good`: a layer with no wrong unit whose every unit reads, at each of its registered reads, the registrant's
  value (`RegAgree`). `rd` records those values, the hidden circuit's rows as registered and the values the verifier
  registers itself (its coins, the comb `coef`).
* `ZkOuter.accepts_le`: an accepted session whose layer is not `Good` is at most `bound` likely: the draw misses every
  bad unit (`miss 1`); or a drawn unit computes wrongly or reads a registered value other than the registrant's at a
  leaf that is the registrant's (`zk_session_regDrawnR`: `ksAvgStrictZ + linkBoundZC + regTermZC`); or a drawn row's
  registered leaf is not the registrant's while the verifier's registered check passes (`Reads.prob_offOn_le`:
  `qR²/2^513`). The units are bad at the strategy alone, so one sampling step covers all three.
* `sessions`: the `S` sessions in a row, each played by a strategy the prover names at its start (`seqC`, `ofBase`);
  every strategy of a session is named (`ZkOuter.ofBase_surj`), so no prover is lost.
* `VBridge`: what V*'s circuits compute: when every session accepts and every layer is `Good`, the messages `wmsg` reads
  from the layers hold an inner transcript the inner verifier accepts on the hidden circuit's statement `x R ω` at the
  rounds' coins.
* `zk_sessions_recursive_upto`: `recursive_sound_upto` at V*'s sessions, its outer term the expected sum over the
  sessions of each one's `boundCR` (`bound` where its finders satisfy `cr/sha-512`, `ZkOuter.CR`, else `1`) at the
  strategy the prover names there.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

set_option genSizeOfSpec false in
set_option genInjectivity false in
/-- **V*'s `--zk --registered` session at one history**: what its soundness takes besides the circuit `c` (fixed: V* is
public), the law `Lv` of V*'s units, the count curve and the prover's strategy. `rd` is the registrant's record of the
registered reads: the hidden circuit's rows from its registration, and the values the verifier registers itself (the
draw, the coins, the comb `coef`). `hown`: at an accepted outcome, the reads at the rows of the drawn units are the
verifier's own (`OwnReads`: its ports, at the positions its program reads). -/
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
  hown : ∀ ω R' o, LiveAcceptsZKR I own (R', Lv.draw ω, wr (Lv.draw ω) R' o) →
    OwnReads own c pub (wr (Lv.draw ω) R' o) (rd.restrict (rowIn (Lv.draw ω))).isRead rd.port rd.readAt

/-- **The names of a session's strategies**: a registration and, at each of the law's coins, a strategy of the live
table; they read the same at every history. -/
abbrev baseGame {nv : ℕ} (Lv : Law nv) (Reg' : Type) (dg : ByteArray → ByteArray) (mPts N : ℕ) : Game (LiveOut mPts) :=
  .send Reg' fun _ => .coin Lv.Ω fun _ => liveTable dg mPts N

namespace ZkOuter

variable {mPts nv : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv Reg' c hU hscope Lv) (dg : ByteArray → ByteArray) (N : ℕ)

/-- V*'s live session at the history. -/
abbrev game : Game (Reg' × Finset (Fin nv) × Exec.Transcript) := auditLive Lv Reg' (liveOf dg mPts N zo.wr)

/-- The registered-reads layout of V*'s session (`laySh`). -/
noncomputable abbrev lay := laySh (Reg := Reg') zo.ht zo.dj zo.y₀ zo.hc hU (pub := zo.pub)

/-- The value binding of V*'s session (`rsZAtS`'s). -/
noncomputable abbrev vb :=
  (rsZAtS (Reg := Reg') zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope).bindingZ
    (rsZAtS_computes zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope)

/-- The strategy a name stands for in V*'s session. -/
def ofBase (τ₀ : Strategy (baseGame Lv Reg' dg mPts N)) : Strategy (zo.game dg N) :=
  ⟨τ₀.1, fun ω => Strategy.map (fun t => (τ₀.1, Lv.draw ω, t)) _ (Strategy.map (zo.wr (Lv.draw ω) τ₀.1) _ (τ₀.2 ω))⟩

/-- **Every strategy of V*'s session has a name.** -/
theorem ofBase_surj (τ : Strategy (zo.game dg N)) : ∃ τ₀, zo.ofBase dg N τ₀ = τ := by
  obtain ⟨R, s⟩ := τ
  refine ⟨⟨R, fun ω => Strategy.ofMap (zo.wr (Lv.draw ω) R) (liveTable dg mPts N)
    (Strategy.ofMap (fun t => (R, Lv.draw ω, t)) (liveOf dg mPts N zo.wr (Lv.draw ω) R) (s ω))⟩,
    congrArg (Sigma.mk R) (funext fun ω => ?_)⟩
  have h1 := map_ofMap (zo.wr (Lv.draw ω) R) (liveTable dg mPts N)
    (Strategy.ofMap (fun t => (R, Lv.draw ω, t)) (liveOf dg mPts N zo.wr (Lv.draw ω) R) (s ω))
  have h2 := map_ofMap (fun t => (R, Lv.draw ω, t)) (liveOf dg mPts N zo.wr (Lv.draw ω) R) (s ω)
  show Strategy.map (fun t => (R, Lv.draw ω, t)) (liveOf dg mPts N zo.wr (Lv.draw ω) R)
    (Strategy.map (zo.wr (Lv.draw ω) R) (liveTable dg mPts N) (Strategy.ofMap (zo.wr (Lv.draw ω) R)
      (liveTable dg mPts N) (Strategy.ofMap (fun t => (R, Lv.draw ω, t)) (liveOf dg mPts N zo.wr (Lv.draw ω) R)
        (s ω)))) = s ω
  rw [h1, h2]

/-- The prover's strategy in V*'s session, simulated in the compiled batched session (`simZ`). -/
noncomputable def sim (τ : Strategy (zo.game dg N)) :=
  simZ execArith hs512 _ zo.I zo.wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ zo.I zo.wr _ _ _ _
    (runsZC_live zo.I zo.ht zo.dj zo.y₀ zo.wr _ tablesAtZK_live zo.hTags zo.hRec zo.hDraw)) τ

/-- The simulation registers what the prover sends. -/
theorem reg_sim (τ : Strategy (zo.game dg N)) : reg (zo.sim dg N τ) = τ.1 :=
  reg_cast (sessionZC_planOf execArith hs512 (tablesAtZK zo.dj zo.y₀)) _

/-- **V*'s value layer at the strategy `τ`**: the decoded witness (`XplurZC`), each wire's bit in its commit string's
plurality value. -/
noncomputable def layer (k Rw : ℕ) (τ : Strategy (zo.game dg N)) : Fin (Layout.keyN c hU nv) → Bool :=
  XplurZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀) (tabZAt zo.dj zo.y₀)
    zo.vb (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw

/-- **`cr/sha-512` for the knowledge finders** of the simulation at the strategy `τ` (`TableCRZ`), at budgets
`qF, qS`. -/
def TabCR (qF qS : ℝ) (τ : Strategy (zo.game dg N)) : Prop :=
  ∀ ω j, TableCRZ execArith hs512 (planZAt zo.dj zo.y₀ (Lv.draw ω) (reg (zo.sim dg N τ))) j
    (zcStrat execArith hs512 (planZAt zo.dj zo.y₀) (cont (zo.sim dg N τ) ω)) qF qS

/-- **`cr/sha-512` at the strategy `τ`**: for the knowledge finders of its simulation (`TabCR`) at budgets `qF, qS`,
for its registered-value finders at `rd` (`RegCRZC`) at budget `qW`, and for the finder that plays the session once and
outputs the extractor's pair of an off-leaf read at a drawn row (`Reads.pick`) at budget `qR`. -/
def CR (k Rw : ℕ) (qF qS qW qR : ℝ) (τ : Strategy (zo.game dg N)) : Prop :=
  zo.TabCR dg N qF qS τ ∧
    RegCRZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀) (tabZAt zo.dj zo.y₀)
      zo.vb (regOpenZ zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope zo.rd) (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw
      qW ∧
    (StrictCR.oneRun (zo.game dg N) τ fun o =>
      (zo.rd.restrict (rowIn o.2.1)).pick o.1 (regPaths c zo.pub o.2.2 zo.rd.port)).CR H512 qR

/-- **The outer bound at the strategy `τ`**: the model's miss at one bad unit (`Lm` the model's law), the drawn units'
`ksAvgStrictZ + linkBoundZC + regTermZC`, and the registered paths' `qR²/2^513`. -/
noncomputable def bound (Lm : Law nv) (k Rw M : ℕ) (ρ t' qF qS qW qR : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  Lm.miss 1 +
    (ksAvgStrictZ (planZAt zo.dj zo.y₀) Lv k qF qS (reg (zo.sim dg N τ)) +
      linkBoundZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀)
        (tabZAt zo.dj zo.y₀) zo.vb k Rw M ρ t' (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) +
      regTermZC execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt (Reg := Reg') zo.dj zo.y₀)
        (tabZAt zo.dj zo.y₀) zo.vb Lv k Rw ρ qW) +
    ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ))

open Classical in
/-- **The outer bound where the strategy's finders satisfy `cr/sha-512`**, else `1`. -/
noncomputable def boundCR (Lm : Law nv) (k Rw M : ℕ) (ρ t' qF qS qW qR : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  if zo.CR dg N k Rw qF qS qW qR τ then zo.bound dg N Lm k Rw M ρ t' qF qS qW qR τ else 1

/-- **A layer agrees with the registrant**: every wire a unit reads at a registered read carries the registrant's bit. -/
def RegAgree (bits : Fin (Layout.keyN c hU nv) → Bool) : Prop :=
  ∀ u, ∀ g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u, zo.rd.isRead (zo.lay.pos g) →
    bits g = zo.lay.bit (zo.rd.wv (zo.lay.pos g)) g

/-- **A good layer**: no wrong unit, and agreement with the registrant. -/
def Good (bits : Fin (Layout.keyN c hU nv) → Bool) : Prop :=
  ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).wrong bits = ∅ ∧ zo.RegAgree bits

open Classical in
/-- The units with a row at a registered read whose leaf under `R` is not the registrant's. -/
noncomputable def offUnits (R : Reg') : Finset (Fin nv) :=
  Finset.univ.filter fun u => ∃ p, zo.rd.isRead p ∧ rowIn {u} p ∧ zo.lay.regLeaf R p ≠ zo.rd.leaf p

/-- **The bad units at the strategy `τ`**: wrong in the layer, reading a registered-value disagreement at the
registrant's leaf, or holding an off-leaf registered read. -/
noncomputable def badUnits (k Rw : ℕ) (τ : Strategy (zo.game dg N)) : Finset (Fin nv) :=
  ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).wrong (zo.layer dg N k Rw τ) ∪
    regWrongZ execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZAt zo.dj zo.y₀) (tabZAt zo.dj zo.y₀)
      zo.vb (regOpenZ zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope zo.rd) (reg (zo.sim dg N τ)) (zo.layer dg N k Rw τ) ∪
    zo.offUnits τ.1

/-- **A layer that is not good has a bad unit.** -/
theorem badUnits_nonempty (k Rw : ℕ) (τ : Strategy (zo.game dg N)) (hG : ¬ zo.Good (zo.layer dg N k Rw τ)) :
    (zo.badUnits dg N k Rw τ).Nonempty := by
  classical
  by_cases hw : ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).wrong (zo.layer dg N k Rw τ) = ∅
  · obtain ⟨u, g, hg, hread, hne⟩ : ∃ u, ∃ g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u,
        zo.rd.isRead (zo.lay.pos g) ∧ zo.layer dg N k Rw τ g ≠ zo.lay.bit (zo.rd.wv (zo.lay.pos g)) g := by
      by_contra h
      refine hG ⟨hw, fun u g hg hread => ?_⟩
      by_contra hne
      exact h ⟨u, g, hg, hread, hne⟩
    by_cases hleaf : zo.lay.regLeaf τ.1 (zo.lay.pos g) = zo.rd.leaf (zo.lay.pos g)
    · refine ⟨u, Finset.mem_union_left _ (Finset.mem_union_right _
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, g, hg, ?_⟩))⟩
      exact regBadZ_of_leaf zo.ht zo.dj zo.y₀ zo.hc hU zo.hpub hscope zo.rd _ _ g hread
        (by rw [zo.reg_sim dg N τ]; exact hleaf) hne
    · obtain ⟨⟨u', q⟩, hp⟩ := Sum.isLeft_iff.1 (zo.rd.read_row _ hread)
      refine ⟨u', Finset.mem_union_right _
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, zo.lay.pos g, hread, ?_, hleaf⟩)⟩
      rw [hp]
      exact Finset.mem_singleton_self u'
  · exact (Finset.nonempty_iff_ne_empty.2 hw).mono (Finset.subset_union_left.trans Finset.subset_union_left)

/-- **An off-leaf registered read at a drawn row**, at an accepted outcome, is at most `qR²/2^513` likely, under
`cr/sha-512` for the finder that plays V*'s session once (`Reads.prob_offOn_le`): the verifier's registered check
holds at the drawn rows (`checked_of_accepts`, on `hown`). -/
theorem off_le {qR : ℝ} (τ : Strategy (zo.game dg N))
    (hR : (StrictCR.oneRun (zo.game dg N) τ fun o =>
      (zo.rd.restrict (rowIn o.2.1)).pick o.1 (regPaths c zo.pub o.2.2 zo.rd.port)).CR H512 qR) :
    prob (fun o => LiveAcceptsZKR zo.I zo.own o ∧ ∃ u ∈ o.2.1, u ∈ zo.offUnits τ.1) (zo.game dg N) τ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  classical
  have hmono : ∀ ω o', (LiveAcceptsZKR zo.I zo.own (τ.1, Lv.draw ω, zo.wr (Lv.draw ω) τ.1 o') ∧
      ∃ u ∈ Lv.draw ω, u ∈ zo.offUnits τ.1) →
      (LiveAcceptsZKR zo.I zo.own (τ.1, Lv.draw ω, zo.wr (Lv.draw ω) τ.1 o') ∧
        ∃ ω' o'', Lv.draw ω = Lv.draw ω' ∧ zo.wr (Lv.draw ω) τ.1 o' = zo.wr (Lv.draw ω') τ.1 o'') ∧
      (zo.rd.restrict (rowIn (Lv.draw ω))).Off τ.1 := by
    rintro ω o' ⟨hacc, u, hu, hoff⟩
    obtain ⟨p, hp, hrow, hne⟩ := (Finset.mem_filter.1 hoff).2
    exact ⟨⟨hacc, ω, o', rfl, rfl⟩, p, ⟨hp, rowIn_mono (Finset.singleton_subset_iff.2 hu) p hrow⟩, hne⟩
  refine (prob_auditLive_liveOf_mono dg N zo.wr τ
    (q := fun o => (LiveAcceptsZKR zo.I zo.own o ∧ ∃ ω o', o.2.1 = Lv.draw ω ∧ o.2.2 = zo.wr (Lv.draw ω) o.1 o') ∧
      (zo.rd.restrict (rowIn o.2.1)).Off o.1) hmono).trans
    (FlockSoundness.Registered.Reads.prob_offOn_le (zo.game dg N) τ (fun o => zo.rd.restrict (rowIn o.2.1))
      (fun o => o.1) (fun o => LiveAcceptsZKR zo.I zo.own o ∧ ∃ ω o', o.2.1 = Lv.draw ω ∧ o.2.2 = zo.wr (Lv.draw ω) o.1 o')
      (fun o => regPaths c zo.pub o.2.2 zo.rd.port) ?_ hR)
  rintro ⟨R', S', tr⟩ ⟨hacc, ω, o', hS', htr⟩
  obtain rfl : S' = Lv.draw ω := hS'
  obtain rfl : tr = zo.wr (Lv.draw ω) R' o' := htr
  exact checked_of_accepts zo.I zo.ht zo.own zo.dj zo.y₀ zo.hc hU zo.hpub hscope zo.hTags _ hacc _
    (zo.hown ω R' o' hacc)

/-- **V*'s session at one history, at one strategy**: an accepted outcome at which V*'s value layer is not `Good` is at
most `bound` likely. The bad units are fixed by the strategy: the draw misses them all (`Law.execOS_escape_le` under
A3, then `miss 1`), or meets a wrong unit or a registered-value disagreement at the registrant's leaf
(`zk_session_regDrawnR`), or an off-leaf registered read (`off_le`). -/
theorem accepts_le {m : ℕ} (σs : Fin nv → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (zo : ZkOuter mPts nv Reg' c hU hscope (Law.execOS (n := nv) src (.stratified Kd (Array.ofFn St)) pop))
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg') zo.dj zo.y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    {qF qS qW qR : ℝ} (τ : Strategy (zo.game dg N)) (hCR : zo.CR dg N k Rw qF qS qW qR τ) :
    prob (fun o => LiveAcceptsZKR zo.I zo.own o ∧ ¬ zo.Good (zo.layer dg N k Rw τ)) (zo.game dg N) τ ≤
      zo.bound dg N (Law.stratified σs ks hks) k Rw M ρ t' qF qS qW qR τ := by
  classical
  by_cases hG : zo.Good (zo.layer dg N k Rw τ)
  · exact (prob_mono (fun o h => h.2 hG) _ _).trans ((prob_false _ _).le.trans zero_le)
  have h1 : prob (fun o => Disjoint o.2.1 (zo.badUnits dg N k Rw τ)) (zo.game dg N) τ ≤
      (Law.stratified σs ks hks).miss 1 :=
    (prob_auditLive_draw _ τ fun S => Disjoint S (zo.badUnits dg N k Rw τ)).trans_le
      ((Law.execOS_escape_le σs ks hks St hS Kd pop Lb src hA3 _).trans
        (Law.escape_le_miss _ (Finset.card_pos.2 (zo.badUnits_nonempty dg N k Rw τ hG))))
  have h2 := zk_session_regDrawnR zo.I zo.ht zo.own zo.dj zo.y₀ zo.wr dg N zo.hc hU zo.hpub hscope St Kd pop Lb src
    zo.hDraw zo.hTags zo.hRec hk hRw hM hρ hr ht' zo.rd τ hCR.1 hCR.2.1
  have h3 := zo.off_le dg N τ hCR.2.2
  refine (prob_mono (fun o h => ?_) _ _).trans
    ((prob_or_le _ _ _ _).trans (add_le_add ((prob_or_le _ _ _ _).trans (add_le_add h1 h2)) h3))
  obtain ⟨hacc, -⟩ := h
  by_cases hd : Disjoint o.2.1 (zo.badUnits dg N k Rw τ)
  · exact Or.inl (Or.inl hd)
  · obtain ⟨u, hu, hub⟩ := Finset.not_disjoint_iff.1 hd
    rcases Finset.mem_union.1 hub with hub | hub
    · exact Or.inl (Or.inr ⟨hacc, Finset.not_disjoint_iff.2 ⟨u, hub, hu⟩⟩)
    · exact Or.inr ⟨hacc, u, hu, hub⟩

end ZkOuter

/-- **The law of a public verifier's units**: the live stratified law on a byte source (`live`), and the model's
stratified law it escapes no more often than (`model`), with the strata `σs`, their draw counts `ks` and the
executable's strata `St`. -/
structure VLaw (nv : ℕ) : Type 1 where
  m : ℕ
  σs : Fin nv → Fin m
  ks : Fin m → ℕ
  hks : ∀ s, ks s ≤ (Law.stratum σs s).card
  St : Fin m → Flock.Draw.Stratum
  Kd : ℕ
  pop : ℕ
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin m → ℕ
  src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256

attribute [instance] VLaw.fintypeΩ VLaw.nonemptyΩ

namespace VLaw

variable {nv : ℕ} (law : VLaw nv)

/-- The live stratified law on the source's bytes. -/
abbrev live : Law nv := Law.execOS (n := nv) law.src (.stratified law.Kd (Array.ofFn law.St)) law.pop

/-- The model's stratified law. -/
abbrev model : Law nv := Law.stratified law.σs law.ks law.hks

end VLaw

/-- **A public verifier**: its circuit `c` over `nv` units and its law. -/
structure VStmt : Type 1 where
  nv : ℕ
  mPts : ℕ
  c : Flock.Circuit
  hU : Built c.unit
  hscope : Layout.scopeOk c = true
  law : VLaw nv

namespace VStmt

/-- A session of the verifier's circuit at one history. -/
abbrev Outer (Q : VStmt) (Reg' : Type) := ZkOuter Q.mPts Q.nv Reg' Q.c Q.hU Q.hscope Q.law.live

/-- A value layer of the verifier's circuit. -/
abbrev Bits (Q : VStmt) : Type := Fin (Layout.keyN Q.c Q.hU Q.nv) → Bool

/-- A name for a strategy of a session of the verifier's circuit. -/
abbrev Name (Q : VStmt) (Reg' : Type) (dg : ByteArray → ByteArray) (N : ℕ) : Type :=
  Strategy (baseGame Q.law.live Reg' dg Q.mPts N)

end VStmt

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- The record of V*'s sessions at one history: each session's name and outcome. -/
abbrev VRec (Vs : Fin S → VStmt) (Reg' : Type) (dg : ByteArray → ByteArray) (N : ℕ) : Type :=
  (j : Fin S) → (Vs j).Name Reg' dg N × (Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript)

/-- **V*'s sessions at one history, in a row**: each played by the strategy the prover names at its start (`seqC`). -/
abbrev sessions (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ) :
    Game (VRec Vs Reg' dg N) :=
  seqC S (T := fun j => (Vs j).Name Reg' dg N) (α := fun j => Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript)
    (fun j => (zs j).game dg N) fun j => (zs j).ofBase dg N

/-- **Every session accepts.** -/
def Accepts (zs : (j : Fin S) → (Vs j).Outer Reg') {dg : ByteArray → ByteArray} {N : ℕ} (b : VRec Vs Reg' dg N) :
    Prop :=
  ∀ j, LiveAcceptsZKR (zs j).I (zs j).own (b j).2

/-- **V*'s layers** at the strategies the record names. -/
noncomputable def layers (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (b : VRec Vs Reg' dg N) : (j : Fin S) → (Vs j).Bits :=
  fun j => (zs j).layer dg N k Rw ((zs j).ofBase dg N (b j).1)

/-- **The expected sum over V*'s sessions** of `F j` at the strategy the prover names for session `j`. -/
noncomputable def sumS (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    (F : (j : Fin S) → Strategy ((zs j).game dg N) → ℝ≥0∞) (s : Strategy (sessions zs dg N)) : ℝ≥0∞ :=
  exG (fun rs => ∑ j, F j ((zs j).ofBase dg N (rs j).1)) (sessions zs dg N) s

/-- **V*'s sessions at one history**: when every outcome at which every session accepts and every layer is `Good`
satisfies `Q` of the layers, the sessions accept with `Q` failing at most the expected sum of the sessions' `boundCR`
(`accepts_le`, `prob_seqC_exists_le`). -/
theorem sessions_le
    (hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ j S' R' i, rateZ (planZAt (Reg := Reg') (zs j).dj (zs j).y₀ S' R') i k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (qF qS qW qR : ℝ) (Q : ((j : Fin S) → (Vs j).Bits) → Prop)
    (hQ : ∀ (τ : (j : Fin S) → Strategy ((zs j).game dg N))
      (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript),
      (∀ j, LiveAcceptsZKR (zs j).I (zs j).own (o j) ∧ (zs j).Good ((zs j).layer dg N k Rw (τ j))) →
      Q fun j => (zs j).layer dg N k Rw (τ j))
    (s : Strategy (sessions zs dg N)) :
    prob (fun b => Accepts zs b ∧ ¬ Q (layers zs dg N k Rw b)) (sessions zs dg N) s ≤
      sumS zs dg N (fun j τ => (zs j).boundCR dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s := by
  classical
  have hmono : ∀ b : VRec Vs Reg' dg N, Accepts zs b ∧ ¬ Q (layers zs dg N k Rw b) →
      ∃ j, LiveAcceptsZKR (zs j).I (zs j).own (b j).2 ∧
        ¬ (zs j).Good ((zs j).layer dg N k Rw ((zs j).ofBase dg N (b j).1)) := by
    rintro b ⟨hacc, hnq⟩
    by_contra hno
    refine hnq (hQ (fun j => (zs j).ofBase dg N (b j).1) (fun j => (b j).2) fun j => ⟨hacc j, ?_⟩)
    by_contra hg
    exact hno ⟨j, hacc j, hg⟩
  have hB : ∀ j τ₀, prob (fun a => LiveAcceptsZKR (zs j).I (zs j).own a ∧
        ¬ (zs j).Good ((zs j).layer dg N k Rw ((zs j).ofBase dg N τ₀))) ((zs j).game dg N) ((zs j).ofBase dg N τ₀) ≤
      (zs j).boundCR dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR ((zs j).ofBase dg N τ₀) := by
    intro j τ₀
    unfold ZkOuter.boundCR
    split_ifs with hc
    · exact ZkOuter.accepts_le (Vs j).law.σs (Vs j).law.ks (Vs j).law.hks (Vs j).law.St (hS j) (Vs j).law.Kd
        (Vs j).law.pop (Vs j).law.Lb (Vs j).law.src (hA3 j) (zs j) dg N hk hRw hM hρ (hr j) ht' _ hc
    · exact prob_le_one _ _ _
  exact (prob_mono hmono _ _).trans
    (prob_seqC_exists_le S _ _ (fun j r => LiveAcceptsZKR (zs j).I (zs j).own r.2 ∧
        ¬ (zs j).Good ((zs j).layer dg N k Rw ((zs j).ofBase dg N r.1)))
      (fun j τ₀ => (zs j).boundCR dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR ((zs j).ofBase dg N τ₀)) hB s)

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **What V*'s circuits compute** (`k, Rw` the count curve's): at every history `(R, ω, t)` and strategies `τ` of V*'s
sessions there, when every session's outcome flock-verify `--zk --registered` accepts and every session's value layer
is `Good`, the messages `wmsg` reads from the layers hold an inner transcript the inner verifier accepts on the hidden
circuit's statement `x R ω` at the rounds' coins. -/
def VBridge (zs : Reg → L.Ω → List (Cm × I.Coin) → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) : Prop :=
  ∀ R ω t (τ : (j : Fin S) → Strategy ((zs R ω t j).game dg N))
    (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript),
    (∀ j, LiveAcceptsZKR (zs R ω t j).I (zs R ω t j).own (o j) ∧
      (zs R ω t j).Good ((zs R ω t j).layer dg N k Rw (τ j))) →
    I.accepts (x R ω) ((wmsg fun j => (zs R ω t j).layer dg N k Rw (τ j)).zip (t.map Prod.snd))

/-- **The recursive audit with V*'s `--zk --registered` sessions as the outer proof is sound, up to what the inner
verifier reads**: `recursive_sound_upto` at V*'s sessions (`sessions_le` at every leaf, on `VBridge`). Stage `i` opens
to `msg i v`, and its transfer bounds the messages V*'s layers hold for round `i` not being `same i` as that. -/
theorem zk_sessions_recursive_upto
    -- V*'s laws: their strata and A3
    (hS : ∀ j, Law.ExecStrata (Vs j).law.σs (Vs j).law.ks (Vs j).law.St)
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history
    (zs : Reg → L.Ω → List (Cm × I.Coin) → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters, and the finders' budgets
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t j S' R' i, rateZ (planZAt (Reg := Reg') (zs R ω t j).dj (zs R ω t j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => sessions (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridge zs dg N k Rw x wmsg)
    -- what the inner verifier reads of each round's message
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    -- the stages: commitments' openings, transfer, and cr/sha-512 for the fork finders
    {V Pos : Type} [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos] (ob : Openings H512 V)
    (cmt : Cm → Pos → ob.Dg) (msg : ℕ → (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy (sessions (zs R ω t) dg N) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy (sessions (zs R ω t) dg N) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    (htr : ∀ i < I.r, AllAt σ fun ω t s => ∀ (v : Pos → V) (e : Cm × I.Coin), t[i]? = some e →
      prob (fun b => Accepts (zs σ.1 ω t) b ∧ ¬ Option.Rel (same i (t.map Prod.snd))
          (wmsg (layers (zs σ.1 ω t) dg N k Rw b))[i]? (some (msg i v))) (sessions (zs σ.1 ω t) dg N) s ≤
        εtr i σ.1 ω t s +
          ∑ pos, κ i pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T i σ.1 ω t s pos).g (T i σ.1 ω t s pos).s)
    (hCR : ∀ i < I.r, ∀ ω pos, (forkAt ob σ cmt (T i) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    prob (fun o => Accepts (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => sessions (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumS (zs σ.1 ω t) dg N
          (fun j τ => (zs σ.1 ω t j).boundCR dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (εtr i σ.1) + (∑ pos, κ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  have h := recursive_sound_upto ob σ (fun R ω t b => Accepts (zs R ω t) b) x
    (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt) (fun R ω t _ b => (layers (zs R ω t) dg N k Rw b, x R ω))
    Prod.snd (fun w => wmsg w.1) same hsame cmt msg T εtr κ q
    (εo := expAt σ fun ω t s => sumS (zs σ.1 ω t) dg N
      (fun j τ => (zs σ.1 ω t j).boundCR dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) (εP := 0)
    ?_ ?_ htr hCR hin
  · rwa [add_zero] at h
  · exact avg_mono fun ω => leafAvg_mono (fun t s => sessions_le hS hA3 (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht'
      qF qS qW qR (fun Xs => I.accepts (x σ.1 ω) ((wmsg Xs).zip (t.map Prod.snd))) (hbr σ.1 ω t) _) _
  · refine le_of_eq ((congrArg avg (funext fun ω => ?_)).trans (avg_const 0))
    refine (leafAvg_congr_len (G := fun _ _ => 0) (fun t s _ => ?_) _).trans (leafAvg_const 0 _)
    exact le_antisymm ((prob_mono (fun b hb => hb.2 rfl) _ _).trans (prob_false _ _).le) zero_le

end FlockSoundness.Discharge.Recursion
