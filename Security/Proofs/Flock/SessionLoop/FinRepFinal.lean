import Proofs.Flock.SessionLoop.FinRepView
import Proofs.Flock.SessionLoop.RepSrc
import Proofs.Flock.Claims.ZK.FinishSwTau
import Proofs.Flock.Claims.ZK.Blind

/-!
# (c.3)'s `FinRepFinal`, in parts

`Loop.FinRepFinal` asks that where the session's rep on the loop's table and request and the run from the views return
proofs equal up to `eraseZ`, their final-only data agree (`Masked.FinalEq`): level 0's opened rows and salts, `τ`, its
salt, and the pads' opened columns and salts. `Loop.FinRepAt P` is that context with `P` of the two proofs in place of
`FinalEq`, so each field is a part, each named here, none a premise of a guarantee:

* **`Loop.FinRepL0`**: level 0's opened rows and salts agree.
* **`Loop.FinRepTau`**: `τ` agrees.
* **`Loop.FinRepTauSalt`**: `τ`'s salt agrees. It holds in the masked scope where the frame draws the tape's pads
  (`Loop.finRepTauSalt`, from `Loop.RepSrc`): each rep fixes its salt in advance (`Finish.repFwC_map_tauSalt`,
  `FinRep.repCut_map_tauSalt`), the views' is the tape's (`Loop.finSrc_tauSalt`: the switched prover moves the pads
  alone, `Coined.swExe_ytau`), and the firewall's is the request's draws', which are the tape's
  (`Masked.pads_frame_eq`).
* **`Loop.FinRepOpen`**: the pads' opened columns and salts agree.

**`Loop.finRepFinal_of`**: the four give `FinRepFinal`.
-/

namespace Flock.Firewall.Finish

section

variable {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The firewall's rep proves under its pads' `τ` salt**, whatever it returns. -/
theorem repFwC_map_tauSalt (io : FlockComplete.Asks M) (hs : Hashes M) (st : Setup) (sch : Flock.Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (g : Array F128) (cm0 : FlockComplete.Commit) (extras : Array Claim)
    (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) :
    (fun z => z.inner.tauSalt) <$> repFwC io hs st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt =
      (fun _ => pads.tauSalt) <$> repFwC io hs st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt := by
  unfold repFwC
  simp only [map_bind, bind_pure_comp, Functor.map_map, Claims.ZKMal.Masked.FinRep.map_ite',
    Claims.ZKMal.Masked.FinRep.innerP_map_tauSalt]

end

end Flock.Firewall.Finish

namespace FlockSoundness.ZK.Table

section

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)

theorem toBase_ytau (v : T.onVals.View) : (T.toBase v).ytau = v.ytau := rfl

theorem toBase_ch (v : T.onVals.View) : (T.toBase v).ch = v.ch := rfl

theorem toBase_cm (v : T.onVals.View) : (T.toBase v).cm = v.cm := rfl

theorem toBase_saltsP (v : T.onVals.View) : (T.toBase v).saltsP = v.saltsP := rfl

end

section

variable {F K W Salt B D Hi VL C : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2]
  {T : Table F K W Salt (B × Salt) D Hi VL} (M : T.Hm96 C)

/-- **A real view's `y_τ` reads the seeds alone**, whatever the witness. -/
theorem Hm96.viewR_ytau (p q : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    (M.viewR p ω ys).ytau = (M.viewR q ω ys).ytau := by
  unfold FlockSoundness.ZK.Table.Hm96.viewR FlockSoundness.ZK.Table.viewWith FlockSoundness.ZK.Table.view
  try dsimp only
  try rfl

/-- **Moving the pads leaves a real view's `y_τ`**: `Table.shift` moves the pads alone. -/
theorem Hm96.viewR_shift_ytau (p q r : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    (M.viewR q (T.shift p q ω) ys).ytau = (M.viewR r ω ys).ytau := by
  unfold FlockSoundness.ZK.Table.Hm96.viewR FlockSoundness.ZK.Table.viewWith FlockSoundness.ZK.Table.view
    FlockSoundness.ZK.Table.shift
  try dsimp only
  try rfl

/-- **A real view's pads columns and their salts read the randomness alone**, whatever the witness. -/
theorem Hm96.viewR_padCols (p q : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    (M.viewR p ω ys).ch = (M.viewR q ω ys).ch ∧ (M.viewR p ω ys).cm = (M.viewR q ω ys).cm ∧
      (M.viewR p ω ys).saltsP = (M.viewR q ω ys).saltsP := by
  refine ⟨?_, ?_, ?_⟩
  all_goals
    unfold FlockSoundness.ZK.Table.Hm96.viewR FlockSoundness.ZK.Table.viewWith FlockSoundness.ZK.Table.view
    try dsimp only
    try rfl

end

end FlockSoundness.ZK.Table

/-- **A switched view's `y_τ` is the honest view's**: the switched prover moves the pads alone (`Table.shift`), and
`y_τ` reads the seeds (`Hm96.viewR_ytau`, `Hm96.viewR_shift_ytau`). -/
theorem FlockSoundness.Discharge.ZkView.Coined.swExe_ytau {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (ω : T.tab.Rand) (ys : T.tab.Hid → FlockSoundness.Discharge.ZkView.Salt) :
    (T.swExe w v ω ys).ytau = (T.viewExe w ω ys).ytau := by
  have e1 := congrArg (fun V => (T.tab.toBase V).ytau)
    (FlockSoundness.ZK.Table.Hm96.swViewR_eq T.hm96.onVals (w, v) (w, T.tab.vals) ω ys)
  have e2 := congrArg (fun V : T.tab.View => V.ytau)
    ((T.hm96.toBase_viewR w ω ys).trans (FlockSoundness.Discharge.ZkView.Coined.viewExe_eq T w ω ys).symm)
  refine e1.trans (Eq.trans ?_ e2)
  beta_reduce
  split_ifs
  · exact (T.tab.toBase_ytau _).trans
      ((FlockSoundness.ZK.Table.Hm96.viewR_ytau T.hm96.onVals (w, v) (w, T.tab.vals) ω ys).trans
        (T.tab.toBase_ytau _).symm)
  · exact (T.tab.toBase_ytau _).trans
      ((FlockSoundness.ZK.Table.Hm96.viewR_shift_ytau T.hm96.onVals (w, v) (w, T.tab.vals) (w, T.tab.vals) ω ys).trans
        (T.tab.toBase_ytau _).symm)

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- A rep run whose result is a proof proves under the salt its every run proves under. -/
theorem tauSalt_of_map1 {m : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.ZkProof} {b : ByteArray}
    (hm : (fun z => z.inner.tauSalt) <$> m = (fun _ => b) <$> m) {s : _root_.Flock.Transcript} {l : Array Msg}
    {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript} (h : ((m.run s).go l).1 = .ok (p, t)) :
    p.inner.tauSalt = b :=
  Masked.FinRep.tauSalt_of_map hm (l' := ((m.run s).go l).2) (Prod.ext h rfl)

open FlockSoundness.Discharge in
/-- **The views' `τ` salt is the tape's pads'**: table `i`'s rep `r` source from the switched views reads `y_τ` of the
switched view, which is the honest view's (`Coined.swExe_ytau`), the honest pads' `τ` salt (`Masked.ytau_saltBytes`). -/
theorem finSrc_tauSalt (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) {i r : ℕ} (hi : i < Masked.jOf x) (hr : r < 2) :
    ((finSrc vs x w ν t ops i).rep r).tauSalt = (Masked.innerPadsE x t ⟨i, hi⟩ ⟨r, hr⟩).tauSalt := by
  unfold finSrc Masked.tabSrcView
  rw [dite_eq_left hi]
  dsimp only
  rw [dite_eq_left hr]
  try dsimp only
  simp only [RefSession.swViewsAt]
  rw [ZkView.Coined.swExe_ytau]
  first
  | exact Masked.ytau_saltBytes x (finCoins x ops) w t ⟨i, hi⟩ ⟨r, hr⟩
  | (have hc : i < Claims.Flock.zkTableCap := lt_of_lt_of_le hi (Masked.jOf_le_cap x)
     show ZkView.saltBytes ((t ⟨i, hc⟩).2.1.2.2 ⟨r, hr⟩) =
       FlockComplete.fitBytes 192
         (Masked.fAt ByteArray.empty (fun r => ZkView.saltBytes ((Masked.cellAt t i).2.1.2.2 r)) r)
     rw [Masked.cellAt_lt t hc, Masked.fAt_lt _ _ hr, Masked.fitBytes_of_size (Masked.saltBytes_size _)])

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **`FinRepFinal`'s context at a property `P` of the two proofs**: at (c.3)'s context, where the session's rep returns
`(p, tr)` and the run from the views returns a proof `p'` equal to `p` up to `eraseZ`, `P p' p`. -/
def FinRepAt (P : _root_.Flock.Zk.ZkProof → _root_.Flock.Zk.ZkProof → Prop) : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs →
    ∀ cms, (∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l)) →
    ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
    ∀ ic l, (padsView vs x w ν t ops schs k).go l = (.ok ic, l) → l.size = Masked.repBase x k →
    ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
    ∀ ic' p tr, (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic' →
      (((_root_.Flock.Firewall.Finish.repFwC
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
          (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
          tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).1 =
        .ok (p, tr) →
      ∀ p' l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p', tr), l') →
        FlockComplete.eraseZ p' = FlockComplete.eraseZ p → P p' p

/-- **Level 0's opened rows and salts from the views are the session's.** -/
def FinRepL0 : Prop :=
  FinRepAt vs L₀ src real wk x w fun p' p =>
    p'.proof.ligerito.initialProof.rows = p.proof.ligerito.initialProof.rows ∧
      p'.proof.ligerito.initialProof.salts = p.proof.ligerito.initialProof.salts

/-- **`τ` from the views is the session's.** -/
def FinRepTau : Prop :=
  FinRepAt vs L₀ src real wk x w fun p' p => p'.inner.tau = p.inner.tau

/-- **`τ`'s salt from the views is the session's.** -/
def FinRepTauSalt : Prop :=
  FinRepAt vs L₀ src real wk x w fun p' p => p'.inner.tauSalt = p.inner.tauSalt

/-- **The pads' opened columns and salts from the views are the session's.** -/
def FinRepOpen : Prop :=
  FinRepAt vs L₀ src real wk x w fun p' p => p'.inner.opened = p.inner.opened ∧ p'.inner.salts = p.inner.salts

/-- **The parts give `FinRepFinal`**: each is one or two of `FinalEq`'s fields. -/
theorem finRepFinal_of (h : FinRepL0 vs L₀ src real wk x w ∧ FinRepTau vs L₀ src real wk x w ∧
    FinRepTauSalt vs L₀ src real wk x w ∧ FinRepOpen vs L₀ src real wk x w) :
    FinRepFinal vs L₀ src real wk x w := by
  obtain ⟨hL, hT, hS, hO⟩ := h
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl tb hi q f hR ic' p tr hic'
    hrun p' l' hv hz
  obtain ⟨h1, h2⟩ := hL ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl tb hi q f hR
    ic' p tr hic' hrun p' l' hv hz
  have h3 := hT ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl tb hi q f hR
    ic' p tr hic' hrun p' l' hv hz
  have h4 := hS ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl tb hi q f hR
    ic' p tr hic' hrun p' l' hv hz
  obtain ⟨h5, h6⟩ := hO ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl tb hi q f hR
    ic' p tr hic' hrun p' l' hv hz
  exact ⟨h1, h2, h3, h4, h5, h6⟩

/-- **`FinRepTauSalt` holds where the frame draws the tape's pads** (`Loop.RepSrc`), at a live statement: the run from
the views proves under the views' `τ` salt (`FinRep.repCut_map_tauSalt`), the tape's (`finSrc_tauSalt`), and the
firewall's rep under its pads' (`Finish.repFwC_map_tauSalt`), the request's draws', which are the tape's
(`Masked.pads_frame_eq`, at the draws `RepSrc` sizes and the layout the frame starts at,
`LeanRelease.lay_ofLoopLink`). -/
theorem finRepTauSalt (hx : Masked.Live x) (hS : Loop.RepSrc L₀ src x w) : FinRepTauSalt vs L₀ src real wk x w := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  intro ν t κ m₀ ρ₀ ops _ s0 spec hsx schs _ cms _ k hk extras _ ic l _ _ tb hi q f hR ic' p tr _ hrun p' l' hv _
  have e1 : p'.inner.tauSalt = ((finSrc vs x w ν t ops (k / 2)).rep (k % 2)).tauSalt :=
    Masked.FinRep.tauSalt_of_map (Masked.FinRep.repCut_map_tauSalt _ _ _ _ _ _ _ _ _ _ _ _) hv
  have e2 : p.inner.tauSalt = (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.tauSalt :=
    tauSalt_of_map1 (_root_.Flock.Firewall.Finish.repFwC_map_tauSalt _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hrun
  obtain ⟨-, -, hT, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    rw [if_pos hx]
  have hj : k / 2 < Masked.jOf x := by omega
  have hr : k % 2 < 2 := Nat.mod_lt _ (by decide)
  have hk2 : Loop.frepOf (k / 2) (k % 2) = k := by
    unfold Loop.frepOf
    omega
  obtain ⟨-, hq, -⟩ := hR
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, hst, hq⟩ := bok hq
  have hqd : q.draws = _root_.Flock.Firewall.Frame.Draws.ofBytes lay (L₀.draws (ν, t) k) := by
    cases hq
    rfl
  have hLk : (LeanRelease.ofLoopLink L₀ src real).lay x k = lay := by
    have e : Loop.startOf x k = _root_.Flock.Firewall.Drive.start (Masked.stsOf x)[k / 2]! := by
      unfold Loop.startOf
      rw [hsx]
      try dsimp only
    show ((Loop.startOf x k).toOption.map (·.1)).getD default = lay
    rw [e]
    first
    | (rw [hst]; all_goals rfl)
    | (simp only [hst]; all_goals rfl)
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := L₀) src real ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hst
  have hd := LeanRelease.draws_ofLoopLink (L₀ := L₀) src real
  have hz := hS.size (LeanRelease.ofLoopLink L₀ src real) hd ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hlay
  have hpads := hS.pads (LeanRelease.ofLoopLink L₀ src real) hd ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hlay
    (Masked.UE.tapeZk t)
  have hq' : q.draws = (LeanRelease.ofLoopLink L₀ src real).drawsAt x (ν, t) (Loop.frepOf (k / 2) (k % 2)) := by
    rw [hk2, hqd]
    show _ = _root_.Flock.Firewall.Frame.Draws.ofBytes ((LeanRelease.ofLoopLink L₀ src real).lay x k)
      (L₀.draws (ν, t) k)
    rw [hLk]
  have hP := Masked.pads_frame_eq (LeanRelease.ofLoopLink L₀ src real) (ν, t) (Masked.UE.tapeZk t) Loop.frepOf x t
    ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ q hq' hlay hz hpads
  rw [e1, e2, finSrc_tauSalt vs x w ν t ops hj hr, hP]
  all_goals rfl

end Loop

end Claims.ZKServer
