import Proofs.Flock.SessionLoop.FinRepFinal
import Proofs.Flock.Claims.ZK.FinPads
import Proofs.Flock.Claims.ZK.FinishInner

/-!
# (c.3)'s `FinRepOpen`: the pads' opened columns and salts

`Loop.FinRepOpen` asks that the run from the views and the session's rep open the pads at the same columns and salts.
Both are answered by the opened coins (`asksCut`), so each opens its pads commitment at `padQs`, its last round's `Q`
coins mod the commitment's size (`FlockComplete.repCut_padsAt`, `Finish.repFwC_padsAt`). Both end at rep `k + 1`'s
first round (the views' by `Loop.FinRepRounds`, the session's as the firewall's run does, `Loop.finRepCut_size`), so the
queried positions are one, and at each the two commitments agree:

* the views' commitment holds the views' pads columns and salts (`FlockComplete.padsTreeCut_cols`). Where table `k / 2`'s
  rows hold at its base (`Loop.FinRows`), the switched view's are the honest view's (`Coined.swExe_padCols`,
  `Loop.finSrc_padCols`), and at a queried position those are the codes of the tape's pads and its fitted leaf salt
  (`Masked.srcV_padQs`);
* the session's commitment holds the codes of the request's pads (`innerCommitP_rows`), which are the tape's
  (`Masked.pads_frame_eq`), under the request's leaf salts, which are the tape's (`Loop.RepSrc.innerSalt`,
  `innerSalt_fit`).

**`Loop.finRepOpen`**: at a live statement where the frame draws the tape's pads (`Loop.RepSrc`), `FinRepOpen` holds
wherever every table's rows hold at its base (`Loop.FinRepRows`) and each rep stream's run from the views ends at the next
one's first round (`Loop.FinRepRounds`).
-/

namespace FlockComplete

open Flock

/-- **A pads commitment from a source holds the source's columns and salts.** -/
theorem padsTreeCut_cols {hs : Hashes PS} {n : ℕ} {S : RepSrc} {l l' : Array Claims.Msg} {ic : InnerCommit}
    (h : (padsTreeCut hs n S).go l = (.ok ic, l')) :
    ic.ch = (Array.range n).map S.ch ∧ ic.cm = (Array.range n).map S.cm ∧
      ic.salts = (Array.range n).map S.padSalt := by
  rw [padsTreeCut_leaf hs n (src := S) (src' := S) rfl, go_map'] at h
  have h1 := congrArg Prod.fst h
  dsimp only at h1
  generalize ((padsTreeCut hs n S).go l).1 = e at h1
  cases e with
  | error _ => first | cases h1 | simp [Except.map] at h1
  | ok a =>
    simp only [Except.map, Except.ok.injEq] at h1
    subst h1
    exact ⟨rfl, rfl, rfl⟩

/-- `RunPost` at a run whose value is known: it holds at the run's last history. -/
theorem RunPost.fst {α : Type} {m : TMW PS α} {P : α → Transcript → Array Claims.Msg → Prop} (h : RunPost m P)
    {s : Transcript} {l : Array Claims.Msg} {a : α} {t : Transcript} (e : ((m.run s).go l).1 = .ok (a, t)) :
    P a t ((m.run s).go l).2 :=
  h s l a t _ (Prod.ext e rfl)

end FlockComplete

namespace Flock.Firewall.Finish

/-- **The firewall's rep answered by `asksCut` opens its pads commitment at the queried positions**, as the session's
does (`FlockComplete.repPZk_padsAt`). -/
theorem repFwC_padsAt (coin : ℕ → ℕ → F128) (n : ℕ) (hs : Hashes FlockComplete.PS) (st : Setup)
    (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128) (cm0 : FlockComplete.Commit)
    (extras : Array Claim) (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) :
    FlockComplete.RunPost (repFwC (FlockComplete.asksCut coin n) hs st sch tPad rep dom g cm0 extras pads ic vals extra
      lanePad salt) fun p _ l => FlockComplete.PadsAt coin ic p.inner l := by
  unfold repFwC
  refine FlockComplete.RunPost.bind fun ⟨_, _⟩ => FlockComplete.RunPost.bind fun ⟨_, _⟩ =>
    FlockComplete.RunPost.bind fun ⟨_, _, _⟩ => FlockComplete.RunPost.bind fun ⟨_, _, _⟩ =>
      FlockComplete.RunPost.bind fun _ => FlockComplete.RunPost.bind fun ⟨_, _⟩ =>
        FlockComplete.RunPost.bind fun _ => ?_
  dsimp only
  refine FlockComplete.RunPost.ite (FlockComplete.RunPost.bind fun _ => ?_) ?_ <;>
    exact FlockComplete.RunPost.bind_pure
      (FlockComplete.RunPost.mono (FlockComplete.innerP_padsAt coin n hs _ pads ic _ _) fun _ _ _ h => h)

end Flock.Firewall.Finish

/-- **Where its rows hold at its base, a switched view's pads columns and salts are the honest view's**: it is the view
at the workers' values (`Hm96.swViewR_eq`), and the pads columns and salts read the pads and seeds alone. -/
theorem FlockSoundness.Discharge.ZkView.Coined.swExe_padCols {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (ω : T.tab.Rand) (ys : T.tab.Hid → FlockSoundness.Discharge.ZkView.Salt)
    (hr : T.tab.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1) :
    (T.swExe w v ω ys).ch = (T.viewExe w ω ys).ch ∧ (T.swExe w v ω ys).cm = (T.viewExe w ω ys).cm ∧
      (T.swExe w v ω ys).saltsP = (T.viewExe w ω ys).saltsP := by
  have e1 : T.swExe w v ω ys = T.tab.toBase (T.hm96.onVals.viewR (w, v) ω ys) :=
    congrArg T.tab.toBase
      ((FlockSoundness.ZK.Table.Hm96.swViewR_eq T.hm96.onVals (w, v) (w, T.tab.vals) ω ys).trans (ite_eq_left hr))
  have e2 : T.viewExe w ω ys = T.tab.toBase (T.hm96.onVals.viewR (w, T.tab.vals) ω ys) :=
    ((T.hm96.toBase_viewR w ω ys).trans (FlockSoundness.Discharge.ZkView.Coined.viewExe_eq T w ω ys).symm).symm
  obtain ⟨g1, g2, g3⟩ := FlockSoundness.ZK.Table.Hm96.viewR_padCols T.hm96.onVals (w, v) (w, T.tab.vals) ω ys
  have h1 : (T.tab.toBase (T.hm96.onVals.viewR (w, v) ω ys)).ch =
      (T.tab.toBase (T.hm96.onVals.viewR (w, T.tab.vals) ω ys)).ch :=
    (T.tab.toBase_ch _).trans (g1.trans (T.tab.toBase_ch _).symm)
  have h2 : (T.tab.toBase (T.hm96.onVals.viewR (w, v) ω ys)).cm =
      (T.tab.toBase (T.hm96.onVals.viewR (w, T.tab.vals) ω ys)).cm :=
    (T.tab.toBase_cm _).trans (g2.trans (T.tab.toBase_cm _).symm)
  have h3 : (T.tab.toBase (T.hm96.onVals.viewR (w, v) ω ys)).saltsP =
      (T.tab.toBase (T.hm96.onVals.viewR (w, T.tab.vals) ω ys)).saltsP :=
    (T.tab.toBase_saltsP _).trans (g3.trans (T.tab.toBase_saltsP _).symm)
  exact ⟨(congrArg (fun V : T.tab.View => V.ch) e1).trans (h1.trans (congrArg (fun V : T.tab.View => V.ch) e2).symm),
    (congrArg (fun V : T.tab.View => V.cm) e1).trans (h2.trans (congrArg (fun V : T.tab.View => V.cm) e2).symm),
    (congrArg (fun V : T.tab.View => V.saltsP) e1).trans
      (h3.trans (congrArg (fun V : T.tab.View => V.saltsP) e2).symm)⟩

namespace Claims.ZKMal.Masked

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge CoinLayer

attribute [local irreducible] Composed.treeEquiv

/-- **At a queried pads position the views read the tape's pads**: where the history's last round is table `i`'s rep
`r`'s query round (`innerOff 3`), each position `padQs` queries is below the pads tree's leaf count, and there table
`i`'s rep `r` source from the views reads the codes of `h ‖ pad_h` and `μ ‖ pad_μ` and the tape's pads salt fitted to
192 bytes (`srcV_pads`, at the opened position `coinedOf_sP_val` places there). -/
theorem srcV_padQs (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) {i r : ℕ}
    (hi : i < jOf x) (hr : r < 2) {L : Array Msg} (hslot : L.size - 1 = repSlot x (2 * i + r) (innerOff 3)) :
    ∀ j ∈ FlockComplete.padQs (fun k i => slotAt c k i) L
        (((layE x ⟨i, hi⟩).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE),
      j < ((layE x ⟨i, hi⟩).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE ∧
      ((srcV x c w τ i).rep r).ch j =
        _root_.Flock.Zk.codeAt ((FlockComplete.padsOf (UE.tapeZk τ) i r (layE x ⟨i, hi⟩)).h ++
          (FlockComplete.padsOf (UE.tapeZk τ) i r (layE x ⟨i, hi⟩)).padH) j ∧
      ((srcV x c w τ i).rep r).cm j =
        _root_.Flock.Zk.codeAt ((FlockComplete.padsOf (UE.tapeZk τ) i r (layE x ⟨i, hi⟩)).mu ++
          (FlockComplete.padsOf (UE.tapeZk τ) i r (layE x ⟨i, hi⟩)).padMu) j ∧
      ((srcV x c w τ i).rep r).padSalt j = FlockComplete.fitBytes 192 ((UE.tapeZk τ).innerSalt i r j) := by
  intro j hj
  set N := ((layE x ⟨i, hi⟩).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE
    with hN
  have hb := padN_bounds x ⟨i, hi⟩
  have hNp : Composed.padNOf (zkTab x ⟨i, hi⟩).table (stAt x ⟨i, hi⟩) = N := by
    rw [hN]
    rfl
  have hNpos : 0 < N := by
    rw [← hNp]
    exact lt_of_lt_of_le (by decide) hb.1
  simp only [FlockComplete.padQs, Array.map_map, Array.mem_map, Array.mem_range, Function.comp_apply] at hj
  obtain ⟨e, he, hj⟩ := hj
  rw [hslot] at hj
  have hjN : j < N := hj ▸ Nat.mod_lt _ hNpos
  have hjq : j ∈ (Composed.bodyOf (zkTab x ⟨i, hi⟩).table (stAt x ⟨i, hi⟩)).padQueries ⟨r, hr⟩
      (czExe PE x c ⟨i, hi⟩).rest := by
    show j ∈ (List.ofFn ((czExe PE x c ⟨i, hi⟩).rest.qP ⟨r, hr⟩)).map
      fun w => w.lo.toNat % Composed.padNOf (zkTab x ⟨i, hi⟩).table (stAt x ⟨i, hi⟩)
    rw [hNp]
    exact List.mem_map.2 ⟨_, List.mem_ofFn.2 ⟨⟨e, he⟩, rfl⟩, hj⟩
  have hlen : ((Composed.bodyOf (zkTab x ⟨i, hi⟩).table (stAt x ⟨i, hi⟩)).padQueries ⟨r, hr⟩
      (czExe PE x c ⟨i, hi⟩).rest).length ≤ _root_.Flock.Zk.Q := by
    show ((List.ofFn ((czExe PE x c ⟨i, hi⟩).rest.qP ⟨r, hr⟩)).map
      fun w => w.lo.toNat % Composed.padNOf (zkTab x ⟨i, hi⟩).table (stAt x ⟨i, hi⟩)).length ≤ _
    simp
  have hL : max (min (Composed.padNOf (zkTab x ⟨i, hi⟩).table (stAt x ⟨i, hi⟩)) (2 ^ 63)) _root_.Flock.Zk.Q = N := by
    rw [min_eq_left hb.2, max_eq_left hb.1, hNp]
  obtain ⟨a, ha⟩ := padPos_of_mem (L := max (min (Composed.padNOf (zkTab x ⟨i, hi⟩).table (stAt x ⟨i, hi⟩))
    (2 ^ 63)) _root_.Flock.Zk.Q) hlen hjq (by rw [hL]; exact hjN)
  have hsp : ((((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩) (.inl a)).val = j :=
    (Composed.coinedOf_sP_val PE _ _ _ hb.1 hb.2 _ _).trans ha
  obtain ⟨s1, s2, s3⟩ := srcV_pads x c w τ hi hr a
  rw [hsp] at s1 s2 s3
  exact ⟨hjN, s1, s2, s3⟩

/-- **A source reads its view's pads columns and salts**: views with the same pads columns and salts at table `i` give
sources with the same ones at each of its reps. -/
theorem tabSrcView_padCols (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (v v' : ∀ j, ((tabC P x j).T (cz j)).tab.View) {i r : ℕ} (hi : i < jOf x) (hr : r < 2)
    (h1 : (v ⟨i, hi⟩).ch = (v' ⟨i, hi⟩).ch) (h2 : (v ⟨i, hi⟩).cm = (v' ⟨i, hi⟩).cm)
    (h3 : (v ⟨i, hi⟩).saltsP = (v' ⟨i, hi⟩).saltsP) :
    ((tabSrcView P x cz v i).rep r).ch = ((tabSrcView P x cz v' i).rep r).ch ∧
      ((tabSrcView P x cz v i).rep r).cm = ((tabSrcView P x cz v' i).rep r).cm ∧
      ((tabSrcView P x cz v i).rep r).padSalt = ((tabSrcView P x cz v' i).rep r).padSalt := by
  unfold tabSrcView
  first
  | simp only [dite_eq_left hi, dite_eq_left hr, h1, h2, h3, and_self]
  | (rw [dite_eq_left hi, dite_eq_left hi]
     dsimp only
     rw [dite_eq_left hr, dite_eq_left hr]
     try dsimp only
     rw [h1, h2, h3]
     exact ⟨rfl, rfl, rfl⟩)

end Claims.ZKMal.Masked

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **Table `j`'s rows hold at its base**: at the opened coins, the workers' values `vs` satisfy table `j`'s rows at the
base the tape gives it, so its switched view is the view at those values (`Hm96.swViewR_eq`). -/
def FinRows (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape) (ops : List (Msg × Msg))
    (j : Fin (Masked.jOf x)) : Prop :=
  ((finS.C x j).T (finS.cz x (finCoins x ops) j)).tab.onVals.RowsAt (finS.wt x w j, vs x w ν (finCoins x ops) j)
    (finS.randAt x (finCoins x ops) t j).1.1 (finS.randAt x (finCoins x ops) t j).1.2.1.2.1

open FlockSoundness.Discharge in
/-- **Where its rows hold, the views' pads columns and salts are the honest views'**: table `i`'s rep `r` source from
the switched views reads the switched view's pads columns and salts (`Masked.tabSrcView_padCols`), which are the honest
view's (`Coined.swExe_padCols`). -/
theorem finSrc_padCols (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) {i r : ℕ} (hi : i < Masked.jOf x) (hr : r < 2) (hrows : FinRows vs x w ν t ops ⟨i, hi⟩) :
    ((finSrc vs x w ν t ops i).rep r).ch = ((Masked.srcV x (finCoins x ops) w t i).rep r).ch ∧
      ((finSrc vs x w ν t ops i).rep r).cm = ((Masked.srcV x (finCoins x ops) w t i).rep r).cm ∧
      ((finSrc vs x w ν t ops i).rep r).padSalt = ((Masked.srcV x (finCoins x ops) w t i).rep r).padSalt := by
  obtain ⟨h1, h2, h3⟩ := ZkView.Coined.swExe_padCols ((finS.C x ⟨i, hi⟩).T (finS.cz x (finCoins x ops) ⟨i, hi⟩))
    (finS.wt x w ⟨i, hi⟩) (vs x w ν (finCoins x ops) ⟨i, hi⟩) (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).1
    (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).2 hrows
  exact Masked.tabSrcView_padCols Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
    (finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t)
    (Masked.viewsE x (finCoins x ops) w t) hi hr h1 h2 h3

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **Every table's rows hold at its base** at a `Finish` `FinishIs` reads (`FinRows`). -/
def FinRepRows : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ j, FinRows vs x w ν t ops j

/-- **The session's rep ends at the next rep's first round**: in the masked scope, where the firewall's rep on rep
stream `k`'s table and request returns at its last coin, `repFwC` on the same inputs answered by the opened coins, from
a history at the rep's first round, ends at rep `k + 1`'s (`repFwC_br`, at the rep's `repRounds` coins). -/
theorem finRepCut_size (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) :
    ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ k < 2 * spec.tables.size, ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
    ∀ ic' p tr, (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic' →
      ((_root_.Flock.Firewall.Finish.repFw (_root_.Flock.Firewall.Finish.asksOf q.coins)
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
          (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0 (_root_.Flock.Firewall.Finish.claimsOf q.xs)
          (_root_.Flock.Firewall.Finish.padsOf q.draws) ic' q.vals tb.extra tb.lanePad
          (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).run 0 =
        .ok ((p, tr), q.coins.size) →
    ∀ l : Array Msg, l.size = Masked.repBase x k →
      (((_root_.Flock.Firewall.Finish.repFwC
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
          (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
          tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).2.size =
        Masked.repBase x (k + 1) := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk tb hi q f hR ic' p tr hic' hrun l hl
  obtain ⟨hend, hcoin⟩ := finRepCoins L₀ src real wk x w hsc ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk tb hi q f hR
  obtain ⟨hms, hps, hab, htw⟩ := finRepPads L₀ src real wk x w hsc ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk tb hi q f hR
  obtain ⟨h25, h35, hsch, hxs⟩ := finRepBounds (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hk hR
  obtain ⟨hn, hqs⟩ := FlockComplete.JsonRT.zkSchedule_small h25 h35 hsch
  rw [_root_.Flock.Firewall.Finish.repFw_eq (M := StateT ℕ V) _ _ tb.st hms tb.sch _ _ _ _ _ _ _ _ _ _ _ _ hps hab htw
    (fun e f => _root_.Flock.Firewall.Finish.throw_bind_sv e f)] at hrun
  have hb := (_root_.Flock.Firewall.Finish.repFwC_br q.coins (Masked.repBase x k) (refK x)
    (fun k i => Masked.slotAt (finCoins x ops) k i) hend hcoin tb.st tb.sch tb.tPad (k % 2) _ tb.g tb.cm0.toC _ _ _
    q.vals tb.extra tb.lanePad _ (by omega) hxs hn hqs _ 0 l (by omega) (p, tr) q.coins.size hrun).2
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  obtain ⟨-, hq, -⟩ := hR
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, -, hq⟩ := bok hq
  have hqc : q.coins =
      ((seenRounds x ((m₀, ρ₀) :: ops) (roundsOf spec (Masked.stsOf x)) k).map fun r => r.2.2).toArray := by
    cases hq
    rfl
  have hrk : (roundsOf spec (Masked.stsOf x)).getD k 0 = FlockDeploy.repRounds true spec (Masked.stsOf x)[k / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  have hsz : q.coins.size = FlockDeploy.repRounds true spec (Masked.stsOf x)[k / 2]! := by
    rw [hqc, List.size_toArray, List.length_map]
    simp only [seenRounds, List.length_map, List.length_range, hrk]
  rw [hsz] at hb
  rw [Masked.repBase_eq hsx (k + 1), FlockComplete.ZkTables.repStart_succ, ← Masked.repBase_eq hsx k]
  exact hb

/-- **`FinRepOpen` holds where the rows hold**: at a live statement where the frame draws the tape's pads
(`Loop.RepSrc`), wherever every table's rows hold at its base (`FinRepRows`) and each rep stream's run from the views
ends at the next one's first round (`FinRepRounds`), the views and the session's rep open the pads at the queried
positions (`repCut_padsAt`, `Finish.repFwC_padsAt`), one set since both runs end at rep `k + 1`'s first round
(`finRepCut_size`), and there the views' commitment (`padsTreeCut_cols`, `finSrc_padCols`, `Masked.srcV_padQs`) and the
session's (`innerCommitP_rows`, `innerCommitP_salts`, at the tape's pads `Masked.pads_frame_eq` and leaf salts
`RepSrc.innerSalt`) agree. -/
theorem finRepOpen (hx : Masked.Live x) (hS : Loop.RepSrc L₀ src x w) (hRows : FinRepRows vs L₀ src real wk x w)
    (hD : FinRepRounds vs L₀ src real wk x w) : FinRepOpen vs L₀ src real wk x w := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl tb hi q f hR ic' p tr hic'
    hrun p' l' hv _
  show p'.inner.opened = p.inner.opened ∧ p'.inner.salts = p.inner.salts
  obtain ⟨-, -, hT, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    rw [if_pos hx]
  have hj : k / 2 < Masked.jOf x := by omega
  have hr : k % 2 < 2 := Nat.mod_lt _ (by decide)
  have hk2 : Loop.frepOf (k / 2) (k % 2) = k := by
    unfold Loop.frepOf
    omega
  have hi' : k / 2 < (Masked.stsOf x).size := Masked.lt_of_jOf ⟨k / 2, hj⟩
  -- the two runs end at rep `k + 1`'s first round
  have hszV := (hD ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl p' tr l' hv).2
  obtain ⟨ic'', p'', tr'', hic'', hrun'', -⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hR.2.2
  have hii : ic'' = ic' := Except.ok.inj (hic''.symm.trans hic')
  rw [hii] at hrun''
  have hszF := finRepCut_size L₀ src real wk x w hx.1 ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk tb hi q f hR ic' p'' tr''
    hic' hrun'' l hl
  -- both open their pads commitments at `padQs`
  obtain ⟨hF1, hF2⟩ := FlockComplete.RunPost.fst
    (_root_.Flock.Firewall.Finish.repFwC_padsAt (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x)
      _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hrun
  have hv' := hv
  unfold viewRep at hv'
  obtain ⟨hV1, hV2⟩ := FlockComplete.repCut_padsAt (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hv'
  -- the layout, schedule and leaf count
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! = _root_.Flock.Setup.ofCircuit (Masked.stAt x ⟨k / 2, hj⟩) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) (by simpa [FlockDeploy.tabsOf] using hi')]
    simp [FlockDeploy.tabsOf, Masked.stAt]
  obtain ⟨schs', hs', hall⟩ := Masked.schedsPZk_live hx
  have hss : schs' = schs := Except.ok.inj (hs'.symm.trans hschs)
  have hsj : schs[k / 2]! = ((Masked.zkTab x ⟨k / 2, hj⟩).sch, (Masked.zkTab x ⟨k / 2, hj⟩).tPad) := by
    rw [← hss]
    exact hall ⟨k / 2, hj⟩
  obtain ⟨N, hN⟩ : ∃ N, N = ((Masked.layE x ⟨k / 2, hj⟩).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
      2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE := ⟨_, rfl⟩
  -- the views' commitment
  unfold padsView at hic
  rw [htab, hsj] at hic
  obtain ⟨hc1, hc2, hc3⟩ := FlockComplete.padsTreeCut_cols hic
  have hc1' : ic.ch = (Array.range N).map ((finSrc vs x w ν t ops (k / 2)).rep (k % 2)).ch := by
    rw [hc1, hN]
    all_goals rfl
  have hc2' : ic.cm = (Array.range N).map ((finSrc vs x w ν t ops (k / 2)).rep (k % 2)).cm := by
    rw [hc2, hN]
    all_goals rfl
  have hc3' : ic.salts = (Array.range N).map ((finSrc vs x w ν t ops (k / 2)).rep (k % 2)).padSalt := by
    rw [hc3, hN]
    all_goals rfl
  have hicN : ic.ch.size = N := by rw [hc1', Array.size_map, Array.size_range]
  obtain ⟨hpc1, hpc2, hpc3⟩ := finSrc_padCols vs x w ν t ops hj hr (hRows ν t κ m₀ ρ₀ ops hctx ⟨k / 2, hj⟩)
  -- the session's commitment, at the tape's pads
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
  have hP : (_root_.Flock.Firewall.Finish.padsOf q.draws).toC =
      FlockComplete.padsOf (Masked.UE.tapeZk t) (k / 2) (k % 2) (Masked.layE x ⟨k / 2, hj⟩) :=
    Masked.pads_frame_eq (LeanRelease.ofLoopLink L₀ src real) (ν, t) (Masked.UE.tapeZk t) Loop.frepOf x t
      ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ q hq' hlay hz hpads
  have hicC : (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (_root_.Flock.Firewall.Finish.padsOf q.draws).toC (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
      V FlockComplete.InnerCommit) = .ok ic'.toC := by
    have e := _root_.Flock.Firewall.Finish.innerCommitP_eq (M := V)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) (_root_.Flock.Firewall.Finish.padsOf q.draws)
      (_root_.Flock.Firewall.Finish.saltAt q.draws.salts)
    rw [hic'] at e
    exact e.symm
  obtain ⟨hch, hcm⟩ := FlockComplete.Replay.innerCommitP_rows hicC
  have hsl := FlockComplete.innerCommitP_salts hicC
  have hpdH : ((FlockComplete.padsOf (Masked.UE.tapeZk t) (k / 2) (k % 2) (Masked.layE x ⟨k / 2, hj⟩)).h ++
      (FlockComplete.padsOf (Masked.UE.tapeZk t) (k / 2) (k % 2) (Masked.layE x ⟨k / 2, hj⟩)).padH).size =
        (Masked.layE x ⟨k / 2, hj⟩).len + _root_.Flock.Zk.K_PAD := by
    rw [Array.size_append, FlockComplete.padsOf_size]
    simp [FlockComplete.padsOf]
  rw [hP, hpdH, ← hN] at hch hcm hsl
  have hic'N : ic'.toC.ch.size = N := by rw [hch, Array.size_map, Array.size_range]
  -- the leaf salts the session commits under are the tape's
  have hsalt : ∀ j, j < N → FlockComplete.fitBytes 192 ((Masked.UE.tapeZk t).innerSalt (k / 2) (k % 2) j) =
      FlockComplete.fitBytes 192 (_root_.Flock.Firewall.Finish.saltAt q.draws.salts j) := by
    intro j hjN
    have hjc : j < _root_.Flock.Firewall.Frame.codeLen (Masked.layE x ⟨k / 2, hj⟩) := by
      rw [hN] at hjN
      exact hjN
    have hE := (hS.innerSalt (LeanRelease.ofLoopLink L₀ src real) hd ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hlay
      (Masked.UE.tapeZk t) j hjc).trans
      (innerSalt_fit (Masked.UE.tapeZk t) ((LeanRelease.ofLoopLink L₀ src real).lay x)
        ((LeanRelease.ofLoopLink L₀ src real).draws (ν, t)) Loop.frepOf (k / 2) (k % 2) j hz
        (by rw [hlay]; exact hjc))
    have hF : _root_.Flock.Firewall.Finish.saltAt q.draws.salts j =
        (_root_.Flock.Firewall.Frame.Draws.ofBytes ((LeanRelease.ofLoopLink L₀ src real).lay x (Loop.frepOf (k / 2) (k % 2)))
          ((LeanRelease.ofLoopLink L₀ src real).draws (ν, t) (Loop.frepOf (k / 2) (k % 2)))).salts.extract
          (_root_.Flock.Firewall.Frame.SALT * j) (_root_.Flock.Firewall.Frame.SALT * j + _root_.Flock.Firewall.Frame.SALT) := by
      rw [hq']
      all_goals rfl
    have e := hE.trans hF.symm
    have hsz : (_root_.Flock.Firewall.Finish.saltAt q.draws.salts j).size = 192 := by
      rw [← e]
      exact FlockComplete.JsonRT.fitBytes_size 192 _
    rw [Claims.ZKServer.fitBytes_of_size _ _ hsz]
    exact e
  -- the queried positions are one
  have hslot : l'.size - 1 = Masked.repSlot x (2 * (k / 2) + k % 2) (Masked.innerOff 3) := by
    have hst' : (Masked.stsOf x)[(2 * (k / 2) + k % 2) / 2]! = Masked.stAt x ⟨k / 2, hj⟩ := by
      rw [show (2 * (k / 2) + k % 2) / 2 = k / 2 by omega]
      exact getElem!_pos (Masked.stsOf x) (k / 2) hi'
    have hz' : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * (k / 2) + k % 2) / 2]!.m =
        .ok ((Masked.zkTab x ⟨k / 2, hj⟩).sch, (Masked.zkTab x ⟨k / 2, hj⟩).tPad) := by
      rw [hst']
      exact (Masked.zkTab x ⟨k / 2, hj⟩).hz
    have hk6 : 6 ≤ (Masked.stsOf x)[(2 * (k / 2) + k % 2) / 2]!.c.kLog := by
      rw [hst']
      exact (Masked.zkTab x ⟨k / 2, hj⟩).wf.1.hk6
    have hrr : FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * (k / 2) + k % 2) / 2]! =
        FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * (k / 2) + k % 2) / 2]! +
          ((Masked.zkTab x ⟨k / 2, hj⟩).sch.k0 + 7 + 11 * ((Masked.zkTab x ⟨k / 2, hj⟩).sch.levels.size - 1)) + 4 := by
      unfold FlockDeploy.repRounds FlockComplete.piopRoundsZk
      simp only [↓reduceIte, hz']
      omega
    have e : 2 * (k / 2) + k % 2 + 1 = k + 1 := by omega
    rw [hszV, ← e, Masked.repBase_eq hsx, FlockComplete.ZkTables.repStart_succ, hrr, Masked.repSlot_eq hsx hz']
    unfold Masked.innerOff
    omega
  unfold FlockComplete.padQs at hF1 hF2 hV1 hV2
  rw [hszF, hic'N] at hF1 hF2
  rw [hszV, hicN] at hV1 hV2
  rw [hV1, hV2, hF1, hF2]
  refine ⟨Array.map_congr_left fun j hj' => ?_, Array.map_congr_left fun j hj' => ?_⟩ <;>
  · have hjq : j ∈ FlockComplete.padQs (fun k i => Masked.slotAt (finCoins x ops) k i) l'
        (((Masked.layE x ⟨k / 2, hj⟩).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
          2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) := by
      unfold FlockComplete.padQs
      rw [hszV, ← hN]
      exact hj'
    obtain ⟨hjN, s1, s2, s3⟩ := Masked.srcV_padQs x (finCoins x ops) w t hj hr hslot j hjq
    have hjN' : j < N := by rw [hN]; exact hjN
    first
    | (rw [hc1', hc2', hch, hcm, FlockSoundness.Refine.getElem!_range_map _ hjN',
        FlockSoundness.Refine.getElem!_range_map _ hjN', FlockSoundness.Refine.getElem!_range_map _ hjN',
        FlockSoundness.Refine.getElem!_range_map _ hjN', hpc1, hpc2]
       exact Prod.ext s1 s2)
    | (rw [hc3', hsl, FlockSoundness.Refine.getElem!_range_map _ hjN',
        FlockSoundness.Refine.getElem!_range_map _ hjN', hpc3, s3]
       exact hsalt j hjN')

end Loop

end Claims.ZKServer
