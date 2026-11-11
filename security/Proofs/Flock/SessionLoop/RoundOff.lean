module

public import Proofs.Flock.SessionLoop.RoundPiop
public import Proofs.Flock.SessionLoop.FinRepPads
public import Proofs.Flock.SessionLoop.RoundOffL0
public import Proofs.Flock.SessionLoop.RoundOffOpenBind
public import Proofs.Flock.SessionLoop.RoundLig
public import Proofs.Flock.Claims.ZK.PadsRoot
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# A rep's scheduled rounds outside its masked phase

`LiveSched`'s equation at the scheduled rounds of rep stream `2j + r` where the frame is outside the masked phase, so
that `RoundFrames` does not cover them: the release of the worker's round is the switched session's message without
its leading domain frame. Each is stated after one history, at which the worker takes `vs`'s values
(`Loop.TakesValsAt`), at `SessionLoopZKDeployed`'s release and its premises on the statement and the loop's inputs, the
round's place in the rep given by the coins before it:

* **`LeanRelease.release_sched_zcBind`**: the rep's first round, the zerocheck's binding with the rep's pads root, at
  `Masked.repBase`. The frame starts the rep fresh (`freshOf_ok`), checks the worker's round against the loop's own, the
  binding behind the stream's domain frame (`Sched.rounds_two`, `head_encs`), and releases the binding with its pads
  root (`roundFree_zcHead_out`); the switched session sends the domain frame, the binding at the views' level-0 cap and
  the pads commitment's root, and `r_skip`'s squeeze (`Loop.enc_zcBind`). The two caps are one tree of `linkIn`
  (`Loop.cap_zcBind`), and the two roots one tree of the frame's samples (`Loop.root_zcBind`), where the rows fix the
  switched views' pads leaves (`Loop.swSrcE_padLeaf`).
* **`LeanRelease.release_sched_openBind`**: the opening's binding after the algebraic phase, at `Masked.repSlot` `0`
  (`LeanRelease.release_sched_openBind_pf`).
* **`LeanRelease.release_sched_l0`**: level 0's rounds `1` to `k0 + 3` (`LeanRelease.L0Sent.release`, and at `k0 + 3`
  `ρ_1`'s squeeze alone, `LeanRelease.release_on` at no frames), at `Masked.repSlot` `i`, where `vs`'s level-0 values
  are the ZK server's (`Loop.L0Fixed`; `LeanRelease.release_sched_l0_pf`).
* **`LeanRelease.release_sched_close`**: level 0's closing round, at `Masked.repSlot` `k0 + 4`: `release_sched_lig`
  there, the cut opening's body from that round on being the frame's scheduled round, its closing label ending level
  0's phase.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal.Masked

open FlockSoundness.Discharge

/-- **A rep's pads leaves read its cut's pads leaves alone**: cuts with the same pads leaves at table `j` give the same
pads leaves at each of its reps. -/
theorem padLeafCut_congr (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    {cut cut' : ∀ j, ((tabC P x j).T (cz j)).tab.Cut} (j : Fin (jOf x)) (r : Fin 2)
    (h : (cut j).leavesP = (cut' j).leavesP) : padLeafCut P x cz cut j r = padLeafCut P x cz cut' j r := by
  unfold padLeafCut
  rw [h]

end Claims.ZKMal.Masked

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree)

/-- **The frame starts a rep fresh after `prior`**: outside every phase, at absorbed position `0`. -/
theorem step_prior_fresh {c : Cfg} {k : ℕ} {b : ByteArray} {a : _root_.Flock.Firewall.Frame.Reply}
    {st' : _root_.Flock.Firewall.Frame.State} (h : _root_.Flock.Firewall.Frame.step c {} (.prior k b) = (a, st')) :
    st'.m.on = false ∧ st'.m.done = false ∧ st'.m.inner = none ∧ st'.m.pos = 0 := by
  have e : (_root_.Flock.Firewall.Frame.step c {} (.prior k b)).2 = st' := (congrArg Prod.snd h).trans rfl
  rw [← e]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> (simp only [_root_.Flock.Firewall.Frame.step]; repeat' split) <;> rfl

/-- Frames none of whose bytes is a value's carry no value byte. -/
theorem valBytes_off : ∀ fs : List Frame, (∀ f ∈ fs, ∀ i, ¬ valByte f i) → ∀ i, valBytes fs i → False
  | [], _, _, h => h
  | f :: fs, hf, i, h => by
    simp only [valBytes] at h
    rcases h with ⟨-, h⟩ | ⟨-, h⟩
    · exact hf f (List.mem_cons_self ..) i h
    · exact valBytes_off fs (fun g hg => hf g (List.mem_cons_of_mem _ hg)) _ h

/-- **The zerocheck's binding round at a rep's start, released**: `roundFree_zcHead`'s round frees no value and releases
the binding with the configuration's pads root, then `r_skip`'s squeeze. -/
theorem roundFree_zcHead_out (c : Cfg) (s : Masker) (digest cap dom : ByteArray)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = false) (hd : s.done = false)
    (hi : s.inner = none) (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hb : s.pos ≤ c.order.size) :
    ∃ out t, roundFree c s
        (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
          .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray) 6 =
        .ok (out, t, #[]) ∧
      out.data.toList = encs (zcBind digest cap c.root) ++ (Frame.sq 2 6).enc := by
  have hm5 : ∀ f ∈ ([.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
      .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] : List Frame), midFrame f := by
    intro f hf
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
    rcases hf with rfl | rfl | rfl | rfl | rfl
    · exact hdom
    · show "flock-r1cs-v0".toUTF8.size < 2 ^ 64; decide
    · exact hdg
    · exact hcap
    · show _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8.size < 2 ^ 64; decide
  obtain ⟨out, fr, hr, ho⟩ := roundFree_zc c s [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap] []
    2 6 hc hon hd hi hm5
    (by
      intro f hf
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl | rfl
      · trivial
      · show plainLabel "flock-r1cs-v0".toUTF8; unfold plainLabel; decide
      · trivial
      · trivial)
    (by simp) (.inl rfl) (by decide) (by decide) hb
  have hfr := roundFree_free c s [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
    .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] 2 6 (by simp [hi]) hm5 (.inl rfl) (by decide) (by decide) hr
  have hoff := valBytes_off [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
      .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8]
    (by
      intro f hf i h
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl | rfl | rfl <;> exact h)
  have hfr0 : fr = #[] := by
    rcases fr with ⟨_ | ⟨p, l⟩⟩
    · rfl
    · exact absurd (hfr p (Array.mem_def.2 (List.mem_cons_self ..)) p le_rfl (by omega)) (hoff p)
  subst hfr0
  refine ⟨out, _, hr, ?_⟩
  rw [ho]
  simp only [maskFrames, zcBind, encs, _root_.Flock.Firewall.Frame.LABEL_ZC, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, List.append_assoc]

/-- The PIOP's round `0` body: the pending bytes, the zerocheck's binding and `r_skip`'s squeeze. -/
theorem piopBody_zero (P : List UInt8) (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout)
    (mk : ℕ → F128) :
    piopBody P m kLog digest cap root lay mk 0 = P ++ encs (zcBind digest cap root) ++ (Frame.sq 2 6).enc := by
  simp [piopBody, piopFr, piopK, piopN, zcBind, _root_.Flock.Firewall.Frame.LABEL_ZC]

namespace Loop

set_option maxHeartbeats 1000000 in
/-- **Where table `j`'s rows hold at its base, the switched source's pads leaves are the views'** at each rep: the
switched cut is the switched views' (`RefSession.cutV_swViewsAt`), whose pads leaves are the honest views'
(`ZkView.Coined.swExe_leavesP`), and a rep's pads leaves read its cut's alone (`Masked.padLeafCut_congr`,
`Masked.tabSrcView_padLeaf`). -/
theorem swSrcE_padLeaf (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (C : Fin (refK x) → finS.Coin) (j : Fin (Masked.jOf x)) (r : Fin 2) (hrows : RowsOK vs x w)
    (hout : ∀ r : Fin 2, ∀ y ∈ (Masked.tabC Masked.PE x j).rOuter (Masked.czExe Masked.PE x C j) r, y ≠ 1) :
    ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val).padLeaf =
      ((Masked.srcV x C w t j.val).rep r.val).padLeaf := by
  have e := congrFun (Claims.ZKMal.RefSession.cutV_swViewsAt finS x C (finS.wt x w) (vs x w ν C) t) j
  have hP : (Masked.viewCut (finS.swViewsAt x C (finS.wt x w) (vs x w ν C) t j)).leavesP =
      (Masked.swCutE x C w (vs x w ν C (finS.seenOf x t)) t j).leavesP := by
    show _ = (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t j).leavesP
    rw [← e]
    rfl
  have e1 : ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val).padLeaf =
      Masked.padLeafCut Masked.PE x (Masked.czExe Masked.PE x C) (Masked.swCutE x C w (vs x w ν C (finS.seenOf x t)) t) j r := by
    exact (congrArg FlockComplete.RepSrc.padLeaf (Masked.swSrcE_rep x C w (vs x w ν C (finS.seenOf x t)) t j r)).trans
      (Masked.repSrcCut_padLeaf _ _ j r)
  have e2 : Masked.padLeafCut Masked.PE x (Masked.czExe Masked.PE x C) (Masked.swCutE x C w (vs x w ν C (finS.seenOf x t)) t) j r =
      Masked.padLeafCut Masked.PE x (Masked.czExe Masked.PE x C)
        (fun j => Masked.viewCut (finS.swViewsAt x C (finS.wt x w) (vs x w ν C) t j)) j r :=
    Masked.padLeafCut_congr Masked.PE x _ (cut := Masked.swCutE x C w (vs x w ν C (finS.seenOf x t)) t)
      (cut' := fun j => Masked.viewCut (finS.swViewsAt x C (finS.wt x w) (vs x w ν C) t j)) j r hP.symm
  have e3 : ((Masked.tabSrcView Masked.PE x (Masked.czExe Masked.PE x C)
      (finS.swViewsAt x C (finS.wt x w) (vs x w ν C) t) j.val).rep r.val).padLeaf =
      Masked.padLeafCut Masked.PE x (Masked.czExe Masked.PE x C)
        (fun j => Masked.viewCut (finS.swViewsAt x C (finS.wt x w) (vs x w ν C) t j)) j r := by
    simp only [Masked.tabSrcView, Masked.tabSrcCut, Masked.repSrcCut, j.isLt, r.isLt, ↓reduceDIte, Fin.eta]
  have hrw : ((finS.C x j).T (finS.cz x C j)).tab.onVals.RowsAt (finS.wt x w j, vs x w ν C (finS.seenOf x t) j)
      (finS.randAt x C t j).1.1 (finS.randAt x C t j).1.2.1 := hrows ν C t j hout
  refine e1.trans (e2.trans (e3.symm.trans ?_))
  exact Masked.tabSrcView_padLeaf Masked.PE x (Masked.czExe Masked.PE x C)
    (finS.swViewsAt x C (finS.wt x w) (vs x w ν C) t) (Masked.viewsE x C w t) j.isLt r.isLt
    (FlockSoundness.Discharge.ZkView.Coined.swExe_leavesP ((finS.C x j).T (finS.cz x C j)) (finS.wt x w j)
      (vs x w ν C (finS.seenOf x t) j) (finS.randAt x C t j).1 (finS.randAt x C t j).2 hrw)

set_option maxHeartbeats 1000000 in
/-- **The switched session's pads root at rep stream `2i + r` is the frame's**: where the rows hold at its base and the
frame samples the tape's pads (`Loop.RepSrc`), the pads commitment of the switched source's rep is the masked prover's on
the tape's pads (`Masked.padsTreeCut_srcV`, its leaves the views', `swSrcE_padLeaf`), whose root is that of the frame's
samples (`padsRoot_tapeOfDraws`). -/
theorem root_zcBind (vs : finS.SwVals) {L₀ : LeanRelease Flock.Wit Tp}
    {src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hS : Loop.RepSrc L₀ src x w) (Lr : LeanRelease Flock.Wit Tp) (hd : Lr.draws = L₀.draws) (ν : finS.Nonce)
    (t : finS.Tape) (C : Fin (refK x) → finS.Coin) {i r : ℕ} (hi : i < Masked.jOf x) (hr : r < 2)
    (hlay : Lr.lay x (Loop.frepOf i r) = Masked.layE x ⟨i, hi⟩) (hrows : RowsOK vs x w)
    (hout : ∀ r : Fin 2, ∀ y ∈ (Masked.tabC Masked.PE x ⟨i, hi⟩).rOuter (Masked.czExe Masked.PE x C ⟨i, hi⟩) r,
      y ≠ 1)
    {ic : FlockComplete.InnerCommit}
    (hic : (FlockComplete.padsTreeCut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.kLog (Masked.zkTab x ⟨i, hi⟩).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i).rep r) : V FlockComplete.InnerCommit) = .ok ic) :
    ic.root = _root_.Flock.Firewall.Frame.padsRoot (Masked.layE x ⟨i, hi⟩)
      (_root_.Flock.Firewall.Frame.Draws.ofBytes (Masked.layE x ⟨i, hi⟩) (Lr.draws (ν, t) (Loop.frepOf i r))) := by
  have hx := Masked.live_of hsc hR
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[i]! = _root_.Flock.Setup.ofCircuit (Masked.stAt x ⟨i, hi⟩) :=
    Masked.tabsOf_stAt x ⟨i, hi⟩
  have hlayE : _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.kLog (Masked.zkTab x ⟨i, hi⟩).sch.k0 = Masked.layE x ⟨i, hi⟩ := by
    rw [htab]
    rfl
  have hz := hS.size Lr hd ν t ⟨i, hi⟩ ⟨r, hr⟩ hlay
  have hpads : FlockComplete.padsOf (Masked.UE.tapeZk t) i r (Masked.layE x ⟨i, hi⟩) =
      FlockComplete.padsOf (tapeOfDraws (Masked.UE.tapeZk t) (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf) i r
        (Masked.layE x ⟨i, hi⟩) :=
    hS.pads Lr hd ν t ⟨i, hi⟩ ⟨r, hr⟩ hlay (Masked.UE.tapeZk t)
  have hpdH : ((FlockComplete.padsOf (Masked.UE.tapeZk t) i r (Masked.layE x ⟨i, hi⟩)).h ++
      (FlockComplete.padsOf (Masked.UE.tapeZk t) i r (Masked.layE x ⟨i, hi⟩)).padH).size =
        (Masked.layE x ⟨i, hi⟩).len + _root_.Flock.Zk.K_PAD := by
    rw [Array.size_append, FlockComplete.padsOf_size]
    simp [FlockComplete.padsOf]
  have hTape : (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (FlockComplete.padsOf (Masked.UE.tapeZk t) i r (Masked.layE x ⟨i, hi⟩))
      ((Masked.UE.tapeZk t).innerSalt i r) : V FlockComplete.InnerCommit) =
      FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (FlockComplete.padsOf (tapeOfDraws (Masked.UE.tapeZk t) (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf) i r
          (Masked.layE x ⟨i, hi⟩))
        ((tapeOfDraws (Masked.UE.tapeZk t) (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf).innerSalt i r) := by
    rw [FlockComplete.innerCommitP_salt_congr (M := V) (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (FlockComplete.padsOf (Masked.UE.tapeZk t) i r (Masked.layE x ⟨i, hi⟩))
      (salt := (Masked.UE.tapeZk t).innerSalt i r)
      (salt' := (tapeOfDraws (Masked.UE.tapeZk t) (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf).innerSalt i r)
      (fun q hq => hS.innerSalt Lr hd ν t ⟨i, hi⟩ ⟨r, hr⟩ hlay (Masked.UE.tapeZk t) q
        (by rw [hpdH] at hq; exact hq)), hpads]
  have hpl : ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i).rep r).padLeaf = ((Masked.srcV x C w t i).rep r).padLeaf :=
    swSrcE_padLeaf vs x w ν t C ⟨i, hi⟩ ⟨r, hr⟩ hrows hout
  have hic1 : (FlockComplete.padsTreeCut Masked.hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.kLog (Masked.zkTab x ⟨i, hi⟩).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i).rep r)).go #[] = (.ok ic, #[]) := by
    rw [FlockComplete.CutEq.padsTreeCut_quiet _ _ _ #[], hic]
  rw [FlockComplete.padsTreeCut_leaf Masked.hsE _ hpl, Masked.padsTreeCut_srcV Masked.simCommit hx hR t C hi hr,
    hlayE, Functor.map_map, FlockComplete.go_map', FlockComplete.innerCommitP_quiet _root_.Flock.Sha512.hash _ _ #[],
    hTape] at hic1
  have hic2 := congrArg Prod.fst hic1
  cases hT : (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (FlockComplete.padsOf (tapeOfDraws (Masked.UE.tapeZk t) (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf) i r
        (Masked.layE x ⟨i, hi⟩))
      ((tapeOfDraws (Masked.UE.tapeZk t) (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf).innerSalt i r) :
      V FlockComplete.InnerCommit) with
  | error e =>
    rw [hT] at hic2
    simp [Except.map] at hic2
  | ok icT =>
    rw [hT] at hic2
    simp only [Except.map, Except.ok.injEq] at hic2
    subst hic2
    have hroot := padsRoot_tapeOfDraws (Masked.UE.tapeZk t) (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf i r hz
      (ic := icT) (by rw [hlay]; exact hT)
    rw [hlay] at hroot
    exact hroot.symm

set_option maxHeartbeats 1000000 in
/-- **The switched session's first round of rep stream `2j + r`**: where the coins show no refusal by it, the session
sends the stream's domain frame, the zerocheck's binding at the switched source's level-0 cap and the root of its pads
commitment, and `r_skip`'s squeeze (`Masked.runCut_piopBody` at round `0`). -/
theorem enc_zcBind (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (C : Fin (refK x) → finS.Coin) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (h0 : Masked.repBase x (2 * j.val + r.val) ≠ 0)
    (hN0 : ¬ Masked.Refused x C (Masked.repBase x (2 * j.val + r.val))) :
    ∃ ic : FlockComplete.InnerCommit,
      (FlockComplete.padsTreeCut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
          (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val) : V FlockComplete.InnerCommit) = .ok ic ∧
      finS.enc x C (Masked.repBase x (2 * j.val + r.val)) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) =
        0 :: (encs (.dom (spec.domain spec.tables[j.val]! r.val).toUTF8 ::
          zcBind (_root_.Flock.Setup.ofCircuit (Masked.stAt x j)).digest
            (FlockComplete.flat (FlockComplete.layerOf (fun a b => _root_.Flock.Sha512.hash (a ++ b))
              (Masked.lv0 x j).d (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).leaf0
              ((Masked.lv0 x j).d - (Masked.lv0 x j).cap)))
            ic.root) ++ (Frame.sq 2 6).enc) := by
  have hj := Masked.lt_of_jOf j
  obtain ⟨schs, hs, hschs⟩ := Masked.schedsPZk_live hx
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hms : ∀ i < (FlockDeploy.tabsOf (Masked.stsOf x)).size,
      (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (Masked.stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  let cmOf : ℕ → FlockComplete.Commit := fun i =>
    Classical.choose (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))
  have hcm : ∀ i, (FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
      _root_.Flock.Hashes V) _root_.Flock.MerkleScheme.hm96Sha512 (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i) : V _) = .ok (cmOf i) := fun i =>
    (Classical.choose_spec (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))).1
  have hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i)).go l =
      (.ok ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf), l) := by
    intro l
    have hq : FlockComplete.quietS.R
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
              _root_.Flock.Hashes FlockComplete.PS)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i)) :=
      _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun i =>
        FlockComplete.CutEq.commit0Cut_quiet _ _ _ _
    rw [hq l, FlockComplete.mapM_ok_mem cmOf _ fun i hi => ?_]
    rw [hms i (by simpa using hi)]
    exact hcm i
  obtain ⟨extras, hext⟩ := Masked.extra_ok hx hsx C hj
  obtain ⟨ic, hic0⟩ := FlockComplete.padsTreeCut_ok
    (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
      _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
    ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val)
  have hrc := Masked.runCut_piopBody hx C (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t) hsx hs hcms j r hN0 hext (ic := ic)
    (fun l => by rw [FlockComplete.CutEq.padsTreeCut_quiet _ _ _ l, hic0]) (Masked.tabsOf_stAt x j) (jj := 0)
    (by unfold piopNR; omega)
  simp only [Nat.add_zero] at hrc
  have hsrc : Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x C) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) :=
    rfl
  have hj' : j.val < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  have hci : ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf)[j.val]! = cmOf j.val := by
    rw [getElem!_pos ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf) j.val (by simpa using hj'),
      Array.getElem_map, Array.getElem_range]
  have hcap' : (cmOf j.val).cap = FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r))
      (schs[j.val]!.1.levels.getD 0 default).d (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).leaf0
      ((schs[j.val]!.1.levels.getD 0 default).d - (schs[j.val]!.1.levels.getD 0 default).cap) :=
    (Classical.choose_spec (FlockComplete.commit0Cut_cap (schs[j.val]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val))).2
  have hlv : schs[j.val]!.1.levels.getD 0 default = Masked.lv0 x j := by
    rw [hschs j]
    try rfl
  refine ⟨ic, hic0, ?_⟩
  rw [enc_finS_of C h0 (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) hsx hN0
    (by rw [← hsrc]; exact hrc.1), ← hsrc, hrc.2, piopBody_zero, FlockSoundness.Refine.init_enc, hci, hcap', hlv]
  simp only [encs_cons, List.append_assoc]

end Loop

namespace LeanRelease

open Loop in
set_option maxHeartbeats 2000000 in
/-- **The frame at a rep's first round**: at a live history whose next round is rep stream `2j + r`'s first
(`Masked.repBase`), the frame starts the rep fresh (`freshOf_ok`, `step_prior_fresh`) and checks the worker's round
against the loop's own, the binding behind the stream's domain frame (`Sched.rounds_two`, `head_encs`), so the worker's
round is that round and the release is the binding with the configuration's pads root and `r_skip`'s squeeze
(`roundFree_zcHead_out`), at the cap of the loop's table there. The round frees no word. -/
theorem zcBind_frame (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoop Masked.dataE L₁ real).toSrv.Worker) {x : Msg} {w : Flock.Wit} (hx : Masked.Live x)
    (hSv : Loop.Serves x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoop Masked.dataE L₁ real).toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)))
    (hlt : ops.length < refK x)
    (hl : (ofLoop Masked.dataE L₁ real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) {v : Msg}
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hn : ops.length = Masked.repBase x (2 * j.val + r.val)) :
    ∃ (tr : Tab) (hir : Array ByteArray), real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) ∧
      (ofLoop Masked.dataE L₁ real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
        0 :: (encs (zcBind (_root_.Flock.Setup.ofCircuit (Masked.stAt x j)).digest
          (_root_.Flock.Firewall.Finish.flat tr.cm0.cap)
          ((ofLoop Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).root) ++ (Frame.sq 2 6).enc) ∧
      (ofLoop Masked.dataE L₁ real).rep x (ops.length + 1) = 2 * j.val + r.val ∧
      (∀ {m₁ : Masker} {out₁ : ByteArray} {s₁ : Masker} {fr₁ : Array ℕ},
        (ofLoop Masked.dataE L₁ real).answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m₁ →
        roundFree ((ofLoop Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m₁ with stopped := false }
          ⟨v.toArray⟩ ((ofLoop Masked.dataE L₁ real).coins x (ops.length + 1)) = .ok (out₁, s₁, fr₁) →
        fr₁ = #[]) ∧
      ∃ lay mask, _root_.Flock.Firewall.Drive.start (Masked.stsOf x)[j.val]! = .ok (lay, mask) := by
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjh : (2 * j.val + r.val) / 2 = j.val := by omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hrk : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  have hR0 : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
      (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    rw [hrk, repRounds_true]
    simp only [hz]
    rw [hst]
    have hA : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
        (Masked.stAt x j).m + ((Masked.stAt x j).c.kLog - 6) := by
      unfold piopNR
      omega
    exact le_trans (le_trans (le_trans hA (Nat.le_add_right _ _)) (Nat.le_add_right _ _)) (Nat.le_add_right _ _)
  have hbase : Masked.repBase x (2 * j.val + r.val) =
      1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
    rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
  obtain ⟨q, hq⟩ : ∃ q, ops.length = q + 1 := ⟨ops.length - 1, by omega⟩
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q = (2 * j.val + r.val, 0) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) 0
      (by rw [hrsl]; exact hq2) (lt_of_lt_of_le (by unfold piopNR; omega) hR0)
    rw [Nat.zero_add, Nat.add_zero] at e
    rw [← e]
    congr 1
    omega
  have hj0 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).2 = 0 := by rw [hloc]
  -- the frame's check
  have hlast : (ofLoop Masked.dataE L₁ real).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ real) x
  have hk : (ofLoop Masked.dataE L₁ real).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, out, s', hA, hck⟩ :=
    check_of_live (ofLoop Masked.dataE L₁ real) wk x w (ν, t) κ ((m₀, ρ₀) :: ops) v hl hk hv
  -- the frame starts the rep fresh
  obtain ⟨e0, h1, s0', spec', sts', q', k', j', hh, hsx', hq', hloc', hcase⟩ :=
    ofLoop_progChecks_round Masked.dataE L₁ real x w (ν, t) ((m₀, ρ₀) :: ops) hA hk ⟨v, s', none, hck⟩
  obtain ⟨-, hh1⟩ := List.cons.inj hh
  rw [← hh1] at hq' hcase
  have hss := Option.some.inj (hsx'.symm.trans hsx)
  simp only [Prod.mk.injEq] at hss
  obtain ⟨-, hspec', hsts'⟩ := hss
  rw [hspec', hsts', show q' = q by omega, hloc] at hloc'
  simp only [Prod.mk.injEq] at hloc'
  obtain ⟨-, hj'⟩ := hloc'
  have hfresh : freshOf x (ops.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m := by
    rcases hcase with ⟨j'', -, -, -, -, -, hjj, -⟩ | ⟨-, hf⟩
    · omega
    · exact hf
  obtain ⟨c', tb, dom, xs, his, alg, hph0, -, hm'⟩ :=
    freshOf_ok x _ hsx (q := q) (by rw [List.length_map]; omega) hfresh
  rw [hloc] at hph0
  dsimp only at hph0
  have hm0 : m.on = false ∧ m.done = false ∧ m.inner = none ∧ m.pos = 0 := by
    rcases hm' with ⟨-, hm1⟩ | ⟨-, st', hst', hm1⟩
    · rw [hm1]
      exact ⟨rfl, rfl, rfl, rfl⟩
    · rw [hm1]
      exact step_prior_fresh hst'
  -- the loop's own round there
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_live Masked.dataE L₁ real wk x w (ν, t) κ (m₀, ρ₀) ops hl hk
    hsx (q := q) hq
  have hre' : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  obtain ⟨hT1, -, -, -, -, hTcap⟩ := Count.real_tabAlike hRT hph0 hre'
  obtain ⟨hts', -, -⟩ := Count.placeholder_tab hph0
  rw [hst] at hts'
  obtain ⟨lay, mask, hds, -, -, -⟩ := placeholder_ok hph0
  have htrm : tr.st.m = (Masked.stAt x j).m := by rw [hT1, hts']; rfl
  have hmT : 14 ≤ tr.st.m := by rw [htrm]; omega
  have hmT' : tr.st.m < 2 ^ 64 := by rw [htrm]; exact lt_of_le_of_lt hm35 (by norm_num)
  obtain ⟨alg', hal'⟩ := schedSoFar_ok_rounds hs'
  have h0' := Sched.rounds_two hmT hmT' hal'
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal' hs'
  obtain ⟨hsz0, -⟩ := Array.getElem?_eq_some_iff.1 h0'.1
  have hlt' : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).2 < sched'.size := by
    rw [hj0, hrest, Array.size_append]
    omega
  have hscheq : (sched'.getD (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).2
      (.empty, 0)).1.data.toList =
      encs [.dom (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 / 2]!)
          ((ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 % 2)).toUTF8,
        .label "flock-r1cs-v0".toUTF8, .bytes tr.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat tr.cm0.cap),
        .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc := by
    rw [hj0, hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hsz0, h0'.1, head_encs]
    simp only [Option.getD_some, List.toList_toArray]
  have hdom := (sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx)
    (k := (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1) (by rw [hloc]; exact hq2)).2.1
  -- the loop's coin counts there
  have hpl := Count.piop_list (kLog := (Masked.stAt x j).c.kLog) (by omega : 14 ≤ (Masked.stAt x j).m)
  have hpL := congrArg List.length hpl
  simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil] at hpL
  have hNR : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      loopNr x (Masked.repBase x (2 * j.val + r.val) + jj) = piopN (Masked.stAt x j).m jj := by
    intro jj hjj'
    unfold piopNR at hjj'
    rw [Masked.repBase_eq hsx, loopNr_plan hsx hq2 hz (by rw [hst]; exact hk6) (i := jj)
      (by unfold FlockComplete.piopRoundsZk; rw [hst]; omega),
      Count.plan_get hz (by rw [List.length_append, hst, hpL]; omega), hst,
      List.getElem?_append_left (by rw [hpL]; omega), hpl]
    exact piopN_list (by omega) hk6 (by unfold piopNR; omega)
  have hcoins : (ofLoop Masked.dataE L₁ real).coins x ((m₀, ρ₀) :: ops).length = 6 := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel, hn]
    simpa [piopN] using hNR 0 (by unfold piopNR; omega)
  rw [ofLoop_scheduled Masked.dataE L₁ real x w (ν, t) (m₀, ρ₀) ops _ hsx (q := q) hq hre hs' hlt',
    hscheq, hcoins] at hck
  -- the worker's round is the loop's, and its release the binding
  have hdg : tr.st.digest.size < 2 ^ 64 := by
    rw [hT1, hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hdgE : tr.st.digest = (_root_.Flock.Setup.ofCircuit (Masked.stAt x j)).digest := by rw [hT1, hts']
  have hc : ∀ p, p < ((ofLoop Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size →
      ((ofLoop Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[p]! <
        ((ofLoop Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.size :=
    Claims.ZKServer.ofDraws_order _ _ _
  obtain ⟨o, tq, hro, ho⟩ := roundFree_zcHead_out ((ofLoop Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    { m with stopped := false } tr.st.digest (_root_.Flock.Firewall.Finish.flat tr.cm0.cap) _ hc hm0.1 hm0.2.1
    hm0.2.2.1 hdg hTcap hdom (by show m.pos ≤ _; rw [hm0.2.2.2]; exact Nat.zero_le _)
  have hvb := passes_eq hro (check_none_passes hck)
  have hrel := release_of_roundFree (ofLoop Masked.dataE L₁ real) x (ν, t) ((m₀, ρ₀) :: ops) false v hA hcoins
    (out := o) (s' := tq) (fr := #[]) (by rw [hvb]; exact hro)
  refine ⟨tr, hir, hre', ?_, ?_, ?_, lay, mask, by rw [hjh] at hds; exact hds⟩
  · rw [hrel, ho, hdgE]
  · show repOf x (ops.length + 1) = _
    rw [show ops.length + 1 = q + 2 by omega, repOf_add_two hsx, hloc]
  · intro m₁ out₁ s₁ fr₁ hA₁ hf
    obtain rfl : m = m₁ := Except.ok.inj (hA.symm.trans hA₁)
    have hc6 : (ofLoop Masked.dataE L₁ real).coins x (ops.length + 1) = 6 := hcoins
    rw [hc6, hvb] at hf
    exact (Prod.mk.inj (Prod.mk.inj (Except.ok.inj (hf.symm.trans hro))).2).2

open Loop in
set_option maxHeartbeats 2000000 in
/-- **The rep's zerocheck binding**: at rep stream `2j + r`'s first round (`Masked.repBase`), the release of the worker's
round is the switched session's message without its leading domain frame. The binding carries the rep's pads root,
which the rows fix (`Loop.RowsOK`). -/
theorem release_sched_zcBind (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (hTab : Loop.finS.Tables) (hTape : Loop.finS.TapeOK) (hRows : Loop.RowsOK vs x w)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hn : ops.length = Masked.repBase x (2 * j.val + r.val)) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  have hx : Masked.Live x := Masked.live_of hSc hR
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hSc hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjh : (2 * j.val + r.val) / 2 = j.val := by omega
  have hrh : (2 * j.val + r.val) % 2 = r.val := by omega
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  obtain ⟨tr, hir, hre, hrel, hrep, -, lay, mask, hds⟩ :=
    zcBind_frame (withFiles (withLink L₀ src) src) real wk hx hSv ν t κ m₀ ρ₀ ops
      (by rw [hreal]; exact Loop.realTabs_of src x w (ν, t) hsx) hlt hl hv j r hn
  have hrel' : (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      0 :: (encs (zcBind (_root_.Flock.Setup.ofCircuit (Masked.stAt x j)).digest
        (_root_.Flock.Firewall.Finish.flat tr.cm0.cap)
        ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).root) ++ (Frame.sq 2 6).enc) := hrel
  have hrep' : (ofLoopLink L₀ src real).rep x ((m₀, ρ₀) :: ops).length = Loop.frepOf j.val r.val := hrep
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := L₀) src real j r hds
  have hL0 : Loop.Level0Src src x w := (show Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w from hS).1
  have hRS : Loop.RepSrc L₀ src x w := (show Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w from hS).2
  -- the coins show no refusal by the rep's first round, and no outer coin `1`
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hRk : Masked.rankSlot x ((2 * j.val + r.val) / 2) ((2 * j.val + r.val) % 2) =
      Masked.repBase x (2 * j.val + r.val) + ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega, show (2 * j.val + r.val) % 2 = r.val by omega,
      show Masked.rankSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) +
        ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) from rfl, hstj]
  have hN0 := unrefused_prefix Masked.dataE (withFiles (withLink L₀ src) src) real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops
    hlt.le hl href (N := Masked.repBase x (2 * j.val + r.val)) (by omega) (q := 2 * j.val + r.val) (by omega) le_rfl
    (by rw [hRk]; omega)
  have hout := outer_ne_one_ofLoopLink L₀ src real wk hSv (ν, t) κ m₀ ρ₀ ops hlt.le hl href j
  -- `Masked.swSrcE` at `finS.pad (ops.map …)` itself exhausts `whnf`; at a variable list it elaborates
  generalize hcs : (ops.map fun p => Loop.finS.decOf p.2) = cs
  rw [hcs] at hN0
  have hout' : ∀ r : Fin 2, ∀ y ∈ (Masked.tabC Masked.PE x j).rOuter
      (Masked.czExe Masked.PE x (Loop.finS.pad (K := refK x) cs) j) r, y ≠ 1 := by
    rw [← hcs]
    exact hout
  have hcl : cs.length = Masked.repBase x (2 * j.val + r.val) := by rw [← hcs, List.length_map]; exact hn
  have hlt' : cs.length < refK x := by rw [← hcs, List.length_map]; exact hlt
  obtain ⟨-, hdom0, -, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  rw [hjh, hrh] at hdom0
  -- the session's round, its cap and its pads root
  obtain ⟨ic, hic, henc⟩ := Loop.enc_zcBind vs w hx hsx ν t (Loop.finS.pad (K := refK x) cs) j r (by omega) hN0
  have hroot := Loop.root_zcBind vs hSc hR hRS (ofLoopLink L₀ src real) (LeanRelease.draws_ofLoopLink src real) ν t
    (Loop.finS.pad (K := refK x) cs) j.isLt r.isLt hlay hRows hout' hic
  have hcfg : ((ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).root = ic.root := by
    rw [cfg_root, hrep']
    unfold LeanRelease.drawsAt
    rw [hlay]
    exact hroot.symm
  rw [hreal] at hre
  have hcap := Loop.cap_zcBind hL0 ν t (Loop.finS.pad (K := refK x) cs)
    (vs x w ν (Loop.finS.pad (K := refK x) cs) (Loop.finS.seenOf x t)) j r hre
  rw [Loop.swMsgAt_enc vs w ν t cs hlt', hcl, henc, dropDom_dom _ _ _ hdom0, hrel', hcfg, hcap]
  all_goals rfl

/-- **The opening's binding**: at rep stream `2j + r`'s round after its algebraic phase and PIOP
(`Masked.repSlot` `0`), the release of the worker's round is the switched session's message without its leading domain
frame. -/
theorem release_sched_openBind (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => 0) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) :=
  release_sched_openBind_pf vs L₀ src real hreal wk hR hSv hSc hS ν t κ m₀ ρ₀ ops hTV hpos hlt hl hok href hv j r hn

/-- **Level 0's rounds**: at rep stream `2j + r`'s level-0 round `i`, `1 ≤ i ≤ k0 + 3` (`Masked.repSlot` `i`; at
`k0 + 3`, `ρ_1`'s squeeze alone), where `vs`'s level-0 values are the ZK server's (`Loop.L0Fixed`), the release of the
worker's round is the switched session's message without its leading domain frame. -/
theorem release_sched_l0 (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (hL0 : Loop.L0Fixed vs x w) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : 1 ≤ i)
    (hi : i ≤ (Masked.zkTab x j).sch.k0 + 3) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) :=
  release_sched_l0_pf vs L₀ src real hreal wk hR hSv hSc hS hL0 ν t κ m₀ ρ₀ ops hTV hpos hlt hl hok href hv j r hn h1 hi

/-- **Level 0's closing round**: at rep stream `2j + r`'s round `k0 + 4` past its PIOP (`Masked.repSlot`), the release of
the worker's round is the switched session's message without its leading domain frame (`release_sched_lig` there). -/
theorem release_sched_close (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun sch => sch.k0 + 4) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  have hj := Masked.lt_of_jOf j
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [Masked.stsOf_half x j r]
    exact (Masked.zkTab x j).hz
  have hn' : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => (Masked.zkTab x j).sch.k0 + 4 :=
    (hn.trans (Masked.repSlot_eq hsx hz fun sch => sch.k0 + 4)).trans
      (Masked.repSlot_eq hsx hz fun _ => (Masked.zkTab x j).sch.k0 + 4).symm
  exact release_sched_lig vs L₀ src real hreal wk hR hSv hSc hS ν t κ m₀ ρ₀ ops hTV hpos hlt hl hok href hv j r hn'
    le_rfl (by omega)

end LeanRelease

end Claims.ZKServer
