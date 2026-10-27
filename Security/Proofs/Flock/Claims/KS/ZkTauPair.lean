import Proofs.Flock.Claims.KS.ZkTauFork

/-!
# The masked `τ` fork's two-run clash

`ZkTauClash` (`ZkTauFork`) is the clash of the live `τ` fork event: two accepting runs whose decoded outputs at table
`j` are accepted and open one cap `d`'s `τ` commitment to different values (`TauForkEvZ`). This file proves it with no
hypothesis, so the `τ` fork bound rests on its alignment alone (`zkSimForkLeTau_of_align`).

The clash reads the model's openings, as at the pads: a `τ` opening that `opTauZ` accepts has `hm96`'s leaf under its
salt equal to `d 0` (`opTauZ_some`), and at SHA-512 that leaf is the executable's `τ` commitment (`leaf_tau`). At an
accepting run's view each rep's `τ` opening is its proof file's inner proof's `τ` and salt (`PadsRep`), and the run's
verifier hashed every input of their commitment (`verifyInner_tau`; `TauLogged`). Two commitments of different values
that agree collide in the two runs' joined logs (`tiedColl_of_tauLogged`): at `τ`'s tagged digest, or at the leaf
(`Hm96.leafPair_collides`).

* `verifyInner_tau`: `Zk.verifyInnerWith`, run with `hashesIn L`, checked that `τ` opens its commitment, and every
  SHA-512 input of that commitment is in `L`.
* `zkViewTauLogged`, `zkAccTau`: an accepting run's view, and a decoded live outcome of the masked simulator, log their
  reps' `τ` openings.
* `zkTauClash`: `ZkTauClash` holds; `zkSimForkLeTau_of_align`: the `τ` fork bound from `ZkTauAlign`.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

attribute [local irreducible] Flock.Sha512.hash

section Log

open Flock Transcript FlockSoundness.Refine

/-- **The SHA-512 inputs of a `τ` commitment**: `τ`'s tagged bytes, then its `hm96-sha512/v1` leaf's under `salt`. -/
def tauIns (tau : F128) (salt : ByteArray) : List ByteArray :=
  [Zk.TAU_TAG ++ tau.toBytes, Hm96.Default512.saltPrefix ++ salt,
    Hm96.Default512.leafPrefix ++ (Hm96.xor64 (Sha512.hash (Zk.TAU_TAG ++ tau.toBytes)) (Hm96.Default512.mask salt) ++
      Sha512.hash (Hm96.Default512.saltPrefix ++ salt))]

/-- **The inner proof's `τ` commitment, logged**: when `Zk.verifyInnerWith` accepts with SHA-512 answered only within
`L`, `τ` opens the inner proof's `τ` commitment under its salt, and every SHA-512 input of that commitment is in `L`. -/
theorem verifyInner_tau {L : List Claims.Msg} {lay : Zk.PadLayout} {pr : Zk.InnerProof} {finalC : F128}
    {cons : Array (String × Zk.Lin)} {t t' : Transcript}
    (h : Zk.verifyInnerWith (hashesIn L) lay pr finalC cons t = .ok ((), t')) :
    Zk.tauCommitment pr.tau pr.tauSalt = pr.tauCommit ∧ ∀ b ∈ tauIns pr.tau pr.tauSalt, Claims.toMsg b ∈ L := by
  unfold Zk.verifyInnerWith at h
  simp only [liftTM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨_, -, h⟩ := label_bind h
  obtain ⟨_, -, h⟩ := absorbF_bind h
  obtain ⟨-, -, _, -, -, h⟩ := sampleWith_bind _ h
  obtain ⟨_, -, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, _, -, -, h⟩ := sampleVecWith_bind _ (by simp [Zk.triple]) h
  obtain ⟨_, -, h⟩ := absorbBytes_bind h
  obtain ⟨-, -, _, -, -, h⟩ := sampleWith_bind _ h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨_, -, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, _, -, -, h⟩ := sampleVecWith_bind _ (by decide) h
  obtain ⟨tc, htc, h⟩ := vlift_bind h
  obtain ⟨heq, -⟩ := check_bind h
  have hr := With.zk_tauCommitmentWith_sim (back L) (back_hashes L) pr.tau pr.tauSalt tc htc
  unfold Zk.tauCommitmentWith at hr
  obtain ⟨x, hx, hr⟩ := Runs.bind hr
  obtain ⟨rfl, hxm⟩ := Runs.hash hx
  obtain ⟨rfl, hsm, hlm⟩ := Runs.leafW hr
  refine ⟨bytes_eq_of_beq heq, fun b hb => ?_⟩
  simp only [tauIns, List.mem_cons, List.not_mem_nil, or_false] at hb
  rcases hb with rfl | rfl | rfl
  · exact hxm
  · exact hsm
  · exact hlm

/-- `Hm96.leafPair`'s inputs at two `τ` commitments are among their inputs. -/
theorem leafPair_tauIns (τ₁ τ₂ : F128) (s₁ s₂ : ByteArray) :
    (Hm96.leafPair Hm96.Default512.leafPrefix Hm96.Default512.saltPrefix Hm96.Default512.mask
        (Sha512.hash (Zk.TAU_TAG ++ τ₁.toBytes)) (Sha512.hash (Zk.TAU_TAG ++ τ₂.toBytes)) s₁ s₂).1 ∈ tauIns τ₁ s₁ ∧
      (Hm96.leafPair Hm96.Default512.leafPrefix Hm96.Default512.saltPrefix Hm96.Default512.mask
        (Sha512.hash (Zk.TAU_TAG ++ τ₁.toBytes)) (Sha512.hash (Zk.TAU_TAG ++ τ₂.toBytes)) s₁ s₂).2 ∈
        tauIns τ₂ s₂ := by
  simp only [Hm96.leafPair]
  split_ifs <;> simp [tauIns]

/-- Two different inputs with one SHA-512 digest, each in its own log, collide in the joined logs. -/
theorem tiedColl_of_hash {L₁ L₂ : List (Claims.Msg × Claims.Msg)} {u v : ByteArray} (hne : u ≠ v)
    (h : Sha512.hash u = Sha512.hash v) (hu : Claims.toMsg u ∈ L₁.map Prod.fst)
    (hv : Claims.toMsg v ∈ L₂.map Prod.fst) : Claims.TiedColl sha512 (L₁ ++ L₂) := by
  refine ⟨Claims.toMsg u, Claims.toMsg v, ⟨fun e => hne (toMsg_injective e), ?_⟩, ?_, ?_⟩
  · show Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg _))) =
      Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg _)))
    rw [Claims.ofMsg_toMsg, Claims.ofMsg_toMsg, h]
  · rw [List.map_append]
    exact List.mem_append_left _ hu
  · rw [List.map_append]
    exact List.mem_append_right _ hv

end Log

section Logged

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.ZkExec (innerN layZ shapeZK shapeZKn innerOpensOf PadsRep hs512 leaf_tau)
open Claims.Bridge.Custody

/-- **A rep's `τ` opening, logged**: `τ` and its salt, every SHA-512 input of whose commitment is in `L`. -/
def TauLogged (L : List (Claims.Msg × Claims.Msg)) (io : ZkLink.InnerOpens 𝔽 Dig (List UInt8)) : Prop :=
  ∃ (tau : F128) (salt : ByteArray), io.tau = (fOf tau, salt.data.toList) ∧
    ∀ b ∈ tauIns tau salt, Claims.toMsg b ∈ L.map Prod.fst

/-- **Two logged `τ` openings of different values with one leaf collide in the joined logs**: at SHA-512 the leaves
are the executable's `τ` commitments (`leaf_tau`), so either `τ`'s tagged digests agree, a collision of the tagged
bytes, or the `hm96-sha512/v1` leaves of different digests agree (`Hm96.leafPair_collides`); every input of each is in
its own log. -/
theorem tiedColl_of_tauLogged {L₁ L₂ : List (Claims.Msg × Claims.Msg)}
    {io₁ io₂ : ZkLink.InnerOpens 𝔽 Dig (List UInt8)} (h₁ : TauLogged L₁ io₁) (h₂ : TauLogged L₂ io₂)
    (hv : hs512.Lτ.leaf io₁.tau.1 io₁.tau.2 = hs512.Lτ.leaf io₂.tau.1 io₂.tau.2) (hne : io₁.tau.1 ≠ io₂.tau.1) :
    Claims.TiedColl sha512 (L₁ ++ L₂) := by
  obtain ⟨τ₁, s₁, he₁, hl₁⟩ := h₁
  obtain ⟨τ₂, s₂, he₂, hl₂⟩ := h₂
  rw [he₁, he₂] at hv hne
  dsimp only at hv hne
  have hτ : τ₁ ≠ τ₂ := fun e => hne (by rw [e])
  have hc : Zk.tauCommitment τ₁ s₁ = Zk.tauCommitment τ₂ s₂ := by
    rw [← leaf_tau, ← leaf_tau]
    exact congrArg Subtype.val hv
  by_cases hx : Sha512.hash (Zk.TAU_TAG ++ τ₁.toBytes) = Sha512.hash (Zk.TAU_TAG ++ τ₂.toBytes)
  · exact tiedColl_of_hash (fun e => hτ (toBytes_inj ((ByteArray.append_right_inj _).mp e))) hx
      (hl₁ _ (by simp [tauIns])) (hl₂ _ (by simp [tauIns]))
  · obtain ⟨hne', heq⟩ := Hm96.leafPair_collides Hm96.Default512.leafPrefix Hm96.Default512.saltPrefix
      Hm96.Default512.mask (Sha512.hash (Zk.TAU_TAG ++ τ₁.toBytes)) (Sha512.hash (Zk.TAU_TAG ++ τ₂.toBytes)) s₁ s₂
      (Sha512.hash_size _) (Sha512.hash_size _) (fun e => hx (Prod.mk.inj e).1) hc
    obtain ⟨m₁, m₂⟩ := leafPair_tauIns τ₁ τ₂ s₁ s₂
    exact tiedColl_of_hash hne' heq (hl₁ _ m₁) (hl₂ _ m₂)

/-- **An accepted masked rep logs its `τ` opening**: where rep `r`'s proof file decodes to the openings `io`
(`PadsRep`), the run's verifier ran that proof's inner proof (`zkVerifyRepWith_inner`), which checked its `τ`
commitment and hashed every input of it (`verifyInner_tau`). -/
theorem zk_tauLogged {L : List (Claims.Msg × Claims.Msg)} {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live} {j rr : ℕ} {sch : Schedule} {tPad : ℕ}
    (hv : Zk.verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) record proofs = .ok ())
    (hlv : liveOf spec record = .ok lv) (hj : j < sts.size) (hsch : Zk.schedule sts[j].m = .ok (sch, tPad))
    (hp2 : spec.points = 2) (hrr : rr < 2) {y : StreamRec} {io : ZkLink.InnerOpens 𝔽 Dig (List UInt8)}
    {qpos : ℕ → ℕ}
    (hpads : PadsRep (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) y proofs[2 * j + rr]! io qpos) :
    TauLogged L io := by
  obtain ⟨hsz, hall⟩ := rep_zk _ hv hlv
  have hkk : 2 * j + rr < lv.streams.size := by omega
  obtain ⟨sess, sch', tPad', stream, c, hpts, hpn, hsch', hrep, -, -⟩ := hall (2 * j + rr) hkk
  have hj' : sts[(2 * j + rr) / 2]! = sts[j] := by
    rw [show (2 * j + rr) / 2 = j by omega]
    exact getElem!_pos sts j hj
  rw [hj', hsch] at hsch'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hsch')
  rw [hj'] at hrep
  have hsch₀ : Zk.schedule (Setup.ofCircuit sts[j]).m = .ok (sch, tPad) := hsch
  obtain ⟨_, extras, zp, finalC, cons, t, t', -, hex, hdec, -, -, hinner⟩ := zkVerifyRepWith_inner _ hsch₀ hrep
  rw [hpts] at hex
  have hx : extras.size = 2 * sts[j].regions.size := by
    rw [extraClaims_size (st := sts[j]) hex, ← hpts, hpn, hp2, Nat.mul_comm]
  have hsh : shapeZK (Setup.ofCircuit sts[j]) sch extras =
      shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) := by
    simp only [shapeZK, shapeZKn, hx]
  rw [hsh] at hdec
  obtain ⟨zp', hdec', hio, -⟩ := hpads
  obtain rfl : zp = zp' := Except.ok.inj (hdec.symm.trans hdec')
  subst hio
  exact ⟨zp.inner.tau, zp.inner.tauSalt, rfl, (verifyInner_tau hinner).2⟩

end Logged

section View

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Custody

namespace Gen

variable [S : ZkSetup]

/-- **An accepting run's view logs its reps' `τ` openings**: at the view the deployed decoders are right at
`OpensRows` (`decodesTableR_deploy`), so each rep's openings are its proof file's inner-proof openings (`PadsRep`), and
the proof file is the run's, so the run's log holds their `τ` commitments' inputs (`zk_tauLogged`). -/
theorem zkViewTauLogged {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} {r : Fin ((V true).k x) → Coin}
    (hacc : (Claims.run sha512 P (V true) x ω r).out = .acc) {s' spec' sts' link c₁ mids f}
    (hact : act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
      .decide s' spec' sts' link c₁ mids f)
    {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (hret : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec' link c₁ f) a) :
    ∀ rr < 2, TauLogged (Claims.run sha512 P (V true) x ω r).vLog ((a.getD j dfltOutZC).opens.inner rr) := by
  obtain ⟨hV, pend, hpl, ha⟩ := hret
  subst a
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hrun : Runs sha512 (decision true ins x link c₁ mids f) ()
      ((Claims.run sha512 P (V true) x ω r).vLog.map Prod.fst) := by
    obtain ⟨s₂, spec₂, sts₂, link₂, c₂, mids₂, f₂, hact₂, hrun⟩ := acc_decide hacc
    rw [hact] at hact₂
    cases hact₂
    exact hrun
  have hjJ : j < (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := by
    rw [tabsJ_length]
    exact hj
  obtain ⟨jj, rfl⟩ : ∃ jj : Fin (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length, jj.val = j :=
    ⟨⟨j, hjJ⟩, rfl⟩
  rw [getD_ofFnJ]
  obtain ⟨-, spec₂, sts₂, sess, sch, tPad, _, _, hj₂, hsA₂, -, hsch, -, -, -, -, -, hp₀, hp₁, -⟩ :=
    Tables.DecodesTable.ok_of_plays (decodesTableR_deploy hs hin hset h0 L hz hk hlay hd
      (viewOf _ spec.mPts spec link c₁ f) jj) ⟨hV.1, rfl⟩ (pend jj) (hpl jj 0) (hpl jj 1)
  have hacc' := run_acc_next sha512 P (V true) x ω r hacc
  replace hacc' : ((next true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := hacc'
  obtain ⟨-, hc⟩ := (next_decide_sha512 hact).1 hacc'
  obtain ⟨hins, hsu⟩ := setupOf_some (act_decide hact)
  obtain ⟨hvr, -⟩ := decision_runs hsu hrun
  have hv' : Zk.verifyWithM (hashesIn ((Claims.run sha512 P (V true) x ω r).vLog.map Prod.fst)) spec
      (tabsOf sts) f.record f.proofs = .ok () := verdict_within hvr
  have hzk : ins.read.zk = true := readOf_zk (x := x) (by simp [readOf, hins])
  have hsA := setupAt_of_setup hsu hzk (publicInputs_none_of_checks hc)
  rw [hd] at hsA
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsA.symm.trans hsA₂))
  obtain ⟨hpr, -⟩ := recordsLiveAtS_wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f) hV.1
  obtain ⟨-, lv, hlv, -⟩ := checks_ok hc
  have hpr' : (wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  have hp2 := (HiddenExec.setupTables_pointsAny hsA).1
  rw [hpr'] at hp₀ hp₁
  intro rr hrr
  obtain rfl | rfl : rr = 0 ∨ rr = 1 := by omega
  · exact zk_tauLogged (rr := 0) hv' hlv hj₂ hsch hp2 (by omega) (by rw [Nat.add_zero]; exact hp₀)
  · exact zk_tauLogged (rr := 1) hv' hlv hj₂ hsch hp2 (by omega) hp₁

end Gen

/-- **An accepting run's view logs its reps' `τ` openings**: at the view the deployed decoders are right at
`OpensRows` (`decodesTableR_deploy`), so each rep's openings are its proof file's inner-proof openings (`PadsRep`), and
the proof file is the run's, so the run's log holds their `τ` commitments' inputs (`zk_tauLogged`). -/
theorem zkViewTauLogged {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} {r : Fin ((V true).k x) → Coin}
    (hacc : (Claims.run sha512 P (V true) x ω r).out = .acc) {s' spec' sts' link c₁ mids f}
    (hact : act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
      .decide s' spec' sts' link c₁ mids f)
    {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (hret : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec' link c₁ f) a) :
    ∀ rr < 2, TauLogged (Claims.run sha512 P (V true) x ω r).vLog ((a.getD j dfltOutZC).opens.inner rr) :=
  Gen.zkViewTauLogged (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (r := r) (hacc := hacc) (s' := s') (spec' := spec') (sts' := sts')
    (link := link) (c₁ := c₁) (mids := mids) (f := f) (hact := hact) (a := a) (hret := hret)

end View

section Pair

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

variable {D Sl : Type}

open Classical in
/-- **A `τ` opening that `opTauZ` accepts**: the output is accepted, and `τ`'s leaf under its salt is the cap's root. -/
theorem opTauZ_some {Hs : Hashes 𝔽 D Sl} {r : ℕ} {c : Model.Cap D} {out : OutZC 𝔽 D Sl}
    (h : (opTauZ Hs r c out ()).isSome) :
    out.accepted = true ∧ Hs.Lτ.leaf (out.opens.inner r).tau.1 (out.opens.inner r).tau.2 = c 0 ∧
      opTauZ Hs r c out () = some (out.opens.inner r).tau.1 := by
  by_cases ha : out.accepted = true ∧ Hs.Lτ.leaf (out.opens.inner r).tau.1 (out.opens.inner r).tau.2 = c 0
  · exact ⟨ha.1, ha.2, by rw [opTauZ, ite_eq_left ha]⟩
  · rw [opTauZ, ite_eq_right ha] at h
    simp at h

end Pair

section Live

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

namespace Gen

variable [S : ZkSetup]

/-- **A decoded live outcome of the masked simulator logs an accepting run's `τ` openings**: where the live verdict
holds, `P`'s run at `zkΦ`'s coins accepts (`zkSimAbortAt`), the outcome is the view of the record it decides
(`zkSimRunAt`), and table `j`'s decoded output's `τ` openings are logged in the run (`zkViewTauLogged`). -/
theorem zkAccTau {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (ha : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r') a) :
    (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc ∧ ∀ rr < 2,
      TauLogged (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).vLog
        ((a.getD j dfltOutZC).opens.inner rr) := by
  have hacc : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc := by
    by_contra hn
    exact zkSimAbortAt x s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r' hn ha.1
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, -⟩ := acc_decide hacc
  have hv := zkSimRunAt x s hs hin ins spec sts hset h0 L hz hk hlay Ω P ω u r' hacc s' spec' sts' link c₁ mids f hact
  rw [hv] at ha
  exact ⟨hacc, zkViewTauLogged hs hin hset h0 L hz hk hlay hd hj hacc hact ha⟩

end Gen

/-- **A decoded live outcome of the masked simulator logs an accepting run's `τ` openings**: where the live verdict
holds, `P`'s run at `zkΦ`'s coins accepts (`zkSimAbortAt`), the outcome is the view of the record it decides
(`zkSimRunAt`), and table `j`'s decoded output's `τ` openings are logged in the run (`zkViewTauLogged`). -/
theorem zkAccTau {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (ha : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r') a) :
    (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc ∧ ∀ rr < 2,
      TauLogged (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).vLog
        ((a.getD j dfltOutZC).opens.inner rr) :=
  Gen.zkAccTau (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (hC := hC) (u := u) (r' := r') (a := a) (ha := ha)

namespace Gen

variable [S : ZkSetup]

/-- **The live `τ` fork event's collision, with no hypothesis**: two live outcomes of the masked simulator at one
tape whose table `j` outputs are accepted and open one cap's `τ` commitment at rep `r` to different values are two
accepting runs of `P` at `zkΦ`'s coins (`zkAccTau`) whose openings there have one leaf, the cap's root
(`opTauZ_some`), so they collide in the two runs' logs (`tiedColl_of_tauLogged`). -/
theorem zkTauClash (x : Claims.Msg) : ZkTauClash x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r₁ r₂ r h _ _
  obtain ⟨a, a', ha, ha', dc, hs₁, hs₂, hne⟩ := h
  dsimp only at hs₁ hs₂ hne
  obtain ⟨-, hl₁⟩ := zkAccTau hs hin hset h0 L hz hk hlay hd hj hC u r₁ ha
  obtain ⟨-, hl₂⟩ := zkAccTau hs hin hset h0 L hz hk hlay hd hj hC u r₂ ha'
  obtain ⟨-, hv₁, he₁⟩ := opTauZ_some hs₁
  obtain ⟨-, hv₂, he₂⟩ := opTauZ_some hs₂
  exact tiedColl_of_tauLogged (hl₁ r.val r.isLt) (hl₂ r.val r.isLt) (hv₁.trans hv₂.symm)
    fun e => hne (by rw [he₁, he₂, e])

end Gen

/-- **The live `τ` fork event's collision, with no hypothesis**: two live outcomes of the masked simulator at one
tape whose table `j` outputs are accepted and open one cap's `τ` commitment at rep `r` to different values are two
accepting runs of `P` at `zkΦ`'s coins (`zkAccTau`) whose openings there have one leaf, the cap's root
(`opTauZ_some`), so they collide in the two runs' logs (`tiedColl_of_tauLogged`). -/
theorem zkTauClash (x : Claims.Msg) : ZkTauClash x :=
  Gen.zkTauClash (S := S₀) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **The fork bound at `τ` from the alignment**: `ZkSimForkLeOn` at the `τ` events from `ZkTauAlign`, the clash
holding outright (`zkTauClash`). -/
theorem zkSimForkLeTau_of_align {x : Claims.Msg} (ha : ZkTauAlign x) :
    ZkSimForkLeOn x fun _ i => ∃ r, i = .inl (.inr (.inr r)) :=
  zkSimForkLeTau_of ha (zkTauClash x)

end Gen

/-- **The fork bound at `τ` from the alignment**: `ZkSimForkLeOn` at the `τ` events from `ZkTauAlign`, the clash
holding outright (`zkTauClash`). -/
theorem zkSimForkLeTau_of_align {x : Claims.Msg} (ha : ZkTauAlign x) :
    ZkSimForkLeOn x fun _ i => ∃ r, i = .inl (.inr (.inr r)) :=
  Gen.zkSimForkLeTau_of_align (S := S₀) (x := x) (ha := ha)

end Live

end FlockDeploy.KS
