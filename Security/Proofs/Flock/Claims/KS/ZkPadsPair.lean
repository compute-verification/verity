import Proofs.Flock.Claims.KS.ZkClashPads

/-!
# The masked pads fork's two-run clash

`ZkSimForkLeOn x IsPadsEvZ` is the masked simulator's fork bound at each rep's pads. It splits as the caps' does
(`ZkFork`): an alignment of the simulated fork at rep `r`'s pads node with the fork at the pads round, and the clash of
the live fork event, two accepting runs whose decoded outputs open a position of one cap `d`'s pads tree to different
pairs (`PadsForkEvZ`). This file proves the clash with no hypothesis and leaves the alignment as a premise
(`ZkPadsAlign`).

The clash reads the model's openings, not a shared root: an opening that `opPadsAtZ` accepts verifies at the query's
position to `d 0` (`opPadsAtZ_some`), and at SHA-512 that is the executable's `hm96-sha512/v1` climb from the pair's
leaf reaching `d 0`'s bytes (`climb_of_verifiesL`). At an accepting run's view each rep's pads openings are its
proof file's inner-proof openings at the positions its stream's last round's coins give (`PadsRep`), and the run's
verifier hashed every input of each of them (`verifyInner_logs`; `PadsLogged`). Two openings at one position climbing
to one top with different pairs collide in the two runs' joined logs (`tiedColl_of_climbs`; `tiedColl_of_padsLogged`).

* `zkViewPadsLogged`, `zkAccPads`: an accepting run's view, and a decoded live outcome of the masked simulator, log
  their reps' pads openings.
* `zkPairPads`: the live pads fork event gives the deployed runs' pair event.
* `opPadsAtZ_nul`: the dead branches' empty openings open no pads position (`2 ≤ N`, `two_le_zkTabsN`).
* `zkSimForkLePads_of`, `zk_KS_of_padsAlign`: the pads fork bound, and masked KS, from the pads alignment.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Logged

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.ZkExec (innerN layZ shapeZK shapeZKn innerOpensOf PadsRep zk_decode_paths hs512
  encPair leaf_pair walk_pair)
open Claims.Bridge.Custody

/-- **A rep's pads openings, logged**: at each of the `Q` queries the opening is a pair, its salt and a path of
digest-width siblings, every SHA-512 input of whose `hm96-sha512/v1` opening at the query's position is in `L`. -/
def PadsLogged (L : List (Claims.Msg × Claims.Msg)) (io : ZkLink.InnerOpens 𝔽 Dig (List UInt8)) (qpos : ℕ → ℕ) :
    Prop :=
  ∀ k < Zk.Q, ∃ (o : F128 × F128) (salt : ByteArray) (path : Array ByteArray),
    io.pads k = (pairOf o, salt.data.toList, path.toList.map toDig) ∧ (∀ b ∈ path.toList, b.size = 64) ∧
      ∀ m ∈ opensIns #[o.1, o.2] salt (qpos k) path.toList, Claims.toMsg m ∈ L.map Prod.fst

/-- **A pads opening the model verifies is an executable climb**: at SHA-512, an opening of the pair `o` under `salt`
along digest-width siblings that verifies at depth `n` to the cap `fun _ => R` has `n` siblings, and its
`hm96-sha512/v1` climb from the pair's leaf reaches `R`'s bytes. -/
theorem climb_of_verifiesL {R : Dig} {n pos : ℕ} {o : F128 × F128} {salt : ByteArray} {path : Array ByteArray}
    (hw : ∀ b ∈ path.toList, b.size = 64)
    (h : Merkle.VerifiesL hs512.H hs512.Ep hs512.Lp (fun _ => R) n pos (pairOf o) salt.data.toList
      (path.toList.map toDig)) :
    path.toList.length = n ∧
      climb MerkleScheme.hm96Sha512 (MerkleScheme.hm96Sha512.leaf #[o.1, o.2] salt) pos path.toList = R.1 := by
  obtain ⟨hlen, hr⟩ := h
  refine ⟨by simpa using hlen, ?_⟩
  have hmap : (path.toList.map toDig).map (·.1) = path.toList := by
    rw [List.map_map]
    exact (List.map_congr_left fun b hb => toDig_val (hw b hb)).trans (List.map_id _)
  have hlf : Hm96.Default512.leaf (Sha512.hash (o.1.toBytes ++ o.2.toBytes)) salt =
      MerkleScheme.hm96Sha512.leaf #[o.1, o.2] salt := by
    show _ = Hm96.Default512.leaf (Sha512.hash (MerkleScheme.rowBytes _)) _
    rw [rowBytes_pair]
  have e : (Merkle.walk H512 encPair pos ((Merkle.Leaf.hm96 H512 encPair K512 hL512).leaf (pairOf o)
      salt.data.toList) (path.toList.map toDig)).1 = R.1 := congrArg Subtype.val hr
  rw [walk_pair, walk_val "", hmap, leaf_pair, ← climb_hm96 "", hlf] at e
  exact e

/-- **Two logged pads openings at one position that verify to one cap with different pairs collide in the joined
logs**: both climb to the cap's bytes along paths of the depth's length (`climb_of_verifiesL`), and every input of
each is in its own log (`tiedColl_of_climbs`). -/
theorem tiedColl_of_padsLogged {L₁ L₂ : List (Claims.Msg × Claims.Msg)}
    {io₁ io₂ : ZkLink.InnerOpens 𝔽 Dig (List UInt8)} {qpos₁ qpos₂ : ℕ → ℕ} (h₁ : PadsLogged L₁ io₁ qpos₁)
    (h₂ : PadsLogged L₂ io₂ qpos₂) {k₁ k₂ : ℕ} (hk₁ : k₁ < Zk.Q) (hk₂ : k₂ < Zk.Q) {R : Dig} {n p : ℕ}
    (hq₁ : qpos₁ k₁ = p) (hq₂ : qpos₂ k₂ = p)
    (hv₁ : Merkle.VerifiesL hs512.H hs512.Ep hs512.Lp (fun _ => R) n p (io₁.pads k₁).1 (io₁.pads k₁).2.1
      (io₁.pads k₁).2.2)
    (hv₂ : Merkle.VerifiesL hs512.H hs512.Ep hs512.Lp (fun _ => R) n p (io₂.pads k₂).1 (io₂.pads k₂).2.1
      (io₂.pads k₂).2.2)
    (hne : (io₁.pads k₁).1 ≠ (io₂.pads k₂).1) : Claims.TiedColl sha512 (L₁ ++ L₂) := by
  obtain ⟨o₁, salt₁, path₁, he₁, hw₁, hl₁⟩ := h₁ k₁ hk₁
  obtain ⟨o₂, salt₂, path₂, he₂, hw₂, hl₂⟩ := h₂ k₂ hk₂
  rw [he₁] at hv₁ hne
  rw [he₂] at hv₂ hne
  rw [hq₁] at hl₁
  rw [hq₂] at hl₂
  obtain ⟨hn₁, hc₁⟩ := climb_of_verifiesL hw₁ hv₁
  obtain ⟨hn₂, hc₂⟩ := climb_of_verifiesL hw₂ hv₂
  refine tiedColl_of_climbs hw₁ hw₂ (hn₁.trans hn₂.symm) (hc₁.trans hc₂.symm) (fun e => hne ?_)
    (fun b hb => by rw [List.map_append]; exact List.mem_append_left _ (hl₁ b hb))
    (fun b hb => by rw [List.map_append]; exact List.mem_append_right _ (hl₂ b hb))
  have e' : o₁ = o₂ := Prod.ext (by simpa using congrArg (fun r : Array F128 => r[0]!) e)
    (by simpa using congrArg (fun r : Array F128 => r[1]!) e)
  rw [e']

/-- **An accepted masked rep logs its pads openings**: where a stream `y` has the coins of the live session's rep
stream `2j + r` and rep `r`'s proof file decodes to pads openings drawn at the positions `y`'s last round's coins give
(`PadsRep`), the run's verifier opened each of those queries at that position (`zkVerifyRepWith_inner`,
`verifyInner_logs`), so every input of each opening is in its log. -/
theorem zk_padsLogged {L : List (Claims.Msg × Claims.Msg)} {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live} {j rr : ℕ} {sch : Schedule} {tPad : ℕ}
    (hv : Zk.verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) record proofs = .ok ())
    (hlv : liveOf spec record = .ok lv) (hj : j < sts.size) (hsch : Zk.schedule sts[j].m = .ok (sch, tPad))
    (hp2 : spec.points = 2) (hrr : rr < 2) {y : StreamRec}
    (hy : ∀ i : ℕ, y.rounds[i]?.map (fun rd : Flock.Round => rd.coins) =
      (lv.streams.getD (2 * j + rr) #[])[i]?.map (fun rd : Flock.Round => rd.coins))
    {io : ZkLink.InnerOpens 𝔽 Dig (List UInt8)} {qpos : ℕ → ℕ}
    (hpads : PadsRep (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) y proofs[2 * j + rr]! io qpos) :
    PadsLogged L io qpos := by
  obtain ⟨hsz, hall⟩ := rep_zk _ hv hlv
  have hkk : 2 * j + rr < lv.streams.size := by omega
  obtain ⟨sess, sch', tPad', stream, c, hpts, hpn, hsch', hrep, hfind, hfa⟩ := hall (2 * j + rr) hkk
  have hj' : sts[(2 * j + rr) / 2]! = sts[j] := by
    rw [show (2 * j + rr) / 2 = j by omega]
    exact getElem!_pos sts j hj
  rw [hj', hsch] at hsch'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hsch')
  rw [hj'] at hrep
  have hsch₀ : Zk.schedule (Setup.ofCircuit sts[j]).m = .ok (sch, tPad) := hsch
  obtain ⟨stream', extras, zp, finalC, cons, t, t', hfind', hex, hdec, htr, htk, hinner⟩ :=
    zkVerifyRepWith_inner _ hsch₀ hrep
  obtain rfl : stream = stream' := Option.some.inj (hfind.symm.trans hfind')
  rw [hpts] at hex
  have hx : extras.size = 2 * sts[j].regions.size := by
    rw [extraClaims_size (st := sts[j]) hex, ← hpts, hpn, hp2, Nat.mul_comm]
  have hsh : shapeZK (Setup.ofCircuit sts[j]) sch extras =
      shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) := by
    simp only [shapeZK, shapeZKn, hx]
  rw [hsh] at hdec
  obtain ⟨zp', hdec', hio, hqpos⟩ := hpads
  obtain rfl : zp = zp' := Except.ok.inj (hdec.symm.trans hdec')
  subst hio
  have hgd : lv.streams.getD (2 * j + rr) #[] = lv.streams[2 * j + rr] := by
    simp [Array.getD, hkk]
  have hcoin : (stream.rounds[stream.rounds.size - 1]!).coins = (y.rounds[y.rounds.size - 1]!).coins := by
    refine coins_last fun i => ?_
    have h1 := forall₂_coins (fun _ _ h => h.2) hfa i
    rw [Array.getElem?_toList, Array.getElem?_toList] at h1
    rw [hy i, hgd]
    exact h1.symm
  have hcoin' : (t.rounds[t.k + 3]!).coins = (y.rounds[y.rounds.size - 1]!).coins := by
    rw [htr, show t.k + 3 = stream.rounds.size - 1 by omega, hcoin]
  obtain ⟨-, -, hps, hlog⟩ := verifyInner_logs hinner
  intro k hk
  obtain ⟨-, -, hl⟩ := hlog k hk
  rw [hcoin', ← hqpos k hk] at hl
  exact ⟨zp.inner.opened[k]!, zp.inner.salts[k]!, zp.inner.paths[k]!, rfl,
    zk_decode_paths hdec k (by omega), hl⟩

end Logged

section Clash

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Custody

namespace Gen

variable [S : ZkSetup]

/-- **An accepting run's view logs its reps' pads openings**: at the view the deployed decoders are right at
`OpensRows` (`decodesTableR_deploy`), so table `j` reads `Q` queries and each rep's pads openings are its proof file's
inner-proof openings at the positions its rep stream's last round's coins give; that stream has the run's rep stream's
coins (`recordsLiveAtS_wrS`) and the proof file is the run's, so the run's log holds their inputs (`zk_padsLogged`). -/
theorem zkViewPadsLogged {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
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
    (zkTabs h0 L hz hk hlay j).q = Zk.Q ∧ ∀ rr < 2,
      PadsLogged (Claims.run sha512 P (V true) x ω r).vLog ((a.getD j dfltOutZC).opens.inner rr)
        (if rr = 0 then (a.getD j dfltOutZC).pend.1 else (a.getD j dfltOutZC).pend.2).qpos := by
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
  obtain ⟨-, spec₂, sts₂, sess, sch, tPad, s₀, s₁, hj₂, hsA₂, hdec, hsch, hs₀, hs₁, -, -, hq, hp₀, hp₁, -⟩ :=
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
  obtain ⟨hpr, hall⟩ := recordsLiveAtS_wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f) hV.1
  obtain ⟨-, hrs⟩ := hall spec sts sess hsA hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs jj).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs jj).2 ⟨1, by omega⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain rfl : y₁ = s₁ := Option.some.inj (hy₁.symm.trans hs₁)
  obtain ⟨-, lv, hlv, -⟩ := checks_ok hc
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (q : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[q]!).toList.map (withSha spec) →
      ∀ i : ℕ, y.rounds[i]?.map (fun rd : Flock.Round => rd.coins) =
        (lv.streams.getD q #[])[i]?.map (fun rd : Flock.Round => rd.coins) := by
    intro q y hy i
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  have hp2 := (HiddenExec.setupTables_pointsAny hsA).1
  rw [hpr'] at hp₀ hp₁
  rw [tabsJ_get] at hq
  refine ⟨hq, fun rr hrr => ?_⟩
  obtain rfl | rfl : rr = 0 ∨ rr = 1 := by omega
  · rw [ite_eq_left rfl]
    exact zk_padsLogged (rr := 0) (y := y₀) hv' hlv hj₂ hsch hp2 (by omega)
      (by rw [Nat.add_zero]; exact hco _ _ hr₀) (by rw [Nat.add_zero]; exact hp₀)
  · rw [ite_eq_right (show (1 : ℕ) ≠ 0 by omega)]
    exact zk_padsLogged (rr := 1) (y := y₁) hv' hlv hj₂ hsch hp2 (by omega) (hco _ _ hr₁) hp₁

end Gen

/-- **An accepting run's view logs its reps' pads openings**: at the view the deployed decoders are right at
`OpensRows` (`decodesTableR_deploy`), so table `j` reads `Q` queries and each rep's pads openings are its proof file's
inner-proof openings at the positions its rep stream's last round's coins give; that stream has the run's rep stream's
coins (`recordsLiveAtS_wrS`) and the proof file is the run's, so the run's log holds their inputs (`zk_padsLogged`). -/
theorem zkViewPadsLogged {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
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
    (zkTabs h0 L hz hk hlay j).q = Zk.Q ∧ ∀ rr < 2,
      PadsLogged (Claims.run sha512 P (V true) x ω r).vLog ((a.getD j dfltOutZC).opens.inner rr)
        (if rr = 0 then (a.getD j dfltOutZC).pend.1 else (a.getD j dfltOutZC).pend.2).qpos :=
  Gen.zkViewPadsLogged (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (r := r) (hacc := hacc) (s' := s') (spec' := spec') (sts' := sts')
    (link := link) (c₁ := c₁) (mids := mids) (f := f) (hact := hact) (a := a) (hret := hret)

end Clash

section Pair

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

/-- **The pads fork event at rep `r` of table `j`**: table `j`'s outputs in both runs open a position of one cap
`d`'s pads tree to different pairs. -/
def PadsForkEvZ (Hs : Hashes 𝔽 D Sl) (N q j r : ℕ) (a a' : List (OutZC 𝔽 D Sl)) : Prop :=
  ∃ (d : Model.Cap D) (p : Fin N), Rewinding.ConflictAt (fun outs : List (OutZC 𝔽 D Sl) =>
    opPadsAtZ Hs N q r d (outs.getD j dfltOutZC)) (a, a') p

/-- Table `j`'s output opens a position of some cap's pads tree at rep `r`. -/
def OpensPadsZ (Hs : Hashes 𝔽 D Sl) (N q j r : ℕ) (a : List (OutZC 𝔽 D Sl)) : Prop :=
  ∃ (d : Model.Cap D) (p : Fin N), (opPadsAtZ Hs N q r d (a.getD j dfltOutZC) p).isSome

theorem padsForkEvZ_opens {Hs : Hashes 𝔽 D Sl} {N q j r : ℕ} {a a' : List (OutZC 𝔽 D Sl)}
    (h : PadsForkEvZ Hs N q j r a a') : OpensPadsZ Hs N q j r a ∧ OpensPadsZ Hs N q j r a' :=
  let ⟨d, p, h₁, h₂, _⟩ := h
  ⟨⟨d, p, h₁⟩, ⟨d, p, h₂⟩⟩

omit [Inhabited D] in
/-- **The dead branches' empty openings open no pads position**: their paths are empty, and a tree of `N ≥ 2` leaves
has depth at least one. -/
theorem opPadsAtZ_nul (Hs : Hashes 𝔽 D Sl) {N : ℕ} (hN : 2 ≤ N) (q r : ℕ) (c : Model.Cap D)
    {out : OutZC 𝔽 D Sl} (h : out.opens = nulOpZ) (p : Fin N) : opPadsAtZ Hs N q r c out p = none := by
  unfold opPadsAtZ
  rw [List.filter_eq_nil_iff.2 fun jj _ => ?_]
  · rfl
  · rw [h]
    intro hd
    simp only [decide_eq_true_eq] at hd
    obtain ⟨-, hv⟩ := hd
    have h1 : ([] : List D).length = Nat.log 2 N := hv.1
    have h2 : 0 < Nat.log 2 N := Nat.log_pos (by norm_num) hN
    simp only [List.length_nil] at h1
    omega

theorem not_opensPadsZ_nul {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts)) {N : ℕ} (hN : 2 ≤ N)
    (q : ℕ) {j : ℕ} (hj : j < ts.length) (r : ℕ) (caps : Fin ts.length → Model.Cap D)
    (outs : List (OutZC 𝔽 D Sl))
    (h : ∀ i : Fin ts.length, ∃ pend, outs.getD i dfltOutZC = outJ Hs ts i (caps i) pend nulOpZ) :
    ¬ OpensPadsZ Hs N q j r outs := by
  rintro ⟨d, p, hs⟩
  obtain ⟨pend, hp⟩ := h ⟨j, hj⟩
  have hp' : outs.getD j dfltOutZC = outJ Hs ts ⟨j, hj⟩ (caps ⟨j, hj⟩) pend nulOpZ := hp
  rw [hp', opPadsAtZ_nul Hs hN q r d rfl] at hs
  exact Bool.false_ne_true hs

/-- The deployed masked tables' pads trees have at least `K_PAD` leaves. -/
theorem two_le_zkTabsN {sts : Array Stmt} {mPts : ℕ} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts) (j : ℕ) :
    2 ≤ (zkTabs h0 L hz hk hlay j).N := by
  have h := (zkTabs h0 L hz hk hlay j).hL
  have e : (zkTabs h0 L hz hk hlay j).nP = Zk.K_PAD := by
    unfold zkTabs
    split <;> rfl
  rw [e, show Zk.K_PAD = 192 from rfl] at h
  omega

end Pair

section Live

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

namespace Gen

variable [S : ZkSetup]

/-- **A decoded live outcome of the masked simulator logs an accepting run's pads openings**: where the live verdict
holds, `P`'s run at `zkΦ`'s coins accepts (`zkSimAbortAt`), the outcome is the view of the record it decides
(`zkSimRunAt`), and table `j`'s decoded output's pads openings are logged in the run (`zkViewPadsLogged`). -/
theorem zkAccPads {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
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
    (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc ∧
      (zkTabs h0 L hz hk hlay j).q = Zk.Q ∧ ∀ rr < 2,
        PadsLogged (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).vLog
          ((a.getD j dfltOutZC).opens.inner rr)
          (if rr = 0 then (a.getD j dfltOutZC).pend.1 else (a.getD j dfltOutZC).pend.2).qpos := by
  have hacc : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc := by
    by_contra hn
    exact zkSimAbortAt x s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r' hn ha.1
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, -⟩ := acc_decide hacc
  have hv := zkSimRunAt x s hs hin ins spec sts hset h0 L hz hk hlay Ω P ω u r' hacc s' spec' sts' link c₁ mids f hact
  rw [hv] at ha
  exact ⟨hacc, zkViewPadsLogged hs hin hset h0 L hz hk hlay hd hj hacc hact ha⟩

end Gen

/-- **A decoded live outcome of the masked simulator logs an accepting run's pads openings**: where the live verdict
holds, `P`'s run at `zkΦ`'s coins accepts (`zkSimAbortAt`), the outcome is the view of the record it decides
(`zkSimRunAt`), and table `j`'s decoded output's pads openings are logged in the run (`zkViewPadsLogged`). -/
theorem zkAccPads {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
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
    (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc ∧
      (zkTabs h0 L hz hk hlay j).q = Zk.Q ∧ ∀ rr < 2,
        PadsLogged (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).vLog
          ((a.getD j dfltOutZC).opens.inner rr)
          (if rr = 0 then (a.getD j dfltOutZC).pend.1 else (a.getD j dfltOutZC).pend.2).qpos :=
  Gen.zkAccPads (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (hC := hC) (u := u) (r' := r') (a := a) (ha := ha)

namespace Gen

variable [S : ZkSetup]

/-- **The live pads fork event's pair**: two live outcomes of the masked simulator at one tape decoding to outputs
that open a position of one cap's pads tree at rep `r` of table `j` to different pairs are two accepting runs of `P`
at `zkΦ`'s coins (`zkAccPads`) whose openings there both climb to the cap's bytes, so they collide in the two runs'
logs (`tiedColl_of_padsLogged`). -/
theorem zkPairPads {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    {u : zkTape x} {r₁ r₂ : Fin (zkN x + 2) → LCoin 65536} {r : Fin 2}
    (h : LiveEvZ (Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!)
        (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))))
      (fun y => PadsForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).N (zkTabs h0 L hz hk hlay j).q j r.val y.1 y.2)
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r₁)
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r₂)) :
    Claims.Flock.pairEv sha512 (V true) P x ω (zkΦ x spec sts (u, r₁)) (zkΦ x spec sts (u, r₂)) := by
  obtain ⟨a, a', ha, ha', dc, p, hs₁, hs₂, hne⟩ := h
  dsimp only at hs₁ hs₂ hne
  obtain ⟨hacc₁, hq, hl₁⟩ := zkAccPads hs hin hset h0 L hz hk hlay hd hj hC u r₁ ha
  obtain ⟨hacc₂, -, hl₂⟩ := zkAccPads hs hin hset h0 L hz hk hlay hd hj hC u r₂ ha'
  obtain ⟨k₁, hk₁, hq₁, hv₁, he₁⟩ := opPadsAtZ_some hs₁
  obtain ⟨k₂, hk₂, hq₂, hv₂, he₂⟩ := opPadsAtZ_some hs₂
  rw [hq] at hk₁ hk₂
  refine ⟨hacc₁, hacc₂, tiedColl_of_padsLogged (hl₁ r.val r.isLt) (hl₂ r.val r.isLt) hk₁ hk₂ hq₁ hq₂ hv₁ hv₂
    fun e => hne ?_⟩
  rw [he₁, he₂, e]

end Gen

/-- **The live pads fork event's pair**: two live outcomes of the masked simulator at one tape decoding to outputs
that open a position of one cap's pads tree at rep `r` of table `j` to different pairs are two accepting runs of `P`
at `zkΦ`'s coins (`zkAccPads`) whose openings there both climb to the cap's bytes, so they collide in the two runs'
logs (`tiedColl_of_padsLogged`). -/
theorem zkPairPads {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    {u : zkTape x} {r₁ r₂ : Fin (zkN x + 2) → LCoin 65536} {r : Fin 2}
    (h : LiveEvZ (Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!)
        (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))))
      (fun y => PadsForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).N (zkTabs h0 L hz hk hlay j).q j r.val y.1 y.2)
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r₁)
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r₂)) :
    Claims.Flock.pairEv sha512 (V true) P x ω (zkΦ x spec sts (u, r₁)) (zkΦ x spec sts (u, r₂)) :=
  Gen.zkPairPads (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (hC := hC) (u := u) (r₁ := r₁) (r₂ := r₂) (r := r) (h := h)

open Classical in
/-- **The simulated fork's alignment at rep `r`'s pads at tape `u`**: against the simulated strategy after `Commit`
at table `j`, the fork at rep `r`'s pads node (`advPadsSZ`, `zkAdvSim`'s pads term) is at most the fork after `t`
coins of `PadsForkEvZ`. -/
def ZkPadsAlignQ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (t : ℕ)
    (u : zkTape x) : Prop :=
  advPadsSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _)) ≤
    forkC (fun a a' => ind (fun p : List (OutZC 𝔽 Refine.Dig (List UInt8)) ×
        List (OutZC 𝔽 Refine.Dig (List UInt8)) =>
      PadsForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).N (zkTabs h0 L hz hk hlay j).q j r.val p.1 p.2) (a, a')) t
      _ (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **The simulated fork's alignment at rep `r`'s pads at tape `u`**: against the simulated strategy after `Commit`
at table `j`, the fork at rep `r`'s pads node (`advPadsSZ`, `zkAdvSim`'s pads term) is at most the fork after `t`
coins of `PadsForkEvZ`. -/
def ZkPadsAlignQ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (t : ℕ)
    (u : zkTape x) : Prop :=
  advPadsSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _ (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _)) ≤
    forkC (fun a a' => ind (fun p : List (OutZC 𝔽 Refine.Dig (List UInt8)) ×
        List (OutZC 𝔽 Refine.Dig (List UInt8)) =>
      PadsForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).N (zkTabs h0 L hz hk hlay j).q j r.val p.1 p.2) (a, a')) t
      _ (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))

end Gen

/-- **The simulated fork is aligned with the masked fork rounds at the pads**: at every masked statement in scope
that sets up at a draw, every prover and tape, every table `j` and table of columns, at each tape the fork at rep
`r`'s pads node against the simulated strategy after `Commit` is at most the fork after `zkRd j i` coins of
`PadsForkEvZ` (`ZkPadsAlignQ`). -/
def ZkPadsAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (u : zkTape x),
    ZkPadsAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r
      (zkRd hset h0 L hz hk hlay j (.inl (.inr (.inl r)))).val u

namespace Gen

variable [S : ZkSetup]

/-- **The simulated fork is aligned with the masked fork rounds at the pads**: at every masked statement in scope
that sets up at a draw, every prover and tape, every table `j` and table of columns, at each tape the fork at rep
`r`'s pads node against the simulated strategy after `Commit` is at most the fork after `zkRd j i` coins of
`PadsForkEvZ` (`ZkPadsAlignQ`). -/
def ZkPadsAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (r : Fin 2) (u : zkTape x),
    ZkPadsAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r
      (zkRd hset h0 L hz hk hlay j (.inl (.inr (.inl r)))).val u

open Classical in
/-- **The fork advantage at rep `r`'s pads from the alignment**: `zkAdvSim` at the pads is at most the simulated fork
at round `t`, where at each tape the fork at the pads node is at most the fork after `t` coins of `PadsForkEvZ`: that
fork is at most the live fork of the decoded event (`forkC_simAtJZ_le`, the dead branches opening no pads position,
`not_opensPadsZ_nul`), the deployed runs' fork at round `t` (`forkC_liveStratZ`), whose event is the runs' pair event
(`zkPairPads`). -/
theorem zkAdvPads_le_of {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (hC : ∀ p, wideJ true x j (C p))
    (r : Fin 2) (t : Fin ((V true).k x))
    (halign : ∀ u, ZkPadsAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r t.val u) :
    zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C (.inl (.inr (.inl r))) ≤
      Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω t.val := by
  have hk' := zkN_add hset
  have hj' : j < (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := (finJ _ hj).isLt
  unfold Claims.Flock.simForkPr
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((halign u).trans (forkC_simAtJZ_le execArith ZkExec.hs512 _
    (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C) (AccTxJ ins.read.I d j (zkRoot P x ω) C)
    (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
    (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω) _
    (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
    (tabsJ_get _ _ _)
    (Ev := fun p => PadsForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).N (zkTabs h0 L hz hk hlay j).q j r.val
      p.1 p.2)
    (Q := OpensPadsZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).N (zkTabs h0 L hz hk hlay j).q j r.val)
    (fun _ h => padsForkEvZ_opens h)
    (fun caps outs h => not_opensPadsZ_nul ZkExec.hs512 _ (two_le_zkTabsN h0 L hz hk hlay j) _ hj' r.val caps outs
      h) _)).trans ?_
  refine le_trans (le_of_eq (forkC_liveStratZ zkN spec.dg
    (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
    (zkSim P x spec sts) x (ω, u) _ ⟨t.val, by have := t.isLt; omega⟩)) ?_
  exact Claims.Ex_mono fun rr => ind_le_indZ fun h => zkPairPads hs hin hset h0 L hz hk hlay hd hj hC h

end Gen

open Classical in
/-- **The fork advantage at rep `r`'s pads from the alignment**: `zkAdvSim` at the pads is at most the simulated fork
at round `t`, where at each tape the fork at the pads node is at most the fork after `t` coins of `PadsForkEvZ`: that
fork is at most the live fork of the decoded event (`forkC_simAtJZ_le`, the dead branches opening no pads position,
`not_opensPadsZ_nul`), the deployed runs' fork at round `t` (`forkC_liveStratZ`), whose event is the runs' pair event
(`zkPairPads`). -/
theorem zkAdvPads_le_of {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) (hC : ∀ p, wideJ true x j (C p))
    (r : Fin 2) (t : Fin ((V true).k x))
    (halign : ∀ u, ZkPadsAlignQ hs hin hset h0 L hz hk hlay hd P ω j hj C r t.val u) :
    zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C (.inl (.inr (.inl r))) ≤
      Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω t.val :=
  Gen.zkAdvPads_le_of (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (hC := hC) (r := r) (t := t) (halign := halign)

namespace Gen

variable [S : ZkSetup]

/-- **The fork bound at the pads from the alignment**: `ZkSimForkLeOn` at `IsPadsEvZ` from `ZkPadsAlign`, the clash
holding outright (`zkAdvPads_le_of`). -/
theorem zkSimForkLePads_of {x : Claims.Msg} (ha : ZkPadsAlign x) : ZkSimForkLeOn x fun _ i => IsPadsEvZ i := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC i hi
  rcases i with ((_ | (r | _)) | ⟨⟩)
  · exact False.elim hi
  · exact zkAdvPads_le_of hs hin hset h0 L hz hk hlay hd P ω j hj C hC r _
      (ha s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C r)
  · exact False.elim hi
  · exact False.elim hi

end Gen

/-- **The fork bound at the pads from the alignment**: `ZkSimForkLeOn` at `IsPadsEvZ` from `ZkPadsAlign`, the clash
holding outright (`zkAdvPads_le_of`). -/
theorem zkSimForkLePads_of {x : Claims.Msg} (ha : ZkPadsAlign x) : ZkSimForkLeOn x fun _ i => IsPadsEvZ i :=
  Gen.zkSimForkLePads_of (S := S₀) (x := x) (ha := ha)

end Live

open Flock

/-- **The deployed masked verifier: KS, from the cap/`τ` fork bound and the pads alignment**, at SHA-512 at the
trimmed record at the deployed relation: `zk_KS_of_forks` at `zkSimForkLePads_of`. -/
theorem zk_KS_of_padsAlign
    (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hpa : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkPadsAlign x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_forks hct fun x hno => zkSimForkLePads_of (hpa x hno)

end FlockDeploy.KS
