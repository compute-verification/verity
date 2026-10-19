import Proofs.Flock.Claims.KS.ZkCaps
import Proofs.Flock.Claims.KS.ZkFork
import Proofs.Flock.Claims.KS.PlainForkClash

/-!
# The masked live fork event's collision

`ZkForkClash` (`ZkFork`) asks that two accepting runs of `P` at the masked simulator's coins whose decoded live
outcomes have `ForkEvZ` at table `j`'s rep `r`, level `ℓ + 1`, collide among the verifier's calls in the two. This file
proves it with no hypothesis, so the masked fork bound at the caps rests on its alignment alone
(`zkSimForkLeCap_of_align`).

The clash reads each run's view, as plain's (`PlainForkClash`) does. At an accepting run's view the deployed decoders
are right at `OpensRows` and at `CapsRows` (`decodesTableR_deploy`, `decodesTableC_deploy`), so rep `r`'s recorded cap
`ℓ` is the `capDig` of its proof file's `ℓ`-th recursive cap, and its level-`ℓ + 1` openings are the rep's
`zkLevelRowsOf` rows on the run's own stream and proof file, each an `Opening` in the run's log against the flattened
cap (`zk_levelOpening`). One recorded cap in both runs is one cap of 64-byte digests (`capDig_inj`), so two rows opened
at one position of it that differ collide in the joined logs (`tiedColl_of_openings`).

* `zk_levelOpening`: an accepted masked session's level-`ℓ` row of table `j`'s rep `k` is an `Opening` in the log
  against the flattened `ℓ − 1`-th recursive cap of that rep's decoded proof.
* `capDig_of_capAtZ`, `opRZ_eq_some`, `forkEvZ_entries`: a recorded cap read off the rep's caps, and `ForkEvZ` as one
  recorded cap and two openings of one position of it to different rows.
* `CapOpens`, `zkViewCapOpens`, `zkAccCapOpens`: an accepting run's view, and a decoded live outcome of the masked
  simulator, record their reps' proof files' caps and open rows of them in the run's log.
* `zkForkClash`: `ZkForkClash` holds; `zkSimForkLeCap_of_align`, `zk_KS_of_forkAlign`: the cap fork bound and masked
  KS without it.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Opening

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.ZkExec (shapeZK shapeZKn)
open Claims.Bridge.Custody

/-- **A masked rep's later-level row is an `Opening` in the log**: where `Zk.verifyWithM`, run with `hashesIn L`,
accepts at table `j`'s `Zk.schedule` schedule and two link points, a level-`ℓ` row `(q, a)` of table `j`'s rep `k`
(`zkLevelRowsOf` on the live session's rep stream `2j + k` and proof file `2j + k`) is a row of an `Opening` of `q` in
`L` against the flattened `ℓ − 1`-th recursive cap of the proof that file decodes to, a cap of `2^cap` 64-byte digests:
the rep passed level `ℓ`'s Merkle check at those positions (`zkLevels_verifyRepWith`), which logged its inputs
(`merkleCheck_logs`). -/
theorem zk_levelOpening {L : List (Claims.Msg × Claims.Msg)} {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live} {j k ℓ : ℕ} {sch : Flock.Schedule} {tPad : ℕ}
    (hv : Zk.verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) record proofs = .ok ())
    (hlv : liveOf spec record = .ok lv) (hj : j < sts.size) (hsch : Zk.schedule sts[j].m = .ok (sch, tPad))
    (hp2 : spec.points = 2) (hk : k < 2) (hms : sts[j].merkle = MerkleScheme.hm96Sha512) {q : ℕ} {a : List 𝔽}
    (ha : (q, a) ∈ zkLevelRowsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
      (lv.streams.getD (2 * j + k) #[]) proofs[2 * j + k]! ℓ) :
    ∃ (zp : Zk.ZkProof) (row : Array F128),
      Zk.decode (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) proofs[2 * j + k]! = .ok zp ∧
      a = row.toList.map FlockLevel3.GF128.of ∧
      zp.proof.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
      (∀ b ∈ zp.proof.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
      Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) L q row := by
  obtain ⟨hsz, hall⟩ := rep_zk _ hv hlv
  have hkk : 2 * j + k < lv.streams.size := by omega
  obtain ⟨sess, sch', tPad', stream, c, hpts, hpn, hsch', hrep, hfind, hfa⟩ := hall (2 * j + k) hkk
  have hj' : sts[(2 * j + k) / 2]! = sts[j] := by
    rw [show (2 * j + k) / 2 = j by omega]
    exact getElem!_pos sts j hj
  rw [hj', hsch] at hsch'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hsch')
  rw [hj'] at hrep
  have hsch₀ : Zk.schedule (Setup.ofCircuit sts[j]).m = .ok (sch, tPad) := hsch
  obtain ⟨stream', extras, zp, hfind', hex, hdec, hlev⟩ :=
    zkLevels_verifyRepWith _ spec (Setup.ofCircuit sts[j]) sch tPad sess _ _ _ c hsch₀ hrep
  obtain rfl : stream = stream' := Option.some.inj (hfind.symm.trans hfind')
  rw [hpts] at hex
  have hx : extras.size = 2 * sts[j].regions.size := by
    rw [extraClaims_size (st := sts[j]) hex, ← hpts, hpn, hp2, Nat.mul_comm]
  have hsh : shapeZK (Setup.ofCircuit sts[j]) sch extras =
      shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) := by
    simp only [shapeZK, shapeZKn, hx]
  have hcoins := forall₂_coins (fun _ _ hab => hab.2) hfa
  have hco : ∀ i : ℕ, (lv.streams.getD (2 * j + k) #[])[i]?.map (fun rd : Flock.Round => rd.coins) =
      stream.rounds[i]?.map (fun rd : Flock.Round => rd.coins) := by
    intro i
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hkk, Option.getD_some,
      ← Array.getElem?_toList, ← Array.getElem?_toList]
    exact hcoins i
  rw [zkLevelRowsOf_congr _ _ _ hco, ← hsh] at ha
  have hℓ : 1 ≤ ℓ ∧ ℓ < sch.levels.size := by
    by_contra hn
    rw [zkLevelRowsOf, ite_eq_right (fun h => hn ⟨h.1, h.2⟩)] at ha
    simp at ha
  obtain ⟨hent, hmc⟩ := hlev ℓ hℓ.1 hℓ.2
  rw [hent] at ha
  obtain ⟨i, hi, -, hiq, hia⟩ := mem_zkRepCols ha
  have hms' : (Setup.ofCircuit sts[j]).merkle = MerkleScheme.hm96Sha512 := hms
  rw [hms'] at hmc
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) MerkleScheme.hm96Sha512)
    MerkleScheme.hm96Sha512 _ _ _ _ _ _ _ _ () hmc
  rw [opsWith_hm96] at hruns
  obtain ⟨hpure, hlog⟩ := merkleCheck_logs hruns
  obtain ⟨hcs, hcw⟩ := merkleCheck_cap hpure
  refine ⟨zp, (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1[i]!, by rw [← hsh]; exact hdec, hia, hcs, hcw,
    ?_⟩
  rw [← hiq]
  exact ⟨Or.inl rfl, zp.proof.ligerito.recursiveCaps[ℓ - 1]!,
    (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.2.getD i ByteArray.empty,
    sibsOf (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.1
      ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap) i,
    hcs, hcw, rfl, merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure i hi, merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure i,
    by rw [opensInsOf_hm96]; exact hlog i hi⟩

end Opening

section Event

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge

/-- **A recorded cap read off the rep's caps**: where rep `rr`'s recorded caps are `f 0, …, f (n − 1)`, its recorded
cap `ℓ` is `f ℓ`. -/
theorem capDig_of_capAtZ {out : OutZC 𝔽 Refine.Dig (List UInt8)} {rr ℓ n : ℕ} {f : ℕ → Model.Cap Refine.Dig}
    {dc : Model.Cap Refine.Dig} (h : (if rr = 0 then out.pend.1 else out.pend.2).lig.caps = (List.range n).map f)
    (hc : CapAtZ rr ℓ out dc) : f ℓ = dc := by
  unfold CapAtZ at hc
  rw [h, List.getElem?_map] at hc
  by_cases hl : ℓ < n
  · rw [List.getElem?_range hl] at hc
    exact Option.some.inj hc
  · rw [List.getElem?_eq_none (by simpa using hl)] at hc
    cases hc

open Classical in
/-- **An opened position at a masked rep's cap is a verifying opening there**: where rep `r`'s level-`ℓ` openings at
`cap` open `p`, one of them is at `p`, opened to that row. -/
theorem opRZ_eq_some {D Sl : Type} {H : List UInt8 → D} {E : Merkle.Enc (List 𝔽) D}
    {Lf : Merkle.Leaf (List 𝔽) Sl D H} {sch : Accounting.Schedule} {r ℓ : ℕ} {cap : Model.Cap D}
    {out : OutZC 𝔽 D Sl} {p : Fin (2 ^ (lvOf sch ℓ).logLen)}
    (h : (opRZ H E Lf sch r ℓ cap out p).isSome) :
    ∃ x ∈ slotOpensZ out r ℓ (lvOf sch ℓ).queries, x.1 = p.val ∧
      opRZ H E Lf sch r ℓ cap out p = some (rowFin (lvOf sch ℓ).lanes x.2.1) := by
  unfold opRZ Exec.opOfL at h ⊢
  rw [Option.isSome_map] at h
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  have hmem := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hmem
  exact ⟨x, hmem.1, hmem.2.1, by rw [hx]; rfl⟩

/-- **The masked fork event's two entries**: `ForkEvZ` at table `j`'s rep `r`, level `ℓ + 1`, is one recorded cap `d`
in both outputs and an opening in each of one position to different rows. -/
theorem forkEvZ_entries {D Sl : Type} [Inhabited D] [Inhabited Sl] {Hs : Hashes 𝔽 D Sl}
    {sch : Accounting.Schedule} {j r ℓ : ℕ} {a a' : List (OutZC 𝔽 D Sl)} (h : ForkEvZ Hs sch j r ℓ a a') :
    ∃ dc, CapAtZ r ℓ (a.getD j dfltOutZC) dc ∧ CapAtZ r ℓ (a'.getD j dfltOutZC) dc ∧
      ∃ x₁ ∈ slotOpensZ (a.getD j dfltOutZC) r (ℓ + 1) (lvOf sch (ℓ + 1)).queries,
      ∃ x₂ ∈ slotOpensZ (a'.getD j dfltOutZC) r (ℓ + 1) (lvOf sch (ℓ + 1)).queries,
        x₁.1 = x₂.1 ∧ x₁.2.1 ≠ x₂.2.1 := by
  obtain ⟨dc, hc₁, hc₂, p, hs₁, hs₂, hne⟩ := h
  obtain ⟨x₁, hx₁, hp₁, he₁⟩ := opRZ_eq_some (out := a.getD j dfltOutZC) hs₁
  obtain ⟨x₂, hx₂, hp₂, he₂⟩ := opRZ_eq_some (out := a'.getD j dfltOutZC) hs₂
  refine ⟨dc, hc₁, hc₂, x₁, hx₁, x₂, hx₂, hp₁.trans hp₂.symm, fun he => hne ?_⟩
  show opRZ Hs.H Hs.E Hs.Lf sch r (ℓ + 1) dc (a.getD j dfltOutZC) p =
    opRZ Hs.H Hs.E Hs.Lf sch r (ℓ + 1) dc (a'.getD j dfltOutZC) p
  rw [he₁, he₂, he]

end Event

section View

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Custody

/-- **A rep's recorded caps and later openings, logged**: at table `j`'s `Zk.schedule` schedule, rep `rr`'s recorded
cap `ℓ` is the `capDig` of a proof's `ℓ`-th recursive cap, and each of its openings `(q, v)` at a level `ℓ ≥ 1` is a
row `v` of an `Opening` of `q` in `L` against the flattened `ℓ − 1`-th of them, a cap of `2^cap` 64-byte digests. -/
def CapOpens (sts : Array Stmt) (sch₀ : Accounting.Schedule) (j rr : ℕ) (out : OutZC 𝔽 Refine.Dig (List UInt8))
    (L : List (Claims.Msg × Claims.Msg)) : Prop :=
  ∃ (sch : Flock.Schedule) (tPad : ℕ) (zp : Zk.ZkProof), Zk.schedule sts[j]!.m = .ok (sch, tPad) ∧
    (∀ ℓ dc, CapAtZ rr ℓ out dc → Refine.capDig zp.proof.ligerito.recursiveCaps[ℓ]! = dc) ∧
    ∀ ℓ, 1 ≤ ℓ → ∀ q v, (q, v) ∈ (slotOpensZ out rr ℓ (lvOf sch₀ ℓ).queries).map (fun x => (x.1, x.2.1)) →
      ∃ row : Array F128, v = row.toList.map FlockLevel3.GF128.of ∧
        zp.proof.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
        (∀ b ∈ zp.proof.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
        Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
          (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) L q row

/-- **An accepting run's view records its reps' proof files' caps and opens rows of them in its log**: at the view the
deployed decoders are right at `OpensRows` and at `CapsRows` (`decodesTableR_deploy`, `decodesTableC_deploy`), at
`wrS`'s transcript, which records the run's rep streams and proof files (`recordsLiveAtS_wrS`). So each rep's
recorded caps are its proof file's recursive caps, and its openings past level 0 are its `zkLevelRowsOf` rows on the
run's rep stream and proof file, `Opening`s in the run's log (`zk_levelOpening`). -/
theorem zkViewCapOpens {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
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
    ∀ rr < 2, CapOpens sts (zkTabs h0 L hz hk hlay j).sch j rr (a.getD j dfltOutZC)
      (Claims.run sha512 P (V true) x ω r).vLog := by
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
  obtain ⟨-, spec₂, sts₂, sess, sch, tPad, s₀, s₁, hj₂, hsA₂, hdec, hsch, hs₀, hs₁, -, hlater, -⟩ :=
    Tables.DecodesTable.ok_of_plays (decodesTableR_deploy hs hin hset h0 L hz hk hlay hd
      (viewOf _ spec.mPts spec link c₁ f) jj) ⟨hV.1, rfl⟩ (pend jj) (hpl jj 0) (hpl jj 1)
  obtain ⟨-, spec₃, sts₃, sch₃, tPad₃, hj₃, hsA₃, hsch₃, hcaps⟩ :=
    Tables.DecodesTable.ok_of_plays (decodesTableC_deploy hs hin hset h0 L hz hk hlay hd
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
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsA.symm.trans hsA₃))
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsch.symm.trans hsch₃))
  obtain ⟨hpr, hall⟩ := recordsLiveAtS_wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f) hV.1
  obtain ⟨-, hrs⟩ := hall spec sts sess hsA hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs jj).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs jj).2 ⟨1, by omega⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain rfl : y₁ = s₁ := Option.some.inj (hy₁.symm.trans hs₁)
  obtain ⟨-, lv, hlv, -⟩ := checks_ok hc
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (p : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[p]!).toList.map (withSha spec) →
      ∀ i : ℕ, y.rounds[i]?.map (fun rd : Flock.Round => rd.coins) =
        (lv.streams.getD p #[])[i]?.map (fun rd : Flock.Round => rd.coins) := by
    intro p y hy i
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  have hp2 := (HiddenExec.setupTables_pointsAny hsA).1
  have hms : sts[jj.val].merkle = MerkleScheme.hm96Sha512 := checks_hm96 hc hsu hzk jj.val hj₂
  have hsch' : Zk.schedule sts[jj.val]!.m = .ok (sch, tPad) := by
    rw [getElem!_pos sts jj.val hj₂]
    exact hsch
  intro rr hrr
  obtain ⟨zp, hzp, hcp⟩ := hcaps rr hrr
  rw [hpr'] at hzp
  refine ⟨sch, tPad, zp, hsch', fun ℓ dc hc' => capDig_of_capAtZ hcp hc', fun ℓ hℓ1 q v hqv => ?_⟩
  obtain ⟨e₀, e₁⟩ := hlater ℓ hℓ1
  rw [tabsJ_get] at e₀ e₁
  obtain rfl | rfl : rr = 0 ∨ rr = 1 := by omega
  · rw [e₀, hpr', zkLevelRowsOf_congr _ _ _ (hco (2 * jj.val) _ hr₀)] at hqv
    obtain ⟨zp', row, hzp', hrow, hcs, hcw, hop⟩ :=
      zk_levelOpening (k := 0) hv' hlv hj₂ hsch hp2 (by omega) hms hqv
    obtain rfl : zp' = zp := Except.ok.inj (hzp'.symm.trans hzp)
    exact ⟨row, hrow, hcs, hcw, hop⟩
  · rw [e₁, hpr', zkLevelRowsOf_congr _ _ _ (hco (2 * jj.val + 1) _ hr₁)] at hqv
    obtain ⟨zp', row, hzp', hrow, hcs, hcw, hop⟩ :=
      zk_levelOpening (k := 1) hv' hlv hj₂ hsch hp2 (by omega) hms hqv
    obtain rfl : zp' = zp := Except.ok.inj (hzp'.symm.trans hzp)
    exact ⟨row, hrow, hcs, hcw, hop⟩

end View

section Live

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

/-- **A decoded live outcome of the masked simulator records an accepting run's proof files' caps and opens rows of
them in its log**: where the live verdict holds, `P`'s run at `zkΦ`'s coins accepts (`zkSimAbortAt`), the outcome is
the view of the record it decides (`zkSimRunAt`), and table `j`'s decoded output's reps record their proof files' caps
and open rows of them in the run's log (`zkViewCapOpens`). -/
theorem zkAccCapOpens {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
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
      CapOpens sts (zkTabs h0 L hz hk hlay j).sch j rr (a.getD j dfltOutZC)
        (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).vLog := by
  have hacc : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc := by
    by_contra hn
    exact zkSimAbortAt x s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r' hn ha.1
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, -⟩ := acc_decide hacc
  have hv := zkSimRunAt x s hs hin ins spec sts hset h0 L hz hk hlay Ω P ω u r' hacc s' spec' sts' link c₁ mids f hact
  rw [hv] at ha
  exact ⟨hacc, zkViewCapOpens hs hin hset h0 L hz hk hlay hd hj hacc hact ha⟩

/-- **The masked live fork event's collision, with no hypothesis**: two live outcomes of the masked simulator at one
tape whose table `j` outputs record one cap `d` at rep `r`'s level `ℓ + 1` and open a position of it to different rows
are two accepting runs of `P` at `zkΦ`'s coins (`zkAccCapOpens`). Each run's recorded cap is the `capDig` of its proof
file's `ℓ`-th recursive cap, so one cap of 64-byte digests in both (`capDig_inj`), and each opening is an `Opening`
against it in its run's log, so the two rows collide in the joined logs (`tiedColl_of_openings`). -/
theorem zkForkClash (x : Claims.Msg) : ZkForkClash x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r₁ r₂ r ℓ h _ _
  obtain ⟨a, a', ha, ha', hF⟩ := h
  replace hF : ForkEvZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j).sch j r.val ℓ.val a a' := hF
  obtain ⟨dc, hc₁, hc₂, x₁, hx₁, x₂, hx₂, hq, hne⟩ := forkEvZ_entries hF
  obtain ⟨-, hl₁⟩ := zkAccCapOpens hs hin hset h0 L hz hk hlay hd hj hC u r₁ ha
  obtain ⟨-, hl₂⟩ := zkAccCapOpens hs hin hset h0 L hz hk hlay hd hj hC u r₂ ha'
  obtain ⟨sch₁, tPad₁, zp₁, hsch₁, hcap₁, hop₁⟩ := hl₁ r.val r.isLt
  obtain ⟨sch₂, tPad₂, zp₂, hsch₂, hcap₂, hop₂⟩ := hl₂ r.val r.isLt
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsch₁.symm.trans hsch₂))
  obtain ⟨row₁, hrow₁, hcs₁, hcw₁, ho₁⟩ := hop₁ (ℓ.val + 1) (by omega) x₁.1 x₁.2.1 (List.mem_map_of_mem hx₁)
  obtain ⟨row₂, hrow₂, hcs₂, hcw₂, ho₂⟩ := hop₂ (ℓ.val + 1) (by omega) x₂.1 x₂.2.1 (List.mem_map_of_mem hx₂)
  simp only [Nat.add_sub_cancel] at hcs₁ hcw₁ ho₁ hcs₂ hcw₂ ho₂
  have hcap : zp₁.proof.ligerito.recursiveCaps[ℓ.val]! = zp₂.proof.ligerito.recursiveCaps[ℓ.val]! :=
    capDig_inj ((hcap₁ ℓ.val dc hc₁).trans (hcap₂ ℓ.val dc hc₂).symm) (hcs₁.trans hcs₂.symm) hcw₁ hcw₂
  rw [hcap, hq] at ho₁
  exact tiedColl_of_openings ho₁ ho₂ fun he => hne (by rw [hrow₁, hrow₂, he])

/-- **The fork bound at the caps from the alignment**: `ZkSimForkLeOn` at the cap events from `ZkForkAlign`, the
clash holding outright (`zkForkClash`). -/
theorem zkSimForkLeCap_of_align {x : Claims.Msg} (ha : ZkForkAlign x) : ZkSimForkLeOn x fun _ i => IsCapZ i :=
  zkSimForkLeCap_of ha (zkForkClash x)

/-- **The masked record the claims read: KS**, from the alignment at the caps, the fork bounds at `τ` and the pads,
and the view's pads clash at the statements some run accepts: `zk_KS_of_align` with the clash at the caps holding
outright (`zkForkClash`). -/
theorem zk_KS_of_forkAlign (ha : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkForkAlign x)
    (ht : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_align ha (fun x _ => zkForkClash x) ht hp hpads

end Live

end FlockDeploy.KS
