import Proofs.Flock.Claims.KS.ZkFork0
import Proofs.Flock.Claims.KS.ZkPadsLog

/-!
# The masked view's pads clash

`PadsZ`, `SelfClashZ`'s pads case, at an accepting run's view: two of a rep's pads openings, both drawn at one
position, that verify there to one root with different pairs. At the view the deployed decoders are right at
`OpensRows`, whose pads openings are rep `r`'s proof file's inner-proof openings at `wrS`'s transcript, drawn at the
positions the rep stream's last round's coins give (`PadsRep`). That stream has the run's rep stream's coins, and the
run's verifier opened each of those queries against the inner proof's root at that position, so two at one position
with different pairs collide in the run's log (`tiedColl_of_verifyInner`). The root the clash reads plays no part.

* `coins_last`: two round arrays with the same coins at every round have the same last round's coins.
* `zk_padsClash_tied`: a pads clash among one accepted rep's inner-proof openings at the positions its last round's
  coins give is a collision in the run's log.
* `zkViewClashQP`: `ZkViewClashQ x PadsZ` at every statement.
* `zk_KS_of_forks`, `zk_tableBoundJOnAccR_of_forks`, `zk_KSFlock_of_forks`: masked KS and the recursion's chain rest on
  the cap/`τ` and pads fork bounds alone.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Pads

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.ZkExec (innerN layZ shapeZK shapeZKn innerOpensOf PadsRep zk_decode_paths)
open Claims.Bridge.Custody

/-- Two round arrays with the same coins at every round have the same last round's coins. -/
theorem coins_last {a b : Array Flock.Round}
    (h : ∀ i : ℕ, a[i]?.map (fun rd : Flock.Round => rd.coins) = b[i]?.map (fun rd : Flock.Round => rd.coins)) :
    (a[a.size - 1]!).coins = (b[b.size - 1]!).coins := by
  have hsz : a.size = b.size := by
    have e : a.map (fun rd : Flock.Round => rd.coins) = b.map (fun rd : Flock.Round => rd.coins) :=
      Array.ext_getElem? fun i => by simpa only [Array.getElem?_map] using h i
    simpa using congrArg Array.size e
  rw [hsz]
  generalize b.size - 1 = i
  by_cases hi : i < b.size
  · have e := h i
    rw [Array.getElem?_eq_getElem (hsz ▸ hi), Array.getElem?_eq_getElem hi] at e
    rw [getElem!_pos a i (hsz ▸ hi), getElem!_pos b i hi]
    simpa using e
  · rw [getElem!_neg a i (by omega), getElem!_neg b i hi]

/-- **Two of an accepted masked rep's pads openings at one position with different pairs collide in the run's log**:
where a stream `y` has the coins of the live session's rep stream `2j + r` and rep `r`'s proof file decodes to pads
openings drawn at the positions `y`'s last round's coins give (`PadsRep`), the run's verifier opened those queries
against the inner proof's root at those positions (`zkVerifyRepWith_inner`), so two of them at one position with
different pairs collide among its calls (`tiedColl_of_verifyInner`). -/
theorem zk_padsClash_tied {L : List (Claims.Msg × Claims.Msg)} {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live} {j rr : ℕ} {sch : Schedule} {tPad : ℕ}
    (hv : Zk.verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) record proofs = .ok ())
    (hlv : liveOf spec record = .ok lv) (hj : j < sts.size) (hsch : Zk.schedule sts[j].m = .ok (sch, tPad))
    (hp2 : spec.points = 2) (hrr : rr < 2) {y : StreamRec}
    (hy : ∀ i : ℕ, y.rounds[i]?.map (fun rd : Flock.Round => rd.coins) =
      (lv.streams.getD (2 * j + rr) #[])[i]?.map (fun rd : Flock.Round => rd.coins))
    {io : ZkLink.InnerOpens 𝔽 Dig (List UInt8)} {qpos : ℕ → ℕ}
    (hpads : PadsRep (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) y proofs[2 * j + rr]! io qpos)
    {k k' : ℕ} (hk : k < Zk.Q) (hk' : k' < Zk.Q) (hq : qpos k = qpos k') (hne : (io.pads k).1 ≠ (io.pads k').1) :
    Claims.TiedColl sha512 L := by
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
  have hpos : ((t.rounds[t.k + 3]!).coins[k]!).lo.toNat % innerN (layZ (Setup.ofCircuit sts[j]) sch) =
      ((t.rounds[t.k + 3]!).coins[k']!).lo.toNat % innerN (layZ (Setup.ofCircuit sts[j]) sch) := by
    rw [hcoin', ← hqpos k hk, ← hqpos k' hk', hq]
  refine tiedColl_of_verifyInner hinner (zk_decode_paths hdec) hk hk' hpos fun e => hne ?_
  show pairOf zp.inner.opened[k]! = pairOf zp.inner.opened[k']!
  rw [e]

end Pads

section Clash

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Custody

namespace Gen

variable [S : ZkSetup]

/-- **A pads clash at an accepting run's view is a collision in its log**: at the view the deployed decoders are right
at `OpensRows` (`decodesTableR_deploy`), so table `j`'s rep `r` reads `Q` queries and its pads openings are its proof
file's inner-proof openings at `wrS`'s transcript, drawn at the positions its rep stream's last round's coins give; that
stream has the run's rep stream's coins (`recordsLiveAtS_wrS`) and the proof file is the run's, so two of them at one
position with different pairs collide in the run's log (`zk_padsClash_tied`). -/
theorem zkViewClashQP (x : Claims.Msg) : ZkViewClashQ x PadsZ := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C _ r hacc s' spec' sts' link c₁ mids f hact hret
  obtain ⟨a, ⟨hV, pend, hpl, ha⟩, hcl⟩ := hret
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
  rw [getD_ofFnJ] at hcl
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
  obtain ⟨rr, hrr, root, k, hkq, k', hkq', p, hqk, hqk', -, -, hne⟩ := hcl
  rw [hq] at hkq hkq'
  obtain rfl | rfl : rr = 0 ∨ rr = 1 := by omega
  · rw [ite_eq_left rfl] at hqk hqk'
    exact Or.inr (zk_padsClash_tied (rr := 0) (y := y₀) hv' hlv hj₂ hsch hp2 (by omega)
      (by rw [Nat.add_zero]; exact hco _ _ hr₀) (by rw [Nat.add_zero]; exact hp₀) hkq hkq' (hqk.trans hqk'.symm) hne)
  · rw [ite_eq_right (show (1 : ℕ) ≠ 0 by omega)] at hqk hqk'
    exact Or.inr (zk_padsClash_tied (rr := 1) (y := y₁) hv' hlv hj₂ hsch hp2 (by omega) (hco _ _ hr₁) hp₁ hkq hkq'
      (hqk.trans hqk'.symm) hne)

end Gen

/-- **A pads clash at an accepting run's view is a collision in its log**: at the view the deployed decoders are right
at `OpensRows` (`decodesTableR_deploy`), so table `j`'s rep `r` reads `Q` queries and its pads openings are its proof
file's inner-proof openings at `wrS`'s transcript, drawn at the positions its rep stream's last round's coins give; that
stream has the run's rep stream's coins (`recordsLiveAtS_wrS`) and the proof file is the run's, so two of them at one
position with different pairs collide in the run's log (`zk_padsClash_tied`). -/
theorem zkViewClashQP (x : Claims.Msg) : ZkViewClashQ x PadsZ :=
  Gen.zkViewClashQP (S := S₀) (x := x)

end Clash

open Flock

/-- **The deployed masked verifier: KS, from the cap/`τ` and pads fork bounds**, at SHA-512 at the trimmed record at
the deployed relation: the view's pads clash holds at every statement (`zkViewClashQP`; `zk_KS_of_kinds`). -/
theorem zk_KS_of_forks (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_kinds hct hp fun x _ => zkViewClashQP x

end FlockDeploy.KS

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **Every masked table's bound with escapes on accepted runs counted per round**, at full-width columns, from the
cap/`τ` and pads fork bounds (`zk_tableBoundJOnAccR_of_kinds` at `zkViewClashQP`): `zk_KSFlockJR`'s premise. -/
theorem zk_tableBoundJOnAccR_of_forks
    (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i) :
    TableBoundJOnAccR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) :=
  zk_tableBoundJOnAccR_of_kinds hct hp fun x _ => zkViewClashQP x

/-- **KS of the recursion over the masked record the claims read** at the deployed relation, at every number of
tables, from the cap/`τ` and pads fork bounds (`zk_KSFlock_of_kinds` at `zkViewClashQP`). -/
theorem zk_KSFlock_of_forks (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hct : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsCapTauZ i)
    (hp : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLeOn x fun _ i => IsPadsEvZ i)
    (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZkTrim Readable.relation).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim Readable.relation) DO :=
  zk_KSFlock_of_kinds D DO opening hF hct hp (fun x _ => zkViewClashQP x) hRef hRounds hN hε hRd hS hR hV hT hL hO
    hAcc hN0

end FlockDeploy.KS
