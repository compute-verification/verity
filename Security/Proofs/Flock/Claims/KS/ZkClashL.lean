import Proofs.Flock.Claims.KS.ZkClash0

/-!
# The masked view's self-clash past level 0

`LaterClashZ`, `SelfClashZ`'s case past level 0, at an accepting run's view: two of a rep's level-`ℓ` openings at one
position that verify with different rows. At the view the deployed decoders are right at `OpensRows`, whose openings
past level 0 are the rep's `zkLevelRowsOf` rows at `wrS`'s transcript, so they are the run's own level-`ℓ` rows, and two
of them at one position that differ collide in the run's log (`zk_levelClash_tiedJ`). The level's cap the clash reads
plays no part: the collision is against the rep's recorded recursive cap.

* `rawLevelRows_zk`: the level-`ℓ` rows table `j`'s rep `k` opens in an accepted masked record are its `zkLevelRowsOf`
  rows on the live session's rep stream `2j + k` and proof file (`rawRows_zk` at one rep and level).
* `zk_laterClash_tiedJ`: a `ClashL` among openings whose rows are the run's level-`ℓ` rows of one rep is a collision
  in the run's log.
* `zkViewClashQL`: `ZkViewClashQ x LaterClashZ` at every statement.
* `zk_KS_of_pads`, `zk_tableBoundJOnAccR_of_pads`, `zk_KSFlock_of_pads`: masked KS and the recursion's chain rest on the
  fork bounds and the view's pads clash.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Rows

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody

namespace Gen

variable [S : ZkSetup]

/-- **An accepted masked run's level-`ℓ` rows of table `j`'s rep `k`**: `rawRows_zk` at one rep and one level, on the
live session's rep stream `2j + k` and proof file `2j + k`. -/
theorem rawLevelRows_zk (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {j : ℕ} {sch : Schedule} {tPad : ℕ} (hf : lastFinal seen = some f)
    (hset : setupOf true f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : Zk.verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hj : j < sts.size)
    (hsch : Zk.schedule sts[j].m = .ok (sch, tPad)) (hp2 : spec.points = 2) (k : ℕ) (hk : k < 2) (ℓ : ℕ) :
    zkRawLevelRowsJ seen j k ℓ =
      zkLevelRowsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
        (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) ℓ := by
  obtain ⟨hsz, hall⟩ := rep_zk hs hv hlv
  obtain ⟨sess, sch', tPad', stream, c, hpts, hpn, hf', hrep, -, -⟩ := hall (2 * j) (by omega)
  obtain ⟨-, extras, -, hex, -⟩ := zkVerifyRepWith_rounds hs hf' hrep
  have hj' : sts[2 * j / 2]! = sts[j] := by
    rw [show 2 * j / 2 = j by omega]
    exact getElem!_pos sts j hj
  rw [hj', hpts] at hex
  have hx : extras.size = 2 * sts[j].regions.size := by
    rw [extraClaims_size (st := sts[j]) hex, ← hpts, hpn, hp2, Nat.mul_comm]
  have hzs : Claims.Flock.zkSched (Setup.ofCircuit sts[j]).m = (sch, tPad) := by
    simp only [Claims.Flock.zkSched]
    rw [show (Setup.ofCircuit sts[j]).m = sts[j].m from rfl, hsch]
  have hsh : Claims.Flock.zkProofShape (Setup.ofCircuit sts[j]).m (Setup.ofCircuit sts[j]).kLog (2 + extras.size)
      (Setup.ofCircuit sts[j]).merkle.digestLen (saltLenOf true (Setup.ofCircuit sts[j])) =
      ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) := by
    simp only [Claims.Flock.zkProofShape, hzs, ZkExec.shapeZKn, saltLenOf, hx, ite_true]
  have hrun : runOf true seen j = some
      { m := (Setup.ofCircuit sts[j]).m, kLog := (Setup.ofCircuit sts[j]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[j]).merkle.digestLen, saltLen := saltLenOf true (Setup.ofCircuit sts[j]),
        reps := [(lv.streams.getD (2 * j) #[], f.proofs.getD (2 * j) ByteArray.empty),
                 (lv.streams.getD (2 * j + 1) #[], f.proofs.getD (2 * j + 1) ByteArray.empty)] } := by
    unfold runOf
    simp [hf, hset, Array.getElem?_eq_getElem hj, hlv, hex, Except.toOption]
  unfold zkRawLevelRowsJ
  rw [hrun, ← hsh]
  rcases k with _ | _ | k
  · rfl
  · rfl
  · omega

end Gen

/-- **An accepted masked run's level-`ℓ` rows of table `j`'s rep `k`**: `rawRows_zk` at one rep and one level, on the
live session's rep stream `2j + k` and proof file `2j + k`. -/
theorem rawLevelRows_zk (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {j : ℕ} {sch : Schedule} {tPad : ℕ} (hf : lastFinal seen = some f)
    (hset : setupOf true f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : Zk.verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hj : j < sts.size)
    (hsch : Zk.schedule sts[j].m = .ok (sch, tPad)) (hp2 : spec.points = 2) (k : ℕ) (hk : k < 2) (ℓ : ℕ) :
    zkRawLevelRowsJ seen j k ℓ =
      zkLevelRowsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
        (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) ℓ :=
  Gen.rawLevelRows_zk (S := S₀) (hs := hs) (seen := seen) (f := f) (ins := ins) (spec := spec) (sts := sts) (lv := lv)
    (j := j) (sch := sch) (tPad := tPad) (hf := hf) (hset := hset) (hlv := hlv) (hv := hv) (hj := hj) (hsch := hsch)
    (hp2 := hp2) (k := k) (hk := hk) (ℓ := ℓ)

end Rows

section Clash

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Custody

namespace Gen

variable [S : ZkSetup]

/-- **A later-level clash among a run's own rows is a collision in its log**: openings whose `(position, row)` pairs are
the level-`ℓ` rows table `j`'s rep `k` opens in an accepting masked run, two of which verify at one position with
different rows (`ClashL`), give two of those rows at one position that differ (`zk_levelClash_tiedJ`). -/
theorem zk_laterClash_tiedJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {j k ℓ : ℕ} {cap : Model.Cap Refine.Dig} {lanes depth : ℕ}
    {xs : List (ℕ × List 𝔽 × List UInt8 × List Refine.Dig)} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hxs : xs.map (fun x => (x.1, x.2.1)) = zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ)
    (hcl : Exec.ClashL ZkExec.hs512.H ZkExec.hs512.E ZkExec.hs512.Lf cap lanes depth xs) :
    Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog := by
  obtain ⟨u, hu, v, hv, huv, -, -, hne⟩ := hcl
  have hu' : (u.1, u.2.1) ∈ zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ := by
    rw [← hxs]
    exact List.mem_map_of_mem hu
  have hv' : (u.1, v.2.1) ∈ zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ := by
    rw [huv, ← hxs]
    exact List.mem_map_of_mem hv
  exact zk_levelClash_tiedJ h hu' hv' hne

end Gen

/-- **A later-level clash among a run's own rows is a collision in its log**: openings whose `(position, row)` pairs are
the level-`ℓ` rows table `j`'s rep `k` opens in an accepting masked run, two of which verify at one position with
different rows (`ClashL`), give two of those rows at one position that differ (`zk_levelClash_tiedJ`). -/
theorem zk_laterClash_tiedJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {j k ℓ : ℕ} {cap : Model.Cap Refine.Dig} {lanes depth : ℕ}
    {xs : List (ℕ × List 𝔽 × List UInt8 × List Refine.Dig)} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hxs : xs.map (fun x => (x.1, x.2.1)) = zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ)
    (hcl : Exec.ClashL ZkExec.hs512.H ZkExec.hs512.E ZkExec.hs512.Lf cap lanes depth xs) :
    Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog :=
  Gen.zk_laterClash_tiedJ (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (j := j) (k := k) (ℓ := ℓ) (cap := cap)
    (lanes := lanes) (depth := depth) (xs := xs) (h := h) (hxs := hxs) (hcl := hcl)

namespace Gen

variable [S : ZkSetup]

/-- **A self-clash past level 0 at an accepting run's view is a collision in its log**: the reps' pendings at the view
play the deployed decoders' tapes, so table `j`'s output there is right at `OpensRows` (`decodesTableR_deploy`); its
rep `k`'s openings at a level `ℓ ≥ 1` are the rep's `zkLevelRowsOf` rows at `wrS`'s transcript, which records the run's
rep streams and proof files (`recordsLiveAtS_wrS`), so they are the run's level-`ℓ` rows (`rawLevelRows_zk`), and two
of them at one position that differ collide in the run's log (`zk_laterClash_tiedJ`). -/
theorem zkViewClashQL (x : Claims.Msg) : ZkViewClashQ x LaterClashZ := by
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
  obtain ⟨-, spec₂, sts₂, sess, sch, tPad, s₀, s₁, hj₂, hsA₂, hdec, hsch, hs₀, hs₁, -, hlater, -⟩ :=
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
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V true) x ω r).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  have hraw := rawLevelRows_zk _ hf (by rw [hx]; exact act_decide hact) hlv hv' hj₂ hsch
    (HiddenExec.setupTables_pointsAny hsA).1
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (q : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[q]!).toList.map (withSha spec) →
      ∀ i : ℕ, y.rounds[i]?.map (fun rd : Flock.Round => rd.coins) =
        (lv.streams.getD q #[])[i]?.map (fun rd : Flock.Round => rd.coins) := by
    intro q y hy i
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  obtain ⟨k, hk2, ℓ, hℓ1, cap, -, hcl⟩ := hcl
  obtain ⟨e₀, e₁⟩ := hlater ℓ hℓ1
  rw [tabsJ_get] at e₀ e₁
  rcases k with _ | _ | k
  · rw [hpr', zkLevelRowsOf_congr _ _ _ (hco (2 * jj.val) _ hr₀), ← getD_empty f.proofs (2 * jj.val)] at e₀
    exact Or.inr (zk_laterClash_tiedJ hacc (e₀.trans (hraw 0 (by omega) ℓ).symm) hcl)
  · rw [hpr', zkLevelRowsOf_congr _ _ _ (hco (2 * jj.val + 1) _ hr₁), ← getD_empty f.proofs (2 * jj.val + 1)]
      at e₁
    exact Or.inr (zk_laterClash_tiedJ hacc (e₁.trans (hraw 1 (by omega) ℓ).symm) hcl)
  · omega

end Gen

/-- **A self-clash past level 0 at an accepting run's view is a collision in its log**: the reps' pendings at the view
play the deployed decoders' tapes, so table `j`'s output there is right at `OpensRows` (`decodesTableR_deploy`); its
rep `k`'s openings at a level `ℓ ≥ 1` are the rep's `zkLevelRowsOf` rows at `wrS`'s transcript, which records the run's
rep streams and proof files (`recordsLiveAtS_wrS`), so they are the run's level-`ℓ` rows (`rawLevelRows_zk`), and two
of them at one position that differ collide in the run's log (`zk_laterClash_tiedJ`). -/
theorem zkViewClashQL (x : Claims.Msg) : ZkViewClashQ x LaterClashZ :=
  Gen.zkViewClashQL (S := S₀) (x := x)

end Clash

open Flock

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the masked simulator's fork bounds and the view's pads clash at the statements some run accepts: the view's clashes at
level 0 and past it hold at every statement (`zkViewClashQ₀`, `zkViewClashQL`; `zk_KS_of_later`). -/
theorem zk_KS_of_pads (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_later hfork (fun x _ => zkViewClashQL x) hpads

namespace Gen

variable [S : ZkSetup]

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the masked simulator's fork bounds and the view's pads clash at the statements some run accepts: the view's clashes at
level 0 and past it hold at every statement (`zkViewClashQ₀`, `zkViewClashQL`; `zk_KS_of_later`). -/
theorem zk_KS_of_pads (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_later hfork (fun x _ => zkViewClashQL x) hpads

end Gen

end FlockDeploy.KS

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **Every masked table's bound with escapes on accepted runs counted per round**, at full-width columns, from the
fork bounds and the view's pads clash at the statements some run accepts (`zk_tableBoundJOnAccR_of_later`):
`zk_KSFlockJR`'s premise. -/
theorem zk_tableBoundJOnAccR_of_pads (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    TableBoundJOnAccR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) :=
  zk_tableBoundJOnAccR_of_later hfork (fun x _ => zkViewClashQL x) hpads

/-- **KS of the recursion over the masked record the claims read** at the deployed relation, at every number of
tables, from the fork bounds and the view's pads clash at the statements some run accepts (`zk_KSFlock_of_later`). -/
theorem zk_KSFlock_of_pads (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) (hRef : Refuses sha512 D)
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
  zk_KSFlock_of_later D DO opening hF hfork (fun x _ => zkViewClashQL x) hpads hRef hRounds hN hε hRd hS hR hV hT hL hO
    hAcc hN0

end FlockDeploy.KS
