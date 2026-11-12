module

public import Proofs.Flock.SessionLoop.FinInner
public import Proofs.Flock.SessionLoop.FinInnerRec
public import Proofs.Flock.SessionLoop.FinInnerCut
public import Proofs.Flock.SessionLoop.FinPiopRet
public import Proofs.Flock.SessionLoop.SwTakeOn
public import Proofs.Flock.Claims.ZK.FinishInner
public import Proofs.Flock.Claims.ZK.FinishStop
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckReplay
public import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRep
public import Proofs.Flock.SessionLoop.InnerUnrefused
public import Proofs.Flock.SessionLoop.FinRepRows

/-!
# The loop's finish at a rep's inner rounds on the worker's own histories, in parts

`LeanRelease.finOf_finalC_own` and `LeanRelease.finOf_tau_own` read the rounds the loop's finish hands a rep's frame
(`Loop.finOf`) at a history `LiveInnerOwn` reads at rep `2j + r`'s inner round `e` (`Loop.InnerHist`), at the release
`withInner` on those rounds (`Loop.relIn`). Each part is named here, none a premise of a guarantee:

* **`Loop.innerHist_fin`**: there the loop's table and request for rep stream `2j + r` return; the table's pad layout
  and the round's configuration are table `j`'s (`Masked.layE`) and the configuration's pads the tape's; and where the
  finish's live inner proof (`finInner`) returns at the request with `final_c = a` (`askOf`) and at no other `final_c`,
  `finOf` is its rounds. The history's next round is the stream's round `nSched + e`, past its scheduled rounds
  (`innerAt_slot`).
* **`Loop.innerReq_own`**: the request's values below `zp 64` but `final_c`'s are `vs`'s (`vsWord`), and its pads the
  tape's. The history's rounds of the rep's PIOP are the frame's releases of the worker's rounds (`OwnHist`), the
  switched session's PIOP bodies where `LiveSchedOwn` holds, which carry `vs`'s values under the pads.
* **`Loop.innerReq_coins`**: the request's coins of the rep's rounds before the history's next are the opened coins from
  the rep's first round (`Masked.repBase`).
* **`Loop.finInner_fc`**: where `finInner` returns, its `final_c'` is the request's `final_c` under its pad
  (`Loop.repCons_rets`, over `Loop.Rets`: what a run that returns returns).
* **`Loop.finInner_accepts_own`**: `finInner` returns at the request with `final_c = a` only at `vs`'s `final_c`: its
  `zerocheck c_eval` constraint holds on the pads (`consWhy`), which at the request's round-1c values and the opened
  coins' zerocheck point fixes `a` (`Loop.FinalCFixed`).
* **`Loop.finInner_inner`**: where `finInner` returns, its `repCons` returns answered by `asksPad`, `consWhy` passes,
  and its result is the proof's `final_c` and the inner proof's `τ` commitment.
* **`Loop.finInner_tau_own`**: at the inner proof's round 2, its `τ` commitment is `vs`'s (`Loop.vsTau`), where the rows
  hold (`Loop.RowsOK`): `repCons` at the session's cut (`Finish.repCons_cut`) has the switched view's rows as its
  constraints (`Masked.repConsC_rows`), and the inner proof commits to `InZ.tau` over them (`Loop.innerP_pad_tau`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec TMW)
open _root_.Flock.Firewall.Frame (INNER_ROUNDS)
open _root_.Flock.Firewall.Finish (Tab RepIn Asks)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- **The session loop's release at its finish's inner rounds**: `withInner` on `finOf` at `L₁`'s frame samples, with the
loop's own link and `Finish` (`ofLoopLink`). -/
noncomputable abbrev relIn (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) : LeanRelease Flock.Wit Tp :=
  LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src real

/-- **A history `LiveInnerOwn` reads at rep `2j + r`'s inner round `e`**, at the release `L`: after `Hello` `(m₀, ρ₀)`
and `0 < ops.length < refK x` openings, live, whose openings pass, no coin refused, the release's own run
(`ZkServer.OwnHist`), at which the worker sends a message, whose next round is an inner proof's (`InnerAt`), rep
`2j + r`'s round `innerOff e`. -/
def InnerHist (L : LeanRelease Flock.Wit Tp) (wk : L.toSrv.Worker)
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce)
    (t : finS.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (e : ℕ) : Prop :=
  0 < ops.length ∧ ops.length < refK x ∧ L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true ∧
    ((allOk (finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true ∧
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false ∧
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) ∧
    (∃ v, wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) ∧
    InnerAt (real x w (ν, t)) x (ops.map fun (p : Msg × Msg) => CoinLayer.rd p.2) ∧
    L.rep x (ops.length + 1) = 2 * j.val + r.val ∧
    ops.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff e)

theorem v_bind_ok {α β : Type} {x : V α} {f : α → V β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- **A rep's inner rounds are its last**: at a `Live` statement, rep stream `k` has at least the inner proof's rounds,
and its scheduled ones (`nSched`) end at its inner round 0 (`Masked.repSlot … (Masked.innerOff 0)`). -/
theorem repSlot_nSched {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {k : ℕ} (hk : k < spec.repStreams.size) :
    INNER_ROUNDS ≤ (roundsOf spec (Masked.stsOf x)).getD k 0 ∧
      Masked.repBase x k + nSched spec (Masked.stsOf x) k = Masked.repSlot x k (Masked.innerOff 0) := by
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hj : k / 2 < (Masked.stsOf x).size := by omega
  obtain ⟨y⟩ := hx.2 (k / 2) hj
  have hst : (Masked.stsOf x)[k / 2]! = (Masked.stsOf x)[k / 2] := getElem!_pos _ (k / 2) hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[k / 2]!.c.kLog := by
    rw [hst]
    exact y.wf.1.hk6
  have hrq : (roundsOf spec (Masked.stsOf x)).getD k 0 =
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[k / 2]! + (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) + 4 := by
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk, Option.map_some,
      Option.getD_some]
    have e1 : (Masked.stsOf x)[k / 2]!.m + (Masked.stsOf x)[k / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[k / 2]!.m + ((Masked.stsOf x)[k / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hsl : Masked.repSlot x k (Masked.innerOff 0) = FlockComplete.repStartZk spec (Masked.stsOf x) k +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[k / 2]! + (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1) + 0) := by
    unfold Masked.repSlot
    rw [hs]
    simp only [hz]
    rfl
  rw [Masked.repBase_eq hs, hsl]
  unfold nSched
  rw [hrq]
  have hI : INNER_ROUNDS = 4 := rfl
  refine ⟨by omega, by omega⟩

/-- **Rep stream `k`'s first round is past the link**: its first rep round (`streamAt`) is one before its first session
round (`Masked.repBase`). -/
theorem streamAt_repBase {x : Msg} {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {k : ℕ} (hk : k < spec.repStreams.size) :
    streamAt (roundsOf spec (Masked.stsOf x)) k + 1 = Masked.repBase x k := by
  rw [Masked.repBase_eq hs k]
  unfold streamAt roundsOf FlockComplete.repStartZk
  rw [take_map_range _ (by omega)]
  omega

/-- **A rep's PIOP rounds are among its scheduled ones**: at a `Live` statement, rep stream `2j + r`'s PIOP rounds
(`piopNR`) number at most its scheduled rounds (`nSched`). -/
theorem piopNR_le_nSched {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (j : Fin (Masked.jOf x)) (r : Fin 2) :
    piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤ nSched spec (Masked.stsOf x) (2 * j.val + r.val) := by
  have hjs := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hk : 2 * j.val + r.val < spec.repStreams.size := by omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog := by
    rw [hst]
    exact (Masked.zkTab x j).wf.1.hk6
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hjs).1
  have hrq : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! +
        ((Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) + 4 := by
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk, Option.map_some,
      Option.getD_some]
    have e1 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + ((Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6) := by
      omega
    rw [repRounds_true]
    simp only [hz]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hk6' : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  unfold nSched
  rw [hrq, hst]
  unfold FlockComplete.piopRoundsZk piopNR
  have hI : INNER_ROUNDS = 4 := rfl
  generalize spec.points * (Masked.stAt x j).regions.size = P
  omega

/-- **The loop's coin count at a rep's PIOP round `jj`** is the PIOP's (`piopN`). -/
theorem loopNr_piop_at {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (j : Fin (Masked.jOf x)) (r : Fin 2) {jj : ℕ}
    (hjj : jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog) :
    loopNr x (Masked.repBase x (2 * j.val + r.val) + jj) = piopN (Masked.stAt x j).m jj := by
  have hjs := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hkk : 2 * j.val + r.val < spec.repStreams.size := by omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hjs).1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hpl := Count.piop_list (kLog := (Masked.stAt x j).c.kLog) (by omega : 14 ≤ (Masked.stAt x j).m)
  have hpL := congrArg List.length hpl
  simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil] at hpL
  unfold piopNR at hjj
  rw [Masked.repBase_eq hs, loopNr_plan hs hkk hz (by rw [hst]; exact hk6) (i := jj)
    (by unfold FlockComplete.piopRoundsZk; rw [hst]; omega),
    Count.plan_get hz (by rw [List.length_append, hst, hpL]; omega), hst,
    List.getElem?_append_left (by rw [hpL]; omega), hpl]
  exact piopN_list (by omega) hk6 (by unfold piopNR; omega)

/-- **A rep's PIOP rounds are scheduled**: after decoded coins `cs` reaching rep stream `2j + r`'s PIOP round `jj`, the
loop schedules a round there (`schedOf`): the inner proof's rounds sit past the rep's scheduled ones (`innerAt_slot`),
among which are the PIOP's (`piopNR_le_nSched`). -/
theorem sched_piop {real : ℕ → V (Tab × Array ByteArray)} {x : Msg} (hx : Masked.Live x) (hSv : Serves x)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} (hs : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    (hRT : Count.RealTabs (Masked.stsOf x) real) (j : Fin (Masked.jOf x)) (r : Fin 2) {cs : List FlockDeploy.Coin}
    {jj : ℕ} (hjj : jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog)
    (hn : cs.length = Masked.repBase x (2 * j.val + r.val) + jj) (hK : cs.length < refK x) :
    ∃ b, schedOf real x cs = some b := by
  have hjs := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hkk : 2 * j.val + r.val < spec.repStreams.size := by omega
  obtain ⟨-, hbase⟩ := repSlot_nSched hx hs hkk
  have hNS := piopNR_le_nSched hx hs j r
  have hstream := streamAt_repBase hs hkk
  unfold streamAt at hstream
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  rcases hsch : schedOf real x cs with _ | b
  · exfalso
    obtain ⟨_, hp, -⟩ := hx.1
    obtain ⟨s0', spec', sts', q, k, i, tr, hi, rsched, hs', hq, hloc, hrk, hrs, hle⟩ :=
      (schedOf_eq_none_iff (decodeRec_none_of_parse hp) (by omega) (by omega)).1 hsch
    rw [hs] at hs'
    obtain ⟨rfl, rfl, rfl⟩ : s0 = s0' ∧ spec = spec' ∧ Masked.stsOf x = sts' := by
      simpa only [Option.some.injEq, Prod.mk.injEq] using hs'
    have hloc' := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) jj (by omega)
      (by unfold nSched at hNS; omega)
    rw [Nat.zero_add, show ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum + jj = q by omega,
      hloc] at hloc'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hloc'
    obtain ⟨j', r', hk, -, hslot, -⟩ := innerAt_slot hx hSv hs hq hK hloc hRT hrk hrs hle
    obtain rfl : j' = j := Fin.ext (by have := r'.isLt; omega)
    obtain rfl : r' = r := Fin.ext (by omega)
    rw [Masked.repSlot_innerOff hs j' r', ← hbase] at hslot
    omega
  · exact ⟨b, rfl⟩

/-- **The outer coins after the rep's first round are not `1`** where no refusal shows by the rep's third round: the
outer coins' round is the rep's second (`outerSlot_eq`). -/
theorem outer_ne_one_two {x : Msg} {c : Fin (refK x) → FlockDeploy.Coin} {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hnr : ¬ Masked.Refused x c (Masked.repBase x (2 * j.val + r.val) + 2)) :
    ∀ k < (Masked.stAt x j).m - 13, Masked.slotAt c (Masked.repBase x (2 * j.val + r.val) + 1) k ≠ 1 := by
  have hj := Masked.lt_of_jOf j
  have hlt : Masked.outerSlot x j.val r.val < Masked.repBase x (2 * j.val + r.val) + 2 := by
    rw [Masked.outerSlot_eq hsx]
    omega
  intro k hk h1
  refine hnr (Or.inl ⟨j.val, hj, r.val, r.isLt, hlt, k, ?_, ?_⟩)
  · rw [getElem!_pos (Masked.stsOf x) j.val hj]
    exact hk
  · have e : Masked.outerAt c j.val r.val k = Masked.slotAt c (Masked.outerSlot x j.val r.val) k := rfl
    rw [e, Masked.outerSlot_eq hsx]
    exact h1

/-- **The loop's words are the opened coins** below the round's count, on a history of at most `refK x` openings. -/
theorem words_slot_le {x : Msg} {ops : List (Msg × Msg)} (hl : ops.length ≤ refK x) {p e : ℕ} (hp : p < ops.length)
    (he : e < loopNr x p) (hc : e < 65536) :
    (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) p)[e]! = Masked.slotAt (finCoins x ops) p e := by
  have hs := wordsAt_size x ops hp le_rfl
  rw [List.take_length] at hs
  have h := wordsAt_slot Masked.dataE x ops hl hp le_rfl (e := e)
    (by rw [List.take_length, hs]; exact Nat.lt_min.mpr ⟨he, hc⟩)
  rw [List.take_length] at h
  exact h

/-- The values a rep's frame took number `zp 64` (`valsOf`). -/
theorem valsOf_size (c : _root_.Flock.Firewall.Frame.Cfg) (M : Array F128) (fc : F128) :
    (valsOf c M fc).size = c.lay.zp 64 := by
  have h1 : ∀ (v : Array F128) (i : ℕ) (a : F128), (v.set! i a).size = v.size := fun v i a => by simp
  unfold valsOf
  dsimp only
  rw [h1, ← Array.foldl_toList, foldl_set_size _ (fun v e => by cases e; exact h1 _ _ _), Array.size_replicate]

set_option maxHeartbeats 1000000 in
/-- **The switched session's PIOP rounds of rep stream `2j + r`, at one binding**: where the coins show no refusal by the
rep's first round, level 0's cap and the pads root the session commits are under `2^64` bytes, and at each PIOP round
`jj` the coins show no refusal by, the session sends the PIOP's body `jj` at the switched source's masked values, table
`j`'s digest, that cap and that root (`Masked.runCut_piopBody`, `enc_finS_of`; level 0's cap is `2^cap` digests,
`commit0Cut_cap_lt`, and the pads root a digest, `padsTreeCut_root_go`). -/
theorem enc_piop_all (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (C : Fin (refK x) → finS.Coin) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hN0 : ¬ Masked.Refused x C (Masked.repBase x (2 * j.val + r.val))) :
    ∃ cp rt : ByteArray, cp.size < 2 ^ 64 ∧ rt.size < 2 ^ 64 ∧
      ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
        ¬ Masked.Refused x C (Masked.repBase x (2 * j.val + r.val) + jj) →
        finS.enc x C (Masked.repBase x (2 * j.val + r.val) + jj) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) =
          0 :: piopBody
            (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8).pending.data.toList
            (Masked.stAt x j).m (Masked.stAt x j).c.kLog (_root_.Flock.Setup.ofCircuit (Masked.stAt x j)).digest cp rt
            (Masked.layE x j) ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val).masked jj := by
  have hj := Masked.lt_of_jOf j
  obtain ⟨schs, hs, hsj⟩ := Masked.schedsPZk_live hx
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
  have hic : ∀ l, (FlockComplete.padsTreeCut Masked.hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val)).go l = (.ok ic, l) := fun l => by
    rw [FlockComplete.CutEq.padsTreeCut_quiet _ _ _ l, hic0]
  -- the bounds
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hst' : Masked.stAt x j = (Masked.stsOf x)[j.val] :=
    hstj.symm.trans (getElem!_pos (Masked.stsOf x) j.val hj)
  have hm1 : 25 ≤ (Masked.stAt x j).m := by
    rw [hst']
    exact (hall j.val hj).1
  have hm2 : (Masked.stAt x j).m ≤ 35 := by
    rw [hst']
    exact (hall j.val hj).2.1
  have hc32 := (FlockComplete.JsonRT.zkSchedule_cap0 hm1 hm2 (Masked.zkTab x j).hz).2
  have hclt := FlockComplete.JsonRT.zkSchedule_cap0_lt hm1 hm2 (Masked.zkTab x j).hz
  obtain ⟨cm, hcm', hlt⟩ :=
    FlockComplete.commit0Cut_cap_lt _ (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val) hclt hc32
  have hjt : j.val < (Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).size := by
    simpa [FlockDeploy.tabsOf] using hj
  have hcj : ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf)[j.val]! = cm := by
    rw [getElem!_pos _ j.val (by simpa using hjt)]
    simp only [Array.getElem_map, Array.getElem_range]
    have e := hcm j.val
    rw [hsj j, hcm'] at e
    exact (Except.ok.inj e).symm
  have hrt : ic.root.size < 2 ^ 64 := FlockComplete.padsTreeCut_root_go (FlockComplete.two_le_padsN _) hic
  refine ⟨FlockComplete.flat cm.cap, ic.root, hlt, hrt, fun jj hjj hN => ?_⟩
  have h1 : 1 ≤ Masked.repBase x (2 * j.val + r.val) := by
    rw [Masked.repBase_eq hsx]
    unfold FlockComplete.repStartZk
    omega
  have hrc := Masked.runCut_piopBody hx C (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t) hsx hs hcms j r hN0
    hext hic (Masked.tabsOf_stAt x j) (jj := jj) hjj
  have hsrc : Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x C) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) :=
    rfl
  rw [enc_finS_of C (by omega) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) hsx hN
    (by rw [← hsrc]; exact hrc.1), ← hsrc, hrc.2, hcj] <;> rfl

/-- The first `n` openings' coins are the padded coins' first `n`, at most `K` openings. -/
theorem map_take_pad (S : RefSession) {K n : ℕ} (ops : List (Msg × Msg)) (hK : ops.length ≤ K)
    (hn : n ≤ ops.length) :
    (ops.take n).map (fun p => S.decOf p.2) = (List.ofFn (S.pad (K := K) (ops.map fun p => S.decOf p.2))).take n := by
  have e1 := S.ofFn_pad_take (K := K) (ops.map fun p => S.decOf p.2) (by simpa using hK)
  simp only [List.length_map, Nat.min_eq_left hK] at e1
  calc (ops.take n).map (fun p => S.decOf p.2) = (ops.map fun p => S.decOf p.2).take n := by
        simp [List.map_take]
    _ = ((List.ofFn (S.pad (K := K) (ops.map fun p => S.decOf p.2))).take ops.length).take n := by
        rw [e1]
    _ = (List.ofFn (S.pad (K := K) (ops.map fun p => S.decOf p.2))).take n := by
        rw [List.take_take, Nat.min_eq_left hn]

set_option maxHeartbeats 1000000 in
/-- **A scheduled round of the worker's own history is the switched session's at the opened coins**: on a live history
of fewer than `refK x` openings whose openings pass, the release's own run (`ZkServer.OwnHist`), where `LiveSchedOwn`
holds and `vs` is causal at the coins `SwOk` admits, message `n + 1` (`1 ≤ n`), whose round the loop schedules
(`schedOf`), is the switched session's round `n` at the opened coins, without its leading domain frame: `LiveSchedOwn`
at the history's prefix, whose coins agree with the opened ones below the round and which `SwOk` admits, as it does the
opened coins, since the live history shows no refusal (`swOk_take_of_live`, `swOk_of_live`, `swMsgAt_take_on`). -/
theorem msg_liveSchedOwn (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hSch : LeanRelease.LiveSchedOwn vs L₀ src real wk x w) (hcaus : finS.SwCausalOn (SwOk Masked.dataE) vs)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hlt : ops.length < refK x)
    (hl : (LeanRelease.ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (hown : (LeanRelease.ofLoopLink L₀ src real).toSrv.OwnHist (LeanRelease.ofLoopLink L₀ src real).fillOf wk x w
      (ν, t) κ ((m₀, ρ₀) :: ops))
    {n : ℕ} (h1 : 1 ≤ n) (hn : n < ops.length)
    (hsch : ∃ b, schedOf (real x w (ν, t)) x ((ops.take n).map fun (p : Msg × Msg) => CoinLayer.rd p.2) = some b) :
    (((m₀, ρ₀) :: ops)[n + 1]'(by simp only [List.length_cons]; omega)).1 =
      dropDom (finS.enc x (finCoins x ops) n
        (finS.swCutOf x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t)) := by
  have hlast : (LeanRelease.ofLoopLink L₀ src real).last x = refK x + 1 :=
    LeanRelease.loop_last loopNr Masked.dataE _ x
  have hj' : n + 1 < ((m₀, ρ₀) :: ops).length := by simp only [List.length_cons]; omega
  have htake : ((m₀, ρ₀) :: ops).take (n + 1) = (m₀, ρ₀) :: ops.take n := rfl
  have hlj : (((m₀, ρ₀) :: ops).take (n + 1)).length = n + 1 := by
    simp only [List.length_take, List.length_cons]
    omega
  have hk : (LeanRelease.ofLoopLink L₀ src real).kindAt x (((m₀, ρ₀) :: ops).take (n + 1)).length = .masked := by
    rw [hlj]
    exact LeanRelease.kindAt_mid hlast (by omega) (by omega)
  have hl' := (LeanRelease.ofLoopLink L₀ src real).live_take wk x w (ν, t) κ _ hl (n + 1)
  obtain ⟨v, hv, -⟩ := (LeanRelease.ofLoopLink L₀ src real).live_some wk x w (ν, t) κ _ hk hl'
  have hu : (LeanRelease.ofLoopLink L₀ src real).toSrv.wkD (LeanRelease.ofLoopLink L₀ src real).fillOf wk κ x w
      (ν, t) (((m₀, ρ₀) :: ops).take (n + 1)) 0 = some (v, false) := by
    simp only [ZkServer.wkD, hl', ↓reduceIte, hv]
    rfl
  have hr := (LeanRelease.ofLoopLink L₀ src real).refusesCoins_of_live wk x w (ν, t) κ _ hk hl'
  have hown' : (LeanRelease.ofLoopLink L₀ src real).toSrv.OwnHist (LeanRelease.ofLoopLink L₀ src real).fillOf wk x w
      (ν, t) κ (((m₀, ρ₀) :: ops).take (n + 1)) := by
    intro i hi
    have hi1 : i ≤ n + 1 := by rw [hlj] at hi; omega
    rw [List.getElem_take, List.take_take, Nat.min_eq_left hi1]
    exact hown i (by simp only [List.length_cons]; omega)
  refine (hown (n + 1) hj').trans ?_
  rw [(LeanRelease.ofLoopLink L₀ src real).frameD_toSrv (LeanRelease.ofLoopLink L₀ src real).fillOf wk x w (ν, t) κ _
    hk hu]
  rw [htake] at hl' hr hv hown'
  rw [htake]
  have hlen' : (ops.take n).length = n := by
    simp only [List.length_take]
    omega
  obtain ⟨b, hb⟩ := hsch
  refine (hSch ν t κ m₀ ρ₀ (ops.take n) v b (by rw [hlen']; omega) (by rw [hlen']; omega) hl' ?_ hr hown' hv
    hb).trans ?_
  · rw [allOk_eval] at hok ⊢
    intro i hi
    simp only [List.length_map] at hi
    have := hok i (by simp only [List.length_map]; omega)
    simpa [List.getElem_map, List.getElem_take] using this
  · congr 1
    have hkF : (LeanRelease.ofLoopLink L₀ src real).kindAt x ((m₀, ρ₀) :: ops).length = .masked := by
      simp only [List.length_cons]
      exact LeanRelease.kindAt_mid hlast (by omega) (by omega)
    have hrF := (LeanRelease.ofLoopLink L₀ src real).refusesCoins_of_live wk x w (ν, t) κ _ hkF hl
    have hO : SwOk Masked.dataE x
        (finS.pad ((List.ofFn (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2))).take n)) := by
      rw [← map_take_pad finS ops hlt.le hn.le]
      exact LeanRelease.swOk_take_of_live Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real
        wk x w (ν, t) κ m₀ ρ₀ ops hlt.le hl hrF n
    have hO' : SwOk Masked.dataE x (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) :=
      LeanRelease.swOk_of_live Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real wk x w
        (ν, t) κ m₀ ρ₀ ops hlt.le hl hrF
    rw [map_take_pad finS ops hlt.le hn.le, finS.swMsgAt_take_on vs (SwOk Masked.dataE) x w ν hcaus t _ n
      (by omega) hO hO'] <;> rfl

variable (L₁ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))

/-- **The finish at a rep's inner round**: at a history `InnerHist` holds at with `e < 4`, on the loop's own tables at a
statement in the masked scope with a witness that the loop serves, the loop's table and request for rep stream `2j + r`
return; the table's pad layout and the round's configuration are table `j`'s, the configuration's pads the tape's
(`Loop.RepSrc`); and where `finInner` on them returns at `final_c = a` after the stream's `nSched + e` rounds of coins,
and at no other `final_c`, `finOf` is its round 0 and round 2. -/
theorem innerHist_fin (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (wk : (relIn L₁ src real).toSrv.Worker) {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk}
    {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2} {e : ℕ} (he : e < 4)
    (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e) :
    ∃ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) (tr : Tab) (hi : Array ByteArray) (q₀ : RepIn),
      FlockDeploy.setupOf true x = some (s0, spec, sts) ∧
      real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi) ∧
      repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀ ∧
      _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.layE x j ∧
      ((relIn L₁ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay = Masked.layE x j ∧
      ((relIn L₁ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h =
        (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h ∧
      ∀ (a fc' : F128) (tc : ByteArray),
        finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
          (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e) a) = .ok (fc', tc) →
        (∀ a', (finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val))
          (nSched spec sts (2 * j.val + r.val))
          (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e) a')).isOk = true → a' = a) →
        finOf real L₁.draws x w (ν, t) ((m₀, ρ₀) :: ops) = #[inner0 fc'.toBytes, .empty, inner2 tc, .empty] := by
  obtain ⟨h0, hlt, hl, -, -, -, ⟨v, hv⟩, hI, hrep, hslot⟩ := hH
  have hx : Masked.Live x := Masked.live_of hsc hR
  obtain ⟨s0, spec, sts, q, k, i, tr, hi, rsched, hxs, hq, hloc, hrk, hrs, hle⟩ := hI
  have hq1 : ops.length = q + 1 := by rw [← hq, List.length_map]
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hxs]
  subst hsts
  have hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)) := by
    subst hreal
    exact realTabs_of src x w (ν, t) hxs
  obtain ⟨-, -, -, -, -, -, -, hi0, -, -⟩ := innerAt_slot hx hSv hxs hq
    (by rw [List.length_map]; exact hlt) hloc hRT hrk hrs hle
  -- the history's rep is the stream's, `2j + r`
  have hrep' : (relIn L₁ src real).rep x (ops.length + 1) = k := by
    have e : (relIn L₁ src real).rep x (ops.length + 1) = (CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 := by
      show Loop.repOf x (ops.length + 1) = _
      rw [show ops.length + 1 = q + 2 by omega, LeanRelease.repOf_add_two hxs]
    rw [e, hloc]
  have hkj : k = 2 * j.val + r.val := hrep'.symm.trans hrep
  subst hkj
  have hr2 := r.isLt
  have hjs := Masked.lt_of_jOf j
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by omega
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hxs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hkk : 2 * j.val + r.val < spec.repStreams.size := by omega
  -- the round is the stream's `nSched + e`
  obtain ⟨-, hbase⟩ := repSlot_nSched hx hxs hkk
  have hstream := streamAt_repBase hxs hkk
  unfold streamAt at hstream
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hsum : (roundsOf spec (Masked.stsOf x)).sum + 1 = refK x := by
    rw [Masked.refK_eq hxs, ← FlockComplete.ZkTables.repStart_last, ← hrsz]
    unfold FlockComplete.repStartZk roundsOf
    omega
  have hinv := locate_inv (roundsOf spec (Masked.stsOf x)) 0 q (by omega)
  rw [hloc] at hinv
  simp only [Nat.sub_zero] at hinv
  obtain ⟨hqs, -⟩ := hinv
  have hoff := Masked.repSlot_innerOff hxs j r e
  have hie : i = nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e := by omega
  -- the table's layout is table `j`'s
  obtain ⟨g, d, l, -, htr⟩ := hRT _ tr hi hrk
  obtain ⟨hts, hsch⟩ := Count.tabOfLevel0_sched htr
  have hst' : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hjs
  have hsc' : _root_.Flock.Zk.schedule (Masked.stAt x j).m = .ok (tr.sch, tr.tPad) := by
    have e : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m = .ok (tr.sch, tr.tPad) := hsch
    rwa [hk2, hst'] at e
  rw [(Masked.zkTab x j).hz] at hsc'
  have htsch : tr.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (Except.ok.inj hsc')).1.symm
  have htst : tr.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := by
    rw [hts, hk2, hst']
  have hLayT : _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.layE x j := by
    rw [htst, htsch]
    rfl
  -- the rep's frame starts, so the request returns
  have hlast : (relIn L₁ src real).last x = refK x + 1 := LeanRelease.loop_last loopNr Masked.dataE _ x
  have hkd : (relIn L₁ src real).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    LeanRelease.kindAt_masked_of _ (by rw [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, -, -, ha, -⟩ :=
    LeanRelease.check_of_live (relIn L₁ src real) wk x w (ν, t) κ ((m₀, ρ₀) :: ops) v hl hkd hv
  have hloc' : CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q = (2 * j.val + r.val, i - 1 + 1) := by
    rw [Nat.sub_add_cancel hi0]
    exact hloc
  obtain ⟨lay, mask, hst0⟩ := LeanRelease.ofLoop_start_at Masked.dataE
    (LeanRelease.withFiles (LeanRelease.withLink (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src)
      src) real x (ν, t) (m₀, ρ₀) ops ha rfl hxs hq1 hloc'
  obtain ⟨q₀, hq₀⟩ : ∃ q₀, repInOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) (2 * j.val + r.val)
      (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀ := by
    unfold repInOf
    simp only [hst0]
    exact ⟨_, rfl⟩
  -- the configuration's layout and pads are the tape's
  have hst := hst0
  rw [hk2] at hst
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src
    real j r hst
  have hd := LeanRelease.draws_ofLoopLink (L₀ := LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src
    real
  have hz := hSrc.2.size (relIn L₁ src real) hd ν t j r hlay
  have hpads := hSrc.2.pads (relIn L₁ src real) hd ν t j r hlay (Masked.UE.tapeZk t)
  obtain ⟨hHc, -, -⟩ := cfg_tapeOfDraws (relIn L₁ src real) x (ν, t) ((m₀, ρ₀) :: ops) (Masked.UE.tapeZk t)
    Loop.frepOf j.val r.val hrep hz
  rw [hlay, ← hpads] at hHc
  have hLay : ((relIn L₁ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay = Masked.layE x j := by
    show (relIn L₁ src real).lay x ((relIn L₁ src real).rep x (ops.length + 1)) = _
    rw [hrep]
    exact hlay
  refine ⟨s0, spec, Masked.stsOf x, tr, hi, q₀, hxs, hrk, hq₀, hLayT, hLay, hHc, ?_⟩
  -- `finOf` is `finInner` at the `final_c` it accepts
  intro a fc' tc hfi huniq
  have hok : (finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val))
      (nSched spec (Masked.stsOf x) (2 * j.val + r.val))
      (askOf q₀ (Masked.layE x j).fc (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e) a)).isOk = true := by
    rw [hfi] <;> rfl
  unfold finOf
  simp only [hxs, List.tail_cons, hq1]
  generalize CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q = L at hloc ⊢
  subst hloc hie
  simp only [hrk, hq₀]
  rw [hLayT]
  split
  · rename_i hA
    rw [huniq _ hA.choose_spec, hfi] <;> rfl
  · rename_i hn
    exact (hn ⟨a, hok⟩).elim

set_option maxHeartbeats 4000000 in
/-- **The loop's request at a rep's inner round, on the worker's own history**: at a history `InnerHist` holds at, where
`LiveSchedOwn` holds and `vs` is causal at `SwOk`, the request for rep stream `2j + r` (`repInOf`) has `zp 64` values,
`vs`'s below it but at `final_c`'s pad (`vsWord`, at the coins the openings open), and the tape's pads. The history's
rounds of the rep's PIOP are scheduled (`sched_piop`) and show no refusal (`LeanRelease.unrefused_prefix`), so they are
the switched session's PIOP bodies at one binding (`msg_liveSchedOwn`, `enc_piop_all`); the loop's frame takes back the
masked values they carry (`maskedOf_piop`), `vs`'s under the tape's pads (`swSrcE_masked_vsWord`), and the request's
pads are the tape's (`Masked.pads_frame_eq`). -/
theorem innerReq_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausalOn (SwOk Masked.dataE) vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hSch : LeanRelease.LiveSchedOwn vs (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src real wk
      x w) {ν : finS.Nonce} {t : finS.Tape}
    {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2}
    {e : ℕ} (he : e < 4) (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀) :
    q₀.vals.size = (Masked.layE x j).zp 64 ∧
      (∀ i < (Masked.layE x j).zp 64, i ≠ (Masked.layE x j).fc →
        q₀.vals[i]! = vsWord vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r i) ∧
      (_root_.Flock.Firewall.Finish.padsOf q₀.draws).toC =
        FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j) := by
  obtain ⟨-, hlt, hl, hok, hrf, hown, -, -, -, hslot⟩ := hH
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hs]
  subst hsts
  have hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)) := by
    subst hreal
    exact realTabs_of src x w (ν, t) hs
  have hjs := Masked.lt_of_jOf j
  have hr2 := r.isLt
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by omega
  have hmod : (2 * j.val + r.val) % 2 = r.val := by omega
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hkk : 2 * j.val + r.val < spec.repStreams.size := by omega
  -- the history's next round is past the rep's scheduled ones, among which are its PIOP's
  obtain ⟨-, hbase⟩ := repSlot_nSched hx hs hkk
  have hNS := piopNR_le_nSched hx hs j r
  rw [Masked.repSlot_innerOff hs j r e, ← hbase] at hslot
  have hstream := streamAt_repBase hs hkk
  unfold streamAt at hstream
  have hR0 : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
      (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    have h := hNS
    unfold nSched at h
    omega
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hjs
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hjs).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hjs).2.1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hNRv : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog =
      6 + ((Masked.stAt x j).m - 6) + ((Masked.stAt x j).c.kLog - 6) := rfl
  -- no refusal shows through the rep's PIOP
  have hRk : Masked.rankSlot x ((2 * j.val + r.val) / 2) ((2 * j.val + r.val) % 2) =
      Masked.repBase x (2 * j.val + r.val) + ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    rw [hk2, hmod, show Masked.rankSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) +
      ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) from rfl, hstj]
  have hunref : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      ¬ Masked.Refused x (finCoins x ops) (Masked.repBase x (2 * j.val + r.val) + jj) := fun jj hjj =>
    LeanRelease.unrefused_prefix Masked.dataE
      (LeanRelease.withFiles (LeanRelease.withLink (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws))
        src) src) real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops hlt.le hl hrf
      (N := Masked.repBase x (2 * j.val + r.val) + jj) (by omega) (q := 2 * j.val + r.val) (by omega)
      (Nat.le_add_right _ _) (by rw [hRk]; omega)
  -- the history's PIOP rounds are the switched session's bodies at one binding
  obtain ⟨cp, rt, hcp, hrt, hbodyE⟩ := enc_piop_all vs w hx hs ν t (finCoins x ops) j r (hunref 0 (by omega))
  have hd := (sizesOk_at (sizesOk_of_serves hSv s0 spec _ hs) hkk).2.1
  rw [hk2, hmod] at hd
  have hbody : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      ((ops.getD (Masked.repBase x (2 * j.val + r.val) + jj) ([], [])).1).tail =
        FlockSoundness.Refine.encs (piopFr (Masked.stAt x j).m (Masked.stAt x j).c.kLog
          (_root_.Flock.Setup.ofCircuit (Masked.stAt x j)).digest cp rt (Masked.layE x j)
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t j.val).rep r.val).masked
            jj) ++
          (FlockSoundness.Refine.Frame.sq (piopK jj) (piopN (Masked.stAt x j).m jj)).enc := by
    intro jj hjj
    have hp : Masked.repBase x (2 * j.val + r.val) + jj < ops.length := by omega
    have hm := msg_liveSchedOwn vs (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src real wk x w
      hSch hcaus hlt hl hok hown (n := Masked.repBase x (2 * j.val + r.val) + jj) (by omega) hp
      (sched_piop hx hSv hs hRT j r hjj (by rw [List.length_map, List.length_take]; omega)
        (by rw [List.length_map, List.length_take]; omega))
    rw [hbodyE jj hjj (hunref jj hjj), dropDom_piopBody _ hd] at hm
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hp, Option.getD_some]
    have hge : (ops[Masked.repBase x (2 * j.val + r.val) + jj]'hp) =
        ((m₀, ρ₀) :: ops)[Masked.repBase x (2 * j.val + r.val) + jj + 1]'(by simp only [List.length_cons]; omega) :=
      rfl
    rw [hge, hm] <;> rfl
  -- the loop's coin counts there
  have hNR : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      loopNr x (Masked.repBase x (2 * j.val + r.val) + jj) = piopN (Masked.stAt x j).m jj :=
    fun _ hjj' => loopNr_piop_at hx hs j r hjj'
  have hwords : ∀ jj, jj + 1 < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog →
      (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + jj)).size =
        piopN (Masked.stAt x j).m jj := by
    intro jj hjj'
    have hp : Masked.repBase x (2 * j.val + r.val) + jj < ops.length := by omega
    have hs' := wordsAt_size x ops hp le_rfl
    rw [List.take_length, hNR jj (by omega)] at hs'
    rw [hs']
    exact Nat.min_eq_left (by unfold piopN _root_.Flock.Firewall.Frames.COIN_MAX; split_ifs <;> omega)
  -- the zerocheck's outer coins are not `1`
  have hout : ∀ e' < (Masked.stAt x j).m - 13,
      ((wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + 1))[e']! +
        1).isZero = false := by
    intro e' he'
    have h1' : loopNr x (Masked.repBase x (2 * j.val + r.val) + 1) = (Masked.stAt x j).m - 13 :=
      hNR 1 (by omega)
    rw [words_slot_le hlt.le (p := Masked.repBase x (2 * j.val + r.val) + 1) (e := e') (by omega)
      (by rw [h1']; exact he') (by omega)]
    exact isZero_add_one (outer_ne_one_two hs j r (hunref 2 (by omega)) e' he')
  -- the rounds the loop released
  have hrs : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      (seenRounds x ((m₀, ρ₀) :: ops) (roundsOf spec (Masked.stsOf x)) (2 * j.val + r.val))[jj]? =
        some (ByteArray.mk (FlockSoundness.Refine.encs (piopFr (Masked.stAt x j).m (Masked.stAt x j).c.kLog
            (_root_.Flock.Setup.ofCircuit (Masked.stAt x j)).digest cp rt (Masked.layE x j)
            ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t j.val).rep
              r.val).masked jj) ++
          (FlockSoundness.Refine.Frame.sq (piopK jj) (piopN (Masked.stAt x j).m jj)).enc).toArray,
          piopN (Masked.stAt x j).m jj,
          wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + jj)) := by
    intro jj hjj'
    unfold seenRounds
    rw [List.getElem?_map, List.getElem?_range (by omega)]
    refine congrArg some ?_
    dsimp only
    rw [List.tail_cons, show streamAt (roundsOf spec (Masked.stsOf x)) (2 * j.val + r.val) + jj + 1 =
      Masked.repBase x (2 * j.val + r.val) + jj by unfold streamAt; omega, hbody jj hjj', hNR jj hjj']
  -- the request's layout is table `j`'s
  unfold repInOf at hq
  obtain ⟨⟨lay, msk⟩, hds, hq'⟩ := v_bind_ok hq
  obtain ⟨sch', tp', hsch', hlay⟩ := start_lay hds
  rw [hz] at hsch'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hsch'
  obtain ⟨rfl, rfl⟩ := hsch'
  have hnM : lay.nMlv = (Masked.stAt x j).m - 6 := by
    have h := start_nMlv hds
    rwa [hst] at h
  rw [hst] at hlay
  have hlayE : lay = Masked.layE x j := hlay
  subst hlayE
  have hnL : (Masked.layE x j).nLc = (Masked.stAt x j).c.kLog - 6 := rfl
  -- the request's pads are the tape's
  have hqd : q₀.draws = _root_.Flock.Firewall.Frame.Draws.ofBytes (Masked.layE x j)
      (L₁.draws (ν, t) (2 * j.val + r.val)) := by
    cases hq'
    rfl
  have hst0 := hds
  rw [hk2] at hst0
  have hlayR := LeanRelease.lay_ofLoopLink (L₀ := LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws))
    src real j r hst0
  have hd' := LeanRelease.draws_ofLoopLink (L₀ := LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws))
    src real
  have hzS := hSrc.2.size (relIn L₁ src real) hd' ν t j r hlayR
  have hpads := hSrc.2.pads (relIn L₁ src real) hd' ν t j r hlayR (Masked.UE.tapeZk t)
  have hq₀d : q₀.draws = (relIn L₁ src real).drawsAt x (ν, t) (Loop.frepOf j.val r.val) := by
    rw [hqd]
    show _ = _root_.Flock.Firewall.Frame.Draws.ofBytes ((relIn L₁ src real).lay x (Loop.frepOf j.val r.val))
      (L₁.draws (ν, t) (Loop.frepOf j.val r.val))
    rw [hlayR] <;> rfl
  have hP := Masked.pads_frame_eq (relIn L₁ src real) (ν, t) (Masked.UE.tapeZk t) Loop.frepOf x t j r q₀ hq₀d hlayR
    hzS hpads
  -- the masked values the loop's frame took, and the request's values
  have hdg : (_root_.Flock.Setup.ofCircuit (Masked.stAt x j)).digest.size < 2 ^ 64 := by
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  obtain ⟨tail, htail⟩ := maskedOf_piop (Masked.layE x j) (L₁.draws (ν, t) (2 * j.val + r.val)) none
    (Masked.stAt x j).m (Masked.stAt x j).c.kLog (by omega) hk6 (lt_of_le_of_lt hm35 (by norm_num)) hnM hnL _ _ _ hdg
    hcp hrt
    ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops) (finS.seenOf x t)) t j.val).rep r.val).masked
    (fun jj => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + jj)) _ hrs
    hwords hout
  have hM := congrArg List.length htail
  simp only [Array.length_toList, List.length_append, List.length_map, List.length_range,
    List.length_range'] at hM
  cases hq'
  refine ⟨valsOf_size _ _ _, fun i hi hne => ?_, hP⟩
  have hiE : i < (Masked.tabC Masked.PE x j).z.nE := by
    rw [Masked.nE_eq_hab]
    exact lt_trans hi (Masked.zp64_lt_hab x j)
  have hPh : (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h =
      (_root_.Flock.Firewall.Frame.Draws.ofBytes (Masked.layE x j) (L₁.draws (ν, t) (2 * j.val + r.val))).h :=
    (congrArg (fun p => p.h) hP).symm
  dsimp only
  rw [valsOf_at (Masked.layE x j) _ none _ _ hi hne (by omega), getElem!_absPos htail hi hne,
    swSrcE_masked_vsWord vs x w ν (finCoins x ops) t j r hiE, hPh, add_pad_pad]
  rfl

/-- **The request's coins** at a history `InnerHist` holds at: it holds a round of coins for each of rep stream
`2j + r`'s rounds, and its rounds before the history's next, `nSched + e` of them, hold the loop's count of coins
(`loopNr`, at most `COIN_MAX`), below which they are the opened coins from the rep's first round (`Masked.repBase`). As
`finRepCoins` at a `Finish`, below the history's length. -/
theorem innerReq_coins (x : Msg) (w : Flock.Wit)
    (wk : (relIn L₁ src real).toSrv.Worker) {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk}
    {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2} {e : ℕ} (he : e < 4)
    (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀) :
    q₀.coins.size = (roundsOf spec sts).getD (2 * j.val + r.val) 0 ∧
      nSched spec sts (2 * j.val + r.val) + e < q₀.coins.size ∧
      ∀ k' < nSched spec sts (2 * j.val + r.val) + e,
        (q₀.coins[k']!).size =
            min (loopNr x (Masked.repBase x (2 * j.val + r.val) + k')) _root_.Flock.Firewall.Frames.COIN_MAX ∧
          ∀ i < (q₀.coins[k']!).size,
            Masked.slotAt (finCoins x ops) (Masked.repBase x (2 * j.val + r.val) + k') i = (q₀.coins[k']!)[i]! := by
  obtain ⟨-, hlt, -, -, -, -, -, -, -, hslot⟩ := hH
  have hx := Masked.live_of_lt j
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hs]
  subst hsts
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hjs := Masked.lt_of_jOf j
  have hr2 := r.isLt
  rw [Masked.repSlot_innerOff hs j r e] at hslot
  generalize hk2 : 2 * j.val + r.val = k at hslot hq ⊢
  have hk : k < spec.repStreams.size := by omega
  obtain ⟨h4, hbase⟩ := repSlot_nSched hx hs hk
  rw [← hbase] at hslot
  have hI : INNER_ROUNDS = 4 := rfl
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, -, hq⟩ := v_bind_ok hq
  have hqc : q₀.coins =
      ((seenRounds x ((m₀, ρ₀) :: ops) (roundsOf spec (Masked.stsOf x)) k).map fun r => r.2.2).toArray := by
    cases hq
    rfl
  have hsz : q₀.coins.size = (roundsOf spec (Masked.stsOf x)).getD k 0 := by
    rw [hqc, List.size_toArray, List.length_map]
    simp only [seenRounds, List.length_map, List.length_range]
  have hstream := streamAt_repBase hs hk
  have hns : nSched spec (Masked.stsOf x) k + INNER_ROUNDS = (roundsOf spec (Masked.stsOf x)).getD k 0 := by
    unfold nSched
    omega
  refine ⟨hsz, by omega, fun k' hk' => ?_⟩
  have hkc : k' < q₀.coins.size := by omega
  have hQ : Masked.repBase x k + k' < ops.length := by omega
  have hQK : Masked.repBase x k + k' < refK x := by omega
  have hent :
      q₀.coins[k']! = wordsAt x (ops.map fun (p : Msg × Msg) => CoinLayer.rd p.2) (Masked.repBase x k + k') := by
    rw [getElem!_pos q₀.coins k' hkc]
    simp only [hqc, List.getElem_toArray, List.getElem_map]
    unfold seenRounds
    simp only [List.getElem_map, List.getElem_range, List.tail_cons]
    congr 1
    omega
  have hw : wordsAt x (ops.map fun (p : Msg × Msg) => CoinLayer.rd p.2) (Masked.repBase x k + k') =
      (CoinLayer.rd ops[Masked.repBase x k + k'].2).vals
        (min (loopNr x (Masked.repBase x k + k')) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold wordsAt
    rw [List.getElem?_map, List.getElem?_eq_getElem hQ]
    rfl
  rw [hent, hw]
  refine ⟨by simp [FlockDeploy.Coin.vals], fun i hi => ?_⟩
  rw [getElem!_pos _ i hi]
  simp only [FlockDeploy.Coin.vals, Array.getElem_map, Array.getElem_range]
  unfold Masked.slotAt
  rw [dite_eq_left hQK]
  show ((ops.map fun (p : Msg × Msg) => CoinLayer.rd p.2).getD (Masked.repBase x k + k') default).at i = _
  rw [List.getD_eq_getElem _ _ (by simpa using hQ), List.getElem_map]

/-- **The request's extra claims**: one per link point and region of rep stream `2j + r`'s table, the link points the
`Commit`'s coin carries (`repInOf`, `sizesOk`). -/
theorem innerReq_xs {x : Msg} (hx : Masked.Live x) (hSv : Serves x) {ops : List (Msg × Msg)} (hpos : 0 < ops.length)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} (hs : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    (j : Fin (Masked.jOf x)) (r : Fin 2) {m₀ ρ₀ : Msg} {dr : ByteArray} {hi : Array ByteArray} {q₀ : RepIn}
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) (2 * j.val + r.val) dr hi = .ok q₀) :
    q₀.xs.size = spec.points * (Masked.stAt x j).regions.size := by
  have hjs := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hkk : 2 * j.val + r.val < spec.repStreams.size := by omega
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, -, hq⟩ := v_bind_ok hq
  have hqx : q₀.xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (wordsAt x (ops.map fun (p : Msg × Msg) => CoinLayer.rd p.2) 0)) := by
    cases hq
    rfl
  obtain ⟨⟨hmp, hmp0, hpc⟩, -, -, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hs) hkk
  have hls : (wordsAt x (ops.map fun (p : Msg × Msg) => CoinLayer.rd p.2) 0).size =
      min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    obtain ⟨p0, rest, hpr⟩ : ∃ p0 rest, ops = p0 :: rest := by
      cases ops with
      | nil => simp at hpos
      | cons p0 rest => exact ⟨p0, rest, rfl⟩
    rw [hpr]
    simp [wordsAt, FlockDeploy.Coin.vals]
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hs]
    rfl
  have hpts : (wordsAt x (ops.map fun (p : Msg × Msg) => CoinLayer.rd p.2) 0).size /
      (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.mPts = spec.points := by
    rw [hls, hl0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0]
  rw [hqx, Count.extras_size, Count.linkPoints_size, hpts, Nat.mul_comm, Masked.stsOf_half x j r]

/-- **`finInner` returns the request's `final_c` under its pad**: the masked `final_c` its round 0 absorbs is
`repCons`'s zerocheck proof's, the PIOP's masked value at `final_c`'s pad. -/
theorem finInner_fc {t : Tab} {rep : ℕ} {dom : ByteArray} {n : ℕ} {q : RepIn} {fc' : F128} {tc : ByteArray}
    (h : finInner t rep dom n q = .ok (fc', tc)) :
    fc' = q.vals[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc]! +
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc]! := by
  have key : ∀ {c : F128} {x : TMW (StateT ℕ V) (F128 × ByteArray)} {tr : _root_.Flock.Transcript} {k : ℕ}
      {a : F128 × ByteArray} {tr' : _root_.Flock.Transcript} {k' : ℕ},
      (x.run tr).run k = .ok ((a, tr'), k') → Rets (fun r => r.1 = c) x → a.1 = c :=
    fun e hx => hx _ _ _ _ _ e
  unfold finInner at h
  obtain ⟨ic, -, h⟩ := v_bind_ok h
  obtain ⟨⟨⟨r, tr'⟩, k'⟩, hrun, h⟩ := v_bind_ok h
  cases h
  refine key hrun (rets_bind' (repCons_rets _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) ?_)
  rintro ⟨p, zl, cons⟩ hp
  repeat (first | refine rets_bind fun _ => ?_ | exact rets_pure hp | exact rets_throw _ | split | dsimp only)

theorem ok_of_isOk {α : Type} {v : V α} (h : v.isOk = true) : ∃ a, v = .ok a := by
  cases v with
  | error e => cases h
  | ok a => exact ⟨a, rfl⟩

/-- **Where `finInner` returns**, its `repCons`, answered by the request's coins and then ones (`asksPad`) from round `0`
and a transcript with no rounds, returns the proof `p` and the constraints `cons` at the transcript `tF` and round `kF`,
where `consWhy` does not refuse them. -/
theorem finInner_parts {t : Tab} {rep : ℕ} {dom : ByteArray} {n : ℕ} {q : RepIn} {res : F128 × ByteArray}
    (h : finInner t rep dom n q = .ok res) :
    ∃ (ic : _root_.Flock.Firewall.Finish.InnerCommit) (p : _root_.Flock.Proof) (zl : _root_.Flock.Zk.Level0Msgs)
      (cons : Array (String × _root_.Flock.Zk.Lin)) (tF : _root_.Flock.Transcript) (kF : ℕ),
      ((_root_.Flock.Firewall.Finish.repCons (asksPad q.coins)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) t.st t.sch t.tPad rep dom t.g t.cm0
        (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws) ic q.vals t.extra
        t.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
        (_root_.Flock.Transcript.init #[] dom)).run 0 = .ok (((p, zl, cons), tF), kF) ∧
      _root_.Flock.Firewall.Finish.consWhy n kF cons (_root_.Flock.Firewall.Finish.padsOf q.draws).h = none := by
  unfold finInner at h
  obtain ⟨ic, -, h⟩ := v_bind_ok h
  obtain ⟨⟨⟨r, tr'⟩, k'⟩, hrun, -⟩ := v_bind_ok h
  obtain ⟨⟨p, zl, cons⟩, tF, kF, h1, h2⟩ := _root_.Flock.Firewall.Finish.tmw_bind_ok hrun
  refine ⟨ic, p, zl, cons, tF, kF, h1, ?_⟩
  try dsimp only at h2
  obtain ⟨k, t₁, k₁, h3, h4⟩ := _root_.Flock.Firewall.Finish.tmw_bind_ok h2
  have e3 : k = kF := by
    cases h3
    rfl
  rcases hcw : _root_.Flock.Firewall.Finish.consWhy n k cons (_root_.Flock.Firewall.Finish.padsOf q.draws).h with
    _ | why
  · rw [← e3]
    exact hcw
  · exfalso
    try dsimp only at h4
    rw [hcw] at h4
    try dsimp only at h4
    cases h4

/-- **`consWhy` passes** only where the rep took the request's rounds and its `zerocheck c_eval` constraints vanish on
the pads. -/
theorem consWhy_none {n k : ℕ} {cons : Array (String × _root_.Flock.Zk.Lin)} {h : Array F128}
    (hw : _root_.Flock.Firewall.Finish.consWhy n k cons h = none) :
    k = n ∧ ∀ q ∈ cons, q.1 = "zerocheck c_eval" → _root_.Flock.Firewall.Finish.linAt q.2 h = 0 := by
  unfold _root_.Flock.Firewall.Finish.consWhy at hw
  split at hw
  · cases hw
  · rename_i hkn
    split at hw
    · cases hw
    · rename_i hf
      refine ⟨by simpa using hkn, fun q hq hname => ?_⟩
      have he : _root_.Flock.Firewall.Finish.failingOn (cons.filter (·.1 == "zerocheck c_eval")) h = #[] := by
        simpa using hf
      exact _root_.Flock.Firewall.Finish.failingOn_eq_empty_iff.1 he q (Array.mem_filter.2 ⟨hq, by simp [hname]⟩)

/-- **Where `finInner` returns, its inner proof**: its `repCons`, answered by `asksPad` from round `0` and a transcript
with no rounds, returns the proof `p` and the constraints `cons` at the transcript `tF` and the request's round `n` (where
`consWhy` passes), and the inner proof on them returns from there; `finInner` returns `p`'s `final_c` and the inner
proof's `τ` commitment. -/
theorem finInner_inner {t : Tab} {rep : ℕ} {dom : ByteArray} {n : ℕ} {q : RepIn} {res : F128 × ByteArray}
    (h : finInner t rep dom n q = .ok res) :
    ∃ (ic : _root_.Flock.Firewall.Finish.InnerCommit) (p : _root_.Flock.Proof) (zl : _root_.Flock.Zk.Level0Msgs)
      (cons : Array (String × _root_.Flock.Zk.Lin)) (tF : _root_.Flock.Transcript) (ip : _root_.Flock.Zk.InnerProof)
      (tG : _root_.Flock.Transcript) (kG : ℕ),
      ((_root_.Flock.Firewall.Finish.repCons (asksPad q.coins)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) t.st t.sch t.tPad rep dom t.g t.cm0
        (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws) ic q.vals t.extra
        t.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
        (_root_.Flock.Transcript.init #[] dom)).run 0 = .ok (((p, zl, cons), tF), n) ∧
      ((_root_.Flock.Firewall.Finish.innerP (asksPad q.coins) (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (_root_.Flock.Firewall.Finish.padsOf q.draws) ic
        p.zerocheck.finalC cons).run tF).run n = .ok ((ip, tG), kG) ∧
      res = (p.zerocheck.finalC, ip.tauCommit) := by
  unfold finInner at h
  obtain ⟨ic, -, h⟩ := v_bind_ok h
  obtain ⟨⟨⟨r, tr'⟩, k'⟩, hrun, hr⟩ := v_bind_ok h
  cases hr
  obtain ⟨⟨p, zl, cons⟩, tF, kF, h1, h2⟩ := _root_.Flock.Firewall.Finish.tmw_bind_ok hrun
  try dsimp only at h2
  obtain ⟨k, t₁, k₁, h3, h4⟩ := _root_.Flock.Firewall.Finish.tmw_bind_ok h2
  have e3 : k = kF := by
    cases h3
    rfl
  have e3t : t₁ = tF := by
    cases h3
    rfl
  have e3k : k₁ = kF := by
    cases h3
    rfl
  rw [e3, e3t, e3k] at h4
  rcases hcw : _root_.Flock.Firewall.Finish.consWhy n kF cons (_root_.Flock.Firewall.Finish.padsOf q.draws).h with
    _ | why
  · obtain ⟨hkn, -⟩ := consWhy_none hcw
    try dsimp only at h4
    rw [hcw] at h4
    try dsimp only at h4
    try simp only [pure_bind] at h4
    obtain ⟨ip, tG, kG, h5, h6⟩ := _root_.Flock.Firewall.Finish.tmw_bind_ok h4
    rw [hkn] at h1 h5
    refine ⟨ic, p, zl, cons, tF, ip, tG, kG, h1, h5, ?_⟩
    cases h6
    rfl
  · exfalso
    try dsimp only at h4
    rw [hcw] at h4
    try dsimp only at h4
    cases h4

theorem eq_of_add_eq_zero' {d f : F128} (h : d + f = 0) : f = d := by
  have e := FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self d f
  rw [h] at e
  refine Eq.trans ?_ e
  apply FlockSoundness.Discharge.Composed.Zerocheck.eq_of_of
  rw [FlockLevel3.of_add, ← FlockLevel3.GF128.zero_eq, zero_add]

set_option maxHeartbeats 1000000 in
/-- **The finish accepts the request only at `vs`'s `final_c`**: at a history `InnerHist` holds at, on the worker's own
values (`innerReq_own`), where `finInner` returns at the request with `final_c = a`, its `zerocheck c_eval` constraint
holds on the pads (`consWhy`): `a` is the c_eval of the request's round-1c values at the replay's zerocheck point, the
opened coins' (`zcE`), which is `vs`'s `final_c` where no outer coin is `1` (`Loop.FinalCFixed`,
`outer_ne_one_ofLoopLink`). -/
theorem finInner_accepts_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausalOn (SwOk Masked.dataE) vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hSch : LeanRelease.LiveSchedOwn vs (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src real wk
      x w) (hC : FinalCFixed vs x w) {ν : finS.Nonce} {t : finS.Tape}
    {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)} {j : Fin (Masked.jOf x)} {r : Fin 2}
    {e : ℕ} (he : e < 4) (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r e)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀)
    (hlayT : _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.layE x j) (a : F128)
    (hok : (finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
      (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + e) a)).isOk = true) :
    a = vsWord vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r (Masked.layE x j).fc := by
  have hH' := hH
  obtain ⟨-, hlt, hl, -, hrf, -, -, -, -, -⟩ := hH'
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hs]
  subst hsts
  have hjs := Masked.lt_of_jOf j
  obtain ⟨-, -, -, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hjs).1
  have hm6 : tr.st.m - 6 = (Masked.stAt x j).m - 6 := congrArg _root_.Flock.Zk.PadLayout.nMlv hlayT
  have hm : 14 ≤ tr.st.m := by omega
  obtain ⟨hvsz, hvals, -⟩ := innerReq_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch he hH hs hr hq
  obtain ⟨-, hnc, hcoins⟩ := innerReq_coins L₁ src real x w wk he hH hs hq
  have hNS := piopNR_le_nSched hx hs j r
  have hP6 : 6 ≤ piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog := by
    unfold piopNR
    omega
  have hNR2 := loopNr_piop_at hx hs j r (jj := 2) (by omega)
  obtain ⟨Q, hQ⟩ : ∃ Q, Q = askOf q₀ (Masked.layE x j).fc (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e) a :=
    ⟨_, rfl⟩
  rw [← hQ] at hok
  obtain ⟨_, hfi⟩ := ok_of_isOk hok
  obtain ⟨ic, p, zl, cons, tF, kF, hrun, hcw⟩ := finInner_parts hfi
  obtain ⟨hkF, hzero⟩ := consWhy_none hcw
  obtain ⟨rs, rp, tq, suf, hrs, hkrs, hrcoins, hrp, hcons, hr1c, hfc⟩ := repCons_parts hrun
  rw [hlayT] at hrp hr1c hfc
  -- the replay's `zerocheck c_eval` constraint vanishes on the pads
  have hzc := (FlockSoundness.Discharge.Composed.ZerocheckMasked.replay_zerocheck tr.st (Masked.layE x j) p ic.root hm
    hrp).1
  have hsz3 := FlockSoundness.Discharge.ZkExec.replay_cons_size tr.st (Masked.layE x j) p ic.root hrp
  have hmem : rp.cons[0]! ∈ cons := by
    rw [hcons, getElem!_pos rp.cons 0 (by omega)]
    exact Array.mem_append_left suf (Array.getElem_mem _)
  have hL := hzero _ hmem (by rw [hzc])
  rw [hzc] at hL
  try dsimp only at hL
  -- there it is `Σ λ_i(z)·r1c_i + final_c` at the request's values, each masked value less its pad
  have hMk : FlockSoundness.Discharge.Composed.ZerocheckMasked.Masks (Masked.layE x j)
      (_root_.Flock.Firewall.Finish.padsOf Q.draws).h
      ⟨(Array.range 64).map fun i =>
          p.zerocheck.round1ab[i]! + (_root_.Flock.Firewall.Finish.padsOf Q.draws).h[(Masked.layE x j).r1ab i]!,
        (Array.range 64).map fun i => Q.vals[(Masked.layE x j).r1c i]!, p.zerocheck.rounds,
        p.zerocheck.finalA + (_root_.Flock.Firewall.Finish.padsOf Q.draws).h[(Masked.layE x j).fa]!,
        p.zerocheck.finalB + (_root_.Flock.Firewall.Finish.padsOf Q.draws).h[(Masked.layE x j).fb]!,
        Q.vals[(Masked.layE x j).fc]!⟩ p.zerocheck 0 :=
    ⟨fun i hi => by simp only [getElem!_map_range hi, FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self],
      fun i hi => by simp only [hr1c, getElem!_map_range hi],
      fun i hi => absurd hi (Nat.not_lt_zero i),
      (FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self _
        (_root_.Flock.Firewall.Finish.padsOf Q.draws).h[(Masked.layE x j).fa]!).symm,
      (FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self _
        (_root_.Flock.Firewall.Finish.padsOf Q.draws).h[(Masked.layE x j).fb]!).symm, hfc⟩
  rw [_root_.Flock.Firewall.Finish.linAt_eq,
    FlockSoundness.Discharge.Composed.ZerocheckMasked.c_eval_eval _ _ _ _ 0 hMk] at hL
  have hfin := eq_of_add_eq_zero' hL
  -- the request's values: `a` at `final_c`, `vs`'s at round 1c
  have hfcz : (Masked.layE x j).fc < (Masked.layE x j).zp 64 := by
    unfold _root_.Flock.Zk.PadLayout.zp _root_.Flock.Zk.PadLayout.lc
    omega
  have hfa : Q.vals[(Masked.layE x j).fc]! = a := by
    rw [hQ]
    exact askOf_vals_fc _ a (by rw [hvsz]; exact hfcz)
  have hR1 : ((Array.range 64).map fun i => Q.vals[(Masked.layE x j).r1c i]!) =
      (Array.range 64).map fun i =>
        vsWord vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r ((Masked.layE x j).r1c i) := by
    refine Array.map_congr_left fun i hi => ?_
    have hi' : i < 64 := by simpa using hi
    have hlt' : (Masked.layE x j).r1c i < (Masked.layE x j).fc := by
      unfold _root_.Flock.Zk.PadLayout.r1c _root_.Flock.Zk.PadLayout.fc _root_.Flock.Zk.PadLayout.fa
      omega
    rw [hQ]
    show (q₀.vals.set! (Masked.layE x j).fc a)[(Masked.layE x j).r1c i]! = _
    rw [Array.getElem!_set!_ne q₀.vals (Masked.layE x j).fc ((Masked.layE x j).r1c i) a (by omega)]
    exact hvals _ (by omega) (by omega)
  -- the replay's zerocheck point is the opened coins'
  have h2rs : 2 < rs.length := by omega
  obtain ⟨hsz2, hsl2⟩ := hcoins 2 (by omega)
  have hpos2 : 0 < (q₀.coins[2]!).size := by
    rw [hsz2, hNR2]
    unfold piopN _root_.Flock.Firewall.Frames.COIN_MAX
    split_ifs <;> omega
  have hzz : ∀ d, FlockSoundness.Discharge.Composed.ZerocheckMasked.zAt (_root_.Flock.Transcript.init tF.rounds d) =
      (Masked.zcE (finCoins x ops) j r).z := by
    intro d
    have htr : tF.rounds = rs.toArray := by rw [← hrs, Array.toArray_toList]
    obtain ⟨hpos, hco⟩ := hrcoins 2 h2rs
    have hQc : Q.coins = q₀.coins.extract 0 (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e) := by
      rw [hQ]
      rfl
    show (tF.rounds[2]!).coins.getD 0 0 = _
    rw [htr, getElem!_pos rs.toArray 2 (by simpa using h2rs), List.getElem_toArray, hco, getD_map_range hpos, hQc]
    exact (congrFun (padCoin_extract (C := q₀.coins) (N := nSched spec (Masked.stsOf x) (2 * j.val + r.val) + e)
      (k := 2) (by omega) (by omega)) 0).trans (hsl2 0 hpos2).symm
  have hout := LeanRelease.outer_ne_one_ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws))
    src real wk hSv (ν, t) κ m₀ ρ₀ ops hlt.le hl hrf j
  -- one name for the padded coins, so no step compares `finCoins` with its unfolding under the coin decoder
  rw [show finCoins x ops = finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2) from rfl] at hzz
  generalize finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2) = c at hout hzz hR1 ⊢
  calc a = Q.vals[(Masked.layE x j).fc]! := hfa.symm
    _ = _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv
          (FlockSoundness.Discharge.Composed.ZerocheckMasked.zAt
            (_root_.Flock.Transcript.init tF.rounds (domOf spec (2 * j.val + r.val)))))
          ((Array.range 64).map fun i => Q.vals[(Masked.layE x j).r1c i]!) := hfin
    _ = _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (Masked.zcE c j r).z)
          ((Array.range 64).map fun i => vsWord vs x w ν c t j r ((Masked.layE x j).r1c i)) := by
        rw [hzz, hR1]
    _ = vsWord vs x w ν c t j r (Masked.layE x j).fc := (hC ν c t j hout r).symm

set_option maxHeartbeats 1000000 in
/-- **The finish's `τ` commitment at a rep's inner round 2**: at a history `InnerHist` holds at with `e = 2`, where the
rows hold (`Loop.RowsOK`), where `finInner` returns at the request, its `τ` commitment is `vs`'s (`Loop.vsTau`): it
accepts only at `vs`'s `final_c` (`finInner_accepts_own`), and its inner proof commits to `InZ.tau` over its constraints
at the coins of the inner proof's rounds 0 and 1, the stream's, which where the request's values are `vs`'s and its pads
the tape's are the switched view's first rows (`Finish.repCons_cut`, `Masked.repConsC_rows`, `innerP_pad_tau`,
`Masked.swTau_of_rows`). -/
theorem finInner_tau_own (vs : finS.SwVals) (hreal : real = realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Src (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src x w)
    (hcaus : finS.SwCausalOn (SwOk Masked.dataE) vs) (wk : (relIn L₁ src real).toSrv.Worker)
    (hSch : LeanRelease.LiveSchedOwn vs (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws)) src real wk
      x w) (hC : FinalCFixed vs x w) (hrows : RowsOK vs x w)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (relIn L₁ src real).toSrv.Chk} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)}
    {j : Fin (Masked.jOf x)} {r : Fin 2} (hH : InnerHist (relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r 2)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {tr : Tab} {hi : Array ByteArray} {q₀ : RepIn}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hr : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi))
    (hq : repInOf x ((m₀, ρ₀) :: ops) spec sts (2 * j.val + r.val) (L₁.draws (ν, t) (2 * j.val + r.val)) hi = .ok q₀)
    {a fc' : F128} {tc : ByteArray}
    (hok : finInner tr ((2 * j.val + r.val) % 2) (domOf spec (2 * j.val + r.val)) (nSched spec sts (2 * j.val + r.val))
      (askOf q₀ (Masked.layE x j).fc (nSched spec sts (2 * j.val + r.val) + 2) a) = .ok (fc', tc)) :
    tc = vsTau vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r := by
  have hH' := hH
  obtain ⟨h0, hlt, hl, -, hrf, -, ⟨v, hv⟩, hI, -, hslot⟩ := hH'
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hs]
  subst hsts
  have hjs := Masked.lt_of_jOf j
  have hr2 := r.isLt
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by omega
  have hmod : (2 * j.val + r.val) % 2 = r.val := by omega
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hkk : 2 * j.val + r.val < spec.repStreams.size := by omega
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hjs).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hjs).2.1
  -- the table is table `j`'s, at its `--zk` schedule, with the tape's packed table
  have hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)) := by
    subst hreal
    exact realTabs_of src x w (ν, t) hs
  obtain ⟨g₀, d₀, l₀, -, htr⟩ := hRT _ tr hi hr
  obtain ⟨hts, hsch⟩ := Count.tabOfLevel0_sched htr
  have hst' : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hjs
  have hz : _root_.Flock.Zk.schedule (Masked.stAt x j).m = .ok (tr.sch, tr.tPad) := by
    have e : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m = .ok (tr.sch, tr.tPad) := hsch
    rwa [hk2, hst'] at e
  obtain ⟨hn6, hq6⟩ := FlockComplete.JsonRT.zkSchedule_small hm25 hm35 hz
  have hn6' : tr.sch.n + 1 ≤ 65536 := hn6
  have hq6' : ∀ i, (tr.sch.levels.getD i default).q + 1 ≤ 65536 := hq6
  have hz' := hz
  rw [(Masked.zkTab x j).hz] at hz'
  have htsch : tr.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (Except.ok.inj hz')).1.symm
  have htst : tr.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := by
    rw [hts, hk2, hst']
  have hLayT : _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.layE x j := by
    rw [htst, htsch]
    rfl
  have hms : tr.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [htst]
    exact (htabs j.val hjs).2.2
  have hm13 : tr.st.m - 13 ≤ 65536 := by
    rw [htst]
    show (Masked.stAt x j).m - 13 ≤ 65536
    omega
  have hR1 : realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tr, hi) := by
    rw [← hreal]
    exact hr
  obtain ⟨-, -, -, hg⟩ := hSrc.1.realOf ν t j r hR1
  -- the request at `a`: `a` is `vs`'s `final_c`, its other values `vs`'s, its pads the tape's, its coins the opened ones
  have ha := finInner_accepts_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch hC (by decide) hH hs hr hq
    hLayT a (by rw [hok]; rfl)
  obtain ⟨hvsz, hvals, hP⟩ :=
    innerReq_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch (by decide) hH hs hr hq
  obtain ⟨-, hnc, hcoins⟩ := innerReq_coins L₁ src real x w wk (by decide) hH hs hq
  obtain ⟨-, hbase⟩ := repSlot_nSched hx hs hkk
  have hslot' : ops.length =
      Masked.repBase x (2 * j.val + r.val) + nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2 := by
    rw [hslot, Masked.repSlot_innerOff hs j r 2, ← hbase]
  obtain ⟨Q, hQ⟩ : ∃ Q, Q = askOf q₀ (Masked.layE x j).fc (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2) a :=
    ⟨_, rfl⟩
  rw [← hQ] at hok
  obtain ⟨ic, p, zl, cons, tF, ip, tG, kG, hrun, hin, hres⟩ := finInner_inner hok
  have htc : tc = ip.tauCommit := (Prod.mk.inj hres).2
  have hQc : Q.coins = q₀.coins.extract 0 (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2) := by
    rw [hQ]; rfl
  have hQsz : Q.coins.size = nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2 := by
    rw [hQc, Array.size_extract]
    omega
  have hQv : Q.vals = q₀.vals.set! (Masked.layE x j).fc a := by
    rw [hQ]; rfl
  have hQx : Q.xs = q₀.xs := by
    rw [hQ]; rfl
  have hPd : (_root_.Flock.Firewall.Finish.padsOf Q.draws).toC =
      FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j) := by
    rw [hQ, askOf_draws]
    exact hP
  have hPQ : (_root_.Flock.Firewall.Finish.padsOf Q.draws).toC = Masked.innerPadsE x t j r :=
    hPd.trans (congrArg (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val) (Masked.innerLayE_layE x j)).symm
  -- as many extra claims as the statement's regions at each link point, below `2^64`
  have hxb : (_root_.Flock.Firewall.Finish.claimsOf Q.xs).size + 2 < 2 ^ 64 := by
    have hreg := (sizesOk_at (sizesOk_of_serves hSv s0 spec _ hs) hkk).2.2.1
    have hq' := hq
    unfold repInOf at hq'
    obtain ⟨⟨lay, ms⟩, -, hq'⟩ := v_bind_ok hq'
    have hqx : q₀.xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
          (wordsAt x (ops.map fun (p : Msg × Msg) => CoinLayer.rd p.2) 0)) := by
      cases hq'
      rfl
    have hex : ∀ (st : Stmt) (pts : Array (Array F128)),
        (_root_.Flock.Firewall.Drive.extras st pts).size = st.regions.size * pts.size := by
      intro st pts
      unfold _root_.Flock.Firewall.Drive.extras
      exact flatMap_size_const _ _ _ fun _ => Array.size_map
    have hlp : ∀ (st : Stmt) (cs : Array F128), (_root_.Flock.Firewall.Drive.linkPoints st cs).size ≤ cs.size := by
      intro st cs
      unfold _root_.Flock.Firewall.Drive.linkPoints
      simp only [Array.size_map, Array.size_range]
      exact Nat.div_le_self _ _
    have hxle : q₀.xs.size ≤
        (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.regions.size * _root_.Flock.Firewall.Frames.COIN_MAX := by
      rw [hqx, hex]
      exact Nat.mul_le_mul (le_refl _) (le_trans (hlp _ _) (le_trans (wordsAt_size_le _ _ _) (le_of_eq rfl)))
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
    rw [hQx]
    exact lt_of_le_of_lt (Nat.add_le_add_right hxle 2) hreg
  -- the request's coins below the history's length are the opened coins from the rep's base
  have hpc : ∀ k < nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2,
      padCoin Q.coins k = fun i => (q₀.coins[k]!)[i]! := by
    intro k hk
    rw [hQc]
    exact padCoin_extract hk (by omega)
  have hQk : ∀ k < nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2, Q.coins[k]! = q₀.coins[k]! := by
    intro k hk
    rw [hQc]
    exact extract_get! hk (by omega)
  have hQget : ∀ k < nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 2, Q.coins[k]? = some q₀.coins[k]! := by
    intro k hk
    rw [getElem?_pos Q.coins k (by omega), ← getElem!_pos Q.coins k (by omega), hQk k hk]
  have hcQ : ∀ k < Q.coins.size, ∀ i < (Q.coins[k]!).size,
      Masked.ligCoin (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2))
        (Masked.repBase x (2 * j.val + r.val) + k) i = (Q.coins[k]!)[i]! := by
    intro k hk i hi
    rw [hQsz] at hk
    rw [hQk k hk] at hi ⊢
    exact (hcoins k hk).2 i hi
  -- the rep's run, as the session's rounds from the rep's base cut at the history's length
  have hrunOf := repCons_ofPad Q.coins tr.st hms tr.sch tr.tPad _ _ tr.g tr.cm0 _ _ ic Q.vals tr.extra tr.lanePad _ _
    hrun (by omega)
  have hcut := _root_.Flock.Firewall.Finish.repCons_cut Q.coins (Masked.repBase x (2 * j.val + r.val)) ops.length
    (Masked.ligCoin (finS.pad (K := refK x) (ops.map fun (p : Msg × Msg) => finS.decOf p.2))) (by omega) hcQ tr.st hms
    tr.sch tr.tPad _ _ tr.g tr.cm0 _ _ ic Q.vals tr.extra tr.lanePad _ hm13 hxb hn6' hq6' _ hrunOf
    (Array.replicate (Masked.repBase x (2 * j.val + r.val)) default) (by simp)
  have hgo := go_of_fst hcut.1
  rw [hmod, htsch] at hgo
  -- its constraints are the worker's view's first rows, where the rows hold (no outer coin `1`)
  have hnr := LeanRelease.innerAt_unrefused L₁ src real (finOf real L₁.draws) x w hx hSv
    (fun ρ s0' spec' sts' h => by rw [hreal]; exact realTabs_of src x w ρ h) wk ν t κ m₀ ρ₀ ops v h0 hlt hl hrf hv hI
  have hout := LeanRelease.outer_ne_one_ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real (finOf real L₁.draws))
    src real wk hSv (ν, t) κ m₀ ρ₀ ops hlt.le hl hrf j
  have hrowsAt := hrows ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j hout
  have hsw : Masked.SwRows x (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) w
      (vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) (finS.seenOf x t)) t j r cons := by
    refine Masked.repConsC_rows x hx w hR t (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) ops.length hnr
      s0 spec hs j r (vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) (finS.seenOf x t)) hrowsAt _
      (by
        simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
        rw [hQx]
        exact innerReq_xs hx hSv h0 hs j r hq) tr.st htst tr.tPad _ tr.g hg _ _ _ Q.vals
      tr.extra tr.lanePad _ (by rfl) (by simp) hgo ?_
    intro k hk
    have hk' : k < (Masked.layE x j).zp 64 := hk
    have hiE : k < (Masked.tabC Masked.PE x j).z.nE := by
      rw [Masked.nE_eq_hab]
      exact lt_trans hk' (Masked.zp64_lt_hab x j)
    rw [swSrcE_masked_vsWord vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) t j r hiE, hPd]
    congr 1
    by_cases hkf : k = (Masked.layE x j).fc
    · subst hkf
      rw [hQ, askOf_vals_fc (q₀ := q₀) (fc := (Masked.layE x j).fc) _ a (by rw [hvsz]; exact hk')]
      exact ha.symm
    · rw [hQv, Array.getElem!_set!_ne q₀.vals (Masked.layE x j).fc k a (Ne.symm hkf)]
      exact (hvals k hk' hkf).symm
  have h7 : cons.size = 7 := hsw.1
  -- the inner proof's `τ` at the coins of its rounds 0 and 1, the stream's
  obtain ⟨hs1, hs2, hipt⟩ := innerP_pad_tau Q.coins _ _ ic _ cons (by rw [h7]; norm_num) hin
  have hsz0 : (q₀.coins[nSched spec (Masked.stsOf x) (2 * j.val + r.val)]!).size = 1 :=
    hs1 _ (hQget (nSched spec (Masked.stsOf x) (2 * j.val + r.val)) (by omega))
  have hsz1 : (q₀.coins[nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 1]!).size = cons.size + 3 :=
    hs2 _ (hQget (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 1) (by omega))
  have he : (FlockComplete.SimInner.cutCoins (padCoin Q.coins) (nSched spec (Masked.stsOf x) (2 * j.val + r.val))).ce 0 =
      (Masked.innerCoinsE (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) j r).ce 0 := by
    show padCoin Q.coins (nSched spec (Masked.stsOf x) (2 * j.val + r.val)) 0 =
      Masked.slotAt (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2))
        (Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 0)) 0
    rw [hpc (nSched spec (Masked.stsOf x) (2 * j.val + r.val)) (by omega), ← hbase]
    exact ((hcoins (nSched spec (Masked.stsOf x) (2 * j.val + r.val)) (by omega)).2 0
      (by rw [hsz0]; norm_num)).symm
  have hgc : ∀ i < cons.size + 3,
      (FlockComplete.SimInner.cutCoins (padCoin Q.coins) (nSched spec (Masked.stsOf x) (2 * j.val + r.val))).cg i =
        (Masked.innerCoinsE (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) j r).cg i := by
    intro i hi
    show padCoin Q.coins (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 1) i =
      Masked.slotAt (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2))
        (Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 0) + 1) i
    rw [hpc (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 1) (by omega), ← hbase]
    have e := (hcoins (nSched spec (Masked.stsOf x) (2 * j.val + r.val) + 1) (by omega)).2 i
      (by rw [hsz1]; exact hi)
    rw [← Nat.add_assoc] at e
    exact e.symm
  -- at the table's layout and the tape's pads, that `τ` is the worker's view's, `vs`'s commitment
  have hLI : _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 = Masked.innerLayE x j :=
    hLayT.trans (Masked.innerLayE_layE x j).symm
  have hsalt : (_root_.Flock.Firewall.Finish.padsOf Q.draws).tauSalt = (Masked.innerPadsE x t j r).tauSalt :=
    congrArg FlockComplete.PadsZk.tauSalt hPQ
  rw [htc, hipt, hLI, hPQ, hsalt, tau_congr (Masked.innerLayE x j) (Masked.innerPadsE x t j r) cons he hgc,
    Masked.swTau_of_rows hs (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) w
      (vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) (finS.seenOf x t)) t j r cons hsw]
  exact (Masked.swSrcE_tauCom x (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) w
    (vs x w ν (finS.pad (ops.map fun (p : Msg × Msg) => finS.decOf p.2)) (finS.seenOf x t)) t j r hrowsAt).symm

end Loop

end Claims.ZKServer
