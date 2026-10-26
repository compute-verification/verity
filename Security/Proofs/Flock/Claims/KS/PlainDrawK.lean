import Proofs.Flock.Claims.KS.PlainDrawKDefs
import Proofs.Flock.Claims.Recursion.FlockKSDeployK
import Proofs.Flock.Claims.Recursion.CallsBridgeE
import Proofs.Flock.Soundness.Discharge.Exec.VerifyK

/-!
# The plain verdict at the statement's draw, at `VK keep`

`PlainDraw`'s verdict at every `keep`, the facts about `PlainDrawKDefs`' definitions: flock-verify's acceptance is
`verifyWithMK keep` at SHA-512 (`With.std`), the record decodes through `Record.decodeWithK keep`, and a recorded round
is the live outcome's up to its digest (`Round.erase`), since at `keep := fun _ => true` the decoder drops S12's
comparison of a retained round's bytes with its recorded digest (`decodeWithK_live`). Every reader of the rounds here
reads their coin counts, coins and bytes only (`plainRepColsOf` reads the coins), so nothing below reads a round's
digest.

* `AcceptsAtDK`, `RecordsLiveAtDK`, `TxNoOffAtK`, `wrPDK` (`PlainDrawKDefs`): at `keep := id` acceptance and the
  reader's premise are `PlainDraw`'s
  (`acceptsAtD_iff_acceptsAtDK_id`, `txNoOffAt_iff_txNoOffAtK_id`), and a record of the outcome there is one here
  (`recordsLiveAtDK_id_of`); the element-level acceptance holds wherever the shipped one does
  (`acceptsAtDK_true_of_acceptsAtD`).
* `recordsLiveAtDK_of_checks`, `acceptsAtDK_of_decide`: an accepting decision of `VK keep` at SHA-512 is accepted at the
  setup at its statement's draw and records its live outcome there up to the rounds' digests.
* `cols_plainK`: `cols_plain` with the verdict read through `keep`.
* `plainVerdictAtK_of_run`: `plainVerdictAtK` (`plainVerdictAt` at `wrPDK`), which every accepting run of `VK keep`
  with no column off `C` has at its live outcome, on every draw-carrying record. At `keep := id` the run is one of
  `V false` (`VK_id`), and at `keep := fun _ => true` one of `VE` (`VK_true`).
-/

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open FlockSoundness.Discharge.Exec (decodeWithK_id_std)
open FlockSoundness.Refine (ptsOf shapeOf ebind_ok)
open FlockDeploy.KS (decodeWithK_live)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

theorem named_of_setupAtD {I : Exec.Inputs} {dr : Option Json} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt}
    (h : setupAtD I dr = .ok (spec, sts)) : ∀ s ∈ sts.map _root_.Flock.Setup.ofCircuit, s.merkle.Named :=
  With.named_ofCircuit (With.setupTables_named _ _ _ _ _ _ _ _ _ _ _ _ _ _ h)

/-- **`AcceptsAtDK id` is `AcceptsAtD`**: on every session the setup at a draw sets up, `verify` is
`verifyWith Sha512.hash`. -/
theorem acceptsAtD_iff_acceptsAtDK_id (I : Exec.Inputs) (dr : Option Json) (t : Exec.Transcript) :
    AcceptsAtD I dr t ↔ AcceptsAtDK id I dr t := by
  unfold AcceptsAtD AcceptsAtDK
  refine exists_congr fun spec => exists_congr fun sts => and_congr_right fun hs => ?_
  rw [verify_eq_verifyWith _ _ _ _ (named_of_setupAtD hs)]
  exact Iff.rfl

/-- **The element-level acceptance at a draw holds wherever `AcceptsAtD` does** (`verifyWithE_of_verifyWith`). -/
theorem acceptsAtDK_true_of_acceptsAtD {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript}
    (h : AcceptsAtD I dr t) : AcceptsAtDK (fun _ => true) I dr t := by
  obtain ⟨spec, sts, hs, hv⟩ := (acceptsAtD_iff_acceptsAtDK_id I dr t).1 h
  exact ⟨spec, sts, hs, verifyWithE_of_verifyWith _ _ _ _ _ hv⟩

/-- **A record of the outcome at `d` is one at `keep := id`**: `RecordsLiveAtD`'s rounds are the outcome's, so they
are up to their digests. -/
theorem recordsLiveAtDK_id_of {I : Exec.Inputs} {d : Json} {J mPts : ℕ} {o : Tables.LiveOutJ J mPts}
    {t : Exec.Transcript} (h : RecordsLiveAtD I d o t) : RecordsLiveAtDK id I d o t := by
  intro hA
  obtain ⟨hpr, hall⟩ := h ((acceptsAtD_iff_acceptsAtDK_id I (some d) t).2 hA)
  refine ⟨hpr, fun spec sts sess hset hdec => ?_⟩
  rw [decodeWithK_id_std] at hdec
  obtain ⟨hpts, hj⟩ := hall spec sts sess hset hdec
  refine ⟨hpts, fun j => ⟨(hj j).1, fun r => ?_⟩⟩
  obtain ⟨s, hs, hr⟩ := (hj j).2 r
  exact ⟨s, hs, by rw [hr]⟩

theorem txNoOffAt_iff_txNoOffAtK_id (I : Exec.Inputs) (d : Json) {N : ℕ} (C : Fin N → List 𝔽) (t : Exec.Transcript) :
    TxNoOffAt I d C t ↔ TxNoOffAtK id I d C t := by
  unfold TxNoOffAt TxNoOffAtK
  simp only [decodeWithK_id_std]

/-- **The verdict refuses a session without proof files, at every `keep`** (`verify_nil_plain`'s proof). -/
theorem verifyWithMK_nil (keep : Bool → Bool) (hs : Hashes _root_.Flock.V) (spec : SessionSpec) (tabs : Array Setup)
    (record : String) : verifyWithMK keep hs spec tabs record #[] ≠ .ok () := by
  intro h
  unfold verifyWithMK at h
  simp only [] at h
  by_cases c1 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c1] at h
    obtain ⟨_, h', _⟩ := ebind_ok h
    cases h'
  rw [ite_eq_right c1] at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  have c3 : ((#[] : Array ByteArray).size != spec.repStreams.size) = true := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, Decidable.not_not] at c1
    simp only [List.size_toArray, List.length_nil, Custody.repStreams_size, bne_iff_ne, ne_eq]
    omega
  rw [ite_eq_left c3] at h
  obtain ⟨_, h', _⟩ := ebind_ok h
  cases h'

theorem not_acceptsAtDK_nil (keep : Bool → Bool) (I : Exec.Inputs) (dr : Option Json) (record : String) :
    ¬ AcceptsAtDK keep I dr ⟨record, #[]⟩ := by
  rintro ⟨spec, sts, -, h⟩
  exact verifyWithMK_nil keep _ _ _ _ h

/-- **`wrPDK` records the live session at the setup at `d`, up to the rounds' digests.** -/
theorem recordsLiveAtDK_wrPDK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {J mPts : ℕ}
    (o : Tables.LiveOutJ J mPts) : RecordsLiveAtDK keep I d o (wrPDK keep I d o) := by
  unfold wrPDK
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h (not_acceptsAtDK_nil _ _ _ _)

/-- **`verifyWithMK keep` accepts what `wrPDK` writes** wherever it accepts some transcript at the setup at `d` that
records the outcome there. -/
theorem acceptsAtDK_wrPDK {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {J mPts : ℕ} {o : Tables.LiveOutJ J mPts}
    {t : Exec.Transcript} (hA : AcceptsAtDK keep I (some d) t) (hR : RecordsLiveAtDK keep I d o t) :
    AcceptsAtDK keep I (some d) (wrPDK keep I d o) := by
  have hex : ∃ t, AcceptsAtDK keep I (some d) t ∧ RecordsLiveAtDK keep I d o t := ⟨t, hA, hR⟩
  unfold wrPDK
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  exact (Classical.choose_spec hex).1

theorem forall₂_map_congr {α β γ : Type} {R : α → β → Prop} {f : α → γ} {g : β → γ} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → (∀ a b, b ∈ l' → R a b → f a = g b) → l.map f = l'.map g
  | _, _, .nil, _ => rfl
  | _, _, .cons hab hl, H => by
    rw [List.map_cons, List.map_cons, H _ _ (by simp) hab,
      forall₂_map_congr hl fun a b hb h => H a b (List.mem_cons_of_mem _ hb) h]

/-- **Custody gives `RecordsLiveAtDK`**: where the custody checks pass on a plain statement that sets up at its draw
`d` with no `--public-inputs`, the last message's record and proof files are a record of the run's live outcome
(`viewOf`) at the setup at `d` through every `keep`: the decoder's rounds are `liveOf`'s in coin count, coins and bytes
(`decodeWithK_live`), and custody retains every live round's bytes, so they agree after `Round.erase`. -/
theorem recordsLiveAtDK_of_checks (keep : Bool → Bool) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {x : Claims.Msg} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} {J mPts : ℕ} {d : Json} (hs : s.setup = .ok (spec, sts)) (hzk : s.read.zk = false)
    (hpi : s.read.publicInputs = none) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size)
    (hc : checks false s spec x link c₁ mids f = .ok ()) :
    RecordsLiveAtDK keep s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  intro _
  refine ⟨rfl, fun spec' sts' sess hset hdec => ?_⟩
  rw [setupAtD_plainInputs (Ins.setup_read hs) hzk hpi hd] at hset
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hset)
  obtain ⟨-, lv, hlv, hcus⟩ := checks_ok hc
  obtain ⟨hroot, -, hpts, hrounds⟩ := custody_ok hcus
  obtain ⟨eroot, epts, esize, estr⟩ := decodeWithK_live keep With.std hdec hlv
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  refine ⟨by simp only [viewOf]; rw [epts, hpts], fun j => ⟨?_, fun r => ?_⟩⟩
  · show sess.rootB[j.val]! = link.1.toArray[j.val]!
    rw [eroot, ← hroot]
  · obtain ⟨hq, hw⟩ := repStreams_getElem spec (i := j.val) (r := r.val) (by omega) r.isLt
    obtain ⟨y, hy, hyr⟩ := estr _ hq
    rw [hw] at hy
    refine ⟨y, hy, ?_⟩
    show y.rounds.toList.map Round.erase =
      (((liveD spec f.record).streams[2 * j.val + r.val]!).toList.map (withSha spec)).map Round.erase
    rw [hD, List.map_map]
    have hq' : 2 * j.val + r.val < lv.streams.size := esize ▸ hq
    refine forall₂_map_congr hyr fun rd rd' hmem ⟨hn, hcn, hb⟩ => ?_
    obtain ⟨am, -, hbytes, -⟩ := forall₂_mem_left hrounds rd' (List.mem_flatMap.2
      ⟨lv.streams[2 * j.val + r.val]!, by rw [getElem!_pos lv.streams _ hq']; exact Array.getElem_mem_toList hq',
        hmem⟩)
    rcases rd with ⟨n, sha, coins, bytes⟩
    rcases rd' with ⟨n', sha', coins', bytes'⟩
    simp only at hn hcn hb hbytes
    subst hn hcn hb hbytes
    rfl

/-- **An accepting decision of `VK keep` at SHA-512**: the statement sets up at `(spec, sts)`, plain and without
`--public-inputs`, `verifyWithMK keep` at SHA-512 accepts the record and proof files there, and the custody checks
pass (`nextK_decide` at the setup `act` read). -/
theorem decideK_sha512 (keep : Bool → Bool) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (ha : act false x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextK keep x as cs).eval sha512).1 = .acc) :
    verifyWithMK keep With.std spec (tabsOf sts) f.record f.proofs = .ok () ∧
      checks false s spec x link c₁ mids f = .ok () ∧ s.setup = .ok (spec, sts) ∧ s.read.zk = false ∧
      s.read.publicInputs = none := by
  obtain ⟨spec', sts', hsu, hv, hc⟩ := (nextK_decide keep sha512 ha).1 hacc
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj ((act_decide_setupWith ha).symm.trans hsu))
  rw [hOf_sha512] at hv
  obtain ⟨hins, hs⟩ := setupOf_some (act_decide ha)
  exact ⟨hv, hc, hs, readOf_zk (x := x) (by simp [readOf, hins]), publicInputs_none_of_checks hc⟩

/-- **An accepting decision of `VK keep` at SHA-512 is accepted through `keep` at the setup at its statement's draw**,
on a record of its live outcome there up to the rounds' digests, whatever draw the record carries
(`acceptsAtD_of_decide` at every `keep`). -/
theorem acceptsAtDK_of_decide (keep : Bool → Bool) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} {J mPts : ℕ} {d : Json}
    (ha : act false x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextK keep x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) :
    AcceptsAtDK keep s.read.I (some d) ⟨f.record, f.proofs⟩ ∧
      RecordsLiveAtDK keep s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  obtain ⟨hv, hc, hs, hzk, hpi⟩ := decideK_sha512 keep ha hacc
  exact ⟨⟨spec, sts, setupAtD_plainInputs (Ins.setup_read hs) hzk hpi hd, hv⟩,
    recordsLiveAtDK_of_checks keep hs hzk hpi hd hJ hc⟩

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **`cols_plain` with the verdict read through `keep`** (`rep_plainK`, `queryRound_plainK`). -/
theorem cols_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {lv : Live} {st : Stmt} {sch : Schedule} (hf : lastFinal seen = some f)
    (hset : setupOf false f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : verifyWithMK keep hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hst : sts[0]? = some st)
    (hsch : fast100 st.m = some sch) (hp2 : spec.points = 2) :
    cols false seen 0 = Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 0 #[])
          (f.proofs.getD 0 ByteArray.empty) ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 1 #[])
          (f.proofs.getD 1 ByteArray.empty)) := by
  obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hst
  have h0' : sts[0]! = sts[0] := getElem!_pos sts 0 h0
  obtain ⟨hsz, hall⟩ := rep_plainK keep hs hv hlv
  have hk : 0 < lv.streams.size := by omega
  obtain ⟨sess, -, -, -, hpts, hpn, -⟩ := hall 0 hk
  obtain ⟨extras, -, -, hex, -⟩ := queryRound_plainK keep hs hv hlv hk (tab := Setup.ofCircuit sts[0])
    (by rw [Nat.zero_div, h0'])
  have hx : extras.size = 2 * sts[0].regions.size := by
    rw [extraClaims_size (st := sts[0]) hex, ← hpts, hpn, hp2, Nat.mul_comm]
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit sts[0]).m = sch := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit sts[0]).m = sts[0].m from rfl, hsch]
    rfl
  have hrun : runOf false seen 0 = some
      { m := (Setup.ofCircuit sts[0]).m, kLog := (Setup.ofCircuit sts[0]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[0]).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit sts[0]),
        reps := [(lv.streams.getD 0 #[], f.proofs.getD 0 ByteArray.empty),
                 (lv.streams.getD 1 #[], f.proofs.getD 1 ByteArray.empty)] } := by
    unfold runOf
    simp [hf, hset, hst, hlv, hex, Except.toOption]
  unfold cols
  rw [hrun]
  simp only [Bool.false_eq_true, ↓reduceIte, Claims.Flock.plainRunCols, Claims.Flock.ZkRun.plainShape,
    Claims.Flock.plainLevel0, hsched, FlockDeploy.plainProofShape_eq hsched, hx, List.flatMap_cons, List.flatMap_nil,
    List.append_nil]

theorem coins_getElem?_of_erase {xs ys : List Round} (h : xs.map Round.erase = ys.map Round.erase) (j : ℕ) :
    xs[j]?.map (fun rd : Round => rd.coins) = ys[j]?.map (fun rd : Round => rd.coins) := by
  have e := congrArg (fun l : List Round => l[j]?.map (fun rd : Round => rd.coins)) h
  simpa only [List.getElem?_map, Option.map_map, Function.comp_def, Round.erase_coins] using e

/-- **The deployed event gives the verdict at the draw, at every `keep`**: an accepting run of `VK keep` at SHA-512
decides on its messages and coins, and at its statement's draw `d`, if the run's reader looks up no column off `C` at
table 0, `plainVerdictAtK keep` holds at the run's live outcome (`viewOf`, at any `mPts`) and its link's root, whatever
draw its record carries (`plainVerdictAt_of_run` at `nextK keep`): `verifyWithMK keep` accepts the last message at the
setup at `d` (`acceptsAtDK_of_decide`), so it accepts what `wrPDK` writes, whose rep streams' coins are the run's
(`coins_getElem?_of_erase`), and those are the reader's columns (`cols_plainK`). -/
theorem plainVerdictAtK_of_run (keep : Bool → Bool) (mPts : ℕ) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {ω : Ω} {r : Fin ((VK keep).k x) → Coin} {N : ℕ} (C : Fin N → List 𝔽)
    (h : (Claims.run sha512 P (VK keep) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act false x ((Claims.run sha512 P (VK keep) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VK keep) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      ∀ d, s.read.draw = some d →
        (¬ ∃ (p : Fin N) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p) →
        plainVerdictAtK keep s.read.I d C link.1.toArray (viewOf 1 mPts spec link c₁ f) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := accK_decide keep h
  refine ⟨s, spec, sts, link, c₁, mids, f, hact, fun d hd hno => ?_⟩
  have hacc := run_acc_next sha512 P (VK keep) x ω r h
  replace hacc : ((nextK keep x ((Claims.run sha512 P (VK keep) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (VK keep) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := hacc
  obtain ⟨hv, hc, hsu, hzk, hpi⟩ := decideK_sha512 keep hact hacc
  have hsetA : setupAtD s.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu) hzk hpi hd
  obtain ⟨-, -, -, htsz, hpos, -⟩ := verifyWithMK_tables_cap keep With.std hv
  have hJ : 1 ≤ spec.tables.size := by rw [← htsz]; exact hpos
  obtain ⟨hA₀, hR₀⟩ := acceptsAtDK_of_decide keep (J := 1) (mPts := mPts) hact hacc hd hJ
  have hA := acceptsAtDK_wrPDK hA₀ hR₀
  refine ⟨hA, rfl, ?_⟩
  intro spec' st sess sch s₀ s₁ hset' hdec' hsch hs₀ hs₁
  obtain ⟨hpr, hall⟩ := recordsLiveAtDK_wrPDK keep s.read.I d (viewOf 1 mPts spec link c₁ f) hA
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hset'))
  obtain ⟨-, hrs⟩ := hall spec #[st] sess hset' hdec'
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨1, by omega⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain rfl : y₁ = s₁ := Option.some.inj (hy₁.symm.trans hs₁)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (VK keep) x ω r).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decisionK_runs keep hsu hrun
  have hv' : verifyWithMK keep (hashesIn ((Claims.run sha512 P (VK keep) x ω r).vLog.map Prod.fst)) spec
      (tabsOf #[st]) f.record f.proofs = .ok () := verdictK_within keep hvr
  have hcols := cols_plainK keep (st := st) _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hsch
    (HiddenExec.setupTables_pointsAny hsetA).1
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (k : ℕ) (y : StreamRec),
      y.rounds.toList.map Round.erase =
        (((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec)).map Round.erase →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro k y hy j
    rw [← Array.getElem?_toList, coins_getElem?_of_erase hy j, hD, show (#[] : Array Round) = default from rfl,
      getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  rw [hpr]
  show ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow _
    (Claims.Flock.plainRepColsOf _ y₀.rounds f.proofs[0]! ++ Claims.Flock.plainRepColsOf _ y₁.rounds f.proofs[1]!))
      p.val = some u ∧ u ≠ C p
  rw [plainRepColsOf_congr _ _ (hco 0 y₀ hr₀), plainRepColsOf_congr _ _ (hco 1 y₁ hr₁), ← getD_empty,
    ← getD_empty, ← hcols]
  exact hno

end FlockDeploy.KS
