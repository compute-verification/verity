import Proofs.Flock.Claims.KS.PlainClash0K
import Proofs.Flock.Claims.KS.PlainClashLaterK
import Proofs.Flock.Claims.KS.PlainPartsK

/-!
# The clashes' run-level half, at `VK keep`

`PlainClash0K` and `PlainClashLaterK` restate the clash files' draw half at `wrPDK keep`; this is their run-level half,
which reads `P`'s run at the verifier: an accepted run of `VK keep` has its reps' raw columns (`rawCols_plainK`,
`rawLevelCols_plainK`, at `verifyWithMK keep`), so a self-clash at the concrete simulator's outcome is a raw clash of
the run (`plainClashQ₀K`, `plainClashQLK`), and the live self-clash's clash holds at every statement
(`plainSimLaterClashAtK`, `plainSimClashAt_of_laterK`). Each is its original's proof at `plainSimK keep`, with the
rep streams' rounds read up to their digests (`coins_getElem?_of_erase`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- `rawCols_plain` at `VK keep`. -/
theorem rawCols_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {st : Stmt} {sch : Flock.Schedule} (hf : lastFinal seen = some f)
    (hset : setupOf false f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : verifyWithMK keep hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hst : sts[0]? = some st)
    (hsch : Flock.fast100 st.m = some sch) (hp2 : spec.points = 2) :
    plainRawCols seen =
      Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 0 #[])
          (f.proofs.getD 0 ByteArray.empty) ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 1 #[])
          (f.proofs.getD 1 ByteArray.empty) := by
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
  unfold plainRawCols
  rw [hrun]
  simp only [Claims.Flock.ZkRun.plainShape, FlockDeploy.plainProofShape_eq hsched, hx, List.flatMap_cons,
    List.flatMap_nil, List.append_nil]

/-- `plainClashQ₀` at `VK keep`. -/
theorem plainClashQ₀K (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        Clash₀L sch (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        EntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQK keep P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (Clash₀L sch) := by
  intro C u r' hq hacc
  have hw := (plain_mPts hs hin hset).1
  have hm1 := (plain_mPts hs hin hset).2.1
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨a, ⟨hV, pend, hpl, rfl⟩, hcl⟩ := hq
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hcl
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := accK_decide keep hacc
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := plainSimRunK keep P ω hset h hw hm1 hp2 u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hV hpl hcl
  obtain ⟨spec'', st', sess, sch', s₀, s₁, hsu, hdec, hf', hs₀, hs₁, q, a₁, b₁, ha, hb, hab⟩ :=
    hE _ hV.1 (pend 0) (hpl 0) hcl
  have hacc' := run_acc_next sha512 P (VK keep) x ω _ hacc
  replace hacc' : ((nextK keep x ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length).map
        (Claims.coinSeq (plainΦ x spec st (u, r'))))).eval sha512).1 = .acc := hacc'
  obtain ⟨hv, hc, hsu₀, hzk, hpi⟩ := decideK_sha512 keep hact hacc'
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
  subst spec'' e₅
  obtain ⟨hpr, hall⟩ := recordsLiveAtDK_wrPDK keep ins.read.I d (viewOf 1 spec.mPts spec link c₁ f) hV.1
  obtain ⟨-, hrs⟩ := hall spec #[st'] sess hsu hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨1, by omega⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain rfl : y₁ = s₁ := Option.some.inj (hy₁.symm.trans hs₁)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decisionK_runs keep hsu₀ hrun
  have hv' : verifyWithMK keep (hashesIn ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).vLog.map
      Prod.fst)) spec (tabsOf #[st']) f.record f.proofs = .ok () := verdictK_within keep hvr
  have hcols := rawCols_plainK keep (st := st') _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hf'
    (HiddenExec.setupTables_pointsAny hsetA).1
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (k : ℕ) (y : StreamRec),
      y.rounds.toList.map Round.erase =
        (((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec)).map Round.erase →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro k y hy j
    rw [← Array.getElem?_toList, coins_getElem?_of_erase hy j, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrPDK keep ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  have hcl' : ∀ {q : ℕ} {v : List 𝔽}, (q, v) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st') sch'
        (2 * st'.regions.size)) y₀.rounds (wrPDK keep ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs[0]! ++
      Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st') sch' (2 * st'.regions.size)) y₁.rounds
        (wrPDK keep ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs[1]! →
      (q, v) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).seen := by
    intro q v hqv
    rw [hpr', plainRepColsOf_congr _ _ (hco 0 y₀ hr₀), plainRepColsOf_congr _ _ (hco 1 y₁ hr₁), ← getD_empty,
      ← getD_empty, ← hcols] at hqv
    exact hqv
  exact Or.inl ⟨q, a₁, b₁, hcl' ha, hcl' hb, hab⟩

/-- `plainClashQ_or` at `VK keep`. -/
theorem plainClashQ_orK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec} {st : Stmt}
    {T : TabSpec spec.mPts} {hw : 2 * spec.mPts ≤ 65536} {I : Exec.Inputs} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8)}
    {Q₁ Q₂ : Model.OutCL 𝔽 Dig (List UInt8) → Prop} (Q : Model.OutCL 𝔽 Dig (List UInt8) → Prop)
    (h₁ : PlainClashQK keep P x ω st T hw I d opn Q₁) (h₂ : PlainClashQK keep P x ω st T hw I d opn Q₂)
    (hQ : ∀ out, Q out → Q₁ out ∨ Q₂ out) : PlainClashQK keep P x ω st T hw I d opn Q := by
  rintro C u r' ⟨a, ha, hq⟩ hacc
  rcases hQ _ hq with hq | hq
  · exact h₁ C u r' ⟨a, ha, hq⟩ hacc
  · exact h₂ C u r' ⟨a, ha, hq⟩ hacc

/-- `rawLevelCols_plain` at `VK keep`. -/
theorem rawLevelCols_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {st : Stmt} {sch : Flock.Schedule} (hf : lastFinal seen = some f)
    (hset : setupOf false f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : verifyWithMK keep hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hst : sts[0]? = some st)
    (hsch : Flock.fast100 st.m = some sch) (hp2 : spec.points = 2) {k : ℕ} (hk : k < 2) (ℓ : ℕ) :
    plainRawLevelCols seen k ℓ =
      plainLevelColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD k #[])
        (f.proofs.getD k ByteArray.empty) ℓ := by
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
  unfold plainRawLevelCols
  rw [hrun]
  rcases k with _ | _ | k
  · simp only [Claims.Flock.ZkRun.plainShape, FlockDeploy.plainProofShape_eq hsched, hx, List.getElem?_cons_zero]
  · simp only [Claims.Flock.ZkRun.plainShape, FlockDeploy.plainProofShape_eq hsched, hx, List.getElem?_cons_succ,
      List.getElem?_cons_zero]
  · omega

/-- `plainClashQL` at `VK keep`. -/
theorem plainClashQLK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        LaterClashL sch (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        LaterEntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQK keep P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (LaterClashL sch) := by
  intro C u r' hq hacc
  have hw := (plain_mPts hs hin hset).1
  have hm1 := (plain_mPts hs hin hset).2.1
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨a, ⟨hV, pend, hpl, rfl⟩, hcl⟩ := hq
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hcl
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := accK_decide keep hacc
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := plainSimRunK keep P ω hset h hw hm1 hp2 u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hV hpl hcl
  obtain ⟨spec'', st', sess, sch', hsu, hdec, hf', k, hk, s₀, hs₀, ℓ, q, a₁, b₁, ha, hb, hab⟩ :=
    hE _ hV.1 (pend 0) (hpl 0) hcl
  have hacc' := run_acc_next sha512 P (VK keep) x ω _ hacc
  replace hacc' : ((nextK keep x ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length).map
        (Claims.coinSeq (plainΦ x spec st (u, r'))))).eval sha512).1 = .acc := hacc'
  obtain ⟨hv, hc, hsu₀, hzk, hpi⟩ := decideK_sha512 keep hact hacc'
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
  subst spec'' e₅
  obtain ⟨hpr, hall⟩ := recordsLiveAtDK_wrPDK keep ins.read.I d (viewOf 1 spec.mPts spec link c₁ f) hV.1
  obtain ⟨-, hrs⟩ := hall spec #[st'] sess hsu hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨k, hk⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decisionK_runs keep hsu₀ hrun
  have hv' : verifyWithMK keep (hashesIn ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).vLog.map
      Prod.fst)) spec (tabsOf #[st']) f.record f.proofs = .ok () := verdictK_within keep hvr
  have hcols := rawLevelCols_plainK keep (st := st') _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hf'
    (HiddenExec.setupTables_pointsAny hsetA).1 hk ℓ
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (k : ℕ) (y : StreamRec),
      y.rounds.toList.map Round.erase =
        (((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec)).map Round.erase →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro k y hy j
    rw [← Array.getElem?_toList, coins_getElem?_of_erase hy j, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrPDK keep ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  have hcl' : ∀ {q : ℕ} {v : List 𝔽}, (q, v) ∈ plainLevelColsOf (shapeOf (Setup.ofCircuit st') sch'
        (2 * st'.regions.size)) y₀.rounds (wrPDK keep ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs[k]! ℓ →
      (q, v) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).seen k ℓ := by
    intro q v hqv
    rw [hpr', plainLevelColsOf_congr _ _ _ (hco k y₀ hr₀), ← getD_empty, ← hcols] at hqv
    exact hqv
  exact Or.inr (Or.inl (plain_levelClashK keep hacc (hcl' ha) (hcl' hb) hab))

/-- `PlainSimLaterClashAt` at `VK keep`. -/
def PlainSimLaterClashAtK (keep : Bool → Bool) (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (d : Json) (_ : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Claims.Prover Ω) (ω : Ω),
    (ins.read.I.tags.hiddenOutputs = false →
      PlainClashQK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)) ∧
    (ins.read.I.tags.hiddenOutputs = true → ins.read.I.tags.merkleLeaf = "hm96-sha512/v1" →
      PlainClashQK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))

/-- `plainSimLaterClashAt` at `VK keep`. -/
theorem plainSimLaterClashAtK (keep : Bool → Bool) (x : Claims.Msg) : PlainSimLaterClashAtK keep x := by
  intro s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω _ _ P ω
  exact ⟨fun hh => plainClashQLK keep hs hin hset h hd _ _ _
      (plain_laterClashAtK keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω,
    fun hh hl => plainClashQLK keep hs hin hset h hd _ _ _
      (plain_laterClashHAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω⟩

/-- `plainSimClashAt_of_later` at `VK keep`. -/
theorem plainSimClashAt_of_laterK (keep : Bool → Bool) {x : Claims.Msg} (hl : PlainSimLaterClashAtK keep x) : PlainSimClashAtK keep x := by
  intro s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω _ _ P ω
  refine ⟨fun hh => ?_, fun hh hleaf => ?_⟩
  · exact plainClashQ_orK keep _ (plainClashQ₀K keep hs hin hset h hd _ _ _
      (plain_clash₀AtK keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω)
      ((hl s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).1 hh) fun _ hq => hq
  · exact plainClashQ_orK keep _ (plainClashQ₀K keep hs hin hset h hd _ _ _
      (plain_clash₀HAtK keep hs hin hset h hh hleaf hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω)
      ((hl s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).2 hh hleaf) fun _ hq => hq

end FlockDeploy.KS
