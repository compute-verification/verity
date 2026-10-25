import Proofs.Flock.Claims.KS.PlainLeafClashK
import Proofs.Flock.Claims.KS.PlainClashRunK

/-!
# The live self-clash at plain SHA-512 leaves, at `VK keep`

`PlainLeafClash`'s run-level half over the verifier `VK keep`: `plainDecQUK`, `PlainClashQUK`, `plainClashQ₀UK`,
`plainClashQ_orUK`, `plainClashQLUK` and `plainSimClashHUK`, from `PlainLeafClashK`'s clashes at the draw.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
  Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- `plainDecQU` at `VK keep`. -/
def plainDecQUK (keep : Bool → Bool) {mPts : ℕ} (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) (T : TabSpec mPts) (I : Exec.Inputs)
    (d : Json) {M : ℕ} (C : Fin M → List 𝔽) (rootB₀ : Array ByteArray)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)) (o : LiveOutJ 1 mPts) :
    Prop :=
  ∃ a, Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
    (plainVerdictAtK keep I d C rootB₀) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Refine.Dig) execArith ([T].get j)) o a ∧
    Q (a.getD 0 dfltOutL)

/-- `PlainClashQU` at `VK keep`. -/
def PlainClashQUK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536),
    plainDecQUK keep Q T I d C (plainRoot P x ω) opn (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec)
      (plainW spec hw) sha512 (plainSimK keep P x spec st) x (ω, u) r') →
    (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).out = .acc →
    PlainRawClashLK keep P x ω (plainΦ x spec st (u, r'))

/-- `plainClashQ₀U` at `VK keep`. -/
theorem plainClashQ₀UK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        Clash₀LU sch (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        EntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQUK keep P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (Clash₀LU sch) := by
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

/-- `plainClashQ_orU` at `VK keep`. -/
theorem plainClashQ_orUK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec} {st : Stmt}
    {T : TabSpec spec.mPts} {hw : 2 * spec.mPts ≤ 65536} {I : Exec.Inputs} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8)}
    {Q₁ Q₂ : Model.OutCL 𝔽 Dig (List UInt8) → Prop} (Q : Model.OutCL 𝔽 Dig (List UInt8) → Prop)
    (h₁ : PlainClashQUK keep P x ω st T hw I d opn Q₁) (h₂ : PlainClashQUK keep P x ω st T hw I d opn Q₂)
    (hQ : ∀ out, Q out → Q₁ out ∨ Q₂ out) : PlainClashQUK keep P x ω st T hw I d opn Q := by
  rintro C u r' ⟨a, ha, hq⟩ hacc
  rcases hQ _ hq with hq | hq
  · exact h₁ C u r' ⟨a, ha, hq⟩ hacc
  · exact h₂ C u r' ⟨a, ha, hq⟩ hacc

/-- `plainClashQLU` at `VK keep`. -/
theorem plainClashQLUK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        LaterClashLU sch (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        LaterEntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQUK keep P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (LaterClashLU sch) := by
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

/-- `plainSimClashHU` at `VK keep`. -/
theorem plainSimClashHUK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (regionsOf st)).m) (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQUK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
      ins.read.I d (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
      (SelfClashL H512 enc512 leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) :=
  plainClashQ_orUK keep _ (plainClashQ₀UK keep hs hin hset h hd _ _ _
      (plain_clash₀HUAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω)
    (plainClashQLUK keep hs hin hset h hd _ _ _
      (plain_laterClashHUAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω) fun _ hq => hq

end FlockDeploy.KS
