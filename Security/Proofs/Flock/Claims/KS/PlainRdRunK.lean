import Proofs.Flock.Claims.KS.PlainRdRun
import Proofs.Flock.Claims.KS.PlainLeafKSK

/-!
# The plain table bound at one tape at `VK keep`, each slot at the run's own level columns

`PlainCapsDep`'s `plain_liveRow` (its `mem_rdAtK` is `PlainCapsDep`'s own), `PlainRawL`'s `plainOffL_le_rd` and `plain_accPr_le_rdL`, and
`PlainRdRun`'s `plainRel_of`, `plain_accPr_le_rdRun` and `plain_accPr_le_rdAll`, over the verifier `VK keep`: the
simulator at `plainSimK keep` and the verdict at the draw `plainVerdictAtK keep`.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Caps

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

/-- `plain_liveRow` at `VK keep`. -/
theorem plain_liveRowK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) {u : plainTape x}
    {r' : Fin (plainN x + 2) → LCoin 65536}
    (hacc : (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).out = .acc)
    (hA : AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN
      (plainDg spec) (plainW spec (plain_mPts hs hin hset).1) sha512 (plainSimK keep P x spec st) x (ω, u) r')))
    {sch : Accounting.Schedule} {out : Model.OutCL 𝔽 Dig (List UInt8)} {k : ℕ} (hk : k < 2)
    (hL : LaterEntriesAtK keep sch ins.read.I d (wrPDK keep ins.read.I d (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN
      (plainDg spec) (plainW spec (plain_mPts hs hin hset).1) sha512 (plainSimK keep P x spec st) x (ω, u) r')) out k) :
    ∀ ℓ, 1 ≤ ℓ → ∀ q v, (q, v) ∈ (slotOpensL out k ℓ (lvOf sch ℓ).queries).map (fun x => (x.1, x.2.1)) →
      (q, v) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).seen k ℓ := by
  have hw := (plain_mPts hs hin hset).1
  have hm1 := (plain_mPts hs hin hset).2.1
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨spec'', st', sess, sch', hsu, hdec, hf', -, s₀, p, hs₀, -, -, hent⟩ := hL
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := accK_decide keep hacc
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := plainSimRunK keep P ω hset h hw hm1 hp2 u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hA hdec hent
  have hacc' := run_acc_next sha512 P (VK keep) x ω _ hacc
  replace hacc' : ((nextK keep x ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length).map
        (Claims.coinSeq (plainΦ x spec st (u, r'))))).eval sha512).1 = .acc := hacc'
  obtain ⟨hv, hc, hsu₀, hzk, hpi⟩ := decideK_sha512 keep hact hacc'
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
  subst spec'' e₅
  obtain ⟨hpr, hall⟩ := recordsLiveAtDK_wrPDK keep ins.read.I d (viewOf 1 spec.mPts spec link c₁ f) hA
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
    (HiddenExec.setupTables_pointsAny hsetA).1 hk
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ y : StreamRec,
      y.rounds.toList.map Round.erase =
        (((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec)).map Round.erase →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro y hy j
    rw [← Array.getElem?_toList, coins_getElem?_of_erase hy j, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrPDK keep ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  rw [hpr'] at hent
  intro ℓ h1 q v hqv
  rw [hent ℓ h1, plainLevelColsOf_congr _ _ _ (hco y₀ hr₀), ← getD_empty, ← hcols ℓ] at hqv
  exact hqv

end Caps

section Raw

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

open Classical in
/-- `plainOffL_le_rd` at `VK keep`. -/
theorem plainOffL_le_rdK (keep : Bool → Bool) {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {st : Stmt}
    (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (T : Model.Oracle 𝔽) (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ)
    (hL : plainN x + 2 = (V false).k x)
    (hslot : ∀ u, SlotRSL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
      hk0 hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) r.val ℓ.val j
      (simAt execArith Refine.H512 Refine.enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
        (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn (plainDg spec)
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
          (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u)))
    (rdD : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (Fin ((V false).k x) → Coin) → Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ c c' : Fin ((V false).k x) → Coin, (∀ i : Fin ((V false).k x), i.val < j → c i = c' i) → w c = w c')
    (hrel : ∀ (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536) a,
      Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
          (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
          (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSimK keep P x spec st) x (ω, u) rr) a →
        ∀ q v, opCapL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val a q = some v →
          v ∈ rdD (plainΦ x spec st (u, rr)) q) :
    (∑ u, Lock.nodeE (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0)
        (slotOf r.val (ℓ.val + 1))
        (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8))) =>
            opRL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val (ℓ.val + 1) d
              (outs.getD 0 dfltOutL))
          (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen)
          (plurSL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r.val (ℓ.val + 1)))
        (lockSessL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
          hk0 hm PT_LOCAL spec.mPts [] [] T (capOfRoot (plainRoot P x ω)[0]!))
        (simAt execArith Refine.H512 Refine.enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
          (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn (plainDg spec)
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
          (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u))) /
        Fintype.card (plainTape x) ≤
      ∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q} := by
  have hR : (∑ q, Claims.Pr fun c : Fin ((V false).k x) → Coin => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q}) =
      Claims.Ex fun p : plainTape x × (Fin (plainN x + 2) → LCoin 65536) =>
        ∑ q, ind (fun c => (rdD c q).Nonempty ∧ rdD c q ≠ {w c q}) (plainΦ x spec st p) := by
    rw [Claims.Bridge.Ex_sum_univ]
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [← Pr_plainΦ (spec := spec) (st := st) hL, Claims.Bridge.Pr_eq_Ex]
    rfl
  rw [hR, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((slotOffRSL_le_offC execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁
    rest ext hk0 hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) T r ℓ.val j _ (hslot u)).trans
    ((offC_simAt_le execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u) _ r.val ℓ.val j).trans
      (Claims.Bridge.Live.offC_liveStrat_le plainN (plainDg spec) (plainW spec hw) sha512 (plainSimK keep P x spec st) x
        (ω, u) _ j (fun rr => w (plainΦ x spec st (u, rr)))
        (fun rr rr' h => hwj _ _ (plainΦ_prefix hL u h))))).trans
    (Claims.Ex_mono fun rr => Finset.sum_le_sum fun q _ => Claims.Bridge.ind_le_ind_imp fun hoff => ?_)
  obtain ⟨hs, hne⟩ := hoff
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
  obtain ⟨a, ha, hav⟩ := Claims.Bridge.liveOp_some hv
  have hd := hrel u rr a ha q v hav
  refine ⟨⟨v, hd⟩, fun he => hne ?_⟩
  rw [he, Finset.mem_singleton] at hd
  rw [hv]
  exact congrArg some hd

open Classical in
/-- `plain_accPr_le_rdL` at `VK keep`. -/
theorem plain_accPr_le_rdLK (keep : Bool → Bool) {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Refine.Dig Refine.H512) (hset : setupOf false x = some (ins, spec, sts))
    (h : tables false x = [(spec, st)]) {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (hp : spec.points = 2) {Ω : Type} (P : Claims.Prover Ω)
    (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (hsc : PlainClashRunK keep P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSimK keep P x spec st)
      (plainΦ x spec st)
      (LiveSelfClash Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Refine.Dig) execArith
          ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j)))
    (hslot : ∀ (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) u,
      SlotRSL execArith Refine.H512 Refine.enc512 Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
        PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) r.val ℓ.val (forkRound spec st k0 r.val ℓ.val)
        (simAt execArith Refine.H512 Refine.enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
          (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn (plainDg spec)
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
          (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u)))
    (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ r ℓ (c c' : Fin ((V false).k x) → Coin),
      (∀ i : Fin ((V false).k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) → w r ℓ c = w r ℓ c')
    (hrel : ∀ r ℓ (u : plainTape x) (rr : Fin (plainN x + 2) → LCoin 65536) a,
      Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Lf cap pend os)
          (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
          (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
          (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
          (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
          (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
            (plainSimK keep P x spec st) x (ω, u) rr) a →
        ∀ q v, opCapL Refine.H512 Refine.enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val a q = some v →
          v ∈ rdD r ℓ (plainΦ x spec st (u, rr)) q) :
    Claims.accPr sha512 (VK keep) P x ω ≤ εtable false x +
      (Claims.Pr fun s : Fin ((V false).k x) → Coin => plainEv₁K keep P x ω s) +
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
        Claims.Pr (fun c : Fin ((V false).k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
  have hL := plainN_add hset
  have htr := plain_first_of_simDK keep P x ω C spec.mPts hset hd plainN (plainDg spec) (plainW spec hw)
    (plainSimK keep P x spec st) (plainΦ x spec st) ⟨Pr_plainΦ hL, plainSimRunK keep P ω hset h hw hm1 hp⟩
  have hmain := plain_le_sessionsRawLK keep Lf h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay P ω C hT (plainTape x)
    (capOfRoot (plainRoot P x ω)[0]!) (fun u => simAt execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw)))
  refine hmain.trans (add_le_add (add_le_add (add_le_add le_rfl ?_) ?_) le_rfl)
  · exact le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Lf
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
      (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw))
      (liveMix_le_Pr (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
        (plainSimK keep P x spec st) x ω _ (plainEv₁K keep P x ω) (plainΦ x spec st) (Pr_plainΦ hL _) hsc)
  · rw [Finset.sum_range]
    refine Finset.sum_le_sum fun r _ => ?_
    rw [Claims.Flock.sum_Icc_one]
    exact Finset.sum_le_sum fun ℓ _ => plainOffL_le_rdK keep Lf L m k0 l₀ l₁ rest ext hsch hk0 hm hlay d opn hdec hw P ω C
      (Claims.Flock.oracleOf C) r ℓ _ hL (hslot r ℓ) (rdD r ℓ) (w r ℓ) (hwj r ℓ) (hrel r ℓ)

end Raw

section Rd

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

/-- `plainRel_of` at `VK keep`. -/
theorem plainRel_ofK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512) (T : TabSpec spec.mPts)
    (sch : Accounting.Schedule) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        ∀ r < 2, LaterEntriesAtK keep sch ins.read.I d (wrPDK keep ins.read.I d o)
          (tableOutL H512 enc512 Lf (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) r)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x)
    (rr : Fin (plainN x + 2) → LCoin 65536) (a : List (Model.OutCL 𝔽 Dig (List UInt8)))
    (ha : Claims.Bridge.RetOK (fun _ cap pend os => tableOutL H512 enc512 Lf cap pend os)
      (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Dig) execArith ([T].get j))
      (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec (plain_mPts hs hin hset).1)
        sha512 (plainSimK keep P x spec st) x (ω, u) rr) a)
    (r : Fin 2) (ℓ : ℕ) (q : Fin (2 ^ (lvOf sch (ℓ + 1)).logLen)) (v : Fin (2 ^ (lvOf sch (ℓ + 1)).lanes) → 𝔽)
    (hv : opCapL H512 enc512 Lf sch r.val ℓ a q = some v) :
    (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, rr))).out = .acc ∧
      ∃ e, (q.val, e) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, rr))).seen r.val
        (ℓ + 1) ∧ rowFin (lvOf sch (ℓ + 1)).lanes e = v := by
  obtain ⟨hV, pend, hpl, rfl⟩ := ha
  have hacc := plain_acc_of_abortK keep (plainSimAbortAtK keep x s hs hin ins spec sts hset st h Ω P ω ins.read.I d C
    (plainRoot P x ω)) hV
  refine ⟨hacc, ?_⟩
  unfold opCapL at hv
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hv
  obtain ⟨dc, -, hdc⟩ := Option.bind_eq_some_iff.1 hv
  obtain ⟨y, hy, hyq, hyv⟩ := opRL_eq_some (Option.isSome_iff_exists.2 ⟨v, hdc⟩)
  refine ⟨y.2.1, ?_, Option.some.inj (hyv.symm.trans hdc)⟩
  rw [← hyq]
  exact plain_liveRowK keep hs hin hset h hd P ω hacc hV.1 r.isLt (hE _ hV.1 (pend 0) (hpl 0) r.val r.isLt) (ℓ + 1)
    (by omega) y.1 y.2.1 (List.mem_map_of_mem hy)

open Classical in
/-- `plain_accPr_le_rdRun` at `VK keep`. -/
theorem plain_accPr_le_rdRunK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (Lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (regionsOf st)).m) (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL H512 enc512 Lf cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 Lf
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL H512 enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
          o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
            (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        ∀ r < 2, LaterEntriesAtK keep ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPDK keep ins.read.I d o)
          (tableOutL H512 enc512 Lf (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) r)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hsc : PlainClashRunK keep P x ω spec.mPts plainN (plainDg spec) (plainW spec (plain_mPts hs hin hset).1)
      (plainSimK keep P x spec st) (plainΦ x spec st)
      (LiveSelfClash H512 enc512 Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Dig) execArith
          ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j)))
    (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
    (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
      Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
        Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽)
    (hwj : ∀ r ℓ (c c' : Fin ((V false).k x) → Coin),
      (∀ i : Fin ((V false).k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) → w r ℓ c = w r ℓ c')
    (hrd : ∀ r ℓ (c : Fin ((V false).k x) → Coin), (Claims.run sha512 P (VK keep) x ω c).out = .acc →
      ∀ q a, (q.val, a) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω c).seen r.val (ℓ.val + 1) →
        rowFin (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes a ∈ rdD r ℓ c q) :
    Claims.accPr sha512 (VK keep) P x ω ≤ εtable false x +
      (Claims.Pr fun s : Fin ((V false).k x) → Coin => plainEv₁K keep P x ω s) +
      ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
        Claims.Pr (fun c : Fin ((V false).k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p :=
  plain_accPr_le_rdLK keep Lf hset h hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay opn hdec (plain_mPts hs hin hset).1
    (plain_mPts hs hin hset).2.1 (plain_mPts hs hin hset).2.2 P ω C hT hsc
    (fun r ℓ u => slotRSL_lead L (plain_mPts hs hin hset).2.2 execArith H512 enc512 Lf m k0 l₀ l₁ rest ext hsch hk0
      hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!)
      (leadsAfter_simAt execArith H512 enc512 Lf (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ _ opn
        (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (fun _ => rfl)
        (plainRoot P x ω) (plainLiveK keep x spec st (plain_mPts hs hin hset).1 P ω u)
        (plainSim_rootK keep P x spec st ω (plain_mPts hs hin hset).1 u)
        (plainLive_schedLK keep hset h hsch (by simp) (plain_mPts hs hin hset).1 P ω u)) r ℓ.val)
    rdD w hwj fun r ℓ u rr a ha q v hv => by
      obtain ⟨hacc, e, he, rfl⟩ := plainRel_ofK keep hs hin hset h hd Lf _ _ opn hE P ω C u rr a ha r ℓ.val q v hv
      exact hrd r ℓ _ hacc q e he

open Classical in
/-- `plain_accPr_le_rdAll` at `VK keep`. -/
theorem plain_accPr_le_rdAllK (keep : Bool → Bool) {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (VK keep) x) :
    ∃ (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (m k0 : ℕ)
      (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ),
      setupOf false x = some (ins, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      schedOf sch = ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧ tables false x = [(spec, st)] ∧
      Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧
      ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽),
        (∀ p, wide false x (C p)) → ¬ Claims.Flock.GoodTable (deployPlainK keep Readable.relation) (wordLen false) x C →
        ∀ (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
            Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
              Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
          (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
            Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
              Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽),
        (∀ r ℓ (c c' : Fin ((V false).k x) → Coin),
          (∀ i : Fin ((V false).k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) →
            w r ℓ c = w r ℓ c') →
        (∀ r ℓ (c : Fin ((V false).k x) → Coin), (Claims.run sha512 P (VK keep) x ω c).out = .acc →
          ∀ q a, (q.val, a) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω c).seen r.val (ℓ.val + 1) →
            rowFin (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes a ∈ rdD r ℓ c q) →
        Claims.accPr sha512 (VK keep) P x ω ≤ εtable false x +
          (Claims.Pr fun s : Fin ((V false).k x) → Coin => plainEv₁K keep P x ω s) +
          ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
            Claims.Pr (fun c : Fin ((V false).k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
          Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
            ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
              Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧
                u ≠ C p := by
  obtain ⟨s, hs, hin⟩ := accK_inScope keep Readable.relation hno
  obtain ⟨ins, spec, sts, d, hset, hd⟩ := accK_draw keep hno
  have h1 := plain_one hs hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  obtain ⟨sch, hF, hF'⟩ := exec_sched hsch
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  have hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlainK keep Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (regionsOf (sts[0]'h0))) ⟨_, k0, l₀ :: l₁ :: rest, ext⟩
        (Claims.Flock.oracleOf C) := fun C hw hg => by
    have := plain_not_committed hs hin hset h0 L C hw fun hg' =>
      hg ((goodTable_plainK keep Readable.relation (wordLen false) x C).2 hg')
    rwa [e] at this
  refine ⟨ins, spec, sts, sts[0]'h0, sch, (sts[0]'h0).m, k0, l₀, l₁, rest, ext, hset,
    Array.getElem?_eq_getElem h0, hF, hF'.symm, ht, hsch, ?_⟩
  intro Ω _ _ P ω C hw hg rdD w hwj hrd
  cases hh : ins.read.I.tags.hiddenOutputs
  · refine plain_accPr_le_rdRunK keep hs hin hset ht hd leaf512 L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay _
      (plain_hdecAtK keep hs hin hset ht hh hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay)
      (plain_laterEntriesAtK keep hs hin hset ht hh hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay) P ω C (hnc C hw hg)
      (plainClashRun_of_rawLK keep (plainRawClashRunL_of_abortK keep (fun u r' hna hq => ?_)
        ((plainSimClashAt_of_laterK keep (plainSimLaterClashAtK keep x) s hs hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁ rest
          ext hsch hm13 hlay Ω P ω).1 hh C)))
      rdD w hwj hrd
    obtain ⟨a, ha, -⟩ := hq
    exact plainSimAbortAtK keep x s hs hin ins spec sts hset _ ht Ω P ω ins.read.I d C (plainRoot P x ω) u r' hna ha.1
  · by_cases hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"
    · refine plain_accPr_le_rdRunK keep hs hin hset ht hd leaf512 L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay _
        (plain_hdecHAtK keep hs hin hset ht hh hl hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay)
        (plain_laterEntriesHAtK keep hs hin hset ht hh hl hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay) P ω C (hnc C hw hg)
        (plainClashRun_of_rawLK keep (plainRawClashRunL_of_abortK keep (fun u r' hna hq => ?_)
          ((plainSimClashAt_of_laterK keep (plainSimLaterClashAtK keep x) s hs hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁
            rest ext hsch hm13 hlay Ω P ω).2 hh hl C)))
        rdD w hwj hrd
      obtain ⟨a, ha, -⟩ := hq
      exact plainSimAbortAtK keep x s hs hin ins spec sts hset _ ht Ω P ω ins.read.I d C (plainRoot P x ω) u r' hna ha.1
    · have hl' := plain_leafU hs hin hset hl
      refine plain_accPr_le_rdRunK keep hs hin hset ht hd leafPlain L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay _
        (plain_hdecHUAtK keep hs hin hset ht hh hl' hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay)
        (plain_laterEntriesHUAtK keep hs hin hset ht hh hl' hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay) P ω C (hnc C hw hg)
        (plainClashRun_of_rawLK keep (plainRawClashRunL_of_abortK keep (fun u r' hna hq => ?_)
          (plainSimClashHUK keep hs hin hset ht hh hl' hd L _ k0 l₀ l₁ rest ext hsch hm13 hlay P ω C)))
        rdD w hwj hrd
      obtain ⟨a, ha, -⟩ := hq
      exact plainSimAbortAtK keep x s hs hin ins spec sts hset _ ht Ω P ω ins.read.I d C (plainRoot P x ω) u r' hna ha.1

end Rd

end FlockDeploy.KS
