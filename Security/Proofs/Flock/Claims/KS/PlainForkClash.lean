import Proofs.Flock.Claims.KS.PlainFork
import Proofs.Flock.Claims.KS.PlainClashRunK

/-!
# The live fork event's collision

`PlainForkClash` asks that two accepting runs of `P` at the concrete simulator's coins whose decoded live outcomes have
`ForkEvL` at rep `r`'s level `ℓ + 1` collide among the verifier's calls in the two. Each live outcome is its run's
(`plainSimRun`), and where `verify` accepts what `wrPD` writes for it, rep `r`'s recorded caps are its proof's
recursive caps and its level-`ℓ + 1` openings its deployed entries (`plain_laterEntriesAt`), each an `Opening` in the
run's log against the flattened `ℓ`-th recursive cap (`plain_levelOpening`). One recorded cap in both runs is one cap of
64-byte digests (`capDig_inj`), so one root, and two rows opened at one position of it that differ collide in the joined
logs (`tiedColl_of_openings`).

* `plain_liveOpening`: at an accepted run, its live outcome's output records its proof's recursive caps, and a
  later-level opening there is an `Opening` in the run's log against one of them.
* `capDig_inj`, `opRL_eq_some`, `forkEvL_entries`: two caps of 64-byte digests with one `capDig` are one, and
  `ForkEvL` is one recorded cap and two openings of one position of it to different rows.
* `plainForkQ_of`, `plainForkClash`: so `PlainForkClash` holds at every statement.
* `plain_liveOpeningK`, `plainForkQ_ofK`, `plainForkClashK`: the same at `VK keep` (`plainSimK keep`, `wrPDK keep`,
  `rawLevelCols_plainK`), so `PlainForkClashK keep` holds at every statement.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **An accepted run's live outcome records its proof's caps and opens rows of them in its log**: at the concrete
simulator's coins where `P`'s run accepts and `verify` accepts what `wrPD` writes for the live outcome, in an output
whose rep `k` records the deployed rep's caps and later openings (`LaterEntriesAt`), the recorded caps are the decoded
proof's recursive caps, and each level-`ℓ ≥ 1` opening `(q, v)` is a row `v` of an `Opening` of `q` against the
flattened `ℓ − 1`-th of them, a cap of `2^cap` 64-byte digests, in the run's log: the live outcome is the run's
(`plainSimRun`), the opening one of the run's level-`ℓ` rows (`rawLevelCols_plain`), an `Opening` there
(`plain_levelOpening`). -/
theorem plain_liveOpening {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) {u : plainTape x}
    {r' : Fin (plainN x + 2) → LCoin 65536}
    (hacc : (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).out = .acc)
    (hA : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN
      (plainDg spec) (plainW spec (plain_mPts hs hin hset).1) sha512 (plainSim P x spec st) x (ω, u) r')))
    {sch : Accounting.Schedule} {out : Model.OutCL 𝔽 Dig (List UInt8)} {k : ℕ} (hk : k < 2)
    (hL : LaterEntriesAt sch ins.read.I d (wrPD ins.read.I d (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN
      (plainDg spec) (plainW spec (plain_mPts hs hin hset).1) sha512 (plainSim P x spec st) x (ω, u) r')) out k) :
    ∃ (st' : Stmt) (sch' : Flock.Schedule) (p : Proof), setupAtD ins.read.I (some d) = .ok (spec, #[st']) ∧
      Flock.fast100 st'.m = some sch' ∧
      (∀ ℓ dc, CapAtL k ℓ out dc → ℓ + 1 < sch'.levels.size ∧ capDig p.ligerito.recursiveCaps[ℓ]! = dc) ∧
      ∀ ℓ, 1 ≤ ℓ → ∀ q v, (q, v) ∈ (slotOpensL out k ℓ (lvOf sch ℓ).queries).map (fun x => (x.1, x.2.1)) →
        ∃ row : Array F128, v = row.toList.map FlockLevel3.GF128.of ∧
          p.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch'.levels.getD ℓ default).cap ∧
          (∀ b ∈ p.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
          Opening (Setup.ofCircuit st').merkle (sch'.levels.getD ℓ default) (2 ^ (sch'.levels.getD ℓ default).lanes)
            (p.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty)
            (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).vLog q row := by
  have hw := (plain_mPts hs hin hset).1
  have hm1 := (plain_mPts hs hin hset).2.1
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨spec'', st', sess, sch', hsu, hdec, hf', -, s₀, p, hs₀, hp, hcaps, hent⟩ := hL
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := acc_decide hacc
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := plainSimRun P ω hset h hw hm1 hp2 u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hA hdec hp hent
  have hacc' := run_acc_next sha512 P (V false) x ω _ hacc
  replace hacc' : ((next false x ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length).map
        (Claims.coinSeq (plainΦ x spec st (u, r'))))).eval sha512).1 = .acc := hacc'
  obtain ⟨hv, hc⟩ := (next_decide_sha512 hact).1 hacc'
  have hpi := publicInputs_none_of_checks hc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hsu₀⟩ := setupOf_some (act_decide hact)
  have hzk : ins.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
  subst spec'' e₅
  obtain ⟨hpr, hall⟩ := recordsLiveAtD_wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f) hA
  obtain ⟨-, hrs⟩ := hall spec #[st'] sess hsu hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨k, hk⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decision_runs hsu₀ hrun
  have hv' : verifyWithM (hashesIn ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).vLog.map
      Prod.fst)) spec (tabsOf #[st']) f.record f.proofs = .ok () := verdict_within hvr
  have hcols := rawLevelCols_plain (st := st') _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hf'
    (HiddenExec.setupTables_pointsAny hsetA).1 hk
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ y : StreamRec,
      y.rounds.toList = ((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec) →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro y hy j
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  rw [hpr'] at hp hent
  refine ⟨st', sch', p, hsu, hf', fun ℓ dc hc => ?_, fun ℓ h1 q v hqv => ?_⟩
  · unfold CapAtL at hc
    rw [hcaps] at hc
    rw [List.getElem?_map] at hc
    by_cases hl : ℓ < sch'.levels.size - 1
    · rw [List.getElem?_range hl] at hc
      exact ⟨by omega, Option.some.inj hc⟩
    · rw [List.getElem?_eq_none (by simpa using hl)] at hc
      cases hc
  rw [hent ℓ h1, plainLevelColsOf_congr _ _ _ (hco y₀ hr₀), ← getD_empty, ← hcols ℓ] at hqv
  obtain ⟨s₁, spec₁, sts₁, f₁, st₁, sch₁, p₁, row, hs₁, hf₁, hst₁, hsch₁, hp₁, hrow, hcs, hcw, hop⟩ :=
    plain_levelOpening hacc hqv
  obtain ⟨rfl, rfl, rfl⟩ : s₁ = ins ∧ spec₁ = spec ∧ sts₁ = #[st'] := by simpa using hs₁.symm.trans hset
  obtain rfl : st' = st₁ := by simpa using hst₁
  obtain rfl := Option.some.inj (hf.symm.trans hf₁)
  obtain rfl := Option.some.inj (hf'.symm.trans hsch₁)
  rw [hp2, Nat.mul_comm] at hp₁
  obtain rfl := Except.ok.inj (hp.symm.trans hp₁)
  exact ⟨row, hrow, hcs, hcw, hop⟩

/-- **Two caps of 64-byte digests, of one size, with one `capDig`, are one.** -/
theorem capDig_inj {c₁ c₂ : Array ByteArray} (h : capDig c₁ = capDig c₂) (hs : c₁.size = c₂.size)
    (h₁ : ∀ b ∈ c₁.toList, b.size = 64) (h₂ : ∀ b ∈ c₂.toList, b.size = 64) : c₁ = c₂ := by
  refine Array.ext hs fun i hi₁ hi₂ => ?_
  have e := congrArg Subtype.val (congrFun h i)
  simp only [capDig, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi₁, Array.getElem?_eq_getElem hi₂,
    Option.getD_some] at e
  rwa [toDig_val (h₁ _ (Array.getElem_mem_toList hi₁)), toDig_val (h₂ _ (Array.getElem_mem_toList hi₂))] at e

open Classical in
/-- **An opened position at a rep's cap is a verifying opening there**: where rep `r`'s level-`ℓ` openings at `cap`
open `p`, one of them is at `p`, opened to that row. -/
theorem opRL_eq_some {D Sl : Type} {Hc : List UInt8 → D} {E : Merkle.Enc (List 𝔽) D}
    {Lf : Merkle.Leaf (List 𝔽) Sl D Hc} {sch : Accounting.Schedule} {r ℓ : ℕ} {cap : Model.Cap D}
    {out : Model.OutCL 𝔽 D Sl} {p : Fin (2 ^ (lvOf sch ℓ).logLen)}
    (h : (opRL Hc E Lf sch r ℓ cap out p).isSome) :
    ∃ x ∈ slotOpensL out r ℓ (lvOf sch ℓ).queries, x.1 = p.val ∧
      opRL Hc E Lf sch r ℓ cap out p = some (rowFin (lvOf sch ℓ).lanes x.2.1) := by
  unfold opRL opOfL at h ⊢
  rw [Option.isSome_map] at h
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  have hmem := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hmem
  exact ⟨x, hmem.1, hmem.2.1, by rw [hx]; rfl⟩

/-- **The fork event's two entries**: `ForkEvL` at rep `r`'s level `ℓ + 1` is one recorded cap `d` in both outputs
and an opening in each of one position to different rows. -/
theorem forkEvL_entries {D Sl : Type} [Inhabited D] [Inhabited Sl] {Hc : List UInt8 → D}
    {E : Merkle.Enc (List 𝔽) D} {Lf : Merkle.Leaf (List 𝔽) Sl D Hc} {sch : Accounting.Schedule} {r ℓ : ℕ}
    {a a' : List (Model.OutCL 𝔽 D Sl)} (h : ForkEvL Hc E Lf sch r ℓ a a') :
    ∃ dc, CapAtL r ℓ (a.getD 0 dfltOutL) dc ∧ CapAtL r ℓ (a'.getD 0 dfltOutL) dc ∧
      ∃ x₁ ∈ slotOpensL (a.getD 0 dfltOutL) r (ℓ + 1) (lvOf sch (ℓ + 1)).queries,
      ∃ x₂ ∈ slotOpensL (a'.getD 0 dfltOutL) r (ℓ + 1) (lvOf sch (ℓ + 1)).queries,
        x₁.1 = x₂.1 ∧ x₁.2.1 ≠ x₂.2.1 := by
  obtain ⟨dc, hc₁, hc₂, p, hs₁, hs₂, hne⟩ := h
  obtain ⟨x₁, hx₁, hp₁, he₁⟩ := opRL_eq_some (out := a.getD 0 dfltOutL) hs₁
  obtain ⟨x₂, hx₂, hp₂, he₂⟩ := opRL_eq_some (out := a'.getD 0 dfltOutL) hs₂
  refine ⟨dc, hc₁, hc₂, x₁, hx₁, x₂, hx₂, hp₁.trans hp₂.symm, fun he => hne ?_⟩
  show opRL Hc E Lf sch r (ℓ + 1) dc (a.getD 0 dfltOutL) p = opRL Hc E Lf sch r (ℓ + 1) dc (a'.getD 0 dfltOutL) p
  rw [he₁, he₂, he]

/-- **The live fork event's collision, at decoders whose outputs record the deployed reps' caps and later openings**:
where `verify`'s acceptance of what `wrPD` writes for an outcome makes the decoded output's reps record the deployed
reps' caps and later entries (`hE`, as `plain_laterEntriesAt` and `plain_laterEntriesHAt` give), two accepting runs of
`P` whose live outcomes decode to `ForkEvL` at rep `r`'s level `ℓ + 1` record one cap `d` there, the `capDig` of each
run's proof's `ℓ`-th recursive cap, so one cap of 64-byte digests (`capDig_inj`), and open one position of it to two
different rows, each an `Opening` against that cap in its run's log (`plain_liveOpening`), which collide in the joined
logs (`tiedColl_of_openings`). -/
theorem plainForkQ_of {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        ∀ r < 2, LaterEntriesAt sch ins.read.I d (wrPD ins.read.I d o)
          (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) r)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainForkQ P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn sch := by
  intro C u r₁ r₂ r ℓ hq hacc₁ hacc₂
  obtain ⟨a₁, a₂, ⟨hV₁, pend₁, hpl₁, rfl⟩, ⟨hV₂, pend₂, hpl₂, rfl⟩, hF⟩ := hq
  obtain ⟨dc, hc₁, hc₂, x₁, hx₁, x₂, hx₂, hq, hne⟩ := forkEvL_entries hF
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hc₁ hc₂ hx₁ hx₂
  obtain ⟨st₁, sch₁, p₁, hsu₁, hf₁, hcap₁, hop₁⟩ := plain_liveOpening hs hin hset h hd P ω hacc₁ hV₁.1 r.isLt
    (hE _ hV₁.1 (pend₁ 0) (hpl₁ 0) r.val r.isLt)
  obtain ⟨st₂, sch₂, p₂, hsu₂, hf₂, hcap₂, hop₂⟩ := plain_liveOpening hs hin hset h hd P ω hacc₂ hV₂.1 r.isLt
    (hE _ hV₂.1 (pend₂ 0) (hpl₂ 0) r.val r.isLt)
  obtain rfl : st₁ = st₂ := by simpa using hsu₁.symm.trans hsu₂
  obtain rfl := Option.some.inj (hf₁.symm.trans hf₂)
  obtain ⟨-, hd₁⟩ := hcap₁ ℓ dc hc₁
  obtain ⟨-, hd₂⟩ := hcap₂ ℓ dc hc₂
  obtain ⟨row₁, hrow₁, hcs₁, hcw₁, ho₁⟩ := hop₁ (ℓ + 1) (by omega) x₁.1 x₁.2.1 (List.mem_map_of_mem hx₁)
  obtain ⟨row₂, hrow₂, hcs₂, hcw₂, ho₂⟩ := hop₂ (ℓ + 1) (by omega) x₂.1 x₂.2.1 (List.mem_map_of_mem hx₂)
  simp only [Nat.add_sub_cancel] at hcs₁ hcw₁ ho₁ hcs₂ hcw₂ ho₂
  have hcap : p₁.ligerito.recursiveCaps[ℓ]! = p₂.ligerito.recursiveCaps[ℓ]! :=
    capDig_inj (hd₁.trans hd₂.symm) (hcs₁.trans hcs₂.symm) hcw₁ hcw₂
  rw [hcap, hq] at ho₁
  exact tiedColl_of_openings ho₁ ho₂ fun he => hne (by rw [hrow₁, hrow₂, he])

/-- **The live fork event's collision holds at every statement**: `plainForkQ_of` at `plain_laterEntriesAt` and
`plain_laterEntriesHAt`. -/
theorem plainForkClash (x : Claims.Msg) : PlainForkClash x := by
  intro s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω
  exact ⟨fun hh => plainForkQ_of hs hin hset h hd _ _ _
      (plain_laterEntriesAt hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω,
    fun hh hl => plainForkQ_of hs hin hset h hd _ _ _
      (plain_laterEntriesHAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω⟩

/-! ## At `VK keep` -/

/-- `plain_liveOpening` at `VK keep`. -/
theorem plain_liveOpeningK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
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
    ∃ (st' : Stmt) (sch' : Flock.Schedule) (p : Proof), setupAtD ins.read.I (some d) = .ok (spec, #[st']) ∧
      Flock.fast100 st'.m = some sch' ∧
      (∀ ℓ dc, CapAtL k ℓ out dc → ℓ + 1 < sch'.levels.size ∧ capDig p.ligerito.recursiveCaps[ℓ]! = dc) ∧
      ∀ ℓ, 1 ≤ ℓ → ∀ q v, (q, v) ∈ (slotOpensL out k ℓ (lvOf sch ℓ).queries).map (fun x => (x.1, x.2.1)) →
        ∃ row : Array F128, v = row.toList.map FlockLevel3.GF128.of ∧
          p.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch'.levels.getD ℓ default).cap ∧
          (∀ b ∈ p.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
          Opening (Setup.ofCircuit st').merkle (sch'.levels.getD ℓ default) (2 ^ (sch'.levels.getD ℓ default).lanes)
            (p.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty)
            (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).vLog q row := by
  have hw := (plain_mPts hs hin hset).1
  have hm1 := (plain_mPts hs hin hset).2.1
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨spec'', st', sess, sch', hsu, hdec, hf', -, s₀, p, hs₀, hp, hcaps, hent⟩ := hL
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := accK_decide keep hacc
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := plainSimRunK keep P ω hset h hw hm1 hp2 u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hA hdec hp hent
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
  rw [hpr'] at hp hent
  refine ⟨st', sch', p, hsu, hf', fun ℓ dc hc => ?_, fun ℓ h1 q v hqv => ?_⟩
  · unfold CapAtL at hc
    rw [hcaps] at hc
    rw [List.getElem?_map] at hc
    by_cases hl : ℓ < sch'.levels.size - 1
    · rw [List.getElem?_range hl] at hc
      exact ⟨by omega, Option.some.inj hc⟩
    · rw [List.getElem?_eq_none (by simpa using hl)] at hc
      cases hc
  rw [hent ℓ h1, plainLevelColsOf_congr _ _ _ (hco y₀ hr₀), ← getD_empty, ← hcols ℓ] at hqv
  obtain ⟨s₁, spec₁, sts₁, f₁, st₁, sch₁, p₁, row, hs₁, hf₁, hst₁, hsch₁, hp₁, hrow, hcs, hcw, hop⟩ :=
    plain_levelOpeningK keep hacc hqv
  obtain ⟨rfl, rfl, rfl⟩ : s₁ = ins ∧ spec₁ = spec ∧ sts₁ = #[st'] := by simpa using hs₁.symm.trans hset
  obtain rfl : st' = st₁ := by simpa using hst₁
  obtain rfl := Option.some.inj (hf.symm.trans hf₁)
  obtain rfl := Option.some.inj (hf'.symm.trans hsch₁)
  rw [hp2, Nat.mul_comm] at hp₁
  obtain rfl := Except.ok.inj (hp.symm.trans hp₁)
  exact ⟨row, hrow, hcs, hcw, hop⟩

/-- `plainForkQ_of` at `VK keep`. -/
theorem plainForkQ_ofK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        ∀ r < 2, LaterEntriesAtK keep sch ins.read.I d (wrPDK keep ins.read.I d o)
          (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) r)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainForkQK keep P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn sch := by
  intro C u r₁ r₂ r ℓ hq hacc₁ hacc₂
  obtain ⟨a₁, a₂, ⟨hV₁, pend₁, hpl₁, rfl⟩, ⟨hV₂, pend₂, hpl₂, rfl⟩, hF⟩ := hq
  obtain ⟨dc, hc₁, hc₂, x₁, hx₁, x₂, hx₂, hq, hne⟩ := forkEvL_entries hF
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hc₁ hc₂ hx₁ hx₂
  obtain ⟨st₁, sch₁, p₁, hsu₁, hf₁, hcap₁, hop₁⟩ := plain_liveOpeningK keep hs hin hset h hd P ω hacc₁ hV₁.1 r.isLt
    (hE _ hV₁.1 (pend₁ 0) (hpl₁ 0) r.val r.isLt)
  obtain ⟨st₂, sch₂, p₂, hsu₂, hf₂, hcap₂, hop₂⟩ := plain_liveOpeningK keep hs hin hset h hd P ω hacc₂ hV₂.1 r.isLt
    (hE _ hV₂.1 (pend₂ 0) (hpl₂ 0) r.val r.isLt)
  obtain rfl : st₁ = st₂ := by simpa using hsu₁.symm.trans hsu₂
  obtain rfl := Option.some.inj (hf₁.symm.trans hf₂)
  obtain ⟨-, hd₁⟩ := hcap₁ ℓ dc hc₁
  obtain ⟨-, hd₂⟩ := hcap₂ ℓ dc hc₂
  obtain ⟨row₁, hrow₁, hcs₁, hcw₁, ho₁⟩ := hop₁ (ℓ + 1) (by omega) x₁.1 x₁.2.1 (List.mem_map_of_mem hx₁)
  obtain ⟨row₂, hrow₂, hcs₂, hcw₂, ho₂⟩ := hop₂ (ℓ + 1) (by omega) x₂.1 x₂.2.1 (List.mem_map_of_mem hx₂)
  simp only [Nat.add_sub_cancel] at hcs₁ hcw₁ ho₁ hcs₂ hcw₂ ho₂
  have hcap : p₁.ligerito.recursiveCaps[ℓ]! = p₂.ligerito.recursiveCaps[ℓ]! :=
    capDig_inj (hd₁.trans hd₂.symm) (hcs₁.trans hcs₂.symm) hcw₁ hcw₂
  rw [hcap, hq] at ho₁
  exact tiedColl_of_openings ho₁ ho₂ fun he => hne (by rw [hrow₁, hrow₂, he])

/-- `plainForkClash` at `VK keep`. -/
theorem plainForkClashK (keep : Bool → Bool) (x : Claims.Msg) : PlainForkClashK keep x := by
  intro s hs hin ins spec sts hset st h d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω
  exact ⟨fun hh => plainForkQ_ofK keep hs hin hset h hd _ _ _
      (plain_laterEntriesAtK keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω,
    fun hh hl => plainForkQ_ofK keep hs hin hset h hd _ _ _
      (plain_laterEntriesHAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω⟩

end FlockDeploy.KS
