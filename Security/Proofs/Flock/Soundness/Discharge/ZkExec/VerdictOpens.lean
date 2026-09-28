import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictStatements
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRep
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysDet

/-!
# A `--zk` rep's requested rows verify

`zk_opens_ok`: the third conclusion of `ZkRepAccepts`, the `--zk` analogue of the non-ZK session's `opensOKL_of`. Every
row a compiled `--zk` rep requests verifies against its level's cap (`OpensOKZ`), at openings that are the proof's rows
and salts.

The run `zk_rep_plays` builds records its requests; `zk_ligerito_reqs` builds the same run as `zk_ligerito_plays` and
tracks, along Ligerito's level loop, that its caps are the proof's and that each request is one whose Merkle check the
executable passed (`ReqOKZ`): level 0's on the `2^k₀ + 4`-lane rows against the commitment's cap (`Zk.queryLevel0`), a
later level's as the non-ZK session's `ReqOK`. `zk_rep_reqs` carries that to the rep, and a run has one result
(`plays_det`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D β : Type}

/-! ## Requests that passed the executable's Merkle checks -/

/-- **A `--zk` request whose Merkle check the executable passed**: at level 0, its paths have level 0's length and the
proof's level-0 rows, `2^lanes + EXTRA_LANES` wide, verify against the commitment's cap (`Zk.queryLevel0`); above,
the non-ZK session's `ReqOK`. -/
def ReqOKZ (ms : MerkleScheme) (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof) (req : Req) : Prop :=
  (req.level = 0 → req.depth = pathLen (levelOf (sch.levels.getD 0 default)) ∧
      merkleCheck ms cap (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d req.positions.toArray
        p.initialProof.rows p.initialProof.siblings p.initialProof.salts
        (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) = .ok ()) ∧
    (req.level ≠ 0 → ReqOK ms sch cap p req)

theorem reqOKZ_zero {ms : MerkleScheme} {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof}
    {ℓ' : ℕ} (he : ℓ' = 0) (hl : LevelSeam (sch.levels.getD 0 default)) (cw : Array F128) (lanes : ℕ)
    (hmc : merkleCheck ms cap (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
      (Flock.positions (sch.levels.getD 0 default) cw) p.initialProof.rows p.initialProof.siblings
      p.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) = .ok ()) :
    ReqOKZ ms sch cap p ⟨ℓ', Model.positions execArith (levelOf (sch.levels.getD 0 default)).queries
      (levelOf (sch.levels.getD 0 default)).logLen (vecOf (levelOf (sch.levels.getD 0 default)).queries cw), lanes,
      pathLen (levelOf (sch.levels.getD 0 default))⟩ := by
  subst he
  refine ⟨fun _ => ⟨rfl, ?_⟩, fun h => absurd rfl h⟩
  dsimp only
  rw [show Model.positions execArith (levelOf (sch.levels.getD 0 default)).queries
      (levelOf (sch.levels.getD 0 default)).logLen (vecOf (levelOf (sch.levels.getD 0 default)).queries cw) =
      (Flock.positions (sch.levels.getD 0 default) cw).toList from Refine.positions_eq _ hl cw, Array.toArray_toList]
  exact hmc

theorem reqOKZ_succ {ms : MerkleScheme} {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof}
    {ℓ ℓ' : ℕ} (he : ℓ' = ℓ) (hℓ0 : ℓ ≠ 0) (hℓ : ℓ < sch.levels.size) (hl : LevelSeam (sch.levels.getD ℓ default))
    (ch : List 𝕂) (hch : ch.length = (sch.levels.getD ℓ default).lanes) (cw : Array F128)
    {cap' : Array ByteArray} {rows : Array (Array F128)} {sibs salts : Array ByteArray}
    (hcap : capAt cap p ℓ = cap') (hopen : openAt p (sch.levels.size - 1) ℓ = (rows, sibs, salts))
    (hmc : merkleCheck ms cap' (sch.levels.getD ℓ default).cap (sch.levels.getD ℓ default).d
      (Flock.positions (sch.levels.getD ℓ default) cw) rows sibs salts (2 ^ (sch.levels.getD ℓ default).lanes) = .ok ()) :
    ReqOKZ ms sch cap p ⟨ℓ', Model.positions execArith (levelOf (sch.levels.getD ℓ default)).queries
      (levelOf (sch.levels.getD ℓ default)).logLen (vecOf (levelOf (sch.levels.getD ℓ default)).queries cw), ch.length,
      pathLen (levelOf (sch.levels.getD ℓ default))⟩ := by
  subst ℓ'
  refine ⟨fun h => absurd h hℓ0, fun _ => ⟨hℓ, hch, rfl, ?_⟩⟩
  dsimp only
  rw [hcap, hopen, hch, show Model.positions execArith (levelOf (sch.levels.getD ℓ default)).queries
      (levelOf (sch.levels.getD ℓ default)).logLen (vecOf (levelOf (sch.levels.getD ℓ default)).queries cw) =
      (Flock.positions (sch.levels.getD ℓ default) cw).toList from Refine.positions_eq _ hl cw, Array.toArray_toList]
  exact hmc

/-- **The proof's openings pass `OpensOKZ`** (§14): for `hm96-sha512/v1`, a rep whose requests passed the executable's
Merkle checks (`ReqOKZ`) and whose caps are the proof's verifies every requested row, level 0's `2^lanes + EXTRA_LANES`
wide, at openings and salts that are the proof's rows, siblings and salts. -/
theorem opensOKZ_of (sch : Flock.Schedule) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (h0 : 0 < sch.levels.size) (cap : Array ByteArray) (p : LigeritoProof) (r : ℕ) (pend : PendL 𝔽 Dig)
    (o : Opens 𝔽 Dig) (sl : Salts (List UInt8))
    (hcaps : pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!))
    (hreqs : ∀ req ∈ pend.reqs, ReqOKZ MerkleScheme.hm96Sha512 sch cap p req)
    (ho : ∀ ℓ j, o r ℓ j = openRow p sch ℓ j) (hs : ∀ ℓ j, sl r ℓ j = saltRow p sch ℓ j) :
    ZkLink.OpensOKZ H512 enc512 leaf512 (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) (capDig cap) r pend
      o sl := by
  intro req hreq
  obtain ⟨hz, hnz⟩ := hreqs req hreq
  by_cases hl0 : req.level = 0
  · obtain ⟨hdepth, hmc⟩ := hz hl0
    refine ⟨capDig cap, by simp [capOf, hl0], fun j hj => ?_⟩
    have hj' : j < req.positions.toArray.size := by simpa using hj
    obtain ⟨hsz, hver⟩ := merkleCheck_verifiesL hmc j hj'
    have hd := (hlv 0 h0).2.2.1
    have hpos : req.positions.getD j 0 = req.positions.toArray[j]! := by
      rw [List.getD_eq_getElem _ _ hj, getElem!_pos _ j hj', List.getElem_toArray]
    have hrow : (p.initialProof.rows[j]!.toList.map fOf) = p.initialProof.rows[j]!.toList :=
      (List.map_congr_left fun _ _ => rfl).trans (List.map_id _)
    rw [ho, hs, ite_eq_left hl0, hl0]
    refine ⟨?_, ?_⟩
    · show (p.initialProof.rows[j]!.toList.map fOf).length = _
      rw [List.length_map, Array.length_toList, hsz]
    · show Merkle.VerifiesL H512 enc512 leaf512 _ _ _ (p.initialProof.rows[j]!.toList.map fOf) _ _
      rw [hrow, hdepth, hpos, ← hd]
      exact hver
  · obtain ⟨hlev, hlanes, hdepth, hmc⟩ := hnz hl0
    refine ⟨capDig (capAt cap p req.level), ?_, fun j hj => ?_⟩
    · unfold capOf capAt
      rw [ite_eq_right hl0, ite_eq_right hl0, hcaps, List.getElem?_map, List.getElem?_range (by omega)]
      rfl
    · have hj' : j < req.positions.toArray.size := by simpa using hj
      obtain ⟨hsz, hver⟩ := merkleCheck_verifiesL hmc j hj'
      have hd := (hlv req.level hlev).2.2.1
      rw [ho, hs, ite_eq_right hl0]
      have hpos : req.positions.getD j 0 = req.positions.toArray[j]! := by
        rw [List.getD_eq_getElem _ _ hj, getElem!_pos _ j hj', List.getElem_toArray]
      have hrow : (((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList.map fOf) =
          ((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList :=
        (List.map_congr_left fun _ _ => rfl).trans (List.map_id _)
      refine ⟨?_, ?_⟩
      · show (((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList.map fOf).length = _
        rw [List.length_map, Array.length_toList, hsz]
      · show Merkle.VerifiesL H512 enc512 leaf512 _ _ _
          (((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList.map fOf) _ _
        rw [hrow, hdepth, hpos, ← hd]
        exact hver

/-! ## The model's phases, with their requests -/

theorem foldRoundsC_len (p : LigeritoProof) :
    ∀ (k a : ℕ) (sM : Opens 𝔽 D → 𝕂) (qM : Opens 𝔽 D → Model.Quad 𝕂) (cs : List (Array F128)),
      cs.length = k → (∀ c ∈ cs, c.size = 2) →
      ∃ out, Plays fOf (Model.foldRoundsC execArith k sM qM) (kMsgs p a k) cs out ∧ out.1.length = k
  | 0, a, sM, qM, cs, hl, _ => by
      rw [List.eq_nil_of_length_eq_zero hl, Model.foldRoundsC, show kMsgs p a 0 = [] by simp [kMsgs]]
      exact ⟨_, Plays.pure _, rfl⟩
  | k + 1, a, sM, qM, c :: cs, hl, hs => by
      rw [Model.foldRoundsC, show kMsgs p a (k + 1) = kMsg p.messages[a]! :: kMsgs p (a + 1) k by
        simp [kMsgs, List.range'_succ]]
      have hk : Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := by
        unfold Model.drawK
        exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c (hs c (by simp)) (by omega)) (Plays.pure _)
          (by simp) (by simp)
      obtain ⟨⟨rs, s'', q'⟩, hrest, hlen⟩ := foldRoundsC_len p k (a + 1)
        (fun o => (qM o).eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)))
        (fun o => Model.fromMsg (gOf p.messages[a]!.1, gOf p.messages[a]!.2)
          ((qM o).eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)))) cs (by simpa using hl)
        fun c' h => hs c' (by simp [h])
      refine ⟨_, Plays.bind' hk (Plays.bind' (Plays.recv _ (gOf p.messages[a]!.1, gOf p.messages[a]!.2))
        (Plays.bind' hrest (Plays.pure _) (List.append_nil _).symm (List.append_nil _).symm) rfl rfl) (by simp [kMsg])
        (by simp), ?_⟩
      simp [hlen]

theorem queryAndBatchPadC_req (r : ℕ) (lv : Accounting.Level) (col : List 𝔽 → ℕ → 𝕂) (start : ℕ)
    (st : LigStateC 𝔽 𝕂 D) (cw cα cβ : Array F128) (hcw : cw.size = lv.queries)
    (hcα : cα.size = Accounting.ceilLog2 lv.queries) (hcβ : cβ.size = 1) (hq1 : lv.queries ≠ 1)
    (hα1 : Accounting.ceilLog2 lv.queries ≠ 1) (m : F256 × F256) :
    ∃ st', Plays fOf (ZkLink.queryAndBatchPadC execArith r lv col start st) [kMsg m] [cw, cα, cβ] st' ∧
      st'.reqs = st.reqs ++ [⟨st.level, Model.positions execArith lv.queries lv.logLen (vecOf lv.queries cw), lv.lanes,
        pathLen lv⟩] := by
  unfold ZkLink.queryAndBatchPadC
  exact ⟨_, Plays.bind' (Plays.drawVec cw hcw hq1) (Plays.bind' (Plays.drawVec cα hcα hα1)
    (Plays.bind' (Plays.recv _ (gOf m.1, gOf m.2)) (Plays.bind' (Plays.drawOne cβ hcβ) (Plays.pure _) rfl rfl)
      rfl rfl) rfl rfl) rfl rfl, rfl⟩

theorem ligTailC_reqs (r : ℕ) (sch : Accounting.Schedule) (tp : ℕ) (c₀ : 𝔽) (e : ℕ) (b : ℕ → 𝔽)
    (lv₀ lv₁ : Accounting.Level) (rest : List Accounting.Level) (z₀ : Fin (sch.m - 7) → 𝔽) (β₀ : 𝔽) (ini : List 𝕂)
    (ρ : Fin e → 𝕂) (ybar : Fin tp → 𝕂) (s : 𝕂) (m' : 𝕂 × 𝕂) (C₁ : Cap D) (req₀ : Req) {Q : PendL 𝔽 D → Prop}
    {msO msQ msR : List Msg} {csO csQ csR : List (Array F128)}
    (htwo : ∀ s q, ∃ out, Plays fOf (Model.twoOodC (D := D) execArith (sch.m - 7 - sch.k0 + 1) 0 s q) msO csO out)
    (hqb : ∀ col (st : LigStateC 𝔽 𝕂 D), st.level = 0 → st.reqs = [] →
      ∃ st', Plays fOf (ZkLink.queryAndBatchPadC (D := D) execArith r lv₀ col 0 st) msQ csQ st' ∧ st'.reqs = [req₀])
    (hrec : ∀ st : LigStateC 𝔽 𝕂 D, st.reqs = [req₀] → st.level = 1 → st.caps = [C₁] →
      ∃ pend, Plays fOf (Model.recursiveRoundsC (D := D) execArith r (sch.m - 7) b ini sch.ext
        (lv₁ :: rest) st) msR csR pend ∧ Q pend) :
    ∃ pend, Plays fOf (ZkLink.ligTailC (D := D) execArith r sch tp c₀ e b lv₀ (lv₁ :: rest) z₀ β₀ ini ρ ybar s)
      (⟨𝕂 × 𝕂, m'⟩ :: ⟨Cap D, C₁⟩ :: (msO ++ (msQ ++ msR))) (csO ++ (csQ ++ csR)) pend ∧ Q pend := by
  unfold ZkLink.ligTailC
  refine exQ_recv m' ?_
  dsimp only
  refine exQ_recv C₁ ?_
  refine exQ_bind (htwo _ _) (fun ⟨s', q', oods⟩ => ?_) rfl rfl
  dsimp only
  exact pq_bindQ (hqb _ _ rfl rfl) (fun st' hst' => hrec _ hst' rfl rfl) rfl rfl

theorem ligeritoPadZC_reqs {np : ℕ} (P : ZkSession.Pads0 np 2) (sch : Accounting.Schedule) (tp : ℕ) (c₀ : 𝔽)
    (r : ℕ) (Tm : 𝔽) (b : ℕ → 𝔽) {lv₀ lv₁ : Accounting.Level} {rest : List Accounting.Level}
    (hl : sch.levels = lv₀ :: lv₁ :: rest) (z₀ cβ₀ : Array F128) (hz₀ : z₀.size = sch.m - 7) (hm7 : sch.m - 7 ≠ 1)
    (hβ₀ : cβ₀.size = 1) (y₀ : 𝔽) (m₀ : 𝕂 × 𝕂) {msF : List Msg} {csF : List (Array F128)}
    (hfold : ∀ s q, ∃ out, Plays fOf (ZkSession.foldRoundsZK execArith P sch.k0 0 s q) msF csF out)
    (ev : Fin 2 → 𝕂) {csρ : List (Array F128)} (hρ : ∃ ρ, Plays fOf (Model.drawKs execArith 2) [] csρ ρ) (T' : 𝕂)
    (ybar : Fin tp → 𝕂) (m' : 𝕂 × 𝕂) (C₁ : Cap D) (req₀ : Req) {Q : PendL 𝔽 D → Prop}
    {msO msQ msR : List Msg} {csO csQ csR : List (Array F128)}
    (htwo : ∀ s q, ∃ out, Plays fOf (Model.twoOodC (D := D) execArith (sch.m - 7 - sch.k0 + 1) 0 s q) msO csO out)
    (hqb : ∀ col (st : LigStateC 𝔽 𝕂 D), st.level = 0 → st.reqs = [] →
      ∃ st', Plays fOf (ZkLink.queryAndBatchPadC (D := D) execArith r lv₀ col 0 st) msQ csQ st' ∧ st'.reqs = [req₀])
    (hrec : ∀ ini (st : LigStateC 𝔽 𝕂 D), st.reqs = [req₀] → st.level = 1 → st.caps = [C₁] →
      ∃ pend, Plays fOf (Model.recursiveRoundsC (D := D) execArith r (sch.m - 7) b ini sch.ext
        (lv₁ :: rest) st) msR csR pend ∧ Q pend) :
    ∃ out, Plays fOf (ZkLink.ligeritoPadZC (D := D) execArith P sch tp c₀ r Tm b)
      (⟨𝔽, y₀⟩ :: ⟨𝕂 × 𝕂, m₀⟩ :: (msF ++ ⟨Fin 2 → 𝕂, ev⟩ :: ⟨𝕂, T'⟩ :: ⟨Fin tp → 𝕂, ybar⟩ :: ⟨𝕂 × 𝕂, m'⟩ ::
        ⟨Cap D, C₁⟩ :: (msO ++ (msQ ++ msR))))
      (z₀ :: cβ₀ :: (csF ++ (csρ ++ (csO ++ (csQ ++ csR))))) out ∧ Q out.2 := by
  unfold ZkLink.ligeritoPadZC
  rw [hl]
  dsimp only
  refine exQ_drawVec z₀ hz₀ hm7 fun z => exQ_recv y₀ (exQ_drawOne cβ₀ hβ₀ fun β => ?_)
  refine exQ_recv m₀ (exQ_bind (hfold _ _) (fun ⟨ini, s⟩ => ?_) rfl rfl)
  dsimp only
  refine exQ_recv ev (exQ_bind0 hρ (fun ρ => ?_) rfl)
  refine exQ_recv T' (exQ_recv ybar ?_)
  exact exQ_ret (ligTailC_reqs r sch tp c₀ 2 b lv₀ lv₁ rest z β ini ρ ybar T' m' C₁ req₀ htwo hqb (hrec ini))
    fun _ h => h

/-! ## The executable's level-0 Merkle check -/

/-- Level 0's queries pass the Merkle check on the proof's level-0 rows at the query coins' positions. -/
theorem queryLevel0_mc {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level} {cap : Array ByteArray}
    {fold : Array F256} (hq : lv.q ≠ 0)
    {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Flock.Transcript} {b : β}
    (h : (Zk.queryLevel0 ms p lv cap fold >>= f) st t = .ok ((b, st'), t')) :
    merkleCheck ms cap lv.cap lv.d (positions lv (t.rounds[t.k]!).coins) p.initialProof.rows p.initialProof.siblings
      p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) = .ok () := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold Zk.queryLevel0 at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, _, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, _, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨t₃', k₃, h₃'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₃)
  obtain ⟨rfl, rfl⟩ := pure_ok h₃'
  obtain ⟨coins, t₄, h₄, h₁⟩ := lift_bind h₁
  obtain ⟨-, hc₄, -, -⟩ := sampleVec_ok hq h₄
  subst hc₄
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  obtain ⟨t₅', k₅, h₅'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₅)
  obtain ⟨rfl, rfl⟩ := pure_ok h₅'
  obtain ⟨alpha, t₆, h₆, h₁⟩ := lift_bind h₁
  cases hmc : merkleCheck ms cap lv.cap lv.d (positions lv (t₃.rounds[t₃.k]!).coins) p.initialProof.rows
      p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    rw [← k₃.2, ← k₃.1]
    exact hmc

/-! ## Ligerito under `--zk`, with its requests -/

set_option maxHeartbeats 4000000 in
/-- **The hiding Ligerito's run records requests that passed** (`zk_ligerito_plays` with its requests): at the
proof's caps, every request the run records is one whose Merkle check the executable passed (`ReqOKZ`). -/
theorem zk_ligerito_reqs (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0) (t t' : Flock.Transcript)
    (rec : Zk.L0Rec) (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1) (hk0 : 1 ≤ sch.k0)
    (hwf : RoundsWF t.rounds) (h : Zk.ligerito ms sch T cap p basisAt zl t = .ok (rec, t')) :
    ∀ {np : ℕ} (P : ZkSession.Pads0 np 2) (c₀ : 𝔽) (r : ℕ) (Tm : 𝔽) (b : ℕ → 𝔽),
      ∃ out : ZkSession.LinK 𝕂 np × PendL 𝔽 Dig,
        Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P (schedOf sch) zl.tPad c₀ r Tm b)
          (ligMsgsZ p capDig sch zl.tPad zl.msgs) (consumed t.rounds t.k t'.k) out ∧
        out.2.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
        ∀ req ∈ out.2.reqs, ReqOKZ ms sch cap p req := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hn3 : 3 ≤ sch.n := by omega
  unfold Zk.ligerito at h
  obtain ⟨hcap, h⟩ := check_bind h
  obtain ⟨hzl, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₃', k₃', h⟩ := label_bind h
  obtain ⟨⟨rc', stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  obtain ⟨z, t₅, h₅, h₄⟩ := lift_bind h₄
  obtain ⟨hn₅, rfl, hr₅, hk₅⟩ := sampleVec_ok (by omega) h₅
  obtain ⟨y₀, -, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨bc, t₈, h₈, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, -, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀, hn₁₀⟩ := lane_coins _ sch.k0 (by
      intro i v st tE bv st' tE' hiB hb
      obtain ⟨r, tR, hR', hb⟩ := lift_bind hb
      obtain ⟨-, hnR, -, hrR, hkR⟩ := sampleK_ok hR'
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
        obtain ⟨f₂k, f₂r⟩ := absorbK_ok h₂
        obtain ⟨_, t₃, h₃, hb⟩ := lift_bind hb
        obtain ⟨f₃k, f₃r⟩ := absorbK_ok h₃
        obtain ⟨ρ₀, t₄, h₄, hb⟩ := lift_bind hb
        obtain ⟨-, hn₄, -, hr₄, hk₄⟩ := sampleK_ok h₄
        obtain ⟨ρ₁, t₅, h₅, hb⟩ := lift_bind hb
        obtain ⟨-, hn₅, -, hr₅, hk₅⟩ := sampleK_ok h₅
        try dsimp only at hb
        by_cases hρ : (!Zk.rhoIndependent (ρ₀, ρ₁)) = true
        · rw [ite_eq_left hρ] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hρ] at hb
        try simp only [pure_bind] at hb
        obtain ⟨_, t₆, h₆, hb⟩ := lift_bind hb
        obtain ⟨f₆k, f₆r⟩ := label_ok h₆
        obtain ⟨_, t₇, h₇, hb⟩ := lift_bind hb
        obtain ⟨f₇k, f₇r⟩ := absorbK_ok h₇
        obtain ⟨_, t₈, h₈, hb⟩ := lift_bind hb
        obtain ⟨f₈k, f₈r⟩ := absorbSlice_ok h₈
        have hb := lmodify_bind hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        refine ⟨_, rfl, by rw [f₉r, f₈r, f₇r, f₆r, hr₅, hr₄, f₃r, f₂r, hrR], by rw [ite_eq_left hj]; omega, hnR,
          fun _ => ⟨?_, ?_⟩⟩
        · rw [show tE.k + 1 = t₃.k by omega, show tE.rounds = t₃.rounds by rw [f₃r, f₂r, hrR]]
          exact hn₄
        · rw [show tE.k + 2 = t₄.k by omega, show tE.rounds = t₄.rounds by rw [hr₄, f₃r, f₂r, hrR]]
          exact hn₅
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        exact ⟨_, rfl, by rw [f₉r, hrR], by rw [ite_eq_right hj]; omega, hnR, fun h => absurd h hj⟩)
    sch.k0 0 _ _ t₉ _ _ t₁₀ (by omega) (by omega) h₁₀
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨hnA, hnB, yA, mA, t₁₂, -, -, hr₁₂, hk₁₂, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  obtain ⟨hnC, hnD, yB, mB, t₁₃, -, -, hr₁₃, hk₁₃, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  have hlv0 := hlv 0 (by omega)
  have hmc0 := queryLevel0_mc hlv0.q_ne_zero h₄
  obtain ⟨hnQ, hnQ', ⟨pos0, al0, e0⟩, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevel0_bind hlv0.q_ne_zero h₄
  dsimp only at h₄
  obtain ⟨extra, stX, tX, hX, h₄'⟩ := lbind_ok h₄
  clear h₄
  rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hX
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hX
  obtain ⟨mq, t₁₅, -, k₁₅, h₄⟩ := nextMsg_bind h₄'
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨bq, t₁₇, h₁₇, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₁₇, rfl, hr₁₇, hk₁₇⟩ := sample_ok h₁₇
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have hfin := finalProgZ_ok hcont
  rw [hfin]
  have er₃ : t₃'.rounds = t.rounds := by rw [k₃'.2, k₃.2, k₂.2, k₁.2]
  have ek₃ : t₃'.k = t.k := by rw [k₃'.1, k₃.1, k₂.1, k₁.1]
  have er₇ : t₇.rounds = t.rounds := by rw [k₇.2, k₆.2, hr₅, er₃]
  have ek₇ : t₇.k = t.k + 1 := by rw [k₇.1, k₆.1, hk₅, ek₃]
  have er₉ : t₉.rounds = t.rounds := by rw [k₉.2, hr₈, er₇]
  have ek₉ : t₉.k = t.k + 2 := by rw [k₉.1, hk₈, ek₇]
  have er₁₁ : t₁₁.rounds = t.rounds := by rw [k₁₁.2, hr₁₀, er₉]
  have ek₁₁ : t₁₁.k = t.k + 2 + sch.k0 + 2 := by rw [k₁₁.1, hk₁₀, ek₉]
  have er₁₂ : t₁₂.rounds = t.rounds := hr₁₂.trans er₁₁
  have ek₁₂ : t₁₂.k = t.k + 2 + sch.k0 + 4 := by rw [hk₁₂, ek₁₁]
  have er₁₃ : t₁₃.rounds = t.rounds := hr₁₃.trans er₁₂
  have ek₁₃ : t₁₃.k = t.k + 2 + sch.k0 + 6 := by rw [hk₁₃, ek₁₂]
  have er₁₆ : t₁₆.rounds = t.rounds := by rw [k₁₆.2, k₁₅.2, hr₁₄, er₁₃]
  have ek₁₆ : t₁₆.k = t.k + 2 + sch.k0 + 8 := by rw [k₁₆.1, k₁₅.1, hk₁₄, ek₁₃]
  have er₁₇ : t₁₇.rounds = t.rounds := hr₁₇.trans er₁₆
  have ek₁₇ : t₁₇.k = t.k + 2 + sch.k0 + 9 := by rw [hk₁₇, ek₁₆]
  have hd1 : (sch.levels.toList.map levelOf).drop 1 =
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := drop_levels sch 1 (by omega)
  have hL : ∀ (r : ℕ) (b : ℕ → 𝔽) (ini : List 𝕂) (stM : LigStateC 𝔽 𝕂 Dig), stM.level = 1 →
      stM.caps = [capDig p.recursiveCaps[0]!] → (∀ req ∈ stM.reqs, ReqOKZ ms sch cap p req) →
      ∃ pend : PendL 𝔽 Dig, t₁₈.rounds = t₁₇.rounds ∧ t₁₇.k ≤ t₁₈.k ∧
      Plays fOf (Model.recursiveRoundsC execArith r (sch.m - 7) b ini sch.ext
        (levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2) stM)
        (recMsgs p capDig sch.ext (sch.levels.size - 1) 0 (sch.k0 + 4) 3) (consumed t₁₇.rounds t₁₇.k t₁₈.k) pend ∧
      pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
      ∀ req ∈ pend.reqs, ReqOKZ ms sch cap p req := by
    intro r b ini stM hlev1 hcaps1 hreqs1
    have hLC := loop_compose _ (sch.levels.size - 1)
      (fun i stM => Model.recursiveRoundsC execArith r (sch.m - 7) b ini sch.ext
        ((sch.levels.toList.map levelOf).drop (i + 1)) stM)
      (fun i v _ tE stM => RoundsWF tE.rounds ∧ v.1 = (sch.levels.getD (i + 1) default).cols + 4 ∧
        v.2.1 = p.recursiveCaps[i]! ∧ stM.level = i + 1 ∧
        stM.caps = (List.range (i + 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
        ∀ req ∈ stM.reqs, ReqOKZ ms sch cap p req)
      (fun i _ => recMsgs p capDig sch.ext (sch.levels.size - 1 - i) i (sch.k0 + 4 + 7 * i) (3 + 2 * i))
      (fun _ _ _ pend => pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
        ∀ req ∈ pend.reqs, ReqOKZ ms sch cap p req) ?hstep ?hlast (sch.levels.size - 1) 0 _ _ _ out st₁₈ t₁₈ stM
      (by omega) (by omega)
      ⟨by rw [er₁₇]; exact hwf, by show sch.n - sch.k0 + 1 = _; omega, rfl, hlev1, by rw [hcaps1]; rfl, hreqs1⟩
      hloop
    case hstep =>
      clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1
      intro i v stE tE bv stE' tE' stM hi hrel hb
      obtain ⟨hwfE, hsplit, hprev, hlevM, hcapsM, hreqsM⟩ := hrel
      have hc1 := hcols (i + 1) (by omega)
      have hl1 := hl4 (i + 1) (by omega) (by omega)
      have hl2 := hl4 (i + 1 + 1) (by omega) (by omega)
      try dsimp only at hb
      rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      obtain ⟨hr₁, hk₁, hn₁⟩ := fold_coins p 4 0 #[] chs stE st₁ tE t₁ hf
      rw [ite_eq_left hi] at hb
      obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
      have k₂ := absorbBytes_ok h₂
      obtain ⟨hnA, hnB, yA, mA, t₃, -, -, hr₃, hk₃, hb⟩ := oodIntro_bind (by omega) hb
      obtain ⟨hnC, hnD, yB, mB, t₄, -, -, hr₄, hk₄, hb⟩ := oodIntro_bind (by omega) hb
      have hlvi := hlv (i + 1) (by omega)
      obtain ⟨hnQ, hnQ', hmcQ, t₅, hr₅, hk₅, hb⟩ := queryLevel_bind hlvi.q_ne_zero hb
      obtain ⟨mq, t₆, -, k₆, hb⟩ := nextMsg_bind hb
      obtain ⟨t₇, k₇, hb⟩ := claimPow_bind hb
      obtain ⟨bq, t₈, h₈, hb⟩ := lift_bind hb
      obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
      have hb := lmodify_bind hb
      rw [ite_eq_right (show ¬ v.1 - 4 + 1 < 4 by omega)] at hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
      have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
      have er₃ : t₃.rounds = tE.rounds := hr₃.trans er₂
      have ek₃ : t₃.k = tE.k + 6 := by rw [hk₃, ek₂]
      have er₄ : t₄.rounds = tE.rounds := hr₄.trans er₃
      have ek₄ : t₄.k = tE.k + 8 := by rw [hk₄, ek₃]
      have er₇ : t₇.rounds = tE.rounds := by rw [k₇.2, k₆.2, hr₅, er₄]
      have ek₇ : t₇.k = tE.k + 10 := by rw [k₇.1, k₆.1, hk₅, ek₄]
      have er₈ : tE'.rounds = tE.rounds := hr₈.trans er₇
      have ek₈ : tE'.k = tE.k + 11 := by rw [hk₈, ek₇]
      have hdim : (levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
          (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes = v.1 - 4 + 1 := by
        show (sch.levels.getD (i + 1 + 1) default).cols + (sch.levels.getD (i + 1 + 1) default).lanes = _
        omega
      obtain ⟨⟨rs, sM', qM'⟩, hfoldM, hrsl⟩ := foldRoundsC_len (D := Dig) p 4 (sch.k0 + 4 + 7 * i) stM.s stM.q
        (consumed tE.rounds tE.k (tE.k + 4)) (by rw [consumed_length]; omega) (consumed_sizes hwfE hn₁)
      obtain ⟨⟨sT, qT, ctxs⟩, htwo⟩ := twoOodC_any (D := Dig)
        ((levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
          (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes) (stM.levels ++ [rs]).length sM' qM'
        (tE.rounds[tE.k + 4]!).coins (tE.rounds[tE.k + 5]!).coins (tE.rounds[tE.k + 6]!).coins
        (tE.rounds[tE.k + 7]!).coins
        (by rw [RoundsWF.get hwfE, hdim, show tE.k + 4 = t₂.k by omega, ← er₂]; exact hnA)
        (by rw [RoundsWF.get hwfE, show tE.k + 5 = t₂.k + 1 by omega, ← er₂]; exact hnB)
        (by rw [RoundsWF.get hwfE, hdim, show tE.k + 6 = t₃.k by omega, ← er₃]; exact hnC)
        (by rw [RoundsWF.get hwfE, show tE.k + 7 = t₃.k + 1 by omega, ← er₃]; exact hnD) (by omega)
        p.oodValues[3 + 2 * i]! p.oodValues[3 + 2 * i + 1]! p.messages[sch.k0 + 4 + 7 * i + 4]!
        p.messages[sch.k0 + 4 + 7 * i + 5]!
      have hcw : (tE.rounds[tE.k + 8]!).coins.size = (levelOf (sch.levels.getD (i + 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 8 = t₄.k by omega, ← er₄, hnQ]; rfl
      have hcα : (tE.rounds[tE.k + 9]!).coins.size =
          Accounting.ceilLog2 (levelOf (sch.levels.getD (i + 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 9 = t₄.k + 1 by omega, ← er₄, hnQ']; exact hlvi.2.2.2.1
      have hcβ : (tE.rounds[tE.k + 10]!).coins.size = 1 := by
        rw [RoundsWF.get hwfE, show tE.k + 10 = t₇.k by omega, ← er₇, hn₈]
      have hmcE : merkleCheck ms v.2.1 (sch.levels.getD (i + 1) default).cap (sch.levels.getD (i + 1) default).d
          (Flock.positions (sch.levels.getD (i + 1) default) (tE.rounds[tE.k + 8]!).coins)
          (p.recursiveProofs[i]!).rows (p.recursiveProofs[i]!).siblings (p.recursiveProofs[i]!).salts
          (2 ^ (sch.levels.getD (i + 1) default).lanes) = .ok () := by
        rw [show tE.k + 8 = t₄.k by omega, ← er₄]; exact hmcQ
      refine ⟨_, { qbOut r (levelOf (sch.levels.getD (i + 1) default)) rs (stM.levels ++ [rs]).length
            { stM with s := sT, q := qT, levels := stM.levels ++ [rs], oods := stM.oods ++ ctxs }
            (tE.rounds[tE.k + 8]!).coins (tE.rounds[tE.k + 9]!).coins (tE.rounds[tE.k + 10]!).coins true
            p.messages[sch.k0 + 4 + 7 * i + 6]! with
          level := stM.level + 1, caps := stM.caps ++ [capDig p.recursiveCaps[i + 1]!],
          prevLevel := levelOf (sch.levels.getD (i + 1 + 1) default) },
        kMsgs p (sch.k0 + 4 + 7 * i) 4 ++ ⟨Cap Dig, capDig p.recursiveCaps[i + 1]!⟩ ::
          [fMsg p.oodValues[3 + 2 * i]!, kMsg p.messages[sch.k0 + 4 + 7 * i + 4]!,
            fMsg p.oodValues[3 + 2 * i + 1]!, kMsg p.messages[sch.k0 + 4 + 7 * i + 5]!] ++
          [kMsg p.messages[sch.k0 + 4 + 7 * i + 6]!],
        rfl, ⟨by rw [er₈]; exact hwfE, by show v.1 - 4 + 1 = _; omega, rfl,
          by show stM.level + 1 = i + 1 + 1; rw [hlevM],
          by
            show stM.caps ++ [capDig p.recursiveCaps[i + 1]!] = _
            rw [hcapsM, List.range_succ (n := i + 1), List.map_append]
            rfl,
          ?reqs⟩, er₈, by omega, ?tape, ?plays⟩
      case reqs =>
        intro req hreq
        simp only [qbOut, List.mem_append, List.mem_singleton] at hreq
        rcases hreq with hreq | rfl
        · exact hreqsM req hreq
        · exact reqOKZ_succ (ℓ := i + 1) hlevM (by omega) (by omega) hlvi rs (by rw [hrsl, hl1])
            (tE.rounds[tE.k + 8]!).coins (by unfold capAt; rw [ite_eq_right (by omega), hprev]; rfl)
            (by unfold openAt; rw [ite_eq_right (by omega), ite_eq_left (by omega)]; rfl) hmcE
      case tape =>
        obtain ⟨j, hj⟩ : ∃ j, sch.levels.size - 1 - i = j + 2 := ⟨sch.levels.size - 1 - i - 2, by omega⟩
        rw [hj, show sch.levels.size - 1 - (i + 1) = j + 1 by omega,
          show sch.k0 + 4 + 7 * (i + 1) = sch.k0 + 4 + 7 * i + 7 by omega,
          show 3 + 2 * (i + 1) = 3 + 2 * i + 2 by omega]
        simp only [recMsgs]
      case plays =>
        intro msR csR pend hrec
        try dsimp only at hrec ⊢
        rw [drop_levels sch (i + 1) (by omega), drop_levels sch (i + 1 + 1) (by omega)]
        rw [drop_levels sch (i + 1 + 1) (by omega)] at hrec
        have hfold' : Plays fOf (Model.foldRoundsC execArith (levelOf (sch.levels.getD (i + 1) default)).lanes
            stM.s stM.q) (kMsgs p (sch.k0 + 4 + 7 * i) 4) (consumed tE.rounds tE.k (tE.k + 4)) (rs, sM', qM') := by
          rw [show (levelOf (sch.levels.getD (i + 1) default)).lanes = 4 from hl1]; exact hfoldM
        have hmain := iterNL_model r (sch.m - 7) b ini sch.ext
          (levelOf (sch.levels.getD (i + 1) default)) (levelOf (sch.levels.getD (i + 1 + 1) default))
          ((sch.levels.toList.map levelOf).drop (i + 1 + 1 + 1)) stM hfold' (capDig p.recursiveCaps[i + 1]!) htwo
          (tE.rounds[tE.k + 8]!).coins (tE.rounds[tE.k + 9]!).coins (tE.rounds[tE.k + 10]!).coins hcw hcα hcβ
          hlvi.q_ne_one (hα1 (i + 1) (by omega)) p.messages[sch.k0 + 4 + 7 * i + 6]! hrec
        have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k (tE.k + 4) ++
            [(tE.rounds[tE.k + 4]!).coins, (tE.rounds[tE.k + 5]!).coins, (tE.rounds[tE.k + 6]!).coins,
              (tE.rounds[tE.k + 7]!).coins] ++
            [(tE.rounds[tE.k + 8]!).coins, (tE.rounds[tE.k + 9]!).coins, (tE.rounds[tE.k + 10]!).coins] := by
          rw [ek₈, consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 11 by omega),
            show tE.k + 11 = tE.k + 4 + 7 by omega, consumed_add tE.rounds (tE.k + 4) 7]
          simp [List.range_succ, Nat.add_assoc]
        rw [hcs]
        exact hmain
    case hlast =>
      clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1
      intro i v stE tE bv stE' tE' stM hi hrel hb
      obtain ⟨hwfE, hsplit, hprev, hlevM, hcapsM, hreqsM⟩ := hrel
      have hl1 := hl4 (i + 1) (by omega) (by omega)
      try dsimp only at hb
      rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      obtain ⟨hr₁, hk₁, hn₁⟩ := fold_coins p 4 0 #[] chs stE st₁ tE t₁ hf
      rw [ite_eq_right (show ¬ i + 1 < sch.levels.size - 1 by omega)] at hb
      have hb := lget_bind hb
      by_cases hmc₀ : (st₁.msg != p.messages.size) = true
      · rw [ite_eq_left hmc₀] at hb
        obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
        cases h0
      rw [ite_eq_right hmc₀] at hb
      by_cases hyr : (p.finalProof.yr.size != 2 ^ (v.1 - 4 + 1)) = true
      · rw [ite_eq_left hyr] at hb
        obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
        cases h0
      rw [ite_eq_right hyr] at hb
      obtain ⟨_, st₂, t₂, h₂, hb₂⟩ := lbind_ok hb
      clear hb
      rw [← Array.forIn_toList] at h₂
      obtain ⟨rfl, k₂⟩ := forIn_absorbF_keeps _ h₂
      have hb := hb₂
      clear hb₂
      have hlvR := hlv (sch.levels.size - 1) (by omega)
      obtain ⟨hnQ, hnQ', hmcQ, t₃, hr₃, hk₃, hb⟩ := queryLevel_bind hlvR.q_ne_zero hb
      obtain ⟨t₄, k₄, hb⟩ := claimPow_bind hb
      obtain ⟨bq, t₅, h₅, hb⟩ := lift_bind hb
      obtain ⟨-, hn₅, rfl, hr₅, hk₅⟩ := sample_ok h₅
      have hb := lmodify_bind hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
      have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
      have er₄ : t₄.rounds = tE.rounds := by rw [k₄.2, hr₃, er₂]
      have ek₄ : t₄.k = tE.k + 6 := by rw [k₄.1, hk₃, ek₂]
      have er₅ : tE'.rounds = tE.rounds := hr₅.trans er₄
      have ek₅ : tE'.k = tE.k + 7 := by rw [hk₅, ek₄]
      rw [hi] at hl1
      obtain ⟨⟨rs, sM', qM'⟩, hfoldM, hrsl⟩ := foldRoundsC_len (D := Dig) p 4 (sch.k0 + 4 + 7 * i) stM.s stM.q
        (consumed tE.rounds tE.k (tE.k + 4)) (by rw [consumed_length]; omega) (consumed_sizes hwfE hn₁)
      have hfold' : Plays fOf (Model.foldRoundsC execArith
          (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes stM.s stM.q)
          (kMsgs p (sch.k0 + 4 + 7 * i) 4) (consumed tE.rounds tE.k (tE.k + 4)) (rs, sM', qM') := by
        rw [show (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes = 4 from hl1]; exact hfoldM
      have hcw : (tE.rounds[tE.k + 4]!).coins.size =
          (levelOf (sch.levels.getD (sch.levels.size - 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 4 = t₂.k by omega, ← er₂, hnQ]; rfl
      have hcα : (tE.rounds[tE.k + 5]!).coins.size =
          Accounting.ceilLog2 (levelOf (sch.levels.getD (sch.levels.size - 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 5 = t₂.k + 1 by omega, ← er₂, hnQ']; exact hlvR.2.2.2.1
      have hcβ : (tE.rounds[tE.k + 6]!).coins.size = 1 := by
        rw [RoundsWF.get hwfE, show tE.k + 6 = t₄.k by omega, ← er₄, hn₅]
      have hmcE : merkleCheck ms v.2.1 (sch.levels.getD (sch.levels.size - 1) default).cap
          (sch.levels.getD (sch.levels.size - 1) default).d
          (Flock.positions (sch.levels.getD (sch.levels.size - 1) default) (tE.rounds[tE.k + 4]!).coins)
          p.finalProof.rows p.finalProof.siblings p.finalProof.salts
          (2 ^ (sch.levels.getD (sch.levels.size - 1) default).lanes) = .ok () := by
        rw [show tE.k + 4 = t₂.k by omega, ← er₂]; exact hmcQ
      have hmain := iterLast_model r (sch.m - 7) b ini sch.ext
        (levelOf (sch.levels.getD (sch.levels.size - 1) default)) stM hfold' (yrOf p sch.ext)
        (tE.rounds[tE.k + 4]!).coins (tE.rounds[tE.k + 5]!).coins (tE.rounds[tE.k + 6]!).coins hcw hcα hcβ
        hlvR.q_ne_one (hα1 (sch.levels.size - 1) (by omega))
      have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k (tE.k + 4) ++
          [(tE.rounds[tE.k + 4]!).coins, (tE.rounds[tE.k + 5]!).coins, (tE.rounds[tE.k + 6]!).coins] := by
        rw [ek₅, consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 7 by omega),
          show tE.k + 7 = tE.k + 4 + 3 by omega, consumed_add tE.rounds (tE.k + 4) 3]
        simp [List.range_succ, Nat.add_assoc]
      have hdrop : (sch.levels.toList.map levelOf).drop (i + 1) =
          [levelOf (sch.levels.getD (sch.levels.size - 1) default)] := by
        rw [hi, drop_levels sch (sch.levels.size - 1) (by omega),
          show sch.levels.size - 1 + 1 = sch.levels.size by omega]
        simp
      exact ⟨_, _, rfl, er₅, by omega, by
        rw [hdrop, show sch.levels.size - 1 - i = 1 by omega, hcs]
        simp only [recMsgs]
        exact hmain, by
        refine ⟨?_, fun req hreq => ?_⟩
        · show stM.caps = _
          rw [hcapsM, hi]
        · simp only [qbOut, List.mem_append, List.mem_singleton] at hreq
          rcases hreq with hreq | rfl
          · exact hreqsM req hreq
          · exact reqOKZ_succ (ℓ := sch.levels.size - 1) (hlevM.trans hi) (by omega) (by omega) hlvR rs
              (by rw [hrsl, hl1]) (tE.rounds[tE.k + 4]!).coins
              (by unfold capAt; rw [ite_eq_right (by omega), hprev, show sch.levels.size - 1 - 1 = i by omega])
              (by unfold openAt; rw [ite_eq_right (by omega), ite_eq_right (lt_irrefl _)]) hmcE⟩
    obtain ⟨pend, hr, hk, hpl, hfinal⟩ := hLC
    refine ⟨pend, hr, hk, ?_, hfinal⟩
    simp only [Nat.zero_add, Nat.sub_zero, Nat.mul_zero, Nat.add_zero] at hpl
    rw [hd1] at hpl
    exact hpl
  obtain ⟨_, hrL, hkL, -⟩ := hL 0 (fun _ => 0) []
    ⟨fun _ => 0, fun _ => (0, 0, 0), 0, [], [], [], 1, [capDig p.recursiveCaps[0]!], [], ⟨0, 0, 0, 0⟩⟩ rfl rfl
    (by simp)
  intro np P c₀ r Tm b
  have hcs : consumed t.rounds t.k t₁₈.k = (t.rounds[t.k]!).coins :: (t.rounds[t.k + 1]!).coins ::
      (consumed t.rounds (t.k + 2) (t.k + 2 + sch.k0) ++
        (consumed t.rounds (t.k + 2 + sch.k0) (t.k + 2 + sch.k0 + 2) ++
          ([(t.rounds[t.k + 2 + sch.k0 + 2]!).coins, (t.rounds[t.k + 2 + sch.k0 + 3]!).coins,
              (t.rounds[t.k + 2 + sch.k0 + 4]!).coins, (t.rounds[t.k + 2 + sch.k0 + 5]!).coins] ++
            ([(t.rounds[t.k + 2 + sch.k0 + 6]!).coins, (t.rounds[t.k + 2 + sch.k0 + 7]!).coins,
                (t.rounds[t.k + 2 + sch.k0 + 8]!).coins] ++
              consumed t₁₇.rounds t₁₇.k t₁₈.k)))) := by
    rw [er₁₇, ek₁₇, consumed_cons _ (show t.k < t₁₈.k by omega), consumed_cons _ (show t.k + 1 < t₁₈.k by omega),
      show t.k + 1 + 1 = t.k + 2 from rfl,
      consumed_append t.rounds (show t.k + 2 ≤ t.k + 2 + sch.k0 by omega)
        (show t.k + 2 + sch.k0 ≤ t₁₈.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 ≤ t.k + 2 + sch.k0 + 2 by omega)
        (show t.k + 2 + sch.k0 + 2 ≤ t₁₈.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 + 2 ≤ t.k + 2 + sch.k0 + 2 + 4 by omega)
        (show t.k + 2 + sch.k0 + 2 + 4 ≤ t₁₈.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 + 2 + 4 ≤ t.k + 2 + sch.k0 + 2 + 4 + 3 by omega)
        (show t.k + 2 + sch.k0 + 2 + 4 + 3 ≤ t₁₈.k by omega),
      consumed_add t.rounds (t.k + 2 + sch.k0 + 2) 4, consumed_add t.rounds (t.k + 2 + sch.k0 + 2 + 4) 3,
      show t.k + 2 + sch.k0 + 2 + 4 + 3 = t.k + 2 + sch.k0 + 9 by omega]
    simp [List.range_succ, Nat.add_assoc]
  have hlv01 : (schedOf sch).levels = levelOf (sch.levels.getD 0 default) ::
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := by
    have e0 := drop_levels sch 0 (by omega)
    simp only [List.drop_zero, Nat.zero_add] at e0
    rw [hd1] at e0
    exact e0
  have hz₀ : (t.rounds[t.k]!).coins.size = (schedOf sch).m - 7 := by
    rw [RoundsWF.get hwf, show t.k = t₃'.k from ek₃.symm, ← er₃, hn₅, hn7]; rfl
  have hβ₀ : (t.rounds[t.k + 1]!).coins.size = 1 := by
    rw [RoundsWF.get hwf, show t.k + 1 = t₇.k from ek₇.symm, ← er₇, hn₈]
  have hFl : (consumed t.rounds (t.k + 2) (t.k + 2 + sch.k0)).length = sch.k0 := by
    rw [consumed_length]; omega
  have hFs : ∀ c ∈ consumed t.rounds (t.k + 2) (t.k + 2 + sch.k0), c.size = 2 :=
    consumed_sizes hwf fun j hj => by
      rw [show t.k + 2 + j = t₉.k + j by omega, ← er₉]; exact hn₁₀ j (by omega)
  have hρl : (consumed t.rounds (t.k + 2 + sch.k0) (t.k + 2 + sch.k0 + 2)).length = 2 := by
    rw [consumed_length]; omega
  have hρs : ∀ c ∈ consumed t.rounds (t.k + 2 + sch.k0) (t.k + 2 + sch.k0 + 2), c.size = 2 :=
    consumed_sizes hwf fun j hj => by
      rw [show t.k + 2 + sch.k0 + j = t₉.k + (sch.k0 + j) by omega, ← er₉]; exact hn₁₀ (sch.k0 + j) (by omega)
  have hdim0 : sch.n - sch.k0 + 1 = (schedOf sch).m - 7 - (schedOf sch).k0 + 1 := by
    show sch.n - sch.k0 + 1 = sch.m - 7 - sch.k0 + 1
    rw [hn7]
  have hmc0' : merkleCheck ms cap (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
      (Flock.positions (sch.levels.getD 0 default) (t.rounds[t.k + 2 + sch.k0 + 6]!).coins) p.initialProof.rows
      p.initialProof.siblings p.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) =
        .ok () := by
    rw [show t.k + 2 + sch.k0 + 6 = t₁₃.k by omega, ← er₁₃]; exact hmc0
  have hmain := ligeritoPadZC_reqs (D := Dig) P (schedOf sch) zl.tPad c₀ r Tm b hlv01
    (t.rounds[t.k]!).coins (t.rounds[t.k + 1]!).coins hz₀ (by show sch.m - 7 ≠ 1; omega) hβ₀
    (fOf p.oodValues[0]!) (gOf p.messages[0]!.1, gOf p.messages[0]!.2)
    (fun s q => foldRoundsZK_any P p sch.k0 0 1 s q _ hFl hFs)
    ![gOf zl.msgs.e0, gOf zl.msgs.e1] (drawKs_any 2 _ hρl hρs) (gOf zl.msgs.tPrime)
    (fun i => gOf zl.msgs.ybar[i.val]!) (gOf p.messages[sch.k0]!.1, gOf p.messages[sch.k0]!.2)
    (capDig p.recursiveCaps[0]!)
    ⟨0, Model.positions execArith (levelOf (sch.levels.getD 0 default)).queries
      (levelOf (sch.levels.getD 0 default)).logLen
      (vecOf (levelOf (sch.levels.getD 0 default)).queries (t.rounds[t.k + 2 + sch.k0 + 6]!).coins),
      (levelOf (sch.levels.getD 0 default)).lanes, pathLen (levelOf (sch.levels.getD 0 default))⟩
    (Q := fun pend => pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!) ∧
      ∀ req ∈ pend.reqs, ReqOKZ ms sch cap p req)
    (fun s q => twoOodC_any _ 0 s q
      (t.rounds[t.k + 2 + sch.k0 + 2]!).coins (t.rounds[t.k + 2 + sch.k0 + 3]!).coins
      (t.rounds[t.k + 2 + sch.k0 + 4]!).coins (t.rounds[t.k + 2 + sch.k0 + 5]!).coins
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 2 = t₁₁.k by omega, ← er₁₁, hnA, hdim0])
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 3 = t₁₁.k + 1 by omega, ← er₁₁, hnB])
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 4 = t₁₂.k by omega, ← er₁₂, hnC, hdim0])
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 5 = t₁₂.k + 1 by omega, ← er₁₂, hnD])
      (by show sch.m - 7 - sch.k0 + 1 ≠ 1; omega)
      p.oodValues[1]! p.oodValues[2]! p.messages[sch.k0 + 1]! p.messages[sch.k0 + 2]!)
    (fun col st hlev hreqs => by
      obtain ⟨st', hpl, hst'⟩ := queryAndBatchPadC_req r (levelOf (sch.levels.getD 0 default)) col 0 st
        (t.rounds[t.k + 2 + sch.k0 + 6]!).coins (t.rounds[t.k + 2 + sch.k0 + 7]!).coins
        (t.rounds[t.k + 2 + sch.k0 + 8]!).coins
        (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 6 = t₁₃.k by omega, ← er₁₃, hnQ]; rfl)
        (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 7 = t₁₃.k + 1 by omega, ← er₁₃, hnQ'];
            exact hlv0.2.2.2.1)
        (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 8 = t₁₆.k by omega, ← er₁₆, hn₁₇])
        hlv0.q_ne_one (hα1 0 (by omega)) p.messages[sch.k0 + 3]!
      exact ⟨st', hpl, by rw [hst', hreqs, hlev]; rfl⟩)
    (fun ini st hreqs hlev hcaps => by
      obtain ⟨pend, _, _, hpl, hQ⟩ := hL r b ini st hlev hcaps (by
        rw [hreqs]
        intro req hreq
        rw [List.mem_singleton] at hreq
        subst hreq
        exact reqOKZ_zero rfl hlv0 _ _ hmc0')
      exact ⟨pend, hpl, hQ⟩)
  rw [hcs]
  simp only [ligMsgsZ, List.append_assoc, List.cons_append, List.nil_append] at hmain ⊢
  obtain ⟨out, hpl, hcaps, hreqs⟩ := hmain
  exact ⟨out, hpl, hcaps, hreqs⟩

/-! ## The rep, with its requests -/

/-- **`repProg`'s run records requests that passed** (`zk_rep_plays` with its requests): the run of the compiled rep
that `zk_rep_plays` builds has the proof's caps, and every request it records passed the executable's Merkle check. -/
theorem zk_rep_reqs {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (extras : Array Flock.Claim) (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript)
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam T st sch tPad)
    (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hwf : RoundsWF rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) :
    ∃ pend : ZkLink.PendZ 𝔽 Dig Sl,
      Plays fOf (ZkLink.repZCAt execArith Hs T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend ∧
      pend.lig.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig zp.proof.ligerito.recursiveCaps[j]!) ∧
      ∀ req ∈ pend.lig.reqs, ReqOKZ st.merkle sch zp.proof.cap zp.proof.ligerito req := by
  obtain ⟨h25, -, -, hk0, -, hdim, hlv, hα1, -⟩ := zk_schedule_seams _ _ _ hsch
  have hTm : T.S.m = st.m := hseam.hm
  have h7 : 7 ≤ T.S.m := by omega
  unfold repProg at hrun
  rw [StateT.run] at hrun
  try dsimp only at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  try dsimp only at hrun
  obtain ⟨⟨Tb, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄
  | ok lg =>
  rw [hv] at hrun
  try dsimp only at hrun
  obtain ⟨lg', t₄, h₄, h₅⟩ := bind_ok hrun
  clear hrun
  obtain ⟨hlg, ht₄⟩ := pure_ok h₄
  rw [hlg, ht₄] at h₅
  have hwf₀ : RoundsWF (Flock.Transcript.init rounds dom).rounds := hwf
  have hcs3 := replay_cons_size _ _ _ _ h₁
  obtain ⟨hr₁, hk₁, hRP⟩ := zk_replay_plays st _ zp.proof zp.inner.root _ t₁ rp T.S hTm.symm hseam.hk.symm
    (by omega) hwf₀ h₁
  obtain ⟨k₁, zc, row, abM, row', hk₁a, hk₁b, hpz, hpl, hc0, hc1⟩ := hRP T.padsA T.tr
  rw [← hTm] at h₂
  obtain ⟨g₀, g₁, g₂⟩ := two_append_get! ({ rp.claims[0]! with value := Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]! } : Flock.Claim)
    ({ rp.claims[1]! with value := Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]! } : Flock.Claim) extras
  obtain ⟨hr₂, hk₂, op, hop, -⟩ := opening_refines T.S.m h7 _
    (⟨fOf (Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]!), abM.skip, abM.x⟩ ::
      ⟨fOf (Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]!), execArith.lagS zc.z, zc.r⟩ ::
        Model.extraClaims T.S T.ptLocal mPts pts)
    (by rw [size_two_append]; simp; omega)
    (fun k hk => by
      rcases k with _ | _ | k
      · rw [g₀]; exact hc0 _
      · rw [g₁]; exact hc1 _
      · have hk' : k < (Model.extraClaims T.S T.ptLocal mPts pts).length := by simp at hk; omega
        rw [g₂ k (by omega)]
        exact hext k hk')
    (by rw [size_two_append]; omega) (by rw [size_two_append]; omega) _ t₁ t₂ Tb sws (hr₁ ▸ hwf₀) h₂
  rw [size_two_append] at hop
  obtain ⟨xo, hxo, hxo1⟩ := openingZK_plays h7 _ _ abM.pub ⟨0, execArith.lagS zc.z, zc.r⟩ _ hop
  obtain ⟨xo1, xo2⟩ := xo
  obtain rfl : xo1 = some op := hxo1
  have hwf₂ : RoundsWF t₂.rounds := hr₂ ▸ hr₁ ▸ hwf₀
  obtain ⟨hr₃, hk₃, -⟩ := zk_ligerito_plays st.merkle sch Tb zp.proof.cap zp.proof.ligerito _ ⟨tPad, r.val, zp.level0⟩
    t₂ t₃ rc hdim hlv hα1 hk0 hwf₂ h₃
  have hlig : ∀ {e : ℕ} (P : ZkSession.Pads0 T.np e), e = 2 → ∀ (Tm : 𝔽) (b : ℕ → 𝔽),
      ∃ out : ZkSession.LinK 𝕂 T.np × PendL 𝔽 Dig,
        Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P T.sch T.t T.c₀ r.val Tm b)
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0) (consumed t₂.rounds t₂.k t₃.k) out ∧
        (out.2.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig zp.proof.ligerito.recursiveCaps[j]!) ∧
          ∀ req ∈ out.2.reqs, ReqOKZ st.merkle sch zp.proof.cap zp.proof.ligerito req) := by
    intro e P he Tm b
    subst he
    rw [hseam.hsch, hseam.ht]
    exact zk_ligerito_reqs st.merkle sch Tb zp.proof.cap zp.proof.ligerito _ ⟨tPad, r.val, zp.level0⟩
      t₂ t₃ rc hdim hlv hα1 hk0 hwf₂ h₃ P T.c₀ r.val Tm b
  have hwf₃ : RoundsWF t₃.rounds := hr₃ ▸ hwf₂
  obtain ⟨hr₅, hk₅, hIP⟩ := zk_inner_plays _ zp.inner zp.proof.zerocheck.finalC _ t₃ tr
    (by rw [Array.size_append, Array.size_append, hcs3, ligeritoConstraints_size hv]; rfl) hwf₃ h₅
  have hM : ∃ o, Plays fOf (ZkLink.repMaskedZC (D := Dig) execArith T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r.val
        (Model.extraClaims T.S T.ptLocal mPts pts))
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC]))))
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) o ∧
      (o.2.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig zp.proof.ligerito.recursiveCaps[j]!) ∧
        ∀ req ∈ o.2.reqs, ReqOKZ st.merkle sch zp.proof.cap zp.proof.ligerito req) := by
    unfold ZkLink.repMaskedZC
    refine pq_map ?_
    refine pq_bind hpz ?_ rfl rfl
    refine pq_map ?_
    refine pq_bind hpl ?_ rfl rfl
    refine pq_map ?_
    unfold ZkLink.restZC
    refine pq_bind hxo ?_ rfl rfl
    obtain ⟨out, hout, hne⟩ := hlig T.pads hseam.he op.T op.b
    refine pq_bind hout ?_ rfl (List.append_nil _).symm
    exact ⟨_, Plays.bind' (Plays.recv _ (fOf zp.proof.zerocheck.finalC)) (Plays.pure _) rfl rfl, hne⟩
  have hmsgs : restMsgsZ st sch (2 + extras.size) tPad zp =
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC])))) ++
      [⟨𝔽 × 𝔽, (fOf zp.inner.rho, fOf zp.inner.sigma)⟩, ⟨Dig, toDig zp.inner.tauCommit⟩,
        ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) zp.inner.y⟩] := by
    rw [hTm, hseam.hk, hseam.hnp, hseam.hnP]
    simp [restMsgsZ, innerMsgs]
  have hcoins : rounds.toList.map (·.coins) =
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) ++ consumed t₃.rounds t₃.k tr.k := by
    rw [hr₃, hr₂, hr₁]
    simp only [List.append_assoc]
    rw [← consumed_append _ hk₃ hk₅, ← consumed_append _ hk₂ (hk₃.trans hk₅),
      ← consumed_append _ hk₁b (hk₂.trans (hk₃.trans hk₅)),
      ← consumed_append _ hk₁a (hk₁b.trans (hk₂.trans (hk₃.trans hk₅))), hk]
    exact (consumed_all _).symm
  unfold ZkLink.repZCAt ZkLink.repZC
  rw [hseam.hq]
  refine exQ_recv (toDig zp.inner.root) ?_
  refine pq_bindQ hM (fun o ho => ?_) hmsgs hcoins
  obtain ⟨v, hv⟩ := hIP Hs T.dom T.np T.nP T.pos o.1 T.tr (toDig zp.inner.root)
  exact ⟨⟨o.2, v⟩, Plays.bind' hv (Plays.pure _) (List.append_nil _).symm (List.append_nil _).symm, ho⟩

/-! ## `ZkRepAccepts`'s openings -/

/-- **A `--zk` rep's requested rows verify** (PROTOCOL.md §14; `ZkRepAccepts`'s third conclusion). At the `--zk`
schedule under `hm96-sha512/v1`, at a compiled table at the executable's dimensions whose extra claims are the
executable's, when a run of rep `r`'s transcript program consumes every round, every result of the compiled rep, played
with the rep's messages and the rounds' coins, verifies every row it requested (`OpensOKZ`, level 0's rows `width₀ T`
wide), at the level-0 cap the proof's cap reads as and openings whose rep-`r` rows and salts are the proof's. -/
theorem zk_opens_ok {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (zp : Zk.ZkProof) (rounds : Array Round)
    (dom : ByteArray) (tr : Flock.Transcript) (cap₀ : Cap Dig) (os : ZkLink.OpensZ 𝔽 Dig (List UInt8))
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hms : st.merkle = MerkleScheme.hm96Sha512)
    (hseam : ZkTabSeam T st sch tPad)
    (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hwf : RoundsWF rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) (hcap : cap₀ = capDig zp.proof.cap)
    (ho : ∀ ℓ j, os.rows r.val ℓ j = openRow zp.proof.ligerito sch ℓ j)
    (hs : ∀ ℓ j, os.salts r.val ℓ j = saltRow zp.proof.ligerito sch ℓ j)
    (pend : ZkLink.PendZ 𝔽 Dig (List UInt8))
    (hp : Plays fOf (ZkLink.repZCAt execArith hs512 T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
      (rounds.toList.map (·.coins)) pend) :
    ZkLink.OpensOKZ H512 enc512 leaf512 (ZkLink.width₀ T) cap₀ r.val pend.lig os.rows os.salts := by
  obtain ⟨-, -, -, -, -, hdim, hlv, -, -⟩ := zk_schedule_seams _ _ _ hsch
  obtain ⟨-, h2, hl0, -⟩ := hdim
  obtain ⟨pend', hp', hcaps, hreqs⟩ :=
    zk_rep_reqs hs512 T pts r st sch tPad extras zp rounds dom tr hsch hseam hlen hext hwf hrun hk
  obtain rfl := plays_det hp hp'
  rw [hms] at hreqs
  have hw₀ : ZkLink.width₀ T = 2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES := by
    unfold ZkLink.width₀
    rw [hseam.hsch, hseam.he, hl0]
    rfl
  rw [hw₀, hcap]
  exact opensOKZ_of sch hlv (by omega) zp.proof.cap zp.proof.ligerito r.val pend.lig os.rows os.salts hcaps hreqs ho
    hs

end FlockSoundness.Discharge.ZkExec
