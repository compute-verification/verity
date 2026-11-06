import Proofs.Flock.Claims.KS.LevelZero
import Proofs.Flock.Claims.KS.PlainLevels

/-!
# Every level's request in an accepted Ligerito run

`ligerito_refines_req0` fixes the model's first request: level 0's, at the positions of round `k + k0 + 6`'s coins.
Here the same run fixes every later one: the requests after it are levels `1, …, R` in order, level `ℓ`'s at the
positions of the coins of round `k + levRound k0 R ℓ` (`LaterAt`, `ligerito_refines_reqs`), so the first request of
level `ℓ ≥ 1` is that one (`find?_later`).

* `LaterAt`: requests that are levels `1, …, n` in order, each at the positions of its query round's coins.
* `laterAt_nil`, `qbOut_reqs_later`: none are, and a query step at the next level appends one.
* `find?_later`: after level 0's request, the first request of level `ℓ ≥ 1` among such requests is the `ℓ`-th.
* `ligerito_refines_reqs`, `ligerito_accepts_reqs`, `ligerito_accepts_opening_reqs`, `rep_refines_reqs`: their
  `_req0` versions, with the later requests `LaterAt`.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **Requests at levels `1, …, n` in order**, each at the positions of the coins of its query round in `rounds`, round
`k + levRound k0 R ℓ` for level `ℓ` of `sch` (`R` its last level), Ligerito starting at round `k`. -/
def LaterAt (sch : Flock.Schedule) (rounds : Array Flock.Round) (k : ℕ) (rest : List Req) : Prop :=
  ∀ (j : ℕ) (hj : j < rest.length), rest[j].level = j + 1 ∧
    rest[j].positions = (Flock.positions (sch.levels.getD (j + 1) default)
      (rounds[k + FlockDeploy.KS.levRound sch.k0 (sch.levels.size - 1) (j + 1)]!).coins).toList

theorem laterAt_nil (sch : Flock.Schedule) (rounds : Array Flock.Round) (k : ℕ) : LaterAt sch rounds k [] :=
  fun _ hj => absurd hj (Nat.not_lt_zero _)

/-- **A query step at level `i + 1` appends that level's request**, at the positions of its query coins: after `i`
requests `LaterAt`, the step's coins those of round `k + levRound k0 R (i + 1)`, the `i + 1` requests after it are. -/
theorem qbOut_reqs_later {D : Type} (r : ℕ) (sch : Flock.Schedule) (i : ℕ)
    (hl : LevelSeam (sch.levels.getD (i + 1) default)) (ch : List 𝕂) (start : ℕ) (stM : LigStateC 𝔽 𝕂 D)
    (cw cα cβ : Array F128) (w : Bool) (m : F256 × F256) {R : Req} {rest : List Req} {rounds : Array Flock.Round}
    {k : ℕ} (h : stM.reqs = R :: rest) (hlen : rest.length = i) (hlev : stM.level = i + 1)
    (hlat : LaterAt sch rounds k rest)
    (hcw : cw = (rounds[k + FlockDeploy.KS.levRound sch.k0 (sch.levels.size - 1) (i + 1)]!).coins) :
    ∃ rest', (qbOut r (levelOf (sch.levels.getD (i + 1) default)) ch start stM cw cα cβ w m).reqs = R :: rest' ∧
      rest'.length = i + 1 ∧ LaterAt sch rounds k rest' := by
  refine ⟨rest ++ [⟨stM.level, Model.positions execArith (levelOf (sch.levels.getD (i + 1) default)).queries
    (levelOf (sch.levels.getD (i + 1) default)).logLen
    (vecOf (levelOf (sch.levels.getD (i + 1) default)).queries cw), ch.length,
    pathLen (levelOf (sch.levels.getD (i + 1) default))⟩], by simp only [qbOut, h, List.cons_append],
    by simp [hlen], fun j hj => ?_⟩
  rcases Nat.lt_or_ge j rest.length with hj' | hj'
  · rw [List.getElem_append_left hj']
    exact hlat j hj'
  · have hji : j = i := by simp at hj; omega
    subst hji
    rw [List.getElem_append_right hj']
    simp only [hlen, Nat.sub_self, List.getElem_cons_zero]
    refine ⟨hlev, ?_⟩
    rw [hcw]
    exact positions_eq _ hl _

/-- **The first request of level `ℓ ≥ 1`** after level 0's, among requests at levels `1, …, n` in order, is the
`ℓ`-th. -/
theorem find?_later {sch : Flock.Schedule} {rounds : Array Flock.Round} {k : ℕ} {R : Req} {rest : List Req}
    (h0 : R.level = 0) (hlat : LaterAt sch rounds k rest) {ℓ : ℕ} (h1 : 1 ≤ ℓ) (hℓ : ℓ ≤ rest.length) :
    (R :: rest).find? (fun q => q.level == ℓ) = some (rest[ℓ - 1]'(by omega)) := by
  rw [List.find?_cons_of_neg (by simp [h0]; omega), List.find?_eq_some_iff_getElem]
  refine ⟨by simp [(hlat (ℓ - 1) (by omega)).1]; omega, ℓ - 1, by omega, rfl, fun j hj => ?_⟩
  simp [(hlat j (by omega)).1]
  omega

set_option maxHeartbeats 4000000 in
/-- **Ligerito's rounds refine the model's, every level's request fixed** (§13.3): `ligerito_refines_req0`, and the
requests after level 0's are levels `1, …, R` in order, each at the positions of the coins of its query round,
`k + levRound k0 R ℓ` (`LaterAt`). -/
theorem ligerito_refines_reqs {D : Type} (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (r : ℕ) (b : ℕ → 𝔽)
    (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1)
    (ho : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
      (o r ℓ j).1 = ((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf)
    {t t' : Transcript} (hwf : RoundsWF t.rounds) (h : Flock.ligerito ms sch T cap p basisAt t = .ok ((), t')) :
    ∃ (pend : PendL 𝔽 D) (initCh : Array F256) (levels : Array (Array F256)) (stF stF' : LState) (tF : Transcript),
      tF.rounds = t.rounds ∧ t.k ≤ tF.k ∧
      Plays fOf (Model.ligeritoC execArith r (schedOf sch) (fOf T) b) (ligMsgs p capD sch)
        (consumed t.rounds t.k tF.k) pend ∧
      pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.recursiveCaps[j]!) ∧
      (∀ req ∈ pend.reqs, ReqOK ms sch cap p req) ∧
      initCh.size = sch.k0 ∧ levels.size = sch.levels.size - 1 ∧ (∀ l ∈ levels.toList, l.size = 4) ∧
      p.finalProof.yr.size = 2 ^ (sch.ext + 1) ∧ (∀ cs ∈ stF.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size) ∧
      finalProg p basisAt sch.n initCh levels sch.ext stF tF = .ok (((), stF'), t') ∧
      pend.verdict o = Model.finalCheck execArith (sch.m - 7) b (initCh.toList.map gOf)
        (levels.toList.map fun l => l.toList.map gOf) (stF.oods.toList.map oodOf) (stF.conss.toList.map consOf)
        sch.ext (yrOf p sch.ext) (gOf stF.s) ∧
      ∃ (R : Req) (rest : List Req), pend.reqs = R :: rest ∧ R.level = 0 ∧
        R.positions = (Flock.positions (sch.levels.getD 0 default) (t.rounds[t.k + 6 + sch.k0]!).coins).toList ∧
        rest.length = sch.levels.size - 1 ∧ LaterAt sch t.rounds t.k rest := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  unfold Flock.ligerito at h
  obtain ⟨hcap, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨⟨u, stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  have hc01 := hcols 0 (by omega)
  have hn3 : 3 ≤ sch.n := by omega
  obtain ⟨z, t₅, h₅, h₄⟩ := lift_bind h₄
  obtain ⟨hn₅, rfl, hr₅, hk₅⟩ := sampleVec_ok (by omega) h₅
  obtain ⟨y₀, hy₀, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨bc, t₈, h₈, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, hm₀, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨initCh, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  have er₉ : t₉.rounds = t.rounds := by rw [k₉.2, hr₈, k₇.2, k₆.2, hr₅, k₃.2, k₂.2, k₁.2]
  have ek₉ : t₉.k = t.k + 2 := by rw [k₉.1, hk₈, k₇.1, k₆.1, hk₅, k₃.1, k₂.1, k₁.1]
  have hs₀ : execArith.embed (fOf T) + execArith.embed (fOf (t₇.rounds[t₇.k]!).coins[0]!) * execArith.embed (fOf y₀) =
      gOf (F256.ofF T + (F256.ofF y₀).smul ((t₇.rounds[t₇.k]!).coins.getD 0 0)) := by
    rw [gOf_add, gOf_smul, gOf_ofF, gOf_ofF, getElem!_eq_getD, mul_comm]; rfl
  have hq₀ : Model.fromMsg (gOf m₀.1, gOf m₀.2) (execArith.embed (fOf T) +
      execArith.embed (fOf (t₇.rounds[t₇.k]!).coins[0]!) * execArith.embed (fOf y₀)) =
      quadOf (Poly.ofMsg m₀ (F256.ofF T + (F256.ofF y₀).smul ((t₇.rounds[t₇.k]!).coins.getD 0 0))) := by
    rw [quadOf_ofMsg, hs₀]
  obtain ⟨hr₁₀, hk₁₀, hmsg₁₀, hood₁₀, -, -, -, hoods₁₀, hconss₁₀, hsz₁₀, -, sF, qF, hfold, hsF, hqF⟩ :=
    fold_refines p o sch.k0 0 #[] initCh _ st₁₀ t₉ t₁₀
      (fun _ => execArith.embed (fOf T) + execArith.embed (fOf (t₇.rounds[t₇.k]!).coins[0]!) * execArith.embed (fOf y₀))
      (fun _ => Model.fromMsg (gOf m₀.1, gOf m₀.2) (execArith.embed (fOf T) +
        execArith.embed (fOf (t₇.rounds[t₇.k]!).coins[0]!) * execArith.embed (fOf y₀))) hs₀ hq₀ (er₉ ▸ hwf) h₁₀
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨hnA, hnB, yA, mA, t₁₂, hyA, hmA, hr₁₂, hk₁₂, h₄⟩ := oodIntro_bind (by omega) h₄
  obtain ⟨hnC, hnD, yB, mB, t₁₃, hyB, hmB, hr₁₃, hk₁₃, h₄⟩ := oodIntro_bind (by omega) h₄
  have hlv0 := hlv 0 (by omega)
  obtain ⟨hnQ, hnQ', hmc, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevel_bind hlv0.q_ne_zero h₄
  obtain ⟨mq, t₁₅, hmq, k₁₅, h₄⟩ := nextMsg_bind h₄
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨bq, t₁₇, h₁₇, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₁₇, rfl, hr₁₇, hk₁₇⟩ := sample_ok h₁₇
  obtain ⟨stL, hstL, h₄⟩ := lmodify_bind' h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have hfin : finalProg p basisAt sch.n initCh out.2.2.1 out.2.2.2 st₁₈ t₁₈ = .ok ((u, stF), t') := hcont
  simp only [Array.size_empty, Array.extract_size] at hfold
  have er₁₁ : t₁₁.rounds = t.rounds := by rw [k₁₁.2, hr₁₀, er₉]
  have ek₁₁ : t₁₁.k = t.k + 2 + sch.k0 := by rw [k₁₁.1, hk₁₀, ek₉]
  have er₁₂ : t₁₂.rounds = t.rounds := hr₁₂.trans er₁₁
  have ek₁₂ : t₁₂.k = t.k + 4 + sch.k0 := by rw [hk₁₂, ek₁₁]; omega
  have er₁₃ : t₁₃.rounds = t.rounds := hr₁₃.trans er₁₂
  have ek₁₃ : t₁₃.k = t.k + 6 + sch.k0 := by rw [hk₁₃, ek₁₂]; omega
  have er₁₆ : t₁₆.rounds = t.rounds := by rw [k₁₆.2, k₁₅.2, hr₁₄, er₁₃]
  have ek₁₆ : t₁₆.k = t.k + 8 + sch.k0 := by rw [k₁₆.1, k₁₅.1, hk₁₄, ek₁₃]; omega
  have er₁₇ : t₁₇.rounds = t.rounds := hr₁₇.trans er₁₆
  have ek₁₇ : t₁₇.k = t.k + 9 + sch.k0 := by rw [hk₁₇, ek₁₆]; omega
  obtain ⟨sT, qT, htwo, hsT, hqT⟩ := twoOodC_plays o (sch.m - 7 - sch.k0 + 1) 0 sF qF _ _ hsF hqF
    (t₁₁.rounds[t₁₁.k]!).coins (t₁₁.rounds[t₁₁.k + 1]!).coins (t₁₂.rounds[t₁₂.k]!).coins
    (t₁₂.rounds[t₁₂.k + 1]!).coins (by rw [RoundsWF.get (er₁₁ ▸ hwf), hnA, hn7])
    (by rw [RoundsWF.get (er₁₁ ▸ hwf), hnB]) (by rw [RoundsWF.get (er₁₂ ▸ hwf), hnC, hn7])
    (by rw [RoundsWF.get (er₁₂ ▸ hwf), hnD]) (by omega) yA yB mA mB
  obtain ⟨stM₀, hstM₀⟩ : ∃ x : LigStateC 𝔽 𝕂 D, x =
      { qbOut r (levelOf (sch.levels.getD 0 default)) (initCh.toList.map gOf) 0
          ⟨sT, qT, sch.m - 7 - sch.k0 + 1, [],
            ⟨List.ofFn (vecOf (sch.m - 7) (t₃.rounds[t₃.k]!).coins), fOf (t₇.rounds[t₇.k]!).coins[0]!, none⟩ ::
              [oodOf ⟨(t₁₁.rounds[t₁₁.k]!).coins, (t₁₁.rounds[t₁₁.k + 1]!).coins.getD 0 0, some 0⟩,
                oodOf ⟨(t₁₂.rounds[t₁₂.k]!).coins, (t₁₂.rounds[t₁₂.k + 1]!).coins.getD 0 0, some 0⟩],
            [], 0, [], [], levelOf (sch.levels.getD 0 default)⟩
          (t₁₃.rounds[t₁₃.k]!).coins (t₁₃.rounds[t₁₃.k + 1]!).coins (t₁₆.rounds[t₁₆.k]!).coins true mq with
        level := 1, caps := [capD p.recursiveCaps[0]!], prevLevel := levelOf (sch.levels.getD 1 default) } :=
    ⟨_, rfl⟩
  simp only at hoods₁₀ hconss₁₀ hood₁₀ hmsg₁₀
  have h0 : 0 < sch.levels.size := Nat.lt_of_lt_of_le (by decide) h2
  have hz₀ : (t₃.rounds[t₃.k]!).coins.size = sch.m - 7 := by
    rw [RoundsWF.get ((k₃.2.trans (k₂.2.trans k₁.2)) ▸ hwf), hn₅, hn7]
  have hopen0 : openAt p (sch.levels.size - 1) 0 =
      (p.initialProof.rows, p.initialProof.siblings, p.initialProof.salts) := by simp [openAt]
  have hmc0 : merkleCheck ms (capAt cap p 0) (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
      (Flock.positions (sch.levels.getD 0 default) (t₁₃.rounds[t₁₃.k]!).coins) (openAt p (sch.levels.size - 1) 0).1
      (openAt p (sch.levels.size - 1) 0).2.1 (openAt p (sch.levels.size - 1) 0).2.2
      (2 ^ (sch.levels.getD 0 default).lanes) = .ok () := by
    rw [show capAt cap p 0 = cap by simp [capAt], hopen0]; exact hmc
  obtain ⟨stQ, hstQ⟩ : ∃ x : LigStateC 𝔽 𝕂 D, x = ⟨sT, qT, sch.m - 7 - sch.k0 + 1, [],
      ⟨List.ofFn (vecOf (sch.m - 7) (t₃.rounds[t₃.k]!).coins), fOf (t₇.rounds[t₇.k]!).coins[0]!, none⟩ ::
        [oodOf ⟨(t₁₁.rounds[t₁₁.k]!).coins, (t₁₁.rounds[t₁₁.k + 1]!).coins.getD 0 0, some 0⟩,
          oodOf ⟨(t₁₂.rounds[t₁₂.k]!).coins, (t₁₂.rounds[t₁₂.k + 1]!).coins.getD 0 0, some 0⟩],
      [], 0, [], [], levelOf (sch.levels.getD 0 default)⟩ := ⟨_, rfl⟩
  rw [← hstQ] at hstM₀
  obtain ⟨hqs, hqq, hqc, hqr, hql, hqo, hqlev, hqcaps⟩ := qbOut_rel ms sch cap p r 0 h0 hlv0 initCh
    (by rw [hsz₁₀, hl0]; simp) 0 stQ (by rw [hstQ]) o (ho 0 h0) (t₁₃.rounds[t₁₃.k]!).coins
    (t₁₃.rounds[t₁₃.k + 1]!).coins (t₁₆.rounds[t₁₆.k]!).coins (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ']) hmc0 true mq
  rw [hopen0] at hqs hqq
  have hrel₀ : LoopRel ms sch cap p capD o 0 (sch.n - sch.k0 + 1, p.recursiveCaps[0]!, #[], 0) stL t₁₇ stM₀ := by
    rw [hstM₀]
    refine ⟨er₁₇ ▸ hwf, ?_, ?_, ?_, rfl, rfl, by simp, ?_, ?_, ?_, ?_, rfl, rfl, ?_⟩
    · dsimp only
      rw [hqs, hstL, hstQ]
      dsimp only
      rw [hsT]
      simp only [gOf_add, gOf_smul]
      rfl
    · dsimp only
      rw [hqq, hstL, hstQ]
      dsimp only
      rw [hqT]
      simp only [ite_true, quadOf_addScaled, quadOf_ofMsg]
      rfl
    · show sch.n - sch.k0 + 1 = _
      rw [← hc0, ← hc01]
    · dsimp only
      rw [hql, hstQ]
      rfl
    · dsimp only
      rw [hqo, hstQ, hstL]
      dsimp only
      rw [hoods₁₀, show List.ofFn (vecOf (sch.m - 7) (t₃.rounds[t₃.k]!).coins) =
        (t₃.rounds[t₃.k]!).coins.toList.map fOf from ofFn_vecOf _ _ hz₀]
      simp [oodOf, getElem!_eq_getD]
    · dsimp only
      rw [hqc, hstQ, hstL]
      dsimp only
      rw [hconss₁₀, hc0]
      simp
    · rw [hstL]
      dsimp only
      rw [hconss₁₀]
      exact conss_push_pos (fun x hx => by simp at hx) (pos_le_alpha _ hlv0 _ _
        (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ']))
    · intro req hreq
      rcases hqr req hreq with h | h
      · rw [hstQ] at h; cases h
      · exact h
  obtain ⟨R, hR0, hRl, hRp⟩ : ∃ R : Req, stM₀.reqs = [R] ∧ R.level = 0 ∧
      R.positions = (Flock.positions (sch.levels.getD 0 default) (t₁₃.rounds[t₁₃.k]!).coins).toList := by
    rw [hstM₀, hstQ]
    exact ⟨_, rfl, rfl, positions_eq _ hlv0 _⟩
  have hL := loop_compose _ (sch.levels.size - 1)
    (fun i stM => Model.recursiveRoundsC execArith r (sch.m - 7) b (initCh.toList.map gOf) sch.ext
      ((sch.levels.toList.map levelOf).drop (i + 1)) stM)
    (fun i v stE tE stM => LoopRel ms sch cap p capD o i v stE tE stM ∧ ∃ rest, stM.reqs = R :: rest ∧
      tE.rounds = t.rounds ∧ tE.k = t.k + 9 + sch.k0 + 11 * i ∧ rest.length = i ∧ LaterAt sch t.rounds t.k rest)
    (fun i stE => recMsgs p capD sch.ext (sch.levels.size - 1 - i) i stE.msg stE.ood)
    (fun v stE _ pend => LoopFinal ms sch cap p capD o b (initCh.toList.map gOf) v stE pend ∧
      ∃ rest, pend.reqs = R :: rest ∧ rest.length = sch.levels.size - 1 ∧ LaterAt sch t.rounds t.k rest)
    ?hstep ?hlast (sch.levels.size - 1) 0 _ _ _ out st₁₈ t₁₈ stM₀ (by omega) (by omega)
    ⟨hrel₀, [], hR0, er₁₇, by rw [ek₁₇]; omega, rfl, laterAt_nil _ _ _⟩ hloop
  case hstep =>
    clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1 ho
    intro i v stE tE bv stE' tE' stM hi hrel hb
    obtain ⟨⟨hwfE, hsE, hqE, hsplit, hprev, hsize, hlanes, hlevels, hoods, hcons, hposE, hlevel, hcaps, hreqs⟩, rest,
      hrest, hrE, hkE, hlenE, hlatE⟩ := hrel
    have hc1 := hcols (i + 1) (by omega)
    have hl1 := hl4 (i + 1) (by omega) (by omega)
    have hl2 := hl4 (i + 1 + 1) (by omega) (by omega)
    dsimp only at hb
    rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
    obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁, hmsg₁, hood₁, -, -, -, hoods₁, hconss₁, hsz₁, -, sM₁, qM₁, hfoldM, hsM₁, hqM₁⟩ :=
      fold_refines p o 4 0 #[] chs stE st₁ tE t₁ stM.s stM.q hsE hqE hwfE hf
    simp only [Array.size_empty, Array.extract_size] at hfoldM
    rw [ite_eq_left hi] at hb
    obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
    have k₂ := absorbBytes_ok h₂
    obtain ⟨hnA, hnB, yA, mA, t₃, hyA, hmA, hr₃, hk₃, hb⟩ := oodIntro_bind (by omega) hb
    obtain ⟨hnC, hnD, yB, mB, t₄, hyB, hmB, hr₄, hk₄, hb⟩ := oodIntro_bind (by omega) hb
    have hlvi := hlv (i + 1) (by omega)
    obtain ⟨hnQ, hnQ', hmc, t₅, hr₅, hk₅, hb⟩ := queryLevel_bind hlvi.q_ne_zero hb
    obtain ⟨mq, t₆, hmq, k₆, hb⟩ := nextMsg_bind hb
    obtain ⟨t₇, k₇, hb⟩ := claimPow_bind hb
    obtain ⟨bq, t₈, h₈, hb⟩ := lift_bind hb
    obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
    obtain ⟨stN, hstN, hb⟩ := lmodify_bind' hb
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
    have hsz : chs.size = 4 := by simpa using hsz₁
    have hlen : (stM.levels ++ [chs.toList.map gOf]).length = i + 1 := by simp [hlevels, hsize]
    have hdim : (levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
        (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes = v.1 - 4 + 1 := by
      show (sch.levels.getD (i + 1 + 1) default).cols + (sch.levels.getD (i + 1 + 1) default).lanes = _
      omega
    obtain ⟨sT, qT, htwo, hsT, hqT⟩ := twoOodC_plays o (v.1 - 4 + 1) (i + 1) sM₁ qM₁ _ _ hsM₁ hqM₁
      (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins (t₃.rounds[t₃.k]!).coins (t₃.rounds[t₃.k + 1]!).coins
      (by rw [RoundsWF.get (er₂ ▸ hwfE), hnA]) (by rw [RoundsWF.get (er₂ ▸ hwfE), hnB])
      (by rw [RoundsWF.get (er₃ ▸ hwfE), hnC]) (by rw [RoundsWF.get (er₃ ▸ hwfE), hnD]) (by omega) yA yB mA mB
    have hcapAt : capAt cap p (i + 1) = v.2.1 := by rw [hprev]; simp [capAt]
    have hopen : openAt p (sch.levels.size - 1) (i + 1) =
        (p.recursiveProofs[i]!.rows, p.recursiveProofs[i]!.siblings, p.recursiveProofs[i]!.salts) := by
      simp [openAt, hi]
    have hmc' : merkleCheck ms (capAt cap p (i + 1)) (sch.levels.getD (i + 1) default).cap
        (sch.levels.getD (i + 1) default).d (Flock.positions (sch.levels.getD (i + 1) default) (t₄.rounds[t₄.k]!).coins)
        (openAt p (sch.levels.size - 1) (i + 1)).1 (openAt p (sch.levels.size - 1) (i + 1)).2.1
        (openAt p (sch.levels.size - 1) (i + 1)).2.2 (2 ^ (sch.levels.getD (i + 1) default).lanes) = .ok () := by
      rw [hcapAt, hopen]; exact hmc
    set stQ : LigStateC 𝔽 𝕂 D := { stM with
        s := sT, q := qT, levels := stM.levels ++ [chs.toList.map gOf],
        oods := stM.oods ++ [oodOf ⟨(t₂.rounds[t₂.k]!).coins, (t₂.rounds[t₂.k + 1]!).coins.getD 0 0, some (i + 1)⟩,
          oodOf ⟨(t₃.rounds[t₃.k]!).coins, (t₃.rounds[t₃.k + 1]!).coins.getD 0 0, some (i + 1)⟩] } with hstQ
    have hq := qbOut_rel ms sch cap p r (i + 1) (by omega) hlvi chs (by rw [hsz, hl1])
      (stM.levels ++ [chs.toList.map gOf]).length stQ hlevel o (ho (i + 1) (by omega)) (t₄.rounds[t₄.k]!).coins
      (t₄.rounds[t₄.k + 1]!).coins (t₇.rounds[t₇.k]!).coins (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ']) hmc' true mq
    rw [hopen] at hq
    obtain ⟨hqs, hqq, hqc, hqr, hql, hqo, hqlev, hqcaps⟩ := hq
    set qb := qbOut r (levelOf (sch.levels.getD (i + 1) default)) (chs.toList.map gOf)
      (stM.levels ++ [chs.toList.map gOf]).length stQ (t₄.rounds[t₄.k]!).coins (t₄.rounds[t₄.k + 1]!).coins
      (t₇.rounds[t₇.k]!).coins true mq with hqb
    have hmsgA := getElem!_of_getElem? hmA
    have hoodA := getElem!_of_getElem? hyA
    have hmsgB := getElem!_of_getElem? hmB
    have hoodB := getElem!_of_getElem? hyB
    have hmsgQ := getElem!_of_getElem? hmq
    simp only [hmsg₁, hood₁, Nat.add_assoc, Nat.reduceAdd] at hmsgA hoodA hmsgB hoodB hmsgQ
    subst hstN
    refine ⟨_, { qb with
        level := stM.level + 1, caps := stM.caps ++ [capD p.recursiveCaps[i + 1]!],
        prevLevel := levelOf (sch.levels.getD (i + 1 + 1) default) },
      kMsgs p stE.msg 4 ++ ⟨Cap D, capD p.recursiveCaps[i + 1]!⟩ ::
        [fMsg p.oodValues[stE.ood]!, kMsg p.messages[stE.msg + 4]!, fMsg p.oodValues[stE.ood + 1]!,
          kMsg p.messages[stE.msg + 5]!] ++ [kMsg p.messages[stE.msg + 6]!],
      rfl, ?rel, er₈, by omega, ?tape, ?plays⟩
    case rel =>
      refine ⟨⟨er₈ ▸ hwfE, ?_, ?_, ?_, rfl, by simp [hsize], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
      · show qb.s o = _
        rw [hqs]
        simp only [hstQ, hsT, gOf_add, gOf_smul]
        rfl
      · show qb.q o = _
        rw [hqq]
        simp only [ite_true, hstQ, hqT, quadOf_addScaled, quadOf_ofMsg]
        rfl
      · show v.1 - 4 + 1 = _
        omega
      · intro l hl
        simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hl
        rcases hl with hl | rfl
        · exact hlanes l hl
        · exact hsz
      · show qb.levels = _
        rw [hql, hstQ]
        simp [hlevels]
      · show qb.oods = _
        rw [hqo, hstQ]
        simp [hoods, hoods₁]
      · show qb.cons = _
        rw [hqc, hstQ, hlen]
        simp [hcons, hconss₁, hsplit]
      · dsimp only
        rw [hconss₁]
        exact conss_push_pos hposE (pos_le_alpha _ hlvi _ _ (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ']))
      · show stM.level + 1 = _
        rw [hlevel]
      · show stM.caps ++ _ = _
        rw [hcaps, List.range_succ (n := i + 1), List.map_append]
        rfl
      · intro req hreq
        rcases hqr req hreq with h | h
        · exact hreqs req h
        · exact h
      · show ∃ rest', qb.reqs = R :: rest' ∧ tE'.rounds = t.rounds ∧ tE'.k = t.k + 9 + sch.k0 + 11 * (i + 1) ∧
          rest'.length = i + 1 ∧ LaterAt sch t.rounds t.k rest'
        obtain ⟨rest', h1, h2, h3⟩ := qbOut_reqs_later r sch i hlvi (chs.toList.map gOf)
          (stM.levels ++ [chs.toList.map gOf]).length stQ (t₄.rounds[t₄.k]!).coins (t₄.rounds[t₄.k + 1]!).coins
          (t₇.rounds[t₇.k]!).coins true mq hrest hlenE hlevel hlatE (by
            rw [er₄, hrE, ek₄, hkE, FlockDeploy.KS.levRound, ite_eq_left hi,
              show t.k + 9 + sch.k0 + 11 * i + 8 = t.k + (sch.k0 + 6 + 11 * (i + 1)) by omega])
        exact ⟨rest', h1, er₈.trans hrE, by rw [ek₈, hkE]; omega, h2, h3⟩
    case tape =>
      obtain ⟨j, hj⟩ : ∃ j, sch.levels.size - 1 - i = j + 2 := ⟨sch.levels.size - 1 - i - 2, by omega⟩
      rw [hj, show sch.levels.size - 1 - (i + 1) = j + 1 by omega]
      simp only [recMsgs, hmsg₁, hood₁, Nat.add_assoc, Nat.reduceAdd, List.append_assoc, List.cons_append]
    case plays =>
      intro msR csR pend hrec
      rw [drop_levels sch (i + 1) (by omega), drop_levels sch (i + 1 + 1) (by omega)]
      rw [drop_levels sch (i + 1 + 1) (by omega)] at hrec
      have hfold' : Plays fOf
          (Model.foldRoundsC execArith (levelOf (sch.levels.getD (i + 1) default)).lanes stM.s stM.q)
          (kMsgs p stE.msg 4) (consumed tE.rounds tE.k t₁.k) (List.map gOf chs.toList, sM₁, qM₁) := by
        rw [show (levelOf (sch.levels.getD (i + 1) default)).lanes = 4 from hl1]; exact hfoldM
      have htwo' : Plays fOf (Model.twoOodC execArith ((levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
          (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes) (stM.levels ++ [chs.toList.map gOf]).length sM₁ qM₁)
          [fMsg yA, kMsg mA, fMsg yB, kMsg mB]
          [t₂.rounds[t₂.k]!.coins, t₂.rounds[t₂.k + 1]!.coins, t₃.rounds[t₃.k]!.coins, t₃.rounds[t₃.k + 1]!.coins]
          (sT, qT, [oodOf ⟨(t₂.rounds[t₂.k]!).coins, (t₂.rounds[t₂.k + 1]!).coins.getD 0 0, some (i + 1)⟩,
            oodOf ⟨(t₃.rounds[t₃.k]!).coins, (t₃.rounds[t₃.k + 1]!).coins.getD 0 0, some (i + 1)⟩]) := by
        rw [hdim, hlen]; exact htwo
      have hmain := iterNL_model r (sch.m - 7) b (initCh.toList.map gOf) sch.ext
        (levelOf (sch.levels.getD (i + 1) default)) (levelOf (sch.levels.getD (i + 1 + 1) default))
        ((sch.levels.toList.map levelOf).drop (i + 1 + 1 + 1)) stM hfold' (capD p.recursiveCaps[i + 1]!) htwo'
        (t₄.rounds[t₄.k]!).coins (t₄.rounds[t₄.k + 1]!).coins (t₇.rounds[t₇.k]!).coins
        (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ]; rfl)
        (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ']; exact hlvi.2.2.2.1)
        (by rw [RoundsWF.get (er₇ ▸ hwfE), hn₈]) hlvi.q_ne_one (hα1 (i + 1) (by omega)) mq hrec
      have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k t₁.k ++
          [t₂.rounds[t₂.k]!.coins, t₂.rounds[t₂.k + 1]!.coins, t₃.rounds[t₃.k]!.coins, t₃.rounds[t₃.k + 1]!.coins] ++
          [t₄.rounds[t₄.k]!.coins, t₄.rounds[t₄.k + 1]!.coins, t₇.rounds[t₇.k]!.coins] := by
        rw [er₂, ek₂, er₃, ek₃, er₄, ek₄, er₇, ek₇, ek₈, hk₁,
          consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 11 by omega),
          show tE.k + 11 = tE.k + 4 + 7 by omega, consumed_add tE.rounds (tE.k + 4) 7]
        simp [List.range_succ, Nat.add_assoc]
      rw [hcs, hmsgA, hoodA, hmsgB, hoodB, hmsgQ]
      simp only [List.append_assoc, List.cons_append, List.nil_append] at hmain ⊢
      exact hmain
  case hlast =>
    clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1 ho
    intro i v stE tE bv stE' tE' stM hi hrel hb
    obtain ⟨⟨hwfE, hsE, hqE, hsplit, hprev, hsize, hlanes, hlevels, hoods, hcons, hposE, hlevel, hcaps, hreqs⟩, rest,
      hrest, hrE, hkE, hlenE, hlatE⟩ := hrel
    have hl1 := hl4 (i + 1) (by omega) (by omega)
    dsimp only at hb
    rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
    obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁, hmsg₁, hood₁, -, -, -, hoods₁, hconss₁, hsz₁, -, sM₁, qM₁, hfoldM, hsM₁, hqM₁⟩ :=
      fold_refines p o 4 0 #[] chs stE st₁ tE t₁ stM.s stM.q hsE hqE hwfE hf
    simp only [Array.size_empty, Array.extract_size] at hfoldM
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
    obtain ⟨hnQ, hnQ', hmc, t₃, hr₃, hk₃, hb⟩ := queryLevel_bind hlvR.q_ne_zero hb
    obtain ⟨t₄, k₄, hb⟩ := claimPow_bind hb
    obtain ⟨bq, t₅, h₅, hb⟩ := lift_bind hb
    obtain ⟨-, hn₅, rfl, hr₅, hk₅⟩ := sample_ok h₅
    obtain ⟨stN, hstN, hb⟩ := lmodify_bind' hb
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
    have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
    have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
    have er₄ : t₄.rounds = tE.rounds := by rw [k₄.2, hr₃, er₂]
    have ek₄ : t₄.k = tE.k + 6 := by rw [k₄.1, hk₃, ek₂]
    have er₅ : tE'.rounds = tE.rounds := hr₅.trans er₄
    have ek₅ : tE'.k = tE.k + 7 := by rw [hk₅, ek₄]
    have hsz : chs.size = 4 := by simpa using hsz₁
    rw [hi] at hsplit hl1
    have hlen : (stM.levels ++ [chs.toList.map gOf]).length = sch.levels.size - 1 := by
      simp [hlevels, hsize]; omega
    have hyr' : p.finalProof.yr.size = 2 ^ (sch.ext + 1) := by
      simp only [bne_iff_ne, ne_eq, not_not] at hyr
      rw [hyr, hext, hsplit, Nat.add_sub_cancel]
    have hcapAt : capAt cap p (sch.levels.size - 1) = v.2.1 := by
      rw [hprev]; simp [capAt, show sch.levels.size - 1 ≠ 0 by omega, show sch.levels.size - 1 - 1 = i by omega]
    have hopen : openAt p (sch.levels.size - 1) (sch.levels.size - 1) =
        (p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts) := by
      simp [openAt, show sch.levels.size - 1 ≠ 0 by omega]
    have hmc' : merkleCheck ms (capAt cap p (sch.levels.size - 1)) (sch.levels.getD (sch.levels.size - 1) default).cap
        (sch.levels.getD (sch.levels.size - 1) default).d
        (Flock.positions (sch.levels.getD (sch.levels.size - 1) default) (t₂.rounds[t₂.k]!).coins)
        (openAt p (sch.levels.size - 1) (sch.levels.size - 1)).1
        (openAt p (sch.levels.size - 1) (sch.levels.size - 1)).2.1
        (openAt p (sch.levels.size - 1) (sch.levels.size - 1)).2.2
        (2 ^ (sch.levels.getD (sch.levels.size - 1) default).lanes) = .ok () := by
      rw [hcapAt, hopen]; exact hmc
    set stQ : LigStateC 𝔽 𝕂 D := { stM with s := sM₁, q := qM₁, levels := stM.levels ++ [chs.toList.map gOf] }
      with hstQ
    have hq := qbOut_rel ms sch cap p r (sch.levels.size - 1) (by omega) hlvR chs (by rw [hsz, hl1])
      (stM.levels ++ [chs.toList.map gOf]).length stQ (by rw [hstQ]; exact hlevel.trans hi) o
      (ho (sch.levels.size - 1) (by omega)) (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins
      (t₄.rounds[t₄.k]!).coins (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ']) hmc' false (0, 0)
    rw [hopen] at hq
    obtain ⟨hqs, hqq, hqc, hqr, hql, hqo, hqlev, hqcaps⟩ := hq
    have hfold' : Plays fOf (Model.foldRoundsC execArith (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes
        stM.s stM.q) (kMsgs p stE.msg 4) (consumed tE.rounds tE.k t₁.k) (List.map gOf chs.toList, sM₁, qM₁) := by
      rw [show (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes = 4 from hl1]; exact hfoldM
    have hmain := iterLast_model r (sch.m - 7) b (initCh.toList.map gOf) sch.ext
      (levelOf (sch.levels.getD (sch.levels.size - 1) default)) stM hfold' (yrOf p sch.ext)
      (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins (t₄.rounds[t₄.k]!).coins
      (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ]; rfl)
      (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ']; exact hlvR.2.2.2.1)
      (by rw [RoundsWF.get (er₄ ▸ hwfE), hn₅]) hlvR.q_ne_one (hα1 (sch.levels.size - 1) (by omega))
    have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k t₁.k ++
        [t₂.rounds[t₂.k]!.coins, t₂.rounds[t₂.k + 1]!.coins, t₄.rounds[t₄.k]!.coins] := by
      rw [er₂, ek₂, er₄, ek₄, ek₅, hk₁,
        consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 7 by omega),
        show tE.k + 7 = tE.k + 4 + 3 by omega, consumed_add tE.rounds (tE.k + 4) 3]
      simp [List.range_succ, Nat.add_assoc]
    have hdrop : (sch.levels.toList.map levelOf).drop (i + 1) =
        [levelOf (sch.levels.getD (sch.levels.size - 1) default)] := by
      rw [hi, drop_levels sch (sch.levels.size - 1) (by omega), show sch.levels.size - 1 + 1 = sch.levels.size by omega]
      simp
    rw [← hstQ] at hmain
    rw [hdrop, show sch.levels.size - 1 - i = 1 by omega, hcs]
    simp only [recMsgs]
    refine ⟨_, _, rfl, er₅, by omega, hmain, ?_⟩
    subst hstN
    refine ⟨⟨by simp [hsize]; omega, ?_, by simp only; omega, hyr', ?_, ?_, ?_, ?_⟩, ?_⟩
    · intro l hl
      simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hl
      rcases hl with hl | rfl
      · exact hlanes l hl
      · exact hsz
    · dsimp only
      rw [hql, hqo, hqc, hqs, hstQ, hlen]
      dsimp only
      rw [hlevels, hoods, hcons, hsM₁, hoods₁, hconss₁,
        show v.1 - 4 = (sch.levels.getD (sch.levels.size - 1) default).cols by omega, gOf_add, gOf_smul]
      simp only [Array.toList_push, List.map_append, List.map_cons, List.map_nil]
      rfl
    · dsimp only
      rw [hqcaps, hstQ]
      simp only [hcaps, hi]
    · intro req hreq
      rcases hqr req hreq with h | h
      · exact hreqs req h
      · exact h
    · dsimp only
      rw [hconss₁]
      exact conss_push_pos hposE (pos_le_alpha _ hlvR _ _ (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ']))
    · have hiR : i + 1 = sch.levels.size - 1 := by omega
      obtain ⟨rest', h1, h2, h3⟩ := qbOut_reqs_later r sch i (by rw [hiR]; exact hlvR) (chs.toList.map gOf)
        (stM.levels ++ [chs.toList.map gOf]).length stQ (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins
        (t₄.rounds[t₄.k]!).coins false (0, 0) hrest hlenE hlevel hlatE (by
          rw [er₂, hrE, ek₂, hkE, FlockDeploy.KS.levRound, ite_eq_right (by omega),
            show t.k + 9 + sch.k0 + 11 * i + 4 = t.k + (sch.k0 + 2 + 11 * (i + 1)) by omega])
      rw [hiR] at h1 h2
      exact ⟨rest', h1, h2, h3⟩
  obtain ⟨pend, hrL, hkL, hplL, ⟨hsizeL, hlanesL, hextL, hyrL, hverL, hcapsL, hreqsL, hposL⟩, restL, hrestL, hlenL,
    hlatL⟩ := hL
  clear hcont h₄ hloop
  have er₃ : t₃.rounds = t.rounds := by rw [k₃.2, k₂.2, k₁.2]
  have ek₃ : t₃.k = t.k := by rw [k₃.1, k₂.1, k₁.1]
  have er₇ : t₇.rounds = t.rounds := by rw [k₇.2, k₆.2, hr₅, er₃]
  have ek₇ : t₇.k = t.k + 1 := by rw [k₇.1, k₆.1, hk₅, ek₃]
  have er₁₀ : t₁₀.rounds = t.rounds := hr₁₀.trans er₉
  have ek₁₀ : t₁₀.k = t.k + 2 + sch.k0 := by rw [hk₁₀, ek₉]
  have ek₁₂' : t₁₂.k = t.k + 2 + sch.k0 + 2 := by rw [ek₁₂]; omega
  have ek₁₃' : t₁₃.k = t.k + 2 + sch.k0 + 4 := by rw [ek₁₃]; omega
  have ek₁₆' : t₁₆.k = t.k + 2 + sch.k0 + 6 := by rw [ek₁₆]; omega
  have ek₁₇' : t₁₇.k = t.k + 2 + sch.k0 + 7 := by rw [ek₁₇]; omega
  have hcs : consumed t.rounds t.k t₁₈.k = (t₃.rounds[t₃.k]!).coins :: (t₇.rounds[t₇.k]!).coins ::
      (consumed t₉.rounds t₉.k t₁₀.k ++
        [(t₁₁.rounds[t₁₁.k]!).coins, (t₁₁.rounds[t₁₁.k + 1]!).coins, (t₁₂.rounds[t₁₂.k]!).coins,
          (t₁₂.rounds[t₁₂.k + 1]!).coins] ++
        [(t₁₃.rounds[t₁₃.k]!).coins, (t₁₃.rounds[t₁₃.k + 1]!).coins, (t₁₆.rounds[t₁₆.k]!).coins] ++
        consumed t₁₇.rounds t₁₇.k t₁₈.k) := by
    rw [er₃, ek₃, er₇, ek₇, er₉, ek₉, ek₁₀, er₁₁, ek₁₁, er₁₂, ek₁₂', er₁₃, ek₁₃', er₁₆, ek₁₆', er₁₇, ek₁₇',
      consumed_append t.rounds (show t.k ≤ t.k + 2 + sch.k0 + 7 by omega) (by rw [← ek₁₇']; exact hkL),
      consumed_cons _ (show t.k < t.k + 2 + sch.k0 + 7 by omega),
      consumed_cons _ (show t.k + 1 < t.k + 2 + sch.k0 + 7 by omega),
      consumed_append t.rounds (show t.k + 1 + 1 ≤ t.k + 2 + sch.k0 by omega)
        (show t.k + 2 + sch.k0 ≤ t.k + 2 + sch.k0 + 7 by omega),
      consumed_add t.rounds (t.k + 2 + sch.k0) 7]
    simp [List.range_succ, Nat.add_assoc]
  have hlv01 : (schedOf sch).levels = levelOf (sch.levels.getD 0 default) ::
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := by
    have e0 := drop_levels sch 0 h0
    have e1 := drop_levels sch 1 (by omega)
    simp only [List.drop_zero, Nat.zero_add, Nat.reduceAdd] at e0 e1
    rw [e1] at e0
    exact e0
  have hd1 : (sch.levels.toList.map levelOf).drop (0 + 1) =
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := drop_levels sch 1 (by omega)
  rw [hd1, hstM₀, hstQ] at hplL
  have hmain := ligeritoC_model r (schedOf sch) (fOf T) b _ _ _ hlv01 (t₃.rounds[t₃.k]!).coins
    (t₇.rounds[t₇.k]!).coins hz₀ (by show sch.m - 7 ≠ 1; omega) (by rw [RoundsWF.get (er₇ ▸ hwf), hn₈]) y₀ m₀ hfold
    (capD p.recursiveCaps[0]!) htwo
    (t₁₃.rounds[t₁₃.k]!).coins (t₁₃.rounds[t₁₃.k + 1]!).coins (t₁₆.rounds[t₁₆.k]!).coins
    (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ]; rfl) (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ']; exact hlv0.2.2.2.1)
    (by rw [RoundsWF.get (er₁₆ ▸ hwf), hn₁₇]) hlv0.q_ne_one (hα1 0 (by omega)) mq hplL
  have hy0 := getElem!_of_getElem? hy₀
  have hm0 := getElem!_of_getElem? hm₀
  have hyA' := getElem!_of_getElem? hyA
  have hmA' := getElem!_of_getElem? hmA
  have hyB' := getElem!_of_getElem? hyB
  have hmB' := getElem!_of_getElem? hmB
  have hmq' := getElem!_of_getElem? hmq
  simp only [hood₁₀, hmsg₁₀] at hy0 hm0 hyA' hmA' hyB' hmB' hmq'
  have hLmsg : stL.msg = sch.k0 + 4 := by rw [hstL]; simp only [hmsg₁₀]; omega
  have hLood : stL.ood = 3 := by rw [hstL]; simp only [hood₁₀]
  refine ⟨pend, initCh, out.2.2.1, st₁₈, stF, t₁₈, hrL.trans er₁₇, by rw [er₁₇] at hrL; omega, ?_, hcapsL,
    hreqsL, by rw [hsz₁₀]; simp, hsizeL, hlanesL, hyrL, hposL, by rw [← hextL]; exact hfin, hverL,
    R, restL, hrestL, hRl, by rw [hRp, er₁₃, ek₁₃], hlenL, hlatL⟩
  rw [hcs]
  unfold ligMsgs
  rw [hLmsg, hLood, Nat.sub_zero] at hmain
  rw [show 0 + 1 + sch.k0 = sch.k0 + 1 by rw [Nat.zero_add, Nat.add_comm]] at hmA'
  rw [show 0 + 1 + sch.k0 + 1 = sch.k0 + 2 by rw [Nat.zero_add, Nat.add_comm 1 sch.k0]] at hmB'
  rw [show 0 + 1 + sch.k0 + 1 + 1 = sch.k0 + 3 by rw [Nat.zero_add, Nat.add_comm 1 sch.k0]] at hmq'
  rw [show (0 + 1 + 1 : ℕ) = 2 from rfl] at hyB'
  rw [show (0 + 1 : ℕ) = 1 from rfl] at hyA' hmain
  rw [hy0, hm0, hyA', hmA', hyB', hmB', hmq']
  simp only [List.append_assoc, List.cons_append, List.nil_append] at hmain ⊢
  exact hmain

/-- **Ligerito refines the model's, every level's request fixed** (§13.3–§13.6): `ligerito_accepts_req0`, and
the requests after level 0's are `LaterAt`. -/
theorem ligerito_accepts_reqs {D : Type} (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (r : ℕ) (b : ℕ → 𝔽)
    (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1)
    (ho : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
      (o r ℓ j).1 = ((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf)
    (hbasis : ∀ (orig : Array F256) (y : ℕ), orig.size + sch.ext = sch.m - 7 → y < 2 ^ sch.ext →
      gOf (basisAt orig sch.ext)[y]! = Model.bHat execArith (sch.m - 7) b (orig.toList.map gOf ++ bitsK sch.ext y))
    {t t' : Transcript} (hwf : RoundsWF t.rounds) (h : Flock.ligerito ms sch T cap p basisAt t = .ok ((), t')) :
    ∃ pend : PendL 𝔽 D, t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      Plays fOf (Model.ligeritoC execArith r (schedOf sch) (fOf T) b) (ligMsgs p capD sch)
        (consumed t.rounds t.k t'.k) pend ∧
      pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.recursiveCaps[j]!) ∧
      (∀ req ∈ pend.reqs, ReqOK ms sch cap p req) ∧ pend.verdict o = true ∧
      ∃ (R : Req) (rest : List Req), pend.reqs = R :: rest ∧ R.level = 0 ∧
        R.positions = (Flock.positions (sch.levels.getD 0 default) (t.rounds[t.k + 6 + sch.k0]!).coins).toList ∧
        rest.length = sch.levels.size - 1 ∧ LaterAt sch t.rounds t.k rest := by
  obtain ⟨pend, initCh, levels, stF, stF', tF, hr, hk, hpl, hcaps, hreqs, -, -, hlanes, hyr, hpos, hfin, hver, hR⟩ :=
    ligerito_refines_reqs ms sch T cap p basisAt r b capD o hdim hlv hα1 ho hwf h
  have hn7 := hdim.1
  obtain ⟨-, -, horig, -⟩ := finalProg_ok hfin
  obtain ⟨rfl, hfc⟩ := final_refines (n := sch.n) b hlanes hyr hpos
    (fun y hy => by rw [hn7]; exact hbasis _ y (by omega) hy) hfin
  refine ⟨pend, hr, hk, hpl, hcaps, hreqs, ?_, hR⟩
  rw [hver, ← hn7]
  exact hfc

/-- **Ligerito on the opening's output refines the model's, every level's request fixed** (§12.4, §13):
`ligerito_accepts_opening_req0`, and the requests after level 0's are `LaterAt`. -/
theorem ligerito_accepts_opening_reqs {D : Type} (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128)
    (sws : Array Switched) (n : ℕ) (op : Model.OpenOut 𝔽) (hop : OpenRel sch.m n T sws op) (cap : Array ByteArray)
    (p : LigeritoProof) (r : ℕ) (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1)
    (ho : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
      (o r ℓ j).1 = ((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf)
    {t t' : Transcript} (hwf : RoundsWF t.rounds)
    (h : Flock.ligerito ms sch T cap p (basis sws) t = .ok ((), t')) :
    ∃ pend : PendL 𝔽 D, t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      Plays fOf (Model.ligeritoC execArith r (schedOf sch) op.T op.b) (ligMsgs p capD sch)
        (consumed t.rounds t.k t'.k) pend ∧
      pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.recursiveCaps[j]!) ∧
      (∀ req ∈ pend.reqs, ReqOK ms sch cap p req) ∧ pend.verdict o = true ∧
      ∃ (R : Req) (rest : List Req), pend.reqs = R :: rest ∧ R.level = 0 ∧
        R.positions = (Flock.positions (sch.levels.getD 0 default) (t.rounds[t.k + 6 + sch.k0]!).coins).toList ∧
        rest.length = sch.levels.size - 1 ∧ LaterAt sch t.rounds t.k rest := by
  rw [hop.T]
  exact ligerito_accepts_reqs ms sch T cap p (basis sws) r op.b capD o hdim hlv hα1 ho
    (fun orig y horig hy => basis_eq hop orig sch.ext horig y hy) hwf h

/-- **One rep refines the model's, every level's request fixed** (§9–§14): `rep_refines_req0`, and the
requests after level 0's are `LaterAt`, Ligerito starting after zerocheck's `3 + (m − 6)` rounds, lincheck's
`3 + (kLog − 6)` and the ring switch's `claims + 1`. -/
theorem rep_refines_reqs {D : Type} (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (h15 : 15 ≤ S.m) (sch : Flock.Schedule) (hmS : st.m = S.m) (hkS : st.kLog = S.kLog)
    (hfold : FoldRealizes st.fold S) (extra : List (Model.Claim 𝔽 S.m)) (sess : RunRecord)
    (hext : ExtraRealizes st S sess.points extra) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1) (hschm : sch.m = S.m)
    (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds) (i r : ℕ) (bytes cap : ByteArray)
    (h : verifyRep spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (p : Proof),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream ∧
      decodeProof (shapeOf st sch extra.length) bytes = .ok p ∧
      cap = p.cap.foldl (· ++ ·) ByteArray.empty ∧ (cap == sess.rootB[i]!) = true ∧
      p.cap.size = 2 ^ (sch.levels.getD 0 default).cap ∧ (∀ b ∈ p.cap.toList, b.size = st.merkle.digestLen) ∧
      ∀ (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D),
        (∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
          (o r ℓ j).1 = ((openAt p.ligerito (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf) →
        ∃ pend : PendL 𝔽 D,
          Plays fOf (Model.repC execArith r S (schedOf sch) hm extra) (repMsgs st sch (2 + extra.length) p capD)
            (stream.rounds.toList.map (·.coins)) pend ∧
          pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.ligerito.recursiveCaps[j]!) ∧
          (∀ req ∈ pend.reqs, ReqOK st.merkle sch p.cap p.ligerito req) ∧ pend.verdict o = true ∧
          ∃ (R : Req) (rest : List Req), pend.reqs = R :: rest ∧ R.level = 0 ∧
            R.positions = (Flock.positions (sch.levels.getD 0 default) (stream.rounds[(3 + (S.m - 6)) +
              (3 + (S.kLog - 6)) + (2 + extra.length + 1) + (sch.k0 + 6)]!).coins).toList ∧
            rest.length = sch.levels.size - 1 ∧
            LaterAt sch stream.rounds ((3 + (S.m - 6)) + (3 + (S.kLog - 6)) + (2 + extra.length + 1)) rest := by
  unfold verifyRep at h
  dsimp only at h
  cases hs : Array.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r)
      sess.streams with
  | none => rw [hs] at h; cases h
  | some stream =>
      rw [hs] at h
      obtain ⟨extras, hex, h⟩ := ebind_ok h
      obtain ⟨p, hp, h⟩ := ebind_ok h
      by_cases c1 : (p.params != pcsParams st.merkle sch) = true
      · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      rw [ite_eq_right c1] at h
      obtain ⟨⟨u, t⟩, hrun, h⟩ := ebind_ok h
      change (if (t.k != stream.rounds.size) = true then _ else _) = _ at h
      by_cases c2 : (t.k != stream.rounds.size) = true
      · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      rw [ite_eq_right c2] at h
      by_cases c3 : (Array.foldl (fun x1 x2 => x1 ++ x2) ByteArray.empty p.cap != sess.rootB[i]!) = true
      · rw [ite_eq_left c3] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      rw [ite_eq_right c3] at h
      cases h
      simp only [bne_iff_ne, ne_eq, not_not] at c2
      have c3' : (Array.foldl (fun x1 x2 => x1 ++ x2) ByteArray.empty p.cap == sess.rootB[i]!) = true := by
        cases hb : (Array.foldl (fun x1 x2 => x1 ++ x2) ByteArray.empty p.cap == sess.rootB[i]!)
        · exact absurd (by simp [bne, hb]) c3
        · rfl
      obtain ⟨hlenx, hcx⟩ := hext extras hex
      have hcapE : p.cap.size = 2 ^ (sch.levels.getD 0 default).cap ∧
          ∀ b ∈ p.cap.toList, b.size = st.merkle.digestLen := by
        rw [StateT.run] at hrun
        obtain ⟨_, _, _, hrun⟩ := bind_ok hrun
        obtain ⟨_, _, _, hrun⟩ := bind_ok hrun
        obtain ⟨⟨_, _⟩, _, _, hlig⟩ := bind_ok hrun
        exact ligerito_cap _ _ _ _ _ _ (hlv 0 (by have := hdim.2.1; omega)).q_ne_zero hlig
      refine ⟨stream, p, rfl, by rw [← hlenx]; exact hp, rfl, c3', hcapE.1, hcapE.2, fun capD o ho => ?_⟩
      have hwf₀ : RoundsWF stream.rounds := hwf stream (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs))
      rw [StateT.run] at hrun
      obtain ⟨zc, t₁, h₁, hrun⟩ := bind_ok hrun
      obtain ⟨lc, t₂, h₂, hrun⟩ := bind_ok hrun
      obtain ⟨⟨T, sws⟩, t₃, h₃, h₄⟩ := bind_ok hrun
      rw [hmS] at h₁
      obtain ⟨hr₁, hk₁, zc', hpl₁, hzc, -⟩ := zerocheck_refines_coins S.m h15 _ _ _ _ _ zc hwf₀ h₁
      have hm₂ := lincheck_moves _ _ _ h₂
      obtain ⟨hr₂, hk₂, ab, hpl₂, hab⟩ := lincheck_refines S st.fold hfold zc zc' hzc p.lincheck t₁ t₂ lc
        (hr₁ ▸ hwf₀) h₂
      rw [hmS, hkS] at h₃
      have hm₃ := ringSwitch_moves _ _ _ (by simp [abcClaims]) h₃
      have hcsz : (abcClaims S.m S.kLog zc lc ++ extras).size = (ab :: Model.cClaim execArith zc' :: extra).length := by
        simp [abcClaims, hlenx]; omega
      obtain ⟨hr₃, hk₃, op, hpl₃, hop⟩ := opening_refines S.m (by omega) _ (ab :: Model.cClaim execArith zc' :: extra)
        hcsz (fun k hk => by
          rcases k with _ | _ | k
          · rw [getElem!_pos _ 0 (by simp [abcClaims]), Array.getElem_append_left (by simp [abcClaims])]
            rw [getElem!_pos _ 0 (by simp [abcClaims])] at hab
            exact hab
          · rw [getElem!_pos _ 1 (by simp [abcClaims]; omega), Array.getElem_append_left (by simp [abcClaims])]
            exact cClaim_rel zc zc' hzc
          · have hk' : k < extra.length := by simp at hk; omega
            have hsz2 : (abcClaims S.m S.kLog zc lc).size = 2 := rfl
            rw [getElem!_pos _ (k + 2) (by rw [Array.size_append, hsz2]; omega),
              Array.getElem_append_right (by rw [hsz2]; omega)]
            simp only [hsz2, Nat.add_sub_cancel]
            rw [← getElem!_pos extras k (by omega)]
            exact hcx k hk')
        (by simp [abcClaims]) (by rw [Array.size_append, show (abcClaims S.m S.kLog zc lc).size = 2 from rfl]; omega)
        p.ringSwitches t₂ t₃ T sws (hr₂ ▸ hr₁ ▸ hwf₀) h₃
      rw [← hschm] at hop
      obtain ⟨pend, hr₄, hk₄, hpl₄, hcaps, hreqs, hver, R, rest, hRe, hRl, hRp, hlenL, hlatL⟩ :=
        ligerito_accepts_opening_reqs
        st.merkle sch T sws _ op hop p.cap p.ligerito r capD o hdim hlv hα1 ho (hr₃ ▸ hr₂ ▸ hr₁ ▸ hwf₀) h₄
      have hsz' : (abcClaims S.m S.kLog zc lc ++ extras).size = 2 + extra.length := by
        simp [abcClaims, hlenx]
      have hidx : t₃.k + 6 + sch.k0 =
          (3 + (S.m - 6)) + (3 + (S.kLog - 6)) + (2 + extra.length + 1) + (sch.k0 + 6) := by
        rw [hm₃.2, hm₂.2, hk₁, hfold.kLog, hsz']
        simp only [Transcript.init]
        omega
      have e₃ : t₃.rounds = stream.rounds := hr₃.trans (hr₂.trans hr₁)
      have hk3 : (3 + (S.m - 6)) + (3 + (S.kLog - 6)) + (2 + extra.length + 1) = t₃.k := by omega
      refine ⟨pend, ?_, hcaps, hreqs, hver, R, rest, hRe, hRl, by rw [hRp, ← hidx, e₃], hlenL,
        by rw [hk3, ← e₃]; exact hlatL⟩
      have hcs : stream.rounds.toList.map (·.coins) =
          consumed stream.rounds 0 t₁.k ++ (consumed t₁.rounds t₁.k t₂.k ++ (consumed t₂.rounds t₂.k t₃.k ++
            consumed t₃.rounds t₃.k t.k)) := by
        have e1 : t₁.rounds = stream.rounds := hr₁
        have e2 : t₂.rounds = stream.rounds := hr₂.trans hr₁
        have e3 : t₃.rounds = stream.rounds := hr₃.trans e2
        rw [e1, e2, e3, ← consumed_append _ hk₃ hk₄, ← consumed_append _ hk₂ (hk₃.trans hk₄),
          ← consumed_append _ (Nat.zero_le t₁.k) (hk₂.trans (hk₃.trans hk₄)), c2]
        exact (consumed_all _).symm
      have hsz : (abcClaims S.m S.kLog zc lc ++ extras).size = 2 + extra.length := by
        simp [abcClaims, hlenx]
      rw [hsz] at hpl₃
      rw [hcs]
      unfold Model.repC
      refine Plays.bind' (ms₂ := lcMsgs S.kLog p.lincheck ++ opMsgs (2 + extra.length) p.ringSwitches ++
        ligMsgs p.ligerito capD sch) hpl₁ ?_ (by simp [repMsgs, hmS, hkS]) rfl
      refine Plays.bind' (ms₂ := opMsgs (2 + extra.length) p.ringSwitches ++ ligMsgs p.ligerito capD sch) hpl₂ ?_
        (by simp) rfl
      exact Plays.bind' hpl₃ hpl₄ rfl rfl

end FlockSoundness.Refine
