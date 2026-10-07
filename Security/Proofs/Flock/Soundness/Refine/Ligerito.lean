import Proofs.Flock.Soundness.Refine.ArithK
import Proofs.Flock.Soundness.Refine.ExecLM
import Proofs.Flock.Soundness.Refine.Schedule

/-!
# Ligerito's rounds refine the model's (PROTOCOL.md §13.3)

The executable's Ligerito verifier carries the running claim `s` and the round polynomial `q` as values; the compiled
model carries them as functions of the openings (`LigStateC`), since the consistency values read the opened rows, which
arrive after the last coin. At the openings the proof carries, the two agree.

Each phase is a pair: the executable's `LM` program, and the model's game, run on the proof's messages and the
consumed rounds' coins.

* `fold_refines`: the folding rounds (`foldRoundsC`, the executable's `challenge` loop).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D : Type}

/-- A sumcheck message of the proof as the model's. -/
def kMsg (m : F256 × F256) : Msg := ⟨𝕂 × 𝕂, (gOf m.1, gOf m.2)⟩

/-- The proof's sumcheck messages `start, …, start + k − 1`. -/
def kMsgs (p : LigeritoProof) (start k : ℕ) : List Msg := (List.range' start k).map fun i => kMsg p.messages[i]!

theorem getElem!_of_getElem? {α : Type} [Inhabited α] {a : Array α} {i : ℕ} {x : α} (h : a[i]? = some x) :
    a[i]! = x := by
  obtain ⟨hi, hx⟩ := Array.getElem?_eq_some_iff.1 h
  rw [getElem!_pos a i hi, hx]

/-- A `K` coin: the round's two coins as `c₀ + c₁·u` (§2.4, §13.2). -/
theorem ofLimbs_coin (c : Array F128) :
    execArith.ofLimbs (fOf c[0]!) (fOf c[1]!) = gOf ⟨c.getD 0 0, c.getD 1 0⟩ := by
  rw [gOf_limbs, getElem!_eq_getD, getElem!_eq_getD]
  rfl

/-- **The folding rounds** (§13.3 step 4 and each recursive round). The executable's `k` challenges, from `stE` and
`t`, are the model's `foldRoundsC`, played with the proof's next `k` sumcheck messages and the `k` rounds consumed; at
the proof's openings `o`, the model's claim and polynomial are the executable's. -/
theorem fold_refines (p : LigeritoProof) (o : Opens 𝔽 D) :
    ∀ (k a : ℕ) (chs chs' : Array F256) (stE stE' : LState) (t t' : Transcript)
      (sM : Opens 𝔽 D → 𝕂) (qM : Opens 𝔽 D → Model.Quad 𝕂), sM o = gOf stE.s → qM o = quadOf stE.q →
      RoundsWF t.rounds →
      forIn (List.range' a k) chs (fun _ chs => challenge p >>= fun x => pure (ForInStep.yield (chs.push x)))
        stE t = .ok ((chs', stE'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + k ∧ stE'.msg = stE.msg + k ∧
        stE'.ood = stE.ood ∧ stE'.qn = stE.qn ∧ stE'.cn = stE.cn ∧ stE'.cbn = stE.cbn ∧ stE'.oods = stE.oods ∧
        stE'.conss = stE.conss ∧ chs'.size = chs.size + k ∧ (∀ i < chs.size, chs'[i]! = chs[i]!) ∧
        ∃ sM' qM', Plays fOf (Model.foldRoundsC execArith k sM qM) (kMsgs p stE.msg k) (consumed t.rounds t.k t'.k)
          (((chs'.extract chs.size).toList.map gOf), sM', qM') ∧ sM' o = gOf stE'.s ∧ qM' o = quadOf stE'.q := by
  intro k
  induction k with
  | zero =>
      intro a chs chs' stE stE' t t' sM qM hs hq _ h
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
      refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, fun _ _ => rfl, sM, qM, ?_, hs, hq⟩
      rw [consumed_self, Model.foldRoundsC]
      simpa [kMsgs] using Plays.pure (ι := fOf) ([], sM, qM)
  | succ k ih =>
      intro a chs chs' stE stE' t t' sM qM hs hq hwf h
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨hlt, hn, m, t₂, hm, hr₂, hk₂, h₁⟩ := challenge_bind h₁
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
      simp only at h₂
      set c := (t.rounds[t.k]!).coins with hc
      set rE : F256 := ⟨c.getD 0 0, c.getD 1 0⟩ with hrE
      have hwf₂ : RoundsWF t₁.rounds := hr₂ ▸ hwf
      have hcsz : c.size = 2 := by rw [hc, RoundsWF.get hwf, hn]
      have hs' : (fun o => (qM o).eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!))) o = gOf (stE.q.eval rE) := by
        simp only; rw [hq, ofLimbs_coin, gOf_eval]
      obtain ⟨hr', hk', hmsg', hood', hqn', hcn', hcbn', hoods', hconss', hsz', hkeep', sM', qM', hpl, hsM', hqM'⟩ :=
        ih (a + 1) (chs.push rE) chs'
          { stE with s := stE.q.eval rE, q := Poly.ofMsg m (stE.q.eval rE), msg := stE.msg + 1 } stE' t₁ t'
          (fun o => (qM o).eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)))
          (fun o => Model.fromMsg (gOf m.1, gOf m.2)
            ((fun o => (qM o).eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!))) o)) hs'
          (by rw [hs', ← quadOf_ofMsg]) hwf₂ h₂
      have hmE : p.messages[stE.msg]! = m := getElem!_of_getElem? hm
      refine ⟨hr'.trans hr₂, by rw [hk', hk₂]; omega, by rw [hmsg']; simp; omega, hood', hqn', hcn', hcbn', hoods',
        hconss', by rw [hsz']; simp; omega, fun i hi => by
          rw [hkeep' i (by simp; omega), getElem!_pos _ i (by simp; omega), Array.getElem_push_lt, getElem!_pos chs i hi],
        sM', qM', ?_, hsM', hqM'⟩
      have hdraw : Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := by
        unfold Model.drawK
        exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c hcsz (by omega)) (Plays.pure _) (by simp) (by simp)
      unfold Model.foldRoundsC
      have hcons : consumed t.rounds t.k t'.k = c :: consumed t₁.rounds t₁.k t'.k := by
        rw [hr₂, hk₂, consumed_append t.rounds (show t.k ≤ t.k + 1 by omega) (by rw [hk', hk₂]; omega),
          consumed_succ]
        rfl
      have hex : (chs'.extract chs.size).toList.map gOf = gOf rE :: ((chs'.extract (chs.push rE).size).toList.map gOf) := by
        have h1 : chs'[chs.size]! = rE := by rw [hkeep' chs.size (by simp)]; simp
        apply List.ext_getElem
        · simp only [List.length_map, Array.length_toList, List.length_cons, Array.size_extract, Array.size_push, hsz']
          omega
        · intro i h₁ h₂
          simp only [List.getElem_map, Array.getElem_toList, Array.getElem_extract]
          rcases i with _ | i
          · simp only [List.getElem_cons_zero, Nat.add_zero]
            rw [← h1, getElem!_pos chs' chs.size (by rw [hsz', Array.size_push]; omega)]
          · simp only [List.getElem_cons_succ, List.getElem_map, Array.getElem_toList, Array.getElem_extract,
              Array.size_push]
            congr 2
            omega
      rw [hcons, hex]
      refine Plays.bind' (ms₂ := kMsgs p stE.msg (k + 1)) (cs₂ := consumed t₁.rounds t₁.k t'.k) hdraw ?_
        (by simp) (by simp)
      refine Plays.bind' (ms₂ := kMsgs p (stE.msg + 1) k) (cs₂ := consumed t₁.rounds t₁.k t'.k)
        (Plays.recv _ (gOf m.1, gOf m.2)) ?_ (by simp [kMsgs, List.range'_succ, kMsg, hmE]) (by simp)
      refine Plays.bind' (ms₂ := []) (cs₂ := []) hpl ?_ (by simp) (by simp)
      rw [ofLimbs_coin]
      exact Plays.pure _

/-- An out-of-domain value of the proof as the model's message. -/
def fMsg (a : F128) : Msg := ⟨𝔽, fOf a⟩

/-- An executable out-of-domain context as the model's. -/
def oodOf (c : Flock.OodCtx) : Model.OodCtx 𝔽 := ⟨c.z.toList.map fOf, fOf c.beta, c.split⟩

theorem ofFn_vecOf (n : ℕ) (c : Array F128) (hc : c.size = n) : List.ofFn (fun i : Fin n => fOf c[i.val]!) =
    c.toList.map fOf := by
  subst hc
  refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
  simp only [List.getElem_ofFn, List.getElem_map, Array.getElem_toList]
  rw [getElem!_pos c i (by simpa using h₁)]

/-- One out-of-domain sample's update of the claim and the polynomial (§13.3 step 6), as the executable computes it. -/
theorem ood_update (sE : F256) (qE : Poly) (y : F128) (m : F256 × F256) (c : Array F128) :
    gOf (sE + (F256.ofF y).smul (c.getD 0 0)) = gOf sE + execArith.embed (fOf c[0]!) * execArith.embed (fOf y) ∧
      quadOf (qE.addScaled (Poly.ofMsg m (F256.ofF y)) (c.getD 0 0)) =
        (quadOf qE).addScaled (Model.fromMsg (gOf m.1, gOf m.2) (execArith.embed (fOf y)))
          (execArith.embed (fOf c[0]!)) := by
  refine ⟨?_, ?_⟩
  · rw [gOf_add, gOf_smul, gOf_ofF, getElem!_eq_getD, mul_comm]; rfl
  · rw [quadOf_addScaled, quadOf_ofMsg, gOf_ofF, getElem!_eq_getD]; rfl

/-- **Two out-of-domain samples** (§13.3 step 6): the model's `twoOodC`, fed the two points' coins, the two values and
intro messages, and the two batching coins, updates the claim and the polynomial as the executable does, and records
the two contexts. -/
theorem twoOodC_plays (o : Opens 𝔽 D) (dim split : ℕ) (sM : Opens 𝔽 D → 𝕂) (qM : Opens 𝔽 D → Model.Quad 𝕂)
    (sE : F256) (qE : Poly) (hs : sM o = gOf sE) (hq : qM o = quadOf qE)
    (z₁ c₁ z₂ c₂ : Array F128) (hz₁ : z₁.size = dim) (hc₁ : c₁.size = 1) (hz₂ : z₂.size = dim) (hc₂ : c₂.size = 1)
    (hd : dim ≠ 1) (y₁ y₂ : F128) (m₁ m₂ : F256 × F256) :
    ∃ sM' qM', Plays fOf (Model.twoOodC execArith dim split sM qM)
      [fMsg y₁, kMsg m₁, fMsg y₂, kMsg m₂] [z₁, c₁, z₂, c₂]
      (sM', qM', [oodOf ⟨z₁, c₁.getD 0 0, some split⟩, oodOf ⟨z₂, c₂.getD 0 0, some split⟩]) ∧
      sM' o = gOf ((sE + (F256.ofF y₁).smul (c₁.getD 0 0)) + (F256.ofF y₂).smul (c₂.getD 0 0)) ∧
      qM' o = quadOf (((qE.addScaled (Poly.ofMsg m₁ (F256.ofF y₁)) (c₁.getD 0 0)).addScaled
        (Poly.ofMsg m₂ (F256.ofF y₂)) (c₂.getD 0 0))) := by
  obtain ⟨u₁, v₁⟩ := ood_update sE qE y₁ m₁ c₁
  obtain ⟨u₂, v₂⟩ := ood_update (sE + (F256.ofF y₁).smul (c₁.getD 0 0))
    (qE.addScaled (Poly.ofMsg m₁ (F256.ofF y₁)) (c₁.getD 0 0)) y₂ m₂ c₂
  refine ⟨fun o' => sM o' + execArith.embed (fOf c₁[0]!) * execArith.embed (fOf y₁) +
      execArith.embed (fOf c₂[0]!) * execArith.embed (fOf y₂),
    fun o' => ((qM o').addScaled (Model.fromMsg (gOf m₁.1, gOf m₁.2) (execArith.embed (fOf y₁)))
      (execArith.embed (fOf c₁[0]!))).addScaled (Model.fromMsg (gOf m₂.1, gOf m₂.2) (execArith.embed (fOf y₂)))
      (execArith.embed (fOf c₂[0]!)), ?_, by simp only; rw [hs, ← u₁, ← u₂], by simp only; rw [hq, v₂, v₁]⟩
  have hctx : [oodOf ⟨z₁, c₁.getD 0 0, some split⟩, oodOf ⟨z₂, c₂.getD 0 0, some split⟩] =
      [⟨List.ofFn (fun i : Fin dim => fOf z₁[i.val]!), fOf c₁[0]!, some split⟩,
        ⟨List.ofFn (fun i : Fin dim => fOf z₂[i.val]!), fOf c₂[0]!, some split⟩] := by
    simp only [oodOf]
    rw [ofFn_vecOf dim z₁ hz₁, ofFn_vecOf dim z₂ hz₂, getElem!_eq_getD c₁ 0, getElem!_eq_getD c₂ 0]
  rw [hctx]
  unfold Model.twoOodC
  refine Plays.bind' (ms₂ := [fMsg y₁, kMsg m₁, fMsg y₂, kMsg m₂]) (cs₂ := [c₁, z₂, c₂]) (Plays.drawVec z₁ hz₁ hd) ?_
    (by simp) (by simp)
  refine Plays.bind' (ms₂ := [kMsg m₁, fMsg y₂, kMsg m₂]) (cs₂ := [c₁, z₂, c₂]) (Plays.recv _ (fOf y₁)) ?_
    (by simp [fMsg]) (by simp)
  refine Plays.bind' (ms₂ := [fMsg y₂, kMsg m₂]) (cs₂ := [c₁, z₂, c₂]) (Plays.recv _ (gOf m₁.1, gOf m₁.2)) ?_
    (by simp [kMsg]) (by simp)
  refine Plays.bind' (ms₂ := [fMsg y₂, kMsg m₂]) (cs₂ := [z₂, c₂]) (Plays.drawOne c₁ hc₁) ?_ (by simp) (by simp)
  refine Plays.bind' (ms₂ := [fMsg y₂, kMsg m₂]) (cs₂ := [c₂]) (Plays.drawVec z₂ hz₂ hd) ?_ (by simp) (by simp)
  refine Plays.bind' (ms₂ := [kMsg m₂]) (cs₂ := [c₂]) (Plays.recv _ (fOf y₂)) ?_ (by simp [fMsg]) (by simp)
  refine Plays.bind' (ms₂ := []) (cs₂ := [c₂]) (Plays.recv _ (gOf m₂.1, gOf m₂.2)) ?_ (by simp [kMsg]) (by simp)
  refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawOne c₂ hc₂) ?_ (by simp) (by simp)
  exact Plays.pure _

/-! ## Query positions (§13.5) -/

theorem shift_or (a b k : ℕ) (hb : b < 2 ^ k) : (a <<< k) ||| b = a * 2 ^ k + b := by
  rw [← Nat.shiftLeft_add_eq_or_of_lt hb, Nat.shiftLeft_eq]

/-- The greedy strata of `Q` sum to `Q`: `Σ 2^{c_i} = Q` (§13.5). -/
theorem strata_go_sum (d : ℕ) : ∀ (fuel rem : ℕ), rem ≤ fuel →
    ((Model.strata.go d fuel rem).map (2 ^ ·)).sum = rem
  | 0, rem, h => by simp [Model.strata.go]; omega
  | fuel + 1, rem, h => by
      unfold Model.strata.go
      split_ifs with h0
      · simp [h0]
      · have hle : 2 ^ min (Nat.log 2 rem) d ≤ rem :=
          (Nat.pow_le_pow_right (by norm_num) (min_le_left _ _)).trans (Nat.pow_log_le_self 2 h0)
        have hpos : 0 < 2 ^ min (Nat.log 2 rem) d := Nat.two_pow_pos _
        rw [List.map_cons, List.sum_cons, strata_go_sum d fuel (rem - 2 ^ min (Nat.log 2 rem) d) (by omega)]
        omega

theorem strata_sum (Q d : ℕ) : ((Model.strata Q d).map (2 ^ ·)).sum = Q := strata_go_sum d Q Q le_rfl

/-- The slots `(c, s)` of the strata, stratum by stratum. -/
def slotsOf (st : List ℕ) : List (ℕ × ℕ) := st.flatMap fun c => (List.range (2 ^ c)).map fun s => (c, s)

theorem slotsOf_length (st : List ℕ) : (slotsOf st).length = (st.map (2 ^ ·)).sum := by
  induction st with
  | nil => rfl
  | cons c st ih => simp_all [slotsOf, List.flatMap_cons]

/-- A slot's position, as the executable computes it. -/
def posAt (d : ℕ) (coins : Array F128) (x : (ℕ × ℕ) × ℕ) : ℕ :=
  (x.1.2 <<< (d - x.1.1)) ||| (coins[x.2]!).lo.toNat % 2 ^ (d - x.1.1)

/-- The executable's positions loop, one stratum: push slot `s`'s position for `s < 2^c`, the counter moving on. -/
theorem positions_inner (d : ℕ) (coins : Array F128) (c : ℕ) :
    ∀ (n a : ℕ) (out : Array ℕ) (j : ℕ),
      ((List.range' a n).foldl (fun (b : Array ℕ × ℕ) s =>
          (b.1.push ((s <<< (d - c)) ||| (coins[b.2]!).lo.toNat % 2 ^ (d - c)), b.2 + 1)) (out, j)).1.toList =
        out.toList ++ (((List.range' a n).map fun s => (c, s)).zipIdx j).map (posAt d coins) ∧
      ((List.range' a n).foldl (fun (b : Array ℕ × ℕ) s =>
          (b.1.push ((s <<< (d - c)) ||| (coins[b.2]!).lo.toNat % 2 ^ (d - c)), b.2 + 1)) (out, j)).2 = j + n
  | 0, a, out, j => by simp
  | n + 1, a, out, j => by
      rw [List.range'_succ, List.foldl_cons]
      obtain ⟨h1, h2⟩ := positions_inner d coins c n (a + 1) _ (j + 1)
      refine ⟨?_, by rw [h2]; omega⟩
      rw [h1]
      simp [List.zipIdx_cons, posAt]

theorem positions_foldl (d : ℕ) (coins : Array F128) : ∀ (st : List ℕ) (out : Array ℕ) (j : ℕ),
    (st.foldl (fun (b : Array ℕ × ℕ) c => (List.range' 0 (2 ^ c)).foldl (fun (b : Array ℕ × ℕ) s =>
        (b.1.push ((s <<< (d - c)) ||| (coins[b.2]!).lo.toNat % 2 ^ (d - c)), b.2 + 1)) b) (out, j)).1.toList =
      out.toList ++ ((slotsOf st).zipIdx j).map (posAt d coins)
  | [], out, j => by simp [slotsOf]
  | c :: st, out, j => by
      rw [List.foldl_cons]
      obtain ⟨h1, h2⟩ := positions_inner d coins c (2 ^ c) 0 out j
      have e : ((List.range' 0 (2 ^ c)).foldl (fun (b : Array ℕ × ℕ) s =>
          (b.1.push ((s <<< (d - c)) ||| (coins[b.2]!).lo.toNat % 2 ^ (d - c)), b.2 + 1)) (out, j)) =
          (((List.range' 0 (2 ^ c)).foldl (fun (b : Array ℕ × ℕ) s =>
            (b.1.push ((s <<< (d - c)) ||| (coins[b.2]!).lo.toNat % 2 ^ (d - c)), b.2 + 1)) (out, j)).1, j + 2 ^ c) :=
        Prod.ext rfl h2
      rw [e, positions_foldl d coins st, h1]
      simp only [slotsOf, List.flatMap_cons, List.zipIdx_append, List.map_append, List.append_assoc,
        List.length_map, List.length_range', List.range_eq_range']

/-- **The query positions** (§13.5): the executable's `positions` is the model's, at every `fast100` level. -/
theorem positions_eq (lv : Flock.Level) (hl : LevelSeam lv) (coins : Array F128) :
    Model.positions execArith (levelOf lv).queries (levelOf lv).logLen (vecOf lv.q coins) =
      (Flock.positions lv coins).toList := by
  obtain ⟨hst, -, -, -, hd⟩ := hl
  have hpos : Flock.positions lv coins = (lv.strata.toList.foldl (fun (b : Array ℕ × ℕ) c =>
      (List.range' 0 (2 ^ c)).foldl (fun (b : Array ℕ × ℕ) s =>
        (b.1.push ((s <<< (lv.d - c)) ||| (coins[b.2]!).lo.toNat % 2 ^ (lv.d - c)), b.2 + 1)) b) (#[], 0)).1 := by
    unfold Flock.positions
    simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', range_size, List.forIn_pure_yield_eq_foldl, pure_bind,
      Array.forIn_pure_yield_eq_foldl, ← Array.foldl_toList]
    rfl
  rw [hpos, positions_foldl]
  dsimp only [levelOf, Model.positions]
  have hd' : ({ logInvRate := lv.rate, queries := lv.q, logCols := lv.cols, lanes := lv.lanes } :
      Accounting.Level).logLen = lv.d := hd.symm
  rw [hd', hst]
  simp only [List.nil_append, slotsOf]
  refine List.map_congr_left fun x hm => ?_
  obtain ⟨⟨c, s⟩, j⟩ := x
  have hj : j < lv.q := by
    have := (List.mem_zipIdx hm).2.1
    have hl := slotsOf_length (strata lv.q lv.d)
    simp only [slotsOf] at hl
    rw [hl, strata_sum] at this
    omega
  simp only [hj, dite_true, vecOf, posAt]
  rw [shift_or _ _ _ (Nat.mod_lt _ (Nat.two_pow_pos _))]
  rfl

/-! ## The consistency value (§13.3 step 10) -/

theorem gOf_foldl_row (fold : Array F256) (row : Array F128) : ∀ (n : ℕ) (acc : F256),
    gOf ((List.range n).foldl (fun acc l => acc + F256.smul (eqTableK fold)[l]! row[l]!) acc) =
      gOf acc + ∑ l ∈ Finset.range n, gOf (eqTableK fold)[l]! * execArith.embed (fOf row[l]!) := by
  intro n
  induction n with
  | zero => intro acc; simp
  | succ n ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, gOf_add, ih, gOf_smul,
        Finset.sum_range_succ, add_assoc]
      rfl

theorem gOf_foldl_cons (g : ℕ → F256) (alpha : Array F128) : ∀ (n : ℕ) (acc : F256),
    gOf ((List.range n).foldl (fun e j => e + F256.smul (g j) (eqTable alpha)[j]!) acc) =
      gOf acc + ∑ j ∈ Finset.range n, gOf (g j) * execArith.embed (fOf (eqTable alpha)[j]!) := by
  intro n
  induction n with
  | zero => intro acc; simp
  | succ n ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, gOf_add, ih, gOf_smul,
        Finset.sum_range_succ, add_assoc]
      rfl

/-- **The consistency value** (§13.3 step 10): the executable's `consE` is the model's `consValueO` on the opened rows,
when every opened row has one entry per folded lane and the consistency coins index every query. -/
theorem consValue_eq (rows : Array (Array F128)) (q : ℕ) (alpha : Array F128) (fold : Array F256) (pos : List ℕ)
    (hpos : pos.length = q) (hrows : ∀ j < q, rows[j]!.size = 2 ^ fold.size) (halpha : q ≤ 2 ^ alpha.size) :
    Model.consValueO execArith (alpha.toList.map fOf) (fold.toList.map gOf) (fun j => rows[j]!.toList.map fOf) pos =
      gOf (consE rows q alpha fold) := by
  unfold consE Model.consValueO Model.foldColumn
  rw [gOf_foldl_cons, gOf_zero, zero_add]
  have hz : (pos.zipIdx).map (fun pj => execArith.embed (Model.eqList (alpha.toList.map fOf) pj.2) *
      ∑ ℓ ∈ Finset.range (2 ^ (fold.toList.map gOf).length), Model.eqList (fold.toList.map gOf) ℓ *
        execArith.embed ((rows[pj.2]!.toList.map fOf).getD ℓ 0)) =
      (List.range q).map fun j => execArith.embed (Model.eqList (alpha.toList.map fOf) j) *
        ∑ ℓ ∈ Finset.range (2 ^ (fold.toList.map gOf).length), Model.eqList (fold.toList.map gOf) ℓ *
          execArith.embed ((rows[j]!.toList.map fOf).getD ℓ 0) := by
    refine List.ext_getElem (by simp [hpos]) fun i h₁ h₂ => ?_
    simp only [List.getElem_map, List.getElem_zipIdx, List.getElem_range, Nat.zero_add]
  rw [hz, FlockLevel3.list_sum_range]
  refine Finset.sum_congr rfl fun j hj => ?_
  rw [Finset.mem_range] at hj
  rw [mul_comm, gOf_foldl_row, gOf_zero, zero_add]
  refine congrArg₂ (· * ·) ?_ ?_
  · rw [List.length_map, Array.length_toList, ← hrows j hj]
    refine Finset.sum_congr rfl fun l hl => ?_
    rw [Finset.mem_range] at hl
    rw [eqTableK_eqList fold l (by rw [← hrows j hj]; exact hl), List.getD_eq_getElem _ _ (by simpa using hl),
      List.getElem_map, Array.getElem_toList, getElem!_pos rows[j]! l hl]
  · rw [eqTable_eqList alpha j (by omega)]

/-! ## Queries and batching (§13.3 steps 7–11) -/

/-- The model's state after `queryAndBatchC` on the coins `cw`, `cα`, `cβ` and, with a message, `m`. -/
noncomputable def qbOut (r : ℕ) (lv : Accounting.Level) (ch : List 𝕂) (start : ℕ) (stM : LigStateC 𝔽 𝕂 D)
    (cw cα cβ : Array F128) (withMessage : Bool) (m : F256 × F256) : LigStateC 𝔽 𝕂 D :=
  let pos := Model.positions execArith lv.queries lv.logLen (vecOf lv.queries cw)
  let e : Opens 𝔽 D → 𝕂 := fun o =>
    Model.consValueO execArith (List.ofFn (vecOf (ceilLog2 lv.queries) cα)) ch (fun j => (o r stM.level j).1) pos
  { stM with
    s := fun o => stM.s o + execArith.embed (fOf cβ[0]!) * e o
    q := if withMessage then fun o => (stM.q o).addScaled (Model.fromMsg (gOf m.1, gOf m.2) (e o))
        (execArith.embed (fOf cβ[0]!)) else stM.q
    cons := stM.cons ++ [⟨lv.logCols, pos, List.ofFn (vecOf (ceilLog2 lv.queries) cα), fOf cβ[0]!, start⟩]
    reqs := stM.reqs ++ [⟨stM.level, pos, ch.length, pathLen lv⟩] }

/-- **Queries and batching**: the model's `queryAndBatchC`, fed the query coins, the consistency coins, the message
when there is one, and the batching coin, ends in `qbOut`. -/
theorem queryAndBatchC_plays (r : ℕ) (lv : Accounting.Level) (ch : List 𝕂) (start : ℕ) (stM : LigStateC 𝔽 𝕂 D)
    (cw cα cβ : Array F128) (hcw : cw.size = lv.queries) (hcα : cα.size = ceilLog2 lv.queries) (hcβ : cβ.size = 1)
    (hq1 : lv.queries ≠ 1) (hα1 : ceilLog2 lv.queries ≠ 1) (withMessage : Bool) (m : F256 × F256) :
    Plays fOf (queryAndBatchC execArith r lv ch start stM withMessage) (if withMessage then [kMsg m] else [])
      [cw, cα, cβ] (qbOut r lv ch start stM cw cα cβ withMessage m) := by
  unfold queryAndBatchC
  refine Plays.bind' (ms₂ := if withMessage then [kMsg m] else []) (cs₂ := [cα, cβ]) (Plays.drawVec cw hcw hq1) ?_
    (by simp) (by simp)
  refine Plays.bind' (ms₂ := if withMessage then [kMsg m] else []) (cs₂ := [cβ]) (Plays.drawVec cα hcα hα1) ?_
    (by simp) (by simp)
  cases withMessage with
  | true =>
      simp only [ite_true]
      refine Plays.bind' (ms₂ := []) (cs₂ := [cβ]) (Plays.recv _ (gOf m.1, gOf m.2)) ?_ (by simp [kMsg]) (by simp)
      refine Plays.bind' (ms₂ := []) (cs₂ := [cβ]) (Plays.pure _) ?_ (by simp) (by simp)
      refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawOne cβ hcβ) ?_ (by simp) (by simp)
      exact Plays.pure _
  | false =>
      simp only [Bool.false_eq_true, ite_false]
      refine Plays.bind' (ms₂ := []) (cs₂ := [cβ]) (Plays.pure _) ?_ (by simp) (by simp)
      refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawOne cβ hcβ) ?_ (by simp) (by simp)
      exact Plays.pure _

/-! ## The recursive rounds (§13.3) -/

/-- **A non-last recursive round of the model** (§13.3), composed from its parts: the folds (`hfold`), the next
level's cap `C'`, the two out-of-domain samples (`htwo`), the queries with their message (`queryAndBatchC_plays`), and
any run of the remaining rounds from the resulting state (`hrec`). -/
theorem iterNL_model (r n : ℕ) (b : ℕ → 𝔽) (initial : List 𝕂) (ext : ℕ) (lv lv' : Accounting.Level)
    (rest : List Accounting.Level) (stM : LigStateC 𝔽 𝕂 D)
    {msF : List Msg} {csF : List (Array F128)} {rs : List 𝕂} {sM' : Opens 𝔽 D → 𝕂} {qM' : Opens 𝔽 D → Model.Quad 𝕂}
    (hfold : Plays fOf (Model.foldRoundsC execArith lv.lanes stM.s stM.q) msF csF (rs, sM', qM'))
    (C' : Cap D) {msO : List Msg} {csO : List (Array F128)} {sT : Opens 𝔽 D → 𝕂} {qT : Opens 𝔽 D → Model.Quad 𝕂}
    {ctxs : List (Model.OodCtx 𝔽)}
    (htwo : Plays fOf (Model.twoOodC execArith (lv'.logCols + lv'.lanes) (stM.levels ++ [rs]).length sM' qM')
      msO csO (sT, qT, ctxs))
    (cw cα cβ : Array F128) (hcw : cw.size = lv.queries) (hcα : cα.size = ceilLog2 lv.queries) (hcβ : cβ.size = 1)
    (hq1 : lv.queries ≠ 1) (hα1 : ceilLog2 lv.queries ≠ 1)
    (mq : F256 × F256) {msR : List Msg} {csR : List (Array F128)} {pend : PendL 𝔽 D}
    (hrec : Plays fOf (Model.recursiveRoundsC execArith r n b initial ext (lv' :: rest)
      { qbOut r lv rs (stM.levels ++ [rs]).length
          { stM with s := sT, q := qT, levels := stM.levels ++ [rs], oods := stM.oods ++ ctxs } cw cα cβ true mq with
        level := stM.level + 1, caps := stM.caps ++ [C'], prevLevel := lv' }) msR csR pend) :
    Plays fOf (Model.recursiveRoundsC execArith r n b initial ext (lv :: lv' :: rest) stM)
      (msF ++ ⟨Cap D, C'⟩ :: msO ++ [kMsg mq] ++ msR) (csF ++ csO ++ [cw, cα, cβ] ++ csR) pend := by
  rw [Model.recursiveRoundsC]
  have hqb := queryAndBatchC_plays r lv rs (stM.levels ++ [rs]).length
    { stM with s := sT, q := qT, levels := stM.levels ++ [rs], oods := stM.oods ++ ctxs } cw cα cβ hcw hcα hcβ hq1 hα1
    true mq
  simp only [ite_true] at hqb
  refine Plays.bind' (ms₂ := ⟨Cap D, C'⟩ :: msO ++ [kMsg mq] ++ msR) (cs₂ := csO ++ [cw, cα, cβ] ++ csR) hfold ?_
    (by simp) (by simp)
  refine Plays.bind' (ms₂ := msO ++ [kMsg mq] ++ msR) (cs₂ := csO ++ [cw, cα, cβ] ++ csR) (Plays.recv _ C') ?_
    (by simp) (by simp)
  refine Plays.bind' (ms₂ := [kMsg mq] ++ msR) (cs₂ := [cw, cα, cβ] ++ csR) htwo ?_ (by simp) (by simp)
  exact Plays.bind' hqb hrec rfl rfl

/-- **The last recursive round of the model** (§13.3, §13.6), composed from its parts: the folds, the clear residual
`yr`, and the final queries (no message). Its verdict is the final check on the resulting state. -/
theorem iterLast_model (r n : ℕ) (b : ℕ → 𝔽) (initial : List 𝕂) (ext : ℕ) (lv : Accounting.Level)
    (stM : LigStateC 𝔽 𝕂 D)
    {msF : List Msg} {csF : List (Array F128)} {rs : List 𝕂} {sM' : Opens 𝔽 D → 𝕂} {qM' : Opens 𝔽 D → Model.Quad 𝕂}
    (hfold : Plays fOf (Model.foldRoundsC execArith lv.lanes stM.s stM.q) msF csF (rs, sM', qM'))
    (yr : Fin (2 ^ (ext + 1)) → 𝔽)
    (cw cα cβ : Array F128) (hcw : cw.size = lv.queries) (hcα : cα.size = ceilLog2 lv.queries) (hcβ : cβ.size = 1)
    (hq1 : lv.queries ≠ 1) (hα1 : ceilLog2 lv.queries ≠ 1) :
    Plays fOf (Model.recursiveRoundsC execArith r n b initial ext [lv] stM)
      (msF ++ [⟨Fin (2 ^ (ext + 1)) → 𝔽, yr⟩]) (csF ++ [cw, cα, cβ])
      (let stF := qbOut r lv rs (stM.levels ++ [rs]).length
          { stM with s := sM', q := qM', levels := stM.levels ++ [rs] } cw cα cβ false (0, 0)
       ⟨fun o => Model.finalCheck execArith n b initial stF.levels stF.oods stF.cons ext yr (stF.s o),
         stF.caps, stF.reqs⟩) := by
  rw [Model.recursiveRoundsC]
  have hqb := queryAndBatchC_plays r lv rs (stM.levels ++ [rs]).length
    { stM with s := sM', q := qM', levels := stM.levels ++ [rs] } cw cα cβ hcw hcα hcβ hq1 hα1 false (0, 0)
  simp only [Bool.false_eq_true, ite_false] at hqb
  refine Plays.bind' (ms₂ := [⟨Fin (2 ^ (ext + 1)) → 𝔽, yr⟩]) (cs₂ := [cw, cα, cβ]) hfold ?_ (by simp) (by simp)
  refine Plays.bind' (ms₂ := []) (cs₂ := [cw, cα, cβ]) (Plays.recv _ yr) ?_ (by simp) (by simp)
  exact Plays.bind' hqb (Plays.pure _) (by simp) (by simp)

/-- **The level loop, composed** (§13.3): iterations `i, …, R − 1` of the executable's loop are the model's rounds
from `i` on (`G i`), when each non-last iteration extends any run of the rounds after it (`hstep`) and the last one is a
run of the last round (`hlast`). -/
theorem loop_compose {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (R : ℕ)
    (G : ℕ → LigStateC 𝔽 𝕂 D → Game (PendL 𝔽 D))
    (Rel : ℕ → σ → LState → Transcript → LigStateC 𝔽 𝕂 D → Prop)
    (tape : ℕ → LState → List Msg) (Final : σ → LState → Transcript → PendL 𝔽 D → Prop)
    (hstep : ∀ i s stE t b stE' t' stM, i + 1 < R → Rel i s stE t stM → body i s stE t = .ok ((b, stE'), t') →
      ∃ s' stM' msS, b = .yield s' ∧ Rel (i + 1) s' stE' t' stM' ∧ t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
        tape i stE = msS ++ tape (i + 1) stE' ∧
        ∀ msR csR pend, Plays fOf (G (i + 1) stM') msR csR pend →
          Plays fOf (G i stM) (msS ++ msR) (consumed t.rounds t.k t'.k ++ csR) pend)
    (hlast : ∀ i s stE t b stE' t' stM, i + 1 = R → Rel i s stE t stM → body i s stE t = .ok ((b, stE'), t') →
      ∃ s' pend, b = .yield s' ∧ t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
        Plays fOf (G i stM) (tape i stE) (consumed t.rounds t.k t'.k) pend ∧ Final s' stE' t' pend) :
    ∀ left i s stE t out stE' t' stM, 0 < left → i + left = R → Rel i s stE t stM →
      forIn (List.range' i left) s body stE t = .ok ((out, stE'), t') →
      ∃ pend, t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
        Plays fOf (G i stM) (tape i stE) (consumed t.rounds t.k t'.k) pend ∧ Final out stE' t' pend
  | 0, _, _, _, _, _, _, _, _, h0, _, _, _ => absurd h0 (lt_irrefl 0)
  | 1, i, s, stE, t, out, stE', t', stM, _, hR, hrel, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨s', pend, rfl, hr, hk, hpl, hfin⟩ := hlast i s stE t b st₁ t₁ stM hR hrel h₁
      simp only [List.range'_zero, List.forIn_nil] at h₂
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₂
      exact ⟨pend, hr, hk, hpl, hfin⟩
  | left + 2, i, s, stE, t, out, stE', t', stM, _, hR, hrel, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨s', stM', msS, rfl, hrel', hr₁, hk₁, htape, hext⟩ := hstep i s stE t b st₁ t₁ stM (by omega) hrel h₁
      obtain ⟨pend, hr, hk, hpl, hfin⟩ :=
        loop_compose body R G Rel tape Final hstep hlast (left + 1) (i + 1) s' st₁ t₁ out stE' t' stM' (by omega)
          (by omega) hrel' h₂
      refine ⟨pend, hr.trans hr₁, hk₁.trans hk, ?_, hfin⟩
      rw [hr₁] at hpl
      rw [htape, consumed_append t.rounds hk₁ hk]
      exact hext _ _ _ hpl

end FlockSoundness.Refine
