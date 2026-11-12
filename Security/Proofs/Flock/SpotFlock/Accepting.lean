module

public import Proofs.Flock.SpotFlock.KS

/-!
# SPOT[Flock] meets `SampledKSTwo`'s premises, at a knowledge error below `1`, and accepts where Flock does

`Flock.SecurityProofs.SampledKSTwo` (`Proofs.Flock.SamplingGeneric.Guarantees`) bounds the two-stage sampled verifier's
knowledge error at Flock's masked record and SHA-512 from eight premises. At SPOT[Flock]'s code (`SpotFlock.code h`),
at every set of the auditor's copies `h`:

* `ref_numbers`: the reference session statement `xS₀` has `1` to `200` tables, their level-0 lengths summed `2^10` to
  `n·2^23` (`zkOk_sound`, `FlockDeploy.N₀_le`);
* `κF_le`: so the sessions' knowledge error `κ_F = 4·ε_table + 2^{−m′} + 2k′/K′ + e^{−k′/8}` is at most `2^{−190}`;
* `slack_le_one_add`, `slack_sq_le_one_add`: the waits' slack factor `c = 2^{2^{−j}}` at `j = slackExp` is at most
  `1 + 2^{−j}`, and `c² = 2^{2^{1−j}}` at most `1 + 2^{1−j}` (`2^t ≤ 1 + t` on `[0, 1]`);
* `κ_le`: the compiled two-stage verifier's `κ_δ = c²(δ + κ_s) + (1 + c)(2k/K + e^{−k/8})` is at most
  `(1 + 2^{−9})·δ + 2^{−125}` at every statement and every `δ ≥ 0`, with `K = 2^128·k` and `k ≥ 2^j` (`κ_le_of`). The
  slack costs `δ` the factor `c² ≤ 1 + 2^{−9}`, never an additive term that would swamp a small `δ`. So `κ_δ < 1` at
  every `δ ≤ 1/2` (`κ_lt_one`);
* `run_proverOf`, `accepts_pos`: a prover that sends `m₀`, `m₁` and a session statement `xS` and then plays a Flock
  prover `P'` at `xS` (`proverOf`) is decided as `P'` is by Flock's masked verifier on the coins of rounds 3 on, at
  coins whose rounds 0 and 1 pass the check of `m₁` and `xS` (`stmtOk`). So where the check passes at some coins and
  Flock accepts `P'` on some coins, SPOT[Flock] accepts `proverOf m₀ m₁ xS P'` on a positive share of its coins;
* `accepting`: seven of the premises hold, A3 at the coins' bytes among them (`callsUniform`); with the eighth, the
  sessions' openings (`SessionOpensTwo`), `CompiledTwo.KS` holds there, at `κ_δ ≤ (1 + 2^{−9})·δ + 2^{−125}`, and the
  run accepts with positive probability wherever Flock does at a statement the check passes.

So the premises hold together at a verifier whose acceptance is Flock's at its session statements. A witness that builds
the instance needs a `Held`, whose `xS₀_ok` needs a masked Flock statement's bytes decided in Flock's masked scope, and
for a run accepted with positive probability, an accepting run of Flock's masked verifier at a statement the check
passes.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling

variable (h : Held)

/-! ## The knowledge error -/

/-- **The reference statement's numbers**: `1` to `200` tables, their masked level-0 lengths summed `2^10` to `n·2^23`
at `n` tables. -/
theorem ref_numbers : 1 ≤ (shapeOf h.xS₀).length ∧ (shapeOf h.xS₀).length ≤ 200 ∧
    (2 : ℝ) ^ 10 ≤ ((shapeOf h.xS₀).map fun m => (2 : ℝ) ^ (Claims.Flock.zkLevel0 m).d).sum ∧
    ((shapeOf h.xS₀).map fun m => (2 : ℝ) ^ (Claims.Flock.zkLevel0 m).d).sum ≤
      ((shapeOf h.xS₀).length : ℝ) * 2 ^ 23 := by
  obtain ⟨s, spec, sts, hset, h1, h2, -, h4⟩ := zkOk_sound h.xS₀_ok
  have hlen : (shapeOf h.xS₀).length = sts.size := by simp [shapeOf, hset]
  have hN : FlockDeploy.N₀ true h.xS₀ =
      ((shapeOf h.xS₀).map fun m => (2 : ℝ) ^ (Claims.Flock.zkLevel0 m).d).sum := by
    unfold FlockDeploy.N₀ shapeOf
    rw [FlockDeploy.tables_eq hset, hset]
    simp [FlockDeploy.tableLen, List.map_map, Function.comp_def]
  obtain ⟨hlo, hhi⟩ := FlockDeploy.N₀_le hset h1 fun st hst =>
    FlockDeploy.tableLen_zk_bounds st (h4 st hst).1 (h4 st hst).2.1
  rw [← hN, hlen]
  exact ⟨Nat.succ_le_of_lt h1, h2, hlo, hhi⟩

theorem inv_two_pow_eq (k : ℕ) : (2 : ℝ) ^ (-(k : ℝ)) = 2⁻¹ ^ k := by
  rw [Real.rpow_neg (by norm_num), Real.rpow_natCast, inv_pow]

section Numbers

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)

/-- **The sessions' knowledge error is at most `2^{−190}`**: `κ_F` at the reference statement's at most `200` tables,
`m′ = 210`, `K′ = 2^256` and `k′ = ⌈4N₀/e⌉` from `1507` to `200·2^24`, at every statement and every Flock record. -/
theorem κF_le (x : Msg) : Claims.Sampling.κF (SamplingDeploy.deployOf F (code h)) x ≤ 2⁻¹ ^ 190 := by
  obtain ⟨hn1, hn200, hN, hN'⟩ := ref_numbers h
  have hn : ((shapeOf h.xS₀).length : ℝ) ≤ 200 := by exact_mod_cast hn200
  have hk1 : 1507 ≤ (SamplingDeploy.deployOf F (code h)).innerK x := Claims.Flock.Numbers.k_ge hN
  have hk2 : ((SamplingDeploy.deployOf F (code h)).innerK x : ℝ) ≤ 200 * 2 ^ 24 :=
    (Claims.Flock.Numbers.k_le_mul (by exact_mod_cast hn1) (le_trans (by positivity) hN) hN').trans (by nlinarith)
  have he : Real.exp (-((SamplingDeploy.deployOf F (code h)).innerK x : ℝ) / 8) ≤ 2⁻¹ ^ 250 := by
    have := Claims.Flock.Numbers.exp_neg_k_le hk1
    rwa [show (-(250 : ℝ)) = -((250 : ℕ) : ℝ) by norm_num, inv_two_pow_eq] at this
  have e205 : (2 : ℝ) ^ (-(205 : ℝ)) = 2⁻¹ ^ 205 := by
    rw [show (-(205 : ℝ)) = -((205 : ℕ) : ℝ) by norm_num, inv_two_pow_eq]
  have hε : (SamplingDeploy.deployOf F (code h)).εtable x = (shapeOf h.xS₀).length * 2 ^ (-(205 : ℝ)) := rfl
  have hw : (2 : ℝ) ^ (-((SamplingDeploy.deployOf F (code h)).innerWaits x : ℝ)) = 2⁻¹ ^ 210 :=
    inv_two_pow_eq 210
  have hcap : ((SamplingDeploy.deployOf F (code h)).innerCap x : ℝ) = 2 ^ 256 := by
    show ((Claims.Flock.Numbers.cap : ℕ) : ℝ) = 2 ^ 256
    norm_num [Claims.Flock.Numbers.cap]
  have h1 : 4 * (((shapeOf h.xS₀).length : ℝ) * 2⁻¹ ^ 205) ≤ 4 * (200 * 2⁻¹ ^ 205) := by gcongr
  have h3 : 2 * ((SamplingDeploy.deployOf F (code h)).innerK x : ℝ) / 2 ^ 256 ≤ 2 * (200 * 2 ^ 24) / 2 ^ 256 := by
    gcongr
  have hsum : 4 * (200 * (2 : ℝ)⁻¹ ^ 205) + 2⁻¹ ^ 210 + 2 * (200 * 2 ^ 24) / 2 ^ 256 + 2⁻¹ ^ 250 ≤ 2⁻¹ ^ 190 := by
    norm_num
  unfold Claims.Sampling.κF
  rw [hε, e205, hw, hcap]
  linarith

end Numbers

/-- **The waits' slack factor is at most `1 + 2^{−j}`** at `j = slackExp`: `2^t ≤ 1 + t` at `t = 2^{−j} ∈ [0, 1]`. -/
theorem slack_le_one_add : Claims.Sampling.slack ≤ 1 + 2⁻¹ ^ slackExp := by
  have ej : (2 : ℝ) ^ (-(slackExp : ℝ)) = 2⁻¹ ^ slackExp := inv_two_pow_eq slackExp
  have h := rpow_one_add_le_one_add_mul_self (s := 1) (by norm_num) (p := (2 : ℝ) ^ (-(slackExp : ℝ)))
    (by positivity) (by rw [ej]; exact pow_le_one₀ (by norm_num) (by norm_num))
  calc Claims.Sampling.slack = (1 + 1 : ℝ) ^ ((2 : ℝ) ^ (-(slackExp : ℝ))) := by
        simp only [Claims.Sampling.slack, one_add_one_eq_two]
    _ ≤ 1 + (2 : ℝ) ^ (-(slackExp : ℝ)) * 1 := h
    _ = 1 + 2⁻¹ ^ slackExp := by rw [ej, mul_one]

/-- **The slack squared is at most `1 + 2^{1−j}`** at `j = slackExp`: `c² = 2^{2·2^{−j}}`, and `2^t ≤ 1 + t` at
`t = 2·2^{−j} ∈ [0, 1]`. Squaring `slack_le_one_add` instead leaves `2^{−2j}` over. -/
theorem slack_sq_le_one_add : Claims.Sampling.slack ^ 2 ≤ 1 + 2 * 2⁻¹ ^ slackExp := by
  have ej : (2 : ℝ) ^ (-(slackExp : ℝ)) = 2⁻¹ ^ slackExp := inv_two_pow_eq slackExp
  have hp : 2 * (2 : ℝ) ^ (-(slackExp : ℝ)) ≤ 1 := by rw [ej]; norm_num [slackExp]
  have hsq : Claims.Sampling.slack ^ 2 = (2 : ℝ) ^ (2 * (2 : ℝ) ^ (-(slackExp : ℝ))) := by
    unfold Claims.Sampling.slack
    rw [← Real.rpow_natCast, ← Real.rpow_mul (by norm_num)]
    congr 1
    push_cast
    ring
  have h := rpow_one_add_le_one_add_mul_self (s := 1) (by norm_num) (p := 2 * (2 : ℝ) ^ (-(slackExp : ℝ)))
    (by positivity) hp
  rw [hsq, ← ej]
  simpa only [one_add_one_eq_two, mul_one] using h

/-- **`κ_δ ≤ (1 + 2^{−9})·δ + 2^{−125}`** at a sampled record whose sessions' knowledge error is at most `2^{−190}`
and whose waits wait for at least `2^j` accepted runs (`j = slackExp`), at every `δ ≥ 0`:
`c²(δ + κ_F) + (1 + c)(2k/K + e^{−k/8})` at `c² ≤ 1 + 2^{−9}`, `c ≤ 1 + 2^{−10}`, `2k/K = 2^{−127}` and
`e^{−k/8} ≤ e^{−128} ≤ 2^{−128}`. The slack multiplies `δ`; what it adds is at most `2^{−125}`. -/
theorem κ_le_of {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) (x : Msg) {δ : ℝ} (hδ0 : 0 ≤ δ)
    (hF0 : 0 ≤ Claims.Sampling.κF D x) (hF : Claims.Sampling.κF D x ≤ 2⁻¹ ^ 190) (hk : 2 ^ slackExp ≤ D.k x) :
    Claims.Sampling.κ D δ x ≤ (1 + 2⁻¹ ^ 9) * δ + 2⁻¹ ^ 125 := by
  have hj : (2 : ℝ)⁻¹ ^ slackExp = 2⁻¹ ^ 10 := by rw [slackExp]
  have hs := slack_le_one_add
  have hs2 := slack_sq_le_one_add
  rw [hj] at hs hs2
  have hs2' : Claims.Sampling.slack ^ 2 ≤ 1 + 2⁻¹ ^ 9 := by
    have : (2 : ℝ) * 2⁻¹ ^ 10 = 2⁻¹ ^ 9 := by norm_num
    linarith
  have hkr : (1024 : ℝ) ≤ (D.k x : ℝ) := by
    have : 1024 ≤ D.k x := by simpa [slackExp] using hk
    exact_mod_cast this
  have hk0 : (0 : ℝ) < D.k x := by linarith
  have hcap : 2 * (D.k x : ℝ) / D.cap x = 2⁻¹ ^ 127 := by
    rw [Claims.Sampling.Deploy.cap]
    push_cast
    rw [div_eq_iff (mul_pos (by positivity) hk0).ne']
    ring
  have he : Real.exp (-(D.k x : ℝ) / 8) ≤ 2⁻¹ ^ 128 := by
    have h128 : Real.exp (-(D.k x : ℝ) / 8) ≤ Real.exp (-128) := by
      rw [Real.exp_le_exp]
      linarith
    refine h128.trans ?_
    have he2 : (2 : ℝ) ^ 128 ≤ Real.exp 128 := by
      rw [show (128 : ℝ) = ((128 : ℕ) : ℝ) by norm_num, ← Real.exp_one_pow]
      exact pow_le_pow_left₀ (by norm_num) (by linarith [Real.add_one_le_exp (1 : ℝ)]) _
    rw [Real.exp_neg, inv_pow]
    exact inv_anti₀ (by positivity) he2
  unfold Claims.Sampling.κ
  rw [hcap]
  have hA : Claims.Sampling.slack ^ 2 * (δ + Claims.Sampling.κF D x) ≤ (1 + 2⁻¹ ^ 9) * (δ + 2⁻¹ ^ 190) :=
    mul_le_mul hs2' (by linarith) (by linarith) (by positivity)
  have hB : (1 + Claims.Sampling.slack) * (2⁻¹ ^ 127 + Real.exp (-(D.k x : ℝ) / 8)) ≤
      (2 + 2⁻¹ ^ 10) * (2⁻¹ ^ 127 + 2⁻¹ ^ 128) :=
    mul_le_mul (by linarith) (by linarith) (by positivity) (by positivity)
  have hC : (1 + (2 : ℝ)⁻¹ ^ 9) * (δ + 2⁻¹ ^ 190) = (1 + 2⁻¹ ^ 9) * δ + (1 + 2⁻¹ ^ 9) * 2⁻¹ ^ 190 := by ring
  have hD : (1 + (2 : ℝ)⁻¹ ^ 9) * 2⁻¹ ^ 190 + (2 + 2⁻¹ ^ 10) * (2⁻¹ ^ 127 + 2⁻¹ ^ 128) ≤ 2⁻¹ ^ 125 := by
    norm_num
  linarith

/-- **`κ_δ ≤ (1 + 2^{−9})·δ + 2^{−125}`** at SPOT[Flock]'s compiled two-stage verifier, at Flock's masked record,
every statement and every `δ ≥ 0`: its `κ_δ` is at most its sampled record's (`κ_compiled_le`), whose sessions' error
is at most `2^{−190}` (`κF_le`) and whose waits wait for `k = 2^j·(2n_PU + 1) ≥ 2^j` accepted runs (`j = slackExp`). -/
theorem κ_le (x : Msg) {δ : ℝ} (hδ0 : 0 ≤ δ) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h)).κ δ x ≤ (1 + 2⁻¹ ^ 9) * δ + 2⁻¹ ^ 125 :=
  (SamplingDeploy.κ_compiled_le (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots δ x).trans
    (κ_le_of _ x hδ0 (Claims.Sampling.κF_nonneg _ x (SamplingDeploy.deployOf_εtable_nonneg _ (code h) x))
      (κF_le h _ x) (by unfold Claims.Sampling.Deploy.k; exact Nat.le_mul_of_pos_right _ (by omega)))

/-- **`κ_δ < 1` at every `δ ≤ 1/2`**: the bound `CompiledTwo.KS` gives SPOT[Flock] is below `1`. -/
theorem κ_lt_one (x : Msg) {δ : ℝ} (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h)).κ δ x < 1 := by
  have := κ_le h x hδ0
  have h1 : (1 + (2 : ℝ)⁻¹ ^ 9) * δ ≤ (1 + 2⁻¹ ^ 9) * (1 / 2) := mul_le_mul_of_nonneg_left hδ (by positivity)
  have h2 : (1 + (2 : ℝ)⁻¹ ^ 9) * (1 / 2) + 2⁻¹ ^ 125 < 1 := by norm_num
  linarith

/-! ## Acceptance -/

/-- **SPOT[Flock]'s prover from a Flock prover**: rounds 0 to 2 send `m₀`, `m₁` and the session's statement `xS`, and
every later round plays `P'` at `xS` on the history after them. -/
def proverOf {Ω : Type} (m₀ m₁ xS : Msg) (P' : Prover Ω) : Prover Ω :=
  ⟨fun _ hist ω => match hist with
    | [] => .ret m₀
    | [_] => .ret m₁
    | [_, _] => .ret xS
    | _ :: _ :: _ :: hist' => P'.next xS hist' ω⟩

/-- **The session's coins**: Flock's round `i` is the run's round `3 + i`. -/
def shiftCoins {x xS : Msg} (r : Fin ((V h).k x) → FlockDeploy.Coin) (hk : (FlockDeploy.V true).k xS = k₀ h) :
    Fin ((FlockDeploy.V true).k xS) → FlockDeploy.Coin :=
  fun i => r ⟨3 + i.val, by have := i.isLt; show 3 + i.val < 3 + k₀ h; omega⟩

theorem coinSeq_shift {x xS : Msg} (r : Fin ((V h).k x) → FlockDeploy.Coin) (hk : (FlockDeploy.V true).k xS = k₀ h)
    (i : ℕ) : coinSeq r (i + 3) = coinSeq (shiftCoins h r hk) i := by
  unfold coinSeq shiftCoins
  by_cases hi : i < k₀ h
  · rw [dite_eq_left (show i + 3 < (V h).k x by show i + 3 < 3 + k₀ h; omega),
      dite_eq_left (show i < (FlockDeploy.V true).k xS by omega)]
    exact congrArg r (Fin.ext (Nat.add_comm i 3))
  · rw [dite_eq_right (show ¬i + 3 < (V h).k x by show ¬i + 3 < 3 + k₀ h; omega),
      dite_eq_right (show ¬i < (FlockDeploy.V true).k xS by omega)]

/-- **The prover from a Flock prover is decided as Flock decides**: at coins whose rounds 0 and 1 pass the check of
`m₁` and `xS` (`stmtOk`), the run of `proverOf m₀ m₁ xS P'` ends as `P'`'s run of Flock's masked verifier at `xS` on
the coins of rounds 3 on. -/
theorem run_proverOf {Ω : Type} (H : Msg → Msg) (x m₀ m₁ xS : Msg) (P' : Prover Ω) (ω : Ω)
    (r : Fin ((V h).k x) → FlockDeploy.Coin)
    (hok : stmtOk h x m₁ xS (r ⟨0, by show 0 < 3 + k₀ h; omega⟩) (r ⟨1, by show 1 < 3 + k₀ h; omega⟩) = true) :
    (run H (proverOf m₀ m₁ xS P') (V h) x ω r).out =
      (run H P' (FlockDeploy.V true) xS ω (shiftCoins h r (k_of_stmtOk h hok))).out := by
  have hk := k_of_stmtOk h hok
  have e0 : coinSeq r 0 = r ⟨0, by show 0 < 3 + k₀ h; omega⟩ :=
    dite_eq_left (show 0 < (V h).k x by show 0 < 3 + k₀ h; omega)
  have e1 : coinSeq r 1 = r ⟨1, by show 1 < 3 + k₀ h; omega⟩ :=
    dite_eq_left (show 1 < (V h).k x by show 1 < 3 + k₀ h; omega)
  have e2 : (List.range 3).map (coinSeq r) = [coinSeq r 0, coinSeq r 1, coinSeq r 2] := rfl
  let P := proverOf m₀ m₁ xS P'
  let vS : List Msg → OC Out := fun as => (V h).next x as ((List.range as.length).map (coinSeq r))
  let vF : List Msg → OC Out := fun as =>
    (FlockDeploy.V true).next xS as ((List.range as.length).map (coinSeq (shiftCoins h r hk)))
  have hv : ∀ bs : List Msg, bs ≠ [] → vS ([m₀, m₁, xS] ++ bs) = vF bs := by
    intro bs hbs
    obtain ⟨b, bs', rfl⟩ := List.exists_cons_of_ne_nil hbs
    have hl : ([m₀, m₁, xS] ++ b :: bs').length = 3 + (b :: bs').length := by
      simp only [List.length_append, List.length_cons, List.length_nil]
    have hm : (List.range (b :: bs').length).map (coinSeq r ∘ (3 + ·)) =
        (List.range (b :: bs').length).map (coinSeq (shiftCoins h r hk)) :=
      List.map_congr_left fun i _ => by
        show coinSeq r (3 + i) = _
        rw [Nat.add_comm]
        exact coinSeq_shift h r hk i
    have hc : (List.range (([m₀, m₁, xS] ++ b :: bs').length)).map (coinSeq r) =
        coinSeq r 0 :: coinSeq r 1 :: coinSeq r 2 ::
          (List.range (b :: bs').length).map (coinSeq (shiftCoins h r hk)) := by
      rw [hl, List.range_add, List.map_append, e2, List.map_map, hm]
      rfl
    show next h x (m₀ :: m₁ :: xS :: b :: bs') ((List.range (([m₀, m₁, xS] ++ b :: bs').length)).map (coinSeq r)) =
      FlockDeploy.next true xS (b :: bs') ((List.range (b :: bs').length).map (coinSeq (shiftCoins h r hk)))
    rw [hc]
    rfl
  obtain ⟨b₀, h1⟩ : ∃ b₀, vS [m₀] = .ret (.msg b₀) := ⟨_, rfl⟩
  obtain ⟨b₁, h2⟩ : ∃ b₁, vS [m₀, m₁] = .ret (.msg b₁) := ⟨_, rfl⟩
  have h3 : vS [m₀, m₁, xS] = .ret (.msg []) := by
    show OC.ret (if stmtOk h x m₁ xS (coinSeq r 0) (coinSeq r 1) then Out.msg [] else Out.rej) = _
    rw [e0, e1, ite_eq_left hok]
  have hP : ∀ hist, P.next x ([(m₀, b₀), (m₁, b₁), (xS, [])] ++ hist) ω = P'.next xS hist ω := fun _ => rfl
  have hs := stepWith_shift H P P' x xS ω ω vS vF [(m₀, b₀), (m₁, b₁), (xS, [])] [m₀, m₁, xS]
    [(m₀, .msg b₀), (m₁, .msg b₁), (xS, .msg [])] [] [] hP hv (k₀ h) [] [] ⟨[], [], [], .rej⟩
  simp only [List.append_nil, List.nil_append] at hs
  have hrun : ∀ n, n = k₀ h + 1 + 1 + 1 →
      (stepWith H P x ω vS n [] [] ⟨[], [], [], .rej⟩).out =
        (stepWith H P' xS ω vF (k₀ h) [] [] ⟨[], [], [], .rej⟩).out := by
    rintro n rfl
    rw [stepWith_msg H P x ω vS (k₀ h + 1 + 1) [] [] ⟨[], [], [], .rej⟩ b₀ h1]
    show (stepWith H P x ω vS (k₀ h + 1 + 1) [(m₀, b₀)] [m₀] ⟨[(m₀, .msg b₀)], [], [], .rej⟩).out = _
    rw [stepWith_msg H P x ω vS (k₀ h + 1) [(m₀, b₀)] [m₀] ⟨[(m₀, .msg b₀)], [], [], .rej⟩ b₁ h2]
    show (stepWith H P x ω vS (k₀ h + 1) [(m₀, b₀), (m₁, b₁)] [m₀, m₁]
      ⟨[(m₀, .msg b₀), (m₁, .msg b₁)], [], [], .rej⟩).out = _
    rw [stepWith_msg H P x ω vS (k₀ h) [(m₀, b₀), (m₁, b₁)] [m₀, m₁] ⟨[(m₀, .msg b₀), (m₁, .msg b₁)], [], [], .rej⟩
      [] h3]
    show (stepWith H P x ω vS (k₀ h) [(m₀, b₀), (m₁, b₁), (xS, [])] [m₀, m₁, xS]
      ⟨[(m₀, .msg b₀), (m₁, .msg b₁), (xS, .msg [])], [], [], .rej⟩).out = _
    have hs' := congrArg Trace.out hs
    exact hs'
  exact (hrun ((V h).k x) (by show 3 + k₀ h = k₀ h + 1 + 1 + 1; omega)).trans
    (congrArg (fun n => (stepWith H P' xS ω vF n [] [] ⟨[], [], [], .rej⟩).out) hk).symm

/-- **SPOT[Flock] accepts with positive probability where Flock does at a statement its check passes**: if the check
passes `m₁` and `xS` at some coins `c₀` and `c₁` of rounds 0 and 1, and Flock's masked verifier accepts a prover `P'`
at `xS` on some coins, then `proverOf m₀ m₁ xS P'` is accepted on a positive share of SPOT[Flock]'s coins. -/
theorem accepts_pos {Ω : Type} (H : Msg → Msg) (x m₀ m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin)
    (hok : stmtOk h x m₁ xS c₀ c₁ = true) (P' : Prover Ω) (ω : Ω)
    (r' : Fin ((FlockDeploy.V true).k xS) → FlockDeploy.Coin)
    (hacc : (run H P' (FlockDeploy.V true) xS ω r').out = .acc) :
    0 < Pr fun r : Fin ((V h).k x) → FlockDeploy.Coin => (run H (proverOf m₀ m₁ xS P') (V h) x ω r).out = .acc := by
  let r : Fin ((V h).k x) → FlockDeploy.Coin := fun t =>
    if t.val = 0 then c₀ else if t.val = 1 then c₁ else
      if ht : 3 ≤ t.val ∧ t.val - 3 < (FlockDeploy.V true).k xS then r' ⟨t.val - 3, ht.2⟩ else default
  have hr0 : r ⟨0, by show 0 < 3 + k₀ h; omega⟩ = c₀ := ite_eq_left rfl
  have hr1 : r ⟨1, by show 1 < 3 + k₀ h; omega⟩ = c₁ := by
    show (if (1 : ℕ) = 0 then c₀ else if (1 : ℕ) = 1 then c₁ else _) = c₁
    rw [ite_eq_right (show ¬((1 : ℕ) = 0) by decide), ite_eq_left rfl]
  have hok' : stmtOk h x m₁ xS (r ⟨0, by show 0 < 3 + k₀ h; omega⟩) (r ⟨1, by show 1 < 3 + k₀ h; omega⟩) = true := by
    rw [hr0, hr1]
    exact hok
  have hshift : shiftCoins h r (k_of_stmtOk h hok') = r' := by
    funext i
    have hi := i.isLt
    show (if 3 + i.val = 0 then c₀ else if 3 + i.val = 1 then c₁ else
      if ht : 3 ≤ 3 + i.val ∧ 3 + i.val - 3 < (FlockDeploy.V true).k xS then r' ⟨3 + i.val - 3, ht.2⟩ else default) =
        r' i
    rw [ite_eq_right (show ¬3 + i.val = 0 by omega), ite_eq_right (show ¬3 + i.val = 1 by omega),
      dite_eq_left (show 3 ≤ 3 + i.val ∧ 3 + i.val - 3 < (FlockDeploy.V true).k xS by omega)]
    exact congrArg r' (Fin.ext (show 3 + i.val - 3 = i.val by omega))
  have hacc' : (run H (proverOf m₀ m₁ xS P') (V h) x ω r).out = .acc := by
    rw [run_proverOf h H x m₀ m₁ xS P' ω r hok', hshift]
    exact hacc
  exact Claims.SamplingGeneric.Pr_pos_of r hacc'

/-! ## The premises together -/

/-- **SPOT[Flock] meets `SampledKSTwo`'s premises, its bound says something, and it accepts where Flock does**, at every
set of the auditor's copies `h`: A3 at the coins' bytes (`callsUniform`, at both stages), the code's facts
(`RegMatchesTwo`, `Code.RefusesTwo`, `Code.InteriorFixed`), the sessions' runs (an accepted run at coins that check a
unit is Flock's masked verifier accepting the session at its round-2 statement) and their scope (`SessionsIn`); with
the sessions' openings (`SessionOpensTwo`), `CompiledTwo.KS` at Flock's masked record and SHA-512, at
`κ_δ ≤ (1 + 2^{−9})·δ + 2^{−125}` at every statement and `δ ≥ 0`; and a prover is accepted on a positive share of the
coins wherever Flock's masked verifier accepts one at a session statement the check passes (`accepts_pos`). -/
theorem accepting
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    SamplingDeploy.CallsUniform (fun _ => L) (fun _ => coinByte) ∧
      SamplingDeploy.RegMatchesTwo ((base h).atOrder (order h)) (reg h) (two h) ∧
      ((base h).atOrder (order h)).RefusesTwo FlockDeploy.sha512 (two h) ∧
      ((base h).atOrder (order h)).InteriorFixed FlockDeploy.sha512 iroots ∧
      SessionRuns FlockDeploy.sha512 (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) ∧
      SamplingDeploy.SessionsIn (code h) ∧
      (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
        (rs h)).KS FlockDeploy.sha512 ∧
      (∀ x δ, 0 ≤ δ →
        (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
          (rs h)).κ δ x ≤ (1 + 2⁻¹ ^ 9) * δ + 2⁻¹ ^ 125) ∧
      ∀ x m₀ m₁ xS c₀ c₁, stmtOk h x m₁ xS c₀ c₁ = true →
        ∀ (Ω : Type) (P' : Prover Ω) (ω : Ω) (r' : Fin ((FlockDeploy.V true).k xS) → FlockDeploy.Coin),
          (run FlockDeploy.sha512 P' (FlockDeploy.V true) xS ω r').out = .acc →
          0 < Pr fun r : Fin ((code h).V.k x) → FlockDeploy.Coin =>
            (run FlockDeploy.sha512 (proverOf m₀ m₁ xS P') (code h).V x ω r).out = .acc :=
  ⟨callsUniform, regMatchesTwo_code h, refusesTwo_code h _, interiorFixed_code h _,
    sessionRuns h _root_.Flock.Readable.relation FlockDeploy.sha512, sessionsIn_code h,
    ksTwo_spot h hopens, fun x _ hδ0 => κ_le h x hδ0,
    fun x m₀ m₁ xS c₀ c₁ hok _ P' ω r' hacc => accepts_pos h FlockDeploy.sha512 x m₀ m₁ xS c₀ c₁ hok P' ω r' hacc⟩

end SpotFlock
