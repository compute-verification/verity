import Proofs.Flock.Claims.ZK.Forced
import Proofs.Flock.Claims.ZK.Rewind

/-!
# The Goldreich–Kahan simulator meets `ZKmalR`

`Proto.gk_zk`: for a committed-coin protocol whose dummy-witness run hides the witness until the last coin
(`PrefixHides`, `δ_A`), whose messages at every fixed coin vector are simulated (`SHVZK`, `δ_B`) and whose opening check
binds in its own calls (`Binding`), the Goldreich–Kahan simulator with `N` rewinds meets `ZKmalR` at

  `ε_zk(x) = 1/(N+1) + 2·δ_A(x) + 2·N·δ_B(x)`,

plus the probability of a collision among the simulator's own SHA-512 calls. At each nonce, the hybrids are:

1. the real run (`runStar`);
2. the real run followed by `N` rewinds of the honest prover on fresh tapes at the coins it opened (`yReal`), output as
   the simulator does (`gkOut`): within `1/(N+1)` plus a binding failure among these rewinds (`rewind_close`, with
   `Forced`'s three facts);
3. the same with the SHVZK simulator's messages in each rewind (`ySim`): `N·δ_B` (`hyb_sim`, by `Pr_hybrid`);
4. the same with the dummy-witness run first, which is the simulator (`sim_out`): `δ_A` (`hyb_prefix`).

The binding failure moves along steps 3 and 4 the same way, and in the simulator it is a collision among its own calls
(`mm_coll`): the extraction run's check of the coin and the rewind's check of the other opening, against one commitment,
since both runs send the same `Hello` to the same verifier.
-/

namespace Claims.ZKMal

open Claims

namespace Proto

variable {W : Type}

/-- The forced run at `c` sending `v`: how it ended, and its transcript. -/
def fRun (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (v : List Msg) : St × List (Msg × Out) :=
  (G.fcls H x ν c (interact ask (G.pmF H x ν c v) (G.k x) []), interact ask (G.pmF H x ν c v) (G.k x) [])

/-- The rewinds with the SHVZK simulator's messages, at every coin vector, on the tapes `s`. -/
def ySim (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce) {N : ℕ}
    (s : Fin N → G.SimRand x) : (Fin (G.K x) → G.Coin) → Fin N → St × List (Msg × Out) :=
  fun c i => G.fRun H ask x ν c ((G.sim x ν c (s i)).eval H).1

/-- The rewinds with the honest prover's messages, at every coin vector, on the fresh tapes `α`. -/
def yReal (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (w : W) (ν : G.Nonce) {N : ℕ}
    (α : Fin N → G.Rand) : (Fin (G.K x) → G.Coin) → Fin N → St × List (Msg × Out) :=
  fun c i => G.fRun H ask x ν c (G.realVec H x w ν (α i) c)

/-! ## What the simulator computes -/

theorem rewindSP_eval (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (s : G.SimRand x) :
    (SP.eval H ask (G.rewindSP x ν c s)).1 = G.fRun H ask x ν c ((G.sim x ν c s).eval H).1 := by
  simp only [rewindSP, spBind_eval, ocSP_eval, spRun_eval, SP.eval]
  rfl

theorem rewindsSP_eval (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) : ∀ (N : ℕ) (s : Fin N → G.SimRand x),
    (SP.eval H ask (G.rewindsSP x ν c N s)).1 = firstSucc [] N (G.ySim H ask x ν s c)
  | 0, _ => by simp [rewindsSP, SP.eval, firstSucc]
  | N + 1, s => by
      simp only [rewindsSP, spBind_eval, rewindSP_eval, firstSucc, ySim]
      by_cases h : (G.fRun H ask x ν c ((G.sim x ν c (s 0)).eval H).1).1 = .succ
      · simp only [h, ite_true, SP.eval]
      · simp only [h, ite_false]
        exact rewindsSP_eval G H ask x ν c N _

/-- **The simulator's output**: the dummy-witness run's transcript if it opened no coin vector, else the first rewind
at its coins that opened them all. -/
theorem sim_out (G : Proto W) (H : Msg → Msg) (N : ℕ) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (x : Msg) (ν : G.Nonce)
    (r : G.Rand) (s : Fin N → G.SimRand x) :
    (G.simulator N).out H Vs θ x (ν, r, s) =
      gkOut [] (G.view0 H x ν (interact (fun as => ((Vs.next x as θ).eval H).1) (G.pmD H x ν r) (G.k x) []))
        (G.ySim H (fun as => ((Vs.next x as θ).eval H).1) x ν s) := by
  simp only [Simulator.out, simulator, spBind_eval, spRun_eval, ocSP_eval]
  unfold view0 cls
  change (SP.eval H _ (match ((clsOC (G.K x) (G.chk x ν) G.dec
    (interact (fun as => ((Vs.next x as θ).eval H).1) (G.pmD H x ν r) (G.k x) [])).eval H).1 with
    | none => _ | some cs => _)).1 = _
  cases ((clsOC (G.K x) (G.chk x ν) G.dec
    (interact (fun as => ((Vs.next x as θ).eval H).1) (G.pmD H x ν r) (G.k x) [])).eval H).1 with
  | none =>
      simp only [SP.eval, gkOut]
      rfl
  | some cs =>
      simp only [gkOut]
      exact G.rewindsSP_eval H _ x ν cs N s

/-! ## A binding failure in the simulator is a collision among its calls -/

theorem fclsGo_mism_inv (H : Msg → Msg) {Coin : Type} [DecidableEq Coin] (chk : Msg → ℕ → Msg → OC Bool)
    (dec : Msg → Coin) (cc : ℕ → Option Coin) (τ : List (Msg × Out)) (ρ : Msg) : ∀ (n j : ℕ),
    ((fclsGo chk dec cc τ ρ j n).eval H).1 = .mism →
      ∃ i < n, ∃ m, resp τ (j + i + 1) = some m ∧ ((chk ρ (j + i) m).eval H).1 = true ∧ cc (j + i) ≠ some (dec m) ∧
        ((chk ρ (j + i) m).eval H).2 ⊆ ((fclsGo chk dec cc τ ρ j n).eval H).2
  | 0, _, h => by simp [fclsGo, OC.eval] at h
  | n + 1, j, h => by
      simp only [fclsGo] at h ⊢
      cases hr : resp τ (j + 1) with
      | none => rw [hr] at h; simp [OC.eval] at h
      | some m =>
          rw [hr] at h
          simp only [ocBind_eval] at h ⊢
          cases hb : ((chk ρ j m).eval H).1
          · rw [hb] at h; simp [OC.eval] at h
          · rw [hb] at h
            simp only [ite_true] at h ⊢
            by_cases hc : cc j = some (dec m)
            · simp only [hc, ite_true] at h ⊢
              obtain ⟨i, hi, m', e1, e2, e3, e4⟩ := fclsGo_mism_inv H chk dec cc τ ρ n (j + 1) h
              refine ⟨i + 1, by omega, m', ?_, ?_, ?_, ?_⟩
              · rw [show j + (i + 1) = j + 1 + i by omega]; exact e1
              · rw [show j + (i + 1) = j + 1 + i by omega]; exact e2
              · rw [show j + (i + 1) = j + 1 + i by omega]; exact e3
              · rw [show j + (i + 1) = j + 1 + i by omega]
                exact List.Subset.trans e4 (List.subset_append_right _ _)
            · simp only [hc, ite_false]
              exact ⟨0, by omega, m, by simpa using hr, by simpa using hb, by simpa using hc,
                by simp [OC.eval]⟩

/-- **A rewind that ended in a binding failure**: some coin's opening passed the check against the rewind's commitment
and read as another coin, and that check is among the classification's calls. -/
theorem fcls_mism_inv (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (τ : List (Msg × Out)) (h : G.fcls H x ν c τ = .mism) :
    ∃ ρ t m, t < G.K x ∧ resp τ 0 = some ρ ∧ resp τ (t + 1) = some m ∧ G.chkP H x ν ρ t m = true ∧
      coinAt c t ≠ some (G.dec m) ∧
      ((G.chk x ν ρ t m).eval H).2 ⊆ ((fclsOC (G.K x) (G.chk x ν) G.dec c τ).eval H).2 := by
  unfold fcls fclsOC at h
  unfold fclsOC
  cases h0 : resp τ 0 with
  | none => rw [h0] at h; simp [OC.eval] at h
  | some ρ =>
      rw [h0] at h
      simp only
      obtain ⟨i, hi, m, e1, e2, e3, e4⟩ := fclsGo_mism_inv H (G.chk x ν) G.dec (coinAt c) τ ρ (G.K x) 0 h
      simp only [Nat.zero_add] at e1 e2 e3 e4
      exact ⟨ρ, i, m, hi, rfl, e1, e2, e3, e4⟩

/-- The rewinds' calls include each rewind's up to the first that opened every coin. -/
theorem rewindsSP_log (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) : ∀ (N : ℕ) (s : Fin N → G.SimRand x) (i : Fin N),
    (∀ j < i, (G.ySim H ask x ν s c j).1 ≠ .succ) →
      spLog H ask (G.rewindSP x ν c (s i)) ⊆ spLog H ask (G.rewindsSP x ν c N s)
  | 0, _, i, _ => i.elim0
  | N + 1, s, i, hlt => by
      simp only [rewindsSP, spBind_log, rewindSP_eval]
      cases i using Fin.cases with
      | zero => exact List.subset_append_left _ _
      | succ i =>
          have h0 : (G.ySim H ask x ν s c 0).1 ≠ .succ := hlt 0 (Fin.succ_pos i)
          simp only [ySim] at h0
          simp only [h0, ite_false]
          refine List.Subset.trans ?_ (List.subset_append_right _ _)
          exact rewindsSP_log G H ask x ν c N (fun j => s j.succ) i fun j hj =>
            hlt j.succ (Fin.succ_lt_succ_iff.mpr hj)

/-- **A binding failure in the simulator is a collision among its own calls.** -/
theorem mm_coll (G : Proto W) (H : Msg → Msg) (N : ℕ) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (x : Msg) (ν : G.Nonce)
    (r : G.Rand) (s : Fin N → G.SimRand x) (hC : G.Binding H)
    (hmm : gkMM (G.view0 H x ν (interact (fun as => ((Vs.next x as θ).eval H).1) (G.pmD H x ν r) (G.k x) []))
      (G.ySim H (fun as => ((Vs.next x as θ).eval H).1) x ν s)) :
    findColl (simLog H (G.simulator N) Vs θ x (ν, r, s)) ≠ none := by
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  set τD := interact ask (G.pmD H x ν r) (G.k x) [] with hτD
  unfold view0 at hmm
  cases hcls : G.cls H x ν τD with
  | none => rw [hcls] at hmm; exact hmm.elim
  | some c =>
      rw [hcls] at hmm
      obtain ⟨i, hi, hlt⟩ := hmm
      -- the simulator's calls
      have hlog : ((clsOC (G.K x) (G.chk x ν) G.dec τD).eval H).2 ++ spLog H ask (G.rewindsSP x ν c N s) ⊆
          simLog H (G.simulator N) Vs θ x (ν, r, s) := by
        unfold simLog
        simp only [simulator, spBind_log, spRun_eval, ocSP_eval, ocSP_log]
        have e : ((clsOC (G.K x) (G.chk x ν) G.dec
            (interact ask (fun h => ((G.dprover.next x h (ν, r)).eval H).1) (G.k x) [])).eval H).1 = some c := hcls
        rw [e]
        exact List.subset_append_right _ _
      -- the rewind's binding failure
      set v := ((G.sim x ν c (s i)).eval H).1 with hv
      set τF := interact ask (G.pmF H x ν c v) (G.k x) [] with hτF
      have hmis : G.fcls H x ν c τF = .mism := hi
      obtain ⟨ρ', t, m, ht, h0', hr, hb, hc⟩ := G.fcls_mism_inv H x ν c τF hmis
      have hlogF : ((fclsOC (G.K x) (G.chk x ν) G.dec c τF).eval H).2 ⊆ spLog H ask (G.rewindsSP x ν c N s) := by
        refine List.Subset.trans ?_ (G.rewindsSP_log H ask x ν c N s i hlt)
        simp only [rewindSP, spBind_log, ocSP_eval, ocSP_log, spRun_eval]
        intro p hp
        simp only [List.mem_append, spLog, List.append_nil]
        exact Or.inr (Or.inr hp)
      -- the extraction run's opening of the same coin
      obtain ⟨ρ, h0, hall⟩ := (G.cls_eq_some_iff H x ν τD c).mp hcls
      obtain ⟨m₀, e1, e2, e3⟩ := hall t ht
      have hρ : ρ' = ρ := by
        have := resp_zero_eq ask (G.pmF H x ν c v) (G.pmD H x ν r) (G.K x + 1)
          (by rw [pmF_nil, pmD_nil])
        have e : resp τF 0 = resp τD 0 := this
        rw [h0', h0] at e
        exact Option.some.inj e
      subst hρ
      have hlog₀ := G.cls_log H x ν τD c hcls ρ' h0 t ht m₀ e1
      have hdec : G.dec m₀ ≠ G.dec m := fun e => hc.1 (by rw [e3, e])
      have hcoll := hC x ν ρ' t m₀ m e2 hb hdec
      rw [findColl_ne_none_iff] at hcoll ⊢
      obtain ⟨p, hp, q, hq, hne, heq⟩ := hcoll
      have sub : ((G.chk x ν ρ' t m₀).eval H).2 ++ ((G.chk x ν ρ' t m).eval H).2 ⊆
          simLog H (G.simulator N) Vs θ x (ν, r, s) := by
        refine List.append_subset.mpr ⟨?_, ?_⟩
        · exact List.Subset.trans hlog₀ (List.Subset.trans (List.subset_append_left _ _) hlog)
        · exact List.Subset.trans hc.2 (List.Subset.trans hlogF
            (List.Subset.trans (List.subset_append_right _ _) hlog))
      exact ⟨p, sub hp, q, sub hq, hne, heq⟩

/-! ## The hybrids -/

theorem viewOf_cls (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) {Ω : Type} (τ : Ω → List (Msg × Out))
    (r : Ω) : viewOf (fun r => G.cls H x ν (τ r)) τ r = G.view0 H x ν (τ r) := by
  unfold viewOf view0
  dsimp only
  cases G.cls H x ν (τ r) <;> rfl

/-- **The rewinds' messages simulated** (step 3), on any event that reads the rewinds only at the first run's coins:
`N·δ_B`. -/
theorem hyb_sim (G : Proto W) (H : Msg → Msg) (R : Rel W) (δB : Msg → ℝ) (hB : G.SHVZK H R δB) (x : Msg)
    (hB0 : 0 ≤ δB x) (w : W) (hR : R x w) (ask : List Msg → Out) (ν : G.Nonce) (N : ℕ)
    (V : G.Rand → List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin))
    (E : List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin) → ((Fin (G.K x) → G.Coin) → Fin N → St × List (Msg × Out)) →
      Prop)
    (hloc : ∀ v Y Y', (∀ c, v = .inr c → Y c = Y' c) → (E v Y ↔ E v Y')) :
    |Pr (fun p : G.Rand × (Fin N → G.Rand) => E (V p.1) (G.yReal H ask x w ν p.2)) -
      Pr (fun p : G.Rand × (Fin N → G.SimRand x) => E (V p.1) (G.ySim H ask x ν p.2))| ≤ N * δB x := by
  rw [Pr_prod_left, Pr_prod_left]
  refine le_trans (Ex_abs_sub_le (b := fun _ => N * δB x) fun r => ?_) (by rw [Ex_const])
  dsimp only
  cases V r with
  | inl t =>
      classical
      have e : ∀ Y, E (.inl t) Y ↔ E (.inl t) fun _ _ => (.abort, []) :=
        fun Y => hloc _ _ _ fun c h => by cases h
      rw [Pr_congr fun α => e (G.yReal H ask x w ν α), Pr_congr fun s => e (G.ySim H ask x ν s), Pr_const,
        Pr_const, sub_self, abs_zero]
      exact mul_nonneg (Nat.cast_nonneg N) hB0
  | inr c =>
      have e1 : ∀ α : Fin N → G.Rand, E (.inr c) (G.yReal H ask x w ν α) ↔
          E (.inr c) fun _ i => G.fRun H ask x ν c (G.realVec H x w ν (α i) c) :=
        fun α => hloc _ _ _ fun c' h => by cases h; rfl
      have e2 : ∀ s : Fin N → G.SimRand x, E (.inr c) (G.ySim H ask x ν s) ↔
          E (.inr c) fun _ i => G.fRun H ask x ν c ((G.sim x ν c (s i)).eval H).1 :=
        fun s => hloc _ _ _ fun c' h => by cases h; rfl
      rw [Pr_congr e1, Pr_congr e2]
      exact Pr_hybrid (fun r => G.realVec H x w ν r c) (fun s => ((G.sim x ν c s).eval H).1) (δB x)
        (fun E' => hB x w hR ν c E') N fun vs => E (.inr c) fun _ i => G.fRun H ask x ν c (vs i)

/-- **The dummy witness in the first run** (step 4), on any event of what it passes on and the rewinds: `δ_A`. -/
theorem hyb_prefix (G : Proto W) (H : Msg → Msg) (R : Rel W) (δA : Msg → ℝ) (hA : G.PrefixHides H R δA)
    (x : Msg) (w : W) (hR : R x w) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = G.k x) (ν : G.Nonce)
    {S Y : Type} [Fintype S] [Nonempty S] (Ys : S → Y) (E : List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin) → Y → Prop) :
    |Pr (fun p : G.Rand × S =>
        E (G.view0 H x ν (interact (fun as => ((Vs.next x as θ).eval H).1) (G.pmR H x w ν p.1) (G.k x) []))
          (Ys p.2)) -
      Pr (fun p : G.Rand × S =>
        E (G.view0 H x ν (interact (fun as => ((Vs.next x as θ).eval H).1) (G.pmD H x ν p.1) (G.k x) []))
          (Ys p.2))| ≤ δA x := by
  rw [Pr_prod_right, Pr_prod_right]
  refine le_trans (Ex_abs_sub_le (b := fun _ => δA x) fun s => ?_) (by rw [Ex_const])
  have := hA x w hR ν Θ Vs θ hk fun u => E u (Ys s)
  simp only [runStar_msgs, hk] at this
  exact this

/-! ## The theorem -/

/-- **The Goldreich–Kahan simulator meets `ZKmalR`** at `ε_zk(x) = 1/(N+1) + 2·δ_A(x) + 2·N·δ_B(x)`, for a protocol
whose dummy-witness run hides the witness until the last coin, whose messages at a fixed coin vector are simulated and
whose opening check binds. -/
theorem gk_zk (G : Proto W) (H : Msg → Msg) (R : Rel W) (N : ℕ) (δA δB : Msg → ℝ) (hA : G.PrefixHides H R δA)
    (hB : G.SHVZK H R δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    ZKmalR H G.k G.prover R (G.simulator N) fun x => 1 / ((N : ℝ) + 1) + 2 * δA x + 2 * N * δB x := by
  intro x w hR Θ Vs θ hk Ev
  beta_reduce
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  have key : ∀ ν : G.Nonce,
      |Pr (fun r : G.Rand => Ev (interact ask (G.pmR H x w ν r) (G.k x) [])) -
          Pr (fun q : G.Rand × (Fin N → G.SimRand x) => Ev ((G.simulator N).out H Vs θ x (ν, q.1, q.2)))| ≤
        (1 / ((N : ℝ) + 1) + 2 * δA x + 2 * N * δB x) +
          Pr (fun q : G.Rand × (Fin N → G.SimRand x) =>
            findColl (simLog H (G.simulator N) Vs θ x (ν, q.1, q.2)) ≠ none) := by
    intro ν
    have c1 : |Pr (fun r : G.Rand => Ev (interact ask (G.pmR H x w ν r) (G.k x) [])) -
        Pr (fun p : G.Rand × (Fin N → G.Rand) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.yReal H ask x w ν p.2)))| ≤
        1 / ((N : ℝ) + 1) + Pr (fun p : G.Rand × (Fin N → G.Rand) =>
          gkMM (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.yReal H ask x w ν p.2)) := by
      have := rewind_close N [] (fun r => G.cls H x ν (interact ask (G.pmR H x w ν r) (G.k x) []))
        (fun r => interact ask (G.pmR H x w ν r) (G.k x) [])
        (fun c r => (G.fRun H ask x ν c (G.realVec H x w ν r c)).1)
        (fun c r => (G.fRun H ask x ν c (G.realVec H x w ν r c)).2)
        (fun c r => G.fcls_succ_iff H ask x w ν r c)
        (fun c r h => (G.real_eq_of_fcls H ask x w ν r c h).symm)
        (fun c r h => G.cls_none_of_abort H ask x w ν r c h) Ev
      simp only [viewOf_cls, Prod.mk.eta] at this
      exact this
    have locOut : ∀ (v : List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin))
        (Y Y' : (Fin (G.K x) → G.Coin) → Fin N → St × List (Msg × Out)),
        (∀ c, v = .inr c → Y c = Y' c) → (Ev (gkOut [] v Y) ↔ Ev (gkOut [] v Y')) := by
      rintro (t | c) Y Y' h
      · rfl
      · simp only [gkOut]; rw [h c rfl]
    have locMM : ∀ (v : List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin))
        (Y Y' : (Fin (G.K x) → G.Coin) → Fin N → St × List (Msg × Out)),
        (∀ c, v = .inr c → Y c = Y' c) → (gkMM v Y ↔ gkMM v Y') := by
      rintro (t | c) Y Y' h
      · rfl
      · simp only [gkMM]; rw [h c rfl]
    have b1 : |Pr (fun p : G.Rand × (Fin N → G.Rand) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.yReal H ask x w ν p.2))) -
        Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2)))| ≤
        N * δB x :=
      G.hyb_sim H R δB hB x (hB0 x) w hR ask ν N
        (fun r => G.view0 H x ν (interact ask (G.pmR H x w ν r) (G.k x) [])) (fun v Y => Ev (gkOut [] v Y)) locOut
    have b2 : |Pr (fun p : G.Rand × (Fin N → G.Rand) =>
          gkMM (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.yReal H ask x w ν p.2)) -
        Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          gkMM (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2))| ≤
        N * δB x :=
      G.hyb_sim H R δB hB x (hB0 x) w hR ask ν N
        (fun r => G.view0 H x ν (interact ask (G.pmR H x w ν r) (G.k x) [])) gkMM locMM
    have a1 : |Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2))) -
        Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmD H x ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2)))| ≤
        δA x :=
      G.hyb_prefix H R δA hA x w hR Vs θ hk ν (G.ySim H ask x ν) fun v Y => Ev (gkOut [] v Y)
    have a2 : |Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          gkMM (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2)) -
        Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          gkMM (G.view0 H x ν (interact ask (G.pmD H x ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2))| ≤
        δA x :=
      G.hyb_prefix H R δA hA x w hR Vs θ hk ν (G.ySim H ask x ν) gkMM
    have m3 : Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          gkMM (G.view0 H x ν (interact ask (G.pmD H x ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2)) ≤
        Pr (fun q : G.Rand × (Fin N → G.SimRand x) =>
          findColl (simLog H (G.simulator N) Vs θ x (ν, q.1, q.2)) ≠ none) :=
      Pr_mono fun p h => G.mm_coll H N Vs θ x ν p.1 p.2 hC h
    have s3 : Pr (fun q : G.Rand × (Fin N → G.SimRand x) => Ev ((G.simulator N).out H Vs θ x (ν, q.1, q.2))) =
        Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmD H x ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2))) :=
      Pr_congr fun q => by rw [G.sim_out H N Vs θ x ν q.1 q.2]
    rw [s3]
    have hc1 := abs_le.mp c1
    have hb1 := abs_le.mp b1
    have hb2 := abs_le.mp b2
    have ha1 := abs_le.mp a1
    have ha2 := abs_le.mp a2
    rw [abs_le]
    constructor <;> linarith [hc1.1, hc1.2, hb1.1, hb1.2, hb2.1, hb2.2, ha1.1, ha1.2, ha2.1, ha2.2, m3]
  have eR : Pr (fun ω : G.Nonce × G.Rand => Ev (runStar H (G.prover w) Vs x ω θ).msgs) =
      Ex fun ν => Pr fun r : G.Rand => Ev (interact ask (G.pmR H x w ν r) (G.k x) []) := by
    rw [Pr_prod_left]
    congr 1
    funext ν
    congr 1
    funext r
    rw [runStar_msgs, hk]
    rfl
  have eS : Pr (fun c : (G.simulator N).C x => Ev ((G.simulator N).out H Vs θ x c)) =
      Ex fun ν => Pr fun q : G.Rand × (Fin N → G.SimRand x) => Ev ((G.simulator N).out H Vs θ x (ν, q.1, q.2)) :=
    Pr_prod_left (α := G.Nonce) (β := G.Rand × (Fin N → G.SimRand x)) fun c => Ev ((G.simulator N).out H Vs θ x c)
  have eC : Pr (fun c : (G.simulator N).C x => findColl (simLog H (G.simulator N) Vs θ x c) ≠ none) =
      Ex fun ν => Pr fun q : G.Rand × (Fin N → G.SimRand x) =>
        findColl (simLog H (G.simulator N) Vs θ x (ν, q.1, q.2)) ≠ none :=
    Pr_prod_left (α := G.Nonce) (β := G.Rand × (Fin N → G.SimRand x))
      fun c => findColl (simLog H (G.simulator N) Vs θ x c) ≠ none
  rw [eR, eS, eC, ← Ex_const_add]
  exact Ex_abs_sub_le key

end Proto

end Claims.ZKMal
