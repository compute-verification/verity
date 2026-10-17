import Proofs.Flock.Claims.ZK.Forced
import Proofs.Flock.Claims.ZK.Rewind

/-!
# The Goldreich–Kahan simulator meets `ZKmalR`

`Proto.gk_zk`: for a committed-coin protocol whose dummy-witness run hides the witness until the last coin
(`PrefixHides`, `δ_A`), whose messages at every fixed coin vector are simulated (`SHVZK`, `δ_B`) and whose opening check
binds in its own calls (`Binding`), the Goldreich–Kahan simulator with `N` rewinds meets `ZKmalR` at

  `ε_zk(x) = 1/(N+1) + δ_A(x) + N·δ_B(x)` and `γ(x) = N`.

At each nonce, the hybrids are:

1. the real run (`runStar`);
2. the real run followed by `N` rewinds of the honest prover on fresh tapes at the coins it opened (`yReal`), output as
   the simulator does (`gkOut`): within `1/(N+1)` plus a binding failure among these rewinds (`rewind_close`, with
   `Forced`'s three facts);
3. the same with the SHVZK simulator's messages in each rewind (`ySim`): `N·δ_B` (`hyb_sim`, by `Pr_hybrid`);
4. the same with the dummy-witness run first, which is the simulator (`sim_out`): `δ_A` (`hyb_prefix`).

The binding failure stays at step 2, in the real game: a rewind on the tape `r'` that opens one of the real run's coins
as another is a collision among the honest prover's calls in its runs on `(ν, r)` and `(ν, r')` (`mism_coll`). The two
runs send one `Hello` to one verifier, agree up to that coin's opening, and each checks its opening against the one
commitment (`real_chk_log`). Over `N` rewinds that is at most `N` times `Fork`'s success (`Pr_exists_coord_le`,
`pr_collR_le`).
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

/-! ## A binding failure in a rewind is a collision in the real game -/

theorem fclsGo_mism_inv (H : Msg → Msg) {Coin : Type} [DecidableEq Coin] (chk : Msg → ℕ → Msg → OC Bool)
    (dec : Msg → Coin) (cc : ℕ → Option Coin) (τ : List (Msg × Out)) (ρ : Msg) : ∀ (n j : ℕ),
    ((fclsGo chk dec cc τ ρ j n).eval H).1 = .mism →
      ∃ i < n, (∀ l < i, ∃ m, resp τ (j + l + 1) = some m ∧ ((chk ρ (j + l) m).eval H).1 = true ∧
          cc (j + l) = some (dec m)) ∧
        ∃ m, resp τ (j + i + 1) = some m ∧ ((chk ρ (j + i) m).eval H).1 = true ∧ cc (j + i) ≠ some (dec m)
  | 0, _, h => by simp [fclsGo, OC.eval] at h
  | n + 1, j, h => by
      simp only [fclsGo] at h
      cases hr : resp τ (j + 1) with
      | none => rw [hr] at h; simp [OC.eval] at h
      | some m =>
          rw [hr] at h
          simp only [ocBind_eval] at h
          cases hb : ((chk ρ j m).eval H).1
          · rw [hb] at h; simp [OC.eval] at h
          · rw [hb] at h
            simp only [ite_true] at h
            by_cases hc : cc j = some (dec m)
            · simp only [hc, ite_true] at h
              obtain ⟨i, hi, hpre, m', e1, e2, e3⟩ := fclsGo_mism_inv H chk dec cc τ ρ n (j + 1) h
              refine ⟨i + 1, by omega, fun l hl => ?_, m', ?_⟩
              · cases l with
                | zero => exact ⟨m, by simpa using hr, by simpa using hb, by simpa using hc⟩
                | succ l =>
                    obtain ⟨m'', a, b, d⟩ := hpre l (by omega)
                    rw [show j + (l + 1) = j + 1 + l by omega]
                    exact ⟨m'', a, b, d⟩
              · rw [show j + (i + 1) = j + 1 + i by omega]
                exact ⟨e1, e2, e3⟩
            · exact ⟨0, by omega, fun l hl => absurd hl (by omega), m, by simpa using hr, by simpa using hb,
                by simpa using hc⟩

/-- **A rewind that ended in a binding failure**: it opened `c`'s coins before some coin `t`, and coin `t`'s opening
passed the check against the rewind's commitment and read as another coin. -/
theorem fcls_mism_inv (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (τ : List (Msg × Out)) (h : G.fcls H x ν c τ = .mism) :
    ∃ ρ t m, t < G.K x ∧ resp τ 0 = some ρ ∧
      (∀ l < t, ∃ m', resp τ (l + 1) = some m' ∧ G.chkP H x ν ρ l m' = true ∧ coinAt c l = some (G.dec m')) ∧
      resp τ (t + 1) = some m ∧ G.chkP H x ν ρ t m = true ∧ coinAt c t ≠ some (G.dec m) := by
  unfold fcls fclsOC at h
  cases h0 : resp τ 0 with
  | none => rw [h0] at h; simp [OC.eval] at h
  | some ρ =>
      rw [h0] at h
      obtain ⟨i, hi, hpre, m, e1, e2, e3⟩ := fclsGo_mism_inv H (G.chk x ν) G.dec (coinAt c) τ ρ (G.K x) 0 h
      simp only [Nat.zero_add] at hpre e1 e2 e3
      exact ⟨ρ, i, m, hi, rfl, hpre, e1, e2, e3⟩

/-- **The honest prover checks every opening so far in its next message's calls**: after a transcript's first `t + 2`
rounds, whose answers are the commitment `ρ` and openings that pass the check before coin `t`, the check of coin `t`'s
opening is among them. -/
theorem chk_log_of_hist (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce) (r : G.Rand)
    (τ : List (Msg × Out)) (ρ : Msg) (t : ℕ) (m : Msg) (hlen : t + 2 < τ.length) (h0 : resp τ 0 = some ρ)
    (hpre : ∀ l < t, ∃ m', resp τ (l + 1) = some m' ∧ G.chkP H x ν ρ l m' = true) (hm : resp τ (t + 1) = some m) :
    ((G.chk x ν ρ t m).eval H).2 ⊆ (((G.prover w).next x (histOf (τ.take (t + 2))) (ν, r)).eval H).2 := by
  rcases τ with _ | ⟨⟨a, b⟩, τ'⟩
  · simp at hlen
  have hb : b = .msg ρ := by
    obtain ⟨_, hb⟩ := resp_eq_some.mp h0
    simpa using hb
  subst hb
  simp only [List.length_cons] at hlen
  rw [show t + 2 = (t + 1) + 1 from rfl, List.take_succ_cons]
  set ops := histOf (τ'.take (t + 1)) with hops
  have hlenOps : ops.length = t + 1 := by simp only [hops, histOf, List.length_map, List.length_take]; omega
  have hget : ∀ l (hl : l < (ops.map Prod.snd).length) m', resp τ' l = some m' → (ops.map Prod.snd)[l] = m' := by
    intro l hl m' h
    obtain ⟨hl', h2⟩ := resp_eq_some.mp h
    simp [hops, histOf, h2, msgOf]
  show _ ⊆ ((strat (G.hello x ν) G.abort G.dec (G.chk x ν) (G.msg x w ν r) ((a, ρ) :: ops)).eval H).2
  refine List.Subset.trans ?_ (strat_log_cons H (G.hello x ν) G.abort G.dec (G.chk x ν) (G.msg x w ν r) a ρ ops)
  have hlt : t < (ops.map Prod.snd).length := by simp [hlenOps]
  have := allOk_log H (G.chk x ν ρ) (ops.map Prod.snd) 0 t hlt fun i hi => by
    obtain ⟨m', e1, e2⟩ := hpre i hi
    rw [resp_cons_succ] at e1
    rw [Nat.zero_add, hget i (by omega) m' e1]
    exact e2
  rw [Nat.zero_add, hget t hlt m (by rwa [resp_cons_succ] at hm)] at this
  exact this

/-- **The honest prover's run checks each opening in its own calls**: if the run's commitment is `ρ`, the openings before
coin `t` pass and coin `t`'s opening is `m`, the check of `m` is among the run's calls of the prover. -/
theorem real_chk_log (G : Proto W) (H : Msg → Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (x : Msg) (w : W)
    (ν : G.Nonce) (r : G.Rand) (hk : Vs.k x = G.k x) (ρ : Msg) (t : ℕ) (m : Msg) (ht : t < G.K x)
    (h0 : resp (runStar H (G.prover w) Vs x (ν, r) θ).msgs 0 = some ρ)
    (hpre : ∀ l < t, ∃ m', resp (runStar H (G.prover w) Vs x (ν, r) θ).msgs (l + 1) = some m' ∧
      G.chkP H x ν ρ l m' = true)
    (hm : resp (runStar H (G.prover w) Vs x (ν, r) θ).msgs (t + 1) = some m) :
    ((G.chk x ν ρ t m).eval H).2 ⊆ (runStar H (G.prover w) Vs x (ν, r) θ).pLog := by
  have hlen : t + 2 < (runStar H (G.prover w) Vs x (ν, r) θ).msgs.length := by
    rw [runStar_msgs] at hm ⊢
    exact interact_length_of_resp _ _ (Vs.k x) [] (t + 1) m hm (by rw [hk]; unfold k; omega)
  exact List.Subset.trans (G.chk_log_of_hist H x w ν r _ ρ t m hlen h0 hpre hm)
    (runStar_round_log H (G.prover w) Vs x (ν, r) θ (t + 2) hlen)

/-- Two real runs on one nonce hold a collision among the honest prover's calls. -/
def collR (G : Proto W) (H : Msg → Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (x : Msg) (w : W) (ν : G.Nonce)
    (r r' : G.Rand) : Prop :=
  findColl ((runStar H (G.prover w) Vs x (ν, r) θ).pLog ++ (runStar H (G.prover w) Vs x (ν, r') θ).pLog) ≠ none

/-- **Over the nonce, two real runs' collision is `Fork`'s.** -/
theorem pr_collR_le (G : Proto W) (H : Msg → Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (x : Msg) (w : W) :
    Ex (fun ν => Pr fun p : G.Rand × G.Rand => G.collR H Vs θ x w ν p.1 p.2) ≤ (Fork H (G.prover w) Vs θ x).Pr H :=
  calc Ex (fun ν => Pr fun p : G.Rand × G.Rand => G.collR H Vs θ x w ν p.1 p.2)
      = Pr (fun s : G.Nonce × G.Rand × G.Rand => G.collR H Vs θ x w s.1 s.2.1 s.2.2) :=
        (Pr_prod_left fun s : G.Nonce × G.Rand × G.Rand => G.collR H Vs θ x w s.1 s.2.1 s.2.2).symm
    _ ≤ _ := Fork_pr_ge H (G.prover w) Vs θ x

/-- **A binding failure in a rewind is a collision in the real game**: if the real run on `(ν, r)` opens `c` and the
forced run at `c` with the honest prover's messages on `r'` opens one of `c`'s coins as another, the honest prover's runs
on `(ν, r)` and `(ν, r')` agree up to that coin's opening and check its two openings against one commitment, so
`Binding` puts a collision among their calls. -/
theorem mism_coll (G : Proto W) (H : Msg → Msg) (hC : G.Binding H) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (x : Msg)
    (w : W) (ν : G.Nonce) (r r' : G.Rand) (hk : Vs.k x = G.k x) (c : Fin (G.K x) → G.Coin)
    (hc : G.cls H x ν (runStar H (G.prover w) Vs x (ν, r) θ).msgs = some c)
    (hm : G.fcls H x ν c (interact (fun as => ((Vs.next x as θ).eval H).1)
      (G.pmF H x ν c (G.realVec H x w ν r' c)) (G.k x) []) = .mism) :
    G.collR H Vs θ x w ν r r' := by
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  set τF := interact ask (G.pmF H x ν c (G.realVec H x w ν r' c)) (G.k x) [] with hτF
  have eR : ∀ r₀ : G.Rand,
      (runStar H (G.prover w) Vs x (ν, r₀) θ).msgs = interact ask (G.pmR H x w ν r₀) (G.k x) [] :=
    fun r₀ => by rw [runStar_msgs, hk]; rfl
  obtain ⟨ρ', t, m, ht, h0', hpre, hr, hb, hcm⟩ := G.fcls_mism_inv H x ν c τF hm
  have htake : (interact ask (G.pmR H x w ν r') (G.k x) []).take (t + 1 + 1) = τF.take (t + 1 + 1) :=
    interact_take_congr ask _ _ (G.k x) [] (t + 1) fun i hi _ => by
      simp only [List.nil_append]
      exact G.agree H x w ν r' c _ (G.good_of_TCons H x ν c τF t ⟨ρ', h0', hpre⟩ i hi)
  have hresp : ∀ l ≤ t + 1, resp (runStar H (G.prover w) Vs x (ν, r') θ).msgs l = resp τF l := fun l hl => by
    rw [eR, ← resp_take (n := t + 1 + 1) (by omega), htake, resp_take (by omega)]
  obtain ⟨ρ, h0, hall⟩ := (G.cls_eq_some_iff H x ν _ c).mp hc
  have hρ : ρ' = ρ := by
    have e : resp τF 0 = resp (runStar H (G.prover w) Vs x (ν, r) θ).msgs 0 := by
      rw [eR]
      exact resp_zero_eq ask _ _ (G.K x + 1) (by rw [pmF_nil, pmR_nil])
    rw [h0', h0] at e
    exact Option.some.inj e
  subst hρ
  obtain ⟨m₀, e1, e2, e3⟩ := hall t ht
  have log₀ := G.real_chk_log H Vs θ x w ν r hk ρ' t m₀ ht h0 (fun l hl => by
    obtain ⟨m', a, b, -⟩ := hall l (by omega)
    exact ⟨m', a, b⟩) e1
  have log₁ := G.real_chk_log H Vs θ x w ν r' hk ρ' t m ht (by rw [hresp 0 (by omega)]; exact h0')
    (fun l hl => by
      obtain ⟨m', a, b, -⟩ := hpre l hl
      exact ⟨m', by rw [hresp (l + 1) (by omega)]; exact a, b⟩)
    (by rw [hresp (t + 1) le_rfl]; exact hr)
  have hdec : G.dec m₀ ≠ G.dec m := fun e => hcm (by rw [← e]; exact e3)
  have hcoll := hC x ν ρ' t m₀ m e2 hb hdec
  unfold collR
  rw [findColl_ne_none_iff] at hcoll ⊢
  obtain ⟨p, hp, q, hq, hne, heq⟩ := hcoll
  have sub : ((G.chk x ν ρ' t m₀).eval H).2 ++ ((G.chk x ν ρ' t m).eval H).2 ⊆
      (runStar H (G.prover w) Vs x (ν, r) θ).pLog ++ (runStar H (G.prover w) Vs x (ν, r') θ).pLog :=
    List.append_subset.mpr ⟨List.Subset.trans log₀ (List.subset_append_left _ _),
      List.Subset.trans log₁ (List.subset_append_right _ _)⟩
  exact ⟨p, sub hp, q, sub hq, hne, heq⟩

/-- **The rewinds' binding failures**, from the real run on `(ν, r)`, against the honest prover's messages on the tapes
`α`: each is a collision of the real run with the rewind's tape. -/
theorem gkMM_coll (G : Proto W) (H : Msg → Msg) (hC : G.Binding H) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (x : Msg)
    (w : W) (ν : G.Nonce) (hk : Vs.k x = G.k x) {N : ℕ} (r : G.Rand) (α : Fin N → G.Rand)
    (h : gkMM (G.view0 H x ν (interact (fun as => ((Vs.next x as θ).eval H).1) (G.pmR H x w ν r) (G.k x) []))
      (G.yReal H (fun as => ((Vs.next x as θ).eval H).1) x w ν α)) :
    ∃ i, G.collR H Vs θ x w ν r (α i) := by
  unfold view0 at h
  cases hcls : G.cls H x ν (interact (fun as => ((Vs.next x as θ).eval H).1) (G.pmR H x w ν r) (G.k x) []) with
  | none => rw [hcls] at h; exact h.elim
  | some c =>
      rw [hcls] at h
      obtain ⟨i, hi, -⟩ := h
      refine ⟨i, G.mism_coll H hC Vs θ x w ν r (α i) hk c ?_ hi⟩
      rw [runStar_msgs, hk]
      exact hcls

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

/-- **The Goldreich–Kahan simulator meets `ZKmalR`** at `ε_zk(x) = 1/(N+1) + δ_A(x) + N·δ_B(x)` and `γ(x) = N`, for a
protocol whose dummy-witness run hides the witness until the last coin, whose messages at a fixed coin vector are
simulated and whose opening check binds. -/
theorem gk_zk (G : Proto W) (H : Msg → Msg) (R : Rel W) (N : ℕ) (δA δB : Msg → ℝ) (hA : G.PrefixHides H R δA)
    (hB : G.SHVZK H R δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    ZKmalR H G.k G.prover R (G.simulator N) (fun x => 1 / ((N : ℝ) + 1) + δA x + N * δB x) fun _ => N := by
  intro x w hR Θ Vs θ hk Ev
  beta_reduce
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  have key : ∀ ν : G.Nonce,
      |Pr (fun r : G.Rand => Ev (interact ask (G.pmR H x w ν r) (G.k x) [])) -
          Pr (fun q : G.Rand × (Fin N → G.SimRand x) => Ev ((G.simulator N).out H Vs θ x (ν, q.1, q.2)))| ≤
        (1 / ((N : ℝ) + 1) + δA x + N * δB x) +
          Pr (fun p : G.Rand × G.Rand => G.collR H Vs θ x w ν p.1 p.2) * N := by
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
    have b1 : |Pr (fun p : G.Rand × (Fin N → G.Rand) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.yReal H ask x w ν p.2))) -
        Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2)))| ≤
        N * δB x :=
      G.hyb_sim H R δB hB x (hB0 x) w hR ask ν N
        (fun r => G.view0 H x ν (interact ask (G.pmR H x w ν r) (G.k x) [])) (fun v Y => Ev (gkOut [] v Y)) locOut
    have a1 : |Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2))) -
        Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmD H x ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2)))| ≤
        δA x :=
      G.hyb_prefix H R δA hA x w hR Vs θ hk ν (G.ySim H ask x ν) fun v Y => Ev (gkOut [] v Y)
    have mm : Pr (fun p : G.Rand × (Fin N → G.Rand) =>
          gkMM (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.yReal H ask x w ν p.2)) ≤
        Pr (fun p : G.Rand × G.Rand => G.collR H Vs θ x w ν p.1 p.2) * N := by
      rw [mul_comm]
      exact (Pr_mono fun p h => G.gkMM_coll H hC Vs θ x w ν hk p.1 p.2 h).trans
        (Pr_exists_coord_le N (G.collR H Vs θ x w ν))
    have s3 : Pr (fun q : G.Rand × (Fin N → G.SimRand x) => Ev ((G.simulator N).out H Vs θ x (ν, q.1, q.2))) =
        Pr (fun p : G.Rand × (Fin N → G.SimRand x) =>
          Ev (gkOut [] (G.view0 H x ν (interact ask (G.pmD H x ν p.1) (G.k x) [])) (G.ySim H ask x ν p.2))) :=
      Pr_congr fun q => by rw [G.sim_out H N Vs θ x ν q.1 q.2]
    rw [s3]
    have hc1 := abs_le.mp c1
    have hb1 := abs_le.mp b1
    have ha1 := abs_le.mp a1
    have hmm0 := Pr_nonneg (fun p : G.Rand × (Fin N → G.Rand) =>
      gkMM (G.view0 H x ν (interact ask (G.pmR H x w ν p.1) (G.k x) [])) (G.yReal H ask x w ν p.2))
    rw [abs_le]
    constructor <;> linarith [hc1.1, hc1.2, hb1.1, hb1.2, ha1.1, ha1.2, mm]
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
  rw [eR, eS]
  refine (Ex_abs_sub_le key).trans ?_
  rw [Ex_const_add, Ex_mul_const, mul_comm (Ex _)]
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_left (G.pr_collR_le H Vs θ x w) (Nat.cast_nonneg N))

end Proto

end Claims.ZKMal
