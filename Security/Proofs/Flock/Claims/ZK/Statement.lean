import Proofs.Flock.Claims.Prob

/-!
# Zero knowledge against a malicious verifier: what the proofs add to the spec's `ZKmalR`

The claim is the spec's `Claims.ZKmalR` (the real run lasts the protocol's `k(x)` rounds, and the bound adds the
probability that the simulator's own SHA-512 calls, `simLog`, hold a collision), and Flock's is `Claims.Flock.ZK`, at the
masked verifier's rounds `D.V.k` plus the `Hello` that commits every coin. This module adds:

* **ZK bits** (`ZKmalR.bits`): under `StrictBirthday` at the finder `Bzk` (the simulator, with every call `V*` makes in
  the answers it is asked) the collision term is at most `q²/2^513`.
* **The worst-case cost** (`SimCostLe`), beside the spec's expected `SimCost`, which it implies (`SimCostLe.simCost`).
-/

namespace Claims.ZKMal

open Claims

/-- The calls of a simulator program, its own and those of the verifier `vnext` in every answer it is asked. -/
def spCostV {α : Type} (H : Msg → Msg) (vnext : List Msg → OC Out) : SP α → ℕ
  | .ret _ => 0
  | .hash q k => spCostV H vnext (k (H q)) + 1
  | .ask as k => spCostV H vnext (k ((vnext as).eval H).1) + 1 + ((vnext as).eval H).2.length

/-- **`Bzk`**: the simulator run against `V*`, outputting the first collision among its SHA-512 calls; its cost is its
calls and every call `V*` makes in the answers it is asked. -/
def Bzk {Θ : Type} (H : Msg → Msg) (S : Simulator) (Vs : VStar Θ) (θ : Θ) (x : Msg) : Finder where
  S := S.C x
  out c := findColl (simLog H S Vs θ x c)
  cost c := spCostV H (fun as => Vs.next x as θ) (S.prog x c)

/-- **ZK bits, proposed**: `ZKmalR` with its collision term bounded by the strict birthday bound at `Bzk`, for every
verifier whose run with the simulator makes at most `q` SHA-512 calls. -/
def ZKmalBits {W Ω : Type} [Fintype Ω] (H : Msg → Msg) (k : Msg → ℕ) (P : W → Prover Ω) (R : Rel W) (S : Simulator)
    (εzk : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ q : ℝ,
    (∀ c, ((Bzk H S Vs θ x).cost c : ℝ) ≤ q) → StrictBirthday H (Bzk H S Vs θ x) → ∀ (Ev : List (Msg × Out) → Prop),
    |Pr (fun ω : Ω => Ev (runStar H (P w) Vs x ω θ).msgs) - Pr (fun c : S.C x => Ev (S.out H Vs θ x c))|
      ≤ εzk x + q ^ 2 / 2 ^ (513 : ℝ)

/-- **The simulator's cost, worst case**: at most `bound(x)` calls (hash and ask) on every coin, against every
verifier. -/
def SimCostLe (H : Msg → Msg) (S : Simulator) (bound : Msg → ℕ) : Prop :=
  ∀ x (c : S.C x) (ask : List Msg → Out), ((S.prog x c).eval H ask).2 ≤ bound x

/-- A worst-case bound is an expected one. -/
theorem SimCostLe.simCost {H : Msg → Msg} {S : Simulator} {bound : Msg → ℕ} (h : SimCostLe H S bound) :
    SimCost H S fun x => (bound x : ℝ) := by
  intro x ask
  unfold Ex
  rw [div_le_iff₀ (by exact_mod_cast Fintype.card_pos)]
  calc (∑ c : S.C x, (((S.prog x c).eval H ask).2 : ℝ)) ≤ ∑ _c : S.C x, (bound x : ℝ) :=
        Finset.sum_le_sum fun c _ => by exact_mod_cast h x c ask
    _ = (bound x : ℝ) * Fintype.card (S.C x) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm]

/-! ## What `findColl` finds -/

/-- `findColl` finds a collision exactly when two calls in the log have different inputs and one output. -/
theorem findColl_ne_none_iff : ∀ l : List (Msg × Msg),
    findColl l ≠ none ↔ ∃ p ∈ l, ∃ q ∈ l, p.1 ≠ q.1 ∧ p.2 = q.2
  | [] => by simp [findColl]
  | (u, a) :: l => by
      have ih := findColl_ne_none_iff l
      unfold findColl
      cases hf : l.find? (fun p => decide (p.1 ≠ u) && decide (p.2 = a)) with
      | some p =>
          have hp := List.find?_some hf
          have hm := List.mem_of_find?_eq_some hf
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
          simp only [ne_eq, reduceCtorEq, not_false_eq_true, true_iff]
          exact ⟨(u, a), List.mem_cons_self .., p, List.mem_cons_of_mem _ hm, fun h => hp.1 h.symm, hp.2.symm⟩
      | none =>
          simp only
          rw [ih]
          have hn : ∀ p ∈ l, ¬ (p.1 ≠ u ∧ p.2 = a) := by
            intro p hp h
            have := List.find?_eq_none.mp hf p hp
            simp only [Bool.and_eq_true, decide_eq_true_eq, not_and] at this
            exact this h.1 h.2
          constructor
          · rintro ⟨p, hp, q, hq, h⟩
            exact ⟨p, List.mem_cons_of_mem _ hp, q, List.mem_cons_of_mem _ hq, h⟩
          · rintro ⟨p, hp, q, hq, hne, heq⟩
            rcases List.mem_cons.mp hp with rfl | hp' <;> rcases List.mem_cons.mp hq with rfl | hq'
            · exact absurd rfl hne
            · exact absurd ⟨fun h => hne h.symm, heq.symm⟩ (hn q hq')
            · exact absurd ⟨hne, heq⟩ (hn p hp')
            · exact ⟨p, hp', q, hq', hne, heq⟩

/-- A collision in a log of `H` calls is a collision of `H`. -/
theorem findColl_collision (H : Msg → Msg) : ∀ l : List (Msg × Msg), (∀ p ∈ l, p.2 = H p.1) →
    ∀ u v, findColl l = some (u, v) → Collision H u v
  | [], _, _, _, h => by simp [findColl] at h
  | (q, a) :: l, hl, u, v, h => by
      unfold findColl at h
      cases hf : l.find? (fun p => decide (p.1 ≠ q) && decide (p.2 = a)) with
      | some p =>
          rw [hf] at h
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          have hp := List.find?_some hf
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
          have ha := hl (q, a) (List.mem_cons_self ..)
          have hb := hl p (List.mem_cons_of_mem _ (List.mem_of_find?_eq_some hf))
          exact ⟨fun h => hp.1 h.symm, by simp only at ha; rw [← ha, ← hb, hp.2]⟩
      | none =>
          rw [hf] at h
          exact findColl_collision H l (fun p hp => hl p (List.mem_cons_of_mem _ hp)) u v h

theorem spLog_calls {α : Type} (H : Msg → Msg) (ask : List Msg → Out) :
    ∀ (p : SP α) (e : Msg × Msg), e ∈ spLog H ask p → e.2 = H e.1
  | .ret _, _, he => by simp [spLog] at he
  | .hash q k, e, he => by
      simp only [spLog, List.mem_cons] at he
      rcases he with rfl | he
      · rfl
      · exact spLog_calls H ask (k (H q)) e he
  | .ask as k, e, he => spLog_calls H ask (k (ask as)) e he

theorem spLog_length_le {α : Type} (H : Msg → Msg) (vnext : List Msg → OC Out) :
    ∀ p : SP α, (spLog H (fun as => ((vnext as).eval H).1) p).length ≤ spCostV H vnext p
  | .ret _ => by simp [spLog, spCostV]
  | .hash q k => by
      have := spLog_length_le H vnext (k (H q))
      simp only [spLog, spCostV, List.length_cons]
      omega
  | .ask as k => by
      have := spLog_length_le H vnext (k ((vnext as).eval H).1)
      simp only [spLog, spCostV]
      omega

/-- **`ZKmalR` gives ZK bits**: the collision term is `Bzk`'s success, which the strict birthday bound caps. -/
theorem ZKmalR.bits {W Ω : Type} [Fintype Ω] {H : Msg → Msg} {k : Msg → ℕ} {P : W → Prover Ω} {R : Rel W}
    {S : Simulator} {εzk : Msg → ℝ} (h : ZKmalR H k P R S εzk) : ZKmalBits H k P R S εzk := by
  intro x w hR Θ Vs θ hk q hq hSB Ev
  refine (h x w hR Θ Vs θ hk Ev).trans (add_le_add le_rfl ?_)
  refine le_trans ?_ (hSB q hq)
  unfold Finder.Pr
  refine Pr_mono fun c hc => ?_
  obtain ⟨⟨u, v⟩, huv⟩ := Option.ne_none_iff_exists'.mp hc
  exact ⟨u, v, huv, findColl_collision H _ (spLog_calls H _ _) u v huv⟩

end Claims.ZKMal
