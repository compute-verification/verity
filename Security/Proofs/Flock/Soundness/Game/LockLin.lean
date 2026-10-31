import Proofs.Flock.Soundness.Game.Lock
import Proofs.Flock.Soundness.LinkFinder

/-!
# The compiled table bound, linearly: a link finder at each slot

`Lock.leafE_off_le` bounds, per slot, the mass of compiled runs that open a position off the table chosen there by
`√(N · Q · forkE)`: the rewinding lemma at the slot's node (`Rewinding.expect_bad_le`, Cauchy–Schwarz over the `N`
positions) and Jensen over the prefixes. Here the same mass is bounded linearly, by the link finder (`LinkFinder`) at a
uniform position:

* `Rewinding.linkG`: at a node, a uniform position `p`, a run, then up to `M` more, stopping at the first that opens `p`
  (`LinkFinder.finderG` at the reading `rdAt op p`). It succeeds (`LinkSucc`) when the first run opens `p` and the
  first further run that opens `p` opens it otherwise: two runs from the node that conflict at `p`
  (`linkSucc_conflict`).
* `Rewinding.expect_bad_le_link`: a run opens a position off a plurality table with probability at most
  `N · Pr[linkG succeeds] + N/(eM)` (`LinkFinder.succ_ge` at each position, then the union bound). The openings per
  run do not enter.
* `Rewinding.expect_linkCost_le`: `linkG` costs at most two runs in expectation, whatever the prover
  (`LinkFinder.expect_finderCost_le` at each position); so under expected-time collision resistance at it the mass is
  at most `N · 2·E[c]/2^256 + N/(eM)` (`expect_bad_le_ecr`).
* `Lock.nodeE`: a functional of the slot node's continuation, averaged over the compiled prefixes (`forkE` is the
  functional `expect₂ Φ`).
* `Lock.leafE_off_le_link`: per slot, the mass off the chosen table is at most `N · nodeE[Pr linkG] + N/(eM)`. The bound
  is linear, so the average over the prefixes needs no Jensen. `Lock.nodeE_linkCost_le`: over the prefixes, the finder
  costs at most two compiled runs.
-/

namespace FlockSoundness

namespace Rewinding

open Game

variable {α P V : Type} [Fintype P] [Fintype V] [DecidableEq V] (op : α → P → Option V)

/-- What a run reads at `p`: its opening there, if any. -/
def rdAt (p : P) (a : α) : Finset V := (op a p).toFinset

/-- **The link finder at a node**: a uniform position `p`, a run, then up to `M` more. -/
def linkG (P : Type) [Fintype P] [Nonempty P] (g : Game α) (M : ℕ) : Game (P × α × List α) :=
  (Game.draw P).bind fun p => (LinkFinder.finderG g M).map fun x => (p, x)

/-- `s` in every run. -/
def linkStrat (P : Type) [Fintype P] [Nonempty P] (g : Game α) (s : Strategy g) (M : ℕ) :
    Strategy (linkG P g M) :=
  Strategy.bind (Game.draw P) _ (drawStrat P) fun p =>
    Strategy.map (fun x => (p, x)) (LinkFinder.finderG g M) (LinkFinder.finderStrat g s M)

/-- **The link finder succeeds**: its first run opens its position `p`, and the first further run that opens `p` opens
it otherwise. -/
def LinkSucc (x : P × α × List α) : Prop := LinkFinder.Succ (rdAt op x.1) x.2

/-- **The link finder's cost**: its first run's, and when that opens `p`, the further runs' up to the first that opens
`p`. -/
noncomputable def linkCost (c : α → ℝ) (x : P × α × List α) : ℝ := LinkFinder.finderCost (rdAt op x.1) c x.2

omit [Fintype P] [Fintype V] [DecidableEq V] in
theorem rdAt_eq_singleton (p : P) (a : α) (v : V) : rdAt op p a = {v} ↔ op a p = some v := by
  unfold rdAt
  cases op a p with
  | none => simp only [Option.toFinset_none, reduceCtorEq, iff_false]; exact (Finset.singleton_ne_empty v).symm
  | some w => simp only [Option.toFinset_some, Finset.singleton_inj, Option.some.injEq]

omit [Fintype P] [Fintype V] [DecidableEq V] in
theorem rdAt_card_le (p : P) (a : α) : (rdAt op p a).card ≤ 1 := by
  unfold rdAt
  cases op a p <;> simp

omit [Fintype P] [Fintype V] [DecidableEq V] in
theorem pRd_rdAt (g : Game α) (s : Strategy g) (p : P) (v : V) :
    LinkFinder.pRd g s (rdAt op p) v = expect (ind fun a => op a p = some v) g s := by
  simp only [LinkFinder.pRd, rdAt_eq_singleton]

omit [Fintype P] [Fintype V] [DecidableEq V] in
theorem sRd_rdAt (g : Game α) (s : Strategy g) (p : P) : LinkFinder.sRd g s (rdAt op p) = 0 := by
  unfold LinkFinder.sRd
  have h : (ind fun a => 2 ≤ (rdAt op p a).card) = fun _ => (0 : ℝ) := funext fun a => by
    have := rdAt_card_le op p a
    simp only [ind, show ¬ 2 ≤ (rdAt op p a).card by omega, ↓reduceIte]
  rw [h, expect_const]

omit [Fintype P] in
/-- The off-table mass at `p`: every value's opening mass, less the table's. -/
theorem expect_off_eq (g : Game α) (s : Strategy g) (vstar : P → V) (p : P) :
    expect (ind fun a => Off op vstar a p) g s =
      (∑ v, expect (ind fun a => op a p = some v) g s) - expect (ind fun a => op a p = some (vstar p)) g s := by
  rw [← expect_sum, ← expect_sub]
  refine congrArg (fun f => expect f g s) (funext fun a => ?_)
  rw [← ind_isSome_eq_sum op p a]
  unfold ind Off
  by_cases h1 : (op a p).isSome <;> by_cases h2 : op a p = some (vstar p) <;> simp_all

omit [Fintype P] [Fintype V] [DecidableEq V] in
/-- **The finder's success is a conflict between two runs from the node**: its first run and the further run it
stopped at open its position differently. -/
theorem linkSucc_conflict (x : P × α × List α) (h : LinkSucc op x) :
    ∃ e ∈ x.2.2, ConflictAt op (x.2.1, e) x.1 := by
  obtain ⟨p, a, aux⟩ := x
  simp only [LinkSucc, LinkFinder.Succ] at h
  rcases h with h2 | ⟨v, hv, e', hn, hne⟩
  · exact absurd ((rdAt_card_le op p a).trans_lt' h2) (by omega)
  · rw [rdAt_eq_singleton] at hv
    unfold LinkFinder.nextRead at hn
    have hmem := List.mem_of_find?_eq_some hn
    have hsome : (rdAt op p e').Nonempty := by simpa using List.find?_some hn
    refine ⟨e', hmem, by simp [hv], ?_, ?_⟩
    · unfold rdAt at hsome
      cases he : op e' p with
      | none => simp [he] at hsome
      | some _ => rfl
    · rw [hv]
      intro heq
      exact hne ((rdAt_eq_singleton op p e' v).2 heq.symm)

/-- The link finder's expectation: the per-position finders', averaged over the positions. -/
theorem expect_linkG [Nonempty P] (g : Game α) (s : Strategy g) (M : ℕ) (F : P × α × List α → ℝ) :
    expect F (linkG P g M) (linkStrat P g s M) =
      (∑ p, expect (fun x => F (p, x)) (LinkFinder.finderG g M) (LinkFinder.finderStrat g s M)) /
        Fintype.card P := by
  unfold linkG linkStrat
  rw [expect_bind, expect_draw]
  congr 1
  exact Finset.sum_congr rfl fun p _ => expect_map _ _ _ _

/-- **Off a plurality table, linearly**: a run opens a position off a plurality table with probability at most `N`
times the link finder's success, plus `N/(eM)`. Where `expect_bad_le` is a root of `N · Q` times two runs' conflict,
this is linear, and the openings per run do not enter. -/
theorem expect_bad_le_link [Nonempty P] (g : Game α) (s : Strategy g) (vstar : P → V)
    (hv : IsPluralityE op g s vstar) (M : ℕ) (hM : 1 ≤ M) :
    expect (ind fun a => ∃ p, Off op vstar a p) g s ≤
      Fintype.card P * expect (ind (LinkSucc op)) (linkG P g M) (linkStrat P g s M) +
        Fintype.card P / (Real.exp 1 * M) := by
  have hpos : ∀ p, expect (ind fun a => Off op vstar a p) g s ≤
      expect (ind (LinkFinder.Succ (rdAt op p))) (LinkFinder.finderG g M) (LinkFinder.finderStrat g s M) +
        1 / (Real.exp 1 * M) := by
    intro p
    have hX : ∀ v, LinkFinder.pRd g s (rdAt op p) v ≤ LinkFinder.pRd g s (rdAt op p) (vstar p) := fun v => by
      rw [pRd_rdAt, pRd_rdAt]; exact hv p v
    have h := LinkFinder.succ_ge g s (rdAt op p) hX M hM
    rw [sRd_rdAt, zero_add] at h
    simp only [pRd_rdAt] at h
    rw [expect_off_eq]
    linarith
  have hunion : expect (ind fun a => ∃ p, Off op vstar a p) g s ≤
      ∑ p, expect (ind fun a => Off op vstar a p) g s := by
    rw [← expect_sum]
    refine expect_mono (fun a => ?_) g s
    by_cases h : ∃ p, Off op vstar a p
    · obtain ⟨p, hp⟩ := h
      calc ind (fun a => ∃ p, Off op vstar a p) a = ind (fun a => Off op vstar a p) a := by
            simp only [ind]; rw [ite_eq_left ⟨p, hp⟩, ite_eq_left hp]
        _ ≤ ∑ p, ind (fun a => Off op vstar a p) a :=
            Finset.single_le_sum (f := fun p => ind (fun a => Off op vstar a p) a)
              (fun p _ => ind_nonneg _ _) (Finset.mem_univ p)
    · rw [show ind (fun a => ∃ p, Off op vstar a p) a = 0 by simp [ind, h]]
      exact Finset.sum_nonneg fun p _ => ind_nonneg _ _
  have hlink := expect_linkG g s M (ind (LinkSucc op))
  have hN : (Fintype.card P : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hs : (∑ p, expect (fun x => ind (LinkSucc op) (p, x)) (LinkFinder.finderG g M)
      (LinkFinder.finderStrat g s M)) = ∑ p, expect (ind (LinkFinder.Succ (rdAt op p)))
        (LinkFinder.finderG g M) (LinkFinder.finderStrat g s M) := rfl
  calc _ ≤ ∑ p, expect (ind fun a => Off op vstar a p) g s := hunion
    _ ≤ ∑ p, (expect (ind (LinkFinder.Succ (rdAt op p))) (LinkFinder.finderG g M)
          (LinkFinder.finderStrat g s M) + 1 / (Real.exp 1 * M)) := Finset.sum_le_sum fun p _ => hpos p
    _ = _ := by
      rw [Finset.sum_add_distrib, hlink, hs, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
        mul_div_cancel₀ _ hN]
      ring

omit [Fintype V] [DecidableEq V] in
/-- **The link finder costs at most two runs** in expectation, at every position and so on average. -/
theorem expect_linkCost_le [Nonempty P] (g : Game α) (s : Strategy g) (c : α → ℝ) (hc : ∀ a, 0 ≤ c a) (M : ℕ) :
    expect (linkCost op c) (linkG P g M) (linkStrat P g s M) ≤ 2 * expect c g s := by
  rw [expect_linkG]
  have hN : (0 : ℝ) < Fintype.card P := by exact_mod_cast Fintype.card_pos
  rw [div_le_iff₀ hN]
  calc ∑ p, expect (fun x => linkCost op c (p, x)) (LinkFinder.finderG g M) (LinkFinder.finderStrat g s M)
      ≤ ∑ _p : P, 2 * expect c g s :=
        Finset.sum_le_sum fun p _ => LinkFinder.expect_finderCost_le g s (rdAt op p) c hc M
    _ = _ := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring

/-- **Off a plurality table under expected-time collision resistance, linearly**: where every success of the link
finder outputs a collision of `Hc` and the finder obeys the generic bound at its cost, a run opens a position off a
plurality table with probability at most `N · 2·E[c]/2^256 + N/(eM)`: weight `N`, two runs. -/
theorem expect_bad_le_ecr {Dh : Type} (Hc : List UInt8 → Dh) [Nonempty P] (g : Game α) (s : Strategy g)
    (vstar : P → V) (hv : IsPluralityE op g s vstar) (M : ℕ) (hM : 1 ≤ M)
    (out : P × α × List α → Option (List UInt8 × List UInt8)) (c : α → ℝ) (hc : ∀ a, 0 ≤ c a)
    (hout : ∀ x, LinkSucc op x → ∃ a b, out x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b)
    (hCR : Assumptions.SHA512CRExpected Hc (linkG P g M) (linkStrat P g s M) out (linkCost op c)) :
    expect (ind fun a => ∃ p, Off op vstar a p) g s ≤
      Fintype.card P * (2 * expect c g s / 2 ^ (256 : ℝ)) + Fintype.card P / (Real.exp 1 * M) := by
  have h1 := expect_bad_le_link op g s vstar hv M hM
  have h2 : expect (ind (LinkSucc op)) (linkG P g M) (linkStrat P g s M) ≤
      (prob (fun x => ∃ a b, out x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b) (linkG P g M)
        (linkStrat P g s M)).toReal := by
    rw [prob_toReal]
    exact expect_mono (ind_mono hout) _ _
  have h3 : expect (linkCost op c) (linkG P g M) (linkStrat P g s M) / 2 ^ (256 : ℝ) ≤
      2 * expect c g s / 2 ^ (256 : ℝ) :=
    div_le_div_of_nonneg_right (expect_linkCost_le op g s c hc M) (by positivity)
  have hN : (0 : ℝ) ≤ Fintype.card P := Nat.cast_nonneg _
  have h4 := mul_le_mul_of_nonneg_left (h2.trans (hCR.trans h3)) hN
  linarith

end Rewinding

namespace Game

namespace Lock

variable {α β : Type} {T D I : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}

/-- **A functional of the slot node's continuation, over the compiled prefixes**: `Ψ d g'' σ''` at the slot's node,
with the cap `d`, the continuation `g''` and its strategy `σ''`; `0` on runs that do not reach the slot. `forkE` is the
functional `expect₂ Φ`. -/
noncomputable def nodeE (plur : Chooser T D β) (slot : ℕ) (Ψ : D → (g : Game β) → Strategy g → ℝ) : {i : I} →
    {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Strategy g' → ℝ
  | _, _, _, .ret _, _ => 0
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF _ _ _ L, σ' =>
      (∑ c : C, nodeE plur slot Ψ (L c) (σ' c)) / (@Fintype.card C iF)
  | _, _, _, .send _ L, σ' => nodeE plur slot Ψ (L σ'.1) σ'.2
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ' =>
      if s = slot then Ψ σ'.1 (k' σ'.1) σ'.2
      else nodeE plur slot Ψ (L (plur s σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2

theorem nodeE_mono (plur : Chooser T D β) (slot : ℕ) {Ψ Ψ' : D → (g : Game β) → Strategy g → ℝ}
    (h : ∀ d g σ, Ψ d g σ ≤ Ψ' d g σ) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      nodeE plur slot Ψ L σ' ≤ nodeE plur slot Ψ' L σ'
  | _, _, _, .ret _, _ => le_rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [nodeE]
      exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => nodeE_mono plur slot h (L c) (σ' c))
        (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ' => nodeE_mono plur slot h (L σ'.1) σ'.2
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ' => by
      simp only [nodeE]
      split_ifs
      · exact h _ _ _
      · exact nodeE_mono plur slot h _ σ'.2

/-- **A continuation's expectation at the slot is at most the run's**: for a nonnegative leaf function, the node's
continuation is the rest of the run. -/
theorem nodeE_le_expect (plur : Chooser T D β) (slot : ℕ) (c : β → ℝ) (hc : ∀ b, 0 ≤ c b) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      nodeE plur slot (fun _ g'' σ'' => expect c g'' σ'') L σ' ≤ expect c g' σ'
  | _, _, _, @Lock.ret _ _ _ _ _ _ _ _ _ b _, _ => hc b
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [nodeE, expect]
      exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun c' _ => nodeE_le_expect plur slot c hc (L c') (σ' c'))
        (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ' => nodeE_le_expect plur slot c hc (L σ'.1) σ'.2
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ' => by
      simp only [nodeE, expect]
      split_ifs
      · exact le_rfl
      · exact nodeE_le_expect plur slot c hc _ σ'.2

/-- **Rewinding at a slot, linearly.** If the table chosen at `slot` (with cap `d`) always reads, through `row`, as a
plurality of the slot's openings `op d` over the continuation, then the probability that a run opens a position off the
table fixed at `slot` is at most `N` times the link finder's success at the slot's node, over the prefixes, plus
`N/(eM)`. Where `leafE_off_le` takes a root at the node and Jensen over the prefixes, this is linear at both. -/
theorem leafE_off_le_link {P V : Type} [Fintype P] [Nonempty P] [Fintype V] [DecidableEq V]
    (op : D → β → P → Option V) (row : T → P → V) (slot : ℕ) (plur : Chooser T D β)
    (hplur : ∀ (d : D) (g'' : Game β) (σ'' : Strategy g''),
      Rewinding.IsPluralityE (op d) g'' σ'' (row (plur slot d g'' σ''))) (M : ℕ) (hM : 1 ≤ M) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      get i slot = none →
        leafE plur (fun i b => ind (fun b => ∃ t d, get i slot = some (t, d) ∧
            ∃ p, Rewinding.Off (op d) (row t) b p) b) L σ' ≤
          Fintype.card P * nodeE plur slot (fun d g'' σ'' => expect (ind (Rewinding.LinkSucc (op d)))
              (Rewinding.linkG P g'' M) (Rewinding.linkStrat P g'' σ'' M)) L σ' +
            Fintype.card P / (Real.exp 1 * M)
  | _, _, _, @Lock.ret _ _ _ _ _ _ _ i _ b _, _, hi => by
      simp only [leafE, nodeE, hi, mul_zero, zero_add]
      have h0 : (0 : ℝ) ≤ Fintype.card P / (Real.exp 1 * M) := by positivity
      refine le_trans (le_of_eq ?_) h0
      simp [ind]
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      simp only [leafE, nodeE]
      have hC : (Fintype.card C : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
      calc (∑ c : C, leafE plur _ (L c) (σ' c)) / (Fintype.card C : ℝ)
          ≤ (∑ c : C, (Fintype.card P * nodeE plur slot _ (L c) (σ' c) +
              Fintype.card P / (Real.exp 1 * M))) / (Fintype.card C : ℝ) :=
            div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ =>
              leafE_off_le_link op row slot plur hplur M hM (L c) (σ' c) hi) (Nat.cast_nonneg _)
        _ = _ := by
            rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
            field_simp
  | _, _, _, .send _ L, σ', hi => leafE_off_le_link op row slot plur hplur M hM (L σ'.1) σ'.2 hi
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s upd _ k' hfresh hset hkeep L, σ', hi => by
      simp only [leafE, nodeE]
      by_cases hs : s = slot
      · subst hs
        rw [ite_eq_left rfl]
        set t := plur s σ'.1 (k' σ'.1) σ'.2
        rw [leafE_fixed plur s (t, σ'.1) (ind fun b => ∃ p, Rewinding.Off (op σ'.1) (row t) b p) _
          (fun i b h => by
            simp only [h, Option.some.injEq, Prod.mk.injEq]
            congr 1
            funext b
            refine propext ⟨fun ⟨t₁, d₁, ⟨h1, h2⟩, h3⟩ => ?_, fun h3 => ⟨_, _, ⟨rfl, rfl⟩, h3⟩⟩
            subst h1; subst h2; exact h3) _ σ'.2 (hset t σ'.1)]
        exact Rewinding.expect_bad_le_link (op σ'.1) (k' σ'.1) σ'.2 (row t) (hplur _ _ _) M hM
      · rw [ite_eq_right hs]
        exact leafE_off_le_link op row slot plur hplur M hM _ σ'.2
          (by rw [hkeep _ _ slot (Ne.symm hs), hi])

/-- **The slot's link finder costs at most two compiled runs** in expectation over the prefixes, for a nonnegative
cost of a run's output. -/
theorem nodeE_linkCost_le {P V : Type} [Fintype P] [Nonempty P] [Fintype V] [DecidableEq V]
    (op : D → β → P → Option V) (slot : ℕ) (plur : Chooser T D β) (c : β → ℝ) (hc : ∀ b, 0 ≤ c b) (M : ℕ)
    {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g') :
    nodeE plur slot (fun d g'' σ'' => expect (Rewinding.linkCost (op d) c) (Rewinding.linkG P g'' M)
      (Rewinding.linkStrat P g'' σ'' M)) L σ' ≤ 2 * expect c g' σ' := by
  calc _ ≤ nodeE plur slot (fun _ g'' σ'' => expect (fun b => 2 * c b) g'' σ'') L σ' :=
        nodeE_mono plur slot (fun d g'' σ'' => by
          rw [expect_const_mul]; exact Rewinding.expect_linkCost_le (op d) g'' σ'' c hc M) L σ'
    _ ≤ expect (fun b => 2 * c b) g' σ' := nodeE_le_expect plur slot _ (fun b => by linarith [hc b]) L σ'
    _ = _ := expect_const_mul 2 c g' σ'

end Lock

end Game

end FlockSoundness
