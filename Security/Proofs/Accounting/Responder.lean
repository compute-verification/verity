import Definitions.Accounting.Guarantees.Responder
import Proofs.Accounting.Actual.Confidence
import Proofs.Accounting.Memory

/-!
# B2, proved

`LengthCode`: the code's last `1` sits at the string's length, so equal codes have equal lengths, then equal bits.
`TimelyTicks`: by induction on `j`, each window adding at most `span`.

`B2` is a choice construction. The responder's state before challenge `j` is `stateOf`: `lengthCode M` of `X`'s state's
content at `t x j`, a `τ`-bit code of `t x j` (injective on the ticks challenge `j` takes on the runs on time, `ticks`),
and the help's `H`-bit code (`exists_code`), packed into `N` bits (`pack`). The preprocessing picks an outcome with the
primitive and the encoding it is given (`pre`); the step at challenge `j` picks an outcome whose state codes to its
state, with content of at most `M` bits, a tick among `ticks j`, and whose challenge `j` is the one asked (`out`), and
returns that run's answer and its next state. At an outcome on time where `mem_X(t_j) ≤ M`, the pick has the same
content (`LengthCode`, with `content_length_le`), the same tick and the same help (the codes are injective), so the same
ingress until the earlier of the two next ticks, so by `CausalFrom` the same `acts` until then, so by `ArrivalRule` the
same next tick (`next_tick_eq`), so the same window, `acts` and answer, and the same state at the next tick
(`image_state_eq_of_acts`: what `X` does during a window includes its state at the window's end). `seqAnswers_eq` and
`seqState_eq` fold that over the challenges.

`B2Audit` applies PoUS's guarantee to that responder at `M = N − 1 − τ − H`: an outcome that passes with
`mem_X(t_j) + 1 + τ + H ≤ N` is on time and one the responder answers correctly. The deployed forms get `TicksIn`
from a fixed first tick (`ticksIn_of_start`: on time, challenge `j`'s tick is in `[t₀, t₀ + j·span]`).
`B2Determines` reads `z` back with a left inverse of the length code (`decodeLen`). `B2_fixed` and `B2Audit_fixed`
check that the fixed-schedule statements are the case `t x = t`, `τ = 0`.
-/

namespace Accounting.SecurityProofs

open Finset
open scoped ENNReal

theorem LengthCode : Guarantees.LengthCode := by
  have len : ∀ (M : ℕ) (l l' : List Bool), l.length ≤ M → lengthCode M l = lengthCode M l' →
      l.length ≤ l'.length := by
    intro M l l' hl h
    by_contra hlt
    have h1 : lengthCode M l ⟨l.length, by omega⟩ = true := by simp [lengthCode]
    have h2 : lengthCode M l' ⟨l.length, by omega⟩ = false := by
      simp [lengthCode, show ¬ l.length < l'.length by omega, show l.length ≠ l'.length by omega]
    rw [h] at h1
    exact absurd (h1.symm.trans h2) (by simp)
  intro M l hl l' hl' h
  change l.length ≤ M at hl
  change l'.length ≤ M at hl'
  have hlen : l.length = l'.length := le_antisymm (len M l l' hl h) (len M l' l hl' h.symm)
  refine List.ext_getElem hlen fun i hi hi' => ?_
  have := congrFun h ⟨i, by omega⟩
  simpa only [lengthCode, hi, hi', dite_true] using this

theorem pr_mono {β : Type} [Fintype β] {P Q : β → Prop} (h : ∀ x, P x → Q x) : Pous.pr P ≤ Pous.pr Q := by
  classical
  unfold Pous.pr
  refine ENNReal.div_le_div_right ?_ _
  exact_mod_cast Finset.card_le_card fun x hx => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
    exact h x hx


theorem TimelyTicks : Guarantees.TimelyTicks := by
  intro t span k ht j hj
  induction j with
  | zero => simp
  | succ j ih =>
    have h1 := ht j (by omega)
    have h2 := ih (by omega)
    rw [Nat.succ_mul]
    omega

namespace Responder

/-- What `X` does during `[t, u)`, `t < u`, fixes its state at `u` as seen (its state at the end of the window's
last tick is among what it does). -/
theorem image_state_eq_of_acts {L : Type*} [LinearOrder L] {Ω : Type*} {S : Ω → Actual.Run L} {X : Finset L}
    {t u : ℕ} (htu : t < u) {r r' : Ω} (h : (S r).acts X (Set.Ico t u) = (S r').acts X (Set.Ico t u)) :
    ((S r).state X u).image (S r).seen = ((S r').state X u).image (S r').seen := by
  have h3 := congrArg (fun a => a.2.2) h
  simp only [Actual.Run.acts] at h3
  have key : ∀ (R : Actual.Run L) p, (u - 1, p) ∈
      {q | ∃ v a, v ∈ Set.Ico t u ∧ a ∈ R.state X (v + 1) ∧ q = (v, R.seen a)} ↔ p ∈ (R.state X u).image R.seen := by
    intro R p
    have e : u - 1 + 1 = u := by omega
    simp only [Set.mem_ofPred_eq, Set.mem_Ico, Prod.mk.injEq, Finset.mem_image]
    constructor
    · rintro ⟨v, a, -, ha, hv, rfl⟩
      subst hv
      exact ⟨a, by rwa [e] at ha, rfl⟩
    · rintro ⟨a, ha, rfl⟩
      exact ⟨u - 1, a, ⟨by omega, by omega⟩, by rwa [e], rfl, rfl⟩
  ext p
  rw [← key (S r) p, ← key (S r') p, h3]


section

variable {L : Type*} [LinearOrder L] {I : Pous.IdealModel} {sch : Pous.Scheme I} {k : ℕ}
  {S : Pous.Outcome sch k → Actual.Run L} {X : Finset L} {t : Pous.Outcome sch k → ℕ → ℕ} {α : Type*}
  {help : Pous.Outcome sch k → α}

/-- **The next tick agrees**: at two outcomes with the same help, challenge `j`, tick `t j` and content there, the next
challenge tick is the same. Until the earlier of the two, what reaches `X` is the same (`IngressFixed`), so what `X`
does is (`CausalFrom`), and the earlier one is then the other's too (`ArrivalRule`). -/
theorem next_tick_eq (hc : CausalFrom S) (hinc : ∀ x (j : Fin k), t x j < t x (j + 1)) (hin : IngressFixed S X t help)
    (har : ArrivalRule S X t help) {x x' : Pous.Outcome sch k} (j : Fin k) (hh : help x = help x')
    (hi : x.2.2 j = x'.2.2 j) (ht : t x j = t x' j)
    (hcont : (S x).content X (t x j) = (S x').content X (t x' j)) : t x (j + 1) = t x' (j + 1) := by
  have hcr := hin x x' j hh hi ht
  rw [← ht] at hcont
  have hacts := hc X (t x j) (min (t x (j + 1)) (t x' (j + 1))) x x' hcont hcr
  have hs : t x j < min (t x (j + 1)) (t x' (j + 1)) := lt_min (hinc x j) (ht ▸ hinc x' j)
  rcases min_choice (t x (j + 1)) (t x' (j + 1)) with hm | hm
  · rw [hm] at hacts hs
    exact (har x x' j _ hh hi ht hs hacts rfl).symm
  · rw [hm] at hacts hs
    rw [ht] at hacts hs
    exact har x' x j _ hh.symm hi.symm ht.symm hs hacts.symm rfl

omit [LinearOrder L] in
/-- On time with the first tick at `t₀`, challenge `j`'s tick is in `[t₀, t₀ + j·span]`, so with `k·span ≤ 2^τ` it
takes at most `2^τ` values. -/
theorem ticksIn_of_start (hinc : ∀ x (j : Fin k), t x j < t x (j + 1)) {t₀ span τ : ℕ} (h0 : ∀ x, t x 0 = t₀)
    (hτ : k * span ≤ 2 ^ τ) : TicksIn t span τ := by
  intro j
  have hsub : {v | ∃ x, Timely (t x) span k ∧ t x j = v} ⊆ (Finset.Icc t₀ (t₀ + j * span) : Set ℕ) := by
    rintro v ⟨x, hx, rfl⟩
    have hle : ∀ i ≤ (j : ℕ), t x 0 ≤ t x i := by
      intro i hi
      induction i with
      | zero => exact le_rfl
      | succ i ih => exact (ih (by omega)).trans (hinc x ⟨i, by omega⟩).le
    have := TimelyTicks (t x) span k hx j j.isLt.le
    simp only [Finset.coe_Icc, Set.mem_Icc]
    exact ⟨h0 x ▸ hle j le_rfl, h0 x ▸ this⟩
  refine (Set.encard_le_encard hsub).trans ?_
  rw [Set.encard_coe_eq_coe_finsetCard, Nat.card_Icc]
  have : t₀ + j * span + 1 - t₀ ≤ 2 ^ τ := by
    rcases Nat.eq_zero_or_pos span with h | h
    · subst h; simpa using Nat.one_le_two_pow
    · have := Nat.mul_le_mul_right span (show (j : ℕ) + 1 ≤ k from j.isLt)
      rw [Nat.succ_mul] at this
      omega
  exact_mod_cast this

/-- The ticks challenge `j` takes on the runs on time. -/
def ticks (t : Pous.Outcome sch k → ℕ → ℕ) (span : ℕ) (j : ℕ) : Set ℕ := {v | ∃ x, Timely (t x) span k ∧ t x j = v}

omit [LinearOrder L] in
/-- With `TicksIn`, each challenge's ticks on the runs on time have a `τ`-bit code, injective on them. -/
theorem exists_tick_code {span τ : ℕ} (h : TicksIn t span τ) :
    ∃ ct : ℕ → ℕ → Pous.Bits τ, ∀ j < k, Set.InjOn (ct j) (ticks t span j) := by
  have : ∀ j : ℕ, ∃ c : ℕ → Pous.Bits τ, j < k → Set.InjOn c (ticks t span j) := by
    intro j
    by_cases hj : j < k
    · have h' : (ticks t span j).encard ≤ ((2 ^ τ : ℕ) : ℕ∞) := h ⟨j, hj⟩
      have hle : (ticks t span j).encard ≤ (Set.univ : Set (Pous.Bits τ)).encard := by
        rw [Set.encard_univ, ENat.card_eq_coe_fintype_card]
        simpa [Fintype.card_fun] using h'
      obtain ⟨c, -, hc⟩ := (Set.finite_of_encard_le_coe h').exists_injOn_of_encard_le hle
      exact ⟨c, fun _ => hc⟩
    · exact ⟨fun _ _ => false, fun h' => absurd h' hj⟩
  choose ct hct using this
  exact ⟨ct, hct⟩

end

/-- `a` then `b`, then `false`s, in `N` bits. -/
def pack (A B N : ℕ) (a : Pous.Bits A) (b : Pous.Bits B) : Pous.Bits N :=
  fun i => if h : i.val < A then a ⟨i, h⟩ else if h' : i.val - A < B then b ⟨i - A, h'⟩ else false

theorem pack_inj {A B N : ℕ} (hN : A + B ≤ N) {a a' : Pous.Bits A} {b b' : Pous.Bits B}
    (h : pack A B N a b = pack A B N a' b') : a = a' ∧ b = b' := by
  refine ⟨funext fun i => ?_, funext fun i => ?_⟩
  · have := congrFun h ⟨i, by omega⟩
    simpa only [pack, i.isLt, dite_true] using this
  · have := congrFun h ⟨A + i, by omega⟩
    have e : A + i.val - A = i.val := by omega
    simp only [pack, show ¬ (A + i.val < A) by omega, dite_false, e, i.isLt, dite_true] at this
    exact this

/-- The first `A` bits of an `N`-bit state. -/
def unpack (A N : ℕ) (hN : A ≤ N) (σ : Pous.Bits N) : Pous.Bits A := fun i => σ ⟨i, by omega⟩

theorem unpack_pack {A B N : ℕ} (hN : A ≤ N) (a : Pous.Bits A) (b : Pous.Bits B) :
    unpack A N hN (pack A B N a b) = a := by
  funext i
  simp [unpack, pack, i.isLt]

open Classical in
/-- A left inverse of `lengthCode M` on strings of at most `M` bits. -/
noncomputable def decodeLen (M : ℕ) (a : Pous.Bits (M + 1)) : List Bool :=
  if h : ∃ l : List Bool, l.length ≤ M ∧ lengthCode M l = a then Classical.choose h else []

theorem decodeLen_lengthCode {M : ℕ} {l : List Bool} (hl : l.length ≤ M) : decodeLen M (lengthCode M l) = l := by
  have h : ∃ l' : List Bool, l'.length ≤ M ∧ lengthCode M l' = lengthCode M l := ⟨l, hl, rfl⟩
  rw [decodeLen, dite_eq_left h]
  exact LengthCode M (Classical.choose_spec h).1 hl (Classical.choose_spec h).2

/-- A help with at most `2^H` values has an `H`-bit code, injective on them. -/
theorem exists_code {β α : Type*} [Finite β] (help : β → α) {H : ℕ}
    (h : (Set.range help).encard ≤ ((2 ^ H : ℕ) : ℕ∞)) : ∃ c : α → Pous.Bits H, Set.InjOn c (Set.range help) := by
  have hle : (Set.range help).encard ≤ (Set.univ : Set (Pous.Bits H)).encard := by
    rw [Set.encard_univ, ENat.card_eq_coe_fintype_card]
    simpa [Fintype.card_fun] using h
  obtain ⟨c, -, hc⟩ := (Set.finite_range help).exists_injOn_of_encard_le hle
  exact ⟨c, hc⟩


theorem seqAnswers_eq {I : Pous.IdealModel} {ℓ N B : ℕ} (f : I.Qry → I.Ans)
    (step : ℕ → Pous.Bits N → Fin B → Pous.OracleAlgorithm I (Pous.Bits ℓ × Pous.Bits N)) (k : ℕ) (σ : ℕ → Pous.Bits N)
    (a : ℕ → Pous.Bits ℓ) (is : ℕ → Fin B)
    (h : ∀ j < k, ((step j (σ j) (is j)).run f).1 = a j ∧
      (j + 1 < k → ((step j (σ j) (is j)).run f).2 = σ (j + 1))) :
    ∀ n m, m + n = k → Pous.seqAnswers f step m (List.ofFn fun i : Fin n => is (m + i)) (σ m) =
      List.ofFn fun i : Fin n => a (m + i) := by
  intro n
  induction n with
  | zero => intro m _; simp [Pous.seqAnswers]
  | succ n ih =>
    intro m hm
    rw [List.ofFn_succ, List.ofFn_succ]
    simp only [Pous.seqAnswers, Fin.val_zero, Nat.add_zero, Fin.val_succ]
    obtain ⟨h1, h2⟩ := h m (by omega)
    rw [h1]
    congr 1
    rcases n with _ | n
    · simp [Pous.seqAnswers]
    · rw [h2 (by omega)]
      have e1 : (fun i : Fin (n + 1) => is (m + (↑i + 1))) = fun i : Fin (n + 1) => is (m + 1 + ↑i) :=
        funext fun i => congrArg is (by omega)
      have e2 : (fun i : Fin (n + 1) => a (m + (↑i + 1))) = fun i : Fin (n + 1) => a (m + 1 + ↑i) :=
        funext fun i => congrArg a (by omega)
      rw [e1, e2]
      exact ih (m + 1) (by omega)

theorem seqState_append {I : Pous.IdealModel} {ℓ N B : ℕ} (f : I.Qry → I.Ans)
    (step : ℕ → Pous.Bits N → Fin B → Pous.OracleAlgorithm I (Pous.Bits ℓ × Pous.Bits N)) (i : Fin B) :
    ∀ (l : List (Fin B)) (m : ℕ) (σ : Pous.Bits N), Pous.Sampled.seqState f step m (l ++ [i]) σ =
      ((step (m + l.length) (Pous.Sampled.seqState f step m l σ) i).run f).2 := by
  intro l
  induction l with
  | nil => intro m σ; simp [Pous.Sampled.seqState]
  | cons a l ih =>
    intro m σ
    simp only [List.cons_append, Pous.Sampled.seqState, List.length_cons]
    rw [ih, show m + 1 + l.length = m + (l.length + 1) by omega]

theorem seqState_eq {I : Pous.IdealModel} {ℓ N B : ℕ} (f : I.Qry → I.Ans)
    (step : ℕ → Pous.Bits N → Fin B → Pous.OracleAlgorithm I (Pous.Bits ℓ × Pous.Bits N)) (k : ℕ) (σ : ℕ → Pous.Bits N)
    (idx : Fin k → Fin B)
    (h : ∀ j (hj : j + 1 < k), ((step j (σ j) (idx ⟨j, by omega⟩)).run f).2 = σ (j + 1)) :
    ∀ j < k, Pous.Sampled.seqState f step 0 ((List.ofFn idx).take j) (σ 0) = σ j := by
  intro j
  induction j with
  | zero => intro _; simp [Pous.Sampled.seqState]
  | succ j ih =>
    intro hj
    have hlen : j < (List.ofFn idx).length := by simp; omega
    rw [List.take_add_one, List.getElem?_eq_getElem hlen, Option.toList_some, seqState_append, ih (by omega)]
    have hl : ((List.ofFn idx).take j).length = j := by simp; omega
    rw [hl, Nat.zero_add, List.getElem_ofFn]
    exact h j hj

variable {L : Type*} [LinearOrder L] {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) {k : ℕ}
  (W : Pous.Bits sch.n) (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : Pous.Outcome sch k → ℕ → ℕ)
  (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ) {α : Type*} (help : Pous.Outcome sch k → α) (span M τ H N : ℕ)
  (c : α → Pous.Bits H) (ct : ℕ → ℕ → Pous.Bits τ)

/-- The responder's state before challenge `j` at the outcome `x`: `X`'s content at `t x j` coded in `M + 1` bits, the
tick's code and the help's code. -/
noncomputable def stateOf (x : Pous.Outcome sch k) (j : ℕ) : Pous.Bits N :=
  pack (M + 1) (τ + H) N (lengthCode M ((S x).content X (t x j))) (pack τ H (τ + H) (ct j (t x j)) (c (help x)))

open Classical in
/-- The preprocessing: the state at the first challenge of an outcome with the given primitive and encoding. -/
noncomputable def pre : Pous.Preprocessor sch N := fun _ pp C ω =>
  if h : ∃ x : Pous.Outcome sch k, x.1 = ω ∧ sch.enc W ω x.2.1 = (pp, C) then
    stateOf sch S X t help M τ H N c ct (Classical.choose h) 0
  else fun _ => false

open Classical in
/-- What the step at challenge `j` returns from the state `σ` and the index `i`: the answer and the next state of an
outcome whose state codes to `σ`, with content of at most `M` bits and a tick among `ticks j`, and whose challenge `j`
is `i`. -/
noncomputable def out (j : ℕ) (σ : Pous.Bits N) (i : Fin sch.B) : Pous.Bits sch.ℓ × Pous.Bits N :=
  if hj : j < k then
    if h : ∃ x : Pous.Outcome sch k, x.2.2 ⟨j, hj⟩ = i ∧ ((S x).content X (t x j)).length ≤ M ∧
        t x j ∈ ticks t span j ∧ stateOf sch S X t help M τ H N c ct x j = σ then
      (ans j ((S (Classical.choose h)).acts X (window (t (Classical.choose h)) j)),
        stateOf sch S X t help M τ H N c ct (Classical.choose h) (j + 1))
    else (fun _ => false, σ)
  else (fun _ => false, σ)

/-- The responder: `out`, with no query. -/
noncomputable def resp : Pous.SeqResponderOn sch N 0 0 0 where
  step _ _ j σ i := .ret (out sch S X t ans help span M τ H N c ct j σ i)
  bounded _ _ _ _ _ := fun _ => by simp [Pous.OracleAlgorithm.mulDepth, Pous.OracleAlgorithm.keyWork, Pous.OracleAlgorithm.ringWork]

variable {sch W S X t ans help span M τ H N c ct}

/-- When the setup fixes `X`'s state at the first challenge, that tick and the help, the preprocessing returns
`stateOf … 0`. -/
theorem pre_spec (hset : SetupFixed W S X t help) (x : Pous.Outcome sch k) :
    pre sch W S X t help M τ H N c ct W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1 =
      stateOf sch S X t help M τ H N c ct x 0 := by
  have h : ∃ x' : Pous.Outcome sch k, x'.1 = x.1 ∧
      sch.enc W x.1 x'.2.1 = ((sch.enc W x.1 x.2.1).1, (sch.enc W x.1 x.2.1).2) := ⟨x, rfl, rfl⟩
  obtain ⟨h1, h2⟩ := Classical.choose_spec h
  obtain ⟨hcont, htick, hhelp⟩ := hset _ _ h1 (by rw [h1]; exact h2)
  simp only [pre, dite_eq_left h, stateOf]
  rw [hcont, htick, hhelp]

section
variable (hc : CausalFrom S) (hinc : ∀ x (j : Fin k), t x j < t x (j + 1)) (hin : IngressFixed S X t help)
  (har : ArrivalRule S X t help) (hct : ∀ j < k, Set.InjOn (ct j) (ticks t span j))
  (hcode : Set.InjOn c (Set.range help)) (hN : M + 1 + τ + H ≤ N)
include hc hinc hin har hct hcode hN

/-- At an outcome on time where `X`'s content at `t x j` has at most `M` bits, the step at challenge `j` answers as `X`
does and, unless `j` is the last challenge, moves to `X`'s next state. -/
theorem out_spec (x : Pous.Outcome sch k) (hx : Timely (t x) span k) (j : ℕ) (hj : j < k)
    (hm : ((S x).content X (t x j)).length ≤ M) :
    (out sch S X t ans help span M τ H N c ct j (stateOf sch S X t help M τ H N c ct x j) (x.2.2 ⟨j, hj⟩)).1 =
        ans j ((S x).acts X (window (t x) j)) ∧
      (j + 1 < k → (out sch S X t ans help span M τ H N c ct j (stateOf sch S X t help M τ H N c ct x j)
        (x.2.2 ⟨j, hj⟩)).2 = stateOf sch S X t help M τ H N c ct x (j + 1)) := by
  have hxt : t x j ∈ ticks t span j := ⟨x, hx, rfl⟩
  have h : ∃ x' : Pous.Outcome sch k, x'.2.2 ⟨j, hj⟩ = x.2.2 ⟨j, hj⟩ ∧ ((S x').content X (t x' j)).length ≤ M ∧
      t x' j ∈ ticks t span j ∧ stateOf sch S X t help M τ H N c ct x' j = stateOf sch S X t help M τ H N c ct x j :=
    ⟨x, rfl, hm, hxt, rfl⟩
  obtain ⟨hi, hl, hR, hs⟩ := Classical.choose_spec h
  set x' := Classical.choose h
  obtain ⟨h1, h2⟩ := pack_inj (by omega) hs
  obtain ⟨h3, h4⟩ := pack_inj le_rfl h2
  have hcont := LengthCode M hl hm h1
  have htick : t x' j = t x j := hct j hj hR hxt h3
  have hhelp : help x' = help x := hcode ⟨_, rfl⟩ ⟨_, rfl⟩ h4
  have hnext : t x' (j + 1) = t x (j + 1) := next_tick_eq hc hinc hin har ⟨j, hj⟩ hhelp hi htick hcont
  have hcr := hin x' x ⟨j, hj⟩ hhelp hi htick
  simp only [hnext, min_self] at hcr
  rw [← htick] at hcont
  have hacts := hc X (t x' j) (t x' (j + 1)) x' x hcont (by rw [hnext]; exact hcr)
  have e : out sch S X t ans help span M τ H N c ct j (stateOf sch S X t help M τ H N c ct x j) (x.2.2 ⟨j, hj⟩) =
      (ans j ((S x').acts X (window (t x') j)), stateOf sch S X t help M τ H N c ct x' (j + 1)) := by
    unfold out
    rw [dite_eq_left hj]
    exact dite_eq_left h
  rw [e]
  have hw : window (t x') j = window (t x) j := by simp only [window, htick, hnext]
  refine ⟨?_, fun _ => ?_⟩
  · rw [← hw]
    exact congrArg (ans j) (by simpa only [window, hnext, htick] using hacts)
  · have hst := image_state_eq_of_acts (hinc x' ⟨j, hj⟩) hacts
    have hc' : (S x').content X (t x' (j + 1)) = (S x).content X (t x (j + 1)) := by
      unfold Actual.Run.content
      rw [hst, hnext]
    simp only [stateOf]
    rw [hc', hnext, hhelp]

/-- **The responder answers as `X` does** at every outcome on time where `mem_X(t_j) ≤ M` at each challenge. -/
theorem seqAnswers_resp (hset : SetupFixed W S X t help) (x : Pous.Outcome sch k) (hx : Timely (t x) span k)
    (hm : ∀ j : Fin k, (S x).mem X (t x j) ≤ M) :
    Pous.seqAnswers ((Pous.ringRO B w p).answer x.1)
        ((resp sch S X t ans help span M τ H N c ct).step W (sch.enc W x.1 x.2.1).1) 0 (List.ofFn x.2.2)
        (pre sch W S X t help M τ H N c ct W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) =
      List.ofFn fun j : Fin k => ans j ((S x).acts X (window (t x) j)) := by
  have hl : ∀ j (hj : j < k), ((S x).content X (t x j)).length ≤ M := fun j hj =>
    (Actual.content_length_le _ _ _).trans (hm ⟨j, hj⟩)
  let is : ℕ → Fin sch.B := fun j => if hj : j < k then x.2.2 ⟨j, hj⟩ else ⟨0, sch.B_pos⟩
  have key := seqAnswers_eq ((Pous.ringRO B w p).answer x.1) ((resp sch S X t ans help span M τ H N c ct).step W
    (sch.enc W x.1 x.2.1).1) k (stateOf sch S X t help M τ H N c ct x)
    (fun j => ans j ((S x).acts X (window (t x) j))) is
    (fun j hj => by
      simp only [resp, Pous.OracleAlgorithm.run, is, dite_eq_left hj]
      exact out_spec hc hinc hin har hct hcode hN x hx j hj (hl j hj)) k 0 (by omega)
  rw [pre_spec hset x]
  simp only [Nat.zero_add] at key
  convert key using 2
  congr 1
  funext i
  simp [is, i.isLt]

/-- **The state before each challenge is `stateOf`** at every outcome on time where `mem_X(t_j) ≤ M` at each
challenge. -/
theorem seqState_resp (hset : SetupFixed W S X t help) (x : Pous.Outcome sch k) (hx : Timely (t x) span k)
    (hm : ∀ j : Fin k, (S x).mem X (t x j) ≤ M) (j : ℕ) (hj : j < k) :
    Pous.Sampled.seqState ((Pous.ringRO B w p).answer x.1)
        ((resp sch S X t ans help span M τ H N c ct).step W (sch.enc W x.1 x.2.1).1) 0 ((List.ofFn x.2.2).take j)
        (pre sch W S X t help M τ H N c ct W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) =
      stateOf sch S X t help M τ H N c ct x j := by
  rw [pre_spec hset x]
  refine seqState_eq _ _ k (stateOf sch S X t help M τ H N c ct x) x.2.2 (fun j hj => ?_) j hj
  simp only [resp, Pous.OracleAlgorithm.run]
  exact (out_spec hc hinc hin har hct hcode hN x hx j (by omega)
    ((Actual.content_length_le _ _ _).trans (hm ⟨j, by omega⟩))).2 hj

end

end Responder

open Responder in
theorem B2 : Guarantees.B2 := by
  intro L _ B w p _ sch k W S X t span ans α help M τ H N hc hinc hset hin har hticks hcount hN
  obtain ⟨c, hcode⟩ := exists_code help hcount
  obtain ⟨ct, hct⟩ := exists_tick_code hticks
  exact ⟨pre sch W S X t help M τ H N c ct, resp sch S X t ans help span M τ H N c ct,
    fun x hx hm => seqAnswers_resp hc hinc hin har hct hcode hN hset x hx hm⟩

theorem B2Audit : Guarantees.B2Audit := by
  intro L _ B w p _ sch N k δ ε hsec W S X t span ans α help τ H hc hinc hset hin har hticks hcount
  by_cases hN : 1 + τ + H ≤ N
  · obtain ⟨A₁, A₂, hA⟩ :=
      B2 sch k W S X t span ans help (N - 1 - τ - H) τ H N hc hinc hset hin har hticks hcount (by omega)
    refine (pr_mono fun x hx => ?_).trans (hsec W A₁ A₂)
    obtain ⟨⟨htime, hp⟩, hm⟩ := hx
    unfold Pous.seqAcceptsOn
    dsimp only
    rw [hA x htime fun j => by have := hm j; omega]
    exact congrArg List.ofFn (funext hp)
  · let A₂ : Pous.SeqResponderOn sch N 0 0 0 :=
      { step := fun _ _ _ σ _ => .ret (fun _ => false, σ)
        bounded := fun _ _ _ _ _ _ => by simp [Pous.OracleAlgorithm.mulDepth, Pous.OracleAlgorithm.keyWork, Pous.OracleAlgorithm.ringWork] }
    refine (pr_mono fun x hx => ?_).trans (hsec W (fun _ _ _ _ _ => false) A₂)
    rcases Nat.eq_zero_or_pos k with hk | hk
    · subst hk
      simp [Pous.seqAcceptsOn, Pous.seqAnswers]
    · have := hx.2 ⟨0, hk⟩
      omega

open Responder in
theorem B2P2Ring : Guarantees.B2P2Ring := by
  intro hsec L _ W S X t t₀ span ans α help τ H hc hinc hset hin har h0 hτ hcount
  exact B2Audit _ _ _ _ _ hsec W S X t span ans help τ H hc hinc hset hin har (ticksIn_of_start hinc h0 hτ) hcount

open Responder in
theorem B2Determines : Guarantees.B2Determines := by
  intro L _ B w p _ sch k q W S X t span ans α help ζ M τ H hc hinc hset hin har hticks hcount htime hm hζ
  choose c hcode using fun z => exists_code (help z) (hcount z)
  choose ct hct using fun z => exists_tick_code (hticks z)
  have hN : M + 1 + τ + H ≤ M + 1 + τ + H := le_rfl
  refine ⟨fun z => pre sch W (S z) X (t z) (help z) M τ H (M + 1 + τ + H) (c z) (ct z),
    fun z => resp sch (S z) X (t z) ans (help z) span M τ H (M + 1 + τ + H) (c z) (ct z),
    fun j σ => ζ j (decodeLen M (unpack (M + 1) (M + 1 + τ + H) (by omega) σ)), fun z x j hj => ?_, fun z x =>
      seqAnswers_resp (hc z) (hinc z) (hin z) (har z) (hct z) (hcode z) hN (hset z) x (htime z x) (hm z x)⟩
  rw [seqState_resp (hc z) (hinc z) (hin z) (har z) (hct z) (hcode z) hN (hset z) x (htime z x) (hm z x) j hj]
  simp only [stateOf]
  rw [unpack_pack, decodeLen_lengthCode ((Actual.content_length_le _ _ _).trans (hm z x ⟨j, hj⟩))]
  exact hζ z x ⟨j, hj⟩

theorem B2Adds : Guarantees.B2Adds := by
  intro L _ B w p _ sch N c δ ε hsec q k W S X t span ans α help ζ M τ H hc hinc hset hin har hticks hcount htime hm hζ
  obtain ⟨A₁, A₂, π, hdet, hans⟩ :=
    B2Determines sch c q W S X t span ans help ζ M τ H hc hinc hset hin har hticks hcount htime hm hζ
  refine (pr_mono fun x hx => ?_).trans
    (Memory.AddsOn sch N 0 0 0 c δ ε hsec (M + 1 + τ) H q k W π A₁ A₂ hdet)
  refine ⟨?_, hx.2⟩
  unfold Pous.seqAcceptsOn
  dsimp only
  rw [hans x.1 x.2]
  exact congrArg List.ofFn (funext hx.1.2)

open Responder in
theorem B2AddsP2Ring : Guarantees.B2AddsP2Ring := by
  intro hsec L _ q k W S X t t₀ span ans α help ζ M τ H hc hinc hset hin har h0 hτ hcount htime hm hζ
  have := B2Adds _ _ _ _ _ hsec q k W S X t span ans help ζ M τ H hc hinc hset hin har
    (fun z => ticksIn_of_start (hinc z) (h0 z) hτ) hcount htime hm hζ
  rwa [Nat.cast_ofNat] at this

/-! ### The fixed-schedule statements

A fixed schedule `t : ℕ → ℕ` is the case `t x = t` for every outcome, with `span = t k` and `τ = 0`; each of the six
fixed-schedule statements is proved here, word for word, from its restatement (its `Passes` is `PassesFixed`). -/

namespace Responder

/-- `Passes` on a fixed schedule `t`. -/
def PassesFixed {L : Type*} {I : Pous.IdealModel} (sch : Pous.Scheme I) {k : ℕ} (W : Pous.Bits sch.n)
    (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : ℕ → ℕ) (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ)
    (x : Pous.Outcome sch k) : Prop :=
  ∀ j : Fin k, ans j ((S x).acts X (window t j)) = (sch.enc W x.1 x.2.1).2 (x.2.2 j)

section
variable {L : Type*} [LinearOrder L] {I : Pous.IdealModel} {sch : Pous.Scheme I} {k : ℕ} {W : Pous.Bits sch.n}
  {S : Pous.Outcome sch k → Actual.Run L} {X : Finset L} {t : ℕ → ℕ} {α : Type*} {help : Pous.Outcome sch k → α}

omit [LinearOrder L] in
theorem fixed_timely (ht : StrictMono t) (k : ℕ) : Timely t (t k) k := fun j hj => by
  have := ht.monotone (show j + 1 ≤ k by omega)
  omega

omit [LinearOrder L] in
theorem fixed_ticksIn : TicksIn (fun _ : Pous.Outcome sch k => t) (t k) 0 := by
  intro j
  refine (Set.encard_le_encard (t := {t j}) ?_).trans (by simp)
  rintro v ⟨x, -, rfl⟩
  rfl

omit [LinearOrder L] in
theorem fixed_arrival : ArrivalRule S X (fun _ => t) help := fun _ _ _ _ _ _ _ _ _ h => h

theorem fixed_ingress (hin : ∀ x x' (j : Fin k), help x = help x' → x.2.2 j = x'.2.2 j →
    (S x).crossing (↑X)ᶜ ↑X (window t j) = (S x').crossing (↑X)ᶜ ↑X (window t j)) :
    IngressFixed S X (fun _ => t) help :=
  fun x x' j hh hi _ => by simpa only [min_self, window] using hin x x' j hh hi

theorem fixed_setup (hset : ∀ x x', x.1 = x'.1 → sch.enc W x.1 x.2.1 = sch.enc W x'.1 x'.2.1 →
    (S x).content X (t 0) = (S x').content X (t 0) ∧ help x = help x') : SetupFixed W S X (fun _ => t) help :=
  fun x x' h1 h2 => ⟨(hset x x' h1 h2).1, rfl, (hset x x' h1 h2).2⟩

end

end Responder

open Responder in
/-- **`B2` on a fixed schedule**, from `B2` at `t x = t`, `span = t k` and `τ = 0`. -/
theorem B2_fixed {L : Type*} [LinearOrder L] {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (k : ℕ)
    (W : Pous.Bits sch.n) (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : ℕ → ℕ)
    (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ) {α : Type*} (help : Pous.Outcome sch k → α) (M H N : ℕ)
    (hc : CausalFrom S) (ht : StrictMono t)
    (hset : ∀ x x', x.1 = x'.1 → sch.enc W x.1 x.2.1 = sch.enc W x'.1 x'.2.1 →
      (S x).content X (t 0) = (S x').content X (t 0) ∧ help x = help x')
    (hin : ∀ x x' (j : Fin k), help x = help x' → x.2.2 j = x'.2.2 j →
      (S x).crossing (↑X)ᶜ ↑X (window t j) = (S x').crossing (↑X)ᶜ ↑X (window t j))
    (hcount : (Set.range help).encard ≤ ((2 ^ H : ℕ) : ℕ∞)) (hN : M + 1 + H ≤ N) :
    ∃ (A₁ : Pous.Preprocessor sch N) (A₂ : Pous.SeqResponderOn sch N 0 0 0),
      ∀ x, (∀ j : Fin k, (S x).mem X (t j) ≤ M) →
        Pous.seqAnswers ((Pous.ringRO B w p).answer x.1) (A₂.step W (sch.enc W x.1 x.2.1).1) 0 (List.ofFn x.2.2)
            (A₁ W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) =
          List.ofFn fun j : Fin k => ans j ((S x).acts X (window t j)) := by
  obtain ⟨A₁, A₂, hA⟩ := B2 sch k W S X (fun _ => t) (t k) ans help M 0 H N hc (fun _ j => ht (Nat.lt_succ_self j))
    (fixed_setup hset) (fixed_ingress hin) fixed_arrival fixed_ticksIn hcount (by omega)
  exact ⟨A₁, A₂, fun x hm => hA x (fixed_timely ht k) hm⟩

open Responder in
/-- **`B2Audit` on a fixed schedule**, from `B2Audit` at `t x = t`, `span = t k` and `τ = 0`. -/
theorem B2Audit_fixed {L : Type*} [LinearOrder L] {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p))
    (N k : ℕ) (δ ε : ℝ≥0∞) (hsec : Pous.AuditSecureOn sch .sequential N 0 0 0 k δ ε)
    (W : Pous.Bits sch.n) (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : ℕ → ℕ)
    (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ) {α : Type*} (help : Pous.Outcome sch k → α) (H : ℕ)
    (hc : CausalFrom S) (ht : StrictMono t)
    (hset : ∀ x x', x.1 = x'.1 → sch.enc W x.1 x.2.1 = sch.enc W x'.1 x'.2.1 →
      (S x).content X (t 0) = (S x').content X (t 0) ∧ help x = help x')
    (hin : ∀ x x' (j : Fin k), help x = help x' → x.2.2 j = x'.2.2 j →
      (S x).crossing (↑X)ᶜ ↑X (window t j) = (S x').crossing (↑X)ᶜ ↑X (window t j))
    (hcount : (Set.range help).encard ≤ ((2 ^ H : ℕ) : ℕ∞)) :
    Pous.pr (fun x => PassesFixed sch W S X t ans x ∧ ∀ j : Fin k, (S x).mem X (t j) + 1 + H ≤ N) ≤ δ + ε :=
  (pr_mono fun _ hx => ⟨⟨fixed_timely ht k, hx.1⟩, hx.2⟩).trans
    (B2Audit sch N k δ ε hsec W S X (fun _ => t) (t k) ans help 0 H hc (fun _ j => ht (Nat.lt_succ_self j))
      (fixed_setup hset) (fixed_ingress hin) fixed_arrival fixed_ticksIn hcount)

/-- **`B2P2Ring` on a fixed schedule**, from `B2Audit_fixed`. -/
theorem B2P2Ring_fixed :
  Pous.P2.Assumptions.P2RingAuditSecure 0 0 0 →
    ∀ {L : Type*} [LinearOrder L] (W : Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).n)
      (S : Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → Actual.Run L) (X : Finset L)
      (t : ℕ → ℕ) (ans : ℕ → Actual.Acts L → Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).ℓ)
      {α : Type*} (help : Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → α) (H : ℕ),
      CausalFrom S → StrictMono t →
      (∀ x x', x.1 = x'.1 →
        (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).enc W x.1 x.2.1 =
          (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).enc W x'.1 x'.2.1 →
        (S x).content X (t 0) = (S x').content X (t 0) ∧ help x = help x') →
      (∀ x x' (j : Fin 105), help x = help x' → x.2.2 j = x'.2.2 j →
        (S x).crossing (↑X)ᶜ ↑X (window t j) = (S x').crossing (↑X)ᶜ ↑X (window t j)) →
      (Set.range help).encard ≤ ((2 ^ H : ℕ) : ℕ∞) →
      Pous.pr (fun x => Responder.PassesFixed (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) W S X t ans x ∧
          ∀ j : Fin 105, (S x).mem X (t j) + 1 + H ≤
            (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ) ≤
        ENNReal.ofReal Pous.Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) := by
  intro hsec L _ W S X t ans α help H hc ht hset hin hcount
  exact B2Audit_fixed _ _ _ _ _ hsec W S X t ans help H hc ht hset hin hcount

open Responder in
/-- **`B2Determines` on a fixed schedule**, from `B2Determines` at `t z x = t`, `span = t k` and `τ = 0`. -/
theorem B2Determines_fixed :
  ∀ {L : Type*} [LinearOrder L] {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (k q : ℕ)
    (W : Pous.Bits sch.n) (S : Pous.Bits q → Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : ℕ → ℕ)
    (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ) {α : Type*} (help : Pous.Bits q → Pous.Outcome sch k → α)
    (ζ : ℕ → List Bool → Pous.Bits q) (M H : ℕ),
    (∀ z, CausalFrom (S z)) → StrictMono t →
    (∀ z x x', x.1 = x'.1 → sch.enc W x.1 x.2.1 = sch.enc W x'.1 x'.2.1 →
      (S z x).content X (t 0) = (S z x').content X (t 0) ∧ help z x = help z x') →
    (∀ z x x' (j : Fin k), help z x = help z x' → x.2.2 j = x'.2.2 j →
      (S z x).crossing (↑X)ᶜ ↑X (window t j) = (S z x').crossing (↑X)ᶜ ↑X (window t j)) →
    (∀ z, (Set.range (help z)).encard ≤ ((2 ^ H : ℕ) : ℕ∞)) →
    (∀ z x (j : Fin k), (S z x).mem X (t j) ≤ M) →
    (∀ z x (j : Fin k), ζ j ((S z x).content X (t j)) = z) →
    ∃ (A₁ : Pous.Bits q → Pous.Preprocessor sch (M + 1 + H))
      (A₂ : Pous.Bits q → Pous.SeqResponderOn sch (M + 1 + H) 0 0 0) (π : ℕ → Pous.Bits (M + 1 + H) → Pous.Bits q),
      (∀ (z : Pous.Bits q) (x : Pous.Outcome sch k) (j : ℕ), j < k →
        π j (Pous.Sampled.seqState ((Pous.ringRO B w p).answer x.1) ((A₂ z).step W (sch.enc W x.1 x.2.1).1) 0
          ((List.ofFn x.2.2).take j) (A₁ z W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1)) = z) ∧
      ∀ z x, Pous.seqAnswers ((Pous.ringRO B w p).answer x.1) ((A₂ z).step W (sch.enc W x.1 x.2.1).1) 0
          (List.ofFn x.2.2) (A₁ z W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) =
        List.ofFn fun j : Fin k => ans j ((S z x).acts X (window t j)) := by
  intro L _ B w p _ sch k q W S X t ans α help ζ M H hc ht hset hin hcount hm hζ
  exact B2Determines sch k q W S X (fun _ _ => t) (t k) ans help ζ M 0 H hc (fun _ _ j => ht (Nat.lt_succ_self j))
    (fun z => fixed_setup (hset z)) (fun z => fixed_ingress (hin z)) (fun _ => fixed_arrival)
    (fun _ => fixed_ticksIn) hcount (fun _ _ => fixed_timely ht k) hm hζ

open Responder in
/-- **`B2Adds` on a fixed schedule**, from `B2Adds` at `t z x = t`, `span = t k` and `τ = 0`. -/
theorem B2Adds_fixed :
  ∀ {L : Type*} [LinearOrder L] {B w p : ℕ} [NeZero p] (sch : Pous.Scheme (Pous.ringRO B w p)) (N c : ℕ)
    (δ ε : ℝ≥0∞), Pous.AuditSecureOn sch .sequential N 0 0 0 c δ ε →
    ∀ (q k : ℕ) (W : Pous.Bits sch.n) (S : Pous.Bits q → Pous.Outcome sch c → Actual.Run L) (X : Finset L)
      (t : ℕ → ℕ) (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ) {α : Type*}
      (help : Pous.Bits q → Pous.Outcome sch c → α) (ζ : ℕ → List Bool → Pous.Bits q) (M H : ℕ),
      (∀ z, CausalFrom (S z)) → StrictMono t →
      (∀ z x x', x.1 = x'.1 → sch.enc W x.1 x.2.1 = sch.enc W x'.1 x'.2.1 →
        (S z x).content X (t 0) = (S z x').content X (t 0) ∧ help z x = help z x') →
      (∀ z x x' (j : Fin c), help z x = help z x' → x.2.2 j = x'.2.2 j →
        (S z x).crossing (↑X)ᶜ ↑X (window t j) = (S z x').crossing (↑X)ᶜ ↑X (window t j)) →
      (∀ z, (Set.range (help z)).encard ≤ ((2 ^ H : ℕ) : ℕ∞)) →
      (∀ z x (j : Fin c), (S z x).mem X (t j) ≤ M) →
      (∀ z x (j : Fin c), ζ j ((S z x).content X (t j)) = z) →
      Pous.pr (fun x : Pous.Bits q × Pous.Outcome sch c =>
          Responder.PassesFixed sch W (S x.1) X t ans x.2 ∧ ¬ q + N < M + 1 + H + k) ≤ δ + ε + c * (2⁻¹ : ℝ≥0∞) ^ k := by
  intro L _ B w p _ sch N c δ ε hsec q k W S X t ans α help ζ M H hc ht hset hin hcount hm hζ
  exact (pr_mono fun x hx => ⟨⟨fixed_timely ht c, hx.1⟩, hx.2⟩).trans
    (B2Adds sch N c δ ε hsec q k W S X (fun _ _ => t) (t c) ans help ζ M 0 H hc
      (fun _ _ j => ht (Nat.lt_succ_self j)) (fun z => fixed_setup (hset z)) (fun z => fixed_ingress (hin z))
      (fun _ => fixed_arrival) (fun _ => fixed_ticksIn) hcount (fun _ _ => fixed_timely ht c) hm hζ)

/-- **`B2AddsP2Ring` on a fixed schedule**, from `B2Adds_fixed`. -/
theorem B2AddsP2Ring_fixed :
  Pous.P2.Assumptions.P2RingAuditSecure 0 0 0 →
    ∀ {L : Type*} [LinearOrder L] (q k : ℕ) (W : Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).n)
      (S : Pous.Bits q → Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → Actual.Run L) (X : Finset L) (t : ℕ → ℕ)
      (ans : ℕ → Actual.Acts L → Pous.Bits (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).ℓ) {α : Type*}
      (help : Pous.Bits q → Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 → α) (ζ : ℕ → List Bool → Pous.Bits q) (M H : ℕ),
      (∀ z, CausalFrom (S z)) → StrictMono t →
      (∀ z x x', x.1 = x'.1 → (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).enc W x.1 x.2.1 = (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).enc W x'.1 x'.2.1 →
        (S z x).content X (t 0) = (S z x').content X (t 0) ∧ help z x = help z x') →
      (∀ z x x' (j : Fin 105), help z x = help z x' → x.2.2 j = x'.2.2 j →
        (S z x).crossing (↑X)ᶜ ↑X (window t j) = (S z x').crossing (↑X)ᶜ ↑X (window t j)) →
      (∀ z, (Set.range (help z)).encard ≤ ((2 ^ H : ℕ) : ℕ∞)) →
      (∀ z x (j : Fin 105), (S z x).mem X (t j) ≤ M) →
      (∀ z x (j : Fin 105), ζ j ((S z x).content X (t j)) = z) →
      Pous.pr (fun x : Pous.Bits q × Pous.Outcome (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) 105 =>
          Responder.PassesFixed (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) W (S x.1) X t ans x.2 ∧ ¬ q + (Pous.p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ < M + 1 + H + k) ≤
        ENNReal.ofReal Pous.Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) + 105 * (2⁻¹ : ℝ≥0∞) ^ k := by
  intro hsec L _ q k W S X t ans α help ζ M H hc ht hset hin hcount hm hζ
  have := B2Adds_fixed _ _ _ _ _ hsec q k W S X t ans help ζ M H hc ht hset hin hcount hm hζ
  rwa [Nat.cast_ofNat] at this

end Accounting.SecurityProofs
