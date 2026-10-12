import Models.Accounting.Guarantees.Responder
import Proofs.Accounting.Actual.Confidence

/-!
# B2, proved

`LengthCode`: the code's last `1` sits at the string's length, so equal codes have equal lengths, then equal bits.

`B2` is a choice construction. The responder's state before challenge `j` is `stateOf`: `lengthCode M` of `X`'s state's
content at `t j`, and the help's `H`-bit code (`exists_code`, from the help's count), packed into `N` bits (`pack`).
The preprocessing picks an outcome with the primitive and the encoding it is given (`pre`); the step at challenge `j`
picks an outcome whose state codes to its state, with content of at most `M` bits, and whose challenge `j` is the one
asked (`out`), and returns that run's answer and its next state. At an outcome where `mem_X(t_j) ≤ M`, the pick has
the same content (`LengthCode`, with `content_length_le`) and the same help (the code is injective on the help's
values), so the same ingress during window `j`, so by `CausalFrom` the same `acts` and answer, and by `Decided` the
same state at `t (j + 1)` (`out_spec`). `decided` derives `Decided` from `CausalFrom`: since v2.6 what `X` does during
a window includes its state at the window's end. `seqAnswers_eq` and `seqState_eq` fold that over the challenges.

`B2Audit` applies PoUS's guarantee to that responder at `M = N − 1 − H`: an outcome that passes with
`mem_X(t_j) + 1 + H ≤ N` is one the responder answers correctly. `B2Determines` reads `z` back with a left inverse of
the length code (`decodeLen`).
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

namespace Responder

/-- **`X`'s state is decided at the challenge ticks**: for each challenge `j + 1 < k`, `X`'s state's content at
`t (j + 1)` is a function of its content at `t j` and of what reached it from outside during window `j`. -/
def Decided {L : Type*} [LinearOrder L] {Ω : Type*} (S : Ω → Actual.Run L) (X : Finset L) (t : ℕ → ℕ) (k : ℕ) :
    Prop :=
  ∀ j, j + 1 < k → ∀ r r', (S r).content X (t j) = (S r').content X (t j) →
    (S r).crossing (↑X)ᶜ ↑X (window t j) = (S r').crossing (↑X)ᶜ ↑X (window t j) →
    (S r).content X (t (j + 1)) = (S r').content X (t (j + 1))

/-- What `X` does during `[t, u)`, `t < u`, fixes its state at `u` as seen (v2.6: its state at the end of the window's
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

/-- **`Decided` from causality** (v2.6): with the challenge ticks increasing, a strategy causal from every tick has
`X`'s state decided at them. -/
theorem decided {L : Type*} [LinearOrder L] {Ω : Type*} {S : Ω → Actual.Run L} (hc : CausalFrom S) (X : Finset L)
    {t : ℕ → ℕ} (ht : StrictMono t) (k : ℕ) : Decided S X t k := by
  intro j _ r r' hcont hx
  have hacts := hc X (t j) (t (j + 1)) r r' hcont hx
  unfold Actual.Run.content
  rw [image_state_eq_of_acts (ht (Nat.lt_succ_self j)) hacts]

/-- `a` then `b`, then `false`s, in `N` bits. -/
def pack (M H N : ℕ) (a : Pous.Bits (M + 1)) (b : Pous.Bits H) : Pous.Bits N :=
  fun i => if h : i.val < M + 1 then a ⟨i, h⟩ else if h' : i.val - (M + 1) < H then b ⟨i - (M + 1), h'⟩ else false

theorem pack_inj {M H N : ℕ} (hN : M + 1 + H ≤ N) {a a' : Pous.Bits (M + 1)} {b b' : Pous.Bits H}
    (h : pack M H N a b = pack M H N a' b') : a = a' ∧ b = b' := by
  refine ⟨funext fun i => ?_, funext fun i => ?_⟩
  · have := congrFun h ⟨i, by omega⟩
    simpa only [pack, i.isLt, dite_true] using this
  · have := congrFun h ⟨M + 1 + i, by omega⟩
    have e : M + 1 + i.val - (M + 1) = i.val := by omega
    simp only [pack, show ¬ (M + 1 + i.val < M + 1) by omega, dite_false, e, i.isLt, dite_true] at this
    exact this

/-- The first `M + 1` bits of an `N`-bit state. -/
def unpack (M N : ℕ) (hN : M + 1 ≤ N) (σ : Pous.Bits N) : Pous.Bits (M + 1) := fun i => σ ⟨i, by omega⟩

theorem unpack_pack {M H N : ℕ} (hN : M + 1 ≤ N) (a : Pous.Bits (M + 1)) (b : Pous.Bits H) :
    unpack M N hN (pack M H N a b) = a := by
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
    (step : ℕ → Pous.Bits N → Fin B → Pous.Prog I (Pous.Bits ℓ × Pous.Bits N)) (k : ℕ) (σ : ℕ → Pous.Bits N)
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
    (step : ℕ → Pous.Bits N → Fin B → Pous.Prog I (Pous.Bits ℓ × Pous.Bits N)) (i : Fin B) :
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
    (step : ℕ → Pous.Bits N → Fin B → Pous.Prog I (Pous.Bits ℓ × Pous.Bits N)) (k : ℕ) (σ : ℕ → Pous.Bits N)
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
  (W : Pous.Bits sch.n) (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : ℕ → ℕ)
  (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ) {α : Type*} (help : Pous.Outcome sch k → α) (M H N : ℕ)
  (c : α → Pous.Bits H)

/-- The responder's state before challenge `j` at the outcome `x`: `X`'s content at `t j` coded in `M + 1` bits, and
the help's code. -/
noncomputable def stateOf (x : Pous.Outcome sch k) (j : ℕ) : Pous.Bits N :=
  pack M H N (lengthCode M ((S x).content X (t j))) (c (help x))

open Classical in
/-- The preprocessing: the state at `t 0` of an outcome with the given primitive and encoding. -/
noncomputable def pre : Pous.Preprocessor sch N := fun _ pp C ω =>
  if h : ∃ x : Pous.Outcome sch k, x.1 = ω ∧ sch.enc W ω x.2.1 = (pp, C) then
    stateOf sch S X t help M H N c (Classical.choose h) 0
  else fun _ => false

open Classical in
/-- What the step at challenge `j` returns from the state `σ` and the index `i`: the answer and the next state of an
outcome whose state codes to `σ` with content of at most `M` bits and whose challenge `j` is `i`. -/
noncomputable def out (j : ℕ) (σ : Pous.Bits N) (i : Fin sch.B) : Pous.Bits sch.ℓ × Pous.Bits N :=
  if hj : j < k then
    if h : ∃ x : Pous.Outcome sch k, x.2.2 ⟨j, hj⟩ = i ∧ ((S x).content X (t j)).length ≤ M ∧
        stateOf sch S X t help M H N c x j = σ then
      (ans j ((S (Classical.choose h)).acts X (window t j)), stateOf sch S X t help M H N c (Classical.choose h) (j + 1))
    else (fun _ => false, σ)
  else (fun _ => false, σ)

/-- The responder: `out`, with no query. -/
noncomputable def resp : Pous.SeqResponderOn sch N 0 0 0 where
  step _ _ j σ i := .ret (out sch S X t ans help M H N c j σ i)
  bounded _ _ _ _ _ := fun _ => by simp [Pous.Prog.mulDepth, Pous.Prog.keyWork, Pous.Prog.ringWork]

variable {sch W S X t ans help M H N c}

/-- When `X`'s state at `t 0` and the help depend on the outcome only through the primitive and the encoding, the
preprocessing returns `stateOf … 0`. -/
theorem pre_spec (hset' : ∀ x x', x.1 = x'.1 → sch.enc W x.1 x.2.1 = sch.enc W x'.1 x'.2.1 →
      (S x).content X (t 0) = (S x').content X (t 0) ∧ help x = help x') (x : Pous.Outcome sch k) :
    pre sch W S X t help M H N c W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1 =
      stateOf sch S X t help M H N c x 0 := by
  have h : ∃ x' : Pous.Outcome sch k, x'.1 = x.1 ∧
      sch.enc W x.1 x'.2.1 = ((sch.enc W x.1 x.2.1).1, (sch.enc W x.1 x.2.1).2) := ⟨x, rfl, rfl⟩
  obtain ⟨h1, h2⟩ := Classical.choose_spec h
  obtain ⟨hcont, hhelp⟩ := hset' _ _ h1 (by rw [h1]; exact h2)
  simp only [pre, dite_eq_left h, stateOf, hcont, hhelp]

section
variable (hc : CausalFrom S) (hd : Decided S X t k)
  (hin : ∀ x x' (j : Fin k), help x = help x' → x.2.2 j = x'.2.2 j →
    (S x).crossing (↑X)ᶜ ↑X (window t j) = (S x').crossing (↑X)ᶜ ↑X (window t j))
  (hcode : Set.InjOn c (Set.range help)) (hN : M + 1 + H ≤ N)
include hc hd hin hcode hN

/-- At an outcome where `X`'s content at `t j` has at most `M` bits, the step at challenge `j` answers as `X` does and,
unless `j` is the last challenge, moves to `X`'s next state. -/
theorem out_spec (x : Pous.Outcome sch k) (j : ℕ) (hj : j < k) (hm : ((S x).content X (t j)).length ≤ M) :
    (out sch S X t ans help M H N c j (stateOf sch S X t help M H N c x j) (x.2.2 ⟨j, hj⟩)).1 =
        ans j ((S x).acts X (window t j)) ∧
      (j + 1 < k → (out sch S X t ans help M H N c j (stateOf sch S X t help M H N c x j) (x.2.2 ⟨j, hj⟩)).2 =
        stateOf sch S X t help M H N c x (j + 1)) := by
  have h : ∃ x' : Pous.Outcome sch k, x'.2.2 ⟨j, hj⟩ = x.2.2 ⟨j, hj⟩ ∧ ((S x').content X (t j)).length ≤ M ∧
      stateOf sch S X t help M H N c x' j = stateOf sch S X t help M H N c x j := ⟨x, rfl, hm, rfl⟩
  obtain ⟨hi, hl, hs⟩ := Classical.choose_spec h
  obtain ⟨h1, h2⟩ := pack_inj hN hs
  have hcont := LengthCode M hl hm h1
  have hhelp := hcode ⟨_, rfl⟩ ⟨_, rfl⟩ h2
  have hx := hin _ _ ⟨j, hj⟩ hhelp hi
  have hacts := hc X (t j) (t (j + 1)) _ _ hcont hx
  have e : out sch S X t ans help M H N c j (stateOf sch S X t help M H N c x j) (x.2.2 ⟨j, hj⟩) =
      (ans j ((S (Classical.choose h)).acts X (window t j)),
        stateOf sch S X t help M H N c (Classical.choose h) (j + 1)) := by
    unfold out
    rw [dite_eq_left hj]
    exact dite_eq_left h
  rw [e]
  exact ⟨congrArg (ans j) hacts, fun hj1 =>
    congrArg₂ (pack M H N) (congrArg (lengthCode M) (hd j hj1 _ _ hcont hx)) (congrArg c hhelp)⟩

/-- **The responder answers as `X` does** at every outcome where `mem_X(t_j) ≤ M` at each challenge. -/
theorem seqAnswers_resp (hset' : ∀ x x', x.1 = x'.1 → sch.enc W x.1 x.2.1 = sch.enc W x'.1 x'.2.1 →
      (S x).content X (t 0) = (S x').content X (t 0) ∧ help x = help x') (x : Pous.Outcome sch k)
    (hm : ∀ j : Fin k, (S x).mem X (t j) ≤ M) :
    Pous.seqAnswers ((Pous.ringRO B w p).answer x.1) ((resp sch S X t ans help M H N c).step W (sch.enc W x.1 x.2.1).1)
        0 (List.ofFn x.2.2) (pre sch W S X t help M H N c W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) =
      List.ofFn fun j : Fin k => ans j ((S x).acts X (window t j)) := by
  have hl : ∀ j (hj : j < k), ((S x).content X (t j)).length ≤ M := fun j hj =>
    (Actual.content_length_le _ _ _).trans (hm ⟨j, hj⟩)
  let is : ℕ → Fin sch.B := fun j => if hj : j < k then x.2.2 ⟨j, hj⟩ else ⟨0, sch.B_pos⟩
  have key := seqAnswers_eq ((Pous.ringRO B w p).answer x.1) ((resp sch S X t ans help M H N c).step W
    (sch.enc W x.1 x.2.1).1) k (stateOf sch S X t help M H N c x) (fun j => ans j ((S x).acts X (window t j))) is
    (fun j hj => by
      simp only [resp, Pous.Prog.run, is, dite_eq_left hj]
      exact out_spec hc hd hin hcode hN x j hj (hl j hj)) k 0 (by omega)
  rw [pre_spec hset' x]
  simp only [Nat.zero_add] at key
  convert key using 2
  congr 1
  funext i
  simp [is, i.isLt]

/-- **The state before each challenge is `stateOf`** at every outcome where `mem_X(t_j) ≤ M` at each challenge. -/
theorem seqState_resp (hset' : ∀ x x', x.1 = x'.1 → sch.enc W x.1 x.2.1 = sch.enc W x'.1 x'.2.1 →
      (S x).content X (t 0) = (S x').content X (t 0) ∧ help x = help x') (x : Pous.Outcome sch k)
    (hm : ∀ j : Fin k, (S x).mem X (t j) ≤ M) (j : ℕ) (hj : j < k) :
    Pous.Sampled.seqState ((Pous.ringRO B w p).answer x.1)
        ((resp sch S X t ans help M H N c).step W (sch.enc W x.1 x.2.1).1) 0 ((List.ofFn x.2.2).take j)
        (pre sch W S X t help M H N c W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1) =
      stateOf sch S X t help M H N c x j := by
  rw [pre_spec hset' x]
  refine seqState_eq _ _ k (stateOf sch S X t help M H N c x) x.2.2 (fun j hj => ?_) j hj
  simp only [resp, Pous.Prog.run]
  exact (out_spec hc hd hin hcode hN x j (by omega)
    ((Actual.content_length_le _ _ _).trans (hm ⟨j, by omega⟩))).2 hj

end

end Responder

open Responder in
theorem B2 : Guarantees.B2 := by
  intro L _ B w p _ sch k W S X t ans α help M H N hc ht hset hin hcount hN
  obtain ⟨c, hcode⟩ := exists_code help hcount
  exact ⟨pre sch W S X t help M H N c, resp sch S X t ans help M H N c,
    fun x hm => seqAnswers_resp hc (decided hc X ht k) hin hcode hN hset x hm⟩

theorem B2Audit : Guarantees.B2Audit := by
  intro L _ B w p _ sch N k δ ε hsec W S X t ans α help H hc ht hset hin hcount
  by_cases hN : 1 + H ≤ N
  · obtain ⟨A₁, A₂, hA⟩ := B2 sch k W S X t ans help (N - 1 - H) H N hc ht hset hin hcount (by omega)
    refine (pr_mono fun x hx => ?_).trans (hsec W A₁ A₂)
    obtain ⟨hp, hm⟩ := hx
    unfold Pous.seqAcceptsOn
    dsimp only
    rw [hA x fun j => by have := hm j; omega]
    exact congrArg List.ofFn (funext hp)
  · let A₂ : Pous.SeqResponderOn sch N 0 0 0 :=
      { step := fun _ _ _ σ _ => .ret (fun _ => false, σ)
        bounded := fun _ _ _ _ _ _ => by simp [Pous.Prog.mulDepth, Pous.Prog.keyWork, Pous.Prog.ringWork] }
    refine (pr_mono fun x hx => ?_).trans (hsec W (fun _ _ _ _ _ => false) A₂)
    rcases Nat.eq_zero_or_pos k with hk | hk
    · subst hk
      simp [Pous.seqAcceptsOn, Pous.seqAnswers]
    · have := hx.2 ⟨0, hk⟩
      omega

theorem B2P2Ring : Guarantees.B2P2Ring := by
  intro hsec L _ W S X t ans α help H hc ht hset hin hcount
  exact B2Audit _ _ _ _ _ hsec W S X t ans help H hc ht hset hin hcount

open Responder in
theorem B2Determines : Guarantees.B2Determines := by
  intro L _ B w p _ sch k q W S X t ans α help ζ M H hc ht hset hin hcount hm hζ
  choose c hcode using fun z => exists_code (help z) (hcount z)
  have hN : M + 1 + H ≤ M + 1 + H := le_rfl
  refine ⟨fun z => pre sch W (S z) X t (help z) M H (M + 1 + H) (c z),
    fun z => resp sch (S z) X t ans (help z) M H (M + 1 + H) (c z),
    fun j σ => ζ j (decodeLen M (unpack M (M + 1 + H) (by omega) σ)), fun z x j hj => ?_, fun z x =>
      seqAnswers_resp (hc z) (decided (hc z) X ht k) (hin z) (hcode z) hN (hset z) x (hm z x)⟩
  rw [seqState_resp (hc z) (decided (hc z) X ht k) (hin z) (hcode z) hN (hset z) x (hm z x) j hj]
  simp only [stateOf]
  rw [unpack_pack, decodeLen_lengthCode ((Actual.content_length_le _ _ _).trans (hm z x ⟨j, hj⟩))]
  exact hζ z x ⟨j, hj⟩

end Accounting.SecurityProofs
