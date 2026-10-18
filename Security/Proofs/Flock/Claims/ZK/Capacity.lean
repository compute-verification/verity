import Proofs.Flock.Claims.ZK.CoinLayer

/-!
# Zero knowledge at the reference session on a class of statements

`RefSession.TapeOK` asks, at every statement, that the session's one finite tape be read with fibers of one size, and
no session with statements of every table count meets it (`RefSession.not_tapeOK`). The masked record's tape is one
finite type too (`Claims.Flock.ZkDeploy.Rest`, read as `tapeOf`, with no statement). Here the tape is asked for only on
a class `A` of statements, as a tape of bounded size can meet it, and the bound outside `A` is `1`, which every pair of
probabilities meets:

* `RefSession.TapeOKOn A` and `RefSession.HonestMsgsOn A R`: `TapeOK` and `HonestMsgs` at the statements in `A`, the
  latter at their witnesses in `R`. Zero knowledge reads the prover at the relation's witnesses only; outside it the
  reference prover stops where its batched constraint fails (`innerP`), a test of the clear values the views mask.
  `RefSession.msgLinkOn` is `TMsgLink` on the relation's statements in `A` from `TapeOKOn A`, as `msgLink` has it from
  `TapeOK`. **`RefSession.tapeOKOn_of_tape`**: `TapeOKOn A` holds when the tape is read with fibers of one size at each
  statement in `A` (`FibersEq (S.tape x)`), since each table's randomness is read from that reading with fibers of one
  size (`fibersEq_tabRand`: `randOf` is a bijection, and `baseWith` drops only the base's revealed salts).
* `deployOn D A`: the record `D` whose parser refuses every statement outside `A`, so its relation is `D`'s on `A`
  (`r_deployOn`).
* `zkmalR_congr_on`: `zkmalR_congr` with the two provers agreeing only on the relation's statements. `zkmalR_on`:
  `ZKmalR` on the relation's statements in `A` is `ZKmalR` on the relation at the bound `1` outside `A`.
* `Flock.zkAt_deployZk_ref_on`: `Claims.Flock.ZKAt` at `refP`'s prover at `deployOn (deployZk rel) A`, under the
  session's Props with `TapeOKOn A` for `TapeOK`. `zk_zkOf_on`: `zk_zkOf` with `Proto.HonestOn A R` at the relation
  for `Proto.Honest`.
* **`Flock.zk_zkD_ref_on`**: `Claims.Flock.ZK` at the masked record `RefSession.zkD`, at `Flock.zk_zkD_ref`'s `ε_zk`
  on `A` and `1` outside it, under `TapeOKOn A` and `HonestMsgsOn A`. **`Flock.zk_deployZk_ref_on`** (and `_gap`): the
  same at the deployed masked record `zkDeployed`, with `TapeOKOn A` from the tables' tape read with fibers of one size
  on `A`, where `Flock.zk_deployZk_ref` asks `TapeOK`, which no tables with statements of every table count meet.
-/

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## `ZKmalR` on a class of statements -/

/-- **`ZKmalR` carries over to a prover with the same messages and more calls where the relation holds**, at `γ ≥ 0`:
`zkmalR_congr` with the two provers agreeing only on the relation's statements. -/
theorem zkmalR_congr_on {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg}
    {k : Msg → ℕ} {P Q : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ} (hγ : ∀ x, 0 ≤ γ x)
    (hv : ∀ w x, R x w → ∀ (hist : List (Msg × Msg)) ω, hist.length < k x →
      (((P w).next x hist ω).eval H).1 = (((Q w).next x hist ω).eval H).1)
    (hl : ∀ w x, R x w → ∀ (hist : List (Msg × Msg)) ω, hist.length < k x →
      (((P w).next x hist ω).eval H).2 ⊆ (((Q w).next x hist ω).eval H).2)
    (h : ZKmalR H k P R S εzk γ) : ZKmalR H k Q R S εzk γ := by
  intro x w hR Θ Vs θ hk Ev
  have e : ∀ ω, (runStar H (Q w) Vs x ω θ).msgs = (runStar H (P w) Vs x ω θ).msgs := fun ω =>
    (runStar_msgs_congr H (P w) (Q w) Vs x ω θ fun hist hh => hv w x hR hist ω (hk ▸ hh)).symm
  simp only [e]
  exact (h x w hR Θ Vs θ hk Ev).trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left
    (fork_pr_mono H (P w) (Q w) Vs θ x (fun hist ω hh => hv w x hR hist ω (hk ▸ hh))
      (fun hist ω hh => hl w x hR hist ω (hk ▸ hh))) (hγ x)))

/-- **`ZKmalR` on the relation's statements in `A` is `ZKmalR` on the relation at the bound `1` outside `A`**, at
`γ ≥ 0`: outside `A` any two probabilities are within `1`. -/
theorem zkmalR_on {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : Msg → ℕ}
    {P : W → Prover (N × Ω)} {R R' : Rel W} {S : Simulator} {εzk γ : Msg → ℝ} (A : Msg → Prop) [DecidablePred A]
    (hγ : ∀ x, 0 ≤ γ x) (hR : ∀ x w, R x w → A x → R' x w) (h : ZKmalR H k P R' S εzk γ) :
    ZKmalR H k P R S (fun x => if A x then εzk x else 1) γ := by
  intro x w hxw Θ Vs θ hk Ev
  beta_reduce
  by_cases hA : A x
  · simp only [hA, ↓reduceIte]
    exact h x w (hR x w hxw hA) Θ Vs θ hk Ev
  · simp only [hA, ↓reduceIte]
    have h0 : 0 ≤ γ x * (Fork H (P w) Vs θ x).Pr H :=
      mul_nonneg (hγ x) ((Pr_nonneg _).trans (Fork_pr_ge H (P w) Vs θ x))
    have h1 := Pr_le_one fun ω : N × Ω => Ev (runStar H (P w) Vs x ω θ).msgs
    have h2 := Pr_nonneg fun ω : N × Ω => Ev (runStar H (P w) Vs x ω θ).msgs
    have h3 := Pr_le_one fun c : S.C x => Ev (S.out H Vs θ x c)
    have h4 := Pr_nonneg fun c : S.C x => Ev (S.out H Vs θ x c)
    rw [abs_le]
    constructor <;> linarith

/-- **The record on the statements in `A`**: `D` with its parser refusing every statement outside `A`. -/
def deployOn {Coin Col T : Type} (D : Flock.Deploy Coin Col T) (A : Msg → Prop) [DecidablePred A] :
    Flock.Deploy Coin Col T :=
  { D with parse := fun x => if A x then D.parse x else none }

theorem r_deployOn {Coin Col T : Type} (D : Flock.Deploy Coin Col T) (A : Msg → Prop) [DecidablePred A] (x : Msg)
    (w : Flock.Wit) : Flock.R (deployOn D A).parse x w ↔ A x ∧ Flock.R D.parse x w := by
  by_cases hA : A x <;> simp [Flock.R, deployOn, hA]

/-! ## The masked record on a class of statements -/

/-- **`G`'s messages are the reference prover's at the statements in `A`**, at their witnesses in `R`: `Proto.Honest`
there. Zero knowledge reads the prover only at a witness in the relation (`zk_zkOf_on`). -/
def Proto.HonestOn {Ω : Type} (G : Proto Flock.Wit) (A : Msg → Prop) (R : Rel Flock.Wit) (H : Msg → Msg)
    (P : Flock.Wit → Prover Ω) (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → Ω) : Prop :=
  ∀ x, A x → ∀ w, R x w → ∀ ν r (ops : List (Msg × Msg)), ops.length ≤ G.K x →
    ((G.msg x w ν r (ops.map fun p => G.dec p.2)).eval H).1 =
        (((P w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, r))).eval H).1 ∧
      ((G.msg x w ν r (ops.map fun p => G.dec p.2)).eval H).2 ⊆
        (((P w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, r))).eval H).2

/-- **The masked record's reference prover sends `G.prover`'s messages and makes its calls** at a statement in `A` and
a witness in `R`, where `G`'s messages are the reference prover's there (`Proto.HonestOn A R`): `zkOf_prover` at one
statement. -/
theorem zkOf_prover_on (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) (H : Msg → Msg)
    (A : Msg → Prop) (R : Rel Flock.Wit) (hG : G.HonestOn A R H FlockComplete.ZkTables.honestPR coinOf tapeOf)
    (w : Flock.Wit) (x : Msg) (hx : A x) (hR : R x w) (hist : List (Msg × Msg)) (ω : G.Nonce × G.Rand)
    (hlen : hist.length < G.k x) :
    (((G.prover w).next x hist ω).eval H).1 = ((((zkOf rel G coinOf tapeOf S).prover w).next x hist ω).eval H).1 ∧
      (((G.prover w).next x hist ω).eval H).2 ⊆ ((((zkOf rel G coinOf tapeOf S).prover w).next x hist ω).eval H).2 := by
  cases hist with
  | nil => exact ⟨rfl, List.Subset.refl _⟩
  | cons p ops =>
    have hops : ops.length ≤ G.K x := by
      simp only [List.length_cons, Proto.k] at hlen
      omega
    obtain ⟨a, ρ⟩ := p
    have e₁ : (G.prover w).next x ((a, ρ) :: ops) ω = (Claims.Flock.allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind
        fun ok => if ok then G.msg x w ω.1 ω.2 (ops.map fun p => G.dec p.2) else .ret G.abort := by
      show ocBind (allOk (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)) _ = _
      rw [ocBind_eq_bind, allOk_eq_allPass]
    have e₂ : ((zkOf rel G coinOf tapeOf S).prover w).next x ((a, ρ) :: ops) ω =
        (Claims.Flock.allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind fun ok =>
          if ok then (FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf ω)
          else .ret G.abort := rfl
    rw [e₁, e₂, OC.eval_bind, OC.eval_bind]
    obtain ⟨h₁, h₂⟩ := hG x hx w hR ω.1 ω.2 ops hops
    cases ((Claims.Flock.allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).eval H).1 with
    | true =>
        simp only [↓reduceIte]
        exact ⟨h₁, List.append_subset.mpr
          ⟨List.subset_append_left _ _, List.Subset.trans h₂ (List.subset_append_right _ _)⟩⟩
    | false => exact ⟨rfl, List.Subset.refl _⟩

/-- **`Claims.Flock.ZK` at the masked record of a committed-coin protocol, at the bound `1` outside `A`**, from `ZKAt`
at its prover on the statements in `A` (`deployOn`): `zk_zkOf` with `G`'s messages the reference prover's on `A` only,
at the relation's witnesses (`Proto.HonestOn`), at `γ ≥ 0`. -/
theorem zk_zkOf_on (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) {H : Msg → Msg}
    {εzk γ : Msg → ℝ} (A : Msg → Prop) [DecidablePred A] (hγ : ∀ x, 0 ≤ γ x)
    (hk : ∀ x, G.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (hG : G.HonestOn A (Flock.R (FlockDeploy.deployZk rel).parse) H FlockComplete.ZkTables.honestPR coinOf tapeOf)
    (h : Claims.Flock.ZKAt H (deployOn (FlockDeploy.deployZk rel) A) G.prover S εzk γ) :
    Claims.Flock.ZK H (zkOf rel G coinOf tapeOf S) (fun x => if A x then εzk x else 1) γ := by
  have h' : ZKmalR H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (zkOf rel G coinOf tapeOf S).prover
      (Flock.R (deployOn (FlockDeploy.deployZk rel) A).parse) S εzk γ :=
    zkmalR_congr_on hγ
      (fun w x hR hist ω hh => (zkOf_prover_on rel G coinOf tapeOf S H A _ hG w x
        ((r_deployOn (FlockDeploy.deployZk rel) A x w).1 hR).1 ((r_deployOn (FlockDeploy.deployZk rel) A x w).1 hR).2
        hist ω (lt_of_lt_of_eq hh (hk x).symm)).1)
      (fun w x hR hist ω hh => (zkOf_prover_on rel G coinOf tapeOf S H A _ hG w x
        ((r_deployOn (FlockDeploy.deployZk rel) A x w).1 hR).1 ((r_deployOn (FlockDeploy.deployZk rel) A x w).1 hR).2
        hist ω (lt_of_lt_of_eq hh (hk x).symm)).2)
      h.2
  exact ⟨h.1, zkmalR_on A hγ (fun x w hR hA => (r_deployOn (FlockDeploy.deployZk rel) A x w).2 ⟨hA, hR⟩) h'⟩

/-! ## Fibers of one size -/

/-- **Every value of `f` is taken equally often.** -/
def FibersEq {α β : Type} (f : α → β) : Prop := ∀ b b', Nat.card {a // f a = b} = Nat.card {a // f a = b'}

/-- A first projection through an equivalence has fibers of one size, the second factor. -/
theorem fibersEq_of_equiv {α β D : Type} {f : α → β} (e : α ≃ β × D) (h : ∀ a, (e a).1 = f a) : FibersEq f := by
  have k : ∀ b, {a // f a = b} ≃ D := fun b =>
    { toFun := fun a => (e a.1).2
      invFun := fun d => ⟨e.symm (b, d), by rw [← h, Equiv.apply_symm_apply]⟩
      left_inv := fun a => Subtype.ext (e.injective (by
        rw [Equiv.apply_symm_apply]
        exact Prod.ext ((h a.1).trans a.2).symm rfl))
      right_inv := fun d => by simp }
  exact fun b b' => by rw [Nat.card_congr (k b), Nat.card_congr (k b')]

theorem fibersEq_of_bijective {α β : Type} {f : α → β} (hf : Function.Bijective f) : FibersEq f := fun b b' => by
  rw [ZkView.card_fiber_of_bijective hf b, ZkView.card_fiber_of_bijective hf b']

/-- Fibers of one size compose, from a finite type: a map with fibers of one size from a nonempty type is onto. -/
theorem FibersEq.comp {α β γ : Type} [Finite α] {f : α → β} {g : β → γ} (hf : FibersEq f) (hg : FibersEq g) :
    FibersEq (g ∘ f) := by
  classical
  rcases isEmpty_or_nonempty α with hα | hα
  · intro c c'
    simp [Nat.card_of_isEmpty]
  obtain ⟨a₀⟩ := hα
  have hs : Function.Surjective f := fun b => by
    have : Nonempty {a // f a = f a₀} := ⟨⟨a₀, rfl⟩⟩
    have h0 : 0 < Nat.card {a // f a = f a₀} := Nat.card_pos
    rw [hf (f a₀) b] at h0
    obtain ⟨a, ha⟩ := (Nat.card_pos_iff.1 h0).1.some
    exact ⟨a, ha⟩
  have : Finite β := Finite.of_surjective f hs
  have : Fintype α := Fintype.ofFinite α
  have : Fintype β := Fintype.ofFinite β
  have e : ∀ c, {a // (g ∘ f) a = c} ≃ Σ b : {b // g b = c}, {a // f a = b.1} := fun c =>
    { toFun := fun a => ⟨⟨f a.1, a.2⟩, ⟨a.1, rfl⟩⟩
      invFun := fun p => ⟨p.2.1, by simp only [Function.comp, p.2.2, p.1.2]⟩
      left_inv := fun a => rfl
      right_inv := by
        rintro ⟨⟨b, hb⟩, ⟨a, ha⟩⟩
        simp only at ha
        subst ha
        rfl }
  intro c c'
  rw [Nat.card_congr (e c), Nat.card_congr (e c'), Nat.card_sigma, Nat.card_sigma]
  simp only [fun b : β => hf b (f a₀), Finset.sum_const, Finset.card_univ, smul_eq_mul]
  have := hg c c'
  rw [Nat.card_eq_fintype_card, Nat.card_eq_fintype_card] at this
  rw [this]

/-- Fibers of one size, coordinatewise over a finite index. -/
theorem FibersEq.pi {ι : Type} [Fintype ι] {α β : ι → Type} {f : ∀ i, α i → β i} (h : ∀ i, FibersEq (f i)) :
    FibersEq fun (a : ∀ i, α i) i => f i (a i) := by
  have k : ∀ r : ∀ i, β i, {a : ∀ i, α i // (fun i => f i (a i)) = r} ≃ ∀ i, {x // f i x = r i} := fun r =>
    (Equiv.subtypeEquivRight fun a => funext_iff).trans (Equiv.subtypePiEquivPi (p := fun i x => f i x = r i))
  intro r r'
  rw [Nat.card_congr (k r), Nat.card_congr (k r'), Nat.card_pi, Nat.card_pi]
  exact Finset.prod_congr rfl fun i _ => h i (r i) (r' i)

/-- The salts of a table's base the final message reveals: level 0's opened leaves', each rep's opened pads leaves',
`y_τ`. -/
abbrev RevealedSalts (z : ZkView.Sizes) :=
  (Fin z.nS → ZkView.Salt) × (Fin 2 → Fin z.nQ → ZkView.Salt) × (Fin 2 → ZkView.Salt)

/-- **A table's randomness and hidden leaves' salts from a reading of the tape**: its base, every pre-final leaf's salt
and seed, the base's revealed salts read from the leaves' (`baseWith`, `hidOf`), as `RefSession.randAt` reads them. -/
noncomputable def tabRand {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z)
    (a : ZkView.Base z Hi × (ZkView.LeafPos z → ZkView.Salt) × (ZkView.LeafPos z → ZkView.Bits × ZkView.Salt)) :
    T.tab.Rand × (T.tab.Hid → ZkView.Salt) :=
  (T.randOf (baseWith T a.1 a.2.1) a.2.2, hidOf T a.2.1)

/-- The leaves' salts from a base's revealed ones and the hidden leaves'. -/
def mergeSalt {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (b : ZkView.Base z Hi)
    (h : T.tab.Hid → ZkView.Salt) : ZkView.LeafPos z → ZkView.Salt :=
  Sum.elim (fun p => Sum.elim b.2.2.2.2.2.2.1 (fun q => h (.inl q)) (T.s0.symm p))
    (Sum.elim (fun rp => Sum.elim (b.2.2.2.2.2.2.2.1 rp.1) (fun k => h (.inr (rp.1, k))) ((T.sP rp.1).symm rp.2))
      b.2.2.2.2.2.2.2.2.1)

/-- A base and the leaves' salts, as the base with the leaves' revealed salts, the hidden leaves' salts and the base's
own revealed salts. -/
def splitSalt {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) :
    ZkView.Base z Hi × (ZkView.LeafPos z → ZkView.Salt) ≃
      ZkView.Base z Hi × (T.tab.Hid → ZkView.Salt) × RevealedSalts z where
  toFun a := (baseWith T a.1 a.2, hidOf T a.2, a.1.2.2.2.2.2.2.1, a.1.2.2.2.2.2.2.2.1, a.1.2.2.2.2.2.2.2.2.1)
  invFun c := ((c.1.1, c.1.2.1, c.1.2.2.1, c.1.2.2.2.1, c.1.2.2.2.2.1, c.1.2.2.2.2.2.1, c.2.2.1, c.2.2.2.1, c.2.2.2.2,
    c.1.2.2.2.2.2.2.2.2.2), mergeSalt T c.1 c.2.1)
  left_inv a := by
    refine Prod.ext rfl (funext fun p => ?_)
    rcases p with p | ⟨r, p⟩ | r
    · obtain ⟨q, rfl⟩ := T.s0.surjective p
      change Sum.elim _ _ (T.s0.symm (T.s0 q)) = _
      rw [Equiv.symm_apply_apply]
      rcases q with q | q <;> rfl
    · obtain ⟨q, rfl⟩ := (T.sP r).surjective p
      change Sum.elim _ _ ((T.sP r).symm ((T.sP r) q)) = _
      rw [Equiv.symm_apply_apply]
      rcases q with q | q <;> rfl
    · rfl
  right_inv c := by
    refine Prod.ext ?_ (Prod.ext (funext fun q => ?_) rfl)
    · simp only [baseWith, mergeSalt, Sum.elim_inl, Sum.elim_inr, Equiv.symm_apply_apply]
      rfl
    · rcases q with q | ⟨r, k⟩
      · simp only [hidOf, mergeSalt, Sum.elim_inl, Equiv.symm_apply_apply]
        rfl
      · simp only [hidOf, mergeSalt, Sum.elim_inl, Sum.elim_inr, Equiv.symm_apply_apply]
        rfl

/-- **A table's randomness is read from the tape's reading with fibers of one size**, `RevealedSalts`: `randOf` is a
bijection, and `baseWith` replaces the base's revealed salts by the leaves'. -/
theorem fibersEq_tabRand {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) :
    FibersEq (tabRand T) := by
  let r : ZkView.Base z Hi × (ZkView.LeafPos z → ZkView.Bits × ZkView.Salt) ≃ T.tab.Rand :=
    Equiv.ofBijective _ ⟨T.randOf_injective, T.randOf_surjective⟩
  let e : ZkView.Base z Hi × (ZkView.LeafPos z → ZkView.Salt) × (ZkView.LeafPos z → ZkView.Bits × ZkView.Salt) ≃
      (T.tab.Rand × (T.tab.Hid → ZkView.Salt)) × RevealedSalts z :=
    (Equiv.prodAssoc _ _ _).symm.trans <| ((splitSalt T).prodCongr (Equiv.refl _)).trans <|
      { toFun := fun c => ((r (c.1.1, c.2), c.1.2.1), c.1.2.2)
        invFun := fun d => (((r.symm d.1.1).1, d.1.2, d.2), (r.symm d.1.1).2)
        left_inv := fun c => by simp
        right_inv := fun d => by simp }
  exact fibersEq_of_equiv e fun _ => rfl

/-! ## The reference session on a class of statements -/

namespace RefSession

/-- **The tape is read with fibers of one size at the statements in `A`**: `TapeOK` there. -/
def TapeOKOn (S : RefSession) (A : Msg → Prop) : Prop :=
  ∀ x, A x → (∀ p p', Nat.card {t // S.eP x t = p} = Nat.card {t // S.eP x t = p'}) ∧
    ∀ c p p', Nat.card {t // S.randAt x c t = p} = Nat.card {t // S.randAt x c t = p'}

theorem tapeOKOn_iff (S : RefSession) : S.TapeOKOn (fun _ => True) ↔ S.TapeOK :=
  ⟨fun h x => h x trivial, fun h x _ => h x⟩

/-- The tape's reading at a statement, by table. -/
def byTable (S : RefSession) (x : Msg) :
    ((∀ j, ZkView.Base (S.C x j).z (S.C x j).Hi) × (ZkView.Leaves (fun j => (S.C x j).z) → ZkView.Salt) ×
      (ZkView.Leaves (fun j => (S.C x j).z) → ZkView.Bits × ZkView.Salt)) ≃
    ∀ j, ZkView.Base (S.C x j).z (S.C x j).Hi × (ZkView.LeafPos (S.C x j).z → ZkView.Salt) ×
      (ZkView.LeafPos (S.C x j).z → ZkView.Bits × ZkView.Salt) where
  toFun a j := (a.1 j, fun p => a.2.1 ⟨j, p⟩, fun p => a.2.2 ⟨j, p⟩)
  invFun g := (fun j => (g j).1, fun l => (g l.1).2.1 l.2, fun l => (g l.1).2.2 l.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-- **`TapeOKOn A` is the tape read with fibers of one size at the statements in `A`**: the tables' randomness is read
from that reading with fibers of one size (`fibersEq_tabRand`), whatever the coins, and so is the pre-final view's part
of it. -/
theorem tapeOKOn_of_tape (S : RefSession) (A : Msg → Prop) (h : ∀ x, A x → FibersEq (S.tape x)) :
    S.TapeOKOn A := by
  intro x hx
  have he : FibersEq ((fun b => (b.1, b.2.1)) ∘ S.tape x) :=
    (h x hx).comp (fibersEq_of_equiv (Equiv.prodAssoc _ _ _).symm fun _ => rfl)
  refine ⟨he, fun c => ?_⟩
  have hr : FibersEq ((fun a j => tabRand ((S.C x j).T (S.cz x c j)) (a j)) ∘ (S.byTable x ∘ S.tape x)) :=
    ((h x hx).comp (fibersEq_of_bijective (S.byTable x).bijective)).comp
      (FibersEq.pi fun j => fibersEq_tabRand ((S.C x j).T (S.cz x c j)))
  exact hr

/-- **`refP`'s messages are the reference prover's at the statements in `A`**, at their witnesses in `R`: `HonestMsgs`
there. -/
def HonestMsgsOn (S : RefSession) (A : Msg → Prop) (R : Rel Flock.Wit) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) : Prop :=
  ∀ x, A x → ∀ w, R x w → ∀ ν t (ops : List (Msg × Msg)), ops.length ≤ refK x →
    S.msgAt x (S.wt x w) t (ops.map fun p => S.decOf p.2) =
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, t))).eval H).1

theorem honestOn_of_msgs (S : RefSession) (A : Msg → Prop) (R : Rel Flock.Wit) (H : Msg → Msg) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk) (h : S.HonestMsgsOn A R H coinOf tapeOf) :
    S.refP.HonestOn A R H FlockComplete.ZkTables.honestPR coinOf tapeOf :=
  fun x hx w hR ν t ops hops => ⟨h x hx w hR ν t ops hops, List.nil_subset _⟩

/-- `preModel_link` at a statement in `A`, from `TapeOKOn A`. -/
theorem preModel_link_on (S : RefSession) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hT : S.TapeOKOn A) (x : Msg) (hx : A x) (w : Flock.Wit) (hR : R x w)
    (ν : S.Nonce) : (S.preModel H x ν).Link H ν w := by
  refine ⟨hV x w hR, fun a j => hD x (S.coinsOf a) j, ?_, fun p => (hT x hx).1 p _, fun t a => funext fun i =>
    S.pre_msg H x ν hc (S.wt x w) t a i, fun t a => funext fun i => S.pre_msg H x ν hc (S.wt₀ x) t a i⟩
  intro wt ω p a a' i h
  show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
  rw [ansHist_congr a a' i h]
  refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
  exact hc x wt ω p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
    congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))

/-- `coinModel_link` at a statement in `A`, from `TapeOKOn A`. -/
theorem coinModel_link_on (S : RefSession) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hT : S.TapeOKOn A) (x : Msg) (hx : A x) (w : Flock.Wit) (hR : R x w)
    (ν : S.Nonce) (c : Fin (refK x) → S.Coin) : (S.coinModel x c).Link H ν c w := by
  unfold coinModel
  split_ifs with hAd
  · refine ⟨hV x w hR, fun p => (hT x hx).2 c p _, fun (t : S.Tape) => ?_⟩
    show (List.range (refK x + 1)).map (fun j => S.msgAt x (S.wt x w) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.cutV x c (S.viewsAt x c (S.wt x w) t)) else S.fin x c (S.viewsAt x c (S.wt x w) t)
    refine List.map_congr_left fun j hj => ?_
    rw [S.msgAt_take x hc _ t c j (List.mem_range.1 hj), S.cutV_viewsAt, ite_eq_left hAd]
  · refine ⟨hV x w hR, fun j => hD x c j, fun p => (hT x hx).1 p _, fun (t : S.Tape) => ?_,
      fun (t : S.Tape) => ?_⟩
    · show (List.range (refK x + 1)).map (fun j => S.msgAt x (S.wt x w) t ((List.ofFn c).take j)) =
        (List.range (refK x + 1)).map fun j =>
          if j < refK x then S.enc x c j (S.cutOf x c (S.wt x w) t) else S.off x c (S.cutOf x c (S.wt x w) t)
      refine List.map_congr_left fun j hj => ?_
      rw [S.msgAt_take x hc _ t c j (List.mem_range.1 hj), ite_eq_right hAd]
    · show (List.range (refK x + 1)).map (fun j => S.msgAt x (S.wt₀ x) t ((List.ofFn c).take j)) =
        (List.range (refK x + 1)).map fun j =>
          if j < refK x then S.enc x c j (S.cutOf x c (S.wt₀ x) t) else S.off x c (S.cutOf x c (S.wt₀ x) t)
      refine List.map_congr_left fun j hj => ?_
      rw [S.msgAt_take x hc _ t c j (List.mem_range.1 hj), ite_eq_right hAd]

/-- **`TMsgLink` holds of `refP`'s models on the relation's statements in `A`**, from `TapeOKOn A`: `msgLink` with
`TapeOK` asked for only there. -/
theorem msgLinkOn (S : RefSession) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x) :
    S.refP.TMsgLink H R (fun x ν => S.preModel H x ν) fun x _ c => S.coinModel x c := fun x w hR ν =>
  ⟨S.preModel_link_on H R A hV hD hc hT x (hRA x w hR) w hR ν,
    fun c => S.coinModel_link_on H R A hV hD hc hT x (hRA x w hR) w hR ν c⟩

end RefSession

/-- **Flock's zero knowledge at the reference prover, on the statements in `A`**: `Flock.zkAt_deployZk_ref` at the
record whose parser refuses every statement outside `A` (`deployOn`), with `TapeOK` asked for only on `A`. -/
theorem Flock.zkAt_deployZk_ref_on (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession) (A : Msg → Prop)
    [DecidablePred A] (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete fun x w => A x ∧ Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hD : S.DummyReg) (hc : S.Causal) (hT : S.TapeOKOn A) :
    Claims.Flock.ZKAt H (deployOn (FlockDeploy.deployZk rel) A) S.refP.prover
      ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  have hV' : S.Complete (Flock.R (deployOn (FlockDeploy.deployZk rel) A).parse) := fun x w hR =>
    hV x w ((r_deployOn (FlockDeploy.deployZk rel) A x w).1 hR)
  have hL := S.msgLinkOn H _ A hV' hD hc hT fun x w hR => ((r_deployOn (FlockDeploy.deployZk rel) A x w).1 hR).1
  have hOK := S.tablesOK H hS
  have hA : ∀ x w, Flock.R (deployOn (FlockDeploy.deployZk rel) A).parse x w → ∀ (ν : S.Nonce)
      (V : FlockSoundness.ZK.Responder Msg Out (refK x + 1)) (E : (Fin (refK x + 1) → Msg) → Prop),
      |Pr (fun r : S.Tape => E (FlockSoundness.ZK.run (S.refP.preStrat H x ν (S.refP.msg x w ν)) V r)) -
        Pr (fun r : S.Tape => E (FlockSoundness.ZK.run (S.refP.preStrat H x ν (S.refP.dmsg x ν)) V r))| ≤
        2 * S.nLeaves x * δ₁.toReal := fun x w hR ν V E =>
    (S.preModel H x ν).runs_close H hT1 hδ₁ ν w (hL x w hR ν).1 (hOK.1 x ν) V E
  rw [← S.refP.withSim_prover fun x _ c => S.coinModel x c]
  exact Flock.zkR_gkE_hm96_bind H (deployOn (FlockDeploy.deployZk rel) A)
    (S.refP.withSim fun x _ c => S.coinModel x c) t ht rfl (fun x => (deployZk_proto_k rel _ (fun _ => rfl) x).symm) δ₁
    S.nLeaves S.nLeaves hA (S.refP.withSim_shvzk H _ hT1 hδ₁ _ S.nLeaves (fun x _ c => S.coinModel_size x c)
      (fun x w hR ν c => (hL x w hR ν).2 c) hOK.2) (S.binding H)

/-- **Flock's zero knowledge at the reference session's masked record, on a class of statements**: `Claims.Flock.ZK`
at `RefSession.zkD`, at `Flock.zk_zkD_ref`'s `ε_zk` on the statements in `A` and `1` outside them, under
`Claims.Hm96Hiding` at the pinned key, `Tables`, `Complete` on `A`, `DummyReg`, `Causal`, and `TapeOK` and `HonestMsgs`
on `A` only (`TapeOKOn`, `HonestMsgsOn`, the latter at the relation's witnesses): a tape of bounded size can meet
`TapeOKOn A` where `A` bounds the tables. -/
theorem Flock.zk_zkD_ref_on (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession) (A : Msg → Prop)
    [DecidablePred A] (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables) (hV : S.Complete fun x w => A x ∧ Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hD : S.DummyReg) (hc : S.Causal) (hT : S.TapeOKOn A) (coinOf : Msg → Msg)
    (tapeOf : S.refP.Nonce × S.refP.Rand → FlockComplete.TapeZk)
    (hH : S.HonestMsgsOn A (Flock.R (FlockDeploy.deployZk rel).parse) H coinOf tapeOf) :
    Claims.Flock.ZK H (S.zkD rel coinOf tapeOf t)
      (fun x => if A x then 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  zk_zkOf_on rel S.refP coinOf tapeOf _ A (fun _ => by positivity) (deployZk_proto_k rel S.refP fun _ => rfl)
    (S.honestOn_of_msgs A _ H coinOf tapeOf hH) (Flock.zkAt_deployZk_ref_on H rel S A t ht hT1 hδ₁ hS hV hD hc hT)

/-- **Flock's zero knowledge at the deployed masked record, on a class of statements**: `Claims.Flock.ZK` at
`zkDeployed`, at `Flock.zk_deployZk_ref`'s `ε_zk` on the statements in `A` and `1` outside them, with the tape asked
for only on `A`, and there only as the tables' tape read with fibers of one size (`RefSession.tapeOKOn_of_tape`).
`Flock.zk_deployZk_ref`'s `TapeOK` holds at no tables with statements of every table count (`RefSession.not_tapeOK`). -/
theorem Flock.zk_deployZk_ref_on (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (A : Msg → Prop) [DecidablePred A] (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : (CoinLayer.session nr T).Tables)
    (hV : (CoinLayer.session nr T).Complete fun x w => A x ∧ Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hD : (CoinLayer.session nr T).DummyReg) (hc : (CoinLayer.session nr T).Causal)
    (hT : ∀ x, A x → FibersEq (T.tape x))
    (hH : (CoinLayer.session nr T).HonestMsgsOn A (Flock.R (FlockDeploy.deployZk rel).parse) H CoinLayer.coinOfD
      fun ω => T.tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed rel nr T t)
      (fun x => if A x then 8 * (CoinLayer.session nr T).nLeaves x * δ₁.toReal +
        8 * t * (CoinLayer.session nr T).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_zkD_ref_on H rel (CoinLayer.session nr T) A t ht hT1 hδ₁ hS hV hD hc
    ((CoinLayer.session nr T).tapeOKOn_of_tape A hT) CoinLayer.coinOfD _ hH

/-- **The same at the pinned key's hiding gap** (`CROnly.hm96Hiding_gap`), with `Claims.Hm96Hiding` discharged. -/
theorem Flock.zk_deployZk_ref_on_gap (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (A : Msg → Prop) [DecidablePred A] (t : ℕ) (ht : 0 < t) (hS : (CoinLayer.session nr T).Tables)
    (hV : (CoinLayer.session nr T).Complete fun x w => A x ∧ Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hD : (CoinLayer.session nr T).DummyReg) (hc : (CoinLayer.session nr T).Causal)
    (hT : ∀ x, A x → FibersEq (T.tape x))
    (hH : (CoinLayer.session nr T).HonestMsgsOn A (Flock.R (FlockDeploy.deployZk rel).parse) H CoinLayer.coinOfD
      fun ω => T.tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed rel nr T t)
      (fun x => if A x then 8 * (CoinLayer.session nr T).nLeaves x *
          (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        8 * t * (CoinLayer.session nr T).nLeaves x *
          (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        2 * (1 / 3) ^ t else 1) fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_ref_on H rel nr T A t ht (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)
    ENNReal.ofReal_ne_top hS hV hD hc hT hH

end Claims.ZKMal
