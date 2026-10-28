import Proofs.Properties.Verity.Network.Legs

/-!
# The network certifier's coupling: S4 and S6 by construction

S4 (`WindowCoinsLive`) and S6's `stmt` and `wit` (`NetworkCouples`) say how a realization at a coupling's run is built
from the play, not what the code checks, so they hold by construction at a realization built from these pieces:

* **S4, the challenger's coins**: window `i`'s prover's tape and its sampled verifier's coins are uniform wherever they
  are an equivalence's image of the network audit's own coin (`windowCoinsLive_of_equiv`). At an own coin of `n`
  blocks of `m` challenger's coins, one block a window (`OwnCoins`), window `i`'s coins are the first `D.V.k` of its
  block and its prover's tape the rest of the own coin (`OwnTape`, `splitOwn`, `windowCoinsLive_split`).
* **S6's `stmt`, fixed before the own coin**: window `i`'s statement is the one its decoding most often gives over the
  network audit's own coin among accepted plays (`coinMode`) where that is in the sampled audit's scope, else a fixed
  in-scope one (`stmtOf`). It reads the coins only off the other coordinates (`stmtOf_update`, `couples_stmt`) and is
  in scope at every coin (`stmtOf_scope`), so `R.scope` holds as stated. The commitment the game's `c` binds holds the
  network's only by digest, so an accepted play may decode to another statement; where every accepted play's is in
  scope, that event's probability is at most the square root of two independent own coins' accepted plays decoding to
  different statements (`stmtOf_collision`, from `coinMode_collision`): two preimages of one binding digest, which the
  coupling's break answers for.
* **S6's `wit`**: a witness in the leak relation wherever the statement has one (`witOf`, `couples_wit`).
* **The run's verdicts**: a coupling's guarantee (`Vocab.Coupling.bound`) reads its run only through the run at the
  code, which sets the verdicts itself, and `BreakBound`, which reads none: `Vocab.Coupling.atCode_withAccepts` and
  `Vocab.Coupling.breakBound_withAccepts`, at the coupling with its run's verdicts replaced
  (`Vocab.Coupling.withAccepts`). So the realization may be built at a run that accepts nothing, where `window_le` is
  the probability of `False` (`pr_rejects_le`), while the network certifier's target still fails for a verifier that
  accepts everything (`Vocab.networkSound_of_nontrivial`).
-/

namespace Verity.Network

open Accounting Accounting.JointBridge Claims Claims.Sampling NetTiming

/-! ## S4: the challenger's coins -/

section Coins

variable {Coin : Type}

/-- **The network audit's own coin** at `n` windows of `m` challenger's coins each. -/
abbrev OwnCoins (Coin : Type) (n m : ℕ) : Type := Fin n → Fin m → Coin

/-- **What window `i`'s prover's tape holds of the own coin**, the window's sampled verifier spending `k` of its block's
`m` coins: the rest of its block and the other windows' blocks. -/
abbrev OwnTape (Coin : Type) (n m : ℕ) (i : Fin n) (k : ℕ) : Type :=
  (Fin (m - k) → Coin) × ({j : Fin n // j ≠ i} → Fin m → Coin)

/-- A block of `m` coins as its first `k` and the rest. -/
def blockSplit {m k : ℕ} (hk : k ≤ m) : (Fin m → Coin) ≃ (Fin k → Coin) × (Fin (m - k) → Coin) :=
  (Equiv.arrowCongr ((finCongr (Nat.add_sub_of_le hk).symm).trans finSumFinEquiv.symm) (Equiv.refl Coin)).trans
    (Equiv.sumArrowEquivProdArrow _ _ _)

/-- **The own coin as window `i`'s prover's tape and its sampled verifier's `k` coins**, the first `k` of its block. -/
def splitOwn {n m k : ℕ} (i : Fin n) (hk : k ≤ m) : OwnCoins Coin n m ≃ OwnTape Coin n m i k × (Fin k → Coin) :=
  ((Equiv.piSplitAt i fun _ => Fin m → Coin).trans ((blockSplit hk).prodCongr (Equiv.refl _))).trans
    ((Equiv.prodAssoc _ _ _).trans (Equiv.prodComm _ _))

end Coins

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
  [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
  [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S}
  {k : Stmt → ℕ} {δ : ℝ} {decode : List (ℕ × Col) → List T} {reg : Stmt → Finset ℕ} {Dj n : ℕ}
  {r : PeriodRun d α C Ω} {K : ℝ} {R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n}

/-- **S4 at a coin map that is an equivalence**: where window `i`'s tape and coins are the network audit's own coin's
image under an equivalence `e ξ i`, they are uniform. -/
theorem windowCoinsLive_of_equiv (W : WindowRuns R)
    (e : ∀ ξ i, Ω (r.own .network) ≃ R.Tape ξ i × (Fin (D.V.k (R.stmt ξ i)) → Coin))
    (he : ∀ ξ i x, W.coins ξ i x = e ξ i x) : WindowCoinsLive W := fun ξ i q =>
  (congrArg Pouw.pr (funext fun x => congrArg q (he ξ i x))).trans (Pouw.pr_equiv (e ξ i) q)

/-- **S4 at an own coin of `n` blocks of `m` challenger's coins**: where the own coin is such a coin (`eo`), no
statement's sampled verifier spends more than `m` coins (`hk`), and window `i`'s prover's tape is the rest of the own
coin (`et`) and its coins the first of its block, the sessions' coins are the challenger's live coins. -/
theorem windowCoinsLive_split (W : WindowRuns R) {m : ℕ} (eo : Ω (r.own .network) ≃ OwnCoins Coin n m)
    (hk : ∀ x, D.V.k x ≤ m) (et : ∀ ξ i, R.Tape ξ i ≃ OwnTape Coin n m i (D.V.k (R.stmt ξ i)))
    (h : ∀ ξ i x, W.coins ξ i x = ((et ξ i).symm.prodCongr (Equiv.refl _)) (splitOwn i (hk (R.stmt ξ i)) (eo x))) :
    WindowCoinsLive W :=
  windowCoinsLive_of_equiv W
    (fun ξ i => (eo.trans (splitOwn i (hk (R.stmt ξ i)))).trans ((et ξ i).symm.prodCongr (Equiv.refl _))) h

/-! ## S6: the statements and witnesses -/

section Mode

variable {β γ : Type} [Fintype β] [Nonempty β]

/-- **The value `g` most often takes where `A` holds**, over a uniform coin. -/
noncomputable def coinMode (A : β → Prop) (g : β → γ) : γ :=
  g (Classical.choose (Finite.exists_max fun x => Pouw.pr fun y => A y ∧ g y = g x))

/-- No value is likelier than the mode where `A` holds. -/
theorem coinMode_max (A : β → Prop) (g : β → γ) (c : γ) :
    Pouw.pr (fun y => A y ∧ g y = c) ≤ Pouw.pr (fun y => A y ∧ g y = coinMode A g) := by
  by_cases hc : ∃ x, g x = c
  · obtain ⟨x, rfl⟩ := hc
    exact Classical.choose_spec (Finite.exists_max fun x => Pouw.pr fun y => A y ∧ g y = g x) x
  · rw [Pouw.pr_eq_zero fun y (hy : A y ∧ g y = c) => hc ⟨y, hy.2⟩]
    exact Pouw.pr_nonneg _

theorem pr_pos_of {P : β → Prop} {x : β} (hx : P x) : 0 < Pouw.pr P := by
  classical
  rw [Pouw.pr_eq_card P (Finset.univ.filter P) fun y => by simp]
  apply div_pos
  · exact_mod_cast Finset.card_pos.2 ⟨x, by simp [hx]⟩
  · exact_mod_cast Fintype.card_pos

/-- Where `A` holds somewhere, some coin where it holds takes the mode. -/
theorem coinMode_attained {A : β → Prop} (g : β → γ) {x : β} (hx : A x) : ∃ y, A y ∧ g y = coinMode A g := by
  by_contra hn
  have h0 : Pouw.pr (fun y => A y ∧ g y = coinMode A g) = 0 := Pouw.pr_eq_zero fun y hy => hn ⟨y, hy⟩
  have h1 := (pr_pos_of (P := fun y => A y ∧ g y = g x) ⟨hx, rfl⟩).trans_le (coinMode_max A g (g x))
  rw [h0] at h1
  exact lt_irrefl _ h1

open Finset in
/-- **Two independent coins' plays where `A` holds disagree on `g` at least as often as the square of one's
disagreeing with the mode**: the counting behind a finder that runs the play at two coins. Every coin disagrees with at
least as many coins in `A` as the mode does, the mode's being the largest of the counts. -/
theorem coinMode_collision (A : β → Prop) (g : β → γ) :
    Pouw.pr (fun y => A y ∧ g y ≠ coinMode A g) ^ 2 ≤
      Pouw.pr (fun z : β × β => (A z.1 ∧ A z.2) ∧ g z.1 ≠ g z.2) := by
  classical
  have hcard : ∀ (P : β → Prop) [DecidablePred P], Pouw.pr P = ((univ.filter P).card : ℝ) / Fintype.card β :=
    fun P _ => Pouw.pr_eq_card P _ fun x => by simp
  have hcard2 : ∀ (P : β × β → Prop) [DecidablePred P],
      Pouw.pr P = ((univ.filter P).card : ℝ) / Fintype.card (β × β) :=
    fun P _ => Pouw.pr_eq_card P _ fun x => by simp
  have hN : (0 : ℝ) < Fintype.card β := by exact_mod_cast Fintype.card_pos
  have hmax : ∀ c, (univ.filter fun y => A y ∧ g y = c).card ≤
      (univ.filter fun y => A y ∧ g y = coinMode A g).card := fun c => by
    have h := coinMode_max A g c
    rw [hcard, hcard] at h
    exact_mod_cast (div_le_div_iff_of_pos_right hN).1 h
  have hsplit : ∀ c, (univ.filter fun y => A y ∧ g y = c).card + (univ.filter fun y => A y ∧ g y ≠ c).card =
      (univ.filter A).card := fun c => by
    rw [card_filter, card_filter, card_filter, ← sum_add_distrib]
    refine sum_congr rfl fun y _ => ?_
    by_cases h1 : A y <;> by_cases h2 : g y = c <;> simp [h1, h2]
  have hfib : ∀ x, (univ.filter fun y => A y ∧ g y ≠ coinMode A g).card ≤
      (univ.filter fun y => A y ∧ g y ≠ g x).card := fun x => by
    have h1 := hsplit (coinMode A g)
    have h2 := hsplit (g x)
    have h3 := hmax (g x)
    omega
  have hpairs : (univ.filter fun z : β × β => (A z.1 ∧ A z.2) ∧ g z.1 ≠ g z.2).card =
      ∑ x ∈ univ.filter A, (univ.filter fun y => A y ∧ g y ≠ g x).card := by
    rw [card_filter, Fintype.sum_prod_type, sum_filter]
    refine sum_congr rfl fun x _ => ?_
    by_cases hx : A x
    · rw [if_pos hx, card_filter]
      exact sum_congr rfl fun y _ =>
        if_congr ⟨fun h => ⟨h.1.2, Ne.symm h.2⟩, fun h => ⟨⟨hx, h.1⟩, Ne.symm h.2⟩⟩ rfl rfl
    · rw [if_neg hx]
      exact sum_eq_zero fun y _ => if_neg fun h => hx h.1.1
  have hB : (univ.filter fun y => A y ∧ g y ≠ coinMode A g).card ≤ (univ.filter A).card := by
    have := hsplit (coinMode A g)
    omega
  have hcount : (univ.filter fun y => A y ∧ g y ≠ coinMode A g).card *
      (univ.filter fun y => A y ∧ g y ≠ coinMode A g).card ≤
      (univ.filter fun z : β × β => (A z.1 ∧ A z.2) ∧ g z.1 ≠ g z.2).card := by
    rw [hpairs]
    calc _ ≤ (univ.filter A).card * (univ.filter fun y => A y ∧ g y ≠ coinMode A g).card := Nat.mul_le_mul hB le_rfl
      _ = ∑ x ∈ univ.filter A, (univ.filter fun y => A y ∧ g y ≠ coinMode A g).card := by
          rw [sum_const, smul_eq_mul]
      _ ≤ _ := sum_le_sum fun x _ => hfib x
  rw [hcard, hcard2, Fintype.card_prod, div_pow, sq, sq, Nat.cast_mul]
  exact div_le_div_of_nonneg_right (by exact_mod_cast hcount) (by positivity)

end Mode

open Classical in
/-- **A statement fixed before the coordinate `o`**: the one `dec` most often gives over `o` where `acc` holds, at the
other coordinates of `ξ`, where that is in the sampled audit's scope at `D`, `lay` and `k`, else `s₀`. -/
noncomputable def stmtOf (D : Deploy Coin Col T) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (s₀ : Msg) (o : I)
    [Nonempty (Ω o)] (acc : ((i : I) → Ω i) → Prop) (dec : ((i : I) → Ω i) → Msg) (ξ : (i : I) → Ω i) : Msg :=
  if ScopeAcc D lay k (coinMode (fun x => acc (Function.update ξ o x)) fun x => dec (Function.update ξ o x)) then
    (coinMode (fun x => acc (Function.update ξ o x)) fun x => dec (Function.update ξ o x))
  else s₀

/-- `stmtOf` doesn't read the coordinate `o`. -/
theorem stmtOf_update (s₀ : Msg) (o : I) [Nonempty (Ω o)] (acc : ((i : I) → Ω i) → Prop)
    (dec : ((i : I) → Ω i) → Msg) (ξ : (i : I) → Ω i) (x : Ω o) :
    stmtOf D lay k s₀ o acc dec (Function.update ξ o x) = stmtOf D lay k s₀ o acc dec ξ := by
  simp only [stmtOf, Function.update_idem]

/-- `stmtOf` is in the sampled audit's scope where `s₀` is. -/
theorem stmtOf_scope {s₀ : Msg} (h₀ : ScopeAcc D lay k s₀) (o : I) [Nonempty (Ω o)] (acc : ((i : I) → Ω i) → Prop)
    (dec : ((i : I) → Ω i) → Msg) (ξ : (i : I) → Ω i) : ScopeAcc D lay k (stmtOf D lay k s₀ o acc dec ξ) := by
  unfold stmtOf
  split_ifs with h
  exacts [h, h₀]

/-- Where every play `acc` accepts decodes to a statement in scope, `stmtOf` is the mode wherever some value of `o`
is accepted. -/
theorem stmtOf_eq_coinMode {acc : ((i : I) → Ω i) → Prop} {dec : ((i : I) → Ω i) → Msg}
    (h : ∀ ξ, acc ξ → ScopeAcc D lay k (dec ξ)) (s₀ : Msg) (o : I) [Nonempty (Ω o)] (ξ : (i : I) → Ω i) (x : Ω o)
    (hx : acc (Function.update ξ o x)) :
    stmtOf D lay k s₀ o acc dec ξ =
      coinMode (fun x => acc (Function.update ξ o x)) fun x => dec (Function.update ξ o x) := by
  obtain ⟨y, hy, he⟩ := coinMode_attained (A := fun x => acc (Function.update ξ o x))
    (fun x => dec (Function.update ξ o x)) hx
  have hs : ScopeAcc D lay k
      (coinMode (fun x => acc (Function.update ξ o x)) fun x => dec (Function.update ξ o x)) := by
    rw [← he]
    exact h _ hy
  unfold stmtOf
  rw [if_pos hs]

/-- **The break `stmtOf` leaves, at the square root of a collision**: where every accepted play decodes to a
statement in scope, the probability over `o` that a play is accepted while its decoded statement isn't `stmtOf`'s,
squared, is at most the probability over two independent values of `o` that both plays are accepted with different
decoded statements. Two statements under one binding digest are a collision, which a finder that runs the play at both
outputs. -/
theorem stmtOf_collision {acc : ((i : I) → Ω i) → Prop} {dec : ((i : I) → Ω i) → Msg}
    (h : ∀ ξ, acc ξ → ScopeAcc D lay k (dec ξ)) (s₀ : Msg) (o : I) [Nonempty (Ω o)] (ξ : (i : I) → Ω i) :
    Pouw.pr (fun x : Ω o => acc (Function.update ξ o x) ∧ dec (Function.update ξ o x) ≠ stmtOf D lay k s₀ o acc dec ξ)
        ^ 2 ≤
      Pouw.pr (fun z : Ω o × Ω o => (acc (Function.update ξ o z.1) ∧ acc (Function.update ξ o z.2)) ∧
        dec (Function.update ξ o z.1) ≠ dec (Function.update ξ o z.2)) := by
  by_cases ha : ∃ x, acc (Function.update ξ o x)
  · obtain ⟨x, hx⟩ := ha
    rw [stmtOf_eq_coinMode h s₀ o ξ x hx]
    exact coinMode_collision (fun x => acc (Function.update ξ o x)) fun x => dec (Function.update ξ o x)
  · rw [Pouw.pr_eq_zero (P := fun x : Ω o => acc (Function.update ξ o x) ∧
      dec (Function.update ξ o x) ≠ stmtOf D lay k s₀ o acc dec ξ) fun x hx => ha ⟨x, hx.1⟩]
    calc (0 : ℝ) ^ 2 = 0 := by norm_num
      _ ≤ _ := Pouw.pr_nonneg _

/-- **S6's `stmt` by construction**: a realization whose window statements are `stmtOf`'s at the network audit's own
coin reads the coins only off the other coordinates. -/
theorem couples_stmt [Nonempty (Ω (r.own .network))] (s₀ : Msg) (acc : ((i : I) → Ω i) → Prop)
    (dec : Fin n → ((i : I) → Ω i) → Msg) (h : ∀ ξ i, R.stmt ξ i = stmtOf D lay k s₀ (r.own .network) acc (dec i) ξ) :
    ∀ ξ i x, R.stmt (Function.update ξ (r.own .network) x) i = R.stmt ξ i := fun ξ i x =>
  (h _ i).trans ((stmtOf_update s₀ _ acc (dec i) ξ x).trans (h ξ i).symm)

open Classical in
/-- **A witness in `Rl` at `x`** wherever there is one. -/
noncomputable def witOf (Rl : Rel Wit) (x : Msg) : Wit :=
  if h : ∃ w, Rl x w then Classical.choose h else ⟨[]⟩

theorem witOf_spec (Rl : Rel Wit) {x : Msg} (h : ∃ w, Rl x w) : Rl x (witOf Rl x) := by
  unfold witOf
  rw [dif_pos h]
  exact Classical.choose_spec h

/-- **S6's `wit` by construction**: a realization whose witnesses are `witOf`'s at the leak relation. -/
theorem couples_wit (h : ∀ ξ i, R.wit ξ i = witOf (Rlt R) (R.stmt ξ i)) :
    ∀ ξ i, (∃ w, Rlt R (R.stmt ξ i) w) → Rlt R (R.stmt ξ i) (R.wit ξ i) := fun ξ i hw => by
  rw [h ξ i]
  exact witOf_spec _ hw

/-! ## The run's verdicts -/

/-- **`window_le` at a run that accepts nothing**: an event under a verdict that never holds is below any `PrBad`. -/
theorem pr_rejects_le {β W Ω' : Type} [Fintype β] [Fintype Ω'] {A B : β → Prop} (hA : ∀ x, ¬ A x)
    (Vf : Verifier Coin) (Rl : Rel W) (E : Extractor Coin W Vf) (x : Msg) (P : Prover Ω') :
    Pouw.pr (fun y => A y ∧ B y) ≤ PrBad H Vf Rl E x P := by
  rw [Pouw.pr_eq_zero fun y hy => hA y hy.1]
  unfold PrBad Claims.Pr
  positivity

end Verity.Network

namespace Verity.Vocab.Coupling

open Accounting

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {V : Vocab d α} {k : Verity.Registration.Composite}
  {a : Audit} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)} (q : V.Coupling k a θ F c σ)

/-- **The coupling with its run's verdicts replaced by `M`**: the run's claims, and so `world`, are unchanged
(`Subprotocol.holds_atCode`). -/
def withAccepts (M : Audit → ((i : q.I) → q.Ω i) → Prop) : V.Coupling k a θ F c σ :=
  { q with
    run := Subprotocol.atCode q.run M
    world := fun ξ h => q.world ξ ((Subprotocol.holds_atCode q.run M _ _ a ξ).1 h) }

/-- **The coupled run at the code doesn't read the run's verdicts.** -/
theorem atCode_withAccepts (M : Audit → ((i : q.I) → q.Ω i) → Prop) : (q.withAccepts M).atCode = q.atCode := rfl

/-- **The error beyond `δ` doesn't read the run's verdicts.** -/
theorem breakBound_withAccepts (M : Audit → ((i : q.I) → q.Ω i) → Prop) :
    (q.withAccepts M).BreakBound ↔ q.BreakBound := Iff.rfl

end Verity.Vocab.Coupling
