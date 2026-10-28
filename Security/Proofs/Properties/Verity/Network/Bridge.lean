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
  network audit's own coin (`coinMode`) where that is in the sampled audit's scope, else a fixed in-scope one
  (`stmtOf`). It reads the coins only off the other coordinates (`stmtOf_update`, `couples_stmt`) and is in scope at
  every coin (`stmtOf_scope`), so `R.scope` holds as stated.
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

/-- **The value `g` most often takes** over a uniform coin. -/
noncomputable def coinMode (g : β → γ) : γ :=
  g (Classical.choose (Finite.exists_max fun x => Pouw.pr fun y => g y = g x))

/-- No value is likelier than the mode. -/
theorem coinMode_max (g : β → γ) (c : γ) : Pouw.pr (fun y => g y = c) ≤ Pouw.pr (fun y => g y = coinMode g) := by
  by_cases hc : ∃ x, g x = c
  · obtain ⟨x, rfl⟩ := hc
    exact Classical.choose_spec (Finite.exists_max fun x => Pouw.pr fun y => g y = g x) x
  · rw [Pouw.pr_eq_zero fun y hy => hc ⟨y, hy⟩]
    exact Pouw.pr_nonneg _

end Mode

open Classical in
/-- **A statement fixed before the coordinate `o`**: the one `dec` most often gives over `o` at the other coordinates
of `ξ`, where that is in the sampled audit's scope at `D`, `lay` and `k`, else `s₀`. -/
noncomputable def stmtOf (D : Deploy Coin Col T) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (s₀ : Msg) (o : I)
    [Nonempty (Ω o)] (dec : ((i : I) → Ω i) → Msg) (ξ : (i : I) → Ω i) : Msg :=
  if ScopeAcc D lay k (coinMode fun x => dec (Function.update ξ o x)) then
    coinMode (fun x => dec (Function.update ξ o x))
  else s₀

/-- `stmtOf` doesn't read the coordinate `o`. -/
theorem stmtOf_update (s₀ : Msg) (o : I) [Nonempty (Ω o)] (dec : ((i : I) → Ω i) → Msg) (ξ : (i : I) → Ω i)
    (x : Ω o) : stmtOf D lay k s₀ o dec (Function.update ξ o x) = stmtOf D lay k s₀ o dec ξ := by
  simp only [stmtOf, Function.update_idem]

/-- `stmtOf` is in the sampled audit's scope where `s₀` is. -/
theorem stmtOf_scope {s₀ : Msg} (h₀ : ScopeAcc D lay k s₀) (o : I) [Nonempty (Ω o)] (dec : ((i : I) → Ω i) → Msg)
    (ξ : (i : I) → Ω i) : ScopeAcc D lay k (stmtOf D lay k s₀ o dec ξ) := by
  unfold stmtOf
  split_ifs with h
  exacts [h, h₀]

/-- **S6's `stmt` by construction**: a realization whose window statements are `stmtOf`'s at the network audit's own
coin reads the coins only off the other coordinates. -/
theorem couples_stmt [Nonempty (Ω (r.own .network))] (s₀ : Msg) (dec : Fin n → ((i : I) → Ω i) → Msg)
    (h : ∀ ξ i, R.stmt ξ i = stmtOf D lay k s₀ (r.own .network) (dec i) ξ) :
    ∀ ξ i x, R.stmt (Function.update ξ (r.own .network) x) i = R.stmt ξ i := fun ξ i x =>
  (h _ i).trans ((stmtOf_update s₀ _ (dec i) ξ x).trans (h ξ i).symm)

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
