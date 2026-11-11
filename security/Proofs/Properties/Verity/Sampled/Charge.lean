module

public import Proofs.Properties.Verity.Sampled.LeafErr
public import Proofs.Properties.Verity.Sampled.Stops
public import Proofs.Properties.Verity.Sampled.Walk

/-!
# Sampled proofs' coupling: the stops' term within the hiding term

The coupling's stops' term (`couplesErr`) charges each stop's opening its gap and empty class: the link's at the
first stop, at each later one the round's whose submission the stop decodes. Each is within the class error of the
node the stop is at (`openErr`), so this file bounds the stops' term by the class errors along a play, and those by
the potential from no tree open, the hiding term:

* **A charge along a play** (`Game.pathCharge`): a reading of the nodes, summed over the nodes a tape's play passes.
  Over a uniform tape it splits at each node (`avg_pathCharge_send`, `avg_pathCharge_coin`), and the walk to a stop
  and past a consumed stop only leaves nodes out (`Pos.pathCharge_walk_le`, `Pos.add_stopCharge_consume_le`).
* **The rounds' error within the charge** (`avg_roundsErr_le`): where `γ` at each stop the projection reaches is
  within the charge at the stop's node, the rounds' error from a position is within the charge from there
  (`Pos.stopCharge`), over the tapes from it.
* **The class errors along a play within the potential** (`avg_pathCharge_relK_le`, `avg_pathCharge_game_le`): over
  a uniform tape, the class errors of the nodes a play of the period's game passes sum to at most the potential from
  no tree open. A fresh frame's coin and the close are no opening's node, a `Hello` keeps the potential on average,
  and an opening pays its round's class error out of it: the device opens each round at most once.
* **The stops' term** (`couplesErr_le_of_charge`, `couplesErr_sampleMap_le`): where each stop's opening asks for the
  count of coins the stop releases (`RoundCounts`, `LinkCounts`), the stops' term at the sample-coupled coin map is
  within the hiding term. A frame of another count leaves every class empty, so the empty class as the coupling
  charges it (`openEmpty`) is then 1.
* **The stops' term at the stops' counts** (`openEmptyAt`, `couplesErr_sampleMap_le_at`): with the empty class held
  only where the frame asks for the stop's count, each stop's gap and empty class are within its node's class error
  at every count, and the stops' term is within the hiding term at every strategy.
-/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

variable {α V ι : Type}

/-! ## A charge along a play -/

/-- **The charge along a play**: `Γ` at each node the play of `s` on the tape `x` passes, from `g` to its end. -/
noncomputable def pathCharge (Γ : Game α → ℝ≥0∞) : (g : Game α) → (s : Strategy g) → CoinTape g s → ℝ≥0∞
  | .ret _, _, _ => 0
  | .send M k, s, x => Γ (.send M k) + pathCharge Γ (k s.1) s.2 x
  | @Game.coin _ C iF iN k, s, x => Γ (@Game.coin _ C iF iN k) + pathCharge Γ (k x.1) (s x.1) (x.2 x.1)

theorem avg_pathCharge_ret (Γ : Game α → ℝ≥0∞) (a : α) (s : Strategy (Game.ret a)) :
    avg (pathCharge Γ (.ret a) s) = 0 :=
  avg_const (C := CoinTape (Game.ret a) s) 0

theorem avg_pathCharge_send (Γ : Game α → ℝ≥0∞) {M : Type} {k : M → Game α} (s : Strategy (.send M k)) :
    avg (pathCharge Γ (.send M k) s) = Γ (.send M k) + avg (pathCharge Γ (k s.1) s.2) := by
  change avg (fun x : CoinTape (k s.1) s.2 => Γ (.send M k) + pathCharge Γ (k s.1) s.2 x) = _
  rw [avg_add, avg_const]

open Classical in
/-- **The charge at a coin, over a uniform tape**: the coin's node, and the charge from each continuation averaged
over the coin. -/
theorem avg_pathCharge_coin (Γ : Game α → ℝ≥0∞) {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (s : Strategy (@Game.coin _ C iF iN k)) :
    avg (pathCharge Γ (@Game.coin _ C iF iN k) s) =
      Γ (@Game.coin _ C iF iN k) + avg fun c => avg (pathCharge Γ (k c) (s c)) := by
  let : ∀ c, Fintype (CoinTape (k c) (s c)) := fun c => CoinTape.fintype (k c) (s c)
  have : ∀ c, Nonempty (CoinTape (k c) (s c)) := fun c => CoinTape.nonempty (k c) (s c)
  change avg (fun x : C × ((c : C) → CoinTape (k c) (s c)) =>
    Γ (@Game.coin _ C iF iN k) + pathCharge Γ (k x.1) (s x.1) (x.2 x.1)) = _
  rw [avg_add, avg_const, avg_prod]
  exact congrArg (Γ (@Game.coin _ C iF iN k) + ·) (congrArg avg (funext fun c =>
    avg_eval (T := fun c => CoinTape (k c) (s c)) c (pathCharge Γ (k c) (s c))))

theorem avg_pathCharge_le_of_eq (Γ : Game α → ℝ≥0∞) {g g' : Game α} (h : g = g') {b : ℝ≥0∞}
    (H : ∀ s, avg (pathCharge Γ g' s) ≤ b) : ∀ s, avg (pathCharge Γ g s) ≤ b := by
  subst h
  exact H

/-- **A message that costs nothing**: the bound from every message the prover may send. -/
theorem avg_pathCharge_send_le (Γ : Game α → ℝ≥0∞) {M : Type} {k : M → Game α} {b : ℝ≥0∞}
    (h0 : Γ (.send M k) = 0) (hk : ∀ m (s : Strategy (k m)), avg (pathCharge Γ (k m) s) ≤ b) :
    ∀ s, avg (pathCharge Γ (.send M k) s) ≤ b := fun s => by
  rw [avg_pathCharge_send, h0, zero_add]
  exact hk s.1 s.2

/-- **A coin**: its node's cost, and the bounds from its continuations averaged over the coin. -/
theorem avg_pathCharge_coin_le (Γ : Game α → ℝ≥0∞) {C : Type} {iF : Fintype C} {iN : Nonempty C}
    {k : C → Game α} {a B : ℝ≥0∞} (b : C → ℝ≥0∞) (h0 : Γ (@Game.coin _ C iF iN k) ≤ a)
    (hk : ∀ c (s : Strategy (k c)), avg (pathCharge Γ (k c) s) ≤ b c) (hB : a + avg b ≤ B) :
    ∀ s, avg (pathCharge Γ (@Game.coin _ C iF iN k) s) ≤ B := fun s => by
  rw [avg_pathCharge_coin]
  exact (add_le_add h0 (avg_mono fun c => hk c (s c))).trans hB

theorem avg_pathCharge_optGame_le {β : Type} (Γ : Game α → ℝ≥0∞) {o : Option β} {n : Game α} {y : β → Game α}
    {b : ℝ≥0∞} (hn : o = none → ∀ s, avg (pathCharge Γ n s) ≤ b)
    (hy : ∀ x, o = some x → ∀ s, avg (pathCharge Γ (y x) s) ≤ b) :
    ∀ s, avg (pathCharge Γ (optGame o n y) s) ≤ b := by
  cases o with
  | none => exact hn rfl
  | some x => exact hy x rfl

theorem avg_pathCharge_ite_le (Γ : Game α → ℝ≥0∞) {c : Prop} {dc : Decidable c} {t u : Game α} {b : ℝ≥0∞}
    (ht : c → ∀ s, avg (pathCharge Γ t s) ≤ b) (hu : ¬ c → ∀ s, avg (pathCharge Γ u s) ≤ b) :
    ∀ s, avg (pathCharge Γ (@ite _ c dc t u) s) ≤ b := by
  by_cases hc : c
  · exact avg_pathCharge_le_of_eq Γ (ite_eq_left hc) (ht hc)
  · exact avg_pathCharge_le_of_eq Γ (ite_eq_right hc) (hu hc)

/-- **A written coin falls short of uniform by at most 1.** -/
theorem shortfall_le_one {P C : Type} [Fintype P] [Fintype C] [Nonempty C] (w : P → C → C) : shortfall w ≤ 1 := by
  unfold shortfall
  refine (Finset.sum_le_sum (g := fun _ => (Fintype.card C : ℝ≥0∞)⁻¹) fun x _ => tsub_le_self).trans_eq ?_
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
    ENNReal.mul_inv_cancel (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)]

namespace Pos

variable {g : Game α} {σ : Strategy g}

/-- **The walk only leaves nodes out**: the charge from where the walk stops is within the charge from where it
starts. -/
theorem pathCharge_walk_le (Γ : Game α → ℝ≥0∞) : ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g')
    (x : CoinTape g' s') (back : CoinTape g' s' → CoinTape g σ),
    pathCharge Γ (walk g' s' Ks x back).g' (walk g' s' Ks x back).s' (walk g' s' Ks x back).x ≤
      pathCharge Γ g' s' x
  | .ret _, _, _, _, _ => le_rfl
  | .send M k, s', Ks, x, back => by
    by_cases hf : ∃ i, (Ks i).isHere
    · exact (congrArg (fun w : Pos g σ V ι => pathCharge Γ w.g' w.s' w.x) (walk_here _ s' Ks x back hf)).le
    · rw [walk_send s' Ks x back fun i => Bool.eq_false_iff.2 fun h' => hf ⟨i, h'⟩]
      exact (pathCharge_walk_le Γ (k s'.1) s'.2 _ x back).trans le_add_self
  | @Game.coin _ C iF iN k, s', Ks, x, back => by
    by_cases hf : ∃ i, (Ks i).isHere
    · exact (congrArg (fun w : Pos g σ V ι => pathCharge Γ w.g' w.s' w.x) (walk_here _ s' Ks x back hf)).le
    · rw [walk_coin s' Ks x back fun i => Bool.eq_false_iff.2 fun h' => hf ⟨i, h'⟩]
      exact (pathCharge_walk_le Γ (k x.1) (s' x.1) _ (x.2 x.1) _).trans le_add_self

/-- **The walk from a node no cut is fixed at leaves that node out**, where the play's end costs nothing. -/
theorem add_pathCharge_walk_le (Γ : Game α → ℝ≥0∞) (hret : ∀ a, Γ (.ret a) = 0) :
    ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g') (x : CoinTape g' s')
      (back : CoinTape g' s' → CoinTape g σ), (∀ i, (Ks i).isHere = false) →
      Γ g' + pathCharge Γ (walk g' s' Ks x back).g' (walk g' s' Ks x back).s' (walk g' s' Ks x back).x ≤
        pathCharge Γ g' s' x
  | .ret a, _, _, _, _, _ => (show Γ (.ret a) + 0 = 0 by rw [hret, add_zero]).le
  | .send M k, s', Ks, x, back, h => by
    rw [walk_send s' Ks x back h]
    exact add_le_add le_rfl (pathCharge_walk_le Γ (k s'.1) s'.2 _ x back)
  | @Game.coin _ C iF iN k, s', Ks, x, back, h => by
    rw [walk_coin s' Ks x back h]
    exact add_le_add le_rfl (pathCharge_walk_le Γ (k x.1) (s' x.1) _ (x.2 x.1) _)

/-- **The charge from the next stop**: along the play from where the walk from the position stops. -/
noncomputable def stopCharge (Γ : Game α → ℝ≥0∞) (r : Pos g σ V ι) : ℝ≥0∞ :=
  pathCharge Γ r.toStop.g' r.toStop.s' r.toStop.x

/-- **Past a consumed stop the walk leaves the stop's node out.** -/
theorem add_stopCharge_consume_le (Γ : Game α → ℝ≥0∞) (hret : ∀ a, Γ (.ret a) = 0) (q : Pos g σ V ι) :
    Γ q.g' + stopCharge Γ q.consume ≤ pathCharge Γ q.g' q.s' q.x :=
  add_pathCharge_walk_le Γ hret q.g' q.s' q.consume.Ks q.x q.back (isHere_consume q)

theorem resample_add_consume (Γ : Game α → ℝ≥0∞) (H : Pos g σ V ι → ℝ≥0∞) (q : Pos g σ V ι) :
    resample (fun r => Γ r.g' + H r.consume) q = Γ q.g' + resample H q.consume := by
  show avg (fun z : CoinTape q.g' q.s' => Γ q.g' + H (q.consume.withTape z)) =
    Γ q.g' + avg fun z : CoinTape q.g' q.s' => H (q.consume.withTape z)
  rw [avg_add, avg_const]

theorem resample_le_of_le {H : Pos g σ V ι → ℝ≥0∞} {q : Pos g σ V ι} {x : ℝ≥0∞}
    (h : ∀ z : CoinTape q.g' q.s', H (q.withTape z) ≤ x) : resample H q ≤ x :=
  (avg_mono h).trans_eq (avg_const x)

/-- **The positions the projection reaches from the play's start at the cuts `Ks₀`**: its first stop on a tape, and
from a position, the next stop, the position with its stop consumed, or with another tape from it. -/
inductive Reached (Ks₀ : ι → DigestCut V g) : Pos g σ V ι → Prop
  | root (x : CoinTape g σ) : Reached Ks₀ (Pos.root Ks₀ x)
  | toStop {p : Pos g σ V ι} : Reached Ks₀ p → Reached Ks₀ p.toStop
  | consume {p : Pos g σ V ι} : Reached Ks₀ p → Reached Ks₀ p.consume
  | withTape {p : Pos g σ V ι} (z : CoinTape p.g' p.s') : Reached Ks₀ p → Reached Ks₀ (p.withTape z)

end Pos

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The rounds' error within the charge -/

/-- **The rounds' error within the charge**: where `γ` at each stop the projection reaches from the cuts `Ks₀`, at
each submission decoded there, is within `Γ` at the stop's node, the rounds' error from a position it reaches is
within the charge from the next stop, over the tapes from the position; the play's end costs nothing. -/
theorem avg_roundsErr_le {g : Game Exchange} {σ : Strategy g} {R : ℕ} {Reg : Type} (D : Decoders R Reg) (J : ℕ)
    (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞) (Γ : Game Exchange → ℝ≥0∞)
    (hret : ∀ a, Γ (.ret a) = 0) (Ks₀ : Fin R → DigestCut Claims.Msg g)
    (hγ : ∀ q, Pos.Reached Ks₀ q → ∀ s, D.sub J q.modes = some s → γ q J s ≤ Γ q.g') (N : ℕ) :
    ∀ p, Pos.Reached Ks₀ p → Pos.resample (roundsErr D J γ N) p ≤ Pos.resample (Pos.stopCharge Γ) p := by
  induction N with
  | zero => exact fun p _ => avg_mono fun _ => zero_le
  | succ N ih =>
    intro p hp
    have hstep : ∀ z : CoinTape p.g' p.s', roundsErr D J γ (N + 1) (p.withTape z) ≤
        Γ (p.withTape z).toStop.g' + Pos.resample (Pos.stopCharge Γ) (p.withTape z).toStop.consume := fun z => by
      show stopErr J γ (roundsErr D J γ N) (p.withTape z).toStop (D.sub J (p.withTape z).toStop.modes) ≤ _
      cases h : D.sub J (p.withTape z).toStop.modes with
      | none => exact zero_le
      | some s =>
        exact add_le_add (hγ _ (.toStop (.withTape z hp)) s h) (ih _ (.consume (.toStop (.withTape z hp))))
    calc Pos.resample (roundsErr D J γ (N + 1)) p
        ≤ avg (fun z : CoinTape p.g' p.s' => Γ (p.withTape z).toStop.g' +
            Pos.resample (Pos.stopCharge Γ) (p.withTape z).toStop.consume) := avg_mono hstep
      _ = avg (fun z : CoinTape p.g' p.s' =>
            Pos.resample (fun r => Γ r.g' + Pos.stopCharge Γ r.consume) (p.withTape z).toStop) :=
          congrArg avg (funext fun z => (Pos.resample_add_consume Γ (Pos.stopCharge Γ) _).symm)
      _ = avg (fun z : CoinTape p.g' p.s' => Γ (p.withTape z).toStop.g' +
            Pos.stopCharge Γ (p.withTape z).toStop.consume) :=
          (Pos.avg_toStop (fun r => Γ r.g' + Pos.stopCharge Γ r.consume) p).symm
      _ ≤ Pos.resample (Pos.stopCharge Γ) p := avg_mono fun z => Pos.add_stopCharge_consume_le Γ hret _

/-! ## The class errors along a play within the potential -/

variable {β : Type} {E : List Release → Game β}

/-- **A node the prover moves at, and not only to the play's end.** -/
def GoesOn (g : Game β) : Prop := ∃ (M : Type) (k : M → Game β) (m : M), g = .send M k ∧ ∀ a, k m ≠ .ret a

theorem goesOn_relK {f : List Release → Game β} (hf : ∀ log, GoesOn (f log)) :
    ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree), GoesOn (relK k fs ts f)
  | 0, _, _ => hf []
  | _ + 1, _, _ => by
    obtain ⟨M, k, m, hk, -⟩ := hf []
    exact ⟨_, _, none, rfl, fun a h => by cases hk.symm.trans h⟩

theorem openingOf_send {M : Type} {k : M → Game β} : openingOf E (.send M k) = none := by
  unfold openingOf
  refine dite_eq_right ?_
  rintro ⟨o⟩
  cases o.eq

theorem openingOf_ret {b : β} : openingOf E (.ret b) = none := by
  unfold openingOf
  refine dite_eq_right ?_
  rintro ⟨o⟩
  cases o.eq

/-- **A coin whose continuations are the device's releases after a frame other than an opening is no opening's
node**, at a close that records its log (`relK_log_eq`). -/
theorem openingOf_coin_none {lg : β → List Release} (hE : RecordsLog lg E) {C : Type} {iF : Fintype C}
    {iN : Nonempty C} {k' : C → Game β}
    (h : ∀ x, ∃ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Release) (r : Release),
      k' x = relK k fs ts (fun log => E (pre ++ r :: log)) ∧ ∀ i s n, r.frame ≠ .opening i s n) :
    openingOf E (@Game.coin β C iF iN k') = none := by
  unfold openingOf
  refine dite_eq_right ?_
  rintro ⟨o⟩
  have h₀ := o.eq
  injection h₀ with hC _ hk
  obtain ⟨d⟩ := o.iN
  obtain ⟨k, fs, ts, pre, r, h₁, h₂⟩ := h (cast hC.symm d)
  have e₁ := (apply_cast_of_heq hC hk (cast hC.symm d)).symm.trans h₁
  exact h₂ o.i o.s o.n (congrArg Release.frame (relK_log_eq hE e₁).2).symm

/-- **A coin whose continuations end the play at the prover's next message is no opening's node**, where every play
of the close goes on past the prover's next message. -/
theorem openingOf_coin_send (hE' : ∀ log, GoesOn (E log)) {C : Type} {iF : Fintype C} {iN : Nonempty C}
    {k' : C → Game β}
    (hk' : ∀ x, ∃ (M : Type) (k : M → Game β), k' x = .send M k ∧ ∀ m, ∃ a, k m = .ret a) :
    openingOf E (@Game.coin β C iF iN k') = none := by
  unfold openingOf
  refine dite_eq_right ?_
  rintro ⟨o⟩
  have h₀ := o.eq
  injection h₀ with hC _ hk
  obtain ⟨d⟩ := o.iN
  obtain ⟨M, k₂, h₃, h₄⟩ := hk' (cast hC.symm d)
  obtain ⟨M', k₃, m, h₁, h₂⟩ := goesOn_relK (fun log => hE' (o.pre ++
    ⟨.opening o.i o.s o.n, o.b, TreeDraw.opened (cast hC (cast hC.symm d)).1 o.n⟩ :: log)) o.k o.fs
    (o.ts.set o.i (o.T.bump o.p.1))
  have e₂ := h₁.symm.trans ((apply_cast_of_heq hC hk (cast hC.symm d)).symm.trans h₃)
  injection e₂ with hM hk₂
  subst hM
  obtain ⟨a, ha⟩ := h₄ m
  exact h₂ a ((congrFun (eq_of_heq hk₂) m).trans ha)

theorem cons_close {f : List Release → Game β} {pre : List Release} (hf : ∀ log, f log = E (pre ++ log))
    (r : Release) : ∀ log, f (r :: log) = E ((pre ++ [r]) ++ log) := fun log => by
  rw [hf, List.append_assoc, List.singleton_append]

/-- **The close after the device's log goes on past the prover's next message.** -/
theorem goesOn_finish (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c a : ByteArray) (log : List Release) :
    GoesOn (finish D θ F c a log) :=
  ⟨ByteArray, _, ByteArray.empty, rfl, fun _ h => by cases h⟩

variable (e : 𝔽 ≃ (Fin 128 → Bool))

theorem openErr_fresh {lg : β → List Release} (hE : RecordsLog lg E) (k : ℕ) (fs : List (String × ℕ))
    (ts : List OpenTree) (f : List Release → Game β) (pre : List Release) (hf : ∀ log, f log = E (pre ++ log))
    (l : String) (b : ByteArray) {iF : Fintype (Fin 512 → Bool)} {iN : Nonempty (Fin 512 → Bool)} :
    openErr e E (@Game.coin β (Fin 512 → Bool) iF iN fun x =>
      relK k fs ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)) = 0 :=
  openErr_none e (openingOf_coin_none hE fun x => ⟨k, fs, ts, pre, ⟨.fresh l, b, List.ofFn x⟩,
    congrArg (relK k fs ts) (funext fun log => hf _), fun _ _ _ h => by cases h⟩)

theorem openErr_hello {lg : β → List Release} (hE : RecordsLog lg E) (k : ℕ) (fs : List (String × ℕ))
    (ts : List OpenTree) (f : List Release → Game β) (pre : List Release) (hf : ∀ log, f log = E (pre ++ log))
    (spec : Flock.CoinTree.Spec) (ν : ByteArray) {iF : Fintype (TreeSample spec)}
    {iN : Nonempty (TreeSample spec)} :
    openErr e E (@Game.coin β (TreeSample spec) iF iN fun t =>
      relK k fs (ts ++ [(⟨ν, spec, t, fun _ => 0⟩ : OpenTree)]) fun log =>
        f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩ :: log)) = 0 :=
  openErr_none e (openingOf_coin_none hE fun t => ⟨k, fs, ts ++ [(⟨ν, spec, t, fun _ => 0⟩ : OpenTree)], pre,
    ⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩,
    congrArg (relK k fs (ts ++ [(⟨ν, spec, t, fun _ => 0⟩ : OpenTree)])) (funext fun log => hf _),
    fun _ _ _ h => by cases h⟩)

/-- **An opening's node costs its round's class error**, at a count its block holds (`openErr_le_roundErr`). -/
theorem openErr_opening_le {lg : β → List Release} (hE : RecordsLog lg E) (k : ℕ) (fs : List (String × ℕ))
    (ts : List OpenTree) (f : List Release → Game β) (pre : List Release) (hf : ∀ log, f log = E (pre ++ log))
    (i s n : ℕ) (b : ByteArray) (T : OpenTree) (p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog))
    (hn : n ≤ T.spec.block) {iF : Fintype (TreeOpenable T.nonce T.spec T.sample p.1 p.2)}
    {iN : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2)} :
    openErr e E (@Game.coin β (TreeOpenable T.nonce T.spec T.sample p.1 p.2) iF iN fun d =>
      relK k fs (ts.set i (T.bump p.1)) fun log => f (⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log)) ≤
      roundErr e T p.1 p.2 := by
  refine openErr_le_roundErr e hE ⟨T, p, i, s, n, b, k, fs, ts, pre, iF, iN, ?_⟩ hn
  exact congrArg (@Game.coin β _ iF iN) (funext fun d =>
    congrArg (relK k fs (ts.set i (T.bump p.1))) (funext fun log => hf _))

/-- **The class errors along the device's releases within the potential**: over a uniform tape, the class errors of
the nodes a play of the releases from the open trees `ts` passes, to a close that records the log and whose own plays
cost nothing, sum to at most the potential at `ts`. A fresh frame's coin costs nothing, a `Hello` keeps the potential
on average (`avg_treePotential_hello_le`), and an opening pays its round's class error out of it
(`roundErr_add_treePotential_le`). -/
theorem avg_pathCharge_relK_le {lg : β → List Release} (hE : RecordsLog lg E)
    (hz : ∀ log (s : Strategy (E log)), avg (pathCharge (openErr e E) (E log) s) ≤ 0) :
    ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β) (pre : List Release),
      (∀ log, f log = E (pre ++ log)) →
        ∀ s : Strategy (relK k fs ts f), avg (pathCharge (openErr e E) (relK k fs ts f) s) ≤ treePotential e ts := by
  intro k
  induction k with
  | zero =>
    intro fs ts f pre hf s
    exact (avg_pathCharge_le_of_eq _ (hf []) (hz _) s).trans zero_le
  | succ k ih =>
    intro fs ts f pre hf
    have hfz : ∀ s : Strategy (f []), avg (pathCharge (openErr e E) (f []) s) ≤ treePotential e ts :=
      fun s => (avg_pathCharge_le_of_eq _ (hf []) (hz _) s).trans zero_le
    refine avg_pathCharge_send_le _ (openErr_none e openingOf_send) ?_
    rintro (_ | ⟨l | spec | ⟨i, s, n⟩, b⟩)
    · exact hfz
    · refine avg_pathCharge_optGame_le _ (fun _ => hfz) fun fs' _ => ?_
      refine avg_pathCharge_coin_le _ (fun _ => treePotential e ts)
        (le_of_eq (openErr_fresh e hE k fs' ts f pre hf l b))
        (fun x => ih fs' ts _ _ (cons_close hf ⟨.fresh l, b, List.ofFn x⟩)) ?_
      simp only [zero_add, avg_const, le_refl]
    · refine avg_pathCharge_ite_le _ (fun hc => ?_) fun _ => hfz
      refine avg_pathCharge_coin_le _ (fun t => treePotential e (ts ++ [(⟨b, spec, t, fun _ => 0⟩ : OpenTree)]))
        (le_of_eq (openErr_hello e hE k fs ts f pre hf spec b))
        (fun t => ih fs (ts ++ [(⟨b, spec, t, fun _ => 0⟩ : OpenTree)]) _ _
          (cons_close hf ⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩)) ?_
      rw [zero_add]
      exact avg_treePotential_hello_le e ts b spec hc.2 hc.1
    · refine avg_pathCharge_ite_le _ (fun hL => ?_) fun _ => hfz
      refine avg_pathCharge_optGame_le _ (fun _ => hfz) fun T hT => ?_
      refine avg_pathCharge_optGame_le _ (fun _ => hfz) fun p hp => ?_
      obtain ⟨-, hp₂, hn⟩ := OpenTree.next_some hp
      have : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2) := ⟨⟨T.sample.2 p.1 p.2, rfl⟩⟩
      refine avg_pathCharge_coin_le _ (fun _ => treePotential e (ts.set i (T.bump p.1)))
        (openErr_opening_le e hE k fs ts f pre hf i s n b T p hn)
        (fun d => ih fs (ts.set i (T.bump p.1)) _ _
          (cons_close hf ⟨.opening i s n, b, TreeDraw.opened d.1 n⟩)) ?_
      have hlt : T.opened p.1 < 2 ^ T.spec.roundsLog := by
        rw [← hp₂]
        exact p.2.isLt
      have hpe : p.2 = ⟨T.opened p.1, hlt⟩ := Fin.ext hp₂
      rw [avg_const, hpe]
      exact roundErr_add_treePotential_le e hT hL p.1 hlt

/-- **The close costs nothing**: its coin is no opening's node, as every play of a close goes on past the prover's
next message. -/
theorem avg_pathCharge_finish_le {E : List Release → Game Exchange} (hE' : ∀ log, GoesOn (E log)) (D : Signer)
    (θ : ByteArray) (F : Array ByteArray) (c a : ByteArray) (log : List Release) :
    ∀ s, avg (pathCharge (openErr e E) (finish D θ F c a log) s) ≤ 0 := by
  refine avg_pathCharge_send_le _ (openErr_none e openingOf_send) fun _ => ?_
  refine avg_pathCharge_coin_le _ (fun _ => 0)
    (le_of_eq (openErr_none e (openingOf_coin_send hE' fun _ => ⟨_, _, rfl, fun _ => ⟨_, rfl⟩⟩)))
    (fun _ => ?_) ?_
  · exact avg_pathCharge_send_le _ (openErr_none e openingOf_send) fun _ s => le_of_eq (avg_pathCharge_ret _ _ s)
  · simp only [zero_add, avg_const, le_refl]

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The class errors along a play of the period's game within the potential from no tree open**: over a uniform
tape, the class errors of the nodes a play of `σ` passes, at its close. -/
theorem avg_pathCharge_game_le (σ : Strategy (V.game θ F c)) :
    avg (pathCharge (openErr e (finish (V.signer θ) θ F c σ.1)) (V.game θ F c) σ) ≤ treePotential e [] := by
  have h := avg_pathCharge_send (openErr e (finish (V.signer θ) θ F c σ.1))
    (k := fun a => (releases (frames (V.families θ)) (V.families θ) []).bind (finish (V.signer θ) θ F c a)) σ
  refine h.trans_le ?_
  rw [openErr_none e openingOf_send, zero_add]
  exact avg_pathCharge_le_of_eq _
    (releases_bind (frames (V.families θ)) (V.families θ) [] (finish (V.signer θ) θ F c σ.1))
    (avg_pathCharge_relK_le e (finish_recordsLog (V.signer θ) θ F c σ.1)
      (fun log => avg_pathCharge_finish_le e (fun log' => goesOn_finish (V.signer θ) θ F c σ.1 log')
        (V.signer θ) θ F c σ.1 log) _ _ [] _ [] fun _ => rfl) σ.2

/-! ## The empty class at a count -/

/-- **An empty class at a count**: `g'` is an opening's coin node in a play that ends in `E`, its frame asks for `n`
coins, and none of the rounds its canonical description flips among is released as `bits`. -/
def EmptyAtCount (E : List Release → Game β) (g' : Game β) (n : ℕ) (bits : List Bool) : Prop :=
  ∃ o, openingOf E g' = some o ∧ o.n = n ∧ ¬ (o.classOf bits).Nonempty

/-- **The opening's empty class at a count**: the probability that the node's frame asks for `n` coins and no round of
the node is released as `bits` of a uniform `P`. -/
noncomputable def openEmptyAt {P : Type} [Fintype P] (E : List Release → Game β) (g' : Game β) (n : ℕ)
    (bits : P → List Bool) : ℝ≥0∞ :=
  prCoin fun c => EmptyAtCount E g' n (bits c)

theorem openEmptyAt_le_openEmpty {P : Type} [Fintype P] {g' : Game β} (n : ℕ) (bits : P → List Bool) :
    openEmptyAt E g' n bits ≤ openEmpty E g' bits := by
  unfold openEmptyAt openEmpty
  exact prCoin_mono fun _ hu => hu.elim fun o h => ⟨o, h.1, h.2.2⟩

/-- **A gap and an empty class at a count within the node's class error**, from the bound at that count: where the
frame asks for another count, the empty class at the count is none, and either the frame's count fits its block, every
class is empty and the written round is the tape's own, or it doesn't and the class error is 1. -/
theorem openGap_add_openEmptyAt_le {P : Type} [Fintype P] [Nonempty P] {g' : Game β} (n : ℕ) (bits : P → List Bool)
    (hl : ∀ u, (bits u).length = 128 * n)
    (h : (∀ o, openingOf E g' = some o → o.n = n) → openGap E g' bits + openEmpty E g' bits ≤ openErr e E g') :
    openGap E g' bits + openEmptyAt E g' n bits ≤ openErr e E g' := by
  by_cases hn : ∀ o, openingOf E g' = some o → o.n = n
  · exact (add_le_add le_rfl (openEmptyAt_le_openEmpty n bits)).trans (h hn)
  · push Not at hn
    obtain ⟨o, ho, hon⟩ := hn
    have h0 : openEmptyAt E g' n bits = 0 := by
      unfold openEmptyAt
      refine le_antisymm ((prCoin_mono fun u hu => ?_).trans_eq prCoin_false) zero_le
      obtain ⟨o', ho', hn', -⟩ := hu
      obtain rfl := Option.some.inj (ho.symm.trans ho')
      exact hon hn'
    rw [h0, add_zero]
    by_cases hb : o.n ≤ o.T.spec.block
    · have hp : (fun u => pickIn (o.classOf (bits u))) = fun (_ : P) (d : o.Openable) => d := by
        funext u d
        unfold pickIn
        refine dite_eq_right fun hne => ?_
        obtain ⟨d', hd'⟩ := hne
        rw [OpeningAt.classOf, Finset.mem_filter] at hd'
        have h1 := congrArg List.length hd'.2
        rw [length_opened, hl, min_eq_left hb] at h1
        exact hon (by omega)
      have hG : openGap E g' bits = 0 := by
        unfold openGap
        rw [ho]
        show shortfall (fun u => pickIn (o.classOf (bits u))) = 0
        rw [hp]
        exact shortfall_id
      rw [hG]
      exact zero_le
    · rw [openErr_eq_one e ho (by omega)]
      unfold openGap
      rw [ho]
      show shortfall (fun u => pickIn (o.classOf (bits u))) ≤ 1
      exact shortfall_le_one _

/-- **A round's gap and empty class at its count within the node's class error**, at every count the frame asks
for. -/
theorem openGap_add_openEmptyAt_le_openErr {g' : Game β} (m : ℕ) :
    openGap E g' (roundBits e : (Fin m → 𝔽) → List Bool) +
      openEmptyAt E g' m (roundBits e : (Fin m → 𝔽) → List Bool) ≤ openErr e E g' :=
  openGap_add_openEmptyAt_le e m _ (length_roundBits_enc e) (openGap_add_openEmpty_le_openErr e)

/-- **A link's gap and empty class at its count within the node's class error**, at every count the frame asks
for. -/
theorem openGap_add_openEmptyAt_le_openErr_link {g' : Game β} (m : ℕ) :
    openGap E g' (linkBits e : (Fin 2 → Fin m → 𝔽) → List Bool) +
      openEmptyAt E g' (m + m) (linkBits e : (Fin 2 → Fin m → 𝔽) → List Bool) ≤ openErr e E g' :=
  openGap_add_openEmptyAt_le e (m + m) _ (fun u => by
    rw [linkBits, List.length_append, length_roundBits_enc, length_roundBits_enc]
    ring) (openGap_add_openEmpty_le_openErr_link e)

/-! ## The stops' term -/

/-- **Each later stop's opening asks for its round's count**: at every stop the projection reaches and every
submission decoded there, the opening frame at the stop's node asks for the submission's count of coins. -/
def RoundCounts (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg) :
    Prop :=
  ∀ q : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds), Pos.Reached (roundCuts V θ F c rd) q →
    ∀ (dr : Finset (Fin A.n)) (s : SubmitJ (A.J dr (regOf V θ F c A rd σ D))),
      D.sub (A.J dr (regOf V θ F c A rd σ D)) q.modes = some s →
        ∀ o, openingOf (finish (V.signer θ) θ F c σ.1) q.g' = some o → o.n = s.n

/-- **The first stop's opening asks for the link's count**: at the first stop of every tape, the opening frame at its
node asks for both link points' coins. -/
def LinkCounts (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) : Prop :=
  ∀ (x : CoinTape (V.game θ F c) σ)
    (o : OpeningAt (finish (V.signer θ) θ F c σ.1) (Pos.root (roundCuts V θ F c rd) x).toStop.g'),
    openingOf (finish (V.signer θ) θ F c σ.1) (Pos.root (roundCuts V θ F c rd) x).toStop.g' = some o →
      o.n = A.mP + A.mP

/-- **The stops' term within the charge along a play**: where `γL` at the first stop and `γ` at each later one the
projection reaches are within `Γ` at the stop's node, the play's end costs nothing, the start leaves the tape from
the first stop fresh and is the uniform tape, the stops' term is within the charge along a play over a uniform tape. -/
theorem couplesErr_le_of_charge (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    {γ : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞}
    {γL : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞} {b : ℝ≥0∞}
    (Γ : Game Exchange → ℝ≥0∞) (hret : ∀ a, Γ (.ret a) = 0) (hF : FirstStopFresh V θ F c A rd σ M)
    (hS : ∀ H : CoinTape (V.game θ F c) σ → ℝ≥0∞,
      avg (fun ω : (A.extend M.widening).L.Ω => H (M.start ω)) = avg H)
    (hγ : ∀ (dr : Finset (Fin A.n)) (q : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds)),
      Pos.Reached (roundCuts V θ F c rd) q → ∀ s : SubmitJ (A.J dr (regOf V θ F c A rd σ D)),
        D.sub (A.J dr (regOf V θ F c A rd σ D)) q.modes = some s →
          γ q (A.J dr (regOf V θ F c A rd σ D)) s ≤ Γ q.g')
    (hγL : ∀ (x : CoinTape (V.game θ F c) σ) z,
      γL ((Pos.root (roundCuts V θ F c rd) x).toStop.withTape z) ≤ Γ (Pos.root (roundCuts V θ F c rd) x).toStop.g')
    (hb : avg (pathCharge Γ (V.game θ F c) σ) ≤ b) :
    couplesErr V θ F c A rd σ D M γ γL ≤ b := by
  unfold couplesErr
  refine le_trans (avg_mono (g := fun ω => Pos.resample (fun r => Γ r.g' + Pos.stopCharge Γ r.consume)
    (firstStop V θ F c A rd σ M ω)) fun ω => ?_) ?_
  · refine (add_le_add ?_ ?_).trans_eq (Pos.resample_add_consume Γ (Pos.stopCharge Γ) _).symm
    · exact Pos.resample_le_of_le (hγL (M.start ω))
    · exact avg_roundsErr_le D _ _ Γ hret (roundCuts V θ F c rd) (hγ _) A.N _
        (.consume (.toStop (.root (M.start ω))))
  · calc avg (fun ω => Pos.resample (fun r => Γ r.g' + Pos.stopCharge Γ r.consume) (firstStop V θ F c A rd σ M ω))
        = avg (fun ω => Γ (firstStop V θ F c A rd σ M ω).g' +
            Pos.stopCharge Γ (firstStop V θ F c A rd σ M ω).consume) :=
          (hF fun _ r => Γ r.g' + Pos.stopCharge Γ r.consume).symm
      _ ≤ avg (fun ω => pathCharge Γ (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s'
            (firstStop V θ F c A rd σ M ω).x) :=
          avg_mono fun ω => Pos.add_stopCharge_consume_le Γ hret _
      _ ≤ avg (fun ω => pathCharge Γ (V.game θ F c) σ (M.start ω)) :=
          avg_mono fun ω => Pos.pathCharge_walk_le Γ (V.game θ F c) σ (roundCuts V θ F c rd) (M.start ω) id
      _ = avg (pathCharge Γ (V.game θ F c) σ) := hS _
      _ ≤ b := hb

/-- **The sample-coupled start averages any reading of the tape to its uniform average**, wherever the sample's coins
are uniform (`avg_fill`). -/
theorem avg_sampleMap_start (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H)
    (H : CoinTape (V.game θ F c) σ → ℝ≥0∞) :
    avg (fun ω : (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.Ω =>
      H ((sampleMap e V θ F c A rd σ label after κ).start ω)) = avg H :=
  (avg_prod (C := A.Ω) (D := CoinTape (V.game θ F c) σ) _).trans
    ((hκ fun κ' => avg fun t =>
      H (fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m κ' t)).trans
      (avg_fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m H))

/-- **The stops' term at the sample-coupled coin map is within the hiding term**, where each stop's opening asks for
the count of coins the stop releases: each stop's gap and empty class are within the class error of its node
(`openGap_add_openEmpty_le_openErr`, `openGap_add_openEmpty_le_openErr_link`), and the class errors along a play are
within the potential from no tree open (`avg_pathCharge_game_le`, `treePotential_nil`). -/
theorem couplesErr_sampleMap_le (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (label : String) (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H)
    (hR : RoundCounts V θ F c A rd σ D) (hL : LinkCounts V θ F c A rd σ) :
    couplesErr V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ)
      (fun q _ s => openGap (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) +
        openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool))
      (fun q => openGap (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool) +
        openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool)) ≤
      A.hidingTerm :=
  couplesErr_le_of_charge V θ F c A rd σ D _ (openErr e (finish (V.signer θ) θ F c σ.1))
    (fun _ => openErr_none e openingOf_ret) (sampleMap_fresh e V θ F c A rd σ label after κ)
    (avg_sampleMap_start e V θ F c A rd σ label after κ hκ)
    (fun dr q hq s hs => openGap_add_openEmpty_le_openErr e (hR q hq dr s hs))
    (fun x _ => openGap_add_openEmpty_le_openErr_link e (hL x))
    ((avg_pathCharge_game_le e V θ F c σ).trans_eq (treePotential_nil e A))

/-- **The stops' term at the sample-coupled coin map, with each stop's empty class at the count of coins the stop
releases, is within the hiding term** at every strategy: each stop's gap and empty class at its count are within its
node's class error at any count the frame asks for (`openGap_add_openEmptyAt_le_openErr`,
`openGap_add_openEmptyAt_le_openErr_link`), and the class errors along a play are within the potential from no tree
open. -/
theorem couplesErr_sampleMap_le_at (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (label : String) (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    couplesErr V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ)
      (fun q _ s => openGap (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) +
        openEmptyAt (finish (V.signer θ) θ F c σ.1) q.g' s.n (roundBits e : (Fin s.n → 𝔽) → List Bool))
      (fun q => openGap (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool) +
        openEmptyAt (finish (V.signer θ) θ F c σ.1) q.g' (A.mP + A.mP)
          (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool)) ≤
      A.hidingTerm :=
  couplesErr_le_of_charge V θ F c A rd σ D _ (openErr e (finish (V.signer θ) θ F c σ.1))
    (fun _ => openErr_none e openingOf_ret) (sampleMap_fresh e V θ F c A rd σ label after κ)
    (avg_sampleMap_start e V θ F c A rd σ label after κ hκ)
    (fun _ _ _ s _ => openGap_add_openEmptyAt_le_openErr e s.n)
    (fun _ _ => openGap_add_openEmptyAt_le_openErr_link e A.mP)
    ((avg_pathCharge_game_le e V θ F c σ).trans_eq (treePotential_nil e A))

end Verity.Sampled
