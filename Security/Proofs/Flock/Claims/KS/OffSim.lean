import Proofs.Flock.Claims.KS.ForkSim
import Proofs.Flock.Claims.KS.OffAlign

/-!
# The plurality off mass after a number of coins, through a simulation and on the round coins

`offC op j` (`OffAlign`) is one run to its `j`-th coin, then the continuation's plurality off mass. Route (c) carries it
from the simulated session to the deployed runs as one run's mass, never as a fork:

* `Rewinding.plurOff_ret`, `offC_ret`, `offC_le_open`, `offC_eqRec`, `offC_cast`: at an end nothing is off a plurality
  table; the off mass is at most the mass that opens anything; it survives a cast.
* `Claims.Bridge.liveOp`: the opening of the model outcome a live outcome is related to, where the relation is
  functional (`liveOp_eq`).
* `Claims.Bridge.Sim.offC_le`: through a simulation whose returns relate the live and model outcomes by `R`, where each
  model opening is the related live outcome's and the dead branches open nothing, the model's off mass after `j` coins
  is at most the live one's. At each node the live plurality table is one the model's plurality beats
  (`Sim.plurOff_le`), and the simulation matches the coins one to one.
* `Claims.Bridge.offX`, `ex_off_le`: on the round coins, the off mass after `j` rounds is at most the probability over
  the coins of opening off a table `W r` that reads only the first `j` coins. At each node after `j` rounds that table
  is one the plurality beats, and the rest of the coins are that node's run (`ex_playOn`).
* `offX_fresh`, `Live.offC_liveStrat_le`: on a game whose every coin answers its own send, `j` rounds are `j` coins, so
  at the live session the off mass after `j` coins is at most that probability at the run's coins.
-/

namespace FlockSoundness

namespace Rewinding

open Game

variable {α P V : Type} [Fintype P] [Fintype V] [DecidableEq V] [Nonempty V] (op : α → P → Option V)

/-- **At an end nothing is off a plurality table**: the end's own openings are a table. -/
theorem plurOff_ret (a : α) (s : Strategy (Game.ret a)) : plurOff op (.ret a) s = 0 := by
  refine le_antisymm ((plurOff_le op _ s fun p => (op a p).getD (Classical.arbitrary V)).trans_eq ?_)
    (plurOff_nonneg op _ s)
  refine Finset.sum_eq_zero fun p _ => ?_
  show ind (fun a' => Off op (fun p => (op a p).getD (Classical.arbitrary V)) a' p) a = 0
  unfold ind
  split_ifs with h
  · exfalso
    obtain ⟨hs, hne⟩ := h
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
    exact hne (by simp only [hv, Option.getD_some])
  · rfl

end Rewinding

namespace Game

variable {α P V : Type} [Fintype P] [Fintype V] [DecidableEq V] [Nonempty V]

theorem offC_ret (op : α → P → Option V) (j : ℕ) (a : α) (s : Strategy (Game.ret a)) :
    offC op j (.ret a) s = 0 := by
  cases j
  · exact Rewinding.plurOff_ret op a s
  · rfl

/-- **The off mass is at most the mass that opens anything.** -/
theorem offC_le_open (op : α → P → Option V) :
    ∀ (j : ℕ) (g : Game α) (s : Strategy g), offC op j g s ≤ ∑ p, expect (ind fun a => (op a p).isSome) g s
  | 0, g, s => Finset.sum_le_sum fun p _ => expect_mono (ind_mono fun _ h => h.1) g s
  | _ + 1, .ret a, s => by
      rw [offC_ret]
      exact Finset.sum_nonneg fun _ _ => expect_nonneg (ind_nonneg _) _ s
  | j + 1, .send _ k, s => by rw [offC_send]; exact offC_le_open op (j + 1) (k s.1) s.2
  | j + 1, @Game.coin _ C _ _ k, s => by
      rw [offC_coin]
      calc (∑ c : C, offC op j (k c) (s c)) / Fintype.card C
          ≤ (∑ c : C, ∑ p, expect (ind fun a => (op a p).isSome) (k c) (s c)) / Fintype.card C :=
            div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => offC_le_open op j (k c) (s c))
              (Nat.cast_nonneg _)
        _ = ∑ p, expect (ind fun a => (op a p).isSome) (.coin C k) s := by
            rw [Finset.sum_comm, Finset.sum_div]
            rfl

theorem offC_eqRec (op : α → P → Option V) {g g' : Game α} (h : g = g') (j : ℕ) (s : Strategy g) :
    offC op j g' (h ▸ s) = offC op j g s := by
  subst h
  rfl

theorem offC_cast (op : α → P → Option V) {g g' : Game α} (h : g = g') (j : ℕ) (s : Strategy g) :
    offC op j g' (cast (congrArg Strategy h) s) = offC op j g s := by
  subst h
  rfl

end Game

end FlockSoundness

namespace Claims.Bridge

open FlockSoundness FlockSoundness.Game

/-! ## Through a simulation -/

section OffSim

variable {β α Pt V : Type} [Fintype Pt] [Fintype V] [DecidableEq V] [Nonempty V] {pL : β → Prop} {pg : α → Prop}

open Classical in
/-- **The opening of the related model outcome**: `op` at the model outcome `b` is related to, none where it is related
to none. -/
noncomputable def liveOp (R : β → α → Prop) (op : α → Pt → Option V) (b : β) (p : Pt) : Option V :=
  if h : ∃ a, R b a then op (Classical.choose h) p else none

omit [Fintype Pt] [Fintype V] [DecidableEq V] [Nonempty V] in
/-- Where the relation is functional, the related outcome's opening. -/
theorem liveOp_eq {R : β → α → Prop} (hR : ∀ b a a', R b a → R b a' → a = a') (op : α → Pt → Option V) {b : β}
    {a : α} (h : R b a) : liveOp R op b = op a := by
  funext p
  unfold liveOp
  split_ifs with he
  · rw [hR b _ a (Classical.choose_spec he) h]
  · exact absurd ⟨a, h⟩ he

theorem ind_le_ind_imp {γ δ : Type} {p : γ → Prop} {q : δ → Prop} {a : γ} {b : δ} (h : p a → q b) :
    ind p a ≤ ind q b := by
  unfold ind
  split_ifs with h1 h2 <;> first | exact absurd (h h1) h2 | norm_num

open Classical in
/-- **The plurality off mass at a node, through a simulation**: the live plurality table is one the model's plurality
beats, and against a fixed table the off mass is one run's, which the simulation carries (`Sim.expect_le`). -/
theorem Sim.plurOff_le {R : β → α → Prop} {Q : α → Prop} {op : α → Pt → Option V} {op' : β → Pt → Option V}
    (hQ : ∀ a p, (op a p).isSome → Q a) (hR : ∀ b a, R b a → ∀ p v, op a p = some v → op' b p = some v)
    {L : Game β} {g : Game α} (d : Sim pL pg L g) (h : Sim.All R (fun g σ => prob Q g σ = 0) d) (s : Strategy L) :
    Rewinding.plurOff op g (d.strategy s) ≤ Rewinding.plurOff op' L s := by
  refine (Rewinding.plurOff_le op g _ (Rewinding.plurOf op' L s)).trans (Finset.sum_le_sum fun p _ => ?_)
  refine Sim.expect_le (f := ind fun a => Rewinding.Off op (Rewinding.plurOf op' L s) a p)
    (f' := ind fun b => Rewinding.Off op' (Rewinding.plurOf op' L s) b p)
    (fun b a hba => ind_le_ind_imp fun hoff => ?_) (fun b => ind_nonneg _ _)
    (fun _ _ hσ => expect_le_zero_of_prob (fun a => ind_le_ind_imp fun h => hQ a p h.1) hσ) d h s
  obtain ⟨hs, hne⟩ := hoff
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
  have hv' := hR b a hba p v hv
  refine ⟨by rw [hv']; rfl, ?_⟩
  rw [hv']
  rw [hv] at hne
  exact hne

open Classical in
/-- **The off mass after `j` coins through a simulation**: as `Sim.forkC_le`, at each node by `Sim.plurOff_le`. -/
theorem Sim.offC_le {R : β → α → Prop} {Q : α → Prop} {op : α → Pt → Option V} {op' : β → Pt → Option V}
    (hQ : ∀ a p, (op a p).isSome → Q a) (hR : ∀ b a, R b a → ∀ p v, op a p = some v → op' b p = some v) :
    ∀ {L : Game β} {g : Game α} (d : Sim pL pg L g), Sim.All R (fun g σ => prob Q g σ = 0) d →
      ∀ (j : ℕ) (s : Strategy L), offC op j g (d.strategy s) ≤ offC op' j L s
  | _, _, @Sim.ret _ _ _ _ _ a _, _, j, s => by
      rw [offC_ret]
      exact offC_nonneg op' j _ s
  | _, _, .model _ d, h, j, s => (offC_send op j _).trans_le (Sim.offC_le hQ hR d h j s)
  | _, _, .live k, h, j, s =>
      (Sim.offC_le hQ hR (k s.1) (h s.1) j s.2).trans_eq (offC_send op' j s).symm
  | _, _, @Sim.coin _ _ _ _ C C' iF iN iF' iN' k k' e f, h, 0, s =>
      Sim.plurOff_le hQ hR (Sim.coin e f) h s
  | _, _, @Sim.coin _ _ _ _ C C' iF iN iF' iN' k k' e f, h, j + 1, s => by
      rw [offC_coin, offC_coin]
      simp only [Sim.strategy]
      rw [← e.sum_comp, @Fintype.card_congr C' C iF' iF e]
      exact div_le_div_of_nonneg_right
        (Finset.sum_le_sum fun c' _ => Sim.offC_le hQ hR (f c') (h c') j (s (e c'))) (Nat.cast_nonneg _)
  | _, _, .dead _ σ, h, j, s =>
      ((offC_le_open op j _ σ).trans (Finset.sum_nonpos fun p _ =>
        expect_le_zero_of_prob (fun a => ind_le_ind_imp fun h => hQ a p h) h)).trans (offC_nonneg op' j _ s)

end OffSim

/-! ## On the round coins -/

section OffX

variable {β Pt V Coin : Type} [Fintype Pt] [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Coin]

/-- The plurality off mass from a position with the pending coin `p`: a pending coin read by a `coin` node. -/
noncomputable def plurOffP (op : β → Pt → Option V) :
    (g : Game β) → (I : Iface Coin g) → RStrat Coin g I → Option Coin → ℝ
  | .ret b, I, s, _ => Rewinding.plurOff op (.ret b) (toStrat _ I s)
  | .send M k, I, s, _ => Rewinding.plurOff op (.send M k) (toStrat _ I s)
  | @Game.coin _ _ _ _ k, I, s, some u => Rewinding.plurOff op (k (I.1.proj u)) (toStrat _ _ (s (I.1.proj u)))
  | @Game.coin _ C iF iN k, I, s, none => Rewinding.plurOff op (@Game.coin _ C iF iN k) (toStrat _ I s)

/-- **The plurality off mass after `j` rounds**: the plays share their first `j` round coins, then the continuation's
plurality off mass. -/
noncomputable def offX (op : β → Pt → Option V) :
    ℕ → (g : Game β) → (I : Iface Coin g) → RStrat Coin g I → Option Coin → ℝ
  | 0, g, I, s, p => plurOffP op g I s p
  | _ + 1, .ret _, _, _, _ => 0
  | j + 1, .send _ k, I, s, _ => Ex fun u : Coin => offX op j (k (I.1 s.1)) (I.2 s.1) s.2 (some u)
  | j + 1, @Game.coin _ _ _ _ k, I, s, some u => offX op (j + 1) (k (I.1.proj u)) (I.2 _) (s _) none
  | j + 1, @Game.coin _ _ _ _ k, I, s, none => Ex fun u : Coin => offX op j (k (I.1.proj u)) (I.2 _) (s _) none

omit [Fintype Coin] in
/-- A function of the first `j + 1` coins, at a first coin `c`, is a function of the first `j` of the rest. -/
theorem prefixDep_cons {n j : ℕ} {X : Type} {W : (Fin (n + 1) → Coin) → X}
    (hW : ∀ r r' : Fin (n + 1) → Coin, (∀ i : Fin (n + 1), i.val < j + 1 → r i = r' i) → W r = W r') (c : Coin) :
    ∀ r r' : Fin n → Coin, (∀ i : Fin n, i.val < j → r i = r' i) → W (Fin.cons c r) = W (Fin.cons c r') := by
  intro r r' h
  refine hW _ _ fun i hi => ?_
  cases i using Fin.cases with
  | zero => simp only [Fin.cons_zero]
  | succ i =>
    rw [Fin.cons_succ, Fin.cons_succ]
    exact h i (by simp only [Fin.val_succ] at hi; omega)

theorem Ex_sum_univ {γ ι : Type} [Fintype γ] [Fintype ι] (f : ι → γ → ℝ) :
    (Ex fun a => ∑ i, f i a) = ∑ i, Ex (f i) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

/-- At a fixed table, the plurality beats it at the position's node. -/
theorem plurOffP_le (op : β → Pt → Option V) (w : Pt → V) :
    ∀ (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I) (p : Option Coin),
      plurOffP op g I s p ≤ ∑ q, expectP (ind fun b => Rewinding.Off op w b q) g I s p
  | .ret _, _, _, _ => Rewinding.plurOff_le op _ _ w
  | .send _ _, _, _, _ => Rewinding.plurOff_le op _ _ w
  | @Game.coin _ _ _ _ _, _, _, some _ => Rewinding.plurOff_le op _ _ w
  | @Game.coin _ _ _ _ _, _, _, none => Rewinding.plurOff_le op _ _ w

variable [Inhabited Coin]

theorem ex_off_le_zero [Nonempty Coin] (op : β → Pt → Option V) (g : Game β) (I : Iface Coin g)
    (s : RStrat Coin g I) (p : Option Coin) (n : ℕ) (hR : RoundsLe g I s p.isSome n)
    (W : (Fin n → Coin) → Pt → V) (hW : ∀ r r' : Fin n → Coin, (∀ i : Fin n, i.val < 0 → r i = r' i) → W r = W r') :
    offX op 0 g I s p ≤ Ex fun r : Fin n → Coin =>
      ∑ q, ind (fun b => Rewinding.Off op (W r) b q) (playOn g I s p (coinSeq r)) := by
  have hw : ∀ r, W r = W default := fun r => hW r _ fun i hi => absurd hi (Nat.not_lt_zero _)
  have hE : (fun r : Fin n → Coin => ∑ q, ind (fun b => Rewinding.Off op (W r) b q) (playOn g I s p (coinSeq r))) =
      fun r => ∑ q, ind (fun b => Rewinding.Off op (W default) b q) (playOn g I s p (coinSeq r)) :=
    funext fun r => by rw [hw r]
  rw [hE, Ex_sum_univ]
  calc offX op 0 g I s p = plurOffP op g I s p := by rw [offX]
    _ ≤ ∑ q, expectP (ind fun b => Rewinding.Off op (W default) b q) g I s p := plurOffP_le op (W default) g I s p
    _ = _ := Finset.sum_congr rfl fun q _ =>
        (ex_playOn (ind fun b => Rewinding.Off op (W default) b q) g I s p n hR).symm

/-- **The off mass after `j` rounds is at most the mass off a table that reads the first `j` coins**: at each node
after `j` rounds that table is one the plurality beats, and the rest of the coins are the node's run. -/
theorem ex_off_le [Nonempty Coin] (op : β → Pt → Option V) :
    ∀ (g : Game β) (I : Iface Coin g) (s : RStrat Coin g I) (p : Option Coin) (n j : ℕ),
      RoundsLe g I s p.isSome n → ∀ W : (Fin n → Coin) → Pt → V,
        (∀ r r' : Fin n → Coin, (∀ i : Fin n, i.val < j → r i = r' i) → W r = W r') →
        offX op j g I s p ≤ Ex fun r : Fin n → Coin =>
          ∑ q, ind (fun b => Rewinding.Off op (W r) b q) (playOn g I s p (coinSeq r)) := by
  intro g
  induction g with
  | ret b =>
    intro I s p n j hR W hW
    cases j with
    | zero => exact ex_off_le_zero op _ I s p n hR W hW
    | succ j => exact Ex_nonneg fun r => Finset.sum_nonneg fun q _ => ind_nonneg _ _
  | send M k ih =>
    intro I s p n j hR W hW
    cases j with
    | zero => exact ex_off_le_zero op _ I s p n hR W hW
    | succ j =>
      cases n with
      | zero => cases p <;> exact hR.elim
      | succ n =>
        have hR' : RoundsLe (k (I.1 s.1)) (I.2 s.1) s.2 true n := by cases p <;> exact hR
        rw [Ex_cons]
        simp only [playOn_send_cons]
        rw [offX]
        exact Ex_mono fun c => ih (I.1 s.1) (I.2 s.1) s.2 (some c) n j hR' (fun r => W (Fin.cons c r)) (prefixDep_cons hW c)
  | coin C k ih =>
    intro I s p n j hR W hW
    cases j with
    | zero => exact ex_off_le_zero op _ I s p n hR W hW
    | succ j =>
      cases p with
      | some u =>
        simp only [playOn]
        rw [offX]
        exact ih _ _ _ none n (j + 1) (hR _) W hW
      | none =>
        cases n with
        | zero => exact hR.elim
        | succ n =>
          rw [Ex_cons]
          simp only [playOn_coin_cons]
          rw [offX]
          exact Ex_mono fun c => ih _ _ _ none n j (hR _) (fun r => W (Fin.cons c r)) (prefixDep_cons hW c)

omit [Inhabited Coin] in
theorem offC_noCoin (op : β → Pt → Option V) : ∀ (g : Game β) (s : Strategy g), NoCoin g → ∀ j j' : ℕ,
    offC op j g s = offC op j' g s
  | .ret _, _, _, j, j' => by rw [offC_ret, offC_ret]
  | .send _ k, s, h, j, j' => by
      rw [offC_send, offC_send]
      exact offC_noCoin op (k s.1) s.2 (noCoin_send h s.1) j j'
  | @Game.coin _ _ _ _ _, _, h, _, _ => by unfold NoCoin at h; exact h.elim

omit [Inhabited Coin] in
theorem offX_noCoin [Nonempty Coin] (op : β → Pt → Option V) : ∀ (g : Game β) (I : Iface Coin g)
    (s : RStrat Coin g I), NoCoin g → ∀ (j : ℕ) (p : Option Coin), offX op j g I s p = offC op 0 g (toStrat g I s)
  | .ret _, _, _, _, 0, _ => rfl
  | .ret b, _, _, _, _ + 1, _ => (Rewinding.plurOff_ret op b _).symm
  | .send _ _, _, _, _, 0, _ => rfl
  | .send _ k, I, s, h, j + 1, _ => by
      rw [offX, offC_send]
      simp only [toStrat]
      rw [show (fun u : Coin => offX op j (k (I.1 s.1)) (I.2 s.1) s.2 (some u)) =
        fun _ => offC op 0 (k (I.1 s.1)) (toStrat _ (I.2 s.1) s.2) from
          funext fun u => offX_noCoin op _ _ _ (noCoin_send h _) j (some u)]
      exact Ex_const _
  | @Game.coin _ _ _ _ _, _, _, h, _, _ => by unfold NoCoin at h; exact h.elim

omit [Inhabited Coin] in
/-- **`j` rounds are `j` coins** where every coin answers its own send: at a position with no round open, the off mass
after `j` rounds is the off mass after `j` coins; right after a send, the pending coin averaged, after `j + 1`. -/
theorem offX_fresh [Nonempty Coin] (op : β → Pt → Option V) : ∀ (g : Game β) (I : Iface Coin g)
    (s : RStrat Coin g I),
    (Fresh true g → ∀ (j : ℕ) (p : Option Coin), offX op j g I s p = offC op j g (toStrat g I s)) ∧
    (Fresh false g → ∀ j : ℕ, Ex (fun u : Coin => offX op j g I s (some u)) = offC op (j + 1) g (toStrat g I s))
  | .ret b, _, _ => ⟨fun _ j _ => by cases j <;> rfl, fun _ j => by
      cases j
      · exact (Ex_const _).trans (Rewinding.plurOff_ret op b _)
      · exact Ex_const _⟩
  | .send _ k, I, s => by
    refine ⟨fun h j p => ?_, fun h j => ?_⟩
    · cases j with
      | zero => rfl
      | succ j =>
        rw [offX, offC_send]
        exact (offX_fresh op _ _ _).2 (fresh_true_send h _) j
    · rw [show (fun u : Coin => offX op j (.send _ k) I s (some u)) = fun _ => offC op 0 (.send _ k) (toStrat _ I s)
        from funext fun u => offX_noCoin op _ I s (fresh_false_send h) j (some u), Ex_const]
      exact offC_noCoin op _ _ (fresh_false_send h) 0 (j + 1)
  | @Game.coin _ C _ _ k, I, s => by
    refine ⟨fun h => by unfold Fresh at h; exact h.elim, fun h j => ?_⟩
    cases j with
    | zero =>
      simp only [offX, plurOffP]
      exact I.1.unif (fun c => Rewinding.plurOff op (k c) (toStrat _ (I.2 c) (s c)))
    | succ j =>
      simp only [offX]
      rw [show (fun u : Coin => offX op (j + 1) (k (I.1.proj u)) (I.2 _) (s _) none) =
        fun u => offC op (j + 1) (k (I.1.proj u)) (toStrat _ (I.2 _) (s _)) from
          funext fun u => (offX_fresh op _ _ _).1 (fresh_false_coin h _) (j + 1) none,
        I.1.unif (fun c => offC op (j + 1) (k c) (toStrat _ (I.2 c) (s c)))]
      rfl

end OffX

namespace Live

open FlockSoundness.Discharge.Tables

variable {Pt V : Type} [Fintype Pt] [Fintype V] [DecidableEq V] [Nonempty V]

/-- **At the live session, the off mass after `j` coins is at most the mass off a table that reads the run's first `j`
coins**, over the run's coins. -/
theorem offC_liveStrat_le {nMax J mPts : ℕ} (N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : Msg → Wire nMax J mPts) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (op : LiveOutJ J mPts → Pt → Option V) (j : ℕ) (W : (Fin (N x + 2) → LCoin nMax) → Pt → V)
    (hW : ∀ r r' : Fin (N x + 2) → LCoin nMax, (∀ i : Fin (N x + 2), i.val < j → r i = r' i) → W r = W r') :
    offC op j (liveSession dg J mPts (N x)) (liveStrat (fun _ => J) (fun _ => mPts) N dg w H P x ω) ≤
      Ex fun r : Fin (N x + 2) → LCoin nMax => ∑ q, ind (fun b => Rewinding.Off op (W r) b q)
        (liveOut (fun _ => J) (fun _ => mPts) N dg w H P x ω r) := by
  rw [liveStrat, offC_eqRec]
  exact ((offX_fresh op _ _ _).1 (fresh_sessionK dg J mPts (N x)) j none).symm.trans_le
    (ex_off_le op _ _ _ none (N x + 2) j (sessionLe (w x) dg (N x) _) W hW)

end Live

end Claims.Bridge
