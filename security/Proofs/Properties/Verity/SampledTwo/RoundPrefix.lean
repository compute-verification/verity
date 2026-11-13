module

public import Proofs.Properties.Verity.SampledTwo.PrefixFixed

/-!
# The play's log before a round is fixed by the coins its history shows

`Messages.session_prefix_fixed` (`PrefixFixed.lean`) fixes the play's log up to a release the locator puts on a
session's round, where the rounds rise along the log. The messages' piece needs as much of every round, and of the
rounds no release is put on: the statements' round, and the round a session is decided on, whose message the device's
log holds a digest of in a release the locator puts on no round.

This module states it of **the log before round `i`** (`CoinMap.before`): the releases up to the first the locator
puts on a round at least `i`, and that release's frame and bytes; the whole log where there is none. Every release of
that part is put on a round below `i` or on none, by its definition, so no order of the rounds along the log is asked.

* `CoinMap.before_append`, `CoinMap.before_first`, `CoinMap.before_all`: the log before round `i` of a log whose first
  releases are put on no round at least `i`; where the next release is put on such a round; and where the log ends.
* `CoinMap.before_sameUntil`, `CoinMap.before_place`: the coin map's plays at two runs' coins that agree at every round
  below `i` a release is put on have the same log before round `i` (`CoinMap.same_place`).
* `Messages.before_shown`: so do the plays at the run's coins and at any coins that agree with what the play's history
  before round `i` shows, on a play accepted whose log the check read is the device's, where each round a release is
  put on answers with its coin (`Messages.ShowsPlaced`).
* `Messages.fixed_of_before`: so whatever a play fixes for round `i` that reads only its log before round `i` is fixed
  by the coins the history before round `i` shows.

What this does not prove: that the log before round `i` holds what round `i`'s message is fixed by. At a round a
release is put on, that the release is the first put on a round at least `i` is the rounds rising along the log
(`SessionCoins.LogRising`).
-/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

open Verity.Game
open Claims

open Classical in
/-- **The log before round `i`**, after the releases `pre`: the releases of the log up to the first the locator puts
on a round at least `i`, and that release's frame and bytes; the whole log, and none, where there is no such
release. -/
noncomputable def before (L : Locate) (i : ℕ) :
    List Verity.Release → List Verity.Release → List Verity.Release × Option (Verity.Frame × ByteArray)
  | _, [] => ([], none)
  | pre, rl :: log =>
    if ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j then ([], some (rl.frame, rl.answers))
    else (rl :: (before L i (pre ++ [rl]) log).1, (before L i (pre ++ [rl]) log).2)

section Before

variable (L : Locate) {K : ℕ}

/-- The log before round `i` ends at a release put on a round at least `i`. -/
theorem before_cons_pos {i : ℕ} {pre : List Verity.Release} {rl : Verity.Release} (log : List Verity.Release)
    (h : ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j) :
    before L i pre (rl :: log) = ([], some (rl.frame, rl.answers)) := by
  simp only [before, ite_eq_left h]

/-- The log before round `i` goes past a release put on a round below `i`, or on none. -/
theorem before_cons_neg {i : ℕ} {pre : List Verity.Release} {rl : Verity.Release} (log : List Verity.Release)
    (h : ¬ ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j) :
    before L i pre (rl :: log) = (rl :: (before L i (pre ++ [rl]) log).1, (before L i (pre ++ [rl]) log).2) := by
  simp only [before, ite_eq_right h]

/-- **The log before round `i` goes past releases put on no round at least `i`**: it is those releases, then the
log before round `i` of what follows them. -/
theorem before_append {i : ℕ} : ∀ (q pre rest : List Verity.Release),
    (∀ a rl b, q = a ++ rl :: b → ¬ ∃ j, locAt L (pre ++ a) rl.frame rl.answers = some j ∧ i ≤ j) →
      (before L i pre (q ++ rest)).1 = q ++ (before L i (pre ++ q) rest).1 ∧
        (before L i pre (q ++ rest)).2 = (before L i (pre ++ q) rest).2
  | [], pre, rest, _ => by simp
  | rl :: q, pre, rest, h => by
    have h0 : ¬ ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j := by
      have h' := h [] rl q rfl
      rw [List.append_nil] at h'
      exact h'
    obtain ⟨h1, h2⟩ := before_append (i := i) q (pre ++ [rl]) rest fun a rl₂ b hq => by
      rw [show pre ++ [rl] ++ a = pre ++ rl :: a by simp]
      exact h (rl :: a) rl₂ b (by simp [hq])
    rw [show pre ++ [rl] ++ q = pre ++ rl :: q by simp] at h1 h2
    have e : before L i pre (rl :: q ++ rest) =
        (rl :: (before L i (pre ++ [rl]) (q ++ rest)).1, (before L i (pre ++ [rl]) (q ++ rest)).2) :=
      before_cons_neg L (q ++ rest) h0
    rw [e]
    exact ⟨congrArg (fun l => rl :: l) h1, h2⟩

/-- **The log before round `i` where a release is the first put on a round at least `i`**: the releases before it,
and its frame and bytes. -/
theorem before_first {i : ℕ} {q rest : List Verity.Release} {rl : Verity.Release} {j : ℕ}
    (hq : ∀ a rl₀ b, q = a ++ rl₀ :: b → ¬ ∃ j, locAt L a rl₀.frame rl₀.answers = some j ∧ i ≤ j)
    (hj : locAt L q rl.frame rl.answers = some j) (hij : i ≤ j) :
    before L i [] (q ++ rl :: rest) = (q, some (rl.frame, rl.answers)) := by
  obtain ⟨h1, h2⟩ := before_append L (i := i) q [] (rl :: rest) fun a rl₀ b hqe => by
    rw [List.nil_append]
    exact hq a rl₀ b hqe
  rw [List.nil_append, before_cons_pos L (i := i) (pre := q) (rl := rl) rest ⟨j, hj, hij⟩] at h1 h2
  exact Prod.ext (h1.trans (List.append_nil q)) h2

/-- **The log before round `i` where no release is put on a round at least `i`**: the whole log. -/
theorem before_all {i : ℕ} {q : List Verity.Release}
    (hq : ∀ a rl₀ b, q = a ++ rl₀ :: b → ¬ ∃ j, locAt L a rl₀.frame rl₀.answers = some j ∧ i ≤ j) :
    before L i [] q = (q, none) := by
  obtain ⟨h1, h2⟩ := before_append L (i := i) q [] [] fun a rl₀ b hqe => by
    rw [List.nil_append]
    exact hq a rl₀ b hqe
  rw [List.append_nil] at h1 h2
  exact Prod.ext (h1.trans (List.append_nil q)) h2

/-- **Two logs that are one until the coins differ have one log before round `i`**, where the coins agree at every
round below `i` a release of the first is put on: a release put on a round at least `i` has the same frame and bytes
in both, and a release before it is written from coins that agree (`ChunkAgree`), or from none. -/
theorem before_sameUntil {E : Prop} {r r' : Fin K → FlockDeploy.Coin} (i : ℕ) :
    ∀ (log pre log' : List Verity.Release), SameUntil L E r r' pre log log' →
      (∀ a rl b, log = a ++ rl :: b → ∀ j : Fin K, locAt L (pre ++ a) rl.frame rl.answers = some j.val →
        j.val < i → r j = r' j) →
      before L i pre log' = before L i pre log
  | [], pre, log', h, _ => by
    obtain ⟨rfl, -⟩ := h
    rfl
  | rl :: log, pre, log', h, hr => by
    obtain ⟨rl', rest, rfl, hf, ha, hstep⟩ := h
    by_cases hp : ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j
    · have hp' : ∃ j, locAt L pre rl'.frame rl'.answers = some j ∧ i ≤ j := by
        rw [hf, ha]
        exact hp
      rw [before_cons_pos L rest hp', before_cons_pos L log hp, hf, ha]
    · have hc : ChunkAgree L r r' pre rl := by
        intro j u hs a _ _
        obtain ⟨hl, -, -⟩ := slot_spec L hs
        have hji : j.val < i := Nat.lt_of_not_le fun hge => hp ⟨j.val, hl, hge⟩
        rw [hr [] rl log rfl j (by rw [List.append_nil]; exact hl) hji]
      obtain ⟨he, hs⟩ := hstep hc
      have ih := before_sameUntil i log (pre ++ [rl]) rest hs fun a rl₂ b hlog j hj hji =>
        hr (rl :: a) rl₂ b (by simp [hlog]) j
          (by rw [show pre ++ rl :: a = pre ++ [rl] ++ a by simp]; exact hj) hji
      rw [he, before_cons_neg L rest hp, before_cons_neg L log hp, ih]

end Before

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The coin map's plays at two runs' coins have one log before round `i`**, where the coins agree at every round
below `i` a release of the first play's log is put on. -/
theorem before_place (L : Locate) {K : ℕ} (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (r r' : Fin K → FlockDeploy.Coin) (ω : Tape (V.game θ F c) σ) (i : ℕ)
    (h : ∀ a rl b, (Tape.out _ σ (place V θ F c σ L r ω)).log = a ++ rl :: b →
      ∀ j : Fin K, locAt L a rl.frame rl.answers = some j.val → j.val < i → r j = r' j) :
    before L i [] (Tape.out _ σ (place V θ F c σ L r' ω)).log =
      before L i [] (Tape.out _ σ (place V θ F c σ L r ω)).log := by
  have hs := same_place L V θ F c σ r r' ω
  unfold SameAt at hs
  rw [List.length_nil, List.drop_zero, List.drop_zero] at hs
  exact before_sameUntil L i _ [] _ hs fun a rl b hlog j hj hji =>
    h a rl b hlog j (by rw [List.nil_append] at hj; exact hj) hji

end Verity.SampledTwo.CoinMap

namespace Verity.SampledTwo.Messages

open Claims

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The play's log before round `i` is fixed by the coins the history before round `i` shows.** At the coin map's
play with the run's coins `r`, accepted (`Acc`) and with the check's log the device's (`Good`), any coins `r'` that
agree with every coin the play's history before round `i` shows give a play with the same log before round `i`
(`CoinMap.before`). Each round a release is put on answers with its coin (`ShowsPlaced`), so the history shows the
coin of every round below `i` a release is put on. -/
theorem before_shown (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L) (ω : TapeOf V θ F c σ)
    (r r' : Coins A x)
    (hacc : Acc (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)))
    (hgood : Good (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω))) (i : ℕ)
    (hs : Shows (playHist A x r
      (playMsgs Rd (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω))) i) r') :
    CoinMap.before L i [] (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r' ω)).log =
      CoinMap.before L i [] (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)).log :=
  CoinMap.before_place L V θ F c σ r r' ω i fun a rl b hlog j hj hji =>
    (hs j _ (shownAt_playHist A x r _ j hji (hShow _ hacc hgood a rl b hlog j.val hj r j.isLt))).symm

/-- **What a play fixes for round `i` is fixed by the coins the history before it shows, where it reads only the log
before round `i`**: at a frame whose coin map is the locator `L`'s, a value `Fx` reads off the play's log before
round `i` (`CoinMap.before`) is the same at the run's coins and at any coins that agree with every coin the play's
history before round `i` shows. -/
theorem fixed_of_before (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Fr : Frame V A x θ F c σ)
    (hFr : ∀ ω r, Fr.place ω r = CoinMap.place V θ F c σ L r ω) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L) {Φ : Type}
    (Fx : Verity.Exchange → ℕ → Option Φ) (i : ℕ)
    (g : List Verity.Release × Option (Verity.Frame × ByteArray) → Option Φ)
    (hFx : ∀ o, Fx o i = g (CoinMap.before L i [] o.log)) (ω : TapeOf V θ F c σ) (r r' : Coins A x)
    (hacc : Acc (Fr.out ω r)) (hgood : Good (Fr.out ω r))
    (hs : Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r') :
    Fx (Fr.out ω r') i = Fx (Fr.out ω r) i := by
  have e : ∀ r₀, Fr.out ω r₀ = Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r₀ ω) := fun r₀ =>
    congrArg (Verity.Game.Tape.out (V.game θ F c) σ) (hFr ω r₀)
  rw [e r] at hacc hgood hs
  rw [hFx, hFx, e r', e r, before_shown V A x θ F c σ L Rd Acc Good hShow ω r r' hacc hgood i hs]

end Verity.SampledTwo.Messages
