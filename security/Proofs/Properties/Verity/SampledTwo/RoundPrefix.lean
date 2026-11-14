module

public import Proofs.Properties.Verity.SampledTwo.PrefixFixed
public import Proofs.Properties.Verity.SampledTwo.MessagesOpen

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
* `Messages.fixed_of_before`: so whatever a play fixes for round `i` that reads, on the frame's plays, only its log
  before round `i` is fixed by the coins the history before round `i` shows.

* `Messages.fixedBefore_of_before`: `Messages.FixedBefore` (`MessagesOpen.lean`) at a frame whose coin map is the
  locator's, for a value read off the log before the round and opened only by accepted plays whose log the check read
  is the device's.
* `Messages.fixedBefore_strings_before`: the same at digests and a context (`Messages.stringsFx`), the digests read
  off the log before the round and the context the same on two accepted plays the history's coins show alike.
* `SessionCoins.before_session`: where the rounds rise along the log (`SessionCoins.LogRising`), the log before a
  session's round a release is put on is the releases before that release, with its frame and bytes.

What this does not prove: that the log before round `i` holds what round `i`'s message is fixed by. At a session's
round a release is put on, the bytes are the ones the release answers; that they are a digest of the round's message
is the form's. At the statements' round and at the round a session is decided on, which no release is put on, where
the releases the message is fixed by lie in the log before the round is not shown here.
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
before round `i`**: at a frame whose coin map is the locator `L`'s, a value `Fx` reads, on the frame's plays, off the
play's log before round `i` (`CoinMap.before`) is the same at the run's coins and at any coins that agree with every
coin the play's history before round `i` shows. `Fx` may read the game's parameters, files and commitment too: they
are the same on every play of the frame. -/
theorem fixed_of_before (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Fr : Frame V A x θ F c σ)
    (hFr : ∀ ω r, Fr.place ω r = CoinMap.place V θ F c σ L r ω) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L) {Φ : Type}
    (Fx : Verity.Exchange → ℕ → Option Φ) (i : ℕ)
    (g : List Verity.Release × Option (Verity.Frame × ByteArray) → Option Φ) (ω : TapeOf V θ F c σ)
    (hFx : ∀ r₀, Fx (Fr.out ω r₀) i = g (CoinMap.before L i [] (Fr.out ω r₀).log)) (r r' : Coins A x)
    (hacc : Acc (Fr.out ω r)) (hgood : Good (Fr.out ω r))
    (hs : Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r') :
    Fx (Fr.out ω r') i = Fx (Fr.out ω r) i := by
  have e : ∀ r₀, Fr.out ω r₀ = Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r₀ ω) := fun r₀ =>
    congrArg (Verity.Game.Tape.out (V.game θ F c) σ) (hFr ω r₀)
  rw [hFx r', hFx r]
  rw [e r] at hacc hgood hs
  rw [e r', e r, before_shown V A x θ F c σ L Rd Acc Good hShow ω r r' hacc hgood i hs]

/-- **`FixedBefore` where the value is read off the log before the round**: at a frame whose coin map is the locator
`L`'s, where a play that opens a value for round `i` is one accepted whose log the check read is the device's, and
what a play of the frame fixes for round `i` is a reading `g` of its tape and its log before round `i`. -/
theorem fixedBefore_of_before (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Fr : Frame V A x θ F c σ)
    (hFr : ∀ ω r, Fr.place ω r = CoinMap.place V θ F c σ L r ω) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L) {Φ : Type}
    (Fx : Verity.Exchange → ℕ → Option Φ) (Opn : Φ → Verity.Exchange → ℕ → Prop) (i : ℕ)
    (hOpn : ∀ φ o, Opn φ o i → Acc o ∧ Good o)
    (g : TapeOf V θ F c σ → List Verity.Release × Option (Verity.Frame × ByteArray) → Option Φ)
    (hFx : ∀ ω r₀, Fx (Fr.out ω r₀) i = g ω (CoinMap.before L i [] (Fr.out ω r₀).log)) :
    FixedBefore Fr Rd Fx Opn i :=
  fixedBefore_of_eq Fr Rd Acc Good Fx Opn i hOpn fun ω r r' hacc hgood hs =>
    fixed_of_before V A x θ F c σ L Fr hFr Rd Acc Good hShow Fx i (g ω) ω (hFx ω) r r' hacc hgood hs

/-- **`FixedBefore` at digests read off the log before the round**: at a frame whose coin map is the locator `L`'s,
where the digests a play of the frame fixes for round `i` are a reading `g` of its tape and its log before round `i`,
and two accepted plays of one tape whose logs are the device's, the second at run coins that agree with the coins the
first's history before round `i` shows, read the same context. -/
theorem fixedBefore_strings_before (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Fr : Frame V A x θ F c σ)
    (hFr : ∀ ω r, Fr.place ω r = CoinMap.place V θ F c σ L r ω) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L) {κ : Type}
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) (i : ℕ)
    (g : TapeOf V θ F c σ → List Verity.Release × Option (Verity.Frame × ByteArray) → Option (List Msg))
    (hDg : ∀ ω r₀, Dg (Fr.out ω r₀) i = g ω (CoinMap.before L i [] (Fr.out ω r₀).log))
    (hctx : ∀ (ω : TapeOf V θ F c σ) (r r' : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
      Acc (Fr.out ω r') → Good (Fr.out ω r') → Shows (playHist A x r (playMsgs Rd (Fr.out ω r)) i) r' →
      ctx (Fr.out ω r') i = ctx (Fr.out ω r) i) :
    FixedBefore Fr Rd (stringsFx Dg ctx) (stringsOpn Acc Good raw ctx) i :=
  fixedBefore_strings Fr Rd Acc Good Dg raw ctx i fun ω r r' ha hg ha' hg' hs =>
    ⟨fixed_of_before V A x θ F c σ L Fr hFr Rd Acc Good hShow Dg i (g ω) ω (hDg ω) r r' ha hg hs,
      hctx ω r r' ha hg ha' hg' hs⟩

end Verity.SampledTwo.Messages

namespace Verity.SampledTwo.SessionCoins

/-- **The log before a session's round ends at the release put on it**: on a play the form accepts whose log the check
read is the device's, where the locator `L` agrees with the session-round locator at the sessions' rounds and puts the
release `rl` on the session's round `i`, the play's log before round `i` is the releases before `rl`, with `rl`'s
frame and bytes. A release before `rl` is put on no round at least `i`: a session's round it is put on is below `i`
by `LogRising`, and any other round is at most the statements'. -/
theorem before_session {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {tab : ℕ → String} {L : CoinMap.Locate}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) (hRise : LogRising A C Good tab) {o : Exchange}
    (hacc : FormAccepts C o) (hgood : Good o) {q rest : List Release} {rl : Release} {i : ℕ}
    (hlog : o.log = q ++ rl :: rest) (hi : CoinMap.locAt L q rl.frame rl.answers = some i)
    (hg : groupOf A i = .sessions) :
    CoinMap.before L i [] o.log = (q, some (rl.frame, rl.answers)) := by
  have hsi := stage_lt_of_sessions hg
  have h2 := (hL q rl.frame rl.answers i hg).1 hi
  have hq : ∀ a r b, q = a ++ r :: b → ¬ ∃ j, CoinMap.locAt L a r.frame r.answers = some j ∧ i ≤ j := by
    intro a r b hqe hex
    obtain ⟨j, hj, hij⟩ := hex
    have hgj : groupOf A j = .sessions := sessions_of_lt (by omega)
    have h1 := (hL a r.frame r.answers j hgj).1 hj
    have hsplit : o.log = a ++ r :: (b ++ rl :: rest) := by
      rw [hlog, hqe]
      simp
    have h2' : locOf A tab (a ++ r :: b) rl.frame = some i := by
      rw [← hqe]
      exact h2
    have hlt : j < i := hRise o hacc hgood a r b rl rest hsplit j i h1 h2'
    omega
  rw [hlog]
  exact CoinMap.before_first L hq hi (Nat.le_refl i)

end Verity.SampledTwo.SessionCoins
