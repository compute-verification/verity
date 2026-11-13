module

public import Proofs.Properties.Verity.SampledTwo.RoundsRise
public import Proofs.Properties.Verity.SampledTwo.SessionDev
public import Proofs.Properties.Verity.SampledTwo.CoinPrefix
public import Proofs.Properties.Verity.SampledTwo.MessagesShown

/-!
# The play before a session's round is fixed by the coins its history shows

The messages' piece fixes a round's message by the run's coins the play's history shows before that round
(`Messages.Shows`). The coin map writes a run's coins into the play one release at a time, so the log before a
release, and that release's frame and bytes, read only the coins of the rounds the earlier releases continue
(`CoinMap.place_prefix_used`). This module joins the two at a second-stage session's round.

Three facts are needed, and the join is `Messages.session_prefix_fixed`:

* **The rounds rise along the play's log** (`LogRising`, `logRising_of`): `LocatedRising` is about the log the check
  read; the play's log is that log after releases of no coin tree, release by release the same frames
  (`InteriorLogged`), and the session-round locator reads only the frames (`Same`).
* **Every round an earlier release continues is an earlier round** (`used_below`): a release before one put on the
  session's round `i` is put on a session's round below `i` (`LogRising`), or on a round at most the statements',
  which is below every session's.
* **Each of those rounds shows its coin** (`Messages.ShowsPlaced`, a hypothesis here): the history shows a round's
  coin only where the verifier's answer at that round is the coin's bytes (`Messages.shownAt`). `Messages.CoinAnswered`
  says so of one round; `Messages.coinAnswered_stage` proves it at the first and second stages' rounds, where the
  form's reading reads the empty message past round 1 (`Messages.coinAnswered_formStage`).

What this does not prove: `ShowsPlaced` at the sessions' rounds. The statements' round and a session's last round
answer with no coin, so it needs that no release is put on either. The run's locator puts none on the statements'
round (`locAt_run_ne_stmt`); that an accepted play has no release on a session's last round, and that a session's
earlier rounds answer with their coins, are not proved here.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Verity.Sampled.Binding (Read)

/-! ## Lists -/

/-- Two lists related entry by entry split together: at a split of the second, the first splits at the same place. -/
theorem forall₂_split_right {α β : Type} {R : α → β → Prop} {a : List β} {y : β} {rest : List β} :
    ∀ {l : List α}, List.Forall₂ R l (a ++ y :: rest) →
      ∃ la x lrest, l = la ++ x :: lrest ∧ List.Forall₂ R la a ∧ R x y ∧ List.Forall₂ R lrest rest := by
  induction a with
  | nil =>
    intro l h
    rw [List.nil_append] at h
    cases h with
    | cons hx hrest => exact ⟨[], _, _, rfl, List.Forall₂.nil, hx, hrest⟩
  | cons z a ih =>
    intro l h
    rw [List.cons_append] at h
    cases h with
    | cons hw h₁ =>
      obtain ⟨la, x, lrest, rfl, hla, hx, hlr⟩ := ih h₁
      exact ⟨_ :: la, x, lrest, rfl, List.Forall₂.cons hw hla, hx, hlr⟩

/-- Releases of the same frames, one by one, keep what the session-round locator reads. -/
theorem same_append_frames {l l' : List Release}
    (hl : List.Forall₂ (fun r r' : Release => r.frame = r'.frame) l l') :
    ∀ {pre pre' : List Release}, Same pre pre' → Same (pre ++ l) (pre' ++ l') := by
  induction hl with
  | nil =>
    intro pre pre' h
    rw [List.append_nil, List.append_nil]
    exact h
  | @cons r r' l l' hr _ ih =>
    intro pre pre' h
    rw [show pre ++ r :: l = pre ++ [r] ++ l by simp, show pre' ++ r' :: l' = pre' ++ [r'] ++ l' by simp]
    exact ih (h.snoc hr)

/-- A frame the session-round locator puts on a round opens a coin tree. -/
theorem openTree_of_locOf {A : Live₂} {tab : ℕ → String} {pre : List Release} {fr : Verity.Frame} {n : ℕ}
    (h : locOf A tab pre fr = some n) : ∃ i, openTree fr = some i := by
  cases fr with
  | fresh _ => simp [locOf, sessLoc] at h
  | hello _ => simp [locOf, sessLoc] at h
  | opening i s c => exact ⟨i, rfl⟩

/-- A release that opens a coin tree, in a log that begins with releases opening none, is past them. -/
theorem split_past {q l' a rest : List Release} {r : Release} (h : q ++ l' = a ++ r :: rest)
    (hq : ∀ x ∈ q, openTree x.frame = none) {i : ℕ} (hr : openTree r.frame = some i) :
    ∃ a', a = q ++ a' ∧ l' = a' ++ r :: rest := by
  rcases List.append_eq_append_iff.1 h with ⟨a', ha, hl⟩ | ⟨c', hq', hc⟩
  · exact ⟨a', ha, hl⟩
  · cases c' with
    | nil =>
      rw [List.append_nil] at hq'
      rw [List.nil_append] at hc
      exact ⟨[], by rw [List.append_nil, hq'], hc.symm⟩
    | cons y c'' =>
      rw [List.cons_append] at hc
      have hy : r = y := (List.cons.inj hc).1
      have hn : openTree y.frame = none := hq y (by rw [hq']; simp)
      rw [← hy, hr] at hn
      cases hn

/-! ## The rounds rise along the play's log -/

/-- **The session-round locator's rounds rise along the play's log**: on a play the form accepts whose log the check
read is the device's (`Good`), of two releases of the play's log the session-round locator puts on rounds, the later
is on the later round. -/
def LogRising (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ a r b r' c, o.log = a ++ r :: (b ++ r' :: c) →
    ∀ n n', locOf A tab a r.frame = some n → locOf A tab (a ++ r :: b) r'.frame = some n' → n < n'

/-- **`LogRising`, from the log the check read** (`LocatedRising`): the play's log is the read log after releases of no
coin tree, the same frames one by one (`InteriorLogged`), and the session-round locator reads the frames alone
(`sessLoc_congr`). -/
theorem logRising_of {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {tab : ℕ → String}
    (hR : LocatedRising A C tab) (hI : InteriorLogged C Good) (hRd : InteriorReads C) :
    LogRising A C Good tab := by
  intro o hacc hgood a r b r' c hsplit n n' h1 h2
  obtain ⟨l, hl⟩ := hRd o hacc
  obtain ⟨q, l', hlog, hq, hF⟩ := hI o hacc hgood l hl
  have hF' : List.Forall₂ (fun x y : Release => x.frame = y.frame) l l' :=
    (List.forall₂_map_left_iff.1 hF).imp fun _ _ h => (of_ofV2 h).1
  obtain ⟨i, hi⟩ := openTree_of_locOf h1
  rw [hlog] at hsplit
  obtain ⟨a', rfl, hl'⟩ := split_past hsplit (fun x hx => (hq x hx).1) hi
  subst hl'
  obtain ⟨la, x, lrest, rfl, hla, hx, hrest⟩ := forall₂_split_right hF'
  obtain ⟨lb, x', lc, rfl, hlb, hx', -⟩ := forall₂_split_right hrest
  have hxf : x.frame = r.frame := hx
  have hxf' : x'.frame = r'.frame := hx'
  have S0 : Same [] q := by simpa using same_append (pre := []) hq
  have S1 : Same la (q ++ a') := by simpa using same_append_frames hla S0
  have S2 : Same (la ++ [x]) (q ++ a' ++ [r]) := S1.snoc hxf
  have S3 : Same (la ++ [x] ++ lb) (q ++ a' ++ [r] ++ b) := same_append_frames hlb S2
  have e1 : locOf A tab la x.frame = some n := by
    unfold locOf at h1 ⊢
    rw [sessLoc_congr S1, hxf]
    exact h1
  have e2 : locOf A tab (la ++ x :: lb) x'.frame = some n' := by
    unfold locOf at h2 ⊢
    rw [show la ++ x :: lb = la ++ [x] ++ lb by simp, sessLoc_congr S3, hxf']
    rw [show q ++ a' ++ r :: b = q ++ a' ++ [r] ++ b by simp] at h2
    exact h2
  exact hR o hacc _ hl la x lb x' lc rfl n n' e1 e2

/-! ## The rounds the earlier releases continue -/

theorem sessions_of_lt {A : Live₂} {n : ℕ} (h : A.h.stage + 1 < n) : groupOf A n = .sessions := by
  unfold groupOf
  rw [if_neg (show ¬ n = 0 by omega), if_neg (show ¬ n ≤ A.h.stage by omega),
    if_neg (show ¬ n = A.h.stage + 1 by omega)]

/-- A round the log `q` gives elements has a release of `q` put on it. -/
theorem exists_of_usedOf {L : CoinMap.Locate} {j : ℕ} : ∀ (q pre : List Release),
    CoinMap.usedOf L j pre q ≠ 0 →
      ∃ a r b, q = a ++ r :: b ∧ CoinMap.locAt L (pre ++ a) r.frame r.answers = some j
  | [], pre, h => by simp [CoinMap.usedOf] at h
  | rl :: q, pre, h => by
    by_cases hl : CoinMap.locAt L pre rl.frame rl.answers = some j
    · exact ⟨[], rl, q, rfl, by rw [List.append_nil]; exact hl⟩
    · have h' : CoinMap.usedOf L j (pre ++ [rl]) q ≠ 0 := by
        intro e
        apply h
        show (if CoinMap.locAt L pre rl.frame rl.answers = some j then CoinMap.widthOf rl.frame else 0) +
          CoinMap.usedOf L j (pre ++ [rl]) q = 0
        simp only [if_neg hl, e, Nat.add_zero]
      obtain ⟨a, r, b, rfl, hr⟩ := exists_of_usedOf q (pre ++ [rl]) h'
      exact ⟨rl :: a, r, b, rfl, by rw [show pre ++ rl :: a = pre ++ [rl] ++ a by simp]; exact hr⟩

/-- **Every round a release before a session's round continues is an earlier round**: on a play the form accepts
whose log the check read is the device's, where the locator `L` agrees with the session-round locator at the
sessions' rounds and puts the release `rl` on the session's round `i`, every round the releases before `rl` give
elements is below `i`. A session's round below `i` by `LogRising`; any other round is at most the statements'. -/
theorem used_below {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {tab : ℕ → String} {L : CoinMap.Locate}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) (hRise : LogRising A C Good tab) {o : Exchange}
    (hacc : FormAccepts C o) (hgood : Good o) {q rest : List Release} {rl : Release} {i : ℕ}
    (hlog : o.log = q ++ rl :: rest) (hi : CoinMap.locAt L q rl.frame rl.answers = some i)
    (hg : groupOf A i = .sessions) {j : ℕ} (hj : CoinMap.usedOf L j [] q ≠ 0) : j < i := by
  obtain ⟨a, r, b, rfl, hr⟩ := exists_of_usedOf q [] hj
  rw [List.nil_append] at hr
  have hsi := stage_lt_of_sessions hg
  by_cases hgj : groupOf A j = .sessions
  · have h1 := (hL a r.frame r.answers j hgj).1 hr
    have h2 := (hL (a ++ r :: b) rl.frame rl.answers i hg).1 hi
    exact hRise o hacc hgood a r b rl rest (by rw [hlog]; simp) j i h1 h2
  · have hnj : ¬ A.h.stage + 1 < j := fun h => hgj (sessions_of_lt h)
    omega

/-- **The run's locator puts no release on the statements' round**: an opening goes past it (`locOf_past`), any other
frame at most on the second stage's last round. -/
theorem locAt_run_ne_stmt (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String) (pre : List Release)
    (fr : Verity.Frame) (b : ByteArray) :
    CoinMap.locAt (runLocate A Ls tab) pre fr b ≠ some (A.h.stage + 1) := by
  intro h
  cases fr with
  | hello sp => cases (show (none : Option ℕ) = some (A.h.stage + 1) from h)
  | opening i s c =>
    have hp := locOf_past (show locOf A tab pre (.opening i s c) = some (A.h.stage + 1) from h)
    omega
  | fresh l =>
    have h' : (Ls pre (.fresh l) b).filter (fun n => decide (n ≤ A.h.stage)) = some (A.h.stage + 1) := h
    obtain ⟨-, hle⟩ := Option.filter_eq_some_iff.1 h'
    have hd := of_decide_eq_true hle
    omega

end Verity.SampledTwo.SessionCoins

namespace Verity.SampledTwo.Messages

open Claims

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The coins the history shows -/

/-- **The play's history shows a round's coin where the verifier answers with it**: at a round `j` before the
history's end whose answer, at the play's messages and the run's coins, is the round's coin's bytes. -/
theorem shownAt_playHist (A : Live₂) (x : Msg) (r : Coins A x) (ms : ℕ → Msg) {i : ℕ}
    (j : Fin (A.L.V.k x)) (hji : j.val < i)
    (h : msgOf ((A.L.V.next x ((List.range (j.val + 1)).map ms)
      ((List.range (j.val + 1)).map (coinSeq r))).eval FlockDeploy.sha512).1 = (r j).bytes) :
    shownAt (playHist A x r ms i) j.val = some (r j) := by
  unfold shownAt playHist
  rw [List.getElem?_map, List.getElem?_range hji]
  exact (congrArg shownCoin h).trans (shownCoin_of _)

/-- **The log before a release, and the release's frame and bytes, are fixed by the coins a history shows**, where
the history `hist` shows the coin of every round a release before it continues: any coins `r'` that agree with what
`hist` shows give the same log up to the release, and its frame and bytes (`CoinMap.place_prefix_used`). -/
theorem prefix_shown (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (ω : TapeOf V θ F c σ) (r r' : Coins A x)
    (hist : List (Msg × Msg)) {q rest : List Verity.Release} {rl : Verity.Release}
    (h : (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)).log = q ++ rl :: rest)
    (hq : ∀ j : Fin (A.L.V.k x), CoinMap.usedOf L j.val [] q ≠ 0 → shownAt hist j.val = some (r j))
    (hs : Shows hist r') :
    ∃ cs rest', (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r' ω)).log =
      q ++ ⟨rl.frame, rl.answers, cs⟩ :: rest' :=
  CoinMap.place_prefix_used L V θ F c σ r r' ω h fun j hj => (hs j _ (hq j hj)).symm

/-! ## The rounds that answer with their coin -/

/-- **The verifier answers round `j` with the round's coin**, at the messages `ms`, whatever the run's coins are. -/
def CoinAnswered (A : Live₂) (x : Msg) (ms : ℕ → Msg) (j : ℕ) : Prop :=
  ∀ (r : Coins A x) (hj : j < A.L.V.k x),
    msgOf ((A.L.V.next x ((List.range (j + 1)).map ms) ((List.range (j + 1)).map (coinSeq r))).eval
      FlockDeploy.sha512).1 = (r ⟨j, hj⟩).bytes

/-- **The first and second stages' rounds answer with their coin**: rounds 0 and 1, and a later round up to the
second stage's last whose message is empty (`SpotFlock.next_coin`). -/
theorem coinAnswered_stage (A : Live₂) (x : Msg) (ms : ℕ → Msg) {j : ℕ} (hj : j ≤ A.h.stage)
    (hm : j ≤ 1 ∨ ms j = []) : CoinAnswered A x ms j := by
  intro r hjk
  have hl : ((List.range (j + 1)).map ms).length = j + 1 := by simp
  have hne : (List.range (j + 1)).map ms ≠ [] := List.ne_nil_of_length_pos (by omega)
  have hcl : ((List.range (j + 1)).map (coinSeq r)).length = ((List.range (j + 1)).map ms).length := by simp
  have hc : ((List.range (j + 1)).map ms).length ≤ 2 ∨
      ((List.range (j + 1)).map ms).getD (((List.range (j + 1)).map ms).length - 1) [] = [] := by
    rw [hl, Nat.add_sub_cancel, getD_range_map ms (Nat.lt_add_one j)]
    exact hm.imp (fun h => by omega) id
  show msgOf ((SpotFlock.next A.h x ((List.range (j + 1)).map ms)
    ((List.range (j + 1)).map (coinSeq r))).eval FlockDeploy.sha512).1 = (r ⟨j, hjk⟩).bytes
  rw [SpotFlock.next_coin A.h x hne hcl (by rw [hl]; omega), if_pos hc]
  simp only [OC.eval, msgOf]
  rw [hl, Nat.add_sub_cancel, getD_range_map (coinSeq r) (Nat.lt_add_one j), coinSeq_lt r hjk]

/-- The form's reading reads the empty message at the second stage's rounds past round 1. -/
theorem playMsgs_stage (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) {j : ℕ} (h1 : 1 < j)
    (hj : j ≤ A.h.stage) : playMsgs (formReading A C S dev) o j = [] := by
  simp only [playMsgs, formReading, if_neg (show ¬ j = 0 by omega), if_neg (show ¬ j = 1 by omega), if_pos hj,
    Option.getD_some]

/-- **At the form's reading, the first and second stages' rounds answer with their coin.** -/
theorem coinAnswered_formStage (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) {j : ℕ} (hj : j ≤ A.h.stage) :
    CoinAnswered A x (playMsgs (formReading A C S dev) o) j := by
  refine coinAnswered_stage A x _ hj ?_
  by_cases h1 : j ≤ 1
  · exact Or.inl h1
  · exact Or.inr (playMsgs_stage A C S dev o (by omega) hj)

/-- **Every round a release is put on answers with its coin**: on a play accepted (`Acc`) whose log the check read is
the device's (`Good`), at the messages the reading `Rd` reads of it, the verifier answers each round the locator `L`
puts a release of the play's log on with that round's coin. The statements' round and a session's last round answer
with no coin, so this says the locator puts no release of such a play on either. -/
def ShowsPlaced (A : Live₂) (x : Msg) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop) (L : CoinMap.Locate) :
    Prop :=
  ∀ o, Acc o → Good o → ∀ a rl b, o.log = a ++ rl :: b → ∀ j, CoinMap.locAt L a rl.frame rl.answers = some j →
    CoinAnswered A x (playMsgs Rd o) j

/-! ## The join -/

/-- **The play up to a session's round's release is fixed by the coins the history before that round shows.** At the
coin map's play with the run's coins `r`, accepted by the form and with the check's log the device's, let `rl` be a
release the locator puts on the session's round `i`, after the releases `q`. Then at any coins `r'` that agree with
every coin the play's history before round `i` shows, the play's log is `q` and then a release of `rl`'s frame and
bytes. The locator agrees with the session-round locator at the sessions' rounds, the rounds rise along the play's
log (`LogRising`), and each round a release is put on answers with its coin (`ShowsPlaced`). -/
theorem session_prefix_fixed (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (C : CaseOf)
    (Good : Verity.Exchange → Prop) (tab : ℕ → String) (Rd : Reading A)
    (hL : SessionCoins.LocatesSessions A tab (CoinMap.locAt L)) (hRise : SessionCoins.LogRising A C Good tab)
    (hShow : ShowsPlaced A x Rd (FormAccepts C) Good L) (ω : TapeOf V θ F c σ) (r r' : Coins A x)
    (hacc : FormAccepts C (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)))
    (hgood : Good (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)))
    {q rest : List Verity.Release} {rl : Verity.Release} {i : ℕ}
    (h : (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω)).log = q ++ rl :: rest)
    (hi : CoinMap.locAt L q rl.frame rl.answers = some i) (hg : groupOf A i = .sessions)
    (hs : Shows (playHist A x r
      (playMsgs Rd (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r ω))) i) r') :
    ∃ cs rest', (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ L r' ω)).log =
      q ++ ⟨rl.frame, rl.answers, cs⟩ :: rest' := by
  refine prefix_shown V A x θ F c σ L ω r r' _ h (fun j hj => ?_) hs
  have hlt : j.val < i := SessionCoins.used_below hL hRise hacc hgood h hi hg hj
  obtain ⟨a, r₀, b, hqe, hr₀⟩ := SessionCoins.exists_of_usedOf q [] hj
  rw [List.nil_append] at hr₀
  have hans := hShow _ hacc hgood a r₀ (b ++ rl :: rest) (by rw [h, hqe]; simp) j.val hr₀
  exact shownAt_playHist A x r _ j hlt (hans r j.isLt)

end Verity.SampledTwo.Messages
