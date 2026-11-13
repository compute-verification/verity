module

public import Proofs.Properties.Verity.SampledTwo.PrefixFixed
public import Proofs.Properties.Verity.SampledTwo.Reached
public import Proofs.Properties.Verity.SampledTwo.Fits
public import Proofs.Properties.Verity.SampledTwo.OfRun
public import Proofs.Properties.Verity.SampledTwo.ProjectionOf

/-!
# A session's messages before its last are coin requests, on an accepted play

`Messages.session_prefix_fixed` (`PrefixFixed.lean`) fixes the play up to a session's round's release by the coins the
history before that round shows, where each round a release is put on answers with its coin. At the form's reading that
is `Messages.SessionMsgsAsk`: a session's slot a release is put on is one whose session's statement sets up and whose
session's messages up to it are of tag `0`. This module proves it from the accept of the run of record, but for one
fact about the locator.

* **The verifier's step** (`session_step_asks`, `session_asks`): at a session's round the verifier answers with a
  message, where no shorter prefix of the session's messages is decided and the messages so far are fewer than the
  session's statement has rounds, Flock's step on the session's messages is a coin request. A refusal ends the run,
  and Flock accepts only as many messages as the statement has rounds (`FlockDeploy.next_acc_length_sha512`). So
  where the verifier answered every round of a session before its last with a message, each is a coin request.
* **An accepted run** (`run_session_asks`): an accepted run of SPOT[Flock] has all its rounds' messages, and every
  round before its last answered a message. So, where each session statement has its template's count of rounds, the
  statements' message sets each session up and each session's message before its last is of tag `0`.
* **At the form** (`sessionAsks_form`): on a play the form accepts and reads at the statement `x`, whose coins as the
  check read them agree with some run's coins (`ReadAt`), some accepted run sends the messages the check read
  (`proverAt`, from `PlayAccepts` and `SessionMsgs`). So the session reading's statements set each session up and its
  message at each slot before a session's last is of tag `0`.
* **`SessionMsgsAsk`** (`sessionMsgsAsk_form`), on the plays `ReadAt` holds of, where the locator puts no release on
  a session's last slot (`LastSlotFree`); and `Messages.session_prefix_fixed` at the form's reading and the run's
  locator with it (`session_prefix_fixed_form`).

`ReadAt` costs the messages' piece nothing: the assembly asks the messages fixed only where the coins the check read
agree with the run's (`MessagesFixedAgreed`), and there the frame's play is read at `x` (`StmtAt`), so
`messagesFixedAgreed_of_readAt` takes the messages fixed on the plays `ReadAt` holds of.

What this does not prove: `LastSlotFree`. The session-round locator puts the `k`-th opening of a session's rep 1
stream on slot `R + 1 + k` at `R` rounds a rep only for `k < R`, so it puts no release on the last slot, `2 R + 1`, of
a session of at least two rounds (`SessionCoins.locOf_ne_last`). `lastSlotFree_run` states it of the session-round
locator alone, and `LastSlot.lean` proves it from the sessions' rounds. The premises `PlayAccepts`, `SessionMsgs` and
`RoundsHeldAt` are the join's.
-/

@[expose] public section

namespace Verity.SampledTwo.Messages

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The verifier's step -/

/-- **A session's round the verifier answers with a message is a coin request**, where no shorter prefix of the
session's messages is decided and the messages so far are not as many as the session's statement has rounds: a
refusal ends the run, and Flock accepts only as many messages as its statement has rounds. -/
theorem session_step_asks (h : SpotFlock.Held) (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) {t : ℕ}
    (has : h.stage + 2 < as.length) (hcs : cs.length = as.length) (hslot : SpotFlock.slotOf h as.length = some t)
    (hund : ∀ j, 0 < j → j < as.length - SpotFlock.off h t →
      SpotFlock.decidedAt (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t) (as.drop (SpotFlock.off h t))
        (cs.drop (SpotFlock.off h t)) j = false)
    (hlen : as.length - SpotFlock.off h t ≠
      (FlockDeploy.V true).k (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t))
    {m : Msg} (hm : ((SpotFlock.next h x as cs).eval FlockDeploy.sha512).1 = .msg m) :
    ∃ c, FlockDeploy.act true (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t) (as.drop (SpotFlock.off h t))
      (cs.drop (SpotFlock.off h t)) = .coin c := by
  rw [SpotFlock.next_long h x as cs has hcs, hslot] at hm
  dsimp only at hm
  rw [SpotFlock.sessStep_eval h _ t _ as cs hund] at hm
  dsimp only at hm
  cases ha : FlockDeploy.act true (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t) (as.drop (SpotFlock.off h t))
      (cs.drop (SpotFlock.off h t)) with
  | coin c => exact ⟨c, rfl⟩
  | rej =>
    rw [FlockDeploy.next_rej FlockDeploy.sha512 ha] at hm
    have hm' : (Out.rej : Out) = .msg m := hm
    cases hm'
  | decide s spec sts link c₁ mids f =>
    exfalso
    cases ho : ((FlockDeploy.next true (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t)
        (as.drop (SpotFlock.off h t)) (cs.drop (SpotFlock.off h t))).eval FlockDeploy.sha512).1 with
    | msg m' =>
      simp only [FlockDeploy.next, ha] at ho
      exact Claims.Bridge.outOf_ne_msg _ _ _ ho
    | acc =>
      refine hlen ?_
      have hd := FlockDeploy.next_acc_length_sha512 true
        (by rw [List.length_drop, List.length_drop, hcs]) ho
      rw [List.length_drop] at hd
      exact hd
    | rej =>
      rw [ho] at hm
      have hm' : (Out.rej : Out) = .msg m := hm
      cases hm'

/-- **Each round of a session before its last is a coin request**, where the verifier answered each of them with a
message and the session's statement has its template's count of rounds: by `session_step_asks`, round by round, the
rounds before each being coin requests and so not decided. -/
theorem session_asks (h : SpotFlock.Held) (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) {t : ℕ}
    (ht : t < h.ms.length) (hcs : cs.length = as.length)
    (hk : (FlockDeploy.V true).k (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t) = SpotFlock.k₀ h t)
    (hmsg : ∀ n, SpotFlock.off h t < n → n < SpotFlock.off h t + SpotFlock.k₀ h t → n ≤ as.length →
      ∃ m, ((SpotFlock.next h x (as.take n) (cs.take n)).eval FlockDeploy.sha512).1 = .msg m) :
    ∀ l, l + 1 < SpotFlock.k₀ h t → SpotFlock.off h t + l < as.length →
      ∃ c, FlockDeploy.act true (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t)
        ((as.drop (SpotFlock.off h t)).take (l + 1)) ((cs.drop (SpotFlock.off h t)).take (l + 1)) = .coin c := by
  have key : ∀ N l, l < N → l + 1 < SpotFlock.k₀ h t → SpotFlock.off h t + l < as.length →
      ∃ c, FlockDeploy.act true (SpotFlock.stmtAt h (as.getD (h.stage + 1) []) t)
        ((as.drop (SpotFlock.off h t)).take (l + 1)) ((cs.drop (SpotFlock.off h t)).take (l + 1)) = .coin c := by
    intro N
    induction N with
    | zero => intro l hl; exact absurd hl (Nat.not_lt_zero l)
    | succ N ih =>
      intro l _ hl hlt
      have hoff := SpotFlock.stage_le_off h t
      obtain ⟨n, hn⟩ : ∃ n, n = SpotFlock.off h t + l + 1 := ⟨_, rfl⟩
      have hlen : (as.take n).length = n := by rw [List.length_take]; omega
      have hcn : (cs.take n).length = (as.take n).length := by rw [List.length_take, List.length_take, hcs]
      obtain ⟨m, hm⟩ := hmsg n (by omega) (by omega) (by omega)
      have hgetD : (as.take n).getD (h.stage + 1) [] = as.getD (h.stage + 1) [] := by
        rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt (by omega)]
      have hdrop : ∀ (β : Type) (l' : List β),
          (l'.take n).drop (SpotFlock.off h t) = (l'.drop (SpotFlock.off h t)).take (l + 1) := by
        intro β l'
        rw [List.drop_take, show n - SpotFlock.off h t = l + 1 by omega]
      have hund : ∀ j, 0 < j → j < (as.take n).length - SpotFlock.off h t →
          SpotFlock.decidedAt (SpotFlock.stmtAt h ((as.take n).getD (h.stage + 1) []) t)
            ((as.take n).drop (SpotFlock.off h t)) ((cs.take n).drop (SpotFlock.off h t)) j = false := by
        intro j hj0 hj
        rw [hlen] at hj
        rw [hgetD, hdrop, hdrop, decidedAt_false_iff, List.take_take, List.take_take,
          Nat.min_eq_left (show j ≤ l + 1 by omega)]
        obtain ⟨c, hc⟩ := ih (j - 1) (by omega) (by omega) (by omega)
        rw [show j - 1 + 1 = j by omega] at hc
        exact ⟨c, hc⟩
      have hne : (as.take n).length - SpotFlock.off h t ≠
          (FlockDeploy.V true).k (SpotFlock.stmtAt h ((as.take n).getD (h.stage + 1) []) t) := by
        rw [hlen, hgetD, hk]
        omega
      have hres := session_step_asks h x (as.take n) (cs.take n) (t := t) (by rw [hlen]; omega) hcn
        (by rw [hlen]; exact SpotFlock.slotOf_eq h ht (by omega) (by omega)) hund hne hm
      rw [hgetD, hdrop, hdrop] at hres
      exact hres
  exact fun l => key (l + 1) l (Nat.lt_succ_self l)

/-! ## An accepted run -/

/-- **In an accepted run of SPOT[Flock], the statements' message sets each session up and each session's message
before its last is of tag `0`**, where the session's statement, read off the statements' message, has its template's
count of rounds: the run has all its rounds' messages (`length_of_acc`), every round before the last answered a
message (`run_msg_lt`), the verifier's on the run's messages and coins so far (`run_answer`), so each round of the
session before its last is a coin request (`session_asks`), which is a setup and a last message of tag `0`
(`act_coin_iff`). -/
theorem run_session_asks (h : SpotFlock.Held) {Ω : Type} (P : Claims.Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((SpotFlock.V h).k x) → FlockDeploy.Coin)
    (hacc : (Claims.run FlockDeploy.sha512 P (SpotFlock.V h) x ω r).out = .acc) {t : ℕ} (ht : t < h.ms.length)
    (hk : ∀ m₂, msgAt (Claims.run FlockDeploy.sha512 P (SpotFlock.V h) x ω r) (h.stage + 1) = some m₂ →
      (FlockDeploy.V true).k (SpotFlock.stmtAt h m₂ t) = SpotFlock.k₀ h t)
    {l : ℕ} (hl : l + 1 < SpotFlock.k₀ h t) :
    ∃ m₂ u, msgAt (Claims.run FlockDeploy.sha512 P (SpotFlock.V h) x ω r) (h.stage + 1) = some m₂ ∧
      (FlockDeploy.setupOf true (SpotFlock.stmtAt h m₂ t)).isSome = true ∧
      msgAt (Claims.run FlockDeploy.sha512 P (SpotFlock.V h) x ω r) (SpotFlock.off h t + l) = some (0 :: u) := by
  have hK := length_of_acc h FlockDeploy.sha512 P x ω r hacc
  have hoff := SpotFlock.stage_le_off h t
  have hmono : SpotFlock.off h t + SpotFlock.k₀ h t ≤ SpotFlock.off h h.ms.length :=
    Claims.Sampling.sessOff_add_le (h.stage + 2) (SpotFlock.k₀ h) ht
  obtain ⟨T, hT⟩ : ∃ T, T = Claims.run FlockDeploy.sha512 P (SpotFlock.V h) x ω r := ⟨_, rfl⟩
  rw [← hT] at hK hk ⊢
  obtain ⟨e₂, he₂⟩ : ∃ e, T.msgs[h.stage + 1]? = some e := ⟨_, List.getElem?_eq_getElem (by omega)⟩
  have hm₂ : msgAt T (h.stage + 1) = some e₂.1 := by
    unfold msgAt
    rw [he₂]
    rfl
  have hgetD : (T.msgs.map Prod.fst).getD (h.stage + 1) [] = e₂.1 := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_map, he₂]
    rfl
  have hcs : ((List.range T.msgs.length).map (coinSeq r)).length = (T.msgs.map Prod.fst).length := by
    rw [List.length_map, List.length_map, List.length_range]
  have hmsg : ∀ n, SpotFlock.off h t < n → n < SpotFlock.off h t + SpotFlock.k₀ h t →
      n ≤ (T.msgs.map Prod.fst).length →
      ∃ m, ((SpotFlock.next h x ((T.msgs.map Prod.fst).take n)
        (((List.range T.msgs.length).map (coinSeq r)).take n)).eval FlockDeploy.sha512).1 = .msg m := by
    intro n h1 h2 h3
    rw [List.length_map] at h3
    obtain ⟨j, rfl⟩ : ∃ j, n = j + 1 := ⟨n - 1, by omega⟩
    have hlt : j + 1 < (Claims.run FlockDeploy.sha512 P (SpotFlock.V h) x ω r).msgs.length := by
      rw [← hT]
      omega
    obtain ⟨e, mm, he, hmm⟩ := run_msg_lt FlockDeploy.sha512 P (SpotFlock.V h) x ω r hlt
    obtain ⟨e', he', hans⟩ := run_answer P (SpotFlock.V h) x ω r (Nat.lt_of_succ_lt hlt)
    rw [he] at he'
    obtain rfl := Option.some.inj he'
    rw [← hT, V_next, List.map_take, ← take_range_map (coinSeq r) h3] at hans
    exact ⟨mm, hans.symm.trans hmm⟩
  have hk' : (FlockDeploy.V true).k (SpotFlock.stmtAt h ((T.msgs.map Prod.fst).getD (h.stage + 1) []) t) =
      SpotFlock.k₀ h t := by
    rw [hgetD]
    exact hk _ hm₂
  have hlt : SpotFlock.off h t + l < (T.msgs.map Prod.fst).length := by
    rw [List.length_map]
    omega
  obtain ⟨c, hc⟩ := session_asks h x (T.msgs.map Prod.fst) ((List.range T.msgs.length).map (coinSeq r)) ht hcs
    hk' hmsg l hl hlt
  rw [hgetD] at hc
  obtain ⟨hset, ⟨u, hu⟩, -⟩ := act_coin_iff.1 ⟨c, hc⟩
  refine ⟨e₂.1, u, hm₂, hset, ?_⟩
  rw [List.getLast?_eq_getElem?, List.length_take, List.length_drop, Nat.min_eq_left (by omega),
    Nat.add_sub_cancel, List.getElem?_take_of_lt (Nat.lt_succ_self l), List.getElem?_drop,
    List.getElem?_map] at hu
  exact hu

/-! ## At the form -/

/-- The form's reading reads the session reading's statements at the statements' round. -/
theorem formMsg_stmt (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) :
    (formReading A C S dev).msg o (A.h.stage + 1) = S.stmts o := by
  have hpos : 0 < A.h.stage := Nat.succ_pos _
  simp only [formReading, ite_eq_right (show ¬ A.h.stage + 1 = 0 by omega),
    ite_eq_right (show ¬ A.h.stage + 1 = 1 by omega), ite_eq_right (show ¬ A.h.stage + 1 ≤ A.h.stage by omega),
    ↓reduceIte]

/-- The form's reading reads the session reading's message of template `t`'s session at that session's rounds. -/
theorem formMsg_session (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (o : Verity.Exchange) {t l : ℕ}
    (ht : t < A.h.ms.length) (h1 : SpotFlock.off A.h t ≤ l) (h2 : l < SpotFlock.off A.h t + SpotFlock.k₀ A.h t) :
    (formReading A C S dev).msg o l = S.msg o t (l - SpotFlock.off A.h t) := by
  have hso := SpotFlock.stage_le_off A.h t
  have hslot : SpotFlock.slotOf A.h (l + 1) = some t := SpotFlock.slotOf_eq A.h ht (by omega) (by omega)
  simp only [formReading, ite_eq_right (show ¬ l = 0 by omega), ite_eq_right (show ¬ l = 1 by omega),
    ite_eq_right (show ¬ l ≤ A.h.stage by omega), ite_eq_right (show ¬ l = A.h.stage + 1 by omega), hslot]

/-- **The play is read at the statement `x`, and the coins the check read of it agree with some run's coins.** -/
structure ReadAt (A : Live₂) (x : Msg) (Rd : Reading A) (o : Verity.Exchange) : Prop where
  stmt : Rd.stmt o = some x
  agrees : ∃ r : Coins A x, ReadAgrees A x Rd o r

/-- **On an accepted play, the session reading's statements set each session up and its message at each slot before
a session's last is of tag `0`**, where the play is read at `x` and the coins the check read agree with the run
coins `r`: some accepted run on `r` sends the messages the check read (`proverAt`), each session statement has its
template's count of rounds (`RoundsHeldAt`), and `run_session_asks` reads the tags off that run. -/
theorem sessionAsks_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P') (hrounds : RoundsHeldAt A C S)
    {o : Verity.Exchange} (hA : FormAccepts C o) (hs : (formReading A C S dev).stmt o = some x) {r : Coins A x}
    (hr : ReadAgrees A x (formReading A C S dev) o r) {t l : ℕ} (ht : t < A.h.ms.length)
    (hl : l + 1 < SpotFlock.k₀ A.h t) :
    (FlockDeploy.setupOf true (SpotFlock.stmtAt A.h ((S.stmts o).getD []) t)).isSome = true ∧
      ∃ u, (S.msg o t l).getD [] = 0 :: u := by
  obtain ⟨Ω, P, ω, hacc, hm⟩ :=
    proverAt A x C S dev P' (coinsCover A.h) (sourcesFit A C) hplay hsess (readsOnly A.h) o hA hs r hr
  have hoff := SpotFlock.stage_le_off A.h t
  have hmono : SpotFlock.off A.h t + SpotFlock.k₀ A.h t ≤ SpotFlock.off A.h A.h.ms.length :=
    Claims.Sampling.sessOff_add_le (A.h.stage + 2) (SpotFlock.k₀ A.h) ht
  have hkx : A.L.V.k x = SpotFlock.off A.h A.h.ms.length + 1 := rfl
  have hi₂ : A.h.stage + 1 < A.L.V.k x := by
    rw [hkx]
    omega
  have hil : SpotFlock.off A.h t + l < A.L.V.k x := by
    rw [hkx]
    omega
  have hstmt : ∀ m₂, msgAt (Claims.run FlockDeploy.sha512 P (SpotFlock.V A.h) x ω r) (A.h.stage + 1) = some m₂ →
      S.stmts o = some m₂ := fun m₂ h2 =>
    (formMsg_stmt A C S dev o).symm.trans ((hm (A.h.stage + 1) hi₂).trans h2)
  obtain ⟨m₂, u, h2, hset, hu⟩ := run_session_asks A.h P x ω r hacc ht
    (fun m₂ h2 => hrounds o m₂ hA (hstmt m₂ h2) t ht) hl
  have hfs := formMsg_session A C S dev o ht (Nat.le_add_right (SpotFlock.off A.h t) l)
    (show SpotFlock.off A.h t + l < SpotFlock.off A.h t + SpotFlock.k₀ A.h t by omega)
  rw [Nat.add_sub_cancel_left] at hfs
  have hmsg : S.msg o t l = some (0 :: u) := hfs.symm.trans ((hm (SpotFlock.off A.h t + l) hil).trans hu)
  refine ⟨?_, u, ?_⟩
  · rw [hstmt m₂ h2]
    exact hset
  · rw [hmsg, Option.getD_some]

/-- **No release is put on a session's last slot**: on a play accepted (`Acc`) whose log the check read is the
device's (`Good`), the locator `L` puts no release of the play's log on the last slot of a template's session, the
slot the session is decided on. -/
def LastSlotFree (A : Live₂) (Acc Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o, Acc o → Good o → ∀ a rl b, o.log = a ++ rl :: b → ∀ t, t < A.h.ms.length →
    CoinMap.locAt L a rl.frame rl.answers ≠ some (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1))

/-- **`SessionMsgsAsk` at the form**, on the plays read at `x` whose coins as the check read them agree with some
run's (`ReadAt`): a slot a release is put on is before the session's last (`LastSlotFree`), and the session's
statement sets up and its messages up to that slot are of tag `0` (`sessionAsks_form`). -/
theorem sessionMsgsAsk_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) (hplay : PlayAccepts A C S P')
    (hsess : SessionMsgs A C S P') (hrounds : RoundsHeldAt A C S) (hlast : LastSlotFree A (FormAccepts C) Good L) :
    SessionMsgsAsk A S (FormAccepts C) (fun o => Good o ∧ ReadAt A x (formReading A C S dev) o) L := by
  intro o hacc hgood a rl b hlog t k ht hk hloc
  obtain ⟨hg, hs, r, hr⟩ := hgood
  have hk' : k + 1 < SpotFlock.k₀ A.h t := by
    by_contra hc
    have hke : k = SpotFlock.k₀ A.h t - 1 := by omega
    rw [hke] at hloc
    exact hlast o hacc hg a rl b hlog t ht hloc
  exact ⟨(sessionAsks_form A x C S dev P' hplay hsess hrounds hacc hs hr ht hk').1, fun l hl =>
    (sessionAsks_form A x C S dev P' hplay hsess hrounds hacc hs hr ht
      (show l + 1 < SpotFlock.k₀ A.h t by omega)).2⟩

/-- **`LastSlotFree` at the run's locator, from the session-round locator**: the run's locator agrees with it at the
sessions' rounds (`SessionCoins.locates_run`). -/
theorem lastSlotFree_run (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (Ls : CoinMap.Locate)
    (tab : ℕ → String)
    (h : ∀ o, FormAccepts C o → Good o → ∀ a rl b, o.log = a ++ rl :: b → ∀ t, t < A.h.ms.length →
      SessionCoins.locOf A tab a rl.frame ≠ some (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1))) :
    LastSlotFree A (FormAccepts C) Good (SessionCoins.runLocate A Ls tab) := by
  intro o hacc hgood a rl b hlog t ht hloc
  have hoff := SpotFlock.stage_le_off A.h t
  have hn0 : ¬ SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) = 0 := by omega
  have hn1 : ¬ SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) ≤ A.h.stage := by omega
  have hn2 : ¬ SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) = A.h.stage + 1 := by omega
  have hg : groupOf A (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) = .sessions := by
    unfold groupOf
    rw [ite_eq_right hn0, ite_eq_right hn1, ite_eq_right hn2]
  exact h o hacc hgood a rl b hlog t ht ((SessionCoins.locates_run A Ls tab a rl.frame rl.answers _ hg).1 hloc)

/-- **The play up to a session's round's release is fixed by the coins the history before that round shows, at the
form's reading and the run's locator**: `session_prefix_fixed` with `ShowsPlaced` from `sessionMsgsAsk_form`, on a
play read at `x` whose coins as the check read them agree with some run's (`ReadAt`). The rounds rise along the
play's log (`LogRising`), the run of record's accept and its sessions' messages are the join's (`PlayAccepts`,
`SessionMsgs`, `RoundsHeldAt`), and no release is put on a session's last slot (`LastSlotFree`). -/
theorem session_prefix_fixed_form (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (Good : Verity.Exchange → Prop) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hRise : SessionCoins.LogRising A C Good tab) (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P')
    (hrounds : RoundsHeldAt A C S)
    (hlast : LastSlotFree A (FormAccepts C) Good (SessionCoins.runLocate A Ls tab))
    (ω : TapeOf V θ F c σ) (r r' : Coins A x)
    (hacc : FormAccepts C
      (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ (SessionCoins.runLocate A Ls tab) r ω)))
    (hgood : Good (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ (SessionCoins.runLocate A Ls tab) r ω)))
    (hread : ReadAt A x (formReading A C S dev)
      (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ (SessionCoins.runLocate A Ls tab) r ω)))
    {q rest : List Verity.Release} {rl : Verity.Release} {i : ℕ}
    (h : (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ (SessionCoins.runLocate A Ls tab) r ω)).log =
      q ++ rl :: rest)
    (hi : CoinMap.locAt (SessionCoins.runLocate A Ls tab) q rl.frame rl.answers = some i)
    (hg : groupOf A i = .sessions)
    (hs : Shows (playHist A x r (playMsgs (formReading A C S dev)
      (Verity.Game.Tape.out _ σ (CoinMap.place V θ F c σ (SessionCoins.runLocate A Ls tab) r ω))) i) r') :
    ∃ cs rest', (Verity.Game.Tape.out _ σ
      (CoinMap.place V θ F c σ (SessionCoins.runLocate A Ls tab) r' ω)).log =
        q ++ ⟨rl.frame, rl.answers, cs⟩ :: rest' :=
  session_prefix_fixed V A x θ F c σ (SessionCoins.runLocate A Ls tab) C
    (fun o => Good o ∧ ReadAt A x (formReading A C S dev) o) tab (formReading A C S dev)
    (SessionCoins.locates_run A Ls tab) (fun o hA hG => hRise o hA hG.1)
    (showsPlaced_form A x C S dev _ Ls tab
      (sessionMsgsAsk_form A x C S dev P' Good _ hplay hsess hrounds hlast))
    ω r r' hacc ⟨hgood, hread⟩ h hi hg hs

/-! ## The messages' piece on the plays `ReadAt` holds of -/

/-- **The messages fixed where the coins agree, from the messages fixed on the plays `ReadAt` holds of**: where the
frame's accepted plays are read at `x` (`StmtAt`), on agreeing coins a run whose messages are the check's at every
round it reaches has all the verifier's messages, and a message the check reads differs from the run's at a round the
run reaches on plays at most `miss` likely among those `ReadAt` holds of, `MessagesFixedAgreed` holds at `miss`. -/
theorem messagesFixedAgreed_of_readAt {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} {Fr : Frame V A x θ F c σ} {Rd : Reading A}
    {Acc Good : Verity.Exchange → Prop} {miss : ℝ≥0∞} (hst : StmtAt V A x θ F c σ Fr Rd Acc)
    (hR : ∀ (ω : TapeOf V θ F c σ) (r : Coins A x), Acc (Fr.out ω r) → Good (Fr.out ω r) →
      ReadAgrees A x Rd (Fr.out ω r) r →
      (∀ i < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length,
        Rd.msg (Fr.out ω r) i = msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r) i) →
      A.L.V.k x ≤ (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).msgs.length)
    (h : MessagesReached V A x θ F c σ Fr Rd Acc (fun o => Good o ∧ ReadAt A x Rd o) miss) :
    MessagesFixedAgreed V A x θ F c σ Fr Rd Acc Good miss := by
  refine le_trans (ENNReal.ofReal_le_ofReal (Claims.Pr_mono fun s hs => ?_)) h
  obtain ⟨hA, hG, hC, i, hi, hne⟩ := hs
  refine ⟨hA, ⟨hG, ⟨hst s.1 s.2 hA, s.2, hC⟩⟩, ?_⟩
  by_contra hnone
  have hall : ∀ j < (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).msgs.length,
      Rd.msg (Fr.out s.1 s.2) j = msgAt (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2) j :=
    fun j hj => Classical.not_not.1 fun hne' => hnone ⟨j, hj, hne'⟩
  exact hne (hall i (lt_of_lt_of_le hi (hR s.1 s.2 hA hG hC hall)))

end Verity.SampledTwo.Messages
