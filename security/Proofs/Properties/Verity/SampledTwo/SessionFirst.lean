module

public import Proofs.Properties.Verity.SampledTwo.SessionMiddle

/-!
# A session's first round

The first round of session `t`, round `SpotFlock.off A.h t`, is the link's. The release the locator puts on it opens
round 0 of the session's tree's `TREE_SESSION` stream, and answers the digest the session's record gives that round
(`SessionCoins.LogAnswers`): the SHA-512 of the link's `root_F` and each table's `root_B` in table-name order
(`Spot.Challenger.linkAfter`). This module names those bytes and shows `Messages.ShowsAt` at the round.

* `linkBytes`, `linkAfter_hash`: the bytes `Spot.Challenger.linkAfter` is the SHA-512 of.
* `linkShown`, `recordDigest_shown`: those bytes of a session's record, and the digest the record gives a round of
  `TREE_SESSION` as their SHA-512.
* `slotStream_zero`: a session's first round is round 0 of `TREE_SESSION`.
* `linkShownAt`, `linkShownAt_of`: those bytes of the `t`-th of the sessions' records the check read.
* `showsAt_session_first`: `Messages.ShowsAt` at a session's first round of the form's reading, with the digest the
  bytes the release that ends the log before the round answers and the string `linkShownAt`.

**Not established here**: `Messages.Explains` at a session's first round. The message there is the link's
(`FlockDeploy.linkBody`, of the live session's `rootB` and `publics`), and the string shown is `root_F` and each
`root_B`: that two accepted plays showing the same string after the same log read the same link message is not shown.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Claims
open Lean (Json)
open Spot Spot.Challenger

/-! ## The bytes a record's link is opened after -/

theorem foldl_congr_fun {α β : Type} {f g : β → α → β} (h : ∀ b a, f b a = g b a) (xs : Array α) (init : β) :
    xs.foldl f init = xs.foldl g init := by
  have e : f = g := funext fun b => funext fun a => h b a
  rw [e]

/-- **The bytes `Spot.Challenger.linkAfter` is the SHA-512 of**: `root_F`, then each table's `root_B` in table-name
order. -/
def linkBytes (rootF : ByteArray) (rootB : Array (String × ByteArray)) : ByteArray :=
  (rootB.qsort (·.1 < ·.1)).foldl (fun acc kv => acc ++ kv.2) rootF

theorem linkAfter_hash (rootF : ByteArray) (rootB : Array (String × ByteArray)) :
    linkAfter rootF rootB = Flock.Sha512.hash (linkBytes rootF rootB) := by
  unfold linkAfter linkBytes
  exact congrArg Flock.Sha512.hash (foldl_congr_fun (fun acc kv => by cases kv; rfl) _ _)

/-- **The bytes a session's record's link points are opened after the SHA-512 of**: the `root_F` its link names, then
each table's `root_B` in table-name order; none where the record names no `root_F`. -/
def linkShown (record : Json) : Option ByteArray :=
  (linkRootF record).map fun rootF => linkBytes rootF (linkRootB record)

/-- The digest a record gives a round of `TREE_SESSION` is the SHA-512 of `linkShown`. -/
theorem recordDigest_shown (record : Json) (k : ℕ) :
    recordDigest record TREE_SESSION k = (linkShown record).map Flock.Sha512.hash := by
  unfold recordDigest linkShown
  rw [ite_eq_left (beq_self_eq_true _)]
  cases linkRootF record with
  | none => rfl
  | some rootF => exact congrArg some (linkAfter_hash rootF (linkRootB record))

/-- A session's first round is round 0 of `TREE_SESSION`. -/
theorem slotStream_zero (tab : String) (R : ℕ) : slotStream tab R 0 = (TREE_SESSION, 0) := by
  unfold slotStream
  rw [ite_eq_left rfl]

/-! ## The sessions' records the check read -/

/-- **The string a play shows at template `t`'s session's first round**: `linkShown` of the `t`-th of the sessions'
records the check read (`sessionRecords`). -/
def linkShownAt (C : CaseOf) (o : Exchange) (t : ℕ) : Option (List Msg) :=
  (sessionRecords C o).bind fun recs => recs[t]?.bind fun rc => (linkShown rc).map fun b => [b.data.toList]

theorem linkShownAt_of {C : CaseOf} {o : Exchange} {t : ℕ} {recs : List Json} {rc : Json} {b : ByteArray}
    (hrecs : sessionRecords C o = some recs) (hrc : recs[t]? = some rc) (hb : linkShown rc = some b) :
    linkShownAt C o t = some [b.data.toList] := by
  unfold linkShownAt
  rw [hrecs, Option.bind_some, hrc, Option.bind_some, hb]
  rfl

end Verity.SampledTwo.SessionCoins

namespace Verity.SampledTwo

open Claims

/-! ## `ShowsAt` at a session's first round -/

/-- **A session's first round of the form's reading shows the bytes its link points are opened after**: at the first
round of template `t`'s session, a session of at least two rounds, where the digest a play fixes for the round is the
bytes the release that ends the log before the round answers (`Messages.endAnswers`) and the string it shows is
`SessionCoins.linkShownAt`, an accepted play whose log the check read is the device's shows a string that has its
digest. The log before the round ends at the release on the round (`SessionCoins.before_session`), which answers the
digest the session's record gives round 0 of `TREE_SESSION` (`SessionCoins.LogAnswers`), the SHA-512 of the string
(`SessionCoins.recordDigest_shown`). -/
theorem showsAt_session_first (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (tab : ℕ → String) (hL : SessionCoins.LocatesSessions A tab (CoinMap.locAt L))
    (hRise : SessionCoins.LogRising A C Good tab) (hOpen : SessionCoins.LogOpened A C Good L)
    (hAns : SessionCoins.LogAnswers A C Good L tab) (hAsk : Messages.SessionMsgsAsk A S (FormAccepts C) Good L)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) {t : ℕ} (ht : t < A.h.ms.length)
    (hK : 2 ≤ SpotFlock.k₀ A.h t)
    (hd : ∀ o, FormAccepts C o → Good o → Dg o (SpotFlock.off A.h t) =
      Messages.endAnswers (Messages.logBefore L o (SpotFlock.off A.h t)))
    (hr : ∀ o, FormAccepts C o → Good o → raw o (SpotFlock.off A.h t) = SessionCoins.linkShownAt C o t) :
    Messages.ShowsAt (formReading A C S dev) (FormAccepts C) Good Dg raw (SpotFlock.off A.h t) := by
  intro o hA hG
  have hj : 0 + 1 < SpotFlock.k₀ A.h t := by omega
  obtain ⟨q, rl, rest, hlog, hloc⟩ := hOpen o hA hG t 0 ht hj
  have hb : CoinMap.before L (SpotFlock.off A.h t) [] o.log = (q, some (rl.frame, rl.answers)) :=
    SessionCoins.before_session hL hRise hA hG hlog hloc (SessionCoins.groupOf_off A t 0)
  obtain ⟨recs, rc, hrecs, hrc, hdig⟩ := hAns o hA hG q rl rest hlog t 0 (by omega) hloc
  obtain ⟨u, -, hm⟩ := session_msg_tag A C S dev Good L hOpen hAsk ht hj o hA hG
  have hm' : (formReading A C S dev).msg o (SpotFlock.off A.h t) = some (0 :: u) := hm
  rw [SessionCoins.slotStream_zero] at hdig
  have hdig' : (SessionCoins.linkShown rc).map Flock.Sha512.hash = some rl.answers :=
    (SessionCoins.recordDigest_shown rc 0).symm.trans hdig
  obtain ⟨b, hshown, hbe⟩ := Option.map_eq_some_iff.1 hdig'
  refine ⟨⟨_, hm'⟩, [rl.answers.data.toList], [b.data.toList], ?_, ?_, ?_⟩
  · exact (hd o hA hG).trans (congrArg (Option.map fun fb : Verity.Frame × ByteArray => [fb.2.data.toList])
      (congrArg Prod.snd hb))
  · exact (hr o hA hG).trans (SessionCoins.linkShownAt_of hrecs hrc hshown)
  · rw [← hbe]
    exact congrArg (fun m : Msg => [m]) (Verity.Sampled.sha512Msg_bytes b)

end Verity.SampledTwo
