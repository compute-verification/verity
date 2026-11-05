module

public import Proofs.Flock.ZkAny.RelayTook
public import Proofs.Flock.ZkAny.RelayTaken
public import Proofs.Flock.ZkAny.RelayRank

/-!
# The model takes the coins after a frame round's history, from what the loop took

The model takes the coins after a history (`LeanRelease.answered`) where the transport's check passes it, the frame
took the earlier rounds' coins, the frame's coin step takes the last, and the round after it is a rep's first at the
next rep's start (`LeanRelease.fresh`) or the same rep's next, where the rank stop passes the coin step's state. At a
frame round's slot the loop holds the first three (`LoopModel.TookAt`), and the rank stop it checks at the opening's
label of the round it releases:

* `frameAnswerR_of_fed`, `answeredR_same`, `answeredR_switch`: the coin step takes the coins, at a later round of the
  same rep or at a rep's first round.
* **`answered_of_took`**: so the model takes them, given the next rep's start at a rep's first round and the rank stop
  at a later round of the same rep.
* `LoopModel.TookAt.opens`: where the session holds a coin tree, the transport's check passes the model's history.
* **`rank_of_release`**: where a frame releases a device round at a state alike the coin step's, at the model's
  configuration, in the place of a round that is the opening's first wherever the rank stop fails, the rank stop
  passes (`Flags.rankFails_of_step`).
-/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock.Firewall.Frame (Cfg Masker Place)

namespace LeanRelease

/-- **Where the frame takes the last coins** (`fed`), at a later round of the same rep, the coin step takes them. -/
theorem frameAnswerR_of_fed {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    (hf : L.fed x ρ h = true) (hn : 3 ≤ h.length) (hs : L.rep x h.length = L.rep x (h.length - 1)) :
    ∃ m, L.frameAnswerR x ρ h = .ok m := by
  unfold fed at hf
  rw [decide_eq_false (show ¬ h.length ≤ 2 by omega), Bool.false_or] at hf
  rw [frameAnswerR_same L x ρ hn hs, cfg_eq_cfgAt_pred L x ρ hs]
  cases ha : (L.prog x h).answer (L.cfgAt x ρ (h.length - 1)) (L.coinsOf x h) with
  | error e =>
    rw [ha] at hf
    first
      | cases hf
      | simp [Except.isOk, Except.toBool] at hf
  | ok m => exact ⟨m, rfl⟩

/-- **At a later round of the same rep the coin step takes the coins** the frame takes (`fed`), past the transport's
check and the earlier rounds' coins. -/
theorem answeredR_same {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    (ho : L.opens x ρ h = true) (ht : L.tookEarlier x ρ h = true) (hf : L.fed x ρ h = true) (hn : 3 ≤ h.length)
    (hs : L.rep x h.length = L.rep x (h.length - 1)) : ∃ m, L.answeredR x ρ h = .ok m := by
  rw [L.answeredR_of_opens x ρ h ho ht]
  exact frameAnswerR_of_fed hf hn hs

/-- **At a rep's first round the coin step is the next rep's start**, past the last rep's taking of the last coins
(`fed`). -/
theorem answeredR_switch {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    (ho : L.opens x ρ h = true) (ht : L.tookEarlier x ρ h = true) (hf : L.fed x ρ h = true) (h2 : 2 ≤ h.length)
    (hs : 3 ≤ h.length → L.rep x h.length ≠ L.rep x (h.length - 1)) {m : Masker} (hm : L.fresh x h = .ok m) :
    L.answeredR x ρ h = .ok m := by
  rw [L.answeredR_of_opens x ρ h ho ht]
  rcases h with _ | ⟨e, t⟩
  · simp at h2
  rw [frameAnswerR_cons]
  unfold takeAtR
  by_cases hn : (e :: t).length ≤ 2
  · rw [if_pos hn]
    exact hm
  · rw [if_neg hn, if_neg (fun hb => hs (by omega) (of_decide_eq_true hb))]
    unfold fed at hf
    rw [decide_eq_false hn, Bool.false_or] at hf
    cases ha : (L.prog x (e :: t)).answer (L.cfgAt x ρ ((e :: t).length - 1)) (L.coinsOf x (e :: t)) with
    | error e' =>
      rw [ha] at hf
      first
        | cases hf
        | simp [Except.isOk, Except.toBool] at hf
    | ok a => exact hm

/-- **The model takes the coins after `h`**: where the transport's check passes `h`, the frame took the earlier
rounds' coins and its coin step takes the last, and the round after `h` is a rep's first at a next rep's start it has
(`hfr`), or the same rep's next where the rank stop passes the coin step's state (`hrk`). -/
theorem answered_of_took {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {h : List (Msg × Msg)}
    (ho : L.opens x ρ h = true) (ht : L.tookEarlier x ρ h = true) (hf : L.fed x ρ h = true) (h2 : 2 ≤ h.length)
    (hfr : (3 ≤ h.length → L.rep x h.length ≠ L.rep x (h.length - 1)) → ∃ m, L.fresh x h = .ok m)
    (hrk : ∀ m, L.answeredR x ρ h = .ok m → 3 ≤ h.length → L.rep x h.length = L.rep x (h.length - 1) →
      rankFails (L.cfg x ρ h) m = false) :
    ∃ m, L.answered x ρ h = .ok m := by
  by_cases hs : 3 ≤ h.length ∧ L.rep x h.length = L.rep x (h.length - 1)
  · obtain ⟨m, hm⟩ := answeredR_same ho ht hf hs.1 hs.2
    exact ⟨m, answered_of_answeredR hm (hrk m hm)⟩
  · have hs' : 3 ≤ h.length → L.rep x h.length ≠ L.rep x (h.length - 1) := fun h3 he => hs ⟨h3, he⟩
    obtain ⟨m, hm⟩ := hfr hs'
    exact ⟨m, answered_of_answeredR (answeredR_switch ho ht hf h2 hs' hm) fun h3 he => absurd he (hs' h3)⟩

end LeanRelease

namespace Relay

open _root_.Flock.Firewall.Session (State)

/-- **Where the session holds a coin tree, the transport's check passes the model's history** the loop took. -/
theorem LoopModel.TookAt.opens {W Rr : Type} {M : LoopModel W Rr} {st : State} {c : Nat} (h : M.TookAt st c)
    {C : _root_.Flock.Firewall.Coins.Release} (hC : st.session = some C) :
    M.L.opens M.x M.ρ (M.hist c) = true :=
  LeanRelease.opens_of_verdictAt (h.1 C hC)

/-- **A frame that releases a device round in the place of the opening's first round, at a state alike `m`, passes
the rank stop at `m`**: the opening's label checks the same rank on the same coins. -/
theorem rankFails_of_release {c : Cfg} {st st' : _root_.Flock.Firewall.Frame.State} {m : Masker} {n : Nat}
    {v b out : ByteArray} {vd : String} (hu : unstop st.m = unstop m) (hq : Flags.PcsRound (b, n))
    (h : _root_.Flock.Firewall.Frame.step c st (.round n v (.sched b)) = (.round out vd, st')) :
    LeanRelease.rankFails c m = false :=
  (Flags.rankFails_unstop c hu).symm.trans (Flags.rankFails_of_step (r := (b, n)) hq h)

/-- **The rank stop at a released round**: where the frame `q` is at the coin step's state after `h` but for
`stopped` (`hu`), at the model's configuration for the round, and releases a device round in the place `pc`, which is
the opening's first round wherever the rank stop fails at the coin step's state (`hal`), the rank stop passes it. -/
theorem rank_of_release {W Rr : Type} {L : LeanRelease W Rr} {x : Msg} {ρ : Rr} {h : List (Msg × Msg)}
    {q : _root_.Flock.Firewall.Frames.Rep} (hu : ∀ m, L.answeredR x ρ h = .ok m → unstop q.st.m = unstop m)
    (hcf : q.cfg = L.cfg x ρ h) {n : Nat} {v out : ByteArray} {pc : Place} {vd : String}
    {st' : _root_.Flock.Firewall.Frame.State}
    (hs : _root_.Flock.Firewall.Frame.step q.cfg q.st (.round n v pc) = (.round out vd, st'))
    (hal : ∀ m, L.answeredR x ρ h = .ok m → LeanRelease.rankFails (L.cfg x ρ h) m = true →
      ∃ b, pc = .sched b ∧ Flags.PcsRound (b, n)) :
    ∀ m, L.answeredR x ρ h = .ok m → LeanRelease.rankFails (L.cfg x ρ h) m = false := by
  intro m hm
  cases hr : LeanRelease.rankFails (L.cfg x ρ h) m with
  | false => rfl
  | true =>
    obtain ⟨b, rfl, hb⟩ := hal m hm hr
    rw [hcf] at hs
    have h' := rankFails_of_release (hu m hm) hb hs
    rw [hr] at h'
    cases h'

end Relay

end Claims.ZKServer
