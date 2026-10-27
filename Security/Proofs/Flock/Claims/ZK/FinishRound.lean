import Proofs.Flock.Claims.ZK.Refine

/-!
# The `Finish` round where neither the frame nor the firewall's own `Finish` stops

`FinalRoundEq` holds the firewall's `Finish` (the session's Lean finish, `Flock.Firewall.Finish`) to the session's
switched last message, which is `fin` of the tables' switched views where every table admits the opened coins and `off`
of the cut where one does not (`RefSession.swMsgAt`). Here it is split at that branch:

* `LeanRelease.FrameAdmits`: on a live history of `refK x` openings whose checks pass and whose last coins the frame takes,
  every table admits the opened coins. A named hypothesis, one of the transport's trusts: the frame refuses each of the
  four coin faults `Admits` excludes (an outer coin `1`, `ρ_0, ρ_1` dependent over F128, a rank-deficient mask slot,
  `β = 0`: `Frame.Masker.answer`, `Frame.maskRank`), but the release reads the frame's state and coins through the
  transport (`prog`, `coinsOf`), so that the state it hands the frame is the frame's own run on the opened coins is
  the session loop's (`Flock.Firewall.Frames`, on `Drive.reqs`), as `ReadsCoins` is. That the frames took every
  round's coins on such a history is proved (`LeanRelease.coins_taken_of_live`, `FrameTaken`), and that a rank the
  frame finds full is `maskRankOk` at the statement's mask words (`Rank.maskRankOk_of_maskRank`, `FrameRank`).
* `LeanRelease.FinishIs`: on such a history where the firewall's own `Finish` does not stop either (`finStops`), its
  files encode `fin` of the tables' switched views. It says nothing where the firewall's `Finish` stops, which the server
  refuses (`refusesFinish`).
* **`LeanRelease.finalRoundEq_of_finishIs`**: `FinalRoundEq` from the two, at a `Finish` one past the last coin: a live
  run's `Finish` is one the firewall's own `Finish` takes (`finStops_of_live`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

namespace LeanRelease

/-- **The frame's coins are coins every table admits**: after every live history of `refK x` openings whose openings
pass and whose last coins the frame takes, every table admits the coins the openings open (`RefSession.Admits`). A named
hypothesis, one of the transport's trusts: the Lean frame refuses each coin fault `Admits` excludes, and the state and
coins the transport hands it (`prog`, `coinsOf`) are its own run on the opened coins (the session loop's rep frame,
`Flock.Firewall.Frames`). -/
def FrameAdmits (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg)
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    ops.length = refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    S.Admits x (S.pad (ops.map fun p => S.decOf p.2))

/-- **The firewall's `Finish` is the session's at admitted coins**: after every live history of `refK x` openings whose
openings pass, whose last coins the frame takes, at which the firewall's own `Finish` does not stop (`finStops`) and
whose messages are the deferred release's own (`ZkServer.OwnHist`), `Final.encode` of the record and proof files the
firewall builds itself is `fin` of the tables' switched views at the coins the openings open. A named hypothesis, one of
the transport's trusts, narrower than `FinalRoundEq`: it says nothing where the frame or the firewall's `Finish` stops,
nor at coins a table does not admit, nor on a history the release did not send, whose rounds its record would read. -/
def FinishIs (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    ops.length = refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false → L.finStops x w (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    FlockDeploy.Final.encode ⟨x, (L.files x w (ν, t) ((m₀, ρ₀) :: ops)).1, (L.files x w (ν, t) ((m₀, ρ₀) :: ops)).2⟩ =
      S.fin x (S.pad (ops.map fun p => S.decOf p.2))
        (S.swViewsAt x (S.pad (ops.map fun p => S.decOf p.2)) (S.wt x w)
          (vs x w ν (S.pad (ops.map fun p => S.decOf p.2))) t)

/-- **`FinalRoundEq` from the frame's coins being admitted and the `Finish` being the session's there**, at a `Finish`
one past the last coin: a live run's `Finish` is one the firewall's own `Finish` takes (`finStops_of_live`), and at
admitted coins the switched last message is `fin` of the switched views (`RefSession.swMsgAt`). -/
theorem finalRoundEq_of_finishIs (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hlast : L.last x = refK x + 1) (hA : FrameAdmits S L H wk x w) (hF : FinishIs S vs L H wk x w) :
    FinalRoundEq S vs L H wk x w := by
  intro ν t κ m₀ ρ₀ ops hl hlive hok hr hown
  have hs := L.finStops_of_live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) hlive
    (by rw [List.length_cons, hl, hlast])
  rw [hF ν t κ m₀ ρ₀ ops hl hlive hok hr hs hown]
  unfold RefSession.swMsgAt
  rw [if_neg (by rw [List.length_map, hl]; exact lt_irrefl _), if_pos (hA ν t κ m₀ ρ₀ ops hl hlive hok hr)]

end LeanRelease

end Claims.ZKServer
