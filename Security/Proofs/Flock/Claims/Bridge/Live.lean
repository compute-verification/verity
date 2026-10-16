import Proofs.Flock.Claims.Bridge.Transport
import Proofs.Flock.Soundness.Discharge.Tables.Sim

/-!
# Flock's live session as a public-coin verifier

`liveSession dg J mPts N` (`Discharge.Tables.Sim`) is the live game the old end-to-end theorems run: `Commit`'s roots,
the two link points, at most `N` rounds on the `2J` rep streams (each round's `n` coins drawn after its bytes), then the
proof files. This file runs it as a `Claims.Verifier` and transports `Claims.run`'s probabilities to it.

* `LCoin nMax`: a round's coin, `nMax` field elements. A round of `n ≤ nMax` coins reads the first `n`; the link points
  read the first `2·mPts` (`splitE`). Both projections are uniform (`CoinIf.ofEquiv`).
* `Wire nMax J mPts`: the wire formats: total decoders of the prover's `Commit`, submissions and proof files, and the
  bytes of the verifier's replies that carry the link points and a round's coins. A decoded submission asks for at most
  `nMax` coins (`sub_le`), and the link points fit in one coin (`pts_le`).
* `sessionK`: `liveSession` with each round's continuation explicit (`liveSession_eq`), and `sessionIf`, its interface
  with `Claims`' messages and coins. One round per prover message: `Commit` (its reply the link points), each
  submission (its reply that round's coins), the stop, the proof files (its reply the decision).
* `sessionLe`: every prover ends within `N + 2` rounds. `liveV … d` is `ofGame` at `k x = N x + 2`, deciding by `d x`
  on the session's outcome.
* `Pr_acc_live`, `Pr_acc_replay_live`, `forkPr_live`: `Claims.run`'s acceptance with an event of the outcome, or of
  what the verifier reads off the transcript, is `liveSession`'s probability of `d` accepting with that event, against
  the strategy the prover's bytes decode to (`liveStrat`); `B`'s fork at round `j` is `forkX` of the session.

**The correspondence.** The old games average over each `coin` node's own type (`prob`, `expect`) against a
`Strategy`, a move at each `send` node that may read every earlier move and coin. `Claims.run H P* (liveV …) x ω r`
averages over `r : Fin (N x + 2) → LCoin nMax`, the tape `ω` fixed. Round `j`'s message `a_j` is `P*.next x hist ω`;
its decoding is the move (`Wire`), and its reply carries the coin the session draws next: Commit's the link points, a
submission's its `n` coins, the proof files' the decision, which is `d x` on the outcome, its `H` queries the verifier's
log. `P*` sees every earlier reply, so the decoded strategy (`stratG`) reads exactly what a `Strategy` reads, and the
coin law (`Coins.ex_playOn`) gives each coin uniform, independent, after the message it answers. A stop (`none`) leaves
the later round coins unread. The round digest `roundOf dg` computes is the coin server's, not an oracle query.
-/

namespace Claims.Bridge.Live

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Tables FlockSoundness.Refine

/-- A round's coin: `nMax` field elements. -/
abbrev LCoin (nMax : ℕ) := Fin nMax → FlockLevel3.GF128

noncomputable instance (nMax : ℕ) : Inhabited (LCoin nMax) := ⟨fun _ => 0⟩

/-- `nMax` field elements as the first `n` and the rest. -/
def splitE (nMax n : ℕ) (h : n ≤ nMax) :
    LCoin nMax ≃ (Fin n → FlockLevel3.GF128) × (Fin (nMax - n) → FlockLevel3.GF128) :=
  (Equiv.arrowCongr ((finCongr (by omega)).trans finSumFinEquiv.symm) (Equiv.refl _)).trans
    (Equiv.sumArrowEquivProdArrow _ _ _)

/-- `2·mPts` field elements as two link points of `mPts` coordinates. -/
def ptsE (mPts : ℕ) : (Fin (2 * mPts) → FlockLevel3.GF128) ≃ (Fin 2 → Fin mPts → FlockLevel3.GF128) :=
  (Equiv.arrowCongr finProdFinEquiv.symm (Equiv.refl _)).trans (Equiv.curry _ _ _)

/-- **The wire formats**: total decoders of the prover's messages, and the bytes of the verifier's replies that carry
the link points and a round's coins. -/
structure Wire (nMax J mPts : ℕ) where
  root : Msg → Array ByteArray
  sub : Msg → Option (SubmitJ J)
  sub_le : ∀ a s, sub a = some s → s.n ≤ nMax
  proofs : Msg → Array ByteArray
  pts_le : 2 * mPts ≤ nMax
  encPts : (Fin 2 → Fin mPts → FlockLevel3.GF128) → Msg
  decPts : Msg → Fin 2 → Fin mPts → FlockLevel3.GF128
  decPts_enc : ∀ c, decPts (encPts c) = c
  encRd : (n : ℕ) → (Fin n → FlockLevel3.GF128) → Msg
  decRd : (n : ℕ) → Msg → Fin n → FlockLevel3.GF128
  decRd_enc : ∀ n c, decRd n (encRd n c) = c

/-- A round's `n` coins: the round coin's first `n` field elements. -/
def rdIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (n : ℕ) (h : n ≤ nMax) :
    CoinIf (LCoin nMax) (Fin n → FlockLevel3.GF128) :=
  CoinIf.ofEquiv (splitE nMax n h) (w.encRd n) (w.decRd n) (w.decRd_enc n)

/-- The link points: the round coin's first `2·mPts` field elements. -/
def ptsIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) : CoinIf (LCoin nMax) (Fin 2 → Fin mPts → FlockLevel3.GF128) :=
  CoinIf.ofEquiv ((splitE nMax (2 * mPts) w.pts_le).trans (Equiv.prodCongr (ptsE mPts) (Equiv.refl _))) w.encPts
    w.decPts w.decPts_enc

/-! ## The live session, each round's continuation explicit -/

/-- After a submission: the proof files on a stop, the round's coins on a round. -/
def branch {J mPts : ℕ} (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → FlockLevel3.GF128)
    (rs : Fin J → Fin 2 → List Flock.Round)
    (K : (s : SubmitJ J) → (Fin s.n → FlockLevel3.GF128) → Game (LiveOutJ J mPts)) :
    Option (SubmitJ J) → Game (LiveOutJ J mPts)
  | none => KLJ rootB pts rs
  | some s => .coin (Fin s.n → FlockLevel3.GF128) (K s)

/-- At most `N` rounds, then the proof files. -/
def roundsK (dg : ByteArray → ByteArray) {J mPts : ℕ} (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) : ℕ → (Fin J → Fin 2 → List Flock.Round) → Game (LiveOutJ J mPts)
  | 0, rs => KLJ rootB pts rs
  | N + 1, rs => .send (Option (SubmitJ J)) (branch rootB pts rs fun s c =>
      roundsK dg rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))

/-- **The live session**, `liveSession` with each round's continuation explicit. -/
def sessionK (dg : ByteArray → ByteArray) (J mPts N : ℕ) : Game (LiveOutJ J mPts) :=
  .send (Array ByteArray) fun rootB => .coin (Fin 2 → Fin mPts → FlockLevel3.GF128) fun pts =>
    roundsK dg rootB pts N fun _ _ => []

theorem bind_rounds (dg : ByteArray → ByteArray) {J mPts : ℕ} (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    ∀ N rs, (liveRoundsJ dg J N rs).bind (KLJ rootB pts) = roundsK dg rootB pts N rs
  | 0, _ => rfl
  | N + 1, rs => by
    show Game.send _ _ = Game.send _ _
    congr 1
    funext o
    cases o with
    | none => rfl
    | some s =>
      show Game.coin _ _ = Game.coin _ _
      congr 1
      funext c
      exact bind_rounds dg rootB pts N _

theorem liveSession_eq (dg : ByteArray → ByteArray) (J mPts N : ℕ) :
    liveSession dg J mPts N = sessionK dg J mPts N := by
  unfold liveSession sessionK
  congr 1
  funext rootB
  congr 1
  funext pts
  exact bind_rounds dg rootB pts N _

/-! ## Its interface with `Claims`' messages and coins -/

/-- The proof files, then the outcome. -/
def kIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) (rs : Fin J → Fin 2 → List Flock.Round) :
    Iface (LCoin nMax) (KLJ rootB pts rs) :=
  Iface.mkSend w.proofs fun a => Iface.mkRet (LiveOutJ.mk rootB pts rs (w.proofs a))

/-- After a decoded submission. -/
def subIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) (rs : Fin J → Fin 2 → List Flock.Round)
    (K : (s : SubmitJ J) → (Fin s.n → FlockLevel3.GF128) → Game (LiveOutJ J mPts))
    (next : (s : SubmitJ J) → (c : Fin s.n → FlockLevel3.GF128) → Iface (LCoin nMax) (K s c)) :
    (o : Option (SubmitJ J)) → (∀ s, o = some s → s.n ≤ nMax) → Iface (LCoin nMax) (branch rootB pts rs K o)
  | none, _ => kIf w rootB pts rs
  | some s, h => Iface.mkCoin (rdIf w s.n (h s rfl)) (next s)

/-- The rounds' interface. -/
def roundsIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (dg : ByteArray → ByteArray) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Iface (LCoin nMax) (roundsK dg rootB pts N rs)
  | 0, rs => kIf w rootB pts rs
  | N + 1, rs => Iface.mkSend w.sub fun a =>
      subIf w rootB pts rs _ (fun _ _ => roundsIf w dg rootB pts N _) (w.sub a) (w.sub_le a)

/-- **The live session's interface**: `Commit`, answered by the link points; each submission, answered by its round's
coins; the stop; the proof files, answered by the decision. -/
def sessionIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (dg : ByteArray → ByteArray) (N : ℕ) :
    Iface (LCoin nMax) (sessionK dg J mPts N) :=
  Iface.mkSend w.root fun a => Iface.mkCoin (ptsIf w) fun pts => roundsIf w dg (w.root a) pts N _

/-! ## Every prover ends within `N + 2` rounds -/

theorem kLe {nMax J mPts : ℕ} (w : Wire nMax J mPts) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) (rs : Fin J → Fin 2 → List Flock.Round)
    (s : RStrat (LCoin nMax) (KLJ rootB pts rs) (kIf w rootB pts rs)) (p : Bool) (n : ℕ) :
    RoundsLe (KLJ rootB pts rs) (kIf w rootB pts rs) s p (n + 1) :=
  trivial

theorem subLe {nMax J mPts : ℕ} (w : Wire nMax J mPts) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) (rs : Fin J → Fin 2 → List Flock.Round)
    (K : (s : SubmitJ J) → (Fin s.n → FlockLevel3.GF128) → Game (LiveOutJ J mPts))
    (next : (s : SubmitJ J) → (c : Fin s.n → FlockLevel3.GF128) → Iface (LCoin nMax) (K s c)) (N : ℕ)
    (ih : ∀ s c (t : RStrat (LCoin nMax) (K s c) (next s c)), RoundsLe (K s c) (next s c) t false (N + 1)) :
    ∀ (o : Option (SubmitJ J)) (ho : ∀ s, o = some s → s.n ≤ nMax)
      (t : RStrat (LCoin nMax) (branch rootB pts rs K o) (subIf w rootB pts rs K next o ho)),
      RoundsLe (branch rootB pts rs K o) (subIf w rootB pts rs K next o ho) t true (N + 1)
  | none, _, t => kLe w rootB pts rs t true N
  | some s, _, t => fun c => ih s c (t c)

theorem roundsLe {nMax J mPts : ℕ} (w : Wire nMax J mPts) (dg : ByteArray → ByteArray) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    ∀ N rs (s : RStrat (LCoin nMax) (roundsK dg rootB pts N rs) (roundsIf w dg rootB pts N rs)),
      RoundsLe (roundsK dg rootB pts N rs) (roundsIf w dg rootB pts N rs) s false (N + 1)
  | 0, rs, s => kLe w rootB pts rs s false 0
  | N + 1, rs, s =>
    subLe w rootB pts rs _ (fun _ _ => roundsIf w dg rootB pts N _) N
      (fun _ _ t => roundsLe w dg rootB pts N _ t) (w.sub s.1) (w.sub_le s.1) s.2

/-- **Every prover ends within `N + 2` rounds**. -/
theorem sessionLe {nMax J mPts : ℕ} (w : Wire nMax J mPts) (dg : ByteArray → ByteArray) (N : ℕ)
    (s : RStrat (LCoin nMax) (sessionK dg J mPts N) (sessionIf w dg N)) :
    RoundsLe (sessionK dg J mPts N) (sessionIf w dg N) s false (N + 2) :=
  fun pts => roundsLe w dg (w.root s.1) pts N _ (s.2 pts)

/-! ## The verifier, and the transport -/

/-- **The live session as a public-coin verifier**: `N x + 2` rounds, deciding by `d x` on the session's outcome. -/
noncomputable def liveV (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) :
    Verifier (LCoin nMax) :=
  ofGame (fun x => N x + 2) (fun x => sessionK dg (J x) (mPts x) (N x)) (fun x => sessionIf (w x) dg (N x)) d

/-- **The strategy the run plays in `liveSession`**: the prover's bytes, decoded. -/
noncomputable def liveStrat {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    Strategy (liveSession dg (J x) (mPts x) (N x)) :=
  (liveSession_eq dg (J x) (mPts x) (N x)).symm ▸
    stratG H P (fun x => sessionK dg (J x) (mPts x) (N x)) (fun x => sessionIf (w x) dg (N x)) x ω

/-- The session's outcome of a run. -/
noncomputable def liveOut {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (N x + 2) → LCoin nMax) : LiveOutJ (J x) (mPts x) :=
  outcome H P (fun x => N x + 2) (fun x => sessionK dg (J x) (mPts x) (N x)) (fun x => sessionIf (w x) dg (N x)) x ω r

/-- The session's outcome the verifier reads off a transcript's messages. -/
def liveRead {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Msg) → Wire nMax (J x) (mPts x))
    (x : Msg) (msgs : List (Msg × Out)) : Option (LiveOutJ (J x) (mPts x)) :=
  replay ⟨sessionK dg (J x) (mPts x) (N x), sessionIf (w x) dg (N x)⟩ msgs

theorem prob_eqRec {β : Type} {g g' : Game β} (h : g = g') (p : β → Prop) (s : Strategy g) :
    prob p g' (h ▸ s) = prob p g s := by
  subst h
  rfl

/-- **The run is the live session**: its verdict and verifier log are `d`'s on the session's outcome, and the verifier
reads that outcome off the run's transcript. -/
theorem run_live (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) (H : Msg → Msg)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (N x + 2) → LCoin nMax) :
    (run H P (liveV nMax J mPts N dg w d) x ω r).out = toOut ((d x (liveOut J mPts N dg w H P x ω r)).eval H).1 ∧
      (run H P (liveV nMax J mPts N dg w d) x ω r).vLog = ((d x (liveOut J mPts N dg w H P x ω r)).eval H).2 ∧
      liveRead J mPts N dg w x (run H P (liveV nMax J mPts N dg w d) x ω r).msgs =
        some (liveOut J mPts N dg w H P x ω r) :=
  run_ofGame H P (fun x => N x + 2) (fun x => sessionK dg (J x) (mPts x) (N x)) (fun x => sessionIf (w x) dg (N x)) d
    x ω r (sessionLe (w x) dg (N x) _)

open Classical in
/-- **Acceptance with an event of the outcome**: `Pr_r[run = acc ∧ Q(outcome)]` is `liveSession`'s probability that
`d x` accepts the outcome and `Q` holds, against `liveStrat`. -/
theorem Pr_acc_live (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) (H : Msg → Msg)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (Q : LiveOutJ (J x) (mPts x) → Prop) :
    Pr (fun r : Fin ((liveV nMax J mPts N dg w d).k x) → LCoin nMax =>
        (run H P (liveV nMax J mPts N dg w d) x ω r).out = .acc ∧ Q (liveOut J mPts N dg w H P x ω r)) =
      (prob (fun o => ((d x o).eval H).1 = true ∧ Q o) (liveSession dg (J x) (mPts x) (N x))
        (liveStrat J mPts N dg w H P x ω)).toReal := by
  rw [liveStrat, prob_eqRec, prob_toReal]
  exact Pr_acc_and H P _ _ _ d x ω (sessionLe (w x) dg (N x) _) Q

open Classical in
/-- **Acceptance**, as `liveSession`'s probability. -/
theorem Pr_acc_live' (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) (H : Msg → Msg)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    Pr (fun r : Fin ((liveV nMax J mPts N dg w d).k x) → LCoin nMax =>
        (run H P (liveV nMax J mPts N dg w d) x ω r).out = .acc) =
      (prob (fun o => ((d x o).eval H).1 = true) (liveSession dg (J x) (mPts x) (N x))
        (liveStrat J mPts N dg w H P x ω)).toReal := by
  rw [liveStrat, prob_eqRec]
  exact Pr_acc H P _ _ _ d x ω (sessionLe (w x) dg (N x) _)

open Classical in
/-- **Acceptance with an event of the transcript**: an event `Q` of the outcome the verifier reads off the run's
messages (`liveRead`) has `liveSession`'s probability; at `Q := (· = some o₀)`, acceptance with outcome `o₀`. -/
theorem Pr_acc_replay_live (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) (H : Msg → Msg)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (Q : Option (LiveOutJ (J x) (mPts x)) → Prop) :
    Pr (fun r : Fin ((liveV nMax J mPts N dg w d).k x) → LCoin nMax =>
        (run H P (liveV nMax J mPts N dg w d) x ω r).out = .acc ∧
          Q (liveRead J mPts N dg w x (run H P (liveV nMax J mPts N dg w d) x ω r).msgs)) =
      (prob (fun o => ((d x o).eval H).1 = true ∧ Q (some o)) (liveSession dg (J x) (mPts x) (N x))
        (liveStrat J mPts N dg w H P x ω)).toReal := by
  rw [liveStrat, prob_eqRec, prob_toReal]
  exact Pr_acc_replay H P _ _ _ d x ω (sessionLe (w x) dg (N x) _) Q

/-- **`B`'s fork at the live session**: the probability that `B`'s two runs, forked at round `j`, collide is the
average over the tape of the session's `forkX` of the collision of the two decisions' `H` logs. -/
theorem forkPr_live (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) (H : Msg → Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (hk : 0 < (liveV nMax J mPts N dg w d).k x)
    (j : Fin ((liveV nMax J mPts N dg w d).k x)) :
    forkPr H (liveV nMax J mPts N dg w d) P x hk j =
      Ex fun ω : Ω => forkX (collOf H (d x)) j (sessionK dg (J x) (mPts x) (N x)) (sessionIf (w x) dg (N x))
        (stratOf H P x ω (sessionK dg (J x) (mPts x) (N x)) (sessionIf (w x) dg (N x)) [] none) none :=
  forkPr_ofGame H P _ _ _ d x (fun _ => sessionLe (w x) dg (N x) _) hk j

end Claims.Bridge.Live
