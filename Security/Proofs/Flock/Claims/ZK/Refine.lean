import Proofs.Flock.Claims.ZK.Release

/-!
# Toward the item-stream refinement at the reference session

`Flock.zk_server_lean` takes `LeanRelease.ItemsRefine` at the switched reference prover `(S.swP vs).prover`: wherever
the deferred release releases a frame, it is that prover's message. That prover is the committed-coin strategy
(`Claims.ZKMal.strat`): `Hello` first, then, while every opening so far passes its coin-tree check, its message at the
coins they open (`RefSession.swMsgAt`: `enc` before the last coin, then `fin` or `off`), and `abort` from the first
opening that fails (`strat_eval_cons`).

* **The coin stops** (`stopRefine_of_sessionStops`): `StopRefine` holds once the session's `abort` is the frame's stop and
  its message is the stop after every history whose openings pass and whose last coins the frame refuses
  (`SessionStops`). A session whose messages are the frame's releases (`SessionFrames`) has the second, by
  `LeanRelease.release_of_refusesCoins` (`sessionStops_of_frames`). At the empty history the frame refuses no coins
  (`refusesCoins_nil`), and the prover's message there is its `Hello`.
* **The openings** (`release_eq_abort_of_liveRefine`): after an opening the prover's coin-tree check refuses, the prover
  sends `abort` at every later round, so `LiveRefine` makes the frame's release of a live round there the `abort`
  message. The frame takes the coins the transport hands it (`LeanRelease.answered`, from the last answer alone) and
  checks no opening: the deployed opening check is Rust's (`coin_tree::check`, in `flock-circuit`'s checked transport).
  So `LiveRefine` and `FillRefine` hold only for a transport whose coins the frame refuses after any opening that check
  refuses, with `abort` the stop.
* **The coin stops read no tape** (`LeanRelease.refusesCoins_tape`): the frame's refusal of the coins (`Masker.answer`)
  reads its configuration's layout and absorbed order alone (`answer_congr`), never the pads or the root, so it is the
  same on every tape. A session, whose round encoding sees the coins and the cut view but not the tape, can stop where
  the frame does.
* **The session with the frame's stops** (`RefSession.withStops`): `S` with every message `stop` at the coins and round
  where `Rf` says, and its `abort` the stop. It keeps `S`'s tables, witnesses, dummy and tape, so `Tables`, `Complete`,
  `DummyReg` and `TapeOK` carry over as they are, and `Causal` and `SwCausal` where `Rf` reads the coins before its round
  only (`StopsCausal`). Where its stops are the frame's refusals (`StopsAt`: on every history whose openings pass, `Rf`
  at the coins they open is `refusesCoins`), `StopRefine` holds at its switched prover
  (`LeanRelease.stopRefine_withStops`). `LiveRefine` and `FillRefine` hold there from two more each
  (`LeanRelease.liveRefine_withStops`, `fillRefine_withStops`): the transport makes the frame refuse after any opening
  that fails (`OpeningsRefused`, today Rust's), and the frame's release of each worker round that passed
  (`LiveRoundEq`), and of each scheduled round in a failed worker's place below the last (`FillRoundEq`), is the
  session's message, `Hello` first: the per-round byte equality. So `Flock.zk_server_lean` holds at that session from
  those alone (`Flock.zk_server_lean_stops`), with no part of `ItemsRefine` assumed.
-/

namespace Claims.ZKMal.RefSession

/-- **A stop rule reads the coins before its round only**: `Rf x c i`, at the padded coins `c` and the round `i`, is the
same at every `c'` that agrees with `c` below `i`. -/
def StopsCausal (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) : Prop :=
  ∀ x (c c' : Fin (refK x) → S.Coin) (i : ℕ), (∀ j : Fin (refK x), j.val < i → c j = c' j) → Rf x c i = Rf x c' i

/-- **The session with stops**: `S`, whose message at the coins `c` and round `i` is `stop` where `Rf x c i` (round
`refK x` for the last message, `fin` or `off`), and whose `abort` is `stop`. -/
def withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg) : RefSession :=
  { S with
    abort := stop
    enc := fun x c i σ => if Rf x c i then stop else S.enc x c i σ
    fin := fun x c V => if Rf x c (refK x) then stop else S.fin x c V
    off := fun x c σ => if Rf x c (refK x) then stop else S.off x c σ }

theorem withStops_tables (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (h : S.Tables) : (S.withStops Rf stop).Tables :=
  h

theorem withStops_complete (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    {R : Rel Flock.Wit} (h : S.Complete R) : (S.withStops Rf stop).Complete R :=
  h

theorem withStops_dummyReg (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (h : S.DummyReg) : (S.withStops Rf stop).DummyReg :=
  h

theorem withStops_tapeOK (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (h : S.TapeOK) : (S.withStops Rf stop).TapeOK :=
  h

theorem withStops_causal (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (hRf : S.StopsCausal Rf) (hc : S.Causal) : (S.withStops Rf stop).Causal := by
  intro x wt ω p c c' i h
  show (if Rf x c i = true then stop else S.enc x c i _) = (if Rf x c' i = true then stop else S.enc x c' i _)
  rw [hRf x c c' i h]
  exact congrArg (fun m => if Rf x c' i = true then stop else m) (hc x wt ω p c c' i h)

theorem withStops_swCausal (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (hRf : S.StopsCausal Rf) (hc : S.SwCausal vs) : (S.withStops Rf stop).SwCausal vs := by
  intro x w ν ω p c c' i h
  show (if Rf x c i = true then stop else S.enc x c i _) = (if Rf x c' i = true then stop else S.enc x c' i _)
  rw [hRf x c c' i h]
  exact congrArg (fun m => if Rf x c' i = true then stop else m) (hc x w ν ω p c c' i h)

open Classical in
/-- **Where the rule stops the coins, the session's switched message is the stop**, before the last coin and after it. -/
theorem swMsgAt_withStops_stop (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals) (t : S.Tape) (cs : List S.Coin)
    (hR : Rf x (S.pad cs) (min cs.length (refK x)) = true) : (S.withStops Rf stop).swMsgAt x wt v t cs = stop := by
  by_cases h : cs.length < refK x
  · rw [Nat.min_eq_left h.le] at hR
    show (if cs.length < refK x then (if Rf x (S.pad cs) cs.length = true then stop else _) else _) = stop
    rw [if_pos h, if_pos hR]
  · rw [Nat.min_eq_right (Nat.le_of_not_lt h)] at hR
    show (if cs.length < refK x then _ else if S.Admits x (S.pad cs) then
        (if Rf x (S.pad cs) (refK x) = true then stop else _) else
        (if Rf x (S.pad cs) (refK x) = true then stop else _)) = stop
    rw [if_neg h]
    by_cases hA : S.Admits x (S.pad cs)
    · rw [if_pos hA, if_pos hR]
    · rw [if_neg hA, if_pos hR]

open Classical in
/-- **Where the rule doesn't stop the coins, the session's switched message is `S`'s.** -/
theorem swMsgAt_withStops_go (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals) (t : S.Tape) (cs : List S.Coin)
    (hR : Rf x (S.pad cs) (min cs.length (refK x)) = false) :
    (S.withStops Rf stop).swMsgAt x wt v t cs = S.swMsgAt x wt v t cs := by
  by_cases h : cs.length < refK x
  · rw [Nat.min_eq_left h.le] at hR
    show (if cs.length < refK x then
        (if Rf x (S.pad cs) cs.length = true then stop
          else S.enc x (S.pad cs) cs.length (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t)) else _) =
      (if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t) else _)
    rw [if_pos h, if_pos h, hR, if_neg Bool.false_ne_true]
  · rw [Nat.min_eq_right (Nat.le_of_not_lt h)] at hR
    show (if cs.length < refK x then _ else if S.Admits x (S.pad cs) then
        (if Rf x (S.pad cs) (refK x) = true then stop
          else S.fin x (S.pad cs) (S.swViewsAt x (S.pad cs) wt (v (S.pad cs)) t)) else
        (if Rf x (S.pad cs) (refK x) = true then stop
          else S.off x (S.pad cs) (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t))) =
      (if cs.length < refK x then _ else if S.Admits x (S.pad cs) then
        S.fin x (S.pad cs) (S.swViewsAt x (S.pad cs) wt (v (S.pad cs)) t)
        else S.off x (S.pad cs) (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t))
    rw [if_neg h, if_neg h, hR, if_neg Bool.false_ne_true, if_neg Bool.false_ne_true]

end Claims.ZKMal.RefSession

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock.Firewall.Frame (Cfg Masker)

/-- **The frame's coin refusal reads its configuration's layout and absorbed order alone**, never its pads or root. -/
theorem answer_congr {c c' : Cfg} (s : Masker) (cs : Array _root_.Flock.F128) (hl : c.lay = c'.lay)
    (ho : c.order = c'.order) : s.answer c cs = s.answer c' cs := by
  unfold Masker.answer
  rw [hl, ho]

namespace LeanRelease

/-- At the empty history the frame has no coins to take, so it refuses none. -/
theorem refusesCoins_nil {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) : L.refusesCoins x ρ [] = false :=
  rfl

/-- **The session's coin stops are the frame's**: after every history whose openings all pass the coin-tree check and
whose last coins the frame refuses, the switched reference prover's message at the coins the openings open is the stop.
-/
def SessionStops (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = true →
      S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) = L.stop

/-- **The session's messages are the frame's releases**: after every history past `Hello`, the switched reference
prover's message at the coins the openings open is the frame's release there of some device round, of a stopped rep or
not. -/
def SessionFrames (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)), ∃ (st : Bool) (v : Msg),
    S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) = L.release x (ν, t) ((m₀, ρ₀) :: ops) st v

/-- **A session whose messages are the frame's releases stops where the frame refuses the coins**
(`release_of_refusesCoins`). -/
theorem sessionStops_of_frames (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (x : Msg) (w : Flock.Wit)
    (hF : SessionFrames S vs L x w) : SessionStops S vs L H x w := by
  intro ν t m₀ ρ₀ ops _ hr
  obtain ⟨st, v, he⟩ := hF ν t m₀ ρ₀ ops
  rw [he]
  exact L.release_of_refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) st v hr

/-- **`StopRefine` at the switched reference prover**, at any worker and last round: from the session's `abort` being
the stop and its coin stops being the frame's. -/
theorem stopRefine_of_sessionStops (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hA : S.abort = L.stop) (hS : SessionStops S vs L H x w) :
    L.StopRefine H (S.swP vs).prover n wk x w := by
  intro ρ κ h _ hr
  obtain ⟨ν, t⟩ := ρ
  cases h with
  | nil =>
    rw [refusesCoins_nil] at hr
    cases hr
  | cons p ops =>
    obtain ⟨m₀, ρ₀⟩ := p
    show ((strat ((S.swP vs).hello x ν) (S.swP vs).abort (S.swP vs).dec ((S.swP vs).chk x ν)
      ((S.swP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = L.stop
    rw [strat_eval_cons]
    split
    next hok => exact hS ν t m₀ ρ₀ ops hok hr
    next => exact hA

/-- **`StopRefine` at a session whose messages are the frame's releases.** -/
theorem stopRefine_of_sessionFrames (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hA : S.abort = L.stop) (hF : SessionFrames S vs L x w) :
    L.StopRefine H (S.swP vs).prover n wk x w :=
  stopRefine_of_sessionStops S vs L H n wk x w hA (sessionStops_of_frames S vs L H x w hF)

/-- **After an opening the prover's check refuses, `LiveRefine` makes the frame's release the `abort` message**: at a
live round past `Hello` whose last coins the frame takes, where some opening so far fails the coin-tree check, the
frame's release of the worker's round is the switched reference prover's `abort`. -/
theorem release_eq_abort_of_liveRefine (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hL : L.LiveRefine H (S.swP vs).prover wk x w) (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg)
    (hl : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hr : L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false))
    (hok : ((allOk ((S.swP vs).chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = false) :
    L.release x (ν, t) ((m₀, ρ₀) :: ops) false v = S.abort := by
  have he := hL (ν, t) κ ((m₀, ρ₀) :: ops) v hl hr hv
  have hm : ((strat ((S.swP vs).hello x ν) (S.swP vs).abort (S.swP vs).dec ((S.swP vs).chk x ν)
      ((S.swP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = S.abort := by
    rw [strat_eval_cons, hok] <;> rfl
  exact he.symm.trans hm

/-- The frame's taking of the coins is the same on every tape. -/
theorem answered_tape {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ ρ' : R) (h : List (Msg × Msg)) :
    L.answered x ρ h = L.answered x ρ' h := by
  unfold answered
  cases h.getLast? with
  | none => rfl
  | some p =>
    obtain ⟨_, cs⟩ := p
    exact answer_congr _ _ (by rfl) (by rfl)

/-- **The coin stops read no tape**: the frame refuses the coins after `h` on every tape or on none. -/
theorem refusesCoins_tape {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ ρ' : R) (h : List (Msg × Msg)) :
    L.refusesCoins x ρ h = L.refusesCoins x ρ' h := by
  unfold refusesCoins
  rw [L.answered_tape x ρ ρ' h]

/-- **The rule's stops are the frame's refusals** at `x`: after every history past `Hello` whose openings all pass the
coin-tree check, `Rf` at the coins they open, padded, at their round, is whether the frame refuses the last coins. After
an opening that fails, the prover sends its `abort`, so the rule need not say there. -/
def StopsAt (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (x : Msg) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) =
      Rf x (S.pad (ops.map fun p => S.decOf p.2)) (min (ops.map fun p => S.decOf p.2).length (refK x))

/-- **`StopRefine` at the session with the frame's stops**, at any worker, last round and values: its switched
prover's message is the stop wherever the frame refuses the coins. -/
theorem stopRefine_withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hR : StopsAt S Rf L H x) :
    L.StopRefine H ((S.withStops Rf L.stop).swP vs).prover n wk x w :=
  stopRefine_of_sessionStops (S.withStops Rf L.stop) vs L H n wk x w rfl fun ν t m₀ ρ₀ ops hok hr =>
    S.swMsgAt_withStops_stop Rf L.stop x _ _ t _ ((hR ν t m₀ ρ₀ ops hok).symm.trans hr)

/-- **The transport refuses a failed opening** at `x`: after every history past `Hello` some opening of which fails the
coin-tree check, the frame refuses the last coins. The frame checks no opening itself; today it is Rust's
`coin_tree::check` in `flock-circuit`'s checked transport that ends the session there. -/
def OpeningsRefused (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg)
    (x : Msg) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = false →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = true

/-- **The live rounds' bytes**, the per-round byte equality at the worker's rounds: at `Hello`, the frame's release of
the worker's live round is the session's `Hello`; after every history whose rounds all passed, whose openings pass and
whose last coins the frame takes, it is the session's switched message at the coins the openings open. -/
def LiveRoundEq (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  (∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (v : Msg), L.toSrv.live wk x w (ν, t) κ [] = true →
    wk x w (ν, t) [] 0 = some (v, false) → L.release x (ν, t) [] false v = S.hello x ν) ∧
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.release x (ν, t) ((m₀, ρ₀) :: ops) false v = S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)

/-- **`LiveRefine` at the session with the frame's stops**, from the stop link, the transport's refusal of a failed
opening and the live rounds' bytes. -/
theorem liveRefine_withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hR : StopsAt S Rf L H x) (hO : OpeningsRefused S L H x) (hE : LiveRoundEq S vs L H wk x w) :
    L.LiveRefine H ((S.withStops Rf L.stop).swP vs).prover wk x w := by
  intro ρ κ h v hl hr hv
  obtain ⟨ν, t⟩ := ρ
  cases h with
  | nil => exact (hE.1 ν t κ v hl hv).symm
  | cons p ops =>
    obtain ⟨m₀, ρ₀⟩ := p
    show ((strat (((S.withStops Rf L.stop).swP vs).hello x ν) ((S.withStops Rf L.stop).swP vs).abort
      ((S.withStops Rf L.stop).swP vs).dec (((S.withStops Rf L.stop).swP vs).chk x ν)
      (((S.withStops Rf L.stop).swP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = _
    rw [strat_eval_cons]
    split
    next hok =>
      show (S.withStops Rf L.stop).swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) = _
      exact (S.swMsgAt_withStops_go Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
        ((hR ν t m₀ ρ₀ ops hok).symm.trans hr)).trans (hE.2 ν t κ m₀ ρ₀ ops v hl hok hr hv).symm
    next hok =>
      have h1 := hO ν t m₀ ρ₀ ops (by simpa using hok)
      cases h1.symm.trans hr

/-- **The stand-in rounds' bytes**, the per-round byte equality at the firewall's scheduled rounds: below the last round
`n x - 1`, once a round failed, at `Hello` the frame's release of the scheduled round, at the values the firewall's
transcript reports and as a stopped rep's, is the session's `Hello`; after every history whose openings pass and whose
last coins the frame takes, it is the session's switched message at the coins the openings open. -/
def FillRoundEq (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  (∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk), 1 < n x → L.toSrv.live wk x w (ν, t) κ [] = false →
    L.release x (ν, t) [] true (L.scheduled x w (ν, t) [] (computedOf (L.computed x w (ν, t) []))) = S.hello x ν) ∧
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    ((m₀, ρ₀) :: ops).length + 1 < n x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.release x (ν, t) ((m₀, ρ₀) :: ops) true
        (L.scheduled x w (ν, t) ((m₀, ρ₀) :: ops) (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops)))) =
      S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)

/-- **`FillRefine` at the session with the frame's stops**, from the stop link, the transport's refusal of a failed
opening and the stand-in rounds' bytes. -/
theorem fillRefine_withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hR : StopsAt S Rf L H x) (hO : OpeningsRefused S L H x)
    (hE : FillRoundEq S vs L H n wk x w) : L.FillRefine H ((S.withStops Rf L.stop).swP vs).prover n wk x w := by
  intro ρ κ h hlt hl hr
  obtain ⟨ν, t⟩ := ρ
  cases h with
  | nil => exact (hE.1 ν t κ hlt hl).symm
  | cons p ops =>
    obtain ⟨m₀, ρ₀⟩ := p
    show ((strat (((S.withStops Rf L.stop).swP vs).hello x ν) ((S.withStops Rf L.stop).swP vs).abort
      ((S.withStops Rf L.stop).swP vs).dec (((S.withStops Rf L.stop).swP vs).chk x ν)
      (((S.withStops Rf L.stop).swP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = _
    rw [strat_eval_cons]
    split
    next hok =>
      show (S.withStops Rf L.stop).swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) = _
      exact (S.swMsgAt_withStops_go Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
        ((hR ν t m₀ ρ₀ ops hok).symm.trans hr)).trans (hE.2 ν t κ m₀ ρ₀ ops hlt hl hok hr).symm
    next hok =>
      have h1 := hO ν t m₀ ρ₀ ops (by simpa using hok)
      cases h1.symm.trans hr

end LeanRelease

open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **`Flock.zk_server_lean` at the session with the frame's stops**: at a session `S` and a stop rule `Rf` that reads
the coins before its round only and is the frame's refusals at `x` wherever the openings pass (`LeanRelease.StopsAt`),
the bound holds at `S.withStops Rf L.stop` from `S`'s six Props, the transport's refusal of a failed opening
(`LeanRelease.OpeningsRefused`) and the per-round byte equality at the worker's rounds and the stand-in rounds
(`LeanRelease.LiveRoundEq`, `LeanRelease.FillRoundEq`): the item-stream refinement is proved from them
(`LeanRelease.stopRefine_withStops`, `liveRefine_withStops`, `fillRefine_withStops`). -/
theorem Flock.zk_server_lean_stops (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession)
    (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (hRf : S.StopsCausal Rf) (vs : S.SwVals) (t : ℕ)
    (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hS : S.Tables) (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hsc : S.SwCausal vs) (hT : S.TapeOK) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hst : LeanRelease.StopsAt S Rf L H x) (hO : LeanRelease.OpeningsRefused S L H x)
    (hE : LeanRelease.LiveRoundEq S vs L H wk x w)
    (hEf : LeanRelease.FillRoundEq S vs L H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk x w)
    {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk =>
        Ev (L.toSrv.timed x (runStar H (L.toSrv.proverD L.fillOf (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 (((S.withStops Rf L.stop).refP.withSim fun x _ c =>
              (S.withStops Rf L.stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).C x =>
          Ev (L.toSrv.timed x ((leaky1 (((S.withStops Rf L.stop).refP.withSim fun x _ c =>
              (S.withStops Rf L.stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).out H Vs θ x c))) +
          (8 * (S.withStops Rf L.stop).nLeaves x * δ₁.toReal + 8 * t * (S.withStops Rf L.stop).nLeaves x * δ₁.toReal +
            2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H (((S.withStops Rf L.stop).swP vs).prover w) Vs θ x).Pr H) :=
  Flock.zk_server_lean H rel (S.withStops Rf L.stop) vs t ht hT1 hδ₁ (S.withStops_tables Rf L.stop hS)
    (S.withStops_complete Rf L.stop hV) (S.withStops_dummyReg Rf L.stop hD) (S.withStops_causal Rf L.stop hRf hc)
    (S.withStops_swCausal Rf L.stop vs hRf hsc) (S.withStops_tapeOK Rf L.stop hT) L wk x w hR
    ⟨L.liveRefine_withStops S Rf vs H wk x w hst hO hE, L.fillRefine_withStops S Rf vs H _ wk x w hst hO hEf,
      L.stopRefine_withStops S Rf vs H _ wk x w hst⟩ Vs θ hk Ev

end Claims.ZKServer
