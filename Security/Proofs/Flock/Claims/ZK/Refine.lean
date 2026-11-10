module

public import Proofs.Flock.Claims.ZK.CoinOpen
public import Proofs.Flock.Claims.ZK.HelloLean
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# Toward the item-stream refinement at the reference session

`Flock.zk_server_lean` takes `LeanRelease.ItemsRefine` at the switched reference prover `(S.swP vs).prover`: wherever
the deferred release releases a frame (the server's `Hello`, the ZK server's own link and `Finish`, or a frame round's
release), it is that prover's message. That prover is the committed-coin strategy
(`Claims.ZKMal.strat`): `Hello` first, then, while every opening so far passes its coin-tree check, its message at the
coins they open (`RefSession.swMsgAt`: `enc` before the last coin, then `fin` or `off`), and `abort` from the first
opening that fails (`strat_eval_cons`).

* **The coin stops** (`stopRefine_of_sessionStops`): `StopRefine` holds once the session's `abort` is the frame's stop
  and its message is the stop after every history whose openings pass and whose last coins the frame refuses
  (`SessionStops`). A session whose messages are the frame's releases (`SessionFrames`) has the second, by
  `LeanRelease.release_of_refusesCoins` (`sessionStops_of_frames`). At the empty history the frame refuses no coins
  (`refusesCoins_nil`), and the prover's message there is its `Hello`.
* **The openings** (`release_eq_abort_of_liveRefine`): after an opening the prover's coin-tree check refuses, the prover
  sends `abort` at every later round, so `LiveRefine` makes the frame's release of a live round there the `abort`
  message. The frame checks no opening; the release reads the transport's check (`LeanRelease.opens`,
  `flock-firewall coins` at the tape's nonce) and is the stop where it refuses. So `LiveRefine` and `FillRefine` hold
  where that check is the session's (`CheckIs`): then the release refuses after any opening the session's check
  refuses, up to the protocol's last coin (`OpeningsRefused`, `openingsRefused_of_checkIs`), with `abort` the stop.
* **The frame's coin stops read no tape** (`LeanRelease.refusesCoins_tape`): on the histories the transport's check
  passes, the frame's refusal of the coins (`Masker.answer`, `rankStop`) reads its configuration's layout, absorbed
  order and mask slot alone (`answer_congr`, `rankStop_congr`), never the pads or the root, and a rep's configuration
  takes them from its layout and the statement's slot (`answer_ofDraws`, `rankStop_ofDraws`), so it is the same on
  every tape. A session, whose round encoding sees the coins and the cut view
  but not the tape, can stop where the frame does. The check itself reads the tape's nonce, as the session's does, so a
  live frame round asks its openings to pass under that nonce only (`checkIs_live`).
* **The session with the frame's stops** (`RefSession.withStops`): `S` with every message `stop` at the coins and round
  where `Rf` says, every frame round's message past the link without its leading domain frame (`dropDom`: the Lean frame
  keeps a rep's domain for its stream and never releases it), and its `abort` the stop. It keeps `S`'s tables,
  witnesses, dummy and tape, so `Tables`, `Complete`, `DummyReg` and `TapeOK` carry over as they are, and `Causal` and
  `SwCausalOn` where `Rf` reads the coins before its round only (`StopsCausal`). A live run's coins were all taken
  (`refusesCoins_of_live`), so the deferred server asks for the stops below its last round only. Where its stops are the
  frame's refusals (`StopsAt`: on every history of at most the protocol's coins whose openings pass, `Rf` at the coins
  they open is `refusesCoins`), `StopRefine` holds at its switched prover (`LeanRelease.stopRefine_withStops`).
  `LiveRefine` and `FillRefine` hold there from more (`LeanRelease.liveRefine_withStops`, `fillRefine_withStops`): the
  release refuses after any opening that fails (`OpeningsRefused`, from `CheckIs`), no frame round past the last coin
  is live (the session ends at its `Finish`, so the deferred server takes no worker value past it,
  `LeanRelease.live_lt_last`), and the frame's release of each worker round that passed, at the deferred release's
  own histories (`LiveRoundEqOwn`, which `LiveRoundEq` gives, `LiveRoundEq.own`), and of each scheduled round in a
  failed worker's place below the last (`FillRoundEq`), is the session's message without its leading domain frame.
  `HelloRefine` and `OwnRefine` hold there where the server's `Hello` is the session's (`HelloEq`, `helloRefine_withStops`) and the ZK server's own link and `Finish` are the
  session's link and last message (`LinkRoundEq`, `FinalRoundEq`, `ownRefine_withStops`): the per-round byte equality.
  So `Flock.zk_server_lean` holds at that session from those alone (`Flock.zk_server_lean_stops`), with no part of
  `ItemsRefine` assumed. `FrameRounds` gives `LiveRoundEq` and `FillRoundEq` wherever the device's frames are the
  session's (`LeanRelease.liveRoundEq_of_frames`, `liveRoundEq_of_honest`, `fillRoundEq_of_frames`).
* **The stops of a transport that reads only the coins** (`LeanRelease.stopsAt_of_readsCoins`): where, on the histories
  its check passes, the round's rep's state and the coins handed to the frame are functions of the coins the openings
  open (`ReadsCoins`), the frame's refusal at those coins or at a prefix of them (`refuseOf` at any samples and the rep's
  mask slot, `stopOf`: a
  refusal stops the session for good) is a stop rule (`rfOf`) that reads the coins before its round only
  (`stopsCausal_rfOf`) and, where the check is the session's, is the release's refusals (`StopsAt`).
* **At the deployed coin layer** (`Flock.zk_server_lean_coinLayer`, at SHA-512): `S` is `CoinLayer.session nr T`,
  whose `Hello`, opening check and coin reading are `coin-tree/hm96-sha512/v2`'s and whose tables are `T`'s. Its
  premises on the transport are the transport's trusts alone, each a named hypothesis for a stated reason: the
  server's `Hello` is `flock-firewall hello`'s at the statement's files (`LeanRelease.HelloIs`, which gives
  `LeanRelease.HelloEq` by `helloEq_of_helloIs`: that command's `Hello` is the coin layer's), the ZK server's own link and
  `Finish` are `T`'s (`LinkRoundEq`, `FinalRoundEq`; the link's roots are `Flock.Firewall.Link`'s, and at the
  deployed tables `LinkRoundEq` follows from the transport handing it the tape's tables, `LinkRecheck`; the `Finish`'s
  proof files are the session's Lean finish's, `Flock.Firewall.Finish`), the transport
  requests each opening at the committed schedule's place (`ReqsFollow`, `CoinOpen.Follows`) and hands the frame a rep
  state and coins read from the coins the openings open (`ReadsCoins`: the state, `ProgReads`, the session loop's rep
  frame's, `Flock.Firewall.Frames`; the
  coins, `CoinsRead`), the `Finish` is one past the
  last coin, and the coin check reads the session's `Hello` with the tape's nonce (`HelloParses`). At the session loop
  (`flock-firewall session`, `Claims/ZK/SessionLean.lean`) `HelloIs`, `HelloParses`, `ReqsFollow`, `CoinsRead` and the
  `Finish`'s round are theorems; under `coin_tree::Checked` `ReqsFollow` is a trust, and `HelloParses` follows from `HelloEq` and
  `HelloDParses` (`helloParses_of_helloEq`). From them `CheckIs` is proved (`LeanRelease.checkIs_coinLayer`, by
  `CoinOpen.coinLayer_verdict`), and the
  stop rule and both its Props (`rfOf`, `stopsCausal_rfOf`, `stopsAt_of_readsCoins`). The rest is the tables' side:
  `S`'s Props (its switched causality at the coin vectors `Ok`, `SwCausalOn`, which take in every coin vector the stop
  rule lets through, `StopsOk`) and that `T`'s frame rounds, without a rep's leading domain frame, are the frame's bytes
  (`LiveRoundEqOwn`, `FillRoundEq`). `Claims/ZK/CoinPass.lean` has, at every tape, a history past the link whose answer
  and first opening the check passes (`LeanRelease.opens_coinPass`), so the premises leave the frame rounds a history
  to be live at.
-/

@[expose] public section

namespace Claims.ZKMal

/-- The bytes a domain frame at the head of `b` takes: its 16-byte header and its payload, padded to 16 bytes, at the
length its header reads. -/
def domLen (b : List UInt8) : ℕ :=
  let n := (_root_.Flock.u64OfLE ⟨b.toArray⟩ 8).toNat
  16 + n + (16 - n % 16) % 16

/-- **A round message without its leading domain frame**: a round message `0 :: b` whose first frame is a domain frame
(op `1`) loses that frame; any other message is as it is. A rep's first round carries its stream's domain, which the
Lean frame keeps for its stream and never releases (`dropDoms`). -/
def dropDom : Msg → Msg
  | [] => []
  | t :: b => if t = 0 ∧ b[0]! = 1 then t :: b.drop (domLen b) else t :: b

end Claims.ZKMal

namespace Claims.ZKMal.RefSession

/-- **A stop rule reads the coins before its round only**: `Rf x c i`, at the padded coins `c` and the round `i`, is the
same at every `c'` that agrees with `c` below `i`. -/
def StopsCausal (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) : Prop :=
  ∀ x (c c' : Fin (refK x) → S.Coin) (i : ℕ), (∀ j : Fin (refK x), j.val < i → c j = c' j) → Rf x c i = Rf x c' i

/-- **The session with stops**: `S`, whose message at the coins `c` and round `i` is `stop` where `Rf x c i` (round
`refK x` for the last message, `fin` or `off`), whose frame rounds (`0 < i`, past the link) leave without a leading
domain frame (`dropDom`), as the Lean frame releases them, and whose `abort` is `stop`. -/
def withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg) : RefSession :=
  { S with
    abort := stop
    enc := fun x c i σ => if Rf x c i then stop else if i = 0 then S.enc x c i σ else dropDom (S.enc x c i σ)
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
  show (if Rf x c i = true then stop else if i = 0 then S.enc x c i _ else dropDom (S.enc x c i _)) =
    (if Rf x c' i = true then stop else if i = 0 then S.enc x c' i _ else dropDom (S.enc x c' i _))
  rw [hRf x c c' i h]
  exact congrArg (fun m => if Rf x c' i = true then stop else if i = 0 then m else dropDom m) (hc x wt ω p c c' i h)

theorem withStops_swCausalOn (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (hRf : S.StopsCausal Rf)
    (hc : S.SwCausalOn Ok vs) : (S.withStops Rf stop).SwCausalOn Ok vs := by
  intro x w ν t p1 c c' i h hO hO'
  show (if Rf x c i = true then stop else if i = 0 then S.enc x c i _ else dropDom (S.enc x c i _)) =
    (if Rf x c' i = true then stop else if i = 0 then S.enc x c' i _ else dropDom (S.enc x c' i _))
  rw [hRf x c c' i h]
  exact congrArg (fun m => if Rf x c' i = true then stop else if i = 0 then m else dropDom m)
    (hc x w ν t p1 c c' i h hO hO')

open Classical in
/-- **Where the rule stops the coins, the session's switched message is the stop**, before the last coin and after
it. -/
theorem swMsgAt_withStops_stop (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, FlockSoundness.Discharge.ZkView.Seen (S.C x j).z) →
      ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin)
    (hR : Rf x (S.pad cs) (min cs.length (refK x)) = true) : (S.withStops Rf stop).swMsgAt x wt v t cs = stop := by
  by_cases h : cs.length < refK x
  · rw [Nat.min_eq_left h.le] at hR
    show (if cs.length < refK x then (if Rf x (S.pad cs) cs.length = true then stop else _) else _) = stop
    rw [ite_eq_left h, ite_eq_left hR]
  · rw [Nat.min_eq_right (Nat.le_of_not_lt h)] at hR
    show (if cs.length < refK x then _ else if S.Admits x (S.pad cs) then
        (if Rf x (S.pad cs) (refK x) = true then stop else _) else
        (if Rf x (S.pad cs) (refK x) = true then stop else _)) = stop
    rw [ite_eq_right h]
    by_cases hA : S.Admits x (S.pad cs)
    · rw [ite_eq_left hA, ite_eq_left hR]
    · rw [ite_eq_right hA, ite_eq_left hR]

open Classical in
/-- **Where the rule doesn't stop the coins, the session's switched message at the link and after the last coin is
`S`'s.** -/
theorem swMsgAt_withStops_go (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, FlockSoundness.Discharge.ZkView.Seen (S.C x j).z) →
      ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin)
    (hR : Rf x (S.pad cs) (min cs.length (refK x)) = false) (h0 : cs.length = 0 ∨ refK x ≤ cs.length) :
    (S.withStops Rf stop).swMsgAt x wt v t cs = S.swMsgAt x wt v t cs := by
  by_cases h : cs.length < refK x
  · rw [Nat.min_eq_left h.le] at hR
    have hz : cs.length = 0 := h0.resolve_right (by omega)
    show (if cs.length < refK x then
        (if Rf x (S.pad cs) cs.length = true then stop
          else if cs.length = 0 then S.enc x (S.pad cs) cs.length (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t)
          else dropDom (S.enc x (S.pad cs) cs.length (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t))) else _) =
      (if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t) else _)
    rw [ite_eq_left h, ite_eq_left h, hR, ite_eq_right Bool.false_ne_true, ite_eq_left hz]
  · rw [Nat.min_eq_right (Nat.le_of_not_lt h)] at hR
    show (if cs.length < refK x then _ else if S.Admits x (S.pad cs) then
        (if Rf x (S.pad cs) (refK x) = true then stop
          else S.fin x (S.pad cs) (S.swViewsAt x (S.pad cs) wt (v (S.pad cs)) t)) else
        (if Rf x (S.pad cs) (refK x) = true then stop
          else S.off x (S.pad cs) (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t))) =
      (if cs.length < refK x then _ else if S.Admits x (S.pad cs) then
        S.fin x (S.pad cs) (S.swViewsAt x (S.pad cs) wt (v (S.pad cs)) t)
        else S.off x (S.pad cs) (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t))
    rw [ite_eq_right h, ite_eq_right h, hR, ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true]

open Classical in
/-- **Where the rule doesn't stop the coins, the session's switched message at a frame round is `S`'s without its
leading domain frame.** -/
theorem swMsgAt_withStops_frame (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, FlockSoundness.Discharge.ZkView.Seen (S.C x j).z) →
      ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin)
    (hR : Rf x (S.pad cs) cs.length = false) (h0 : 0 < cs.length) (h : cs.length < refK x) :
    (S.withStops Rf stop).swMsgAt x wt v t cs = dropDom (S.swMsgAt x wt v t cs) := by
  show (if cs.length < refK x then
      (if Rf x (S.pad cs) cs.length = true then stop
        else if cs.length = 0 then S.enc x (S.pad cs) cs.length (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t)
        else dropDom (S.enc x (S.pad cs) cs.length (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t))) else _) =
    dropDom (if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t)
      else _)
  rw [ite_eq_left h, ite_eq_left h, hR, ite_eq_right Bool.false_ne_true, ite_eq_right (by omega)]

/-- The coins at most `K` read back from their padding. -/
theorem ofFn_pad_take (S : RefSession) {K : ℕ} (cs : List S.Coin) (h : cs.length ≤ K) :
    (List.ofFn (S.pad (K := K) cs)).take (min cs.length K) = cs := by
  rw [Nat.min_eq_left h]
  apply List.ext_getElem
  · simp [h]
  · intro i _ h2
    simp [pad, h2]

/-- Two coin vectors that agree below `i` have the same first `i`. -/
theorem ofFn_take_congr {α : Type} {K : ℕ} (c c' : Fin K → α) (i : ℕ) (h : ∀ j : Fin K, j.val < i → c j = c' j) :
    (List.ofFn c).take i = (List.ofFn c').take i := by
  apply List.ext_getElem
  · simp
  · intro k h1 _
    simp only [List.length_take, List.length_ofFn] at h1
    simpa using h ⟨k, by omega⟩ (show k < i by omega)

/-! ## The session with stops at values causal where it computes -/

/-- **`Ok` holds wherever the session with stops `Rf` computes a round**: at every opened prefix of at most the
protocol's coins, padded, whose round the rule doesn't stop, and at every coin vector the tables admit. -/
def StopsOk (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) : Prop :=
  (∀ x (cs : List S.Coin), cs.length ≤ refK x → Rf x (S.pad cs) cs.length = false → Ok x (S.pad cs)) ∧
    ∀ x c, S.Admits x c → Ok x c

/-- Every rule's `StopsOk` holds at every coin vector. -/
theorem stopsOk_true (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) :
    S.StopsOk Rf fun _ _ => True :=
  ⟨fun _ _ _ _ => trivial, fun _ _ _ => trivial⟩

theorem stopsOk_pad (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (hRf : S.StopsCausal Rf) (hOk : S.StopsOk Rf Ok) {x : Msg}
    (cs : List S.Coin) (hl : cs.length ≤ refK x) (c : Fin (refK x) → S.Coin)
    (h₀ : ∀ k : Fin (refK x), k.val < cs.length → S.pad cs k = c k) (hr : Rf x c cs.length = false) :
    Ok x (S.pad cs) :=
  hOk.1 x cs hl ((hRf x _ c _ h₀).trans hr)

/-- The last round, up to the last coin, at which the rule doesn't stop the coins `c` (`0` where it stops every
round). -/
noncomputable def lastGo (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (x : Msg)
    (c : Fin (refK x) → S.Coin) : ℕ :=
  Nat.findGreatest (fun i => Rf x c i = false) (refK x)

theorem lastGo_le (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (x : Msg)
    (c : Fin (refK x) → S.Coin) : S.lastGo Rf x c ≤ refK x :=
  Nat.findGreatest_le _

theorem le_lastGo (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) {x : Msg}
    {c : Fin (refK x) → S.Coin} {i : ℕ} (hi : i ≤ refK x) (h : Rf x c i = false) : i ≤ S.lastGo Rf x c :=
  Nat.le_findGreatest hi h

theorem lastGo_go (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) {x : Msg}
    {c : Fin (refK x) → S.Coin} {i : ℕ} (hi : i ≤ refK x) (h : Rf x c i = false) : Rf x c (S.lastGo Rf x c) = false :=
  Nat.findGreatest_spec (P := fun i => Rf x c i = false) hi h

/-- **The coins `c` cut at the last round the rule doesn't stop** (`lastGo`), padded past it: the coins the session
with stops computes its rounds at before its stop. -/
noncomputable def cutCoins (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (x : Msg)
    (c : Fin (refK x) → S.Coin) : Fin (refK x) → S.Coin :=
  S.pad ((List.ofFn c).take (S.lastGo Rf x c))

/-- The cut coins agree with `c` before every round, up to the last coin, the rule doesn't stop. -/
theorem cutCoins_agree (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) {x : Msg}
    {c : Fin (refK x) → S.Coin} {i : ℕ} (hi : i ≤ refK x) (h : Rf x c i = false) (k : Fin (refK x))
    (hk : k.val < i) : S.cutCoins Rf x c k = c k :=
  S.pad_take c _ k (lt_of_lt_of_le hk (S.le_lastGo Rf hi h))

/-- Where the rule doesn't stop some round of `c`, its cut coins are `Ok`'s (`StopsOk`). -/
theorem cutCoins_ok (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (hRf : S.StopsCausal Rf) (hOk : S.StopsOk Rf Ok) {x : Msg}
    {c : Fin (refK x) → S.Coin} {i : ℕ} (hi : i ≤ refK x) (h : Rf x c i = false) : Ok x (S.cutCoins Rf x c) := by
  have hm : ((List.ofFn c).take (S.lastGo Rf x c)).length = S.lastGo Rf x c := by
    simp only [List.length_take, List.length_ofFn]
    exact Nat.min_eq_left (S.lastGo_le Rf x c)
  refine S.stopsOk_pad Rf Ok hRf hOk _ (hm.trans_le (S.lastGo_le Rf x c)) c (fun k hk => S.pad_take c _ k (hm ▸ hk))
    ?_
  rw [hm]
  exact S.lastGo_go Rf hi h

/-- Where the rule doesn't stop the last message, the cut coins are `c`. -/
theorem cutCoins_last (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) {x : Msg}
    {c : Fin (refK x) → S.Coin} (h : Rf x c (refK x) = false) : S.cutCoins Rf x c = c := by
  have : S.lastGo Rf x c = refK x := le_antisymm (S.lastGo_le Rf x c) (S.le_lastGo Rf le_rfl h)
  unfold cutCoins
  rw [this]
  exact S.pad_ofFn c

/-- Where the rule stops, the session with stops' round is the stop. -/
theorem withStops_enc_stop (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (c : Fin (refK x) → S.Coin) (i : ℕ) (σ : ∀ j, ((S.C x j).T (S.cz x c j)).tab.Cut) :
    (if Rf x c i then stop else (S.withStops Rf stop).enc x c i σ) = (S.withStops Rf stop).enc x c i σ := by
  cases hr : Rf x c i
  · exact ite_eq_right Bool.false_ne_true
  · rw [ite_eq_left rfl]
    show stop = if Rf x c i = true then stop else _
    rw [hr, ite_eq_left rfl]

/-- **Round `i` of the session with stops at coins `c₀` that agree with `c` before `i`, at the switched cut view**: the
stop where the rule stops `c` there, else its round `i` at `c₁`, any coin vector that agrees with `c` before `i`, where
`c₀` and `c₁` are `Ok`'s (`SwCausalOn`). -/
theorem withStops_enc_swCut (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (hRf : S.StopsCausal Rf)
    (hsc : S.SwCausalOn Ok vs) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (t : S.Tape) (c c₀ c₁ : Fin (refK x) → S.Coin)
    (i : ℕ) (h₀ : ∀ k : Fin (refK x), k.val < i → c₀ k = c k)
    (h₁ : Rf x c i = false → (∀ k : Fin (refK x), k.val < i → c₁ k = c k) ∧ Ok x c₀ ∧ Ok x c₁) :
    (S.withStops Rf stop).enc x c₀ i (S.swCutOf x c₀ (S.wt x w) (vs x w ν c₀) t) =
      if Rf x c i then stop else (S.withStops Rf stop).enc x c₁ i (S.swCutOf x c₁ (S.wt x w) (vs x w ν c₁) t) := by
  cases hr : Rf x c i
  · obtain ⟨h1, hO₀, hO₁⟩ := h₁ hr
    rw [ite_eq_right Bool.false_ne_true]
    exact (S.withStops Rf stop).enc_swCutOf_on vs Ok x w ν (S.withStops_swCausalOn Rf stop vs Ok hRf hsc) t c₀ c₁ i
      (fun k hk => (h₀ k hk).trans (h1 k hk).symm) hO₀ hO₁
  · rw [ite_eq_left rfl]
    show (if Rf x c₀ i = true then stop else _) = stop
    rw [hRf x c₀ c i h₀, hr, ite_eq_left rfl]

/-- **The same at the honest cut view** (`Causal`), at any `c₁` that agrees with `c` before `i`. -/
theorem withStops_enc_cut (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (hRf : S.StopsCausal Rf) (hc : S.Causal) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (c c₀ c₁ : Fin (refK x) → S.Coin) (i : ℕ) (h₀ : ∀ k : Fin (refK x), k.val < i → c₀ k = c k)
    (h₁ : Rf x c i = false → ∀ k : Fin (refK x), k.val < i → c₁ k = c k) :
    (S.withStops Rf stop).enc x c₀ i (S.cutOf x c₀ wt t) =
      if Rf x c i then stop else (S.withStops Rf stop).enc x c₁ i (S.cutOf x c₁ wt t) := by
  cases hr : Rf x c i
  · rw [ite_eq_right Bool.false_ne_true]
    exact (S.withStops Rf stop).enc_cutOf x (S.withStops_causal Rf stop hRf hc) wt t c₀ c₁ i
      fun k hk => (h₀ k hk).trans (h₁ hr k hk).symm
  · rw [ite_eq_left rfl]
    show (if Rf x c₀ i = true then stop else _) = stop
    rw [hRf x c₀ c i h₀, hr, ite_eq_left rfl]

/-- The reference prover's message after `n` coins, `n` before the last: round `n`'s encoding at them, padded. -/
theorem msgAt_pre (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape)
    (cs : List S.Coin) {n : ℕ} (hl : cs.length = n) (hn : n < refK x) :
    S.msgAt x wt t cs = S.enc x (S.pad cs) n (S.cutOf x (S.pad cs) wt t) := by
  subst hl
  unfold msgAt
  rw [ite_eq_left hn]

/-- The switched prover's message after `n` coins, `n` before the last: round `n`'s encoding at them, padded. -/
theorem swMsgAt_pre (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → (∀ j, FlockSoundness.Discharge.ZkView.Seen (S.C x j).z) →
      ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) (cs : List S.Coin) {n : ℕ} (hl : cs.length = n) (hn : n < refK x) :
    S.swMsgAt x wt v t cs = S.enc x (S.pad cs) n (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t) := by
  subst hl
  unfold swMsgAt
  rw [ite_eq_left hn]

open Classical in
/-- **The reference prover's message after a prefix of a coin vector, before the last coin**: round `j`'s encoding at
the prefix, padded. -/
theorem msgAt_take_pre (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x) :
    S.msgAt x wt t ((List.ofFn c).take j) =
      S.enc x (S.pad ((List.ofFn c).take j)) j (S.cutOf x (S.pad ((List.ofFn c).take j)) wt t) := by
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  unfold msgAt
  rw [hl, ite_eq_left hj]

open Classical in
/-- **The reference prover's message after a whole coin vector**: `fin` where the tables admit it, else `off`. -/
theorem msgAt_take_last (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) :
    S.msgAt x wt t ((List.ofFn c).take (refK x)) =
      if S.Admits x c then S.fin x c (S.viewsAt x c wt t) else S.off x c (S.cutOf x c wt t) := by
  have hl : ((List.ofFn c).take (refK x)).length = refK x := by simp only [List.length_take, List.length_ofFn]; omega
  unfold msgAt
  rw [hl, ite_eq_right (Nat.lt_irrefl _), S.pad_ofFn c]

/-- **The session with stops' switched message after a prefix of `c`, before the last coin**, at values causal on `Ok`
(`SwCausalOn`, `StopsOk`): the stop where the rule stops `c` at round `j`, else round `j` at `c₁`, any coin vector `Ok`
admits that agrees with `c` before `j`. -/
theorem withStops_swMsgAt_pre (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (hRf : S.StopsCausal Rf)
    (hsc : S.SwCausalOn Ok vs) (hOk : S.StopsOk Rf Ok) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (t : S.Tape)
    (c c₁ : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x)
    (h₁ : Rf x c j = false → (∀ k : Fin (refK x), k.val < j → c₁ k = c k) ∧ Ok x c₁) :
    (S.withStops Rf stop).swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j) =
      if Rf x c j then stop else (S.withStops Rf stop).enc x c₁ j (S.swCutOf x c₁ (S.wt x w) (vs x w ν c₁) t) := by
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  have h₀ : ∀ k : Fin (refK x), k.val < j → S.pad ((List.ofFn c).take j) k = c k := fun k hk => S.pad_take c j k hk
  refine ((S.withStops Rf stop).swMsgAt_take_pre x (S.wt x w) (vs x w ν) t c j hj).trans ?_
  exact S.withStops_enc_swCut Rf stop vs Ok hRf hsc x w ν t c _ c₁ j h₀ fun hr =>
    ⟨(h₁ hr).1, S.stopsOk_pad Rf Ok hRf hOk ((List.ofFn c).take j) (hl.trans_le hj.le) c
      (fun k hk => h₀ k (lt_of_lt_of_eq hk hl)) ((congrArg (Rf x c) hl).trans hr), (h₁ hr).2⟩

/-- **The session with stops' message after a prefix of `c`, before the last coin** (`Causal`): the stop where the rule
stops `c` at round `j`, else round `j` at any `c₁` that agrees with `c` before `j`. -/
theorem withStops_msgAt_pre (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (hRf : S.StopsCausal Rf) (hc : S.Causal) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (t : S.Tape) (c c₁ : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x)
    (h₁ : Rf x c j = false → ∀ k : Fin (refK x), k.val < j → c₁ k = c k) :
    (S.withStops Rf stop).msgAt x wt t ((List.ofFn c).take j) =
      if Rf x c j then stop else (S.withStops Rf stop).enc x c₁ j (S.cutOf x c₁ wt t) := by
  refine ((S.withStops Rf stop).msgAt_take_pre x wt t c j hj).trans ?_
  exact S.withStops_enc_cut Rf stop hRf hc x wt t c _ c₁ j (fun k hk => S.pad_take c j k hk) h₁

/-- **The stopped round at the cut coins**: the stop where the rule stops `c` at round `i`, else the session with
stops' round `i` at `c`'s cut coins (`cutCoins`). -/
noncomputable def encCut (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (x : Msg) (c : Fin (refK x) → S.Coin) (i : ℕ)
    (σ : ∀ j, ((S.C x j).T (S.cz x (S.cutCoins Rf x c) j)).tab.Cut) : Msg :=
  if Rf x c i then stop else (S.withStops Rf stop).enc x (S.cutCoins Rf x c) i σ

/-- **Before the last coin, the session with stops' switched prover sends the stopped round at the cut coins of what
the answers open** (`SwCausalOn`, `StopsOk`). -/
theorem swPre_msg_stops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (H : Msg → Msg) (x : Msg) (w : Flock.Wit)
    (ν : S.Nonce) (hRf : S.StopsCausal Rf) (hsc : S.SwCausalOn Ok vs) (hOk : S.StopsOk Rf Ok) (t : S.Tape)
    (a : Fin (refK x + 1) → Out) (i : Fin (refK x + 1)) :
    (S.withStops Rf stop).refP.pmG H x ν
        (fun cs => .ret ((S.withStops Rf stop).swMsgAt x (S.wt x w) (vs x w ν) t cs)) (ansHist a i) =
      (S.withStops Rf stop).refP.pmG H x ν (fun _ => .ret (S.encCut Rf stop x (S.coinsOf a) (i.val - 1)
        (S.swCutOf x (S.cutCoins Rf x (S.coinsOf a)) (S.wt x w) (vs x w ν (S.cutCoins Rf x (S.coinsOf a))) t)))
        (ansHist a i) := by
  unfold Proto.pmG
  congr 2
  refine strat_congr_gen _ _ _ _ _ _ _ fun hne => ?_
  have hi : 0 < i.val := by
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · exact absurd (by simp [ansHist, h0]) hne
    · exact h0
  have hl : ((ansHist a i).tail.map fun p => S.decOf p.2).length = i.val - 1 := by
    simp only [List.length_map, List.length_tail, ansHist, List.length_ofFn]
  have hlt : i.val - 1 < refK x := by omega
  have h₀ : ∀ k : Fin (refK x), k.val < i.val - 1 →
      S.pad ((ansHist a i).tail.map fun p => S.decOf p.2) k = S.coinsOf a k := fun k hk => S.pad_ansHist a i k hk
  show OC.ret ((S.withStops Rf stop).swMsgAt x (S.wt x w) (vs x w ν) t
    ((ansHist a i).tail.map fun p => S.decOf p.2)) = OC.ret _
  exact congrArg OC.ret (((S.withStops Rf stop).swMsgAt_pre x (S.wt x w) (vs x w ν) t _ hl hlt).trans
    (S.withStops_enc_swCut Rf stop vs Ok hRf hsc x w ν t (S.coinsOf a) _ _ (i.val - 1) h₀
      fun hr => ⟨fun k hk => S.cutCoins_agree Rf hlt.le hr k hk,
        S.stopsOk_pad Rf Ok hRf hOk _ (hl.trans_le hlt.le) _ (hl ▸ h₀) (hl ▸ hr),
        S.cutCoins_ok Rf Ok hRf hOk hlt.le hr⟩))

/-- **Before the last coin, the session with stops' prover sends the stopped round at the cut coins of what the answers
open** (`Causal`). -/
theorem pre_msg_stops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (H : Msg → Msg) (x : Msg) (ν : S.Nonce) (hRf : S.StopsCausal Rf) (hc : S.Causal)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape) (a : Fin (refK x + 1) → Out)
    (i : Fin (refK x + 1)) :
    (S.withStops Rf stop).refP.pmG H x ν (fun cs => .ret ((S.withStops Rf stop).msgAt x wt t cs)) (ansHist a i) =
      (S.withStops Rf stop).refP.pmG H x ν (fun _ => .ret (S.encCut Rf stop x (S.coinsOf a) (i.val - 1)
        (S.cutOf x (S.cutCoins Rf x (S.coinsOf a)) wt t))) (ansHist a i) := by
  unfold Proto.pmG
  congr 2
  refine strat_congr_gen _ _ _ _ _ _ _ fun hne => ?_
  have hi : 0 < i.val := by
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · exact absurd (by simp [ansHist, h0]) hne
    · exact h0
  have hl : ((ansHist a i).tail.map fun p => S.decOf p.2).length = i.val - 1 := by
    simp only [List.length_map, List.length_tail, ansHist, List.length_ofFn]
  have hlt : i.val - 1 < refK x := by omega
  show OC.ret ((S.withStops Rf stop).msgAt x wt t ((ansHist a i).tail.map fun p => S.decOf p.2)) = OC.ret _
  exact congrArg OC.ret (((S.withStops Rf stop).msgAt_pre x wt t _ hl hlt).trans
    (S.withStops_enc_cut Rf stop hRf hc x wt t (S.coinsOf a) _ _ (i.val - 1)
      (fun k hk => S.pad_ansHist a i k hk) fun hr k hk => S.cutCoins_agree Rf hlt.le hr k hk))

/-- **The switched model before the last coin, at the cut coins**: `swPreModel` of the session with stops, its tables
and its stopped round at the cut coins of what the answers open (`encCut`). -/
noncomputable def swPreCut (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (ν : S.Nonce) : ((S.withStops Rf stop).swP vs).TPreModel x where
  J := S.J x
  C := S.C x
  cz a := S.cz x (S.cutCoins Rf x (S.coinsOf a))
  out a σ i := (S.withStops Rf stop).refP.pmG H x ν
    (fun _ => .ret (S.encCut Rf stop x (S.coinsOf a) (i.val - 1) σ)) (ansHist a i)
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

/-- **`swPreCut`'s link**: before the last coin the session with stops' switched prover is the cut model's, from the
session's causality, the switched causality on `Ok` (`SwCausalOn`) and `StopsOk`. -/
theorem swPreCut_link (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (H : Msg → Msg) (R : Rel Flock.Wit)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (hRf : S.StopsCausal Rf) (hsc : S.SwCausalOn Ok vs)
    (hOk : S.StopsOk Rf Ok) (hT : S.TapeOK) (x : Msg) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) :
    (S.swPreCut Rf stop vs H x ν).SwLink H ν w := by
  refine ⟨hV x w hR, fun a j => hD x _ j, ?_, fun p => (hT x).1 p _,
    ⟨fun a => vs x w ν (S.cutCoins Rf x (S.coinsOf a)), ?_,
      fun t a => funext fun i => S.swPre_msg_stops Rf stop vs Ok H x w ν hRf hsc hOk t a i⟩,
    fun t a => funext fun i => S.pre_msg_stops Rf stop H x ν hRf hc (S.wt₀ x) t a i⟩
  · intro wt ω p a a' i h
    show (S.withStops Rf stop).refP.pmG H x ν _ (ansHist a i) = (S.withStops Rf stop).refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => (S.withStops Rf stop).refP.pmG H x ν g (ansHist a' i))
      (funext fun _ => congrArg OC.ret ?_)
    have hag : ∀ k : Fin (refK x), k.val < i.val - 1 → S.coinsOf a k = S.coinsOf a' k := fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))
    have hi : i.val - 1 ≤ refK x := by
      have : i.val < refK x + 1 := i.isLt
      omega
    unfold encCut
    rw [hRf x (S.coinsOf a) (S.coinsOf a') (i.val - 1) hag]
    cases hr : Rf x (S.coinsOf a') (i.val - 1)
    · rw [ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true]
      have hr' : Rf x (S.coinsOf a) (i.val - 1) = false := (hRf x _ _ _ hag).trans hr
      exact (S.withStops_causal Rf stop hRf hc) x wt ω p _ _ (i.val - 1) fun k hk =>
        ((S.cutCoins_agree Rf hi hr' k hk).trans (hag k hk)).trans (S.cutCoins_agree Rf hi hr k hk).symm
    · rw [ite_eq_left rfl, ite_eq_left rfl]
  · intro ω y0 p1 a a' i h
    obtain ⟨t, rfl, ht⟩ := S.exists_tape x (hT x).1 ω y0
    obtain rfl : y0 = fun l0 => (S.eP x t).2 ((leaves01 (fun j => (S.C x j).z)).symm (.inl l0)) := (funext ht).symm
    show (S.withStops Rf stop).refP.pmG H x ν _ (ansHist a i) = (S.withStops Rf stop).refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => (S.withStops Rf stop).refP.pmG H x ν g (ansHist a' i))
      (funext fun _ => congrArg OC.ret ?_)
    have hag : ∀ k : Fin (refK x), k.val < i.val - 1 → S.coinsOf a k = S.coinsOf a' k := fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))
    have hi : i.val - 1 ≤ refK x := by
      have : i.val < refK x + 1 := i.isLt
      omega
    unfold encCut
    rw [hRf x (S.coinsOf a) (S.coinsOf a') (i.val - 1) hag]
    cases hr : Rf x (S.coinsOf a') (i.val - 1)
    · rw [ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true]
      have hr' : Rf x (S.coinsOf a) (i.val - 1) = false := (hRf x _ _ _ hag).trans hr
      exact (S.withStops_swCausalOn Rf stop vs Ok hRf hsc) x w ν t p1 _ _ (i.val - 1)
        (fun k hk => ((S.cutCoins_agree Rf hi hr' k hk).trans (hag k hk)).trans (S.cutCoins_agree Rf hi hr k hk).symm)
        (S.cutCoins_ok Rf Ok hRf hOk hi hr') (S.cutCoins_ok Rf Ok hRf hOk hi hr)
    · rw [ite_eq_left rfl, ite_eq_left rfl]

/-- **The switched model where the prover refuses at `c`, at the cut coins**: the session with stops' off-model, its
tables at `c`'s cut coins, its rounds the stopped rounds there (`encCut`) and its last message the stop where the rule
stops `c` there, else `off` at the cut coins. -/
noncomputable def swOffCut (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) : ((S.withStops Rf stop).swP vs).TOffModel x where
  J := S.J x
  C := S.C x
  cz := S.cz x (S.cutCoins Rf x c)
  out σ := (List.range (refK x + 1)).map fun j =>
    if j < refK x then S.encCut Rf stop x c j σ else if Rf x c (refK x) then stop else S.off x (S.cutCoins Rf x c) σ
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

/-- **`swOffCut`'s link** where the tables don't admit `c`. -/
theorem swOffCut_link (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (H : Msg → Msg) (R : Rel Flock.Wit)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (hRf : S.StopsCausal Rf) (hsc : S.SwCausalOn Ok vs)
    (hOk : S.StopsOk Rf Ok) (hT : S.TapeOK) (x : Msg) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce)
    (c : Fin (refK x) → S.Coin) (hA : ¬ (S.withStops Rf stop).Admits x c) :
    (S.swOffCut Rf stop vs x c).SwLink H ν c w := by
  refine ⟨hV x w hR, fun j => hD x _ j, fun p => (hT x).1 p _,
    ⟨vs x w ν (S.cutCoins Rf x c), fun (t : S.Tape) => ?_⟩, fun (t : S.Tape) => ?_⟩
  · show (List.range (refK x + 1)).map (fun j =>
          (S.withStops Rf stop).swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j)) =
        (List.range (refK x + 1)).map fun j => if j < refK x then
          S.encCut Rf stop x c j (S.swCutOf x (S.cutCoins Rf x c) (S.wt x w) (vs x w ν (S.cutCoins Rf x c)) t)
          else if Rf x c (refK x) then stop else
            S.off x (S.cutCoins Rf x c) (S.swCutOf x (S.cutCoins Rf x c) (S.wt x w) (vs x w ν (S.cutCoins Rf x c)) t)
    refine List.map_congr_left fun j hj => ?_
    by_cases hlt : j < refK x
    · rw [ite_eq_left hlt]
      exact S.withStops_swMsgAt_pre Rf stop vs Ok hRf hsc hOk x w ν t c _ j hlt fun hr =>
        ⟨fun k hk => S.cutCoins_agree Rf hlt.le hr k hk, S.cutCoins_ok Rf Ok hRf hOk hlt.le hr⟩
    · obtain rfl : j = refK x := by have := List.mem_range.1 hj; omega
      rw [ite_eq_right hlt]
      refine ((S.withStops Rf stop).swMsgAt_take_last x (S.wt x w) (vs x w ν) t c).trans ?_
      rw [ite_eq_right hA]
      show (if Rf x c (refK x) = true then stop else S.off x c (S.swCutOf x c (S.wt x w) (vs x w ν c) t)) =
        (if Rf x c (refK x) = true then stop else
          S.off x (S.cutCoins Rf x c) (S.swCutOf x (S.cutCoins Rf x c) (S.wt x w) (vs x w ν (S.cutCoins Rf x c)) t))
      cases hr : Rf x c (refK x)
      · rw [ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true, S.cutCoins_last Rf hr]
      · rw [ite_eq_left rfl, ite_eq_left rfl]
  · show (List.range (refK x + 1)).map (fun j => (S.withStops Rf stop).msgAt x (S.wt₀ x) t ((List.ofFn c).take j)) =
        (List.range (refK x + 1)).map fun j => if j < refK x then
          S.encCut Rf stop x c j (S.cutOf x (S.cutCoins Rf x c) (S.wt₀ x) t)
          else if Rf x c (refK x) then stop else S.off x (S.cutCoins Rf x c) (S.cutOf x (S.cutCoins Rf x c) (S.wt₀ x) t)
    refine List.map_congr_left fun j hj => ?_
    by_cases hlt : j < refK x
    · rw [ite_eq_left hlt]
      exact S.withStops_msgAt_pre Rf stop hRf hc x (S.wt₀ x) t c _ j hlt fun hr k hk =>
        S.cutCoins_agree Rf hlt.le hr k hk
    · obtain rfl : j = refK x := by have := List.mem_range.1 hj; omega
      rw [ite_eq_right hlt]
      refine ((S.withStops Rf stop).msgAt_take_last x (S.wt₀ x) t c).trans ?_
      rw [ite_eq_right hA]
      show (if Rf x c (refK x) = true then stop else S.off x c (S.cutOf x c (S.wt₀ x) t)) =
        (if Rf x c (refK x) = true then stop else S.off x (S.cutCoins Rf x c) (S.cutOf x (S.cutCoins Rf x c) (S.wt₀ x) t))
      cases hr : Rf x c (refK x)
      · rw [ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true, S.cutCoins_last Rf hr]
      · rw [ite_eq_left rfl, ite_eq_left rfl]

/-- **The session with stops' switched message link at each coin vector**, at values causal on `Ok`: where its tables
admit `c`, `swCoinModel`'s own (`c` is `Ok`'s, `StopsOk`); where they don't, `swOffCut` at the cut coins. -/
theorem swCoinModel_link_stops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg)
    (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (H : Msg → Msg) (R : Rel Flock.Wit)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (hRf : S.StopsCausal Rf) (hsc : S.SwCausalOn Ok vs)
    (hOk : S.StopsOk Rf Ok) (hT : S.TapeOK) (x : Msg) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce)
    (c : Fin (refK x) → S.Coin) : ((S.withStops Rf stop).swCoinModel vs x c).SwLink H ν c w := by
  unfold swCoinModel
  split_ifs with hA
  · refine ⟨vs x w ν c, hV x w hR, fun p => (hT x).2 c p _, fun (t : S.Tape) => ?_⟩
    refine Eq.trans ?_ (congrArg (fun V => (List.range (refK x + 1)).map fun j =>
      if j < refK x then (S.withStops Rf stop).enc x c j ((S.withStops Rf stop).cutV x c V)
      else (S.withStops Rf stop).fin x c V)
      ((S.withStops Rf stop).swViewsAt_eq x c (S.wt x w) (vs x w ν c) t))
    show (List.range (refK x + 1)).map (fun j =>
        (S.withStops Rf stop).swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then (S.withStops Rf stop).enc x c j
          ((S.withStops Rf stop).cutV x c ((S.withStops Rf stop).swViewsAt x c (S.wt x w) (vs x w ν c) t))
        else (S.withStops Rf stop).fin x c ((S.withStops Rf stop).swViewsAt x c (S.wt x w) (vs x w ν c) t)
    refine List.map_congr_left fun j hj => ?_
    by_cases hlt : j < refK x
    · rw [ite_eq_left hlt]
      refine ((S.withStops_swMsgAt_pre Rf stop vs Ok hRf hsc hOk x w ν t c c j hlt fun _ =>
        ⟨fun _ _ => rfl, hOk.2 x c hA⟩).trans (S.withStops_enc_stop Rf stop x c j _)).trans ?_
      exact congrArg ((S.withStops Rf stop).enc x c j)
        ((S.withStops Rf stop).cutV_swViewsAt x c (S.wt x w) (vs x w ν c) t).symm
    · obtain rfl : j = refK x := by have := List.mem_range.1 hj; omega
      rw [ite_eq_right hlt]
      refine ((S.withStops Rf stop).swMsgAt_take_last x (S.wt x w) (vs x w ν) t c).trans ?_
      rw [ite_eq_left hA]
  · exact ⟨S.swOffCut Rf stop vs x c, S.swOffCut_link Rf stop vs Ok H R hV hD hc hRf hsc hOk hT x w hR ν c hA, id,
      le_rfl⟩

end Claims.ZKMal.RefSession

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Flock's zero knowledge at the session with stops, at values causal where it computes, at any masked record whose
rounds on its relation are the session's** (`refK x + 1`): `Flock.zkAt_sw` at `S.withStops Rf stop` with `refP`'s
simulator there, its switched round encoding causal only on `Ok` (`RefSession.SwCausalOn`), which holds wherever the
session with stops computes a round (`RefSession.StopsOk`). Before the last coin its prover is the cut model's
(`RefSession.swPreCut_link`); at each coin vector, where its tables admit it, `swCoinModel`'s own, else the off-model at
the cut coins (`RefSession.swCoinModel_link_stops`); at the record with those rounds at every statement, carried back
by `Flock.zkAt_withK`. -/
theorem Flock.zkAt_sw_stops {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (hzk : D.zk = true)
    (hkD : ∀ x w, Flock.R D.parse x w → D.V.k x = refK x + 1) (S : RefSession)
    (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg) (vs : S.SwVals)
    (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R D.parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hRf : S.StopsCausal Rf) (hsc : S.SwCausalOn Ok vs) (hOk : S.StopsOk Rf Ok) (hT : S.TapeOK) :
    Claims.Flock.ZKAt H D ((S.withStops Rf stop).swP vs).prover
      (((S.withStops Rf stop).refP.withSim fun x _ c => (S.withStops Rf stop).coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * (S.withStops Rf stop).nLeaves x * δ₁.toReal +
        8 * t * (S.withStops Rf stop).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zkAt_withK (fun x w hR => (hkD x w hR).symm)
    (Flock.zkAt_sw_of_links H (D.withK fun x => refK x + 1) hzk (S.withStops Rf stop) vs (fun _ => rfl) t ht hT1 hδ₁
      (S.withStops_tables Rf stop hS) (fun x ν => S.swPreCut Rf stop vs H x ν) (fun _ _ => le_rfl) (fun x _ => hS x)
      (fun x w hR ν => S.swPreCut_link Rf stop vs Ok H _ hV hD hc hRf hsc hOk hT x w hR ν)
      fun x w hR ν c => S.swCoinModel_link_stops Rf stop vs Ok H _ hV hD hc hRf hsc hOk hT x w hR ν c)

/-- **Flock's zero knowledge at the session with stops, at values causal where it computes**: `Flock.zkAt_sw_stops` at
`deployZk`, whose rounds on its relation are the session's (`V_k_of_R`). -/
theorem Flock.zk_deployZk_sw_stops (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession)
    (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (stop : Msg) (vs : S.SwVals)
    (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hRf : S.StopsCausal Rf) (hsc : S.SwCausalOn Ok vs) (hOk : S.StopsOk Rf Ok) (hT : S.TapeOK) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) ((S.withStops Rf stop).swP vs).prover
      (((S.withStops Rf stop).refP.withSim fun x _ c => (S.withStops Rf stop).coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * (S.withStops Rf stop).nLeaves x * δ₁.toReal +
        8 * t * (S.withStops Rf stop).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zkAt_sw_stops H _ rfl (fun _ _ hR => V_k_of_R hR) S Rf stop vs Ok t ht hT1 hδ₁ hS hV hD hc hRf hsc hOk hT

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock.Firewall.Frame (Cfg Masker Draws MaskSlot)

/-- **The frame's coin refusal reads its configuration's layout and absorbed order alone**, never its pads or root. -/
theorem answer_congr {c c' : Cfg} (s : Masker) (cs : Array _root_.Flock.F128) (hl : c.lay = c'.lay)
    (ho : c.order = c'.order) : s.answer c cs = s.answer c' cs := by
  unfold Masker.answer
  rw [hl, ho]

/-- **At a rep's configuration the frame's coin refusal reads its layout alone**, whatever its samples and mask slot. -/
theorem answer_ofDraws (s : Masker) (cs : Array _root_.Flock.F128) (l : _root_.Flock.Zk.PadLayout) (d d' : Draws)
    (m m' : Option MaskSlot) : s.answer (Cfg.ofDraws l d m) cs = s.answer (Cfg.ofDraws l d' m') cs :=
  answer_congr s cs rfl rfl

/-- At a rep's configuration the rank stop reads its layout and mask slot alone, whatever its samples. -/
theorem rankStop_ofDraws (l : _root_.Flock.Zk.PadLayout) (d d' : Draws) (m : Option MaskSlot) :
    LeanRelease.rankStop (Cfg.ofDraws l d m) = LeanRelease.rankStop (Cfg.ofDraws l d' m) :=
  funext fun s => LeanRelease.rankStop_congr s rfl rfl rfl

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
  intro ρ κ h _ _ hr
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
live frame round of the deferred release's own run whose last coins the frame takes, where some opening so far fails
the coin-tree check, the frame's release of the worker's round is the switched reference prover's `abort`. -/
theorem release_eq_abort_of_liveRefine (S : RefSession) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hL : L.LiveRefine H (S.swP vs).prover wk x w) (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (hk : L.kindAt x ((m₀, ρ₀) :: ops).length = .masked)
    (hl : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hr : L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hown : L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops))
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false))
    (hok : ((allOk ((S.swP vs).chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = false) :
    L.release x (ν, t) ((m₀, ρ₀) :: ops) false v = S.abort := by
  have he := hL (ν, t) κ ((m₀, ρ₀) :: ops) v hk hl hr hown hv
  have hm : ((strat ((S.swP vs).hello x ν) (S.swP vs).abort (S.swP vs).dec ((S.swP vs).chk x ν)
      ((S.swP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = S.abort := by
    rw [strat_eval_cons, hok]
    rfl
  exact he.symm.trans hm

/-- Who takes the coins reads the configurations' layouts alone, and the next round's rep's mask slot (`rankStop`). -/
theorem takeAt_ofDraws (n : ℕ) (b : Bool) (lN lP : _root_.Flock.Zk.PadLayout) (dN dN' dP dP' : Draws)
    (mN mP mP' : Option MaskSlot) (s : Masker) (f : Except String Masker) (cs : Array _root_.Flock.F128) :
    takeAt n b (Cfg.ofDraws lN dN mN) (Cfg.ofDraws lP dP mP) s f cs =
      takeAt n b (Cfg.ofDraws lN dN' mN) (Cfg.ofDraws lP dP' mP') s f cs := by
  unfold takeAt
  rw [answer_ofDraws s cs lN dN dN' mN mN, answer_ofDraws s cs lP dP dP' mP mP', rankStop_ofDraws lN dN dN' mN]

/-- The frame's own taking of the coins is the same on every tape. -/
theorem frameAnswer_tape {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ ρ' : R) (h : List (Msg × Msg)) :
    L.frameAnswer x ρ h = L.frameAnswer x ρ' h := by
  unfold frameAnswer cfg cfgAt
  cases h.getLast? with
  | none => rfl
  | some _ => exact takeAt_ofDraws _ _ _ _ _ _ _ _ _ _ _ _ _ _

theorem tookEarlier_tape {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ ρ' : R) (h : List (Msg × Msg)) :
    L.tookEarlier x ρ h = L.tookEarlier x ρ' h :=
  List.all_congr rfl fun _ => congrArg Except.isOk (L.frameAnswer_tape x ρ ρ' _)

/-- On the histories the transport's coin check passes at both tapes, the coins the frame takes are the same. -/
theorem answered_tape {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ ρ' : R) (h : List (Msg × Msg))
    (ho : L.opens x ρ h = true) (ho' : L.opens x ρ' h = true) : L.answered x ρ h = L.answered x ρ' h := by
  unfold answered
  rw [ho, ho', L.tookEarlier_tape x ρ ρ' h, L.frameAnswer_tape x ρ ρ' h]

/-- **The coin stops read no tape** past the transport's check: on a history the check passes at both tapes, the frame
refuses the coins after it on both or on neither. The check reads the tape's nonce (`opens`), so off those histories
the release refuses where it does. -/
theorem refusesCoins_tape {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ ρ' : R) (h : List (Msg × Msg))
    (ho : L.opens x ρ h = true) (ho' : L.opens x ρ' h = true) : L.refusesCoins x ρ h = L.refusesCoins x ρ' h := by
  unfold refusesCoins
  rw [L.answered_tape x ρ ρ' h ho ho']

/-- Where the frame takes a device round after `h`, it took the stream's last coins. -/
theorem refusesCoins_of_takes {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (v : Msg)
    (ht : L.takes x ρ h v = true) : L.refusesCoins x ρ h = false := by
  unfold refusesCoins
  cases ha : L.answered x ρ h with
  | ok _ => rfl
  | error _ =>
    unfold takes step at ht
    rw [ha] at ht
    change false = true at ht
    cases ht

/-- **A live run's coins were all taken**: while the worker's run is live, the frame took the frame round, so it
refused none of the stream's last coins. -/
theorem refusesCoins_of_live {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hl : L.toSrv.live wk x w ρ κ h = true) : L.refusesCoins x ρ h = false := by
  obtain ⟨v, -, -, -, ht⟩ := L.live_some wk x w ρ κ h hk hl
  exact L.refusesCoins_of_takes x ρ h v ht

/-- **The rule's stops are the frame's refusals** at `x`: after every history past `Hello` of at most the protocol's
`refK x` openings, all passing the coin-tree check, `Rf` at the coins they open, padded, at their round, is whether the
release refuses the coins (`refusesCoins`: the frame refused the last coins or an earlier round's). After an opening
that fails, the prover sends its `abort`, so the rule need not say there;
past the last coin no frame round is live (`LeanRelease.live_lt_last`). -/
def StopsAt (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (x : Msg) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)), ops.length ≤ refK x →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) =
      Rf x (S.pad (ops.map fun p => S.decOf p.2)) (min (ops.map fun p => S.decOf p.2).length (refK x))

/-- **`StopRefine` at the session with the frame's stops**, at any worker and values and a last round `n x` past the
protocol's coins: its switched prover's message is the stop wherever the frame refuses the coins. A live run's coins
were all taken (`refusesCoins_of_live`), so only the rounds below the last ask, and there the rule is the frame's. -/
theorem stopRefine_withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hn : n x ≤ refK x + 2) (hR : StopsAt S Rf L H x) :
    L.StopRefine H ((S.withStops Rf L.stop).swP vs).prover n wk x w := by
  intro ρ κ h hk hc hr
  obtain ⟨ν, t⟩ := ρ
  cases h with
  | nil =>
    rw [refusesCoins_nil] at hr
    cases hr
  | cons p ops =>
    obtain ⟨m₀, ρ₀⟩ := p
    have hlen : ops.length ≤ refK x := by
      rcases hc with hc | hc
      · simp only [List.length_cons] at hc
        omega
      · rw [L.refusesCoins_of_live wk x w _ κ _ hk hc] at hr
        cases hr
    show ((strat (((S.withStops Rf L.stop).swP vs).hello x ν) ((S.withStops Rf L.stop).swP vs).abort
      ((S.withStops Rf L.stop).swP vs).dec (((S.withStops Rf L.stop).swP vs).chk x ν)
      (((S.withStops Rf L.stop).swP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = L.stop
    rw [strat_eval_cons]
    split
    next hok =>
      show (S.withStops Rf L.stop).swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) = L.stop
      exact S.swMsgAt_withStops_stop Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
        ((hR ν t m₀ ρ₀ ops hlen hok).symm.trans hr)
    next => rfl

/-- **The transport's coin check is the session's** at `x`: after every history past `Hello` of at most the protocol's
`refK x` openings, the release's check (`opens`: `flock-firewall coins` on the server's `Hello` at the tape's nonce,
its answer and the openings) passes exactly where the session's check passes every opening under that nonce. It reads
the tape's nonce as the session's check does. Proved at the deployed coin layer (`checkIs_coinLayer`). -/
def CheckIs (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (x : Msg) :
    Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)), ops.length ≤ refK x →
    L.opens x (ν, t) ((m₀, ρ₀) :: ops) = ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1

/-- **The release refuses a failed opening** at `x`: after every history past `Hello` of at most `refK x` openings,
some opening of which fails the session's check under the tape's nonce, the release refuses the last coins. The frame
checks no opening itself; the transport's check does (`opens`), and the release is the stop past its refusal
(`refusesCoins_of_not_opens`). So it follows from that check being the session's (`openingsRefused_of_checkIs`). -/
def OpeningsRefused (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg)
    (x : Msg) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)), ops.length ≤ refK x →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = false →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = true

/-- **`OpeningsRefused` from the transport's check being the session's.** -/
theorem openingsRefused_of_checkIs (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (x : Msg) (hC : CheckIs S L H x) : OpeningsRefused S L H x :=
  fun ν t m₀ ρ₀ ops hl hf => L.refusesCoins_of_not_opens x (ν, t) _ ((hC ν t m₀ ρ₀ ops hl).trans hf)

/-- **A live frame round's openings pass under the tape's nonce**, where the transport's check is the session's: a live
run's coins were all taken (`refusesCoins_of_live`), so the check passed the history, at the tape's own nonce only. -/
theorem checkIs_live (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hC : CheckIs S L H x) (ν : S.Nonce)
    (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hlen : ops.length ≤ refK x)
    (hk : L.kindAt x ((m₀, ρ₀) :: ops).length = .masked)
    (hl : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) :
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true := by
  rw [← hC ν t m₀ ρ₀ ops hlen]
  cases ho : L.opens x (ν, t) ((m₀, ρ₀) :: ops) with
  | true => rfl
  | false =>
    have h1 := L.refusesCoins_of_not_opens x (ν, t) _ ho
    rw [L.refusesCoins_of_live wk x w (ν, t) κ _ hk hl] at h1
    cases h1

/-- **The live frame rounds' bytes**, the per-round byte equality at the worker's frame rounds: after every history of
`0 < ops.length < refK x` openings, past the link and before the last coin, whose rounds all passed, whose openings
pass and whose last coins the frame takes, the frame's release of the worker's round is the session's switched message
at the coins the openings open without its leading domain frame (`dropDom`). -/
def LiveRoundEq (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

/-- **The live frame rounds' bytes at the worker's own histories**: `LiveRoundEq`'s equation after every history it
reads whose messages are the deferred release's own (`ZkServer.OwnHist`, placed as `FinalRoundEq` places it), the only
histories the deferred release meets. -/
def LiveRoundEqOwn (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

/-- `LiveRoundEq` gives its equation at the own histories. -/
theorem LiveRoundEq.own {S : RefSession} {vs : S.SwVals} {L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)}
    {H : Msg → Msg} {wk : L.toSrv.Worker} {x : Msg} {w : Flock.Wit} (hE : LiveRoundEq S vs L H wk x w) :
    LiveRoundEqOwn S vs L H wk x w :=
  fun ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr _ hv => hE ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv

/-- At a frame round `ops.length + 1`, with the `Finish` one past the last coin, the openings are past the link and
before the last coin. -/
theorem ops_of_masked {W R : Type} {L : LeanRelease W R} {x : Msg} {ops : List (Msg × Msg)} {p : Msg × Msg}
    (hlast : L.last x = refK x + 1) (hk : L.kindAt x (p :: ops).length = .masked) :
    0 < ops.length ∧ ops.length ≠ refK x := by
  obtain ⟨-, h1, h2⟩ := kindAt_masked hk
  simp only [List.length_cons, hlast] at h1 h2
  omega

/-- **`LiveRefine` at the session with the frame's stops**, at every worker, from the stop link, the transport's refusal
of a failed opening and the live frame rounds' bytes at the worker's own histories (`LiveRoundEqOwn`), with the
`Finish` one past the last coin. A live run's frame rounds are below the `Finish` (`LeanRelease.live_lt_last`), so no
history of more than `refK x` openings is live. -/
theorem liveRefine_withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg)
    (w : Flock.Wit) (hlast : L.last x = refK x + 1) (hR : StopsAt S Rf L H x) (hO : OpeningsRefused S L H x)
    (hE : LiveRoundEqOwn S vs L H wk x w) :
    L.LiveRefine H ((S.withStops Rf L.stop).swP vs).prover wk x w := by
  intro ρ κ h v hk hl hr hown hv
  obtain ⟨ν, t⟩ := ρ
  cases h with
  | nil => exact Kind.noConfusion (hk : Kind.own = Kind.masked)
  | cons p ops =>
    obtain ⟨m₀, ρ₀⟩ := p
    obtain ⟨h0, hne⟩ := ops_of_masked hlast hk
    show ((strat (((S.withStops Rf L.stop).swP vs).hello x ν) ((S.withStops Rf L.stop).swP vs).abort
      ((S.withStops Rf L.stop).swP vs).dec (((S.withStops Rf L.stop).swP vs).chk x ν)
      (((S.withStops Rf L.stop).swP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = _
    rw [strat_eval_cons]
    split
    next hok =>
      by_cases hlen : ops.length ≤ refK x
      · have hlt : ops.length < refK x := by omega
        have hRf := (hR ν t m₀ ρ₀ ops hlen hok).symm.trans hr
        simp only [List.length_map, Nat.min_eq_left hlen] at hRf
        show (S.withStops Rf L.stop).swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) = _
        exact (S.swMsgAt_withStops_frame Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
          (by simpa using hRf) (by simpa using h0) (by simpa using hlt)).trans
          (hE ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hown hv).symm
      · have hlt := L.live_lt_last wk x w (ν, t) κ _ hk hl
        simp only [List.length_cons, hlast] at hlt
        omega
    next hok =>
      by_cases hlen : ops.length ≤ refK x
      · have h1 := hO ν t m₀ ρ₀ ops hlen (Bool.eq_false_of_ne_true hok)
        cases h1.symm.trans hr
      · have hlt := L.live_lt_last wk x w (ν, t) κ _ hk hl
        simp only [List.length_cons, hlast] at hlt
        omega

/-- **The stand-in frame rounds' bytes**, the per-round byte equality at the ZK server's scheduled rounds: after every
history of `0 < ops.length < refK x` openings on which a round failed, whose openings pass and whose last coins the
frame takes, the frame's release of the scheduled round, at the values the ZK server's transcript reports and as a
stopped rep's, is the session's switched message at the coins the openings open without its leading domain frame. -/
def FillRoundEq (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.release x (ν, t) ((m₀, ρ₀) :: ops) true
        (L.scheduled x w (ν, t) ((m₀, ρ₀) :: ops) (computedOf (L.computed x w (ν, t) ((m₀, ρ₀) :: ops)))) =
      dropDom (S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2))

/-- **`FillRefine` at the session with the frame's stops**, at a last round `n x` past the protocol's coins, from the
stop link, the transport's refusal of a failed opening and the stand-in frame rounds' bytes. -/
theorem fillRefine_withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hn : n x ≤ refK x + 2) (hlast : L.last x = refK x + 1) (hR : StopsAt S Rf L H x)
    (hO : OpeningsRefused S L H x) (hE : FillRoundEq S vs L H wk x w) :
    L.FillRefine H ((S.withStops Rf L.stop).swP vs).prover n wk x w := by
  intro ρ κ h hk hlt hl hr
  obtain ⟨ν, t⟩ := ρ
  cases h with
  | nil => exact Kind.noConfusion (hk : Kind.own = Kind.masked)
  | cons p ops =>
    obtain ⟨m₀, ρ₀⟩ := p
    obtain ⟨h0, -⟩ := ops_of_masked hlast hk
    have hlt' : ops.length < refK x := by
      simp only [List.length_cons] at hlt
      omega
    show ((strat (((S.withStops Rf L.stop).swP vs).hello x ν) ((S.withStops Rf L.stop).swP vs).abort
      ((S.withStops Rf L.stop).swP vs).dec (((S.withStops Rf L.stop).swP vs).chk x ν)
      (((S.withStops Rf L.stop).swP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = _
    rw [strat_eval_cons]
    split
    next hok =>
      have hRf := (hR ν t m₀ ρ₀ ops hlt'.le hok).symm.trans hr
      simp only [List.length_map, Nat.min_eq_left hlt'.le] at hRf
      show (S.withStops Rf L.stop).swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) = _
      exact (S.swMsgAt_withStops_frame Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
        (by simpa using hRf) (by simpa using h0) (by simpa using hlt')).trans (hE ν t κ m₀ ρ₀ ops h0 hlt' hl hok hr).symm
    next hok =>
      have h1 := hO ν t m₀ ρ₀ ops hlt'.le (Bool.eq_false_of_ne_true hok)
      cases h1.symm.trans hr

/-- **The `Hello` is the session's**: at every tape the server's own `Hello` is the session's at its nonce. At the
deployed coin layer the session's is `CoinLayer.helloD`, the verifier's own canonical `Hello` (what
`CoinTree.checkHello` compares byte for byte), and this follows from the server's `Hello` being `flock-firewall hello`'s
(`LeanRelease.HelloIs`, `helloEq_of_helloIs`). -/
def HelloEq (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (x : Msg) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape), L.hello x (ν, t) = S.hello x ν

/-- **`HelloEq` at the deployed coin layer from `flock-firewall hello`**: where the server's `Hello` is that command's
message (`HelloIs`), it is the session's (`CoinLayer.helloD_eq`). -/
theorem helloEq_of_helloIs (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (x : Msg) (hL : HelloIs nr T L x) : HelloEq (CoinLayer.session nr T) L x :=
  fun ν t => (hL ν t).trans (CoinLayer.helloD_eq x ν).symm

/-- **`HelloRefine` at the session with the frame's stops**, from the `Hello` being the session's. -/
theorem helloRefine_withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (x : Msg) (w : Flock.Wit)
    (hH : HelloEq S L x) : L.HelloRefine H ((S.withStops Rf L.stop).swP vs).prover x w := by
  intro ρ
  obtain ⟨ν, t⟩ := ρ
  show ((strat (((S.withStops Rf L.stop).swP vs).hello x ν) ((S.withStops Rf L.stop).swP vs).abort
    ((S.withStops Rf L.stop).swP vs).dec (((S.withStops Rf L.stop).swP vs).chk x ν)
    (((S.withStops Rf L.stop).swP vs).msg x w ν t) []).eval H).1 = _
  rw [strat_eval_nil, hH ν t]
  rfl

/-- **The link is the session's**: after `Hello`'s answer, where the protocol has coins and the frame takes the
answer's, the ZK server's own link (its recheck's roots) is the session's switched link at the tape. A named hypothesis,
one of the transport's trusts: the link (`link`) is the recheck's, the level-0 roots `Flock.Firewall.Link` commits from
the leaves under the salts on the ZK server's tape (`Flock.Firewall.Link.root`), which the session loop's `Level0` runs
(else the session's Lean finish, at the table's level-0 request) and `Firewall::root` sends (`flock-circuit`). At the deployed tables it follows from the transport handing that command the
tape's tables and framing its roots (`LeanRelease.LinkIs`, `LeanRelease.linkRoundEq_of_linkIs` in `LinkRecheck`). -/
def LinkRoundEq (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (m₀ ρ₀ : Msg), 0 < refK x → L.refusesCoins x (ν, t) [(m₀, ρ₀)] = false →
    L.link x w (ν, t) [(m₀, ρ₀)] = S.swMsgAt x (S.wt x w) (vs x w ν) t []

/-- **The `Finish` is the session's**: after every history of `refK x` openings whose rounds all passed, whose
openings pass, whose last coins the frame takes and whose messages are the deferred release's own (`ZkServer.OwnHist`,
the only histories it meets), `Final.encode` of the record and proof files the ZK server builds
itself is the session's switched last message at the coins the openings open. A live run's `Finish` is one the
ZK server's own `Finish` takes (`LeanRelease.finStops_of_live`), so it says nothing where that stops. A named hypothesis,
one of the transport's trusts: the proof files in `files` are the session's Lean finish's (`Flock.Firewall.Finish`,
`proofFile`, each rep's `repFw` at the values the ZK server passed and every round's coins), which `flock-circuit` sends
as the session's `Proof`s. That they encode the session's last message is not yet proved about that code: it follows
from `FrameAdmits` and `FinishIs` (`finalRoundEq_of_finishIs`, `FinishRound`). -/
def FinalRoundEq (S : RefSession) (vs : S.SwVals) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : S.Nonce) (t : S.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    ops.length = refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk (S.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval H).1 = true →
    L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    FlockDeploy.Final.encode ⟨x, (L.files x w (ν, t) ((m₀, ρ₀) :: ops)).1, (L.files x w (ν, t) ((m₀, ρ₀) :: ops)).2⟩ =
      S.swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)

/-- **`OwnRefine` at the session with the frame's stops**, at a last round `n x` one past the `Finish`'s, from the
stop link, the transport's refusal of a failed opening, and the link and the `Finish` being the session's. -/
theorem ownRefine_withStops (S : RefSession) (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (vs : S.SwVals)
    (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (H : Msg → Msg) (n : Msg → ℕ) (wk : L.toSrv.Worker)
    (x : Msg) (w : Flock.Wit) (hn : n x = refK x + 2) (hlast : L.last x = refK x + 1) (hR : StopsAt S Rf L H x)
    (hO : OpeningsRefused S L H x) (hLk : LinkRoundEq S vs L x w) (hF : FinalRoundEq S vs L H wk x w) :
    L.OwnRefine H ((S.withStops Rf L.stop).swP vs).prover n wk x w := by
  intro ρ κ h hk hc hown
  obtain ⟨ν, t⟩ := ρ
  cases h with
  | nil => exact Kind.noConfusion (hk : Kind.own = Kind.recomputed)
  | cons p ops =>
    obtain ⟨m₀, ρ₀⟩ := p
    have hk' := kindAt_recomputed hk
    simp only [List.length_cons, hlast] at hk'
    have hlen : ops.length ≤ refK x := by omega
    show ((strat (((S.withStops Rf L.stop).swP vs).hello x ν) ((S.withStops Rf L.stop).swP vs).abort
      ((S.withStops Rf L.stop).swP vs).dec (((S.withStops Rf L.stop).swP vs).chk x ν)
      (((S.withStops Rf L.stop).swP vs).msg x w ν t) ((m₀, ρ₀) :: ops)).eval H).1 = _
    rw [strat_eval_cons]
    split
    next hok =>
      show (S.withStops Rf L.stop).swMsgAt x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2) = _
      have hRf := hR ν t m₀ ρ₀ ops hlen hok
      unfold ownMsg
      cases hr : L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) with
      | true =>
        exact S.swMsgAt_withStops_stop Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
          (hRf.symm.trans hr)
      | false =>
        refine (S.swMsgAt_withStops_go Rf L.stop x (S.wt x w) (vs x w ν) t (ops.map fun p => S.decOf p.2)
          (hRf.symm.trans hr) (by simp only [List.length_map]; omega)).trans ?_
        simp only [Bool.false_eq_true, ↓reduceIte]
        split_ifs with hif
        · have hf : ops.length = refK x := by
            simp only [List.length_cons, hlast] at hif
            omega
          have hlv : L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true := by
            rcases hc with hc | hc
            · simp only [List.length_cons, hn] at hc
              omega
            · exact hc
          exact (hF ν t κ m₀ ρ₀ ops hf hlv hok hr hown).symm
        · simp only [List.length_cons, hlast] at hif
          have h0 : ops = [] := List.eq_nil_of_length_eq_zero (by omega)
          subst h0
          exact (hLk ν t m₀ ρ₀ (by simp only [List.length_nil] at hif; omega) hr).symm
    next hok =>
      have h1 := hO ν t m₀ ρ₀ ops hlen (Bool.eq_false_of_ne_true hok)
      unfold ownMsg
      rw [h1]
      rfl

/-! ## The stops of a transport that reads only the coins -/

/-- **The rep's state reads the history only through the decoded coins** at `x`, up to the protocol's last coin and on
the histories its check passes: after every history past `Hello` of at most `refK x` openings that the transport's coin
check passes at the tape `ρ` (`opens`), the frame state of the rep that released the last round is `pc x` of the coins
the openings open, and the next round's rep's first-round state `fc x` of them. Past the check's refusal the transport
hands the frame nothing, so the premise says nothing there. A named hypothesis, one of the transport's trusts: the
frame's state (`prog`, `fresh`) is what the session loop's requests to the rep's frame make it (`Flock.Firewall.Frames`,
on `Drive.reqs`: the rep's coins and the device and scheduled rounds), which the release takes as given. At the loop's
own frames it holds by definition (`LeanRelease.ofLoop_progReads`). The coins the frame takes are `CoinsRead`, a
theorem at the session loop (`LeanRelease.loop_coinsRead`). -/
def ProgReads (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (pc : Msg → List S.Coin → Masker) (fc : Msg → List S.Coin → Except String Masker) (x : Msg) : Prop :=
  ∀ (ρ : S.refP.Nonce × S.refP.Rand) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)), ops.length ≤ refK x →
    L.opens x ρ ((m₀, ρ₀) :: ops) = true →
    L.prog x ((m₀, ρ₀) :: ops) = pc x (ops.map fun p => S.decOf p.2) ∧
      L.fresh x ((m₀, ρ₀) :: ops) = fc x (ops.map fun p => S.decOf p.2)

/-- **The coins the frame takes are a function of the decoded coins** at `x`: after every history past `Hello`, `cc x`
of the coins its openings open. -/
def CoinsRead (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (cc : Msg → List S.Coin → Array _root_.Flock.F128) (x : Msg) : Prop :=
  ∀ (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)), L.coinsOf x ((m₀, ρ₀) :: ops) = cc x (ops.map fun p => S.decOf p.2)

/-- **The transport reads the history only through the decoded coins** at `x`: the frame's state (`ProgReads`) and the
coins it takes (`CoinsRead`). -/
def ReadsCoins (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (pc : Msg → List S.Coin → Masker) (fc : Msg → List S.Coin → Except String Masker)
    (cc : Msg → List S.Coin → Array _root_.Flock.F128) (x : Msg) : Prop :=
  ProgReads S L pc fc x ∧ CoinsRead S L cc x

/-- **The frame's refusal at the coins `cs`** under the transport's reading `pc`, `fc`, `cc`: whether the frame refuses
`cc x cs` where round `cs.length + 1` takes them (`takeAt`), at the states `pc x cs` and `fc x cs`, the layouts of
the reps of rounds `cs.length` and `cs.length + 1` and the mask slot of the latter, at any samples
(`takeAt_ofDraws`). -/
def refuseOf {W R Coin : Type} (L : LeanRelease W R) (pc : Msg → List Coin → Masker)
    (fc : Msg → List Coin → Except String Masker) (cc : Msg → List Coin → Array _root_.Flock.F128) (x : Msg)
    (cs : List Coin) : Bool :=
  !(takeAt (cs.length + 1) (decide (L.rep x (cs.length + 1) = L.rep x cs.length))
      (Cfg.ofDraws (L.lay x (L.rep x (cs.length + 1))) default (L.slot x (L.rep x (cs.length + 1))))
      (Cfg.ofDraws (L.lay x (L.rep x cs.length)) default none) (pc x cs) (fc x cs) (cc x cs)).isOk

/-- **The transport's stop at the coins `cs`** under its reading: whether the frame refused the coins at `cs` or at one
of its prefixes (`refuseOf`), as a refusal stops the session for good (`refusesCoins_of_take`). -/
def stopOf {W R Coin : Type} (L : LeanRelease W R) (pc : Msg → List Coin → Masker)
    (fc : Msg → List Coin → Except String Masker) (cc : Msg → List Coin → Array _root_.Flock.F128) (x : Msg)
    (cs : List Coin) : Bool :=
  (List.range (cs.length + 1)).any fun k => L.refuseOf pc fc cc x (cs.take k)

/-- **The stop rule of the transport's reading**: at the padded coins `c` and round `i`, the transport's stop at `c`'s
first `i`. -/
def rfOf (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (pc : Msg → List S.Coin → Masker) (fc : Msg → List S.Coin → Except String Masker)
    (cc : Msg → List S.Coin → Array _root_.Flock.F128) :
    (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool :=
  fun x c i => L.stopOf pc fc cc x ((List.ofFn c).take i)

/-- The frame's taking of coins at a rep's configuration reads the history's length alone. -/
theorem answer_cfg {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg)) (s : Masker)
    (cs : Array _root_.Flock.F128) (n : ℕ) (hn : h.length = n) :
    s.answer (L.cfg x ρ h) cs = s.answer (Cfg.ofDraws (L.lay x (L.rep x n)) default none) cs := by
  subst hn
  exact answer_ofDraws _ _ _ _ _ _ _

/-- **Under the transport's reading, the frame takes the coins after a history where `refuseOf` does not refuse
them**, at every tape whose check passes the history. -/
theorem frameAnswer_of_readsCoins (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (pc : Msg → List S.Coin → Masker) (fc : Msg → List S.Coin → Except String Masker)
    (cc : Msg → List S.Coin → Array _root_.Flock.F128) (x : Msg)
    (hT : ReadsCoins S L pc fc cc x) (ρ : S.refP.Nonce × S.refP.Rand) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hl : ops.length ≤ refK x) (ho : L.opens x ρ ((m₀, ρ₀) :: ops) = true) :
    (L.frameAnswer x ρ ((m₀, ρ₀) :: ops)).isOk = !L.refuseOf pc fc cc x (ops.map fun p => S.decOf p.2) := by
  obtain ⟨hp, hf⟩ := hT.1 ρ m₀ ρ₀ ops hl ho
  have hc := hT.2 m₀ ρ₀ ops
  unfold frameAnswer refuseOf cfg cfgAt
  cases hg : ((m₀, ρ₀) :: ops).getLast? with
  | none => simp at hg
  | some p =>
    dsimp only
    rw [hp, hf, hc]
    simp only [List.length_cons, List.length_map, Nat.add_sub_cancel, Bool.not_not]
    exact congrArg Except.isOk (takeAt_ofDraws _ _ _ _ _ _ _ _ _ _ _ _ _ _)

/-- **Under the transport's reading, the release refuses the coins where the transport's stop does** (`stopOf`), at
every tape whose check passes the history: the check passes each prefix (`opens_take`), and at each the frame's
refusal is `refuseOf`'s. -/
theorem refusesCoins_of_readsCoins (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (pc : Msg → List S.Coin → Masker) (fc : Msg → List S.Coin → Except String Masker)
    (cc : Msg → List S.Coin → Array _root_.Flock.F128) (x : Msg)
    (hT : ReadsCoins S L pc fc cc x) (ρ : S.refP.Nonce × S.refP.Rand) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hl : ops.length ≤ refK x) (ho : L.opens x ρ ((m₀, ρ₀) :: ops) = true) :
    L.refusesCoins x ρ ((m₀, ρ₀) :: ops) = L.stopOf pc fc cc x (ops.map fun p => S.decOf p.2) := by
  have hk : ∀ k, k ≤ ops.length → (L.frameAnswer x ρ (((m₀, ρ₀) :: ops).take (k + 1))).isOk =
      !L.refuseOf pc fc cc x ((ops.map fun p => S.decOf p.2).take k) := fun k hk => by
    rw [List.take_succ_cons, ← List.map_take]
    exact frameAnswer_of_readsCoins S L pc fc cc x hT ρ m₀ ρ₀ (ops.take k) (by rw [List.length_take]; omega)
      (L.opens_take x ρ ((m₀, ρ₀) :: ops) (k + 1) ho)
  cases hr : L.refusesCoins x ρ ((m₀, ρ₀) :: ops) <;>
    cases hs : L.stopOf pc fc cc x (ops.map fun p => S.decOf p.2)
  · rfl
  · obtain ⟨-, hall⟩ := (L.refusesCoins_eq_false x ρ _).1 hr
    obtain ⟨k, hk', hkr⟩ := List.any_eq_true.1 hs
    rw [List.mem_range, List.length_map] at hk'
    have hkr' : L.refuseOf pc fc cc x ((ops.map fun p => S.decOf p.2).take k) = true := hkr
    have h1 := hall (k + 1) (by simp only [List.length_cons]; omega)
    rw [hk k (by omega), hkr'] at h1
    exact absurd h1 (by decide)
  · have hf : L.refusesCoins x ρ ((m₀, ρ₀) :: ops) = false := by
      refine (L.refusesCoins_eq_false x ρ _).2 ⟨ho, fun j hj => ?_⟩
      rcases j with _ | k
      · rfl
      · simp only [List.length_cons] at hj
        rw [hk k (by omega)]
        have h1 := (List.any_eq_false.1 hs) k (List.mem_range.2 (by rw [List.length_map]; omega))
        rw [Bool.not_eq_true] at h1
        simp only [h1, Bool.not_false]
    rw [hf] at hr
    cases hr
  · rfl

/-- **The stops of a transport that reads only the coins are the frame's refusals**: `StopsAt` at its rule, where its
check is the session's, so that every history whose openings pass the session's check passes the transport's. -/
theorem stopsAt_of_readsCoins (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (H : Msg → Msg) (pc : Msg → List S.Coin → Masker) (fc : Msg → List S.Coin → Except String Masker)
    (cc : Msg → List S.Coin → Array _root_.Flock.F128) (x : Msg)
    (hT : ReadsCoins S L pc fc cc x) (hC : CheckIs S L H x) : StopsAt S (rfOf S L pc fc cc) L H x := by
  intro ν t m₀ ρ₀ ops hl hok
  rw [refusesCoins_of_readsCoins S L pc fc cc x hT (ν, t) m₀ ρ₀ ops hl ((hC ν t m₀ ρ₀ ops hl).trans hok)]
  show _ = L.stopOf pc fc cc x ((List.ofFn (S.pad (K := refK x) (ops.map fun p => S.decOf p.2))).take
    (min (ops.map fun p => S.decOf p.2).length (refK x)))
  rw [S.ofFn_pad_take (K := refK x) (ops.map fun p => S.decOf p.2) (by simpa using hl)]

/-- **The rule reads the coins before its round only.** -/
theorem stopsCausal_rfOf (S : RefSession) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (pc : Msg → List S.Coin → Masker) (fc : Msg → List S.Coin → Except String Masker)
    (cc : Msg → List S.Coin → Array _root_.Flock.F128) :
    S.StopsCausal (rfOf S L pc fc cc) := by
  intro x c c' i h
  show L.stopOf pc fc cc x ((List.ofFn c).take i) = L.stopOf pc fc cc x ((List.ofFn c').take i)
  rw [RefSession.ofFn_take_congr c c' i h]

/-! ## The transport's check at the deployed coin layer -/

/-- **The coin check reads the session's `Hello` with the tape's nonce** at `x`: at every tape, where it reads one
(`helloChk`), the nonce it reads is the bytes of the tape's, and where it reads none, the schedule has no place for coin
`0` (`CoinLayer.slot`), so the session's check refuses every opening too. The session loop's is a theorem
(`SessionLean`); a transport that parses the `Hello`'s bytes has it from the server's `Hello` being the coin layer's
(`HelloEq`) and `Hello.parse` reading that one (`HelloDParses`, `helloParses_of_helloEq`). -/
def HelloParses (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand)) (x : Msg) :
    Prop :=
  ∀ (ν : (CoinLayer.session nr T).Nonce) (t : (CoinLayer.session nr T).Tape),
    match L.helloChk x (ν, t) with
    | .ok hl => hl.nonce = CoinLayer.nb ν
    | .error _ => CoinLayer.slot x 0 = none

/-- **`flock-firewall coins` reads the coin layer's `Hello` with its nonce** at `x`: at every nonce, `Hello.parse`
reads `CoinLayer.helloD` (`coinHello`), and the nonce it reads is the nonce's bytes. A statement about Lean functions
alone, all total (`Hello.parse` reads the JSON through `Flock.JsonParse.parse`, and `CoinLayer.helloD` is itself built
through it, `CoinLayer.helloJson`). The parser reads `canon`'s output back alike (`FlockComplete.JsonRT.parse_canon`);
it stays a named hypothesis because `Hello.parse` accepting what it reads depends on the statement's own `Hello`.
`verity/ml/flock/tests/test_lean_coin_open.py` runs it on the `Hello`s of the sessions Rust's `Checked` is held to
(`verity/ml/flock/live/src/coin_open_vectors.json`). -/
def HelloDParses (x : Msg) : Prop :=
  ∀ ν : CoinLayer.Nonce,
    ∃ hl : _root_.Flock.Firewall.Coins.Hello, coinHello (CoinLayer.helloD x ν) = .ok hl ∧ hl.nonce = CoinLayer.nb ν

/-- **`HelloParses` of a transport whose coin check parses the `Hello`'s bytes** (`coinHello`, `coin_tree::Checked`), from
the server's `Hello` being the coin layer's and `Hello.parse` reading the coin layer's. -/
theorem helloParses_of_helloEq (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand)) (x : Msg)
    (hR : ∀ ρ, L.helloChk x ρ = coinHello (L.hello x ρ))
    (hH : HelloEq (CoinLayer.session nr T) L x) (hD : HelloDParses x) : HelloParses nr T L x := by
  intro ν t
  obtain ⟨hl, h1, h2⟩ := hD ν
  have e : L.helloChk x (ν, t) = .ok hl := (hR (ν, t)).trans ((congrArg coinHello (hH ν t)).trans h1)
  show (match L.helloChk x (ν, t) with
    | .ok hl => hl.nonce = CoinLayer.nb ν
    | .error _ => CoinLayer.slot x 0 = none)
  rw [e]
  exact h2

/-- **The transport requests each opening at the committed schedule's place** at `x` (`CoinOpen.Follows` at
`CoinLayer.place`): at every tape, where the coin check reads the session's `Hello` as `hl` (`helloChk`), the requests
(`LeanRelease.req`) of at most the protocol's `refK x` openings name, in order, the places the schedule gives them. The
session loop's is a theorem (`SessionLean`: its requests are `Flock.Firewall.Session.reqAt`); under `coin_tree::Checked`,
which asks for the coins of each `Req::Commit` and `Req::Round` by its stream and count, a named trust. -/
def ReqsFollow (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand)) (x : Msg) :
    Prop :=
  ∀ (ν : (CoinLayer.session nr T).Nonce) (t : (CoinLayer.session nr T).Tape) (hl : _root_.Flock.Firewall.Coins.Hello),
    L.helloChk x (ν, t) = .ok hl → ∀ ms : List Msg, ms.length ≤ refK x →
      CoinOpen.Follows (CoinLayer.place nr x) hl.spec hl.links 0 [] (reqsFrom (L.req x) 0 ms)

/-- **At the deployed coin layer the transport's check is the session's**, at SHA-512: from `flock-firewall coins`
reading the server's `Hello` with the tape's nonce and the requests following the schedule, `CheckIs` at
`CoinLayer.session nr T` (`CoinOpen.coinLayer_verdict`). -/
theorem checkIs_coinLayer (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand)) (x : Msg)
    (hP : HelloParses nr T L x) (hF : ReqsFollow nr T L x) : CheckIs (CoinLayer.session nr T) L CoinOpen.shaH x := by
  intro ν t m₀ ρ₀ ops hl
  cases ops with
  | nil => rfl
  | cons o ops =>
    have hPνt := hP ν t
    cases hp : L.helloChk x (ν, t) with
    | ok hlo =>
      rw [hp] at hPνt
      have hlen : ((o :: ops).map Prod.snd).length ≤ refK x := by
        rw [List.length_map]; exact hl
      have hv := CoinOpen.coinLayer_verdict nr T x ν hlo ρ₀ (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd)) hPνt
        (hF ν t hlo hp ((o :: ops).map Prod.snd) hlen) (List.cons_ne_nil _ _)
      exact (L.opens_of_hello x (ν, t) m₀ ρ₀ o ops hp).trans (hv.trans (congrArg
        (fun ms => ((allOk ((CoinLayer.session nr T).refP.chk x ν ρ₀) 0 ms).eval CoinOpen.shaH).1)
        (reqsFrom_map (L.req x) 0 ((o :: ops).map Prod.snd))))
    | error e =>
      rw [hp] at hPνt
      have hpl : CoinLayer.place nr x 0 = { ok := false, pos := [], idx := 0, n := 0, depth := 0 } := by
        unfold CoinLayer.place
        rw [hPνt]
      have hpd : CoinLayer.parseD (CoinLayer.place nr x 0) ρ₀ o.2 = none := by
        rw [hpl]
        rfl
      have hop : L.opens x (ν, t) ((m₀, ρ₀) :: o :: ops) = false := by
        show (coinVerdict (L.helloChk x (ν, t)) ρ₀ (reqsFrom (L.req x) 0 ((o :: ops).map Prod.snd))).isOk = false
        unfold coinVerdict
        rw [hp]
        rfl
      rw [hop]
      show false =
        (OC.eval CoinOpen.shaH (allOk (CoinOpen.chkAt (CoinLayer.place nr x) _ ρ₀) 0 ((o :: ops).map Prod.snd))).1
      have hc : ∀ ν', CoinOpen.chkAt (CoinLayer.place nr x) ν' ρ₀ 0 o.2 = .ret false := fun ν' => by
        simp only [CoinOpen.chkAt, hpd, treeChk]
      simp only [List.map_cons, allOk, hc]
      all_goals rfl

end LeanRelease

open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- At a statement in the relation, the deployed record's last round is one past the `Finish`, which is one past the
protocol's last coin. -/
theorem deployZk_last_eq {rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation} {x : Msg}
    {w : Flock.Wit} (hR : Flock.R (FlockDeploy.deployZk rel).parse x w) :
    (FlockDeploy.deployZk rel).V.k x + 1 = refK x + 2 := by
  have := V_k_of_R hR
  omega

/-- **`Flock.zk_server_lean` at the session with the frame's stops**: at a session `S` and a stop rule `Rf` that reads
the coins before its round only and is the frame's refusals at `x` up to the protocol's last coin wherever the openings
pass (`LeanRelease.StopsAt`), and a release whose `Finish` is one past the last coin, the bound holds at
`S.withStops Rf L.stop` from `S`'s Props, its switched causality only at the coin vectors `Ok` holds at
(`SwCausalOn`), which hold at every coin vector the stops let through and every one the session admits (`StopsOk`),
the transport's coin check being the session's up to the last coin
(`LeanRelease.CheckIs`, which gives the release's refusal of a failed opening, `LeanRelease.openingsRefused_of_checkIs`),
the server's `Hello`, the ZK server's own link and `Finish` being the session's (`LeanRelease.HelloEq`, `LinkRoundEq`,
`FinalRoundEq`) and the per-round byte equality at the worker's frame rounds, at its own histories, and the stand-in
frame rounds (`LeanRelease.LiveRoundEqOwn`, `LeanRelease.FillRoundEq`, each against the session's message without its
leading domain frame): the item-stream refinement is proved from them
(`LeanRelease.helloRefine_withStops`, `ownRefine_withStops`, `liveRefine_withStops`, `fillRefine_withStops`,
`stopRefine_withStops`). A transport that reads the history only through the decoded coins on the histories its check
passes gives `Rf` and both its Props (`LeanRelease.rfOf`, `stopsCausal_rfOf`, `stopsAt_of_readsCoins`). The stop
leaks `Flock.zk_server_lean`'s one bit a session, `1[a round failed ∨ L.finStops]`, `0` for honest workers. -/
theorem Flock.zk_server_lean_stops (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession)
    (Rf : (x : Msg) → (Fin (refK x) → S.Coin) → ℕ → Bool) (hRf : S.StopsCausal Rf) (vs : S.SwVals) (t : ℕ)
    (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hS : S.Tables) (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop) (hsc : S.SwCausalOn Ok vs) (hOk : S.StopsOk Rf Ok)
    (hT : S.TapeOK) (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hlast : L.last x = refK x + 1) (hst : LeanRelease.StopsAt S Rf L H x)
    (hC : LeanRelease.CheckIs S L H x) (hH : LeanRelease.HelloEq S L x) (hLk : LeanRelease.LinkRoundEq S vs L x w)
    (hF : LeanRelease.FinalRoundEq S vs L H wk x w) (hE : LeanRelease.LiveRoundEqOwn S vs L H wk x w)
    (hEf : LeanRelease.FillRoundEq S vs L H wk x w)
    {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × L.toSrv.Chk =>
        Ev (L.toSrv.timed x
          (runStar H (L.toSrv.proverD L.fillOf (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 (((S.withStops Rf L.stop).refP.withSim fun x _ c =>
              (S.withStops Rf L.stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).C x =>
          Ev (L.toSrv.timed x ((leaky1 (((S.withStops Rf L.stop).refP.withSim fun x _ c =>
              (S.withStops Rf L.stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).out H Vs θ x c))) +
          (8 * (S.withStops Rf L.stop).nLeaves x * δ₁.toReal + 8 * t * (S.withStops Rf L.stop).nLeaves x * δ₁.toReal +
            2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H (((S.withStops Rf L.stop).swP vs).prover w) Vs θ x).Pr H) :=
  Flock.zk_server_lean_of H rel (S.withStops Rf L.stop) vs t δ₁
    (Flock.zk_deployZk_sw_stops H rel S Rf L.stop vs Ok t ht hT1 hδ₁ hS hV hD hc hRf hsc hOk hT) L wk x w hR
    ⟨L.helloRefine_withStops S Rf vs H x w hH,
      L.ownRefine_withStops S Rf vs H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk x w (deployZk_last_eq hR)
        hlast hst (LeanRelease.openingsRefused_of_checkIs S L H x hC) hLk hF,
      L.liveRefine_withStops S Rf vs H wk x w hlast hst (LeanRelease.openingsRefused_of_checkIs S L H x hC) hE,
      L.fillRefine_withStops S Rf vs H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk x w
        (deployZk_last_eq hR).le hlast hst (LeanRelease.openingsRefused_of_checkIs S L H x hC) hEf,
      L.stopRefine_withStops S Rf vs H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk x w
        (deployZk_last_eq hR).le hst⟩ Vs θ hk Ev

/-- **`Flock.zk_server_lean_stops` at the deployed coin layer** (`CoinLayer.session nr T`), at SHA-512
(`CoinOpen.shaH`), from the transport's trusts alone. `Hello`, the opening check (the keyed coin-tree walk at the round's place under
`Hello`'s answer, `CoinLayer.treeD`, `CoinLayer.parseD`) and the coin reading are the deployed ones, and the tables,
with their round encoding `enc`, `fin` and `off`, are `T`'s, whose Props (the switched causality at the coin vectors
`Ok`, which the stop rule's `StopsOk` takes in) and the frame rounds' bytes
(`LeanRelease.LiveRoundEqOwn`, `FillRoundEq`: `T`'s frame rounds, without their leading domain frame, are the frame's) are
the tables' side. Each premise on the transport is a named trust or, at the session loop (`Claims/ZK/SessionLean.lean`),
a theorem:
* the server's `Hello` is the Lean `Hello` at the tape's nonce (`LeanRelease.HelloIs`; that `Hello` is `CoinLayer.helloD`,
  `CoinLayer.helloD_eq`, so `LeanRelease.HelloEq` follows, `helloEq_of_helloIs`), and the coin check reads the session's
  `Hello` with that nonce (`LeanRelease.HelloParses`);
* the transport requests each opening at the schedule's place (`LeanRelease.ReqsFollow`);
* the ZK server's own link and `Finish` are `T`'s (`LeanRelease.LinkRoundEq`: `Flock.Firewall.Link`'s roots, which
  `Firewall::root` sends, from `LeanRelease.LinkIs` at the deployed tables; `FinalRoundEq`: the session's Lean finish's
  proof files, `Flock.Firewall.Finish`, from `FrameAdmits` and `FinishIs`);
* the transport hands the frame rep states and coins that are functions `pc`, `fc`, `cc` of the coins the openings
  open, on the histories its check passes (`LeanRelease.ReadsCoins`: the states, `ProgReads`, are the session loop's
  rep frames', `Flock.Firewall.Frames`, by definition at `LeanRelease.ofLoop`; the coins,
  `CoinsRead`, the session loop's own, `loop_coinsRead`);
* the `Finish` is one past the last coin (`L.last`, the transport's schedule).
From them the release's coin check is the session's (`LeanRelease.checkIs_coinLayer`, from `CoinOpen.coinLayer_verdict`,
at `LeanRelease.helloParses_of_helloEq`), the stop rule is the frame's refusal under the transport's reading
(`LeanRelease.rfOf`), which reads the coins before its round only (`stopsCausal_rfOf`) and is the release's refusals
(`stopsAt_of_readsCoins`), and a live frame round is below the `Finish` (`LeanRelease.live_lt_last`). The stop leaks
`Flock.zk_server_lean`'s one bit a session, `1[a round failed ∨ L.finStops]`: whether the session stopped at its
`Finish`, where the ZK server's own `Finish` stops on workers' values that fail a constraint at the sampled pads; `0` for
honest workers. -/
theorem Flock.zk_server_lean_coinLayer
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (vs : (CoinLayer.session nr T).SwVals) (t : ℕ) (ht : 0 < t)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hS : (CoinLayer.session nr T).Tables)
    (hV : (CoinLayer.session nr T).Complete (Flock.R (FlockDeploy.deployZk rel).parse))
    (hD : (CoinLayer.session nr T).DummyReg) (hc : (CoinLayer.session nr T).Causal)
    (Ok : (x : Msg) → (Fin (refK x) → (CoinLayer.session nr T).Coin) → Prop)
    (hsc : (CoinLayer.session nr T).SwCausalOn Ok vs) (hT : (CoinLayer.session nr T).TapeOK)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hlast : L.last x = refK x + 1)
    (pc : Msg → List (CoinLayer.session nr T).Coin → Masker)
    (fc : Msg → List (CoinLayer.session nr T).Coin → Except String Masker)
    (cc : Msg → List (CoinLayer.session nr T).Coin → Array _root_.Flock.F128)
    (hOk : (CoinLayer.session nr T).StopsOk (LeanRelease.rfOf (CoinLayer.session nr T) L pc fc cc) Ok)
    (hRd : LeanRelease.ReadsCoins (CoinLayer.session nr T) L pc fc cc x)
    (hH : LeanRelease.HelloIs nr T L x) (hP : LeanRelease.HelloParses nr T L x)
    (hFw : LeanRelease.ReqsFollow nr T L x)
    (hLk : LeanRelease.LinkRoundEq (CoinLayer.session nr T) vs L x w)
    (hF : LeanRelease.FinalRoundEq (CoinLayer.session nr T) vs L CoinOpen.shaH wk x w)
    (hE : LeanRelease.LiveRoundEqOwn (CoinLayer.session nr T) vs L CoinOpen.shaH wk x w)
    (hEf : LeanRelease.FillRoundEq (CoinLayer.session nr T) vs L CoinOpen.shaH wk x w)
    {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand) × L.toSrv.Chk =>
        Ev (L.toSrv.timed x
          (runStar CoinOpen.shaH (L.toSrv.proverD L.fillOf (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x ω
            θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((((CoinLayer.session nr T).withStops
                (LeanRelease.rfOf (CoinLayer.session nr T) L pc fc cc) L.stop).refP.withSim fun x _ c =>
              ((CoinLayer.session nr T).withStops
                (LeanRelease.rfOf (CoinLayer.session nr T) L pc fc cc) L.stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).C x =>
          Ev (L.toSrv.timed x ((leaky1 ((((CoinLayer.session nr T).withStops
                (LeanRelease.rfOf (CoinLayer.session nr T) L pc fc cc) L.stop).refP.withSim fun x _ c =>
              ((CoinLayer.session nr T).withStops
                (LeanRelease.rfOf (CoinLayer.session nr T) L pc fc cc) L.stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).out CoinOpen.shaH Vs θ x c))) +
          (8 * ((CoinLayer.session nr T).withStops (LeanRelease.rfOf (CoinLayer.session nr T) L pc fc cc)
              L.stop).nLeaves x * δ₁.toReal +
            8 * t * ((CoinLayer.session nr T).withStops (LeanRelease.rfOf (CoinLayer.session nr T) L pc fc cc)
              L.stop).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t *
            (Fork CoinOpen.shaH ((((CoinLayer.session nr T).withStops
                (LeanRelease.rfOf (CoinLayer.session nr T) L pc fc cc) L.stop).swP vs).prover w) Vs θ x).Pr
              CoinOpen.shaH) :=
  have hHe := LeanRelease.helloEq_of_helloIs nr T L x hH
  have hC := LeanRelease.checkIs_coinLayer nr T L x hP hFw
  Flock.zk_server_lean_stops CoinOpen.shaH rel (CoinLayer.session nr T)
    (LeanRelease.rfOf (CoinLayer.session nr T) L pc fc cc) (LeanRelease.stopsCausal_rfOf _ L pc fc cc) vs t ht hT1 hδ₁ hS hV
    hD hc Ok hsc hOk hT L wk x w hR hlast (LeanRelease.stopsAt_of_readsCoins _ L CoinOpen.shaH pc fc cc x hRd hC) hC hHe hLk hF
    hE hEf Vs θ hk Ev

end Claims.ZKServer
