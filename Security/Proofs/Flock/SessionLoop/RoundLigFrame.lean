import Proofs.Flock.SessionLoop.SchedAt
import Proofs.Flock.SessionLoop.LiveInner
import Proofs.Flock.Claims.ZK.L0PinBind
import Proofs.Flock.Claims.ZK.FramePhase

/-!
# The frame at a rep's Ligerito rounds past level 0

At rep stream `2j + r`'s Ligerito rounds after level 0's closing round (`Masked.repSlot` `k0 + 5` up to the last
before the inner proof) the frame is outside every masked phase, so its check passes the worker's round only as the
round it schedules, and releases that round as it came (`check_offAt`). The scheduled round is the rep's opening fed the
history's coins (`openStep`), which past the ring switch is the cut run of `ligeritoPZk` at the rep's real table, the
switch's target and basis, zero pads, the tape's extras, lane pads and salts and no nonces (`openStep_l0`'s form, past
level 0).

* **`LeanRelease.release_lig_frame`**: there, for every coin function `coin` that is the history's coins from the rep's
  opening on (`CoinAt`), `ligeritoPZk`'s cut run at `coin`, cut at the round, sends at the round the body whose release
  is the worker's round, with no leading domain frame; it squeezes at each round before it as many coins as the history's
  coins there hold; and its basis is the switch's at the rep's algebraic coins and link points.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

/-- **The frame at a rep's Ligerito rounds past level 0**: at rep stream `2j + r`'s round `i` past its PIOP
(`Masked.repSlot` `i`), from `k0 + 5` to the last before the inner proof, the rep's real table is `tb` with salts
`hi`, and the history's coins from round `base` on are `cs`, the rep's opening's (its ring switch's, `γ`'s and
Ligerito's), after its algebraic phase's `algC`, the coins from the rep's first round (`Masked.repBase`). For every coin function that is `cs` from `base` on, `ligeritoPZk`'s
cut run at it, cut at the round and started past the ring switch (at Ligerito's first round, `Masked.repSlot` `0`),
at the switch's target `T` and basis, the table's `cm0`, `g`, extras, lane pads and salts, zero pads and no nonces,
logs `L`, where:

* the release of the worker's round is `L`'s body at the round, tagged `0`, which has no leading domain frame;
* each body `L` holds before the round squeezes as many coins as the history's coins at its round hold;
* the switch's basis is `Rs.sws` at the rep's claims (the algebraic phase's at `algC`, then the link's at the
  history's first coin) and `cs`'s first `xs.size + 3` coins. -/
theorem release_lig_frame (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) {v : Msg} (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : (Masked.zkTab x j).sch.k0 + 5 ≤ i)
    (h2 : i < (Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) :
    ∃ (tb : Tab) (hi : Array ByteArray) (xs : Array (Array F128)) (algC : Array (Array F128)) (base : ℕ)
      (cs : List (Array F128)),
      real x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi) ∧
      xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[j.val]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[j.val]!
          (Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) ∧
      algC.size ≤ base ∧
      Masked.repBase x (2 * j.val + r.val) + algC.size = base ∧
      (∀ a (ha : a < algC.size),
        algC[a] = Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (base - algC.size + a)) ∧
      base + cs.length = ops.length ∧
      (∀ a (ha : a < cs.length), cs[a] = Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (base + a)) ∧
      base + (xs.size + 2) + 1 = Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) ∧
      ∀ coin : ℕ → ℕ → F128, FlockComplete.LigLock.Fed.CoinAt coin base cs →
        ∃ (T : F128) (sws : Array _root_.Flock.Switched) (tr₂ : _root_.Flock.Transcript) (l₂ L : Array Claims.Msg),
          l₂.size = base + (xs.size + 2) + 1 ∧
          (∃ zc lc, FlockComplete.LigLock.Fed.PtsAt tb.st.m tb.st.fold.kLog
              (FlockComplete.LigLock.Fed.recCoin algC) 0 zc lc ∧
            sws = FlockComplete.Replay.Rs.sws
              (_root_.Flock.abcClaims tb.st.m tb.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf xs)
              ⟨fun k i => (cs.getD k #[])[i]!, fun i => (cs.getD (xs.size + 2) #[])[i]!⟩) ∧
          (((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin ops.length)
              (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) _root_.Flock.MerkleScheme.hm96Sha512 tb.sch T
              tb.cm0.toC tb.g (FlockComplete.basisVals sws (tb.st.m - 7))
              (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0)
              (Array.replicate (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).len 0) r.val tb.tPad
              tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty))
              (fun _ _ => 0)).run tr₂).go l₂).2 = L ∧
          (∀ p, l₂.size ≤ p → p < ops.length → ∀ (t' : _root_.Flock.Transcript) (kind : UInt8) (k : ℕ),
            k < 2 ^ 64 → L[p]? = some (Claims.toMsg (t'.framed kind k)) →
            (Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) p).size = k) ∧
          (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v = 0 :: L[ops.length]! ∧
          dropDom (0 :: L[ops.length]!) = 0 :: L[ops.length]! := by
  sorry

end LeanRelease

end Claims.ZKServer
