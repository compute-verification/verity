import Proofs.Flock.SessionLoop.Statement
import Proofs.Flock.Claims.ZK.FinishSwFrame
import Proofs.Flock.Claims.ZK.FrameRankFault

/-!
# The rep draws as the finish reads them

`Loop.RepSrc` (`Loop.Src`'s rep-draws half) gives, at a release whose draws are `L₀`'s and whose layout at rep stream
`2j + r` is the tape's (`Masked.layE`), the premises `Masked.swSrcE_repCut` takes of the frame's draws and the tape, at
the frame reps `Loop.frepOf`:

* `Loop.RepSrc.size`: the draws are `drawBytes` long (`hz`).
* `Loop.RepSrc.pads`: the tape's pads are those of the tape of the frames' draws (`hpads`).
* `Loop.RepSrc.h`: below `h_ab`, the tape's `h` is the tape of the frames' draws' (`htape`).
* `Loop.RepSrc.innerSalt`: at each leaf of the pads tree, the masked prover's fitted leaf salts agree.

At the loop's own release (`ofLoopLink`) the draws are `L₀`'s (`LeanRelease.draws_ofLoopLink`), and where the rep's frame
starts, its layout is the tape's (`LeanRelease.lay_ofLoopLink`, `swSrcE_repCut`'s `hlayR`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- The frame rep of table `i`'s rep `r`: rep stream `2i + r`. -/
def Loop.frepOf (i r : ℕ) : ℕ := 2 * i + r

variable {L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}

/-- **The frame's draws are `drawBytes` long** at the release `Lr`, whose draws are `L₀`'s and whose layout at the rep is
the tape's. -/
theorem Loop.RepSrc.size (hS : Loop.RepSrc L₀ src x w)
    (Lr : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (hd : Lr.draws = L₀.draws) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hlay : Lr.lay x (Loop.frepOf j.val r.val) = Masked.layE x j) :
    (Lr.draws (ν, t) (Loop.frepOf j.val r.val)).size =
      _root_.Flock.Firewall.Frame.drawBytes (Lr.lay x (Loop.frepOf j.val r.val)) := by
  rw [hd, hlay]
  exact (hS ν t j r).1

/-- **The tape's pads are the frame's**: table `j`'s rep `r`'s pads on the tape are those on the tape of the frames'
draws (`tapeOfDraws`, at any other entries `tz`). -/
theorem Loop.RepSrc.pads (hS : Loop.RepSrc L₀ src x w)
    (Lr : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (hd : Lr.draws = L₀.draws) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hlay : Lr.lay x (Loop.frepOf j.val r.val) = Masked.layE x j) (tz : FlockComplete.TapeZk) :
    FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j) =
      FlockComplete.padsOf (tapeOfDraws tz (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf) j.val r.val (Masked.layE x j) := by
  have h := padsOf_drawsAt Lr x (ν, t) tz Loop.frepOf j.val r.val (hS.size Lr hd ν t j r hlay)
  rw [hlay] at h
  rw [h]
  unfold LeanRelease.drawsAt
  rw [hlay, hd]
  exact (hS ν t j r).2.1

/-- **Below `h_ab`, the tape's `h` is the frame's**: table `j`'s rep `r`'s `h` on the tape is that on the tape of the
frames' draws, where `padsOf` reads it as it stands (`padsOf_get`). -/
theorem Loop.RepSrc.h (hS : Loop.RepSrc L₀ src x w)
    (Lr : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (hd : Lr.draws = L₀.draws) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hlay : Lr.lay x (Loop.frepOf j.val r.val) = Masked.layE x j) (tz : FlockComplete.TapeZk) :
    ∀ k < (Masked.layE x j).hab, (Masked.UE.tapeZk t).h j.val r.val k =
      (tapeOfDraws tz (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf).h j.val r.val k := by
  intro k hk
  have hlen : (Masked.layE x j).len = (Masked.layE x j).tw + 1 := rfl
  have htw : (Masked.layE x j).tw = (Masked.layE x j).hab + 3 := rfl
  rw [← FlockComplete.padsOf_get (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j) k (by omega) (by omega) (by omega),
    hS.pads Lr hd ν t j r hlay tz,
    FlockComplete.padsOf_get _ j.val r.val (Masked.layE x j) k (by omega) (by omega) (by omega)]

/-- **The pads tree's leaf salts are the frame's**: at each leaf, the masked prover's `fitBytes 192` of the tape's leaf
salt is that of the tape of the frames' draws. -/
theorem Loop.RepSrc.innerSalt (hS : Loop.RepSrc L₀ src x w)
    (Lr : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (hd : Lr.draws = L₀.draws) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hlay : Lr.lay x (Loop.frepOf j.val r.val) = Masked.layE x j) (tz : FlockComplete.TapeZk) :
    ∀ q < _root_.Flock.Firewall.Frame.codeLen (Masked.layE x j),
      FlockComplete.fitBytes 192 ((Masked.UE.tapeZk t).innerSalt j.val r.val q) =
        FlockComplete.fitBytes 192 ((tapeOfDraws tz (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf).innerSalt j.val r.val q) := by
  intro q hq
  rw [innerSalt_fit tz (Lr.lay x) (Lr.draws (ν, t)) Loop.frepOf j.val r.val q (hS.size Lr hd ν t j r hlay)
    (by rw [hlay]; exact hq), hlay, hd]
  exact (hS ν t j r).2.2.1 q hq

/-- The release at the loop's own frames and link draws `L₀`'s bytes. -/
theorem LeanRelease.draws_ofLoopLink (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) :
    (LeanRelease.ofLoopLink L₀ src real).draws = L₀.draws := rfl

/-- **The loop lays table `j`'s rep `r` out at the tape's layout** (`Masked.layE`), where its frame starts
(`Drive.start`, the loop's `Frames.startRep`). -/
theorem LeanRelease.lay_ofLoopLink (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (j : Fin (Masked.jOf x)) (r : Fin 2) {lay : _root_.Flock.Zk.PadLayout} {mask : _root_.Flock.Firewall.Frame.MaskSlot}
    (hs : _root_.Flock.Firewall.Drive.start (Masked.stsOf x)[j.val]! = .ok (lay, mask)) :
    (LeanRelease.ofLoopLink L₀ src real).lay x (Loop.frepOf j.val r.val) = Masked.layE x j := by
  have hst : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val (Masked.lt_of_jOf j)
  have hk : Loop.frepOf j.val r.val / 2 = j.val := by
    unfold Loop.frepOf
    omega
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) (Masked.lt_of_jOf j))
  have e : Loop.startOf x (Loop.frepOf j.val r.val) = _root_.Flock.Firewall.Drive.start (Masked.stsOf x)[j.val]! := by
    unfold Loop.startOf
    rw [hsx]
    dsimp only
    rw [hk]
  obtain ⟨sch, tp, hsch, hlay, -⟩ := Loop.start_eq hs
  rw [hst] at hsch hlay
  rw [(Masked.zkTab x j).hz] at hsch
  simp only [Except.ok.injEq, Prod.mk.injEq] at hsch
  obtain ⟨hsch, -⟩ := hsch
  subst hsch
  show ((Loop.startOf x (Loop.frepOf j.val r.val)).toOption.map (·.1)).getD default = Masked.layE x j
  rw [e, hs]
  exact hlay

end Claims.ZKServer
