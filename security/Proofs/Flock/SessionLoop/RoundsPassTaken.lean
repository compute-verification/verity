module

public import Proofs.Flock.SessionLoop.RoundsPassAt
public import Proofs.Flock.SessionLoop.RoundsPassFrame

/-!
# A taken round's scheduled round passes the frame's checks

`Loop.RoundsPass` at `Loop.valsDev` holds at a history wherever the frame takes the scheduled round there, of frames none
the inner proof's label, at the state the stream's coins leave (`LeanRelease.roundOk_valsDev`). This file states that
premise; each phase of a rep's rounds meets it in a file of its own (`RoundsPassPiop` and the like), without liveness:

* `LeanRelease.SchedTaken`: the frame answered the history's coins at a state outside the inner proof, and takes the
  scheduled round there, at a rep not stopped, as frames (`Sched.MidAll`) and the squeeze of the round's coins.
  `LeanRelease.roundOk_of_schedTaken`: it gives `roundOk` at `valsDev` wherever `cross` passes the device round.
  `LeanRelease.RoundTaken`: the same of given round bytes, which `LeanRelease.schedTaken_of` turns into `SchedTaken`
  at the scheduled round.
* `LeanRelease.answered_of_unrefused`: where the frame takes the coins after a history it answered them.
* `midAll_of_plain`, `ne_inner_of_isLabel`: frames inside a phase, and labels other than the inner proof's, are none
  the inner proof's label. `order_lt_of_alike`: alike configurations pad the same positions.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree isLabel)
open FlockSoundness.Refine (Frame encs)

theorem ne_inner_of_isLabel {b : ByteArray} (h : isLabel b _root_.Flock.Zk.LABEL_INNER = false) :
    Frame.label b ≠ .label _root_.Flock.Zk.LABEL_INNER.toUTF8 := by
  intro he
  injection he with he
  subst he
  rw [isLabel_inner] at h
  cases h

theorem midAll_of_plain {fs : List Frame} (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) :
    Sched.MidAll fs := by
  intro f hf
  refine ⟨hm f hf, ?_⟩
  rintro rfl
  have h3 : plainLabel _root_.Flock.Zk.LABEL_INNER.toUTF8 := hp _ hf
  obtain ⟨-, -, h4, -, -⟩ := h3
  rw [isLabel_inner] at h4
  cases h4

theorem order_lt_of_alike {c c' : Cfg} (h : CfgAlike c c')
    (hc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size) : ∀ p, p < c.order.size → c.order[p]! < c.h.size := by
  rw [h.order, h.size]
  exact hc

namespace LeanRelease

theorem answered_of_unrefused {W R : Type} {L : LeanRelease W R} {x : Msg} {ρ : R} {g : List (Msg × Msg)}
    (hr : L.refusesCoins x ρ g = false) : ∃ m, L.answered x ρ g = .ok m := by
  unfold refusesCoins at hr
  cases hA : L.answered x ρ g with
  | error e => rw [hA] at hr; cases hr
  | ok m => exact ⟨m, rfl⟩

/-- **The frame takes the scheduled round after a history**: it answered the history's coins at a state outside the
inner proof, the scheduled round is frames none the inner proof's label (`Sched.MidAll`) and a squeeze of the round's
coins that `roundFree` takes, and the frame takes it at that state, at a rep not stopped. -/
def SchedTaken (L : LeanRelease Flock.Wit Loop.Tp) (x : Msg) (w : Flock.Wit) (ρ : Loop.Tp) (g : List (Msg × Msg)) :
    Prop :=
  ∃ (m : Masker) (fs : List Frame) (k : UInt8) (o : ByteArray) (t : Masker) (fr : Array ℕ),
    L.answered x ρ g = .ok m ∧ m.inner = none ∧
    L.scheduled x w ρ g (computedOf (L.computed x w ρ g)) = encs fs ++ (Frame.sq k (L.coins x g.length)).enc ∧
    Sched.MidAll fs ∧ (k = 2 ∨ (k = 1 ∧ L.coins x g.length = 1)) ∧ ¬ (L.coins x g.length = 0 ∧ k = 2) ∧
    L.coins x g.length < 2 ^ 64 ∧
    roundFree (L.cfg x ρ g) { m with stopped := false }
      (ByteArray.mk (encs fs ++ (Frame.sq k (L.coins x g.length)).enc).toArray) (L.coins x g.length) = .ok (o, t, fr)

/-- **The frame takes round bytes `B` from state `m` after a history**: `B` is frames none the inner proof's label and a
squeeze of the round's coins that `roundFree` takes from `m`, at a rep not stopped. -/
def RoundTaken {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (g : List (Msg × Msg)) (m : Masker) (B : Msg) :
    Prop :=
  ∃ (fs : List Frame) (k : UInt8) (o : ByteArray) (t : Masker) (fr : Array ℕ),
    B = encs fs ++ (Frame.sq k (L.coins x g.length)).enc ∧
    Sched.MidAll fs ∧ (k = 2 ∨ (k = 1 ∧ L.coins x g.length = 1)) ∧ ¬ (L.coins x g.length = 0 ∧ k = 2) ∧
    L.coins x g.length < 2 ^ 64 ∧
    roundFree (L.cfg x ρ g) { m with stopped := false } (ByteArray.mk B.toArray) (L.coins x g.length) = .ok (o, t, fr)

theorem schedTaken_of {L : LeanRelease Flock.Wit Loop.Tp} {x : Msg} {w : Flock.Wit} {ρ : Loop.Tp}
    {g : List (Msg × Msg)} {m : Masker} (hA : L.answered x ρ g = .ok m) (hin : m.inner = none)
    (h : L.RoundTaken x ρ g m (L.scheduled x w ρ g (computedOf (L.computed x w ρ g)))) : L.SchedTaken x w ρ g := by
  obtain ⟨fs, k, o, t, fr, hB, hM, hk, h0, hn, hr⟩ := h
  rw [hB] at hr
  exact ⟨m, fs, k, o, t, fr, hA, hin, hB, hM, hk, h0, hn, hr⟩

/-- **Where the frame takes the scheduled round, the frame round passes the frame's checks at `valsDev`**, wherever
`cross` passes its device round: `roundOk_valsDev`. -/
theorem roundOk_of_schedTaken (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    {L : LeanRelease Flock.Wit Loop.Tp} {x : Msg} {w : Flock.Wit} {ρ : Loop.Tp} {g : List (Msg × Msg)}
    (h : L.SchedTaken x w ρ g)
    (hcr : L.cross x g.length w ρ g (Loop.valsDev vs L x w ρ g (computedOf (L.computed x w ρ g))) = true) :
    (L.withDevice (Loop.valsDev vs L)).roundOk x w ρ g = true := by
  obtain ⟨m, fs, k, o, t, fr, hm, hin, hr, hM, hk, h0, hn, hq⟩ := h
  exact roundOk_valsDev vs L hm hin hr hM hk h0 hn hq hcr

end LeanRelease

end Claims.ZKServer
