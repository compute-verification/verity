import Proofs.Flock.Claims.Complete.Replay.RingSwitch

/-!
# Ligerito replays the honest prover's rounds

`Flock.Ligerito`, `Flock.Merkle` at the honest run: the verifier's Ligerito (`ligeritoWith`) on the rounds the session recorded.
-/

namespace FlockComplete.Replay

open Flock Transcript

/-! ## Where the ring switch leaves the rep -/

/-- Table `i`'s rep `r`'s transcript after the prover's ring switch. -/
def _root_.FlockComplete.Run.rsEnd {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : Transcript :=
  Rs.tEnd (o.trace i) (o.claims i r) (o.rsCoins i r) (o.lcEnd i r)

theorem _root_.FlockComplete.Run.rsEnd_k {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : (o.rsEnd i r).k = (o.rsEnd i r).rounds.size := by
  simp only [Run.rsEnd, Rs.tEnd_k, Rs.tEnd_size, o.lcEnd_k]

/-- **The ring switch's value at a run**: the prover's `T` and switched claims, and the transcript where the prover's
ring switch stopped. -/
theorem _root_.FlockComplete.Run.sw_eq {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (hR1 : TraceR1CS H rel) (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) (hzc : ZcAt o i r)
    (hlc : LcAt o i r) (hal : LcAlgebra H rel) (hrs : RsAt o i r) (hra : RsAlgebra H rel) :
    o.sw i r = ((Rs.T (o.trace i) (o.claims i r) (o.rsCoins i r), Rs.sws (o.claims i r) (o.rsCoins i r)),
      ⟨(o.stream i r).rounds, (o.rsEnd i r).k, (o.rsEnd i r).pending⟩) := by
  obtain ⟨hp, hag⟩ := hrs
  have h := ringSwitchWith_of o.hs o.tabs[i]!.m (o.trace i) (o.claims i r) (o.rsCoins i r) (o.lcEnd i r)
    (o.lcEnd_k i r) (o.claims_size_ne i r) (hag.mono (Nat.zero_le _)) (hra o i hi r hr)
  unfold Run.sw
  rw [o.lc_eq hR1 hi hr hzc hlc hal, o.zc_eq hR1 hi hr hzc, hp]
  exact congrArg val h

end FlockComplete.Replay
