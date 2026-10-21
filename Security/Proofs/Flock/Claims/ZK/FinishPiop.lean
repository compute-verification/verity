import Proofs.Flock.Claims.ZK.FinishCommit
import Proofs.Flock.Claims.ZK.CutSession

/-!
# The firewall's PIOP is the masked session's from its source

`Flock.Firewall.Finish`'s transcript, zerocheck, lincheck and ring switch are copies of the analysis's, its sends an
`Asks` of its own (`Flock.Firewall.Finish.Asks`, the same two fields, read through `Asks.toC`). Each is the analysis's
by unfolding:

* `squeezeP_eq`, `sampleP_eq`, `sampleVecP_eq`, `sampleKP_eq`: the transcript's squeezes are `FlockComplete`'s.
* **`zerocheckFw_eq`**, **`lincheckFw_eq`**: the masked zerocheck and lincheck sending the masked values `mk` are
  `FlockComplete.zerocheckCut`'s and `lincheckCut`'s (the PIOP of `repCut`, `piopCut`, at `src.masked = mk`).
* **`ringSwitchP_eq`**: the ring switch sending claim `k`'s vector `sv k` is `FlockComplete.ringSwitchCut`'s.
* `basisVals_eq`: the batched basis is `FlockComplete.basisVals`.
-/

namespace Flock.Firewall.Finish

/-- The port's sends as the analysis's: the same two fields. -/
def Asks.toC {M : Type → Type} (io : Asks M) : FlockComplete.Asks M := ⟨io.link, io.round⟩

section

variable {M : Type → Type} [Monad M]

theorem squeezeP_eq (io : Asks M) (kind : UInt8) (n : ℕ) :
    squeezeP io kind n = FlockComplete.squeezeP io.toC kind n := rfl

theorem sampleP_eq (io : Asks M) : sampleP io = FlockComplete.sampleP io.toC := rfl

theorem sampleVecP_eq (io : Asks M) (n : ℕ) : sampleVecP io n = FlockComplete.sampleVecP io.toC n := rfl

theorem sampleKP_eq (io : Asks M) : sampleKP io = FlockComplete.sampleKP io.toC := rfl

/-- **The firewall's zerocheck** is the masked session's from its source, at the masked values `mk`. -/
theorem zerocheckFw_eq [MonadLiftT V M] (io : Asks M) (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (mk : ℕ → F128) :
    zerocheckFw io m digest cap root lay mk = FlockComplete.zerocheckCut io.toC m digest cap root lay mk := rfl

/-- **The firewall's lincheck** is the masked session's from its source, at the masked values `mk`. -/
theorem lincheckFw_eq [MonadExceptOf String M] [MonadLiftT V M] (io : Asks M) (c : CircuitFold) (m : ℕ)
    (lay : Zk.PadLayout) (mk : ℕ → F128) (zc : ZcOut) :
    lincheckFw io c lay mk zc = FlockComplete.lincheckCut io.toC c m lay mk zc := rfl

/-- **The firewall's ring switch** is the masked session's from its source, at the switched vectors `sv`. -/
theorem ringSwitchP_eq [MonadLiftT V M] (io : Asks M) (sv : ℕ → Array F128) (claims : Array Claim) :
    ringSwitchP io sv claims = FlockComplete.ringSwitchCut io.toC sv claims := rfl

end

theorem basisVals_eq : basisVals = FlockComplete.basisVals := rfl

end Flock.Firewall.Finish
