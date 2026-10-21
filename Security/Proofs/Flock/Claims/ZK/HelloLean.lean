import Proofs.Flock.Claims.ZK.Release
import Proofs.Flock.Claims.ZK.CoinLayer
import Flock.Firewall.Hello

/-!
# The `Hello` at Lean's `flock-firewall hello`

`flock-firewall hello` (`Flock.Firewall.Hello`) is the code of record of a firewalled session's `Hello`: the verifier's
own session parameters at the session's statement, with the nonce it draws in `coins.nonce`, framed as `Req::Hello`
(live `PROTOCOL.md` §10). Here that function is held to the coin layer's `Hello`.

* `Flock.Firewall.Hello.json_eq`: its `Hello` is `CoinLayer.helloJson`, by `rfl`.
* `CoinLayer.helloMsg`: its message at the statement `x`'s setup (`FlockDeploy.setupOf`) and a nonce;
  `CoinLayer.helloD_eq`: that is `CoinLayer.helloD`, the session's `Hello`.
* `LeanRelease.HelloIs`: the server's `Hello` is `helloMsg`, a named trust narrower than `HelloEq`. It is a fact about
  what the transport hands `flock-firewall hello` (`x`'s statement files and the verifier's draw) and how it frames the
  `Hello` it answers, not about how a `Hello` is made. `Refine`'s `LeanRelease.helloEq_of_helloIs` gives `HelloEq`
  from it.
-/

namespace Flock.Firewall.Hello

/-- **`flock-firewall hello`'s `Hello` is the coin layer's.** -/
theorem json_eq : json = Claims.ZKMal.CoinLayer.helloJson := rfl

end Flock.Firewall.Hello

namespace Claims.ZKMal.CoinLayer

open Claims

/-- **What `flock-firewall hello` answers at `x`**, framed: the `Hello` message (`Flock.Firewall.Hello.msg`) of the
session `x`'s setup makes, at the nonce's bytes; the empty message where `x` sets up no session. -/
def helloMsg (x : Msg) (ν : Nonce) : Msg :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, _) => _root_.Flock.Firewall.Hello.msg spec ⟨(nb ν).toArray⟩
  | none => []

/-- **`flock-firewall hello`'s message is the session's `Hello`.** -/
theorem helloD_eq (x : Msg) (ν : Nonce) : helloD x ν = helloMsg x ν := rfl

end Claims.ZKMal.CoinLayer

namespace Claims.ZKServer

open Claims Claims.ZKMal

namespace LeanRelease

/-- **The server's `Hello` is `flock-firewall hello`'s**: at every tape, the server's own `Hello` is `CoinLayer.helloMsg`
at the tape's nonce. A named hypothesis, one of the transport's trusts, narrower than `HelloEq`: that the transport hands
`flock-firewall hello` the statement `x`'s files (the circuit, the verifier's own public file, the lookup tables, the
table count, the partition and the program) and the verifier's draw, that the nonce it answers at is the tape's, and that
`flock-circuit` frames the text it answers as `Req::Hello` (`firewall::LeanHello`); never how a `Hello` is made. -/
def HelloIs (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Claims.Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (x : Msg) : Prop :=
  ∀ (ν : (CoinLayer.session nr T).Nonce) (t : (CoinLayer.session nr T).Tape), L.hello x (ν, t) = CoinLayer.helloMsg x ν

end LeanRelease

end Claims.ZKServer
