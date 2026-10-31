import Definitions.Pouw.Lead.Gap

/-!
# A forward of the lead, as the network certifier counts it (trusted definitions)

The network certifier counts the node's bits per forward, not per call. A forward is open from its first call's `A`
root to its words root, and the certifier counts every bucket of the node's links once, against the open forward; at
the words root it releases the links, and the release carries `⌈log₂(r + 1)⌉` bits once, at the window's rate `r`.

* **A forward** (`Forward`): its calls, each at the strategy that served it and the coins the certifier released to it
  (`FwdCall`), the bits the certifier counted over its span (`count`) and its release's bits (`release`).
* **Its bits** (`Forward.bits`): its calls' bits summed, `cap · Σ rows · n · K / fwdMacs`, a whole forward of `T`
  tokens' `cap · T`.
* **Accepted** (`Forward.Accepted`): the certifier refuses a forward whose counted and release bits pass its bits.
* **Its network term** (`Forward.networkRegs`): each call's `betaLead` a bit its execution exchanged, in registers.
  That the calls' bits are the forward's counted ones is a premise (`Pouw.Assumptions.Lead.SerialForwards`).
-/

namespace Pouw.Lead

/-- **A call of a forward**: its layout and commitment format, the strategy that served it, and the coins the
certifier released to it. -/
structure FwdCall where
  /-- The call's layout. -/
  L : Layout
  /-- The call's commitment format. -/
  C : Commit L
  /-- The strategy that served the call. -/
  σ : Strategy L
  /-- The coins the certifier released to the call. -/
  ω : Coins L

/-- The bits the call's execution exchanged. -/
def FwdCall.egress (c : FwdCall) : ℕ := (c.σ.exec c.ω).egress

/-- **A forward** as the network certifier counts it: its calls, from its first `A` root to its words root; the bits
the certifier counted over that span, each bucket once; and the bits its release at the words root carries. -/
structure Forward where
  /-- The forward's calls. -/
  calls : List FwdCall
  /-- The bits the certifier counted while the forward was open. -/
  count : ℕ
  /-- The bits the release at the forward's words root carries, `⌈log₂(r + 1)⌉`. -/
  release : ℕ

/-- **A forward's bits**: its calls' bits summed (`Layout.callBits`), `cap · Σ rows · n · K / fwdMacs`. -/
def Forward.bits (F : Forward) : ℚ := (F.calls.map fun c => c.L.callBits).sum

/-- **The forward is accepted**: the bits the certifier counted over its span and its release's bits are at most its
bits. The certifier refuses every other forward. -/
def Forward.Accepted (F : Forward) : Prop := ((F.count + F.release : ℕ) : ℚ) ≤ F.bits

/-- **A forward's network term** in registers: `betaLead` a bit each call's execution exchanged, over 32. -/
def Forward.networkRegs (F : Forward) : ℚ := (F.calls.map fun c => betaLead c.L c.C * c.egress).sum / 32

end Pouw.Lead
