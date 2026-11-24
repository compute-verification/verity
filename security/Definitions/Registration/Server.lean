module

public import Definitions.Registration.Physical
public import Mathlib.Tactic.DeriveFintype

/-!
# A server of GPUs as hardware of the actual-circuit model (trusted)

The hardware whose limits are numbers per leaf and per link (`Hardware.ofLeaves`): every set of leaves is a location,
doing its leaves' work a tick and carrying their bits, a gate reads only gates placed no later than itself, and there
is no ring. A machine with several resources (a pipe's lanes, the issue slots) is one such hardware per resource, each
with that resource's pricing of the gates and its units a tick.

A server of `n` GPUs (`server`) is that hardware at the leaves `Loc n`: each GPU's die and its device memory, the host,
the switch between the GPUs and the world outside, with a link from each die to its device memory, to the host and to
the switch, and one from the host to the world outside. Device memory computes nothing.
-/

@[expose] public section

namespace ResourceAccounting

open Finset

/-- **The hardware with limits per leaf and per link**: a location does its leaves' work `compute` a tick and carries
their bits `M`, the links `ends` carry `β` bits a tick each way, and a gate reads gates placed no later than itself. -/
def Hardware.ofLeaves {L E : Type*} (compute M : L → ℕ) (ends : E → L × L) (β : E → ℕ) : Hardware L E where
  units := Set.univ
  ℓ _ _ := 0
  compute X := ∑ p ∈ X, compute p
  M X := ∑ p ∈ X, M p
  ends := ends
  β := β
  ring := ∅
  D := 0
  span := 0
  cert := ∅
  trusted := ∅
  bucket := id
  jitter := 0
  record _ := ∅

/-- **The leaves of a server of `n` GPUs**: each GPU's die and its device memory, the host, the switch between the
GPUs, and the world outside. -/
inductive Loc (n : ℕ)
  | die (i : Fin n)
  | mem (i : Fin n)
  | host
  | switch
  | outside
  deriving DecidableEq, Fintype

/-- **The links of a server of `n` GPUs**: each die's to its device memory (`store`), to the host (`bus`) and to the
switch (`peer`), and the host's to the world outside (`net`). -/
inductive Wire (n : ℕ)
  | store (i : Fin n)
  | bus (i : Fin n)
  | peer (i : Fin n)
  | net
  deriving DecidableEq, Fintype

/-- The leaves a server's link joins. -/
def Wire.ends {n : ℕ} : Wire n → Loc n × Loc n
  | .store i => (.mem i, .die i)
  | .bus i => (.host, .die i)
  | .peer i => (.switch, .die i)
  | .net => (.outside, .host)

/-- **A server of `n` GPUs**, for one resource: a die does `die` of it a tick, the host `host` and the switch
`fabric`; a die carries `chip` bits, a device memory `dram`, the host `ram`, the switch `queue` and the world outside
`world`; and the links carry `store`, `bus`, `peer` and `net` bits a tick each way. -/
def server (n die host fabric chip dram ram queue world store bus peer net : ℕ) : Hardware (Loc n) (Wire n) :=
  Hardware.ofLeaves
    (fun | .die _ => die | .host => host | .switch => fabric | _ => 0)
    (fun | .die _ => chip | .mem _ => dram | .host => ram | .switch => queue | .outside => world)
    Wire.ends
    (fun | .store _ => store | .bus _ => bus | .peer _ => peer | .net => net)

/-- The dies of a server. -/
def dies (n : ℕ) : Finset (Loc n) := univ.image Loc.die

/-- The device memories of a server. -/
def drams (n : ℕ) : Finset (Loc n) := univ.image Loc.mem

end ResourceAccounting
