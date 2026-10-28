import PouwWindow.Side
import PouwWindow.Record
import Identity.Domains

/-!
# The tile instances' public inputs (`window.py`, "The tile instances' public inputs")

A window's K′ drawn tiles are proved in S fixed-shape sessions of `cap` instances (`sessionCap?`, `sessions?`); session σ's
instance s is tile `drawn[σ·cap + s]`, or N − 1 past the draw (`instanceTs?`). `publicInputs?` is F_σ, the verifier's own
`--public-inputs` file of the session: a header line, then each instance's ports (`PUBLIC_PORTS`: t as a u128, the window
root, its domain, side A's and side B's domains, the pass's -h3 domain), little-endian, instance-major. The registration is
one-stage's record as JSON, read as Python reads it.
-/

namespace PouwWindow

open Lean (Json)

def PUBLIC_INPUTS_FORMAT : String := Domains.FLOCK_PUBLIC_INPUTS
/-- `M_MAX`: C-Flock's largest session, m = k_log + nbl ≤ 35. -/
def mMax : Nat := 35

/-- `PUBLIC_PORTS`: (port, words, seed, derivation). -/
def publicPorts : List (String × Nat × String × String) := [
  ("t", 8, "drawn", Domains.POUW_PERIOD_T),
  ("root", 32, "registration", Domains.POUW_PERIOD_ROOT),
  ("domain", 32, "registration", Domains.POUW_PERIOD_DOMAIN),
  ("domain_a", 32, "registration", Domains.POUW_PERIOD_SIDE_DOMAIN_A),
  ("domain_b", 32, "registration", Domains.POUW_PERIOD_SIDE_DOMAIN_B),
  ("domain_h3", 32, "pass", Domains.POUW_PERIOD_DOMAIN_H3)]

/-- `session_cap`: the most instances one session holds when each block of 2^`kLog` bits holds `perBlock` of them. -/
def sessionCap? (kLog perBlock : Int) : Except String Nat :=
  if 1 ≤ kLog && kLog ≤ (mMax : Int) - 3 && 1 ≤ perBlock then .ok (perBlock.toNat * 2 ^ (mMax - kLog.toNat))
  else .error s!"a block of 2^{kLog} bits does not fit a session of m <= {mMax}"

/-- `sessions`: S, the sessions of `cap` instances that hold K′ = `k` drawn tiles. -/
def sessions? (k cap : Int) : Except String Nat :=
  if 1 ≤ k && 1 ≤ cap then .ok ((k.toNat + cap.toNat - 1) / cap.toNat)
  else .error s!"K′ and the cap are positive, not {k} and {cap}"

/-- `_registered_root`: the registration's one root of port `entries`, and its domain id. -/
def registeredRoot? (reg : Json) : Except String (ByteArray × ByteArray) := do
  let items := ((pyGet? reg "roots").bind pyList?).getD []
  let roots := items.filter fun r => match r, pyGet? r "port" with
    | .obj _, some (.str p) => p == port
    | _, _ => false
  let [r] := roots | throw s!"the registration names {roots.length} roots of port '{port}', not one"
  let hex (k : String) : Except String ByteArray := match pyGet? r k with
    | some (.str s) => PouwCommit.pyFromHex s
    | _ => throw s!"the window root's {k} is not hex"
  let root ← hex "root"
  let dom ← hex "domain"
  if root.size != 64 || dom.size != 64 then throw "the window root and its domain id are 64 bytes"
  pure (root, dom)

/-- `instance_ts`: session σ's tile indices. -/
def instanceTs? (reg : Json) (drawn : List Json) (sigma : Json) (cap : Int) : Except String (List Nat) := do
  let pop := pyGet? reg "population"
  let some n := pop.bind strictInt?
    | throw s!"the registration's population {(pop.map pyRepr).getD "None"} is not a tile count of at most 2^{tBits}"
  if !(1 ≤ n && n ≤ 2 ^ tBits) then throw s!"the registration's population {n} is not a tile count of at most 2^{tBits}"
  let law := pyGet? reg "law"
  let k := match law with
    | some l@(.obj _) => if pyGet? l "law" == some (Json.str "subset") then (pyGet? l "k").bind strictInt? else none
    | _ => none
  let some k := k.filter (1 ≤ ·) | throw "the registration's law is not subset:K′"
  let some ds := drawn.mapM strictInt? | throw s!"the draw is not {k} distinct ascending indices of [0, {n})"
  if (ds.length : Int) != k || !increasing ds || !(ds.all fun t => 0 ≤ t && t < n) then
    throw s!"the draw is not {k} distinct ascending indices of [0, {n})"
  let some σ := pyInt? sigma | throw s!"session {pyRepr sigma} is not one of the window's sessions"
  let s ← sessions? k cap
  if !(0 ≤ σ && σ < s) then throw s!"session {σ} is not one of the window's {s}"
  let σ := σ.toNat
  let cap := cap.toNat
  pure ((List.range cap).map fun i => if σ * cap + i < k.toNat then (ds[σ * cap + i]!).toNat else n.toNat - 1)

/-- The header `json.dumps({"format", "instances", "ports"}, sort_keys=True)` writes, with its default separators. -/
def publicHeader (cap : Nat) : String :=
  let ports := publicPorts.map fun (p, w, s, d) =>
    s!"\{\"derivation\": \"{d}\", \"port\": \"{p}\", \"seed\": \"{s}\", \"words\": {w}}"
  s!"\{\"format\": \"{PUBLIC_INPUTS_FORMAT}\", \"instances\": {cap}, \"ports\": [{", ".intercalate ports}]}"

/-- `public_inputs`: F_σ, the verifier's own `--public-inputs` file for session σ's `cap` tile instances. -/
def publicInputs? (reg : Json) (drawn : List Json) (sigma : Json) (cap : Int) (domainH3 : ByteArray) :
    Except String ByteArray := do
  let (root, dom) ← registeredRoot? reg
  if domainH3.size != 64 then throw "the pass's -h3 domain id is 64 bytes"
  let common := root ++ dom ++ (← sideDomain? dom "a").id ++ (← sideDomain? dom "b").id ++ domainH3
  let ts ← instanceTs? reg drawn sigma cap
  let body := ts.foldl (fun acc t => acc ++ Entry.le 16 t ++ common) .empty
  pure ((publicHeader cap.toNat ++ "\n").toUTF8 ++ body)

end PouwWindow
