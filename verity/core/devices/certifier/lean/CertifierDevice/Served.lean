import NetworkCertifier.Schedule

/-!
# A served link's admission rule

The executable forms of `served.py`'s reference that the proxy's ingress side reads (`active.Bucketer(served=…)`): a
served link's public sizes (`ServedProfile`), each request's anchor (`anchors`), the window's egress counts derived from
its ingress counts under a sync (`derive`), and whether a window still ends idle with one more request admitted
(`admits`). `handing_buckets` is `NetworkCertifier.handingBuckets` with one window of advice; `grid.released` is
`releasedCounts`.
-/

namespace CertifierDevice

open NetworkCertifier (ClockSync LogicalFrame handingBuckets)

/-- `ServedProfile`: every response's frames F (FIXLEN), every request's frames F_in (a power of two), the link's
windows, and `own`, the ingress certifier's own frames at the head of every window's first bucket (1 with the salt on). -/
structure ServedProfile where
  responseFrames : Nat
  requestFrames : Nat
  windows : Nat
  own : Nat := 0

/-- `ServedProfile.__post_init__`'s check. -/
def ServedProfile.check (p : ServedProfile) : Except String Unit :=
  if p.responseFrames < 1 || p.windows < 1 || p.requestFrames < 1 || p.requestFrames &&& (p.requestFrames - 1) != 0 ||
      p.own > 1 then
    throw "a served profile has F ≥ 1 frames a response, a power of two of frames a request, one or more windows and 0 \
      or 1 own frames a window"
  else pure ()

/-- `anchors`: each whole request's anchor in the window, the bucket whose release completes its `n` frames. -/
def anchors (ingress : List Nat) (n : Nat) : List Nat := Id.run do
  let mut out := #[]
  let mut done := 0
  let mut total := 0
  for (c, t) in ingress.zipIdx do
    total := total + c
    let k := total / n
    out := out ++ Array.replicate (k - done) t
    done := max done k
  return out.toList

/-- `grid.released`: bucket t releases min(queue + enqueued[t], r) frames, and the rest wait, in order. -/
def releasedCounts (allocation : Nat) (enqueued : List Nat) : List Nat :=
  (enqueued.foldl (fun (acc : Array Nat × Nat) e =>
    let n := min (acc.2 + e) allocation
    (acc.1.push n, acc.2 + e - n)) (#[], 0)).1.toList

/-- `derive`: the window's egress counts from its ingress counts under `sync`, each request answered by F frames, handed
by `handingBuckets` and released at the egress allocation from an idle start, the first `own` frames of bucket 0 the
certifier's own; none when they aren't there or the window doesn't end idle. -/
def derive (ingress : List Nat) (sync : ClockSync) (p : ServedProfile) (allocation : Nat) : Option (List Nat) := do
  let T := ingress.length
  let ingress ← if p.own = 0 then some ingress else
    match ingress with
    | [] => none
    | c :: rest => if c < p.own then none else some ((c - p.own) :: rest)
  let total := ingress.sum
  if total % p.requestFrames ≠ 0 || total / p.requestFrames * p.responseFrames > T * allocation then none
  let frames : List LogicalFrame :=
    (anchors ingress p.requestFrames).flatMap fun a => (List.range p.responseFrames).map fun k => ⟨[], a, k⟩
  let hands := handingBuckets T 1 (fun _ => sync) frames
  if hands.any (· ≥ T) then none
  let counts := releasedCounts allocation ((List.range T).map fun t => hands.count t)
  if counts.sum = frames.length then some counts else none

/-- `admits`: whether the window still ends idle with `released` the ingress counts of its buckets so far and `queued`
frames, the request's being admitted included, released FIFO at `ingressAllocation` a bucket from the next bucket on:
false when one would be released after the window's last bucket. -/
def admits (released : Array Nat) (queued : Nat) (sync : ClockSync) (p : ServedProfile)
    (allocation ingressAllocation T : Nat) : Bool :=
  let s := released.size
  if 0 < queued && s + (queued - 1) / ingressAllocation ≥ T then false else
  let counts := (released.toList ++ List.replicate (T - s) 0).zipIdx.map fun (c, t) =>
    if t < s then c else c + min ingressAllocation (queued - (t - s) * ingressAllocation)
  (derive counts sync p allocation).isSome

/-- `Admission`: the admission rule at the proxy, under each window's declared sync where the proxy holds it (`syncs`),
else the slowest of Σ_sync. -/
structure Admission where
  profile : ServedProfile
  allocation : Nat
  syncs : Array ClockSync
  slowest : ClockSync

def Admission.sync (a : Admission) (w : Nat) : ClockSync := a.syncs[w]?.getD a.slowest

end CertifierDevice
