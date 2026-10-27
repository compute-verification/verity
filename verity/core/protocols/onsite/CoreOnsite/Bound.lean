/-
The joint bound over a monitoring period of server-time blocks, evaluated on onsite's answers: the right-hand side of
`Accounting.Guarantees.JointHorizonBlocks` (`Security/Definitions/Accounting/Guarantees/JointBlocks.lean`),
$$\sum_b \sum_a \delta_{b,a} + \sum_b \varepsilon_{\mathrm{wipe},b} + \sum_b \delta_{\mathrm{PoUS},b},$$
and beside it each resource's pooled figure, from the statement that pools it (`CoreOnsite.Pooled`).

Each block is one accepted onsite answer. Its terms are each subprotocol's error, the probability its accepted claim
fails, read where that subprotocol publishes it (`SOURCES`): the network certifier's, PoUS's, the remote bandwidth
audit's and the wipe's `eps`, PoUW's period `delta` (the δ its periods' ε are stated at), and sampled proofs' `delta`
from its profile. The wipe's is `ε_wipe`; the others are the audits' `δ_{b,a}`. No subprotocol publishes the memory
term's `δ_PoUS` (weights-only PoUS at a share of 0.07), so that sum is not evaluated and `OPEN` names it.

`evaluate` gives each block's terms and their sum, the run's total (its blocks' sums added), and for each count `B`
of a monitoring period whose every block is audited as the run's are, `B` times the run's largest block sum, which
bounds the sum over those `B` blocks. An error of 1 or more is vacuous: a union over the blocks says nothing past about
`1/δ` of them. So each period also gives, at each `δ` asked for, its pooled reading (`horizons`): the pooled figures,
whose statements pay `δ` once over the period rather than in every block. They are integrity's faulty proof blocks
(`IntegrityHorizon`) with the per-block threshold beside them (`IntegrityPeriod` at `δ/B`), PoUS's certified sampling
periods (`pooledShort_log`) and the network certifier's leak (`HorizonLeakAccepted`), each at the run's worst block. A
resource whose inputs no block gives says why. `OPEN` lists the figures no guarantee states. A block that wasn't accepted
establishes no claim, so it has no terms, and `evaluate` refuses it.
-/
import CoreProtocol.Protocol
import CoreProtocol.Ratio
import CoreOnsite.Pooled

namespace Core.Onsite.Bound

open Lean (Json)
open Flock (V)

def STATEMENT : String := "Accounting.Guarantees.JointHorizonBlocks"

def SAMPLED_PROOFS : String := "verity/sampled-proofs/one-stage/v2"
def POUS : String := "verity/pous/p2/v4/run"
def CERTIFIER : String := "verity/certifier/v0"
def POUW : String := "verity/pouw/period/v1"

/-- What the report would give and no Lean guarantee states, each with what is missing. -/
def OPEN : List (String × String) :=
  [("joint: δ_PoUS", "the memory term's δ_PoUS (weights-only PoUS at a share of 0.07): no subprotocol publishes it"),
    ("bandwidth", "no guarantee pools the remote bandwidth audit's per-period eps over a monitoring period (the " ++
      "union B·eps is stated in no Lean)"),
    ("space", "proved, not locked: pooledShort_log (Security/Proofs/Pous/Pooled.lean) is a proved theorem, evaluated " ++
      "here, but no lock lists a guarantee stating it"),
    ("communication per machine", "no guarantee joins the machines' HorizonLeakAccepted by a union over the " ++
      "deployment's machines; the pooled figure takes every machine's blocks as one sequence")]

/-- Where a subprotocol's answer states its error: a path into its public outputs, or into its profile. -/
inductive Source where
  | pub (path : List String)
  | profile (path : List String)

/-- Each subprotocol's error, by its id, in the order of the ids. -/
def SOURCES : List (String × Source) :=
  [(CERTIFIER, .pub ["eps"]), (POUS, .pub ["eps"]), (POUW, .pub ["delta"]),
    ("verity/remote-bandwidth/v0", .pub ["eps"]), (SAMPLED_PROOFS, .profile ["delta"]), ("verity/wipe/v0", .pub ["eps"])]

/-- `a + b` in lowest terms. -/
def add (a b : Ratio) : Ratio :=
  let n := a.num * b.den + b.num * a.den
  let d := a.den * b.den
  ⟨n / Nat.gcd n d, d / Nat.gcd n d⟩

/-- `k · r` in lowest terms. -/
def scale (k : Nat) (r : Ratio) : Ratio :=
  let n := k * r.num
  ⟨n / Nat.gcd n r.den, r.den / Nat.gcd n r.den⟩

def sum (rs : List Ratio) : Ratio := rs.foldl add ⟨0, 1⟩

def largest (rs : List Ratio) : Ratio := rs.foldl (fun m r => if m.le r then r else m) ⟨0, 1⟩

private def at? (j : Json) (path : List String) (ctx : String) : V Json :=
  path.foldlM (fun j k => Flock.J.get j k ctx) j

/-- An accepted answer of the composition `id`: its public outputs and its profile, each keyed by subprotocol. -/
def parts (id : String) (a : Answer) : V (Json × Json) := do
  if a.protocol != id then throw s!"an answer of {a.protocol}, not {id}"
  let (.accept, some pub, some prof) := (a.verdict, a.published, a.profile)
    | throw s!"a {a.verdict.name} block establishes no claim: {a.reason.getD ""}"
  pure (pub, prof)

/-- A block's terms: each subprotocol's error, read from its public outputs `pub` and profile `prof`. -/
def termsOf (pub prof : Json) : V (List (String × Ratio)) :=
  SOURCES.mapM fun (c, src) => do
    let (j, path, where_) := match src with
      | .pub p => (pub, p, "public")
      | .profile p => (prof, p, "profile")
    let ctx := s!"{c}'s {where_}"
    let .str s ← at? (← Flock.J.get j c where_) path ctx | throw s!"{ctx}: its error is not a ratio"
    pure (c, ← Ratio.ofString s ctx)

/-- Integrity's inputs: the least fraction `q = kd/n` of its proof blocks a block's draw covers, with that block's `n`
and `kd` (the largest `n` at that `q`). -/
structure Draw where
  q : Ratio
  n : Nat
  kd : Nat

/-- PoUS's inputs: the certified blocks' largest `eps` and least storage. -/
structure Store where
  p : Ratio
  sBits : Nat

/-- The network certifier's inputs: the least fraction a block's audit draws covers, and the block whose own terms
(schedules, cells, each audit's free inputs and jitter) cost most, with its proof blocks and its whole `K`. -/
structure Leak where
  q : Ratio
  every : Nat
  units : Nat
  K : Nat

private def ratioOf (n d : Nat) : Ratio := (Ratio.mk' n d).getD ⟨0, 1⟩

/-- A block's integrity draw: sampled proofs' checked level (its last) and the proof blocks drawn at it. -/
def drawOf (prof : Json) : V Draw := do
  let ctx := s!"{SAMPLED_PROOFS}'s profile"
  let p ← Flock.J.get prof SAMPLED_PROOFS "profile"
  let some l := (← Flock.J.arr p "levels" ctx).back? | throw s!"{ctx}: no level"
  let name ← Flock.J.str l "name" s!"{ctx}.levels"
  let n ← Flock.J.nat l "size" s!"{ctx}.levels"
  let kd ← Flock.J.nat (← Flock.J.get p "drawn" ctx) name s!"{ctx}.drawn"
  if n == 0 || kd == 0 then throw s!"{ctx}: a draw of {kd} of {n} proof blocks covers no fraction of them"
  pure ⟨ratioOf kd n, n, kd⟩

/-- A block's PoUS storage, `none` when its certificate didn't certify. -/
def storeOf (pub : Json) : V (Option Store) := do
  let ctx := s!"{POUS}'s public"
  let j ← Flock.J.get pub POUS "public"
  if !(← Flock.J.bool j "certified" ctx) then return none
  pure (some ⟨← Ratio.ofString (← Flock.J.str j "eps" ctx) s!"{ctx}.eps", ← Flock.J.nat j "s_bits" ctx⟩)

/-- A block's leak terms, from its period check's itemized `K`: each audit's proof blocks and draw, and the free inputs and
jitter of its terms (the scans' in the rows form). -/
def leakOf (pub : Json) : V Leak := do
  let ctx := s!"{CERTIFIER}'s public"
  let j ← Flock.J.get pub CERTIFIER "public"
  let audits ← Flock.J.arr j "audits" ctx
  let terms ← if Flock.J.has j "scans" then Flock.J.arr j "scans" ctx else pure audits
  let every ← terms.foldlM (fun s t => do
    pure (s + (← Flock.J.nat t "free" s!"{ctx}.audits") + (← Flock.J.nat t "jitter" s!"{ctx}.audits")))
    ((← Flock.J.nat j "cells" ctx) + (← Flock.J.nat j "schedules" ctx))
  let draws ← audits.mapM fun a => do
    pure ((← Flock.J.nat a "units" s!"{ctx}.audits"), (← Flock.J.nat a "drawn" s!"{ctx}.audits"))
  let some q := (draws.filter (·.1 > 0)).foldl (fun m (n, k) =>
      let r := ratioOf k n; match m with | some m => some (if m.le r then m else r) | none => some r) none
    | throw s!"{ctx}: no audit with proof blocks, so no fraction drawn"
  if q.num == 0 then throw s!"{ctx}: an audit drew none of its proof blocks"
  pure ⟨q, every, draws.foldl (· + ·.1) 0, ← Flock.J.nat j "K" ctx⟩

private def leastDraw (a b : Draw) : Draw :=
  if a.q.le b.q && (!b.q.le a.q || b.n ≤ a.n) then a else b

private def worstLeak (a b : Leak) : Leak :=
  let q := if a.q.le b.q then a.q else b.q
  if b.every > a.every || (b.every == a.every && b.K > a.K) then { b with q } else { a with q }

/-- Each resource's inputs over every block, or why it has none. -/
structure Inputs where
  draw : V Draw
  store : V Store
  leak : V Leak

private def every {α} (ps : List (Json × Json)) (f : Json × Json → V α) (join : α → α → α) : V α := do
  let some x := (← ps.mapM f).foldl (fun m a => some (match m with | some m => join m a | none => a)) none
    | throw "no accepted block"
  pure x

def inputsOf (ps : List (Json × Json)) : Inputs :=
  { draw := every ps (drawOf ·.2) leastDraw
    store := do
      let some s := (← ps.mapM (storeOf ·.1)).filterMap id |>.foldl (fun m (s : Store) => some (match m with
          | some m => ⟨if m.p.le s.p then s.p else m.p, min m.sBits s.sBits⟩ | none => s)) none
        | throw "no block's PoUS certified, so no eps to pool"
      pure s
    leak := every ps (leakOf ·.1) worstLeak }

private def ratioJ (r : Ratio) : Json := .str r.toString
private def natJ (n : Nat) : Json := Lean.toJson n
private def why (e : String) : Json := Json.mkObj [("why", .str e)]

private def figure {α} (x : V α) (f : α → List (String × Json)) : Json :=
  match x with
  | .ok a => Json.mkObj (f a)
  | .error e => why e

/-- The pooled figures over a monitoring period of `B` blocks at `δ`, each block audited as the run's worst. -/
def pooledJ (i : Inputs) (B : Nat) (δ : Ratio) : Json :=
  Json.mkObj [
    ("integrity", figure i.draw fun d => [("statement", .str "Flock.Guarantees.IntegrityHorizon"),
      ("tied_by", .str "Properties.Onsite.integrityHorizonCovers"),
      ("faulty_blocks_at_most", natJ (Pooled.faultyBlocks d.q δ)), ("proof_blocks", natJ (B * d.n)),
      ("except_with", .str "δ + Σ_t (ε_ks,t + ε_bind,t); the run computes neither ε_ks nor ε_bind")]),
    ("integrity_per_block", figure i.draw fun d => [("statement", .str "Flock.Guarantees.IntegrityPeriod"),
      ("tied_by", .str "Properties.Onsite.integrityPeriodCovers"),
      ("at", .str "δ/B a block"), ("faulty_blocks_at_most", natJ (Pooled.threshold d.n d.kd ⟨δ.num, δ.den * B⟩ - 1)),
      ("proof_blocks", natJ d.n)]),
    ("space", figure i.store fun s => [("statement", .str "Pous.Period.SecurityProofs.pooledShort_log"),
      ("status", .str "proved, not locked"), ("tied_by", .str "Properties.Onsite.pooledShortCovers"),
      ("certified_at_least", natJ (Pooled.certified B s.p δ)), ("periods", natJ B), ("s_bits_each", natJ s.sBits),
      ("given", .str "FixedBefore, and PeriodHeld's premises CoinCommitHides and P2RingAuditSecure")]),
    ("communication", figure i.leak fun l =>
      let (k, f, faulty) := Pooled.leakBits B l.every l.units l.q δ
      [("statement", .str "NetTiming.Guarantees.HorizonLeakAccepted"),
        ("tied_by", .str "Properties.Onsite.horizonLeakCovers"), ("K_bits", natJ k), ("every_block_bits", natJ l.every),
        ("faulty_blocks_at_most", natJ f), ("faulty_once_bits", natJ faulty), ("proof_blocks", natJ (B * l.units)),
        ("except_with", .str "δ + Σ_i e_i + Σ_i e'_i"),
        ("conditional", .str ("on hesc and hleak (SampledKSLive at each block's history, the live " ++
          "SampledLeakAccepted), not yet discharged")),
        ("jitter_term", .str "assumed (J): NetTiming.JitterBounded at jitter bound 1, which the certifier doesn't meet")])]

private def horizonJ (blocks : Nat) (e : Ratio) : List (String × Json) :=
  [("blocks", natJ blocks), ("error", ratioJ e), ("vacuous", .bool ((⟨1, 1⟩ : Ratio).le e))]

private def inputsJ (i : Inputs) : Json :=
  Json.mkObj [
    ("integrity", figure i.draw fun d => [("q", ratioJ d.q), ("n", natJ d.n), ("kd", natJ d.kd)]),
    ("space", figure i.store fun s => [("p", ratioJ s.p), ("s_bits", natJ s.sBits)]),
    ("communication", figure i.leak fun l =>
      [("q", ratioJ l.q), ("every_block_bits", natJ l.every), ("proof_blocks", natJ l.units), ("K", natJ l.K)])]

/-- The bound on the accepted answers `as` of the composition `id`, and over monitoring periods of each of `counts`
blocks with the pooled figures at each of `deltas`. -/
def evaluate (id : String) (as : List Answer) (counts : List Nat) (deltas : List Ratio := []) : V Json := do
  let ps ← as.mapM (parts id)
  let terms ← ps.mapM fun (pub, prof) => termsOf pub prof
  let sums := terms.map fun ts => sum (ts.map (·.2))
  let i := inputsOf ps
  pure <| Json.mkObj [("statement", .str STATEMENT),
    ("open", .arr (OPEN.map fun (f, m) => Json.mkObj [("figure", .str f), ("missing", .str m)]).toArray),
    ("blocks", .arr (terms.zip sums |>.map fun (ts, s) =>
      Json.mkObj [("error", ratioJ s), ("terms", Json.mkObj (ts.map fun (c, r) => (c, ratioJ r)))]).toArray),
    ("inputs", inputsJ i),
    ("run", Json.mkObj (horizonJ as.length (sum sums))),
    ("horizons", .arr (counts.map fun b => Json.mkObj (horizonJ b (scale b (largest sums)) ++
      [("pooled", Json.mkObj (deltas.map fun d => (d.toString, pooledJ i b d)))])).toArray)]

private def natOf? (s : String) : Option Nat := if !s.isEmpty && s.all Char.isDigit then some s.toNat! else none

/-- `--delta R` options and block counts, each `0 < R < 1` and each count positive. -/
def parse : List String → Option (List Ratio × List Nat)
  | [] => some ([], [])
  | "--delta" :: r :: rest => do
    let (ds, bs) ← parse rest
    let d ← (Ratio.ofString r "--delta").toOption
    if 0 < d.num && d.num < d.den then some (d :: ds, bs) else none
  | b :: rest => do
    let (ds, bs) ← parse rest
    let n ← natOf? b
    if n == 0 then none else some (ds, n :: bs)

/-- `onsite bound ANSWERS [--delta R]… [BLOCKS…]`: `evaluate` on the answer lines of the file `ANSWERS`, one
`onsite check` outcome a line (JSON, in the shape `Answer.toJson` writes), over monitoring periods of each count of `BLOCKS` with
the pooled figures at each `R`. Prints the bound's canonical JSON line and exits 0, or the reason it is refused on
stderr and exits 1; 64 on a usage error. -/
def main (id : String) (args : List String) : IO UInt32 := do
  let usage : IO UInt32 := do IO.eprintln s!"usage: {id} bound ANSWERS [--delta R]… [BLOCKS…]"; return 64
  let file :: rest := args | usage
  let some (deltas, counts) := parse rest | usage
  let lines := (← IO.FS.readFile file).splitOn "\n" |>.filter (!·.all Char.isWhitespace)
  let got : V String := do
    let as ← lines.mapM fun l => do Answer.ofJson (← Json.parse l)
    Flock.canon (← evaluate id as counts deltas)
  match got with
  | .ok line => IO.println line; return 0
  | .error e => IO.eprintln s!"{id} bound: {e}"; return 1

#guard (add ⟨1, 100⟩ ⟨1, 1000000⟩, scale 4 ⟨1, 6⟩, sum [], largest [⟨1, 3⟩, ⟨1, 2⟩, ⟨1, 4⟩]) ==
  (⟨10001, 1000000⟩, ⟨2, 3⟩, ⟨0, 1⟩, ⟨1, 2⟩)
#guard parse ["--delta", "1/100", "3", "--delta", "1/1000000", "50"] == some ([⟨1, 100⟩, ⟨1, 1000000⟩], [3, 50])
#guard (parse ["--delta", "1"], parse ["0"], parse ["--delta"], parse ["x"]) == (none, none, none, none)

private def accepted (pub prof : List (String × Json)) : Answer :=
  { protocol := "verity/onsite/v0", verdict := .accept, reason := none, digest := none, published := some (Json.mkObj pub),
    profile := some (Json.mkObj prof) }

private def audit (units drawn : Nat) : Json :=
  Json.mkObj [("units", natJ units), ("drawn", natJ drawn), ("free", natJ 3), ("jitter", natJ 2)]

private def block (eps : String) (certified : Bool := true) : Answer :=
  accepted [(CERTIFIER, Json.mkObj [("eps", "1/100"), ("K", natJ 40), ("cells", natJ 4), ("schedules", natJ 1),
        ("audits", .arr #[audit 1024 64, audit 0 0])]),
      (POUS, Json.mkObj [("eps", .str (if certified then "1/100" else "0")), ("certified", .bool certified),
        ("s_bits", natJ (if certified then 4096 else 0))]),
      (POUW, Json.mkObj [("delta", "1/1000000"), ("eps", .arr #["1/2"])]),
      ("verity/remote-bandwidth/v0", Json.mkObj [("eps", .str eps)]), (SAMPLED_PROOFS, .null),
      ("verity/wipe/v0", Json.mkObj [("eps", .str s!"1/{2 ^ 128}")])]
    [(SAMPLED_PROOFS, Json.mkObj [("delta", "1/100"), ("drawn", Json.mkObj [("unit", natJ 64)]),
      ("levels", .arr #[Json.mkObj [("name", "step"), ("size", natJ 2)], Json.mkObj [("name", "unit"), ("size", natJ 1024)]])])]

-- a block's error is its terms summed; the run's is its blocks' summed, and a period's B times the largest block's
private def demo : Json :=
  (evaluate "verity/onsite/v0" [block "0", block "1/100" false] [3, 10080] [⟨1, 100⟩]).toOption.getD .null
private def horizon (i : Nat) (k : String) : Json :=
  (((demo.getObjValD "horizons").getArrVal? i).toOption.map (·.getObjValD k)).getD .null
#guard (demo.getObjValD "run").getObjValD "error" == .str (sum [⟨6, 100⟩, ⟨2, 1000000⟩, ⟨2, 2 ^ 128⟩]).toString
#guard (demo.getObjValD "run").getObjValD "blocks" == natJ 2
#guard horizon 0 "error" == .str (scale 3 (sum [⟨3, 100⟩, ⟨1, 1000000⟩, ⟨1, 2 ^ 128⟩])).toString
#guard (horizon 0 "vacuous", horizon 1 "vacuous") == (.bool false, .bool true)
-- beside it, each pooled figure from its statement at the run's worst block: 16·ln 100 = 73.68 faulty proof blocks,
-- the threshold K₀ = 69 at δ/3, PoUS's 9,948 of 10,080 periods (at p = 1/100), and the certifier's faulty blocks once
private def pooled (i : Nat) (r k : String) : Json :=
  ((horizon i "pooled").getObjValD "1/100" |>.getObjValD r).getObjValD k
#guard pooled 1 "integrity" "faulty_blocks_at_most" == natJ 73
#guard pooled 0 "integrity_per_block" "faulty_blocks_at_most" == natJ (Pooled.threshold 1024 64 ⟨1, 300⟩ - 1)
#guard pooled 1 "space" "certified_at_least" == natJ (Pooled.certified 10080 ⟨1, 100⟩ ⟨1, 100⟩)
#guard pooled 1 "communication" "K_bits" == natJ (Pooled.leakBits 10080 15 1024 ⟨1, 16⟩ ⟨1, 100⟩).1
#guard (demo.getObjValD "inputs").getObjValD "space" == Json.mkObj [("p", "1/100"), ("s_bits", natJ 4096)]
-- a resource no block gives inputs for says why, and the rest stand
#guard ((((evaluate "verity/onsite/v0" [block "0" false] [3] [⟨1, 100⟩]).toOption.getD .null).getObjValD "inputs"
  ).getObjValD "space").getObjValD "why" == .str "no block's PoUS certified, so no eps to pool"
-- a block that wasn't accepted, or of another composition, is refused
private def rejected : Answer :=
  { block "0" with verdict := .reject, reason := some "x", published := none, profile := none }
#guard (evaluate "verity/onsite/v0" [rejected] []).toOption.isNone
#guard (evaluate "verity/remote/v0" [block "0"] []).toOption.isNone

end Core.Onsite.Bound
