import Flock.Canon
import PousCheck.Merkle
import PousCheck.Challenge
import PousCheck.Certificate
import PousCheck.Prim.PyJson

/-!
# The timed audit's verdict on its record

`protocol.TimedAudit` with `audit.UnsignedRounds` (`verity/core/protocols/pous/protocol.py`, `audit.py`), as the GPU
harness's memory challenger runs it strict (`benchmarks/pous/band_gpu/memory_challenger.py --strict`, P2 v3's arm) on the
registered scheme's framing (`schemes/p2.py`, `p2-<w>/v4`), judged from what that process records: `READY.json` (the salt,
the scheme and its manifest), `LEAVES.bin` (one 64-byte leaf digest a block, `verifier.leaf_hash`) and its audit document
(the root, `Δ`, the RTT allowance, the limit, `strict`, `k`, and each audit's challenge `source` with each answer's index,
elapsed time and block). `verdict` is `memory_challenger.verdict`, which drives `TimedAudit` on the same record:

* `fault "ready"`: not P2 v4 (`p2-<w>/v4` at a manifest `P2Params` gives back, keys sorted or not, with a 24-byte salt);
* `fault "leaves"`: not one digest a block;
* `fault "record"`: the document's root, `deadline_ns`, `rtt_allowance_ns`, `limit_ns`, `strict` or `records`, or `k` when
  present, missing or of another type;
* `fault "root"`: the root is not the leaves' (`Merkle.root`);
* `fault "limit"`: the limit is not `Δ` plus the allowance;
* `refused`: strict, and the certificate or the spec refuses `k` at `Δ` and the allowance (`Certificate.gate`), with
  `TimedAudit`'s message;
* then audit by audit: `fault "record"` (no nonempty hex `source`, no `answers` list), and round by round `fault "short"`
  (no answer left), `fault "record"` (the answer's `index`, `elapsed_ns` or hex `block` missing), `fault "index"` (an answer
  to another index than round `j`'s, `key.uniform(B, j)` under `derive(source, CHALLENGE_DOMAIN, {root, salt})`); and
  `fault "long"` (answers past `k`).

Otherwise each answer is correct when its block has the width and its leaf (`Merkle.packageLeaf`) is the leaf digest of
the revealed index, and on time when its elapsed time is at most `Δ + RTT`; an audit is accepted when every answer is both,
the record when it has an audit and every audit is; and the audit of record (the first), when accepted, gives its
StorageProfile (`Certificate.profileJson`). Without `k` (a refusal recorded before refusals recorded it), `k` is the
certificate's, else `audit.CHALLENGES`.
-/

namespace PousCheck.TimedAudit

open Lean (Json JsonNumber)
open Flock

def CHALLENGE_DOMAIN : String := "verity/pous/audit-challenge/v1"
/-- `audit.CHALLENGES`: `k` where a scheme has no certificate. -/
def CHALLENGES : Nat := 132
/-- The setup salt's bytes (`labels.SALT_BITS / 8`). -/
def SALT : Nat := 24
def MANIFEST_KEYS : List String := ["blocks_per_segment", "c0", "expander", "profile", "scheme", "segments", "w"]

/-- A verdict's computation: `throw part` names the part of the record it does not judge. -/
abbrev R := Except String

def need (ok : Bool) (part : String) : R Unit := if ok then pure () else throw part

def natOf (j : Json) (part : String) : R Nat := match j.getNat? with
  | .ok n => pure n
  | .error _ => throw part

def strOf (j : Json) (part : String) : R String := match j.getStr? with
  | .ok s => pure s
  | .error _ => throw part

def hexOf' (j : Json) (part : String) : R ByteArray := do
  match unhex (← strOf j part) with
  | .ok b => pure b
  | .error _ => throw part

def field (j : Json) (k part : String) : R Json := match j.getObjVal? k with
  | .ok v => pure v
  | .error _ => throw part

/-- The store of a `READY.json`, and its salt: P2 v4, at a manifest `P2Params` takes and gives back exactly. -/
def store (ready : Json) : R (P2.Params × ByteArray) := do
  let p := "ready"
  let m ← field ready "params" p
  let keys ← match m.getObj? with
    | .ok o => pure o.keys
    | .error _ => throw p
  need (keys == MANIFEST_KEYS) p
  let n k := do natOf (← field m k p) p
  let (w, c0, bps, segments) := (← n "w", ← n "c0", ← n "blocks_per_segment", ← n "segments")
  need (w % 8 == 0 && w > 16 && c0 % 4 == 1 && c0 < 2 ^ 16 && 1 ≤ bps && bps ≤ 2 ^ 16 && 1 ≤ segments) p
  let s : P2.Params := ⟨w, c0, bps, segments⟩
  need ((← strOf (← field m "scheme" p) p) == "p2" && (← strOf (← field m "profile" p) p) == s.profile
    && (← strOf (← field m "expander" p) p) == Certificate.EXPANDER) p
  need ((← strOf (← field ready "scheme" p) p) == s.name) p
  let salt ← hexOf' (← field ready "salt" p) p
  need (salt.size == SALT) p
  pure (s, salt)

/-- One audit's answers, `[correct, on time]` each, judged against the leaves under `root`. -/
def audit (s : P2.Params) (salt root : ByteArray) (leaves : Array ByteArray) (k limit : Nat) (rec : Json) :
    R (Array (Bool × Bool)) := do
  let p := "record"
  let source ← hexOf' (← field rec "source" p) p
  need (source.size > 0) p
  let answers ← match (← field rec "answers" p).getArr? with
    | .ok a => pure a
    | .error _ => throw p
  let key := FlockSoundness.Randomness.derive source CHALLENGE_DOMAIN [("root", .bytes root), ("salt", .bytes salt)]
  let mut out : Array (Bool × Bool) := #[]
  for j in List.range k do
    need (j < answers.size) "short"
    let a := answers[j]!
    let index ← natOf (← field a "index" p) p
    let elapsed ← natOf (← field a "elapsed_ns" p) p
    let block ← hexOf' (← field a "block" p) p
    let i := (Challenge.index key s.blocks j).getD s.blocks
    need (index == i) "index"
    let correct := block.size == s.w / 8 &&
      Merkle.packageLeaf salt (i / s.blocksPerSegment) (i % s.blocksPerSegment) block == leaves[i]!
    out := out.push (correct, decide (elapsed ≤ limit))
  need (answers.size == k) "long"
  pure out

def verdictOf (ready : Json) (leafBytes : ByteArray) (doc : Json) : R Json := do
  let (s, salt) ← store ready
  need (leafBytes.size == s.blocks * Merkle.DIGEST) "leaves"
  let leaves := (Array.range s.blocks).map fun i => leafBytes.extract (Merkle.DIGEST * i) (Merkle.DIGEST * (i + 1))
  let p := "record"
  let root ← hexOf' (← field doc "root" p) p
  let n k := do natOf (← field doc k p) p
  let (delta, rtt, limit) := (← n "deadline_ns", ← n "rtt_allowance_ns", ← n "limit_ns")
  let strict ← match (← field doc "strict" p).getBool? with
    | .ok b => pure b
    | .error _ => throw p
  let records ← match (← field doc "records" p).getArr? with
    | .ok a => pure a
    | .error _ => throw p
  let k? ← match doc.getObjVal? "k" with
    | .ok j => some <$> natOf j p
    | .error _ => pure none
  need (some root == Merkle.root leaves) "root"
  need (limit == delta + rtt) "limit"
  let cert := Certificate.p2 s.w s.c0 s.blocksPerSegment
  let k := k?.getD ((cert.map (·.k)).getD CHALLENGES)
  let (_, refused, uncovered) := Certificate.gate s k delta rtt
  if strict && !refused.isEmpty then
    return Json.mkObj [("refused", .str s!"the audit is outside the certificate: {"; ".intercalate refused} (research=True runs it anyway)")]
  let audits ← records.mapM (audit s salt root leaves k limit)
  let acc := audits.map fun a => a.all fun (c, t) => c && t
  let nat (x : Nat) : Json := .num (JsonNumber.fromNat x)
  pure <| Json.mkObj [
    ("k", nat k), ("limit_ns", nat limit), ("gaps", .arr ((refused ++ uncovered).map Json.str).toArray),
    ("audits", .arr ((audits.zip acc).map fun (a, ok) => Json.mkObj [("accepted", .bool ok),
      ("answers", .arr (a.map fun (c, t) => .arr #[.bool c, .bool t]))])),
    ("audits_accepted", nat (acc.filter id).size), ("accepted", .bool (!acc.isEmpty && acc.all id)),
    ("profile", if acc[0]?.getD false then Certificate.profileJson s k delta rtt else .null)]

/-- The verdict: `{"fault": part}` for a record it does not judge, else `verdictOf`'s. -/
def verdict (ready : Json) (leaves : ByteArray) (doc : Json) : Json :=
  match verdictOf ready leaves doc with
  | .ok v => v
  | .error part => Json.mkObj [("fault", .str part)]

/-- `verdict` on the difftest's line: `{"ready": READY, "leaves": hex, "audit": DOC}`. -/
def verdictFn (inst : Json) : Except String Json := do
  let leaves ← unhex (← J.str inst "leaves" "instance")
  pure (verdict (← J.get inst "ready" "instance") leaves (← J.get inst "audit" "instance"))

/-- `pous-timed-audit READY.json LEAVES.bin AUDIT.json`: the verdict as one canonical JSON line; exit 0 iff accepted. A file
is read as Python's `json` reads it (`Prim.PyJson.scan`: a float, which `natOf` never takes, as `null`), and a file that
is not JSON as `null`. -/
def run (readyPath leavesPath auditPath : String) : IO UInt32 := do
  let read (path : String) : IO Json :=
    return (Json.parse (Prim.PyJson.scan (← IO.FS.readFile path)).1).toOption.getD .null
  let v := verdict (← read readyPath) (← IO.FS.readBinFile leavesPath) (← read auditPath)
  match canon v with
  | .ok text => IO.println text
  | .error e => IO.eprintln e; return 2
  return if v.getObjValD "accepted" == .bool true then 0 else 1

end PousCheck.TimedAudit
