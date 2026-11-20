module

public import Poub.Session
public import Spot.OneStageZk

/-!
# A window decided from accepted proofs: the span in place of the opened blocks

`checkSpan` decides window t from the proof system's accepted runs of PoUB's opening unit (`BlockOpeningUnit_v2`,
`block_check.py`), one unit a sample, in place of the openings `checkWindow` reads; it never reads a block, a payload, a
receipt leaf or a path. A span is one or more statements of the sampled proof system's one-stage audit
(`Spot.OneStageZk`, every unit proved), each its docket, its evidence and its public file, with each instance's verdict
word and the salt of its `o0` row; the blobs hold the files the evidence names.

The auditor's side (`Auditor`) is its own: the one-stage audit's parameters (its challenger, delta, program, partition,
the registered opening circuit's SHA-512 and the values it expects registered, `pub` and `signs` among them) and that
circuit's ports in order, each with its row's bits (`Layout`). Of a statement only its public file's SHA-512 is the
prover's, which the audit holds to the docket.

Each sample (i, p), in order, is first a unit's (`sampleOk`): its pass's plan the identity and its block j = p registered
(v2 has no plan row and this check no appended block); else the window rejects `malformed` at (i, p). Unit k's `pub` and
`signs` rows are then the verifier's own: sample k's (`publicWords`, `signWords`: `block_check.opening_public_v2`'s),
which it gives every statement's audit as the supplied values `pub` and `signs`, row k each (`supplied`,
`Spot.Supplied`), so R7 holds the record's entries for them to the root it commits them to under public salts, and no
salt of theirs is the prover's. The window rejects `malformed`, at no sample, unless every statement's audit accepts
(`accepted`) with every instance of its public file proved and the statements' units, in order, are the population's
units 0, 1, … one a sample. Then unit k's `o0` row must open to its verdict word under the salt the prover sends, else
the window rejects `malformed` at sample k. The word then fails the sample as `block_check.verdict` names it
(`verdictFailure`), and the first sample that fails rejects the window there, as `checkWindow` does.

`Session.verifySpan` is `Session.verifyBy` with `checkSpan`. It is no `Session.Event`: `Session.step`, `Session.run` and
the verifier's process don't reach it, and `poub-difftest` (`span`, `span_held`, `span_public`) is its one
entry point.
-/

@[expose] public section

namespace Poub.Span

open Lean (Json)
open Flock (V)

/-- The registered opening circuit's ports, in the circuit's order: each port's name and its row's bits. -/
structure Layout where
  ports : Array (String × Nat)

/-- The auditor's own inputs: the one-stage audit's parameters as JSON (`Spot.OneStageZk.paramsOfJson`; their
`proofs.public` replaced by each statement's) and the registered circuit's layout. -/
structure Auditor where
  params : Json
  layout : Layout

/-- One proof block's unit as the prover sends it: its verdict word and its `o0` row's salt. -/
structure Instance where
  verdict : Nat
  verdictSalt : ByteArray
deriving Inhabited

/-- One statement of the span: its public file's SHA-512 (lowercase hex), its docket's and its evidence's
bytes, and its instances in the public file's order. -/
structure Statement where
  pub : String
  record : ByteArray
  evidence : ByteArray
  instances : Array Instance

structure Span where
  statements : Array Statement
  blobs : Array ByteArray

/-- A byte string's u16 words, little-endian pairs. -/
def u16s (b : ByteArray) : Array Nat :=
  (Array.range (b.size / 2)).map fun i => (b.get! (2 * i)).toNat + 256 * (b.get! (2 * i + 1)).toNat

/-- `x`'s low `16 n` bits as `n` u16 words, low first. -/
def u16sOf (x n : Nat) : Array Nat := (Array.range n).map fun i => (x >>> (16 * i)) % 65536

/-- `block_check.opening_public_v2`'s `pub` row for sample (i, p) of window `w`, a registered block j = p: block j's
codec key under the registration's seed (unbound: `bound` 0, `coin` zero), `t0 = τ W j mod p`, the pass's sketch
prefix's chaining value (`mid`), `pos` = j, the pass's receipt root, its tree's domain id and depth, the registration's
payload root, its domain id, rank j and depth, and the value tree's fields zero. -/
def publicWords (reg : Registration) (w : WindowRecord) (i j : Nat) : Array Nat :=
  let f := reg.codec.field
  let fw := f.n / 16
  let k := reg.key
  let pr := w.passes[i]!
  let mid := (Flock.Sha512.compress Flock.Sha512.iv (receiptPrefix SKETCH_RECEIPT_DOMAIN (w.seedOf i)) 0).foldl
    (fun acc v => acc ++ u16sOf v.toNat 4) #[]
  let rdom := treeDomain (w.seedOf i) pr.plan
  u16sOf j 4 ++ #[0] ++ u16s reg.codecSeed ++ u16sOf 0 16 ++ u16sOf k.tau fw ++ u16sOf (k.tau * f.blockWords * j % f.p) fw
    ++ k.c.foldl (fun acc c => acc ++ u16sOf c fw) #[] ++ (match k.mask with | some m => u16sOf m fw | none => #[])
    ++ mid ++ u16sOf j 2 ++ u16s pr.root ++ u16s rdom.id ++ #[Flock.log2ceil rdom.count]
    ++ u16s reg.payloadRoot ++ u16s reg.payloadDomain.id ++ u16sOf j 2 ++ #[Flock.log2ceil reg.blocks]
    ++ u16sOf 0 32 ++ u16sOf 0 32 ++ u16sOf 0 2 ++ #[0]

/-- The `signs` row for pass i of window `w`: its sign rows (`signBits`) at the registration's rows, as u16 words. -/
def signWords (reg : Registration) (w : WindowRecord) (i : Nat) : Array Nat :=
  (signBits (w.seedOf i) reg.rows).foldl (fun acc r => acc ++ u16s r) #[]

/-- Sample (i, p) is a unit's: its pass's plan the identity, p a position of it and a registered block, under
sketch-sha512. -/
def sampleOk (reg : Registration) (w : WindowRecord) (i p : Nat) : Bool :=
  match w.passes[i]? with
  | some pr => pr.plan.runs == #[(0, pr.plan.length)] && p < pr.plan.length && p < reg.blocks && reg.receipt == .sketch
  | none => false

/-- u16 words w0, w1, … as big-endian 32-bit halves w1·2^16 + w0, …: the bytes `Spot.Supplied.bytesLeaves` reads back
as 16-bit leaves w0, w1, …; `none` for an odd count. -/
def halves (ws : Array Nat) : Option ByteArray :=
  if ws.size % 2 ≠ 0 then none
  else some ((Array.range (ws.size / 2)).foldl (fun acc k => acc ++ be 4 (ws[2 * k + 1]! * 65536 + ws[2 * k]!)) .empty)

/-- The supplied values the verifier gives each statement's audit (`Spot.Supplied.sources`' form): `pub` and `signs`,
row k each sample k's, its words' `halves` as one `bytes` source; `none` where a row's words are odd. -/
def supplied (reg : Registration) (w : WindowRecord) (sampled : Array (Nat × Nat)) : Option Json := do
  let row (ws : Array Nat) : Option Json := do
    pure (.arr #[Json.mkObj [("bytes", .str (Flock.hexOf (← halves ws)))]])
  let pub ← sampled.mapM fun (i, p) => row (publicWords reg w i p)
  let signs ← sampled.mapM fun (i, _) => row (signWords reg w i)
  pure (Json.mkObj [("pub", .arr pub), ("signs", .arr signs)])

/-- A row's `b ‖ c` as the proof system commits a unit's row: hm96-sha512 over the `sha512/row/v2` digest of the row of
`bits` bits its words hold (u16, low bit first, zero words to fill it), under `salt`. `none` for words past the row,
a word past 16 bits, a bit set past the row, or a salt not 192 bytes. -/
def rowBc (bits : Nat) (words : Array Nat) (salt : ByteArray) : Option ByteArray :=
  let n := (bits + 15) / 16
  if words.size > n || words.any (· ≥ 65536) || salt.size ≠ SALT_BYTES then none
  else
    let ws := words ++ Array.replicate (n - words.size) 0
    if bits % 16 ≠ 0 && ws.back?.any (· >>> (bits % 16) ≠ 0) then none
    else
      let bytes := (ws.foldl (fun acc x => acc ++ le 2 x) ByteArray.empty).extract 0 ((bits + 7) / 8)
      some (Flock.Hm96.Default512.commitString (sha512 (pad128 (tag0 "verity/sha512-row/v2" ++ be 8 bits) ++ bytes)) salt)

/-- `block_check.verdict`: none for the word 1, else the first failed check its bits 1 … 4 name, `malformed` for a
word no unit returns. -/
def verdictFailure (word : Nat) : Option Failure :=
  if word = 1 then none
  else if (word >>> 1) % 2 = 1 then some .receiptPath
  else if (word >>> 2) % 2 = 1 then some .codec
  else if (word >>> 3) % 2 = 1 then some .payloadPath
  else if (word >>> 4) % 2 = 1 then some .receipt
  else some .malformed

def BC_BYTES : Nat := 128

/-- A public file's `b ‖ c`, instance-major and ports in order, for a circuit of `np` ports: its header's
`instances` rows of `np`, the file's whole body (no shared rows, no output words). -/
def bcsOf (pubB : ByteArray) (np : Nat) : V (Nat × Array ByteArray) := do
  let h ← Spot.FilePart.header pubB
  if Flock.J.has h "shared_rows" then throw "a public file of shared rows"
  let n ← Flock.J.nat h "instances" "public file"
  let off := (Spot.FilePart.headerLine pubB).size + 1
  let body := pubB.extract off pubB.size
  if off > pubB.size || body.size ≠ BC_BYTES * n * np then throw "public file: not one b ‖ c a port an instance"
  pure (n, (Array.range (n * np)).map fun x => body.extract (BC_BYTES * x) (BC_BYTES * (x + 1)))

/-- Port `name`'s index in the layout and its row's bits. -/
def Layout.port? (l : Layout) (name : String) : Option (Nat × Nat) :=
  (l.ports.findIdx? (·.1 == name)).map fun q => (q, l.ports[q]!.2)

/-- The first sample that is no unit's (`sampleOk`), if any. -/
def firstNoUnit (reg : Registration) (w : WindowRecord) (sampled : Array (Nat × Nat)) : Option (Nat × Nat) :=
  sampled.find? fun (i, p) => !sampleOk reg w i p

/-- Unit k's `o0` row's `b ‖ c` as the public file holds it, and what the prover sends of the unit. -/
structure Held where
  verdict : ByteArray
  inst : Instance
deriving Inhabited

/-- A unit's failure: `none` if its `o0` row opens to its verdict word under its salt and the word says accepted. -/
def unitFails (verdictBits : Nat) (u : Held) : Option Failure :=
  if rowBc verdictBits #[u.inst.verdict] u.inst.verdictSalt != some u.verdict then some .malformed
  else verdictFailure u.inst.verdict

/-- The accepted public files' units, in order, held to the window's samples: `malformed` at the first sample that is no
unit's (`firstNoUnit`); `malformed` at no sample for a layout with no `o0`, a file that isn't one `b ‖ c` a port an
instance, another count of instances than its statement sends, or units that are not the population's 0, 1, … one a
sample; else the first sample whose unit fails (`unitFails`), there, or accept. -/
def spanHeld (l : Layout) (reg : Registration) (w : WindowRecord) (sampled : Array (Nat × Nat))
    (held : Array (ByteArray × Array Instance)) : Verdict := Id.run do
  if let some ip := firstNoUnit reg w sampled then return .reject .malformed (some ip)
  let some (qo, verdictBits) := l.port? "o0" | return .reject .malformed none
  let np := l.ports.size
  let mut units : Array Held := #[]
  let mut indices : Array Nat := #[]
  for (pubB, insts) in held do
    match bcsOf pubB np, Spot.FilePart.unitIndices pubB with
    | .ok (n, bcs), .ok is =>
      if n ≠ insts.size then return .reject .malformed none
      indices := indices ++ is
      for u in [0:n] do
        units := units.push ⟨bcs[u * np + qo]!, insts[u]!⟩
    | _, _ => return .reject .malformed none
  if units.size ≠ sampled.size || indices != Array.range sampled.size then return .reject .malformed none
  for k in [0:sampled.size] do
    if let some f := unitFails verdictBits units[k]! then return .reject f (some sampled[k]!)
  return .accept

/-- A statement's public file, once its one-stage audit accepts at the auditor's parameters, with the statement's public
file and the verifier's `supplied` values `sup` in place of any of the same names (`Spot.OneStageZk.checkBlobs`, its
record and evidence read by `Spot.OneStage.commitmentOf` and `Spot.OneStageZk.evidenceOf`), and proves every instance
the file holds; else why not. -/
def accepted (a : Auditor) (sup : Json) (st : Statement) (bs : Core.Blobs) : V ByteArray := do
  let theirs := match a.params.getObjValD "supplied" with | .obj kvs => kvs.toArray | _ => #[]
  let ours := match sup with | .obj kvs => kvs.toArray | _ => #[]
  let merged := theirs.filter (fun (k, _) => !ours.any (·.1 == k)) ++ ours
  let p ← Spot.OneStageZk.paramsOfJson (a.params.setObjVal! "supplied" (Json.mkObj merged.toList))
  let p := { p with own := { p.own with pub := st.pub } }
  match Spot.OneStageZk.checkBlobs p (← Spot.OneStage.commitmentOf st.record) (← Spot.OneStageZk.evidenceOf st.evidence) bs with
  | .accept pub _ =>
    let pubB ← bs.get st.pub "public file"
    if pub.proved != (← Spot.FilePart.unitIndices pubB) then throw "the audit proved other units than the file's"
    pure pubB
  | .reject why => throw why
  | .undecided why => throw why

/-- **Window `w`'s verdict from the span**: `malformed` at the first sample that is no unit's; else `malformed` at no
sample unless every statement is `accepted` with the window's `supplied` rows; else `spanHeld` on their public files. -/
def checkSpan (a : Auditor) (reg : Registration) (w : WindowRecord) (sampled : Array (Nat × Nat)) (sp : Span) :
    Verdict := Id.run do
  if let some ip := firstNoUnit reg w sampled then return .reject .malformed (some ip)
  let some sup := supplied reg w sampled | return .reject .malformed none
  let bs := Core.Blobs.ofArray sp.blobs
  let mut held := #[]
  for st in sp.statements do
    match accepted a sup st bs with
    | .ok pubB => held := held.push (pubB, st.instances)
    | .error _ => return .reject .malformed none
  return spanHeld a.layout reg w sampled held

end Poub.Span

namespace Poub.Session

/-- `verify` with window t decided from the span (`Span.checkSpan`) in place of openings. -/
def verifySpan (s : State) (t : Int) (a : Span.Auditor) (sp : Span.Span) : Result × State :=
  verifyBy s t (Span.checkSpan a s.reg (s.record t.toNat) · sp)

/-- `verify` with window t decided by `Span.spanHeld` on public files taken as accepted: the span's check past the
proof system's audit, which `poub-difftest`'s `span_held` holds on public files a test commits itself. -/
def verifyHeld (s : State) (t : Int) (l : Span.Layout) (held : Array (ByteArray × Array Span.Instance)) :
    Result × State :=
  verifyBy s t (Span.spanHeld l s.reg (s.record t.toNat) · held)

end Poub.Session
