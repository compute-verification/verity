module

public import RemoteAudit.Session
public import Spot.OneStageZk

/-!
# A window decided from accepted proofs: the span in place of the opened blocks

`checkSpan` decides window t from the proof system's accepted runs of PoUB's opening unit (`BlockOpeningCheck_v2`,
`block_check.py`), one unit a draw, in place of the openings `checkWindow` reads; it never reads a block, a payload, a
receipt leaf or a path. A span is one or more statements of the sampled proof system's one-stage audit
(`Spot.OneStageZk`, every unit proved), each its registration record, its evidence and its public file, with each
instance's verdict word and the salts of its three opened rows; the blobs hold the files the evidence names.

The auditor's side (`Auditor`) is its own: the one-stage audit's parameters (its challenger, delta, program, partition
and the registered opening circuit's SHA-512) and that circuit's ports in order, each with its row's bits (`Layout`). Of
a statement only its public file's SHA-512 is the prover's, which the audit holds to the registration record.

The window rejects `malformed`, at no draw, unless every statement's audit accepts (`accepted`) with every instance of
its public file proved and the statements' instances, in order, are the window's draws, one each. Then each draw
(i, p), in order, is held (`unitFails`): its pass's plan the identity and its block j = p registered (v2 has no plan
row and this check no appended block); its `pub` and `signs` rows the verifier's own (`publicWords`, `signWords`:
`block_check.opening_public_v2`'s) opening under their salts to its instance's `b ‖ c` in the public file; and its `o0`
row the verdict word under its salt. Else the window rejects `malformed` at (i, p). The word then fails the draw as
`block_check.verdict` names it (`verdictFailure`), and the first draw that fails rejects the window there, as
`checkWindow` does.

`Session.verifySpan` is `Session.verifyBy` with `checkSpan`. It is no `Session.Event`: `Session.step`, `Session.run` and
the verifier's process don't reach it, and `remote-audit-difftest` (`span`, `span_held`, `span_public`) is its one
entry point.
-/

@[expose] public section

namespace RemoteAudit.Span

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

/-- One proof block's unit as the prover sends it: its verdict word and the salts of its `pub`, `signs` and `o0` rows. -/
structure Instance where
  verdict : Nat
  pubSalt : ByteArray
  signsSalt : ByteArray
  verdictSalt : ByteArray
deriving Inhabited

/-- One statement of the span: its public file's SHA-512 (lowercase hex), its registration record's and its evidence's
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

/-- `block_check.opening_public_v2`'s `pub` row for draw (i, p) of window `w`, a registered block j = p: block j's codec
key under the registration's seed (unbound: `bound` 0, `coin` zero), `t0 = τ W j mod p`, the pass's sketch prefix's
chaining value (`mid`), `pos` = j, the pass's receipt root, its tree's domain id and depth, the registration's payload
root, its domain id, rank j and depth, and the value tree's fields zero. -/
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

/-- One unit's rows, as the public file holds them: `pub`, `signs` and `o0`'s `b ‖ c`. -/
structure Held where
  pub : ByteArray
  signs : ByteArray
  verdict : ByteArray
  inst : Instance
deriving Inhabited

/-- Draw (i, p)'s unit: `none` if it opens to the verifier's own rows and its word says accepted, else its failure. -/
def unitFails (l : Layout) (reg : Registration) (w : WindowRecord) (i p : Nat) (u : Held) : Option Failure := Id.run do
  let some (_, pubBits) := l.port? "pub" | return some .malformed
  let some (_, signsBits) := l.port? "signs" | return some .malformed
  let some (_, verdictBits) := l.port? "o0" | return some .malformed
  let some pr := w.passes[i]? | return some .malformed
  if pr.plan.runs != #[(0, pr.plan.length)] || p ≥ pr.plan.length || p ≥ reg.blocks || reg.receipt != .sketch then
    return some .malformed
  let pubW := publicWords reg w i p
  let signsW := signWords reg w i
  if 16 * pubW.size ≠ pubBits || 16 * signsW.size ≠ signsBits then return some .malformed
  if rowBc pubBits pubW u.inst.pubSalt != some u.pub || rowBc signsBits signsW u.inst.signsSalt != some u.signs
      || rowBc verdictBits #[u.inst.verdict] u.inst.verdictSalt != some u.verdict then
    return some .malformed
  return verdictFailure u.inst.verdict

/-- The accepted public files' units, in order, held to the window's draws: `malformed` at no draw for a file that
isn't one `b ‖ c` a port an instance, another count of instances than its statement sends, or another count of units
than draws; else the first draw whose unit fails (`unitFails`), there, or accept. -/
def spanHeld (l : Layout) (reg : Registration) (w : WindowRecord) (drawn : Array (Nat × Nat))
    (held : Array (ByteArray × Array Instance)) : Verdict := Id.run do
  let (some (qp, _), some (qs, _), some (qo, _)) := (l.port? "pub", l.port? "signs", l.port? "o0")
    | return .reject .malformed none
  let np := l.ports.size
  let mut units : Array Held := #[]
  for (pubB, insts) in held do
    match bcsOf pubB np with
    | .error _ => return .reject .malformed none
    | .ok (n, bcs) =>
      if n ≠ insts.size then return .reject .malformed none
      for u in [0:n] do
        units := units.push ⟨bcs[u * np + qp]!, bcs[u * np + qs]!, bcs[u * np + qo]!, insts[u]!⟩
  if units.size ≠ drawn.size then return .reject .malformed none
  for k in [0:drawn.size] do
    let (i, p) := drawn[k]!
    if let some f := unitFails l reg w i p units[k]! then return .reject f (some (i, p))
  return .accept

/-- A statement's public file, once its one-stage audit accepts at the auditor's parameters with the statement's public
file (`Spot.OneStageZk.checkBlobs`, its record and evidence read by `Spot.OneStage.commitmentOf` and
`Spot.OneStageZk.evidenceOf`) and proves every instance the file holds; else why not. -/
def accepted (a : Auditor) (st : Statement) (bs : Core.Blobs) : V ByteArray := do
  let p ← Spot.OneStageZk.paramsOfJson a.params
  let p := { p with own := { p.own with pub := st.pub } }
  match Spot.OneStageZk.checkBlobs p (← Spot.OneStage.commitmentOf st.record) (← Spot.OneStageZk.evidenceOf st.evidence) bs with
  | .accept pub _ =>
    let pubB ← bs.get st.pub "public file"
    if pub.proved != (← Spot.FilePart.unitIndices pubB) then throw "the audit proved other units than the file's"
    pure pubB
  | .reject why => throw why
  | .undecided why => throw why

/-- **Window `w`'s verdict from the span**: `malformed` at no draw unless every statement is `accepted`, else
`spanHeld` on their public files. -/
def checkSpan (a : Auditor) (reg : Registration) (w : WindowRecord) (drawn : Array (Nat × Nat)) (sp : Span) :
    Verdict := Id.run do
  let bs := Core.Blobs.ofArray sp.blobs
  let mut held := #[]
  for st in sp.statements do
    match accepted a st bs with
    | .ok pubB => held := held.push (pubB, st.instances)
    | .error _ => return .reject .malformed none
  return spanHeld a.layout reg w drawn held

end RemoteAudit.Span

namespace RemoteAudit.Session

/-- `verify` with window t decided from the span (`Span.checkSpan`) in place of openings. -/
def verifySpan (s : State) (t : Int) (a : Span.Auditor) (sp : Span.Span) : Result × State :=
  verifyBy s t (Span.checkSpan a s.reg (s.record t.toNat) · sp)

/-- `verify` with window t decided by `Span.spanHeld` on public files taken as accepted: the span's check past the
proof system's audit, which `remote-audit-difftest`'s `span_held` holds on public files a test commits itself. -/
def verifyHeld (s : State) (t : Int) (l : Span.Layout) (held : Array (ByteArray × Array Span.Instance)) :
    Result × State :=
  verifyBy s t (Span.spanHeld l s.reg (s.record t.toNat) · held)

end RemoteAudit.Session
