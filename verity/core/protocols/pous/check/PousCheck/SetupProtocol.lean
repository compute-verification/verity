import CoreProtocol.Cli
import PousCheck.Setup

/-!
# The public-encoder setup as a core protocol

`verity/pous/p2/v4/setup` is `Setup.verifierKeyFromEncoding` behind `Core.Protocol`: the verifier decodes the server's
`C*`, accepts iff `Dec(C*)` is the `W` the server committed to, and publishes `vk`'s root, which the timed audit then
takes as its commitment. Its three inputs:

~~~text
params      the codec's manifest (`P2Params.manifest()`), canonical JSON with exactly its seven keys:
            {"blocks_per_segment":n,"c0":c0,"expander":"sha512-ctr-v4","profile":"p2-<payload bytes>B",
             "scheme":"p2","segments":m,"w":w}
commitment  w_commitment[64] ‖ salt          the server's commitment to W (`setup.commit`), then the verifier's salt;
                                             the digest is these bytes, and the check refuses a salt not of 24 bytes
evidence    u32be(segments) ‖ segment…       C*, segment by segment, in order
segment     u32be(blocks) ‖ block…
block       u32be(len) ‖ len bytes
~~~

Each count is at most a quarter of the bytes left (an element takes at least 4), and no byte may follow the last segment.
The evidence's form carries any shape, so a segment or block of the wrong size reaches the check and is refused there,
as Python's `verifier_key_from_encoding` refuses it.

On acceptance the public outputs are `{"blocks":B,"commitment":hex,"root":hex,"salt":hex}` (`Public.ofJson` reads them
back) and the profile is `{"mode":"public"}`: Python's setup records nothing beyond its mode (`protocol.PUBLIC`).
-/

namespace PousCheck.SetupProtocol

open Lean (Json)
open Flock (V)

def ID : String := "verity/pous/p2/v4/setup"

/-- The server's commitment to `W`, a SHA-512 digest. -/
def W_COMMITMENT_BYTES : Nat := 64

structure Commitment where
  w : ByteArray
  salt : ByteArray

/-- What acceptance publishes: the audit's commitment. -/
structure Public where
  blocks : Nat
  commitment : ByteArray
  root : ByteArray
  salt : ByteArray
  deriving BEq

def Public.toJson (pub : Public) : Json :=
  Json.mkObj [("blocks", .num pub.blocks), ("commitment", .str (Flock.hexOf pub.commitment)),
    ("root", .str (Flock.hexOf pub.root)), ("salt", .str (Flock.hexOf pub.salt))]

def Public.ofJson (j : Json) : V Public := do
  let ctx := "the setup's public outputs"
  return ⟨← Flock.J.nat j "blocks" ctx, ← Flock.unhex (← Flock.J.str j "commitment" ctx),
    ← Flock.unhex (← Flock.J.str j "root" ctx), ← Flock.unhex (← Flock.J.str j "salt" ctx)⟩

/-- The parameters: the four numbers, valid (`P2Params.__post_init__`), and the JSON exactly their manifest. -/
def params (j : Json) : V P2.Params := do
  let ctx := "the P2 parameters"
  let q ← P2.Params.validate
    ⟨← Flock.J.nat j "w" ctx, ← Flock.J.nat j "c0" ctx, ← Flock.J.nat j "blocks_per_segment" ctx,
      ← Flock.J.nat j "segments" ctx⟩
  let want ← Flock.canon (Commit.manifestJson q)
  if (← Flock.canon j) != want then throw s!"not the manifest of p2-16448/v4 at these numbers, {want}"
  return q

def commitment (b : ByteArray) : V Commitment := do
  if b.size < W_COMMITMENT_BYTES then
    throw s!"the commitment to W is {W_COMMITMENT_BYTES} bytes, then the salt; {b.size} bytes is too short"
  return ⟨b.extract 0 W_COMMITMENT_BYTES, b.extract W_COMMITMENT_BYTES b.size⟩

open Flock.Reader in
private def u32be : M Nat := do return Prim.natOfBE (← bytes 4)

open Flock.Reader in
private def count (what : String) : M Nat := do
  let n ← u32be
  let r ← get
  if 4 * n > r.data.size - r.pos then throw s!"{what} {n} exceeds the bytes left"
  return n

open Flock.Reader in
private def segment : M (Array ByteArray) := do
  let n ← count "a segment's block count"
  let mut out := Array.mkEmpty n
  for _ in [0:n] do out := out.push (← bytes (← u32be))
  return out

/-- `C*` from its bytes: segments of blocks, of any shape. -/
def evidence (b : ByteArray) : V (Array (Array ByteArray)) := do
  let read : Flock.Reader.M (Array (Array ByteArray)) := do
    let n ← count "the segment count"
    let mut out := Array.mkEmpty n
    for _ in [0:n] do out := out.push (← segment)
    return out
  let (segments, r) ← read.run ⟨b, 0⟩
  if r.pos != b.size then throw s!"{b.size - r.pos} bytes follow the last segment"
  return segments

/-- `C*` as bytes, the form `evidence` reads. -/
def encodeEvidence (segments : Array (Array ByteArray)) : ByteArray :=
  segments.foldl (init := Prim.beOfNat segments.size 4) fun out seg =>
    seg.foldl (init := out ++ Prim.beOfNat seg.size 4) fun out block => out ++ Prim.beOfNat block.size 4 ++ block

/-- The setup check. It refuses parameters `P2.Params.validate` rejects (which `params` never returns), so validity is
part of what an acceptance says. -/
def check (q : P2.Params) (c : Commitment) (e : Array (Array ByteArray)) : Core.Outcome Public Unit :=
  match q.validate with
  | .error m => .reject s!"params: {m}"
  | .ok _ =>
    match Setup.verifierKeyFromEncoding q c.w c.salt e with
    | .ok vk => .accept ⟨q.blocks, c.w, vk.root, c.salt⟩ ()
    | .error r => .reject (r.message q)

def protocol : Core.Protocol where
  id := ID
  Params := P2.Params
  Commitment := Commitment
  Evidence := Array (Array ByteArray)
  Public := Public
  Profile := Unit
  params := params
  commitment := commitment
  evidence := evidence
  digest c := c.w ++ c.salt
  check := check
  publicJson := Public.toJson
  profileJson _ := Json.mkObj [("mode", .str "public")]

/-! A setup Python accepts: `w = 64`, `c0 = 189`, one segment of two blocks, `W = 01 02 … 0c`, salt `64 65 … 7b`, the
blocks `encode_segment`'s, the root `vk.root`. -/

private def MANIFEST : String :=
  "{\"blocks_per_segment\":2,\"c0\":189,\"expander\":\"sha512-ctr-v4\",\"profile\":\"p2-6B\",\"scheme\":\"p2\"," ++
  "\"segments\":1,\"w\":64}"
def COMMITMENT : String :=
  "6d5b09e192a58823e26422f4e6d3c34410236172c5ddb9886114759c094fbeee4a9c1d42141a3f5007f556b8913182bce67cd5c07db82f7f21b1" ++
  "34aea9f7acf3" ++ "6465666768696a6b6c6d6e6f707172737475767778797a7b"
def ROOT : String :=
  "e276477790a18d9f3f1044d19863367ebde2fdf6bce07c5d03f651c4462bd2de912ac9bc6a156686831bbbe6b63967a251453153481383a8cb" ++
  "bb49db8d35e1e6"

private def run (blocks : Array String) (params : String := MANIFEST) (commitment : String := COMMITMENT)
    (trailing : ByteArray := .empty) : Core.Answer :=
  let segment := (blocks.mapM Flock.unhex).toOption.getD #[]
  protocol.answer params.toUTF8 ((Flock.unhex commitment).toOption.getD .empty) (encodeEvidence #[segment] ++ trailing)

def HONEST : Array String := #["03c7dc4c7c6466f2", "16f7feeb615c1874"]

#guard (run HONEST).published.map (·.getObjValD "root") == some (.str ROOT)
#guard ((run HONEST).published.bind fun j => (Public.ofJson j).toOption).map (·.blocks) == some 2
#guard (run #["03c7dc4c7c6466f3", "16f7feeb615c1874"]).verdict == .reject
#guard (run HONEST (commitment := "6c" ++ COMMITMENT.drop 2)).reason == some "Dec(C*) does not match the commitment to W"
#guard (run #["ffffffffffffffff", "16f7feeb615c1874"]).reason == some "segment 0: block 0 is not below p"
#guard (run #["03c7dc4c7c6466f2"]).reason == some "segment 0 of C* is not 2 blocks of 8 bytes"
#guard (run HONEST (trailing := ⟨#[0]⟩)).reason == some "evidence: 1 bytes follow the last segment"
#guard (run HONEST (commitment := COMMITMENT ++ "00")).reason ==
  some "the salt is 24 bytes, drawn by the verifier after W is committed"
#guard (run HONEST (commitment := "00")).digest.isNone
#guard ((run HONEST (params := MANIFEST.replace "p2-6B" "p2-7B")).reason.map (·.startsWith "params: not the manifest"))
  == some true
#guard match check ⟨64, 187, 2, 1⟩ ⟨.empty, .empty⟩ #[] with
  | .reject r => r == "params: c0 ≡ 1 mod 4 (so p ≡ 3 mod 4) and below 2^16"
  | _ => false

end PousCheck.SetupProtocol
