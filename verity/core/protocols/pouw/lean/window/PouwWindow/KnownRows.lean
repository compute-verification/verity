import PouwWindow.Lanes

/-!
# A drawn H1 leaf's known rows

The rows the verifier registers itself at their known content (`Source.known`): lane `l`'s `pub` and `admit`, and every
statement's `o0`, each one row of 16-bit words (`hm96-sha512/row/v2`) at the port that holds it (`lanePorts`,
`okPort`). This is their content, which the known-content check (`Flock.Registered.checkKnownAll`) holds the registered
roots to, as a function of the leaf's public inputs:

* `pub<l>`: its fields' bits in `lanePorts`' order, each least significant bit first: the salt (`digestBits`), the
  position (`positionBits`), the roots (`2 · digestBits`) and the index `l` (`laneBits`), 1616 bits in 101 words;
* `admit`: the leaf's admit bits, row `i`'s at bit `i`;
* `o0`: the row 1 (`ok = 1`, bit 0, zero-padded to one word).

Each field is a natural number whose bit `t` is the field's bit `t`. A digest's is `halvesNat` of its bytes (its
big-endian u32 halves, half `h` at bits `[32h, 32h + 32)`, `seeds.leaves`), a position's `positionNat` of the u64
(its two big-endian halves, the high one first, `seeds.position_leaves`), and the roots' `rootsNat` (`roots[0]`'s
halves, then `roots[1]`'s).

`tests/test_known_rows.py` holds `knownRows` to the Python that writes the served path's stand-in known rows
(`benchmarks/pouw/h1_lanes/run.py`'s `known_rows`: `pub_words`, `bits_words` and `h1unit.OK`). Proofs read it through
`Accounting.TS.Lanes.knownOf` (`Security/Proofs/Accounting/TS/Session/Lanes/KnownOf.lean`), the known content
`Accounting.Guarantees.TS.LeafLanesSound` hands each session.
-/

namespace PouwWindow.Lanes

/-- `n` bits of `x`, least significant first. -/
def natBits (n x : Nat) : List Bool := (List.range n).map x.testBit

/-- The natural number bits spell, least significant first. -/
def bitsNat : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + 2 * bitsNat bs

/-- **A bit row's 16-bit words** (`bits_words`): bit `t` is bit `t % 16` of word `t / 16`, the last word zero-padded. -/
def rowWords (bs : List Bool) : Array Nat :=
  ((List.range ((bs.length + 15) / 16)).map fun w => bitsNat ((bs.drop (16 * w)).take 16)).toArray

/-- The big-endian u32 at byte `i`. -/
def be32 (b : ByteArray) (i : Nat) : Nat :=
  (b.get! i).toNat * 2 ^ 24 + (b.get! (i + 1)).toNat * 2 ^ 16 + (b.get! (i + 2)).toNat * 2 ^ 8 + (b.get! (i + 3)).toNat

/-- **A digest's field** (`seeds.leaves`): its big-endian u32 halves, half `h` at bits `[32h, 32h + 32)`. -/
def halvesNat (b : ByteArray) : Nat := ((List.range (b.size / 4)).map fun h => be32 b (4 * h) * 2 ^ (32 * h)).sum

/-- **A position's field** (`seeds.position_leaves`): the u64's big-endian halves, the high one first. -/
def positionNat (p : Nat) : Nat := p / 2 ^ 32 + 2 ^ 32 * (p % 2 ^ 32)

/-- **The roots' field**: `roots[0]`'s halves, then `roots[1]`'s. -/
def rootsNat (r0 r1 : ByteArray) : Nat := halvesNat r0 + 2 ^ digestBits * halvesNat r1

/-- Lane `l`'s `pub` row's bits: the salt, the position, the roots and the index, each at its field's width. -/
def pubBits (salt position roots lane : Nat) : List Bool :=
  natBits digestBits salt ++ natBits positionBits position ++ natBits (2 * digestBits) roots ++ natBits laneBits lane

/-- Lane `l`'s `pub` row. -/
def pubRow (salt position roots lane : Nat) : Array Nat := rowWords (pubBits salt position roots lane)

/-- The `admit` row: the admit bits. -/
def admitRow (admit : List Bool) : Array Nat := rowWords admit

/-- Every statement's `o0` row: `ok = 1`. -/
def okRow : Array Nat := rowWords [true]

/-- **A statement's known rows**, by port, each one row under its salt (`salts` by port name): lane `l`'s (`some l`)
`admit`, `pub` and `o0`, and the combine's (`none`) `o0`. -/
def knownRows (admit : List Bool) (salt position roots : Nat) (salts : String → ByteArray) :
    Option Nat → Array (String × Array (Array Nat × ByteArray))
  | some l => #[("admit", #[(admitRow admit, salts "admit")]), ("pub", #[(pubRow salt position roots l, salts "pub")]),
      ("o0", #[(okRow, salts "o0")])]
  | none => #[("o0", #[(okRow, salts "o0")])]

end PouwWindow.Lanes
