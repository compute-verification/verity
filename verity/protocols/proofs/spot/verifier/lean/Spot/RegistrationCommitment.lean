module

public import Spot.OneStage
public import Flock.Hm96

/-!
# The registration's hiding commitment

The whole registration digest covers the docket (`Docket.digest`: the circuit as `program` and `window.circuit`, the
partition, the sampling design as `law`, the window's compiler) and the registration log's compiler as the verifier reads
it, its digest and the advice's length cap ℓ:

    whole      = SHA-512(TAG ‖ 0 ‖ canon({"compiler": {"cap": ℓ, "digest": d} | null, "record": hex(Docket.digest rec)}))
    commitment = hm96-sha512 leaf of `whole` under a private salt (`Flock.Hm96.Default512.leaf`)

`compiler` is null only where the record names no compiler: a record that names one (`window.compiler.digest`) under a
verifier that read no log's compiler, or a log that registers none, is refused (`COMPILER`), forming and opening alike.
`opens` recomputes `whole` from the record and the compiler the verifier reads and checks the opening it is handed
against the published commitment; its hiding is the hm96 leaf's, and no guarantee states it. `boundCoin` is a coin
derived from a challenger coin and the commitment.

This form is built and wired into nothing: no verifier, challenger or runner calls `opens`, `whole` or `boundCoin`.
That the commitment itself precedes the coins is held by nothing until the challenger's log takes it as a coin's
`after`, with a negative that refuses a commitment recorded after the first coin.
-/

@[expose] public section

namespace Spot.RegistrationCommitment

open Lean (Json)
open Flock (V)

def TAG : String := "verity/registration-commitment/v1"
def COIN_TAG : String := "verity/registration-commitment/coin/v1"

/-- The refusal of an opening whose salt and digest give another leaf than the published commitment. -/
def OPENING : String := "registration-commitment-opening"
/-- The refusal of a digest handed to the verifier that is not the whole digest of the registration it reads. -/
def DIGEST : String := "registration-commitment-digest"
/-- The refusal of a record that names a compiler where the commitment's compiler would be null. -/
def COMPILER : String :=
  "registration-commitment-compiler: the record names a compiler (window.compiler.digest) and the verifier read no " ++
    "registration log's compiler, or a log that registers none: null commits only a log that registers no compiler"

/-- The record names a compiler: its `window.compiler.digest` is not null. -/
def namesCompiler (rec : Json) : Bool := !(getD (getD (getD rec "window") "compiler") "digest" matches .null)

/-- The registration log's compiler as the commitment covers it: `{cap, digest}`; null where the record names no
compiler and the verifier read no log's compiler or the log registers none; else `COMPILER`. -/
def compilerJson (rec : Json) : Option (Option (String × Nat)) → V Json
  | some (some (d, ℓ)) => pure (Json.mkObj [("cap", Lean.toJson ℓ), ("digest", .str d)])
  | _ => if namesCompiler rec then throw COMPILER else pure Json.null

/-- The whole registration digest of the record `rec` and the compiler `compiler` (`OneStage.paramsParsed`'s: none when
the verifier read no log's compiler, `some none` when the log registers none). -/
def whole (rec : Json) (compiler : Option (Option (String × Nat))) : V ByteArray := do
  let body := Json.mkObj [("compiler", ← compilerJson rec compiler), ("record", .str (Flock.hexOf (← Docket.digest rec)))]
  return Flock.WindowRecord.digestOf TAG (← Flock.canon body).toUTF8

/-- The hiding leaf of the whole digest `d` under `salt`. -/
def commitment (d salt : ByteArray) : ByteArray := Flock.Hm96.Default512.leaf d salt

/-- The published commitment `c` opens, under the digest `d` and salt `salt` handed to the verifier privately, to the
registration it reads (`rec`, `compiler`): `COMPILER` where `whole` refuses the record's compiler, `DIGEST` when `d` is
not `whole rec compiler`, `OPENING` when the salt is not 192 bytes or the leaf is not `c`. -/
def opens (c d salt : ByteArray) (rec : Json) (compiler : Option (Option (String × Nat))) : V Unit := do
  let mine ← whole rec compiler
  if Flock.hexOf d != Flock.hexOf mine then throw DIGEST
  if salt.size != Flock.Hm96.sha512.saltBytes || Flock.hexOf (commitment mine salt) != Flock.hexOf c then throw OPENING

/-- SHA-512(COIN_TAG ‖ 0 ‖ coin ‖ commitment): a coin the challenger released after `commitment`, made a function of it. -/
def boundCoin (coin c : ByteArray) : ByteArray := Flock.Sha512.hash (COIN_TAG.toUTF8.push 0 ++ coin ++ c)

/-- The field `k` of `j`, read as lowercase hex. -/
def hexField (j : Json) (k : String) : V ByteArray := do
  let .str s := getD j k | throw s!"{k} is not hex"
  Flock.unhex s

/-- `{record, compiler, digest, salt, commitment, coin}` to `{formed, whole, commitment, opens, coin}`: `whole`'s
refusal (empty when it forms), the whole digest and its leaf under `salt` (empty where it refuses or the salt is not 192
bytes), `opens`'s refusal (empty when it holds) and `boundCoin coin commitment`, all lowercase hex. `compiler` absent is
a verifier that read no log's compiler, null a log that registers none, else `{digest, cap}`. -/
def piece (j : Json) : V Json := do
  let compiler ← match field? j "compiler" with
    | none => pure none
    | some .null => pure (some none)
    | some b => match getD b "digest", nat? (getD b "cap") with
      | .str d, some ℓ => pure (some (some (d, ℓ)))
      | _, _ => throw "compiler is null or {digest, cap}"
  let rec_ := getD j "record"
  let salt ← hexField j "salt"
  let c ← hexField j "commitment"
  let (formed, wholeHex, leaf) := match whole rec_ compiler with
    | .ok mine => ("", Flock.hexOf mine,
        if salt.size == Flock.Hm96.sha512.saltBytes then Flock.hexOf (commitment mine salt) else "")
    | .error e => (e, "", "")
  let why := match opens c (← hexField j "digest") salt rec_ compiler with
    | .ok () => ""
    | .error e => e
  return Json.mkObj [("formed", .str formed), ("whole", .str wholeHex), ("commitment", .str leaf),
    ("opens", .str why), ("coin", .str (Flock.hexOf (boundCoin (← hexField j "coin") c)))]

end Spot.RegistrationCommitment
