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

`opens` recomputes `whole` from the record and the compiler the verifier reads and checks the opening it is handed
against the published commitment. No verifier reads it yet, and no guarantee states it; its hiding is the hm96 leaf's.
`boundCoin` is a coin derived from a challenger coin and the commitment, which no verifier reads yet either.
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

/-- The registration log's compiler as the commitment covers it: `{cap, digest}`, or null when the log registers none or
the verifier reads none. -/
def compilerJson : Option (Option (String × Nat)) → Json
  | some (some (d, ℓ)) => Json.mkObj [("cap", Lean.toJson ℓ), ("digest", .str d)]
  | _ => Json.null

/-- The whole registration digest of the record `rec` and the compiler `compiler` (`OneStage.paramsParsed`'s). -/
def whole (rec : Json) (compiler : Option (Option (String × Nat))) : V ByteArray := do
  let body := Json.mkObj [("compiler", compilerJson compiler), ("record", .str (Flock.hexOf (← Docket.digest rec)))]
  return Flock.WindowRecord.digestOf TAG (← Flock.canon body).toUTF8

/-- The hiding leaf of the whole digest `d` under `salt`. -/
def commitment (d salt : ByteArray) : ByteArray := Flock.Hm96.Default512.leaf d salt

/-- The published commitment `c` opens, under the digest `d` and salt `salt` handed to the verifier privately, to the
registration it reads (`rec`, `compiler`): `DIGEST` when `d` is not `whole rec compiler`, `OPENING` when the salt is not
192 bytes or the leaf is not `c`. -/
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

/-- `{record, compiler, digest, salt, commitment, coin}` to `{whole, commitment, opens, coin}`: the whole digest, the
leaf of it under `salt`, `opens`'s refusal (empty when it holds) and `boundCoin coin commitment`, all lowercase hex.
`compiler` absent is none, null is a log that registers none, else `{digest, cap}`. -/
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
  let mine ← whole rec_ compiler
  let why := match opens c (← hexField j "digest") salt rec_ compiler with
    | .ok () => ""
    | .error e => e
  let leaf := if salt.size == Flock.Hm96.sha512.saltBytes then Flock.hexOf (commitment mine salt) else ""
  return Json.mkObj [("whole", .str (Flock.hexOf mine)), ("commitment", .str leaf),
    ("opens", .str why), ("coin", .str (Flock.hexOf (boundCoin (← hexField j "coin") c)))]

end Spot.RegistrationCommitment
