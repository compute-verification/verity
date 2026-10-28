/-
The interface every core protocol implements (PoUW, PoUS, the network certificate check, sampled proofs' one-stage
audit, the composition).

* `params`: the protocol's public parameters (k, Δ, the law, ...), canonical JSON;
* `commitment`: what the prover committed to before the challenge, and the digest that binds it;
* `evidence`: everything else the check reads (the session record, transcript, proofs, certificate);
* `check`: one total function from those three to an `Outcome`;
* the outcome: a verdict, and on acceptance alone the public outputs and the profile.

`Protocol.answer` is the whole of what a caller runs. It decodes the three byte strings, refuses non-canonical
parameters, runs the check, and turns every decoding failure into a rejection, so every input gets a verdict.
`Answer.toJson` is the outcome's one canonical form, the one the difftests compare and a composition reads.
-/
import Lean.Data.Json
import Flock.Bytes
import Flock.Canon

namespace Core

open Lean (Json)
open Flock (V)

/-- `undecided` does not reject the prover. The evidence doesn't settle the question (a link-window with no
certificate yet), and nothing is made public. -/
inductive Verdict where
  | accept
  | reject
  | undecided
  deriving BEq, DecidableEq, Repr, Inhabited

def Verdict.name : Verdict → String
  | .accept => "accept"
  | .reject => "reject"
  | .undecided => "undecided"

/-- Public outputs and the profile exist only on acceptance, so what a protocol makes public is a property of its
`accept` case. -/
inductive Outcome (Public Profile : Type) where
  | accept (published : Public) (profile : Profile)
  | reject (reason : String)
  | undecided (reason : String)

structure Protocol where
  /-- The protocol's id and version, e.g. `verity/pouw/v1`. Every outcome names it. -/
  id : String
  Params : Type
  Commitment : Type
  Evidence : Type
  Public : Type
  Profile : Type
  /-- The decoders. A refusal's message needn't name the input: `answer` prefixes it with `params: `, `commitment: `
  or `evidence: `. -/
  params : Json → V Params
  commitment : ByteArray → V Commitment
  evidence : ByteArray → V Evidence
  /-- The digest that binds the commitment, which a record or certificate cites. -/
  digest : Commitment → ByteArray
  check : Params → Commitment → Evidence → Outcome Public Profile
  /-- Never `null`, so that `Answer.ofJson` reads an accepted answer back. -/
  publicJson : Public → Json
  profileJson : Profile → Json

/-- The check accepts: the predicate a protocol's properties (`Security/Properties/<protocol>.lean`) are stated over. -/
def Protocol.Accepts (P : Protocol) (p : P.Params) (c : P.Commitment) (e : P.Evidence) : Prop :=
  ∃ pub prof, P.check p c e = .accept pub prof

/-- What `answer` returns: the verdict, the reason when it isn't `accept`, the commitment's digest once it decoded,
and the public outputs and profile as JSON on acceptance. -/
structure Answer where
  protocol : String
  verdict : Verdict
  reason : Option String
  digest : Option ByteArray
  published : Option Json
  profile : Option Json

/-- Canonical JSON: ASCII without DEL (every byte below `0x7f`) that `Flock.canon` writes back byte for byte, so every
string in it is ASCII without DEL too (`canon` refuses any other, an escaped one included). The parameters must be
canonical, and a protocol whose commitment or evidence is JSON reads it with this too. -/
def canonicalJson (b : ByteArray) : V Json := do
  unless b.data.all (· < 0x7f) do throw "not ASCII without DEL (a byte at or above 0x7f)"
  let some s := String.fromUTF8? b | throw "not UTF-8"
  let j ← match Json.parse s with
    | .ok j => pure j
    | .error e => throw s!"not JSON ({e})"
  if (← Flock.canon j) != s then throw "not canonical JSON (compact, keys sorted, integers only)"
  pure j

def Protocol.answer (P : Protocol) (params commitment evidence : ByteArray) : Answer :=
  let refuse (reason : String) (digest : Option ByteArray) : Answer :=
    { protocol := P.id, verdict := .reject, reason := some reason, digest, published := none, profile := none }
  match canonicalJson params >>= P.params with
  | .error e => refuse s!"params: {e}" none
  | .ok p =>
    match P.commitment commitment with
    | .error e => refuse s!"commitment: {e}" none
    | .ok c =>
      let d := P.digest c
      match P.evidence evidence with
      | .error e => refuse s!"evidence: {e}" (some d)
      | .ok ev =>
        match P.check p c ev with
        | .accept pub prof => { protocol := P.id, verdict := .accept, reason := none, digest := some d,
                                published := some (P.publicJson pub), profile := some (P.profileJson prof) }
        | .reject r => refuse r (some d)
        | .undecided r => { protocol := P.id, verdict := .undecided, reason := some r, digest := some d,
                            published := none, profile := none }

private def opt {α} (f : α → Json) : Option α → Json
  | some a => f a
  | none => Json.null

def Answer.toJson (a : Answer) : Json :=
  Json.mkObj [("protocol", Json.str a.protocol), ("verdict", Json.str a.verdict.name), ("reason", opt Json.str a.reason),
    ("commitment", opt (fun d => Json.str (Flock.hexOf d)) a.digest), ("public", opt id a.published),
    ("profile", opt id a.profile)]

/-- The outcome's line in canonical JSON's form (`Flock.canonLine`, which prints a reason's free text whatever its
characters), the only form a caller or a difftest reads. It fails only when a protocol's `publicJson` or `profileJson`
writes a non-integer number, which the protocol's own tests must rule out. -/
def Answer.canonical (a : Answer) : V String := Flock.canonLine a.toJson

def Verdict.ofName : String → Option Verdict
  | "accept" => some .accept
  | "reject" => some .reject
  | "undecided" => some .undecided
  | _ => none

/-- The inverse of `toJson`, for a protocol that consumes another's answer (the profile is `Integrity.ofJson`'s to
read). It accepts only what `toJson` writes, and only the shapes `answer` returns: an `accept` has no reason and both its
public outputs and its profile, which are therefore never `null`; a `reject` or `undecided` has a reason and neither. -/
def Answer.ofJson (j : Json) (ctx : String := "answer") : V Answer := do
  let some verdict := Verdict.ofName (← Flock.J.str j "verdict" ctx)
    | throw s!"{ctx}.verdict: not accept, reject or undecided"
  let opt (k : String) : V (Option Json) := do
    match ← Flock.J.get j k ctx with
    | Json.null => pure none
    | v => pure (some v)
  let a : Answer := {
    protocol := ← Flock.J.str j "protocol" ctx, verdict,
    reason := ← (← opt "reason").mapM (Flock.J.asStr · s!"{ctx}.reason"),
    digest := ← (← opt "commitment").mapM fun v => do Flock.unhex (← Flock.J.asStr v s!"{ctx}.commitment"),
    published := ← opt "public", profile := ← opt "profile" }
  match verdict, a.reason, a.published, a.profile with
  | .accept, none, some _, some _ => pure ()
  | .accept, _, _, _ => throw s!"{ctx}: an accept has no reason, and both its public outputs and its profile"
  | _, some _, none, none => pure ()
  | v, _, _, _ => throw s!"{ctx}: a {v.name} has a reason, and neither public outputs nor a profile"
  if (← Flock.canonLine a.toJson) != (← Flock.canonLine j) then throw s!"{ctx}: not in the form Answer.toJson writes"
  pure a

end Core
