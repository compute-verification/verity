/-
`ProvedScope.servedForm`, the last of `verify`'s refusals: a session is taken only at `verity/flock-circuit` or
`verity/flock-circuit+plain-leaves` on live coins. Every other version `Tags.all` knows (the superseded hm96 versions
and the selftest builds' `+seed-injection` ones among them) and a plain session on seed coins are refused, each with its
reason; a version before hm96 rows, or typed, is refused earlier by `ProvedScope.check`, and the version is named first.
-/
import Flock.ProvedScope

namespace Flock.Test.Served

open ProvedScope

def why (t : Tags) (c : Coins) : String :=
  match servedForm t c with
  | .error e => e
  | .ok () => ""

def tree : Coins := .tree ⟨#["session", "circuit/rep0", "circuit/rep1"], 9, 512⟩

#guard served == ["verity/flock-circuit", "verity/flock-circuit+plain-leaves"]
#guard why Tags.circuit .os == ""
#guard why Tags.circuit tree == ""
#guard why Tags.circuitPlainLeaves .os == ""
#guard why Tags.circuit .seed == SEED_COINS
#guard why Tags.circuitPlainLeaves .seed == SEED_COINS
#guard why Tags.circuitSelftest tree == "refused: verity/flock-circuit+seed-injection, a selftest build's statement, " ++
  "whose coins the prover injected: verify takes only verity/flock-circuit and verity/flock-circuit+plain-leaves"
#guard why Tags.circuit967b8d06Selftest .seed == "refused: verity/flock-circuit@967b8d06+seed-injection, a selftest " ++
  "build's statement, whose coins the prover injected: verify takes only verity/flock-circuit and " ++
  "verity/flock-circuit+plain-leaves"
#guard why Tags.circuitCec3e347 tree == "refused: verity/flock-circuit@cec3e347, a superseded statement version: verify " ++
  "takes only verity/flock-circuit and verity/flock-circuit+plain-leaves"

/-- The versions `ProvedScope.check` admits (hm96 rows, untyped) that `verify` still refuses: the six superseded ones and
five `+seed-injection` ones; the other two `+seed-injection` versions, `@b3727bd9`'s and `@cec3e347`'s, are gone. -/
def refused : List String :=
  ((Tags.all.filter fun t => t.hm96Rows && !t.typed).filter fun t => (versionWhy t).isSome).map (·.name)

#guard refused == ["verity/flock-circuit@eb90718f", "verity/flock-circuit@eb90718f+seed-injection",
  "verity/flock-circuit@e51e2b86", "verity/flock-circuit@e51e2b86+seed-injection", "verity/flock-circuit@967b8d06",
  "verity/flock-circuit@967b8d06+seed-injection", "verity/flock-circuit@f5df3bbc",
  "verity/flock-circuit@f5df3bbc+seed-injection", "verity/flock-circuit@b3727bd9", "verity/flock-circuit@cec3e347",
  "verity/flock-circuit+seed-injection"]
#guard (Tags.all.filter fun t => (versionWhy t).isNone).map (·.name) == served
#guard (Tags.find? "verity/flock-circuit@b3727bd9+seed-injection").isNone
#guard (Tags.find? "verity/flock-circuit@cec3e347+seed-injection").isNone

end Flock.Test.Served
