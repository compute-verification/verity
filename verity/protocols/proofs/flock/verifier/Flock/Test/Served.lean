/-
`ProvedScope.servedForm`, the last of `verify`'s refusals: a session is taken only at `verity/flock-circuit`,
`verity/flock-circuit+plain-leaves`, `verity/flock-circuit+wide-leaf` or the templated `verity/flock-circuit/tmpl` and
`verity/flock-circuit/tmpl+wide-leaf` on the challenger's keyed coin tree. Every
other version `Tags.all` knows (the superseded hm96 versions and the selftest builds' `+seed-injection` ones among them)
and a plain session on seed coins are refused, each with its reason; a version before hm96 rows, or typed, is refused
earlier by `ProvedScope.check`, and the version is named first.
-/
module

public import Flock.ProvedScope
meta import Flock.ProvedScope

@[expose] public section

namespace Flock.Test.Served

open ProvedScope

def why (t : Tags) (c : Coins) : String :=
  match servedForm t c with
  | .error e => e
  | .ok () => ""

def tree : Coins := .tree ⟨#["session", "circuit/rep0", "circuit/rep1"], 9, 512⟩

/-- The templated statement's version: `Tags.circuitTmpl`'s name, without building its identity. -/
def tmpl : Tags := { Tags.circuit with name := Tags.tmplName }

/-- The templated wide-leaf statement's version, by name as `tmpl` is. -/
def tmplWide : Tags := { Tags.circuit with name := Tags.tmplWideLeafName }

/-- The versions `verify` takes, as its refusals name them. -/
def TAKEN : String :=
  "verity/flock-circuit, verity/flock-circuit+plain-leaves, verity/flock-circuit+wide-leaf, verity/flock-circuit/tmpl " ++
  "and verity/flock-circuit/tmpl+wide-leaf"

#guard served == ["verity/flock-circuit", "verity/flock-circuit+plain-leaves", "verity/flock-circuit+wide-leaf",
  "verity/flock-circuit/tmpl", "verity/flock-circuit/tmpl+wide-leaf"]
#guard why Tags.circuit tree == ""
#guard why Tags.circuitPlainLeaves tree == ""
#guard why Tags.circuitWideLeaf tree == ""
#guard why tmpl tree == ""
#guard why tmplWide tree == ""
-- a wide-leaf statement is taken under --zk only, and in no recursive session; any other statement wherever it was
#guard (wideLeafWhy Tags.circuitWideLeaf true false).isNone && (wideLeafWhy tmplWide true false).isNone
#guard (wideLeafWhy Tags.circuitWideLeaf false false).isSome && (wideLeafWhy Tags.circuitWideLeaf true true).isSome
#guard (wideLeafWhy tmplWide false false).isSome && (wideLeafWhy tmplWide true true).isSome
#guard Tags.all.all fun t => t.wideLeaf || [true, false].all fun zk => [true, false].all fun r => (wideLeafWhy t zk r).isNone
#guard (Tags.all ++ Tags.templated).all fun t => t.wideLeaf == t.name.endsWith "+wide-leaf"
-- a wide-leaf statement's rep domains are its own, one table or several
#guard [(Tags.circuit, Tags.circuitWideLeaf), (Tags.circuitTmpl, Tags.circuitTmplWideLeaf)].all fun (t, w) =>
  [1, 2].all fun count => (t.tableNames count).all fun tab => [0, 1].all fun r =>
    t.tableDomain count tab r != w.tableDomain count tab r &&
      w.tableDomain count tab r == (t.tableDomain count tab r).replace "/fast100x2/" "/fast100-wide-leafx2/"
#guard Tags.circuitWideLeaf.sigmaTag == Tags.circuit.sigmaTag && Tags.circuitWideLeaf.statement == Tags.circuit.statement
#guard why Tags.circuit .seed == SEED_COINS
#guard why Tags.circuitPlainLeaves .seed == SEED_COINS
#guard why tmpl .seed == SEED_COINS
#guard why Tags.circuitSelftest tree == "refused: verity/flock-circuit+seed-injection, a selftest build's statement, " ++
  "whose coins the prover injected: verify takes only " ++ TAKEN
#guard why Tags.circuit967b8d06Selftest .seed == "refused: verity/flock-circuit@967b8d06+seed-injection, a selftest " ++
  "build's statement, whose coins the prover injected: verify takes only " ++ TAKEN
#guard why Tags.circuitCec3e347 tree == "refused: verity/flock-circuit@cec3e347, a superseded statement version: verify " ++
  "takes only " ++ TAKEN

/-- The versions `ProvedScope.check` admits (hm96 rows, untyped) that `verify` still refuses: the six superseded ones and
five `+seed-injection` ones; the other two `+seed-injection` versions, `@b3727bd9`'s and `@cec3e347`'s, are gone. -/
def refused : List String :=
  ((Tags.all.filter fun t => t.hm96Rows && !t.typed).filter fun t => (versionWhy t).isSome).map (·.name)

#guard refused == ["verity/flock-circuit@eb90718f", "verity/flock-circuit@eb90718f+seed-injection",
  "verity/flock-circuit@e51e2b86", "verity/flock-circuit@e51e2b86+seed-injection", "verity/flock-circuit@967b8d06",
  "verity/flock-circuit@967b8d06+seed-injection", "verity/flock-circuit@f5df3bbc",
  "verity/flock-circuit@f5df3bbc+seed-injection", "verity/flock-circuit@b3727bd9", "verity/flock-circuit@cec3e347",
  "verity/flock-circuit+seed-injection"]
#guard (Tags.all.filter fun t => (versionWhy t).isNone).map (·.name) ++ Tags.templated.map (·.name) == served
#guard (Tags.find? "verity/flock-circuit@b3727bd9+seed-injection").isNone
#guard (Tags.find? "verity/flock-circuit@cec3e347+seed-injection").isNone

end Flock.Test.Served
