/-
`Session.State.request`, the session loop's decision on each request after the prover's `Register` and `Hello`, at a
session of two rep streams: along the honest prover's order (the first rep stream's `Open`, `Commit`, `Link`, its rounds,
the second's `Open` and rounds, `Finish`) each leaves once, at its place; a second `Register`, `Hello`, `Link` or `Open`,
and each of them out of its place, is refused; after a refusal every request is refused for its reason; and a `Commit`
or `Round` the auditor answers without coins stops the session (`Session.State.answer`), so it never leaves again.
-/
module

public import Flock.Firewall.Session
meta import Flock.Firewall.Session

@[expose] public section

namespace Flock.Test.SessionPlace

open Flock.Firewall.Session

def A : String := "t/rep0"
def B : String := "t/rep1"

def stop (why : String) : String := Flock.Firewall.Coins.STOP ++ why

/-- A session past `Hello`'s answer whose rep streams `A` then `B` have the rounds `plan`. -/
def past (plan : Array (String × Nat)) : State :=
  { sent := true, plan, labels := #[(A, "dA"), (B, "dB")], tables := #[("t", 20)], points := 2,
    session := some { hello := ⟨⟨#[Flock.Firewall.Coins.SESSION, A, B], 4, 1⟩, 2, []⟩, key := .empty, root := [],
                      opened := [] } }

def S₀ : State := past #[(A, 2), (A, 3), (B, 4)]

/-- The prover's `Register` left. -/
def P₀ : Placed := { register := true }

/-- What the prover receives in a refusal's place: its reason. -/
def reasonOf (out : ByteArray) : String := ((getBytes out 1).bind fun (b, _) => String.fromUTF8? b).getD "?"

/-- `S` with one more opening counted, as `State.coins` counts it once the coins open the tree. -/
def opened (S : State) : State := { S with session := S.session.map fun C => { C with opened := C.opened ++ [0] } }

/-- The loop's decision on each request of `ks` in turn, from `S` once `P` left, as `relay` asks it of every request but
a `Ping` and a `Proof` before a stop, which it answers itself: `ok` where it leaves (a `Commit` or `Round` then opened),
else the reason of its refusal. -/
def decisions (S : State) (P : Placed) : List Flock.Firewall.Session.Kind → List String
  | [] => []
  | k :: ks =>
    match S.request P k with
    | some (S', out) => reasonOf out :: decisions S' P ks
    | none => "ok" :: decisions (match k with | .commit | .round _ _ => opened S | _ => S) (P.after k) ks

def honest : List Flock.Firewall.Session.Kind :=
  [.openStream A, .commit, .link, .round A 2, .round A 3, .openStream B, .round B 4, .finish]

#guard decisions S₀ P₀ honest == List.replicate honest.length "ok"
#guard decisions {} {} [.register] == ["ok"]

/-- The honest order's first `n` requests, then `k`: what the loop decides of `k`. -/
def decideAt (n : Nat) (k : Flock.Firewall.Session.Kind) : String :=
  ((decisions S₀ P₀ (honest.take n ++ [k])).drop n).headD "?"

#guard decideAt 0 .register == stop "a second Register"
#guard decideAt 3 .register == stop "a second Register"
#guard decideAt 0 .hello == stop "a second Hello"
#guard decisions {} {} [.hello] == [stop "a Hello before the Register"]
#guard decisions {} P₀ [.link] == [stop "a Link before the session's Hello"]
#guard decisions {} P₀ [.openStream A] == [stop s!"an Open of {A}, which is none of the session's rep streams"]
#guard decideAt 0 (.openStream "x") == stop "an Open of x, which is none of the session's rep streams"
#guard decideAt 0 (.openStream B) == stop s!"an Open of {B} before the Open of {A}"
#guard decideAt 1 (.openStream A) == stop s!"a second Open of {A}"
#guard decideAt 0 .commit == stop s!"a Commit before the Open of {A}"
#guard decideAt 1 .link == stop "a Link at opening 0, where its place is just after the step-0 coin commitment"
#guard decideAt 2 (.round A 2) == stop s!"a round of {A} before the Link"
#guard decideAt 2 (.openStream B) == stop s!"an Open of {B} before the Link"
#guard decideAt 3 .link == stop "a second Link"
#guard decideAt 5 .link == stop "a second Link"
#guard decideAt 4 (.openStream B) == stop s!"an Open of {B} before the last round of {A}"
#guard decideAt 5 (.round B 4) == stop s!"a round of {B} before its Open"
#guard decideAt 6 (.openStream B) == stop s!"a second Open of {B}"
#guard decideAt 6 (.openStream A) == stop s!"a second Open of {A}"
#guard decideAt 7 .register == stop "a second Register"
#guard decideAt 3 .other == stop "a request the session loop does not serve"

/-- With no round of `B`'s, `Finish` comes once `A`'s rounds are past and `B` opened, and not before. -/
def S₁ : State := past #[(A, 2)]

#guard decisions S₁ P₀ [.openStream A, .commit, .link, .round A 2, .openStream B, .finish] == List.replicate 6 "ok"
#guard decisions S₁ P₀ [.openStream A, .commit, .link, .round A 2, .finish] ==
  ["ok", "ok", "ok", "ok", stop s!"a Finish before the Open of {B}"]

/-- After a refusal every request is refused for its reason, the auditor asked nothing more. -/
def after : List Flock.Firewall.Session.Kind :=
  [.register, .hello, .openStream A, .openStream "x", .commit, .link, .round A 2, .proof, .other, .ping, .finish]

#guard [0, 1, 3, 5].all fun n =>
  let why := decideAt n (.round B 4)
  why.startsWith Flock.Firewall.Coins.STOP &&
    (decisions S₀ P₀ (honest.take n ++ Flock.Firewall.Session.Kind.round B 4 :: after)).drop n ==
      List.replicate (after.length + 1) why
#guard decisions {} {} (Flock.Firewall.Session.Kind.round A 2 :: after) ==
  List.replicate (after.length + 1) (stop "coins before the step-0 coin commitment")

/-- What the prover receives for the auditor's answer `a` to `k` at its place (`A` opened, `S₀`'s first opening), and
what the loop then decides of `k` sent again. -/
def answerThen (k : Flock.Firewall.Session.Kind) (a : ByteArray) : String × String :=
  let (S', out) := S₀.answer k a
  (reasonOf out, ((S'.request (P₀.after (.openStream A)) k).map (reasonOf ·.2)).getD "ok")

/- A `Commit` or `Round` the auditor answers with anything but coins stops the session, refused, the auditor's reason
kept where it gave one: sent again, it is refused for that reason without asking the auditor, so it leaves once. -/
#guard answerThen .commit (ByteArray.mk #[1]) ==
  (stop "the auditor answered the step-0 coin commitment without coins",
   stop "the auditor answered the step-0 coin commitment without coins")
#guard answerThen .commit (refused "no") ==
  (stop "the auditor answered the step-0 coin commitment without coins: no",
   stop "the auditor answered the step-0 coin commitment without coins: no")
#guard answerThen (.round A 2) (ByteArray.mk #[1]) ==
  (stop s!"the auditor answered a round of {A} without coins",
   stop s!"the auditor answered a round of {A} without coins")
#guard (S₀.request (P₀.after (.openStream A)) .commit).isNone

end Flock.Test.SessionPlace
