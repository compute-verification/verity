/-
U4 (`Registered.checkFull`) on a five-unit population of two strata, `a = [0, 1)` and `b = [1, 5)`: `stratified:2` draws `a`
in full (`k_a = 1 = n_a`) and two of `b`'s four units. A unit reading a round's commitment must be in `a`.
-/
import Flock.Registered

namespace Flock.Test.Full

open Lean (Json)
open Flock.Registered

def n (x : Nat) : Json := Lean.toJson x

def port (value : String) (reads : Array (Option Nat)) : Registered.Port :=
  { name := "m0", value, root := ByteArray.empty, leaves := 1, words := none, wordBits := 16,
    domain := ByteArray.empty, reads }

def stratified : Json := Json.mkObj [("law", "stratified"), ("population", (n 5)), ("k", (n 2)),
  ("strata", Json.arr #[
    Json.mkObj [("template", "a"), ("units", Json.arr #[Json.arr #[(n 0), (n 1)]]), ("k", (n 1))],
    Json.mkObj [("template", "b"), ("units", Json.arr #[Json.arr #[(n 1), (n 5)]]), ("k", (n 2))]]),
  ("units", Json.arr #[(n 0), (n 1), (n 3)])]

def subset (k : Nat) : Json := Json.mkObj [("law", "subset"), ("population", (n 5)), ("k", n k),
  ("units", Json.arr ((List.range k).toArray.map fun u => Lean.toJson u))]

def ok (own : Option (Array Registered.Port)) (draw : Option Json) : Bool := (checkFull own draw).isOk

def onlyA : Array (Option Nat) := #[some 0, none, none, none, none]
def aAndB : Array (Option Nat) := #[some 0, none, none, some 0, none]

#guard roundCommitment "rec-c0" && roundCommitment "rec-c47" && roundCommitment "rec-top-L3"
#guard !roundCommitment "rec-coef" && !roundCommitment "rec-acc" && !roundCommitment "rec-c" &&
  !roundCommitment "rec-top-L" && !roundCommitment "rec-rows-L1" && !roundCommitment "rec-residuals-zero" &&
  !roundCommitment "cs_table"
#guard roundCommitment "rec-opening-c0" && roundCommitment "rec-opening-c31" && roundCommitment "rec-top-stacked-L0" &&
  roundCommitment "rec-top-stacked-L5" && roundCommitment "rec-sparse-pre0" && roundCommitment "rec-sparse-sc12" &&
  roundCommitment "rec-sparse-v2" && roundCommitment "rec-sparse-u" && roundCommitment "rec-sparse-vt"
#guard !roundCommitment "rec-opening-c" && !roundCommitment "rec-top-stacked-L" && !roundCommitment "rec-sparse-pre" &&
  !roundCommitment "rec-sparse-v" && !roundCommitment "rec-sparse-zero" && !roundCommitment "rec-sparse-tau" &&
  !roundCommitment "rec-claim" && !roundCommitment "rec-opening-c2x" && !roundCommitment "rec-opening/m22-c2-p0"

#guard ok (some #[port "rec-c0" onlyA]) (some stratified)
#guard ok (some #[port "rec-top-L0" onlyA]) (some stratified)
#guard !ok (some #[port "rec-c0" aAndB]) (some stratified)
#guard !ok (some #[port "rec-top-L1" aAndB]) (some stratified)
#guard ok (some #[port "rec-opening-c0" onlyA, port "rec-top-stacked-L1" onlyA]) (some stratified)
#guard !ok (some #[port "rec-opening-c0" aAndB]) (some stratified)
#guard !ok (some #[port "rec-top-stacked-L1" aAndB]) (some stratified)
#guard !ok (some #[port "rec-sparse-u" aAndB]) (some stratified)
#guard ok (some #[port "rec-coef" aAndB, port "cs_table" aAndB]) (some stratified)
#guard !ok (some #[port "rec-coef" aAndB, port "rec-c3" aAndB]) (some stratified)
#guard ok (some #[port "rec-c0" aAndB]) none
#guard ok none (some stratified)
#guard ok (some #[port "rec-c0" aAndB]) (some (subset 5))
#guard !ok (some #[port "rec-c0" aAndB]) (some (subset 4))

end Flock.Test.Full
