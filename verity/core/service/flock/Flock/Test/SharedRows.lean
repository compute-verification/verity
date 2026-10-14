/-
`RecVStar.checkShared` (`SharedOwnRoot`): a session's port reading a value the verifier's shared rows (`--rec-rows`)
register passes only at that registration (the first entry reading the value; the reads may differ), and a port reading
R's tops `rec-top-stacked-L<l>` passes only where the shared rows register them. A port reading a value the shared rows
do not register, other than R's tops, is not checked here. RecSparse and R's opening fail closed unless the rows register
`rec-claim`.
-/
import Flock.RecVStar

namespace Flock.Test.SharedRows

def port (name value : String) (root : UInt8 := 1) (leaves : Nat := 64) (domain : UInt8 := 7)
    (reads : Array (Option Nat) := #[some 0]) : Registered.Port :=
  { name, value, root := ⟨#[root]⟩, leaves, words := some 64, wordBits := 16, domain := ⟨#[domain]⟩, reads }

def level : VStar.Stmt := .recOpenStacked 1 16 4 53

def top : String := "rec-top-stacked-L1"

def rows : Array Registered.Port := #[port "m0" "rec-c0", port top top (leaves := 128), port "acc0" "rec-acc"]

def passes (registered : Array Registered.Port) (rows : Option (Array Registered.Port)) : Bool :=
  (RecVStar.checkShared level registered rows).toBool

#guard passes #[port "out" top (leaves := 128) (reads := #[some 3]), port "acc" "rec-acc" (reads := #[some 9])] (some rows)
#guard passes #[port "coef" "rec-coef" (root := 2)] (some rows)
#guard passes #[port "coef" "rec-coef" (root := 2)] none
#guard passes #[port "m0" "rec-c0", port "h" "rec-c0" (reads := #[none])] (some rows)

#guard !(RecVStar.checkShared (.sparse ⟨18, 13, 6, 0⟩ 38) #[] (some rows)).toBool
#guard (RecVStar.checkShared (.sparse ⟨18, 13, 6, 0⟩ 38) #[]
  (some (rows.push (port "claim" RecVStar.CLAIM_VALUE)))).toBool

#guard !passes #[port "out" top (leaves := 128) (root := 2)] (some rows)
#guard !passes #[port "out" top] (some rows)
#guard !passes #[port "out" top (leaves := 128) (domain := 8)] (some rows)
#guard !passes #[port "out" top (leaves := 128)] none
#guard !passes #[port "out" "rec-top-stacked-L0"] (some rows)
#guard !passes #[port "h" "rec-c0" (root := 3)] (some rows)
#guard !passes #[port "acc" "rec-acc", { port "acc_out" "rec-acc" with wordBits := 8 }] (some rows)
#guard !passes #[port "m0" "rec-c0"] (some (#[port "m0'" "rec-c0" (root := 5)] ++ rows))

def why (registered : Array Registered.Port) (rows : Option (Array Registered.Port)) : String :=
  match RecVStar.checkShared level registered rows with
  | .error e => e
  | .ok () => ""

#guard why #[port "out" top (leaves := 128) (root := 2)] (some rows) ==
  "rec-open-stacked/l16-h4: its port out reads rec-top-stacked-L1, which the verifier's shared rows (--rec-rows) register as rec-top-stacked-L1: root 02, against 01"
#guard why #[port "out" top (leaves := 128)] none ==
  "rec-open-stacked/l16-h4: its port out reads R's tops rec-top-stacked-L1; pass the verifier's shared rows (--rec-rows F), which register it"

end Flock.Test.SharedRows
