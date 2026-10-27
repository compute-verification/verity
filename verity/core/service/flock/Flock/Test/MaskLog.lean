/-
`Zk.maskLogOf` reads `mask_log` from the circuit file's second line, the META line, whether or not a line follows it, and
is 14 when the file has one line, when META names no `mask_log` or isn't JSON, and when the META line isn't UTF-8. Bytes
past the META line are never read, so a file that isn't UTF-8 there still gives its META's `mask_log`.
-/
import Flock.Zk

namespace Flock.Test.MaskLog

open Flock

def file (line : String) (rest : ByteArray := "row 0\n".toUTF8) : ByteArray :=
  s!"flock-ir-unit/v2\n{line}\n".toUTF8 ++ rest

#guard Zk.maskLogOf (file "META {\"mask_log\": 12}") == 12
#guard Zk.maskLogOf (file "META {\"mask_log\": 9, \"k\": 1}" .empty) == 9
#guard Zk.maskLogOf "flock-ir-unit/v2\nMETA {\"mask_log\": 11}".toUTF8 == 11
#guard Zk.maskLogOf "flock-ir-unit/v2".toUTF8 == 14
#guard Zk.maskLogOf .empty == 14
#guard Zk.maskLogOf (file "META {\"k\": 1}") == 14
#guard Zk.maskLogOf (file "META not json") == 14
#guard Zk.maskLogOf ("flock-ir-unit/v2\nMETA {\"mask_log\": 12".toUTF8 ++ ⟨#[0xff]⟩ ++ "}\n".toUTF8) == 14
#guard Zk.maskLogOf (file "META {\"mask_log\": 12}" ⟨#[0xff, 0xfe]⟩) == 12

end Flock.Test.MaskLog
