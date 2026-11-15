module

public import Proofs.Flock.SessionLoop.Concrete.Stmt
import Proofs.Flock.SessionLoop.Concrete.Fwd
import Flock.JsonParse
import all Init.Data.String.Defs

/-! # The concrete statement, drawing its one unit

`concreteStmt` carries no draw, so the masked verifier refuses it at every run (`ProvedScope.sampleDesign` on `none`)
and `FlockComplete.readyZk` is false there. `concreteStmtD` is the same statement with a subset draw of its
population's one unit (`DRAW`), which the masked verifier's scope admits (`ProvedScope.samplingDesignWhy`: a subset
draw under `--zk`). Drawing unit `0` of a population of `1` keeps `n = 1`, so its table keeps `m = k_log + 3 = 25`.

* `concreteRawD_fits`, `concreteStmtD_decode`: it decodes back to `concreteRawD`, at the two fixed nets' build
  (`Properties.Flock.Nets.fixedNetsBuild`).
* `jsonOf_draw`: its draw parses, to `drawJ`.
-/

@[expose] public section

namespace FlockDeploy

open Flock
open Lean (Json)
open Properties.Flock.Nets (FixedNetsBuild)

/-- The draw's text: a subset draw of unit `0` of a population of one, its keys in sorted order. -/
def drawChars : List Char :=
  ['{', '"', 'k', '"', ':', '1', ',', '"', 'l', 'a', 'w', '"', ':', '"', 's', 'u', 'b', 's', 'e', 't', '"', ',', '"',
   'p', 'o', 'p', 'u', 'l', 'a', 't', 'i', 'o', 'n', '"', ':', '1', ',', '"', 'u', 'n', 'i', 't', 's', '"', ':', '[',
   '0', ']', '}']

def DRAW : String := String.ofList drawChars

/-- The draw, parsed. -/
def drawJ : Json := (JsonParse.parseList drawChars).toOption.getD .null

theorem draw_parseList : JsonParse.parseList drawChars = .ok drawJ := by kernel_rfl'

theorem draw_parse : JsonParse.parse DRAW = .ok drawJ := by
  rw [JsonParse.parse_eq, DRAW, String.toList_ofList]
  exact draw_parseList

/-- **The drawn concrete statement**, decoded: `concreteRaw` with `DRAW` as its draw. -/
def concreteRawD : Raw := { concreteRaw with draw := some DRAW.toByteArray }

/-- **The drawn concrete `--zk` statement**: `concreteRawD`'s encoding. -/
def concreteStmtD : Claims.Msg := concreteRawD.encode

theorem draw_size : DRAW.toByteArray.size < 2 ^ 64 := by decide +kernel

theorem concreteRawD_fits (hn : FixedNetsBuild) : concreteRawD.Fits :=
  have h := concreteRaw_fits hn
  ⟨h.statement, h.circuit, h.publicFile, h.tables, h.tableItems, h.count,
    fun b hb => (Option.some.inj (hb : some DRAW.toByteArray = some b)) ▸ draw_size, h.partition, h.program, h.work,
    h.workTable, h.stratified, h.closure, h.registered, h.publicInputs⟩

/-- **The drawn concrete statement decodes** to the `Raw` it encodes. -/
theorem concreteStmtD_decode (hn : FixedNetsBuild) : decode concreteStmtD = some concreteRawD :=
  decode_encode_of_fits (concreteRawD_fits hn)

/-- **Its draw parses**, as `Raw.ins` reads it. -/
theorem jsonOf_draw : jsonOf "draw" DRAW.toByteArray = .ok drawJ := by
  unfold jsonOf
  rw [Concrete.fromUTF8?_toByteArray]
  show (JsonParse.parse DRAW).mapError _ = _
  rw [draw_parse]
  rfl

end FlockDeploy
