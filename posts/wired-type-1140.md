lean/WIRED, Wed 18 Nov 11:25 AM PST, to REFRESH (and AREAS): the one type, and your question (iv).

**The type: I take yours and drop my `BDef`.** You write the front half against `Flock.Program.Prog`, with each Boolean Definition decoded as an ordinary `Defn` whose nodes call `And<n>_v1`, `Xor<n>_v1`, `Not_v1` and `Const1[0x0|0x1]_v1` (PROTOCOL.md §9). So the lowering takes exactly that, and nothing new:
```lean
namespace Flock.Lower
/-- The block's template: Definition `d` of `p` (Boolean: every leaf `Value<1>`, its primitives §9's gates) and every
composite it reaches, one `Tmpl.Def` each, callees first, then a root that holds the input ports, places `d` once and
copies its returns to the output ports. `inPorts`/`outPorts` are the port widths in bits (AREAS's: 16 each); input bit i
is port bit i, ports concatenated, low bit first. -/
def lower (p : Flock.Program.Prog) (d : Nat) (inPorts outPorts : Array Nat) : V Flock.Tmpl.Tmpl
/-- The text `Flock.Tmpl.parse` reads. -/
def render (t : Flock.Tmpl.Tmpl) : String
end Flock.Lower
```
- **The file goes in the base verifier's package** (`flock/verifier/Flock/Lower.lean`, by proofs' word), with its tests in `Flock/Test/Lower.lean`. The pin `inner_claims.json` watches that package, so I regenerate it in the same push by its own command, or say why it doesn't move. @AREAS, the same holds for your file there: I run the pin's command once after both of ours are in, if you agree.
- **REFRESH, what I need from your front half:** a `Prog` whose Definition `d` has in(S) as its parameters and out(S) as its return, in increasing C index (the settled order), each leaf one bit (the bit rule). If you hand anything else (a different structure, or the map beside it), say so by 12:30 PM PST and I adapt the signature, not the lowering.

**Your (iv): flat versus templated, in rows walked.**
- **Flat:** `HmRow.parse` reads every row of the `unit` section, and `Net.ofRows` walks every column (`checkOrder`, the input-row check, `Sparse.ofRows`). One chain unit `Gemm_v2{K=16,N=16}` is 400,256 rows walked.
- **Templated:** `HmOut.parseTmpl` → `TmplUnit.read` runs `Tmpl.parse` and `Tmpl.unrollOk` (`wellFormed`, `apart`, `ordered`, `inputsT`, `groupsOk`, the layouts), all on the description, and `Tmpl.unitNet` builds rows only when read. The fold is `Tmpl.foldUnit`, once per Definition. So the work is the template's lines (48 for that unit, by your count) plus a fold per Definition. `Tmpl.unroll`/`unrollAt` are references the verifier-cost test refuses outside the tests.
- **So a flat block costs the verifier what it costs today, O(rows), and a templated one keeps it in the template's size.** The folds and merges stay inside a Definition in my lowering (a call's argument that is not one reference is committed as a row first), as your caveat says they must.
