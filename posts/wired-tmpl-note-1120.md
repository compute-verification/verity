lean/WIRED, Wed 18 Nov 11:16 AM PST, to REFRESH and AREAS: the templated form's writer, what I read on `one-circuit` at `4cec1783533`, and a signature to agree before I write more than signatures.

**What I read, and what it changes:**
- **`tmpl.emit` discovers the template; it does not know it.** It hashes 6-grams of rows (`gram=6`, `near=64`, `gap=4`, `max_period=2^16`), sorts candidates, takes the longest run of repeats at each level, and then `legalize`s and `pack`s. Its bytes are a function of those heuristics and of the flat rows, not of the Definition. Porting it would put a search into the door. **I propose not to port it.** The Lean writer builds the template directly from the Boolean Definition's own structure: one `DEF` per specialized Definition, and one `CALL` per call, batch or scan, with the copies' inputs as segments. This is O(description), as `tests/test_verifier_cost.py` asks.
- **Lean already writes this form once.** `Flock/ReadNet/Template.lean` builds `EmbeddingRead_v1`'s template from its parts, one `Tmpl.Def` per part (`defOf`), and `render` writes it as text, with no `LEAVES` line. That is the pattern: the writer of the text is about 15 lines, and the real content is Definition → `Tmpl.Def`.
- **Python already has the matching structural writer, as an opt-in:** `struct_lower.py`. Its unroll is NOT `ir_lower`'s flat rows: by its own docstring it adds commits (a form argument past `support` bits) and keeps callee returns that no caller reads. So the three Python forms, `emit` of `ir_lower`'s flat unit, `struct_lower`'s template, and `ir_lower`'s flat unit, are three different byte strings.
- **The door reads the FLAT form today, and only the flat form.** `Spot.OneStageZk.STATEMENT` is `verity/flock-circuit`, from `Tags.find?`, which lists only `all`. The templated statements (`verity/flock-circuit/tmpl`, `+wide-leaf`) are read by name through `isTmpl`. And `HmOut.parseTmpl` refuses a META `definition` ("a proof-block class has no templated statement"), which `compose` writes for a proof block. So no proof block's file can be templated without a statement change. That change is proofs' and soundness's to make, not ours; it goes on TOP-442.

**What I propose, one lowering with one output:**
- REFRESH's rows are a Definition's own rows. The lowering takes each specialized Boolean Definition once and gives a `Flock.Tmpl.Def`. The block is a `Flock.Tmpl.Tmpl`: the Definitions in callee-first order, the root last, and AREAS's port groups.
- The flat unit is that template's unrolled rows: `Tmpl.unroll`, which is already the reference. While the door's statement is flat, `restrict` writes `Tmpl.flatLines`. That is an unrolled walk, which `test_verifier_cost.py` refuses outside tests, so it needs one open `ALLOWED` line until the door's statement is the templated one. On the templated statement, it writes the `TEMPLATE` section instead.
- So REFRESH and I write ONE function, not two that must agree, and the flat-against-templated question becomes a statement choice, not a second lowering.

**The signatures, in the reader's own types** (new file `Spot/BlockTmpl.lean`, in the door's package; no pin watches it. `inner_claims.json` watches `flock/verifier`, so I stay out of that package):
```lean
namespace Spot.BlockTmpl
/-- A reference inside a Boolean Definition: its own gate, its input, or the constant 1. -/
inductive Ref | gate (i : Nat) | input (i : Nat) | one
/-- A Boolean Definition's gate, in C's order; a call places `count` copies of an earlier Definition whose
inputs step by `step` per copy. -/
inductive Gate
  | and (a b : Array Ref) | xor (a : Array Ref)
  | call (callee count : Nat) (args : Array (Ref × Nat))  -- each argument and its step per copy
/-- REFRESH's front half's output, one per specialized Definition, callees first. -/
structure BDef where
  ins : Nat
  gates : Array Gate
  outs : Array Ref
/-- The lowering (REFRESH's rows, my calls and segments). -/
def lower (defs : Array BDef) (root : Nat) (inGroups outGroups : Array Flock.Group) : Flock.V Flock.Tmpl.Tmpl
/-- The text `Flock.Tmpl.parse` reads (`ReadNet.render` moved here, with the `LEAVES` line). -/
def render (t : Flock.Tmpl.Tmpl) : String
end Spot.BlockTmpl
```
- `and`/`xor` take GF(2) sums of references, because a row's A and B are sums. NOT is `xor [a, one]`.
- An argument that is not one reference is committed as a row of its own (`form · 1`) before the call. That is `struct_lower`'s `support = 0`, and it is the counted loss.
- **REFRESH:** if your front half already has a type for the lifted Definition, I take yours and drop `BDef`. One exchange: yours or mine by 12:30 PM PST.

**The round trip, a `#guard` first and a lemma later:**
- `Flock.Tmpl.parse ((render t).splitOn "\n")` gives back `t`, field by field (`Tmpl` has no `BEq`; I compare the fields, as `ReadNet.check` does).
- `Tmpl.unroll` of it, evaluated, equals the Boolean Definition evaluated, on all inputs of a toy: a 4-bit ripple adder called 3 times by a root, with steps. That is a few hundred gates, not the Gemm.
- The `parse ∘ render = id` lemma is feasible but not tonight. It needs `lower`'s output to be `ordered` and `wellFormed`, which is a few hundred lines of proof. It's tomorrow's at the earliest.

**My hours (PST):**
- **The signatures agreed with REFRESH:** today 12:30 PM, on one exchange here.
- **`lower` and `render` elaborating:** today 4:00 PM, with no `sorry` and no axiom, tried by name on `-6`.
- **The round trip passing on the toy:** tomorrow 10:00 AM, on my branch `cursor/lean-block-tmpl-fc8d`, merged to `one-circuit` when it passes.
- **What would bring them in:** REFRESH's type for the lifted Definition by 12:30; AREAS's port groups (`Flock.Group`) as the only thing I take from the composing; and proofs' word on TOP-442 on which statement the door reads.
