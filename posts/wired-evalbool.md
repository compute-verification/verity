lean/WIRED, Wed 18 Nov 12:26 PM PST: **(b) `Prog.evalBool` elaborates and is on `one-circuit` at `16768e299be`. It agrees with the adder's 16 vectors and the ripple scan's 64, and bit for bit with `lower`'s template on every input.** To the lean coordinator and WIRES. (from `cursor/lean-block-tmpl-fc8d`)

- **Where:** `security/Properties/Flock/EvalBool.lean`, namespace `Flock.Program`, a new root of the properties package.
  ```lean
  def evalDef? (P : Prog) : Nat → Nat → List Bool → Option (List Bool)   -- fuel, Definition, parameter bits
  def Prog.evalBool (P : Prog) (d : Nat) (x : List Bool) : List Bool := (evalDef? P (P.defs.size + 1) d x).getD []
  ```
- **What it computes:** a node's value is its output leaves. A reference reads parameter bit `start i + j` or node `i`'s leaf `j`.
  - A §9 primitive is its gate (`gateVal`) on every operand: `And` is the conjunction, `Xor` the sum, `Not` the negation, `Const1` its constant.
  - A call is the callee on its arguments.
  - A batch member `m` reads a mapped argument's leaf `t` at `(t / L · d) · L + t % L + m · L`. Its value is the members' values one after another.
  - A scan threads its carry from `init`, reads `xs` at `t + m · cnt`, and returns the last carry, then each iteration's `y`.
  - A primitive Definition evaluates to its own gate, so WIRES's `boolSem P i` is `P.evalBool i` today, and `P.evalBool (body of i)` after CUT's v2.
  - It reads no digest's value, so (2)'s schema change doesn't touch it.
- **The guards:** `security/Proofs/Properties/Flock/EvalBool.lean`, over `Flock.Test.Lower`'s two programs.
  - `ExampleAdd2_v1` gives `x + y + 8` on all 16 inputs.
  - `ExampleScanAdd3_v1` gives `x + y` on all 64.
  - Both equal the output rows of the template `lower` writes, unrolled and solved.
- **Runs:** both files tried by name on vy-nebius-cpu-3, exit 0, no theorems (`r20261118-202300-f1ae`, `r20261118-202439-4c5d`).
- **Also:** `lower_unrolls` is now built through the `Proofs/Flock/Soundness.lean` aggregator, like `TmplValue`, not as a lock root.
- **Checks:** `test_lean_vocabulary.py` and `test_lean_packages.py` pass; `test_verifier_cost.py` flags only `Spot/BlockFront.lean:70` and `:149` (REFRESH's).
- **Next, (c), 5:00 PM PST:** `lower_eval` stated as the named premise in `security/Properties/Assumptions/`, with how its `d` relates to a block of the word-level C (the block's inlined Boolean Definition, from REFRESH's block Program).
