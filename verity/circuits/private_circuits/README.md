# verity.circuits.private_circuits

Sometimes the circuit itself is the secret: its gates and wiring are a value, like the data it runs on. The universal
unit, `UniversalUnit_v1{G, N_IN, N_OUT}`, is one public Boolean Definition that runs any private circuit of its size
class (G gates over N_IN inputs, N_OUT outputs). It takes the private circuit's description as an input beside the
data, so its own structure depends on the size class alone.

It is not how a private circuit is hidden: circuit hiding is recursion (Flock's `--hide`).

## Try one

~~~python
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.private_circuits.universal import PrivateCircuit, UniversalUnit, arguments, interpret

half_adder = PrivateCircuit(n_in=2, gates=((0, 0, 1), (1, 0, 1)), outputs=(2, 3))   # a XOR b, a AND b
interpret(half_adder, [1, 1])                                  # [0, 1], run gate by gate
unit = bind(UniversalUnit, G=2, N_IN=2, N_OUT=2)               # the public Definition for its size class
evaluate(unit, *arguments(half_adder, [1, 1]))                 # (0, 1), the same outputs
~~~

A gate is `(op, a, b)`, with op 0 to 3 for XOR, AND, XNOR and NAND of two earlier wires. The module's docstring gives
the description's form, the design and its cost, about 2 N_IN + G ANDs per gate of the private circuit.

The tests are in `tests/verity/circuits/private_circuits/`.
