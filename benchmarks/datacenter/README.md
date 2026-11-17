# datacenter

A datacenter as Verity accounts for it, in a small Python API: a Program, the hardware it runs on, the binding between
them, the verifier's plan, a run with what the trusted devices record, the audit's claims, and the account. Think of it
in two layers. The logical layer is where the cryptography happens: the Program, the commitments to its registered
inputs and its values, and the claims the audit establishes. The physical layer reads those claims against registered
hardware, named assumptions and the network attesters and memory attesters, and turns them into credit, the covert
budget and the export.

It is a model, not a protocol: nothing under `verity/` imports it, and it is not a member of the uv workspace, so an
importer puts this directory on `sys.path`.

## Try it

~~~sh
cd benchmarks/datacenter
uv run python -m datacenter.demo
~~~

It needs the workspace (`uv sync --all-packages`) and no GPU, and takes seconds. It runs a night on a two-node rack,
honestly and then with an adversary, and prints each account: each block's ANDs and reads, the covert budget, the
export and where each store was held, then what the audit refused in the adversary's night.

## In Python

~~~python
from datacenter import *
from datacenter.examples import rack
from datacenter.models import WEIGHTS, X, two_stage

program = two_stage()                                  # x -> h = x @ W1^T -> y = h @ W2^T, the weights registered
p = place(program, rack(nodes=2), "n0.gpu0", registration=register(program, WEIGHTS), deliver="world", x="world",
          h="n0.gpu0", y="n1.gpu0")
plan = Plan(Interval(0, 16))
profile = p.public(plan)
print(account(audit(run(p, {"x": X}, plan), profile), profile))
~~~

`python -m datacenter.demo` runs a night on a two-node rack, honestly and then with an adversary, and
`PYTHONPATH=. python ../../tests/benchmarks/datacenter/test_walkthrough.py` prints the walkthrough ("Accounting for a datacenter, from a circuit up") section by
section.
