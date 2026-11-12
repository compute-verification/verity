# redteam

Attempts to get out of the system what its protocols exist to deny. Training is the first, and the narrowest: a
training run hidden in a datacenter held to the inference-only policy.

## Training

The inference-only policy is worth what it costs the cheapest training run that hides under it. The red team looks for
that run, and each attempt is one point: how good a model it trains, and what it costs under the policy.

**A task** is fixed before any submission: starting weights θ, training data, a held-out set, how a bit string is read
as a model, and the reference training algorithm, the best known way to train with no protocol.

**A submission** is two things:

1. **A training circuit**: a Program, the circuit language's format (`verity/circuits/PROTOCOL.md`). Its inputs are θ
   and the training data, and its output is a bit string. A loop is a `scan` node, so a long run is one loop body's
   description and a count.
2. **Its output**: the bit string the circuit outputs on the task's inputs.

**What we compute from it:**

1. **Loss**: the output read as a model, on the held-out set.
2. **Slowdown**: the policy admits only circuits cut into blocks of at most c gates that each read at most k bits of
   fresh input. The slowdown is a lower bound on what any such cut of the training circuit costs, its blocks times c,
   over the circuit's size. The slowdown calculator reads it off the Program, a loop's body once, under two named
   conjectures (`Hardness` and `Exports`, `security/Definitions/Assumptions.lean`).
3. **A point**: loss against cost, the circuit's size in AND gates times its slowdown, beside the reference training
   algorithm's curve under no policy. Cost is also given as a share of one datacenter's and of the world's compute
   over a week, a month and a year.

The plot of the points is the result. At a target loss, the cheapest point's cost over the reference curve's is the
slowdown that survives the red team. A submission wins by lowering it, or with a set of its gates that refutes a
conjecture.

### Making a submission

Nobody writes a Program by hand, and nobody evaluates one at a task's size. A red-teamer writes training in PyTorch,
as a loop, and:

1. **prices the loop body first**: the slowdown reads structure and shapes, never weights or data, so it takes no
   training run and no GPU;
2. **trains what prices well**, in PyTorch, scoring it on the search pool, the held-out set's open part;
3. **submits what is worth a point**: the loop compiled to a Program, with the trained weights as its output.

### What is trusted

A red-teamer is trusted to submit the circuit they ran. Nothing checks that a submission's output is its circuit's, or
that its Program computes what its PyTorch did. One thing is kept from them: the final set, the held-out set's closed
part, which only a submission's loss reads.

### Target, not yet

The code is `benchmarks/training_slowdown/`, whose README has its commands. It differs from this page in four ways:

- It takes PyTorch under a contract of its own (a route and a config), counted and sandboxed as an adversary's, where
  this page takes a circuit and its output.
- It prices a trace of the PyTorch loop body, each op at its Definition's gate count, where this page prices a Program.
- The compiler from a PyTorch loop body to a Program (`adapters/vllm/verity_vllm/program/frontend/train_step.py`, in
  the vLLM adapter until there is a torch adapter) takes one optimizer, AdamW.
- It grades one submission at a time and makes no plot.
