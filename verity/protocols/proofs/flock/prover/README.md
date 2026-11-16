# Flock's prover (`verity-flock`)

Flock is Verity's proof system, and this is its prover: the code that makes the proofs [Flock's
verifier](../verifier/README.md) checks. It proves a committed relation, the commitment's hash leaves included, in one
binary-field proof, on upstream Flock (`succinctlabs/flock` at `b684b12`, patched here).

It has two halves. The Python package `verity_flock` turns a circuit into Flock statements: it lowers each proof block
to a binary-field circuit and stages the statement. The Rust crate `flock-live` (in `live/`) is dropped into an
upstream checkout on a GPU pod, and proves: its `flock-circuit` is the circuit prover, and `flock-circuit prove --zk` is
Flock with zero knowledge. Under `--zk` with `FC_ZK_SERVER=1` it is the workers' side of the ZK server, whose Lean is in
the verifier. [`live/PROTOCOL.md`](live/PROTOCOL.md) is the protocol the two sides speak.

Python returns no accept of its own. Every question a verifier answers is the Lean verifier's, and a Python path that
once answered one is the difftest's reference or raises `NotImplementedError` naming its goal.

## Try it

~~~sh
uv run tools/check/suites.py verity-flock --quick
~~~

This runs the prover's tests without the slow ones, on the CPU. The proofs themselves run on GPU pods, through
`research run` (`pod/`).

[`DESIGN.md`](DESIGN.md) has the package's layout, the hill-climb sessions on GPU, the table of which Lean answers
each question, and the full test commands.
