# Private-circuit runs

A private circuit can be proved without showing it by making its description a value: the public universal unit
`UniversalUnit_v1{G, N_IN, N_OUT}` (`verity.circuits.private_circuits.universal`) reads a circuit of its size class as
input and runs it. This benchmark runs a random hidden circuit that way end to end, under Flock's circuit prover with
`--zk` and the Lean verifier, and records what the unit costs per hidden gate. The universal unit is not the path to
hiding a circuit: circuit hiding is recursion's (`--hide`, the recursive session's private-circuit form:
`Flock.RecVStar`, `Flock.RecCompose`). These runs measure the unit for comparison.

## Try it

Stage the smallest class, 16 gates, 16 inputs and 8 outputs, on eight instances, without proving:

~~~sh
uv run python benchmarks/private_circuit/universal.py --cls 16,16,8 --n 8 --stage-only --out /tmp/private_circuit
~~~

It checks that the hidden circuit, the unit running it and the staged transcript agree on every instance, and prints the
benchmark result (the unit's ANDs, ANDs per hidden gate, the statement's size), also written to
`/tmp/private_circuit/rows.jsonl`. It takes about ten seconds.

The proved runs build Flock's circuit prover and the Lean verifier on a CPU node (`pod.sh` lists its settings, among
them `WIRED=1` for the wired circuit of four units):

~~~sh
research run --on vy-nebius-1 --cwd source --project verity --declared-output 'out/*' -- \
  bash benchmarks/private_circuit/pod.sh CLS="16,16,8 64,64,32" N=64 CPUS=128-159
~~~

`check` runs nothing here.
