# Private-circuit runs

Protocol 2 (the private circuit) end to end: the private circuit's description is a value of the public universal unit
`UniversalUnit_v1{G, N_IN, N_OUT}` (`verity.ml.private_circuits.universal`), proved by C-Flock's circuit prover under `--zk` and
verified by the Lean verifier.

- `universal.py`: the driver; its docstring says what it builds, what it checks and what a row records.
- `nonzero_cost.py`: option 4 (pure recursion), V\* reading a unit's matrices as data: the ANDs per nonzero of the circuit
  fold and of the registration read, measured on the repository's gadgets, with the nonzeros of real units' lowerings. It
  runs in a few minutes on one core (`python benchmarks/private_circuit/nonzero_cost.py --out DIR`).
- `pod.sh`: the entry point on a CPU node. It builds `flock-circuit` and `flock-verify` once per `FLOCK_WORK`, from this
  checkout:

~~~sh
research run --on vy-nebius-1 --cwd source --project verity --declared-output 'out/*' -- \
  bash benchmarks/private_circuit/pod.sh CLS="16,16,8 64,64,32" N=64 CPUS=128-159
~~~

`WIRED=1` runs each class as the wired circuit of four units (`universal.py --wired`) after the one-unit statement at the
same instance count, with its cut wires linked (PROTOCOL.md §16.13) and two cheating provers both verifiers must refuse.
Linking binds a public topology; it does not hide the topology: the program descriptor is public in this run, which leaks
the circuit's wiring beyond the size class.

`check` runs nothing here.
