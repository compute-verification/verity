# Private-circuit runs

Protocol 2 (the private circuit) end to end: the private circuit's description is a value of the public universal unit
`UniversalUnit_v1{G, N_IN, N_OUT}` (`verity.ml.private_circuits.universal`), proved by C-Flock's circuit prover under `--zk` and
verified by the Lean verifier.

- **Protocol:** protocol 2, the private circuit. It has no package under `verity/core/protocols/` yet. Its unit is proved
  and verified like sampled proofs' units are, so `example.toml` names `sampled_proofs`.
- **Workload:** none. Each circuit is a random hidden netlist, padded to its size class (`CLS`).
- **Placement:** one CPU node holds both parties, the prover (`flock-circuit prove --zk`) and the Lean verifier
  (`flock-verify`): `pod.sh` on vy-nebius-1's CPUs.
- **Command:** `pod.sh`, below.
- **Owner:** proofs. Its rung on the ladder is `example.toml`.

- `universal.py`: the driver; its docstring says what it builds, what it checks and what a row records.
- `pod.sh`: the entry point on a CPU node. It builds `flock-circuit` and `flock-verify` once per `FLOCK_WORK`, from this
  checkout:

~~~sh
research run --on vy-nebius-1 --cwd source --project verity --declared-output 'out/*' -- \
  bash examples/private_circuit/pod.sh CLS="16,16,8 64,64,32" N=64 CPUS=128-159
~~~

`WIRED=1` runs each class as the wired circuit of four units (`universal.py --wired`) after the one-unit statement at the
same instance count, with its cut wires linked (PROTOCOL.md §16.13) and two cheating provers both verifiers must refuse.
Linking binds a public topology; it does not hide the topology: the program descriptor is public in this run, which leaks
the circuit's wiring beyond the size class.

`check` runs nothing here.
