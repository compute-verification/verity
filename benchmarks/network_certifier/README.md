# Network certifier benchmarks

The network certifier's benchmarks in one directory (TOP-103 item 4), each subdirectory with its own suite and runner.
network-accounting's.

- `e2e/`: network accounting end to end on a model vLLM serves: a certifier-serve rehearsal's runs through N2, the head
  checks proved behind the Lean firewall, and the negatives, in one report (`bash benchmarks/network_certifier/e2e/run.sh`,
  its README has the commands).
- `sim/`: a mock datacenter of Linux network namespaces on one host, each block behind the real certifier, its records
  audited, on the workload it serves (`python3 benchmarks/network_certifier/sim/run.py --out DIR` on the lab host).

The certifier's rehearsals, scans and proofs of N2 are still `benchmarks/network_traces/`, with `chain_record.py`: it is the
challenger host's pinned tree (`tools/research/src/research/pods/nebius/challenger/pin.sh`), and the auditor runs the
certifier, warden and load services from it. It joins here as `traces/` once the challenger is pinned from the new path.
The accounting layer's model of a datacenter, which `sim/` reads, is network's example, `examples/network/`.
