# network_e2e

The network chain's steps after N2 and the proofs, over varied traffic, held to their contracts on fakes and on recorded
endpoint runs. Nothing here serves or proves: the network certifier's runner is
`benchmarks/network_traces/chain_record.py`, whose endpoint step reads `chain.served_advice`.

- `chain.py`: the steps. The advice (I1 v0) read in each of its forms; its bits per link-window, v0's beside the derived
  rules' residual (`encodings/derived.py`); the endpoint's own advice (`served_advice`: `verity-vllm advice` over a
  `verity-vllm serve-openai` endpoint's records, `chain.ENDPOINT`, bound to a run's certified I/O); and a
  `prove_path.py audit` run's output adopted and held by content to that advice (`prove_path`, `held_to`). Its docstring
  has each step's contract.
- `encodings/`: the advice's encodings, measured.
- `traces/`: the vLLM scheduler traces they are measured on (`traces/README.md`).
- `tests/`: the steps on fakes and on recorded endpoint runs (`tests/data`, `test_chain.py`), the prove path's adoption
  held by content (`test_e2e_serve_run_adoption.py`), and the stop rule's Definitions held to the derived rule
  (`test_stop_rule_definition.py`).
