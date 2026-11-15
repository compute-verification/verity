# prove_path

What the prove path (`verity/protocols/proofs/spot/prover/audit/`, its README) reads from the measurements, and the
served-rows tie's runs on a sealed served window:

- `sources.json` and the checkpoint manifests it names: the served models' sources, which `prove_path.py` reads.
- The class probes (`llama32_1b_*.json`, `glm53_silu_mul_v2_*.json`), each a Program lowered at a small class, which
  `window_cost.py` projects (`verity/protocols/proofs/spot/prover/cost/`).
- `served_tie_window.py`: the served rows' tie (`served_tie.ServedRowsTie_v2`) on a sealed served window's real rows.
- `served_tie_zk.py`, `served_statics_zk.py`, `served_tie_zk.sh`: the routed served-rows tie, and the served rows held to
  their Programs' statics, proved in zero knowledge on a sealed served window, on a CPU machine.
- `e2e_chains/`: the status of every protocol's end-to-end chain behind the prove path's "What an accepted audit means".
