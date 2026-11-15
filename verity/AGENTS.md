# verity/

The distribution `verity` (the root `pyproject.toml`; its module list is `verity/README.md`). The root `AGENTS.md`
holds everywhere; this file adds what holds only here. Import from the owning module
(`from verity.circuits import Program`); `verity` re-exports nothing.

## Layout

- `party.py`, `channel.py`, `device.py`, `protocol.py`, `config.py`: what a party, a channel (the messages every party
  speaks), the registered hardware (the readers of `hardware/`), a protocol and a deployment configuration are.
  `runtime/` runs parties over channels and imports no protocol; no protocol or adapter imports it.
- `circuits/`: the circuit language (format in `circuits/PROTOCOL.md`) and the circuits written in it: `ops` (the
  primitive circuits' Definitions) and `silicon` (the GPUs' exact semantics), which are the catalog, `kernels`,
  `private_circuits` (the universal circuit), and the evaluator and a circuit's requirement in Lean (`evaluate`,
  `requirement`). The served kernels' circuits are the vLLM adapter's (`adapters/vllm/verity_vllm/definitions/`).
- `commitments/`: the commitment schemes, with `crypto`, `randomness` (the one challenge derivation), `lms` and
  `identity` (Lean), `attest` (the auditor's hardware's signed period roots), `window.py` (what an adapter hands the
  prover for each execution phase) and the security claims' ids (`claim_ids.py`).
- Three base Lean packages hold the shared Lean, which Flock's verifier and the packages requiring it load:
  `circuits/base/` (bytes, hashes and parsing), `circuits/lean/` (the circuit format) and `commitments/lean/`
  (frame-v3 and the sampling of units).

## The protocols, by layer

The directories `accounting/` and `policy/` move to `resource_accounting/` and `workload_classification/`; the paths
below are today's.

- **Proofs** (`protocols/proofs/`), what the other protocols call to prove; it uses neither layer above it. `flock/` is
  the base argument system: its Lean verifier and the ZK server (`flock/verifier/`, `Flock.ZkServer`), its prover
  (`flock/prover/`: `verity_flock`, its own distribution) and the agreement with upstream's verifier
  (`flock/prover/agreement/`). `spot/verifier/` is SPOT, sampling, with the integrity profile (`profile`).
  `recursion/` is the recursion layer. `spot_flock/verifier/lean/` is SPOT[Flock], the argument SPOT builds over Flock,
  and `spot_flock_recursion/` composes it with the recursion, the recursive audit.
- **Resource accounting** (`protocols/accounting/`): `pouw`, `pous`, `poub` and `certifier` (compute, memory and
  network) and `registration` (the registration log, checked once, before any challenge; its Lean is still in
  `interface/`).
- **Workload classification** (`protocols/policy/`): `inference_only`, on resource accounting. Its check
  (`inference_only/verifier/lean/`) is Lean only; its Python is its manifest alone
  (`inference_only/verifier/parties.py`: the verifier, no channel).
- **Deployment patterns** (`protocols/compose/`), on top of the audits. Its `verifier/` holds its Lean (`lean/`) and
  the two lists of protocols it accepts: `remote/`, what an auditor with no hardware of its own at the datacenter runs,
  and `onsite/`, what an auditor with its hardware there runs; both are Lean only. Its `device/` holds the auditor
  host's challenger service (`device/live_challenger/`).
- `interface/` is the Lean interface each protocol's check implements. `experimental/NAME/` is a whole protocol no
  configuration lists; it keeps a `PROTOCOL.md` while experimental.

Inside a protocol, `verifier/` is trusted code (a Python verifier until its Lean verifier, its `lean/`, replaces it),
`prover/` is what its prover builds and its verifier doesn't read, and `device/` is the auditor's own hardware (the
network certifier, the memory challenger, the challenger). Four directories are distributions of their own, kept apart
from `verity`'s places: `pouw/prover/clear/`, `pouw/verifier/receipts/`, `poub/device/live_verifier/` and
`poub/verifier/remote_session/` (a place of its own).

## The import listing

The root `AGENTS.md` states the rule; this is how `tests/import_order.toml` applies it. Its order, its distributions and
each place's paths are kept by hand, and its tables are the tree's own, written by `uv run tools/check/import_order.py`.

- A file loads only what its own row of the listing or an earlier row holds, in Python and in Lean, but for the ties the
  listing names by file (`against`, which only shrinks). `infra/` is the last row, so nothing outside it loads it but a
  listed tie.
- `verity`'s own Python is in places, each the directories and files its `place` entry names, and a place's Python
  imports only its own place and the places its `loads` lists. Python of `verity` in no place only groups packages and
  imports the standard library alone.
- A trusted place is one under a `verifier/` or a `device/`, or one the listing marks `trusted` (the wire, the
  deployment layer, the Site, the record, the Lean of the verifiers' side with its tests); the rest is the provers' side
  and the Python that mirrors the Lean for speed. Until its protocol's verifier is Lean, a trusted place loads and
  reaches only the places the listing froze for it, and from outside the trusted places only the files its `loaded_by`
  names import one, a test included; these lists only shrink, and the generator writes a line into one only under
  `--grow`. The soundness path's places (the root `AGENTS.md`) hold a few files outside their directories too, which
  their `paths` name. The rest can't affect soundness.
- A directory under `verity/` with an import root of its own is a distribution the listing names with its owner:
  `verity` loads none of them by name (a load by file is read only into `benchmarks/`) and imports no other
  distribution of the workspace.
- `benchmarks/` code is imported by no path put on `sys.path` or loaded by file either (`BENCHMARKS_IMPORTERS` names
  today's importers).
- Target, not yet enforced: `circuits/` imports nothing and `commitments/` only `circuits/`; no code imports
  `security/` or the hardware data; inside `protocols/`, flock is at the bottom, spot and recursion on it, spot_flock
  and the audits on those, and compose on top; no protocol imports a configuration.
