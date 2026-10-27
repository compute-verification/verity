# toy_network

The toy (`integrations/torch_loop`, a W8A8 MLP on CPU) served and audited by sampled proofs, with the network
certifier to join. `config.toml` is the example's one Verity artifact: the subprotocols it runs (`protocols`), the
workload's shape, the proof service's selection (`[service] law`), each role's machine and principal (`[[party]]`), and
the channels between them with their ordering (`[[channel]]`'s `after`; `private` for a channel that carries the
prover's witness). It holds no ε or δ, because what a run's draw establishes is in its verdict. `verity config new |
check | hash` writes, checks and hashes a configuration, and `verity run CONFIG --party ROLE` runs a role (`--local`:
every role, on this machine).

`protocols` lists only `sampled-proofs` until the network certifier's role exists. A configuration that lists
`certifier` before then gets `"certifier": "absent"` in its verdict line, and the product rejects.

## Sampled proofs, from a fresh clone

`config.toml` runs the toy's integrity by sampled proofs. Every role runs on CPU.

~~~text
git clone https://github.com/compute-verification/verity && cd verity && git fetch origin main
uv sync --all-packages --extra torch-cpu
verity run examples/toy_network/config.toml --local
~~~

Until `verity run` is in the tree, the parties' own entry point runs the same thing:

~~~text
PYTHONPATH=benchmarks/all_protocols:integrations/torch_loop uv run python -m verity_all_protocols.parties \
    examples/toy_network/config.toml --state STATE --local
~~~

The parties are sampled proofs' own (`verity/core/protocols/sampled_proofs/parties.py`): next-message machines on the
deployment layer (`verity.core.deploy`), their messages the wire's, each run's transcripts and record under
`STATE/runs/`. What a machine does for them is a capability it gives them, and on the toy that is
`benchmarks/all_protocols/verity_all_protocols/parties.py`'s. The first run builds what they run into `STATE/build`
(`benchmarks/all_protocols/build.sh`: C-Flock's `flock-circuit`, the Lean `flock-verify`, `flock-firewall`,
`sampled-proofs-verifier` and `product`), and `--tools FILE` reuses a build. On a 192-core node the build took 82 s
and a block about 3 minutes (r20261027-003126-da4c).

1. **challenger** (the auditor's). Its host, `benchmarks/one_stage/challenger_party.py`: a development challenger
   host under `STATE/challenger/host`, with its own LMS key, signed ledger and pins log, pinned on main's first-parent
   history (`STATE/main.txt`, from `origin/main`). The challenger tells the verifier where its host listens
   (`Challenger {address, pin}`) and stops it on the verifier's `Done`. The host opens one session per sampling
   block's root, never two at once and never a root twice, and hands the verifier only the session's public record.
2. **firewall** (the prover's). The Lean `flock-firewall` every drawn member's outer proof goes through, one at a time.
   It tells the verifier the binary's SHA-512 (`Firewall`), and the verifier's audit refuses any other.
3. **workers** (the prover's). The toy serves each sampling block into `STATE/blocks/w<i>/`: the verifier registers
   the circuit from the loop's shape first, then the loop writes its window, root and values. The rows and salts stay
   there; the block's public files are in `STATE/public/w<i>/`. The workers send the verifier each block's
   registration (`Registration`).
4. **verifier** (the auditor's). For each block, once its registration is in, the one-stage audit
   (`benchmarks/prove_path/records_audit.py`, the Lean verifier of record) draws on the challenger's coins. C-Flock
   `--zk` proves the drawn members behind the firewall, and an accepting audit must be of the registration the workers
   sent. The audit writes its answer and case to `STATE/record/w<i>/sampled-proofs/`. Then `Core.Composition.product`
   runs over exactly the configuration's `protocols`, and the verifier prints the block's line (also in
   `STATE/record/verdicts.jsonl`):

   ~~~text
   VERDICT {"block": 0, "coins": "simulated", "components": {"sampled-proofs": "accept"}, "product": "verity/product/v0", "reason": null, "registration": "…", "run": "r20261027-003126-da4c", "verdict": "accept"}
   ~~~

   A listed component with no answer for the block is `absent`, and the product rejects. For example, a configuration
   with `protocols = ["sampled-proofs", "certifier"]` gets `"certifier": "absent"` until the network certifier's
   party runs. With every party on one machine (`--local`), the challenger's host is the prover's, so the coins are
   `simulated`.

`--party ROLE` runs only that party, and its peers meet it over TCP, each channel's handshake signed with a party key
made for the run. The challenger, the firewall and the workers each listen for the verifier at their `--at
ROLE=HOST:PORT`, and the verifier, started once they listen, dials them; every process is given the same `--at`
(`benchmarks/all_protocols/parties.sh`'s `apart` case). `--tamper 0` (the entry point only) makes block 0's workers
record outputs that are not the model's, and the verifier rejects it.

Until the audit's draw, the witnesses and the proofs are messages of their own (the configuration's `draw`, `witness`
and `proof` channels), the audit runs the prover's sessions itself, so the verifier and the workers share the machine
the block's rows are on. The capabilities stay in `benchmarks/`, because `verity.core` may import none of them: the toy
loop (`runner.loop_window`), the records audit (`records_audit.py`, `legs/sampled_proofs.py`), the development
challenger host (`challenger_party.py`, on `a5_coins.Local`) and the Lean product's runner (`compose.onsite_check`).
