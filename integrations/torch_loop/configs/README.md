# The toy's configurations

A configuration is the one Verity artifact a run needs: the subprotocols it runs (`protocols`), the workload's shape, the
proof service's selection (`[service] law`), each role's machine and principal (`[[party]]`), and the channels between
them with their ordering (`[[channel]]`'s `after`; `private` for a channel that carries the prover's witness). It holds
no ε or δ: what a run's draw establishes is in its verdict. `verity config new | check | hash` writes, checks and hashes
one, and `verity run CONFIG --party ROLE` runs a role (`--local`: every role, on this machine).

## Sampled proofs, from a fresh clone

`sampled-proofs.toml` runs the toy's integrity by sampled proofs. Every role runs on CPU.

~~~text
git clone https://github.com/compute-verification/verity && cd verity && git fetch origin main
uv sync --all-packages --extra torch-cpu
verity run integrations/torch_loop/configs/sampled-proofs.toml --local
~~~

Until `verity run` is in the tree, the roles' own entry point runs the same thing:

~~~text
PYTHONPATH=benchmarks/all_protocols uv run python -m verity_all_protocols.parties \
    integrations/torch_loop/configs/sampled-proofs.toml --state STATE --local
~~~

The first run builds what the roles run into `STATE/build` (`benchmarks/all_protocols/build.sh`: C-Flock's
`flock-circuit`, the Lean `flock-verify`, `flock-firewall`, `sampled-proofs-verifier` and `product`), about half an hour
on a 16-core CPU. `--tools FILE` reuses a build. The roles run in this order, and each can run alone with `--party ROLE`
on the same `STATE`:

1. **challenger** (the auditor's). A process of its own, `benchmarks/one_stage/challenger_party.py`: a development
   challenger host under `STATE/challenger/host`, with its own LMS key, signed ledger and pins log, pinned on main's
   first-parent history (`STATE/main.txt`, from `origin/main`). It prints `CHALLENGER {"address", "pin"}` and writes
   `STATE/challenger/address.json` once it listens. It opens one session per sampling block's root, never two at once
   and never a root twice, and hands the verifier only the session's public record.
2. **firewall** (the prover's). The Lean `flock-firewall` every drawn member's outer proof goes through, one at a time.
   It pins the binary in `STATE/firewall/ready.json`.
3. **workers** (the prover's). The toy serves each sampling block into `STATE/blocks/w<i>/`: the verifier registers
   the circuit from the loop's shape first, then the loop writes its window, root and values. The rows and salts stay
   there; the block's public files are in `STATE/public/w<i>/`.
4. **verifier** (the auditor's). For each block, the one-stage audit (`benchmarks/prove_path/records_audit.py`, the
   Lean verifier of record) draws on the challenger's coins once the registration is in. C-Flock `--zk` proves the
   drawn members behind the firewall. The audit writes its answer and case to `STATE/record/w<i>/sampled-proofs/`.
   Then `Core.Composition.product` runs over exactly the configuration's `protocols`, and the verifier prints the
   block's line (also in `STATE/record/verdicts.jsonl`):

   ~~~text
   VERDICT {"block": 0, "coins": "simulated", "components": {"sampled-proofs": "accept"}, "product": "verity/product/v0", "reason": null, "run": "…", "verdict": "accept"}
   ~~~

   A listed component whose role wrote no answer for the block is `absent`, and the product rejects. For example, a
   configuration with `protocols = ["sampled-proofs", "certifier"]` gets `"certifier": "absent"` until the network
   certifier's role runs. With every role on one machine (`--local`), the challenger's host is the prover's, so the
   coins are `simulated`.

`--tamper 0` (the roles' entry point only) makes block 0's workers record outputs that are not the model's, and the
verifier rejects it. In step 0 the roles share `STATE` on one machine. Running them on separate machines, with their
channels on the wire, comes with the roles as deploy-core `Party`s.
