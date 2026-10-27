# PoUW (`verity.core.protocols.pouw`)

Proof of useful work: certifies that a served model's declared matmuls were computed, by sampled proofs over committed,
noised checked values. One protocol carries several named schemes behind one interface. The proof service registers
an audit window, draws its proof blocks with live coins, proves and verifies them; PoUW states what each window
commits, the Program each drawn proof block must pass, and what the outcome credits.

The verifier of record is Lean: `lean/check` (`pouw-check`), `lean/commit` (`pouw-commit`) and `lean/window`
(`pouw-window`), whose verdicts `benchmarks/pouw/served_zk/of_record.py` publishes. The Python modules of the same names
are their references, held to them by the difftests.

## Where things are

- `PROTOCOL.md`: the spec while it is experimental: the interface, the lifecycle, epoch coins, registered weights, the
  proof's identifier, the served audit window, each scheme's section, and the compile pass.
- Dither, PoUW's no-fold protocol under the ACCEL model, has its own `PROTOCOL.md` in its directory beside this one.
- `schemes/`: the named schemes (`ncp-v1`, `pearl-fp8-v4`, the Pearl-C family, `pearl-c-nvfp4-v0`).
- `circuit/`: PoUW as `verity.ml.circuits` Programs: the tile and lane Definitions, the compile pass over a served
  Program (`compile`), and the draw plan (`plan`). Each Definition has its `circuit-check` sidecar.
- `lean/check`, `lean/commit`, `lean/window`: the Lean verifier of record, each with its README and difftest.
- `tests/`: the package's suite, with pinned vectors under `tests/vectors`.
- `benchmarks/pouw/`: the measuring side and the clear reference lifecycle; nothing here imports it.

## Core API

~~~python
from verity.core.protocols.pouw import schemes, audit, coins, core_check, credit, salt, window, verifier
from verity.core.protocols.pouw.protocol import PoUWScheme

scheme = schemes.get("pearl-c-sm120-v1")      # a configured PoUWScheme
root = audit.weights_root(weight_roots)        # the weights' root a period's open signs
verdict = core_check.check(params, commitment, evidence)   # one audit window, as the Lean `pouw-check` answers
credited = credit.window(...)                  # a served window's credited set and its drawn proof blocks' outcome
period = credit.period_credit(...)             # the period a window counts toward, closed and complete
~~~

- `protocol`: the `PoUWScheme` interface and its data.
- `audit`: the commitments, the epoch and its salt, and the tile order, the same for every scheme.
- `coins`: the challenger's live epoch coins, committed before the epoch and each opened once.
- `identifier`: the proof's identifier (the scheme, salt and workload a proof is for).
- `window`: what an audit window makes public, its grid and its draw size.
- `salt`: PoUW's read of the network certifier's per-window salt.
- `registration`, `weights`, `operands`: registered weights and a window's committed operand rows.
- `verifier`: the verifier of record's entry points and its coins.
