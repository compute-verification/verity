# Protocols (prototype, not for merge)

Three minimal protocols that compose over one Program, to let the protocol interface emerge from code:

- `sampled_proofs/` (`verity_sampled_proofs`): commit a transcript, sample replay units and then verification units
  from beacon rounds after each commitment, check sampled units by exact replay, emit an `IntegrityAllowance`.
- `pouw/` (`verity_pouw`): device-bound noise seeds, the `NoisedDot` Definition, lottery tickets over noised
  outputs, and a work claim that charges the allowance's worst case.
- `pous/` (`verity_pous`): design C with auditor-secret filler, the timed linear-MAC audit and a CPU reference
  challenger on a simulated clock.

Protocols import only the core (`tests/test_protocol_boundaries.py`). The application composes them; the worked
composition is `example.py`:

~~~text
uv run python protocols/example.py
uv run pytest protocols
~~~

Shared core pieces added for them, each because two protocols use it: `verity.randomness.derive` (all three) and
`verity.resource_accounting.IntegrityAllowance` (produced by sampled proofs, consumed by PoUW).
