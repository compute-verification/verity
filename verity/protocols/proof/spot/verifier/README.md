# spot/verifier

SPOT's verifier side: the docket's checks, the sample on the challenger's coins, the one-stage audit and its integrity
profile, the challenger service, and the Lean verifier of record (`lean/`). The proof layer's README (`../../README.md`)
is the architecture; each rule, format and refusal is in its module's docstring and the Lean that decides it:

- the docket: `one_stage/docket.py`, `Spot.Docket.check` (R1 to R8, stated by `Properties.Spot.R1` to `R8`);
- registered values: `one_stage/registered.py`, `Spot.Registered`;
- the sample and its designs: `one_stage/draw.py`, `Flock.Draw`; the levels: `levels.py`, `Spot.Levels`;
- the audit record and the profile: `one_stage/audit.py`, `profile.py`, `Spot.Record`;
- the stream and the fleet: `service.py`, `lean/Spot/Stream.lean`, `lean/Spot/Fleet.lean`;
- the coins: `challenger/coins.py`, `lean/Spot/Challenger.lean`; the parties: `parties.py`; readers: `consumers.py`.

The challenger's premises: `ChallengerLogsCoins`, `FlipsAfterRecorded` and `VerifierReadsLog`
(`security/Definitions/Core/Assumptions/Challenger.lean`), `CoinsConfidential` (`Pouw.Assumptions.CoinsConfidential`),
and `OneClock`, which no Lean states yet: the challenger's release times and the network attester's ingress and receipt
times are read on one clock, the auditor device's; a challenger on a device of its own keeps its clock within 1.03 s of
the network attester's.

~~~sh
uv run pytest -q tests/verity/protocols/proof/spot/verifier/test_one_stage.py
~~~
