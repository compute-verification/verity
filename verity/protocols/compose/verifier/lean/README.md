# core_composition: the onsite and remote protocols as compositions of subprotocols

An auditor wants one verdict, not one per protocol. Compose's two configurations, onsite and remote, each take a list
of subprotocols, run every one's check on its own case, and give one verdict over all of them. Any rejection rejects,
even when another subprotocol is undecided: otherwise a prover could hide a rejection by leaving another subprotocol
undecided. The subprotocols that read the run's record must all commit to the same one.

The rule is written once in Lean, over any list of subprotocols, and the two composites are values of it that differ in
their id and their subprotocols. Each composite's executable is the core command line (`onsite id | check | serve`,
`remote id | check | serve`).

Their security argument, layer by layer from one proof to the training slowdown, is the remote protocol's README
(`verity/protocols/compose/verifier/remote/README.md`, "The security argument, end to end"); onsite shares its layers and adds its own subprotocols'.

## Try it

~~~sh
uv run tools/check/suites.py verity-core-composition
~~~

This builds `onsite` and `remote` in Lean and holds every verdict each one gives to a Python reference of the rule, on
cases Hypothesis samples (The difftest, below). It needs no GPU, but the build is a large Lean build: run it on a
machine with room for one.

## Subprotocols

| subprotocol | onsite | remote | `Core.Protocol` | id |
| --- | --- | --- | --- | --- |
| network attestation (over the period: its execution phases of transfer blocks certified and proved, and the bound `K` on what left the site) | yes | | `NetworkCheck.Period.protocol` (`network_check`, `../../../resource_accounting/network/verifier/lean`) | `verity/certifier/v0` |
| PoUW (its period credit: the period's windows credited all or nothing) | yes | yes | `PouwCheck.Period.protocol` (`pouw_check`, `../../../resource_accounting/compute/pouw/verifier/lean/check`) | `verity/pouw/period/v1` |
| SPOT (the audit on the proofs themselves, one-stage v2's, which it reads as blobs; one-stage v1 and the recursive audit, which its soundness doesn't cover, refused, and the two-stage audit read, checked and refused last while its switch is off) | yes | yes | `VStar.Recursive.Sampled.protocol` (`core_vstar_recursive`, `../../../proofs/spot_flock/verifier/lean`) | `verity/sampled-proofs/v1` |
| the inference-only policy (its check: the registered circuit's division into inference blocks within their gate and fresh-input budgets, and the provenance of the registered information) | yes | yes | `InferenceOnlyCheck.Protocol.protocol` (`inference_only_check`, `../../../workload_classification/training_detection/inference_only/verifier/lean`) | `verity/inference-only/v0` |
| PoUS (its check of a run) | yes | | `PousCheck.RunProtocol.protocol` (`pous_check`, `../../../resource_accounting/memory/pous/verifier/check`) | `verity/pous/p2/v4/run` |
| PoUB-P2 (its session over PoUS's store) | yes | | `Poub.P2Protocol.protocol` (`poub`, `../../../resource_accounting/memory/poub/verifier/lean`) | `verity/remote-bandwidth/p2/v0` |
| the wipe (the memory attester's streamed-label wipe of the period) | yes | | `MemoryChallenger.WipeProtocol.protocol` (`core_memory_challenger`, `../../../resource_accounting/memory/pous/attester/lean`) | `verity/wipe/v0` |

## The difftest

`uv run tools/check/suites.py verity-core-composition` builds `onsite` and `remote` and drives each one's `serve`
through the harness's `ProtocolPort` (`../../../../../tools/difftest/python/verity_difftest`), holding every field of each outcome but its reason to
`tests/verity/protocols/compose/verifier/lean/reference.py`: the composition rule above, over the harness's framing of the core interface
(`verity_difftest.protocol.answer`), and each subprotocol's own Python, which that subprotocol's own difftest holds its
Lean to:

- PoUW's period: `period_check.case`, with `credit.commit_of` reading each window's Commit record for remote's tie;
- SPOT: its form's split (`spot`) over one-stage v2's decoders and checks before the proof check
  (the challenger's coins and coin trees, the registered file, the proofs' blobs) and then the one-stage difftest's
  `check`, on v2's audits whose sample at the challenger's coins is empty, those with one break each (`V2_BREAKS`, among
  them v1's params or evidence, refused by name) and v1's audits with one or two changes under the form `one-stage`
  (refused as v1's), the recursive form's params refused by name, whether or not they name its files, and the
  two-stage form's params through its own difftest's twin (`sampled_two.py`) and its evidence up to its keys;
- network attestation: its period difftest's `reference`, the Python mirror over the certifier's certificates and
  one-stage's answers, on small signed periods from its `build`;
- the inference-only policy: its protocol difftest's `reference`, the policy's Python check (`check_theta`) behind
  the protocol's decoders, on θ's provenance difftest's declarations with at most one change, from its `drawn`, or on
  the one-stage audit's own registered circuit and record (`policy_on`);
- the wipe: its mirror, `wipe_protocol_case.answer`, on signed wipes from its `case`;
- PoUS and PoUB-P2: their Python verifiers' answers on one committed `C*` (`tests/verity/protocols/compose/verifier/lean/periods.py`: `pous.period_prover.audit`
  signed at its deployed parameters, honest and cheating, and an honest run on another store; PoUB-P2's sessions over
  that `C*`, honest, cheating and unverified, as `p2_case` answers them), whose Lean their packages' difftests hold to
  those verifiers (`../../../../../tests/verity/protocols/resource_accounting/memory/pous/verifier/check/test_run_protocol.py`, `../../../../../tests/verity/protocols/resource_accounting/memory/poub/verifier/lean/test_p2_case.py`).

Hypothesis samples 1000 cases a product (seed 0), and fewer in the short form each thousand-case test has (`FEWER`),
which a check that leaves the test out still runs: each subprotocol's case accepts, rejects, is undecided or fails to decode
(one sample in six every one aimed at acceptance, one in six all but one, which is aimed at undecided; the inference-only
policy is never undecided), every case hands the subprotocols the same blobs, and the composition's inputs, its blobs or
the case itself are often broken. SPOT's cases are all refused (v1's audits, refused by name, and v1's audits
with one or two changes), so every product's case rejects in the difftest and no tie decides a verdict there; the
ties' refusals are pinned by each product's `#guard`s. `test_spot_refuses_one_stage_v1` shows remote and
onsite refusing v1 by name with every other subprotocol aimed at acceptance, in Lean and Python. Often the tied
subprotocols are put on one record (`one_record`), and otherwise they mostly commit to two, which the product refuses.
On one record the policy declares the one-stage audit's own circuit on its record, which it accepts. An honest period's
windows each name their own Commit record, not the audit's, so the `windows` tie, remote's and onsite's, refuses them,
and a period whose windows are put on the audit's record (`windows_on`) refuses itself.
`test_remote_refuses_two_records_and_else_keeps_the_products_verdict` shows that the record checks change no other
verdict of remote's. Onsite's PoUS and PoUB-P2 cases are on one store, but for PoUS's honest run on another
(`periods.OTHER_STORE`, a quarter of its cases aimed at acceptance), which onsite refuses beside PoUB-P2's unless it
rejects already; `test_onsite_refuses_a_stray_tie_and_else_keeps_its_verdict` shows that its two ties' refusals change
no other verdict. A case that names PoUB-P2's by the generic audit's id is refused, as no subprotocol.
`test_a_reference_without_reject_dominance_is_caught` shows the difftest tells the rule above from one in which a
rejection need not reject the whole: such a reference disagrees with the Lean on a case where one subprotocol rejects and another is undecided.
`product` names the composites and refuses every other subset of their subprotocols
(`test_product_names_only_the_composites`, H1's development composite's among them), and over onsite's
subprotocols its `serve` gives onsite's verdicts (`test_product_over_onsites_subprotocols_is_onsite`); remote v1's has
no difftest until Pearl+'s run has reference cases.

A bare `uv run` inside this directory stops its workspace search at `tests/verity/protocols/pyproject.toml`; run the suite
through `suites.py`, or `pytest` with the workspace's `.venv`.
