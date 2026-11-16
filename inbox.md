[2026-11-16T19:23Z] a082ffd2490c
TRN-158, Part 1 sizing, read-only, from main 5f7249690586. What a tied serve of Llama 3.2 1B must write for `verity bench inference_only --from DIR` to print a number above 1.

The reader (benchmarks/inference_only/record.py `read`, `share`, `tied_of`) needs these files under out/:
- record/verdicts.json, inference-only-policy.json, inference-only-limits.json, one block-counts-*.json;
- registered/program.json, registered/circuit.json, registered/log.jsonl.

The check (served_record_case.py `served_case`, then the reference/Lean `trusted_of` and `front`) then needs:
- the log is a chain from its first record, whose `previous` is null, each later record's `previous` the digest of the one before;
- the log's first record registers the weights (θ, `theta`);
- the last record R names `program` = the Program's digest and `window.circuit` = SHA-512 of circuit.json;
- the circuit record's `program` is that same digest;
- R's `previous` is W's digest, and PoUW's accept ties to W (`share`).

What serve.sh on main does, checked on the newest accepted compute run, r20261115-184938-55f5 (done, rc 0):
- it wrote: the policy with k_by_kind, record/circuit-record.json with program df8733642a8ff191…, W chained after nothing, the Pearl+ Python and Lean accepts tied to W, and verdicts.json with the work row `missing`;
- it did not write: program.json, limits, count, any registration after W, or registered/ at all (that run predates serve.sh's `registered` step).

| piece | writer (file, function) | owner | invasiveness | without a GPU? | on main today |
|---|---|---|---|---|---|
| (a) Program on the tied path | rebuilt after the serve: `window_circuit.class_circuit(registered_of_serve(serve-phi), cls, PearlTables(scheme)).program()[0]`, then `canonical_json(encode_program(.))`. That is the call serve.sh's `circuit` step already makes through prove_path.py `class_circuit_of_serve`. Its digest is `partition_object._program` = `codec.program_digest`, which is the circuit record's `program`. | adapters/vllm (vllm) and benchmarks/prove_path (network-accounting); the step itself can be training's | no change to their code: a new step calls the public functions. CPU, about 2 min on 96 cores in that run. | yes: rebuild from a stored run's serve-phi, class and scheme and compare to its circuit record's digest | code there; no serve writes program.json. The tied serve registers no digest at all: W has no `program` field, so the only digest is circuit-record.json's. On the serve-openai Pearl+ path, window-commit's R names the digest of `WC.lower(...).circuit`, the same function as `class_circuit` (`class_circuit` = `lower(...).circuit`), equal when serve.json's φ is serve-phi's (by the docstring; not run by me). |
| (b) window registration R chained after W, and the log [θ registration, W, R] | R: prove_path.py `window_commit_of_builds` with `--pearl-passes` (`PWR.reuse`, `previous = digest(W)`). W after θ: served_passes.py `retain --previous-record WREG`. | network-accounting (prove_path); compute-accounting (served_passes, serve.sh) | HIGH. R commits every output row of the class circuit from the served activations: it needs serve-openai's advice, Commit bundle and row dir, the checkpoint, the kept network-attester run and the committer key. `served_passes.py serve` writes none of those, so R cannot be assembled after compute's serve. Either the tied serve goes through serve-openai's Pearl+ arm (served_verdicts.sh PEARLPLUS_TIE with `serve_tie`), or compute's serve gains a Commit and advice. serve.sh also never passes `--previous-record`, so W is the chain's head. | no: needs a GPU serve to test | the code is there (served_verdicts.sh PEARLPLUS_TIE); compute's serve.sh does neither |
| (c) out/registered/circuit.json | serve.sh `registered()`, a copy of record/circuit-record.json | compute-accounting | none | yes | yes, serve.sh writes it |
| (d) the PoUW row an accept of record | benchmarks/one_stage/served_verdicts.py `record`, via `pearlplus_of_record` | proofs (served_verdicts.py); compute-accounting (serve.sh) | HIGH. serve.sh writes `missing` from its own `verdicts()` heredoc and never runs `record`. `pearlplus_of_record` requires: live coins, with in-process failing condition 1 (so r20261115-184938-55f5 can never qualify); a challenger key equal to `--key-sha512`; the lane's own reserve; a Pearl-C period whose open committed the SHA-512 of this circuit record; a live sample session reserved first after the close; R chained right after W, with records_audit's summary tied to W; and the Lean verdict. "Deferred" means no SPOT records audit runs on the tied path, so there is no R and no audit summary for it to read. | no | no |
| (e) k_by_kind and one block-counts-*.json | k_by_kind: one_stage/inference_only.py `policy`. Count: `verity_vllm.program.predict.block_counts` `main` with `--window circuit.json -` (or R), `--pins WREG`, `--declaration` the policy, `--day`. | proofs (policy); vllm (block_counts) | count: none to their code, a post-serve call. Reads per template, CPU. | yes | k_by_kind yes (that run: linears plus scan_state = k); count no, written only by served_verdicts.sh `block_counts` on the H1 path |
| (f) inference-only-limits.json (not in your list; the reader needs it) | served_verdicts.sh `limits()`: α = k/c and Γ = c from the policy | proofs | trivial, a function of the policy | yes | no on the tied path |

Bottom line: (a), (c), (e) and (f) can be done after the serve, on CPU, from what serve.sh writes today, and I'm starting them in Part 2 as a training-owned step. (b) and (d) cannot come from compute's serve.sh as it is.
- Both need the tied serve to be the one-stage serve-openai Pearl+ path, with W chained after θ, window-commit's R, records_audit tied to W, live coins with a Pearl-C period, and `served_verdicts.py record`.
- Or compute's serve gains all of that.
- Either way it is a GPU run, and owners outside training.

Until (b) and (d) exist, any number above 1 is from a made record: a test, not a result.

