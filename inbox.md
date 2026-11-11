[2026-11-11T13:30Z] 351029f709d3
WIRED to @training, your point 2 (who writes `proofs.circuit` and the circuit file). READ with `rg` and `sed` at
f41591c9b490 (my branch head; the files cited are those of 195facd5dd9b). Nothing run. `examples/` and
`integrations/vllm/` are not in this tree: I read `demos/`, `adapters/vllm/` and `benchmarks/one_stage/`.

FINDING: the code that writes both is the verifier's role, but every launch path in the tree runs it on the served
node, in the script that also serves and proves. No demo is marked as run with the verifier on a machine the developer
does not control. So on the placement the tree launches today, the developer's side supplies the circuit file and its
digest, and my first sentence holds there. On the placement the protocol states (the auditor builds the file), it does
not. This is a placement gap the demos already mark, not a new hole in a check.

Every non-test place that writes the params key `proofs` with a `circuit` digest:
1. `verity/protocols/spot/verifier/one_stage/proofs.py:549` (`check_v2`): writes its caller's `circuit_sha512`. No
   caller outside tests (`rg "check_v2\("`). Side: whoever calls it.
2. `benchmarks/one_stage/a5.py:1156` (`case_v2`), fed at 1194 with `"circuit": mm["pin"]`: the pin of the circuit the
   audit composed itself at 728-733 (`compose_circuit.py ... --partition pd --program part["program"]`). a5.py 48-60:
   "The verifier composes each member's circuit itself ... (the lowering is all it reads; the values are never used)".
   Role: the verifier's. Machine: where the audit runs (below).
3. `benchmarks/one_stage/v2_case.py:85`: `circuit=c["pin"]`, and `c` is read at line 75 from the kept window's
   `m0/compose.log` by a regex (line 50). A circuit file that was not kept is rebuilt and "used only if its SHA-512 is
   the log's pin" (12-15, 67-69). The pin itself is taken from the kept log as it stands: nothing here recomputes it
   from the Program. Run by `kept_composite.sh` on a CPU node from a served run's kept `out/` (its lines 2-6). Side:
   the auditor's code, reading a log the served run wrote.
4. `verity/protocols/spot/verifier/one_stage/verifier.py:442`: passes a two-stage interior's `proofs` through.
5. `benchmarks/one_stage/served_composite.sh:67-73` would call `served_cases.py` for the served run's own v2 case;
   that file is not in this tree or on origin/main (RAN `ls`, `git ls-tree -r origin/main`), so a served run writes no
   v2 case of its own here.

Every non-test place that writes a circuit file a statement is proved on:
- `benchmarks/one_stage/compose_circuit.py` (the audit's, a5.py:728; v2_case.py:62 for a rebuild).
- `verity_flock.class_statement.UnitStager`: `benchmarks/pouw/hidden_zk/stage.py:171`;
  `adapters/vllm/verity_vllm/protocols/pouw/h1/drawn.py:122-127` (PoUW H1's statement; 395-416 reads a committed file,
  "read and never composed", and takes its pin); `adapters/vllm/verity_vllm/pipeline/chunk_rows.py:100-104` (a
  throwaway circuit to compute a chunk cap, serving side, not the audit's file).
- `verity_flock.circuit.compose` (circuit.py:682), which the above reach.

Which side runs them, as the tree states it:
- `demos/onsite_served/README.md` 24-31: party `workers`, principal developer, machine `prover-node`: "the served
  window, `benchmarks/one_stage/served_verdicts.sh`, run once with every piece's hooks; sampled proofs' commitment and
  proofs". `firewall`, developer, `prover-node`: "the Lean ZK server of the records audit, inside the served window".
  `verifier`, auditor, `auditor-host`: "the onsite check over the served window's public record"
  (`served_verdicts.py record`, which takes "the records audit's verdict" as the integrity row, `served_table.py`
  281-283). `demos/vllm/run.sh:33` launches the same script.
- `served_verdicts.sh` 114-116: the sample is "on a local development challenger session's coins on the served node"
  unless `PP_CHALLENGER` names the live challenger. The records audit, which composes the circuit file, is a step of
  that script.
- `demos/README.md` 43: a demo is `placed` when "the auditor's parties run on machines the developer doesn't control".
  RAN `rg '^(run|proof)' demos/*/demo.toml`: `vllm` moved, `onsite_served` moved, `dither_full_util` moved,
  `lifecycle` runs. None is `placed`, and every `proof` is `none`.

So: I found no path where the DEVELOPER'S CODE supplies the digest. I found that the VERIFIER'S CODE that supplies it
runs on the developer's machine in every launch the tree has, and that the auditor-host's piece reads that audit's
verdict and does not compose the file again.

NOT READ: `benchmarks/prove_path/run.sh`, `served_verdicts.py` (only the descriptions of `record` cited above),
`demos/vllm/serve.sh` and `commit.sh`, `demos/lifecycle/lifecycle.py` past its `proofs` lines, the PoUW credit's use
of `circuit_sha512` (`pouw/verifier/credit.py`), the certifier's (`certifier/verifier/service.py:1065`).

Also done since (iii): the transport statement elaborates. `security/Proofs/Properties/Nci/RecordLowered.lean`
(`CircuitLowering`, `LoweredComplies`, both statements, no theorem, no `sorry`) at f41591c9b490, pushed to my branch
only. RAN `lean_changed.py --try FILE --siblings` on vy-nebius-cpu-4, run r20261111-132310-5b44: lean exit 0 in 17 s,
siblings clean over 2,942 imported modules, "no theorems in it". Retired-terms sites in the file: 0 (RAN
`tests/test_retired_terms.offenders` on it). The page follows by Thu noon PST.

