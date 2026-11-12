[2026-11-12T23:35Z] 7443680b4c6c
BACKWARD2, Thu 12 Nov 3:36 PM PST.

**The second launch was refused too, so I've stopped and launched nothing else: vy-nebius-2 takes a `research run` only through its check slots, and the launch needs `--pool research`.**

- RAN, 23:35Z, r20261112-233520-3a70: the same command with `--project verity`. Its own line: "research: run r20261112-233520-3a70 refused on vy-nebius-2 (Refused): this machine takes a `research run` only through its check slots (/etc/vy/runs-via-slot): add `--pool proofs|research|soundness` to `research run` or `verity lean try`, which sends CMD through that pool's slots".
- INFERRED: a third launch with `--pool research` added, nothing else changed, would wait for a research place (infra's 32 GB need) rather than be refused. I launch it only on your word.
- Done meanwhile (READ r20261112-212456-3c41, rc 0, at `1178eb5439`): "TOTAL deterministic heads 32 words a gradient 2097152 equal dq/dk/dv: 2097152 2097152 2097152". So I pushed `a809ac0598`: `"fit"` (art:82458cb58b136a75e0202b95d68ad96dc5e2b6b0b3953fefc9e4c0730a2bca1e) in `evidence.json`, and CONFORMANCE says every word of the captured deterministic heads at H = 8, 12, 16 and 24 and cites `_FA2['fit']`. Against `1178eb5439`, the only change under `verity/definitions/registry` is that CONFORMANCE string; no Definition's gates change. I'm launching circuit-check and the suites at `a809ac0598` now (CPU pool, `--on auto`).

