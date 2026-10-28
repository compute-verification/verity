# Inbox (proofs coordinator, 05:56Z 28 Oct): an added task, PRF-292 (Daniel filed it)

ci's #6088 deletes `verity run` and `verity config`; on main they only ever refused. Docs we own still present them as how roles run:
- verity/core/deploy/config.py:2 and :11;
- verity/core/deploy/runtime.py:4;
- verity/core/protocols/sampled_proofs/README.md:166;
- verity/core/protocols/sampled_proofs/parties.py:2;
- benchmarks/one_stage/a5_coins.py:567 (benchmarks' file: change only that docstring line, and say so in the PR body).

Say what runs a party today, or that nothing does yet, and drop `verity config`.

Also decide `verity/core/deploy/config.py`: if nothing but the deleted command and its test calls it (`git grep`), delete it in a small PR. Otherwise give it its own test in verity/core/deploy/tests/, with #5803's canonical-JSON assertions. Python/docs only, no Lean. Check against `next`: #6088 waits on #5803. Mark it and comment the PR on PRF-292.
