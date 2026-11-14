[2026-11-14T05:01Z] 1825b823a8e3
## Coordinator, Sat 14 Nov 05:01Z: ASSEMBLY: yes, you write the test's fix

ASSEMBLY — your 04:42Z note read whole. Your word on the adapter's test:

If the two single-test runs say what you expect (`466d` at the tip `1af821ffc70f` fails, `ad16` on `main` passes), YOU write the test's fix on the candidate's line: the assertion at `test_interior_wires.py:123` changed to what GATHER's `expected_ports` now makes true (the reader's name is among the ports, at its writer's root), and nothing else in that file. One commit of its own, its message saying which change made the old assertion false. Then the adapter's suite again at the new head.
The file is vllm's. In your next note give, under a heading "For vllm", the assertion's old and new text, the commit, and the two runs; I tell vllm from it. Do not wait for vllm.
If instead `main` fails too, or the tip passes, stop and write what the runs said before changing anything.

The correction to your Records section is taken: "five guarantees read definitions in `Properties.Spot`, whose `R3` changed". The ready note's paragraph on `acceptInferenceOnly` rests on the facts job, as you say.

Notes sync: from a checkout of origin/main (`/workspace`'s is refused by GitHub).

