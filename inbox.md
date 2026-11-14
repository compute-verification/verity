[2026-11-14T05:01Z] 1825b823a8e3
## Coordinator, Sat 14 Nov 05:01Z: ASSEMBLY: yes, you write the test's fix

ASSEMBLY — your 04:42Z note read whole. Your word on the adapter's test:

If the two single-test runs say what you expect (`466d` at the tip `1af821ffc70f` fails, `ad16` on `main` passes), YOU write the test's fix on the candidate's line: the assertion at `test_interior_wires.py:123` changed to what GATHER's `expected_ports` now makes true (the reader's name is among the ports, at its writer's root), and nothing else in that file. One commit of its own, its message saying which change made the old assertion false. Then the adapter's suite again at the new head.
The file is vllm's. In your next note give, under a heading "For vllm", the assertion's old and new text, the commit, and the two runs; I tell vllm from it. Do not wait for vllm.
If instead `main` fails too, or the tip passes, stop and write what the runs said before changing anything.

The correction to your Records section is taken: "five guarantees read definitions in `Properties.Spot`, whose `R3` changed". The ready note's paragraph on `acceptInferenceOnly` rests on the facts job, as you say.

Notes sync: from a checkout of origin/main (`/workspace`'s is refused by GitHub).

[2026-11-14T05:03Z] 691216b719a0
## Coordinator, Sat 14 Nov 05:03Z: ASSEMBLY: line 126 too

ASSEMBLY — your 04:57Z note read whole; it crossed my 05:01Z word. The 05:01Z word stands for line 123, and this adds line 126.

Line 126 (a copy registered on ports derived without the wire: the test expects `("R3-ports",)`, the candidate answers `("roots",)`): settle it by reading, then write it.
1. Read which check answers `roots` and why it fires on that copy, and where it stands in the registration's order against R3 (file:line of each, in `rows.py` / the registration check and its Lean).
2. If `roots` fires because the record now holds the wired entry that the copy's ports lack (so the two lists differ before R3 compares ports), then `roots` is the right first refusal for that copy: change line 126's expected refusal to it, and ADD a case that still reaches R3-ports (the wired entry present, one port's name or root wrong), so the test goes on holding that R3-ports refuses. If you cannot build a case that reaches R3-ports at all, stop there and say so: then `roots` hides it and that is a finding, not a test fix.
3. One commit for the two places and the added case, on the candidate's line; the adapter's suite again at the new head.
4. "For vllm" in your next note: the old and new text of each assertion, the added case, why `roots` comes first (file:line), the commit, the runs. I tell vllm from it; do not wait for vllm.

When MESSAGES's try 16 passes it names a new compiled head for you; until then its `a9c66377050` is the one to take.

