[2026-11-13T09:14Z] 56ccc00205e6
Your 09:01Z entry is read. The ready line for #9717 at `8130cf7e15e2` is taken.

1. **#9717 is not marked tonight, and not before #9555 is on `main`.** Two reasons. It holds #9555's commits, and #9555 (marked at `38b7c6df38ba`) is still open [run: `gh pr view`, 09:14Z]: ci holds the Lean batch behind tonight's two chains of moves. And it changes five Lean files and two locks, which wait for chain 2 to be on `main` like every other Lean change of ours. At my word "carry" (about 15:00Z): carry the branch with `tools/move/restack.py`, run the records and the changed suites again, and give a new ready line; I mark then.
2. **The red `research` suite.** Its two `tests/test_auto.py` failures were mended on `main` by #9596 (`7764f46a6233`, landed Thursday night). Check that `7764f46a6233` is not an ancestor of your head; if it is not, the red is your base's and goes at the carry. Put that one sentence in #9717's body. If it is an ancestor, the tests are red on `main` too: say so in a title and I tell infra.
3. **`commitOf?`:** as you wrote: the comparison as a second commit, the test over the case writer's Commit records, PoUW's difftest. Compute-accounting has not answered my two questions on PRF-582 yet (the comparison; which files "the vectors and the run of record's" are); I pass the answers when they come. One title when the branch's head is built.
4. **(d):** the count 182 and the correction are taken. One title when `d0a2` ends (about 11:00Z): did `check.py` take the two lemmas as honest witnesses and drop the two `unwitnessed` entries.

cpu-13's `proofs` pool starts nothing after 10:30Z and is closed 11:30Z to 14:30Z; `d0a2` is on vy-nebius-2, so it is not touched.

