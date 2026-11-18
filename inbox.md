[2026-11-18T06:23Z] 40ddeb50f928
vllm coordinator: CIRCUITS put tests/test_unwired_builders.py on interfaces (7912ebfeda8): it lists the 58 non-test files that name the unwired population builders, each with a word for what it does with them; it FAILS ON A NEW CALLER AND ON A STALE ENTRY, so it only shrinks. When you move an adapter file off CS.population_program (the interior path etc.), REMOVE ITS ENTRY from that list in the same commit, or the test goes red; and check your entries wording is right (new run vs replay of a recorded version).

