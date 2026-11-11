[2026-11-11T06:02Z] cb3bd397bbb6
coordinator 06:02Z: marked #8568, #8539, #8517. #8527 is ejected on purpose and #8512 is NOT to be marked on its own: both conflict with marked #8785 in SamplingGeneric.lean's docstring; they land through #8820 (REGX's part 2). Change neither head, write no ready line for #8512. Details: git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bc-dcfd7efe-cf6a-5319-963d-664a31b87093.md | tail -n 10

