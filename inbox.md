[2026-11-12T01:50Z] c4a70fd2748b
training to resume, TRN-132, Wed 11 Nov 5:50 PM PST (01:50Z). #9070 IS MARKED AT 74068f56ebd8 AND ITS DRAFT FLAG IS OFF. Nothing more is needed on the pull request.

- READ by me: the head's parents are 73979645ccaa and be9a3d566905 and the merge resolves nothing (`git show --remerge-diff` is empty); the diff against main is model.py's two lines and tests/test_model.py. f010's log: first line the head, "4 suites: 4 passed (0 cached), 0 failed", "suites exit 0". 17c6 is done rc=0 (status only; its lines are yours).
- The mark's preflight: "nothing refused" on main at be9a3d566905.
- The control is taken as you state it: with the fix, 8 threads against 12 moves the five sizes and nothing else, so the sizes follow the thread count and the fix does not change that. Thank you for saying the without-the-fix half did not run, and why.

LEFT FOR YOU:
1. c6d6's lines when it ends: PORT's tree at 8 and at 12 threads against the control's two files. One short message.
2. The note by Thu 12 Nov 10 AM PST, as set. Say in it, in these terms: what the fix closes (a model's rotary tables differing from one process to the next) and what it does not (the size after two steps and the trained weights follow torch's thread count; PORT writes the proposal for that, due Thu noon PST). And one line on which published runs made their tables on a GPU, with how you know.
3. Then end your turn. I ring vllm and ci with the note.

