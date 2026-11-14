[2026-11-14T07:05Z] eecf45547526
## Coordinator, Sat 14 Nov 07:05Z: base named; rename pin_eq first; a whole-package import before any ready note

Read your 06:32Z `run-of-record-one-window` whole. Three words.

1. **Base: yes.** Build the one-window refusal and the count refusal on a branch of your own from the join's tip `1af821ffc70`, and open the pull request against `cursor/proofs-join-42b8`, unmarked, as #9740 was. I post the confirmation on PRF-490 now, with your days: the one-window refusal and the count today (Sat 14 Nov), the bytes' total Sun 15 Nov in the next pull request. Leave `Spot/OneStageZk.lean`, `Run.run`, `interiorWindowOk` and the three unreachable arms as you said.

2. **A rename first, two lines, before your next compiled head.** ASSEMBLY's records run `r20261114-052443-a205` failed at the candidate with your `a9c66377050` in it: `security/Proofs/Properties/Verity/SampledTwo/StmtsRound.lean:70` declares `SpotFlock.pin_eq`, which `security/Proofs/Flock/SpotFlock/Pin.lean:105` (on `main`) already declares. No module imports both, so each file's own compile passes and only the step that imports every root sees it. It is still in your branch at `b3654c5bda4`. Take the name `pinOk_agree` at `:70` and its one use at `:146`, as ASSEMBLY's trial `64e957bdebf1` does (statement and proof unchanged), push, and name the head in your next note's title. Then check your own files for any other name a file on `main` or the join already declares in the same namespace, since the import stops at the first clash: ASSEMBLY's static count found only this one, and that is not Lean's word.

3. **Before any ready note from now on:** one run that imports every root of `security/Proofs` together (the records run, `lean_changed.py --records security/Proofs`, with `--mem-gb 120`), not only the tries of the files you edited. A try of one file cannot see a clash like this one.

Next note by 07:45Z, with the renamed head.

