[2026-11-11T03:46Z] b5f861ad55e6
From circuits' coordinator, 03:44Z. THE FULLER ORDER; it replaces the ORDER section of the phase B brief. Everything else in the brief stands (the recipe, plain fast-forward pushes, the rules).

WHY: top's rule for every lead (TOP-398, 03:34Z): ci crosses each marked head that crosses clean with `research queue cross` at about 05:00Z, and its mark and land verdict carry; owners push nothing to such a head before that. Owners cross by hand the unmarked heads and the heads ci reports as stopped. I read ci's two dry runs (`r20261111-024824-df4e`, `r20261111-032059-0023`, `out/cross.json`) for what ci does with each of ours.

A. PUSH NOW, by hand (unmarked, nothing below them off `main`): #8594 with #8774 as one stack; #8426; #8370; #8430, #8463, #8464 with your mends. One line on CIR-223 when #8594 and #8774 are pushed, with both heads.

B. PUSH NOW, by hand: #8694 with your mend (marked, but ci's dry run refuses it at the moved merge, so ci does not push it). One line on CIR-223 naming the files of the mend: ci's reviewer reads only those.

C. NEVER PUSH (ci pushes them, about 05:00Z): #8520, #8362, #8432, #8618, #8673, #8599. Do one thing for them now: the repository's root suite (`tests/`, quick tier) on each one's crossed tree. top found a head that crosses clean and is then red there (CI-524, 03:42Z: a line naming an old path inside a file the map does not rewrite). `main`'s tree is `178f49c90c89`'s tree, so your phase A results already have the trees ci's crossings will have: run the suite on those, in as few runs as you can. Report a failing test on CIR-223 at once with its name; I withdraw that head's mark before a batch takes it.

D. AFTER ci HAS PUSHED A PARENT (its branch's head holds both moves; look every ten minutes from 04:50Z): cross its children over THAT head's `Restack-move:` commits, your stack recipe, and push them. ci skips these as stacks and says their owners restack them. #8765 on #8520. #8447, then #8534 on it, and #8397 and #8394, on #8362. #8451 on #8432. Before each, check what the child holds besides its parent: if it holds another open pull request's head that is not on `main`, name it and wait for that one too. One line on CIR-223 when #8765 is pushed, and one when #8362's children are.

E. #8461 and #8466: as before, only after vllm's #8441 holds both moves on its own branch (ci's dry run refuses #8441, so vllm crosses it by hand).

F. #8651: yours by hand after all, last of the pushes, by 05:15Z. ci's second dry run crosses it clean (as `5a17dce5c4bb`) but refuses to carry its mark ("its merge with b5354b9c2344 ... changes its own change, so no mark or verdict would carry"), so ci's step does not push it. Cross it the way ci's run did: the vllm-allowlist merge driver registered, and the pre-move attribute lines given through `core.attributesFile` (a file holding `b5354b9c2344`'s two vLLM lines of `.gitattributes`). If ci's branch for it has moved when you come to push, leave it and say so.

DEADLINES: A and B by 05:00Z. C's runs started by 04:15Z. D within 40 minutes of each parent's push. If ci has pushed no parent by 06:30Z, return with what is done and say what waits. Return by 07:00Z at the latest. The RETURN of the brief stands, with one more column: who pushed the head, you or ci.

