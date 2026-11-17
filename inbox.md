[2026-11-17T18:02Z] 1ffb7a49f4dd
KCARRY, from the kernels lead, Tue 17 Nov 18:02Z. The queue merged three of your seven at their OLD heads at 17:56Z, as they were: #10633 (b5e45ff220f), #10750 (661fda6e057) and #10741 (5af4b43ffee) are MERGED. Do not push your carries of those three; drop them.
What is left, in this order, and push and mark each the minute its own suites pass, without waiting for the others (Daniel's rule today: a fix that can be made now is made now):
1. #10809, then #10821, then #10823: merge today's origin/main into each carried head first (main now holds #10750 and #10741), check the diff against main is still only the PR's own change, push, `research queue ready N`.
2. #10674: its carry sat on #10633's carry; rebase that away by merging today's origin/main (which holds #10633), check `benchmarks/kernels/s8_linear/wgmma.py` and the README conflict again against main as it is now, then push and mark.
#10566 stays unbuilt; say what a rebuild takes in your return.
If a suite run started before 17:56Z is still going on a head that now needs main merged in, do not wait for it: merge main, and run only the suites the PR's own change touches (`suites.py --quick --changed origin/main`).

