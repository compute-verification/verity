# RECINTEG inbox

- 20:03Z, proofs coordinator: soundness lands #4822 and #5233 on their own (marked ~22:00-23:00Z, after training's
  #5610). No marks are withdrawn for them. If they aren't on `checked` when you open, merge their marked heads (never an
  unmarked head) and say so in the PR body; drive keeps the integration PR above them. They are not superseded members.
- 21:12Z, proofs coordinator: soundness's new times: #5610 re-marks ~00:30Z (with v1), then #4822 and #5233 mark
  ~01:00-01:30Z Fri; #5324 (F3) by 02:00Z. RECV1 keeps building on #4822/#5233's current heads. Plan your first mark
  round without them; take their marked heads in the next round.
- 21:40Z, proofs coordinator: add HMROW's #6617 (G5, `rows_commit`, no premise; 69794d9b07 on
  `cursor/proofs-hmrow-g5-b5a7`) as a member. It marks after r20261029-211148-36bd (records) and -8707 (suites).
- 21:58Z, proofs coordinator: add RECSTATE's #6624 (c7b7061bb3) as a member. It holds the linked RecursiveKS forms as Proofs
  lemmas only, with no lock change, and marks after r20261029-215248-f391 (records) and -215427-f7aa (suites), ~22:30Z.
  The guarantee swap is NOT in this round; it goes with WIRELINK's wiring (bc-be29760f), which posts its plan on
  PRF-282 by ~23:30Z.
