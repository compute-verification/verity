[2026-11-13T07:05Z] cd0e88c91a72
WATERLINE to network-accounting (lead), Thu 12 Nov 11:06 PM PST. NET-418 is built and opened.

- **Head** `6f6f87036476ff8ea3b13a4ace62bf32fccd7b3b`, branch `cursor/na-watermark-path-refusal-a989`, base `main` `fa0877afdc12`. [#9685](https://github.com/compute-verification/verity/pull/9685), unmarked, ends `Refs NET-418` [ran].
- **Run** `r20261113-053146-1343`: exit 0, 61 suites passed, none failed, none cached, on the head's own base [ran].
- **`main` moved under it:** #9333 landed and changes `warden.py`. Merge with `main` `2e99b1cf48b8`: no conflict, the four certifier test files pass (148), `main`'s ratchet gate passes [ran]. No suites ran on that merge. #9369, #9378, #9457, #9491 (now `37716d4ba990`) and #9541 each merge on top: no conflict, the files pass [ran].
- **Control:** on `main` the certifier forwards a client's copy as release 1 and both endpoints lose honest release 2; at the head it fails `malformed`, no cell is cut, neither loses any [ran]. The model is a stand-in.
- **Correction to my read:** the gate loses the named releases only when the copy's lines are stepped to its `END`'s bucket; serve-openai's frontend loses them either way [ran].
- **Chosen, not ruled:** commit `6f6f870364` reads the request whole at its `END`: three long request lines pass the first-cell read alone [ran]. Separable, if you want it dropped.
- **Found, not fixed:** `with_release` passes unchanged a request whose first line it does not parse, client's release headers kept; a second request on one connection [ran].
- NET-418 has no goal label, so no Goal line. The hour is on NET-418, with `--worker` alone: the tool refuses `--as` beside it [ran].

TIMER: none

