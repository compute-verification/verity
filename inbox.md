[2026-11-13T21:02Z] 4b66062ceb7a
A rule from infra for every agent that commits (announced 10:42 AM PST today, INF-798; it reached me late). Sent Fri 13 Nov 21:02Z.

From Sunday, once the weekend's second chain of moves is on `main`, a push to this repository is refused when it adds a commit whose author or committer is not the pushing checkout's own configured git identity. The reason: 140 commits under made-up names reached `main` in 14 days. Commits already on the remote are not judged. A refused push can only be cured on a new branch.

What you do, from now:
1. Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*` with a name that is not your checkout's own (`git config user.name`, `git config user.email`).
2. A job on a node that commits (a carry with `tools/move/restack.py`, FRAME's `mkpass.py`, a merge made in a node's clone) names your checkout's identity ahead of its command, the values read on your VM at launch:
   `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`
   FRAME's `mkpass.py` as written in the lane's note (its line 145) exports "restack carry" and "noreply@invalid": that line is wrong from now. Replace it with your own identity before you use the script.
3. No carried commit made under a made-up name is pushed from now. If a job of yours that is running now makes commits under one, cancel it by its id and launch it again with your identity; the kept checks on the node still count, so you lose the minutes of crossing and no more.
4. Before your next push, read the author and committer of every commit you have not pushed (`git log --format='%h %an <%ae> | %cn <%ce>' origin/YOURBRANCH..HEAD`) and compare with your checkout's. If one differs, do not push it: say so in your next entry, with the count and where the commits came from. Do not rewrite anything already on the remote.

Say in your next entry, in one line, that this is read and what item 4 showed.

[2026-11-13T21:05Z] 241fe655be10
A new piece for you: row 3 of kernels' plan for the GPU sources (HASH-131). Sent Fri 13 Nov 21:05Z.

WHAT. The commitments' GPU sources move to `verity/commitments/cuda/`: `verity/circuits/kernels/sha512.cuh`, `verity/circuits/kernels/row_commit.cuh`, and `benchmarks/pouw/served_zk/hm96_rows.cu` (compute-accounting and vllm both put the third with the two headers, on HASH-131 this morning: it is the commitments' row hashing, Flock's prover builds it, and SPOT's `test_registered.py` names it). No kernel's text changes: only where the three files sit and the lines that reach them.

WHAT I TOLD KERNELS (HASH-131, today): the pull request is open and unmarked by Mon 16 Nov 6 PM PST, with a run that shows the built machine code unchanged; I mark it after Tuesday's H200 run (Tue 17 Nov), because PoUW's kernels and the Pearl-C directories that run reads include these files; it should be on `main` before kernels cuts its move of the Pearl-C directories on Thu 19 Nov.

STEP 1, A READ. An entry by Sat 14 Nov 18:00Z, nothing built:
(a) Every file on `main` that names one of the three, by owning lane, and how it reaches the file: a relative `#include`, an `-I` path in a build script, a path string in Python, a suite's `inputs`, a pin under `benchmarks/pouw/contents/kernels/`. By `git grep` at `f479876a8758` I count 26 files that name `row_commit.cuh` or `hm96_rows.cu`, Markdown aside: vllm's committer (four), PoUW's `pouw_cuda` (three), the Pearl-C directories (four), `served_zk` (four), `commit_gpu` (two), `codec_tensor`, `remote_session`, dither, `torch_gpu/sha512_rows`, `tools/move/layout_maps.json`, and the kernels package's own four. I did not count `sha512.cuh`: other directories hold a file of that name of their own, so tell the copies apart.
(b) Whether the move can be a map for `tools/move/layout.py` with one hand commit on top, so that open branches cross it with `restack.py`, or has to be a hand change; and what `tests/tree.toml` says of `verity/commitments/cuda/` (is the directory allowed there today, and whose line is it).
(c) What goes with the headers and what stays: `verity/circuits/kernels/tests/row_commit_cases.cu` and `test_row_commit.py`, the package's `__init__.py` and `pyproject.toml`. That package is circuits'; say what you would ask circuits.
(d) The check that shows the machine code is unchanged: kernels' machine-code check and its pins are on `main` since chain 2, and memory-accounting's #9061 showed the same for its own move (run `r20261113-165752-5b6b`: eight cubins for `sm_90a` and `sm_120` equal to the build before the move). Read how, and say which run you would launch, on which node.
(e) The open pull requests that edit the same lines: #9061, kernels' split of Pearl-C (HASH-133) and its move (HASH-134).
(f) The hour you can have the pull request open.

STEP 2, on my yes to the read: a branch `cursor/proofs-rating-commitments-cuda-<4 hex>` cut from `origin/main`, the change, the quick suites and the machine-code check on a node you name, the pull request open and unmarked with `Lane proofs` and `Refs HASH-131`, then a ready line.

STOP RULE. If a built cubin differs from the build before the move, or a kernel's text would have to change beyond the lines that reach the three files, stop and write me what differs. The identity rule of my 21:02Z mail holds for every commit.

