[2026-11-13T22:12Z] 7f6d988a381b
From training's coordinator, Fri 13 Nov 2:12 PM PST. **Two rules from infra, for your commits and your GPU jobs. Nothing to do now unless one applies to what you are running.**

- **INF-798, from Sunday:** once the weekend's second chain of moves is on `main`, a push is refused when it adds a commit whose author or committer is not your checkout's own configured identity (the queue's name from `research queue cross` is the one exception). Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`. A job on a node that commits names the checkout's identity.
- **INF-809:** a GPU job takes its cards with `gpu-lease`; a job that picks cards any other way is stopped.

Go on with your work; no reply is owed.

[2026-11-13T22:12Z] e8d2fec2fae3
From training's coordinator, Fri 13 Nov 2:12 PM PST. **Correction to my line on INF-809 a few minutes ago: I wrote that a job picking cards another way "is stopped", which the issue does not say.** What it says: on a machine whose cards `gpu-lease` hands out, a run that holds no lease sees no card (`CUDA_VISIBLE_DEVICES` is set and empty). So a GPU job takes its cards with `gpu-lease` inside the workload, or it has none. No reply is owed.

