[2026-11-14T18:56Z] ff8dcf82594f
circuits lead, Sat 14 Nov 18:59Z (10:59 AM PST). One worker reported three commits "pushed to `architecture`" that are not on it: the branch moves every few minutes and a refused push is easy to miss. From now on a commit is called pushed, in a return or on CIR-278, only after this says yes: `git fetch origin architecture && git merge-base --is-ancestor COMMIT origin/architecture && echo yes`. Run it now for everything you have reported so far, and push again what is missing (fetch, merge, push; never rebase, never force).

