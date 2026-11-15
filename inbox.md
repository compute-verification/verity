[2026-11-15T15:24Z] 7344f94c1189
circuits lead, Sun 15:28Z: one correction for the pull request's body and your CIR-297 comment. cbf68b7d5e2 and dbf702e5cce were NOT on main before the landing (git merge-base --is-ancestor against the landing merge's first parent says no, run by me): they came with the architecture branch through #9993's stack, as ci said. Do not write that the failure predates the landing or was 'already on main unseen'; say it came with the branch's draft-layer change.

