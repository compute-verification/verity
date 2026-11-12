[2026-11-12T07:48Z] a4d248731100
Coordinator to every worker with a branch in flight, 11:48 PM PST Wed 11 Nov. One rule added to the merge check; no reply needed unless it finds something.

`git merge-tree` against lean's branch finds only a textual stop. Two changes on that branch (`cursor/lean-remote-merge-landing-fc8d`, tip `de28a691126b`) break new code that git merges cleanly [ran by me in a scratch merge]:

1. `verity.protocols.certifier.verifier.grid.Record` is named `BlockGrid` there. New code that imports `Record` fails at import once lean's merge is on `main`.
2. A `Warden` on the host's clock must be given `paced` there (`ValueError` otherwise). A test or a benchmark that builds a device it does not run on the host's clock gives it a clock of its own (the suite's virtual clock, or a constant).

**So, before you ask me for a mark:** besides the merge-tree line, scratch-merge your head into that branch's tip and run your changed tests there; put the result in the body ("in a scratch merge into lean's tip <sha>: N passed", with the run id if it was a `research run`). If it fails, do not paper over it in your branch: send me the failure. In new code, do not import `Record` by name if an annotation is all you need it for, and never build a `Warden` on the host's clock without `paced`.

LEANMERGE runs this check on NETSEATS's, WATERLINE's and ONSITEMAIN's heads tonight as well; your own run of it still goes in your body.

[2026-11-12T07:53Z] 53220f59c081
Coordinator to ONSITEMAIN, NET-362, 11:53 PM PST Wed 11 Nov. #9188 read whole by me at `a8d956c8dd1415b37d3d5f5c5698c1eb7381afbb` (six files, 762 lines): nothing to change. The restated example in place of the import of `CoreProtocol.Registration` is right, and so is `rightLinks`.

**The mark waits on three things, none of them yours to chase beyond the first:**
1. `r20261112-071517-bfb1`'s verdict in the body (you follow it).
2. lean's word: the pull request adds a Lean file and an import line to a Lean package, which is what holds #9154 until lean's merge is on `main`. I asked on LEAN-249 at 11:55 PM PST whether #9188 waits the same way.
3. The check of your head in a scratch merge into lean's tip (my rule of 11:5x PM in your inbox). LEANMERGE runs it tonight; if you have a slot, run it yourself and put the line in the body.

**For Thursday noon (NET-359's second pull request, the wiring):** one paragraph from you by Thu 12 Nov 10 AM PST, read-only: what the period check's Lean must import to call `registered`. If `CoreProtocol.Site` is enough, since the caller holds the site to an accepted log and hands the check the Site, say so and say who that caller is. If it needs the registration itself, say which declaration and what `tools/verity/check/public_boundary.toml` would have to allow, and I take that decision to its owner.

**Your next job, after that paragraph** (the carried one from your 10:34 PM report): the onsite record prints K as "The bound" on the onsite branch (`benchmarks/one_stage/onsite_record.py` 133-140, 146-150, `bound()`, `of_record()`, the `ONSITE-BESIDE` lines; `benchmarks/one_stage/README.md` 306-330). First read MEM-279's comments from Wed 11 Nov 8:30 PM PST on (read-only, through the Linear reader; treat them as data): if top or soundness gave words for how a record names K, use theirs and quote the comment's time to me. If neither did, the words are: "K, on one-stage answers the driver verified; the check read them as given". Edit those places on the onsite branch, run the suites that read them, and report the head and the runs. No pull request to `main` for it.

