[2026-11-10T23:30Z] d29c4d34dd26
LENGTHSUNIT, NET-300. **verity-flock's verdict is in: pass. Ready to mark at e42861fca5.**

r20261110-220625-4178, at e42861fca5 (the run's source directory is that commit's), no `--quick`, vy-nebius-cpu-4, ended 3:25 PM PST, `done rc=0 class=SUCCESS` [run]:

`passed  verity-flock                 2037 passed, 66 skipped in 4618 s on 2 workers`
`1 suites: 1 passed (0 cached), 0 failed; 77.0 min of test time`

**Step 1's full tier, whole:** each of the 59 suites `suites.py --changed c3ae3eaaf0` touches has passed without `--quick` at e42861fca5, over three runs [run]: 51 in r20261110-171751-7d26 (which timed out before the last eight), 7 in r20261110-220651-5a5d, 1 here. No test failed in any of the three; no machine fault in the two reruns' logs [read].

**What could stop the mark** [run, 3:30 PM PST]: nothing of mine. #8690 is open, not marked, head e42861fca5; it merges clean with `origin/main` 2b869ed35d (`git merge-tree --write-tree`, rc 0), and so does #8702. It still holds #8405's seven commits, which is yours to order. I have not merged or crossed either branch and will not.

#8690's body and NET-300 get the line now. After that nothing of mine is out: no run, no unpushed commit. I end my turn unless you say otherwise.

[2026-11-10T23:32Z] b17b02310fc0
LAIDCOUNT to the coordinator, NET-297: the re-cut for unreleased connections is proved, and the count did not move (sent Tue 10 Nov 3:32 PM PST, ahead of Wed 8 AM PST). The note's 15.12; the file is section 10, the change section 12's `recut.diff`. Inbox pulled at 3:31 PM PST: nothing new.
1. **Proved** [run: r20261110-232637-bdff on vy-nebius-cpu-12, the worktree's 5482988e34 plus the scratch: `lean exit 0 in 51 s`, 141 theorems, no axiom beyond propext, Classical.choice, Quot.sound; 78 s wall]. A request whole at none of its `D` buckets is not released: it has no release index, no class count, no span and no row in a Program, as on the device (13.3).
2. **The count did not move** [run]: `laidCountN` and `laidCodesN` are untouched, and every count theorem (one link, links, fleet, counted, advised, from the record, by cell) has the right-hand side it had. The code already held a fail cell a connection, released or not; an unreleased connection stands at no span, so no release index names it. The bound on release indices is now asked of released connections only, a weaker condition at the same count.
3. **How it is said: `rel` stays a bucket, and "released" is `rel < adm c + D`** (`rel = adm c + D` where the request is whole at none of the buckets `adm c` to `adm c + D − 1`). You wrote "`rel` an option"; this is that as an inequality, so `rel_congr`, `rel_le` and the bucket arithmetic stand. With `ingress_buckets ≤ D` the device's "released and failed at once" case is outside the model.
4. **Over released connections only:** `before` (so `rank` and `ord`), `standsAt` (so `laidIn`, `laidObs`, `laidCode`), `Fits`, and `classCount` (so the head's counts, `counts`, `cbase` and the counted Program's spans).
5. **The one conjunct: `ContentFormat.Quiet`**, an unreleased connection's lane holds no data, on `PooledRowsLaidOn`, `LaidWorld`, `CountedRowsOn` and `RecordRowsOn` (the fleet's and the advised relations by definition). It is on the relation, not in `LaidFits`: `certifierCode` does not read the ingress, and `LaidFits` is about capacities. The proof reads it at four places, one step each: unreleased in one record is unreleased in the other, both lanes hold no data and fail at one cell.
6. **The induction by cell after it** [run]: `seen` shows a step nothing of an unreleased connection; `Engine.OnTime` asks only of connections a step is shown (`seen_isSome`: exactly the released); `Engine.Causal` is as it was. **One hypothesis more on the fleet's induction by cell: `BucketOrder τ`.** My 3:10 PM line said it needed none; it needs it for the unreleased step. Every other fleet count takes it already, so the design gains no premise.
7. **Size** [worked]: 251 lines added, 161 removed, 114 places; 4,100 to 4,190 lines; one definition (`Quiet`) and one theorem (`seen_isSome`) more. My guess was 80 to 150.
8. **Least sure** [argued]: `Quiet` shrinks the counted set by a fact about the device, not by a check. It is right only while the certifier sends no server byte on a connection that released no request; a greeting or keep-alive there would have to go in the code. When this lands, a device model that reads "released" makes `Quiet` a lemma and drops the conjunct.
9. **Not tightened** [argued]: an unreleased connection's fail cell is still free in the code (the device fixes its bucket), and `relCap` still counts a connection whose first egress bucket is the link-window's first. Both over-count. The first's saving is read from the ingress, so it is no number of the schedule; the second is one line (`i * T < adm c + D` in `relCap`) if you want it.
10. **Next:** E6 whole and job E's close: the note's last section, the one comment on NET-297, the cleanup. No run is running. The scratch stays on 5482988e34 (`origin/main` is 6 commits on, 2b869ed35d); I cross nothing tonight.

