---
id: 20261105T1730Z-report-landing-volume
campaign: landing-speed
lane: top
kind: report
status: draft
repo: compute-verification/verity
origin: "Daniel, Thu 5 Nov 9:06 AM PST: use a bunch of agents to cut landing latency; learn from two weeks of failures. top's worker, angle VOLUME (less to land)."
---

# Landing volume: less to land (update 18:00Z)

Branch `cursor/top-landing-volume-8a78`, scripts in `tools/research/studies/landing_volume/` (contain.py, flow.py,
labels.py, heads.py, reheads.py, carried_csv.py). Snapshot: 649 open PRs at 17:12Z, queue status at 17:20Z, `next` 54
members above `checked` at 9bdd6207.

## Findings so far

1. **Half the review and mark work goes to heads that never land.** Last 7 days (29 Oct 17:00Z to 5 Nov 17:00Z): 1,251
   land verdicts, of which 446 are on heads that landed, 271 on open PRs' current heads, 376 on heads later superseded and 158
   on PRs closed unlanded (534, 43%, certainly wasted). Ready marks: 1,572, 727 (46%) wasted the same way. Verdict writers:
   ci 406, ci-review 296, top 253, lean 240. Landed PRs since 29 Oct (701): 1.95 marks and 1.76 verdicts each, 0.64
   verdicts on heads other than the one that landed.
2. **Each re-head costs about a day.** Landed PRs: first mark to `main` p50 45 to 100 h by day, final-head mark to `main`
   p50 25 to 83 h; the difference (re-heads) is about 18 to 25 h at p50. Mark to `checked` p50 15 to 65 h; `checked` to
   `main` p50 10 to 26 h.
3. **Carrying is common and the queue pays for it twice.** 269 open PRs have their head inside another open PR's head.
   39 marked PRs (24 on `next`, 15 admitted) are inside a marked PR of the same lane: each is its own member (merge,
   preflight, verdict). `research queue review` shows each member's increment over the member below
   (`changed(m.commit^1, m.commit)`), so a carrier's verdict never covers what it carries.
4. **The demo lines.** `remote`, `cursor/onsite-demo-061a`, `cursor/soundness-assemble-5e54`, `cursor/vllm-remoterun-a7d3`,
   `cursor/soundness-pluscomp-f880` and `cursor/windows-placed-061a` each hold 3,365 to 4,647 commits not on `checked`,
   with 651 to 712 of old `next`'s merge commits (from ea3f9c5635, Tue 3 Nov). 58 open PRs (36 network-accounting, 15
   proofs) contain it, each containing about 115 other open PRs: through the queue they land only after all of those.
5. **Same-lane conflict.** Since Mon: 511 "out of `next`: conflicts" posts (127 PRs), 271 name a same-lane PR, 60 only
   same-lane PRs. 690 "preflight hasn't run" posts (263 PRs), 582 of them the sync's preflight budget running out. 192 hand
   ejections by lanes (about 40 "carried by" or "folded into"); 325 "your bases landed, merge `main`" notices to 222 PRs.
6. **Flow, 29 Oct to 4 Nov (PST days):** 130.6 opened, 97.3 landed, 57.0 closed unlanded a day; 649 open is 6.7 days of
   landing (memory-accounting 14.5 days, network-accounting 13.4, infra 10.7, circuits 8.2, proofs 7.1).

## In progress

Re-head classification (what each re-head after a mark or verdict changed: pure sync of `main`, regenerated locks, or an
edit), the carried/superseded CSV as an art:, and the ranked levers.
