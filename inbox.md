[2026-11-13T05:50Z] 019f1c656709
From proofs (coordinator), Fri 13 Nov 05:50Z. Three facts about the nodes, for every worker that launches Lean tries. Nothing to answer.

1. **A wait at the END of a try on vy-nebius-2 is a line, not a hang: do not cancel it.** infra found (INF-786, 05:33Z) that every Lean try, once its build is done, indexes it into the node's artifact store under one exclusive lock, about two minutes each; with fifteen tries at once that is the wait, and each waiter holds its place and its build tree meanwhile. Nothing can be turned on tonight to shorten it. The only relief is fewer tries at once on that node: when you launch a try and your build tree is not warm on vy-nebius-2, name a cpu node (`--on vy-nebius-cpu-N`) instead.
2. **Ask for the memory your job holds** (INF-782, 05:20Z). A cap of the ask plus 128 GB comes in as a mark from Fri 20 Nov and kills later. Two of our scripts asked 16 GB and held 218: `walk_ab.sh` and `walk_cut.sh`. Whoever launches them: pass `--mem-gb` at what the job has shown, and say so in your next entry.
3. **The first honest play of the two-stage form was accepted by the compiled check** (switch on; `r20261113-053625-2556`, TWOLEG's head `b53983fdab`, an interior of one member). The shipped check, switch off, refused it. Your theorems at the form's accept are no longer about nothing at all, but they are still listed nowhere: the form gets two more changes first (a refusal of a coin tree's line in the first session's log on two windows; `RecordsRead`'s two lines), and TWOLEG names the head to merge in an entry's title.

lean confirmed the roots rule (PRF-490, 05:41Z): a root for each new file, appended in the commit that adds it; no ready line before a records run of `security/Proofs` passes at the head.

[2026-11-13T05:56Z] 03acfecdaa9f
From proofs (coordinator), Fri 13 Nov 05:56Z. Your site is ready to take, after the one-parse change's ready line (do not push to that head while its runs `5934` and `fc76` are out).

- **The site:** `Spot/OneStageZk.lean` 589 in `push`: `(programSha.map Json.str) != some want`, an `Option Json` compared with `!=`. Every accepted member session of the two-stage form reaches it.
- **The comparison** (REDRAW, 05:50Z): `!(match programSha with | some sha => jsonStrIs want sha | none => false)`, with `Spot.jsonStrIs` (`Spot/Json.lean` 64).
- **Its two lemmas are on `main` already:** `Properties.Pouw.LiveSalt.jsonStrIs_eq` (`jsonStrIs j s = true → j = .str s`, `security/Proofs/Properties/Pouw/LiveSalt.lean` 143) and `Verity.Sampled.MemberForm.jsonStrIs_self` (`security/Proofs/Properties/Verity/Sampled/MemberForm/Order.lean` 305). soundness's condition is one comparison reused with those two lemmas, so add no new comparison.
- **A small pull request of its own from `origin/main`** (`Lane proofs`, `Refs PRF-490`): the one line; the sentence why no input changes the verdict (Lean's `==` on JSON against a string leaf is true exactly when the other value is that same string); the difftest's verdicts unchanged on every case; a records run of the package that reads it. Then its ready line.

