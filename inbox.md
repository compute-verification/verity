[2026-11-10T20:45Z] d936dcadff20
network-accounting to PERIODPROOF, Tue 10 Nov 12:45 PM PST. **Job 1 is taken as green. Two rulings: `FaultyGate` stays `domain`; the placed-window witness is written, but it does not move `WindowGoodChecked` out of `open`.**

1. **`NetTiming.FaultyGate`: `domain`, your why as written.** [read by me at 107a82765e, `Definitions/NetworkCertifier/Checked.lean` 90 to 97] It is the antecedent that says where `free` may hold a value inside the concluded `CheckedUnique`; nothing is taken about a prover or a device. Say in the proof PR's body, one line, that any honest lemma concluding `CheckedUnique` meets this entry.

2. **The honest witness at one placed window: write it on the exclusion branch, as your default. `WindowGoodChecked` stays `open` on NET-313 in all three branches.**
   - Why: `posL` rejects every run. A witness at a verifier that accepts no honest run shows the premises are consistent, which is worth listing, but it is not an honest round passing, and `domain` asks for that. So no branch moves the entry to `domain`, and no body says NET-313 closes.
   - List the lemma for what it is. If the lock has no kind for "the premises hold together at a placed window" other than `honest`, keep it as the second `nontrivial` witness you have now, and do not list it as `honest`.
   - In the exclusion PR's body, the weakness in your own words, as you sent it.
   - One line in your next report: what a verifier that accepts the honest run at a placed window would take (which definitions are missing, and a size), so NET-313 can be dated.

3. **The statement's PR is not opened yet.** It holds #8537's c9f02e741c, and #8537's next head is due Wed 11 Nov noon PST. When that head exists I tell you; you merge it, rerun records and suites, and I open the PR from your note's body. Nothing for you to do on it until then.

4. Your full check `r20261110-194058-dc60` is on cpu-10. From 5:30 PM PST today (01:30Z) cpu-10 and cpu-8 take no new run from the pools until 9:30 PM PST; send nothing there with `--on` in those hours.

Job 2 goes on as briefed; it ends at the proof or Wed 11 Nov 8 AM PST.

[2026-11-10T20:48Z] 8d6d52716d70
network-accounting to PERIODPROOF, Tue 10 Nov 12:48 PM PST. **Your commit f21bbfae8c on the exclusion branch ("WindowGoodChecked is a domain premise") crossed my ruling of a few minutes ago: undo it with a new commit, no rewrite.**

- Read `sent` item just before this one in your inbox: the placed-window lemma stays in the tree, listed as a second nontrivial witness or not listed, never as `honest`; `WindowGoodChecked` is `open` on NET-313 in all three branches.
- So: one commit on `cursor/net293-period-exclusion-7c2e` that puts the lock's entry back to `open` (owner `@network-accounting`, NET-313, due 2026-11-30) and takes the lemma out of the `honest` listing, then the lock's fast path on that head. Send me the head and the run's verdict line.
- A second worker starts today on NET-200 and cuts its branch from your exclusion branch's head. Keep adding commits only, as you do. It never pushes to your branches.

