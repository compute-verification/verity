[2026-11-11T20:38Z] e89c0e876dec
ONSITEMAIN to the coordinator, Wed 11 Nov 12:40 PM PST. The NET-330 job is done; the full report is my run's last message and the note `lanes/network-accounting/20261111T2035Z-job-net330-memory-reads.md` (synced).
- Part 1: branch `cursor/net-onsite-memory-reads-84f9`, head `0bc8dda6890c2b76bdbc523d4cdfd44f5535dbdc`, one commit on `9068336929e0`, pushed, no pull request. The record's tests pass on my VM, 280 cases; `of_record` is unchanged. One comment is on NET-330.
- Part 2: proofs' "one member" is right of Monday's draw and wrong of Monday's window. On `main`'s verifier a class window has no accepted sampled-proofs case, with place 2 in or out; only a run from the onsite branch has one. The four answers are in the note.

[2026-11-11T20:39Z] a996abafbaf8
TERMGB to coordinator (network-accounting). Job 6, #8777. Written Wed 11 Nov 12:40 PM PST (read from the clock).

**Head: `7b314b7dcaca7fd2b807da2a563b8fe0e65777b2`** [run]. Two commits on `6dadd19ca332`, pushed as fast-forwards to my branch only: `aa3dfe00f3fa6aaafb60b2861d5ff5c676055a33` (the change, 3 files, 124 added, 33 removed) and `7b314b7dcaca` (two headers I had missed, 3 files, 5 added, 3 removed). PR open, no mark, no label [read].

**The default document, the plan's request count (1.7784e16 requests a world year)** [run, at the head]:

| the scan's term | D_j = 2, the term of record | D_j = 1, "with a set-time card (not yet tested)" | the jitter alone at D_j = 2 |
| --- | ---: | ---: | ---: |
| world's year | 59,645,643.5 GB (198,819 x 300 GB) | 57,798,028.1 GB (192,660 x) | 1,847,615.4 GB (6,159 x) |
| datacenter's year | 15,907.2 GB (53.02 x) | 15,414.5 GB (51.38 x) | 492.75 GB (1.643 x) |

`value` is 59645643.511189334 and `with_set_time_card` is `{"value": 57798028.111189336, "unit": "GB", "label": "with a set-time card (not yet tested)"}`.

**The change, items 1 to 7** [run unless marked]:

1. `document`: `value` is the scan's world-year term at D_j = 2; `name` ends "as the scan's check charges it today at two release schedules (D_j = 2)". The new key is `with_set_time_card`. Rows' keys and numbers stand.
2. `render`: the rule's line is the second line of text, in your words with "a network certifier whose network card". Every D_j = 1 header ends ", D_j = 1, set-time card (not yet tested)": jitter and term in both tables, a link's phase and a request in the bits table, and the term's `x target` and `x weights` in the GB table. Those two named no D_j before; I missed them in the first commit, which is why there is a second.
3. `CHANGES["timing"]` (the `timing` row's text): "the release times taken once, by the compute side's tick: the jitter is 0, which rests on a set-time card, a network card of the network certifier that releases at a set time, and no timing test has shown one". `all three`: "... and the set-time card the timing's change rests on". The docstring and the README say what the card is once each, at first use.
4. The line reads: "With the lengths held by a stop block and the release times taken once by a network certifier whose network card releases at a set time (not yet tested), the world's year meets the target at a failure rate of at most 4.316e-06. Without that card the jitter alone, at D_j = 2, is 1,847,615 GB over the world's year, 6,159 times the target." The two numbers are `value` less `with_set_time_card.value`, and that over the target: a document cut by `--scope` has no world-year row to read. The README's sentence says the same without numbers.
5. README: a new paragraph on `value` and `with_set_time_card`, and the jitter and term bullets, state the rule.
6. Tests: 18 pass (17 before). The assertion near line 365 and the one near line 414 pin the new headline. The new test is `test_the_headline_is_the_scans_world_year_term_at_two_release_schedules_and_d_j_1_carries_its_label`.
7. No date, name, lane, issue, PR number or run id in the tree's text [read: the diff].

**Every term of every row is equal number for number** [run]: the four documents at `6dadd19ca332` and at the head each hold 72 rows, equal field for field in the same order (360 of 360 terms). The fields that differ are `commit`, `name`, `value`, `with_set_time_card` and the legend texts of `timing` and `all three`. In the print, the tables' 144 body lines are equal cell for cell; only padding moves.

It is one commit of substance and not two: the card is named once at first use, and in the docstring and the README the first use is in the rule itself, so the text does not separate from the headline.

**Item 8: where else the certifier's term or K is printed or recorded with the jitter at D_j = 1 only** [read; nothing changed]:

In `benchmarks/network_traces/leak_bounds.py`, the scaling by links:
- line 25, the module docstring's list of K's terms: "Dj = 1, so the jitter term is 0 (assumed, `NetTiming.JitterBounded`)".
- lines 194 and 218: `terms` and `clock_terms` default `dj=1`. `table` passes each D_j itself, so the default reaches only the readers below.
- lines 302-307, `LinkLoad.terms`: calls `terms` with no `dj`. Lines 309-317, `LinkLoad.bits`: K = certifier + entries from it. That K is what `per_link` (378), `datacenter` (390) and `at_link_rate` (427) return as `Bound.bits` and `.share`.

In `benchmarks/network_traces/chain_record.py`:
- lines 804-892, `leak`, the `chain_record.py leak` command (1215-1219, which writes `--out` as JSON and Markdown and prints it): each row's `pooled`, `pooled_share`, `per_period`, `per_period_share` and `per_link_window`, for a link, the datacenter and the world over a week, a month and a year, all through the three functions above. Line 865 records `"jitter": 0, "jitter_term": "assumed (J)"`, and line 875 the condition "the jitter term is 0 at D_j = 1, assumed (J), not met until a run shows ...".
- lines 895-910, `leak_markdown`: the table "K pooled, δ = 1% | K pooled, δ = 1e-6 | K per period | ε | a sampling block", under "(the jitter term 0, assumed (J))".
- line 712, `audit_of`: each compute audit's `"jitter": 0`. Lines 735-750, `pooled_k`: K sums them, with `"jitter_term": "assumed (J)"`; the report prints it at 1145-1147, "K at δ = ...: ... the jitter term assumed (J)".

The period check's own K (Lean, `NetworkCheck/Period.lean`) takes the jitter bound from its case (lines 805-806) and charges ⌈log2 Dj^steps⌉ (1198-1207), so it prints what the case hands it:
- `benchmarks/network_traces/scan_period.py` line 94 hands `"jitter": 1` (its docstring, line 12: "jitter bound 1 (K's jitter term 0, assumed (J), not met at 1)"). `certifier_piece.py` 473-483 builds its case with it.
- Readers of that answer's K: `block_certifier.py` 66-72 (`block_leg`: the block record's `quantity` K and its `at` each δ); `benchmarks/one_stage/interfaces/verity_e2e_demo/block_record.py` 98; `benchmarks/one_stage/onsite_periods.py` 78-93, which prints K for the node, a datacenter and the world over a week, a month and a year.
- The fixture `benchmarks/network_traces/tests/fixtures/certifier-period-answers.json` holds two recorded answers, each with `jitter` 0 in its audits' and scans' terms.

Prose that states it: `benchmarks/network_traces/README.md` 236-239 and 292-293; `verity/protocols/certifier/verifier/PROTOCOL.md` 1064 ("times `D^T` for jitter, 1 under J"). `verity/protocols/certifier/verifier/capacity.py` 44 defaults `jitter_classes=1`; only its tests call it.

Not readers: `benchmarks/certifier_timing/README.md` line 5 already says the jitter is 0 only at D_j = 1. No card under `benchmarks/**/cards` or `verity/hardware/cards` names the jitter. `benchmarks/README.md`'s index states no D_j.

**Item 9: #8779** [read, at its head `9b5d80511bdf`, by `gh pr diff 8779` and its files; not touched]. It reads none of the three: not the certifier document's `value`, not its `name`, not the print's headers. `certifier()` (`benchmarks/harness/verity_bench/exfiltration.py` 329-336) calls `lb.document`, and the code takes only `inputs`, `rows`, `weights.label`, `legend`, `entries_rule` and `phase` (lines 382, 415, 419, 432, 451, 455-456). So nothing in it breaks, and its copy of the legend picks up the two new texts. What it prints at D_j = 1 is its own:
- 441-442: `together[name]` sums `r["term"]["1"]`.
- 538-541: "the network certifier's part: ... GB" from `scan['term']['1']`, ending "each at D_j = 1".
- 581-584: its own table, "term at D_j = 1, GB" first; 588: "The network certifier's term is at D_j = 1".
- its test, 374 and 380-384 (57,798,028; 10,104; 0.0, 0.0) and 428-429; its README, 101-105.

One line for benchmarks: "#8777's new head changes only the certifier document's `value`, `name` and print headers, none of which #8779 reads; #8779's own sums and headline are at D_j = 1 (exfiltration.py 441-442, 538-541, 581-588)."

**The run**: `r20261111-201623-1eff`, at the head, by `research run --on auto` [run]. Launched 12:16 PM PST. Its pool's slots were all held: it was still waiting for one at 12:29 PM PST and was running at 12:39 PM PST, 23 minutes after launch. It has no verdict yet. That is past the 20 minutes, so I end my turn on it. It runs the five suites by name whose inputs the two commits change (`verity-graph`, `verity-network-traces`, `verity-network-e2e`, `verity-network-sim`, `benchmarks/one_stage`) and the nine root files. The PR's body names it as running; its verdict lines go there when it ends. Local, at the head [run]: `test_certifier_term.py` 18 passed, `test_leak_bounds.py` 9 passed, `test_chain_record.py -k leak` 2 passed, ruff clean, the nine root files 112 passed with 8 bookkeeping warnings, none naming a file of this change.

**What I am least sure of**:
1. `timing` and `all three` still print a jitter of 0 in their D_j = 2 columns. The timing change is the set-time card, so under the rule those cells are card figures too. Their texts now say so; their cells are not under the label. `all three`'s term of record prints as 0, where without the card it is 1,847,615 GB over the world's year. You asked for text only there, so I left the numbers.
2. The tables still put the D_j = 1 columns before the D_j = 2 ones. I changed headers only.
3. The schema name is still `certifier-term/v1` though `value` holds another figure. Nothing in the tree reads the name [read: one grep].
4. The new sentence's number is a float difference of the two headline entries. A test holds it to the row's jitter to 1e-9; at an exact rounding boundary its fourth figure could differ from the table's cell.

**Where I did not look**: the writer of the onsite demo's record (`verity/onsite-network-record/v1`) is not in the tree, so I did not see which jitter bound its case hands the check. I found no Python outside tests that writes a clock-form case's `jitter`. I read the Lean only for where the bound enters, not its proofs.

I cancelled my own first run, r20261111-200743-ac04, by `research cancel` with its reason: it was at `aa3dfe00f3fa`, had waited 8 minutes for a slot and had not started. It holds no verdict.

The PR's title and body carry the new headline, the figures above, the two tables with the term of record first, and the run. No mark.

TIMER: none

