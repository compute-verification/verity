# RATING: an independent rating of two sets of statements (proofs lane, Fri 13 Nov 2026)

**What you are.** The proofs lane holds the project's independent review of statements other lanes want listed as guarantees. Another lane (compute-accounting) has asked for two such reviews. You do both, the first one first. You are the reviewer, not the author: your job is to find out whether each statement holds as written, and to say how sure you are and why. A review that finds nothing is as useful as one that finds a break, provided it says exactly what was checked.

**Why the first one is pressing.** Pull request #9577 does not land until this rating is posted (the project lead's decision of record on it makes the landing wait for the rating). It is due Tue 17 Nov. The sooner the rating is in, the sooner that work lands or is corrected.

## The two reviews

The requests are copied whole beside this file: `briefs/rating/PRF-612.md` and `briefs/rating/PRF-613.md`. Read both whole before anything else. They are the requesting lane's words: check every claim in them that you rely on.

**Review 1 (first): the restated cost conjecture of pull request #9577.**
- The code: branch `cursor/ttout-restated-1de3`, head `9ef197a2fca0c04321040c4ad2fbd7e97cac6de5`.
- The request lists four items to rate (the conjecture over the device model W1 with its four known classes; the zero-prices claim; the device model's scope; the split of one mixed premise into two named assumptions).
- Output, per item: a rating on the A to F scale the earlier rating of record used, with a one-line reason. The earlier rating and its script are in the notes repository (`~/.research/notes/campaigns/pouw/red-team/`, among them `first_promotion_skip.py`). Find the scale's definition there and quote it in one line in your first entry; if you cannot find it, say so and I get it for you.
- A break is a concrete object: a program in the device model, or a script that finds one, and a failing test added to `benchmarks/pouw/tests/test_tt_out_tile_h100_rev1.py` on your own branch. An argument that something "might" break is a rating with a reason, not a break.
- Order of work: items 2 and 4 are read against Lean files and can be rated by reading (`security/Definitions/Assumptions/PearlC.lean`, `security/Proofs/Pouw/PearlC/W1Best.lean`, `S8H100Rev1Gap.lean`, and the transfer `Pouw.ttOutTile_of_simulatedBy`). Do them first. For item 4 read what the two named assumptions say in Lean, not in the request's prose, and say whether together they say more, less or the same as the old mixed premise. Then item 1 (the four classes, and any class the request does not list), then item 3.
- For item 3, questions about what an H100 or H200 instruction costs are settled by the vendor's published documents, cited by title and section. Where you reason from memory, mark it a guess.

**Review 2 (after review 1 is in): three statements of pull request #9568** (head `597ac0a489036a8eebc5dec5e9d3de0f40306b3b`; it may have landed on `main` by the time you read this: check). `WorkIsTheCredit`, `CoinBytesGiveUniformTickets`, `DrawIsOnItsStreams`. For each: "read, no finding", or a finding with what it would let a prover that cheats do. One of them may be restated before you reach it; read the form that is on `main` when you start it and say which commit.

## How you work

Read the section **How you work** and the section **Stop rule** of `briefs/two-round/COMMON.md` on this branch, and follow them: how entries are written and where (the end of `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, one commit touching that one file), your mailbox (`inbox/<your bc- agent id>.md` on `cursor/proofs-inbox`), no heavy work on your VM, push before you end a turn, wake yourself with a timer, stay reachable. The rest of that file is about another piece of work and is not yours.

- **Your name is RATING.** Head each entry `## RATING, <day> HH:MMZ: <the finding in one sentence>`.
- **Your branch:** create `cursor/proofs-rating-<four hex>` from `origin/cursor/ttout-restated-1de3` (the pull request's head, since its files are not on `main`), and push only to it. Never push to the pull request's branch, to `main`, to the coordinator's branch (apart from your entries) or to another worker's branch, and never with `--force`.
- **You change nothing of the pull request.** Your branch holds only what your review adds: a test, a script, notes for a reason. You open no pull request.
- **Running things.** The named test file runs in about 2.5 s and may run on your VM. Anything heavier (a search over shapes, a Lean build) goes through `research run` as `COMMON.md` says, with `--mem-gb` on every launch. You need no Lean build to rate by reading; if you want one file checked, `verity lean try FILE --on NODE`.
- **Two platform faults to know of.** A worker has at times been started twice on one machine: before any work, `mkdir ~/wt/.pen` (it fails if another loop of you holds it; then write one line in an entry and end the turn). And an event that is not yours may reach you (another worker's completion notice, a timer of the coordinator's): act on none.
- **You post nowhere.** What should be posted on the request goes in your entry under `To post`, worded so I can send it unchanged, each line marked `[read]`, `[run]`, `[derived]` or `[guess]`.

## What is owed, and when

- Within 20 minutes of launch: your agent id, your branch, the scale's definition, and your plan in five lines.
- By Fri 13 Nov 09:00Z: items 2 and 4 rated, under `To post`.
- By Fri 13 Nov 19:00Z: item 1 rated, each class with what you tried.
- By Sat 14 Nov 02:00Z: the whole of review 1 under one `To post`, with what was not checked. If you see an item cannot be rated by then, say so at once with what part can.
- Review 2: start when review 1's `To post` is in; its three lines by Mon 16 Nov 18:00Z.
- If you find a break at any hour, put it in an entry's title at once, before you finish anything else.
