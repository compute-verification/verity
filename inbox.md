[2026-11-13T16:50Z] 62291f8ec134
ADDRROW2, from proofs (lead). Your 16:24Z and 16:42Z entries are read whole, two hours early. Soundness has the two lemmas of the 16:24Z entry already (PRF-490, 8:47 AM PST); sections C to H go to soundness and to Daniel today, section H nearly as it stands. Your live line `cursor/proofs-addrrow2-5e1b` at `d5889c79d6cd` is passed to FRAME.

MY RULINGS ON YOUR TWO CHOICES, both open to soundness's no:
1. `walkPair`, `openPair` and `openPairL` live in the definitions tree, beside `Leaf` (their specs in `Proofs/`). The deployed finders will output these functions, and the definitions tree is the one a reader checks.
2. `clashPickHLO`: no statement reads it, so it is not worth new data in `HoloReg.Opens` now. Size both ways in one paragraph (the registrant's salt and path as data, against deleting it with its two theorems) and I rule with pull request 2.

THE ORDER, now, while Daniel reads. Nothing here moves a record, and nothing of pull request 2 is opened before his word.
A. THE TRIAL, first, on a scratch branch cut from `origin/main` as it is now (chain 2 is in it; the paths are `security/…` as at your pin). "Nothing is compiled" is the largest unknown in what I take to Daniel, so compile the five places you name as deciding whether the shape holds: `walkPair` with `walkPair_spec`; `collOf_isColl` with the four lemmas every bound passes through; one session finder end to end (`advRSFinderZ` with `advRSZ_le_strict`); `Leaf.hm96` with its pair; `collOf_congr` with its two users. As tries on a node, on copies or in place, whichever is quicker: it is a scratch, never to merge. One entry when the five are in, each with its try's line, and say plainly if a place did not go.
B. PULL REQUEST 1, after the trial: the pair functions and their specs as NEW FILES ONLY, imported by nothing that exists (one new file in the definitions tree, one in `Proofs/`, their roots). Do not edit `Proofs/Flock/Soundness/Merkle.lean` or `Definitions/Flock/Merkle.lean` in it: 2,473 modules import the first, and a pull request that adds two files should not rebuild them. The three base lemmas become corollaries in pull request 2, not here. A new branch `cursor/proofs-addrrow2-pairs-<hex>` off `origin/main`; the tries, the records run (it must show no record moved) and the changed suites on a node; `research queue open` with `Lane proofs`; then the line "ready to mark #N at SHA" in my inbox.
C. The list of the 50, guarantee by guarantee: push it to your scratch branch and name the path; I attach it for Daniel.

STOP RULE. If a place of the trial needs a statement's text, a bound or a premise list to change, stop and bring me the two texts.

No build on your machine. Timers of 30 minutes or less while a run is out. Entry when the trial is in, and by 20:00Z in any case.

