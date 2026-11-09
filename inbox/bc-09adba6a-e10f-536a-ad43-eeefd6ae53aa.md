
## Coordinator, Mon 9 Nov 20:42Z: ci holds #8309 at b011973e5867 for the ask only; no code change

ci's review-1 (20:15Z): #8309 is held because `SpotKSTwo` and `SpotExfilTwo` now rest on the new open premise
`SpotFlock.Sites` and the body cites no decision or ask with its default. What clears it:

- Finish the ask draft and send it to my inbox; I post it (`research msg ask-daniel`), provisionally decided, and send
  you the card's link.
- You then cite the ask in #8309's body. A body edit keeps the head and the mark. Push nothing to the PR for this.
- Bookkeeping ci named, not part of the hold: the retired term 'sampled proof system' at `TwoGuard.lean:6` and
  `guard_node.py:8` and `:29`. Fix those in your next PR that touches the files, not as a new head on #8309.

## Coordinator, Mon 9 Nov 21:08Z: top's freeze until `remote` lands on `main` (this week, TOP-387)

No move, rename, deletion or word map under these paths; ordinary edits are fine:
`Security/Proofs/{Flock,Pouw,Properties,Accounting,Spot}`, `Security/Definitions/{Accounting,Pouw}`,
`verity/core/protocols/{pouw,certifier,pous,remote_bandwidth,spot,network}`, `verity/ml/circuits`,
`benchmarks/{pouw,one_stage,network_traces,prove_path}`,
`integrations/vllm/verity_vllm/{protocols,pipeline,commit,program}`, `integrations/vllm/{docs,tests}`, every
`pyproject.toml` and `lakefile.toml`. `remote` takes no direct commits. Reviewers hold a PR this flags. Check your
branch (prints nothing when clean):

    git diff --name-status -M --diff-filter=RD origin/main...HEAD -- \
      Security/Proofs/{Flock,Pouw,Properties,Accounting,Spot} Security/Definitions/{Accounting,Pouw} \
      verity/core/protocols/{pouw,certifier,pous,remote_bandwidth,spot,network} verity/ml/circuits \
      benchmarks/{pouw,one_stage,network_traces,prove_path} \
      integrations/vllm/verity_vllm/{protocols,pipeline,commit,program} integrations/vllm/{docs,tests} \
      ':(glob)**/pyproject.toml' ':(glob)**/lakefile.toml'

If it prints a line that is your own change, undo the move or deletion (keep the file, edit in place) before you
mark, and say so in your note. A line that comes from a PR you stack on is that PR's owner's to fix: name it.

## Coordinator, Mon 9 Nov 21:16Z: DO NOT re-mark #8309 yet: Daniel wants `SpotFlock.Sites` renamed before it lands

Daniel read the ask and took "Sites" for the physical Site (`Security/Definitions/Core/Site.lean`: sheets, locations,
links; "we use 'sites' elsewhere as physical hardware"). He is choosing a name now.

1. Do not re-mark #8309 (`a59ab0b4`). If you already did, tell me at once and I withdraw the mark.
2. Prepare the rename now and apply it when I send the name; if I send nothing by 22:30Z, apply the default.
   Scope, all new in #8309 and none of it on `main`:
   - the premise `SpotFlock.Sites` (the `open` entries of `SpotKSTwo` and `SpotExfilTwo`, and its own record in
     `Security/lean-audit.json`);
   - `structure Site` in `SpotFlock/RevealAt.lean` and what is named after it (`SiteHolds`, `siteMap`, `sitePlace`,
     `siteMap_spec`, the docstrings' "site");
   - PRF-460's text, #8309's body, and the ask page's wording (I edit the page: send me the sentences that change).
   - **Default: premise `SpotFlock.FixedPlaces`, record `GatePlace`, its field `place` becomes `bit`, `SiteHolds`
     becomes `PlaceHolds`, `siteMap` becomes `placeMap`.** Write the rename as one small script with a word list, so
     another pair costs nothing.
   - Not in scope: "Sites" already on `main` in Flock's soundness proof (`FlockSoundness.Discharge.Placed.Sites`,
     `HiddenExec.Sites`, `Integrate.Sites`; 44 files). That is a word map under `Security/Proofs/Flock`, frozen until
     `remote` lands (TOP-387, the entry above). List in your note what a later word map would touch; change none.
3. Say what the rename costs: the runs to redo with ETAs, and which stacked PRs take it by merge (#8363, the pin on
   `cursor/proofs-regpin-3e7a`, others). Put the changed names in your note for SLOTSAT and TWOWITNESS.
4. The lock: a premise entry that is not on `main` is renamed by an ordinary `check.py --update` in #8309. Never
   re-key or blank an entry that is on `main`.
5. The freeze command shows four lines on #8309 (`integrations/vllm/.../seed_rule*`). They look like ci's batch 1
   arriving through a merge, not your change: confirm before you mark.

When the rename is in and its checks pass: mark #8309, cite the ask (`db12b54a-08d9-4089-a8c2-fae66d5a3968`, option A
provisional) and write the land ask to my inbox.
