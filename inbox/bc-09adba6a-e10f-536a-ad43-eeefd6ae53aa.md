
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
