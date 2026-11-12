[2026-11-12T14:15Z] a293c7930ce6
ci, Thu 12 Nov 6:15 AM PST: infra stored the three more work dirs (CI-526, 6:06 AM PST comment, art: ids there; check ready:57d5b5bb7717, submission s20261112T120249Z-9bcce4, run r20261112-122254-9d96). Infra's reading: the vLLM shard 2/2 fails the same way as Flock's, in another package: the link of pouw-check-difftest has undefined symbols. So it's likely one cause (staged builds without the modules' objects) across Lean packages: cover both in your CI-526 causes (due 2 PM PST).

