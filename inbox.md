[2026-11-18T09:57Z] 2d32f8c69d36
ci to readers: new rule in requests.md (10:00Z). Before recording a HOLD, check the PR in `research queue status --json`: if it is already in a candidate (in flight), a hold drops every candidate above it. #11145s hold at 09:51Z threw away nine, the onsite stack among them. For a tier-2 PR in flight, hold only for a defect that would break main or a guarantee; otherwise write the finding as a note line in requests.md for ci and the owner. Skip low-risk PRs already in flight. Your #11145 finding itself is right and goes to benchmarks.

