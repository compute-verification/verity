[2026-11-13T05:23Z] 159b6c301396
Coordinator: ci ratchet-gate on #9403 (e8915b820130): it adds 47 retired-term sites to the files it changes, as tests/lint/test_repo_retired_terms.py counts them (benchmarks/README.md 0->1, benchmarks/served_errors/README.md, ...). Nothing is held; fix it forward: after #9429, on a new branch from #9403 head, reword each site to the Glossary term, run the lint and suite remotely, open with research queue open (Lane compute-accounting in body), mark if green. Report sites before/after and the PR number.

