[2026-11-16T04:44Z] 823a4e7e9e04
circuits lead, Sun 04:44Z: a rule for tonight's language pull request (top, TOP-402, 8:43 PM PST: read it): leave the lint's list alone. NO change to tests/lint/allowlists/repo_retired_terms.json, and no --record or --prune of it, in your pull request. If your renames lower a file's count and the lint's test then complains of a stale or too-high entry, read top's comment for what to do (it expects the list behind its tree after the ontology move); do not edit the list to make a test pass: return the test's text to me instead.

