[2026-11-15T00:00Z] 23eea1097105
From the lead, narrowing your task: ci (the tool's owner, 4:00 PM PST) takes the map-committed-with-its-move case itself: restack.py will read the map from the move commit when its parent lacks it. So do NOTHING for 0f6f8e90edf, 3b93447e9e3, 7ef1e26c04e, 35d39c1bdd3. The rows go in tools/move/move_lines.toml (fields line, hand, reason). Your task is only the --no-lock rows for ours among ce0e54799c2, 4f6e3df45fb, 473d8eefa27, verified and pushed. Re-run your command and carry on.

