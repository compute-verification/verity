[2026-11-16T23:59Z] dd7f46290fbf
The grade draft is open: #10665, head 252f9efdc5bc6509c556634787f3c68beac6baf2, from main at 6469852e760 plus nothing. It removes more than it adds: 23 files, +884/−2,961. main's newer commits, up to 220767e4865, touch none of its files.

- S = p × ρ, with ρ labelled in the page's words, as printed: "the bound on the route's own traced step; S = p × ρ". The body quotes the page's definitions of p, ρ and S. It says evaluating ρ on one traced step is the page's method, not a theorem about one step.
- per_period is kept in both the grade and the library, computed from ρ. calculator.periods and calculator.fleets keep their callers.
- p against the page (read; inferred where marked): the code's p is C/H in counted gates, each config's mean over its seeds that reach the gain.
  - `honest` is the config the attack names. All six named honest configs are full Muon with the embedding frozen, the page's reference. The code takes no minimum over honest trainings, so it is the page's "cheapest" only if that config is the cheapest (inferred).
  - `any` is the least C over every config given, the attack included.
  - 6 of 64 configs name `honest`. One that names none has no grade.
  - p is unchanged in this PR.
- Overlap with #10597: none on lines.
- Ran at 252f9efdc5b, fresh: training_slowdown 763 passed; grader 36 passed, 3 skipped; verity-graph 248 passed; root invariants and tests/lint 210 passed; ruff clean.
- Not run: the repository suite whole and the whole check.

#10597's fix is pushed at 23bbfd79a34, in my previous message: four reference.toml tables restored for spot_figures' figure1.py.

