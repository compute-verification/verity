# Mark to `main`

This benchmark measures how long the team's own changes wait to land. For each pull request the verified queue landed on
`main` in the last 7 days, it takes the hours from the ready mark at the head that landed to the move of `main` that
holds it. It measures the landing path, not a protocol.

The queue lands plain changes and Lean changes in separate batches, and the benchmark splits the pull requests the same
way. Each class gives four numbers, and the two p90s are the metrics of the goals that a pull request lands within two
hours of its mark (`metrics.toml`):

| Measurement | Unit |
| --- | --- |
| `ci/plain-mark-to-main-p90`, `ci/lean-mark-to-main-p90` | hours, the 90th percentile, with the target of 2 h |
| `ci/plain-mark-to-main-median`, `ci/lean-mark-to-main-median` | hours |
| `ci/plain-landed`, `ci/lean-landed` | the PRs counted |
| `ci/plain-within-2h`, `ci/lean-within-2h` | the share of them on `main` within 2 h of the mark |

`landing.json`, beside the result, holds every landing: the pull request, its head, the `main` it landed on, its mark,
its hours and its class. What counts as a landing, a mark and a Lean change, and what is left out of the count, are in
`landing.py`'s docstring.

## Run it

```sh
research run --project verity --cwd source -- bash benchmarks/landing/run.sh
```

It reads the control pod, GitHub and the evidence store's labels, which an agent VM with the team's keys reaches and a
CPU node does not, so it runs as a local research run from a checkout there. It reads for about a minute and computes
nothing heavy. A run that can't read one of its inputs writes no measurement and fails its validation.

Its tests run the computation on a small fixture, with stand-ins for the pod, GitHub, the store and git, and need
nothing:

```sh
uv run pytest tests/benchmarks/landing/test_landing.py
```
