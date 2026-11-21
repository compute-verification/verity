<!-- A protocol's or subsystem's README, in its own directory: its architecture on one screen. -->
<!-- The title and these four sections, in this order, and no other heading at level 2: at most 70 lines and 6,000 bytes. -->
<!-- No history, dates, results, run ids, lists of refusals or security argument: the guarantee is its Lean statement, named in one line. -->
<!-- What is not yet true is one line beginning "Not yet:". Each cell of the grid is served, exists, removed or never written. -->
<!-- removed: written once and taken out; the line under the grid names the commit whose parent last holds it. -->
<!-- A sub-directory's README is at most 25 lines (what the directory is, one thing to try) or it goes. -->
<!-- Nested code uses ~~~ fences. `tests/lint/test_readme_pattern.py` holds the form; delete these comment lines. -->
# <name>

<What it establishes about the hardware or the computation, and from what; for a subsystem that is not a protocol,
what it is for.> <The second sentence.>

The guarantee: `<its Lean statement>` (`<its file>`).

Not yet: <what is not yet true, in one line>.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the prover | <what it builds and commits> | `<path>` |
| the Lean verifier | <what it decides> | `<path>` |
| the auditor's device | <what it signs> | `<path>` |
| its circuits | <the Definitions it reads> | `<path>` |
| its device code | <what runs on the GPU> | `<path>` |

## What varies

| implementation | <number format> | <GPU architecture> | <codec, store ...> |
| --- | --- | --- | --- |
| <name> | served / exists / removed / never written | ... | ... |

A new implementation supplies: <what it brings>.
Fixed for every implementation: <what none changes>.

## Interfaces

- From the workload: <what it takes>.
- From `run`: <the public parameters, the coins, the order>.
- It gives: <what goes to the policy, to the proof layer>.

## Try it

~~~sh
<one command that needs no GPU>
~~~

<What it prints.>
