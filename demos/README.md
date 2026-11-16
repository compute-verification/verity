# Demos

A demo is Verity run end to end on real machines. A developer serves a model, the auditor's side challenges it, and
the verifier returns one verdict.

Each demo is a single configuration file. It names the protocols to run and which party runs on which machine, and
that is all: a demo holds no code and is never tuned.

There are two settings. In the **remote** demo the auditor has no hardware at the developer's site. In the **onsite**
demo it does: a network attester on each node's uplink, and a memory attester beside it. Onsite comes in three sizes,
each adding protocols to the one before.

| Demo | Protocols | How far it runs |
| --- | --- | --- |
| `remote` | SPOT, PoUW (Pearl+), inference-only | Not yet: Pearl+ has no party to run. |
| `onsite_network` | SPOT, network attester, memory wipe | Not yet. |
| `onsite_compute` | SPOT, network attester, memory wipe, PoUW, inference-only | Not yet. It is the default. |
| `onsite_full` | SPOT, network attester, memory wipe, PoUW, inference-only, PoUS, PoUB | As a dry run. No run of record yet. |

Only `onsite_full` is the onsite protocol itself. The two smaller sizes run part of it.
[`verity/README.md`](../verity/README.md) says what each protocol tells the auditor.

## Try one

~~~sh
verity demo onsite_full --dry-run
~~~

This resolves the demo's parties and the channels between them, prints them, and starts nothing. It needs no GPU.

`verity demo NAME` runs a demo for real. That takes machines bound to the roles its file names, which the
[`verity` command's README](../tools/verity/cli/README.md) describes.
