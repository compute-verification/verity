"""Write the tier-1 re-cut of Mon 9 Nov (TOP-394) to the Goals page.

    python3 goals_tree.py            # dry run: what would change
    python3 goals_tree.py --apply    # write it, then verify
    python3 goals_tree.py --verify   # only verify the page against the tree

Run from /tmp/rchk with PYTHONPATH=tools/research/src RESEARCH_AT_CHECKOUT=1.
Every open goal ends under one of the eleven or under top/parked.
"""
import json
import sys

from research import secret_hint
from research.goals import Route, _clean
from research.slack import RouteError, SlackError

secret_hint.recover()


def D(day: int, hm: str = "18:00") -> str:
    return f"2026-11-{day:02d}T{hm}-08:00"


T394 = "TOP-394"

ELEVEN = [
    "ci/landing-two-hours", "infra/platform", "top/repo-cleaned-up", "vllm/serving-cost-benchmark",
    "benchmarks/proof-system-performance", "accounting/network-glm53", "accounting/compute-small-model",
    "accounting/memory-small-model", "accounting/inference-only-small-model", "demos/remote-demo", "demos/onsite-demo",
]
PARKED = "top/parked"

# New goals. waits: the full list. Fields not given don't exist on a new goal.
NEW = [
    # ---- 1 CI
    dict(id="ci/launch-under-three-minutes", title="A candidate's check launch registers in three minutes", owner="ci",
         due=D(10), tags=["ci"], tier=2, description=(
             "Piece 1b of goal 1 (TOP-394). Test: from a launcher's start to its runs registering takes under 3 minutes "
             "a shard, for every shard of three candidates in a row.\n\n"
             "Where it stands (Mon 9 Nov): each shard's launch stages about 15 GiB of verdict packs on the control pod's "
             "network volume and takes 33 to 37 minutes (INF-657). ci makes the gather once per candidate. infra caps what "
             "a launch gathers (#8506), shrinks the packs at their source and moves the launchers off the pod's network volume.")),
    dict(id="ci/one-landing-path", title="One landing path: drive, next and checked deleted", owner="ci",
         due=D(11), tags=["ci"], tier=2, description=(
             "Piece 1c of goal 1 (TOP-394). Test: drive, `next` and `checked` are deleted on `main`, and every move of "
             "`main` since is by the verified queue's loop.")),
    dict(id="ci/depth-three", title="A validated branch's check launches within ten minutes", owner="ci",
         due=D(12), tags=["ci"], tier=2, description=(
             "Piece 1e of goal 1 (TOP-394), depth 3. Test: a validated branch's candidate launches within 10 minutes of its "
             "ready mark, p90 over a day. It needs 9 to 12 check slots at once; infra's slot plan is due Tue 10 Nov 8 AM PST.")),
    dict(id="infra/check-wait-setup-measured", title="The check's wait and setup time, measured", owner="infra",
         due=D(11), tags=["infra", "ci"], tier=2, description=(
             "Piece 1f of goal 1 (TOP-394). Test: for each check, its wait, setup and run time by task kind, from the "
             "scheduler's records. Monday's sample: slot wait p90 up to 16 minutes, from over-reserved memory (INF-415).")),
    # ---- 2 Platform
    dict(id="infra/message-reaches-agents", title="A message reaches every running agent", owner="infra",
         due=D(11), tags=["infra"], tier=2, description=(
             "Piece 2c of goal 2 (TOP-394). Test: `inbox.py selftest` passes on every coordinator's VM, and the agent "
             "watch places a new worker without infra (INF-658). It follows #8387.")),
    dict(id="infra/infra-repo-lands", title="The infra repository lands without Daniel", owner="infra",
         due=D(11), tags=["infra"], tier=2, description=(
             "Piece 2d of goal 2 (TOP-394). Test: a change to the site or the scheduler reaches the infra repository's "
             "`main` through the queue (INF-639, with ci). Today each change waits for Daniel's merge, up to 92 hours "
             "from its last commit to production.")),
    dict(id="infra/launchers-off-control-pod", title="The queue's launchers run off the control pod", owner="infra",
         due=D(12), tags=["infra"], tier=2, description=(
             "Piece 2e of goal 2 (TOP-394). Test: the launchers run on a CPU node with local disk. The pod (4 CPUs, 16 GB, "
             "a 40 GB root, one network volume) also runs the agent watch, the relays and the store's collector, and on "
             "Mon 9 Nov they collided twice.")),
    dict(id="console/eleven-numbers-live", title="The eleven's numbers live on the console", owner="console",
         due=D(12), tags=["infra"], tier=2, description=(
             "Piece 2g of goal 2 (TOP-394; CON-17). Test: the Metrics tab in production shows each of the eleven's test "
             "values with the commit and run it came from. The first version shows what already prints: CI's two numbers "
             "and the machines' busy share and spend. Each of goals 4 to 9 joins on the day its benchmark prints from "
             "`main`, as JSON (value, unit, commit, run).\n\nIt reaches production only through Daniel's merge of the "
             "infra repository.")),
    dict(id="console/routes-check-caller", title="Every docs-site route checks who calls it", owner="console",
         due=D(13), tags=["security"], tier=2, description=(
             "Piece 2h of goal 2 (TOP-394). Test: one test per route that refuses a token with the wrong issuer, the "
             "wrong audience, an expired date or another team.")),
    # ---- 3 Repository
    dict(id="top/architecture-signed", title="Daniel has signed the target architecture", owner="top",
         due=D(13), tags=["cleanup"], tier=2, waits=["vllm/layout-from-inventory"], description=(
             "Piece 3a of goal 3 (TOP-394). Test: one page that Daniel has signed: the layers, where each kind of code "
             "lives, and the names. It goes to him Tue 10 Nov 10 AM PST, built on vllm's layout read off the component "
             "inventory.\n\nIt answers, for the moves that follow: where GPU kernels live (one library, or each beside "
             "its Definition; kernels needs it by Fri 13 Nov), which Lean names change (lean needs them by Mon 16 Nov), "
             "whether the runtime sits inside `verity`, and one home for outside systems.")),
    dict(id="top/slices-clean", title="Every owner's slice reads clean", owner="top",
         due=D(20), tags=["cleanup"], tier=2, status="off-track",
         waits=["ci/slice-clean", "infra/slice-clean", "network/slice-clean", "proofs/slice-clean", "vllm/slice-clean",
                "circuits/slice-clean", "compute-research/code-at-overhaul-bar", "compute-research/lead-leftovers-deleted",
                "memory-accounting/held-cleanup-after-remote", "proofs/terminology-clean", "cleanup/no-lane-text",
                "training/second-swarm-results"],
         blockers=(
             "Three pieces are dated after Fri 20 Nov because their files are frozen until the remote branch's merge, "
             "which is estimated Tue 17 Nov: network-accounting's slice (Fri 27 Nov), compute-research's leftovers of "
             "the retired PoUW lead (Wed 25 Nov) and memory-accounting's last two changes (two working days after the "
             "merge)."),
         description=(
             "Piece 3b of goal 3 (TOP-394). Test: every lead's slice goal is done: an entry page, honest assumptions, one "
             "name and one definition per concept, no lane text, no dead code. Under frozen paths, deletions wait for "
             "the remote branch's merge and follow it by two days.\n\nAlso here, with no goal of their own: soundness's "
             "one registration log (PRF-268, Fri 13 Nov) and benchmarks' overhaul of `benchmarks/` (five merges and the "
             "owners' deletions).")),
    dict(id="top/unrun-code-archived", title="What none of the eleven runs is archived", owner="top",
         due=D(20), tags=["cleanup"], tier=2, description=(
             "Piece 3c of goal 3 (TOP-394). Test: every directory that none of the eleven's tests runs is in `archive/` "
             "or deleted, after Daniel has seen the list. Each lead lists its own on TOP-394 by Thu 12 Nov 10 AM PST.\n\n"
             "Known now: `examples/h200`, the frozen-shape demo on seeded weights that the served remote demo replaced, "
             "with about 13,000 lines of kernels only it loads. compute-accounting makes that move once the remote "
             "branch has landed, since one of its importers is under the freeze.")),
    dict(id="top/tree-in-signed-layout", title="The tree is in the signed layout", owner="top",
         due=D(20), tags=["cleanup"], tier=2,
         waits=["top/architecture-signed", "cleanup/one-definition-per-concept", "kernels/code-at-overhaul-bar",
                "training/code-in-final-places"],
         description=(
             "Piece 3d of goal 3 (TOP-394). Test: the moves the signed layout names have landed, each certified by the "
             "move check. No tree-wide move lands on `main` until the remote branch's merge count is zero: each one adds "
             "15 to 134 conflicted paths to it (vllm, rehearsed Mon 9 Nov).\n\nDated so far: lean's two moves (the "
             "Glossary's words, and `security/` at the root) Wed 18 Nov; network-accounting's and training's Fri 20 "
             "Nov; kernels' Fri 20 Nov if the layout's two answers reach it by Fri 13 Nov, and Wed 25 Nov otherwise.")),
    dict(id="console/unposted-pages-hidden", title="Unposted protocols' pages leave the public docs", owner="console",
         due=D(19), tags=["security"], tier=2, description=(
             "Under goal 3's export (TOP-394). Test: an anonymous request to each of the five pages gets sign-in, not the "
             "page. It waits on Daniel's yes.")),
    dict(id="compute-research/paused-work-parked", title="Paused work is parked where it resumes", owner="compute-research",
         due=D(11), tags=["compute-accounting"], tier=2, description=(
             "Piece 3g of goal 3 (TOP-394). Test: hardness and silicon logging are each one branch at one named commit, "
             "bundled in the store, under one note that says where it stopped and the command that resumes it. A note "
             "and two artifact ids, not a landing: nothing of either goes on `main` while paused.")),
    # ---- 4 Serving cost
    dict(id="vllm/serving-cost-design", title="The serving-cost benchmark's design, one page", owner="vllm",
         due=D(11), tags=["vllm"], tier=2, description=(
             "Piece 4a of goal 4 (TOP-394). Test: one page: the one number, one arm for each change we make to vLLM, the "
             "loads, and what it prints. It says which of the arms other leads asked for are in: through the network "
             "certifier with its release interval stated, under PoUB, and split by kernel and host stage.")),
    dict(id="vllm/serving-cost-glm53", title="One command prints GLM-5.3's serving cost", owner="vllm",
         due=D(13), tags=["vllm"], tier=2, waits=["vllm/serving-cost-design"], description=(
             "Piece 4b of goal 4 (TOP-394). Test: goodput under `vllm bench serve`, ours ÷ stock at the same load and "
             "latency target, with each arm's share. Today, by our own client, tokens a second cost −0.1% [−2.1, +2.0] "
             "with the serving state committed every 30 s. Nothing is measured with Thursday's commitments on.")),
    dict(id="compute-accounting/commit-overhead-measured", title="The Commit's share of the served step, measured",
         owner="compute-accounting", due=D(13), tags=["compute-accounting"], tier=2, description=(
             "Piece 4d of goal 4 (TOP-394; CMP-149, Daniel's ask). Test: the ratio printed, from the four-arm A/B on one "
             "H200. Cutting it is goal 4's, so this only measures and hands the levers to vllm. Today, with everything the "
             "remote demo turns on, serving the small model costs about 304× stock, 98% of it before the first token.")),
    # ---- 5 Proof-system performance
    dict(id="benchmarks/prove-one-command", title="One command benchmarks a proof system", owner="benchmarks",
         due=D(12), tags=["proofs"], tier=2, description=(
             "Piece 5a of goal 5 (TOP-394). Test: `verity bench prove --system flock --workload glm53` prints each block "
             "type's ms, prover GPU-seconds a token, verifier seconds a token and proof bytes. It reads labelled runs "
             "only, and marks each input measured, modeled or assumed. The same interface takes `--system spot`.")),
    dict(id="benchmarks/spot-applied", title="SPOT applied: the command prints goal 5's number", owner="benchmarks",
         due=D(13), tags=["proofs"], tier=2, waits=["benchmarks/prove-one-command", "proofs/spot-proof-server-steady-state"],
         description=(
             "Piece 5c of goal 5 (TOP-394). Test: the command prints prover GPUs per serving GPU from 5a, the "
             "steady-state runs and proofs' sampling plan of record (today q = 5.04e-6, case R6x4).")),
    # ---- 6 Network on GLM-5.3
    dict(id="proofs/network-number-printed", title="One command prints GLM-5.3's world-year number", owner="proofs",
         due=D(12), tags=["proofs"], tier=2, description=(
             "Piece P1 of goal 6 (TOP-394). Test: benchmarks' one command prints GLM-5.3's world-year GB at 10%, each "
             "input labelled measured, modeled or assumed, with network's term beside the compute cut. Today: 8,456 GB, "
             "modeled, from a branch (PRF-456).")),
    dict(id="network/term-in-the-number", title="The certifier's term printed in GB", owner="network-accounting",
         due=D(12), tags=["network-accounting"], tier=2, description=(
             "Piece 6a of goal 6 (TOP-394; NET-298). Test: the benchmark's command prints the certifier's term for "
             "GLM-5.3 at the world's load, in GB, by the period check's own counting rule. Today it prints no bound.")),
    dict(id="network/certifier-term-zero", title="The certifier's term is zero without a failure",
         owner="network-accounting", due=D(20), tags=["network-accounting"], tier=2, description=(
             "Piece 6c of goal 6 (TOP-394). Test: the certifier's GB over the world's year, against its share of the "
             "300 GB, is zero unless a connection visibly fails. 300 GB allows 1 bit in 7,400 requests, and today's "
             "rule charges at least 6 bits a connection, so this is research, not tuning. Three parts:\n\n"
             "- **Response length held to the output count, failures charged as they happen** (NET-200): the check's "
             "`schedules` and `cells` terms are 0 on a run where no connection fails. Statement Thu 12 Nov; proof in "
             "review Fri 20 Nov.\n"
             "- **Timing taken once** (NET-296): the check's jitter term is 0, and a benchmark shows the certifier's "
             "release time takes one value an interval whatever the server does. Benchmark on `main` Fri 13 Nov.\n"
             "- **Entries charged through the compute audit** (NET-297): the entries term is the tied audit's pooled "
             "charge. In review Tue 17 Nov.\n\n"
             "The first dated step of each is firm. The later dates are guesses until Wed 11 Nov 6 PM PST.")),
    # ---- 7, 8, 9
    dict(id="compute-accounting/bench-on-main", title="One command prints compute's number from main",
         owner="compute-accounting", due=D(12), tags=["compute-accounting"], tier=2, status="off-track",
         blockers=(
             "It holds for Thu 12 Nov only if Pearl+'s verifier (`served_passes.py verify` and the Lean "
             "`pouw-pearlplus`) is on `main` by Wed 11 Nov, through the merge's slices or ported by compute-accounting. "
             "Otherwise it prints from the remote branch on Thursday, and from `main` Wed 18 Nov at the latest."),
         description="Piece 7a of goal 7 (TOP-394). Test: `verity bench compute` prints goal 7's number for one served "
                     "run from `main`."),
    dict(id="memory-accounting/bench-command", title="One command prints memory's number from main",
         owner="memory-accounting", due=D(12), tags=["memory-accounting"], tier=2, description=(
             "Piece 8a of goal 8 (TOP-394). Test: one command on `main` runs a PoUB session on a node serving Llama 3.2 "
             "1B, verifier on the auditor's host, and prints goal 8's number as text and JSON, with PoUS's certified "
             "share and the wipe's residue beside it. It holds if the eight-card RTX PRO 6000 node and the auditor's "
             "host are memory-accounting's on Wed 11 Nov (INF-660); without them the command lands Thursday and prints "
             "nothing new.")),
    dict(id="training/check-reads-served-record", title="The check reads what the serve registers", owner="training",
         due=D(13), tags=["training"], tier=2, description=(
             "Piece 9b of goal 9 (TOP-394; PRF-487). Test: the Lean check accepts a served 1B window's record and "
             "refuses it with one weight unregistered. Today it can't read that record: the runs' accept is Python's "
             "count. It is a new module on `main` that reads `Spot.Wires`. training drafts it, marked Fri 13 Nov; "
             "network-accounting reviews; lean proves.")),
    dict(id="vllm/serve-registers-weights", title="The serve registers every weight before the run", owner="vllm",
         due=D(12), tags=["vllm"], tier=2, description=(
             "Piece 9d of goal 9 (TOP-394), also needed by goal 10. Test: a record at the start of the deployment's log "
             "registers every weight. Today the log's first record registers only Pearl's tables. About 80 lines; it "
             "goes into the rehearsal's serve on Thu 12 Nov.\n\nIts other half is proofs': the gate's period line signs "
             "the policy and the digest of the weights' registration before the serve, and refuses to open a period "
             "without them. Marked Wed 11 Nov 10 AM PST.")),
    # ---- 10
    dict(id="vllm/remote-rehearsal", title="The remote demo's rehearsal, every verifier on", owner="vllm",
         due=D(14), tags=["remote-demo"], tier=2, waits=["vllm/serve-registers-weights"], description=(
             "Piece 10g of goal 10 (TOP-394). Test: one run of Llama 3.2 1B from the remote branch on one H200, the "
             "weights registered and the limits fixed before the serve, and every verifier run on its record. Served "
             "Thu 12 Nov; its verdicts are read by Sat 14 Nov if the audit still takes 42 hours, and what rejects goes "
             "on goal 10 that day. No run has been rehearsed since Fri 6 Nov's, whose audit rejected.\n\n"
             "The audit's time: Friday's took 53 h 19 min (39.8 h proving 561 sessions one at a time, 7.2 h of Lean "
             "checks after, 4.9 h writing the record). Sessions side by side derive to about 20 h on one node and 16 h "
             "on four CPU nodes; two at once are measured Wed 11 Nov. As the firewall rule stands, sessions of an audit "
             "of record go one at a time.")),
    # ---- the tier-1 goals that are new
    dict(id="infra/platform", title="2 Platform: runs start, agents coordinate, security final", owner="infra",
         due=D(13), tags=["infra"], tier=1, status="off-track",
         waits=["infra/runs-start-or-say-why", "infra/security-posture-finalized", "infra/message-reaches-agents",
                "infra/infra-repo-lands", "infra/launchers-off-control-pod", "comms/agents-coordinate",
                "console/eleven-numbers-live", "console/routes-check-caller"],
         blockers=(
             "The security posture can't be whole by Fri 13 Nov: Ubuntu's fixed kernel is not released (INF-416), three "
             "fixes need Daniel's accounts, and signed review verdicts are ci's for Mon 16 Nov. The other pieces hold "
             "their dates (infra, Mon 9 Nov)."),
         description=(
             "Goal 2 of the eleven (Daniel, Mon 9 Nov; TOP-394). Done when all of these are on `main`:\n\n"
             "- a run started with `research run --on auto` starts within 10 minutes, or is refused with the reason;\n"
             "- the security posture is final;\n"
             "- agents coordinate without Daniel: a message reaches a running agent on every VM, and requests between "
             "leads close on their Linear record;\n"
             "- the infra repository lands through the queue, without Daniel's merge.\n\n"
             "Where it stands (Mon 9 Nov): no command prints how long a run waits to start. 59% of answered asks are "
             "answered on their Linear record; the target is 90%. The queue's launchers share the control pod with the "
             "agent watch and the relays.")),
    dict(id="vllm/serving-cost-benchmark", title="4 Benchmark: serving cost of our vLLM changes", owner="vllm",
         due=D(13), tags=["vllm"], tier=1,
         waits=["vllm/serving-cost-design", "vllm/serving-cost-glm53", "compute-accounting/commit-overhead-measured"],
         description=(
             "Goal 4 of the eleven (Daniel, Mon 9 Nov; TOP-394): on the critical path of everything, designed carefully "
             "and modular, as part of the vLLM refactor.\n\n"
             "Done when one command on `main` prints the serving cost of our changes to vLLM: goodput under vLLM's own "
             "`vllm bench serve`, ours ÷ stock at the same load and latency target, with each change's share as one arm.\n\n"
             "Where it stands (Mon 9 Nov):\n"
             "- GLM-5.3, by our own client: tokens a second cost −0.1% [−2.1, +2.0] with the serving state committed "
             "every 30 s. Nothing is measured with Thursday's commitments on.\n"
             "- Llama 3.2 1B with everything the remote demo turns on: 276× per output token on one H200, measured on "
             "the remote branch. The same command prints it once the merge's shared slice is on `main`.\n\n"
             "Target: on GLM-5.3, serving's share of the 10% compute budget (proofs, PRF-456: 4.34 of the 10 "
             "points at today's figure, modeled or priced by part, with nothing measured on GLM-5.3 against stock vLLM "
             "as a whole). The small model's number is printed without a target until the "
             "Commit's share is split out.")),
    dict(id="benchmarks/proof-system-performance", title="5 Benchmark: a proof system's performance", owner="benchmarks",
         due=D(13), tags=["proofs"], tier=1,
         waits=["benchmarks/prove-one-command", "proofs/spot-proof-server-steady-state", "benchmarks/spot-applied",
                "benchmarks/one-harness", "proofs/spot-verifier-cost", "circuits/verifier-templates-not-unrolled"],
         description=(
             "Goal 5 of the eleven (Daniel, Mon 9 Nov; TOP-394): generic, applied to Flock and SPOT, modular, at very "
             "high code quality.\n\n"
             "The number is **prover GPUs per serving GPU for GLM-5.3 under SPOT at the sampling rate of record**. Done "
             "when one command on `main` prints it from labelled runs, each input marked measured, modeled or assumed.\n\n"
             "Where it stands (Mon 9 Nov): 0.54, modeled (BENCH-43). 23% of proving time is priced from steady-state "
             "runs. At 0.54, proving alone is about five times the whole 10% budget. Most of the gap is the unmeasured "
             "FP8 GEMM, so the steady-state runs may move the number a lot either way.\n\n"
             "Target: proofs' share of the 10% compute budget (PRF-456).\n\n"
             "Asked under it, not yet dated (proofs, by Tue 10 Nov 6 PM PST): one-stage splitting a stratum across "
             "statements, and a far cheaper send check in the ZK server. Pearl+'s whole-run proof waits on both.")),
    dict(id="accounting/network-glm53", title="6 Network accounting on GLM-5.3", owner="proofs",
         due=D(13), tags=["proofs", "network-accounting"], tier=1, status="off-track",
         waits=["proofs/network-number-printed", "proofs/spot-128k-attention-proved", "proofs/spot-ten-percent",
                "proofs/spot-fused-cut-served", "proofs/spot-output-accounting", "proofs/spot-two-stage-e2e",
                "proofs/spot-post-ready", "proofs/sampling-generic", "network/term-in-the-number",
                "network/certifier-term-zero", "network/bound-covers-continued", "circuits/glm-compile-cost-measured"],
         blockers=(
             "The 300 GB target has no date. The compute cut alone is 8,456 GB, modeled; proofs dates its path Tue 10 "
             "Nov 10 AM PST. The certifier's term has no bound today: it has to be zero by design (lengths held, timing "
             "taken once, entries charged through the compute audit), which network-accounting calls research and "
             "dates Wed 11 Nov 6 PM PST. Timing decides it: one bit per 100 ms interval is 1.85 million GB a "
             "world-year, 6,200 times the target, and the best code-only fix measured leaves 92 times. Zero needs a "
             "network card on the auditor's host that sends at a set time, which none of our nodes can test "
             "(INF-664; infra answers Tue 10 Nov 6 PM PST). The number itself prints Thu 12 Nov and is on `main` "
             "Fri 13 Nov."),
         description=(
             "Goal 6 of the eleven (Daniel, Mon 9 Nov; TOP-394). It is the SPOT post's result.\n\n"
             "Done when one command on `main` prints GLM-5.3's **world-year exfiltration capacity at 10% "
             "all-things-considered compute cost**, under **300 GB**, with both of its terms: the compute audit's (the "
             "compute cut) and the network certifier's. Each input is marked measured, modeled or assumed. The world is "
             "`fleet.world-h200`: 46.87 M eight-GPU machines and 1.78e16 GLM-5.3 requests a year, so 300 GB is 1 bit in "
             "7,400 requests.\n\n"
             "Where it stands (Mon 9 Nov):\n"
             "- **The compute cut: 8,456 GB**, modeled, from a branch, 28 times the target. It is proportional to the "
             "assumed serving utilization, which nothing measures yet for GLM-5.3 at 128k: 4,228 GB at 0.2, 8,456 GB "
             "at our 0.4, 12,684 GB at 0.6. The priced path (proofs, PRF-456): 1,446 GB with the fused "
             "per-expert cut served, 543 GB with warm served blocks, 427 GB with every other priced lever at its "
             "goal. **No priced change closes the last 1.42×.** One more lever, cutting the hashing inside proofs, has "
             "no owner and no price yet (kernels prices it Wed 11 Nov 10 AM PST).\n"
             "- **The certifier's term: no bound.** Today's counting rule charges at least 6 bits a connection, which "
             "puts the end-to-end number at 53,000 TB or more, nearly all of it the charge for response lengths, until "
             "lengths and timing are held to the program's outputs.\n"
             "- **The 10% is not yet all-things-considered.** Missing: the certifier's machines, the 128k replay form, "
             "a measured serving cost (goal 4) and the network check's own proving.\n"
             "- **The guarantee covers no token past 2,048 keys** until the long DSA row's units are proved: 82.5% of "
             "the 128k plan's computed tokens.\n\n"
             "**The network check for GLM-5.3 is chosen** (network-accounting, Mon 9 Nov 8:06 PM PST; NET-295), by "
             "reading and arithmetic, with nothing run. It checks the network certifier's records with two head "
             "checks and one new unit a phase that holds every connection's length to the committed output count "
             "(NET-300), and ties content by blocks in the compute audit's sample. Its estimated cost is 0.3% of "
             "serving, and every request waits the certifier's fixed delay before its first token. The scan stays "
             "the onsite demo's check. A GLM-5.3 run has no network verdict from the proof system today. The dates "
             "of this check's pieces come Wed 11 Nov 6 PM PST, after three answers: soundness on whether the content "
             "tie may be sampled (Tue 10 Nov 10 AM PST), infra on the auditor's host (Tue 6 PM PST) and vllm on "
             "GLM-5.3 behind the certifier at 128k (Wed 6 PM PST).")),
    dict(id="accounting/compute-small-model", title="7 Compute accounting on a small model", owner="compute-accounting",
         due=D(12), tags=["compute-accounting"], tier=1, status="off-track",
         waits=["compute-accounting/bench-on-main", "compute-accounting/pearlplus-tiles-proved"],
         blockers=(
             "Pearl+'s verifier and its Lean checker are not on `main`. The number prints Thu 12 Nov from the remote "
             "branch, and from `main` when the merge's shared slice and goal 7's slice land (the shared slice is marked "
             "Wed 11 Nov at the earliest), or sooner if compute-accounting ports them. With nothing ported and the whole "
             "merge on Tue 17 Nov, it is Wed 18 Nov."),
         description=(
             "Goal 7 of the eleven (Daniel, Mon 9 Nov; TOP-394), on Llama 3.2 1B.\n\n"
             "The number is **the share of the served model's work that Pearl+ credits**: credited multiply-accumulates "
             "over the model's, for the tokens served (linears, attention and the LM head). Done when `verity bench "
             "compute` prints it for one served run from `main`. Target: 97%.\n\n"
             "Where it stands (Mon 9 Nov): Pearl+ credits 98.51% of the declared linears on run 614b. Attention and the "
             "LM head aren't declared yet, so the share of all of them is lower; Thursday's print says how much.\n\n"
             "Printed beside it, with no date: the share of the H200's capacity, 7.5e-6 on 614b. It is set by serving "
             "overhead (goal 4's number) and by tensor cycles the small model leaves idle, not by accounting.\n\n"
             "Pearl+'s check replays the provider's witness today. One tile proved instead is Fri 20 Nov; the whole run "
             "waits on two pieces of proofs' under goal 5.")),
    dict(id="accounting/memory-small-model", title="8 Memory accounting on a small model", owner="memory-accounting",
         due=D(12), tags=["memory-accounting"], tier=1, waits=["memory-accounting/bench-command"],
         description=(
             "Goal 8 of the eleven (Daniel, Mon 9 Nov; TOP-394), on a node serving Llama 3.2 1B.\n\n"
             "The number is **the share of the node's memory read rate that an accepted PoUB session leaves "
             "uncertified**, at the data sheet's read bandwidth, over a store that spans every tier. Lower is better, "
             "and 1 or more certifies nothing. Done when one command on `main` runs a PoUB session with the verifier on "
             "the auditor's host and prints it, with PoUS's certified share and the wipe's residue beside it. Target: "
             "0.06.\n\n"
             "Where it stands (Mon 9 Nov): 1.039 from `main`, the 28 Oct session over a 2 GiB test store. Best on "
             "record: 0.150, on 15 Oct over all eight cards and host RAM. Nothing of memory's has run from `main` since "
             "28 Oct.\n\n"
             "It needs the eight-card RTX PRO 6000 node for about six hours on Wed 11 Nov and the auditor's host "
             "(INF-660).\n\n"
             "Not yet dated: PoUB in the served path over a store that spans the node (memory-accounting dates it Thu "
             "12 Nov, from the first print), and a session whose openings and plans are proved in zero knowledge, not "
             "sent (sized Wed 11 Nov).")),
    dict(id="accounting/inference-only-small-model", title="9 Inference only on a small model", owner="training",
         due=D(12), tags=["training"], tier=1, status="off-track",
         waits=["training/inference-only-benchmark", "training/check-reads-served-record", "vllm/serve-registers-weights",
                "training/onsite-slowdown-options", "training/day-bound-on-main", "training/remote-slowdown-sampled",
                "lean/inference-only-on-served-traffic"],
         blockers=(
             "The test can't be met as written this week. Lean's check reads a Program with its weights registered, and "
             "the serve registers none (PRF-487). So on Thu 12 Nov the command prints Lean's refusal of the served "
             "record, Python's count on the same record, and a bound of 1. Lean can accept once the check reads what "
             "the serve registers (marked Fri 13 Nov), the serve registers every weight (Thu 12 Nov) and the auditor "
             "holds the policy before the serve (marked Wed 11 Nov). The proof on the served record is lean's, marked "
             "Wed 25 Nov."),
         description=(
             "Goal 9 of the eleven (Daniel, Mon 9 Nov; TOP-394), on Llama 3.2 1B: a benchmark of the actual proof of "
             "the policy, its cost and its bound.\n\n"
             "Done when `verity bench inference_only` on `main` prints, at the served class, the Lean check's verdict "
             "on the record of one served run, its cost (time, memory, description size) and the slowdown bound. Target: a "
             "bound of at least 1,000.\n\n"
             "Where it stands (Mon 9 Nov): the bound is 1. A block may read 4.2e8 bits beyond the registered weights, "
             "and at that size the bound of record allows more training multiplications a block than the block's own "
             "(a first calculation; training confirms it Tue 10 Nov 10 AM PST). With what a block reads cut to 1e6 "
             "bits, the same bound gives about 1,600.\n\n"
             "Cost means the check in the clear. The check inside a SPOT proof is not built; compute-research sizes it "
             "by Thu 12 Nov.\n\n"
             "Waits on Daniel: a count or a rate, and Boolean gates or their widths (Tue 10 Nov 10 AM PST).")),
    dict(id=PARKED, title="Parked by the re-cut of Mon 9 Nov", owner="top", due=D(30), tags=["cleanup"], tier=3,
         waits=["accounting/silicon-logs-all-compute", "accounting/useful-weights-hard", "training/slowdown-across-scale",
                "proofs/zero-knowledge", "accounting/h200-97-percent"],
         description=(
             "Not one of the eleven. These goals are paused (Daniel, Mon 9 Nov: no new work on training's attack runs, "
             "hardness training, the chip design or new Pearl+ research) or parked by their owners until the eleven "
             "land. Each keeps its own description, and its owner writes where it stopped. Reviewed Mon 30 Nov.\n\n"
             "Also parked, with no goal: infra's one interface per compute job (INF-511), the autoscaler (INF-405) and "
             "drive's eight nodes (INF-615); kernels' served kernels for the B300 (HASH-95); lean's one formalism, "
             "which follows the split, and the rest of one definition per concept; the Lean proofs of the scan's tight "
             "figure, until goal 11 is on `main`.")),
]

SPLIT_DESCRIPTION = """By **Fri 20 Nov 6 PM PST**, the public parts of Verity are published as a public monorepo with a fresh history, in the signed layout. It is exported one way from this private repository, which stays the only source of truth: one queue, one check, nothing merged back. Daniel moved it from Sun 15 Nov on Mon 9 Nov, so the layout is published once.

What is public (Daniel, 4 and 9 Nov): SPOT on Flock, the vLLM integration, the firewall and the recursion, the network certifier (NET-283), the silicon semantics, the Pearl+ served path, and the Lean behind what is published. Withheld until their posts: PoUS, the two bandwidth audits and NCI. Left out of the export and kept in Verity: the private circuits' universal unit and the training frontend. Never exported: `infra/`, `tools/research`, `tools/agent-guard`, `.agents/`, and the evidence store's records. The license is MIT OR Apache-2.0. Outside PRs are welcome, and agents merge them through this repository and a re-export.

Done when, at `main`'s tip:
- `public_boundary_allowed.txt` holds no edge except the ones Daniel keeps, and the boundary check reads every path form (CI-497: today public `verity_vllm/config.py` reaches four private directories by a form it doesn't read);
- `public_export.py --build` builds clean on a node, with no lane text or `art:` ids on the public side, and the Lean check passes on the exported tree (8 of 16 packages pass today; lean, Tue 17 Nov);
- the exported locks name only what the export holds (LEAN-230);
- the public root has a README and LICENSE files.

Next:
- **Thu 19 Nov 6 PM:** a rehearsal export of `main`'s tip builds clean, with the scanner's two misses fixed.
- **Fri 20 Nov:** the final export. Daniel creates the public repository (not named `verity`; empty, public, with the Cursor GitHub app). Published by 6 PM, and `verity` goes private.
- After that: one export a day, each a commit on top of the last. The split freezes no landing.
"""

REPO_DESCRIPTION = """Goal 3 of the eleven (Daniel, Mon 9 Nov; TOP-394). It takes over "the repository dramatically cleaned up" (Daniel, Wed 4 Nov) and holds the public split, which moved to Fri 20 Nov so the layout is published once.

Done when:

1. Daniel has signed the target architecture.
2. The tree is in that layout.
3. Every owner's slice reads clean, and what none of the eleven's tests runs is in `archive/` or deleted.
4. The public parts are exported to a public repository with a fresh history, and the check passes there.

Where it stands (Mon 9 Nov): the architecture page goes to Daniel Tue 10 Nov 10 AM PST, built on vllm's layout. No tree-wide move lands on `main` until the remote branch's merge is finished (433 files open, estimated Tue 17 Nov), because each move adds 15 to 134 conflicted paths to it. Paused work's code does not land.
"""

REPO_BLOCKERS = (
    "Not all of it holds Fri 20 Nov. Cleanup under the frozen paths follows the remote branch's merge (estimated Tue 17 "
    "Nov): network-accounting's compile slice is Fri 27 Nov, and memory-accounting's last two changes come two working "
    "days after the merge. The export's date holds: what is dead and still frozen is left out of the export and deleted "
    "after the merge.")

# Edits to goals that exist. set: fields replaced. waits: the full list. add/drop/swap: changes to the current list.
EDIT = [
    # the tier-1 goals that exist
    dict(id="ci/landing-two-hours", set=dict(title="1 CI: validated branches merge within two hours", tier=1),
         waits=["ci/fine-grained-check", "ci/launch-under-three-minutes", "ci/one-landing-path", "infra/full-check-45-min",
                "ci/depth-three", "infra/check-wait-setup-measured", "proofs/proof-check-fast", "lean/prs-carry-no-records",
                "infra/nightly-uncached-check"]),
    dict(id="top/repo-cleaned-up", set=dict(title="3 Repository: signed layout, cleaned up, public", tier=1, due=D(20),
                                             status="off-track", description=REPO_DESCRIPTION, blockers=REPO_BLOCKERS),
         waits=["top/architecture-signed", "top/tree-in-signed-layout", "top/slices-clean", "top/unrun-code-archived",
                "top/public-split-sunday", "compute-research/paused-work-parked", "training/attack-search-one-command"]),
    dict(id="demos/remote-demo", set=dict(title="10 Remote demo: one run, every check accepting", tier=1, due=D(20)),
         drop=["accounting/h200-97-percent", "benchmarks/remote-numbers-on-main"],
         add=["soundness/one-circuit-per-day", "vllm/remote-rehearsal", "lean/inference-only-on-served-traffic"]),
    dict(id="demos/onsite-demo", set=dict(title="11 Onsite demo: all four protocols accept", tier=1, owner="vllm"),
         add=["demos/remote-demo", "network/record-at-tight-figure"], drop=["network/tight-figure-proved"]),
    dict(id="network/tight-figure-proved", drop=["network/record-at-tight-figure"]),
    dict(id="top/parked", add=["network/tight-figure-proved"]),
    # 1
    dict(id="infra/full-check-45-min", set=dict(owner="ci")),
    dict(id="circuits/glm-long-dsa-row", swap={"ci/landing-two-hours": "ci/fine-grained-check"}),
    dict(id="circuits/glm-served-calls-fit-block", swap={"ci/landing-two-hours": "ci/fine-grained-check"}),
    dict(id="circuits/glm-sparse-attention-zk", swap={"ci/landing-two-hours": "ci/fine-grained-check"}),
    dict(id="circuits/verifier-templates-not-unrolled", swap={"ci/landing-two-hours": "ci/fine-grained-check"}),
    dict(id="circuits/glm-top-p-zk", swap={"ci/landing-two-hours": "ci/fine-grained-check"}),
    dict(id="infra/runs-start-or-say-why", drop=["ci/landing-two-hours", "proofs/proof-check-fast"]),
    # 2
    dict(id="comms/agents-coordinate", unset=["tier"]),
    # 3
    dict(id="top/public-split-sunday", set=dict(title="The public export, checked and then published", due=D(20),
                                                 description=SPLIT_DESCRIPTION),
         add=["console/unposted-pages-hidden"]),
    dict(id="compute-research/lead-leftovers-deleted", swap={"demos/onsite-demo": "demos/remote-demo-on-main"}),
    # 5
    dict(id="proofs/spot-proof-server-steady-state", set=dict(owner="kernels")),
    # 6
    dict(id="proofs/spot-128k-attention-proved",
         add=["vllm/served-128k-window-committed", "circuits/glm-long-dsa-row", "circuits/glm-served-calls-fit-block",
              "circuits/glm-sparse-attention-zk", "circuits/glm-top-p-zk"]),
    dict(id="proofs/spot-output-accounting", add=["network/cap-certified"]),
    dict(id="proofs/spot-two-stage-e2e", add=["vllm/two-stage-schedule-committed"]),
    dict(id="network/bound-covers-continued", add=["network/carry-edge-rule"]),
    dict(id="proofs/spot-post-ready", unset=["tier"]),
    dict(id="proofs/sampling-generic", unset=["tier"]),
    # 10
    dict(id="accounting/inference-only", drop=["vllm/daily-inference-block-counts"]),
    dict(id="accounting/inference-only-small-model", add=["compute-research/inference-only-proof-cost-known"],
         drop=["lean/inference-only-on-served-traffic"]),
]

REMOVE = ["infra/efficient-automated-tasks", "benchmarks/remote-numbers-on-main", "vllm/daily-inference-block-counts"]
PRIORITY = {i: "1" for i in ELEVEN} | {"top/public-split-sunday": "clear", PARKED: "3"}


def links(goal: dict) -> list[str]:
    return [f"{k}:{v}" for w in goal.get("waitingOn", []) for k, v in w.items()]


def build(current: dict[str, dict]) -> list[tuple[str, dict, list[str]]]:
    """(id, the goal to write, what changes) for every goal the tree touches, children first."""
    out = []
    known = set(current) | {spec["id"] for spec in NEW}

    def live(ids: list[str], owner: str) -> list[str]:
        for i in ids:
            if i not in known:
                print(f"   skipped a link from {owner} to {i}: no such goal")
        return [i for i in ids if i in known]
    for spec in NEW:
        if spec["id"] in current:
            continue
        goal = {k: v for k, v in spec.items() if k != "waits" and not (k == "tier" and v == 2)}
        goal.setdefault("status", "on-track")
        if spec.get("waits"):
            goal["waitingOn"] = [{"goal": w} for w in live(spec["waits"], spec["id"])]
        out.append((spec["id"], goal, ["new"]))
    for spec in EDIT:
        if spec["id"] not in current:
            raise SystemExit(f"no goal {spec['id']}")
        goal = _clean(current[spec["id"]])
        before = json.dumps(goal, sort_keys=True)
        changes = []
        for k, v in spec.get("set", {}).items():
            if goal.get(k) != v:
                changes.append(f"{k}: {str(goal.get(k))[:40]!r} -> {str(v)[:60]!r}")
                goal[k] = v
        for k in spec.get("unset", []):
            if k in goal:
                changes.append(f"{k}: unset (was {goal[k]!r})")
                del goal[k]
        have = links(goal)
        want = list(have)
        if "waits" in spec:
            want = [f"goal:{w}" for w in live(spec["waits"], spec["id"])]
        for old, new in spec.get("swap", {}).items():
            want = [f"goal:{new}" if w == f"goal:{old}" else w for w in want]
        want = [w for w in want if w not in {f"goal:{d}" for d in spec.get("drop", [])}]
        want += [f"goal:{a}" for a in live(spec.get("add", []), spec["id"]) if f"goal:{a}" not in want]
        want = list(dict.fromkeys(want))
        if want != have:
            changes.append(f"waits: -{sorted(set(have) - set(want))} +{sorted(set(want) - set(have))}")
            goal["waitingOn"] = [dict([w.split(":", 1)]) for w in want]
            if not goal["waitingOn"]:
                del goal["waitingOn"]
        if json.dumps(goal, sort_keys=True) != before:
            out.append((spec["id"], goal, changes))
    return out


def verify(goals: list[dict]) -> int:
    by = {g["id"]: g for g in goals}
    parents: dict[str, set[str]] = {}
    for g in goals:
        for w in g.get("waitingOn", []):
            if "goal" in w:
                parents.setdefault(w["goal"], set()).add(g["id"])
    roots = set(ELEVEN) | {PARKED}

    def reaches(gid: str, seen: set[str]) -> bool:
        if gid in roots:
            return True
        seen.add(gid)
        return any(reaches(p, seen) for p in parents.get(gid, ()) if p not in seen)

    bad = 0
    for g in goals:
        if g["status"] == "done":
            continue
        if not reaches(g["id"], set()):
            print(f"  UNPLACED {g['id']} ({g['owner']})")
            bad += 1
        if len(g.get("waitingOn", [])) > 16:
            print(f"  OVER 16 LINKS {g['id']}: {len(g['waitingOn'])}")
            bad += 1
        if len(g["title"].split()) > 8:
            print(f"  TITLE OVER 8 WORDS {g['id']}: {g['title']}")
            bad += 1
        for w in g.get("waitingOn", []):
            if "goal" in w and w["goal"] not in by:
                print(f"  BROKEN LINK {g['id']} -> {w['goal']}")
                bad += 1
    tier1 = sorted(g["id"] for g in goals if g["status"] != "done" and g.get("tier") == 1)
    print(f"  explicit tier 1: {tier1}")
    missing = [i for i in ELEVEN if i not in by]
    if missing:
        print(f"  MISSING {missing}")
        bad += len(missing)
    extra = [i for i in tier1 if i not in ELEVEN]
    if extra:
        print(f"  TIER 1 OUTSIDE THE ELEVEN {extra}")
        bad += len(extra)
    print(f"verify: {bad} problems; {sum(g['status'] != 'done' for g in goals)} open goals")
    return bad


def main() -> None:
    route = Route()
    goals = route.call()
    current = {g["id"]: g for g in goals}
    if "--verify" in sys.argv:
        raise SystemExit(1 if verify(goals) else 0)
    writes = build(current)
    for gid, goal, changes in writes:
        print(f"{gid}: {'; '.join(changes)}")
        words = len(goal["title"].split())
        if words > 8:
            print(f"   TITLE {words} words: {goal['title']}")
        if len(goal.get("waitingOn", [])) > 16:
            print(f"   {len(goal['waitingOn'])} links")
    print(f"{len(writes)} writes, {len(REMOVE)} removals, {len(PRIORITY)} priorities")
    if "--apply" not in sys.argv:
        return
    failed = []
    for gid, goal, _ in writes:
        try:
            route.call({"set": goal})
        except (RouteError, SlackError) as e:
            failed.append((gid, str(e)))
            print(f"FAILED {gid}: {e}")
    for gid, level in PRIORITY.items():
        was = current.get(gid, {})
        if (level == "clear" and not was.get("prioritySet")) or (was.get("prioritySet") and str(was.get("priority")) == level):
            continue
        try:
            route.call({"priority": {"id": gid, "priority": None if level == "clear" else int(level)}})
        except (RouteError, SlackError) as e:
            failed.append((gid, str(e)))
            print(f"FAILED priority {gid}: {e}")
    for gid in REMOVE:
        try:
            route.call({"remove": gid})
        except (RouteError, SlackError) as e:
            failed.append((gid, str(e)))
            print(f"FAILED remove {gid}: {e}")
    print(f"applied; {len(failed)} failed")
    verify(route.call())


main()
