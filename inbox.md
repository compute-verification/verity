[2026-11-15T05:41Z] 46d26ea6ed80
From top (your launcher), Sat 14 Nov ~9:45 PM PST. Daniel sent three things at 9:32 PM PST that change your brief. Read all of it, then go on.

1. THE ONTOLOGY SOURCE IS DANIEL'S WHITEPAPER (Notion page "Verity whitepaper"; you may not be able to open it, so its skeleton is here). Where our files and the whitepaper differ, the whitepaper's words win in anything you draft.
- What Verity is, in his words: "Verity is a system by which an AI developer can make verifiable claims to external auditors about how it is using its datacenters, without revealing its algorithms or data." The central claim it enables: "I am not using my datacenters to meaningfully increase the capabilities of AI systems."
- Each party trusts only: conservative cryptographic assumptions (SHA-512's collision resistance); properties of the other party's hardware that ongoing physical inspection can establish, even if the hardware was adversarially designed; a small amount of hardware of its own.
- Alongside the paper: a reference implementation (Lean), a machine-checked security proof, a benchmark.
- Actors: verifier, prover, workload, attester. Parties: the developer and the auditor.
- Three design goals: inefficiency for prohibited workloads; efficiency for approved workloads; confidentiality.
- Layers, in his section order: Proofs (base argument system, zero-knowledge, sampling, recursion, the composite argument system); Resource accounting (compute, network, memory); Workload classification (training detection, research detection); Relating computations to capabilities progress; Deployment patterns (remote, onsite).

2. TWO RENAMES, DANIEL'S: "policy" is now WORKLOAD CLASSIFICATION, and "accounting" is RESOURCE ACCOUNTING. The directories on `architecture` still say verity/protocols/policy/ and verity/protocols/accounting/ and will move after the landing (planned names: verity/protocols/workload_classification/, verity/protocols/resource_accounting/). In every draft, write the new words in prose, and write paths as they are on the tip today with one line saying they move. Do not rename anything in the tree yourself. More renames may follow from the whitepaper (I have put a list to Daniel); I send them when he answers.
3. FOR AGENTSMD ONLY. Your drafts (root AGENTS.md, verity/AGENTS.md, the skill, AGENTS.platform.draft.md) use the whitepaper's ontology above for the map and the design rules' vocabulary. A rule about a guarantee still keeps its words. In your table of every rule, add one column: does the rule's wording use a term the whitepaper replaces (policy, accounting, device for the auditor's hardware, deployment, composite), yes or no. Your hours are unchanged.

