
## 19:00Z, proofs coordinator: proofs has dedicated Lean nodes now; use them

infra (INF-453): vy-nebius-cpu-11, cpu-12 and cpu-7 are proofs' Lean nodes, with proofs pools of 8-10 build slots each,
until Fri 13 Nov 6 PM PST; ci's hand shards may share them until about 20:30Z. cpu-1 and cpu-9 stay up until Fri 6 Nov
6 PM PST, and cpu-9 refuses runs while its disk is low. Launch new runs with `--on vy-nebius-cpu-11 --pool proofs`,
`--on vy-nebius-cpu-12 --pool proofs` or `--on vy-nebius-cpu-7 --pool proofs`, spread across the three. Move any run of yours
that hasn't started, and cancel the old copy once the new one registers. The first build on each node is cold, so
put Security/Proofs builds there early.

## 19:30Z, proofs coordinator: scope change from Daniel, the outer system is generic too

Daniel prefers the outer proof system generic as well as the inner. Extend the blueprint and the statement:
1. **An abstract outer system:** a knowledge-sound argument for the relation "openings of the ZK server's commitments exist on
   which the inner verifier circuits accept", with its extractor's interface stated explicitly (rewinding or straight-line,
   expected or strict running time, the hash calls it makes), and zero knowledge as a simulator for the outer view.
   `RecursiveZK` already takes the outer view's zero knowledge as a premise (`OuterViewZK`), so the ZK side is generic in both.
2. **An abstract ZK server:** the properties the server must have (commit to every inner message under hiding leaves, take
   live coins, release rules, refusal on a malformed request), so the theorem doesn't depend on C-Flock's message formats.
   The deployed server stays C-Flock's and is proved to meet the interface as part of the instance.
3. **The main theorem:** compile(inner, outer, server) is knowledge sound with error inner + outer + collision terms (with the
   extractors' composition explicit), and zero knowledge by `RecursiveZK`. The instance is C-Flock for both, with the
   deployed server, recovering today's bounds or better.
4. **Risks** (in your (g) section): composing a rewinding outer extractor with the inner one, the blow-up in running time and
   hash calls, and what the server interface must guarantee for ZK when the worker misbehaves (today's any-worker premise:
   the worker's values satisfy the circuit).

Same deadline for a reviewable statement: Wed 18:00Z. If the outer generalization would push it, say so in your next
checkpoint, and deliver the inner-generic statement on time with the outer as the immediate next section.

## 19:50Z, proofs coordinator: Daniel's terminology, and the sampled side

**Terminology** (Daniel's, per GLOSSARY.md on main). Use it in the blueprint's prose and in new Lean names, and refer to
existing identifiers as they are:
- sampling design, not law;
- the sample, a sampled proof block, to sample, not draw or drawn;
- proof block, not unit;
- trace for a run's values;
- ε_bind, not δ_link;
- acceptance probability P_a, not escape;
- subprotocols and composition, not components or product;
- Flock, not C-Flock;
- the ZK server, not the firewall;
- the verifier, not 'verifier of record'.

**The sampled side.** GENSAMPLE (new worker) states the sampled audit as a compiler over the same abstract proof
system. You own the shared interface: keep it in your note under a stable heading, and check GENSAMPLE's note
(`*-draft-generic-sampling.md`) for a 'For GENERIC' section asking for additions.
